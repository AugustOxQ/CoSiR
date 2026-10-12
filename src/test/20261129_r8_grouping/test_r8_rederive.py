"""C13: worked examples, guard tests and a schema probe against ticket 10's real pending layout.

The tests import ticket 10's ``r8_gate`` only to PRODUCE a pipeline pending file; the re-derivation under test
(``rederive/r8_rederive.py``) never imports it.
"""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import r8_common as R
import r8_gate as G
from rederive import r8_rederive as RD

P, S, N = 40, 16, 12                    # scorer-train paintings (one row each), selection rows, anchors
TOTAL = P + S
LAB = np.arange(P) % 2                  # NB100 painting labels; every half holds both groups
GROUPS = np.arange(TOTAL)
GROUPS[P:] = np.arange(P, TOTAL) // 2   # selection rows share a painting in pairs (leakage-group codes)
DROPPED = {x: f"late drop of {x}" for x in
           ["V100", "V67", "V33", "NB67", "NB33", "NBC", "AL", "DI", "SG", *RD.REFERENCES, *[f"FC{i}" for i in range(1, 7)]]}
_CACHE = {}


def onehot(lab, k=2):
    return np.eye(k, dtype=np.float32)[lab]


def m13_draws(index=4):
    """ZNB100 by the rule's seeds 7100 + i, 7200 + i, 7300 + i (i = 4), written out here without RD's code."""
    sizes = np.bincount(LAB)
    shares = sizes / P
    perm = np.random.default_rng(7100 + index).permutation(P)
    labels = np.empty(P, np.int64)
    labels[perm] = np.repeat(np.arange(2), sizes)
    train = np.random.default_rng(7200 + index).choice(2, size=P, p=shares)
    n_sel = len(np.unique(GROUPS[P:]))
    sel = np.random.default_rng(7300 + index).choice(2, size=n_sel, p=shares)
    return labels, train, sel


def sha_of(path):
    return RD.sha(path)


def write_json(path, rec):
    Path(path).write_text(json.dumps(R.jsonable(rec)))
    return Path(path)


def store(root):
    """Stored files for NB100 and its control ZNB100 (tiny, zero scores), a dropped V100 and a drops-only file."""
    root = Path(root)
    for folder in ("spaces", "heads", "groupings", "seed42"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    st, sel = np.arange(P, dtype=np.int64), np.arange(P, TOTAL, dtype=np.int64)
    halves = np.array([0] * (P // 2) + [1] * (P // 2))
    z = {"scorer_train": st, "g_st": st, "halves": halves, "selection": sel, "groups": GROUPS, "painting_image": st,
         "row_to_image": np.arange(TOTAL), "image_index": np.arange(TOTAL, dtype=np.int32),
         "anchor": np.full(N, sel[0]), "candidates": np.tile(sel[1:14], (N, 1)),
         "cl": np.arange(N) // 2, "pair_index": np.repeat(np.arange(3), N // 3), "parity": np.arange(N) % 2}
    for c in RD.CONDS:
        for d in RD.DIRS:
            z[f"cos__{c}__{d}"] = np.zeros((N, 13), np.float32)
            z[f"tu__{c}__{d}"] = np.zeros((N, 13), np.float32)
    half_posterior = np.full((TOTAL, 2), np.nan, np.float32)
    half_posterior[sel] = [.5, .5]
    for g in RD.A0:
        for m in ("img", "txt"):
            z[f"e2__{g}__{m}"] = half_posterior
        np.savez(root / "heads" / f"post_{g}.npz", img=half_posterior, txt=half_posterior, rows=sel)
    np.savez(root / "heads" / "post_R0.npz", img=half_posterior, txt=half_posterior, rows=sel)   # legacy R0 (B1)
    np.savez(root / "fixture.npz", **z)
    # NB100: φ of a painting is its label's basis vector, so the centres are basis vectors and the maps are the labels.
    phi = np.zeros((TOTAL, 2), np.float32)
    phi[:P] = onehot(LAB)
    phi[P:] = onehot((np.arange(S) // 3) % 2)
    np.savez(root / "spaces/NB100.npz", phi=phi, image_index=z["image_index"])
    zlab, ztrain, zsel = m13_draws()
    _, inverse = np.unique(GROUPS[P:], return_inverse=True)
    np.savez(root / "groupings/r8_groupings.npz", scorer_train=st, g_st=st, NB100=LAB, ZNB100=zlab, V100=LAB, ZV100=zlab)
    write_json(root / "groupings/r8_groupings.json",
               {"groupings": {x: {"K": 2} for x in ("NB100", "ZNB100", "V100", "ZV100")}, "dropped": {}})
    write_json(root / "groupings/r8_drops_main.json", {"groupings": {}, "dropped": DROPPED})
    for g, own, bank in (("NB100", phi[P:], onehot(LAB)), ("ZNB100", onehot(zsel[inverse]), onehot(ztrain))):
        img = np.full((TOTAL, 2), np.nan, np.float32)
        img[sel] = own
        np.savez(root / f"heads/post_{g}.npz", img=img, txt=half_posterior, rows=sel)
        np.savez(root / f"heads/xheads_{g}.npz", img=bank, txt=np.full((P, 2), .5, np.float32),
                 filled_by_half=(1 - halves).astype(np.int8), classes_half0=np.arange(2), classes_half1=np.arange(2))

    def reader(p):
        srt = np.sort(p, axis=1)
        return {f"{key}__{c}": value for c in RD.CONDS
                for key, value in (("P", p), ("pick", p.argmax(1)), ("m", srt[:, -1] - srt[:, -2]))}

    np.savez(root / "seed42/reader_A0.npz", **reader(np.tile([.6, .2, .2], (N, 1))))
    for g in ("NB100", "ZNB100"):
        np.savez(root / f"seed42/reader_{g}.npz", **reader(np.tile([.4, .1, .1, .4], (N, 1))))
    write_json(root / "seed42/taus_main.json", {"batch": "main", "seed": 9001, "smoke": True, "n_margins": 2 * N,
                                                "taus": {"NB100": [0., 0., 0., 0.], "ZNB100": [0., 0., 0., 0.]}})
    write_stages(root)
    return root / "fixture.npz"


def write_stages(root):
    root = Path(root)
    np.save(root / "AL_embeddings.npy", np.zeros((TOTAL, 256), np.float32))
    np.save(root / "AL_index.npy", np.arange(TOTAL, dtype=np.int64))
    ok = {"ok": True}
    write_json(root / "stage0_regression.json", {"pass": True, "smoke": True})
    write_json(root / "stageE_AL.json", {
        "pass": True, "smoke": True, "reason": None, "n": TOTAL,
        "esc1": {"a_weights_sha256": ok, "a_strict_load": ok, "b_shape": ok, "c_finite": ok, "d_index_hash": ok, "e_repeat": ok},
        "evidence": {"embeddings": {"path": str(root / "AL_embeddings.npy"), "sha256": sha_of(root / "AL_embeddings.npy")},
                     "image_index": {"path": str(root / "AL_index.npy"), "sha256": sha_of(root / "AL_index.npy")}}})
    for name in ("DI", "SG"):    # the weights did not load: a drop record, no evidence
        write_json(root / f"stageE_{name}.json", {"pass": False, "smoke": True, "reason": "weights do not load",
                                                  "esc1": {"a_strict_load": {"ok": False}}, "evidence": {}})
    write_fcmi(root, [1.0, .5, .25, .125])


def write_fcmi(root, losses):
    root = Path(root)
    log = root / "fcmi_losses.jsonl"
    log.write_text("".join(json.dumps({"epoch": 0, "batch": i, "total_loss": v, "finite": v is not None}) + "\n"
                           for i, v in enumerate(losses)))
    np.savez(root / "fcmi_labels.npz", labels=np.arange(P) % 3, painting_index=np.arange(P))
    write_json(root / "stage1_fcmi.json", {
        "pass": False, "wiring_pass": True, "smoke": True,
        "tests": {"process": True, "finite_loss": all(v is not None for v in losses), "clusters": True, "wall_time": True},
        "clusters_before_merge": 3, "steps": len(losses), "error": None,
        "loss_log": str(log), "loss_log_sha256": sha_of(log),
        "labels_before_merge": str(root / "fcmi_labels.npz"), "labels_sha256": sha_of(root / "fcmi_labels.npz")})


def sources_of(root):
    root = Path(root)
    taus = root / "seed42/taus_main.json"
    return {g: {"taus": taus, "taus_sha256": sha_of(taus), "reader": root / f"seed42/reader_{g}.npz",
                "reader_sha256": sha_of(root / f"seed42/reader_{g}.npz")} for g in ("NB100", "ZNB100")}


def groupings_of(root):
    return [Path(root) / "groupings/r8_groupings.json", Path(root) / "groupings/r8_drops_main.json"]


def derived_arrays(root, fixture):
    """What the pipeline would store (here: RD's own arrays on unmodified files), computed once per session."""
    if "arrays" not in _CACHE:
        got = RD.derive_files(root, fixture=fixture, smoke=True, groupings=groupings_of(root), sources=sources_of(root))
        assert not got["disagreements"]
        _CACHE["arrays"] = got["arrays"]
    return copy.deepcopy(_CACHE["arrays"])


STAGE_FILES = ("stage0_regression.json", "stageE_AL.json", "stageE_DI.json", "stageE_SG.json", "stage1_fcmi.json")


def publish(root, fixture, step, *, stored=None, pending_arrays=None, edit=None):
    """Write the seed-42 batch, its index and ticket 10's pending file (made by 10's own gate_stats)."""
    root = Path(root)
    arrays = derived_arrays(root, fixture)
    stored = arrays if stored is None else stored
    pending_arrays = stored if pending_arrays is None else pending_arrays
    batch = root / "seed42/arrays_main.npz"
    np.savez(batch, **stored)
    taus = root / "seed42/taus_main.json"
    write_json(root / "seed42/index.json", {"batches": [{
        "file": batch.name, "sha256": sha_of(batch), "batch": "main", "seed": 9001, "smoke": True, "n_episodes": N,
        "groupings": ["NB100", "ZNB100"], "dropped": {}, "taus_file": taus.name, "taus_sha256": sha_of(taus),
        "readers_sha256": {x: sha_of(root / f"seed42/reader_{x}.npz") for x in ("A0", "NB100", "ZNB100")}}]})
    probe = None
    include_nbc = False
    if step == "gate":
        trigger = G.gate_stats(pending_arrays, dropped=DROPPED, include_nbc=False)["trigger"]
        probe = write_json(root / "probe_trigger.json", {**trigger, "smoke": True, "step": "trigger", "input_sha256": {}})
        include_nbc = trigger["triggered"]
    stats = G.gate_stats(pending_arrays, dropped=DROPPED, include_nbc=include_nbc)
    listed = [root / n for n in STAGE_FILES] + [root / "seed42/index.json", batch, *groupings_of(root), R.RULE] + ([probe] if probe else [])
    records = json.loads((root / "groupings/r8_groupings.json").read_text())["groupings"]
    payload = {**stats, "step": step, "smoke": True, "groupings": records, "dropped": DROPPED,
               "stage_sha256": {n: sha_of(root / n) for n in STAGE_FILES},
               "probe_trigger_sha256": sha_of(probe) if probe else None,
               "input_sha256": {str(Path(p).resolve()): sha_of(p) for p in listed},
               "report_flags": {"AL": "text"}, "rule_sha256": sha_of(R.RULE), "written": "now"}
    if edit:
        edit(payload)
    return write_json(root / f"{step}_pending.json", payload)


def run(root, fixture, step, pending, groupings=None):
    return RD.check_files(root, pending, step, fixture=fixture, smoke=True, groupings=groupings or groupings_of(root))


@pytest.fixture
def fx(tmp_path):
    return tmp_path, store(tmp_path)


def where(got):
    return {r["where"] for r in got["disagreements"]}


# ------------------------------------------------------------------------------------------------ derivations

def test_own_map_and_crossfit_use_paintings_and_opposite_half():
    phi = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], np.float32)
    labels = np.array([0, 1, 0, 1], np.int64)
    got = RD.placements(phi, labels, np.array([0, 0, 1, 1]), np.array([[1, 0], [0, 1], [1, 1]], np.float32))
    assert got["selection"].tolist() == [[1, 0], [0, 1], [1, 0]]    # the [1, 1] tie goes to the lowest group
    assert got["bank"].tolist() == [[1, 0], [0, 1], [1, 0], [0, 1]]
    opposed = RD.placements(phi[[0, 1, 3, 2]], labels, np.array([0, 0, 1, 1]), phi)
    assert opposed["bank"].argmax(1).tolist() == [1, 0, 1, 0]
    with pytest.raises(AssertionError, match="empty half"):
        RD.placements(phi, labels, np.array([0, 1, 0, 1]), phi)


def test_224_cells_strict_ties_and_threshold_equality():
    n = 6
    B = {c: {d: np.zeros((n, 13), np.float32) for d in RD.DIRS} for c in RD.CONDS}
    P4 = {c: np.tile([.4, .1, .1, .4], (n, 1)) for c in RD.CONDS}
    stack = {d: np.zeros((n, 4, 13), np.float32) for d in RD.DIRS}
    arrays, detail = RD.family(B, stack, P4, [0., .25, .5, .75], np.arange(n) % 2, "XAFF")
    assert arrays["gate_a"].tolist() == [[1.] * n, [0.] * n, [0.] * n, [0.] * n]
    assert arrays["fused_cells"].tolist() == [0, 0]
    assert arrays["cf_cells"].tolist() == [0, 0]
    assert arrays["sigma"].tolist() == [0, 0]
    assert detail["rho_f"].shape == (224, 2) and detail["rho_f"].dtype == np.int64
    assert not detail["rho_f"].any()
    assert all(not arrays[m].any() and not arrays["cf__" + m].any() for m in RD.METRICS)


def test_224_cell_fused_winner_counterpart_tie_and_opposite_parity_assembly():
    n, parity = 6, np.arange(6) % 2
    B = {c: {d: np.zeros((n, 13), np.float32) for d in RD.DIRS} for c in RD.CONDS}
    Pr = {"a": np.tile([1., 0., 0., 0.], (n, 1)), "b": np.tile([0., 0., 0., 1.], (n, 1))}
    stack = {d: np.zeros((n, 4, 13), np.float32) for d in RD.DIRS}
    for d in RD.DIRS:
        stack[d][:, 0, 0] = 1
        stack[d][:, 3, 1] = 1
    arrays, detail = RD.family(B, stack, Pr, [1.] * 4, parity, "XAFF")
    assert arrays["fused_cells"].tolist() == [1, 1]
    assert arrays["cf_cells"].tolist() == [0, 0]
    assert detail["rho_f"][1].tolist() == [12, 12]
    assert detail["gam_f"][1].tolist() == [12, 12]
    assert arrays["r1"].tolist() == arrays["gain"].tolist() == [1.] * n
    assert not arrays["cf__r1"].any()
    for d in RD.DIRS:
        stack[d][parity == 1] = 0
    arrays, detail = RD.family(B, stack, Pr, [1.] * 4, parity, "XAFF")
    assert arrays["fused_cells"].tolist() == [1, 0]
    assert not arrays["r1"].any()    # each half is scored by the other half's cell


def test_random_control_seeds_and_selection_painting_order():
    labels, got = RD.random_inputs(np.array([0, 1, 0, 1]), np.array([40, 10, 30, 20, 40]), 4)
    assert labels.tolist() == [0, 1, 1, 0]
    assert got["bank"].argmax(1).tolist() == [1, 1, 0, 0]
    assert got["selection"].argmax(1).tolist() == [1, 0, 0, 0, 1]


# ------------------------------------------------------------------------------------------------ statistics

def verdict_arrays(n=12, names=("NB100", "NB67", "ZNB100")):
    a = {"cl": np.arange(n) // 2, "pair_index": np.repeat(np.arange(3), n // 3), "parity": np.arange(n) % 2}
    for name in ("AFF", "B", "B0", "B1"):
        a[name + "__r1"] = np.zeros(n)
    for g in names:
        a[f"BX_{g}__r1"] = np.zeros(n)
        for kind in RD.KINDS:
            name = kind + "_" + g
            a[name + "__r1"] = np.full(n, .25)
            a[name + "__gain"] = np.full(n, .25)
            a[name + "_cf__r1"] = np.zeros(n)
            a[name + "_cf__gain"] = np.zeros(n)
    return a


def others_dropped(*keep):
    return {x: f"drop {x}" for x in [*RD.MAIN_GROUPS, *RD.SCREEN_GROUPS, *RD.REFERENCES] if x not in keep}


def stats_of(a, evaluated, dropped, nbc_allowed=True, strict=True):
    return RD.statistics(a, set(evaluated), dropped, nbc_allowed=nbc_allowed, strict=strict)


def test_pick_trigger_flag_and_interval_follow_rule():
    got = stats_of(verdict_arrays(), {"NB100", "NB67", "ZNB100"}, others_dropped("NB100", "NB67"), strict=False)
    assert got["main"]["pick"] == "R1_NB100"
    assert got["main"]["random_control_flag"] is True
    assert got["main"]["promising"] is False
    assert got["main"]["head_target"] == {"grouping": "NB100", "scorer_kind": "R1", "scorer": "R1_HNB100", "source_scorer": "R1_NB100"}
    assert got["screen"]["pick"] is None
    assert got["trigger"]["triggered"] is False and got["trigger"]["strength"] is None
    assert got["trigger"]["reason"] == "an NB scorer passes"
    assert got["fallback"]["main_has_pick"] is True and got["fallback"]["screen_eligible_at_gate"] is False
    rec = got["main"]["candidates"]["R1_NB100"]
    assert rec["SC3"] == {"sum4d": 12, "n": 12, "point": 25., "ci95": [25., 25.], "pass": True}
    assert rec["bar_comparator"] == "B1"
    assert rec["S1"]["pass"] is True    # smoke bar ceil(0.5 * 12 / 25) = 1, not the fixed 246
    assert rec["gain_sum4d"] == 12 and rec["r7_pass"] is True
    assert got["main"]["candidates"]["R1_V100"]["dropped"] is True


def test_no_nb_scorer_evaluated_is_not_triggered_with_tens_reason():
    a = {k: v for k, v in verdict_arrays().items() if not k.startswith(("R1_", "XAFF_", "BX_"))}
    got = stats_of(a, set(), others_dropped())
    assert got["trigger"]["triggered"] is False and got["trigger"]["strength"] is None
    assert got["trigger"]["source_scorer"] is None
    assert got["trigger"]["reason"] == "no NB scorer was evaluated; NBC does not run"
    assert len(got["trigger"]["nb_records"]) == 6 and all(r["dropped"] for r in got["trigger"]["nb_records"].values())
    assert got["main"]["head_target"] is None and got["main"]["main_line_fails_at_gate"] is True


def test_smoke_integer_bars_scale_with_ceil_a_n_over_25():
    assert (RD.need("-0.3", 4096), RD.need("0.5", 12288), RD.need("0", 4096)) == (-49, 246, 0)
    assert (RD.need("-0.3", 100), RD.need("0.5", 300), RD.need("0.5", 12)) == (-1, 6, 1)
    a = verdict_arrays(300, ("NB100", "ZNB100"))
    a["AFF__r1"][:] = .25
    for kind in RD.KINDS:
        s = np.full(300, .25)
        s[0] = 0.    # pair 0 sum4d = -1: the bar for 100 anchors is -1
        a[f"{kind}_NB100__r1"] = s
    names = {"NB100", "ZNB100"}
    rec = stats_of(a, names, others_dropped("NB100"), strict=False)["main"]["candidates"]["R1_NB100"]
    assert rec["SC4"]["emotion__style"]["sum4d"] == -1 and rec["SC4"]["emotion__style"]["pass"] is True
    for kind in RD.KINDS:
        a[f"{kind}_NB100__r1"][1] = 0.
    rec = stats_of(a, names, others_dropped("NB100"), strict=False)["main"]["candidates"]["R1_NB100"]
    assert rec["SC4"]["emotion__style"]["sum4d"] == -2 and rec["SC4"]["emotion__style"]["pass"] is False
    assert rec["SC4"]["pass"] is False and rec["pass"] is False


def test_trigger_gain_tie_and_strength():
    a = verdict_arrays()
    a["AFF__r1"][:] = .5
    for kind in RD.KINDS:
        a[f"{kind}_NB67__gain"][:] = .5
    names = {"NB100", "NB67", "ZNB100"}
    got = stats_of(a, names, others_dropped("NB100", "NB67"), strict=False)
    assert got["trigger"]["triggered"] is True and got["trigger"]["strength"] == .67
    assert got["trigger"]["source_scorer"] == "R1_NB67"
    assert got["trigger"]["reason"] == "none of the six NB scorers passes"
    assert got["main"]["pick"] is None
    # at the trigger step NBC is not part of the roster; at the gate it is (here: a recorded drop)
    assert "R1_NBC" in got["main"]["candidates"]
    early = stats_of(a, names, others_dropped("NB100", "NB67"), nbc_allowed=False, strict=False)
    assert "R1_NBC" not in early["main"]["candidates"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_statistics_equal_ticket_10_gate_stats_on_random_and_crafted_arrays(seed):
    n = 300
    drops = others_dropped("NB100", "NB67", "AL")    # NBC stays dropped
    names = ("NB100", "NB67", "ZNB100", "ZNB67", "AL", "ZAL")
    a = verdict_arrays(n, names)
    rng = np.random.default_rng(seed)
    if seed:    # random quarter-point arrays
        for k in list(a):
            if k.endswith("__r1"):
                a[k] = rng.integers(0, 5, n) / 4.
            elif k.endswith("__gain") and "_cf" not in k:
                a[k] = rng.integers(-2, 5, n) / 4.
    if seed == 2:    # a clear winner over AFF
        a["AFF__r1"][:] = 0.
        for kind in RD.KINDS:
            a[f"{kind}_NB67__r1"] = np.full(n, .75)
    include = G.gate_stats(a, dropped=drops, include_nbc=False)["trigger"]["triggered"]
    want = G.gate_stats(a, dropped=drops, include_nbc=include)
    want["fallback"] = {k: v for k, v in want["fallback"].items() if k not in RD.FALLBACK_CONSTANT_STRINGS}
    got = stats_of(a, names, drops)
    files = RD.Files()
    RD.compare_json(files, "stats", got, json.loads(json.dumps(R.jsonable(want))))
    assert not files.disagreements, files.disagreements[:3]


# ------------------------------------------------------------------------------------------------ guard tests (review M3)

def test_s2_scope_is_pair_two_and_sc3_needs_a_strictly_positive_lower_end():
    a = verdict_arrays(12, ("NB100", "ZNB100"))
    pair = a["pair_index"]
    for kind in RD.KINDS:    # s - cf is +.25 on pairs 0 and 1 and -.25 on pair 2
        a[f"{kind}_NB100__r1"] = np.where(pair == 2, 0., .25)
        a[f"{kind}_NB100_cf__r1"] = np.where(pair == 2, .25, 0.)
    rec = stats_of(a, {"NB100", "ZNB100"}, others_dropped("NB100"), strict=False)["main"]["candidates"]["R1_NB100"]
    assert rec["S2"]["sum4d"] == -4 and rec["S2"]["pass"] is False    # pairs 0 and 1 alone would give +8
    assert rec["S2"]["n"] == 4
    # SC3 with every difference zero: the interval is [0, 0] and 0 > 0 is false
    a = verdict_arrays(12, ("NB100", "ZNB100"))
    for kind in RD.KINDS:
        a[f"{kind}_NB100__r1"] = np.zeros(12)
    rec = stats_of(a, {"NB100", "ZNB100"}, others_dropped("NB100"), strict=False)["main"]["candidates"]["R1_NB100"]
    assert rec["SC3"]["ci95"] == [0., 0.] and rec["SC3"]["pass"] is False and rec["pass"] is False


def test_pending_record_with_a_missing_or_an_extra_key_disagrees(fx):
    root, fixture = fx

    def edit(payload):
        del payload["main"]["candidates"]["R1_NB100"]["SC3"]["n"]
        payload["main"]["candidates"]["R1_NB100"]["S1"]["surprise"] = 1
        del payload["screen"]["candidates"]["R1_AL"]["bar_comparator"]
        del payload["fallback"]
        payload["bogus"] = 1

    got = run(root, fixture, "trigger", publish(root, fixture, "trigger", edit=edit))
    assert got["status"] == "disagree"
    reasons = {(r["where"], r["reason"]) for r in got["disagreements"]}
    assert ("pending.main.candidates.R1_NB100.SC3.n", "missing record key") in reasons
    assert ("pending.main.candidates.R1_NB100.S1.surprise", "extra record key") in reasons
    assert ("pending.screen.candidates.R1_AL.bar_comparator", "missing record key") in reasons
    assert ("pending.fallback", "missing pending key") in reasons
    assert ("pending.bogus", "extra pending key") in reasons


def test_declared_input_with_a_wrong_sha256_disagrees(fx):
    root, fixture = fx
    (root / "listed.json").write_text("{}")
    key = str((root / "listed.json").resolve())

    def edit(payload):
        payload["input_sha256"][key] = "0" * 64

    got = run(root, fixture, "trigger", publish(root, fixture, "trigger", edit=edit))
    assert got["status"] == "disagree"
    row = next(r for r in got["disagreements"] if r["where"] == key)
    assert row["reason"] == "input SHA-256 differs" and row["expected"] == "0" * 64
    assert key in got["input_sha256"]    # an unused declared input is still hashed


def test_stage_and_probe_hashes_and_undeclared_inputs_are_checked(fx):
    root, fixture = fx
    drops_key = str((root / "groupings/r8_drops_main.json").resolve())

    def edit(payload):
        payload["stage_sha256"]["stageE_AL.json"] = "1" * 64
        del payload["input_sha256"][drops_key]

    got = run(root, fixture, "trigger", publish(root, fixture, "trigger", edit=edit))
    assert "pending.stage_sha256.stageE_AL.json" in where(got)
    assert {"where": drops_key, "reason": "input is not listed in the pending record's input_sha256"} in got["disagreements"]

    def edit_gate(payload):
        payload["probe_trigger_sha256"] = "2" * 64

    got = run(root, fixture, "gate", publish(root, fixture, "gate", edit=edit_gate))
    assert "pending.probe_trigger_sha256" in where(got)
    # the pending trigger must also be the one in probe_trigger.json
    probe = json.loads((root / "probe_trigger.json").read_text())
    probe["strength"] = .33
    write_json(root / "probe_trigger.json", probe)
    pending = json.loads((root / "gate_pending.json").read_text())
    pending["probe_trigger_sha256"] = sha_of(root / "probe_trigger.json")
    pending["input_sha256"][str((root / "probe_trigger.json").resolve())] = sha_of(root / "probe_trigger.json")
    write_json(root / "gate_pending.json", pending)
    assert "probe_trigger.json.strength" in where(run(root, fixture, "gate", root / "gate_pending.json"))


def test_batch_without_a_required_array_or_with_an_extra_array_disagrees(fx):
    root, fixture = fx
    full = derived_arrays(root, fixture)
    missing = {k: v for k, v in full.items() if k != "R1_NB100__r1"}
    got = run(root, fixture, "trigger", publish(root, fixture, "trigger", stored=missing, pending_arrays=full))
    assert got["status"] == "disagree"
    assert {"where": "R1_NB100__r1", "reason": "required pipeline array missing"} in got["disagreements"]
    extra = {**full, "XAFF_NB100__surprise": np.zeros(N)}
    got = run(root, fixture, "trigger", publish(root, fixture, "trigger", stored=extra, pending_arrays=full))
    assert {"where": "XAFF_NB100__surprise", "reason": "pipeline array was not independently derived"} in got["disagreements"]


def test_control_uses_the_rule_seeds_7104_7204_7304_for_znb100(fx):
    root, fixture = fx
    zlab, ztrain, zsel = m13_draws(4)
    other = m13_draws(5)
    assert not np.array_equal(zlab, other[0]) and not np.array_equal(ztrain, other[1])
    assert RD.CONTROL_INDEX["NB100"] == 4
    got = run(root, fixture, "trigger", publish(root, fixture, "trigger"))
    assert got["status"] == "agree", got["disagreements"][:5]
    assert any("post_ZNB100" in p for p in got["input_sha256"])
    # stored ZNB100 labels from the next index (seed 7105) are named
    np.savez(root / "groupings/r8_groupings.npz", scorer_train=np.arange(P), g_st=np.arange(P), NB100=LAB,
             ZNB100=other[0], V100=LAB, ZV100=zlab)
    got = run(root, fixture, "trigger", publish(root, fixture, "trigger"))
    assert "M13 labels ZNB100" in where(got)


# ------------------------------------------------------------------------------------------------ stored files, ticket 10 layout

@pytest.mark.parametrize("step", ["trigger", "gate"])
def test_ticket_10_pending_layout_agrees_with_the_rederivation(fx, step):
    """Schema probe: 10's own gate_stats output in 10's pending layout, on identical arrays, at both steps."""
    root, fixture = fx
    pending = publish(root, fixture, step)
    got = run(root, fixture, step, pending)
    assert got["status"] == "agree", got["disagreements"][:5]
    assert str(pending.resolve()) in got["input_sha256"]
    assert str((root / "heads/post_NB100.npz").resolve()) in got["input_sha256"]
    assert all(Path(p).is_absolute() and len(v) == 64 for p, v in got["input_sha256"].items())
    assert got["stage_checks"]["AL"] == {"b_shape": True, "c_finite": True, "d_index_hash": True}
    assert got["stage_checks"]["DI"]["status"] == "not produced"
    if step == "gate":    # all-zero scores: no NB scorer passes, so NBC runs (here: a recorded drop)
        assert got["rederived"]["trigger"]["triggered"] is True and "R1_NBC" in got["rederived"]["main"]["candidates"]
    else:
        assert "R1_NBC" not in got["rederived"]["main"]["candidates"]


def test_grouping_files_in_any_order_and_a_drops_only_file_need_no_npz(fx):
    root, fixture = fx
    pending = publish(root, fixture, "trigger")
    assert not (root / "groupings/r8_drops_main.npz").exists()
    got = run(root, fixture, "trigger", pending, groupings=list(reversed(groupings_of(root))))
    assert got["status"] == "agree", got["disagreements"][:5]
    # V100 has labels in the first file and a late drop in the second: the drop wins; ZV100 (parent dropped) is skipped
    assert got["rederived"]["main"]["candidates"]["R1_V100"]["dropped"] is True
    assert "R1_ZV100" not in got["rederived"]["main"]["controls"]


def test_one_anchor_change_is_named_even_with_refreshed_input_hash(fx):
    root, fixture = fx
    stored = derived_arrays(root, fixture)
    stored["R1_NB100__r1"] = stored["R1_NB100__r1"].copy()
    stored["R1_NB100__r1"][3] += .25
    got = run(root, fixture, "gate", publish(root, fixture, "gate", stored=stored))
    assert got["status"] == "disagree"
    row = next(r for r in got["disagreements"] if r["where"] == "R1_NB100__r1")
    assert row["changed"] == 1 and row["first_index"] == [3]


@pytest.mark.parametrize("mutation,name", [("own", "M11 post_NB100 img"), ("bank", "M12 xheads_NB100 img"),
                                           ("tau", "tau NB100"), ("reader", "reader_NB100 pick a")])
def test_stored_own_map_bank_tau_and_reader_changes_are_named(fx, mutation, name):
    root, fixture = fx
    derived_arrays(root, fixture)    # the arrays of the unmodified files
    if mutation == "tau":
        write_json(root / "seed42/taus_main.json", {"batch": "main", "taus": {"NB100": [0., .25, .5, .75], "ZNB100": [0.] * 4}})
    elif mutation == "reader":
        with np.load(root / "seed42/reader_NB100.npz") as f:
            z = {k: f[k] for k in f.files}
        z["pick__a"] = z["pick__a"].copy()
        z["pick__a"][0] = 1
        np.savez(root / "seed42/reader_NB100.npz", **z)
    else:
        path = root / ("heads/post_NB100.npz" if mutation == "own" else "heads/xheads_NB100.npz")
        with np.load(path) as f:
            z = {k: f[k] for k in f.files}
        row = P if mutation == "own" else 0
        z["img"][row] = z["img"][row, ::-1]
        np.savez(path, **z)
    got = run(root, fixture, "trigger", publish(root, fixture, "trigger"))
    assert got["status"] == "disagree" and name in where(got)


def test_c13_interval_tolerance_and_exact_integer_fields():
    files = RD.Files()
    RD.compare_json(files, "pending", {"ci95": [1., 2.], "sum4d": 1}, {"ci95": [1. + 5e-10, 2.], "sum4d": 1})
    assert not files.disagreements
    RD.compare_json(files, "pending", {"ci95": [1., 2.], "sum4d": 1}, {"ci95": [1. + 2e-9, 2.], "sum4d": 2})
    assert {r["where"] for r in files.disagreements} == {"pending.ci95[0]", "pending.sum4d"}


# ------------------------------------------------------------------------------------------------ stage evidence

def declared_of(root):
    return {str((Path(root) / n).resolve()): sha_of(Path(root) / n) for n in STAGE_FILES}


def test_stage_evidence_is_recomputed_from_the_stage_records_files(fx):
    root, fixture = fx
    index = np.arange(TOTAL, dtype=np.int64)
    files = RD.Files()
    got, shas = RD.stage_checks(files, root, declared_of(root), index, True)
    assert got["AL"] == {"b_shape": True, "c_finite": True, "d_index_hash": True}
    assert got["DI"]["status"] == got["SG"]["status"] == "not produced"
    assert got["FCMI"] == {"finite_loss": True, "clusters": True, "clusters_before_merge": 3}
    assert set(shas) == set(STAGE_FILES) and not files.disagreements
    # an expected index with one swap fails ESC1 (d); the record says it passed
    files = RD.Files()
    bad = index.copy()
    bad[[0, 1]] = bad[[1, 0]]
    RD.stage_checks(files, root, declared_of(root), bad, True)
    assert "ESC1 AL.d_index_hash" in where({"disagreements": files.disagreements})
    # non-finite embeddings fail ESC1 (c)
    emb = np.zeros((TOTAL, 256), np.float32)
    emb[3, 3] = np.nan
    np.save(root / "AL_embeddings.npy", emb)
    rec = json.loads((root / "stageE_AL.json").read_text())
    rec["evidence"]["embeddings"]["sha256"] = sha_of(root / "AL_embeddings.npy")
    write_json(root / "stageE_AL.json", rec)
    files = RD.Files()
    RD.stage_checks(files, root, declared_of(root), index, True)
    assert "ESC1 AL.c_finite" in where({"disagreements": files.disagreements})


def test_fcmi_finite_loss_is_recomputed_from_the_losses_not_the_stored_flag(fx):
    root, fixture = fx
    write_fcmi(root, [1.0, None, .25])    # a null total_loss is a non-finite step
    rec = json.loads((root / "stage1_fcmi.json").read_text())
    rec["tests"]["finite_loss"] = True    # the record claims finite
    write_json(root / "stage1_fcmi.json", rec)
    files = RD.Files()
    got, _ = RD.stage_checks(files, root, declared_of(root), np.arange(TOTAL, dtype=np.int64), True)
    assert got["FCMI"]["finite_loss"] is False
    assert "SC2.finite_loss" in where({"disagreements": files.disagreements})


# ------------------------------------------------------------------------------------------------ CLI

def test_smoke_cli_writes_c13_contract_and_disagreement_exit(tmp_path):
    root = tmp_path / "results/smoke/gate"
    fixture = store(root)
    pending = publish(root, fixture, "trigger")
    env = {**os.environ, "R8_RESULTS": str(tmp_path / "results"), "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}
    command = [sys.executable, str(Path(RD.__file__)), "--step", "trigger", "--root", str(root), "--pending", str(pending),
               "--groupings", *map(str, groupings_of(root)), "--index", str(root / "seed42/index.json"),
               "--fixture", str(fixture), "--smoke"]
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=200)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((root / "rederive/trigger_rederive.json").read_text())
    assert report["status"] == "agree" and report["step"] == "trigger" and report["smoke"] is True
    assert str(pending.resolve()) in report["input_sha256"] and len(report["rule_sha256"]) == 64
    stored = derived_arrays(root, fixture)
    stored["R1_NB100__r1"] = stored["R1_NB100__r1"].copy()
    stored["R1_NB100__r1"][0] += .25
    publish(root, fixture, "trigger", stored=stored)    # a smoke record may be rewritten
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=200)
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert json.loads((root / "rederive/trigger_rederive.json").read_text())["status"] == "disagree"
