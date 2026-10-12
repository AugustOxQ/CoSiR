"""Round 7 runner: DECISION_RULE L10–L16, §§5, 6, 7, 9 (prep, pending, independent finalize, stage 3).

Stages, one function each behind one entry point (rule §9 order): prep, ceiling [--finalize], gate [--finalize],
sensitivity (§7.1), smoke (§9, rule smoke), build (§7.2), test [--finalize] (§7.3 to §7.7).
Chain smoke (`--chain-smoke`) runs the whole chain in order on dev 9001 and tests 9002, 9003 with fixture groupings,
the public head/reader/bundle/fusion paths and the index's smoke sizes; the re-derivation runs as separate processes.
It never writes to reused folders except through run_baselines.py (new files only). Re-derivation is never imported.

Rule smoke (`smoke`, §9): stage-3 code end to end on seeds 9001 to 9003 with the frozen real objects (heads, readers,
taus.json, gate_verdict.json read from RESULTS), outputs under RESULTS/smoke/rule/. It prints no metric.
WIRING MUTATION (chosen here, rule §9): the first pick's posteriors in the bundle are replaced by its random control Zk's
(Zk has Vk's group sizes, so no shape changes) just before the pick is scored. The named assertion that must fail is
`wiring[pick_posteriors_are_the_frozen_head]` in `check_wiring`, which compares the bundle's posteriors for the pick's
grouping with heads/post_{X}.npz. If it does not fail (or another error does), the smoke record says passed false and the
run exits non-zero; `build` refuses without a passed smoke record.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import r7_common as R
import r7_groupings as G
import r7_nrkmeans as N
import r7_heads as H
import r7_reader as Q
import r7_bundle as B
import r7_fusion as F
import r7_gate as S
import r7_chain as Ch

GROUPINGS = Q.GROUPINGS + ("T",)
REFERENCES = tuple(f"{kind}_{x}" for x in ("R0", "N") + tuple(f"Z{k}" for k in range(1, 7))
                   for kind in S.KINDS)
METRICS = ("r1", "gain", "other", "swap", "strict")
CORE = {"ceiling": ("sum4d", "n", "point", "ci95", "zone"),
        "gate": ("candidates", "references", "picks", "n", "alpha", "promising", "random_control_flag"),
        "test": ("chains", "FV")}
HERE = Path(__file__).resolve().parent
PICK_POSTERIORS = "wiring[pick_posteriors_are_the_frozen_head]"   # the rule smoke's named assertion


def read_record(path, smoke=None):
    rec = json.loads(Path(path).read_text())
    assert isinstance(rec, dict), f"{path}: expected object"
    assert rec.get("rule_sha256") == R.rule_sha(), f"{path}: rule SHA differs"
    if smoke is not None:
        assert rec.get("smoke") == smoke, f"{path}: smoke role differs"
    return rec


def read_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        arr = {k: z[k] for k in z.files}
    assert {"cl", "pair_index", "parity"} <= arr.keys(), f"{path}: missing episode IDs"
    n = len(arr["cl"])
    for k in ("cl", "pair_index", "parity"):
        assert arr[k].dtype == np.int64 and arr[k].shape == (n,), f"{path}/{k}: IDs differ"
    assert n > 0 and np.isin(arr["pair_index"], (0, 1, 2)).all(), f"{path}: invalid pairs"
    assert np.isin(arr["parity"], (0, 1)).all(), f"{path}: invalid parity"
    for k, a in arr.items():
        if k.endswith(tuple(f"__{m}" for m in METRICS)):
            assert a.shape == (n,) and a.dtype == np.float64 and np.isfinite(a).all(), f"{path}/{k}: metrics differ"
        elif k.endswith(("__gate_a", "__gate_b")):
            assert a.shape == (4, n) and a.dtype == np.float32 and np.isin(a, (0, 1)).all(), f"{path}/{k}: gates differ"
        elif k.endswith(("__fused_cells", "__cf_cells")):
            assert a.shape == (2,) and a.dtype == np.int64 and np.all((a >= 0) & (a < 224)), f"{path}/{k}: cells differ"
        elif k.endswith("__sigma"):
            assert a.shape == (2,) and a.dtype == np.float64 and np.isfinite(a).all(), f"{path}/{k}: sigma differs"
    return arr


def _group_records(root):
    out = {}
    for stem in ("r7_groupings", "r7_grouping_N"):
        rec = read_record(root / "groupings" / f"{stem}.json")
        assert "groupings" in rec, f"{stem}: missing records"
        with np.load(root / "groupings" / f"{stem}.npz", allow_pickle=False) as z:
            assert {"scorer_train", "g_st"} <= set(z.files), f"{stem}: missing alignment"
            st, g = z["scorer_train"], z["g_st"]
            assert st.dtype == g.dtype == np.int64 and st.ndim == 1 and g.shape == st.shape, f"{stem}: row shape/dtype"
            assert len(st) > 0 and np.all(np.diff(st) > 0) and (g >= 0).all(), f"{stem}: row order/codes"
            if stem == "r7_groupings":
                alignment = (st, g)
            else:
                assert all(np.array_equal(a, b) for a, b in zip(alignment, (st, g))), "N: grouping alignment differs"
            for name, item in rec["groupings"].items():
                assert type(item["dropped"]) is bool, f"{name}: missing drop status"
                assert item["dropped"] or name in z.files, f"{name}: missing labels"
                if not item["dropped"]:
                    labels = z[name]
                    assert labels.dtype == np.int64 and labels.shape == st.shape, f"{name}: label shape/dtype"
                    assert np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))), f"{name}: noncompact labels"
        out.update(rec["groupings"])
    assert set(GROUPINGS) <= out.keys(), "prep: missing grouping records"
    return out


def prep(role):
    root = role.root
    if role.smoke:
        smoke_groupings(role)
    else:
        for stem, build in (("r7_groupings", G.build), ("r7_grouping_N", N.build)):
            if not (root / "groupings" / f"{stem}.json").exists():
                build(out=root / "groupings")
    records = _group_records(root)
    assert not any(records[x]["dropped"] for x in ("R0", "T")), "required R0/T grouping dropped"
    rb = R.load("rb_build")
    data, sp, st, groups = rb.splits()
    halves = H.load_halves(st, groups)
    dropped = [x for x, rec in records.items() if rec["dropped"]]
    for name in Q.BASE + GROUPINGS:
        if name in dropped:
            continue
        labels = H.load_groupings((name,), st, halves["g_st"], root)[name]
        rec = H.build_heads(name, data, sp, labels, halves, root=root, smoke=role.smoke)
        if rec["dropped"]:
            assert name not in Q.BASE + ("R0", "T"), f"required head {name} dropped"
            dropped.append(name)
        else:
            ctx = SimpleNamespace(groups=groups, selection=sp.selection)
            if name in Q.BASE:
                with np.load(root / "heads" / f"post_{name}.npz", allow_pickle=False) as z:
                    assert {"rows", "img", "txt"} <= set(z.files), f"{name}: posterior keys differ"
                    assert np.array_equal(z["rows"], sp.selection), f"{name}: selection differs"
                    B._posterior_ok({m: z[m] for m in ("img", "txt")}, ctx, name)
            else:
                B.load_posteriors(ctx, name, root)
            assert (root / "heads" / f"head_{name}.pkl").is_file(), f"{name}: incomplete head"
            if name != "T":
                labels = H.load_groupings((name,), st, halves["g_st"], root)[name]
                Q.load_crossfit(name, labels, halves, root)
    tasks = []
    for name in Q.GROUPINGS:
        if name in dropped:
            continue
        path = root / "readers" / f"reader_{name}.json"
        if path.exists():
            labels = H.load_groupings(Q.BASE + (name,), st, halves["g_st"], root)
            rec = Q.validate_reader(name, labels, halves, root, role.smoke)
            if rec["dropped"]:
                dropped.append(name)
        else:
            tasks.append(name)
    # The public reader entry point runs three fits at once, eight threads each.
    if tasks:
        subprocess.run([sys.executable, str(Path(__file__).with_name("r7_reader.py")),
                        "--grouping", *tasks, "--root", str(root), "--jobs", "3",
                        *(["--smoke"] if role.smoke else [])], check=True)
        for name in tasks:
            rec = read_record(root / "readers" / f"reader_{name}.json", role.smoke)
            if rec["dropped"]:
                dropped.append(name)
    assert "R0" not in dropped, "G3: R0 reader dropped"
    path = root / "prep.json"
    if not path.exists():
        R.write_json_once(path, {"smoke": role.smoke, "dropped": sorted(set(dropped))})
    else:
        assert read_record(path, role.smoke)["dropped"] == sorted(set(dropped)), "prep drop record changed"
    print(f"prep: {len(GROUPINGS)} grouping records checked")


def _ready(role, stage):
    if stage == "ceiling":
        rec = read_record(role.root / "stage0_regression.json")
        assert rec.get("pass") is True, "stage 0 regression did not pass"  # run_r7_stage0 writes "pass"
    else:
        if role.smoke:
            # Smoke never decides: exercise gate wiring even after a smoke K1 STOP.
            read_record(role.root / "ceiling_pending.json", True)
        else:
            rec = read_record(role.root / "ceiling_verdict.json", False)
            assert rec["zone"] in ("WEAK", "CLEAR"), "K1 STOP: gate forbidden"
    return read_record(role.root / "prep.json", role.smoke)


def episode_arrays(bundle):
    return {k: getattr(bundle, k) for k in ("cl", "pair_index", "parity")}


def add_metrics(arr, name, pa):
    assert set(pa) == set(METRICS), f"{name}: missing metrics"
    arr.update({f"{name}__{k}": v for k, v in pa.items()})


def save_reader(role, bundle, name, rd):
    save_reader_to(role.root, role.dev, name, rd)


def save_reader_to(root, seed, name, rd):
    path = Path(root) / f"seed{seed}" / f"reader_{name}.npz"
    arrays = {f"{k}__{c}": rd[k][c] for k in ("P", "m", "pick") for c in ("a", "b")}
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            assert set(arrays) <= set(z.files), f"{path}: missing reader keys"
            assert all(np.array_equal(v, z[k]) for k, v in arrays.items()), f"{path}: reader changed"
    else:
        R.write_npz_once(path, **arrays)


def ceiling(role):
    _ready(role, "ceiling")
    path = role.root / f"seed{role.dev}" / "arrays_ceiling.npz"
    if path.exists():
        arr = read_arrays(path)
    else:
        bundle = B.build_bundle(role.dev, role.smoke, extra=("T",), root=role.root)
        scored = F.told(bundle, B.A0 + ("T",), {"emotion": "affect", "style": "T", "genre": "image"})
        save_reader(role, bundle, "T", scored["reader"])
        arr = episode_arrays(bundle) | scored["arrays"]
        R.write_npz_once(path, **arr)
    assert len(arr["cl"]) == (192 if role.smoke else 12288), "ceiling episode count differs"
    mask = arr["pair_index"] == 2
    assert mask.sum() == (64 if role.smoke else 4096), "K1 must get exactly style x genre episodes"
    pending = S.ceiling_stats(arr["told_T__r1"][mask], arr["told_T_cf__r1"][mask], arr["cl"][mask])
    write_pending(role, "ceiling", pending)
    print(f"ceiling: {path.name}, {int(mask.sum())} style x genre episodes checked")


def gate(role):
    prep_rec = _ready(role, "gate")
    path = role.root / f"seed{role.dev}" / "arrays.npz"
    dropped = prep_rec["dropped"]
    active = tuple(x for x in Q.GROUPINGS if x not in dropped)
    if path.exists():
        arr = read_arrays(path)
    else:
        bundle = B.build_bundle(role.dev, role.smoke, extra=active, root=role.root)
        arr = episode_arrays(bundle)
        add_metrics(arr, "B", bundle.pB)
        add_metrics(arr, "B0", bundle.pBp)
        add_metrics(arr, "B1", B.condition_free(bundle, B.A1)[0])
        arr.update(F.score_aff(bundle)["arrays"])
        save_reader(role, bundle, "A0", F.reader(bundle, B.A0))
        tau_path = role.root / "taus.json"
        taus = {}
        for x in active:
            parts = B.A0 + (x,)
            rd = F.reader(bundle, parts, role.root / "readers" / f"reader_{x}.pkl")
            save_reader(role, bundle, x, rd)
            taus[x], n = F.thresholds(rd["m"])
            assert n == 2 * bundle.n == (384 if role.smoke else 24576), "L12 margin count differs"
        if tau_path.exists():
            stored = read_record(tau_path)
            assert all(stored.get(x) == tau for x, tau in taus.items()), "frozen taus changed"
        else:
            R.write_json_once(tau_path, taus)
        for x in active:
            parts, reader = B.A0 + (x,), role.root / "readers" / f"reader_{x}.pkl"
            add_metrics(arr, f"BX_{x}", B.condition_free(bundle, parts)[0])
            arr.update(F.score_r1(bundle, parts, reader, taus[x])["arrays"])
            arr.update(F.score_x_aff(bundle, parts, reader, taus[x])["arrays"])
        R.write_npz_once(path, **arr)
    assert len(arr["cl"]) == (192 if role.smoke else 12288), "gate episode count differs"
    frozen = read_record(role.root / "taus.json")
    for x in active:
        tau = np.asarray(frozen.get(x), dtype=np.float64)
        assert tau.shape == (4,) and np.isfinite(tau).all() and (np.diff(tau) >= 0).all(), f"{x}: frozen taus differ"
    pending = S.gate_stats(arr, references=REFERENCES, dropped=dropped)
    write_pending(role, "gate", pending)
    print(f"gate: {path.name}, {len(arr)} arrays checked")


def write_pending(role, stage, rec):
    path = role.root / f"{stage}_pending.json"
    if path.exists():
        old = read_record(path, role.smoke)
        assert all(old[k] == rec[k] for k in CORE[stage]), f"{stage}: pending changed"
    else:
        R.write_json_once(path, {**rec, "smoke": role.smoke})


def differences(a, b, path=""):
    """Compare only the pipeline's core schema; reporting extras are ignored recursively."""
    if isinstance(a, dict):
        if not isinstance(b, dict):
            return [path]
        return [p for k, v in a.items() for p in
                ([f"{path}.{k}"] if k not in b else differences(v, b[k], f"{path}.{k}"))]
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            return [path]
        return [p for i, v in enumerate(a) for p in differences(v, b[i], f"{path}[{i}]")]
    if type(a) is float:
        ok = type(b) in (int, float) and np.isfinite(b) and abs(a - b) <= 1e-9
    else:
        ok = type(a) is type(b) and a == b
    return [] if ok else [path]


def core_test(rec):
    """The test verdict's core keys (index): reporting extras of 10's chain records are dropped."""
    def one(r, keys):
        return {k: r[k] for k in (("status", "point", "ci95") if r.get("status") == "not tested" else keys)}
    plain = ("n_j", "sum4d", "point", "ci95", "ci_level")
    chains = {}
    for fam, ch in rec["chains"].items():
        f1 = ch["F1"]
        chains[fam] = {"pick": ch["pick"], "alpha": ch["alpha"],
                       "F1": {"checks": {n: one(c, plain) for n, c in f1["checks"].items()},
                              "holm_order": f1["holm_order"], "passes": f1["passes"], "pass": f1["pass"]},
                       "F2": one(ch["F2"], plain + ("pass",)), "F3": one(ch["F3"], plain + ("pass",))}
    return {"chains": chains, "FV": rec["FV"]}


def required_keys(stage, pending):
    """Arrays the re-derivation must hold and equal, per stage (names of the index)."""
    ids = {"cl", "pair_index", "parity"}  # rule §9, C6: the episode IDs are compared exactly on every stage
    if stage == "ceiling":
        return ids | {"told_T__r1", "told_T_cf__r1"}
    required = ids | {f"{name}__{m}" for name in ("B", "B0", "B1", "AFF") for m in METRICS}
    if stage == "gate":
        names = [n for n in S.CANDIDATES if not pending["candidates"][n]["dropped"]]
    else:
        names = [ch["pick"] for ch in pending["chains"].values()]
    for name in names:
        x = name.split("_", 1)[1]
        required.update(f"{prefix}__{m}" for prefix in (name, f"{name}_cf", f"BX_{x}") for m in METRICS)
        required.update(f"{name}__{k}" for k in ("fused_cells", "cf_cells", "sigma", "gate_a", "gate_b"))
    return required


def phase_diffs(arr, phase_path, required, label):
    """Rule §9: re-derived arrays equal the pipeline's exactly (values and dtype); required keys present in both."""
    diff = [f"{label}.pipeline:{k}:missing" for k in sorted(required - set(arr))]
    with np.load(phase_path, allow_pickle=False) as z:
        diff.extend(f"{phase_path.name}.{k}:missing" for k in sorted(required - set(z.files)))
        for k in sorted(set(z.files) & set(arr)):
            if z[k].dtype != arr[k].dtype or not np.array_equal(z[k], arr[k]):
                diff.append(f"{phase_path.name}.{k}")
    return diff


def force_picks(rec):
    """Chain smoke only: n = 0 on fixture groupings, so take the first non-dropped candidate of each family in rule
    order (V1..V6, R1 before X-AFF), so that the stage-3 wiring still runs. Real stages keep every refusal."""
    assert rec["n"] == 0 and not any(rec["picks"].values()), "forced picks only when n = 0"
    picks = {fam: next((n for n in S.CANDIDATES if n.split("_")[1] in vs and not rec["candidates"][n]["dropped"]), None)
             for fam, vs in S.FAMILIES.items()}
    n = sum(p is not None for p in picks.values())
    return {**rec, "picks": picks, "n": n, "alpha": {f: 0.0125 if n == 2 else 0.025 for f, p in picks.items() if p},
            "forced_pick": True, "n_before_force": 0}


def finalize(role, stage):
    _ready(role, stage)
    verdict = role.root / f"{stage}_verdict.json"
    assert not verdict.exists(), f"{verdict}: refusing to overwrite verdict"
    pending = read_record(role.root / f"{stage}_pending.json", role.smoke)
    stats_path = role.root / "rederive" / f"stats_{stage}.json"
    stats = json.loads(stats_path.read_text())
    assert isinstance(stats, dict), f"{stats_path}: expected object"
    core = core_test(pending) if stage == "test" else {k: pending[k] for k in CORE[stage]}
    diff = differences(core, stats, stage)
    # the re-derivation may not hold a key the pipeline lacks, for every stage
    diff.extend(f"reverse{p}" for p in differences({k: stats[k] for k in CORE[stage] if k in stats}, core, ""))
    required = required_keys(stage, pending)
    if stage == "ceiling":
        items = [(f"seed{role.dev}/arrays_ceiling.npz", "phaseK.npz")]
    elif stage == "gate":
        items = [(f"seed{role.dev}/arrays.npz", f"phaseA_seed{role.dev}.npz")]
    else:
        items = [(f"seed{s}/arrays.npz", f"phaseB_seed{s}.npz") for s in role.tests]
    for name, phase in items:
        arr = read_arrays(role.root / name)
        diff.extend(phase_diffs(arr, role.root / "rederive" / phase, required, name))
    if diff:
        path = R.fix_path(role.root / f"{stage}_disagreement.json")
        R.write_json_once(path, {"stage": stage, "differing_keys": diff, "smoke": role.smoke})
        raise SystemExit(f"re-derivation disagreement: {len(diff)} keys; {path}")
    rec = {k: v for k, v in pending.items() if k not in ("rule_sha256", "written")}
    if stage == "gate" and role.smoke and rec["n"] == 0:
        rec = force_picks(rec)
        print(f"gate: smoke only, n = 0 on fixture groupings; forced picks {rec['picks']} (forced_pick true)")
    R.write_json_once(verdict, rec)
    print(f"{stage}: finalize comparisons passed; {verdict.name}")
    if stage == "ceiling" and not role.smoke:
        if pending["zone"] == "STOP":
            print("STOP: design L ends; FV NO-GO; no stage 2 or fresh-seed build.")
        elif pending["zone"] == "WEAK":
            print("WEAK: ceiling is weak; continue with the report's first line flagged and a run-chat phone note.")
    if stage == "test":
        print(f"FV {pending['FV']}")


def smoke_groupings(role):
    """The only fixture: painting labels on the real scorer-train rows (L1/C6)."""
    assert role.smoke and role.root == R.SMOKE_CHAIN, "fixture requires chain smoke"
    rb = R.load("rb_build")
    data, sp, st, groups = rb.splits()
    halves = H.load_halves(st, groups)
    g = halves["g_st"]
    rng = np.random.default_rng(9001)
    labels = {name: rng.integers(0, 27 if name == "T" else 8, size=36518, dtype=np.int64)[g]
              for name in GROUPINGS if name != "R0"}
    labels["R0"] = R.load("r6_heads").csd_labels(st, len(groups))[st]
    for stem, names in (("r7_groupings", G.IDS), ("r7_grouping_N", ("N",))):
        arrays = {"scorer_train": st, "g_st": g, **{name: labels[name] for name in names}}
        path = role.root / "groupings" / f"{stem}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                assert set(arrays) <= set(z.files), f"{path}: fixture keys differ"
                assert all(np.array_equal(z[k], v) for k, v in arrays.items()), f"{path}: fixture changed"
        else:
            R.write_npz_once(path, **arrays)
        record = {"fixture": True, "smoke": True, "requested": list(names), "n_rows": len(st),
                  "n_paintings": 36518, "scorer_train_sha256": G.sha_array(st), "g_st_sha256": G.sha_array(g),
                  "groupings": {name: {"dropped": False, "fixture": True} for name in names}}
        if path.with_suffix(".json").exists():
            old = read_record(path.with_suffix(".json"), True)
            assert all(old[k] == v for k, v in record.items()), f"{path}: fixture record differs"
        else:
            R.write_json_once(path.with_suffix(".json"), record)


# ------------------------------------------------------------------------------------------------ stage 3 (rule §7)
AB = R.TEST / "20261030_aspect_baselines"
EP_KINDS = ("episodes", "per_anchor", "baselines")
PAIR_ORDER = ("emotion__style", "emotion__genre", "style__genre")
EP_FIELDS = ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")
EARLIER = (42, 43, 45, 47, 48, 49, 50, 51)
CODES_KEY = "20261030_aspect_baselines/results/codes_provenance.json"
LEDGER_LINE = "55, 56, 57 test, round 7; 58 and later free"
# Held ledger row H5 (docs/superpowers/held_ledger.md, seeds 52 to 54, 4,096 per pair): the nine per-pair SHA-256s,
# pairs in the order emotion x style, emotion x genre, style x genre.
H5 = {52: ("f56c357dbfa4f666daf3f2849ce7a6e77676769f62e18ce704c7c7004f770542",
           "58798701e01e4a8e20235252bbf985b52735c0de2a54127221178fc0d6da17d0",
           "e9c591dca0e082f8c63810430f0ffd0c3262f55f062f1bc33d665423a175db50"),
      53: ("ae0cb9c34dfe55d99cc54862f48725d8ec7a0bcc42b23f723ac6512994422c57",
           "850790fbc11995509550bc9cf061ee2868b5253af892ed9aa40cb09623a9fcaf",
           "88138225cff600134cbd020ba29187afec3bd78062266a795f0156d668595404"),
      54: ("dc81db77128edd184df2c55a93529d1873f7f805bac1bbbf297d4b2f6cd34508",
           "439425c52220b820954ede4c03cc2f94ee4b4ce30f6fc8822f728fb0382d86f6",
           "8d7c415c3d46cf2744891e468d1ac0757a7c11e8aae5a67dc9e75ffa3ac7e6c2")}
assert len(H5) == 3 and all(len(set(v)) == 3 for v in H5.values()), "H5: nine hashes"


def n_episodes(role, per_seed_pairs=3):
    return per_seed_pairs * (64 if role.smoke else 4096)


def stage3_ready(role, stage):
    """Rule §9 order and §7.7: stage 3 never runs after a K1 STOP or n = 0; sensitivity comes before smoke, build, test.
    Chain smoke only continues past a smoke STOP (smoke never decides). -> (gate record, ceiling zone)."""
    cv = read_record(role.root / "ceiling_verdict.json", role.smoke)
    if not role.smoke:
        assert cv["zone"] in ("WEAK", "CLEAR"), "K1 STOP: stage 3 forbidden (FV NO-GO)"
    gate = read_record(role.root / "gate_verdict.json", role.smoke)
    picked = {f: p for f, p in gate["picks"].items() if p is not None}
    assert gate["n"] == len(picked) >= 1, "n = 0: design L ends (FV NO-GO); stage 3 forbidden"
    assert set(gate["alpha"]) == set(picked), "gate verdict: alpha families differ from picks"
    if stage != "sensitivity":
        read_record(role.root / "sensitivity.json", role.smoke)
    return gate, cv["zone"]


def sensitivity_record(arr, picks, smoke):
    """Ticket 10's record. Smoke only: 64 episodes per pair can have as many anchor paintings as episodes, and the
    decomposition needs more episodes than paintings; that check then gets x = 0 and a flag (smoke never decides)."""
    try:
        return Ch.sensitivity_record(arr, picks, smoke=smoke)
    except AssertionError as e:
        if not smoke or "more episodes than paintings" not in str(e):
            raise
    chosen, alpha = Ch._picks(picks)
    chains = {}
    for family, pick in chosen.items():
        diffs, cl, pi = Ch._differences(arr, pick, smoke=True)
        mask = pi == 2

        def one(d, c, stage):
            try:
                return Ch.sensitivity(d, c, alpha, stage, smoke=True)
            except AssertionError as err:
                assert "more episodes than paintings" in str(err), str(err)
                return {"x": 0.0, "n": len(d), "smoke_degenerate_clusters": True}
        chains[family] = {"pick": pick, "alpha": alpha,
                          "F1": {n: one(diffs[n], cl, "F1") for n in Ch.CHECKS},
                          "F2": one(diffs["AFF"], cl, "F2"), "F3": one(diffs["B1"][mask], cl[mask], "F3")}
    return {"chains": chains}


def sensitivity(role):
    """§7.1: R3 §6.1's decomposition per check of each chain, on the pick's seed-42 per-episode differences."""
    gate, _ = stage3_ready(role, "sensitivity")
    arr = read_arrays(role.root / f"seed{role.dev}" / "arrays.npz")
    assert len(arr["cl"]) == n_episodes(role), "sensitivity: seed-42 episode count differs"
    rec = R.jsonable(sensitivity_record(arr, gate["picks"], role.smoke))
    assert set(rec["chains"]) == {f for f, p in gate["picks"].items() if p}, "sensitivity: chains differ from picks"
    path = role.root / "sensitivity.json"
    if path.exists():
        assert read_record(path, role.smoke)["chains"] == rec["chains"], "sensitivity changed"
    else:
        R.write_json_once(path, {**rec, "smoke": role.smoke,
                                 "gate_verdict_sha256": R.sha_file(role.root / "gate_verdict.json")})
    print(f"sensitivity: {len(rec['chains'])} chains, {sum(len(c['F1']) + 2 for c in rec['chains'].values())} checks")


def pick_parts(name):
    kind, x = name.split("_")
    assert kind in ("R1", "XAFF") and x in S.VARIANTS, f"invalid pick {name}"
    return kind, x


def check_wiring(bundle, frozen, x):
    """The rule smoke's named assertion: the pick's grouping is scored with its own frozen head's posteriors."""
    with np.load(Path(frozen) / "heads" / f"post_{x}.npz", allow_pickle=False) as z:
        for side in ("img", "txt"):
            assert np.array_equal(bundle.post[x][side], z[side], equal_nan=True), \
                f"{PICK_POSTERIORS}: {x}/{side} differs from heads/post_{x}.npz"


def score_seed(frozen, out, seed, smoke, picks, taus, *, reuse=True, bundle=None):
    """§7.3 and §7.4 for one seed: frozen objects from `frozen`, this seed's own parity halves, arrays and reader
    probabilities under `out`/seed{seed}/. -> (arrays, bundle or None when reused from file)."""
    frozen, out = Path(frozen), Path(out)
    chosen = [p for p in picks.values() if p is not None]
    assert chosen, "no pick to score"
    groups = sorted({pick_parts(p)[1] for p in chosen})
    assert all(np.asarray(taus[x]).shape == (4,) for x in groups), "frozen taus: four values per pick grouping"
    path = out / f"seed{seed}" / "arrays.npz"
    if reuse and path.exists():
        return read_arrays(path), None
    bundle = bundle or B.build_bundle(seed, smoke, extra=tuple(groups), root=frozen)
    for x in groups:
        check_wiring(bundle, frozen, x)
    arr = episode_arrays(bundle)
    add_metrics(arr, "B", bundle.pB)
    add_metrics(arr, "B0", bundle.pBp)
    add_metrics(arr, "B1", B.condition_free(bundle, B.A1)[0])
    aff = F.score_aff(bundle)
    arr.update(aff["arrays"])
    save_reader_to(out, seed, "A0", aff["reader"])
    for name in chosen:
        kind, x = pick_parts(name)
        parts = B.A0 + (x,)
        scored = (F.score_r1 if kind == "R1" else F.score_x_aff)(bundle, parts, frozen / "readers" / f"reader_{x}.pkl", taus[x])
        arr.update(scored["arrays"])
        save_reader_to(out, seed, x, scored["reader"])
        add_metrics(arr, f"BX_{x}", B.condition_free(bundle, parts)[0])
    n = 192 if smoke else 12288
    assert len(arr["cl"]) == n and np.array_equal(arr["pair_index"], np.repeat([0, 1, 2], n // 3)), \
        f"seed {seed}: episode count or aspect-pair blocks differ"
    if path.exists():
        old = read_arrays(path)
        assert old.keys() == arr.keys() and all(np.array_equal(old[k], v) for k, v in arr.items()), \
            f"{path}: arrays changed on rerun"
    else:
        R.write_npz_once(path, **arr)
    read_arrays(path)
    return arr, bundle


def frozen_shas(frozen, picks):
    frozen = Path(frozen)
    files = [frozen / "gate_verdict.json", frozen / "taus.json"]
    for name in picks.values():
        if name is not None:
            x = pick_parts(name)[1]
            files += [frozen / "readers" / f"reader_{x}.pkl", frozen / "heads" / f"post_{x}.npz"]
    return {str(p.relative_to(frozen)): R.sha_file(p) for p in files}


def test(role):
    """§7.3 to §7.6: per seed the picks' arrays, then pooled F1 to F3 and FV -> test_pending.json."""
    gate, zone = stage3_ready(role, "test")
    for s in role.tests:
        verify_build_record(role, s)
    taus = read_record(role.root / "taus.json")
    sens = read_record(role.root / "sensitivity.json", role.smoke)
    arrays = {}
    for s in role.tests:
        arrays[s], _ = score_seed(role.root, role.root, s, role.smoke, gate["picks"], taus)
        print(f"test: seed {s}: {len(arrays[s]['cl'])} episodes, {len(arrays[s])} arrays checked")
    rec = Ch.test_record(arrays, gate["picks"], sens, smoke=role.smoke)
    # §7.7 and the index: pass the ceiling zone; chain smoke alone continues past a smoke STOP (it never decides)
    rec["FV"] = Ch.friday_verdict(rec["chains"], ceiling_zone=None if role.smoke else zone)
    rec = R.jsonable(rec)
    assert set(rec) == {"chains", "FV"} and rec["FV"] in ("GO", "NO-GO"), "test record keys differ"
    write_pending(role, "test", {**rec, "frozen_sha256": frozen_shas(role.root, gate["picks"])})
    print(f"test: pooled F1 to F3 for {len(rec['chains'])} chains written to test_pending.json")


# ------------------------------------------------------------------------------------------------ seed builds (§7.2)
def seed_files(seed, smoke):
    d = AB / "results" / ("smoke" if smoke else "")
    return {k: d / f"{k}_seed{seed}.{'json' if k == 'baselines' else 'npz'}" for k in EP_KINDS}


def run_baselines(seed, smoke, log_file):
    """run_baselines.py as a subprocess: new files only, never --overwrite; output to a file nobody reads."""
    script = R.path_of("20261030_aspect_baselines/run_baselines.py")
    assert R.sha_file(script) == R.CODE["20261030_aspect_baselines/run_baselines.py"], "run_baselines.py SHA differs"
    assert seed in (R.SMOKE_SEEDS if smoke else R.TEST_SEEDS), f"seed {seed} may not be built here"
    cmd = [sys.executable, str(script), "--episodes-seed", str(int(seed))] + (["--smoke"] if smoke else [])
    assert "--overwrite" not in cmd
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
           "PYTHONDONTWRITEBYTECODE": "1"}
    with open(log_file, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(R.MAIN), env=env).returncode


def pair_hashes(npz):
    """Per-pair SHA-256 of an episodes file, as run_baselines.py and run_gonogo.EvalContext compute it."""
    from dataclasses import fields
    import importlib
    R.to_front((R.MAIN,))
    mod = importlib.import_module("src.eval.aspect_episodes")
    assert Path(mod.__file__).resolve().is_relative_to(R.MAIN / "src"), "aspect_episodes is not MAIN's"
    assert tuple(f.name for f in fields(mod.AspectEpisodes)[2:]) == EP_FIELDS, "AspectEpisodes fields differ"
    with np.load(npz) as z:
        assert tuple(str(x) for x in z["pair_order"]) == PAIR_ORDER, f"{Path(npz).name}: pair order differs"
        return {p: mod.episodes_sha256(mod.AspectEpisodes(*p.split("__"), *(z[f"{p}__{k}"].astype(np.int64)
                                                                           for k in EP_FIELDS)))
                for p in PAIR_ORDER}


def recorded_hashes(path):
    rec = json.loads(Path(path).read_text())
    h = rec["episodes_sha256"]
    assert tuple(h) == PAIR_ORDER == tuple(rec["pair_order"]), f"{Path(path).name}: pair order differs"
    return {p: str(h[p]) for p in PAIR_ORDER}, int(rec["episodes_seed"]), int(rec["n_per_pair"])


def earlier_hashes():
    """{seed: {pair: SHA-256}} of baselines_seed{42,43,45,47,48,49,50,51}.json (SHA-256 asserted where a record has it)."""
    r3 = R.load("r3_common")
    out, shas = {}, {}
    for s in EARLIER:
        path = AB / "results" / f"baselines_seed{s}.json"
        shas[s] = R.sha_file(path)
        key = f"20261030_aspect_baselines/results/baselines_seed{s}.json"
        if key in r3.INPUTS:
            assert shas[s] == r3.INPUTS[key], f"baselines_seed{s}.json differs from round 3's D15 SHA-256"
        else:
            b3 = R.R3D / "results" / f"build_seed{s}.json"
            if b3.is_file():
                assert shas[s] == json.loads(b3.read_text())["sha256"]["baselines"], f"baselines_seed{s}.json SHA differs"
        h, seed, _ = recorded_hashes(path)
        assert seed == s, f"baselines_seed{s}.json is seed {seed}"
        out[s] = h
    return out, shas


def hash_problems(new, others_new, earlier):
    """§7.2: the new per-pair hashes differ from each other, from the other new seeds', from every earlier seed's
    and from H5's nine."""
    problems = []
    if len(set(new.values())) != len(new):
        problems.append("two aspect pairs of this seed share an episode SHA-256")
    for label, group in (("new seed", others_new), ("earlier seed", earlier), ("H5 seed", {s: dict(zip(PAIR_ORDER, v))
                                                                                          for s, v in H5.items()})):
        for s, hs in group.items():
            problems += [f"{q} equals {label} {s} {p}" for p, h in hs.items() for q, v in new.items() if v == h]
    return problems


def build_seed(role, seed, others_new, earlier, check_only):
    """One seed: build (or, chain smoke only, check the existing smoke episodes), then §7.2's checks -> record."""
    rec_path = role.root / f"build_seed{seed}.json"
    files = seed_files(seed, check_only)
    n_per_pair = 64 if check_only else 4096
    codes = R.path_of(CODES_KEY)
    record = {"seed": seed, "smoke": role.smoke, "check_only": check_only, "attempts": [], "ledger_edit_due": LEDGER_LINE,
              "files": {k: str(p) for k, p in files.items()}}

    def stop(msg):
        R.write_json_once(rec_path, {**record, "passed": False, "stop": msg})
        raise SystemExit(f"seed {seed}: {msg}; stop and report (rule §7.2)")

    present = [k for k in EP_KINDS if files[k].exists()]
    if check_only:
        assert len(present) == 3, f"smoke seed {seed}: episodes missing ({present}); chain smoke does not build"
        record["codes_provenance_before"] = record["codes_provenance_after"] = R.sha_file(codes)
        assert record["codes_provenance_before"] == R.INPUTS[CODES_KEY], "codes_provenance.json differs"
    else:
        assert not present, f"seed {seed}: {present} exist; a seed is never rebuilt and never --overwrite; stop"
        for attempt in (1, 2):
            log = role.root / (f"build_seed{seed}.log" if attempt == 1 else f"build_seed{seed}_attempt{attempt}.log")
            att = {"attempt": attempt, "log": log.name, "codes_provenance_before": R.sha_file(codes)}
            record["attempts"].append(att)
            record.setdefault("codes_provenance_before", att["codes_provenance_before"])
            if att["codes_provenance_before"] != R.INPUTS[CODES_KEY]:
                stop("codes_provenance.json differs from its SHA-256 before the build")
            print(f"build: seed {seed} attempt {attempt} -> {log.name}", flush=True)
            att["exit_status"] = int(run_baselines(seed, False, log))
            att["codes_provenance_after"] = record["codes_provenance_after"] = R.sha_file(codes)
            if att["codes_provenance_after"] != R.INPUTS[CODES_KEY]:
                stop("codes_provenance.json changed during the build")
            if att["exit_status"] == 0:
                break
            if files["baselines"].exists():
                stop(f"the build exited {att['exit_status']} after writing baselines (not a crash before it)")
            if files["episodes"].exists():
                att["episodes_sha256"] = R.sha_file(files["episodes"])
                try:
                    att["episode_pair_sha256"] = pair_hashes(files["episodes"])
                except Exception as e:  # noqa: BLE001  an unreadable partial file is recorded, not repaired
                    att["episode_pair_sha256_error"] = type(e).__name__
            if attempt == 2:
                stop("the second build attempt failed too")
            # crash rule (§7.2): partial outputs are set aside under new names (nothing is deleted), then one repeat
            att["set_aside"] = []
            for k in EP_KINDS:
                if files[k].exists():
                    aside = files[k].with_name(f"{files[k].stem}.crash{attempt}{files[k].suffix}")
                    assert not aside.exists(), f"{aside}: exists"
                    files[k].rename(aside)
                    att["set_aside"].append(aside.name)
        missing = [k for k in EP_KINDS if not files[k].exists()]
        if missing:
            stop(f"the build exited 0 but {missing} are missing")
        first = record["attempts"][0]
        if len(record["attempts"]) == 2 and "episode_pair_sha256" in first:
            if first["episode_pair_sha256"] != pair_hashes(files["episodes"]):
                stop("the two runs' per-pair episode SHA-256s differ (crash rule)")
            record["crash_rule_episodes_equal"] = True
    record["sha256"] = {k: R.sha_file(p) for k, p in files.items()}
    recorded, rec_seed, rec_n = recorded_hashes(files["baselines"])
    record["episode_pair_sha256"] = recorded
    problems = hash_problems(recorded, others_new, earlier)
    checks = {"baselines_record_is_this_seed": rec_seed == seed, "n_per_pair": rec_n == n_per_pair,
              "episode_pair_hashes_match_baselines_json": pair_hashes(files["episodes"]) == recorded,
              "episode_hashes_distinct": not problems}
    record["checks"], record["hash_problems"] = checks, problems
    for k, v in checks.items():
        print(f"build: seed {seed}: CHECK {k} {'PASS' if v else 'FAIL'}")
    if not all(checks.values()):
        stop(f"build checks failed: {[k for k, v in checks.items() if not v]} {problems}")
    record["passed"] = True
    R.write_json_once(rec_path, record)
    return record


def verify_build_record(role, seed):
    """Every later step: the seed's build record passed under this rule and its files still have the recorded SHA-256s."""
    rec = read_record(role.root / f"build_seed{seed}.json", role.smoke)
    assert rec.get("seed") == seed and rec.get("passed") is True and rec.get("check_only") is role.smoke, \
        f"seed {seed}: build record did not pass"
    for k, p in seed_files(seed, role.smoke).items():
        assert p.is_file() and R.sha_file(p) == rec["sha256"][k], f"seed {seed}: {p.name} differs from its build record"
    return rec


def build(role):
    """§7.2: seeds 55, 56, 57 one at a time (chain smoke: check-only on the existing smoke episodes, no run_baselines)."""
    check_only = role.smoke
    stage3_ready(role, "build")
    if not check_only:
        smoke = read_record(R.SMOKE_RULE / "smoke_record.json", True)
        assert smoke.get("passed") is True, "build refused: the rule smoke record did not pass"
        assert smoke["gate_verdict_sha256"] == R.sha_file(role.root / "gate_verdict.json"), \
            "build refused: the smoke ran on another gate verdict"
    earlier, earlier_shas = earlier_hashes()
    done = {}
    for s in role.tests:
        if (role.root / f"build_seed{s}.json").exists():
            rec = verify_build_record(role, s)  # resumed run: a seed already built and checked is not touched
        else:
            rec = build_seed(role, s, dict(done), earlier, check_only)
        done[s] = rec["episode_pair_sha256"]
        print(f"build: seed {s} PASS (build_seed{s}.json)")
    print(f"build: seed ledger edit due (run chat, docs only): \"{LEDGER_LINE}\"")


# ------------------------------------------------------------------------------------------------ rule smoke (§9)
def smoke_episodes(seeds):
    """Smoke episodes reused from run_baselines' results/smoke/ when all three files of a seed exist (SHA-256s
    recorded), else built there with run_baselines.py --smoke --episodes-seed s."""
    out = {}
    for s in seeds:
        files = seed_files(s, True)
        have = [k for k in EP_KINDS if files[k].exists()]
        assert len(have) in (0, 3), f"smoke seed {s}: partial outputs {have}; stop"
        if not have:
            R.SMOKE_RULE.mkdir(parents=True, exist_ok=True)
            assert run_baselines(s, True, R.SMOKE_RULE / f"build_smoke_seed{s}.log") == 0, f"smoke seed {s}: build failed"
        out[s] = {"built_here": not have, "sha256": {k: R.sha_file(p) for k, p in seed_files(s, True).items()}}
    return out


def mutated(bundle, frozen, x):
    """The wiring mutation: grouping x's posteriors replaced by its random control's (same class count, no shape
    change); a V grouping with no Z control falls back to another V head of the same class count."""
    own = bundle.post[x]
    for other in (f"Z{x[1:]}",) + tuple(v for v in S.VARIANTS if v != x):
        if (Path(frozen) / "heads" / f"post_{other}.npz").is_file():
            post = B.load_posteriors(bundle.ctx, other, frozen)
            if post["img"].shape == own["img"].shape and not np.array_equal(post["img"], own["img"], equal_nan=True):
                return SimpleNamespace(**{**vars(bundle), "post": {**bundle.post, x: post}}), other
    raise SystemExit(f"rule smoke: no other head of {x}'s class count to mutate with")


def separate(script, *args):
    """Stage 0 and the re-derivation run as separate processes from this folder; never imported (rule §9)."""
    subprocess.run([sys.executable, str(HERE / script), *args], check=True, cwd=str(HERE))


def rule_smoke(role):
    """§9 smoke: stage-3 code end to end on seeds 9001 to 9003 with the frozen real objects; prints no metric."""
    assert role.name == "rule" and role.root == R.SMOKE_RULE and role.smoke, "rule smoke role differs"
    if not __debug__:
        raise SystemExit("rule smoke: assertions are off (python -O); the mutation could not fail")
    frozen, out, seeds = R.RESULTS, role.root, (role.dev,) + tuple(role.tests)
    gate, zone = stage3_ready(SimpleNamespace(root=frozen, smoke=False), "smoke")
    picks, taus = gate["picks"], read_record(frozen / "taus.json")
    rec = {"smoke": True, "seeds": list(seeds), "picks": picks, "checks": {}, "frozen_sha256": frozen_shas(frozen, picks),
           "gate_verdict_sha256": R.sha_file(frozen / "gate_verdict.json"), "passed": False}
    path = out / "smoke_record.json"

    def step(name, fn):
        try:
            res = fn()
        except BaseException as e:  # noqa: BLE001  the smoke records any failure, then stops
            rec["checks"][name] = False
            rec["failure"] = f"{name}: {type(e).__name__}: {str(e)[:300]}"
            R.write_json_once(path, rec)
            print(f"smoke: {name} FAIL")
            raise SystemExit(f"rule smoke failed at {name}; {path}")
        rec["checks"][name] = True
        print(f"smoke: {name} PASS")
        return res

    rec["episodes"] = step("smoke episodes (reused or built)", lambda: smoke_episodes(seeds))
    arrays, bundles = {}, {}
    for s in seeds:
        arrays[s], bundles[s] = step(f"seed {s} stage-3 scoring", lambda s=s: score_seed(
            frozen, out, s, True, picks, taus, reuse=False))
    sens = step("sensitivity on the dev smoke seed", lambda: R.jsonable(
        sensitivity_record(arrays[role.dev], picks, True)))
    R.write_json_once(out / "sensitivity.json", {**sens, "smoke": True})
    test_rec = step("pooled F1 to F3", lambda: R.jsonable(Ch.test_record(arrays, picks, sens, smoke=True)))
    R.write_json_once(out / "test_pending.json", {**test_rec, "smoke": True})

    def rederive():
        need = required_keys("test", test_rec)
        for s in seeds:
            if not (out / "rederive" / f"phaseB_seed{s}.npz").exists():
                separate("rederive/rd7_arrays.py", "--phase", "B", "--seed", str(s))
            diff = phase_diffs(arrays[s], out / "rederive" / f"phaseB_seed{s}.npz", need, f"seed{s}")
            assert not diff, f"re-derived arrays differ: {diff[:5]}"
    step("Phase B re-derivation equals the pipeline's arrays", rederive)

    def mutation():
        x = pick_parts(next(p for p in picks.values() if p))[1]
        bad, other = mutated(bundles[role.dev], frozen, x)
        rec["mutation"] = {"description": f"{x}'s posteriors replaced by {other}'s before scoring",
                           "named_assertion": PICK_POSTERIORS}
        try:
            score_seed(frozen, out / "mutation", role.dev, True, picks, taus, reuse=False, bundle=bad)
        except AssertionError as e:
            assert str(e).startswith(PICK_POSTERIORS), f"another assertion fired: {str(e)[:120]}"
            rec["mutation"]["message_head"] = str(e)[:len(PICK_POSTERIORS)]
            return
        raise AssertionError("the wiring mutation did not fail the named assertion")
    step("wiring mutation fails the named assertion", mutation)
    rec["passed"] = all(rec["checks"].values())
    R.write_json_once(path, rec)
    print(f"smoke: record passed {rec['passed']} ({path.name}); no metric printed")


def chain_smoke(role):
    """The whole chain in order on dev 9001 and tests 9002, 9003 (a step whose output exists is skipped, so the run
    resumes). Smoke never decides: it continues past a STOP, and finalize's forced picks (n = 0) keep stage 3 wired."""
    assert role.smoke and role.root == R.SMOKE_CHAIN, "chain smoke root differs"
    root, dev = role.root, role.dev

    def once(out, fn, *args):
        if not (root / out).exists():
            fn(*args)

    once("stage0_regression.json", separate, "run_r7_stage0.py", "--chain-smoke")
    prep(role)
    ceiling(role)
    once(f"rederive/phaseK.npz", separate, "rederive/rd7_arrays.py", "--chain-smoke", "--phase", "K", "--seed", str(dev))
    once("rederive/stats_ceiling.json", separate, "rederive/rd7_stats.py", "--chain-smoke", "--stage", "ceiling")
    once("ceiling_verdict.json", finalize, role, "ceiling")
    gate(role)
    once(f"rederive/phaseA_seed{dev}.npz", separate, "rederive/rd7_arrays.py", "--chain-smoke", "--phase", "A",
         "--seed", str(dev))
    once("rederive/stats_gate.json", separate, "rederive/rd7_stats.py", "--chain-smoke", "--stage", "gate")
    once("gate_verdict.json", finalize, role, "gate")
    once("sensitivity.json", sensitivity, role)
    build(role)
    test(role)
    for s in role.tests:
        once(f"rederive/phaseB_seed{s}.npz", separate, "rederive/rd7_arrays.py", "--chain-smoke", "--phase", "B",
             "--seed", str(s))
    once("rederive/stats_test.json", separate, "rederive/rd7_stats.py", "--chain-smoke", "--stage", "test")
    once("test_verdict.json", finalize, role, "test")
    print("chain smoke: prep -> ceiling -> gate -> sensitivity -> build (check-only) -> test assertions passed; "
          "no metrics printed")


STAGES = {"prep": prep, "ceiling": ceiling, "gate": gate, "sensitivity": sensitivity, "smoke": rule_smoke,
          "build": build, "test": test}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=tuple(STAGES))
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--chain-smoke", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="use chain-smoke fixtures and dev seed 9001")
    args = parser.parse_args()
    R.assert_rule()
    R.assert_threads()
    kind = "rule" if args.stage == "smoke" else "chain" if args.smoke or args.chain_smoke else "real"
    role = R.roles(kind)
    if args.chain_smoke:
        assert args.stage is None and not args.finalize, "chain smoke takes no stage or finalize"
        chain_smoke(role)
    else:
        assert args.stage is not None, "a stage is required"
        if args.finalize:
            assert args.stage in CORE, "this stage has no verdict"
            finalize(role, args.stage)
        else:
            STAGES[args.stage](role)


if __name__ == "__main__":
    main()
