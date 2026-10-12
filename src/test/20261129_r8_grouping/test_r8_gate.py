"""Ticket 10: synthetic per-anchor arrays; no r8_common or real data needed."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r8_gate as G
import run_r8_gate as W


class BoundBoot:
    """Deterministic bootstrap boundary, returning fractions as the real API does."""
    def __init__(self, lower=None):
        self.lower = lower

    def __call__(self, values, clusters):
        point = float(np.mean(values))
        lower = self.lower if self.lower is not None else (0.001 if point > 0 else -0.001)
        return {"point": point, "ci95": [lower, 1.0]}


def world(spec, n_pair=24):
    """spec maps names to (three AFF-difference sums in quarter units, gain sum)."""
    n = 3 * n_pair
    arr = {"cl": np.arange(n) % 8, "pair_index": np.repeat([0, 1, 2], n_pair),
           "parity": np.arange(n) % 2, "AFF__r1": np.full(n, 0.5)}
    for k in ("B", "B0", "B1"):
        arr[f"{k}__r1"] = np.full(n, 0.25)
    for name, (sums, gain) in spec.items():
        s = np.full(n, 0.5)
        for p, units in enumerate(sums):
            assert abs(units) <= n_pair
            s[p * n_pair:p * n_pair + abs(units)] += 0.25 * np.sign(units)
        arr[f"{name}__r1"] = s
        arr[f"{name}__gain"] = np.zeros(n)
        arr[f"{name}__gain"][:abs(gain)] = 0.25 * np.sign(gain)
        arr[f"{name}_cf__r1"] = np.full(n, 0.25)
        arr[f"{name}_cf__gain"] = np.zeros(n)
        arr[f"BX_{name.split('_', 1)[1]}__r1"] = np.full(n, 0.25)
    return arr


@pytest.mark.parametrize("lower, passes", [(0.0, False), (-1e-12, False), (1e-12, True)])
def test_sc3_lower_bound_is_strict(lower, passes):
    arr = world({"R1_V100": ((1, 1, 1), 4)})
    rec = G.evaluate_scorer(arr, "R1_V100", boot=BoundBoot(lower))
    assert rec["SC3"]["sum4d"] == 3
    assert rec["SC3"]["point"] == pytest.approx(25 * 3 / 72)
    assert rec["SC3"]["pass"] is passes
    assert rec["pass"] is passes


@pytest.mark.parametrize("pair", [0, 1])
@pytest.mark.parametrize("units, passes", [(-49, True), (-50, False)])
def test_sc4_minus49_passes_minus50_fails_in_each_emotion_pair(pair, units, passes):
    sums = [0, 0, 200]
    sums[pair] = units
    arr = world({"R1_V100": (sums, 8)}, n_pair=4096)
    rec = G.evaluate_scorer(arr, "R1_V100", boot=BoundBoot())
    assert rec["SC4"][G.PAIRS[pair]]["sum4d"] == units
    assert rec["SC4"]["pass"] is passes
    assert rec["S3"]["pass"] is passes
    assert rec["pass"] is passes


def records(spec, lower=None):
    arr = world(spec)
    return {name: G.evaluate_scorer(arr, name, BoundBoot(lower)) for name in spec}


@pytest.mark.parametrize("track, earlier, later", [
    ("main", "V100", "V67"), ("main", "V67", "V33"), ("main", "V33", "NB100"),
    ("main", "NB100", "NB67"), ("main", "NB67", "NB33"), ("main", "NB33", "FC1"),
    *[("main", f"FC{i}", f"FC{i+1}") for i in range(1, 6)], ("main", "FC6", "NBC"),
    ("screen", "AL", "DI"), ("screen", "DI", "SG"),
])
def test_pick_group_order_precedes_kind_order(track, earlier, later):
    recs = records({f"R1_{later}": ((1, 1, 1), 4), f"XAFF_{earlier}": ((1, 1, 1), 4)})
    assert G.select_scorer(recs, track) == f"XAFF_{earlier}"


@pytest.mark.parametrize("track, grouping", [("main", "FC2"), ("screen", "DI")])
def test_pick_final_tie_prefers_r1(track, grouping):
    recs = records({f"XAFF_{grouping}": ((1, 1, 1), 4), f"R1_{grouping}": ((1, 1, 1), 4)})
    assert G.select_scorer(recs, track) == f"R1_{grouping}"


def test_pick_pooled_aff_margin_then_gain_and_only_passers():
    recs = records({"R1_V100": ((1, 1, 1), 20), "XAFF_FC6": ((2, 1, 1), 1),
                    "R1_NB33": ((-1, 0, 24), 40)})  # largest pooled margin, but fails SC4
    assert G.select_scorer(recs, "main") == "XAFF_FC6"
    recs = records({"R1_V100": ((1, 1, 1), 4), "XAFF_NBC": ((1, 1, 1), 5)})
    assert G.select_scorer(recs, "main") == "XAFF_NBC"
    assert G.select_scorer(records({"R1_V100": ((0, 0, -1), 4)}), "main") is None


def nb_records(spec=None):
    base = {f"{kind}_{x}": ((0, 0, 0), 4)
            for x in ("NB100", "NB67", "NB33") for kind in ("R1", "XAFF")}
    base.update(spec or {})
    return records(base, lower=0.0)


@pytest.mark.parametrize("spec, source, strength", [
    ({}, "R1_NB100", 1.0),
    ({"XAFF_NB33": ((2, 1, 1), 1)}, "XAFF_NB33", 0.33),
    ({"XAFF_NB67": ((0, 0, 0), 5)}, "XAFF_NB67", 0.67),
    ({"XAFF_NB67": ((1, 0, 0), 4), "R1_NB33": ((1, 0, 0), 4)}, "XAFF_NB67", 0.67),
    ({"XAFF_NB100": ((1, 0, 0), 4), "R1_NB67": ((1, 0, 0), 4)}, "XAFF_NB100", 1.0),
    ({"XAFF_NB33": ((1, 0, 0), 4), "R1_NB33": ((1, 0, 0), 4)}, "R1_NB33", 0.33),
])
def test_trigger_strength_uses_every_tie_break(spec, source, strength):
    out = G.probe_trigger(nb_records(spec))
    assert out["triggered"] is True
    assert out["source_scorer"] == source
    assert out["strength"] == strength
    assert len(out["nb_records"]) == 6


def test_trigger_suppressed_by_any_nb_pass_or_no_evaluated_nb():
    recs = nb_records()
    recs.update(records({"XAFF_NB33": ((1, 0, 0), 4)}))
    out = G.probe_trigger(recs)
    assert out["triggered"] is False and out["strength"] is None
    dropped = G.evaluate_all({}, tuple(recs), {x: "missing bank block" for x in ("NB100", "NB67", "NB33")})
    out = G.probe_trigger(dropped)
    assert out["triggered"] is False and out["strength"] is None
    assert "no NB scorer" in out["reason"]
    assert all(not r["pass"] and r["reason"] == "missing bank block" for r in dropped.values())


@pytest.mark.parametrize("track, grouping", [("main", "V100"), ("screen", "AL")])
def test_control_flag_uses_pick_kind_and_pick_stands(track, grouping):
    cands = records({f"XAFF_{grouping}": ((1, 1, 1), 4)})
    controls = records({f"R1_Z{grouping}": ((2, 2, 2), 5), f"XAFF_Z{grouping}": ((0, 0, 0), 4)})
    out = G.track_verdict(cands, track, controls=controls)
    assert out["pick"] == f"XAFF_{grouping}" and out["random_control_flag"] is False
    controls.update(records({f"XAFF_Z{grouping}": ((1, 1, 1), 4)}))
    out = G.track_verdict(cands, track, controls=controls)
    assert out["pick"] == f"XAFF_{grouping}" and out["random_control_flag"] is True
    controls = G.evaluate_all({}, tuple(controls), {f"Z{grouping}": "missing text head"})
    out = G.track_verdict(cands, track, controls=controls)
    assert out["random_control_flag"] == "not evaluable"
    with pytest.raises(AssertionError, match="control"):
        G.track_verdict(cands, track)


def test_promising_requires_all_r7_checks_and_no_main_passer():
    # No win over AFF, but B/CF are lower and the gain is positive: S1-S3 all pass.
    cands = records({"R1_V100": ((0, 0, 0), 4)})
    out = G.track_verdict(cands, "main")
    assert out["promising"] is True and out["pick"] is None
    assert out["head_target"] == {"grouping": "V100", "scorer_kind": "R1",
                                   "scorer": "R1_HV100", "source_scorer": "R1_V100"}
    cands.update(records({"XAFF_NB33": ((1, 1, 1), 4)}))
    out = G.track_verdict(cands, "main", controls=records({"XAFF_ZNB33": ((0, 0, 0), 4)}))
    assert out["promising"] is False and out["head_target"]["scorer"] == "XAFF_HNB33"
    assert out["main_line_fails_at_gate"] is False
    assert G.track_verdict(records({"R1_V100": ((0, 0, 0), 0)}), "main")["promising"] is False
    arr = world({"R1_V100": ((0, 0, 0), 4)})
    arr["R1_V100_cf__r1"][:] = 0.5
    arr["R1_V100_cf__r1"][48] = 0.75  # S2 fails even with positive pooled bar/gain bounds.
    cands = {"R1_V100": G.evaluate_scorer(arr, "R1_V100", BoundBoot())}
    assert G.track_verdict(cands, "main")["promising"] is False


def test_screen_has_no_promising_and_references_controls_pick_nothing():
    cands = records({"R1_AL": ((0, 0, 0), 4)})
    refs = records({"R1_R0own": ((2, 2, 2), 4)})
    out = G.track_verdict(cands, "screen", references=refs, controls=records({"R1_ZAL": ((3, 3, 3), 4)}))
    assert out["pick"] is None
    assert "promising" not in out and "head_target" not in out
    assert out["references"]["R1_R0own"]["pass"] is True
    assert out["controls"]["R1_ZAL"]["pass"] is True


def test_head_target_uses_pick_ties_when_no_pass_and_handles_all_dropped():
    cands = records({"XAFF_NBC": ((0, 0, 0), 5), "R1_V33": ((0, 0, 0), 4)})
    out = G.track_verdict(cands, "main")
    assert out["head_target"]["scorer"] == "XAFF_HNBC"
    dropped = G.evaluate_all({}, tuple(cands), {"NBC": "bank block", "V33": "view check"})
    out = G.track_verdict(dropped, "main")
    assert out["head_target"] is None and out["promising"] is False
    assert out["main_line_fails_at_gate"] is True


def test_sc3_sc4_gate_does_not_require_r7_s1_or_s2():
    arr = world({"R1_V100": ((1, 1, 1), 0)})
    arr["R1_V100_cf__r1"][48:] = 0.75
    rec = G.evaluate_scorer(arr, "R1_V100", BoundBoot())
    assert rec["S1"]["pass"] is False and rec["S2"]["pass"] is False
    assert rec["pass"] is True


def test_rederivation_other_input_sha_blocks_the_verdict_file():
    import json
    root = Path(__file__).resolve().parent / "results" / "smoke" / "ticket10"
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="c13_", dir=root))
    source = folder / "arrays_main.npz"
    source.write_bytes(b"synthetic per-anchor file")
    pending = folder / "gate_pending.json"
    pending.write_text(json.dumps({"step": "gate", "smoke": True,
                                   "input_sha256": {str(source): W.sha_file(source)}, "main": {}, "screen": {}}))
    check = folder / "gate_rederive.json"
    check.write_text(json.dumps({"step": "gate", "status": "agree", "smoke": True,
                                 "input_sha256": {str(source): "0" * 64, str(pending): W.sha_file(pending)}}))
    verdict = folder / "gate_verdict.json"
    def writer(path, rec):
        with open(path, "x") as f:
            json.dump(rec, f)
    with pytest.raises(AssertionError, match="SHA-256"):
        W.publish_checked(pending, check, verdict, writer)
    assert not verdict.exists()
    check.write_text(json.dumps({"step": "gate", "status": "agree", "smoke": True,
                                 "input_sha256": {str(source): W.sha_file(source), str(pending): W.sha_file(pending)}}))
    W.publish_checked(pending, check, verdict, writer)
    assert json.loads(verdict.read_text())["main"] == {}
    with pytest.raises(AssertionError, match="overwrite"):
        W.publish_checked(pending, check, verdict, writer)


@pytest.mark.parametrize("status, step, covered", [("disagree", "gate", True), ("agree", "trigger", True),
                                                  ("agree", "gate", False)])
def test_rederivation_requires_agreement_step_and_every_input(status, step, covered):
    expected = {"/stored/arrays.npz": "a" * 64}
    record = {"status": status, "step": step, "smoke": False,
              "input_sha256": expected if covered else {}}
    with pytest.raises(AssertionError):
        W.require_rederivation(record, expected, "gate", False)


def test_batches_union_requires_identical_anchor_alignment_and_shared_arrays():
    arr = world({"R1_V100": ((1, 1, 1), 4), "XAFF_NB33": ((2, 1, 1), 5)})
    left = {k: v for k, v in arr.items() if "NB33" not in k}
    right = {k: v for k, v in arr.items() if "V100" not in k}
    merged = G.merge_batches([left, right])
    assert G.evaluate_scorer(merged, "XAFF_NB33", BoundBoot())["SC3"]["sum4d"] == 4
    for key in ("cl", "pair_index", "parity", "AFF__r1"):
        changed = {k: v.copy() for k, v in right.items()}
        changed[key][0] += 1
        with pytest.raises(AssertionError, match=key):
            G.merge_batches([left, changed])


def test_full_gate_has_separate_roles_and_records_drops_without_common():
    dropped = {x: "not built in synthetic example" for x in (*G.MAIN[:-1], *G.SCREEN, "NRK", "TP")
               if x not in ("NB100", "AL")}
    spec = {f"{kind}_{x}": ((1, 1, 1), 4)
            for x in ("NB100", "AL", "R0own", "ZNB100", "ZAL") for kind in G.KINDS}
    out = G.gate_stats(world(spec), dropped=dropped, boot=BoundBoot())
    assert out["main"]["pick"] == "R1_NB100" and out["screen"]["pick"] == "R1_AL"
    assert out["trigger"]["triggered"] is False
    assert out["main"]["candidates"]["R1_FC3"]["reason"] == "not built in synthetic example"
    assert set(out["main"]["references"]) == {"R1_R0own", "XAFF_R0own", "R1_NRK", "XAFF_NRK", "R1_TP", "XAFF_TP"}
    assert set(out["screen"]["references"]) == {"R1_R0own", "XAFF_R0own"}
    assert out["fallback"]["screen_read_if_main_fails"] is True
    assert out["fallback"]["screen_eligible_at_gate"] is False


def test_real_cluster_bootstrap_on_small_constant_differences():
    import importlib.util
    spec = importlib.util.spec_from_file_location("r8_test_aspect_metrics", "/project/CoSiR/src/eval/aspect_metrics.py")
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    arr = world({"R1_V100": ((24, 24, 24), 72)})
    rec = G.evaluate_scorer(arr, "R1_V100", metrics.cluster_bootstrap)
    assert rec["SC3"]["ci95"] == [25.0, 25.0]
    assert rec["SC3"]["point"] == 25.0 and rec["pass"] is True
    assert rec["S1"]["ci95"] == [50.0, 50.0]
    assert rec["S1"]["gain_ci95"] == [25.0, 25.0]


@pytest.mark.parametrize("units, passes", [(245, False), (246, True)])
def test_recorded_r7_s1_integer_bar_boundary(units, passes):
    arr = world({"R1_V100": ((0, 0, 0), 4)}, n_pair=4096)
    arr["R1_V100__r1"][:] = 0.25
    arr["R1_V100__r1"][:units] = 0.5
    rec = G.evaluate_scorer(arr, "R1_V100", BoundBoot())
    assert rec["S1"]["sum4d"] == units and rec["S1"]["pass"] is passes


def test_promising_rejects_s3_failure_even_when_s1_s2_pass():
    recs = records({"R1_V100": ((-1, 0, 0), 4)})
    assert recs["R1_V100"]["S1"]["pass"] is True and recs["R1_V100"]["S2"]["pass"] is True
    assert G.track_verdict(recs, "main")["promising"] is False
