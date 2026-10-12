"""Synthetic checks of DECISION_RULE.md §7.1, §7.5–§7.7 and §11."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

import r7_chain as C


@pytest.mark.parametrize("alpha,boundaries", [
    (0.0125, (11, 14, 19, 30, 61)),
    (0.025, (24, 30, 40, 61, 124)),
])
@pytest.mark.parametrize("k", (1, 2, 3, 4, 5))
def test_holm_integer_boundary_at_each_rank(alpha, boundaries, k):
    names = ("B", "B0", "B1", "BX", "CF")
    for count, expected in ((boundaries[k - 1], True), (boundaries[k - 1] + 1, False)):
        counts = {name: 0 if rank < k else count for rank, name in enumerate(names, 1)}
        ranks = C.holm(counts, alpha)
        assert ranks[k - 1]["passes"] is expected
        assert all(r["passes"] for r in ranks[:k - 1])
        if not expected:
            assert not any(r["passes"] for r in ranks[k:])


def test_holm_ties_follow_rule_order_and_stop_later_checks():
    ranks = C.holm({"CF": 12, "BX": 12, "B1": 12, "B0": 12, "B": 12}, 0.0125)
    assert [r["name"] for r in ranks] == ["B", "B0", "B1", "BX", "CF"]
    assert [r["passes"] for r in ranks] == [False] * 5
    assert [r["own_count_passes"] for r in ranks] == [False, True, True, True, True]


def test_holm_orders_counts_before_tie_priority():
    ranks = C.holm({"B": 4, "B0": 2, "B1": 0, "BX": 2, "CF": 0}, 0.025)
    assert [r["name"] for r in ranks] == ["B1", "CF", "B0", "BX", "B"]
    assert all(r["passes"] for r in ranks)


@pytest.mark.parametrize("count", (-1, 5001, 1.5, True))
def test_holm_rejects_non_counts(count):
    with pytest.raises(AssertionError, match="count"):
        C.holm(dict.fromkeys(("B", "B0", "B1", "BX", "CF"), count), 0.025)


def source_function(path, name, **namespace):
    """Read the exact reference function, without importing a reused module or its loaders.

    Ticket 01's r7_common is not present on this independent branch. Compiling only
    the named pure function lets the copy check run without importing worktree code.
    """
    path = Path("/project/CoSiR") / path
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {"np": np, **namespace}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def reference_bootstrap():
    return source_function("src/eval/aspect_metrics.py", "cluster_bootstrap")


def test_draw_copy_matches_cluster_bootstrap_with_unequal_clusters():
    cl = np.array([101, 7, 101, 300, 7, 300, 7, 300, 300])
    values = np.array([0.25, -0.5, 0.75, 0, 0.5, -0.25, -0.25, 1, 0.25])
    draws, le0 = C.bootstrap_draws(values, cl)
    ref = reference_bootstrap()(values, cl)
    assert [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))] == ref["ci95"]
    assert draws.shape == le0.shape == (5000,)
    np.testing.assert_array_equal(le0, draws <= 0)


def test_all_checks_share_draws_and_keep_cluster_ids_across_seeds():
    cl = np.tile([91, 8, 91, 8, 22, 22], 3)
    values = np.tile([0.25, -0.5, 0.75, 0, 0.5, -0.25], 3)
    many, le0 = C.bootstrap_draws(np.column_stack([values, -values, 2 * values]), cl)
    one, _ = C.bootstrap_draws(values, cl)
    np.testing.assert_array_equal(many[:, 0], one)
    np.testing.assert_array_equal(many[:, 1], -one)
    np.testing.assert_array_equal(many[:, 2], 2 * one)
    np.testing.assert_array_equal(le0, many <= 0)


def test_zero_counts_use_integer_quarters_despite_float_noise():
    draws, le0 = C.bootstrap_draws(np.full(6, 1e-12), np.repeat([4, 9, 2], 2))
    assert np.all(draws > 0)
    assert int(le0.sum()) == 5000


@pytest.mark.parametrize("values,cl,match", [
    ([0.1, 0.25], [1, 2], "quarter"),
    ([25, 0], [1, 2], "fraction"),
    ([np.nan, 0], [1, 2], "finite"),
    ([0, 0], [1], "shape"),
    ([], [], "empty"),
    ([0, 0], [1, 1], "two clusters"),
])
def test_draws_assert_bad_boundaries(values, cl, match):
    with pytest.raises(AssertionError, match=match):
        C.bootstrap_draws(values, cl)


@pytest.mark.parametrize("alpha,stage,n,mult", [
    (0.0125, "F1", 12288, 3.649),
    (0.025, "F1", 12288, 3.417),
    (0.0125, "F2", 12288, 3.083),
    (0.025, "F2", 12288, 2.802),
    (0.0125, "F3", 4096, 3.083),
    (0.025, "F3", 4096, 2.802),
])
def test_sensitivity_copies_r3_decomposition_and_uses_chain_multiplier(alpha, stage, n, mult):
    d = np.resize([0, 0.25, -0.5, 0.5, -0.25, 0.75], n)
    cl = np.resize([100, 100, 200, 200, 300, 300], n)
    reference = source_function("src/test/20261121_round3_affect_gate/r3_stats.py", "sensitivity",
                                cluster_bootstrap=reference_bootstrap(), Z_HALF=1.96, K_DETECT=2.80)
    old = reference(d, cl)
    new = C.sensitivity(d, cl, alpha, stage)
    for key in ("SE", "half_width", "seed42_half_width", "sigma_a2", "sigma_e2", "n0", "n", "P"):
        assert new[key] == pytest.approx(old[key], abs=1e-12)
    assert new["mult"] == mult
    assert new["x"] == mult * new["SE"]
    assert new["n"] == n


def test_sensitivity_truncates_negative_painting_variance_to_zero():
    r = C.sensitivity([0, 0.5, 0, 0.5], [1, 1, 2, 2], 0.025, "F3", smoke=True)
    assert r["sigma_a2"] == 0
    assert r["sigma_e2"] == 1250
    assert r["SE"] == pytest.approx(np.sqrt(1250 / 12))


def test_sensitivity_asserts_scope_size_for_real_runs():
    with pytest.raises(AssertionError, match="12288"):
        C.sensitivity([0, 0.25, 0, 0.25], [1, 1, 2, 2], 0.025, "F1")


@pytest.mark.parametrize("alpha,boundary", [(0.0125, 61), (0.025, 124)])
def test_single_check_integer_boundary(alpha, boundary):
    assert C.single_pass(boundary, alpha)
    assert not C.single_pass(boundary + 1, alpha)


def synthetic_chain(fail=None):
    """Three paintings per pair; positive constant margins unless a stage fails."""
    cl = np.tile([10, 20, 30, 10, 20, 30], 3)
    pair_index = np.repeat([0, 1, 2], 6)
    diffs = {name: np.full(18, 0.25) for name in C.CHECKS}
    diffs["AFF"] = np.full(18, 0.25)
    if fail == "F1":
        diffs["B"][:] = 0
    elif fail == "F2":
        diffs["AFF"][:] = -0.25
    elif fail == "F3":
        diffs["B1"][pair_index != 2] = 0.75
        diffs["B1"][pair_index == 2] = -0.25
    sens = {"F1": {name: {"x": 0.75} for name in C.CHECKS}, "F2": {"x": 0.5}, "F3": {"x": 0.5}}
    return diffs, cl, pair_index, sens


@pytest.mark.parametrize("fail,stage,fv", [("F1", "none", "NO-GO"), ("F2", "F1", "GO"),
                                          ("F3", "F2", "GO"), (None, "F3", "GO")])
def test_fixed_sequence_reports_only_reached_stages(fail, stage, fv):
    diffs, cl, pi, sens = synthetic_chain(fail)
    record = C.chain_record(diffs, cl, pi, "R1_V1", 0.025, sens)
    assert record["last_stage_passed"] == stage
    assert C.friday_verdict({"csd": record}) == fv
    assert record["replaces_AFF"] is (stage in ("F2", "F3"))
    assert record["F1"]["checks"]["BX"]["sum4d"] == 18
    assert record["F1"]["checks"]["BX"]["point"] == 25
    assert record["F1"]["checks"]["BX"]["ci95"] == [25, 25]
    if stage == "none":
        assert record["F2"] == record["F3"] == "not tested"
        assert record["not_tested"] == {"F2": {"point": 25, "ci95": [25, 25]},
                                        "F3": {"point": 25, "ci95": [25, 25]}}
    elif stage == "F1":
        assert record["F2"]["pass"] is False
        assert record["F3"] == "not tested"
        assert record["not_tested"] == {"F3": {"point": 25, "ci95": [25, 25]}}
    else:
        assert record["F2"]["pass"] is True
        assert record["F3"]["pass"] is (stage == "F3")
        assert record["F3"]["sum4d"] == (-6 if fail == "F3" else 6)
    json.dumps(record, allow_nan=False)


def test_failed_positive_checks_and_holm_stops_have_c11_readings():
    diffs, cl, pi, sens = synthetic_chain()
    for name in C.CHECKS:
        diffs[name] = np.tile([-0.25, 0.25, 0.25], 6)
    record = C.chain_record(diffs, cl, pi, "R1_V1", 0.0125, sens)
    checks = record["F1"]["checks"]
    assert checks["B"]["status"] == "failed"
    assert checks["B"]["reading"] == "inconclusive at a detectable margin of 0.75"
    assert checks["B"]["half_width"] == 25
    for name in ("B0", "B1", "BX", "CF"):
        assert checks[name]["status"] == "not reached: Holm stopped at B"
        assert checks[name]["reading"] == "inconclusive at a detectable margin of 0.75"
    assert record["F1"]["passes"] == dict.fromkeys(C.CHECKS, False)


@pytest.mark.parametrize("diff", (-0.25, 0))
def test_nonpositive_failed_checks_name_the_pick_and_comparator(diff):
    diffs, cl, pi, sens = synthetic_chain()
    diffs["B"][:] = diff
    record = C.chain_record(diffs, cl, pi, "R1_V1", 0.025, sens)
    assert record["F1"]["checks"]["B"]["reading"] == "R1_V1 did not beat B on fresh episodes"


@pytest.mark.parametrize("alpha", (0.0125, 0.025))
def test_intervals_at_each_checks_own_level(alpha):
    diffs, cl, pi, sens = synthetic_chain()
    values = np.tile([-0.25, 0.25, 0.25], 6)
    diffs["B"] = values
    diffs["AFF"] = values
    record = C.chain_record(diffs, cl, pi, "R1_V1", alpha, sens)
    draws, _ = C.bootstrap_draws(values, cl)
    assert record["F1"]["holm_order"] == ["B0", "B1", "BX", "CF", "B"]
    # B is rank 5: its own one-sided level is alpha, interval coverage 1 - 2 alpha.
    assert record["F1"]["checks"]["B"]["ci_level"] == [
        100 * np.percentile(draws, 100 * alpha), 100 * np.percentile(draws, 100 * (1 - alpha))]
    assert record["F1"]["checks"]["B0"]["level_two_sided"] == 1 - 2 * alpha / 5


def test_friday_go_needs_any_f1_and_stops_force_no_go():
    passed = {"F1": {"pass": True}}
    failed = {"F1": {"pass": False}}
    assert C.friday_verdict({}) == "NO-GO"
    assert C.friday_verdict({"csd": failed, "generic": failed}) == "NO-GO"
    assert C.friday_verdict({"csd": failed, "generic": passed}) == "GO"
    assert C.friday_verdict({"csd": passed}, ceiling_zone="STOP") == "NO-GO"


def synthetic_arrays(pick="R1_V1", value=0.5):
    x = pick.split("_", 1)[1]
    return {"cl": np.tile([10, 20, 30, 10, 20, 30], 3),
            "pair_index": np.repeat([0, 1, 2], 6),
            **{f"{name}__r1": np.full(18, 0.25)
               for name in ("B", "B0", "B1", f"BX_{x}", f"{pick}_cf", "AFF")},
            f"{pick}__r1": np.full(18, value)}


def test_sensitivity_record_uses_each_comparator_and_f3_scope():
    arrays = synthetic_arrays()
    arrays["AFF__r1"][:] = 0.5
    sens = C.sensitivity_record(arrays, {"csd": "R1_V1", "generic": None}, smoke=True)
    chain = sens["chains"]["csd"]
    assert chain["pick"] == "R1_V1"
    assert chain["alpha"] == 0.025
    assert set(chain["F1"]) == set(C.CHECKS)
    assert all(r["n"] == 18 and r["mult"] == 3.417 for r in chain["F1"].values())
    assert chain["F2"]["n"] == 18 and chain["F2"]["x"] == 0
    assert chain["F3"]["n"] == 6 and chain["F3"]["mult"] == 2.802


def test_test_record_pools_in_seed_order_with_shared_painting_ids():
    arrays = {57: synthetic_arrays(value=0.75), 55: synthetic_arrays(value=0.25),
              56: synthetic_arrays(value=0.5)}
    picks = {"csd": "R1_V1", "generic": None}
    # Real-size synthetic inputs avoid a smoke bypass of the real seed-role guard.
    large = {seed: dict(record) for seed, record in arrays.items()}
    for seed, record in large.items():
        for key in record:
            if key == "pair_index":
                record[key] = np.repeat([0, 1, 2], 4096)
            elif key == "cl":
                record[key] = np.tile([10, 20, 30, 10], 3072)
            else:
                record[key] = np.full(12288, arrays[seed][key][0])
    sens = C.sensitivity_record(large[56], picks)
    result = C.test_record(large, picks, sens)
    chain = result["chains"]["csd"]
    expected = C.chain_record({name: np.repeat([0, 0.25, 0.5], 12288)
                               for name in C.CHECKS + ("AFF",)},
                              np.tile([10, 20, 30, 10], 9216), np.tile(np.repeat([0, 1, 2], 4096), 3),
                              "R1_V1", 0.025, sens["chains"]["csd"])
    assert chain == expected
    assert chain["F1"]["checks"]["B"]["sum4d"] == 36864
    assert chain["F1"]["checks"]["B"]["point"] == 25
    assert result["FV"] == "GO"
    json.dumps(result, allow_nan=False)


def test_two_picks_get_two_chains_at_half_level_on_smoke_seeds():
    arrays = synthetic_arrays()
    other = synthetic_arrays("XAFF_V5")
    arrays.update(other)
    picks = {"csd": "R1_V1", "generic": "XAFF_V5"}
    sens = C.sensitivity_record(arrays, picks, smoke=True)
    assert all(chain["alpha"] == 0.0125 for chain in sens["chains"].values())
    assert sens["chains"]["generic"]["F1"]["CF"]["mult"] == 3.649
    result = C.test_record({9003: arrays, 9001: arrays, 9002: arrays}, picks, sens, smoke=True)
    assert result["FV"] == "GO"
    assert all(chain["alpha"] == 0.0125 for chain in result["chains"].values())
    # Chain smoke uses development 9001 and fresh 9002, 9003 (index, Seed roles).
    assert C.test_record({9003: arrays, 9002: arrays}, picks, sens, smoke=True)["FV"] == "GO"


def test_no_picks_returns_no_go_without_reading_arrays():
    assert C.test_record({}, {"csd": None, "generic": None}, {}) == {"chains": {}, "FV": "NO-GO"}
    assert C.sensitivity_record({}, {}) == {"chains": {}}


@pytest.mark.parametrize("bad", ("seed42", "missing_seed", "shape", "pair", "key", "points", "sensitivity"))
def test_runner_boundary_assertions_reject_bad_inputs(bad):
    picks = {"csd": "R1_V1"}
    arrays = synthetic_arrays()
    sens = C.sensitivity_record(arrays, picks, smoke=True)
    per_seed = {9001: arrays, 9002: arrays, 9003: arrays}
    if bad == "seed42":
        per_seed[42] = per_seed.pop(9001)
    elif bad == "missing_seed":
        per_seed.pop(9002)
    elif bad == "shape":
        arrays["B__r1"] = np.zeros(17)
    elif bad == "pair":
        arrays["pair_index"][0] = 2
    elif bad == "key":
        arrays.pop("BX_V1__r1")
    elif bad == "points":
        arrays["R1_V1__r1"][:] = 50
    else:
        sens["chains"]["csd"]["alpha"] = 0.0125
    with pytest.raises(AssertionError):
        C.test_record(per_seed, picks, sens, smoke=True)


def test_secondary_intervals_use_their_own_level():
    diffs, cl, pi, sens = synthetic_chain()
    diffs["B1"][pi != 2] = 0.75
    diffs["B1"][pi == 2] = [-0.25, 0.25, 0.25, -0.25, 0.25, 0.25]
    record = C.chain_record(diffs, cl, pi, "R1_V1", 0.0125, sens)
    draws, _ = C.bootstrap_draws(diffs["B1"][pi == 2], cl[pi == 2])
    assert record["F3"]["ci_level"] == [100 * np.percentile(draws, 1.25), 100 * np.percentile(draws, 98.75)]
    assert record["F3"]["level_two_sided"] == 0.975


@pytest.mark.parametrize("alpha", (0.0125, 0.025))
def test_all_holm_ranks_report_their_own_intervals(alpha):
    diffs, cl, pi, sens = synthetic_chain()
    for name in C.CHECKS:
        diffs[name] = np.tile([-0.25, 0.25, 0.25], 6)
    result = C.chain_record(diffs, cl, pi, "R1_V1", alpha, sens)
    draws, _ = C.bootstrap_draws(diffs["B"], cl)
    for k, name in enumerate(("B", "B0", "B1", "BX", "CF"), 1):
        level = alpha / (6 - k)
        assert result["F1"]["checks"][name]["ci_level"] == pytest.approx([
            100 * np.percentile(draws, 100 * level), 100 * np.percentile(draws, 100 * (1 - level))])


def test_chain_record_accepts_sequence_values():
    diffs, cl, pi, sens = synthetic_chain()
    record = C.chain_record({name: values.tolist() for name, values in diffs.items()},
                            cl, pi, "R1_V1", 0.025, sens)
    assert record["last_stage_passed"] == "F3"


def test_runner_asserts_missing_sensitivity_stage():
    picks = {"csd": "R1_V1"}
    arrays = synthetic_arrays()
    sens = C.sensitivity_record(arrays, picks, smoke=True)
    del sens["chains"]["csd"]["F2"]
    with pytest.raises(AssertionError, match="sensitivity keys"):
        C.test_record({9001: arrays, 9002: arrays, 9003: arrays}, picks, sens, smoke=True)
