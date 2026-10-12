"""Tests of rederive/rd7_stats.py on synthetic per-anchor arrays with hand-computed answers (ticket 15).

Every integer boundary of rule L16, K1, S1 to S3, the Holm form (80 or 40 times (n+1) times (6-k) at most 5,001), every
tie rule, and the chain counts n = 0, 1, 2. The bootstrap loop is checked against a brute-force loop over the same
random stream and against `cluster_bootstrap`. Run from the round folder:
    python -m pytest -q -p no:cacheprovider --rootdir=. rederive/test_rd7_stats.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import r7_common as R  # noqa: E402
import rd7_stats as rd  # noqa: E402

AM = R.load("src.eval.aspect_metrics")

NP_ = 4096                       # episodes per pair
N = 3 * NP_
PAIR = np.repeat([0, 1, 2], NP_)  # emotion__style, emotion__genre, style__genre
CL = np.arange(N) // 41           # 300 painting clusters
GROUPS = ["V1", "V2", "V3", "V4", "V5", "V6"]
REFS = ["R0", "N"] + [f"Z{k}" for k in range(1, 7)]
KINDS = ["R1", "XAFF"]


def spread(c, lo=0, hi=N, shift=0):
    """c distinct, evenly spread positions in [lo, hi)."""
    return lo + (np.arange(c) * (hi - lo)) // c + shift


def blank():
    A = {"cl": CL.copy(), "pair_index": PAIR.copy(), "parity": np.arange(N) % 2}
    names = ["AFF", "B", "B0", "B1"] + [f"BX_{g}" for g in GROUPS + REFS]
    for g in GROUPS + REFS:
        for k in KINDS:
            names += [f"{k}_{g}", f"{k}_{g}_cf"]
    for nm in names:
        A[f"{nm}__r1"] = np.zeros(N)
        A[f"{nm}__gain"] = np.zeros(N)
    return A


def good(A, name, c=300, g=300, shift=0):
    """Scorer `name` has r1 0.25 at c spread positions (comparators stay 0) and gain 0.25 at g positions."""
    A[f"{name}__r1"][spread(c, shift=shift)] = 0.25
    A[f"{name}__gain"][spread(g, shift=shift)] = 0.25


def drop(A, g):
    for k in KINDS:
        for suffix in ("", "_cf"):
            for part in ("r1", "gain"):
                A.pop(f"{k}_{g}{suffix}__{part}")
    for part in ("r1", "gain"):
        A.pop(f"BX_{g}__{part}")


# ------------------------------------------------------------------ L16 helpers

def test_need_matches_rule_values():
    assert rd.need("0.5", 12288) == 246
    assert rd.need("0.5", 4096) == 82
    assert rd.need("1.0", 4096) == 164
    assert rd.need("0.3", 4096) == 50
    assert rd.need("0", 4096) == 0
    assert rd.need("-0.3", 4096) == -49


def test_point_is_25_sum_over_n():
    assert rd.point(164, 4096) == 25 * 164 / 4096
    assert rd.point(-49, 4096) == -25 * 49 / 4096


def test_sum4_integer_and_rejects_off_grid():
    assert rd.sum4(np.array([0.25, -0.5, 1.0, 0.0])) == 3
    assert isinstance(rd.sum4(np.array([0.25])), int)
    with pytest.raises(AssertionError):
        rd.sum4(np.array([0.1]))


# ------------------------------------------------------------------ K1

def ceiling_arrays(total, two_cluster=False):
    n = NP_
    fused = np.zeros(n)
    cf = np.zeros(n)
    fused[spread(total, hi=n)] = 0.25
    return fused, cf, np.arange(n) // 16


@pytest.mark.parametrize("total,zone", [(0, "STOP"), (81, "STOP"), (82, "WEAK"), (163, "WEAK"), (164, "CLEAR"),
                                         (400, "CLEAR")])
def test_k1_zone_boundaries(total, zone):
    fused, cf, cl = ceiling_arrays(total)
    out = rd.ceiling_stats(fused, cf, cl)
    assert out["sum4d"] == total and out["n"] == NP_
    assert out["zone"] == zone
    assert out["point"] == 25 * total / NP_
    want = AM.cluster_bootstrap(fused - cf, cl)["ci95"]
    assert out["ci95"] == [100 * want[0], 100 * want[1]]


def test_k1_lower_bound_exactly_zero_is_stop():
    # two clusters of equal size; the second has d = 0: a quarter of the resamples draw it twice, so the 2.5th
    # percentile is exactly 0 although the sum is far above 164
    fused = np.zeros(NP_)
    fused[spread(300, hi=NP_ // 2)] = 0.25
    cl = np.repeat([0, 1], NP_ // 2)
    out = rd.ceiling_stats(fused, np.zeros(NP_), cl)
    assert out["sum4d"] == 300 and out["ci95"][0] == 0.0
    assert out["zone"] == "STOP"


def test_k1_negative_lower_bound_with_big_sum_is_stop():
    fused = np.zeros(NP_)
    cf = np.zeros(NP_)
    fused[:500] = 0.25                 # cluster 0 : +125 in d units (sum4d +500)
    cf[NP_ // 2: NP_ // 2 + 100] = 0.25   # cluster 1 : sum4d -100
    cl = np.repeat([0, 1], NP_ // 2)
    out = rd.ceiling_stats(fused, cf, cl)
    assert out["sum4d"] == 400 and out["ci95"][0] < 0 and out["zone"] == "STOP"


def test_k1_negative_point():
    fused, cf = np.zeros(NP_), np.zeros(NP_)
    cf[spread(10, hi=NP_)] = 0.25
    out = rd.ceiling_stats(fused, cf, np.arange(NP_) // 16)
    assert out["sum4d"] == -10 and out["zone"] == "STOP"


def test_ceiling_scope_selection_and_size():
    r1 = np.zeros(N)
    cf = np.zeros(N)
    r1[NP_ * 2 + 5] = 0.25
    A = {"cl": CL, "pair_index": PAIR, "told_T__r1": r1, "told_T_cf__r1": cf}
    f, c, cl = rd.ceiling_scope(A, per_pair=NP_)
    assert len(f) == NP_ and f.sum() == 0.25 and len(cl) == NP_
    # a file that holds only the 4,096 style x genre episodes is also accepted
    sel = PAIR == 2
    B = {"cl": CL[sel], "pair_index": PAIR[sel], "told_T__r1": r1[sel], "told_T_cf__r1": cf[sel]}
    f2, _, _ = rd.ceiling_scope(B, per_pair=NP_)
    assert np.array_equal(f, f2)
    with pytest.raises(AssertionError):
        rd.ceiling_scope({k: v[:-1] for k, v in A.items()}, per_pair=NP_)


# ------------------------------------------------------------------ L15 bar comparator

def set_means(A, **counts):
    """Give comparators r1 = 0.25 at counts[name] positions (names B1, BX, CF, B0, B for scorer R1_V1)."""
    m = {"B1": "B1__r1", "BX": "BX_V1__r1", "CF": "R1_V1_cf__r1", "B0": "B0__r1", "B": "B__r1"}
    for k, c in counts.items():
        A[m[k]][spread(c, shift=7)] = 0.25
    A["BX_R0__r1"] = A["B1__r1"].copy()          # B1 is the reference R0's own B' (BX_R0)


@pytest.mark.parametrize("counts,want", [
    (dict(B1=5), "B1"), (dict(BX=5), "BX"), (dict(CF=5), "CF"), (dict(B0=5), "B0"), (dict(B=5), "B"),
    (dict(B1=5, BX=5), "B1"), (dict(BX=5, CF=5), "BX"), (dict(CF=5, B0=5, B=5), "CF"),
    (dict(B0=5, B=5), "B0"), (dict(B1=4, BX=5, CF=5, B0=5, B=5), "BX"), (dict(B1=5, B=6), "B"),
    (dict(), "B1"),
])
def test_bar_comparator_largest_mean_ties_to_earliest(counts, want):
    A = blank()
    set_means(A, **counts)
    out = rd.gate_stats(A)
    assert out["candidates"]["R1_V1"]["bar_comparator"] == want


def test_bar_margin_is_paired_against_the_bar_comparator():
    A = blank()
    A["R1_V1__r1"][spread(300)] = 0.25
    A["B0__r1"][spread(100, shift=3)] = 0.25
    out = rd.gate_stats(A)["candidates"]["R1_V1"]
    assert out["bar_comparator"] == "B0"
    assert out["S1"]["sum4d"] == 300 - 100


# ------------------------------------------------------------------ S1

@pytest.mark.parametrize("c,ok", [(245, False), (246, True), (300, True)])
def test_s1_sum_boundary(c, ok):
    A = blank()
    good(A, "R1_V1", c=c)
    s1 = rd.gate_stats(A)["candidates"]["R1_V1"]["S1"]
    assert s1["sum4d"] == c
    assert s1["pass"] is ok
    assert s1["ci95"][0] > 0 and s1["gain_ci95"][0] > 0


def test_s1_gain_lower_bound_not_above_zero_fails():
    A = blank()
    good(A, "R1_V1", c=300, g=0)       # gain statistic identically 0: lower bound 0
    s1 = rd.gate_stats(A)["candidates"]["R1_V1"]["S1"]
    assert s1["gain_sum4d"] == 0 and s1["gain_ci95"][0] == 0.0
    assert s1["pass"] is False


def test_s1_bar_lower_bound_not_above_zero_fails():
    A = blank()
    A["cl"] = np.repeat(np.arange(2), N // 2)       # two clusters: second has d = 0
    A["R1_V1__r1"][spread(400, hi=N // 2)] = 0.25
    A["R1_V1__gain"][spread(800)] = 0.25             # the gain bound is above 0, so only the bar bound can fail
    s1 = rd.gate_stats(A)["candidates"]["R1_V1"]["S1"]
    assert s1["sum4d"] == 400 and s1["ci95"][0] == 0.0 and s1["gain_ci95"][0] > 0 and s1["pass"] is False


def test_s1_gain_uses_cf_gain_zero_assertion():
    A = blank()
    good(A, "R1_V1")
    A["R1_V1_cf__gain"][3] = 0.25
    with pytest.raises(AssertionError):
        rd.gate_stats(A)


def test_s1_gain_sum_is_scorer_minus_cf_gain():
    A = blank()
    good(A, "R1_V1", c=300, g=123)
    assert rd.gate_stats(A)["candidates"]["R1_V1"]["S1"]["gain_sum4d"] == 123


# ------------------------------------------------------------------ S2, S3

def test_s2_boundary_zero():
    A = blank()
    good(A, "R1_V1", c=300)
    on = np.flatnonzero((PAIR == 2) & (A["R1_V1__r1"] == 0.25))
    A["R1_V1_cf__r1"][on] = 0.25                  # CF equals s wherever s is up: s - CF = 0 on style x genre
    out = rd.gate_stats(A)["candidates"]["R1_V1"]
    assert out["S2"] == {"sum4d": 0, "pass": True}
    off = np.flatnonzero((PAIR == 2) & (A["R1_V1__r1"] == 0))
    A["R1_V1_cf__r1"][off[:1]] = 0.25             # one episode where only CF is up: -1
    out = rd.gate_stats(A)["candidates"]["R1_V1"]
    assert out["S2"] == {"sum4d": -1, "pass": False} and out["pass"] is False


def test_s2_is_style_genre_only():
    A = blank()
    good(A, "R1_V1")
    off = np.flatnonzero((PAIR == 0) & (A["R1_V1__r1"] == 0))
    A["R1_V1_cf__r1"][off[:50]] = 0.25                      # CF better on emotion x style: S2 unaffected
    out = rd.gate_stats(A)["candidates"]["R1_V1"]["S2"]
    assert out["sum4d"] == int((A["R1_V1__r1"][PAIR == 2] * 4).round().sum()) and out["pass"] is True


@pytest.mark.parametrize("pair,pname", [(0, "emotion__style"), (1, "emotion__genre")])
def test_s3_boundary_minus_49_and_minus_50(pair, pname):
    A = blank()
    A["R1_V1__gain"][spread(300)] = 0.25
    # s is 0 on this emotion pair and up on style x genre, so s - AFF on the pair is exactly minus the AFF count
    A["R1_V1__r1"][spread(300, lo=2 * NP_)] = 0.25
    lo = pair * NP_
    A["AFF__r1"][lo: lo + 49] = 0.25
    out = rd.gate_stats(A)["candidates"]["R1_V1"]
    assert out["S3"][pname] == {"sum4d": -49, "pass": True} and out["S3"]["pass"] is True
    A["AFF__r1"][lo + 49] = 0.25
    out = rd.gate_stats(A)["candidates"]["R1_V1"]
    assert out["S3"][pname] == {"sum4d": -50, "pass": False}
    assert out["S3"]["pass"] is False and out["pass"] is False
    assert out["S1"]["pass"] is True and out["S2"]["pass"] is True


def test_pass_needs_all_three():
    A = blank()
    good(A, "R1_V1", c=300)
    assert rd.gate_stats(A)["candidates"]["R1_V1"]["pass"] is True
    B = blank()
    good(B, "R1_V1", c=245)
    assert rd.gate_stats(B)["candidates"]["R1_V1"]["pass"] is False


# ------------------------------------------------------------------ dropped, references, controls

def test_dropped_grouping_does_not_pass_and_has_the_dropped_form():
    A = blank()
    drop(A, "V3")
    good(A, "R1_V1")
    out = rd.gate_stats(A)
    want = {"bar_comparator": None, "S1": None, "S2": None, "S3": None, "dropped": True, "pass": False}
    assert out["candidates"]["R1_V3"] == want and out["candidates"]["XAFF_V3"] == want
    assert out["candidates"]["R1_V1"]["dropped"] is False
    assert set(out["candidates"]) == {f"{k}_{g}" for g in GROUPS for k in KINDS}
    assert set(out["references"]) == {f"{k}_{g}" for g in REFS for k in KINDS}


def test_partially_missing_scorer_is_an_error():
    A = blank()
    A.pop("XAFF_V2__r1")
    with pytest.raises(AssertionError):
        rd.gate_stats(A)


def test_references_get_the_same_quantities_and_pick_nothing():
    A = blank()
    good(A, "R1_R0")
    good(A, "XAFF_N")
    out = rd.gate_stats(A)
    assert out["references"]["R1_R0"]["pass"] is True
    assert out["references"]["XAFF_N"]["pass"] is True
    assert out["references"]["XAFF_R0"]["pass"] is False
    assert out["n"] == 0 and out["picks"] == {"csd": None, "generic": None}


# ------------------------------------------------------------------ P1, n, alpha

def test_n_zero_one_two_and_alpha():
    A = blank()
    out = rd.gate_stats(A)
    assert out["n"] == 0 and out["alpha"] == {}
    assert out["picks"] == {"csd": None, "generic": None}
    assert out["random_control_flag"] == {}
    good(A, "R1_V2")
    out = rd.gate_stats(A)
    assert out["n"] == 1 and out["alpha"] == {"csd": 0.025}
    assert out["picks"] == {"csd": "R1_V2", "generic": None}
    good(A, "XAFF_V5", c=290)
    out = rd.gate_stats(A)
    assert out["n"] == 2 and out["alpha"] == {"csd": 0.0125, "generic": 0.0125}
    assert out["picks"] == {"csd": "R1_V2", "generic": "XAFF_V5"}


def test_p1_largest_pooled_bar_margin_wins():
    A = blank()
    good(A, "R1_V1", c=300)
    good(A, "XAFF_V3", c=301)
    good(A, "R1_V4", c=299)
    assert rd.gate_stats(A)["picks"]["csd"] == "XAFF_V3"


def test_p1_candidates_in_other_family_do_not_compete():
    A = blank()
    good(A, "R1_V1", c=300)
    good(A, "R1_V5", c=900)
    out = rd.gate_stats(A)
    assert out["picks"] == {"csd": "R1_V1", "generic": "R1_V5"}


def test_p1_non_passing_larger_margin_is_ignored():
    A = blank()
    good(A, "R1_V1", c=300)
    good(A, "R1_V2", c=900, g=0)       # larger margin, fails the gain bound
    out = rd.gate_stats(A)
    assert out["candidates"]["R1_V2"]["pass"] is False
    assert out["picks"]["csd"] == "R1_V1"


def test_p1_tie_goes_to_larger_gain_sum():
    A = blank()
    good(A, "R1_V1", c=300, g=300)
    good(A, "R1_V2", c=300, g=301, shift=1)
    out = rd.gate_stats(A)
    assert out["candidates"]["R1_V1"]["S1"]["sum4d"] == out["candidates"]["R1_V2"]["S1"]["sum4d"]
    assert out["picks"]["csd"] == "R1_V2"


def test_p1_tie_on_both_goes_to_earlier_grouping_then_r1():
    A = blank()
    for nm in ("R1_V2", "XAFF_V2", "R1_V3", "XAFF_V3"):
        good(A, nm)
    assert rd.gate_stats(A)["picks"]["csd"] == "R1_V2"
    B = blank()
    for nm in ("XAFF_V2", "R1_V2", "R1_V3"):
        good(B, nm)
    assert rd.gate_stats(B)["picks"]["csd"] == "R1_V2"
    C = blank()
    good(C, "XAFF_V2")
    good(C, "R1_V3")
    assert rd.gate_stats(C)["picks"]["csd"] == "XAFF_V2"
    D = blank()
    good(D, "XAFF_V6")
    good(D, "R1_V5")
    assert rd.gate_stats(D)["picks"]["generic"] == "R1_V5"


def test_p1_tie_across_bar_comparators_uses_the_margin_sums():
    # same scorer r1 but different bar comparators: compared on their own bar-margin sums
    A = blank()
    good(A, "R1_V1", c=400)
    good(A, "R1_V2", c=400)
    A["BX_V1__r1"][spread(50, shift=2)] = 0.25      # V1's bar is BX_V1, so its margin is 350
    assert rd.gate_stats(A)["picks"]["csd"] == "R1_V2"


# ------------------------------------------------------------------ flag, promising

def test_random_control_flag_true_false_and_not_evaluable():
    A = blank()
    good(A, "R1_V1")
    good(A, "R1_Z1")
    out = rd.gate_stats(A)
    assert out["picks"]["csd"] == "R1_V1" and out["random_control_flag"] == {"csd": True}
    B = blank()
    good(B, "XAFF_V2")
    good(B, "R1_Z2")                         # the control passes with the other scorer kind only
    out = rd.gate_stats(B)
    assert out["picks"]["csd"] == "XAFF_V2" and out["random_control_flag"] == {"csd": False}
    C = blank()
    good(C, "XAFF_V2")
    drop(C, "Z2")
    out = rd.gate_stats(C)
    assert out["random_control_flag"] == {"csd": "not evaluable"}


def test_flag_generic_family_uses_its_own_control():
    A = blank()
    good(A, "R1_V6")
    good(A, "R1_Z6")
    out = rd.gate_stats(A)
    assert out["picks"] == {"csd": None, "generic": "R1_V6"}
    assert out["random_control_flag"] == {"generic": True}


def promising_arrays(sg_minus_aff, sg_minus_cf_ok=True, pooled=True):
    """Candidate R1_V1 with style x genre s - AFF = sg_minus_aff and a small pooled margin; no pass (S1 < 246)."""
    A = blank()
    sg = np.flatnonzero(PAIR == 2)
    A["R1_V1__r1"][sg[spread(sg_minus_aff, hi=len(sg))]] = 0.25
    A["R1_V1__gain"][sg[spread(100, hi=len(sg))]] = 0.25
    if not sg_minus_cf_ok:
        A["R1_V1_cf__r1"][sg[spread(sg_minus_aff + 5, hi=len(sg), shift=0)]] = 0.25
    return A


@pytest.mark.parametrize("m,want", [(49, False), (50, True), (60, True)])
def test_promising_threshold_50(m, want):
    out = rd.gate_stats(promising_arrays(m))
    assert out["n"] == 0
    assert out["promising"] is want


def test_promising_needs_cf_and_positive_pooled_margin():
    assert rd.gate_stats(promising_arrays(60, sg_minus_cf_ok=False))["promising"] is False
    A = promising_arrays(60)
    A["B0__r1"][:2000] = 0.25            # bar comparator B0: pooled margin 60 - 500 < 0
    assert rd.gate_stats(A)["promising"] is False
    B = promising_arrays(60)                         # pooled margin sum exactly 0 is not above 0
    B["B0__r1"][:60] = 0.25
    out = rd.gate_stats(B)["candidates"]["R1_V1"]
    assert out["bar_comparator"] == "B0" and out["S1"]["sum4d"] == 0
    assert rd.gate_stats(B)["promising"] is False
    C = promising_arrays(60)
    C["B0__r1"][:59] = 0.25                          # margin sum 1 > 0
    assert rd.gate_stats(C)["promising"] is True


def test_promising_matched_control_boundary_is_at_or_above():
    A = promising_arrays(60)
    A["R1_V1__r1"][:10] = 0.25                           # s is also up on 10 emotion x style episodes
    sg = PAIR == 2
    A["R1_V1_cf__r1"][sg] = A["R1_V1__r1"][sg]           # s - CF = 0 on style x genre: "at or above" holds
    out = rd.gate_stats(A)["candidates"]["R1_V1"]
    assert out["S2"]["sum4d"] == 0 and out["S1"]["sum4d"] == 10 and out["bar_comparator"] == "CF"
    assert rd.gate_stats(A)["promising"] is True


def test_promising_false_when_a_scorer_passes():
    A = promising_arrays(60)
    good(A, "R1_V2")
    out = rd.gate_stats(A)
    assert out["n"] == 1 and out["promising"] is False


def test_promising_ignores_dropped_and_references():
    A = blank()
    sg = np.flatnonzero(PAIR == 2)
    A["R1_R0__r1"][sg[spread(80, hi=len(sg))]] = 0.25
    assert rd.gate_stats(A)["promising"] is False


# ------------------------------------------------------------------ bootstrap loop

def brute(values, cl, n_boot=5000, chunk=250, seed=42):
    ids = np.unique(cl)
    k = len(ids)
    members = [np.flatnonzero(cl == i) for i in ids]
    q = np.rint(4 * np.asarray(values)).astype(np.int64)
    rng = np.random.default_rng(seed)
    n_le0 = 0
    means = []
    for s in range(0, n_boot, chunk):
        draws = rng.integers(0, k, size=(min(chunk, n_boot - s), k))
        for row in draws:
            idx = np.concatenate([members[j] for j in row])
            if q[idx].sum() <= 0:
                n_le0 += 1
            means.append(values[idx].mean())
    return n_le0, np.array(means)


def test_draw_loop_matches_brute_force_and_cluster_bootstrap():
    rng = np.random.default_rng(3)
    cl = np.repeat(np.arange(12), [3, 4, 2, 5, 3, 3, 4, 2, 6, 3, 4, 3])
    d = rng.integers(-1, 3, size=len(cl)) * 0.25
    e = rng.integers(-2, 2, size=len(cl)) * 0.25
    res = rd.bootstrap_checks({"d": d, "e": e}, cl)
    for name, v in (("d", d), ("e", e)):
        n_le0, means = brute(v, cl)
        assert res[name]["n_j"] == n_le0
        assert np.allclose(res[name]["b"], means, rtol=0, atol=1e-12)
        want = AM.cluster_bootstrap(v, cl)["ci95"]
        assert list(np.percentile(res[name]["b"], [2.5, 97.5])) == want


def test_n_j_extremes_all_positive_and_all_zero():
    cl = np.arange(60) % 10
    res = rd.bootstrap_checks({"pos": np.full(60, 0.25), "zero": np.zeros(60), "neg": np.full(60, -0.25)}, cl)
    assert res["pos"]["n_j"] == 0
    assert res["zero"]["n_j"] == 5000 and res["neg"]["n_j"] == 5000


def test_draw_loop_integer_decision_not_float():
    # a cluster sum of exactly 0 in integers (d = +0.25 and -0.25 cancel) counts as <= 0
    cl = np.repeat(np.arange(4), 2)
    d = np.array([0.25, -0.25] * 4)
    assert rd.bootstrap_checks({"d": d}, cl)["d"]["n_j"] == 5000


def test_single_check_k_less_than_two_clusters_is_an_error():
    with pytest.raises(Exception):
        rd.bootstrap_checks({"d": np.zeros(4)}, np.zeros(4, dtype=int))


# ------------------------------------------------------------------ Holm and single-check integers

NAMES = ["B", "B0", "B1", "BX", "CF"]
LAST = {80: {1: 11, 2: 14, 3: 19, 4: 30, 5: 61}, 40: {1: 24, 2: 30, 3: 40, 4: 61, 5: 124}}


def test_boundaries_from_the_formula():
    for c, per_k in LAST.items():
        for k, nmax in per_k.items():
            assert c * (nmax + 1) * (6 - k) <= 5001 < c * (nmax + 2) * (6 - k)


@pytest.mark.parametrize("c", [80, 40])
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_holm_kth_boundary(c, k):
    nmax = LAST[c][k]
    for n_k, ok in ((nmax, True), (nmax + 1, False)):
        ns = {nm: 0 for nm in NAMES}
        order_names = NAMES[:]                  # check k (1-based) is NAMES[k-1]; earlier ones get n = 0
        for j in range(k, 5):
            ns[order_names[j]] = 5000
        ns[order_names[k - 1]] = n_k
        order, passes = rd.holm(ns, c)
        assert order == NAMES
        assert passes[order_names[k - 1]] is ok
        # later checks never pass after a failure
        if not ok:
            assert all(passes[order_names[j]] is False for j in range(k - 1, 5))
        assert all(passes[order_names[j]] is True for j in range(k - 1))


def test_holm_sorts_by_n_and_breaks_ties_in_name_order():
    ns = {"B": 7, "B0": 3, "B1": 7, "BX": 3, "CF": 0}
    order, _ = rd.holm(ns, 80)
    assert order == ["CF", "B0", "BX", "B", "B1"]


def test_holm_a_failure_stops_everything_after_it():
    ns = {"B": 0, "B0": 12, "B1": 0, "BX": 0, "CF": 0}     # B0 sorts last-ish but k=5 allows n up to 61
    order, passes = rd.holm(ns, 80)
    assert order == ["B", "B1", "BX", "CF", "B0"] and all(passes.values())
    ns = {"B": 0, "B0": 62, "B1": 0, "BX": 0, "CF": 0}
    order, passes = rd.holm(ns, 80)
    assert passes["B0"] is False and passes["B"] and passes["CF"]
    ns = {"B": 12, "B0": 12, "B1": 12, "BX": 12, "CF": 12}   # k = 1 needs n <= 11 at c = 80
    assert not any(rd.holm(ns, 80)[1].values())
    assert all(rd.holm(ns, 40)[1].values())                  # c = 40: 40 * 13 * 5 = 2600 <= 5001, and so on


@pytest.mark.parametrize("c,nmax", [(80, 61), (40, 124)])
def test_single_check_boundary(c, nmax):
    assert rd.single_pass(nmax, c) is True
    assert rd.single_pass(nmax + 1, c) is False
    assert rd.single_pass(0, c) is True and rd.single_pass(5000, c) is False


def test_c_from_alpha():
    assert rd.c_of(0.0125) == 80 and rd.c_of(0.025) == 40
    with pytest.raises(AssertionError):
        rd.c_of(0.03)


# ------------------------------------------------------------------ chains

SEEDS = (55, 56, 57)
NP2 = 64
NN = 3 * NP2


def fresh(seed_idx, s_up=0.25, aff=0.0, b0=0.0, b1_sg=0.0, bx=0.0, cf=0.0, b=0.0, pick="R1_V1"):
    """One fresh-seed arrays dict: scorer r1 = s_up everywhere, comparators constant (multiples of 0.25)."""
    pair = np.repeat([0, 1, 2], NP2)
    cl = (np.arange(NN) * 7 + seed_idx * 3) % 50
    A = {"cl": cl, "pair_index": pair, "parity": np.arange(NN) % 2}
    X = pick.split("_", 1)[1]
    cols = {pick: s_up, f"{pick}_cf": cf, "AFF": aff, "B": b, "B0": b0, "B1": np.where(pair == 2, b1_sg, 0.0),
            f"BX_{X}": bx}
    for nm, v in cols.items():
        A[f"{nm}__r1"] = np.zeros(NN) + v
    return A


def chain_inputs(**kw):
    return [fresh(i, **kw) for i in range(3)]


def test_chain_all_pass_f1_f2_f3():
    out = rd.chain_stats(chain_inputs(), "R1_V1", 0.0125)
    assert out["pick"] == "R1_V1" and out["alpha"] == 0.0125
    f1 = out["F1"]
    assert f1["pass"] is True and f1["holm_order"] == NAMES
    assert f1["passes"] == {nm: True for nm in NAMES}
    for nm in NAMES:
        chk = f1["checks"][nm]
        assert chk["n_j"] == 0 and chk["sum4d"] == NN * 3 and chk["point"] == 25.0
        assert set(chk) == {"n_j", "sum4d", "point", "ci95", "ci_level"}
    assert out["F2"]["pass"] is True and out["F2"]["n_j"] == 0
    assert out["F3"]["pass"] is True and out["F3"]["sum4d"] == 3 * NP2
    assert set(out["F2"]) == {"n_j", "sum4d", "point", "ci95", "ci_level", "pass"}


def test_chain_f1_fails_f2_f3_not_tested():
    out = rd.chain_stats(chain_inputs(b0=0.25), "R1_V1", 0.025)
    f1 = out["F1"]
    assert f1["checks"]["B0"]["n_j"] == 5000 and f1["checks"]["B0"]["point"] == 0.0
    assert f1["holm_order"] == ["B", "B1", "BX", "CF", "B0"]
    assert f1["passes"]["B0"] is False and f1["passes"]["B"] is True and f1["pass"] is False
    assert out["F2"]["status"] == "not tested" and set(out["F2"]) == {"status", "point", "ci95"}
    assert out["F3"]["status"] == "not tested" and set(out["F3"]) == {"status", "point", "ci95"}
    # the unreached check still carries its numbers: s - AFF = 25 points, s - B1 on style x genre = 25 points
    assert out["F2"]["point"] == 25.0 and out["F3"]["point"] == 25.0


def test_chain_cf_check_uses_the_matched_counterpart():
    out = rd.chain_stats(chain_inputs(cf=0.25), "R1_V1", 0.0125)
    assert out["F1"]["checks"]["CF"]["n_j"] == 5000 and out["F1"]["holm_order"][-1] == "CF"
    assert out["F1"]["pass"] is False and out["F1"]["passes"]["B"] is True


def test_chain_f2_fails_f3_not_tested():
    out = rd.chain_stats(chain_inputs(aff=0.25), "R1_V1", 0.0125)
    assert out["F1"]["pass"] is True
    assert out["F2"]["pass"] is False and out["F2"]["n_j"] == 5000
    assert out["F3"]["status"] == "not tested" and out["F3"]["point"] == 25.0


def test_chain_f3_fails_after_f2():
    # B1 equals the scorer on style x genre only: F1 still passes (the other pairs carry B1 check), F3 fails
    A = chain_inputs(b1_sg=0.25)
    out = rd.chain_stats(A, "R1_V1", 0.0125)
    assert out["F1"]["pass"] is True and out["F2"]["pass"] is True
    assert out["F3"]["pass"] is False and out["F3"]["n_j"] == 5000 and out["F3"]["sum4d"] == 0


def test_chain_intervals_match_cluster_bootstrap_and_levels():
    rng = np.random.default_rng(5)
    A = chain_inputs()
    for a in A:                               # make the scorer noisy so the intervals are not degenerate
        a["R1_V1__r1"] = rng.integers(0, 3, size=NN) * 0.25
        a["B0__r1"] = rng.integers(0, 2, size=NN) * 0.25
    out = rd.chain_stats(A, "R1_V1", 0.025)
    cl = np.concatenate([a["cl"] for a in A])
    s = np.concatenate([a["R1_V1__r1"] for a in A])
    b0 = np.concatenate([a["B0__r1"] for a in A])
    want = AM.cluster_bootstrap(s - b0, cl)["ci95"]
    got = out["F1"]["checks"]["B0"]["ci95"]
    assert got == [100 * want[0], 100 * want[1]]
    assert out["F1"]["checks"]["B0"]["point"] == 25 * int(round(4 * (s - b0).sum())) / len(s)
    # the interval at the check's own level: k-th in Holm order, one-sided alpha / (6 - k)
    k = out["F1"]["holm_order"].index("B0") + 1
    a_k = 0.025 / (6 - k)
    bb = rd.bootstrap_checks({"x": s - b0}, cl)["x"]["b"]
    assert out["F1"]["checks"]["B0"]["ci_level"] == [100 * float(np.percentile(bb, 100 * a_k)),
                                                     100 * float(np.percentile(bb, 100 * (1 - a_k)))]
    assert out["F2"]["ci_level"][0] == 100 * float(np.percentile(
        rd.bootstrap_checks({"x": s - np.concatenate([a["AFF__r1"] for a in A])}, cl)["x"]["b"], 100 * 0.025))


def test_chain_pools_seeds_in_order():
    A = chain_inputs()
    # a different painting id layout per seed: the pooled cluster count is the union of ids
    out1 = rd.chain_stats(A, "R1_V1", 0.0125)
    out2 = rd.chain_stats(A[::-1], "R1_V1", 0.0125)       # same values, other pooling order
    assert out1["F1"]["checks"]["B"]["sum4d"] == out2["F1"]["checks"]["B"]["sum4d"]


def test_chain_uses_pick_grouping_bx_and_xaff_name():
    A = chain_inputs(pick="XAFF_V5", bx=0.25)
    out = rd.chain_stats(A, "XAFF_V5", 0.025)
    assert out["F1"]["checks"]["BX"]["n_j"] == 5000 and out["F1"]["checks"]["B"]["n_j"] == 0
    assert out["F1"]["pass"] is False


# ------------------------------------------------------------------ whole test stage

def gate_rec(picks, alpha):
    return {"picks": picks, "alpha": alpha}


def test_test_stats_fv_go_and_no_go():
    out = rd.test_stats(chain_inputs(), {"csd": "R1_V1", "generic": None}, {"csd": 0.025})
    assert set(out["chains"]) == {"csd"} and out["FV"] == "GO"
    out = rd.test_stats(chain_inputs(b0=0.25), {"csd": "R1_V1", "generic": None}, {"csd": 0.025})
    assert out["FV"] == "NO-GO"


def test_test_stats_two_chains_go_if_either_f1_passes():
    A = [dict(fresh(i), **{"R1_V5__r1": np.zeros(NN), "R1_V5_cf__r1": np.zeros(NN), "BX_V5__r1": np.zeros(NN)})
         for i in range(3)]
    out = rd.test_stats(A, {"csd": "R1_V1", "generic": "R1_V5"}, {"csd": 0.0125, "generic": 0.0125})
    assert out["chains"]["csd"]["F1"]["pass"] is True
    assert out["chains"]["generic"]["F1"]["pass"] is False
    assert out["FV"] == "GO"


def test_test_stats_without_pick_is_an_error():
    with pytest.raises(AssertionError):
        rd.test_stats(chain_inputs(), {"csd": None, "generic": None}, {})


# ------------------------------------------------------------------ stage runner (files)

def save(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def test_run_ceiling_and_gate_and_test_files(tmp_path):
    A = blank()
    good(A, "R1_V1")
    save(tmp_path / "seed42" / "arrays.npz", A)
    ceil_arrays = {"cl": CL, "pair_index": PAIR, "told_T__r1": np.where(PAIR == 2, 0.25, 0.0),
                   "told_T_cf__r1": np.zeros(N)}
    save(tmp_path / "seed42" / "arrays_ceiling.npz", ceil_arrays)
    rd.run_stage("ceiling", tmp_path, dev=42, tests=SEEDS, per_pair=NP_)
    c = json.loads((tmp_path / "rederive" / "stats_ceiling.json").read_text())
    assert c["zone"] == "CLEAR" and c["n"] == NP_ and c["sum4d"] == NP_
    rd.run_stage("gate", tmp_path, dev=42, tests=SEEDS, per_pair=NP_)
    g = json.loads((tmp_path / "rederive" / "stats_gate.json").read_text())
    assert g["n"] == 1 and g["picks"]["csd"] == "R1_V1" and g["alpha"] == {"csd": 0.025}
    # the pipeline's gate_verdict.json supplies picks and alpha for the test re-derivation
    (tmp_path / "gate_verdict.json").write_text(json.dumps(
        {"picks": {"csd": "R1_V1", "generic": None}, "alpha": {"csd": 0.025}, "n": 1}))
    for i, s in enumerate(SEEDS):
        save(tmp_path / f"seed{s}" / "arrays.npz", fresh(i))
    rd.run_stage("test", tmp_path, dev=42, tests=SEEDS, per_pair=NP2)
    t = json.loads((tmp_path / "rederive" / "stats_test.json").read_text())
    assert t["FV"] == "GO" and t["chains"]["csd"]["F1"]["pass"] is True
    # write-once
    with pytest.raises(SystemExit):
        rd.run_stage("test", tmp_path, dev=42, tests=SEEDS, per_pair=NP2)


def test_run_test_stage_refuses_alpha_that_does_not_match_the_picks(tmp_path):
    for i, s in enumerate(SEEDS):
        save(tmp_path / f"seed{s}" / "arrays.npz", fresh(i))
    (tmp_path / "gate_verdict.json").write_text(json.dumps(
        {"picks": {"csd": "R1_V1", "generic": None}, "alpha": {"csd": 0.0125}, "n": 1}))
    with pytest.raises(AssertionError):
        rd.run_stage("test", tmp_path, dev=42, tests=SEEDS, per_pair=NP2)
    (tmp_path / "gate_verdict.json").write_text(json.dumps(
        {"picks": {"csd": "R1_V1", "generic": None}, "alpha": {"csd": 0.025, "generic": 0.025}, "n": 1}))
    with pytest.raises(AssertionError):
        rd.run_stage("test", tmp_path, dev=42, tests=SEEDS, per_pair=NP2)


def test_overlay_must_agree_where_both_files_have_a_key(tmp_path):
    A = blank()
    good(A, "R1_V1")
    save(tmp_path / "seed42" / "arrays.npz", A)
    B = {"R1_V1__r1": A["R1_V1__r1"].copy()}
    save(tmp_path / "rederive" / "phaseA_seed42.npz", B)
    out = rd.load_arrays(tmp_path / "seed42" / "arrays.npz", tmp_path / "rederive" / "phaseA_seed42.npz")
    assert np.array_equal(out["R1_V1__r1"], A["R1_V1__r1"]) and "B1__r1" in out
    B["R1_V1__r1"][0] += 0.25
    save(tmp_path / "rederive" / "phaseA_bad.npz", B)
    with pytest.raises(AssertionError):
        rd.load_arrays(tmp_path / "seed42" / "arrays.npz", tmp_path / "rederive" / "phaseA_bad.npz")


def test_name_normalisation():
    assert rd.norm_name("R1[V1]") == "R1_V1" and rd.norm_name("X-AFF[V3]") == "XAFF_V3"
    assert rd.norm_name("XAFF_V6") == "XAFF_V6" and rd.norm_name("R1_V2") == "R1_V2"


# ------------------------------------------------------------------ --overlay: required keys per phase (A2)

def is_ref(key):
    return rd.is_reference_key(key)


def gate_files(tmp_path, drop_v6=False):
    """A pipeline gate file with every array, and a phase A file holding all but the reference scorers' arrays."""
    A = blank()
    good(A, "R1_V1")
    if drop_v6:
        drop(A, "V6")
    save(tmp_path / "seed42" / "arrays.npz", A)
    P = {k: v for k, v in A.items() if not is_ref(k)}
    save(tmp_path / "rederive" / "phaseA_seed42.npz", P)
    return A, P


def test_overlay_gate_takes_only_reference_keys_from_the_pipeline_file(tmp_path):
    A, P = gate_files(tmp_path)
    assert not any(is_ref(k) for k in P) and any(is_ref(k) for k in A)
    got = rd.run_stage("gate", tmp_path, dev=42, tests=SEEDS, per_pair=NP_, overlay=True)
    assert got["picks"]["csd"] == "R1_V1" and not got["references"]["R1_R0"]["dropped"]
    loaded = rd._arrays(tmp_path, "seed42/arrays.npz", "rederive/phaseA_seed42.npz", True, "gate")
    assert "R1_Z3_cf__gain" in loaded and "BX_N__r1" in loaded and set(P) <= set(loaded)
    # a pipeline-only key that is not a reference scorer's is not passed on to the statistics
    A["B__other"] = np.zeros(N)
    save(tmp_path / "seed42" / "arrays.npz", A)
    loaded = rd._arrays(tmp_path, "seed42/arrays.npz", "rederive/phaseA_seed42.npz", True, "gate")
    assert "B__other" not in loaded and "R1_Z3_cf__gain" in loaded


@pytest.mark.parametrize("key", ["R1_V1__gain", "XAFF_V3_cf__gain", "R1_V2_cf__r1", "BX_V4__r1", "B1__r1", "B__r1",
                                 "AFF__r1", "parity", "cl", "pair_index"])
def test_overlay_gate_refuses_a_phase_file_missing_a_required_key(tmp_path, key):
    A, P = gate_files(tmp_path)
    del P[key]
    save(tmp_path / "rederive" / "phaseA_seed42.npz", P)
    with pytest.raises(AssertionError, match=key):
        rd._arrays(tmp_path, "seed42/arrays.npz", "rederive/phaseA_seed42.npz", True, "gate")
    # the pipeline's file holds the key, so a silent merge would have run
    assert key in A


def test_overlay_gate_does_not_require_a_grouping_both_files_lack(tmp_path):
    gate_files(tmp_path, drop_v6=True)
    got = rd.run_stage("gate", tmp_path, dev=42, tests=SEEDS, per_pair=NP_, overlay=True)
    assert got["candidates"]["R1_V6"]["dropped"] and got["candidates"]["XAFF_V6"]["dropped"]


def test_overlay_gate_refuses_a_grouping_the_phase_file_lacks_but_the_pipeline_has(tmp_path):
    A, P = gate_files(tmp_path)
    save(tmp_path / "rederive" / "phaseA_seed42.npz", {k: v for k, v in P.items() if "_V6" not in k})
    with pytest.raises(AssertionError, match="V6"):
        rd._arrays(tmp_path, "seed42/arrays.npz", "rederive/phaseA_seed42.npz", True, "gate")


def test_overlay_ceiling_and_test_take_nothing_from_the_pipeline_file(tmp_path):
    full = {"cl": CL, "pair_index": PAIR, "parity": np.arange(N) % 2, "told_T__r1": np.zeros(N),
            "told_T_cf__r1": np.zeros(N)}
    save(tmp_path / "seed42" / "arrays_ceiling.npz", full)
    short = {k: v for k, v in full.items() if k != "told_T_cf__r1"}
    save(tmp_path / "rederive" / "phaseK.npz", short)
    with pytest.raises(AssertionError, match="told_T_cf__r1"):
        rd._arrays(tmp_path, "seed42/arrays_ceiling.npz", "rederive/phaseK.npz", True, "ceiling")
    save(tmp_path / "rederive" / "phaseK.npz", full)
    assert set(rd._arrays(tmp_path, "seed42/arrays_ceiling.npz", "rederive/phaseK.npz", True, "ceiling")) == set(full)
    F = fresh(0)
    F.update({k: np.zeros(NN) for k in ("B__r1", "B0__r1", "B1__r1", "AFF__r1", "R1_V1_cf__r1")})
    save(tmp_path / "seed55" / "arrays.npz", F)
    for key in ("R1_V1__r1", "R1_V1_cf__r1", "BX_V1__r1", "B0__r1", "parity"):
        P = {k: v for k, v in F.items() if k != key}
        save(tmp_path / "rederive" / "phaseB_seed55.npz", P)
        with pytest.raises(AssertionError, match=key):
            rd._arrays(tmp_path, "seed55/arrays.npz", "rederive/phaseB_seed55.npz", True, "test", ["R1_V1"])
    save(tmp_path / "rederive" / "phaseB_seed55.npz", F)
    assert rd._arrays(tmp_path, "seed55/arrays.npz", "rederive/phaseB_seed55.npz", True, "test", ["R1_V1"])
