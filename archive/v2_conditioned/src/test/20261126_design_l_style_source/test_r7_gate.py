"""Tests of r7_gate (ticket 09; rule L15, L16, §5 K1, §6 S1 to S3, P1, chains, flags). Synthetic arrays only.

The world has 12,288 pooled episodes, 4,096 per pair (pair_index 0 emotion__style, 1 emotion__genre, 2 style__genre),
quarter-valued R@1. Every quantity the rule thresholds is set in exact quarter units, so each integer boundary is hit
on both sides. The bootstrap is replaced by a recorder (`FakeBoot`) except in the last tests, which run the real one.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r7_gate as G  # noqa: E402

N, NP = 12288, 4096


def gs(*a, **k):
    k.setdefault("require_controls", False)
    return G.gate_stats(*a, **k)


# ------------------------------------------------------------------------------------------------------- helpers
class FakeBoot:
    """Stands in for `cluster_bootstrap`: lower bound = +0.1 (mean > 0) or -0.1 (mean <= 0) in fractions, unless the
    queue `lowers` (in call order, fractions) gives one. Records each call's values and clusters."""

    def __init__(self, lowers=()):
        self.lowers = list(lowers)
        self.calls = []

    def __call__(self, values, clusters):
        values = np.asarray(values)
        self.calls.append((values.copy(), np.asarray(clusters).copy()))
        m = float(values.mean())
        lo = self.lowers.pop(0) if self.lowers else (0.001 if m > 0 else -0.001)
        return {"point": m, "ci95": [lo, lo + 0.002], "n_clusters": int(len(np.unique(clusters)))}


def spread(units, lo, hi, n=N):
    """Vector of n zeros with |units| entries of 0.25 * sign(units) at positions lo, lo+1, ... (< hi)."""
    assert abs(units) <= hi - lo
    v = np.zeros(n)
    v[lo:lo + abs(units)] = 0.25 * np.sign(units)
    return v


def world(spec, aff_extra=None):
    """arrays dict for scorers `spec` = {name: dict(bar, gain, cf, e, h, sg)}: name -> sums in quarter units of
    s-B1 (pooled), s gain (pooled), s-CF (style x genre), s-AFF (emotion x style, emotion x genre, style x genre)."""
    rng = np.random.default_rng(0)
    arr = {"pair_index": np.repeat([0, 1, 2], NP), "cl": rng.integers(0, 60, N)}
    aff = rng.integers(0, 5, N) / 4
    arr["AFF__r1"] = aff
    arr["AFF__gain"] = np.zeros(N)
    for name, sp in spec.items():
        s = aff + spread(sp.get("e", 0), 0, NP) + spread(sp.get("h", 0), NP, 2 * NP) + spread(sp.get("sg", 0), 2 * NP, N)
        arr[f"{name}__r1"] = s
        arr[f"{name}__gain"] = spread(sp.get("gain", 10), 0, N)
        cf = s - 0.75                       # far below B1 on emotion pairs, so B1 stays the bar comparator
        cf[2 * NP:] = s[2 * NP:] - spread(sp.get("cf", 0), 2 * NP, N)[2 * NP:]
        arr[f"{name}_cf__r1"] = cf
        arr[f"{name}_cf__gain"] = np.zeros(N)
        b1 = s - spread(sp.get("bar", 300), 0, N)
        x = name.split("_", 1)[1]
        arr[f"BX_{x}__r1"] = b1 - 0.25
        arr.setdefault("B1__r1", b1)        # the first scorer fixes B1 (bar margins of later ones are checked apart)
        arr.setdefault("B0__r1", b1 - 0.25)
        arr.setdefault("B__r1", b1 - 0.25)
    return arr


def good(**kw):
    return dict(dict(bar=300, gain=10, cf=0, e=0, h=0, sg=0), **kw)


def run(spec, cands=None, refs=(), dropped=(), boot=None, arr=None):
    arr = arr if arr is not None else world(spec)
    cands = list(spec) if cands is None else cands
    return gs(arr, candidates=cands, references=refs, dropped=dropped, boot=boot or FakeBoot(),
                        require_controls=False)


# ------------------------------------------------------------------------------------------- L16: integer form
def test_need_matches_rule_numbers():
    assert G.need(0.5, 12288) == 246
    assert G.need(0.5, 4096) == 82
    assert G.need(1.0, 4096) == 164
    assert G.need(0.3, 4096) == 50
    assert G.need(0, 4096) == 0
    assert G.need(-0.3, 4096) == -49


def test_need_is_exact_at_integer_quotients():
    assert G.need(0.5, 50) == 1          # 0.5 * 50 / 25 = 1 exactly, not 2
    assert G.need(0.25, 100) == 1
    assert G.need(-0.5, 50) == -1
    assert G.need(0.3, 50) == 1          # 0.6 -> 1
    assert G.need(1, 25) == 1


def test_sum4_integer_and_multiples_check():
    assert G.sum4(np.array([0.25, 0.25, -0.5, 1.0])) == 4 * 1.0
    assert isinstance(G.sum4(np.array([0.25])), int)
    assert G.sum4(np.array([0.25 + 5e-10])) == 1       # float noise below 1e-9 is allowed
    with pytest.raises(AssertionError):
        G.sum4(np.array([0.25, 0.3]))
    with pytest.raises(AssertionError):
        G.sum4(np.array([0.125]))


def test_point_is_25_sum_over_n():
    d = spread(246, 0, N)
    assert G.point_of(246, N) == pytest.approx(25 * 246 / N)
    assert G.point_of(G.sum4(d), N) == pytest.approx(100 * d.mean())


def test_interval_scales_to_points_and_uses_scope_clusters():
    fb = FakeBoot(lowers=[0.004])
    cl = np.arange(N) % 7
    d = spread(100, 0, N)
    pt, ci = G.interval(d, cl, fb)
    assert pt == pytest.approx(25 * 100 / N)
    assert ci == pytest.approx([0.4, 0.6])
    assert np.array_equal(fb.calls[0][1], cl)


# ----------------------------------------------------------------------------------------------------- K1 zones
@pytest.mark.parametrize("sum4d, zone", [(81, "STOP"), (82, "WEAK"), (163, "WEAK"), (164, "CLEAR"), (5000, "CLEAR"),
                                         (0, "STOP"), (-5, "STOP")])
def test_k1_integer_boundaries(sum4d, zone):
    assert G.k1_zone(sum4d, 0.1, NP) == zone


@pytest.mark.parametrize("sum4d", [82, 163, 164, 1000])
def test_k1_lower_bound_exactly_zero_or_below_stops(sum4d):
    assert G.k1_zone(sum4d, 0.0, NP) == "STOP"
    assert G.k1_zone(sum4d, -1e-12, NP) == "STOP"
    assert G.k1_zone(sum4d, 1e-12, NP) != "STOP"


def test_ceiling_stats_end_to_end():
    cl = np.arange(NP) % 50
    cf = np.random.default_rng(1).integers(0, 5, NP) / 4
    for units, zone in [(81, "STOP"), (82, "WEAK"), (164, "CLEAR")]:
        out = G.ceiling_stats(cf + spread(units, 0, NP, NP), cf, cl, boot=FakeBoot())
        assert out["sum4d"] == units and out["n"] == NP and out["zone"] == zone
        assert out["point"] == pytest.approx(25 * units / NP)
        assert len(out["ci95"]) == 2
    out = G.ceiling_stats(cf + spread(300, 0, NP, NP), cf, cl, boot=FakeBoot(lowers=[0.0]))
    assert out["zone"] == "STOP"                          # sum large, lower bound exactly 0


def test_ceiling_needs_matching_lengths():
    with pytest.raises(AssertionError):
        G.ceiling_stats(np.zeros(8), np.zeros(9), np.zeros(8, int), boot=FakeBoot())


# --------------------------------------------------------------------------------------------- L15 comparator
def test_bar_comparator_largest_mean_wins():
    r = {"B1": np.full(8, 0.25), "BX": np.full(8, 0.5), "CF": np.full(8, 0.25), "B0": np.full(8, 0.25),
         "B": np.full(8, 0.75)}
    assert G.bar_comparator(r) == "B"


def test_bar_comparator_ties_to_earliest_in_order():
    base = np.full(8, 0.5)
    r = {k: base.copy() for k in G.BAR_ORDER}
    assert G.BAR_ORDER == ("B1", "BX", "CF", "B0", "B")
    assert G.bar_comparator(r) == "B1"
    r["B1"] = base - 0.25
    assert G.bar_comparator(r) == "BX"
    r["BX"] = base - 0.25
    assert G.bar_comparator(r) == "CF"
    r["CF"] = base - 0.25
    assert G.bar_comparator(r) == "B0"
    r["B0"] = base - 0.25
    assert G.bar_comparator(r) == "B"
    r["B"] = base - 0.25
    assert G.bar_comparator(r) == "B1"                    # all equal again


def test_bar_comparator_tie_is_exact_not_by_float_noise():
    a = np.array([0.25, 0.5, 0.75, 1.0] * 1000)
    r = {"B1": a, "BX": a[::-1].copy(), "CF": a - 0.25, "B0": a - 0.25, "B": a - 0.25}
    assert G.bar_comparator(r) == "B1"                    # same multiset: same sum, tie to the earlier


def test_bar_comparator_a_one_quarter_gap_decides():
    a = np.zeros(N)
    r = {k: a.copy() for k in G.BAR_ORDER}
    r["B"][5] = 0.25                                      # one extra quarter anywhere: B is the strict maximum
    assert G.bar_comparator(r) == "B"


def test_bar_comparator_over_scope_not_pooled():
    # B1 higher on the first half, BX on the second: the pooled and the scoped answers differ
    r = {k: np.zeros(8) for k in G.BAR_ORDER}
    r["B1"][:4] = 1.0
    r["BX"][4:] = 0.75
    assert G.bar_comparator(r) == "B1"
    assert G.bar_comparator({k: v[4:] for k, v in r.items()}) == "BX"


# --------------------------------------------------------------------------------------------------- S1 bounds
@pytest.mark.parametrize("bar, ok", [(244, False), (245, False), (246, True), (247, True)])
def test_s1_integer_boundary_245_246(bar, ok):
    out = run({"R1_V1": good(bar=bar)})["candidates"]["R1_V1"]
    assert out["S1"]["sum4d"] == bar
    assert out["bar_comparator"] == "B1"
    assert out["S1"]["pass"] is ok and out["pass"] is ok


def test_s1_bar_lower_bound_exactly_zero_fails():
    out = run({"R1_V1": good(bar=500)}, boot=FakeBoot(lowers=[0.0, 0.01]))["candidates"]["R1_V1"]
    assert out["S1"]["ci95"][0] == 0.0 and out["S1"]["pass"] is False
    out = run({"R1_V1": good(bar=500)}, boot=FakeBoot(lowers=[1e-9, 0.01]))["candidates"]["R1_V1"]
    assert out["S1"]["pass"] is True


def test_s1_gain_lower_bound_exactly_zero_fails():
    out = run({"R1_V1": good(bar=500)}, boot=FakeBoot(lowers=[0.01, 0.0]))["candidates"]["R1_V1"]
    assert out["S1"]["gain_ci95"][0] == 0.0 and out["S1"]["pass"] is False
    assert out["S1"]["gain_sum4d"] == 10


def test_s1_fields_and_units():
    out = run({"R1_V1": good(bar=400, gain=20)}, boot=FakeBoot(lowers=[0.004, 0.002]))["candidates"]["R1_V1"]
    s1 = out["S1"]
    assert s1["sum4d"] == 400 and isinstance(s1["sum4d"], int)
    assert s1["gain_sum4d"] == 20 and isinstance(s1["gain_sum4d"], int)
    assert s1["ci95"] == pytest.approx([0.4, 0.6])        # fractions x 100 = R@1 points
    assert s1["gain_ci95"] == pytest.approx([0.2, 0.4])


def test_s1_bootstrap_runs_on_the_pooled_episodes_and_all_clusters():
    arr = world({"R1_V1": good()})
    fb = FakeBoot()
    gs(arr, candidates=["R1_V1"], references=(), boot=fb)
    assert len(fb.calls[0][0]) == N and np.array_equal(fb.calls[0][1], arr["cl"])


def test_gain_statistic_requires_cf_gain_zero():
    arr = world({"R1_V1": good()})
    arr["R1_V1_cf__gain"][3] = 0.25
    with pytest.raises(AssertionError, match="gain"):
        gs(arr, candidates=["R1_V1"], references=(), boot=FakeBoot())


def test_bar_margin_uses_the_other_comparator_when_it_is_larger():
    arr = world({"R1_V1": good(bar=300)})
    arr["BX_V1__r1"] = arr["R1_V1__r1"] - spread(250, 0, N)    # BX mean now above B1's: bar margin = s - BX
    out = gs(arr, candidates=["R1_V1"], references=(), boot=FakeBoot())["candidates"]["R1_V1"]
    assert out["bar_comparator"] == "BX" and out["S1"]["sum4d"] == 250 and out["S1"]["pass"] is True


def test_bar_margin_is_paired_per_anchor_not_a_difference_of_means():
    arr = world({"R1_V1": good(bar=300)})
    out = gs(arr, candidates=["R1_V1"], references=(), boot=FakeBoot())["candidates"]["R1_V1"]
    d = arr["R1_V1__r1"] - arr["B1__r1"]
    assert out["S1"]["sum4d"] == int(np.rint(4 * d).sum()) == 300


# ---------------------------------------------------------------------------------------------------- S2 and S3
@pytest.mark.parametrize("cf, ok", [(-1, False), (0, True), (1, True), (-50, False)])
def test_s2_boundary_minus1_0(cf, ok):
    out = run({"R1_V1": good(cf=cf)})["candidates"]["R1_V1"]
    assert out["S2"]["sum4d"] == cf and out["S2"]["pass"] is ok and out["pass"] is ok


@pytest.mark.parametrize("units, ok", [(-50, False), (-49, True), (-48, True), (0, True), (-1000, False)])
def test_s3_boundary_minus50_minus49_each_pair(units, ok):
    for key, pair in (("e", "emotion__style"), ("h", "emotion__genre")):
        out = run({"R1_V1": good(**{key: units})})["candidates"]["R1_V1"]
        assert out["S3"][pair]["sum4d"] == units and out["S3"][pair]["pass"] is ok
        other = "emotion__genre" if key == "e" else "emotion__style"
        assert out["S3"][other] == {"sum4d": 0, "pass": True}
        assert out["S3"]["pass"] is ok and out["pass"] is ok


def test_s3_style_genre_vs_aff_does_not_enter_s3():
    out = run({"R1_V1": good(sg=-500)})["candidates"]["R1_V1"]
    assert out["pass"] is True


def test_pass_needs_all_three():
    assert run({"R1_V1": good()})["candidates"]["R1_V1"]["pass"] is True
    assert run({"R1_V1": good(bar=245)})["candidates"]["R1_V1"]["pass"] is False
    assert run({"R1_V1": good(cf=-1)})["candidates"]["R1_V1"]["pass"] is False
    assert run({"R1_V1": good(e=-50)})["candidates"]["R1_V1"]["pass"] is False


# --------------------------------------------------------------------------------------------- dropped grouping
def test_dropped_grouping_does_not_pass_or_pick():
    spec = {"R1_V1": good(bar=900), "XAFF_V1": good(bar=900), "R1_V2": good(bar=300)}
    arr = world(spec)
    del arr["R1_V1__r1"], arr["R1_V1_cf__r1"]                   # a dropped grouping has no arrays at all
    del arr["XAFF_V1__r1"], arr["XAFF_V1_cf__r1"]
    out = gs(arr, candidates=list(spec), references=(), dropped=["V1"], boot=FakeBoot())
    for name in ("R1_V1", "XAFF_V1"):
        assert out["candidates"][name]["dropped"] is True and out["candidates"][name]["pass"] is False
    assert out["candidates"]["R1_V2"]["dropped"] is False
    assert out["picks"]["csd"] == "R1_V2"


def test_dropped_even_with_good_arrays_does_not_pass():
    spec = {"R1_V1": good(bar=900)}
    out = gs(world(spec), candidates=["R1_V1"], references=(), dropped=["V1"], boot=FakeBoot())
    assert out["candidates"]["R1_V1"]["pass"] is False and out["picks"]["csd"] is None


def test_missing_arrays_of_a_scorer_that_is_not_dropped_assert():
    arr = world({"R1_V1": good()})
    with pytest.raises(AssertionError, match="XAFF_V1"):
        gs(arr, candidates=["R1_V1", "XAFF_V1"], references=(), boot=FakeBoot())


# ------------------------------------------------------------------------------------------------------ P1 pick
def test_pick_largest_bar_margin_among_passers_per_family():
    spec = {"R1_V1": good(bar=300), "XAFF_V2": good(bar=400), "R1_V3": good(bar=900, cf=-1),    # V3 fails S2
            "R1_V5": good(bar=260), "XAFF_V6": good(bar=270)}
    out = _pick(spec)
    assert out["picks"] == {"csd": "XAFF_V2", "generic": "XAFF_V6"}
    assert out["n"] == 2 and out["alpha"] == {"csd": 0.0125, "generic": 0.0125}


def _own_bars(spec, refs=(), dropped=()):
    """world() where each scorer has its own bar margin. BX_x is shared by R1_x and XAFF_x (it belongs to the
    grouping): the first scorer of x fixes BX_x = s - bar; a later one has its s (and its CF, so S2 is unchanged)
    shifted on the style x genre episodes to give it its own margin. BX is the bar comparator: B1, B0 and B are
    pushed 1.0 below it."""
    arr = world(spec)
    first = {}
    for name, sp in spec.items():
        x = name.split("_", 1)[1]
        bar = sp.get("bar", 300)
        if x not in first:
            first[x] = bar
            arr[f"BX_{x}__r1"] = arr[f"{name}__r1"] - spread(bar, 0, N)
        else:
            shift = spread(bar - first[x], 2 * NP, N)
            arr[f"{name}__r1"] = arr[f"{name}__r1"] + shift
            arr[f"{name}_cf__r1"] = arr[f"{name}_cf__r1"] + shift
    low = np.min([arr[f"BX_{x}__r1"] for x in first], axis=0) - 1.0
    for k in ("B1__r1", "B0__r1", "B__r1"):
        arr[k] = low.copy()
    return (arr,), dict(candidates=[c for c in spec if c not in refs], references=refs, dropped=dropped,
                        boot=FakeBoot(), require_controls=bool(refs))


def _pick(spec, **kw):
    args, kwargs = _own_bars(spec, **kw)
    return gs(*args, **kwargs)


def test_own_bars_helper_gives_each_scorer_its_bar_margin():
    out = _pick({"R1_V1": good(bar=300), "R1_V2": good(bar=777)})
    assert out["candidates"]["R1_V1"]["S1"]["sum4d"] == 300 and out["candidates"]["R1_V2"]["S1"]["sum4d"] == 777
    assert out["candidates"]["R1_V1"]["bar_comparator"] == "BX"


def test_pick_families_csd_v1_to_v4_generic_v5_v6():
    assert G.FAMILIES == {"csd": ("V1", "V2", "V3", "V4"), "generic": ("V5", "V6")}
    out = _pick({"R1_V4": good(bar=300), "R1_V5": good(bar=250)})
    assert out["picks"] == {"csd": "R1_V4", "generic": "R1_V5"}


def test_pick_tie_on_bar_goes_to_larger_gain_sum():
    out = _pick({"R1_V1": good(bar=300, gain=10), "XAFF_V2": good(bar=300, gain=11)})
    assert out["picks"]["csd"] == "XAFF_V2"
    out = _pick({"R1_V1": good(bar=300, gain=11), "XAFF_V2": good(bar=300, gain=10)})
    assert out["picks"]["csd"] == "R1_V1"


def test_pick_tie_on_bar_and_gain_goes_to_earlier_variant():
    out = _pick({"R1_V3": good(), "R1_V2": good(), "R1_V4": good()})
    assert out["picks"]["csd"] == "R1_V2"
    out = _pick({"R1_V6": good(), "R1_V5": good()})
    assert out["picks"]["generic"] == "R1_V5"


def test_pick_tie_variant_order_comes_before_kind_order():
    out = _pick({"R1_V2": good(), "XAFF_V1": good()})
    assert out["picks"]["csd"] == "XAFF_V1"


def test_pick_tie_on_all_goes_to_r1_before_xaff():
    out = _pick({"XAFF_V3": good(), "R1_V3": good()})
    assert out["picks"]["csd"] == "R1_V3"


def test_pick_larger_bar_beats_larger_gain():
    out = _pick({"R1_V1": good(bar=301, gain=1), "XAFF_V1": good(bar=300, gain=200)})
    assert out["picks"]["csd"] == "R1_V1"


def test_bar_margin_compared_on_integer_sum_not_on_interval():
    out = _pick({"R1_V1": good(bar=300), "R1_V2": good(bar=301)})
    assert out["picks"]["csd"] == "R1_V2"


@pytest.mark.parametrize("passers, n, alpha", [
    ({}, 0, {}),
    ({"R1_V1": good()}, 1, {"csd": 0.025}),
    ({"R1_V5": good()}, 1, {"generic": 0.025}),
    ({"R1_V1": good(), "R1_V6": good()}, 2, {"csd": 0.0125, "generic": 0.0125}),
])
def test_n_and_alpha(passers, n, alpha):
    spec = dict(passers) or {"R1_V1": good(bar=0)}
    out = _pick(spec)
    assert out["n"] == n and out["alpha"] == alpha
    picked = {f: p for f, p in out["picks"].items() if p}
    assert len(picked) == n and set(out["picks"]) == {"csd", "generic"}
    if n == 0:
        assert out["picks"] == {"csd": None, "generic": None}


def test_failing_scorer_is_not_picked():
    out = _pick({"R1_V1": good(bar=245), "R1_V2": good(cf=-1), "R1_V3": good(e=-50)})
    assert out["picks"]["csd"] is None and out["n"] == 0


# ------------------------------------------------------------------------------------- references and controls
def test_references_get_the_same_quantities_but_pick_nothing():
    spec = {"R1_V1": good(bar=300), "R1_R0": good(bar=900), "R1_Z1": good(bar=901), "R1_N": good(bar=902)}
    out = _pick(spec, refs=["R1_R0", "R1_Z1", "R1_N"])
    assert set(out["candidates"]) == {"R1_V1"}
    assert set(out["references"]) == {"R1_R0", "R1_Z1", "R1_N"}
    assert out["references"]["R1_R0"]["S1"]["sum4d"] == 900 and out["references"]["R1_R0"]["pass"] is True
    assert out["picks"]["csd"] == "R1_V1" and out["picks"]["generic"] is None


# ----------------------------------------------------------------------------------------- random-control flag
def test_random_control_flag_true_false_and_not_evaluable():
    spec = {"R1_V1": good(), "XAFF_V5": good(), "R1_Z1": good(), "XAFF_Z5": good(cf=-1), "R1_Z5": good()}
    out = _pick(spec, refs=["R1_Z1", "XAFF_Z5", "R1_Z5"])
    assert out["picks"] == {"csd": "R1_V1", "generic": "XAFF_V5"}
    assert out["random_control_flag"] == {"csd": True, "generic": False}      # XAFF_Z5 fails S2; R1_Z5 is the wrong kind
    out = _pick(spec, refs=["R1_Z1", "XAFF_Z5", "R1_Z5"], dropped=["Z1"])
    assert out["random_control_flag"]["csd"] == "not evaluable"
    assert out["random_control_flag"]["generic"] is False


def test_random_control_flag_uses_the_picks_scorer_kind():
    spec = {"XAFF_V2": good(), "R1_Z2": good(), "XAFF_Z2": good(cf=-1)}
    out = _pick(spec, refs=["R1_Z2", "XAFF_Z2"])
    assert out["picks"]["csd"] == "XAFF_V2" and out["random_control_flag"]["csd"] is False


def test_random_control_flag_only_for_picked_families():
    out = _pick({"R1_V1": good(), "R1_Z1": good()}, refs=["R1_Z1"])
    assert out["random_control_flag"] == {"csd": True}
    out = _pick({"R1_V1": good(bar=0)})
    assert out["random_control_flag"] == {}


def test_missing_control_of_a_pick_asserts():
    with pytest.raises(AssertionError, match="R1_Z1"):
        G.gate_stats(world({"R1_V1": good()}), candidates=["R1_V1"], references=(), boot=FakeBoot())


def test_dropped_variant_with_its_control_still_reported():
    out = _pick({"R1_V1": good(), "R1_Z1": good()}, refs=["R1_Z1"], dropped=["V1"])
    assert out["picks"]["csd"] is None and out["random_control_flag"] == {}


# ------------------------------------------------------------------------------------------------- "promising"
@pytest.mark.parametrize("sg, cf, bar, expect", [
    (50, 0, 1, True), (49, 0, 1, False), (50, -1, 1, False), (50, 0, 0, False), (500, 3, 300, False),
])
def test_promising_boundaries(sg, cf, bar, expect):
    # a candidate that fails S1 may still be promising: bar > 0 but below 246 is enough; a passer (bar 300) makes
    # n = 1, and "promising" is stated only at n = 0 (rule §6)
    out = _pick({"R1_V1": good(sg=sg, cf=cf, bar=bar)})
    assert out["promising"] is expect


def test_promising_true_even_when_nothing_passes_and_n_zero():
    out = _pick({"R1_V1": good(sg=60, bar=100)})
    assert out["n"] == 0 and out["promising"] is True


def test_promising_ignores_references_and_dropped():
    out = _pick({"R1_V1": good(sg=60, bar=100), "R1_R0": good(sg=60, bar=100)}, refs=["R1_R0"], dropped=["V1"])
    assert out["promising"] is False
    out = _pick({"R1_V1": good(sg=0, bar=100), "R1_R0": good(sg=60, bar=100)}, refs=["R1_R0"])
    assert out["promising"] is False


# ------------------------------------------------------------------------------------------- shape and asserts
def test_scope_checks_on_pair_index():
    arr = world({"R1_V1": good()})
    arr["pair_index"] = arr["pair_index"].copy()
    arr["pair_index"][0] = 2
    with pytest.raises(AssertionError):
        gs(arr, candidates=["R1_V1"], references=(), boot=FakeBoot())


def test_off_grid_value_asserts():
    arr = world({"R1_V1": good()})
    arr["R1_V1__r1"] = arr["R1_V1__r1"] + 0.1
    with pytest.raises(AssertionError):
        gs(arr, candidates=["R1_V1"], references=(), boot=FakeBoot())


def test_scorer_names_parse():
    assert G.split_name("R1_V3") == ("R1", "V3")
    assert G.split_name("XAFF_Z6") == ("XAFF", "Z6")
    assert G.split_name("R1_R0") == ("R1", "R0")
    assert G.CANDIDATES == tuple(f"{k}_V{i}" for i in range(1, 7) for k in ("R1", "XAFF"))
    assert len(G.CANDIDATES) == 12


# ------------------------------------------------------------------------------- the real bootstrap, small data
def test_default_bootstrap_is_cluster_bootstrap_from_main():
    pytest.importorskip("numpy")
    rng = np.random.default_rng(5)
    cl = rng.integers(0, 30, 600)
    d = rng.integers(-2, 3, 600) / 4
    pt, ci = G.interval(d, cl)
    from importlib import util
    spec = util.spec_from_file_location("am", "/project/CoSiR/src/eval/aspect_metrics.py")
    am = util.module_from_spec(spec)
    spec.loader.exec_module(am)
    ref = am.cluster_bootstrap(d, cl)
    assert pt == pytest.approx(100 * ref["point"], abs=1e-12)
    assert ci == pytest.approx([100 * ref["ci95"][0], 100 * ref["ci95"][1]], abs=1e-12)


def test_ceiling_with_the_real_bootstrap_smoke():
    rng = np.random.default_rng(6)
    cl = rng.integers(0, 80, NP)
    cf = rng.integers(0, 5, NP) / 4
    f = cf + spread(400, 0, NP, NP)
    out = G.ceiling_stats(f, cf, cl)
    assert out["sum4d"] == 400 and out["ci95"][0] < out["point"] < out["ci95"][1]
