"""Tests of r5_stats.py (rule DECISION_RULE.md D8, D9, D10, section 5 items 5 to 8, section 10 list A item 6).
Synthetic only: hand-made per-anchor arrays and stub family dicts shaped like round 3's run_family output.
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider test_r5_stats.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r5_common as R5  # noqa: E402
import r5_stats as S  # noqa: E402

N = 48
CL = np.repeat(np.arange(12), 4)
PAIR = np.tile(np.arange(3), 16)
TAUS = R5.R3.TAUS


def pa(r1, other=None, gain=None):
    r1 = np.asarray(r1, np.float64)
    other = np.zeros_like(r1) if other is None else np.asarray(other, np.float64)
    gain = r1 - other if gain is None else np.asarray(gain, np.float64)
    z = np.zeros_like(r1)
    return {"r1": r1, "gain": gain, "other": other, "swap": z.copy(), "strict": z.copy()}


def cfpa(r1):
    r1 = np.asarray(r1, np.float64)
    return {"r1": r1, "gain": np.zeros_like(r1), "other": r1.copy(), "swap": np.zeros_like(r1),
            "strict": np.zeros_like(r1)}


def fam(fused, cf):
    return {"fpick": {0: 3, 1: 4}, "cpick": {0: 5, 1: 6}, "sigma": {0: 0.0, 1: 0.0}, "fused": fused, "cf": cf,
            "details": {}, "ctrl": {}}


def quarter(rng, n=N, p=0.5):
    return rng.integers(0, 5, n) / 4.0


# ---------------------------------------------------------------- comparators and the bar comparator (D8)

def test_comparator_order():
    pB, p0, pG, cf = (pa(np.zeros(N)) for _ in range(4))
    labels = [l for l, _ in S.comparators(pB, p0, pG, cf)]
    assert labels == ["Bprime_G", "Bprime_A0", "counterpart", "B"]
    comps = S.comparators(pB, p0, pG, cf)
    assert comps[0][1] is pG
    assert comps[1][1] is p0
    assert comps[2][1] is cf
    assert comps[3][1] is pB


def test_bar_comparator_ties_to_earliest():
    same = np.full(N, 0.25)
    comps = S.comparators(pa(same), pa(same), pa(same), cfpa(same))
    assert S.bar_comparator(comps)[0] == "Bprime_G"


def test_bar_comparator_scope_choices_three_way():
    n = 12
    g = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], float)
    a0 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0], float)
    cfv = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], float)
    comps = S.comparators(pa(np.zeros(n)), pa(a0), pa(g), cfpa(cfv))
    s1 = np.arange(n) < 6
    odd = np.arange(n) % 2 == 1
    assert S.bar_comparator(comps, s1)[0] == "counterpart"
    assert S.bar_comparator(comps, ~s1)[0] == "Bprime_A0"
    assert S.bar_comparator(comps)[0] == "Bprime_A0"
    assert S.bar_comparator(comps, odd)[0] == "Bprime_A0"
    assert S.bar_comparator(comps, ~odd)[0] == "Bprime_A0"
    assert S.bar_comparator(comps, np.arange(n) == 0)[0] == "Bprime_G"
    labels = {S.bar_comparator(comps, m)[0] for m in (s1, ~s1, odd, ~odd, np.arange(n) == 0)}
    assert len(labels) == 3


# ---------------------------------------------------------------- D10

def _rec(point, lo_bar, lo_gain, name="G-T"):
    return {"name": name, "bar_margin": {"point": point, "ci95": [lo_bar, point + 1]},
            "gain_statistic": {"point": 1.0, "ci95": [lo_gain, 2.0]}}


def test_d10_all_clear():
    r = S.d10(_rec(0.6, 0.1, 0.1))
    assert r["clauses"]["c1"] is True
    assert r["clauses"]["c2"] is True
    assert r["clauses"]["c3"] is True
    assert r["clauses"]["clears"] is True
    assert r["boundaries"] == []


def test_d10_point_half_passes_and_flags():
    r = S.d10(_rec(0.5, 0.1, 0.1))
    assert r["clauses"]["c1"] is True
    assert r["clauses"]["clears"] is True
    assert len(r["boundaries"]) == 1
    assert "clause 1" in r["boundaries"][0]


def test_d10_point_049_fails_without_flag():
    r = S.d10(_rec(0.49, 0.1, 0.1))
    assert r["clauses"]["c1"] is False
    assert r["clauses"]["clears"] is False
    assert r["boundaries"] == []


def test_d10_bar_lower_bound_exactly_zero_fails_and_flags():
    r = S.d10(_rec(0.9, 0.0, 0.1))
    assert r["clauses"]["c2"] is False
    assert r["clauses"]["clears"] is False
    assert any("clause 2" in b for b in r["boundaries"])


def test_d10_clause3_reads_gain_statistic_not_bar():
    r = S.d10(_rec(0.9, 0.2, -0.1))
    assert r["clauses"]["c2"] is True
    assert r["clauses"]["c3"] is False
    assert r["clauses"]["clears"] is False
    r2 = S.d10(_rec(0.9, -0.2, 0.3))
    assert r2["clauses"]["c2"] is False
    assert r2["clauses"]["c3"] is True


def test_d10_flag_within_1e12_only():
    r = S.d10(_rec(0.5 - 5e-13, 0.1, 0.1))
    assert r["clauses"]["c1"] is False
    assert len(r["boundaries"]) == 1
    assert S.d10(_rec(0.5 - 1e-9, 0.1, 0.1))["boundaries"] == []
    assert len(S.d10(_rec(0.9, 0.1, 3e-13))["boundaries"]) == 1


# ---------------------------------------------------------------- delta_k (D9)

def test_delta_k_from_integers():
    a = np.array([0.25, 0.5, 1.0, 0.0])
    b = np.array([0.0, 0.5, 0.25, 0.25])
    d = S.delta_k(fam(pa(a), cfpa(a)), fam(pa(b), cfpa(b)))
    assert d == 1 + 0 + 3 - 1
    assert type(d) is int


def test_delta_k_refuses_non_quarter():
    a = np.array([0.25, 0.3])
    b = np.array([0.0, 0.25])
    with pytest.raises(AssertionError):
        S.delta_k(fam(pa(a), cfpa(a)), fam(pa(b), cfpa(b)))
    with pytest.raises(AssertionError):
        S.delta_k(fam(pa(b), cfpa(b)), fam(pa(a), cfpa(a)))


def test_delta_k_is_sum_not_rounded_mean():
    n = 100_000
    a = np.full(n, 0.25)
    b = np.zeros(n)
    assert S.delta_k(fam(pa(a), cfpa(a)), fam(pa(b), cfpa(b))) == n


# ---------------------------------------------------------------- dev_record

def _stub(seed=0):
    rng = np.random.default_rng(seed)
    aff_f = quarter(rng)
    cand_f = np.clip(aff_f + rng.integers(-1, 2, N) * 0.25, 0, 1)
    cf_a = quarter(rng)
    cf_c = quarter(rng)
    return (fam(pa(cand_f, other=np.zeros(N)), cfpa(cf_c)), fam(pa(aff_f), cfpa(cf_a)),
            pa(quarter(rng)), pa(quarter(rng)), pa(quarter(rng)), pa(quarter(rng)))


def test_dev_record_structure():
    f, a, pB, p0, pG, p1 = _stub()
    r = S.dev_record("G-T", f, a, pB, p0, pG, p1, CL, PAIR, TAUS)
    assert r["name"] == "G-T"
    assert type(r["delta_int"]) is int
    assert r["bar_comparator"] in ("Bprime_G", "Bprime_A0", "counterpart", "B")
    assert set(r["d10"]) == {"c1", "c2", "c3", "clears"}
    assert set(r["per_pair_bar_margin"]) == set(R5.C.POOLED_ORDER)
    assert set(r["comparator_means"]) == {"Bprime_G", "Bprime_A0", "counterpart", "B"}
    assert r["delta"]["point"] == pytest.approx(100.0 * r["delta_int"] / (4 * N))
    assert "Bprime_G_minus_Bprime_A0" in r
    assert "beside_Bprime_A1" in r
    assert r["cell_text"]["fused"].keys() == {0, 1}
    assert r["cells"]["fpick"] == {0: 3, 1: 4}


def test_dev_record_identical_to_aff_has_delta_zero_flag():
    f, a, pB, p0, pG, p1 = _stub()
    r = S.dev_record("G-TF", a, a, pB, p0, pG, p1, CL, PAIR, TAUS)
    assert r["delta_int"] == 0
    assert any("exactly 0" in b for b in r["boundaries"])


def test_dev_record_bar_ties_to_bprime_g_first():
    same = np.full(N, 0.25)
    f = fam(pa(np.full(N, 0.75)), cfpa(np.full(N, 0.25)))
    a = fam(pa(np.full(N, 0.5)), cfpa(np.full(N, 0.25)))
    r = S.dev_record("G-T", f, a, pa(same), pa(same), pa(same), pa(same), CL, PAIR, TAUS)
    assert r["bar_comparator"] == "Bprime_G"


def test_dev_record_refuses_unknown_name_and_nonzero_cf_gain():
    f, a, pB, p0, pG, p1 = _stub()
    with pytest.raises(ValueError):
        S.dev_record("V4", f, a, pB, p0, pG, p1, CL, PAIR, TAUS)
    f["cf"]["gain"] = np.full(N, 0.25)
    with pytest.raises(AssertionError):
        S.dev_record("G-T", f, a, pB, p0, pG, p1, CL, PAIR, TAUS)


def test_dev_record_without_aff_has_no_delta():
    f, a, pB, p0, pG, p1 = _stub()
    r = S.dev_record("G-T", f, None, pB, p0, pG, p1, CL, PAIR, TAUS)
    assert r["delta_int"] is None
    assert r["delta"] is None


# ---------------------------------------------------------------- carry (items 7 and 8)

SHA = "ab" * 32


def crec(delta, clears=True, bnd=None, name="G-T"):
    """A record with the fields carry recomputes D10 from (it ignores any stored 'clears')."""
    lo_gain = 0.1 if clears else -0.1
    return {"name": name, "delta_int": delta, "bar_margin": {"point": 0.9, "ci95": [0.1, 1.5]},
            "gain_statistic": {"point": 1.0, "ci95": [lo_gain, 2.0]}, "boundaries": list(bnd or []),
            "d10": {"clears": True}}


def two(g, h):
    return {"G-T": g, "G-TF": {**h, "name": "G-TF"}}


def test_carry_tie_band_24_inclusive_25_exclusive():
    c = S.carry(two(crec(100), crec(124)), SHA)
    assert c["M"] == 124
    assert c["tied"] == ["G-T", "G-TF"]
    assert c["carried"] == "G-T"
    assert any("exactly 24" in b for b in c["boundaries"])
    c = S.carry(two(crec(100), crec(125)), SHA)
    assert c["tied"] == ["G-TF"]
    assert c["carried"] == "G-TF"
    assert c["boundaries"] == []


def test_carry_tie_goes_to_gt_regardless_of_who_is_larger():
    assert S.carry(two(crec(10), crec(11)), SHA)["carried"] == "G-T"
    assert S.carry(two(crec(11), crec(10)), SHA)["carried"] == "G-T"


def test_carry_delta_zero_not_in_e():
    c = S.carry(two(crec(0), crec(5)), SHA)
    assert c["E"] == ["G-TF"]
    assert c["carried"] == "G-TF"
    c = S.carry(two(crec(0), crec(-3)), SHA)
    assert c["E"] == []


def test_carry_requires_all_clauses_recomputed_not_stored():
    r = crec(500, clears=False)                 # stored d10 says clears True; the gain lower bound is -0.1
    c = S.carry(two(r, crec(1)), SHA)
    assert c["E"] == ["G-TF"]
    assert c["carried"] == "G-TF"
    assert c["candidates"]["G-T"]["d10"]["c3"] is False
    r = crec(500)
    r["bar_margin"] = {"point": 0.49, "ci95": [0.1, 1.5]}
    assert S.carry(two(r, crec(1)), SHA)["E"] == ["G-TF"]
    r = crec(500)
    r["bar_margin"] = {"point": 0.9, "ci95": [0.0, 1.5]}
    assert S.carry(two(r, crec(1)), SHA)["E"] == ["G-TF"]


def test_carry_empty_is_kill():
    c = S.carry(two(crec(50, clears=False), crec(-4)), SHA)
    assert c["E"] == []
    assert c["M"] is None
    assert c["tied"] == []
    assert c["carried"] is None
    assert c["kill"] is True
    assert S.console_line(c) == "KILL (pending the phase-1 agreement, rule §8)"


def test_carry_console_line_and_fields():
    c = S.carry(two(crec(7), crec(1)), SHA)
    assert c["kill"] is False
    assert S.console_line(c) == "CARRY G-T (pending the phase-1 agreement, rule §8)"
    assert c["dev_seed42_sha256"] == SHA
    assert c["rule_sha256"] == R5.RULE_SHA
    assert c["candidates"]["G-T"]["delta_int"] == 7
    assert c["candidates"]["G-TF"]["d10"]["clears"] is True


def test_carry_requires_a_valid_dev_sha():
    for bad in (None, "", "ab" * 31, "AB" * 32, "zz" * 32, 5):
        with pytest.raises(ValueError):
            S.carry(two(crec(1), crec(2)), bad)
    with pytest.raises(TypeError):
        S.carry(two(crec(1), crec(2)))


def test_carry_refuses_float_delta_and_wrong_set():
    with pytest.raises(AssertionError):
        S.carry(two(crec(1.0), crec(2)), SHA)
    with pytest.raises(AssertionError):
        S.carry(two(crec(np.int64(1)), crec(2)), SHA)
    with pytest.raises(AssertionError):
        S.carry({"G-T": crec(1)}, SHA)


def test_carry_collects_record_boundaries():
    c = S.carry(two(crec(7, bnd=["G-T: x"]), crec(1, bnd=["G-TF: y"])), SHA)
    assert "G-T: x" in c["boundaries"]
    assert "G-TF: y" in c["boundaries"]


# ---------------------------------------------------------------- D10 through dev_record (round 4's D10_CASES)

def world(n=48):
    rng = np.random.default_rng(5)
    cl = np.repeat(np.arange(12), n // 12)
    pi = np.arange(n) % 3
    q = lambda: rng.integers(0, 5, n) / 4.0
    fused = pa(q(), other=q())
    cf = cfpa(q())
    aff = pa(q(), other=q())
    return cl, pi, fam(fused, cf), fam(aff, cfpa(q())), pa(q()), pa(q()), pa(q()), pa(q())


D10_CASES = [
    ((0.5, 0.1, 0.9), (1.0, 0.2, 2.0), {"c1": True, "c2": True, "c3": True}, {1}),
    ((0.49, 0.1, 0.9), (1.0, 0.2, 2.0), {"c1": False, "c2": True, "c3": True}, set()),
    ((0.6, 0.0, 0.9), (1.0, 0.2, 2.0), {"c1": True, "c2": False, "c3": True}, {2}),
    ((0.6, 0.1, 0.9), (1.0, 0.0, 2.0), {"c1": True, "c2": True, "c3": False}, {3}),
    ((0.6, -0.1, 0.9), (1.0, 0.2, 2.0), {"c1": True, "c2": False, "c3": True}, set()),
]


@pytest.mark.parametrize("bar, gain, want, at", D10_CASES)
def test_d10_clauses_through_dev_record(monkeypatch, bar, gain, want, at):
    """Round 4's D10_CASES: C.point_ci returns `bar` for the bar margin's per-episode values and C.compare returns
    `gain` for fused-minus-counterpart gain; every other interval is computed as usual."""
    cl, pi, f, a, pB, p0, pG, p1 = world()
    _, comp, _ = S.bar_comparator(S.comparators(pB, p0, pG, f["cf"]))
    bar_v = f["fused"]["r1"] - comp["r1"]
    seen = []
    point_ci, compare = R5.C.point_ci, R5.C.compare

    def fake_point_ci(values, clusters):
        x = np.asarray(values, np.float64)
        if x.shape == bar_v.shape and np.array_equal(x, bar_v):
            seen.append("bar")
            return {"point": bar[0], "ci95": [bar[1], bar[2]]}
        return point_ci(values, clusters)

    def fake_compare(per_a, per_b, clusters, metric):
        if metric == "gain" and per_a is f["fused"] and per_b is f["cf"]:
            seen.append("gain")
            return {"point": gain[0], "ci95": [gain[1], gain[2]]}
        return compare(per_a, per_b, clusters, metric)

    monkeypatch.setattr(R5.C, "point_ci", fake_point_ci)
    monkeypatch.setattr(R5.C, "compare", fake_compare)
    rec = S.dev_record("G-T", f, a, pB, p0, pG, p1, cl, pi, TAUS)
    assert seen.count("bar") == 1
    assert seen.count("gain") == 1
    assert rec["bar_margin"] == {"point": bar[0], "ci95": [bar[1], bar[2]]}
    assert rec["gain_statistic"] == {"point": gain[0], "ci95": [gain[1], gain[2]]}
    assert rec["d10"] == {**want, "clears": want["c1"] and want["c2"] and want["c3"]}
    flagged = {i for i in (1, 2, 3) if any(f"D10 clause {i} " in s for s in rec["boundaries"])}
    assert flagged == at


# ---------------------------------------------------------------- dev_record values recomputed from the inputs

def perpair_world(n=48):
    """B'(A0) leads overall (0.5 vs 1/3) but B'_G leads in pair 0 (1.0 vs 0.5): a per-pair re-choice would differ."""
    rng = np.random.default_rng(11)
    cl = np.repeat(np.arange(12), n // 12)
    pi = np.arange(n) % 3
    pG = pa((pi == 0).astype(float))
    p0 = pa(np.full(n, 0.5))
    pB = pa(rng.integers(0, 3, n) / 4.0)
    p1 = pa(rng.integers(0, 5, n) / 4.0)
    f = fam(pa(rng.integers(0, 5, n) / 4.0, other=rng.integers(0, 3, n) / 4.0), cfpa(rng.integers(0, 5, n) / 4.0))
    a = fam(pa(rng.integers(0, 5, n) / 4.0, other=rng.integers(0, 3, n) / 4.0), cfpa(rng.integers(0, 5, n) / 4.0))
    return cl, pi, f, a, pB, p0, pG, p1


def test_dev_record_matches_hand_numbers():
    cl, pi, f, a, pB, p0, pG, p1 = perpair_world()
    n = len(cl)
    rec = S.dev_record("G-T", f, a, pB, p0, pG, p1, cl, pi, TAUS)
    C = R5.C
    fu, cf = f["fused"], f["cf"]
    label, comp, _ = S.bar_comparator(S.comparators(pB, p0, pG, cf))
    assert label == "Bprime_A0"
    assert rec["bar_comparator"] == "Bprime_A0"
    v = fu["r1"] - comp["r1"]
    want = C.point_ci(v, cl)
    assert rec["bar_margin"]["point"] == want["point"]
    assert rec["bar_margin"]["ci95"] == want["ci95"]
    m = C.point_ci(fu["r1"] - cf["r1"], cl)
    assert rec["margin_vs_counterpart"]["point"] == m["point"]
    assert rec["margin_vs_counterpart"]["ci95"] == m["ci95"]
    g = C.point_ci(fu["gain"] - cf["gain"], cl)
    assert rec["gain_statistic"]["point"] == g["point"]
    assert rec["gain_statistic"]["ci95"] == g["ci95"]
    e = fu["r1"] + fu["other"] - cf["r1"] - cf["other"]
    assert rec["either_change"] == pytest.approx(100 * e.mean(), abs=1e-12)
    assert rec["either_change"] != 0
    d = int(round(4 * (fu["r1"] - a["fused"]["r1"]).sum()))
    assert rec["delta_int"] == d
    assert type(rec["delta_int"]) is int
    assert rec["delta"]["point"] == pytest.approx(100 * d / (4 * n), abs=1e-12)
    assert rec["delta"]["ci95"] == C.point_ci(fu["r1"] - a["fused"]["r1"], cl)["ci95"]
    assert rec["Bprime_G_minus_Bprime_A0"] == C.point_ci(pG["r1"] - p0["r1"], cl)
    assert rec["beside_Bprime_A1"]["mean_r1"] == 100 * float(np.mean(p1["r1"]))
    assert rec["beside_Bprime_A1"]["candidate_minus"] == C.point_ci(fu["r1"] - p1["r1"], cl)
    for i, p in enumerate(C.POOLED_ORDER):
        w = C.point_ci(v[pi == i], cl[pi == i])
        assert rec["per_pair_bar_margin"][p]["point"] == w["point"]
        assert rec["per_pair_bar_margin"][p]["ci95"] == w["ci95"]
    # the seed's comparator, not pair 0's own best (B'_G): pair 0 would give a different number
    alt = C.point_ci((fu["r1"] - pG["r1"])[pi == 0], cl[pi == 0])
    assert rec["per_pair_bar_margin"][C.POOLED_ORDER[0]]["point"] != alt["point"]


def test_dev_record_cell_text_uses_the_candidates_taus():
    cl, pi, f, a, pB, p0, pG, p1 = perpair_world()
    f["fpick"], f["cpick"] = {0: 100, 1: 200}, {0: 100, 1: 200}
    taus = (0.11, 0.22, 0.33, 0.44)
    assert tuple(taus) != tuple(R5.R3.TAUS)
    rec = S.dev_record("G-TF", f, a, pB, p0, pG, p1, cl, pi, taus)
    for k in ("fused", "cf"):
        for h, c in rec["cell_text"][k].items():
            assert c["tau"] == taus[c["tau_index"]]
    assert rec["cell_text"]["fused"][0]["tau_index"] == 1
    assert rec["cell_text"]["fused"][1]["tau_index"] == 3
    with pytest.raises(TypeError):
        S.dev_record("G-TF", f, a, pB, p0, pG, p1, cl, pi)
