"""Unit tests of r4_fusion.py and r4_stats.py (synthetic data only; no seed data is ever loaded). Run from this folder:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q test_r4_fusion.py -p no:cacheprovider
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r4_common as R4  # noqa: E402
import r4_fusion as RF4  # noqa: E402
import r4_stats as RS4  # noqa: E402

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS  # noqa: E402
from src.eval.aspect_nested import _zdict  # noqa: E402

RF3, RS3, C, F = R4.RF3, R4.RS3, R4.C, R4.RF3.F


# ---------------------------------------------------------------- stubs

class StubScaler:
    def transform(self, X):
        return np.asarray(X, dtype=np.float64)


class StubModel:
    """predict_proba returns a fixed (n, 4) array whatever the input."""
    classes_ = np.arange(4)

    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=np.float64)

    def predict_proba(self, X):
        assert len(X) == len(self.probs)
        return self.probs


def stub_readers(p0, p1):
    return {"halves": [{"scaler": StubScaler(), "model": StubModel(p0)},
                       {"scaler": StubScaler(), "model": StubModel(p1)}],
            "feature_names": R4.R3.rf.feature_names(R4.A1)}


def ga(*rows):
    return [{c: np.asarray(r, np.float32) for c in CONDITIONS} for r in rows]


# ---------------------------------------------------------------- reader_a1

def test_reader_a1_matches_hand_mean_and_ties_go_to_affect():
    p0 = np.array([[0.3, 0.3, 0.2, 0.2],       # tie incl. affect -> affect
                   [0.2, 0.3, 0.3, 0.2],       # tie not incl. affect -> image (first of the tied)
                   [0.1, 0.2, 0.3, 0.4],
                   [0.4, 0.1, 0.1, 0.4]])      # affect ties csd -> affect
    p1 = p0.copy()
    p1[2] = [0.1, 0.2, 0.4, 0.3]
    bundle = SimpleNamespace(F1={c: np.zeros((4, 24)) for c in CONDITIONS})
    out = RF4.reader_a1(bundle, readers=stub_readers(p0, p1))
    manual = (p0 + p1) / 2
    for c in CONDITIONS:
        np.testing.assert_allclose(out["P"][c], manual, rtol=0, atol=1e-15)
        assert out["P"][c].dtype == np.float64 and out["P"][c].shape == (4, 4)
        assert out["pick"][c].dtype == np.int64
        assert out["pick"][c].tolist() == [0, 1, 2, 0]
    # hand-computed mean of the third row: (0.3 + 0.4) / 2
    assert out["P"]["a"][2, 2] == pytest.approx(0.35, abs=1e-15)


def test_reader_a1_defaults_to_bundle_readers_and_checks_layout():
    p = np.tile([0.4, 0.3, 0.2, 0.1], (3, 1))
    bundle = SimpleNamespace(F1={c: np.zeros((3, 24)) for c in CONDITIONS}, readers_a1=stub_readers(p, p))
    assert RF4.reader_a1(bundle)["pick"]["a"].tolist() == [0, 0, 0]
    bad = stub_readers(p, p)
    bad["feature_names"] = list(reversed(bad["feature_names"]))
    with pytest.raises(AssertionError):
        RF4.reader_a1(bundle, readers=bad)


def test_reader_a1_rejects_probabilities_not_summing_to_one():
    p = np.tile([0.4, 0.3, 0.2, 0.2], (3, 1))
    bundle = SimpleNamespace(F1={c: np.zeros((3, 24)) for c in CONDITIONS})
    with pytest.raises(AssertionError):
        RF4.reader_a1(bundle, readers=stub_readers(p, p))


def test_pi_tie_goes_to_affect_too():
    """M28: round 3's A0 pick rule, used unchanged by this round, resolves a tie that includes affect to affect."""
    P = np.array([[0.4, 0.4, 0.2], [0.2, 0.4, 0.4], [0.4, 0.2, 0.4]])
    pick, margin = R4.R3.rf.picks_and_margins(P)
    assert pick.tolist() == [0, 1, 0]
    m = {c: margin for c in CONDITIONS}
    g = RF3.gates_aff(m, {c: pick for c in CONDITIONS}, (0.0, 0.1, 0.2, 0.3))
    assert g[0]["a"].tolist() == [1, 0, 1]


# ---------------------------------------------------------------- abstain and gates

def test_abstain_is_strict_less_than_float32():
    v = np.array([0.0, R4.V75, np.nextafter(R4.V75, 0), 0.5])
    a = RF4.abstain(v)
    assert a.dtype == np.float32 and a.tolist() == [1, 0, 1, 0]
    assert RF4.abstain(v, v75=0.6).tolist() == [1, 1, 1, 1]


def make_gates(n=64, seed=0):
    rng = np.random.default_rng(seed)
    g_aff = [{c: (rng.random(n) < 0.8 - 0.15 * t).astype(np.float32) for c in CONDITIONS} for t in range(4)]
    pick = {c: rng.integers(0, 4, n).astype(np.int64) for c in CONDITIONS}
    v = rng.random(n) * 0.04
    return g_aff, pick, RF4.abstain(v)


def test_candidate_gate_algebra():
    g_aff, pick, keep = make_gates()
    g4 = RF4.gates_candidate("V4", g_aff, keep=keep)
    g2 = RF4.gates_candidate("V2", g_aff, pick_a1=pick)
    g24 = RF4.gates_candidate("V24", g_aff, pick_a1=pick, keep=keep)
    gaff = RF4.gates_candidate("AFF", g_aff)
    assert len(g4) == len(g2) == len(g24) == len(gaff) == 4
    for t in range(4):
        for c in CONDITIONS:
            for g in (g4, g2, g24, gaff):
                assert g[t][c].dtype == np.float32
                assert set(np.unique(g[t][c])) <= {0.0, 1.0}
                assert np.all(g[t][c] <= g_aff[t][c])                       # closed wherever AFF's is closed
            np.testing.assert_array_equal(gaff[t][c], g_aff[t][c])           # AFF unchanged
            np.testing.assert_array_equal(g4[t][c], g_aff[t][c] * keep)
            np.testing.assert_array_equal(g2[t][c], g_aff[t][c] * (pick[c] == 0))
            np.testing.assert_array_equal(g24[t][c], g4[t][c] * g2[t][c])    # V24 = V4 * V2
    assert any((g4[0][c] < g_aff[0][c]).any() for c in CONDITIONS)           # the vetoes close something
    assert any((g2[0][c] < g_aff[0][c]).any() for c in CONDITIONS)


def test_both_factors_one_returns_aff_exactly():
    g_aff, pick, keep = make_gates()
    ones = np.ones(len(keep), np.float32)
    all_affect = {c: np.zeros(len(keep), np.int64) for c in CONDITIONS}
    for name, kw in (("V4", {"keep": ones}), ("V2", {"pick_a1": all_affect}),
                     ("V24", {"keep": ones, "pick_a1": all_affect})):
        out = RF4.gates_candidate(name, g_aff, **kw)
        for t in range(4):
            for c in CONDITIONS:
                np.testing.assert_array_equal(out[t][c], g_aff[t][c])
                assert out[t][c].dtype == np.float32


def test_gate_guards_fire():
    g_aff, pick, keep = make_gates()
    with pytest.raises(ValueError):
        RF4.gates_candidate("V4", g_aff)                                    # missing keep
    with pytest.raises(ValueError):
        RF4.gates_candidate("V2", g_aff)                                    # missing pick
    with pytest.raises(ValueError):
        RF4.gates_candidate("V24", g_aff, keep=keep)                        # missing pick
    with pytest.raises(ValueError):
        RF4.gates_candidate("V7", g_aff)
    bad = [{c: g[c] * 0.5 for c in CONDITIONS} for g in g_aff]               # not 0/1
    with pytest.raises(AssertionError):
        RF4.gates_candidate("AFF", bad)
    bad64 = [{c: g[c].astype(np.float64) for c in CONDITIONS} for g in g_aff]
    with pytest.raises(AssertionError):
        RF4.gates_candidate("V4", bad64, keep=keep)
    with pytest.raises(AssertionError):                                      # a keep that is not 0/1
        RF4.gates_candidate("V4", g_aff, keep=keep * 2)


def test_imgabst_r1_uses_the_same_factor_function_as_v4(monkeypatch):
    g_aff, pick, keep = make_gates()
    g_r1 = [{c: np.maximum(g[c], (np.arange(len(keep)) % 2).astype(np.float32)) for c in CONDITIONS} for g in g_aff]
    out = RF4.gates_imgabst_r1(g_r1, keep)
    for t in range(4):
        for c in CONDITIONS:
            np.testing.assert_array_equal(out[t][c], g_r1[t][c] * keep)
            assert out[t][c].dtype == np.float32
    calls = []
    real = RF4.apply_keep

    def spy(g, k):
        calls.append(id(g))
        return real(g, k)

    monkeypatch.setattr(RF4, "apply_keep", spy)
    RF4.gates_candidate("V4", g_aff, keep=keep)
    n_v4 = len(calls)
    RF4.gates_imgabst_r1(g_r1, keep)
    assert n_v4 >= 1 and len(calls) == 2 * n_v4                              # both go through apply_keep


# ---------------------------------------------------------------- comparators and the bar comparator

def pa(r1, gain=None):
    r1 = np.asarray(r1, np.float64)
    return {"r1": r1, "gain": np.zeros_like(r1) if gain is None else np.asarray(gain, np.float64)}


def test_comparators_order_per_candidate():
    A, B0, CF, BB = pa([.25]), pa([.5]), pa([.75]), pa([1.0])
    for name in ("V4", "AFF"):
        comps = RS4.comparators(name, BB, B0, A, CF)
        assert [l for l, _ in comps] == ["Bprime_A0", "counterpart", "B"]
        assert comps[0][1] is B0 and comps[1][1] is CF and comps[2][1] is BB
    for name in ("V2", "V24"):
        comps = RS4.comparators(name, BB, B0, A, CF)
        assert [l for l, _ in comps] == ["Bprime_A1", "Bprime_A0", "counterpart", "B"]
        assert comps[0][1] is A
    with pytest.raises(ValueError):
        RS4.comparators("V9", BB, B0, A, CF)


def test_bar_comparator_four_way_order_differs_by_scope():
    """T2-M4: per-seed, pooled and per-pair choices differ; ties go to the earliest in D8's order."""
    n = 12
    seg = np.repeat(np.arange(4), 3)

    def arr(win_seg, other):
        a = np.full(n, other)
        a[seg == win_seg] = 0.75
        return pa(a)

    A1, A0, CF, B = arr(0, .25), arr(1, .25), arr(2, .25), arr(3, .5)
    comps = RS4.comparators("V2", B, A0, A1, CF)
    assert RS4.bar_comparator(comps)[0] == "B"                               # pooled: B (2.25 against 1.5)
    labels = [RS4.bar_comparator(comps, mask=(seg == s))[0] for s in range(4)]
    assert labels == ["Bprime_A1", "Bprime_A0", "counterpart", "B"]          # per-seed or per-pair choices differ
    two = RS4.bar_comparator(comps, mask=(seg <= 1))                         # A1 and A0 tie at 0.5 over two segments
    assert two[0] == "Bprime_A1"                                             # tie -> earliest
    assert two[1] is A1
    tie = [("Bprime_A1", pa([.5, .5])), ("Bprime_A0", pa([.5, .5])), ("counterpart", pa([.5, .5])), ("B", pa([.5, .5]))]
    assert RS4.bar_comparator(tie)[0] == "Bprime_A1"
    v4 = RS4.comparators("V4", B, A0, A1, CF)                                # V4 does not see B'(A1)
    assert RS4.bar_comparator(v4, mask=(seg == 0))[0] in ("Bprime_A0", "counterpart", "B")
    # full precision: a difference of 1e-12 decides
    t = [("x", pa([.5])), ("y", pa([.5 + 1e-12]))]
    assert RS4.bar_comparator(t)[0] == "y"


# ---------------------------------------------------------------- the development record

def make_family(n=40, seed=0, shift=0):
    rng = np.random.default_rng(seed)
    r1 = rng.integers(0, 5, n) / 4.0
    r1 = np.clip(r1 + shift * 0.25 * (rng.random(n) < 0.3), 0, 1)
    gain = rng.integers(-4, 5, n) / 4.0
    fused = {"r1": r1, "gain": gain, "other": rng.integers(0, 5, n) / 4.0, "swap": r1 * 0, "strict": r1 * 0}
    cf = {"r1": rng.integers(0, 5, n) / 4.0, "gain": np.zeros(n), "other": rng.integers(0, 5, n) / 4.0,
          "swap": r1 * 0, "strict": r1 * 0}
    return {"fpick": {0: 3, 1: 9}, "cpick": {0: 4, 1: 8}, "sigma": {0: 0.0, 1: 0.0}, "fused": fused, "cf": cf}


def make_world(n=40):
    rng = np.random.default_rng(5)
    cl = rng.integers(0, 10, n)
    pi = np.arange(n) % 3
    pB = pa(rng.integers(0, 5, n) / 4.0)
    pBp0 = pa(rng.integers(0, 5, n) / 4.0)
    pBp1 = pa(rng.integers(0, 5, n) / 4.0)
    return cl, pi, pB, pBp0, pBp1


def test_dev_record_matches_hand_numbers():
    n = 40
    cl, pi, pB, pBp0, pBp1 = make_world(n)
    fam, aff = make_family(n, 1, shift=1), make_family(n, 2)
    for name in ("V4", "V2"):
        rec = RS4.dev_record(name, fam, aff, pB, pBp0, pBp1, cl, pi)
        comps = RS4.comparators(name, pB, pBp0, pBp1, fam["cf"])
        label, comp, _ = RS4.bar_comparator(comps)
        v = fam["fused"]["r1"] - comp["r1"]
        want = C.point_ci(v, cl)
        assert rec["bar_comparator"] == label
        assert rec["bar_margin"]["point"] == want["point"] and rec["bar_margin"]["ci95"] == want["ci95"]
        assert rec["fused_r1"] == 100 * float(np.mean(fam["fused"]["r1"]))
        assert rec["cf_r1"] == 100 * float(np.mean(fam["cf"]["r1"]))
        assert rec["cells"] == {"fpick": fam["fpick"], "cpick": fam["cpick"]} and rec["sigma"] == fam["sigma"]
        m = C.point_ci(fam["fused"]["r1"] - fam["cf"]["r1"], cl)
        assert rec["margin_vs_counterpart"]["point"] == m["point"] and rec["margin_vs_counterpart"]["ci95"] == m["ci95"]
        g = C.point_ci(fam["fused"]["gain"] - fam["cf"]["gain"], cl)
        assert rec["gain_statistic"]["point"] == g["point"] and rec["gain_statistic"]["ci95"] == g["ci95"]
        e = fam["fused"]["r1"] + fam["fused"]["other"] - fam["cf"]["r1"] - fam["cf"]["other"]
        assert rec["either_change"] == pytest.approx(100 * e.mean(), abs=1e-12)
        d = int(round(4 * (fam["fused"]["r1"] - aff["fused"]["r1"]).sum()))
        assert rec["delta_int"] == d and type(rec["delta_int"]) is int
        assert rec["delta"]["point"] == pytest.approx(100 * d / (4 * n), abs=1e-12)
        dd = C.point_ci(fam["fused"]["r1"] - aff["fused"]["r1"], cl)
        assert rec["delta"]["ci95"] == dd["ci95"]
        assert rec["d10"] == {"c1": bool(want["point"] >= 0.5), "c2": bool(want["ci95"][0] > 0),
                              "c3": bool(g["ci95"][0] > 0),
                              "clears": bool(want["point"] >= 0.5 and want["ci95"][0] > 0 and g["ci95"][0] > 0)}
        assert set(rec["per_pair_bar_margin"]) == set(C.POOLED_ORDER)
        for i, p in enumerate(C.POOLED_ORDER):
            assert rec["per_pair_bar_margin"][p]["point"] == C.point_ci(v[pi == i], cl[pi == i])["point"]


def test_dev_record_counterpart_mutation_is_caught():
    """Replacing the counterpart by the fused arrays changes the margin; the hand number above would fail."""
    n = 40
    cl, pi, pB, pBp0, pBp1 = make_world(n)
    fam, aff = make_family(n, 1), make_family(n, 2)
    good = RS4.dev_record("V4", fam, aff, pB, pBp0, pBp1, cl, pi)
    mut = dict(fam, cf=fam["fused"])
    with pytest.raises(AssertionError):                                      # fused gain is not 0, so it is not a counterpart
        RS4.dev_record("V4", mut, aff, pB, pBp0, pBp1, cl, pi)
    mut2 = dict(fam, cf=dict(fam["cf"], r1=fam["fused"]["r1"]))
    bad = RS4.dev_record("V4", mut2, aff, pB, pBp0, pBp1, cl, pi)
    assert bad["margin_vs_counterpart"]["point"] == 0.0 != good["margin_vs_counterpart"]["point"]
    assert bad["cf_r1"] != good["cf_r1"]


def test_delta_int_non_multiple_of_quarter_raises():
    n = 40
    cl, pi, pB, pBp0, pBp1 = make_world(n)
    fam, aff = make_family(n, 1), make_family(n, 2)
    fam["fused"] = dict(fam["fused"], r1=fam["fused"]["r1"] + 0.1)
    with pytest.raises(AssertionError, match="multiple of 0.25"):
        RS4.dev_record("V4", fam, aff, pB, pBp0, pBp1, cl, pi)


def test_delta_summed_from_integers_not_float_means():
    n = 40
    cl, pi, pB, pBp0, pBp1 = make_world(n)
    fam = make_family(n, 1)
    aff = make_family(n, 2)
    aff["fused"] = dict(aff["fused"], r1=fam["fused"]["r1"].copy())
    aff["fused"]["r1"][3] = np.clip(aff["fused"]["r1"][3] - 0.25, 0, 1) if fam["fused"]["r1"][3] > 0 else 0.0
    rec = RS4.dev_record("V4", fam, aff, pB, pBp0, pBp1, cl, pi)
    assert rec["delta_int"] == (1 if fam["fused"]["r1"][3] > 0 else 0)
    rec0 = RS4.dev_record("V4", fam, None, pB, pBp0, pBp1, cl, pi)           # no AFF family (regression of item 3)
    assert rec0["delta_int"] is None and rec0["delta"] is None


def test_rho_ctrl_criterion_sigma_star_and_rho_ctrl_by_hand():
    """M06: round 2's control_choice on a 2-half example against a hand computation. (1 + sigma) z(B) ranks as z(B),
    so every control sum ties and sigma* is the smallest (0); rho_ctrl is the integer 4*R@1 sum over each tune half."""
    n = 16
    rng = np.random.default_rng(3)
    base = {d: rng.normal(size=(n, 13)).astype(np.float32) for d in DIRECTIONS}
    B = {c: {d: base[d].copy() for d in DIRECTIONS} for c in CONDITIONS}
    parity = np.arange(n) % 2
    got = F.control_choice(_zdict(B), parity)

    def hit(s, col):
        return (s[:, col] > np.delete(s, col, axis=1).max(axis=1)).astype(np.int64)

    # condition a looks for column 0, b for column 1; B is condition-free, so both read the same scores
    rho = sum(hit(base[d], 0) + hit(base[d], 1) for d in DIRECTIONS)
    for h in (0, 1):
        assert got[h] == (0.0, int(rho[parity == h].sum()))
    assert any(got[h][1] > 0 for h in (0, 1))


# ---------------------------------------------------------------- carry

def rec_(delta, clears=True):
    return {"delta_int": delta, "d10": {"c1": clears, "c2": clears, "c3": clears, "clears": clears}}


def test_carry_rules():
    r = RS4.carry({"V4": rec_(10), "V2": rec_(30), "V24": rec_(40, clears=False)})
    assert r == {"E": ["V4", "V2"], "M": 30, "tied": ["V4", "V2"], "carried": "V4", "boundaries": []}      # 30 - 10 = 20 <= 24
    r = RS4.carry({"V4": rec_(10), "V2": rec_(34), "V24": rec_(5)})
    assert r["tied"] == ["V4", "V2"] and r["carried"] == "V4"                           # gap exactly 24 is tied
    r = RS4.carry({"V4": rec_(10), "V2": rec_(35), "V24": rec_(5)})
    assert r["E"] == ["V4", "V2", "V24"] and r["M"] == 35 and r["tied"] == ["V2"] and r["carried"] == "V2"
    r = RS4.carry({"V4": rec_(0), "V2": rec_(-3), "V24": rec_(7, clears=False)})        # Delta must be > 0 and clear
    assert r == {"E": [], "M": None, "tied": [], "carried": None, "boundaries": []}
    r = RS4.carry({"V4": rec_(1), "V2": rec_(30), "V24": rec_(30)})
    assert r["tied"] == ["V2", "V24"] and r["carried"] == "V2"                          # order V4, V2, V24
    with pytest.raises(AssertionError):
        RS4.carry({"V4": rec_(10.0), "V2": rec_(3), "V24": rec_(2)})                    # integers only
    with pytest.raises(AssertionError):
        RS4.carry({"V4": rec_(10), "V2": rec_(3)})                                      # all three candidates


# ---------------------------------------------------------------- GO checks

def make_per_seed(n=30, n_seeds=3, csd=True, seed=0, offset=0.25):
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n_seeds):
        cl = rng.integers(0, 12, n) + 100 * 0
        r = lambda: rng.integers(0, 5, n) / 4.0
        fused = {"r1": np.clip(r() + offset, 0, 1), "gain": rng.integers(0, 5, n) / 4.0}
        out.append({"cl": cl, "fused": fused,
                    "cf": {"r1": r(), "gain": np.zeros(n)}, "aff_fused": {"r1": r(), "gain": r()},
                    "B": {"r1": r(), "gain": r()}, "Bp0": {"r1": r(), "gain": r()},
                    "Bp1": {"r1": r(), "gain": r()} if csd else None,
                    "cosine": {"r1": r(), "gain": r()}, "rca": {"r1": r(), "gain": r()}})
    return out


def test_go_checks_count_order_and_aff_last():
    ps = make_per_seed()
    r4 = RS4.go_checks("V4", ps)
    assert len(r4["checks"]) == 8 and list(r4["checks"])[-1] == "vs_AFF"
    assert "vs_Bprime_A1" not in r4["checks"]
    for name in ("V2", "V24"):
        r = RS4.go_checks(name, ps)
        assert len(r["checks"]) == 9 and list(r["checks"])[-1] == "vs_AFF"
        assert list(r["checks"]) == ["vs_cosine", "vs_rca", "vs_B", "vs_Bprime_A0", "vs_Bprime_A1", "vs_counterpart",
                                     "gain_statistic", "gain_vs_rca", "vs_AFF"]
    assert list(r4["checks"]) == ["vs_cosine", "vs_rca", "vs_B", "vs_Bprime_A0", "vs_counterpart", "gain_statistic",
                                  "gain_vs_rca", "vs_AFF"]
    ps_no = make_per_seed(csd=False)
    assert len(RS4.go_checks("V4", ps_no)["checks"]) == 8                  # Bp1 None is fine for V4
    with pytest.raises(ValueError):
        RS4.go_checks("V2", ps_no)                                         # CSD candidate needs B'(A1)


def test_go_checks_match_pooled_check_semantics():
    ps = make_per_seed(seed=4)
    r = RS4.go_checks("V2", ps)
    cls = [s["cl"] for s in ps]
    f = lambda k, m="r1": [np.asarray(s["fused"][m]) - np.asarray(s[k][m]) for s in ps]
    exp = RS3.pooled_check([np.asarray(s["fused"]["r1"]) - np.asarray(s["aff_fused"]["r1"]) for s in ps], cls)
    assert r["checks"]["vs_AFF"] == exp
    assert r["checks"]["vs_Bprime_A1"] == RS3.pooled_check(f("Bp1"), cls)
    assert r["checks"]["vs_counterpart"] == RS3.pooled_check(f("cf"), cls)
    assert r["checks"]["gain_statistic"] == RS3.pooled_check(
        [np.asarray(s["fused"]["gain"]) - np.asarray(s["cf"]["gain"]) for s in ps], cls)
    assert r["checks"]["gain_vs_rca"] == RS3.pooled_check(f("rca", "gain"), cls)
    for v in r["checks"].values():
        assert set(v) == {"point", "ci95", "pass"} and v["pass"] == bool(v["ci95"][0] > 0)
    assert r["go"] == all(v["pass"] for v in r["checks"].values())


def test_go_fails_when_only_the_aff_check_fails():
    ps = make_per_seed(offset=0.5, seed=7)
    for s in ps:                                  # the candidate is far above every comparator but equal to AFF
        s["aff_fused"] = {"r1": s["fused"]["r1"].copy(), "gain": s["fused"]["gain"].copy()}
        s["fused"]["gain"] = np.ones(len(s["cl"]))          # a clear condition gain over the counterpart and RCA
        s["rca"]["gain"] = np.zeros(len(s["cl"]))
        s["fused"]["r1"] = np.clip(s["fused"]["r1"] + 0.25, 0, 1)
        s["aff_fused"]["r1"] = s["fused"]["r1"].copy()
        s["aff_fused"]["r1"][0] = 1.0 - s["aff_fused"]["r1"][0]
    r = RS4.go_checks("V4", ps)
    assert [k for k, v in r["checks"].items() if not v["pass"]] == ["vs_AFF"]
    assert r["go"] is False
    assert RS4.aff_only_failed_reading(r["checks"], "V4").startswith("V4 works, but no improvement over AFF")
    r["checks"]["vs_cosine"]["pass"] = False
    assert RS4.aff_only_failed_reading(r["checks"], "V4") is None


def test_go_checks_guards():
    ps = make_per_seed()
    ps[0]["cf"]["gain"] = ps[0]["cf"]["gain"] + 0.25
    with pytest.raises(AssertionError):
        RS4.go_checks("V4", ps)                                            # counterpart gain must be exactly 0
    ps = make_per_seed()
    ps[1]["aff_fused"] = ps[1]["fused"]
    with pytest.raises(AssertionError):
        RS4.go_checks("V4", ps)                                            # AFF's arrays swapped for the candidate's
    ps = make_per_seed()
    ps[0]["B"] = {"r1": ps[0]["B"]["r1"][:-1], "gain": ps[0]["B"]["gain"][:-1]}
    with pytest.raises(ValueError):
        RS4.go_checks("V4", ps)


def test_reading_sentences():
    s = RS4.reading("vs_AFF", 0.04, 0.123456, 0.2, candidate="V4")
    assert s == "inconclusive at a detectable margin of 0.123 pp (realised pooled half-width 0.200 pp)"
    assert RS4.reading("vs_AFF", 0.0, 0.1, 0.2, candidate="V4") == "V4 did not beat AFF on fresh episodes"
    assert RS4.reading("vs_cosine", -0.3, 0.1, 0.2, candidate="V2") == "V2 did not beat cosine on fresh episodes"
    assert RS4.reading("vs_Bprime_A1", -1, 0.1, 0.2, candidate="V2") == "V2 did not beat B'(A1) on fresh episodes"
    assert "the condition-free comparators on condition gain" in RS4.reading("gain_statistic", -1, 0.1, 0.2, "V4")
    assert "RCA on condition gain" in RS4.reading("gain_vs_rca", 0, 0.1, 0.2, "V4")
    for k in ("vs_rca", "vs_B", "vs_Bprime_A0", "vs_counterpart"):
        assert RS4.reading(k, -1, 1, 1, "V4").startswith("V4 did not beat ")
    with pytest.raises(KeyError):
        RS4.reading("nope", -1, 1, 1)


def test_sensitivity_all_covers_every_check_and_matches_rs3():
    n = 200
    rng = np.random.default_rng(9)
    cl = rng.integers(0, 40, n)
    r = lambda: rng.integers(0, 5, n) / 4.0
    seed42 = {"cl": cl, "fused": {"r1": r(), "gain": r()}, "cf": {"r1": r(), "gain": np.zeros(n)},
              "aff_fused": {"r1": r(), "gain": r()}, "B": {"r1": r(), "gain": r()}, "Bp0": {"r1": r(), "gain": r()},
              "Bp1": {"r1": r(), "gain": r()}, "cosine": {"r1": r(), "gain": r()}, "rca": {"r1": r(), "gain": r()}}
    out = RS4.sensitivity_all("V24", seed42)
    names = list(RS4.go_checks("V24", [seed42])["checks"])
    assert list(out) == names and len(out) == 9
    assert out["vs_AFF"] == RS3.sensitivity(seed42["fused"]["r1"] - seed42["aff_fused"]["r1"], cl)
    assert out["gain_vs_rca"] == RS3.sensitivity(seed42["fused"]["gain"] - seed42["rca"]["gain"], cl)
    for v in out.values():
        assert v["x"] == pytest.approx(2.8 * v["SE"], rel=1e-12)
    assert len(RS4.sensitivity_all("V4", seed42)) == 8


# ---------------------------------------------------------------- review fixes

def make_bundle(n=40, seed=0):
    rng = np.random.default_rng(seed)
    base = {d: rng.normal(size=(n, 13)).astype(np.float32) for d in DIRECTIONS}
    B = {c: {d: base[d].copy() for d in DIRECTIONS} for c in CONDITIONS}
    return SimpleNamespace(B=B, parity=np.arange(n) % 2)


def make_T(n=40, seed=1):
    rng = np.random.default_rng(seed)
    T = {c: {} for c in CONDITIONS}
    for c, col in (("a", 0), ("b", 1)):
        for d in DIRECTIONS:
            t = rng.normal(size=(n, 13)).astype(np.float32)
            t[:, col] += 1.5 * (rng.random(n) < 0.6)
            T[c][d] = t
    return T


def test_candidate_counterpart_is_built_from_the_candidates_own_gates(monkeypatch):
    """I1 (Review focus 3): G_cf of V4 comes from V4's gates, not AFF's."""
    n = 40
    b, T = make_bundle(n), make_T(n)
    g_aff = [{c: np.ones(n, np.float32) for c in CONDITIONS} for _ in range(4)]
    keep = (np.arange(n) % 2 == 0).astype(np.float32)                          # V4 vetoes half of the episodes
    cand = RF4.run_candidate(b, T, "V4", g_aff, keep=keep)
    aff = RF3.run_family(b, T, g_aff)
    own = RF3.run_family(b, T, RF4.gates_candidate("V4", g_aff, keep=keep))
    for m in ("r1", "gain", "other"):
        np.testing.assert_array_equal(np.asarray(cand["cf"][m]), np.asarray(own["cf"][m]))
        np.testing.assert_array_equal(np.asarray(cand["fused"][m]), np.asarray(own["fused"][m]))
    assert any(not np.array_equal(np.asarray(cand["cf"][m]), np.asarray(aff["cf"][m])) for m in ("r1", "other"))
    assert cand["cpick"] != aff["cpick"] or not np.array_equal(cand["cf"]["r1"], aff["cf"]["r1"])
    # the mutation (AFF's gates reach the counterpart) reproduces AFF's counterpart, which the assertion above rejects
    monkeypatch.setattr(RF4, "gates_candidate", lambda name, g, pick_a1=None, keep=None: g)
    mut = RF4.run_candidate(b, T, "V4", g_aff, keep=keep)
    for m in ("r1", "gain", "other"):
        np.testing.assert_array_equal(np.asarray(mut["cf"][m]), np.asarray(aff["cf"][m]))
    assert any(not np.array_equal(np.asarray(mut["cf"][m]), np.asarray(own["cf"][m])) for m in ("r1", "other"))
    monkeypatch.undo()
    fo = RF4.run_candidate(b, T, "V4", g_aff, keep=keep, fused_only=True)
    assert fo["cf"] is None and fo["cpick"] is None


def test_select_fused_uses_rho_ctrl_in_the_min_margin_criterion():
    """I2 / M06: min(rho - rho_ctrl, gamma) and min(rho, gamma) choose different cells on both halves."""
    parity = np.array([0, 1, 0, 1])
    fri = np.array([[3, 2, 3, 2], [2, 2, 2, 2], [3, 3, 2, 3]], np.int8)
    fgi = np.array([[1, 1, 0, 1], [2, 1, 1, 1], [1, 1, 1, 0]], np.int8)
    ctrl = {0: (0.0, 3), 1: (0.0, 4)}
    assert F.select_fused(fri, fgi, ctrl, parity) == {0: 2, 1: 2}
    ignoring = {0: (0.0, 0), 1: (0.0, 0)}                                       # the mutant: rho_ctrl ignored
    assert F.select_fused(fri, fgi, ignoring, parity) == {0: 1, 1: 0}
    tie = np.zeros((3, 4), np.int8)
    assert F.select_fused(tie, tie, {0: (0.0, 0), 1: (0.0, 0)}, parity) == {0: 0, 1: 0}   # ties to the lowest cell


def test_aff_gate_takes_no_factors_and_returns_copies():
    g_aff, pick, keep = make_gates()
    with pytest.raises(ValueError):
        RF4.gates_candidate("AFF", g_aff, keep=keep)
    with pytest.raises(ValueError):
        RF4.gates_candidate("AFF", g_aff, pick_a1=pick)
    out = RF4.gates_candidate("AFF", g_aff)
    for t in range(4):
        for c in CONDITIONS:
            assert out[t][c] is not g_aff[t][c] and not np.shares_memory(out[t][c], g_aff[t][c])
            np.testing.assert_array_equal(out[t][c], g_aff[t][c])


def test_factor_shapes_must_match_the_gate():
    g_aff, pick, keep = make_gates()
    with pytest.raises(AssertionError):
        RF4.gates_candidate("V4", g_aff, keep=np.zeros(1, np.float32))
    with pytest.raises(AssertionError):
        RF4.gates_candidate("V2", g_aff, pick_a1={c: np.zeros(1, np.int64) for c in CONDITIONS})
    with pytest.raises(AssertionError):
        RF4.gates_imgabst_r1(g_aff, keep[:-1])


def test_reader_a1_layout_check_is_unconditional():
    p = np.tile([0.4, 0.3, 0.2, 0.1], (3, 1))
    rd = stub_readers(p, p)
    del rd["feature_names"]
    with pytest.raises(KeyError):
        RF4.reader_a1(SimpleNamespace(F1={c: np.zeros((3, 24)) for c in CONDITIONS}), readers=rd)


def test_boundaries_are_flagged():
    n = 40
    cl, pi, pB, pBp0, pBp1 = make_world(n)
    fam = make_family(n, 1)
    same = RS4.dev_record("V4", fam, fam, pB, pBp0, pBp1, cl, pi)                # Delta = 0 exactly
    assert same["delta_int"] == 0 and any("exactly 0" in x for x in same["boundaries"])
    assert "cell_text" in same and same["cell_text"]["fused"][0]["tau_index"] == RF3.describe(3)["tau_index"]
    other = RS4.dev_record("V4", fam, make_family(n, 2), pB, pBp0, pBp1, cl, pi)
    assert not any("exactly 0" in x for x in other["boundaries"]) or other["delta_int"] == 0
    r = RS4.carry({"V4": rec_(10), "V2": rec_(34), "V24": rec_(5)})
    assert r["boundaries"] == ["V4: tie gap M - Delta_k is exactly 24"]
    r = RS4.carry({"V4": dict(rec_(0, clears=False), boundaries=["V4: Delta_k is exactly 0"]), "V2": rec_(3), "V24": rec_(2)})
    assert r["boundaries"] == ["V4: Delta_k is exactly 0"]
    # a D10 clause within 1e-12 of its threshold: patch C.point_ci so the bar point sits 5e-13 above 0.5
    orig = C.point_ci
    calls = {"n": 0}

    def fake(values, clusters):
        calls["n"] += 1
        r = orig(values, clusters)
        return {"point": 0.5 + 5e-13, "ci95": [0.1, 0.9]} if calls["n"] == 1 else r

    try:
        RS4.C.point_ci = fake
        rec = RS4.dev_record("V4", fam, make_family(n, 2), pB, pBp0, pBp1, cl, pi)
    finally:
        RS4.C.point_ci = orig
    assert any("clause 1" in x for x in rec["boundaries"])


def test_comparators_accept_r1_and_imgabst_with_a0_order():
    A, B0, CF, BB = pa([.25]), pa([.5]), pa([.75]), pa([1.0])
    for name in ("R1", "IMGABST"):
        assert [l for l, _ in RS4.comparators(name, BB, B0, A, CF)] == ["Bprime_A0", "counterpart", "B"]


def test_sensitivity_all_unknown_name_is_value_error():
    with pytest.raises(ValueError):
        RS4.sensitivity_all("V9", {"cl": np.arange(3)})
