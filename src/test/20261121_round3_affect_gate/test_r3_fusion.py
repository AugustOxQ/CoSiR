"""Unit tests of round 3's readers, gates, 224-cell families and pooled statistics (synthetic data only). Run from
/project/CoSiR:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/20261121_round3_affect_gate/test_r3_fusion.py -q -p no:cacheprovider
Each guard has a test that fails when the guard is deleted (mutations are listed in the task report)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r3_common as R  # noqa: E402
import r3_fusion as RF  # noqa: E402
import r3_stats as RS  # noqa: E402

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, cluster_bootstrap  # noqa: E402
from src.eval.aspect_nested import NESTED_A, NESTED_U, _zdict, control_sums  # noqa: E402

C, K, F = R.C, R.K, R.F


# ---------------------------------------------------------------- synthetic data

def make_bundle(n=40, seed=0):
    rng = np.random.default_rng(seed)
    base = {d: rng.normal(size=(n, 13)).astype(np.float32) for d in DIRECTIONS}
    B = {c: {d: base[d].copy() for d in DIRECTIONS} for c in CONDITIONS}
    return SimpleNamespace(n=n, cl=rng.integers(0, 10, size=n), parity=(np.arange(n) % 2), pair_index=np.arange(n) % 3,
                           B=B, Bp=B, F=None, stack=None)


def make_T(n=40, seed=1, zero=False):
    """Condition-dependent terms: T^a favours column 0, T^b column 1 (plus noise)."""
    rng = np.random.default_rng(seed)
    T = {c: {} for c in CONDITIONS}
    for c, col in (("a", 0), ("b", 1)):
        for d in DIRECTIONS:
            t = rng.normal(size=(n, 13)).astype(np.float32)
            t[:, col] += 1.5 * (rng.random(n) < 0.6)
            T[c][d] = np.zeros_like(t) if zero else t
    return T


def make_margins(n=40, seed=2):
    rng = np.random.default_rng(seed)
    P = {c: rng.dirichlet(np.ones(3), size=n) for c in CONDITIONS}
    return P, {c: C.top_two_margin(P[c]) for c in CONDITIONS}, {c: P[c].argmax(axis=1) for c in CONDITIONS}


# ---------------------------------------------------------------- independent brute force of the family

def zs(x):
    x = np.asarray(x, np.float64)
    s = x.std(axis=1, keepdims=True)
    return np.where(s > 0, (x - x.mean(axis=1, keepdims=True)) / np.where(s > 0, s, 1), 0.0)


def hit(s, col):
    others = np.delete(s, col, axis=1).max(axis=1)
    return (s[:, col] > others).astype(np.int64)


def ints(sc):
    """(4*R@1, 4*gain) per episode as ints from {c: {d: scores}}."""
    rho = np.zeros(len(sc["a"]["i2t"]), np.int64)
    gam = np.zeros_like(rho)
    for d in DIRECTIONS:
        haa, hbb, hab, hba = hit(sc["a"][d], 0), hit(sc["b"][d], 1), hit(sc["b"][d], 0), hit(sc["a"][d], 1)
        rho += haa + hbb
        gam += haa + hbb - hab - hba
    return rho, gam


def brute(bundle, T, gates):
    zB = {d: zs(bundle.B["a"][d]) for d in DIRECTIONS}
    zT = {c: {d: zs(T[c][d]) for d in DIRECTIONS} for c in CONDITIONS}
    par = bundle.parity
    cells = [(t, lu, la) for t in range(4) for lu in NESTED_U for la in NESTED_A]
    fused, cf = [], []
    for t, lu, la in cells:
        gt = {c: {d: gates[t][c][:, None].astype(np.float64) * zT[c][d] for d in DIRECTIONS} for c in CONDITIONS}
        Gm = {d: 0.5 * (gt["a"][d] + gt["b"][d]) for d in DIRECTIONS}
        sf = {c: {d: zB[d] + lu * zB[d] + la * gt[c][d] for d in DIRECTIONS} for c in CONDITIONS}
        sc = {c: {d: zB[d] + lu * zB[d] + la * Gm[d] for d in DIRECTIONS} for c in CONDITIONS}
        fused.append((ints(sf), sf))
        cf.append((ints(sc), sc))
    ctrl = {}
    for h in (0, 1):
        tune = par == h
        rho_c = int(ints({c: {d: zB[d] for d in DIRECTIONS} for c in CONDITIONS})[0][tune].sum())
        ctrl[h] = rho_c                                    # (1 + sigma) z(B) ranks as z(B): sigma* = 0
    fpick, cpick = {}, {}
    for h in (0, 1):
        tune = par == h
        crit = [min(int(r[tune].sum()) - ctrl[h], int(g[tune].sum())) for (r, g), _ in fused]
        fpick[h] = max(range(len(cells)), key=lambda i: (crit[i], -i))         # ties to the lowest cell
        rho = [int(r[tune].sum()) for (r, _), _ in cf]
        cpick[h] = max(range(len(cells)), key=lambda i: (rho[i], -i))
    fr1 = np.zeros(len(par))
    for h in (0, 1):
        app = par != h
        fr1[app] = fused[fpick[h]][0][0][app] / 4.0
    return fpick, cpick, fr1


# ---------------------------------------------------------------- gates

def test_aff_gate_is_r1_gate_times_affect_pick():
    P, m, pick = make_margins()
    taus = R.TAUS
    g1, ga = RF.gates_r1(m, taus), RF.gates_aff(m, pick, taus)
    assert len(g1) == len(ga) == 4
    for t in range(4):
        for c in CONDITIONS:
            assert ga[t][c].dtype == np.float32 and g1[t][c].dtype == np.float32
            np.testing.assert_array_equal(g1[t][c], (m[c] >= taus[t]).astype(np.float32))
            np.testing.assert_array_equal(ga[t][c], g1[t][c] * (pick[c] == 0))
    # some rows are open under R1 and closed under AFF (the pick is not affect)
    assert any((g1[2][c] > ga[2][c]).any() for c in CONDITIONS)
    assert any(ga[2][c].any() for c in CONDITIONS)


def test_argmax_tie_including_affect_counts_as_affect():
    P = np.array([[0.4, 0.4, 0.2],     # affect ties image -> affect
                  [0.2, 0.4, 0.4],     # image ties caption, affect lower -> not affect
                  [0.4, 0.2, 0.4]])    # affect ties caption -> affect
    pick, margin = R.rf.picks_and_margins(P)
    assert pick.tolist() == [0, 1, 0]
    m = {c: margin for c in CONDITIONS}
    pk = {c: pick for c in CONDITIONS}
    taus = (0.0, 0.1, 0.2, 0.3)                                  # tau_0 = 0 so a zero margin (the tie) is open for R1
    g1, ga = RF.gates_r1(m, taus), RF.gates_aff(m, pk, taus)
    assert g1[0]["a"].tolist() == [1, 1, 1]
    assert ga[0]["a"].tolist() == [1, 0, 1]


def test_counterpart_from_aff_gates_is_condition_free_and_differs_from_r1s():
    P, m, pick = make_margins()
    T = make_T()
    zT = _zdict(T)
    g1, ga = RF.gates_r1(m, R.TAUS), RF.gates_aff(m, pick, R.TAUS)
    gat1, gata = K.gated_terms(zT, g1[2]), K.gated_terms(zT, ga[2])
    G1, Ga = K.g_cf(gat1), K.g_cf(gata)
    for d in DIRECTIONS:
        np.testing.assert_array_equal(Ga["a"][d].numpy(), Ga["b"][d].numpy())            # condition-free
        assert not np.array_equal(Ga["a"][d].numpy(), G1["a"][d].numpy())                # built from AFF's own gates
    # the wiring mistake (passing the gated term where G_cf is expected) is caught by the condition-free assertion
    from src.eval.aspect_quick_checks import _require_condition_free
    with pytest.raises(Exception):
        _require_condition_free({c: {d: gata[c][d].numpy() for d in DIRECTIONS} for c in CONDITIONS}, "gated")
    # _terms builds G from the gate set it is given
    bundle = make_bundle()
    _, _, Gt = RF._terms(bundle, T, ga)
    for t in range(4):
        np.testing.assert_array_equal(Gt[t]["a"]["i2t"].numpy(), K.g_cf(K.gated_terms(zT, ga[t]))["a"]["i2t"].numpy())


def test_gates_random_reproducible_and_condition_a_drawn_first():
    n = 500
    g_r1 = [{c: np.ones(n, np.float32) for c in CONDITIONS} for _ in range(4)]
    share = {"a": 0.9, "b": 0.1}
    out1, out2 = RF.gates_random(g_r1, share, 123, n), RF.gates_random(g_r1, share, 123, n)
    gen = np.random.default_rng(123)
    ua = gen.random(n)
    ub = gen.random(n)
    exp = {"a": (ua < 0.9).astype(np.float32), "b": (ub < 0.1).astype(np.float32)}
    for t in range(4):
        for c in CONDITIONS:
            np.testing.assert_array_equal(out1[t][c], out2[t][c])
            np.testing.assert_array_equal(out1[t][c], exp[c])
    other = RF.gates_random(g_r1, share, 124, n)
    assert not np.array_equal(other[0]["a"], out1[0]["a"])
    # times R1's gate at every tau index
    g_half = [{c: (np.arange(n) % 2).astype(np.float32) for c in CONDITIONS} for _ in range(4)]
    o = RF.gates_random(g_half, share, 123, n)
    np.testing.assert_array_equal(o[3]["a"], exp["a"] * (np.arange(n) % 2))


def test_affect_shares_are_integer_counts_over_E():
    n = 8
    ga = [{"a": np.array([1, 1, 0, 0, 1, 0, 0, 0], np.float32), "b": np.array([0, 1, 0, 0, 0, 0, 0, 0], np.float32)}
          for _ in range(4)]
    s = RF.affect_shares(ga)
    assert s == {"a": 3 / 8, "b": 1 / 8}
    assert RF.open_count(ga[0], "a") == 3


# ---------------------------------------------------------------- family

def test_run_family_matches_hand_integer_crossfit_with_ties_to_lowest_cell():
    bundle = make_bundle(40, seed=3)
    T = make_T(40, seed=4)
    P, m, pick = make_margins(40, seed=5)
    gates = RF.gates_aff(m, pick, R.TAUS)
    res = RF.run_family(bundle, T, gates)
    fpick, cpick, fr1 = brute(bundle, T, gates)
    assert res["fpick"] == fpick
    assert res["cpick"] == cpick
    assert res["sigma"] == {0: 0.0, 1: 0.0}
    np.testing.assert_array_equal(np.asarray(res["fused"]["r1"]), fr1)
    assert np.all(np.asarray(res["cf"]["gain"]) == 0)                 # condition-free counterpart
    for h in (0, 1):                                                  # the details are integers of the chosen cell
        assert isinstance(res["details"]["fused"][str(h)]["rho"], int)
    fo = RF.run_family(bundle, T, gates, fused_only=True)
    assert fo["cpick"] is None and fo["cf"] is None and fo["fpick"] == fpick


def test_run_family_all_ties_go_to_cell_zero():
    bundle = make_bundle(40, seed=6)
    T = make_T(40, zero=True)
    P, m, pick = make_margins(40, seed=7)
    res = RF.run_family(bundle, T, RF.gates_r1(m, R.TAUS))
    assert res["fpick"] == {0: 0, 1: 0}
    assert res["cpick"] == {0: 0, 1: 0}


def test_score_frozen_applies_cell_of_tune_half_to_other_half():
    bundle = make_bundle(40, seed=8)
    T = make_T(40, seed=9)
    P, m, pick = make_margins(40, seed=10)
    gates = RF.gates_aff(m, pick, R.TAUS)
    res = RF.run_family(bundle, T, gates)
    fr = RF.score_frozen(bundle, T, gates, res["fpick"], res["cpick"])
    for k in ("r1", "gain", "other", "swap", "strict"):
        np.testing.assert_array_equal(np.asarray(fr["fused"][k]), np.asarray(res["fused"][k]))
        np.testing.assert_array_equal(np.asarray(fr["cf"][k]), np.asarray(res["cf"][k]))
    fr2 = RF.score_frozen(bundle, T, gates, (39, 119), (149, 10))
    assert np.asarray(fr2["fused"]["r1"]).shape == (40,)
    d = RF.describe(119)
    assert (d["tau_index"], d["lambda_u"], d["lambda_a"]) == (2, 0.0, 16.0)
    assert RF.describe(39)["tau_index"] == 0 and RF.describe(39)["lambda_u"] == 4.0


def test_open_shares_label_free_counts():
    pair_index = np.arange(12) % 3
    g = [{c: np.ones(12, np.float32) for c in CONDITIONS} for _ in range(4)]
    s = RF.open_shares(g, pair_index)
    assert s["tau_0"]["overall"] == 100.0 and s["tau_3"]["open_count"] == {"a": 12, "b": 12}
    assert set(s["tau_1"]["per_pair"]) == set(C.POOLED_ORDER)


# ---------------------------------------------------------------- reader (injected half-readers)

def fake_readers(seed=0):
    rng = np.random.default_rng(seed)
    halves = []
    for _ in range(2):
        X = rng.normal(size=(300, 18))
        y = rng.integers(0, 3, size=300)
        sc = StandardScaler().fit(X)
        halves.append({"scaler": sc, "model": LogisticRegression(max_iter=200).fit(sc.transform(X), y)})
    return {"halves": halves}


def test_reader_matches_manual_computation_and_never_uses_smoke(monkeypatch):
    rng = np.random.default_rng(11)
    n = 30
    bundle = SimpleNamespace(F={c: rng.normal(size=(n, 18)) for c in CONDITIONS},
                             stack={d: rng.normal(size=(n, 3, 13)).astype(np.float32) for d in DIRECTIONS})
    pk = fake_readers()
    out = RF.reader(bundle, readers=pk)
    for c in CONDITIONS:
        manual = np.mean([h["model"].predict_proba(h["scaler"].transform(bundle.F[c])) for h in pk["halves"]], axis=0)
        np.testing.assert_allclose(out["P"][c], manual, rtol=0, atol=1e-15)
        assert out["P"][c].dtype == np.float64 and out["P"][c].shape == (n, 3)
        np.testing.assert_array_equal(out["pick"][c], out["P"][c].argmax(axis=1))
        np.testing.assert_array_equal(out["m"][c], C.top_two_margin(out["P"][c]))
        for d in DIRECTIONS:
            exp = np.einsum("nh,nhk->nk", out["P"][c], bundle.stack[d].astype(np.float64)).astype(np.float32)
            np.testing.assert_array_equal(out["T"][c][d], exp)
            assert out["T"][c][d].dtype == np.float32
    calls = []

    def fake_load(config, smoke):
        calls.append((config, smoke))
        return (pk, None, None)

    monkeypatch.setattr(RF.rb, "load_readers", fake_load)
    RF.reader(bundle)
    assert calls == [("A0", False)]


# ---------------------------------------------------------------- statistics

def test_pooled_check_shared_painting_is_one_cluster():
    rng = np.random.default_rng(12)
    P_, per = 40, 6
    base_cl = np.repeat(np.arange(P_), per)                       # same paintings on every seed
    cls = [base_cl.copy() for _ in range(3)]
    vals = [rng.normal(0.1, 1.0, size=len(base_cl)) for _ in range(3)]
    got = RS.pooled_check(vals, cls)
    cat_v, cat_c = np.concatenate(vals), np.concatenate(cls)
    ref = C.point_ci(cat_v, cat_c)
    assert got["point"] == ref["point"] and got["ci95"] == ref["ci95"]
    assert got["pass"] == bool(ref["ci95"][0] > 0)
    distinct = C.point_ci(cat_v, np.concatenate([c + 1000 * s for s, c in enumerate(cls)]))
    assert distinct["ci95"] != got["ci95"]                         # treating them as different clusters would change it
    # the interval is the 5,000-resample seed-42 painting bootstrap in percentage points
    raw = cluster_bootstrap(cat_v, cat_c, n_boot=5000, seed=42, chunk=250)
    assert got["ci95"] == [100 * raw["ci95"][0], 100 * raw["ci95"][1]]


def test_pooled_check_pass_is_strict():
    cl = np.repeat(np.arange(10), 3)
    zero = RS.pooled_check([np.zeros(30)], [cl])
    assert zero["ci95"] == [0.0, 0.0] and zero["pass"] is False    # lower bound equal to 0 does not pass
    up = RS.pooled_check([np.ones(30)], [cl])
    assert up["pass"] is True


def _per_seed(rng, shift_aff=0.3, n=600, P_=300, n_seeds=3):
    out = []
    for s in range(n_seeds):
        cl = rng.integers(0, P_, size=n)

        def pa(mu_r, mu_g):
            return {"r1": rng.normal(mu_r, 0.2, size=n), "gain": rng.normal(mu_g, 0.2, size=n), "other": np.zeros(n)}
        aff = pa(0.5 + shift_aff, 0.5 + shift_aff)
        out.append({"cl": cl, "pair_index": np.arange(n) % 3, "aff": aff, "cf": {**pa(0.5, 0.0), "gain": np.zeros(n)},
                    "r1": pa(0.5, 0.4), "cosine": pa(0.1, 0.0), "rca": pa(0.2, 0.2), "B": pa(0.4, 0.0), "Bp": pa(0.45, 0.0)})
    return out


def test_go_checks_seven_checks_and_secondary():
    ps = _per_seed(np.random.default_rng(13))
    res = RS.go_checks(ps)
    assert tuple(res["checks"]) == RS.GO_CHECKS and len(res["checks"]) == 7
    cls = [s["cl"] for s in ps]

    def direct(f):
        return RS.pooled_check([f(s) for s in ps], cls)
    assert res["checks"]["r1_vs_cosine"] == direct(lambda s: s["aff"]["r1"] - s["cosine"]["r1"])
    assert res["checks"]["r1_vs_rca"] == direct(lambda s: s["aff"]["r1"] - s["rca"]["r1"])
    assert res["checks"]["r1_vs_B"] == direct(lambda s: s["aff"]["r1"] - s["B"]["r1"])
    assert res["checks"]["r1_vs_Bprime"] == direct(lambda s: s["aff"]["r1"] - s["Bp"]["r1"])
    assert res["checks"]["r1_vs_counterpart"] == direct(lambda s: s["aff"]["r1"] - s["cf"]["r1"])
    assert res["checks"]["gain_statistic"] == direct(lambda s: s["aff"]["gain"] - 0.0)
    assert res["checks"]["gain_vs_rca"] == direct(lambda s: s["aff"]["gain"] - s["rca"]["gain"])
    assert res["secondary"] == direct(lambda s: s["aff"]["r1"] - s["r1"]["r1"])
    assert res["go"] is True and all(c["pass"] for c in res["checks"].values())


def test_go_checks_one_failure_is_no_go_and_secondary_never_changes_go():
    rng = np.random.default_rng(14)
    ps = _per_seed(rng)
    for s in ps:                                                  # AFF no better than B' on R@1
        s["Bp"]["r1"] = s["aff"]["r1"] + rng.normal(0, 0.05, size=len(s["cl"]))
        s["r1"]["r1"] = s["aff"]["r1"] + 1.0                      # secondary fails
    res = RS.go_checks(ps)
    assert res["checks"]["r1_vs_Bprime"]["pass"] is False and res["go"] is False
    assert res["secondary"]["pass"] is False
    ps2 = _per_seed(np.random.default_rng(15))
    for s in ps2:
        s["r1"]["r1"] = s["aff"]["r1"] + 1.0
    r2 = RS.go_checks(ps2)
    assert r2["secondary"]["pass"] is False and r2["go"] is True    # secondary failing does not change GO


def test_go_checks_asserts_counterpart_gain_is_exactly_zero():
    ps = _per_seed(np.random.default_rng(16))
    ps[1]["cf"]["gain"] = ps[1]["cf"]["gain"] + 1e-9
    with pytest.raises(AssertionError):
        RS.go_checks(ps)


def test_bar_info_pooled_chooses_comparator_once_over_pooled_episodes():
    ps = _per_seed(np.random.default_rng(17))
    v, info = RS.bar_info_pooled(ps)
    cat = lambda k: {m: np.concatenate([s[k][m] for s in ps]) for m in ps[0][k]}   # noqa: E731
    cl = np.concatenate([s["cl"] for s in ps])
    pi = np.concatenate([s["pair_index"] for s in ps])
    v2, info2 = C.bar_info(cat("aff"), cat("cf"), cat("Bp"), cat("B"), cl, pi)
    np.testing.assert_array_equal(v, v2)
    assert info["comparator"] == info2["comparator"] == "counterpart" and info["r1"] == info2["r1"]
    assert len(v) == 1800


def test_sensitivity_recovers_variance_components_and_formula():
    rng = np.random.default_rng(18)
    P_, m, s_a2, s_e2 = 100_000, 3, 4.0, 9.0
    cl = np.repeat(np.arange(P_), m)
    diff = rng.normal(0, np.sqrt(s_a2), size=P_)[cl] + rng.normal(0, np.sqrt(s_e2), size=len(cl))
    r = RS.sensitivity(diff, cl)
    assert r["sigma_a2"] == pytest.approx(s_a2, rel=0.05) and r["sigma_e2"] == pytest.approx(s_e2, rel=0.05)
    n = len(cl)
    se_true = np.sqrt((s_a2 * (9 * P_ * m * m - 6 * n) + s_e2 * 3 * n) / (3 * n) ** 2)
    assert r["SE"] == pytest.approx(se_true, rel=0.02)
    assert r["half_width"] == pytest.approx(1.96 * r["SE"], rel=1e-12)
    assert r["x"] == pytest.approx(2.80 * r["SE"], rel=1e-12)
    assert r["n0"] == pytest.approx(3.0, rel=1e-3)
    ci = cluster_bootstrap(diff, cl)["ci95"]
    assert r["seed42_half_width"] == 0.5 * (ci[1] - ci[0])


def test_sensitivity_clamps_negative_between_component_to_zero():
    rng = np.random.default_rng(19)
    cl = np.repeat(np.arange(500), 2)
    diff = rng.normal(size=len(cl))
    diff[1::2] = -diff[0::2]                                      # between-painting variance far below within
    r = RS.sensitivity(diff, cl)
    assert r["sigma_a2"] == 0.0 and r["SE"] > 0
