"""Tests of r6_stats (ticket 04; rule §3, §4, §6.5, §8.2, §8.5). Synthetic data of the real shapes only (36,864 pooled
held-shaped episodes over about 10,000 anchor paintings, 12,288 per seed, quarter-valued differences); no held row.

Check helpers `_check_*` take the module under test, so the mutation tests at the end run them against mutated copies
in tmp_path (never in place).
"""
import importlib.util
import inspect
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r6_stats as S  # noqa: E402

N_SEED = 12288          # 4,096 per pair x 3 pairs
N_HELD = 3 * N_SEED     # 36,864 pooled
SHA = "ab" * 32


# --------------------------------------------------------------------------------------------------------- helpers
def _clusters(rng, n, pool=12000):
    """Painting group ids (large, non-contiguous, as groups[anchor]); ~10,000 distinct over 36,864 draws."""
    ids = rng.choice(np.arange(10_000_000, 10_000_000 + 40 * pool, 40), size=pool, replace=False)
    return ids[rng.integers(0, pool, size=n)].astype(np.int64)


def _quarters(rng, n, p):
    """Mean of four Bernoulli(p) hits (two directions x two conditions): a multiple of 0.25 in [0, 1]."""
    return rng.binomial(4, p, size=n).astype(np.float64) / 4.0


def _per_anchor(r1, other):
    z = np.zeros_like(r1)
    return {"r1": r1, "gain": r1 - other, "other": other, "swap": z.copy(), "strict": z.copy()}


def _seed_scores(rng, cl):
    """A score_seed-shaped dict with the keys pass_record reads (fractions, float64)."""
    n = len(cl)
    out = {"cl": cl, "pair_index": np.repeat(np.arange(3), n // 3).astype(np.int64)}
    for key, p_hit, p_oth in (("aff_fused", 0.20, 0.06), ("cosine", 0.15, 0.12), ("rca", 0.16, 0.11),
                              ("B", 0.195, 0.07), ("B0", 0.197, 0.07), ("B1", 0.199, 0.07),
                              ("r1_fused", 0.1995, 0.065)):
        out[key] = _per_anchor(_quarters(rng, n, p_hit), _quarters(rng, n, p_oth))
    r1 = _quarters(rng, n, 0.19)
    out["aff_cf"] = _per_anchor(r1, r1.copy())          # condition-free: gain exactly 0
    return out


def _extra(seeds):
    return {"episodes_sha256": {s: {p: SHA for p in S.R6C.PAIR_NAMES} for s in seeds},
            "runner_sha256": SHA, "module_sha256": {"r6_stats.py": SHA}}


def _draws_independent(v, cl, n_boot=5000, seed=42, chunk=250):
    """The test's own re-derivation of cluster_bootstrap's resampling: (means, exact integer sums of quarters)."""
    _, idx = np.unique(cl, return_inverse=True)
    k = idx.max() + 1
    s = np.zeros(k)
    np.add.at(s, idx, v)
    c = np.bincount(idx, minlength=k).astype(float)
    q = np.zeros(k, dtype=np.int64)
    np.add.at(q, idx, np.rint(4 * v).astype(np.int64))
    rng = np.random.default_rng(seed)
    means, isum = [], []
    for start in range(0, n_boot, chunk):
        d = rng.integers(0, k, size=(min(chunk, n_boot - start), k))
        means.append(s[d].sum(1) / c[d].sum(1))
        isum.append(q[d].sum(1))
    return np.concatenate(means), np.concatenate(isum)


@pytest.fixture(scope="module")
def held_like():
    rng = np.random.default_rng(20261125)
    cl = _clusters(rng, N_HELD)
    v = _quarters(rng, N_HELD, 0.2) - _quarters(rng, N_HELD, 0.195)     # an R@1 difference near 0
    return v, cl


@pytest.fixture(scope="module")
def held_record():
    rng = np.random.default_rng(52)
    pool_cl = _clusters(rng, N_HELD)
    per_seed = [_seed_scores(rng, pool_cl[i * N_SEED:(i + 1) * N_SEED]) for i in range(3)]
    rec = S.pass_record(per_seed, "held", [52, 53, 54], _extra([52, 53, 54]))
    return per_seed, rec


# ------------------------------------------------------------------------------------------------ paths and imports
def test_main_and_earlier_modules_resolve_under_the_main_checkout():
    assert S.MAIN == Path("/project/CoSiR").resolve()
    assert Path(S._AM.__file__).resolve() == S.MAIN / "src/eval/aspect_metrics.py"
    assert Path(S.RS.__file__).resolve().parent == S.MAIN / "src/test/20261121_round3_affect_gate"
    assert Path(S.C.__file__).resolve().parent == S.MAIN / "src/test/20261117_reader_fix_csd"
    assert S._cluster_bootstrap is S._AM.cluster_bootstrap
    assert S.rule_sha256() == "7444a5e338838d837673b82b047c82b033eb1e2d1e0c3ed4e388dd8673070724"


# ------------------------------------------------------------------------------------------------------ the draws
def test_copy_reproduces_cluster_bootstrap_on_two_real_shaped_arrays(held_like):
    v, cl = held_like
    rng = np.random.default_rng(7)
    cl2 = _clusters(rng, N_SEED, pool=4000)                                       # one seed, ~3,800 paintings
    g_aff = _quarters(rng, N_SEED, 0.3) - _quarters(rng, N_SEED, 0.1)
    v2 = g_aff - (_quarters(rng, N_SEED, 0.25) - _quarters(rng, N_SEED, 0.15))  # a gain difference in [-2, 2]
    for vals, cls, k_min in ((v, cl, 9000), (v2, cl2, 3000)):
        b, int_le0 = S.bootstrap_draws(vals, cls)
        ref = S._AM.cluster_bootstrap(vals, cls)
        assert ref["n_clusters"] > k_min
        assert [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] == ref["ci95"]
        means, isum = _draws_independent(vals, cls)
        assert np.array_equal(b, means)                       # the draws themselves, bit for bit
        assert np.array_equal(int_le0, isum <= 0)
        assert b.dtype == np.float64 and b.shape == (5000,) and int_le0.dtype == bool
    assert not np.array_equal(v[:N_SEED], v2)


def _check_ci95_guard(mod, monkeypatch):
    """The assertion lives in the code: a reference that differs by a hair, or uses other draws, must fire it."""
    rng = np.random.default_rng(3)
    cl = _clusters(rng, N_SEED, pool=4000)
    v = _quarters(rng, N_SEED, 0.3) - _quarters(rng, N_SEED, 0.29)
    real = S._AM.cluster_bootstrap

    def shifted(*a, **k):
        r = real(*a, **k)
        return {**r, "ci95": [r["ci95"][0], np.nextafter(r["ci95"][1], 1.0)]}

    def other_seed(values, clusters, n_boot=5000, seed=42, chunk=250):
        return real(values, clusters, n_boot=n_boot, seed=seed + 1, chunk=chunk)

    for fake in (shifted, other_seed):
        monkeypatch.setattr(mod, "_cluster_bootstrap", fake)
        with pytest.raises(AssertionError, match="ci95"):
            mod.bootstrap_draws(v, cl)
    monkeypatch.setattr(mod, "_cluster_bootstrap", real)
    mod.bootstrap_draws(v, cl)


def test_ci95_assertion_fires_when_the_copy_diverges(monkeypatch):
    _check_ci95_guard(S, monkeypatch)


def test_integer_and_float_decisions_agree_away_from_zero(held_like):
    v, cl = held_like
    rng = np.random.default_rng(8)
    v0 = _quarters(rng, N_HELD, 0.2) - _quarters(rng, N_HELD, 0.2)        # a true difference of 0
    ns = []
    for vals in (v, v0, v + 0.25 * (np.arange(N_HELD) % 2 == 0), v - 0.25 * (np.arange(N_HELD) % 3 == 0)):
        b, int_le0 = S.bootstrap_draws(vals, cl)
        assert np.array_equal(int_le0, b <= 0)
        ns.append(int(int_le0.sum()))
    assert 100 < ns[1] < 4900                                # v0: many draws on both sides of 0
    assert ns[2] == 0 and ns[3] == 5000


def _check_rounding_hair(mod):
    """Real shape, all differences 0 except one painting at 0.25 + 2^-54 (a float hair above a quarter) and one at
    -0.25. A resample with as many draws of each has exact sum 0 (counts as <= 0) but a float sum a hair above 0."""
    rng = np.random.default_rng(11)
    cl = _clusters(rng, N_HELD)
    u = np.unique(cl)
    v = np.zeros(N_HELD)
    i_up = np.flatnonzero(cl == u[17])[0]
    i_dn = np.flatnonzero(cl == u[4242])[0]
    v[i_up] = np.nextafter(0.25, 1.0)
    v[i_dn] = -0.25
    assert v[i_up] - 0.25 == 2.0 ** -54
    b, int_le0 = mod.bootstrap_draws(v, cl)
    _, isum = _draws_independent(v, cl)
    assert np.array_equal(int_le0, isum <= 0)                # decided on the exact integer sums
    hair = (b > 0) & (isum == 0)
    assert hair.sum() > 100                                  # the float form says > 0 there; the integer form decides
    assert int(int_le0.sum()) == int((isum <= 0).sum()) > int((b <= 0).sum())


def test_a_float_hair_from_zero_is_decided_by_the_integer_form():
    _check_rounding_hair(S)


def _check_quarter_guard(mod):
    rng = np.random.default_rng(5)
    cl = _clusters(rng, N_SEED, pool=4000)
    v = _quarters(rng, N_SEED, 0.3) - _quarters(rng, N_SEED, 0.2)
    bad = v.copy()
    bad[123] = 0.1
    with pytest.raises(AssertionError, match="multiple of 0.25"):
        mod.bootstrap_draws(bad, cl)


def test_guards_on_the_values():
    _check_quarter_guard(S)
    rng = np.random.default_rng(6)
    cl = _clusters(rng, N_SEED, pool=4000)
    v = _quarters(rng, N_SEED, 0.3) - _quarters(rng, N_SEED, 0.2)
    with pytest.raises(AssertionError, match="fraction units"):
        S.bootstrap_draws(100 * v + 100 * (v == 0), cl)          # R@1 points instead of fractions
    nan = v.copy()
    nan[0] = np.nan
    with pytest.raises(AssertionError, match="finite"):
        S.bootstrap_draws(nan, cl)
    with pytest.raises(ValueError):
        S.bootstrap_draws(v, cl[:-1])
    with pytest.raises(ValueError):
        S.bootstrap_draws(v, np.zeros_like(cl))
    S.bootstrap_draws(v + 1e-12 * (np.arange(N_SEED) % 2), cl)   # float noise within QUARTER_TOL is accepted


# ------------------------------------------------------------------------------------------------------------ Holm
def _exact_holm(counts, names, m):
    """Holm on exact p = (n + 1)/5001 against 0.025/(m + 1 - k), with Fractions (the rule's definition)."""
    order = sorted(names, key=lambda nm: (counts[nm], names.index(nm)))
    out, ok = [], True
    for k, nm in enumerate(order, 1):
        own = Fraction(counts[nm] + 1, 5001) <= Fraction(25, 1000) / (m + 1 - k)
        ok = ok and own
        out.append((nm, k, ok, own))
    return out


def _view(rows):
    return [(r["name"], r["k"], r["passes"], r["own_count_passes"]) for r in rows]


def test_boundaries_and_near_boundary():
    assert [S.boundary(k, 7) for k in range(1, 8)] == [16, 19, 24, 30, 40, 61, 124]
    assert [S.boundary(k, 2) for k in (1, 2)] == [61, 124]
    for m in (7, 2):
        for k in range(1, m + 1):
            nb = S.boundary(k, m)
            assert [S.near_boundary(n, k, m) for n in (nb - 2, nb - 1, nb, nb + 1, nb + 2)] == [
                False, True, True, True, False]
    for bad in ((0, 7), (8, 7), (3, 2)):
        with pytest.raises(ValueError):
            S.boundary(*bad)


def _check_boundary_counts(mod):
    """For every rank k of both families: the check at rank k passes with n*_k and fails with n*_k + 1."""
    for names, m in ((S.CHECKS, 7), (S.SECONDARY, 2)):
        for k in range(1, m + 1):
            nb = S.boundary(k, m)
            for n, want in ((nb, True), (nb + 1, False)):
                counts = {nm: (0 if i < k - 1 else n if i == k - 1 else nb + 1000 + i) for i, nm in enumerate(names)}
                counts = {nm: min(c, 5000) for nm, c in counts.items()}
                row = mod.holm(counts, names, m)[k - 1]
                assert (row["name"], row["k"], row["n"]) == (names[k - 1], k, n)
                assert row["own_count_passes"] is want and row["passes"] is want, (names, k, n)
                assert row["level_two_sided"] == 1 - 0.05 / (m + 1 - k)


def test_holm_exact_boundary_counts_for_every_rank():
    _check_boundary_counts(S)


def test_holm_all_pass_first_rank_failure_and_middle_failure():
    P = S.CHECKS
    allpass = S.holm(dict(zip(P, (16, 19, 24, 30, 40, 61, 124))), P, 7)
    assert _view(allpass) == [(nm, k, True, True) for k, nm in enumerate(P, 1)]
    assert all(r["passes"] for r in S.holm(dict.fromkeys(P, 0), P, 7))

    first = S.holm(dict(zip(P, (30, 17, 20, 124, 61, 41, 18))), P, 7)    # rank 1 has 17 > 16
    assert _view(first) == [("P2", 1, False, False), ("P7", 2, False, True), ("P3", 3, False, True),
                            ("P1", 4, False, True), ("P6", 5, False, False), ("P5", 6, False, True),
                            ("P4", 7, False, True)]

    mid = S.holm(dict(zip(P, (0, 24, 40, 31, 5, 61, 124))), P, 7)         # rank 4 (P4) has 31 > 30
    assert _view(mid) == [("P1", 1, True, True), ("P5", 2, True, True), ("P2", 3, True, True),
                          ("P4", 4, False, False), ("P3", 5, False, True), ("P6", 6, False, True),
                          ("P7", 7, False, True)]

    last = S.holm(dict(zip(P, (0, 1, 2, 3, 4, 5, 125))), P, 7)            # rank 7 has 125 > 124
    assert _view(last) == [(nm, k, True, True) for k, nm in enumerate(P[:6], 1)] + [("P7", 7, False, False)]


def _check_ties(mod):
    """Ties keep the order P1..P7 (S1 before S2). A tie straddling a boundary shows which check gets the stricter
    rank: with P1 and P2 both at 17, P1 takes rank 1 (fails, 17 > 16) and P2 rank 2 (own count passes: not reached)."""
    P = S.CHECKS
    rows = mod.holm(dict(zip(P, (17, 17, 0, 0, 0, 0, 0))), P, 7)
    assert _view(rows) == [("P3", 1, True, True), ("P4", 2, True, True), ("P5", 3, True, True),
                           ("P6", 4, True, True), ("P7", 5, True, True), ("P1", 6, True, True),
                           ("P2", 7, True, True)]
    rows = mod.holm(dict(zip(P, (17, 17, 200, 200, 200, 200, 200))), P, 7)
    assert _view(rows)[:2] == [("P1", 1, False, False), ("P2", 2, False, True)]
    rows = mod.holm(dict(zip(P, (9, 9, 9, 9, 9, 9, 9))), P, 7)
    assert [r["name"] for r in rows] == list(P)
    rows = mod.holm({"S1": 62, "S2": 62}, S.SECONDARY, 2)
    assert _view(rows) == [("S1", 1, False, False), ("S2", 2, False, True)]
    rows = mod.holm({"S2": 3, "S1": 3}, S.SECONDARY, 2)
    assert [r["name"] for r in rows] == ["S1", "S2"]


def test_holm_ties_follow_the_given_order():
    _check_ties(S)


def test_holm_matches_the_exact_fraction_form_on_random_count_sets():
    rng = np.random.default_rng(99)
    for names, m in ((S.CHECKS, 7), (S.SECONDARY, 2)):
        for _ in range(3000):
            hi = int(rng.choice([20, 70, 130, 5001]))
            counts = {nm: int(c) for nm, c in zip(names, rng.integers(0, hi, size=m))}
            assert _view(S.holm(counts, names, m)) == _exact_holm(counts, names, m)


def _check_integer_form_in_source(mod):
    """The float form p <= 0.025/(m + 1 - k) gives the same answers as the integer form for every count (5,001 is odd,
    so no exact tie exists, and the nearest case is 1/5,001 apart), so only the source can show which one decides."""
    src = inspect.getsource(mod.holm)
    assert "own = 40 * (n + 1) * (m + 1 - k) <= 5001" in src
    assert "/ 5001" not in src and "0.025" not in src.split("own =")[1].split("\n")[0]


def test_holm_decides_in_integers():
    _check_integer_form_in_source(S)


def test_holm_input_guards():
    P = S.CHECKS
    with pytest.raises(ValueError):
        S.holm(dict.fromkeys(P[:6], 0), P, 7)
    with pytest.raises(ValueError):
        S.holm({**dict.fromkeys(P, 0), "P1": 3.0}, P, 7)
    with pytest.raises(ValueError):
        S.holm({**dict.fromkeys(P, 0), "P1": True}, P, 7)
    with pytest.raises(ValueError):
        S.holm({**dict.fromkeys(P, 0), "P1": 5001}, P, 7)
    with pytest.raises(ValueError):
        S.holm(dict.fromkeys(P, 0), P, 2)


# ----------------------------------------------------------------------------------------------------- pass record
CHECK_KEYS = {"quantity", "n", "point", "ci95", "holm_k", "ci_holm", "level_two_sided", "passes",
              "own_count_passes", "near_boundary"}
SECONDARY_KEYS = CHECK_KEYS - {"passes", "own_count_passes"}
TOP_KEYS = {"rule_sha256", "mode", "seeds", "n_episodes", "n_clusters", "checks", "holm_order", "secondary",
            "episodes_sha256", "runner_sha256", "module_sha256", "time"}


def test_pass_record_fills_every_field_of_the_contract(held_record):
    per_seed, rec = held_record
    assert set(rec) == TOP_KEYS
    assert rec["mode"] == "held" and rec["seeds"] == [52, 53, 54] and rec["n_episodes"] == N_HELD
    cl = np.concatenate([s["cl"] for s in per_seed])
    assert rec["n_clusters"] == len(np.unique(cl)) > 9000
    assert rec["rule_sha256"] == S.R6C.RULE_SHA256
    assert list(rec["checks"]) == list(S.CHECKS) and list(rec["secondary"]) == list(S.SECONDARY)
    assert sorted(rec["holm_order"]) == sorted(S.CHECKS)
    assert set(rec["episodes_sha256"]) == {"52", "53", "54"}
    assert json.loads(json.dumps(rec)) == rec                # plain JSON types throughout
    diffs = S.check_diffs(per_seed)
    rows = {r["name"]: r for r in S.holm({c: rec["checks"][c]["n"] for c in S.CHECKS}, S.CHECKS, 7)}
    assert [r["name"] for r in S.holm({c: rec["checks"][c]["n"] for c in S.CHECKS}, S.CHECKS, 7)] == rec["holm_order"]
    for fam, keys, m in ((S.CHECKS, CHECK_KEYS, 7), (S.SECONDARY, SECONDARY_KEYS, 2)):
        for c in fam:
            r = rec["checks" if m == 7 else "secondary"][c]
            assert set(r) == keys, c
            d, cl_c = diffs[c]
            b, int_le0 = S.bootstrap_draws(d, cl_c)
            assert r["n"] == int(int_le0.sum()) and isinstance(r["n"], int)
            ref = S.C.point_ci(d, cl_c)
            assert r["point"] == ref["point"] and r["ci95"] == ref["ci95"]
            k = r["holm_k"]
            lo_q, hi_q = 100 * 0.025 / (m + 1 - k), 100 * (1 - 0.025 / (m + 1 - k))
            assert r["ci_holm"] == [100 * float(np.percentile(b, lo_q)), 100 * float(np.percentile(b, hi_q))]
            assert r["level_two_sided"] == 1 - 0.05 / (m + 1 - k)
            assert r["near_boundary"] == (abs(r["n"] - S.boundary(k, m)) <= 1)
            if k == m:
                assert r["ci_holm"] == r["ci95"]
            else:
                assert r["ci_holm"][0] <= r["ci95"][0] and r["ci_holm"][1] >= r["ci95"][1]
            if m == 7:
                assert (r["passes"], r["own_count_passes"]) == (rows[c]["passes"], rows[c]["own_count_passes"])
    aff = np.concatenate([s["aff_fused"]["gain"] for s in per_seed])
    assert np.array_equal(diffs["P6"][0], aff)                # P6 = AFF's gain (CF's is 0)
    rca = np.concatenate([s["rca"]["r1"] for s in per_seed])
    assert np.array_equal(diffs["P2"][0], np.concatenate([s["aff_fused"]["r1"] for s in per_seed]) - rca)
    assert {c: q for c, (_, _, q) in S.QUANTITIES.items()} == {c: rec[f][c]["quantity"] for f, fam in (
        ("checks", S.CHECKS), ("secondary", S.SECONDARY)) for c in fam}


def test_pass_record_regression_and_smoke_modes():
    rng = np.random.default_rng(42)
    s42 = _seed_scores(rng, _clusters(rng, N_SEED, pool=4000))
    rec = S.pass_record([s42], "regression", [42], _extra([42]))
    assert rec["mode"] == "regression" and rec["seeds"] == [42] and rec["n_episodes"] == N_SEED
    smoke = [_seed_scores(rng, _clusters(rng, 192, pool=300)) for _ in range(3)]
    rec = S.pass_record(smoke, "smoke", [9001, 9002, 9003], _extra([9001, 9002, 9003]))
    assert rec["n_episodes"] == 576 and set(rec) == TOP_KEYS
    with pytest.raises(AssertionError, match="reads seeds"):
        S.pass_record(smoke, "held", [9001, 9002, 9003], _extra([9001, 9002, 9003]))
    with pytest.raises(AssertionError, match="episodes"):
        S.pass_record(smoke, "held", [52, 53, 54], _extra([52, 53, 54]))
    with pytest.raises(ValueError):
        S.pass_record([s42], "verdict", [42], _extra([42]))
    with pytest.raises(ValueError):
        S.pass_record([s42], "regression", [42], _extra([43]))
    with pytest.raises(ValueError):
        S.pass_record([s42], "regression", [42], {**_extra([42]), "runner_sha256": "xyz"})
    with pytest.raises(ValueError):
        S.pass_record([s42], "regression", [42], {**_extra([42]), "time": "now"})


def _check_cf_refusal(mod):
    rng = np.random.default_rng(1)
    per_seed = [_seed_scores(rng, _clusters(rng, N_SEED, pool=4000))]
    mod.check_diffs(per_seed)
    per_seed[0]["aff_cf"]["gain"] = per_seed[0]["aff_cf"]["gain"].copy()
    per_seed[0]["aff_cf"]["gain"][4097] = 0.25
    with pytest.raises(AssertionError, match="CF's condition gain"):
        mod.check_diffs(per_seed)


def test_pass_record_refuses_a_cf_gain_that_is_not_zero():
    _check_cf_refusal(S)
    rng = np.random.default_rng(2)
    s42 = _seed_scores(rng, _clusters(rng, N_SEED, pool=4000))
    s42["aff_cf"]["gain"] = s42["aff_cf"]["gain"] + 1e-300 * (np.arange(N_SEED) == 7)
    with pytest.raises(AssertionError, match="CF's condition gain"):
        S.pass_record([s42], "regression", [42], _extra([42]))


# ----------------------------------------------------------------------------------------------------- sensitivity
def _hand_sigma(d_pp, cl):
    groups = {}
    for x, c in zip(d_pp.tolist(), cl.tolist()):
        groups.setdefault(c, []).append(x)
    n, P = len(d_pp), len(groups)
    grand = sum(d_pp.tolist()) / n
    ssw = sum(sum((x - sum(g) / len(g)) ** 2 for x in g) for g in groups.values())
    ssb = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups.values())
    n0 = (n - sum(len(g) ** 2 for g in groups.values()) / n) / (P - 1)
    se2 = ssw / (n - P)
    return max(0.0, (ssb / (P - 1) - se2) / n0), se2


def test_sigma_split_equals_round3_sensitivity_parts(held_like):
    v, cl = held_like
    rng = np.random.default_rng(12)
    seed_cl = cl[:N_SEED]
    painting_effect = {c: 0.25 * rng.integers(-1, 2) for c in np.unique(seed_cl)}
    d = v[:N_SEED] + np.array([painting_effect[c] for c in seed_cl]) * (rng.random(N_SEED) < 0.5)
    got = S.sigma_split(d, seed_cl)
    ref = S.RS.sensitivity(d, seed_cl)
    assert got == {"sigma_a2": ref["sigma_a2"], "sigma_eps2": ref["sigma_e2"]}
    a2, e2 = _hand_sigma(100.0 * d, seed_cl)                    # R@1 points squared
    assert got["sigma_a2"] == pytest.approx(a2, rel=1e-9) and got["sigma_a2"] > 0
    assert got["sigma_eps2"] == pytest.approx(e2, rel=1e-9)
    with pytest.raises(AssertionError):
        S.sigma_split(100.0 * d + 3, seed_cl)


def test_detectable_matches_a_hand_computation_with_unequal_counts(held_like):
    sig = {"sigma_a2": 2.0, "sigma_eps2": 100.0}
    M = np.array([1, 2, 3], dtype=np.int64)              # sum M^2 = 14, N = 6
    se = math.sqrt((2.0 * 14 + 100.0 * 6) / 36)
    assert S.detectable(sig, M, 6) == {"SE": se, "x": 3.532 * se, "x95": 2.80 * se}
    assert S.detectable(sig, M, 6, family="S") == {"SE": se, "x2": 3.083 * se, "x95": 2.80 * se}
    _, cl = held_like
    M = np.unique(cl, return_counts=True)[1]
    assert M.min() != M.max() and M.sum() == N_HELD
    sig = {"sigma_a2": 3.7, "sigma_eps2": 1234.5}
    se = math.sqrt((3.7 * sum(int(m) ** 2 for m in M) + 1234.5 * N_HELD) / N_HELD ** 2)
    got = S.detectable(sig, M, N_HELD)
    assert got["SE"] == pytest.approx(se, rel=1e-12) and got["x"] == pytest.approx(3.532 * se, rel=1e-12)
    assert got["x95"] == pytest.approx(2.80 * se, rel=1e-12)
    with pytest.raises(AssertionError):
        S.detectable(sig, M, N_HELD - 1)
    with pytest.raises(ValueError):
        S.detectable(sig, M.astype(float), N_HELD)
    with pytest.raises(ValueError):
        S.detectable(sig, M, N_HELD, family="Q")


# --------------------------------------------------------------------------------------------- mutations (copies)
_HARNESS = "HERE = Path(__file__).resolve().parent"
MUTANTS = {
    "ci95 assertion removed": (
        'if got != ref["ci95"] or ref["n_clusters"] != k or b.shape != (n_boot,) or float(v.mean()) != ref["point"]:',
        "if False:", "ci95"),
    "tie order flipped": ("names.index(nm)))", "-names.index(nm)))", "ties"),
    "float p <= 0.025/(8-k)": ("own = 40 * (n + 1) * (m + 1 - k) <= 5001", "own = (n + 1) / 5001 <= 0.025 / (8 - k)",
                               "boundary"),
    "float p <= 0.025/(m+1-k)": ("own = 40 * (n + 1) * (m + 1 - k) <= 5001",
                                 "own = (n + 1) / 5001 <= 0.025 / (m + 1 - k)", "source"),
    "naive p = n/5000": ("own = 40 * (n + 1) * (m + 1 - k) <= 5001", "own = n / 5000 <= 0.025 / (m + 1 - k)",
                         "boundary"),
    "float sign decision": ("le0.append(isums[draws].sum(axis=1) <= 0)",
                            "le0.append(sums[draws].sum(axis=1) / counts[draws].sum(axis=1) <= 0)", "hair"),
    "quarter assertion removed": ("if np.abs(4.0 * v - q).max() > QUARTER_TOL:", "if False:", "quarter"),
    "CF refusal removed": ('if not np.all(np.asarray(s[CF]["gain"], dtype=np.float64) == 0):', "if False:", "cf"),
}
CHECKERS = {"ci95": _check_ci95_guard, "ties": _check_ties, "boundary": _check_boundary_counts,
            "source": _check_integer_form_in_source, "hair": _check_rounding_hair, "quarter": _check_quarter_guard,
            "cf": _check_cf_refusal}


def _load_mutant(tmp_path, name, old, new):
    src = Path(S.__file__).read_text()
    assert src.count(old) == 1, f"mutation site of {name!r} not found exactly once"
    assert src.count(_HARNESS) == 1
    src = src.replace(old, new).replace(_HARNESS, f"HERE = Path({str(HERE)!r})")
    path = tmp_path / "r6_stats_mutant.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(f"r6_stats_mutant_{abs(hash(name))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", list(MUTANTS))
def test_each_mutant_on_a_copy_is_caught(tmp_path, monkeypatch, name):
    old, new, which = MUTANTS[name]
    check = CHECKERS[which]
    args = (monkeypatch,) if which == "ci95" else ()
    check(S, *args)                                          # the real module passes the check
    mutant = _load_mutant(tmp_path, name, old, new)
    assert Path(mutant.__file__).parent == tmp_path and Path(S.__file__).parent == HERE
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        check(mutant, *args)
