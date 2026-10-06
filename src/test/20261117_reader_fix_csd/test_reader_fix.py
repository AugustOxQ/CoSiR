"""Fast unit tests of the reader-fix code on synthetic data (no real data is loaded). Run from /project/CoSiR:
    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python -m pytest -q \
        src/test/20261117_reader_fix_csd/test_reader_fix.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import rc_core as K  # noqa: E402
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS  # noqa: E402
from src.eval.aspect_nested import _combine, _zdict, nested_cells  # noqa: E402
from src.eval.aspect_quick_checks import _require_condition_free  # noqa: E402


def rand_scores(rng, E=40, Kc=13, same_cond=False):
    base = {d: rng.normal(size=(E, Kc)).astype(np.float32) for d in DIRECTIONS}
    return {c: {d: (base[d] if same_cond else rng.normal(size=(E, Kc)).astype(np.float32)) for d in DIRECTIONS}
            for c in CONDITIONS}


def test_rule_sha_matches():
    C.assert_rule()


def test_sigma_matches_bruteforce_var_of_delta():
    rng = np.random.default_rng(0)
    E, H = 60_000, 3
    scale = np.array([0.5, 1.0, 2.0])
    sup = rng.normal(size=(E, 4, H)) * scale
    con = rng.normal(size=(E, 4, H)) * scale
    sig = C.sigma_from_agreements(sup, con)
    delta = sup.mean(axis=1) - con.mean(axis=1)
    brute = delta.std(axis=0, ddof=1)                       # Var(Delta) estimated across episodes
    assert np.allclose(sig, brute, rtol=0.02)
    assert np.allclose(sig, scale / np.sqrt(2), rtol=0.02)  # analytic: Var(Delta) = v / 2
    # the formula itself, term by term
    vs, vc = sup.var(axis=1, ddof=1), con.var(axis=1, ddof=1)
    assert np.array_equal(sig, np.sqrt(np.mean((vs + vc) / 4.0, axis=0)))


def test_scaled_delta_antisymmetry_and_margin():
    rng = np.random.default_rng(1)
    delta = rng.normal(size=(500, 4))
    sigma = np.array([0.3, 1.0, 2.0, 0.7])
    picks, margins, scaled = C.scaled_delta_picks(delta, sigma)
    assert np.array_equal(scaled["b"], -scaled["a"])
    assert np.array_equal(picks["a"], (delta / sigma).argmax(axis=1))
    assert np.array_equal(picks["b"], (delta / sigma).argmin(axis=1))      # pick under b = arg min under a
    for c in CONDITIONS:
        s = np.sort(scaled[c], axis=1)
        assert np.allclose(margins[c], s[:, -1] - s[:, -2]) and (margins[c] >= 0).all()
    # ties go to the first grouping
    p, _, _ = C.scaled_delta_picks(np.zeros((3, 4)), np.ones(4))
    assert (p["a"] == 0).all() and (p["b"] == 0).all()


def test_pair_agreements_and_match_and_terms():
    rng = np.random.default_rng(2)
    n_rows, E = 30, 5
    post = {}
    for h, k in (("x", 3), ("y", 4)):
        post[h] = {m: rng.dirichlet(np.ones(k), size=n_rows).astype(np.float32) for m in ("img", "txt")}
    si, st, ci, ct = (rng.integers(0, n_rows, size=(E, 4)) for _ in range(4))
    sup, con = C.pair_agreements(post, si, st, ci, ct, ("x", "y"))
    assert sup.shape == con.shape == (E, 4, 2)
    e, s, h = 2, 1, 1
    assert np.isclose(sup[e, s, h], float(post["y"]["img"][si[e, s]] @ post["y"]["txt"][st[e, s]]))
    assert np.isclose(con[e, s, 0], float(post["x"]["img"][ci[e, s]] @ post["x"]["txt"][ct[e, s]]))
    match = C.support_argmax_match(post, si, st, ("x", "y"))
    assert match.shape == (E, 4, 2) and match.dtype == bool
    assert match[e, s, h] == (post["y"]["img"][si[e, s]].argmax() == post["y"]["txt"][st[e, s]].argmax())
    stack = {d: rng.normal(size=(E, 2, 13)).astype(np.float32) for d in DIRECTIONS}
    picks = {"a": np.array([0, 1, 0, 1, 1]), "b": np.array([1, 1, 0, 0, 1])}
    T = C.hard_term(stack, picks)
    assert np.array_equal(T["a"]["i2t"][1], stack["i2t"][1, 1]) and np.array_equal(T["b"]["t2i"][3], stack["t2i"][3, 0])
    probs = {"a": np.array([[1.0, 0.0]] * E), "b": np.array([[0.25, 0.75]] * E)}
    X = C.expected_term(stack, probs)
    assert np.allclose(X["a"]["i2t"], stack["i2t"][:, 0], atol=1e-6)
    assert np.allclose(X["b"]["i2t"], 0.25 * stack["i2t"][:, 0] + 0.75 * stack["i2t"][:, 1], atol=1e-6)


def pa(r1):
    r1 = np.asarray(r1, dtype=np.float64)
    return {"r1": r1, "gain": r1 * 0, "other": r1 * 0, "swap": r1 * 0, "strict": r1 * 0}


def test_bar_comparator_choice_and_tie_order():
    hi, mid, lo = pa([1, 1, 0, 0]), pa([1, 0, 0, 0]), pa([0, 0, 0, 0])
    assert C.bar_comparator(pBp=hi, pc=mid, pB=lo)[0] == "B_prime"
    assert C.bar_comparator(pBp=mid, pc=hi, pB=lo)[0] == "counterpart"
    assert C.bar_comparator(pBp=lo, pc=mid, pB=hi)[0] == "B"
    same = pa([1, 0, 1, 0])
    assert C.bar_comparator(same, same, same)[0] == "B_prime"                  # ties: B', counterpart, B
    assert C.bar_comparator(pa([0, 0, 0, 0]), same, same)[0] == "counterpart"
    # full precision: a difference of one part in a million decides
    a, b = pa(np.full(1000, 0.5)), pa(np.full(1000, 0.5 + 1e-6))
    assert C.bar_comparator(a, b, a)[0] == "counterpart"
    # the bar margin uses that comparator, paired per anchor, per pair with the same comparator
    cl = np.repeat(np.arange(50), 4)
    pi = np.tile(np.arange(3), 200)[:200] % 3
    rng = np.random.default_rng(3)
    fused, cf, bp, b = (pa(rng.integers(0, 2, 200)) for _ in range(4))
    v, info = C.bar_info(fused, cf, bp, b, cl, pi)
    name, comp, _ = C.bar_comparator(bp, cf, b)
    assert info["comparator"] == name and np.array_equal(v, fused["r1"] - comp["r1"])
    assert set(info["per_pair_r1"]) == set(C.POOLED_ORDER)


def test_clears_bar_clauses_full_precision():
    ok = {"point": 0.5, "ci95": [0.01, 1.0]}
    g = {"point": 1.0, "ci95": [0.2, 1.8]}
    assert C.clears_bar(ok, g)["clears_bar"]
    assert not C.clears_bar({"point": 0.4999999999, "ci95": [0.01, 1.0]}, g)["clears_bar"]
    assert not C.clears_bar({"point": 0.6, "ci95": [0.0, 1.0]}, g)["clears_bar"]          # lower bound must be > 0
    assert not C.clears_bar(ok, {"point": 1.0, "ci95": [0.0, 1.8]})["clears_bar"]


def test_gcf_identical_under_both_conditions_and_condition_free():
    rng = np.random.default_rng(4)
    T = rand_scores(rng)
    zT = _zdict(T)
    margins = {"a": rng.random(40), "b": rng.random(40)}
    taus, n = K.thresholds(margins)
    assert n == 80 and taus == sorted(taus)
    g = K.gates(margins, taus)
    for k in range(4):
        gated = K.gated_terms(zT, g[k])
        G = K.g_cf(gated)
        _require_condition_free({c: {d: G[c][d].numpy() for d in DIRECTIONS} for c in CONDITIONS}, "G_cf")
        for d in DIRECTIONS:
            assert torch.equal(G["a"][d], G["b"][d])
            ref = 0.5 * (g[k]["a"][:, None] * zT["a"][d].numpy().astype(np.float64)
                         + g[k]["b"][:, None] * zT["b"][d].numpy().astype(np.float64))
            assert np.allclose(G["a"][d].numpy(), ref, atol=1e-6)
    assert g[0]["a"].all() and g[0]["b"].all()                      # tau_0 is the minimum: always open on this data
    assert (g[3]["a"].mean() + g[3]["b"].mean()) / 2 <= 0.26        # 75th percentile: about a quarter open


def test_tau0_cells_reproduce_ungated_combine_exactly():
    rng = np.random.default_rng(5)
    B, T = rand_scores(rng, same_cond=True), rand_scores(rng)
    zB, zT = _zdict(B), _zdict(T)
    margins = {"a": rng.random(40), "b": rng.random(40)}
    taus, _ = K.thresholds(margins)
    g = K.gates(margins, taus)
    gated0 = K.gated_terms(zT, g[0])
    cells = K.rc_cells()
    assert len(cells) == 224 and cells[:56] == [(0, u, a) for (u, a) in nested_cells()]
    assert cells[56][0] == 1 and cells[223][0] == 3                    # tau outer, then lambda_u, then lambda_a
    for (_, u, a) in cells[:56]:
        mine, ref = _combine(zB, zB, gated0, u, a), _combine(zB, zB, zT, u, a)
        assert all(np.array_equal(mine[c][d], ref[c][d]) for c in CONDITIONS for d in DIRECTIONS)
    # a closed gate removes the term: lambda_a > 0 with an all-closed gate equals lambda_a = 0
    closed = K.gated_terms(zT, {c: np.zeros(40, np.float32) for c in CONDITIONS})
    s1, s2 = _combine(zB, zB, closed, 1.0, 4.0), _combine(zB, zB, zT, 1.0, 0.0)
    assert all(np.array_equal(s1[c][d], s2[c][d]) for c in CONDITIONS for d in DIRECTIONS)


def test_rc_crossfit_tie_order_and_criterion():
    E = 20
    parity = np.arange(E) % 2
    n = 224
    fr1 = np.full((n, E), 0.25)
    fg = np.full((n, E), 0.1)
    cr1 = np.full((n, E), 0.25)
    ctrl = {0: (1.0, 0.2), 1: (1.0, 0.2)}
    assert K.select_fused(fr1, fg, ctrl, parity, range(n)) == {0: 0, 1: 0}          # ties go to the first cell
    assert K.select_cf(cr1, parity, range(n)) == {0: 0, 1: 0}
    # the min of (R@1 - control, gain) decides: cell 7 has higher R@1 but zero gain, cell 9 balances both
    fr1[7], fg[7] = 0.9, 0.0
    fr1[9], fg[9] = 0.4, 0.15
    assert K.select_fused(fr1, fg, ctrl, parity, range(n)) == {0: 9, 1: 9}
    # exact ties among later cells keep the first of them
    fr1[100], fg[100] = fr1[9], fg[9]
    assert K.select_fused(fr1, fg, ctrl, parity, range(n))[0] == 9
    cr1[50] = 0.5
    cr1[60] = 0.5
    assert K.select_cf(cr1, parity, range(n)) == {0: 50, 1: 50}
    assert K.select_cf(cr1, parity, range(55, n)) == {0: 60, 1: 60}              # restriction to a cell subset


def test_pick_statistics_and_shares():
    pi = np.repeat(np.arange(3), 4)
    cl = np.arange(12) // 2
    parts = C.CONFIGS["A1"]
    ti = C.told_index(parts, C.TOLD["A1"], pi)
    assert ti["a"][0] == 0 and ti["b"][0] == parts.index("csd") and ti["b"][8] == parts.index("image")
    acc, share = C.pick_statistics(ti, parts, C.TOLD["A1"], pi, cl)
    assert acc["correct_share"]["point"] == 100.0 and abs(share["overall"]["affect"] - 100 * 8 / 24) < 1e-9
    assert abs(sum(share["overall"].values()) - 100) < 1e-9
