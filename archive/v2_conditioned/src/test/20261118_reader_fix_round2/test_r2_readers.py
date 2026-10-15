"""Tests of r2_readers.py (R2 and R3's pure pieces, DECISION_RULE.md §4.3 and §4.4) on synthetic arrays and small slices
of the real D14 banks. Run from /project/CoSiR:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/20261118_reader_fix_round2/test_r2_readers.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2_common as R  # noqa: E402
import r2_readers as RR  # noqa: E402

rf, rbe = R.rf, R.rbe


# ------------------------------------------------------------------ synthetic half-readers (R2)

def _halves(H=3, F=6, n=900, seed=0):
    """Two half-readers as round 1's recipe builds them: StandardScaler on the half's data, multinomial LR."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    halves = []
    for j in range(2):
        y = np.repeat(np.arange(H), n // H)
        X = rng.normal(size=(len(y), F)) * (1 + j) + 3 * j
        X[:, :H] += 2.0 * np.eye(H)[y]
        sc = StandardScaler().fit(X)
        m = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000).fit(sc.transform(X), y)
        halves.append({"scaler": sc, "model": m, "C": 1.0})
    return halves


def _features(F=6, n=400, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, F)) * 1.5 + 0.7
    return {"a": a, "b": -a + rng.normal(size=(n, F)) * 0.1}


def _r1(halves, F, H):
    """R1 exactly as rb_eval.stage_config computes it."""
    return {c: rf.average_probs(rbe.half_reader_probs({"halves": halves}, F[c], H)) for c in RR.CONDITIONS}


def test_code_check_own_stats_no_em_equals_r1():
    halves, F = _halves(), _features()
    P1 = _r1(halves, F, 3)
    Pown = RR.reader_probs(halves, F, RR.scaler_stats(halves), 3)
    rec = RR.code_check(P1, Pown)
    assert rec["passed"]
    for c in RR.CONDITIONS:
        assert rec[c]["max_abs_diff"] <= 1e-12 and rec[c]["picks_identical"]
        np.testing.assert_array_equal(rf.picks_and_margins(Pown[c])[0], rf.picks_and_margins(P1[c])[0])


def test_code_check_stops_on_wrong_statistics():
    halves, F = _halves(), _features()
    P1 = _r1(halves, F, 3)
    st = RR.scaler_stats(halves)
    wrong = RR.reader_probs(halves, F, [st[0], st[0]], 3)          # half 0's statistics for both halves
    with pytest.raises(SystemExit):
        RR.code_check(P1, wrong)


def test_code_check_tolerance_is_1e12_absolute():
    halves, F = _halves(), _features()
    P1 = _r1(halves, F, 3)
    near = {c: P1[c].copy() for c in RR.CONDITIONS}
    near["a"][5, 1] += 5e-13                                          # within 1e-12, same picks
    assert RR.code_check(P1, near)["passed"]
    far = {c: P1[c].copy() for c in RR.CONDITIONS}
    far["b"][7, 0] += 1e-11                                           # beyond 1e-12
    with pytest.raises(SystemExit):
        RR.code_check(P1, far)


def test_code_check_requires_identical_picks():
    P1 = {"a": np.array([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]]), "b": np.array([[0.1, 0.8, 0.1], [0.4, 0.4, 0.2]])}
    P2 = {c: P1[c].copy() for c in RR.CONDITIONS}
    P2["a"][0] = [0.5 - 1e-13, 0.5 + 1e-13, 0.0]                      # within tolerance, pick 1 instead of 0
    with pytest.raises(SystemExit):
        RR.code_check(P1, P2)


def test_restandardise_stats_follow_the_rule():
    F = _features()
    F["a"][:, 2] = 4.0
    F["b"][:, 2] = 4.0                                                # a constant column: sigma42 exactly 0
    mu, sd, zero = RR.restandardise_stats(F["a"], F["b"])
    X = np.vstack([F["a"], F["b"]])
    np.testing.assert_array_equal(mu, X.mean(axis=0))
    want = X.std(axis=0)
    want[2] = 1.0
    np.testing.assert_array_equal(sd, want)
    assert zero == [2]


def test_r2_probabilities_use_mu42_sigma42_for_both_halves():
    halves, F = _halves(), _features()
    mu, sd, _ = RR.restandardise_stats(F["a"], F["b"])
    P = RR.reader_probs(halves, F, [(mu, sd), (mu, sd)], 3)
    for c in RR.CONDITIONS:
        want = np.mean([h["model"].predict_proba((F[c] - mu) / sd) for h in halves], axis=0)
        np.testing.assert_array_equal(P[c], want)


# ------------------------------------------------------------------ EM class-prior correction (R2)

def _gauss_posteriors(prior, n=200_000, seed=0):
    """Exact posteriors under a uniform training prior of three unit Gaussians at -2, 0, 2, on a sample drawn with
    ``prior``."""
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=n, p=prior)
    x = rng.normal(size=n) + np.array([-2.0, 0.0, 2.0])[y]
    ll = -0.5 * (x[:, None] - np.array([-2.0, 0.0, 2.0])[None, :]) ** 2
    P = np.exp(ll - ll.max(axis=1, keepdims=True))
    return P / P.sum(axis=1, keepdims=True), y


def _em_reference(P, pi_train, tol=1e-10, max_iter=10_000):
    """The rule's loop, written from its text (§4.3 b)."""
    pi, changes = np.asarray(pi_train, float).copy(), []
    for s in range(max_iter):
        num = P * pi / pi_train
        new = (num / num.sum(axis=1, keepdims=True)).mean(axis=0)
        changes.append(np.max(np.abs(new - pi)))
        if changes[-1] < tol or s + 1 == max_iter:
            return new, s + 1, changes
        pi = new


def test_em_recovers_a_known_prior():
    true = np.array([0.6, 0.3, 0.1])
    P, y = _gauss_posteriors(true)
    pi_hat, rec = RR.em_prior(P, np.full(3, 1 / 3))
    assert rec["converged_by_criterion"] and not rec["cap_reached"] and rec["n_iter"] < RR.EM_MAX_ITER
    assert np.max(np.abs(pi_hat - true)) < 0.01
    assert np.max(np.abs(pi_hat - np.bincount(y) / len(y))) < 0.01
    assert np.max(np.abs(P.mean(axis=0) - true)) > 0.05                # the unadapted mean does not recover it


def test_em_stops_by_the_rule_criterion():
    P, _ = _gauss_posteriors(np.array([0.5, 0.35, 0.15]), n=20_000, seed=3)
    pt = np.full(3, 1 / 3)
    pi_hat, rec = RR.em_prior(P, pt)
    ref, n_ref, changes = _em_reference(P, pt)
    assert rec["n_iter"] == n_ref and np.array_equal(pi_hat, ref)
    assert changes[-1] < 1e-10 <= changes[-2]                          # the first s that meets the criterion
    assert rec["last_max_abs_change"] == changes[-1]


def test_em_cap():
    P, _ = _gauss_posteriors(np.array([0.7, 0.2, 0.1]), n=5_000, seed=4)
    pt = np.full(3, 1 / 3)
    pi_hat, rec = RR.em_prior(P, pt, max_iter=3)
    ref, n_ref, changes = _em_reference(P, pt, max_iter=3)
    assert rec["n_iter"] == 3 == n_ref and rec["cap_reached"] and not rec["converged_by_criterion"]
    assert np.array_equal(pi_hat, ref) and changes[-1] >= 1e-10


def test_em_and_adapt_leave_probabilities_unchanged_at_the_training_prior():
    rng = np.random.default_rng(5)
    row = rng.dirichlet(np.ones(3), size=50)
    P = np.vstack([row[:, p] for p in ([0, 1, 2], [1, 2, 0], [2, 0, 1])])   # class-symmetric: mean P is uniform
    pt = np.full(3, 1 / 3)
    pi_hat, rec = RR.em_prior(P, pt)
    np.testing.assert_allclose(pi_hat, pt, rtol=0, atol=1e-15)
    np.testing.assert_allclose(RR.adapt(P, pi_hat, pt), P, rtol=0, atol=1e-15)
    skew = np.array([0.5, 0.3, 0.2])                                   # a non-uniform training prior
    np.testing.assert_allclose(RR.adapt(P, skew, skew), P, rtol=0, atol=1e-15)


def test_adapt_formula():
    P = np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
    pi, pt = np.array([0.5, 0.25, 0.25]), np.array([0.2, 0.4, 0.4])
    num = P * pi / pt
    np.testing.assert_array_equal(RR.adapt(P, pi, pt), num / num.sum(axis=1, keepdims=True))


# ------------------------------------------------------------------ R3: draws, impure banks, D(k), k*

def _toy_world(n_rows=400, n_paintings=12, N=300, seed=0):
    """Few paintings so that the anchor and image constraints bind often."""
    rng = np.random.default_rng(seed)
    paint = rng.integers(0, n_paintings, size=n_rows)
    rows = np.sort(rng.choice(n_rows, size=n_rows // 2, replace=False)).astype(np.int64)
    anchor = rng.choice(rows, size=N).astype(np.int64)
    base = {k: rng.choice(rows, size=(N, 4)).astype(np.int64) for kk in RR.PAIR_KEYS for k in kk}
    return paint, rows, anchor, base


def _draws_reference(A, Rw, paint, seed):
    """§4.4 b, written from the rule's text."""
    g = np.random.default_rng(seed)
    N = len(A)
    U = g.random((N, 2, 4))
    order = np.argsort(U, axis=2, kind="stable")
    img = Rw[g.integers(0, len(Rw), size=(N, 2, 4))]
    while True:
        bad = paint[img] == paint[A][:, None, None]
        if not bad.any():
            break
        img[bad] = Rw[g.integers(0, len(Rw), size=bad.sum())]
    cap = Rw[g.integers(0, len(Rw), size=(N, 2, 4))]
    while True:
        bad = (paint[cap] == paint[A][:, None, None]) | (paint[cap] == paint[img])
        if not bad.any():
            break
        cap[bad] = Rw[g.integers(0, len(Rw), size=bad.sum())]
    return order, img, cap


def test_draws_bit_reproducible_and_match_the_rule_text():
    paint, rows, anchor, _ = _toy_world()
    o1, i1, c1, info = RR.replacement_draws(anchor, rows, paint, 21_700)
    o2, i2, c2, _ = RR.replacement_draws(anchor, rows, paint, 21_700)
    for x, y in ((o1, o2), (i1, i2), (c1, c2)):
        assert x.dtype == np.int64 and x.shape == (len(anchor), 2, 4) and np.array_equal(x, y)
    ro, ri, rc = _draws_reference(anchor, rows, paint, 21_700)
    assert np.array_equal(o1, ro) and np.array_equal(i1, ri) and np.array_equal(c1, rc)
    assert info["img_redraw_rounds"] > 0 and info["cap_redraw_rounds"] > 0     # the constraints did bind
    o3, i3, c3, _ = RR.replacement_draws(anchor, rows, paint, 21_800)
    assert not (np.array_equal(o1, o3) and np.array_equal(i1, i3))


def test_draws_respect_the_painting_constraints():
    paint, rows, anchor, _ = _toy_world(seed=2)
    _, img, cap, _ = RR.replacement_draws(anchor, rows, paint, 7)
    pa = paint[anchor][:, None, None]
    assert (paint[img] != pa).all()
    assert (paint[cap] != pa).all()
    assert (paint[cap] != paint[img]).all()
    assert np.isin(img, rows).all() and np.isin(cap, rows).all()


def _mask_reference(order, k):
    m = np.zeros(order.shape, dtype=bool)
    for n in range(order.shape[0]):
        for s in range(2):
            for r in range(4 - k):
                m[n, s, order[n, s, r]] = True
    return m


def test_impure_banks_follow_the_rule_and_are_nested():
    paint, rows, anchor, base = _toy_world(seed=3)
    order, img, cap, _ = RR.replacement_draws(anchor, rows, paint, 11)
    banks = {k: RR.impure_pairs(base, order, img, cap, k) for k in RR.PURITIES}
    for k in RR.PURITIES:
        m = RR.replaced_mask(order, k)
        assert np.array_equal(m, _mask_reference(order, k))
        assert (m.sum(axis=2) == 4 - k).all()
        for s, (ki, kt) in enumerate(RR.PAIR_KEYS):
            ms = m[:, s, :]
            assert np.array_equal(banks[k][ki][ms], img[:, s, :][ms]) and np.array_equal(banks[k][kt][ms], cap[:, s, :][ms])
            assert np.array_equal(banks[k][ki][~ms], base[ki][~ms]) and np.array_equal(banks[k][kt][~ms], base[kt][~ms])
    for k in RR.PURITIES:                                    # a pair replaced at k is replaced identically at every k' < k
        mk = RR.replaced_mask(order, k)
        for k2 in RR.PURITIES:
            if k2 < k:
                assert (RR.replaced_mask(order, k2) | ~mk).all()
                for s, (ki, kt) in enumerate(RR.PAIR_KEYS):
                    ms = mk[:, s, :]
                    assert np.array_equal(banks[k][ki][ms], banks[k2][ki][ms])
                    assert np.array_equal(banks[k][kt][ms], banks[k2][kt][ms])


def test_purity_4_leaves_the_bank_unchanged():
    paint, rows, anchor, base = _toy_world(seed=4)
    order, img, cap, _ = RR.replacement_draws(anchor, rows, paint, 13)
    b4 = RR.impure_pairs(base, order, img, cap, 4)
    for kk in RR.PAIR_KEYS:
        for k in kk:
            assert np.array_equal(b4[k], base[k]) and b4[k].dtype == np.int64
    assert not RR.replaced_mask(order, 4).any()


def test_choose_k_ties_to_the_larger_k():
    assert RR.choose_k({1: 0.3, 2: 0.2, 3: 0.2, 4: 0.25}) == 3
    assert RR.choose_k({1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1}) == 4
    assert RR.choose_k({1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}) == 1
    assert RR.choose_k({1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}) == 4


def test_d_table_values_and_nonfinite_smd_stops():
    rng = np.random.default_rng(6)
    x42 = rng.normal(size=(500, 4))
    banks = {k: rng.normal(size=(800, 4)) + 0.1 * k for k in RR.PURITIES}
    tab = RR.d_table(x42, banks)
    for k in RR.PURITIES:
        np.testing.assert_array_equal(tab[k][0], rf.smd(x42, banks[k]))
        assert tab[k][1] == float(np.mean(np.abs(rf.smd(x42, banks[k]))))
    x_const, bank_const = x42.copy(), {k: v.copy() for k, v in banks.items()}
    x_const[:, 1] = 1.0
    bank_const[2][:, 1] = 2.0                                   # zero variance on both sides, different means -> NaN
    with pytest.raises(SystemExit):
        RR.d_table(x_const, bank_const)


def test_purity4_checks_are_exact():
    a = np.linspace(0, 1, 12).reshape(4, 3)
    assert RR.require_equal("x", a, a.copy())
    b = a.copy()
    b[2, 1] = np.nextafter(b[2, 1], 2.0)                        # one ulp
    with pytest.raises(SystemExit):
        RR.require_equal("x", b, a)
    with pytest.raises(SystemExit):
        RR.require_equal("x", a.astype(np.float32), a)
    names = ["f0", "f1", "f2"]
    s = np.array([0.1, -0.25, 0.0])
    stored = dict(zip(names, s.tolist()))
    assert RR.require_smd_equal(s, names, stored)
    off = dict(stored)
    off["f1"] = float(np.nextafter(-0.25, 0.0))
    with pytest.raises(SystemExit):
        RR.require_smd_equal(s, names, off)
    with pytest.raises(SystemExit):
        RR.require_smd_equal(s, names, {**stored, "f2": None})


def test_bank_matrix_layout_and_mean_abs_delta():
    rng = np.random.default_rng(8)
    parts = ("affect", "image", "caption")
    post = {h: {m: rng.dirichlet(np.ones(5), size=60).astype(np.float32) for m in ("img", "txt")} for h in parts}
    pairs = {k: rng.integers(0, 60, size=(20, 4)) for kk in RR.PAIR_KEYS for k in kk}
    ya, yb = np.zeros(20, np.int64), np.ones(20, np.int64)
    X, y, epi = RR.bank_matrix(post, parts, pairs, ya, yb)
    from types import SimpleNamespace
    F = rf.both_conditions(post, parts, SimpleNamespace(**pairs))
    np.testing.assert_array_equal(X, np.vstack([F["a"], F["b"]]))
    np.testing.assert_array_equal(y, np.r_[ya, yb])
    np.testing.assert_array_equal(epi, np.r_[np.arange(20), np.arange(20)])
    mad = RR.mean_abs_delta(X, parts)
    assert mad["image"] == float(np.mean(np.abs(X[:, 6 + 2])))
    idx = RR.subset_index(3, 10, 2)
    np.testing.assert_array_equal(idx, [0, 1, 10, 11, 20, 21])
    np.testing.assert_array_equal(RR.stacked_rows([0, 3], 5), [0, 3, 5, 8])


# ------------------------------------------------------------------ small slices of the real D14 banks

REAL = ["results/rb_halves.npz", "results/rb_halves.json", "results/rb_bank_A0_half0.npz",
        "results/rb_bank_A0_half0.json", "results/rb_reader_A0.npz", "results/rb_reader_A0.json"]


@pytest.fixture(scope="module")
def real_half0():
    if not all(R.r1_path(n).exists() for n in REAL):
        pytest.skip("round 1's D14 files are not present")
    R.assert_inputs(REAL)
    return np.load(R.r1_path("results/rb_halves.npz")), np.load(R.r1_path("results/rb_bank_A0_half0.npz"))


def test_real_half0_draws(real_half0):
    H, bank = real_half0
    A, Rw, paint = bank["anchor"], H["local_rows_half0"], H["painting_of_local_row"]
    assert len(A) == RR.BANK_N["A0"] and len(Rw) == RR.HALF_ROWS[0]
    order, img, cap, _ = RR.replacement_draws(A, Rw, paint, RR.R3_SEEDS[0])
    pa = paint[A][:, None, None]
    assert (paint[img] != pa).all() and (paint[cap] != pa).all() and (paint[cap] != paint[img]).all()
    assert (H["half_of_local_row"][img] == 0).all() and (H["half_of_local_row"][cap] == 0).all()
    o2, i2, c2, _ = RR.replacement_draws(A, Rw, paint, RR.R3_SEEDS[0])
    assert np.array_equal(order, o2) and np.array_equal(img, i2) and np.array_equal(cap, c2)
    base = {k: bank[k] for kk in RR.PAIR_KEYS for k in kk}
    b4 = RR.impure_pairs(base, order, img, cap, 4)
    assert all(np.array_equal(b4[k], base[k]) for k in base)


def test_real_half0_purity4_features_equal_round1(real_half0):
    """Purity 4 on a slice (first 200 episodes of each block) equals round 1's half0__X rows exactly."""
    H, bank = real_half0
    names = [f"results/rb_heads_{h}.npz" for h in ("affect", "image", "caption")]
    if not all(R.r1_path(n).exists() for n in names):
        pytest.skip("round 1's heads are not present")
    R.assert_inputs(names)
    parts = R.C.CONFIGS["A0"]
    post = {h: {m: np.load(R.r1_path(f"results/rb_heads_{h}.npz"))[m] for m in ("img", "txt")} for h in parts}
    blocks = [tuple(s.split("__")) for s in bank["block_pairs"].tolist()]
    N, n = len(bank["anchor"]), int(bank["block_size"])
    idx = RR.subset_index(len(blocks), n, 200)
    order, img, cap, _ = RR.replacement_draws(bank["anchor"], H["local_rows_half0"], H["painting_of_local_row"],
                                              RR.R3_SEEDS[0])
    base = {k: bank[k][idx] for kk in RR.PAIR_KEYS for k in kk}
    ya, yb = rf.bank_labels(blocks, n, parts)
    X4, y4, _ = RR.bank_matrix(post, parts, RR.impure_pairs(base, order[idx], img[idx], cap[idx], 4), ya[idx], yb[idx])
    rz = np.load(R.r1_path("results/rb_reader_A0.npz"))
    rows = RR.stacked_rows(idx, N)
    assert RR.require_equal("half0__X slice", X4, rz["half0__X"][rows])
    assert np.array_equal(y4, rz["half0__y"][rows])
    X1, _, _ = RR.bank_matrix(post, parts, RR.impure_pairs(base, order[idx], img[idx], cap[idx], 1), ya[idx], yb[idx])
    assert not np.array_equal(X1, X4)
