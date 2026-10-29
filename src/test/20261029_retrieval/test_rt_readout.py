"""Tests for rt_readout: menu, whitening, cross-fit, selection and the round log's Numeric convention."""
import numpy as np
import pytest

import rt_metrics
import rt_readout as ro_mod
from rt_readout import R0, ReadOut

ASPECTS = ('emotion', 'style', 'genre')


def unit_rows(rng, n, d):
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# ---- ReadOut, menu, name ------------------------------------------------------------------------------------------

def test_r0_and_names():
    assert R0 == ReadOut('none', None, 0)
    assert ro_mod.name(R0) == 'none+k0'
    assert ro_mod.name(ReadOut('center', None, 5)) == 'center+k5'
    assert ro_mod.name(ReadOut('whiten', 256, 3)) == 'whiten256+k3'


def test_readout_rejects_invalid_fields():
    with pytest.raises(AssertionError):
        ReadOut('whiten', None, 0)
    with pytest.raises(AssertionError):
        ReadOut('center', 64, 0)
    with pytest.raises(AssertionError):
        ReadOut('none', None, 2)
    with pytest.raises(AssertionError):
        ReadOut('pca', None, 0)


@pytest.mark.parametrize('width,sizes', [(2048, {64, 256, 1024}), (1152, {64, 256, 1024}), (768, {64, 256, 768})])
def test_menu_wide_arms_have_25(width, sizes):
    m = ro_mod.menu(width)
    assert len(m) == 25 and len(set(m)) == 25
    assert {r.size for r in m if r.w == 'whiten'} == sizes
    assert {r.k for r in m} == {0, 1, 3, 5, 10}
    assert {r.w for r in m} == {'none', 'center', 'whiten'}


@pytest.mark.parametrize('width', [40, 50])
def test_menu_narrow_arms_have_15(width):
    m = ro_mod.menu(width)
    assert len(m) == 15 and len(set(m)) == 15
    assert {r.size for r in m if r.w == 'whiten'} == {width}


def test_menu_size_equal_to_width_is_whiten_at_width():
    # 1024 >= 1024: whiten at 1024 (the full width); 64 and 256 stay.
    m = ro_mod.menu(1024)
    assert {r.size for r in m if r.w == 'whiten'} == {64, 256, 1024}
    assert len(m) == 25


def test_menu_is_in_tie_order_and_starts_with_r0():
    m = ro_mod.menu(2048)
    assert m[0] == R0
    assert m == sorted(m, key=ro_mod.tie_key)
    # Hand check of the first 6 and the first expansion entries.
    assert [ro_mod.name(r) for r in m[:5]] == ['none+k0', 'center+k0', 'whiten1024+k0', 'whiten256+k0', 'whiten64+k0']
    assert [ro_mod.name(r) for r in m[5:8]] == ['none+k1', 'none+k3', 'none+k5']
    assert ro_mod.name(m[-1]) == 'whiten64+k10'


# ---- fit and apply ------------------------------------------------------------------------------------------------

def anisotropic(rng, n, d):
    a = rng.standard_normal((d, d))
    return (rng.standard_normal((n, d)) @ a + rng.standard_normal(d) * 3).astype(np.float32)


def test_whitening_gives_identity_covariance_on_kept_dims():
    rng = np.random.default_rng(0)
    x = anisotropic(rng, 4000, 12)
    p = ro_mod.fit(ReadOut('whiten', 8, 0), x)
    z = ro_mod._transform(p, x)  # before the L2 normalization
    assert z.shape == (4000, 8)
    cov = np.cov(z.astype(np.float64), rowvar=False)
    # eps = 1e-6 * lambda_1 shrinks each variance to lambda_i / (lambda_i + eps): identity to ~1e-5 here.
    assert np.allclose(cov, np.eye(8), atol=1e-4)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-4)


def test_whitening_formula_and_epsilon_on_axis_aligned_data():
    # Exactly uncorrelated columns with variances 9, 4, 1e-5 and 0 (a constant column), in shuffled column order.
    # Eigenvalues are those variances (n-1 convention) and U is a signed permutation, so for the eigen-direction
    # with variance v the output is +-(x_col - mu_col) / sqrt(v + 1e-6 * 9).
    rng = np.random.default_rng(1)
    n = 500
    q, _ = np.linalg.qr(rng.standard_normal((n, 3)) - rng.standard_normal((n, 3)).mean(axis=0))
    q = q - q.mean(axis=0)
    q, _ = np.linalg.qr(q)  # orthonormal and zero-mean columns (the mean direction is removed first)
    q = q * np.sqrt(n - 1)  # unit variance in the n-1 convention, exactly uncorrelated
    cols = np.concatenate([q * np.array([3.0, 2.0, np.sqrt(1e-5)]), np.full((n, 1), 5.0)], axis=1)  # variances 9, 4, 1e-5, 0
    x = cols[:, [2, 3, 0, 1]].astype(np.float32)  # variances 1e-5, 0, 9, 4
    p = ro_mod.fit(ReadOut('whiten', 4, 0), x)
    out = ro_mod._transform(p, x)
    xc = x.astype(np.float64) - x.astype(np.float64).mean(axis=0)
    eps = 1e-6 * 9.0
    for i, (col, v) in enumerate([(2, 9.0), (3, 4.0), (0, 1e-5)]):  # descending eigenvalues
        assert xc[:, col].var(ddof=1) == pytest.approx(v, rel=1e-3)
        # v = 1e-5 is of the order of eps (9e-6): dropping or mis-scaling eps changes this column by ~40%
        assert np.allclose(np.abs(out[:, i]), np.abs(xc[:, col]) / np.sqrt(v + eps), rtol=1e-3, atol=1e-3)
    assert np.abs(out[:, 3]).max() < 1e-3  # the constant column carries no variance: output ~0, not blown up


def test_whiten_keeps_the_leading_eigen_directions():
    rng = np.random.default_rng(2)
    x = anisotropic(rng, 800, 10)
    p4 = ro_mod.fit(ReadOut('whiten', 4, 0), x)
    p10 = ro_mod.fit(ReadOut('whiten', 10, 0), x)
    assert np.allclose(ro_mod._transform(p4, x), ro_mod._transform(p10, x)[:, :4], atol=1e-6)


def test_center_subtracts_the_fit_mean_then_normalizes():
    rng = np.random.default_rng(3)
    fit_x = unit_rows(rng, 100, 6) + np.float32(2.0)
    new_x = unit_rows(rng, 20, 6)
    p = ro_mod.fit(ReadOut('center', None, 0), fit_x)
    out = ro_mod.apply(p, new_x)
    ref = new_x.astype(np.float64) - fit_x.astype(np.float64).mean(axis=0)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    assert out.dtype == np.float32
    assert np.allclose(out, ref, atol=1e-6)


def test_apply_output_is_float32_and_unit_norm_for_every_w():
    rng = np.random.default_rng(4)
    x = anisotropic(rng, 300, 9)
    for ro in (R0, ReadOut('center', None, 0), ReadOut('whiten', 5, 0), ReadOut('whiten', 9, 0)):
        out = ro_mod.apply(ro_mod.fit(ro, x), x)
        assert out.dtype == np.float32 and out.shape == (300, 9 if ro.size != 5 else 5)
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_none_is_the_normalized_input():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((30, 7)).astype(np.float32) * 3
    out = ro_mod.apply(ro_mod.fit(R0, x), x)
    assert np.allclose(out, x / np.linalg.norm(x, axis=1, keepdims=True), atol=1e-6)


def test_numeric_convention_projects_then_scales_and_casts_only_after_normalization():
    rng = np.random.default_rng(25)
    x = anisotropic(rng, 80, 12)
    new = anisotropic(rng, 30, 12)
    mu, lam, u = ro_mod._eig(x)
    params = ro_mod.fit(ReadOut('whiten', 8, 0), x, _eig_cache=(mu, lam, u))
    want = ((new.astype(np.float64) - mu) @ u[:, :8]) / np.sqrt(lam[:8] + 1e-6 * lam[0])
    np.testing.assert_array_equal(ro_mod._transform(params, new), want)
    want /= np.linalg.norm(want, axis=1, keepdims=True)
    np.testing.assert_array_equal(ro_mod.apply(params, new), want.astype(np.float32))


def test_fit_rejects_size_above_width():
    x = np.random.default_rng(6).standard_normal((50, 8)).astype(np.float32)
    with pytest.raises(AssertionError):
        ro_mod.fit(ReadOut('whiten', 9, 0), x)


def test_stage5_apply_fits_on_validation_and_applies_to_held():
    rng = np.random.default_rng(7)
    x_val = anisotropic(rng, 400, 8)
    x_held = anisotropic(rng, 100, 8)
    ro = ReadOut('whiten', 5, 3)
    out = ro_mod.stage5_apply(ro, x_val, x_held)
    assert np.array_equal(out, ro_mod.apply(ro_mod.fit(ro, x_val), x_held))
    assert out.dtype == np.float32 and out.shape == (100, 5)


# ---- halves and cross-fit -----------------------------------------------------------------------------------------

def test_halves_by_parity_of_position():
    even, odd = ro_mod.halves(7)
    assert even.tolist() == [0, 2, 4, 6] and odd.tolist() == [1, 3, 5]
    even, odd = ro_mod.halves(6152)
    assert len(even) == len(odd) == 3076
    assert sorted(np.concatenate([even, odd]).tolist()) == list(range(6152))


def toy(rng, n=120, d=10, n_seeds=5, per_aspect=True):
    """xs[(aspect, seed)] and labels[aspect]; genre has unlabelled rows. Rows are anisotropic, off-centre."""
    labels = {'emotion': rng.integers(0, 3, size=n), 'style': rng.integers(0, 4, size=n),
              'genre': rng.integers(0, 3, size=n)}
    labels['genre'][rng.random(n) < 0.3] = -1
    labels = {a: v.astype(np.int64) for a, v in labels.items()}
    seeds = [None] if n_seeds == 0 else list(range(n_seeds))
    xs = {}
    for s in seeds:
        base = anisotropic(rng, n, d) / 3 + 1
        for a in ASPECTS:
            xs[(a, s)] = (anisotropic(rng, n, d) / 3 + 1) if per_aspect else base
    return xs, labels


def manual_crossfit(xs, labels, ro):
    """The rule written out directly: fit on the other half (all rows), apply to this half, retrieval, mean."""
    even, odd = ro_mod.halves(len(next(iter(labels.values()))))
    out = {}
    for a in ASPECTS:
        per_seed = []
        for s in sorted({k[1] for k in xs}, key=lambda v: -1 if v is None else v):
            x = xs[(a, s)]
            vals = []
            for fit_idx, sc_idx in ((odd, even), (even, odd)):
                params = ro_mod.fit(ro, x[fit_idx])
                xh = ro_mod.apply(params, x[sc_idx])
                vals.append(rt_metrics.retrieval(xh, labels[a][sc_idx], qe_k=ro.k, workers=1)['ap'].mean())
            per_seed.append(np.mean(vals))
        out[a] = float(np.mean(per_seed))
    return out


@pytest.mark.parametrize('ro', [R0, ReadOut('center', None, 0), ReadOut('whiten', 6, 0), ReadOut('whiten', 10, 3),
                                ReadOut('center', None, 5)])
def test_crossfit_matches_the_rule_written_out(ro):
    rng = np.random.default_rng(10)
    xs, labels = toy(rng)
    got = ro_mod.crossfit(xs, labels, ro, workers=1)
    want = manual_crossfit(xs, labels, ro)
    assert set(got) == set(ASPECTS)
    for a in ASPECTS:
        assert got[a] == pytest.approx(want[a], abs=1e-12)


def test_crossfit_scores_the_half_it_did_not_fit_on():
    # A whitening fit on the scored half itself would give a different number than the fit on the other half.
    rng = np.random.default_rng(11)
    xs, labels = toy(rng)
    ro = ReadOut('whiten', 6, 0)
    even, odd = ro_mod.halves(120)
    x, lab = xs[('style', 0)], labels['style']
    in_sample = np.mean([rt_metrics.retrieval(ro_mod.apply(ro_mod.fit(ro, x[h]), x[h]), lab[h], workers=1)['ap'].mean()
                         for h in (even, odd)])
    got = ro_mod.crossfit(xs, labels, ro, workers=1)['style']
    assert got != pytest.approx(in_sample, abs=1e-6)
    assert got == pytest.approx(manual_crossfit(xs, labels, ro)['style'], abs=1e-12)


def test_crossfit_fit_uses_unlabelled_rows_too():
    # genre has -1 rows; the fit set is every painting of the other half (the manual rule fits on x[fit_idx]).
    rng = np.random.default_rng(12)
    xs, labels = toy(rng)
    assert (labels['genre'] < 0).any()
    ro = ReadOut('whiten', 8, 1)
    got = ro_mod.crossfit(xs, labels, ro, workers=1)['genre']
    assert got == pytest.approx(manual_crossfit(xs, labels, ro)['genre'], abs=1e-12)


def test_crossfit_averages_seeds_after_halves():
    rng = np.random.default_rng(13)
    xs, labels = toy(rng)
    ro = ReadOut('center', None, 1)
    got = ro_mod.crossfit(xs, labels, ro, workers=1)
    for a in ASPECTS:
        per_seed = [ro_mod.crossfit({(b, None): xs[(b, s)] for b in ASPECTS}, labels, ro, workers=1)[a]
                    for s in range(5)]
        assert got[a] == pytest.approx(np.mean(per_seed), abs=1e-12)


def test_crossfit_aspect_free_arm_may_share_one_matrix():
    rng = np.random.default_rng(14)
    xs, labels = toy(rng, n_seeds=0, per_aspect=False)
    assert xs[('emotion', None)] is xs[('genre', None)]
    ro = ReadOut('whiten', 6, 3)
    got = ro_mod.crossfit(xs, labels, ro, workers=1)
    assert got == pytest.approx(manual_crossfit(xs, labels, ro), abs=1e-12)


def test_crossfit_all_equals_crossfit_per_readout():
    rng = np.random.default_rng(15)
    xs, labels = toy(rng)
    ros = [R0, ReadOut('whiten', 6, 0), ReadOut('whiten', 6, 5), ReadOut('center', None, 3)]
    allres = ro_mod.crossfit_all(xs, labels, ros, workers=1)
    assert list(allres) == ros
    for ro in ros:
        assert allres[ro] == ro_mod.crossfit(xs, labels, ro, workers=1)


def test_crossfit_asserts_missing_matrix():
    rng = np.random.default_rng(16)
    xs, labels = toy(rng)
    del xs[('style', 1)]
    with pytest.raises(AssertionError):
        ro_mod.crossfit(xs, labels, R0, workers=1)


def test_half_dropped_counts_queries_without_a_relevant_painting_in_the_half():
    # positions 0..7: even half = 0,2,4,6; odd half = 1,3,5,7. Class 1 is a singleton in the even half (dropped);
    # the -1 rows are not queries and are not counted.
    lab = np.array([0, 0, 0, -1, 1, 0, 0, 1], dtype=np.int64)
    # even half labels 0,0,1,0 -> one dropped; odd half labels 0,-1,0,1 -> the single class-1 row is dropped
    assert ro_mod.half_dropped({'style': lab}) == {'style': [1, 1]}
    lab2 = np.array([0, 0, 0, -1, 1, 0, 0, 0], dtype=np.int64)
    assert ro_mod.half_dropped({'style': lab2, 'genre': np.full(8, -1)}) == {'style': [1, 0], 'genre': [0, 0]}


def test_crossfit_all_checks_retrieval_dropped_against_half_dropped(monkeypatch):
    rng = np.random.default_rng(17)
    xs, labels = toy(rng)
    labels['style'][:] = labels['style'] % 3
    labels['style'][0] = 3        # a singleton class in the even half (position 0): one dropped query
    assert ro_mod.half_dropped(labels)['style'][0] >= 1
    got = ro_mod.crossfit_all(xs, labels, [R0], workers=1)   # the counts agree: no assertion error
    assert set(got[R0]) == set(ASPECTS)
    real = ro_mod.half_dropped
    monkeypatch.setattr(ro_mod, 'half_dropped',
                        lambda lab: {a: [d + 1 for d in v] for a, v in real(lab).items()})
    with pytest.raises(AssertionError):
        ro_mod.crossfit_all(xs, labels, [R0], workers=1)


@pytest.mark.parametrize('seeds', [(0, 1, 2), (0, 1, 2, 3), (0, 1, 2, 3, 4, 5), (1, 2, 3, 4, 5), (None, 0, 1, 2, 3, 4)])
def test_crossfit_asserts_the_seed_set(seeds):
    rng = np.random.default_rng(18)
    xs, labels = toy(rng, n_seeds=1)
    xs = {(a, s): xs[(a, 0)] for a in ASPECTS for s in seeds}
    with pytest.raises(AssertionError):
        ro_mod.crossfit(xs, labels, R0, workers=1)


# ---- score, guard, select -----------------------------------------------------------------------------------------

R_A = {'emotion': 0.01, 'style': 0.02, 'genre': 0.04}
BASE = {'emotion': 0.25, 'style': 0.5, 'genre': 0.75}  # R0's mAP: exact binary fractions


def row_with_score(s, base=BASE, ranges=R_A):
    """A per-aspect mAP dict whose score is s: the same gain s in r_a units on every aspect."""
    return {a: base[a] + s * ranges[a] for a in ASPECTS}


def rows_from(scores):
    rows = {R0: dict(BASE)}
    for ro, s in scores.items():
        rows[ro] = row_with_score(s)
    return rows


def test_score_hand_example():
    ro = ReadOut('center', None, 0)
    rows = {R0: {'emotion': 0.20, 'style': 0.40, 'genre': 0.60},
            ro: {'emotion': 0.22, 'style': 0.40, 'genre': 0.63}}
    sc = ro_mod.score(rows, {'emotion': 0.02, 'style': 0.04, 'genre': 0.03})
    assert sc[R0] == 0.0
    assert sc[ro] == pytest.approx((1.0 + 0.0 + 1.0) / 3, abs=1e-12)  # (0.02/0.02 + 0/0.04 + 0.03/0.03) / 3


def test_score_requires_r0_row():
    with pytest.raises(AssertionError):
        ro_mod.score({ReadOut('center', None, 0): dict(BASE)}, R_A)


def test_guard_drops_any_aspect_more_than_r_below_r0():
    keep = ReadOut('center', None, 0)
    at_limit = ReadOut('center', None, 1)    # style exactly r_a below R0: kept
    below = ReadOut('center', None, 3)       # genre just under R0 - r_a: dropped, though the others are far above
    rows = {R0: dict(BASE),
            keep: {'emotion': 0.26, 'style': 0.50, 'genre': 0.75},
            at_limit: {'emotion': 0.30, 'style': 0.50 - 0.02, 'genre': 0.80},
            below: {'emotion': 0.50, 'style': 0.60, 'genre': 0.75 - 0.04 - 0.001}}
    g = ro_mod.guard(rows, R_A)
    assert g[R0] and g[keep] and g[at_limit] and not g[below]


def test_select_never_returns_a_guarded_out_readout():
    big = ReadOut('whiten', 64, 0)
    rows = {R0: dict(BASE), big: {'emotion': 0.9, 'style': 0.9, 'genre': 0.75 - 0.05}}
    assert ro_mod.score(rows, R_A)[big] > 10
    assert ro_mod.select(rows, R_A) == R0


def test_select_picks_the_best_when_clearly_better():
    a, b = ReadOut('center', None, 0), ReadOut('whiten', 256, 0)
    assert ro_mod.select(rows_from({a: 0.5, b: 0.502}), R_A) == b  # 0.002 above: outside the tie band


def test_tie_r0_beats_a_better_by_0_0009_readout():
    a = ReadOut('whiten', 256, 3)
    assert ro_mod.select(rows_from({a: 0.0009}), R_A) == R0
    assert ro_mod.select(rows_from({a: 0.0011}), R_A) == a


def test_tie_band_is_measured_from_the_best_not_from_r0():
    a, b = ReadOut('center', None, 0), ReadOut('whiten', 256, 0)
    # best 0.0015 (b); a at 0.0010 is tied with it; R0 (0) is 0.0015 below: not tied. First in tie order: a.
    assert ro_mod.select(rows_from({a: 0.0010, b: 0.0015}), R_A) == a


def test_tie_no_expansion_before_expansion():
    exp, plain = ReadOut('none', None, 1), ReadOut('whiten', 64, 0)
    assert ro_mod.select(rows_from({exp: 0.5, plain: 0.4995}), R_A) == plain


def test_tie_none_before_center_before_whiten():
    c, w, n = ReadOut('center', None, 0), ReadOut('whiten', 1024, 0), ReadOut('none', None, 3)
    assert ro_mod.select(rows_from({w: 0.5, c: 0.4995}), R_A) == c
    # among expanded read-outs W comes before k: none+k3 beats center+k1
    assert ro_mod.select(rows_from({n: 0.5, ReadOut('center', None, 1): 0.5, ReadOut('whiten', 64, 1): 0.5}),
                         R_A) == n


def test_tie_smaller_k():
    k3, k1, k10 = (ReadOut('center', None, k) for k in (3, 1, 10))
    assert ro_mod.select(rows_from({k3: 0.5, k1: 0.4995, k10: 0.5}), R_A) == k1


def test_tie_larger_whitening_size():
    s64, s256, s1024 = (ReadOut('whiten', s, 0) for s in (64, 256, 1024))
    assert ro_mod.select(rows_from({s64: 0.5, s256: 0.4995, s1024: 0.4991}), R_A) == s1024
    # and with expansion at equal W and k
    e = [ReadOut('whiten', s, 5) for s in (64, 256)]
    assert ro_mod.select(rows_from({e[0]: 0.5, e[1]: 0.5}), R_A) == e[1]


def test_tie_smaller_k_before_larger_whitening_size():
    small_k, big_size = ReadOut('whiten', 64, 1), ReadOut('whiten', 1024, 3)
    # both kept, tied within 0.001; the larger size is the (slightly) better one
    assert ro_mod.select(rows_from({small_k: 0.4995, big_size: 0.5}), R_A) == small_k


def test_select_does_not_depend_on_row_order():
    ros = [ReadOut('whiten', 64, 0), ReadOut('center', None, 0), ReadOut('whiten', 256, 0)]
    scores = {ro: 0.3 for ro in ros}
    rows = rows_from(scores)
    rev = {k: rows[k] for k in reversed(list(rows))}
    assert ro_mod.select(rows, R_A) == ro_mod.select(rev, R_A) == ReadOut('center', None, 0)


def test_select_uses_c_star_ranges_for_the_arm_given():
    # The ranges argument is used as is (the runner passes c*'s r_a for every arm); a wider r_a lets a read-out pass.
    a = ReadOut('center', None, 0)
    rows = {R0: dict(BASE), a: {'emotion': 0.25 - 0.015, 'style': 0.9, 'genre': 0.9}}
    assert ro_mod.select(rows, R_A) == R0                     # emotion 0.015 below R0 > r_emotion 0.01: guarded out
    assert ro_mod.select(rows, {**R_A, 'emotion': 0.02}) == a  # r_emotion 0.02: allowed
