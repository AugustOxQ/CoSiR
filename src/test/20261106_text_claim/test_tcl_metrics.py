"""Metric checks for text-claim design §3.4 and scoring replay §3.5(iii)."""
import numpy as np
import pytest

import tx_data as data
import tcl_metrics as m
import tx_metrics
import rt_metrics


def run(q, g, ql, gl, *, qp=None, gp=None, self_match=False, readout='r0', means=None, **entry):
    labels = dict(test=dict(q=np.asarray(ql, dtype=np.int64), g=np.asarray(gl, dtype=np.int64), **entry))
    return m.evaluate(q, g, qp, gp, labels, self_match, readout, 'cpu', block=2, means=means)


def test_ap_four_by_four():
    q = np.tile(np.asarray([[1, 0]], dtype=np.float32), (4, 1))
    g = data.unit(np.asarray([[4, 1], [3, 2], [2, 3], [1, 4]], dtype=np.float32))
    got = run(q, g, [0, 1, 0, 1], [0, 1, 0, 1])
    np.testing.assert_allclose(got['ap']['test'], [5 / 6, 1 / 2, 5 / 6, 1 / 2])
    assert got['chance']['test'] == pytest.approx(0.5)


def test_self_removal_and_first_rank():
    x = data.unit(np.asarray([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32))
    pid = np.asarray([0, 0, 1, 1], dtype=np.int64)
    got = run(x, x, pid, pid, qp=pid, gp=pid, self_match=True)
    np.testing.assert_array_equal(got['ap']['test'], np.ones(4))
    np.testing.assert_array_equal(got['first_rank'], np.ones(4))
    assert got['chance']['test'] == pytest.approx(1 / 3)
    pair = m.summarize_pair(got['first_rank'])
    assert pair == dict(r1=1, r5=1, r10=1, medr=1, mrr=1, n_query=4)


def test_invalid_gallery_skipped_and_query_drop():
    q = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    g = data.unit(np.asarray([[4, 1], [3, 2], [2, 3], [1, 4]], dtype=np.float32))
    got = run(q, g, [0, -1], [-1, 1, 0, 0], g_keep=np.asarray([True, False, True, True]))
    assert got['ap']['test'][0] == 1
    assert np.isnan(got['ap']['test'][1]) and got['dropped']['test'] == 0
    assert got['chance']['test'] == 1
    none = run(q[:1], g, [2], [0, 0, 0, 0])
    assert np.isnan(none['ap']['test'][0]) and none['n_kept']['test'] == 0 and none['dropped']['test'] == 1
    assert np.isnan(none['chance']['test'])


def test_cross_painting_and_pair_ranks():
    q = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    g = data.unit(np.asarray([[4, 1], [3, 2], [2, 3], [1, 4]], dtype=np.float32))
    qp, gp = np.asarray([0, 9], dtype=np.int64), np.asarray([0, 0, 1, 2], dtype=np.int64)
    got = run(q, g, [0, 0], [0, 1, 1, 0], qp=qp, gp=gp, exclude_same_painting=True)
    assert got['ap']['test'][0] == pytest.approx(0.5)
    assert got['first_rank'][0] == 1 and np.isnan(got['first_rank'][1])
    assert m.summarize_pair(got['first_rank'])['mrr'] == 0.5


def test_center_matches_manual_and_expansion_runs():
    rng = np.random.default_rng(4)
    q, g = data.unit(rng.normal(size=(4, 8))), data.unit(rng.normal(size=(13, 8)))
    qp, gp = np.arange(4, dtype=np.int64), np.arange(13, dtype=np.int64)
    ql, gl = qp % 2, gp % 2
    centered = run(q, g, ql, gl, qp=qp, gp=gp, readout='center')
    manual = run(data.unit(q - q.mean(0)), data.unit(g - g.mean(0)), ql, gl, qp=qp, gp=gp)
    np.testing.assert_allclose(centered['ap']['test'], manual['ap']['test'])
    assert np.isfinite(run(q, g, ql, gl, qp=qp, gp=gp, readout='center_k10')['ap']['test']).all()
    words = run(q, g, ql, gl, readout='center')
    expected = run(data.unit(q - g.mean(0)), data.unit(g - g.mean(0)), ql, gl)
    np.testing.assert_allclose(words['ap']['test'], expected['ap']['test'])


def test_rt_metrics_parity_500():
    rng = np.random.default_rng(0)
    x = data.unit(rng.normal(size=(500, 32)))
    y = np.arange(500, dtype=np.int64) % 5
    pid = np.arange(500, dtype=np.int64)
    got = run(x, x, y, y, qp=pid, gp=pid, self_match=True)
    expected = rt_metrics.retrieval(x, y, workers=1)
    np.testing.assert_allclose(got['ap']['test'], expected['ap'], atol=1e-5, rtol=0)


def test_macro_and_painting_bootstrap():
    ap = np.asarray([0, 1, 0.2, np.nan, 1], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1, -1], dtype=np.int64)
    assert m.macro_map(ap, labels) == pytest.approx(0.35)
    pid = np.asarray([0, 0, 1, 2], dtype=np.int64)
    values = np.asarray([1, 3, 4, np.nan])
    mean, lo, hi = m.painting_bootstrap(values, pid, 3)
    assert mean == pytest.approx(8 / 3) and lo <= mean <= hi
    assert (mean, lo, hi) == m.painting_bootstrap(values, pid, 3)
    assert m.painting_bootstrap(np.ones(4), pid, 3) == (1, 1, 1)


def test_entries_share_ranking_but_have_independent_masks():
    x = np.eye(4, dtype=np.float32)
    pid = np.arange(4, dtype=np.int64)
    labels = np.asarray([0, 0, 1, -1], dtype=np.int64)
    result = m.evaluate(x, x, pid, pid, dict(all=dict(q=labels, g=labels),
                        masked=dict(q=labels, g=labels, q_keep=np.asarray([False, True, True, True]))),
                        True, 'r0', 'cpu')
    assert result['n_kept'] == dict(all=2, masked=1)
    assert np.isnan(result['first_rank']).all()


def test_fit_means_accumulates_in_float64_and_returns_float32():
    t = np.asarray([[1e8, 1], [1, 4], [-1e8, 7]], dtype=np.float32)
    i = np.asarray([[3, -2], [5, 6]], dtype=np.float32)
    tm, im = m.fit_means(t, i)
    assert tm.dtype == im.dtype == np.float32
    np.testing.assert_array_equal(tm, np.asarray([1 / 3, 4], dtype=np.float32))
    np.testing.assert_array_equal(im, [4, 2])


def assert_scores_equal(got, expected):
    for field in ('ap', 'chance'):
        for key in expected[field]:
            np.testing.assert_allclose(got[field][key], expected[field][key], atol=1e-5, rtol=0)
    np.testing.assert_allclose(got['first_rank'], expected['first_rank'], atol=1e-5, rtol=0)
    assert got['n_kept'] == expected['n_kept'] and got['dropped'] == expected['dropped']


@pytest.mark.parametrize('readout', ['r0', 'center', 'center_k10'])
@pytest.mark.parametrize('self_match', [False, True])
def test_validation_means_match_r22_replay(readout, self_match):
    rng = np.random.default_rng(14)
    g = data.unit(rng.normal(size=(17, 8)))
    q = g if self_match else data.unit(rng.normal(size=(6, 8)))
    qp, gp = np.arange(len(q), dtype=np.int64) // 2, np.arange(len(g), dtype=np.int64) // 2
    ql, gl = qp % 3, gp % 3
    gl[::4] = -1
    labels = dict(test=dict(q=ql, g=gl))
    args = q, g, qp, gp, labels, self_match, readout, 'cpu'
    expected = tx_metrics.evaluate(*args, block=2)
    assert_scores_equal(m.evaluate(*args, block=2), expected)
    means = m.fit_means(q, g)
    assert_scores_equal(m.evaluate(*args, block=2, means=means), expected)


@pytest.mark.parametrize('readout', ['center', 'center_k10'])
def test_held_means_match_manual_centering_and_expansion(readout):
    rng = np.random.default_rng(7)
    q, g = data.unit(rng.normal(size=(6, 4))), data.unit(rng.normal(size=(17, 4)))
    qm = np.asarray([0.4, -0.2, 0.3, 0.1], dtype=np.float32)
    gm = np.asarray([-0.3, 0.2, -0.4, 0.1], dtype=np.float32)
    qp, gp = np.arange(6, dtype=np.int64), np.arange(17, dtype=np.int64)
    ql, gl = qp % 2, gp % 2
    gl[::3] = -1  # These rows must still participate in ranking and expansion.
    cq, cg = data.unit(q - qm), data.unit(g - gm)
    if readout == 'center_k10':
        order = np.argsort(-(cq @ cg.T), axis=1)
        cq = data.unit(cq + cg[order[:, :10]].sum(1))
    expected = run(cq, cg, ql, gl, qp=qp, gp=gp)
    got = run(q, g, ql, gl, qp=qp, gp=gp, readout=readout, means=(qm, gm))
    assert_scores_equal(got, expected)
    fitted = run(q, g, ql, gl, qp=qp, gp=gp, readout=readout)
    assert not np.allclose(got['ap']['test'], fitted['ap']['test'])


def test_raw_readout_does_not_apply_supplied_means():
    q = np.eye(2, dtype=np.float32)
    means = (np.ones(2, dtype=np.float32), np.zeros(2, dtype=np.float32))
    got = run(q, q, [0, 1], [0, 1], means=means)
    np.testing.assert_array_equal(got['ap']['test'], [1, 1])


@pytest.mark.parametrize('seeds, expected', [([1, 1], 13 / 150), ([0, 1], 1 / 30), ([0, 0], -0.02)])
def test_replicate_six_paintings_counts_repeated_units_and_seeds(seeds, expected):
    counts = np.asarray([1, 2, 3, 1, 2, 4], dtype=np.float64)
    cand = np.asarray([[0.2, 1.0, 1.5, 0.4, 1.6, 1.2],
                       [0.8, 0.6, 2.1, 0.9, 0.2, 2.8]], dtype=np.float64)
    comp = np.asarray([0.1, 0.8, 1.2, 0.6, 0.8, 2.0], dtype=np.float64)
    units = np.asarray([0, 1, 1, 4, 5, 5], dtype=np.int64)
    # The draw has 15 queries; seed numerators are 6.2 and 7.8, comparator 6.5.
    got = m.replicate(cand, comp, counts, units, np.asarray(seeds, dtype=np.int64))
    assert got == pytest.approx(expected, abs=1e-12, rel=0)


def loop_interval(cand, comp, pid, draws):
    """Independent oracle: physically repeat query rows instead of aggregating painting sums."""
    units = np.unique(pid)
    k, n_units = len(cand), len(units)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(draws):
        I = rng.integers(0, n_units, size=n_units)
        S = rng.choice([0, 1, 2, 3, 4] if k == 5 else list(range(k)), size=k, replace=True)
        queries = np.concatenate([np.flatnonzero(pid == units[j]) for j in I])
        boot.append(cand[S][:, queries].mean(1).mean() - comp[queries].mean())
    lower, upper = np.percentile(boot, [2.5, 97.5])
    return cand.mean() - comp.mean(), lower, upper, n_units


@pytest.mark.parametrize('k', [1, 2, 5])
def test_painting_interval_matches_plain_loop_across_chunks(k):
    rng = np.random.default_rng(23)
    # Gaps and out-of-order queries expose grouping by first appearance or by all possible IDs.
    pid = np.asarray([20, 2, 20, 9, 2, 60, 9, 9, 100, 60, 4, 4, 4], dtype=np.int64)
    cand, comp = rng.random((k, len(pid))), rng.random(len(pid))
    expected = loop_interval(cand, comp, pid, 513)
    got = m.painting_interval(cand, comp, pid, draws=513)
    np.testing.assert_allclose(got, expected, atol=1e-12, rtol=0)
    assert got[3] == 6
    assert got == m.painting_interval(cand, comp, pid, draws=513)


def test_painting_interval_default_draws_and_constant_values():
    cand, comp = np.full((2, 3), 0.75), np.full(3, 0.25)
    pid = np.asarray([100, 100, 300], dtype=np.int64)
    assert m.painting_interval(cand, comp, pid) == (0.5, 0.5, 0.5, 2)


def test_painting_interval_one_painting_still_resamples_seeds():
    cand = np.asarray([[0.0, 0.2], [0.8, 1.0]], dtype=np.float64)
    comp = np.asarray([0.1, 0.3], dtype=np.float64)
    pid = np.asarray([9, 9], dtype=np.int64)
    got = m.painting_interval(cand, comp, pid, draws=100)
    np.testing.assert_allclose(got, (0.3, -0.1, 0.7, 1), atol=1e-12, rtol=0)


@pytest.mark.parametrize('lower, seed_range, expected', [
    (0.1, 0.2, 'WIN'), (np.nextafter(0.0, 1.0), 0.0, 'WIN'),
    (0.0, 0.2, 'LEVEL'), (-0.1, 0.2, 'LEVEL'),
    (np.nextafter(-0.2, 0.0), 0.2, 'LEVEL'),
    (-0.2, 0.2, 'BELOW'), (-0.3, 0.2, 'BELOW'), (0.0, 0.0, 'BELOW'),
])
def test_outcome_strict_boundaries(lower, seed_range, expected):
    assert m.outcome(lower, seed_range) == expected


@pytest.mark.parametrize('lower, seed_range', [(np.nan, 0.1), (np.inf, 0.1),
                                              (0.1, -0.1), (0.1, np.nan), (0.1, np.inf)])
def test_outcome_rejects_invalid_inputs(lower, seed_range):
    with pytest.raises(AssertionError):
        m.outcome(lower, seed_range)


@pytest.mark.parametrize('field, value', [
    ('cand', np.ones((2, 3), dtype=np.float32)),
    ('cand', np.ones(3, dtype=np.float64)),
    ('cand', np.ones((0, 3), dtype=np.float64)),
    ('cand', np.full((2, 3), np.nan)),
    ('comp', np.ones(3, dtype=np.float32)),
    ('comp', np.full(3, np.inf)),
    ('comp', np.ones(2, dtype=np.float64)),
    ('pid', np.asarray([0, -1, 2], dtype=np.int64)),
    ('pid', np.asarray([0, 1, 2], dtype=np.float64)),
    ('pid', np.arange(2, dtype=np.int64)),
    ('draws', 0), ('draws', 1.5),
])
def test_painting_interval_rejects_invalid_inputs(field, value):
    args = dict(cand=np.ones((2, 3), dtype=np.float64), comp=np.zeros(3, dtype=np.float64),
                pid=np.arange(3, dtype=np.int64), draws=3)
    args[field] = value
    with pytest.raises(AssertionError):
        m.painting_interval(**args)


@pytest.mark.parametrize('means', [
    (np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float32)),
    (np.zeros(2, dtype=np.float32), np.zeros(3, dtype=np.float32)),
    (np.full(3, np.nan, dtype=np.float32), np.zeros(3, dtype=np.float32)),
    (np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)),
])
def test_self_match_rejects_invalid_or_distinct_means(means):
    x, pid = np.eye(3, dtype=np.float32), np.arange(3, dtype=np.int64)
    with pytest.raises(AssertionError):
        run(x, x, pid, pid, qp=pid, gp=pid, self_match=True, readout='center', means=means)


def test_fit_means_rejects_empty_or_nonfinite_rows():
    x = np.eye(2, dtype=np.float32)
    for bad in (x[:0], np.full_like(x, np.inf)):
        with pytest.raises(AssertionError):
            m.fit_means(bad, x)


@pytest.mark.parametrize('counts, units, seeds', [
    ([0, 1], [0, 1], [0, 1]), ([1, np.nan], [0, 1], [0, 1]),
    ([1, 1], [-1, 1], [0, 1]), ([1, 1], [0, 2], [0, 1]),
    ([1, 1], [0, 1], [-1, 1]), ([1, 1], [0, 1], [0, 2]),
])
def test_replicate_rejects_invalid_counts_or_indices(counts, units, seeds):
    with pytest.raises(AssertionError):
        m.replicate(np.ones((2, 2), dtype=np.float64), np.zeros(2, dtype=np.float64),
                    np.asarray(counts, dtype=np.float64), np.asarray(units), np.asarray(seeds))
