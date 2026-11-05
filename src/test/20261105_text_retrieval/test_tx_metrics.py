"""Hand-calculated retrieval cases and parity with the frozen i2i scorer."""
import numpy as np
import pytest

import tx_data as data
import tx_metrics as m
import rt_metrics


def run(q, g, ql, gl, *, qp=None, gp=None, self_match=False, readout='r0', **entry):
    labels = dict(test=dict(q=np.asarray(ql, dtype=np.int64), g=np.asarray(gl, dtype=np.int64), **entry))
    return m.evaluate(q, g, qp, gp, labels, self_match, readout, 'cpu', block=2)


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
