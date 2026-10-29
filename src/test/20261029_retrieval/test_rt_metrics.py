"""Tests for rt_metrics (verdict code). Tie-free random inputs unless a test says otherwise."""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, ndcg_score

import rt_metrics


def unit_rows(rng, n, d):
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_ap_and_p10_match_sklearn():
    rng = np.random.default_rng(1)
    x = unit_rows(rng, 60, 8)
    labels = rng.integers(0, 3, size=60)
    out = rt_metrics.retrieval(x, labels, workers=1)
    assert out['ap'].shape == (60,) and out['p10'].shape == (60,)
    s = x.astype(np.float64) @ x.astype(np.float64).T
    for q in range(60):
        g = np.arange(60) != q
        rel = (labels[g] == labels[q]).astype(int)
        assert out['ap'][q] == pytest.approx(average_precision_score(rel, s[q, g]), abs=1e-6)
        top = np.argsort(-s[q, g], kind='stable')[:10]
        assert out['p10'][q] == pytest.approx(rel[top].sum() / 10, abs=1e-12)


def test_tie_rule_keeps_gallery_row_order():
    # All rows identical, so every similarity ties. Query = row 0; gallery rows 1..5 have labels 0,1,1,0,1 and the
    # query's label is 0 -> relevant gallery rows are 1 and 4 (gallery ranks 1 and 4 in row order).
    # AP = (1/1 + 2/4) / 2 = 0.75 (scikit-learn groups the ties and would give 2/5 = 0.4); P@10 = 2/10.
    x = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (6, 1))
    labels = np.array([0, 0, 1, 1, 0, 1])
    out = rt_metrics.retrieval(x, labels, workers=1)
    assert out['ap'][0] == pytest.approx(0.75, abs=1e-12)
    assert out['p10'][0] == pytest.approx(0.2, abs=1e-12)


def test_tie_rule_on_thousands_of_identical_rows():
    # N identical rows, so every similarity ties; the order must stay the gallery row order (a non-stable sort
    # scrambles it once there are more than a handful of rows). Row 0 is the query (label 0); rows 1..N-R have
    # label 1 and the last R-1 rows have label 0, so the query's R-1 relevant gallery items sit at the last
    # R-1 gallery ranks g-R+2..g (g = N-1). The k-th of them has precision k / (g-R+1+k).
    n, r = 3000, 400
    x = np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (n, 1))
    labels = np.ones(n, dtype=np.int64)
    labels[0] = 0
    labels[n - (r - 1):] = 0
    g = n - 1
    want = np.mean([k / (g - r + 1 + k) for k in range(1, r)])
    assert want == pytest.approx(0.0699, abs=1e-4)  # far from 1 (what keeping the relevant items in front gives)
    out = rt_metrics.retrieval(x, labels, workers=1)
    assert out['rows'][0] == 0
    assert out['ap'][0] == pytest.approx(want, abs=1e-12)
    assert out['p10'][0] == 0.0
    # query n-1 (last row, label 0): relevant gallery items are row 0 and rows n-r+1..n-2
    # at ranks g-r+3..g, plus row 0 at rank 1 (row order); the k-th of those has precision (k+1) / (g-r+2+k)
    k = np.arange(1, r - 1)
    want_last = (1.0 + np.sum((k + 1) / (g - r + 2 + k))) / (r - 1)
    assert out['ap'][-1] == pytest.approx(want_last, abs=1e-12)


def test_query_is_never_in_its_own_gallery():
    # Four identical rows (all ties), labels 0,1,0,1. Gallery of query q = the other three rows in row order:
    # q0: labels 1,0,1 -> rel [0,1,0] AP 1/2; q1: 0,0,1 -> [0,0,1] AP 1/3; q2: 0,1,1 -> [1,0,0] AP 1;
    # q3: 0,1,0 -> [0,1,0] AP 1/2. With a query left in its gallery (first on a tie) every value would differ.
    x = np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (4, 1))
    out = rt_metrics.retrieval(x, np.array([0, 1, 0, 1]), workers=1)
    assert out['ap'] == pytest.approx([0.5, 1 / 3, 1.0, 0.5], abs=1e-12)


def test_unlabelled_rows_and_dropped_queries():
    # Labels 0,0,1,-1,2,0: classes 1 and 2 have a single painting, so those two queries have no relevant gallery
    # item and are dropped (counted); row 3 (label -1) is neither query nor gallery. Row 3 equals the query
    # direction of row 0 so that, were it in the gallery, it would change the AP.
    x = np.array([[1, 0], [0.8, 0.6], [0, 1], [1, 0], [-1, 0], [0.6, 0.8]], dtype=np.float32)
    labels = np.array([0, 0, 1, -1, 2, 0])
    out = rt_metrics.retrieval(x, labels, workers=1)
    assert out['dropped'] == 2
    assert out['rows'].tolist() == [0, 1, 5]
    # query 0: gallery rows 1,5,2,4 with sims .8,.6,0,-1 -> rel [1,1,0,0] -> AP 1
    # query 1 (0.8,0.6): sims to rows 0,5,2,4 = .8, .48+.48=.96, .6, -.8 -> order 5,0,2,4 -> rel [1,1,0,0] AP 1
    # query 5 (0.6,0.8): sims to rows 0,1,2,4 = .6, .96, .8, -.6 -> order 1,2,0,4 -> rel [1,0,1,0] -> (1+2/3)/2
    assert out['ap'] == pytest.approx([1.0, 1.0, (1 + 2 / 3) / 2], abs=1e-6)
    assert out['p10'] == pytest.approx([0.2, 0.2, 0.2], abs=1e-12)


def test_query_expansion_hand_example():
    # 2-d unit rows at angles a=0 [label 0], b=60 [1], c=25 [0], d=90 [1], e=45 [0]; no ties anywhere.
    # Query b (60 deg), relevant gallery row = d only.
    # Plain: angular distance a 60, c 35, d 30, e 15 -> order e,d,c,a -> rel [0,1,0,0] -> AP 1/2.
    # k=1: top-1 neighbour e (45); new query = normalize(b + e) = bisector at 52.5 deg; distances a 52.5, c 27.5,
    #   d 37.5, e 7.5 -> order e,c,d,a -> rel [0,0,1,0] -> AP 1/3.
    # k=2: neighbours e,d; new query = normalize(b+e+d) at atan2(.866+.7071+1, .5+.7071+0) = 64.9 deg; distances
    #   a 64.9, c 39.9, d 25.1, e 19.9 -> order e,d,c,a -> rel [0,1,0,0] -> AP 1/2.
    # The query stays out of its gallery, so b cannot be its own neighbour.
    t = np.deg2rad([0, 60, 25, 90, 45])
    x = np.stack([np.cos(t), np.sin(t)], axis=1).astype(np.float32)
    labels = np.array([0, 1, 0, 1, 0])
    for k, ap_b in ((0, 1 / 2), (1, 1 / 3), (2, 1 / 2)):
        out = rt_metrics.retrieval(x, labels, qe_k=k, workers=1)
        assert out['ap'][1] == pytest.approx(ap_b, abs=1e-6), k


def test_query_expansion_takes_neighbours_from_the_plain_ranking():
    # 2-d unit rows at angles 330 [label 1, the query], 210 [1], 190 [0], 80 [1], 100 [0], 170 [0] degrees;
    # k = 2. Angular distances from 330: row3 110, row1 120, row4 130, row2 140, row5 160.
    # Plain: order 3,1,4,2,5 -> rel [1,1,0,0,0] -> AP 1. Neighbours are rows 3 and 1.
    # k=2: query = normalize(u330 + u80 + u210), pointing at -5 deg (355); distances row3 85, row4 105, row1 145,
    #   row2 165, row5 175 -> order 3,4,1,2,5 -> rel [1,0,1,0,0] -> AP (1 + 2/3)/2 = 5/6.
    # Wrong: neighbours taken from that expanded ranking (rows 3 and 4) give the query u330 + u80 + u100 at
    #   59.5 deg; distances row3 20.5, row4 40.5, row5 110.5, row2 130.5, row1 150.5 -> rel [1,0,0,0,1] -> AP 0.7.
    t = np.deg2rad([330, 210, 190, 80, 100, 170])
    x = np.stack([np.cos(t), np.sin(t)], axis=1).astype(np.float32)
    labels = np.array([1, 1, 0, 1, 0, 0])
    for k, want in ((0, 1.0), (2, 5 / 6)):
        out = rt_metrics.retrieval(x, labels, qe_k=k, workers=1)
        assert out['rows'][0] == 0
        assert out['ap'][0] == pytest.approx(want, abs=1e-6), k


def test_ndcg_matches_sklearn_per_query():
    rng = np.random.default_rng(2)
    x = unit_rows(rng, 50, 8)
    dist = rng.dirichlet(np.ones(9), size=50)
    out = rt_metrics.graded(x, dist, workers=1)
    assert out.shape == (50,) and out.dtype == np.float64
    s = x.astype(np.float64) @ x.astype(np.float64).T
    for q in range(50):
        g = np.arange(50) != q
        gain = np.minimum(dist[q][None, :], dist[g]).sum(axis=1)
        want = ndcg_score(gain[None, :], s[q, g][None, :], k=None, ignore_ties=False)
        assert out[q] == pytest.approx(want, abs=1e-9)


def test_ndcg_hand_example():
    # Query 0 = (1,0); row 1 = (0,1) (sim 0), row 2 = (.99,.141) (sim .99) -> ranking: row 2, then row 1.
    # Gains for query 0: row 1 -> min(1,.5)+min(0,.5) = .5; row 2 -> min(1,.2)+min(0,.8) = .2.
    # DCG = .2/log2(2) + .5/log2(3); ideal = .5/log2(2) + .2/log2(3).
    x = np.array([[1, 0], [0, 1], [0.99, 0.141]], dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    dist = np.array([[1, 0], [.5, .5], [.2, .8]], dtype=np.float64)
    out = rt_metrics.graded(x, dist, workers=1)
    want = (0.2 + 0.5 / np.log2(3)) / (0.5 + 0.2 / np.log2(3))
    assert out[0] == pytest.approx(want, abs=1e-9)


def _dup_data(seed, n=300, d=256):
    # Wide rows with duplicates: in float32 the products of identical rows differ in the last bits depending on
    # the block shape and the number of threads, which breaks the tie order.
    rng = np.random.default_rng(seed)
    x = unit_rows(rng, n, d)
    src = rng.choice(n, size=80, replace=False)
    dst = rng.choice(np.setdiff1d(np.arange(n), src), size=80, replace=False)
    x[dst] = x[src]
    labels = rng.integers(-1, 6, size=n)
    dist = rng.dirichlet(np.ones(9), size=n)
    dist[dst] = dist[src]
    return x, labels, dist


def test_blocks_and_workers_do_not_change_results(monkeypatch):
    x, labels, dist = _dup_data(3)
    ref = rt_metrics.retrieval(x, labels, qe_k=3, workers=1)
    ref_g = rt_metrics.graded(x, dist, workers=1)
    for block, w in ((256, 8), (7, 1), (7, 8), (50, 3)):
        monkeypatch.setattr(rt_metrics, 'BLOCK', block)
        monkeypatch.setattr(rt_metrics, 'GRADED_BLOCK', block)
        got = rt_metrics.retrieval(x, labels, qe_k=3, workers=w)
        for key in ('ap', 'p10', 'rows'):
            np.testing.assert_array_equal(got[key], ref[key])
        assert got['dropped'] == ref['dropped']
        np.testing.assert_array_equal(rt_metrics.graded(x, dist, workers=w), ref_g)
    # and the same for plain ranking
    ref0 = rt_metrics.retrieval(x, labels, qe_k=0, workers=1)
    monkeypatch.setattr(rt_metrics, 'BLOCK', 7)
    np.testing.assert_array_equal(rt_metrics.retrieval(x, labels, qe_k=0, workers=8)['ap'], ref0['ap'])


def test_graded_rejects_negative_distribution_values():
    x = np.eye(3, dtype=np.float32)
    dist = np.array([[1, 0], [0.5, 0.5], [-0.1, 1.1]], dtype=np.float64)
    with pytest.raises(AssertionError):
        rt_metrics.graded(x, dist, workers=1)
