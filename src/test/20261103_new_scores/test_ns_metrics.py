"""Tests for ns_metrics (r17 ticket 01): class-count metrics, contingency forms, partitions at K, gate (iii).

Verdict code: each test fails when the logic it guards is broken (see the notes in each test).
Run: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -m pytest -q test_ns_metrics.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score)
from threadpoolctl import threadpool_info

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_paths  # noqa: E402
import ns_metrics as M  # noqa: E402

TOL = ns_paths.TOL_METRIC
PRE = ns_paths.PRE
SK = {'ami': adjusted_mutual_info_score, 'ari': adjusted_rand_score,
      'nmi': lambda y, c: normalized_mutual_info_score(y, c, average_method='arithmetic')}


def brute_acc(y, c):
    """Best one-to-one matching by trying every assignment of clusters to classes (padded to square)."""
    ys, cs = np.unique(y), np.unique(c)
    m = max(len(ys), len(cs))
    table = np.zeros((m, m))
    for i, a in enumerate(ys):
        for j, b in enumerate(cs):
            table[i, j] = np.sum((y == a) & (c == b))
    best = max(sum(table[i, p[i]] for i in range(m)) for p in itertools.permutations(range(m)))
    return best / len(y)


def random_case(rng, n, ky, kc):
    return rng.integers(0, ky, n), rng.integers(0, kc, n)


CASES = [(30, 3, 3), (40, 5, 5), (25, 4, 6), (50, 7, 7), (60, 6, 3), (35, 2, 7), (20, 7, 5)]


# ----------------------------------------------------------------------------------------------- array forms
@pytest.mark.parametrize('n,ky,kc', CASES)
@pytest.mark.parametrize('seed', range(3))
def test_array_forms_equal_sklearn(n, ky, kc, seed):
    y, c = random_case(np.random.default_rng(seed), n, ky, kc)
    for name in ('ami', 'nmi', 'ari'):
        assert abs(getattr(M, name)(y, c) - SK[name](y, c)) <= TOL


def test_nmi_is_arithmetic_not_geometric():
    # y has 2 classes, c has 5 clusters nested in them: arithmetic and geometric/max/min normalisations differ
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    c = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    arith = normalized_mutual_info_score(y, c, average_method='arithmetic')
    for other in ('geometric', 'min', 'max'):
        assert abs(arith - normalized_mutual_info_score(y, c, average_method=other)) > 1e-3
    assert abs(M.nmi(y, c) - arith) <= TOL
    C = M.contingency(y, c)
    assert abs(M.from_contingency(C, 'nmi') - arith) <= TOL


@pytest.mark.parametrize('n,ky,kc', CASES)
@pytest.mark.parametrize('seed', range(4))
def test_acc_equals_brute_force_best_permutation(n, ky, kc, seed):
    y, c = random_case(np.random.default_rng(100 + seed), n, ky, kc)
    assert abs(M.acc(y, c) - brute_acc(y, c)) <= TOL


def test_acc_greedy_trap_known_value():
    # table rows = classes, cols = clusters: [[5, 4], [4, 0]]. Greedy takes 5 then 0 = 5; optimum 4 + 4 = 8.
    C = np.array([[5, 4], [4, 0]])
    assert M.from_contingency(C, 'acc') == pytest.approx(8 / 13)
    y = np.array([0] * 9 + [1] * 4)
    c = np.array([0] * 5 + [1] * 4 + [0] * 4)
    assert M.acc(y, c) == pytest.approx(8 / 13) and brute_acc(y, c) == pytest.approx(8 / 13)


def test_acc_unmatched_cluster_counts_zero_and_denominator_is_labelled_rows():
    # 3 clusters, 2 classes: the third cluster has no match; denominator is all 6 labelled rows
    y = np.array([0, 0, 1, 1, 1, 1])
    c = np.array([0, 0, 1, 1, 2, 2])
    assert M.acc(y, c) == pytest.approx(4 / 6)
    # a cluster with no labelled row (empty column) counts zero
    C = np.array([[2, 0, 0], [0, 2, 2]])
    C2 = np.array([[2, 0, 0, 0], [0, 2, 2, 0]])
    assert M.from_contingency(C2, 'acc') == pytest.approx(M.from_contingency(C, 'acc'))


# ------------------------------------------------------------------------------------- contingency forms
def table_of(y, c):
    return M.contingency(y, c)


@pytest.mark.parametrize('metric', ('ami', 'nmi', 'ari', 'acc'))
@pytest.mark.parametrize('n,ky,kc', CASES)
@pytest.mark.parametrize('seed', range(3))
def test_contingency_form_equals_array_form_with_repeats(metric, n, ky, kc, seed):
    rng = np.random.default_rng(200 + seed)
    y, c = random_case(rng, n, ky, kc)
    w = rng.integers(0, 4, n)                              # bootstrap-like multiplicities, zeros included
    if w.sum() == 0:
        w[0] = 1
    C = M.contingency(y, c, weights=w, shape=(ky, kc))
    assert C.sum() == w.sum()
    yr, cr = np.repeat(y, w), np.repeat(c, w)
    expected = getattr(M, metric)(yr, cr)
    assert abs(M.from_contingency(C, metric) - expected) <= TOL
    if metric != 'acc':
        assert abs(M.from_contingency(C, metric) - SK[metric](yr, cr)) <= TOL


def test_contingency_with_empty_rows_and_columns_equals_array_form():
    # classes 2 and 4 and cluster 1 never occur in the draw: zero rows and columns in a fixed-shape table
    y = np.array([0, 0, 1, 3, 3, 3, 1, 0])
    c = np.array([0, 2, 0, 2, 2, 0, 2, 0])
    C = M.contingency(y, c, shape=(5, 3))
    for metric in M.METRICS:
        assert abs(M.from_contingency(C, metric) - getattr(M, metric)(y, c)) <= TOL


def test_contingency_masks_unlabelled_rows_and_rejects_bad_input():
    y = np.array([0, -1, 1, 1, -1])
    c = np.array([0, 1, 1, 1, 0])
    C = M.contingency(y, c, shape=(2, 2))
    assert C.tolist() == [[1, 0], [0, 2]]
    w = np.array([1, 5, 1, 2, 7])
    assert M.contingency(y, c, weights=w, shape=(2, 2)).tolist() == [[1, 0], [0, 3]]
    with pytest.raises(AssertionError):
        M.contingency(y, np.array([2, 1, 1, 1, 0]), shape=(2, 2))          # cluster id outside the shape
    with pytest.raises(AssertionError):
        M.contingency(y, c, weights=np.array([1, 1, 1]))                    # weights of the wrong length
    with pytest.raises(AssertionError):
        M.contingency(y, c, weights=np.array([1.5, 1, 1, 1, 1]))            # non-integer weights
    with pytest.raises(AssertionError):
        M.from_contingency(np.array([[1.5, 0], [0, 2]]), 'ami')             # non-integer table
    with pytest.raises(AssertionError):
        M.from_contingency(np.array([[1, 0], [0, 2]]), 'map')               # not a class-count metric


@pytest.mark.parametrize('y,c', [
    (np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])),           # both single: sklearn gives 1.0
    (np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0])),           # one cluster
    (np.array([0, 0, 0, 0]), np.array([0, 1, 2, 3])),           # one class, all singletons
    (np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3])),           # all singletons, perfect
    (np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])),           # perfect up to naming
])
def test_degenerate_cases_equal_sklearn(y, c):
    C = M.contingency(y, c)
    for name in ('ami', 'nmi', 'ari'):
        assert abs(M.from_contingency(C, name) - SK[name](y, c)) <= TOL
        assert abs(getattr(M, name)(y, c) - SK[name](y, c)) <= TOL


def test_array_forms_refuse_unlabelled_rows():
    y, c = np.array([0, -1, 1]), np.array([0, 1, 1])
    for name in M.METRICS:
        with pytest.raises(AssertionError):
            getattr(M, name)(y, c)


def test_class_count():
    assert M.class_count(np.array([-1, -1, 3, 3, 0, 7])) == 3
    assert M.class_count(np.array([-1, -1])) == 0
    assert M.class_count(np.array([2, 2, 2])) == 1


def test_score_partitions_masks_after_clustering_and_averages_states():
    rng = np.random.default_rng(5)
    n = 80
    y = rng.integers(-1, 4, n)
    parts = rng.integers(0, 4, (3, n)).astype(np.int32)
    for metric in M.METRICS:
        manual = np.mean([getattr(M, metric)(y[y >= 0], p[y >= 0]) for p in parts])
        assert abs(M.score_partitions(metric, y, parts) - manual) <= TOL
    # the unlabelled rows must not be passed as a class: scoring all rows with -1 as its own class differs
    assert abs(M.score_partitions('ami', y, parts) - np.mean([adjusted_mutual_info_score(y, p) for p in parts])) > 1e-6
    with pytest.raises(AssertionError):
        M.score_partitions('ami', y, parts[0])                # needs [states, n]


# ------------------------------------------------------------------------------------------------ partitions
def unit_rows(path, n=2000):
    # stored rows cast to float32, NOT renormalised: partitions() asserts near-unit rows and must not repair them
    x = np.load(path, mmap_mode='r')[:n].astype(np.float32)
    assert np.abs(np.linalg.norm(x, axis=1) - 1).max() < 1e-3
    return x


@pytest.fixture(scope='module')
def old():
    ns_paths.add_rounds()
    import ct_score
    import pav_score
    import bs_emoset_score
    return ct_score, pav_score, bs_emoset_score


@pytest.fixture(scope='module')
def painting_rows():
    return unit_rows(PRE / 'artelingo_full/clip_l14/img.npy')


@pytest.fixture(scope='module')
def photo_rows():
    return unit_rows(PRE / 'emoset/clip_l14/img.npy')


def test_partitions_artelingo_k20_equals_ct_score(old, painting_rows):
    ct_score = old[0]
    got = M.partitions(painting_rows, 20, 'artelingo')
    assert got.dtype == np.int32 and got.shape == (3, len(painting_rows))
    ref = ct_score.output_partitions(painting_rows, 'k1')['r20']
    assert np.array_equal(got, ref)
    assert np.array_equal(M.partitions(painting_rows, 20, 'mmart'), ref)      # same 8-thread route for MMArt


def test_partitions_painting_other_k_equals_ct_route_at_8_threads(old, painting_rows):
    from threadpoolctl import threadpool_limits
    pav_score = old[1]
    with threadpool_limits(limits=8):
        ref = np.asarray(pav_score.partitions(painting_rows, 9))
    assert np.array_equal(M.partitions(painting_rows, 9, 'artelingo'), ref)
    assert not np.array_equal(M.partitions(painting_rows, 9, 'artelingo'), M.partitions(painting_rows, 10, 'artelingo'))


@pytest.mark.parametrize('k', (8, 9, 20))
def test_partitions_photo_equals_bs_emoset_partitions_k(old, photo_rows, k):
    bs_emoset_score = old[2]
    ref = bs_emoset_score.partitions_k(photo_rows, k)
    for name in ('emoset', 'affection'):
        got = M.partitions(photo_rows, k, name)
        assert got.dtype == np.int32 and np.array_equal(got, ref)


def test_partitions_use_threads_from_the_table_not_the_environment(monkeypatch, painting_rows):
    from sklearn.cluster import MiniBatchKMeans
    seen = []
    real_fit = MiniBatchKMeans.fit

    def spy(self, X, *a, **kw):
        seen.append({i['internal_api']: i['num_threads'] for i in threadpool_info()})
        return real_fit(self, X, *a, **kw)
    monkeypatch.setattr(MiniBatchKMeans, 'fit', spy)
    x = painting_rows[:300]
    for name, threads in ns_paths.KMEANS_THREADS.items():
        seen.clear()
        M.partitions(x, 5, name)
        assert len(seen) == 3
        # every threadpool that has a thread count (openmp, blas) is capped at the table value
        assert all(v <= threads for s in seen for v in s.values()), (name, seen)
        if threads > 1:
            assert any(v == threads for s in seen for v in s.values()), (name, seen)


def test_partitions_do_not_renormalise_rows(monkeypatch, painting_rows):
    from sklearn.cluster import MiniBatchKMeans
    seen = []
    real_fit = MiniBatchKMeans.fit

    def spy(self, X, *a, **kw):
        seen.append(np.array(X))
        return real_fit(self, X, *a, **kw)
    monkeypatch.setattr(MiniBatchKMeans, 'fit', spy)
    x = painting_rows[:400] * np.float32(1.0005)          # within the 1e-3 check, visibly not unit
    M.partitions(x, 7, 'affection')
    assert len(seen) == 3 and all(s.dtype == np.float32 and np.array_equal(s, x) for s in seen)
    seen.clear()
    probs = np.random.default_rng(0).random((100, 40)).astype(np.float32)
    M.percept_partitions(probs, 5, 'artelingo')            # PercepT rows are normalised (D16), by design
    assert np.allclose(np.linalg.norm(seen[0], axis=1), 1, atol=1e-6)


def test_partitions_painting_equals_mt_diag_kmeans_k(old, painting_rows):
    import mt_diag
    for k in (9, 20):
        assert np.array_equal(M.partitions(painting_rows, k, 'mmart'), np.asarray(mt_diag.kmeans_k(painting_rows, k)))


def test_partitions_refuse_bad_input(painting_rows):
    x = painting_rows[:200]
    with pytest.raises(AssertionError):
        M.partitions(x * 1.1, 20, 'artelingo')                       # not unit norm
    with pytest.raises(AssertionError):
        M.partitions(x, 20, 'semart')                                # unknown set
    bad = x.copy()
    bad[0, 0] = np.nan
    with pytest.raises(AssertionError):
        M.partitions(bad, 20, 'artelingo')
    with pytest.raises(AssertionError):
        M.partitions(x, 1, 'artelingo')


def test_percept_partitions_equals_ct_score_r20_reading(old):
    ct_score = old[0]
    probs = np.load('/project/CoSiR/src/test/20261020_claim_test/outputs/stage2/full/held/percept/core__k40/P3__s0.npz'
                    )['prob'][:2000].astype(np.float32)
    got = M.percept_partitions(probs, 20, 'artelingo')
    assert got.dtype == np.int32 and got.shape == (3, 2000)
    ref = ct_score.output_partitions(probs, 'percept')['r20']
    assert np.array_equal(got, ref)
    # it clusters the L2-normalised probabilities, not the raw ones
    raw = M.partitions(probs / np.linalg.norm(probs, axis=1, keepdims=True), 20, 'artelingo')
    assert np.array_equal(raw, got)
    assert not np.array_equal(got, M.percept_partitions(probs, 9, 'artelingo'))


def test_percept_partitions_refuse_bad_input():
    with pytest.raises(AssertionError):
        M.percept_partitions(np.full((50, 40), 1.5, np.float32), 20, 'artelingo')      # not probabilities
    with pytest.raises(AssertionError):
        M.percept_partitions(np.zeros((50, 40), np.float32), 20, 'artelingo')          # zero rows


# ------------------------------------------------------------------------------------------------- gate (iii)
def test_gate_iii_ok():
    out = M.gate_iii_metrics()
    for metric in ns_paths.CLASS_METRICS:
        assert out[metric] == 'ok', out['evidence']
    ev = out['evidence']
    assert set(ev) >= set(ns_paths.CLASS_METRICS)
    for metric in ns_paths.CLASS_METRICS:
        assert ev[metric]['cases'] >= 20 and ev[metric]['max_abs_diff'] <= TOL


def test_gate_iii_fails_when_a_metric_is_broken(monkeypatch):
    # a greedy ACC and a geometric NMI must each be caught, and only their own metric
    def greedy_acc(C):
        C = np.array(C, dtype=float)
        total, used = 0.0, C.sum()
        while C.size and C.max() > 0:
            i, j = np.unravel_index(C.argmax(), C.shape)
            total += C[i, j]
            C[i, :] = 0
            C[:, j] = 0
        return total / used
    monkeypatch.setattr(M, '_acc_from_table', greedy_acc)
    out = M.gate_iii_metrics()
    assert out['acc'] == 'failed' and out['ami'] == 'ok' and out['nmi'] == 'ok' and out['ari'] == 'ok'


def test_gate_iii_fails_when_nmi_is_not_arithmetic(monkeypatch):
    # _generalized_average also feeds AMI's contingency form, so AMI fails too; only NMI and ARI are asserted.
    real = M._generalized_average
    monkeypatch.setattr(M, '_generalized_average', lambda u, v: float(np.sqrt(u * v)))
    out = M.gate_iii_metrics()
    assert out['nmi'] == 'failed' and out['ari'] == 'ok'
    assert real is not None
