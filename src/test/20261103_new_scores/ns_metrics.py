"""r17 ticket 01: class-count clustering metrics (AMI, NMI, ARI, ACC), contingency forms, partitions at K, gate (iii).

Spec docs/superpowers/specs/2026-11-03-new-scores-design.md §3.4 (class-count metrics), D6, D16, D20, stage 0 (iii).

- Array forms `ami`, `nmi`, `ari`, `acc` take labelled 1-D integer arrays (no -1 allowed): AMI, NMI and ARI are
  scikit-learn's (NMI with `average_method='arithmetic'`); ACC is the Hungarian matching that maximises matched rows
  on the contingency table (clusters without a match count zero; the denominator is all labelled rows).
- `contingency(y, c, weights, shape)` builds the table (rows = classes, columns = clusters) from rows, with -1 labels
  dropped and integer row weights (a bootstrap draw's multiplicities); `from_contingency(C, metric)` computes each metric
  from such a table without materialising repeated rows and equals the array form on the repeated rows (tested to
  `ns_paths.TOL_METRIC`). Empty rows and columns are dropped first, as scikit-learn's own contingency matrix omits them.
- `partitions(x, k, set_name)` and `percept_partitions(probs, k, set_name)`: r10's MiniBatchKMeans (n_init 3, batch
  4,096, states 0 to 2) at any K, threads from `ns_paths.KMEANS_THREADS[set_name]` set in code (D20).
- `gate_iii_metrics()`: stage 0 (iii) metric part: test cases against scikit-learn and a brute-force best permutation.
"""
import itertools
import math

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import mutual_info_score
from sklearn.metrics.cluster._expected_mutual_info_fast import expected_mutual_information
from threadpoolctl import threadpool_limits
import sklearn

# The contingency forms copy scikit-learn 1.6.1's private code paths (_supervised.py); pin the version they copy.
assert sklearn.__version__ == '1.6.1', f'ns_metrics copies scikit-learn 1.6.1 internals; found {sklearn.__version__}'

import ns_paths

METRICS = tuple(ns_paths.CLASS_METRICS)          # ('ami', 'nmi', 'ari', 'acc')
N_INIT = 3                                       # r10's MiniBatchKMeans settings (rs_vh: N_INIT, BATCH, STATES)
BATCH = 4096
EPS = 1e-12                                      # the floor of `apc_data.l2`, which ct_score uses for PercepT rows


# ------------------------------------------------------------------------------------------------ contingency
def _labels(a, name, n=None):
    a = np.asarray(a)
    assert a.ndim == 1 and np.issubdtype(a.dtype, np.integer), f'{name}: need a 1-D integer array, got {a.dtype} {a.shape}'
    assert n is None or len(a) == n, f'{name}: {len(a)} rows, expected {n}'
    return a.astype(np.int64)


def contingency(y, c, weights=None, shape=None):
    """Contingency table int64 [classes, clusters] of labels y (-1 = unlabelled, dropped) and cluster ids c (>= 0).
    `weights` (non-negative integers per row, e.g. a bootstrap draw's multiplicities) count each row that many times.
    `shape` fixes the table size (otherwise the largest id + 1 of each); an id outside it is refused."""
    y = _labels(y, 'y')
    c = _labels(c, 'c', len(y))
    assert (c >= 0).all(), 'cluster ids must be >= 0'
    if weights is None:
        w = np.ones(len(y), dtype=np.int64)
    else:
        wf = np.asarray(weights)
        assert wf.shape == y.shape, f'weights {wf.shape} against {y.shape}'
        w = np.rint(wf).astype(np.int64)
        assert (w == wf).all() and (w >= 0).all(), 'weights must be non-negative integers'
    keep = y >= 0
    y, c, w = y[keep], c[keep], w[keep]
    if shape is None:
        shape = (int(y.max()) + 1 if len(y) else 0, int(c.max()) + 1 if len(c) else 0)
    ky, kc = shape
    assert (y < ky).all() and (c < kc).all(), f'an id lies outside the table shape {shape}'
    flat = np.bincount(y * kc + c, weights=w, minlength=ky * kc)          # float64 sums of integers: exact below 2**53
    return np.rint(flat).astype(np.int64).reshape(ky, kc)


def _clean(C):
    """Integer table with empty rows and columns dropped."""
    C = np.asarray(C)
    assert C.ndim == 2, C.shape
    Ci = np.rint(C).astype(np.int64)
    assert (Ci == C).all() and (Ci >= 0).all(), 'a contingency table must hold non-negative integers'
    Ci = Ci[Ci.sum(axis=1) > 0][:, Ci.sum(axis=0) > 0]
    assert Ci.size > 0, 'an empty contingency table'
    return Ci


def _generalized_average(u, v):
    """Arithmetic mean of the two entropies (D6; scikit-learn's `average_method='arithmetic'`)."""
    return np.mean([u, v])


def _entropy(counts):
    """scikit-learn's `entropy` of a labelling, from its class sizes."""
    pi = np.asarray(counts, dtype=np.float64)
    pi_sum = np.sum(pi)
    return float(-np.sum((pi / pi_sum) * (np.log(pi) - math.log(pi_sum))))


def _acc_from_table(C):
    """Hungarian matching that maximises matched rows; unmatched clusters or classes count zero; / all rows."""
    C = np.asarray(C)
    r, k = linear_sum_assignment(C, maximize=True)
    return float(C[r, k].sum() / C.sum())


def _ari_from_table(C):
    """scikit-learn's `adjusted_rand_score` from pair counts, in exact integer arithmetic."""
    n = int(C.sum())
    n_c = C.sum(axis=1)
    n_k = C.sum(axis=0)
    ss = int((C ** 2).sum())
    tp = ss - n
    fp = int((C @ n_k).sum()) - ss
    fn = int((C.T @ n_c).sum()) - ss
    tn = n * n - fp - fn - ss
    if fn == 0 and fp == 0:
        return 1.0
    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))


def from_contingency(C, metric):
    """One of `METRICS` from a (weighted) contingency table [classes, clusters] of non-negative integers. Equal to the
    array form on the same rows with repeats."""
    assert metric in METRICS, f'{metric!r} is not a class-count metric {METRICS}'
    Ci = _clean(C)
    if metric == 'acc':
        return _acc_from_table(Ci)
    if metric == 'ari':
        return _ari_from_table(Ci)
    n = int(Ci.sum())
    h_true, h_pred = _entropy(Ci.sum(axis=1)), _entropy(Ci.sum(axis=0))
    if Ci.shape == (1, 1):
        return 1.0                                  # both labellings constant: perfect match (scikit-learn)
    if metric == 'ami':
        sparse = sp.csr_matrix(Ci)
        mi = mutual_info_score(None, None, contingency=sparse)
        emi = expected_mutual_information(sparse, n)
        denominator = _generalized_average(h_true, h_pred) - emi
        if denominator < 0:
            denominator = min(denominator, -np.finfo('float64').eps)
        else:
            denominator = max(denominator, np.finfo('float64').eps)
        return float((mi - emi) / denominator)
    mi = mutual_info_score(None, None, contingency=sp.csr_matrix(Ci.astype(np.float64)))
    if mi == 0:
        return 0.0
    return float(mi / _generalized_average(h_true, h_pred))


# ------------------------------------------------------------------------------------------------ array forms
def _pair(y, c):
    y = _labels(y, 'y')
    c = _labels(c, 'c', len(y))
    assert len(y) > 0 and (y >= 0).all(), 'array forms take labelled rows only: mask y >= 0 first'
    return y, c


def ami(y, c):
    y, c = _pair(y, c)
    return float(adjusted_mutual_info_score(y, c))


def nmi(y, c):
    y, c = _pair(y, c)
    return float(normalized_mutual_info_score(y, c, average_method='arithmetic'))


def ari(y, c):
    y, c = _pair(y, c)
    return float(adjusted_rand_score(y, c))


def acc(y, c):
    y, c = _pair(y, c)
    return _acc_from_table(contingency(y, c))


def class_count(labels):
    """Number of classes with at least one labelled row (-1 = no label)."""
    labels = _labels(labels, 'labels')
    return int(len(np.unique(labels[labels >= 0])))


def score_partitions(metric, y, parts):
    """Mean over the states of `metric` on the labelled rows: clustering ran on every row, the mask (y >= 0) is applied
    to labels and partitions here, after clustering. parts int [states, n]."""
    assert metric in METRICS
    parts = np.asarray(parts)
    assert parts.ndim == 2, f'parts must be [states, n], got {parts.shape}'
    y = _labels(y, 'y', parts.shape[1])
    keep = y >= 0
    return float(np.mean([globals()[metric](y[keep], p[keep]) for p in parts]))


# ------------------------------------------------------------------------------------------------ partitions
def _fit_states(x, k, set_name):
    assert set_name in ns_paths.KMEANS_THREADS, f'unknown set {set_name!r}'
    x = np.asarray(x).astype(np.float32)
    assert x.ndim == 2 and x.shape[0] >= k >= 2, (x.shape, k)
    assert np.isfinite(x).all(), 'non-finite rows'
    assert np.abs(np.linalg.norm(x, axis=1) - 1).max() < ns_paths.TOL_UNIT, 'rows are not unit norm'
    out = []
    with threadpool_limits(limits=ns_paths.KMEANS_THREADS[set_name]):
        for state in ns_paths.KMEANS_STATES:
            km = MiniBatchKMeans(n_clusters=k, random_state=state, n_init=N_INIT, batch_size=BATCH)
            out.append(km.fit(x).labels_)
    return np.stack(out).astype(np.int32)


def partitions(x, k, set_name):
    """r10's MiniBatchKMeans partitions of unit rows with k clusters: int32 [3, n] (n_init 3, batch 4,096, states 0 to
    2, float32). Threads `KMEANS_THREADS[set_name]` (D20), set here whatever the environment says. All rows are
    clustered; labels are masked later."""
    return _fit_states(x, k, set_name)


def percept_partitions(probs, k, set_name):
    """k-means on the L2-normalised PercepT probabilities [n, d] (D16): int32 [3, n]; the r20 reading of
    `ct_score.output_partitions(probs, 'percept')` with k in place of 20."""
    p = np.asarray(probs).astype(np.float32)
    assert p.ndim == 2 and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'need probabilities in [0, 1]'
    norm = np.linalg.norm(p, axis=1, keepdims=True)
    assert (norm > 0).all(), 'a zero probability row'
    return _fit_states(p / np.maximum(norm, EPS), k, set_name)


# ------------------------------------------------------------------------------------------------ gate (iii)
GATE_CASES = ((30, 3, 3), (40, 5, 5), (25, 4, 6), (50, 7, 7), (60, 6, 3), (35, 2, 7), (20, 7, 5))
GATE_EDGES = ((np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])), (np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0])),
              (np.array([0, 0, 0, 0]), np.array([0, 1, 2, 3])), (np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3])),
              (np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])), (np.array([0, 0, 1, 1, 1, 1]), np.array([0, 0, 1, 1, 2, 2])))


def _reference(metric, y, c):
    """The independent value: scikit-learn, or a brute-force best permutation for ACC (table padded to square)."""
    if metric == 'ami':
        return float(adjusted_mutual_info_score(y, c))
    if metric == 'nmi':
        return float(normalized_mutual_info_score(y, c, average_method='arithmetic'))
    if metric == 'ari':
        return float(adjusted_rand_score(y, c))
    ys, cs = np.unique(y), np.unique(c)
    m = max(len(ys), len(cs))
    table = np.zeros((m, m))
    for i, a in enumerate(ys):
        for j, b in enumerate(cs):
            table[i, j] = np.sum((y == a) & (c == b))
    rows = np.arange(m)
    return max(table[rows, list(p)].sum() for p in itertools.permutations(range(m))) / len(y)


def gate_iii_metrics():
    """Stage 0 (iii), metric part. Returns {'ami': 'ok'|'failed', 'nmi': ..., 'ari': ..., 'acc': ..., 'evidence':
    {metric: {cases, max_abs_diff, error}}}. Each metric is checked in its array form and in its contingency form (with
    repeated rows, empty classes and clusters) against the independent value, within `TOL_METRIC`; ACC against a
    brute-force best permutation (K up to 7)."""
    cases = [(np.asarray(y), np.asarray(c), None) for y, c in GATE_EDGES]
    for seed in range(4):
        rng = np.random.default_rng(1000 + seed)
        for n, ky, kc in GATE_CASES:
            y, c = rng.integers(0, ky, n), rng.integers(0, kc, n)
            w = rng.integers(0, 4, n)
            w[0] = max(w[0], 1)
            cases.append((y, c, (w, ky, kc)))
    out, evidence = {}, {}
    for metric in METRICS:
        worst, error, count = 0.0, None, 0
        try:
            for y, c, extra in cases:
                ref = _reference(metric, y, c)
                worst = max(worst, abs(globals()[metric](y, c) - ref),
                            abs(from_contingency(contingency(y, c), metric) - ref))
                count += 1
                if extra is not None:                  # weighted table against the repeated rows
                    w, ky, kc = extra
                    ref = _reference(metric, np.repeat(y, w), np.repeat(c, w))
                    worst = max(worst, abs(from_contingency(contingency(y, c, w, (ky + 2, kc + 1)), metric) - ref))
        except Exception as e:          # a crash is a failure with its message as evidence
            error, worst = f'{type(e).__name__}: {e}', float('inf')
        evidence[metric] = {'cases': count, 'max_abs_diff': worst, 'error': error}
        out[metric] = 'ok' if error is None and worst <= ns_paths.TOL_METRIC else 'failed'
    out['evidence'] = evidence
    return out
