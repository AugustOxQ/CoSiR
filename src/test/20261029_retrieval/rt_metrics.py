"""Retrieval metrics for the r11 retrieval round (spec 2026-10-29-retrieval-design, 3.4 and 6).

Pure array functions, no file reading. Rows are used as given (callers pass L2-normalized float32 rows), so
cosine = dot product. Ranking: the gallery excludes the query; descending similarity by a stable sort over the
gallery in row order, so ties keep gallery row order.

Speed: queries go in blocks; `workers` > 1 runs blocks in forked processes, each limited to one BLAS thread.
"""
import multiprocessing as mp

import numpy as np
from threadpoolctl import threadpool_limits

BLOCK = 256        # queries per block in retrieval(): sims 256 x n float32 (12 MB at n = 12,281) plus the sort
GRADED_BLOCK = 64  # queries per block in graded(): gain matrices 64 x n float64

_SHARED = {}       # set before the fork; read by the workers


def _ranked(sim, rows):
    """sim: (B, n) similarities of B queries to all n rows; rows: each query's own row index.
    Returns (B, n-1) gallery row indices, best first, the query itself removed (ties: row order)."""
    sim = sim.copy()
    sim[np.arange(len(rows)), rows] = -np.inf  # -(-inf) = +inf sorts last: only the query's own entry
    order = np.argsort(-sim, axis=1, kind='stable')
    assert (order[:, -1] == rows).all()
    return order[:, :-1]


def _retrieval_block(rows):
    x, labels, qe_k = _SHARED['x'], _SHARED['labels'], _SHARED['qe_k']
    order = _ranked(x[rows] @ x.T, rows)
    if qe_k > 0:
        q = x[rows] + x[order[:, :qe_k]].sum(axis=1)  # itself + top-k (the mean's scale drops out in the norm)
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        order = _ranked(q @ x.T, rows)
    rel = labels[order] == labels[rows][:, None]
    prec = np.cumsum(rel, axis=1) / np.arange(1, rel.shape[1] + 1)
    ap = (prec * rel).sum(axis=1) / rel.sum(axis=1)
    p10 = rel[:, :10].sum(axis=1) / 10.0
    return ap, p10


def _graded_block(rows):
    x, dist = _SHARED['x'], _SHARED['dist']
    sim = x[rows] @ x.T
    order = _ranked(sim, rows)
    gain = np.zeros(sim.shape)
    for c in range(dist.shape[1]):
        gain += np.minimum(dist[rows, c][:, None], dist[None, :, c])
    gain[np.arange(len(rows)), rows] = -1.0  # below any real gain (>= 0): last in the ideal sort
    got = np.take_along_axis(gain, order, axis=1)
    ideal = -np.sort(-gain, axis=1)[:, :-1]
    disc = 1.0 / np.log2(np.arange(2, got.shape[1] + 2))
    idcg = ideal @ disc
    assert (idcg > 0).all()
    return (got @ disc) / idcg


def _init_worker():
    threadpool_limits(1)


def _run(fn, rows, block, workers):
    chunks = [rows[i:i + block] for i in range(0, len(rows), block)]
    if workers <= 1:
        res = [fn(c) for c in chunks]
    else:
        with mp.get_context('fork').Pool(workers, initializer=_init_worker) as pool:
            res = pool.map(fn, chunks, chunksize=1)
    return res


def retrieval(x, labels, qe_k=0, workers=8):
    """AP and P@10 per query, over the rows with labels >= 0 (queries in row order; gallery = the same rows minus
    the query). A query with no relevant gallery item is dropped and counted.
    Returns {'ap': float64[nq], 'p10': float64[nq], 'dropped': int, 'rows': int64[nq]} where rows are the
    indices into the given x / labels of the kept queries."""
    x = np.asarray(x)
    labels = np.asarray(labels)
    assert x.dtype == np.float32 and x.ndim == 2 and labels.shape == (len(x),)
    assert qe_k >= 0
    keep = np.flatnonzero(labels >= 0)
    xs, ls = x[keep], labels[keep].astype(np.int64)
    n_rel = np.bincount(ls)[ls] - 1
    qrows = np.flatnonzero(n_rel > 0)
    assert len(qrows) > 0
    _SHARED.update(x=xs, labels=ls, qe_k=int(qe_k))
    try:
        res = _run(_retrieval_block, qrows, BLOCK, workers)
    finally:
        _SHARED.clear()
    return {'ap': np.concatenate([r[0] for r in res]).astype(np.float64),
            'p10': np.concatenate([r[1] for r in res]).astype(np.float64),
            'dropped': int((n_rel == 0).sum()), 'rows': keep[qrows]}


def graded(x, dist, workers=8):
    """nDCG per query, every row a query, gallery = all other rows. gain(q, g) = sum_c min(dist[q, c], dist[g, c]);
    DCG = sum_r gain_r / log2(r + 1) over the full similarity ranking; nDCG = DCG / DCG of the gallery sorted by
    gain. Returns float64[n]."""
    x = np.asarray(x)
    dist = np.asarray(dist)
    assert x.dtype == np.float32 and x.ndim == 2 and dist.ndim == 2 and len(dist) == len(x)
    dist = dist.astype(np.float64)
    _SHARED.update(x=x, dist=dist)
    try:
        res = _run(_graded_block, np.arange(len(x)), GRADED_BLOCK, workers)
    finally:
        _SHARED.clear()
    return np.concatenate(res).astype(np.float64)
