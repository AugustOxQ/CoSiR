"""Read-out menu, cross-fit and the selection rule for the r11 retrieval round (spec 2026-10-29-retrieval-design,
section 6 "Selection rule", 3.2 last paragraph, D8, D13).

Pure array functions, no file reading. A read-out is a map W (none, centering, PCA whitening to `size` dimensions,
each followed by L2 normalization) crossed with average query expansion k (0 = none).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

import rt_metrics

ASPECTS = ('emotion', 'style', 'genre')
W_KINDS = ('none', 'center', 'whiten')
W_RANK = {w: i for i, w in enumerate(W_KINDS)}
SIZES = (64, 256, 1024)
KS = (0, 1, 3, 5, 10)
TIE_TOL = 0.001     # scores within this of the best are tied
EPS_REL = 1e-6      # whitening epsilon = EPS_REL * (largest eigenvalue)


@dataclass(frozen=True)
class ReadOut:
    w: str                 # 'none' | 'center' | 'whiten'
    size: Optional[int]    # kept dimensions for 'whiten', else None
    k: int                 # query expansion neighbours, 0 = none

    def __post_init__(self):
        assert self.w in W_KINDS, self.w
        assert (self.size is not None) == (self.w == 'whiten'), self
        assert self.size is None or (isinstance(self.size, int) and self.size > 0), self
        assert self.k in KS, self.k


R0 = ReadOut('none', None, 0)


def name(ro):
    return f'{ro.w}{ro.size if ro.w == "whiten" else ""}+k{ro.k}'


def tie_key(ro):
    """Lexicographic tie order: R0 first (it is the unique minimum), no expansion before expansion,
    none < centering < whitening, smaller k, larger whitening size."""
    return (ro != R0, ro.k > 0, W_RANK[ro.w], ro.k, -(ro.size or 0))


def menu(width):
    """The read-outs of an arm whose embeddings are `width` wide, in the tie order (R0 first).
    A whitening size at or above the width becomes whitening at the full width; repeats are dropped."""
    sizes = sorted({min(s, width) for s in SIZES})
    ws = [('none', None), ('center', None)] + [('whiten', s) for s in sizes]
    ros = [ReadOut(w, s, k) for (w, s) in ws for k in KS]
    assert len(set(ros)) == len(ros)
    return sorted(ros, key=tie_key)


# ---- fit and apply ------------------------------------------------------------------------------------------------

def _eig(x):
    """mean, eigenvalues (descending) and eigenvectors (columns) of the covariance of x, in float64.
    The covariance uses the 1/(n-1) convention (np.cov); the convention only rescales all eigenvalues and eps
    together, so after the L2 normalization the read-out does not depend on it."""
    x64 = np.asarray(x, dtype=np.float64)
    mu = x64.mean(axis=0)
    xc = x64 - mu
    cov = xc.T @ xc / (len(x64) - 1)
    lam, u = np.linalg.eigh(cov)
    return mu, lam[::-1].copy(), u[:, ::-1].copy()


def fit(ro, x, _eig_cache=None):
    """Fit the read-out's map on x (float32 rows, no labels). Returns params for `apply`.
    whiten: x -> diag(lam_i + eps)^(-1/2) U_d^T (x - mu), eps = 1e-6 * lam_1, lam descending from eigh in float64."""
    x = np.asarray(x)
    assert x.dtype == np.float32 and x.ndim == 2 and len(x) > 1
    if ro.w == 'none':
        return {'w': 'none'}
    if ro.w == 'center':
        return {'w': 'center', 'mu': x.astype(np.float64).mean(axis=0)}
    assert ro.size <= x.shape[1], (ro, x.shape)
    mu, lam, u = _eig_cache if _eig_cache is not None else _eig(x)
    assert lam[0] > 0
    eps = EPS_REL * lam[0]
    d = ro.size
    assert (lam[:d] + eps > 0).all()
    return {'w': 'whiten', 'mu': mu, 'proj': u[:, :d] / np.sqrt(lam[:d] + eps)}


def _transform(params, x):
    """The map before the L2 normalization, float64."""
    x64 = np.asarray(x, dtype=np.float64)
    if params['w'] == 'none':
        return x64
    if params['w'] == 'center':
        return x64 - params['mu']
    return (x64 - params['mu']) @ params['proj']


def apply(params, x):
    """Map x with the fitted params and L2-normalize: float32 rows of norm 1."""
    x = np.asarray(x)
    assert x.dtype == np.float32 and x.ndim == 2
    z = _transform(params, x)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    return z.astype(np.float32)


def stage5_apply(ro, x_val, x_held):
    """Stage 5: fit the pick's W on all validation rows (no labels), apply it to the held rows. The k is used later
    by rt_metrics.retrieval(qe_k=ro.k) on the returned rows."""
    return apply(fit(ro, x_val), x_held)


# ---- cross-fit ----------------------------------------------------------------------------------------------------

def halves(n):
    """The two fixed validation halves: positions of the sorted validation list with even / odd parity."""
    return np.arange(0, n, 2), np.arange(1, n, 2)


def half_dropped(labels):
    """{aspect: [dropped in half 0, dropped in half 1]}: the queries rt_metrics.retrieval drops (a labelled row with
    no other labelled row of its class in its half). Depends only on the labels and the halves."""
    out = {}
    for a, lab in labels.items():
        lab = np.asarray(lab)
        counts = []
        for idx in halves(len(lab)):
            l = lab[idx]
            l = l[l >= 0].astype(np.int64)
            counts.append(int((np.bincount(l)[l] == 1).sum()))
        out[a] = counts
    return out


def crossfit_all(xs, labels, ros, workers=8):
    """Cross-fit scores of several read-outs at once (the eigendecomposition of a fit half is done once and shared
    by the whitening sizes; the same applied rows are shared by the k values).
    xs: {(aspect, seed): float32 [n, width]} on the sorted validation list (seed None for unseeded arms; one
    matrix may serve every aspect); labels: {aspect: int64 [n]}, -1 = unlabelled.
    Per seed, aspect and half h: fit on the other half (every painting, no labels), apply to half h, then
    rt_metrics.retrieval on half h with the read-out's k; AP mean; mean over the two halves; mean over seeds.
    Returns {ro: {aspect: mAP}}, keys in the order of `ros`."""
    ros = list(ros)
    assert len(set(ros)) == len(ros) and ros
    aspects = tuple(labels)
    seeds = sorted({s for (_, s) in xs}, key=lambda v: (v is not None, v))
    for a in aspects:
        for s in seeds:
            assert (a, s) in xs, ('missing matrix', a, s)
        a_seeds = sorted({s for (b, s) in xs if b == a}, key=lambda v: (v is not None, v))
        assert a_seeds == [None] or a_seeds == [0, 1, 2, 3, 4], ('seed set', a, a_seeds)
    want_dropped = half_dropped(labels)
    checked = set()  # (aspect, half) whose dropped count was compared
    n = len(labels[aspects[0]])
    halves_ = halves(n)
    acc = {ro: {a: [] for a in aspects} for ro in ros}  # per-seed means
    need_eig = any(ro.w == 'whiten' for ro in ros)
    for s in seeds:
        eigs = {}  # (id of matrix, half) -> eigendecomposition; one seed at a time bounds the memory
        for a in aspects:
            x = xs[(a, s)]
            assert x.shape[0] == n and labels[a].shape == (n,)
            per_half = {ro: [] for ro in ros}
            for h in (0, 1):
                score_idx, fit_idx = halves_[h], halves_[1 - h]
                x_fit = x[fit_idx]
                key = (id(x), h)
                if need_eig and key not in eigs:
                    eigs[key] = _eig(x_fit)
                applied = {}  # (w, size) -> mapped rows of this half
                for ro in ros:
                    wk = (ro.w, ro.size)
                    if wk not in applied:
                        params = fit(ro, x_fit, _eig_cache=eigs.get(key))
                        applied[wk] = apply(params, x[score_idx])
                    res = rt_metrics.retrieval(applied[wk], labels[a][score_idx], qe_k=ro.k, workers=workers)
                    if (a, h) not in checked:
                        assert res['dropped'] == want_dropped[a][h], ('dropped', a, h, res['dropped'], want_dropped[a][h])
                        checked.add((a, h))
                    per_half[ro].append(res['ap'].mean())
            for ro in ros:
                acc[ro][a].append(np.mean(per_half[ro]))
    return {ro: {a: float(np.mean(acc[ro][a])) for a in aspects} for ro in ros}


def crossfit(xs, labels, ro, workers=8):
    """Cross-fit mAP per aspect of one read-out: {aspect: mAP}. See crossfit_all."""
    return crossfit_all(xs, labels, [ro], workers)[ro]


# ---- selection rule -----------------------------------------------------------------------------------------------

SLACK = 1e-12   # absorbs float rounding at the two boundaries (guard: exactly r_a below; tie: exactly 0.001 below)


def _aspects(rows, ranges):
    assert R0 in rows, 'the R0 row is needed'
    aspects = tuple(rows[R0])
    assert set(aspects) == set(ASPECTS) and set(ranges) >= set(aspects), (aspects, tuple(ranges))
    assert all(set(r) == set(aspects) for r in rows.values())
    assert all(ranges[a] > 0 for a in aspects)
    return aspects


def score(rows, ranges):
    """r10's mean gain per read-out: (1/3) sum_a (mAP_a - mAP_a(R0)) / r_a.
    rows: {ReadOut: {aspect: mAP}} (cross-fit values, seeds averaged), R0 among them; ranges: {aspect: r_a} (c*'s
    validation seed range, used for every arm). Returns {ReadOut: float}."""
    aspects = _aspects(rows, ranges)
    base = rows[R0]
    return {ro: sum((m[a] - base[a]) / ranges[a] for a in aspects) / len(aspects) for ro, m in rows.items()}


def guard(rows, ranges):
    """{ReadOut: bool}, True = kept. A read-out is dropped when any aspect is below mAP_a(R0) - r_a."""
    aspects = _aspects(rows, ranges)
    base = rows[R0]
    return {ro: all(m[a] >= base[a] - ranges[a] - SLACK for a in aspects) for ro, m in rows.items()}


def select(rows, ranges):
    """The pick: among the read-outs passing the guard, those with a score within 0.001 of the best; the first of
    them in the tie order (R0, no expansion before expansion, none < center < whiten, smaller k, larger size).
    R0 always passes the guard, so a pick exists."""
    sc, keep = score(rows, ranges), guard(rows, ranges)
    assert keep[R0]
    cands = [ro for ro in rows if keep[ro]]
    best = max(sc[ro] for ro in cands)
    tied = [ro for ro in cands if sc[ro] >= best - TIE_TOL - SLACK]
    return min(tied, key=tie_key)
