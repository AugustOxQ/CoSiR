"""Text-claim rankings and intervals (2026-11-06 text-claim design §3.4, §3.5(iii))."""
import numpy as np
import torch
from torch.nn import functional as F

import tcl_paths  # noqa: F401; installs frozen round imports
from tx_metrics import macro_map, painting_bootstrap, summarize_pair


def fit_means(t, i):
    """Fit validation means over every row, with float64 accumulation."""
    assert t.ndim == i.ndim == 2 and t.shape[1] == i.shape[1]
    assert len(t) and len(i) and t.shape[1]
    assert t.dtype == i.dtype == np.float32
    assert np.isfinite(t).all() and np.isfinite(i).all()
    return t.mean(0, dtype=np.float64).astype(np.float32), i.mean(0, dtype=np.float64).astype(np.float32)


def _rows(x, device):
    assert x.ndim == 2 and x.dtype == np.float32 and np.isfinite(x).all()
    assert np.allclose(np.linalg.norm(x, axis=1), 1, atol=2e-4)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _label(entry, side, n, device):
    x = entry[side]
    assert x.shape == (n,) and x.dtype == np.int64
    keep = entry.get(side + '_keep')
    if keep is not None:
        assert keep.shape == (n,) and keep.dtype == bool
    valid = x >= 0 if keep is None else (x >= 0) & keep
    return (torch.as_tensor(x, device=device), torch.as_tensor(valid, device=device))


@torch.no_grad()
def evaluate(q, g, q_pid, g_pid, labels, self_match, readout, device, block=512, means=None):
    """r22 ranking, optionally centered with supplied validation (query, gallery) means."""
    assert readout in ('r0', 'center', 'center_k10') and block > 0
    nq, ng = len(q), len(g)
    assert nq and ng and q.shape[1] == g.shape[1]
    assert (q_pid is None) == (g_pid is None)
    if q_pid is not None:
        assert q_pid.shape == (nq,) and g_pid.shape == (ng,)
        assert q_pid.dtype == g_pid.dtype == np.int64
    if self_match:
        assert q.shape == g.shape and q_pid is not None and np.array_equal(q_pid, g_pid)
        assert np.array_equal(q, g)
    qt, gt = _rows(q, device), _rows(g, device)
    if means is not None:
        assert len(means) == 2
        assert all(x.shape == (q.shape[1],) and x.dtype == np.float32 and np.isfinite(x).all()
                   for x in means)
        if self_match:
            assert np.array_equal(means[0], means[1]), 'text to text needs one shared mean'
    if readout != 'r0':
        if means is None:
            # Word queries have no pair relation and use only the gallery mean.
            qm = gt.mean(0) if q_pid is None else qt.mean(0)
            gm = qm if self_match else gt.mean(0)
        else:
            qm, gm = (torch.as_tensor(x, dtype=torch.float32, device=device) for x in means)
        qt, gt = F.normalize(qt - qm, dim=1), F.normalize(gt - gm, dim=1)
    entries = {k: (_label(v, 'q', nq, device), _label(v, 'g', ng, device))
               for k, v in labels.items()}
    if any(v.get('exclude_same_painting', False) for v in labels.values()):
        assert q_pid is not None
    qp = None if q_pid is None else torch.as_tensor(q_pid, device=device)
    gp = None if g_pid is None else torch.as_tensor(g_pid, device=device)
    ap = {k: np.full(nq, np.nan, dtype=np.float32) for k in labels}
    chances = {k: [] for k in labels}
    first = None if qp is None else np.full(nq, np.nan, dtype=np.float32)
    for start in range(0, nq, block):
        stop = min(start + block, nq)
        rows = torch.arange(start, stop, device=device)
        sim = qt[start:stop] @ gt.T
        if self_match:
            sim[torch.arange(len(rows), device=device), rows] = -torch.inf
        order = sim.argsort(dim=1, descending=True)
        if self_match:
            order = order[:, :-1]
        if readout == 'center_k10':
            expanded = F.normalize(qt[start:stop] + gt[order[:, :10]].sum(1), dim=1)
            sim = expanded @ gt.T
            if self_match:
                sim[torch.arange(len(rows), device=device), rows] = -torch.inf
            order = sim.argsort(dim=1, descending=True)
            if self_match:
                order = order[:, :-1]
        if first is not None:
            same = gp[order] == qp[start:stop, None]
            has = same.any(1)
            ranks = same.to(torch.int64).argmax(1).float() + 1 if order.shape[1] else torch.zeros(len(rows), device=device)
            first[start:stop] = torch.where(has, ranks, torch.nan).cpu().numpy()
        for key, ((ql, qv), (gl, gv)) in entries.items():
            valid = gv[order]
            if labels[key].get('exclude_same_painting', False):
                valid = valid & (gp[order] != qp[start:stop, None])
            rel = valid & (gl[order] == ql[start:stop, None])
            nr, nv = rel.sum(1), valid.sum(1)
            precision = rel.cumsum(1).float() / valid.cumsum(1).clamp_min(1)
            kept = qv[start:stop] & (nr > 0)
            values = (precision * rel).sum(1) / nr.clamp_min(1)
            ap[key][start:stop] = torch.where(kept, values, torch.nan).cpu().numpy()
            chances[key].append((nr[kept].float() / nv[kept]).cpu().numpy())
    chance = {k: float(np.mean(np.concatenate(v))) if any(len(x) for x in v) else float('nan')
              for k, v in chances.items()}
    n_kept = {k: int(np.isfinite(v).sum()) for k, v in ap.items()}
    # Filtered queries leave the task; only eligible queries without a hit are dropped.
    dropped = {k: int(entries[k][0][1].sum().item()) - n for k, n in n_kept.items()}
    return dict(ap=ap, chance=chance, n_kept=n_kept, dropped=dropped, first_rank=first)


def replicate(sum_c, sum_m, counts, I, S):
    """One painting/seed draw, weighting query sums and counts by painting multiplicity."""
    assert sum_c.ndim == 2 and sum_c.dtype == np.float64
    k, m = sum_c.shape
    assert k > 0 and m > 0
    assert sum_m.shape == counts.shape == (m,) and sum_m.dtype == counts.dtype == np.float64
    assert np.isfinite(sum_c).all() and np.isfinite(sum_m).all() and np.isfinite(counts).all()
    assert (counts > 0).all() and (counts == np.floor(counts)).all()
    I, S = np.asarray(I), np.asarray(S)
    assert I.shape == (m,) and S.shape == (k,)
    assert np.issubdtype(I.dtype, np.integer) and np.issubdtype(S.dtype, np.integer)
    assert ((I >= 0) & (I < m)).all() and ((S >= 0) & (S < k)).all()
    w = np.bincount(I, minlength=m).astype(np.float64)
    den = w @ counts
    values = (sum_c @ w) / den
    return float(values[S].mean() - (w @ sum_m) / den)


_CHUNK = 250  # Bound the draw-by-painting weight matrix, as in rt_stats.paired.


def painting_interval(cand, comp, pid, draws=10000):
    """Return (point, lower, upper, n_units) for scored queries, resampling paintings then seeds."""
    assert cand.ndim == 2 and cand.dtype == np.float64
    k, nq = cand.shape
    assert k > 0 and nq > 0
    assert comp.shape == pid.shape == (nq,) and comp.dtype == np.float64 and pid.dtype == np.int64
    assert np.isfinite(cand).all() and np.isfinite(comp).all(), 'pass scored queries only'
    assert (pid >= 0).all()
    assert isinstance(draws, (int, np.integer)) and draws > 0
    # np.unique orders the scored painting positions, omitting paintings without a scored query.
    units, inverse, counts = np.unique(pid, return_inverse=True, return_counts=True)
    m = len(units)
    counts = counts.astype(np.float64)
    sum_c = np.stack([np.bincount(inverse, weights=row, minlength=m) for row in cand])
    sum_m = np.bincount(inverse, weights=comp, minlength=m)
    cols = np.vstack((sum_c, sum_m))
    rng = np.random.default_rng(0)
    seeds = list(range(k))
    boot = np.empty(draws, dtype=np.float64)
    for lo in range(0, draws, _CHUNK):
        n = min(_CHUNK, draws - lo)
        weights = np.empty((n, m), dtype=np.float64)
        sel = np.empty((n, k), dtype=np.int64)
        for b in range(n):  # Preserve the plain loop's I, then S RNG stream across chunks.
            I = rng.integers(0, m, size=m)
            sel[b] = rng.choice(seeds, size=k, replace=True)
            weights[b] = np.bincount(I, minlength=m)
        den = weights @ counts
        values = (weights @ cols.T) / den[:, None]
        boot[lo:lo + n] = np.take_along_axis(values[:, :k], sel, axis=1).mean(1) - values[:, k]
    lower, upper = np.percentile(boot, [2.5, 97.5])
    return float(cand.mean() - comp.mean()), float(lower), float(upper), m


def outcome(lower, seed_range):
    """Apply §3.4's strict lower-bound comparisons with the validation seed range."""
    assert np.isfinite(lower) and np.isfinite(seed_range) and seed_range >= 0
    if lower > 0:
        return 'WIN'
    if lower > -seed_range:
        return 'LEVEL'
    return 'BELOW'
