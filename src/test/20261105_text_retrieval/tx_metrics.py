"""Blocked float32 cosine rankings, label AP and paired painting bootstrap."""
import numpy as np
import torch
from torch.nn import functional as F


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
def evaluate(q, g, q_pid, g_pid, labels, self_match, readout, device, block=512):
    """One ranking serves all entries; invalid gallery positions never count as misses."""
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
    if readout != 'r0':
        # Word queries have no pair relation and use only the gallery mean.
        qm = gt.mean(0) if q_pid is None else qt.mean(0)
        gm = qm if self_match else gt.mean(0)
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


def macro_map(ap, q_labels):
    assert ap.shape == q_labels.shape
    keep = (q_labels >= 0) & np.isfinite(ap)
    means = [float(ap[keep & (q_labels == c)].mean()) for c in np.unique(q_labels[keep])]
    return float(np.mean(means)) if means else float('nan')


def summarize_pair(first_rank):
    """Pair rates include missing hits as failures; MedR and MRR need finite ranks."""
    assert first_rank.ndim == 1
    valid = np.isfinite(first_rank)
    n = len(first_rank)
    return dict(r1=float(np.mean(first_rank <= 1)), r5=float(np.mean(first_rank <= 5)),
                r10=float(np.mean(first_rank <= 10)),
                medr=float(np.median(first_rank[valid])) if valid.any() else float('nan'),
                mrr=float(np.where(valid, 1 / first_rank, 0).sum() / n) if n else float('nan'),
                n_query=n)


def painting_bootstrap(diff_ap, q_pid, n_pid, n=1000, seed=0):
    assert diff_ap.shape == q_pid.shape and q_pid.dtype == np.int64
    assert n_pid > 0 and n > 0 and ((q_pid >= 0) & (q_pid < n_pid)).all()
    keep = np.isfinite(diff_ap)
    if not keep.any():
        return (float('nan'),) * 3
    sums = np.bincount(q_pid[keep], weights=diff_ap[keep], minlength=n_pid)
    counts = np.bincount(q_pid[keep], minlength=n_pid)
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, n, 100):
        w = rng.multinomial(n_pid, np.full(n_pid, 1 / n_pid), size=min(100, n - start))
        den = w @ counts
        draws.extend(((w @ sums)[den > 0] / den[den > 0]).tolist())
    assert draws, 'no finite bootstrap replicate'
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(sums.sum() / counts.sum()), float(lo), float(hi)
