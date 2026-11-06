"""Label-only r23 evaluation: same rankings and AP; no pair statistics are computed.

The AP operations mirror tcl_metrics.evaluate. Painting ids are needed only by
intervals, outside this evaluator. The self row is removed for text-to-text.
"""
import numpy as np
import torch
from torch.nn import functional as F

from mc_common import P
from tcl_metrics import _rows, _label


@torch.no_grad()
def evaluate(q, g, labels, self_match, readout, device, block=512, means=None, self_indices=None):
    assert readout in P.READOUTS and block > 0
    nq, ng = len(q), len(g)
    assert nq and ng and q.shape[1] == g.shape[1]
    if self_match:
        if self_indices is None:
            self_indices = np.arange(nq)
        assert self_indices.shape == (nq,) and np.array_equal(q, g[self_indices])
        assert len(np.unique(self_indices)) == nq
    qt, gt = _rows(q, device), _rows(g, device)
    # Frozen validation means are mandatory even for smoke, never fitted here.
    assert means is not None and len(means) == 2
    assert all(x.shape == (q.shape[1],) and x.dtype == np.float32 and np.isfinite(x).all() for x in means)
    if self_match:
        assert np.array_equal(*means)
    if readout != 'r0':
        qm, gm = (torch.as_tensor(x, dtype=torch.float32, device=device) for x in means)
        qt, gt = F.normalize(qt - qm, dim=1), F.normalize(gt - gm, dim=1)
    entries = {k: (_label(v, 'q', nq, device), _label(v, 'g', ng, device)) for k, v in labels.items()}
    ap = {k: np.full(nq, np.nan, dtype=np.float32) for k in labels}
    for start in range(0, nq, block):
        stop = min(start + block, nq)
        rows = torch.arange(start, stop, device=device)
        sim = qt[start:stop] @ gt.T
        if self_match:
            sim[torch.arange(len(rows), device=device), torch.as_tensor(self_indices[start:stop], device=device)] = -torch.inf
        order = sim.argsort(dim=1, descending=True)
        if self_match:
            order = order[:, :-1]
        if readout == 'center_k10':
            expanded = F.normalize(qt[start:stop] + gt[order[:, :10]].sum(1), dim=1)
            sim = expanded @ gt.T
            if self_match:
                sim[torch.arange(len(rows), device=device), torch.as_tensor(self_indices[start:stop], device=device)] = -torch.inf
            order = sim.argsort(dim=1, descending=True)
            if self_match:
                order = order[:, :-1]
        for key, ((ql, qv), (gl, gv)) in entries.items():
            valid = gv[order]
            rel = valid & (gl[order] == ql[start:stop, None])
            nr = rel.sum(1)
            precision = rel.cumsum(1).float() / valid.cumsum(1).clamp_min(1)
            kept = qv[start:stop] & (nr > 0)
            values = (precision * rel).sum(1) / nr.clamp_min(1)
            ap[key][start:stop] = torch.where(kept, values, torch.nan).cpu().numpy()
    return ap
