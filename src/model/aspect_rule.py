"""The training-free agreement rule for aspect conditions (CVPR plan spec §6) and per-episode z-fusion."""

import torch
from torch import Tensor
from torch.nn import functional as F


def agreement_weights(sup_img: Tensor, sup_txt: Tensor, con_img: Tensor, con_txt: Tensor) -> Tensor:
    """(E,S,F) x4 -> (E,F): ReLU(mean_S img*txt - mean_C img*txt), L1-normalized; an all-zero row stays zero.
    An episode whose codes contain any non-finite value gets an all-NaN weight row (a miss, never a silent hit)."""
    gap = (sup_img * sup_txt).mean(dim=-2) - (con_img * con_txt).mean(dim=-2)
    bad = ~(torch.isfinite(sup_img).flatten(1).all(-1) & torch.isfinite(sup_txt).flatten(1).all(-1)
            & torch.isfinite(con_img).flatten(1).all(-1) & torch.isfinite(con_txt).flatten(1).all(-1))
    w = F.relu(gap)
    total = w.sum(dim=-1, keepdim=True)
    w = torch.where(total > 0, w / total.clamp_min(1e-12), torch.zeros_like(w))
    return torch.where(bad[:, None], torch.full_like(w, float("nan")), w)


def zscore_rows(x: Tensor) -> Tensor:
    """Per-row z-score over candidates (population std); a constant finite row becomes all zeros and a row that is
    not entirely finite becomes all NaN."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False)
    z = torch.where(std > 0, (x - mean) / std.clamp_min(1e-12), torch.zeros_like(x))
    return torch.where(torch.isfinite(x).all(dim=-1, keepdim=True), z, torch.full_like(z, float("nan")))


def zfuse(cos: Tensor, term: Tensor, lam: float) -> Tensor:
    """z(cos) + lam * z(term); lam = inf means z(term) alone, lam = 0 means z(cos) alone.
    Those two cases depend only on the input they use; otherwise a non-finite row in either input gives a NaN row."""
    if lam == float("inf"):
        return zscore_rows(term)
    if lam == 0:
        return zscore_rows(cos)
    return zscore_rows(cos) + lam * zscore_rows(term)
