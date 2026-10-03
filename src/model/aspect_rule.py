"""The training-free agreement rule for aspect conditions (CVPR plan spec §6) and per-episode z-fusion."""

import torch
from torch import Tensor
from torch.nn import functional as F


def agreement_weights(sup_img: Tensor, sup_txt: Tensor, con_img: Tensor, con_txt: Tensor) -> Tensor:
    """(E,S,F) x4 -> (E,F): ReLU(mean_S img*txt - mean_C img*txt), L1-normalized; an all-zero row stays zero."""
    gap = (sup_img * sup_txt).mean(dim=-2) - (con_img * con_txt).mean(dim=-2)
    w = F.relu(gap)
    total = w.sum(dim=-1, keepdim=True)
    return torch.where(total > 0, w / total.clamp_min(1e-12), torch.zeros_like(w))


def zscore_rows(x: Tensor) -> Tensor:
    """Per-row z-score over candidates (population std); a constant row becomes all zeros."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False)
    return torch.where(std > 0, (x - mean) / std.clamp_min(1e-12), torch.zeros_like(x))


def zfuse(cos: Tensor, term: Tensor, lam: float) -> Tensor:
    """z(cos) + lam * z(term); lam = inf means z(term) alone, lam = 0 means z(cos) alone."""
    if lam == float("inf"):
        return zscore_rows(term)
    if lam == 0:
        return zscore_rows(cos)
    return zscore_rows(cos) + lam * zscore_rows(term)
