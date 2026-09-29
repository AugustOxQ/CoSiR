"""Nonparametric condition weights and the conditional score s(I, T | c).

Task 9 of the condition-interface plan showed the L1-normalized naive rule is
the scale-fair baseline and that the learned recovery head adds nothing; this
module is the reusable form of both for stage (d).
"""

import torch
from torch import Tensor
from torch.nn import functional as F


def pair_codes(img_codes: Tensor, txt_codes: Tensor) -> Tensor:
    return 0.5 * (img_codes + txt_codes)


def naive_condition_weights(support_pair_codes: Tensor, contrast_pair_codes: Tensor | None = None,
                            top_k: int | None = None) -> Tensor:
    gap = support_pair_codes.mean(dim=-2)
    if contrast_pair_codes is not None:
        gap = gap - contrast_pair_codes.mean(dim=-2)
    weights = F.relu(gap)
    if top_k is not None:
        if not 1 <= top_k <= weights.shape[-1]:
            raise ValueError("top_k must be in [1, num_factors]")
        kept = torch.zeros_like(weights).scatter(-1, weights.topk(top_k, dim=-1).indices, 1.0)
        weights = weights * kept
    total = weights.sum(dim=-1, keepdim=True)
    return torch.where(total > 0, weights / total.clamp_min(1e-12), torch.zeros_like(weights))


def conditional_score(query_feat: Tensor, cand_feat: Tensor, query_codes: Tensor, cand_codes: Tensor,
                      weights: Tensor, beta: float) -> Tensor:
    cosine = F.cosine_similarity(query_feat[:, None, :], cand_feat, dim=-1)
    factor = (weights[:, None, :] * query_codes[:, None, :] * cand_codes).sum(dim=-1)
    return beta * cosine + factor
