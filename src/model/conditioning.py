"""Nonparametric condition weights and the conditional score s(I, T | c).

Task 9 of the condition-interface plan showed the L1-normalized naive rule is
the scale-fair baseline and that the learned recovery head adds nothing; this
module is the reusable form of both for stage (d).

Shapes: F factors, D CLIP feature dims, Q queries (episodes), K candidates per
query, S supports, C contrasts; leading ``...`` dims are batch dims. Codes are
the non-negative factor activations of ``SharedFactorEncoder``.

beta couples to the raw code scale. The factor term of ``conditional_score``
is a weighted sum of products of two raw codes (weights sum to 1), so its size
grows with the square of the code scale; the cosine term lies in [-1, 1]. A
beta selected for one factor model is therefore NOT comparable across factor
models (e.g. R0 vs R3, or two seeds), and must be re-selected per model.
"""

import torch
from torch import Tensor
from torch.nn import functional as F


def pair_codes(img_codes: Tensor, txt_codes: Tensor) -> Tensor:
    """(..., F), (..., F) -> (..., F): one item's pair code, the mean of its image and text codes."""
    return 0.5 * (img_codes + txt_codes)


def naive_condition_weights(support_pair_codes: Tensor, contrast_pair_codes: Tensor | None = None,
                            top_k: int | None = None) -> Tensor:
    """Naive rule: (..., S, F) supports [, (..., C, F) contrasts] -> (..., F) condition weights.

    ReLU(mean support pair code - mean contrast pair code) per factor; with ``top_k`` only the
    ``top_k`` largest gaps are kept; then L1-normalized so each row sums to 1. A row with no
    positive gap stays all zero ("no factor favoured"; the score then reduces to beta * cosine).
    The weights are scale-free, but the codes they multiply are not (see the module docstring).
    """
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
    """s(q, c | w) = beta * cos(query_feat, cand_feat) + sum_l w_l * query_code_l * cand_code_l.

    Shapes: query_feat (Q, D), cand_feat (Q, K, D), query_codes (Q, F), cand_codes (Q, K, F),
    weights (Q, F) -> scores (Q, K). The query and the candidates come from opposite modalities
    (i2t: image features/codes of the query, text features/codes of the candidates; t2i the
    reverse), so the two directions are different scores, not one score transposed. At beta=0 the
    ranking is invariant to rescaling the weights; for beta > 0 the balance between the two terms
    depends on the raw code scale, so beta does not transfer across factor models.
    """
    cosine = F.cosine_similarity(query_feat[:, None, :], cand_feat, dim=-1)
    factor = (weights[:, None, :] * query_codes[:, None, :] * cand_codes).sum(dim=-1)
    return beta * cosine + factor
