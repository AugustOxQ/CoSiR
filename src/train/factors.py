"""Pure, differentiable losses for shared image/text factor discovery.

Graph consistency uses one unordered pair per sampled node pair. A smooth
margin-like contrast compares the mean cosine similarity of graph neighbors
with the mean similarity of in-batch non-neighbors. This avoids treating other
true neighbors as false negatives, which a direct symmetric InfoNCE over graph
edges would do. A batch with no positive or negative pair contributes zero;
callers should sample graph edges to obtain a useful graph signal.
"""

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch import Tensor
from torch.nn import functional as F

from src.model.factors import SharedFactorEncoder


def reconstruction_loss(
    encoder: SharedFactorEncoder, img_feat: Tensor, txt_feat: Tensor
) -> Tensor:
    """Average the two modalities' elementwise round-trip MSE losses."""
    img_reconstructed = encoder.reconstruct_image(encoder.encode_image(img_feat))
    txt_reconstructed = encoder.reconstruct_text(encoder.encode_text(txt_feat))
    return 0.5 * (
        F.mse_loss(img_reconstructed, img_feat)
        + F.mse_loss(txt_reconstructed, txt_feat)
    )


def paired_agreement_loss(img_codes: Tensor, txt_codes: Tensor) -> Tensor:
    """Mean 1 - cosine similarity for same-row pairs.

    Cosine compares factor activation patterns without penalizing a uniform
    difference in code magnitude, which the anti-split loss handles separately.
    """
    return (1 - F.cosine_similarity(img_codes, txt_codes, dim=1)).mean()


def graph_neighbor_consistency_loss(
    codes: Tensor, graph: csr_matrix, sample_idx: np.ndarray
) -> Tensor:
    """Softplus((mean non-neighbor cosine - mean neighbor cosine) / 0.1).

    Only the sampled induced subgraph is inspected. The softplus ranking loss
    remains smooth when positives already outrank negatives. Empty positive or
    negative sets yield a differentiable zero, so a sparse batch does not fail.
    """
    sample_idx = np.asarray(sample_idx, dtype=np.int64)
    sampled_codes = codes[torch.as_tensor(sample_idx, device=codes.device)]
    adjacency = graph[sample_idx][:, sample_idx].toarray()
    positive = torch.as_tensor(
        np.triu(adjacency > 0, k=1), dtype=torch.bool, device=codes.device
    )
    upper = torch.triu(
        torch.ones(positive.shape, dtype=torch.bool, device=codes.device), diagonal=1
    )
    negative = upper & ~positive
    if not positive.any() or not negative.any():
        return sampled_codes.sum() * 0.0

    normalized = F.normalize(sampled_codes, dim=1)
    similarity = normalized @ normalized.T
    return F.softplus((similarity[negative].mean() - similarity[positive].mean()) / 0.1)


def sparsity_penalty(codes: Tensor) -> Tensor:
    """Mean per-sample L1 norm, so every active factor has a cost."""
    return codes.abs().sum(dim=1).mean()


def anti_split_penalty(img_codes: Tensor, txt_codes: Tensor) -> Tensor:
    """Mean relative modality imbalance over factors active in either modality.

    For factor l, let I_l and T_l be its batch-mean nonnegative activations.
    Its imbalance is |I_l - T_l| / (I_l + T_l + 1e-8), near zero when both
    modalities use it similarly and near one when only one does. Factors with
    combined mean <= 1e-8 are omitted so dead factors cannot dilute a split.
    An all-dead dictionary receives zero; reconstruction and sparsity must
    separately prevent that collapse.
    """
    mean_img = img_codes.mean(dim=0)
    mean_txt = txt_codes.mean(dim=0)
    combined = mean_img + mean_txt
    active = combined > 1e-8
    if not active.any():
        return (img_codes.sum() + txt_codes.sum()) * 0.0
    imbalance = (mean_img - mean_txt).abs() / (combined + 1e-8)
    return imbalance[active].mean()
