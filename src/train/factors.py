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


def cross_modal_infonce_loss(
    img_codes: Tensor,
    txt_codes: Tensor,
    temperature: float = 0.1,
    group_ids: Tensor | None = None,
) -> Tensor:
    """Symmetric image<->text InfoNCE on L2-normalized codes.

    Unlike paired_agreement_loss, a single code direction shared by every item
    scores log(B), not 0: matched pairs must beat in-batch negatives. Off-diagonal
    pairs with equal group_ids (same painting / image) are excluded as negatives.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = F.normalize(img_codes, dim=1) @ F.normalize(txt_codes, dim=1).T / temperature
    if group_ids is not None:
        group_ids = torch.as_tensor(group_ids, device=logits.device)
        if group_ids.shape != (len(img_codes),):
            raise ValueError("group_ids must have one entry per row")
        same = group_ids[:, None] == group_ids[None, :]
        same.fill_diagonal_(False)
        logits = logits.masked_fill(same, float("-inf"))
    targets = torch.arange(len(logits), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def painting_infonce_loss(
    img_codes: Tensor,
    txt_codes: Tensor,
    painting_ids: Tensor | np.ndarray,
    temperature: float = 0.1,
) -> Tensor:
    """Symmetric InfoNCE between each painting's mean image code and its mean caption code.

    Rows with the same painting id are averaged per modality. A painting's rows share one image, so its
    image mean is that image's code; its caption mean is the average code of its captions. The means are
    matched against the batch's other paintings with ``cross_modal_infonce_loss``, so one caption's code may
    differ from the image code as long as the painting's mean caption code matches it (factor-learning spec §5).
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    ids = painting_ids if isinstance(painting_ids, Tensor) else torch.as_tensor(np.asarray(painting_ids))
    ids = ids.to(img_codes.device)
    if ids.shape != (len(img_codes),) or txt_codes.shape != img_codes.shape:
        raise ValueError("painting_ids must have one entry per row, and both code arrays the same shape")
    _, inverse = torch.unique(ids, return_inverse=True)
    n = int(inverse.max()) + 1
    counts = torch.zeros(n, device=img_codes.device, dtype=img_codes.dtype).index_add_(
        0, inverse, torch.ones(len(inverse), device=img_codes.device, dtype=img_codes.dtype))[:, None]
    img_mean = torch.zeros(n, img_codes.shape[1], device=img_codes.device, dtype=img_codes.dtype).index_add_(
        0, inverse, img_codes) / counts
    txt_mean = torch.zeros(n, txt_codes.shape[1], device=txt_codes.device, dtype=txt_codes.dtype).index_add_(
        0, inverse, txt_codes) / counts
    return cross_modal_infonce_loss(img_mean, txt_mean, temperature)


def decorrelation_penalty(codes: Tensor, eps: float = 1e-6) -> Tensor:
    """Mean squared off-diagonal batch correlation over factors with non-zero std.

    Constant factors are excluded (no NaN); the dead/constant gates in
    src/eval/factor_gates.py catch a model that escapes by going constant.
    """
    if len(codes) < 2:
        return codes.sum() * 0.0
    std = codes.std(dim=0)
    varying = std > eps
    k = int(varying.sum())
    if k < 2:
        return codes.sum() * 0.0
    z = (codes[:, varying] - codes[:, varying].mean(dim=0)) / std[varying]
    corr = z.T @ z / (len(codes) - 1)
    off = corr - torch.diag(torch.diagonal(corr))
    return (off**2).sum() / (k * (k - 1))


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


def usage_balance_penalty(img_codes: Tensor, txt_codes: Tensor) -> Tensor:
    """Negative entropy of combined mean factor usage; lower means more even usage."""
    eps = 1e-8
    mean_activation = 0.5 * (img_codes.mean(dim=0) + txt_codes.mean(dim=0))
    p = mean_activation / (mean_activation.sum() + eps)
    return (p * (p + eps).log()).sum()
