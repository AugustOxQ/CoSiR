"""Regularization functions used by LabelContrastiveLoss_enhance."""

import torch
import torch.nn.functional as F


def boundary_penalty(embeddings, radius=1.0, alpha=0.1):
    """Penalise embeddings whose L2 norm exceeds `radius`."""
    norms = torch.norm(embeddings, p=2, dim=1)
    penalty = torch.where(
        norms > radius, (norms - radius) ** 2, torch.zeros_like(norms)
    )
    return alpha * torch.mean(penalty)


def manifold_smoothness_loss_sparse(
    conditions, text_emb, conditional_text_pos, k=3, model=None, alpha=0.1
):
    """Penalise inconsistent modulation between k-nearest neighbours in condition space."""
    batch_size = len(conditions)

    dist_matrix = torch.cdist(conditions, conditions)

    text_emb_normalized = F.normalize(text_emb, p=2, dim=1)

    mask = torch.eye(batch_size, device=dist_matrix.device).bool()
    dist_matrix = dist_matrix.masked_fill(mask, float("inf"))

    _, neighbor_indices = torch.topk(dist_matrix, k, largest=False, dim=1)

    device = conditions.device
    random_neighbor_idx = torch.randint(0, k, (batch_size,), device=device)
    selected_neighbors = neighbor_indices[torch.arange(batch_size, device=device), random_neighbor_idx]

    neighbor_conditions = conditions[selected_neighbors]
    conditional_text_from_neighbor = model.combine(text_emb, None, neighbor_conditions)

    delta_current = conditional_text_pos - text_emb_normalized
    delta_neighbor = conditional_text_from_neighbor - text_emb_normalized

    smoothness = F.cosine_similarity(delta_current, delta_neighbor, dim=-1)

    distances = dist_matrix.gather(1, selected_neighbors.unsqueeze(1)).squeeze(1)
    distances = torch.clamp(distances, min=1e-8, max=10.0)
    weights = torch.exp(-distances + 1e-8)

    L_smooth_weighted = ((1 - smoothness) * weights).mean()

    return alpha * L_smooth_weighted
