"""Utilities for label embedding manipulation."""

import torch
import torch.nn.functional as F


def replace_with_most_different(
    label_embeddings: torch.Tensor, k: int = 10
) -> torch.Tensor:
    """Replace each label embedding with one of the k most different embeddings.

    Args:
        label_embeddings: Tensor of shape (batch_size, embedding_dim)
        k: Number of most different embeddings to consider

    Returns:
        Tensor of same shape with replaced embeddings
    """
    batch_size = label_embeddings.size(0)

    # Handle case where we have fewer than k+1 embeddings
    if batch_size <= k:
        k = max(1, batch_size - 1)

    # Normalize embeddings
    normalized = F.normalize(label_embeddings, p=2, dim=-1)

    # Compute cosine similarity and distance
    cosine_sim = torch.matmul(normalized, normalized.transpose(-2, -1))
    cosine_dist = 1 - cosine_sim

    # Fill diagonal with large negative to avoid self-match
    # Handle both 2D and higher-dimensional tensors
    if cosine_dist.dim() == 2:
        cosine_dist.fill_diagonal_(-float("inf"))
    else:
        # For higher dimensional tensors, set diagonal elements manually
        diag_indices = torch.arange(min(cosine_dist.shape[-2:]), device=cosine_dist.device)
        cosine_dist[..., diag_indices, diag_indices] = -float("inf")

    # Get indices of top-k most dissimilar embeddings
    topk_indices = torch.topk(cosine_dist, k=k, dim=1).indices  # [B, k]

    # Sample one index from top-k for each row
    rand_indices = torch.randint(0, k, (batch_size,), device=label_embeddings.device)
    selected_indices = topk_indices[torch.arange(batch_size), rand_indices]

    # Gather new embeddings
    new_embeddings = label_embeddings[selected_indices]

    return new_embeddings


def sample_label_embeddings(label_embeddings: torch.Tensor) -> torch.Tensor:
    """Sample new label embeddings ensuring no embedding is at its original index.

    Args:
        label_embeddings: Tensor of shape (batch_size, embedding_dim)

    Returns:
        Tensor of sampled embeddings with same shape
    """
    batch_size = label_embeddings.size(0)

    # Find unique label embeddings
    unique_label_embeddings = torch.unique(label_embeddings, dim=0)
    num_unique = unique_label_embeddings.size(0)

    if num_unique == 1:
        return label_embeddings

    # Initialize new embeddings tensor
    sampled_label_embeddings = torch.empty_like(label_embeddings)

    # For each embedding, sample a different one from unique embeddings
    for i in range(batch_size):
        original_embedding = label_embeddings[i]

        # Find embeddings different from original
        different_mask = ~torch.all(
            unique_label_embeddings == original_embedding, dim=1
        )
        available_choices = unique_label_embeddings[different_mask]

        if available_choices.size(0) > 0:
            # Sample from available choices
            idx = torch.randint(0, available_choices.size(0), (1,))
            sampled_label_embeddings[i] = available_choices[idx]
        else:
            # Fallback: use original if no different embeddings available
            sampled_label_embeddings[i] = original_embedding

    return sampled_label_embeddings


def random_sample_with_replacement(label_embedding: torch.Tensor) -> torch.Tensor:
    """Randomly sample label embeddings with replacement, avoiding self-matches.

    Args:
        label_embedding: Tensor of shape (n_labels, embedding_dim)

    Returns:
        Tensor of sampled embeddings with same shape
    """
    size = label_embedding.size(0)
    random_indices = torch.randint(0, size, (size,), device=label_embedding.device)

    # Ensure sampled index is different from original
    for i in range(size):
        while random_indices[i] == i:
            random_indices[i] = torch.randint(
                0, size, (1,), device=label_embedding.device
            )

    return label_embedding[random_indices]
