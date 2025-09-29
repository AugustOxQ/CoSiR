import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np


def replace_with_most_different(label_embeddings, k=10):
    batch_size = label_embeddings.size(0)

    # Normalize the embeddings
    normalized = F.normalize(label_embeddings, p=2, dim=1)

    # Compute cosine similarity and distance
    cosine_sim = torch.matmul(normalized, normalized.T)
    cosine_dist = 1 - cosine_sim

    # Fill diagonal with large negative to avoid self-match
    cosine_dist.fill_diagonal_(-float("inf"))

    # Get indices of top-k most dissimilar embeddings
    topk_indices = torch.topk(cosine_dist, k=k, dim=1).indices  # [B, k]

    # Sample one index from top-k for each row
    rand_indices = torch.randint(0, k, (batch_size,), device=label_embeddings.device)
    selected_indices = topk_indices[torch.arange(batch_size), rand_indices]

    # Gather new embeddings
    new_embeddings = label_embeddings[selected_indices]

    return new_embeddings


def get_representatives(label_embeddings, k=10):
    # Check if label_embeddings are with dimention 2
    assert label_embeddings.shape[1] == 2, "Label embeddings must be with dimention 2"
    kmeans = KMeans(n_clusters=min(k, label_embeddings.shape[0])).fit(
        label_embeddings.cpu().numpy()
    )
    centroids = kmeans.cluster_centers_
    # Find closest real embedding to each centroid
    indices = np.argmin(cdist(centroids, label_embeddings), axis=1)
    representatives = label_embeddings[indices]
    return representatives
