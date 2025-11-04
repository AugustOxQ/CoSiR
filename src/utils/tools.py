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


def get_representatives_polar_grid(
    learned_conditions=None, num_angles=10, num_radii=3, max_radius=None
):
    """
    在极坐标下均匀采样
    Args:
        num_angles: 角度方向数 (m)
        num_radii: 半径层数 (n)
        max_radius: 最大半径，如果None则从learned_conditions推断
        learned_conditions: [N, 2] 训练学到的conditions
    Returns:
        sampled_conditions: [k, 2] where k = num_angles * num_radii
    """
    if max_radius is None and learned_conditions is not None:
        # 从learned conditions推断合理的radius范围
        radii = torch.norm(learned_conditions, dim=1)
        max_radius = radii.quantile(0.95)  # 用95分位数，避免outlier
        print(f"Inferred max_radius: {max_radius:.3f}")
    elif max_radius is None:
        max_radius = 2.0  # 默认值

    # 生成角度: [0, 2π) 均匀分布
    angles = torch.linspace(0, 2 * torch.pi, num_angles + 1)[:-1]  # 不包括2π

    # 生成半径: [0, max_radius] 均匀或对数分布
    # 选项1: 线性分布（包括0）
    radii = torch.linspace(0, max_radius, num_radii)

    # 选项2: 不包括0（如果你发现0附近的condition不重要）
    # radii = torch.linspace(0.1 * max_radius, max_radius, num_radii)

    # 生成网格
    sampled_conditions = []
    for r in radii:
        for theta in angles:
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            sampled_conditions.append([x.item(), y.item()])

    sampled_conditions = torch.tensor(sampled_conditions)
    print(
        f"Sampled {len(sampled_conditions)} conditions "
        f"({num_angles} angles × {num_radii} radii)"
    )

    return sampled_conditions
