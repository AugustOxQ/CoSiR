import torch
import numpy as np
from scipy.stats import spearmanr

from src.model.cosirmodel import retrieve_with_condition
from src.metrics.loss import compute_retrieval_divergence


def compute_angular_monotonicity(conditions_2d, model, test_set):
    """
    测量角度差异和语义差异的相关性
    理想值: 0.7-0.9 (高度相关但不完全线性)
    """
    n_samples = 100

    # 随机采样condition pairs
    idx1 = torch.randperm(len(conditions_2d))[:n_samples]
    idx2 = torch.randperm(len(conditions_2d))[:n_samples]

    # 计算角度差
    angles1 = torch.atan2(conditions_2d[idx1, 1], conditions_2d[idx1, 0])
    angles2 = torch.atan2(conditions_2d[idx2, 1], conditions_2d[idx2, 0])
    angle_diffs = torch.abs(angles1 - angles2)
    angle_diffs = torch.min(angle_diffs, 2 * np.pi - angle_diffs)

    # 计算语义差异
    semantic_diffs = []
    for i in range(n_samples):
        c1, c2 = conditions_2d[idx1[i]], conditions_2d[idx2[i]]
        retrieval1 = retrieve_with_condition(model, test_set, c1)
        retrieval2 = retrieve_with_condition(model, test_set, c2)

        # 用检索结果的差异衡量语义差异
        diff = compute_retrieval_divergence(retrieval1, retrieval2)
        semantic_diffs.append(diff)

    semantic_diffs = torch.tensor(semantic_diffs)

    # 计算Spearman相关系数
    correlation = spearmanr(angle_diffs.numpy(), semantic_diffs.numpy())[0]

    return correlation
    # 期望: correlation > 0.7


def compute_radius_strength_correlation(conditions_2d, model, test_set):
    """
    测量半径和调制强度的关系
    理想值: 0.8-0.95 (强相关)
    """
    radii = torch.norm(conditions_2d, dim=1)

    strengths = []
    for i, c in enumerate(conditions_2d):
        # 计算该condition造成的变化幅度
        text_emb = get_text_embeddings(test_set)
        text_emb_cond = model.modulate(
            text_emb, c.unsqueeze(0).repeat(len(text_emb), 1)
        )

        # 平均变化幅度
        strength = torch.norm(text_emb_cond - text_emb, dim=1).mean()
        strengths.append(strength.item())

    strengths = torch.tensor(strengths)

    correlation = pearsonr(radii.numpy(), strengths.numpy())[0]

    return correlation
    # 期望: correlation > 0.8


def compute_local_smoothness(conditions_2d, model, test_set, k=5):
    """
    测量condition space的局部平滑性
    理想值: > 0.85 (高平滑度)
    """
    smoothness_scores = []

    # 对每个condition
    for i, c in enumerate(conditions_2d):
        # 找k近邻
        dists = torch.norm(conditions_2d - c, dim=1)
        _, neighbors = torch.topk(dists, k + 1, largest=False)
        neighbors = neighbors[1:]  # 排除自己

        # 当前condition的效果
        effect_i = get_condition_effect(model, test_set, c)

        # 近邻的平均效果
        neighbor_effects = []
        for j in neighbors:
            effect_j = get_condition_effect(model, test_set, conditions_2d[j])
            neighbor_effects.append(effect_j)

        neighbor_avg = torch.stack(neighbor_effects).mean(0)

        # 相似度
        smoothness = F.cosine_similarity(effect_i, neighbor_avg, dim=0)
        smoothness_scores.append(smoothness.item())

    return np.mean(smoothness_scores)
    # 期望: smoothness > 0.85


def compute_cluster_quality(conditions_2d, n_clusters=8):
    """
    测量自然形成的聚类质量
    理想值: Silhouette > 0.5, Davies-Bouldin < 1.0
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(conditions_2d.cpu().numpy())

    # Silhouette score: [-1, 1], 越大越好
    silhouette = silhouette_score(conditions_2d.cpu().numpy(), labels)

    # Davies-Bouldin index: [0, ∞), 越小越好
    davies_bouldin = davies_bouldin_score(conditions_2d.cpu().numpy(), labels)

    return {"silhouette": silhouette, "davies_bouldin": davies_bouldin}
    # 期望: silhouette > 0.5, davies_bouldin < 1.0


def compute_angular_semantic_consistency(conditions_2d, model, test_set, n_bins=8):
    """
    测量每个角度区域的语义一致性
    理想值: 每个bin内的检索结果应该有共同的语义特征
    """
    angles = torch.atan2(conditions_2d[:, 1], conditions_2d[:, 0])

    # 分成n_bins个角度区间
    angle_bins = torch.linspace(-np.pi, np.pi, n_bins + 1)

    bin_consistencies = []

    for i in range(n_bins):
        # 找到这个角度区间的所有conditions
        bin_mask = (angles >= angle_bins[i]) & (angles < angle_bins[i + 1])
        bin_conditions = conditions_2d[bin_mask]

        if len(bin_conditions) < 2:
            continue

        # 对bin内的所有conditions进行检索
        retrievals = []
        for c in bin_conditions[:20]:  # 采样最多20个
            retrieval = retrieve_with_condition(model, test_set, c)
            retrievals.append(retrieval)

        # 计算bin内检索结果的一致性
        # 方法1: 用文本相似度
        consistency = compute_intra_bin_consistency(retrievals)
        bin_consistencies.append(consistency)

    return {
        "mean_consistency": np.mean(bin_consistencies),
        "std_consistency": np.std(bin_consistencies),
    }
    # 期望: mean_consistency > 0.6
