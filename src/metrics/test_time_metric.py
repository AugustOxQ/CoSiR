import torch
import numpy as np
from scipy.stats import spearmanr

from src.model.cosirmodel import retrieve_with_condition
from src.metrics.loss import compute_retrieval_divergence


def compute_angular_monotonicity(conditions_2d, model, test_set):
    """Measure correlation between angular difference and semantic difference.

    Expected: Spearman r in 0.7–0.9 (high but not perfectly linear).
    """
    n_samples = 100

    idx1 = torch.randperm(len(conditions_2d))[:n_samples]
    idx2 = torch.randperm(len(conditions_2d))[:n_samples]

    angles1 = torch.atan2(conditions_2d[idx1, 1], conditions_2d[idx1, 0])
    angles2 = torch.atan2(conditions_2d[idx2, 1], conditions_2d[idx2, 0])
    angle_diffs = torch.abs(angles1 - angles2)
    angle_diffs = torch.min(angle_diffs, 2 * np.pi - angle_diffs)  # shortest arc

    semantic_diffs = []
    for i in range(n_samples):
        c1, c2 = conditions_2d[idx1[i]], conditions_2d[idx2[i]]
        retrieval1 = retrieve_with_condition(model, test_set, c1)
        retrieval2 = retrieve_with_condition(model, test_set, c2)
        diff = compute_retrieval_divergence(retrieval1, retrieval2)
        semantic_diffs.append(diff)

    semantic_diffs = torch.tensor(semantic_diffs)
    correlation = spearmanr(angle_diffs.numpy(), semantic_diffs.numpy())[0]
    return correlation  # expect > 0.7


def compute_radius_strength_correlation(conditions_2d, model, test_set):
    """Measure correlation between condition radius and modulation strength.

    Expected: Pearson r in 0.8–0.95 (strong correlation).
    """
    radii = torch.norm(conditions_2d, dim=1)

    strengths = []
    for i, c in enumerate(conditions_2d):
        text_emb = get_text_embeddings(test_set)
        text_emb_cond = model.modulate(
            text_emb, c.unsqueeze(0).repeat(len(text_emb), 1)
        )
        strength = torch.norm(text_emb_cond - text_emb, dim=1).mean()
        strengths.append(strength.item())

    strengths = torch.tensor(strengths)
    correlation = pearsonr(radii.numpy(), strengths.numpy())[0]
    return correlation  # expect > 0.8


def compute_local_smoothness(conditions_2d, model, test_set, k=5):
    """Measure local smoothness of the condition space.

    Expected: mean cosine similarity > 0.85.
    """
    smoothness_scores = []

    for i, c in enumerate(conditions_2d):
        dists = torch.norm(conditions_2d - c, dim=1)
        _, neighbors = torch.topk(dists, k + 1, largest=False)
        neighbors = neighbors[1:]  # exclude self

        effect_i = get_condition_effect(model, test_set, c)

        neighbor_effects = [
            get_condition_effect(model, test_set, conditions_2d[j]) for j in neighbors
        ]
        neighbor_avg = torch.stack(neighbor_effects).mean(0)

        smoothness = F.cosine_similarity(effect_i, neighbor_avg, dim=0)
        smoothness_scores.append(smoothness.item())

    return np.mean(smoothness_scores)  # expect > 0.85


def compute_cluster_quality(conditions_2d, n_clusters=8):
    """Measure quality of naturally formed clusters in condition space.

    Expected: Silhouette > 0.5, Davies-Bouldin < 1.0.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(conditions_2d.cpu().numpy())

    # Silhouette: [-1, 1], higher is better
    silhouette = silhouette_score(conditions_2d.cpu().numpy(), labels)
    # Davies-Bouldin: [0, ∞), lower is better
    davies_bouldin = davies_bouldin_score(conditions_2d.cpu().numpy(), labels)

    return {"silhouette": silhouette, "davies_bouldin": davies_bouldin}


def compute_angular_semantic_consistency(conditions_2d, model, test_set, n_bins=8):
    """Measure semantic consistency within each angular region.

    Expected: mean_consistency > 0.6 (retrievals within the same angular bin should share themes).
    """
    angles = torch.atan2(conditions_2d[:, 1], conditions_2d[:, 0])
    angle_bins = torch.linspace(-np.pi, np.pi, n_bins + 1)

    bin_consistencies = []

    for i in range(n_bins):
        bin_mask = (angles >= angle_bins[i]) & (angles < angle_bins[i + 1])
        bin_conditions = conditions_2d[bin_mask]

        if len(bin_conditions) < 2:
            continue

        retrievals = [
            retrieve_with_condition(model, test_set, c)
            for c in bin_conditions[:20]  # cap at 20 samples per bin
        ]

        consistency = compute_intra_bin_consistency(retrievals)
        bin_consistencies.append(consistency)

    return {
        "mean_consistency": np.mean(bin_consistencies),
        "std_consistency": np.std(bin_consistencies),
    }
