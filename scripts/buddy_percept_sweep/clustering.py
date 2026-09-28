"""Leiden re-clustering (post-hoc, on the trained Stage-1 embedding) and
small-community merging. Mirrors `run_candidate3_k_sweep_pilot.py` and
`run_candidate1_min_occupancy_pilot.py`'s validated logic, reimplemented
here (fresh module) rather than importing those pilot scripts, since this
must run standalone inside a long-lived sweep-agent process without their
CLI/report-writing side effects.
"""
import igraph as ig
import leidenalg
import numpy as np
from scipy.sparse import csr_matrix

from src.conditional_buddy.buddy_graph import mutual_knn


def leiden_partition(
    embedding: np.ndarray,
    resolution: float,
    seed: int = 42,
    k_neighbors: int = 20,
    device: str = "cpu",
) -> np.ndarray:
    adjacency: csr_matrix = mutual_knn(embedding, K=k_neighbors, backend="auto", device=device)
    sources, targets = adjacency.nonzero()
    graph = ig.Graph(n=adjacency.shape[0], edges=list(zip(sources.tolist(), targets.tolist())))
    partition = leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution, seed=seed,
    )
    labels = np.array(partition.membership, dtype=np.int64)
    # Relabel to 0-indexed, no gaps (membership from leidenalg is already
    # gap-free by construction, but this guards against future changes).
    unique = sorted(set(labels.tolist()))
    remap = {old: new for new, old in enumerate(unique)}
    return np.array([remap[label] for label in labels], dtype=np.int64)


def merge_small_communities(embedding: np.ndarray, labels: np.ndarray, min_fraction: float) -> tuple[np.ndarray, dict]:
    n = len(labels)
    unique_labels = sorted(set(labels.tolist()))
    if min_fraction <= 0.0:
        return labels.copy(), {label: label for label in unique_labels}

    counts = {label: int((labels == label).sum()) for label in unique_labels}
    threshold = min_fraction * n
    small = [label for label, count in counts.items() if count < threshold]
    large = [label for label in unique_labels if label not in small]

    centroids = {}
    for label in unique_labels:
        members = embedding[labels == label]
        centroid = members.mean(axis=0)
        centroids[label] = centroid / (np.linalg.norm(centroid) + 1e-12)

    label_map = {label: label for label in large}
    if not large:
        # Everything is "small" (degenerate/uniform partition) -- nothing
        # to merge into; leave labels untouched.
        return labels.copy(), {label: label for label in unique_labels}
    for label in small:
        similarities = {target: float(centroids[label] @ centroids[target]) for target in large}
        best = max(similarities, key=similarities.get)
        label_map[label] = best

    remap_targets = sorted(set(label_map.values()))
    final_map = {old: new for new, old in enumerate(remap_targets)}
    resolved_map = {old: final_map[label_map[old]] for old in unique_labels}
    merged = np.array([resolved_map[label] for label in labels.tolist()], dtype=np.int64)
    return merged, resolved_map
