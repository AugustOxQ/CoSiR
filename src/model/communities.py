"""Leiden communities over the trained student's output embedding space."""

import igraph as ig
import leidenalg
import numpy as np
from sklearn.neighbors import NearestNeighbors


def detect_communities(
    embeddings: np.ndarray, k: int = 20, seed: int = 42
) -> np.ndarray:
    """Cluster a fresh, unweighted kNN graph of learned embeddings.

    Each sample chooses its k nearest *other* samples. An undirected edge is
    kept if either endpoint chose the other (simple kNN union). This keeps
    nodes connected to their chosen neighbors even when the choice is not
    mutual, unlike mutual kNN, which can isolate points in sparse regions.
    """
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (N, D)")
    if k < 1:
        raise ValueError("k must be positive")

    n_samples = len(embeddings)
    if n_samples == 0:
        return np.empty(0, dtype=np.int64)
    if n_samples == 1:
        return np.zeros(1, dtype=np.int64)

    neighbors = NearestNeighbors(n_neighbors=min(k, n_samples - 1))
    neighbors.fit(embeddings)
    indices = neighbors.kneighbors(return_distance=False)
    edges = [(i, int(j)) for i, row in enumerate(indices) for j in row]
    graph = ig.Graph(n=n_samples, edges=edges, directed=False)
    graph.simplify(multiple=True, loops=True)

    partition = leidenalg.find_partition(
        graph, leidenalg.ModularityVertexPartition, seed=seed
    )
    labels = np.asarray(partition.membership, dtype=np.int64)
    _, compact_labels = np.unique(labels, return_inverse=True)
    return compact_labels.astype(np.int64, copy=False)


def community_stats(labels: np.ndarray) -> dict:
    """Summarize community occupancy; empty_count flags gaps in labels."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("labels must have shape (N,)")
    if len(labels) == 0:
        return {
            "num_communities": 0,
            "sizes": [],
            "min_size": 0,
            "max_size": 0,
            "empty_count": 0,
        }

    sizes = np.bincount(labels)
    return {
        "num_communities": int(len(sizes)),
        "sizes": sizes.tolist(),
        "min_size": int(sizes.min()),
        "max_size": int(sizes.max()),
        "empty_count": int(np.count_nonzero(sizes == 0)),
    }
