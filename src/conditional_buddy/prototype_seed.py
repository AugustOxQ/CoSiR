"""CPU Leiden community detection over the buddy union graph, for PrototypeBank
seeding (Experiment 18). Deliberately avoids cuml/cugraph — this project's
libllvmlite.so import chain is documented broken locally (CLAUDE.md) — using
leidenalg + python-igraph instead, both pure CPU, no RAPIDS dependency.

See docs/superpowers/specs/2026-09-15-buddy-prototype-conditioning-design.md §3-4.
"""
from typing import Optional

import igraph as ig
import leidenalg
import numpy as np
from scipy.sparse import csr_matrix


def detect_communities(E: csr_matrix, seed: int = 42) -> np.ndarray:
    """Leiden community detection (modularity objective) over a binary union graph.

    E: (N, N) symmetric binary sparse adjacency (buddy_graph.union_graph's output).
    Returns: (N,) int64 array of community labels, 0-indexed, no gaps.
    """
    E_coo = E.tocoo()
    mask = E_coo.row < E_coo.col  # undirected: one edge per pair
    edges = list(zip(E_coo.row[mask].tolist(), E_coo.col[mask].tolist()))
    g = ig.Graph(n=E.shape[0], edges=edges)
    partition = leidenalg.find_partition(
        g, leidenalg.ModularityVertexPartition, seed=seed,
    )
    return np.array(partition.membership, dtype=np.int64)


def community_mean_features(
    labels: np.ndarray, img_feats: np.ndarray, txt_feats: np.ndarray,
) -> np.ndarray:
    """Per-community mean of the (img, txt) feature average.

    labels: (N,) from detect_communities, 0-indexed, no gaps.
    img_feats/txt_feats: (N, D), L2-normalized, same D as PrototypeBank's
    condition_dim (see Task 4 for the projection that ensures this).
    Returns: (C, D) float32, C = labels.max() + 1.
    """
    n_communities = int(labels.max()) + 1
    mean_feat = 0.5 * (img_feats + txt_feats)
    sums = np.zeros((n_communities, mean_feat.shape[1]), dtype=np.float32)
    counts = np.zeros(n_communities, dtype=np.int64)
    np.add.at(sums, labels, mean_feat)
    np.add.at(counts, labels, 1)
    if (counts == 0).any():
        raise ValueError("every community label 0..C-1 must have at least one member")
    return sums / counts[:, None]


def coarsen_to_prototype_count(
    community_means: np.ndarray, num_prototypes: int, seed: int = 42,
) -> np.ndarray:
    """KMeans-coarsen community means down to num_prototypes rows (spec §6).

    No-op (returns community_means unchanged) if it already has
    <= num_prototypes rows.
    """
    if community_means.shape[0] <= num_prototypes:
        return community_means
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=num_prototypes, random_state=seed, n_init=10)
    km.fit(community_means)
    return km.cluster_centers_.astype(np.float32)
