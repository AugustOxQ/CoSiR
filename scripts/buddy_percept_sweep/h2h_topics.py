"""Buddy topic formation for the matched-topic-count head-to-head (spec R3):
a graph on the trained Stage-1 train embedding, Leiden (RB-configuration,
tunable resolution) on it, and a bisection over log(resolution) that hits a
target topic count after small-community merging.

At resolution 1.0 `leiden_on_graph` is the pilots' modularity Leiden
(`src.conditional_buddy.prototype_seed.detect_communities`) on the same
graph and seed, so `pilot_repaired` + resolution 1.0 reproduces the pilots'
train topics.
"""
import math

import igraph as ig
import leidenalg
import numpy as np
from scipy.sparse import csr_matrix

from scripts.buddy_percept_sweep.clustering import merge_small_communities
from src.conditional_buddy.buddy_graph import mutual_knn

GRAPH_KINDS = ("mknn", "pilot_repaired")


def build_topic_graph(embedding: np.ndarray, kind: str, pilot, device: str, k_neighbors: int = 20) -> csr_matrix:
    """`mknn`: the §6i harness graph (mutual kNN, as `clustering.leiden_partition`);
    `pilot_repaired`: the pilots' repaired single-modality graph (degree and
    connectivity repair), built with the train pipeline module."""
    if kind == "mknn":
        return mutual_knn(embedding, K=k_neighbors, backend="auto", device=device)
    if kind == "pilot_repaired":
        return pilot.single_modality.build_single_modality_graph(
            "train-topics", embedding, pilot.pipeline, pilot.affect_pilot, device,
            expected_nodes=len(embedding),
        )
    raise ValueError(f"graph kind must be one of {GRAPH_KINDS}, got {kind!r}")


def leiden_on_graph(graph: csr_matrix, resolution: float, seed: int) -> np.ndarray:
    """RB-configuration Leiden on the undirected, unweighted graph (one edge
    per upper-triangle pair, as `clustering.leiden_partition`); labels 0..n-1."""
    sources, targets = graph.nonzero()
    upper = sources < targets
    igraph = ig.Graph(n=graph.shape[0], edges=list(zip(sources[upper].tolist(), targets[upper].tolist())))
    partition = leidenalg.find_partition(
        igraph, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution, seed=seed,
    )
    membership = np.asarray(partition.membership, dtype=np.int64)
    _, labels = np.unique(membership, return_inverse=True)   # gap-free guard, order-preserving
    return labels.astype(np.int64)


def target_k_partition(embedding: np.ndarray, graph: csr_matrix, k_target: int, tolerance: int,
                       merge_threshold: float, seed: int, max_steps: int = 12, lo: float = 0.02,
                       hi: float = 20.0) -> tuple[np.ndarray, dict]:
    """Bisection on log(resolution), starting at 1.0: each step runs Leiden,
    merges communities smaller than `merge_threshold` of the nodes, and counts
    K. Hit when |K - k_target| <= tolerance. Never raises on a miss: returns
    the closest-K result seen (earliest on ties) with `hit=False`.

    info: resolution, k_raw, k_after_merge, steps, hit (of the returned
    labels, `steps` = Leiden runs made) and trace [(resolution, k_raw, k_after_merge)]."""
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    log_lo, log_hi = math.log(lo), math.log(hi)
    resolution = 1.0
    best = None
    trace = []
    for step in range(1, max_steps + 1):
        raw = leiden_on_graph(graph, resolution, seed)
        merged = merge_small_communities(embedding, raw, merge_threshold)[0]
        k_raw, k_merged = int(len(np.unique(raw))), int(len(np.unique(merged)))
        trace.append((resolution, k_raw, k_merged))
        distance = abs(k_merged - k_target)
        if best is None or distance < best[0]:
            best = (distance, merged, {"resolution": resolution, "k_raw": k_raw, "k_after_merge": k_merged})
        if distance <= tolerance:
            break
        if k_merged > k_target + tolerance:
            log_hi = math.log(resolution)
        else:
            log_lo = math.log(resolution)
        resolution = math.exp((log_lo + log_hi) / 2)
    distance, labels, info = best
    return labels, {**info, "steps": len(trace), "hit": bool(distance <= tolerance), "trace": trace}
