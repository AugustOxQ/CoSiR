"""Held-out label transfer and Stage 2 target construction. Reimplements
the validated logic from `run_heldout_label_transfer_pilot.py` and
`run_candidate4_rich_multilabel_pilot.py` as a standalone module (same
reasoning as Task 4: must run inside the long-lived sweep-agent process).
"""
from typing import Union

import numpy as np
from sklearn.neighbors import NearestNeighbors


def one_hot(labels: np.ndarray, n_topics: int) -> np.ndarray:
    targets = np.zeros((len(labels), n_topics), dtype=np.float32)
    targets[np.arange(len(labels)), labels] = 1.0
    return targets


def _fit_knn(train_embeddings: np.ndarray, k: int) -> NearestNeighbors:
    knn = NearestNeighbors(n_neighbors=k, metric="cosine")
    knn.fit(train_embeddings)
    return knn


def _batched_kneighbors(knn: NearestNeighbors, query_embeddings: np.ndarray):
    distances = []
    indices = []
    for start in range(0, len(query_embeddings), 128):
        batch_distances, batch_indices = knn.kneighbors(query_embeddings[start:start + 128])
        distances.append(batch_distances)
        indices.append(batch_indices)
    return np.concatenate(distances), np.concatenate(indices)


def assign_to_train_communities(train_embeddings: np.ndarray, train_labels: np.ndarray,
                                 query_embeddings: np.ndarray, k: int) -> np.ndarray:
    knn = _fit_knn(train_embeddings, k)
    _, indices = _batched_kneighbors(knn, query_embeddings)
    neighbor_labels = train_labels[indices]
    n_topics = int(train_labels.max()) + 1
    votes = np.zeros((len(query_embeddings), n_topics), dtype=np.int64)
    for topic in range(n_topics):
        votes[:, topic] = (neighbor_labels == topic).sum(axis=1)
    return votes.argmax(axis=1)


def cosine_vote_fractions(train_embeddings: np.ndarray, train_labels: np.ndarray,
                           query_embeddings: np.ndarray, n_topics: int, k: int) -> np.ndarray:
    knn = _fit_knn(train_embeddings, k)
    _, indices = _batched_kneighbors(knn, query_embeddings)
    neighbor_labels = train_labels[indices]
    fractions = np.zeros((len(query_embeddings), n_topics), dtype=np.float32)
    for topic in range(n_topics):
        fractions[:, topic] = (neighbor_labels == topic).sum(axis=1) / k
    row_sums = fractions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # guard against a degenerate all-zero row
    return fractions / row_sums


def build_targets(fractions_or_hard: np.ndarray, n_topics: int,
                   target_cutoff: Union[str, float]) -> np.ndarray:
    if target_cutoff == "single_label":
        return one_hot(fractions_or_hard.astype(np.int64), n_topics)
    fractions = fractions_or_hard
    row_max = fractions.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    targets = (fractions > target_cutoff * row_max).astype(np.float32)
    # Guarantee at least one positive label per row (the row's own argmax),
    # matching candidate 4's convention -- avoids an all-zero target row.
    empty_rows = targets.sum(axis=1) == 0
    if empty_rows.any():
        argmax_topic = fractions[empty_rows].argmax(axis=1)
        targets[np.flatnonzero(empty_rows), argmax_topic] = 1.0
    return targets
