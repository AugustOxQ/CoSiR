"""Pseudo-partitions over training rows and the pseudo-aspect episode bank (CVPR plan spec §6). No evaluation label
is used: partitions are k-means clusters of features, or of GoEmotions affect probabilities (distant supervision)."""

from itertools import combinations

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from src.eval.aspect_episodes import PaintingValueIndex, build_aspect_episodes, concat_episodes


def kmeans_partition(features, rows, k: int = 64, seed: int = 42) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    x = np.asarray(features, dtype=np.float32)[rows]
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    fit = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=4096).fit_predict(x)
    labels = np.full(len(features), -1, dtype=np.int64)
    labels[rows] = fit
    return labels


def build_episode_bank(partitions: dict, groups, rows, n_per_pair: int, seed: int,
                       min_paintings: int = 30) -> "AspectEpisodes":
    names = sorted(partitions)
    index = PaintingValueIndex(partitions, groups)
    blocks = []
    for i, (a, b) in enumerate(combinations(names, 2)):
        rest = [n for n in names if n not in (a, b)]
        third = rest[0] if len(rest) == 1 else None
        blocks.append(build_aspect_episodes(partitions, groups, rows, a, b, n_per_pair, seed + i, third=third,
                                            min_paintings=min_paintings, index=index))
    return concat_episodes(blocks)
