"""Aspect-episode metrics (CVPR plan spec §5.1, §10): R@1, condition gain, other-aspect rate, swap, strict swap,
and a bootstrap that resamples whole clusters (paintings, species, reference images)."""

import numpy as np

CONDITIONS = ("a", "b")
DIRECTIONS = ("i2t", "t2i")
METRICS = ("r1", "gain", "other", "swap", "strict")


def first_place(scores: np.ndarray, column: int) -> np.ndarray:
    """1.0 where ``column`` scores strictly above every other candidate; ties and non-finite rows are misses."""
    s = np.asarray(scores, dtype=np.float64)
    target = s[:, column:column + 1]
    others = np.delete(s, column, axis=1)
    finite = np.isfinite(s).all(axis=1)
    return ((others < target).all(axis=1) & finite).astype(np.float64)


def per_anchor(scores: dict) -> dict:
    """Per-anchor metrics averaged over the two directions. gain = R@1 - other-aspect rate."""
    acc = {m: [] for m in METRICS}
    for d in DIRECTIONS:
        s_a, s_b = np.asarray(scores["a"][d], float), np.asarray(scores["b"][d], float)
        hit_aa, hit_bb = first_place(s_a, 0), first_place(s_b, 1)
        hit_ab, hit_ba = first_place(s_b, 0), first_place(s_a, 1)     # the other aspect's candidate wins
        r1 = 0.5 * (hit_aa + hit_bb)
        other = 0.5 * (hit_ba + hit_ab)
        finite = np.isfinite(s_a).all(1) & np.isfinite(s_b).all(1)
        swap = ((s_a[:, 0] > s_a[:, 1]) & (s_b[:, 1] > s_b[:, 0]) & finite).astype(np.float64)
        for name, value in (("r1", r1), ("gain", r1 - other), ("other", other), ("swap", swap),
                            ("strict", hit_aa * hit_bb)):
            acc[name].append(value)
    return {m: 0.5 * (v[0] + v[1]) for m, v in acc.items()}


def cluster_bootstrap(values, clusters, n_boot: int = 5000, seed: int = 42, chunk: int = 250) -> dict:
    values = np.asarray(values, dtype=np.float64)
    _, idx = np.unique(np.asarray(clusters), return_inverse=True)
    k = int(idx.max()) + 1
    if k < 2:
        raise ValueError("cluster_bootstrap needs at least two clusters")
    sums = np.bincount(idx, weights=values, minlength=k)
    counts = np.bincount(idx, minlength=k).astype(np.float64)
    rng = np.random.default_rng(seed)
    boots = []
    for start in range(0, n_boot, chunk):
        draws = rng.integers(0, k, size=(min(chunk, n_boot - start), k))
        boots.append(sums[draws].sum(axis=1) / counts[draws].sum(axis=1))
    boots = np.concatenate(boots)
    return {"point": float(values.mean()), "ci95": [float(np.percentile(boots, 2.5)),
                                                    float(np.percentile(boots, 97.5))], "n_clusters": k}


def _points(result: dict) -> dict:
    return {**result, "point": 100 * result["point"], "ci95": [100 * c for c in result["ci95"]]}


def summarize(per_anchor_values: dict, clusters) -> dict:
    return {m: _points(cluster_bootstrap(per_anchor_values[m], clusters)) for m in METRICS}


def compare(per_a: dict, per_b: dict, clusters, metric: str) -> dict:
    return _points(cluster_bootstrap(np.asarray(per_a[metric]) - np.asarray(per_b[metric]), clusters))
