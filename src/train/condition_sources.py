"""Self-generated condition sources for stage (d) (spec §3).

A source defines groups of rows that "share a condition". Every group is built only from the rows the
source was fitted on (scorer-train rows): no human label and no row outside ``rows`` is ever used.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans

MIN_SWAP_OVERLAP = 20


@dataclass(frozen=True)
class Condition:
    source: str
    key: tuple
    inside: np.ndarray            # sorted global row ids sharing the condition (subset of fit rows)
    outside: np.ndarray           # sorted global row ids clearly lacking it (subset of fit rows)


class FactorComboSource:
    """Condition = high on a random positive mix of 1-3 R3 factors (top decile vs bottom half)."""

    name = "factor_combo"
    swap_capable = True

    def __init__(self, pair_codes, rows, top_frac=0.10, bottom_frac=0.50, max_factors=3,
                 min_group_rows=200, max_tries=100):
        self.rows = np.sort(np.asarray(rows, dtype=np.int64))
        self.fit_codes = np.asarray(pair_codes, dtype=np.float32)[self.rows]
        self.num_factors = self.fit_codes.shape[1]
        self.top_frac, self.bottom_frac, self.max_factors = top_frac, bottom_frac, max_factors
        self.min_group_rows, self.max_tries = min_group_rows, max_tries

    def _random_factors(self, rng, exclude=()):
        available = np.setdiff1d(np.arange(self.num_factors), np.asarray(exclude, dtype=np.int64))
        k = int(rng.integers(1, min(self.max_factors, len(available)) + 1))
        return np.sort(rng.choice(available, k, replace=False)), rng.dirichlet(np.ones(k))

    def _condition(self, factors, weights):
        score = self.fit_codes[:, factors] @ weights
        hi = np.quantile(score, 1.0 - self.top_frac)
        lo = np.quantile(score, self.bottom_frac)
        inside_mask = (score >= hi) & (score > 0)
        outside_mask = (score <= lo) & ~inside_mask
        if inside_mask.sum() < self.min_group_rows or outside_mask.sum() < self.min_group_rows:
            return None
        key = (tuple(int(f) for f in factors), tuple(round(float(w), 6) for w in weights))
        return Condition(self.name, key, self.rows[inside_mask], self.rows[outside_mask])

    def sample_condition(self, rng) -> Condition:
        for _ in range(self.max_tries):
            condition = self._condition(*self._random_factors(rng))
            if condition is not None:
                return condition
        raise RuntimeError("FactorComboSource found no valid condition in max_tries")

    def sample_swap(self, rng) -> tuple[Condition, Condition]:
        for _ in range(self.max_tries):
            a = self.sample_condition(rng)
            for _ in range(self.max_tries):
                b = self._condition(*self._random_factors(rng, exclude=a.key[0]))
                if b is not None and len(np.intersect1d(a.inside, b.inside)) >= MIN_SWAP_OVERLAP:
                    return a, b
        raise RuntimeError("FactorComboSource found no valid swap pair in max_tries")


class _PartitionSource:
    """Condition = one group of a partition (per view); outside = the other groups of that view."""

    swap_capable = False

    def _setup(self, labels_by_view: dict, rows, min_group_rows, max_tries):
        self.rows = np.sort(np.asarray(rows, dtype=np.int64))
        self.labels_by_view = {v: np.asarray(l, dtype=np.int64) for v, l in labels_by_view.items()}
        self.max_tries = max_tries
        self._members, self._outside = {}, {}
        for view, labels in self.labels_by_view.items():
            fit_labels = labels[self.rows]
            for group in np.unique(fit_labels):
                if group < 0:
                    continue
                inside = self.rows[fit_labels == group]
                if len(inside) >= min_group_rows and len(self.rows) - len(inside) >= min_group_rows:
                    self._members[(view, int(group))] = inside
        if not self._members:
            raise RuntimeError(f"{self.name}: no group has at least {min_group_rows} rows")
        self.valid_keys = sorted(self._members)

    def _condition(self, key) -> Condition:
        if key not in self._outside:
            self._outside[key] = np.setdiff1d(self.rows, self._members[key], assume_unique=True)
        return Condition(self.name, key, self._members[key], self._outside[key])

    def sample_condition(self, rng) -> Condition:
        return self._condition(self.valid_keys[int(rng.integers(len(self.valid_keys)))])

    def sample_swap(self, rng) -> tuple[Condition, Condition]:
        raise NotImplementedError(f"{self.name} has one group per item; it cannot form swap pairs")


class ClipClusterSource(_PartitionSource):
    """k-means on L2-normalized raw CLIP image features and, separately, caption features."""

    name = "clip_cluster"
    swap_capable = True

    def __init__(self, img_features, txt_features, rows, n_clusters=64, seed=42, min_group_rows=200,
                 max_tries=100):
        rows = np.sort(np.asarray(rows, dtype=np.int64))
        labels_by_view = {}
        for view, feats in (("image", img_features), ("caption", txt_features)):
            x = np.asarray(feats, dtype=np.float32)[rows]
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
            fit = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=3,
                                  batch_size=4096).fit_predict(x)
            labels = np.full(len(feats), -1, dtype=np.int64)
            labels[rows] = fit
            labels_by_view[view] = labels
        self._setup(labels_by_view, rows, min_group_rows, max_tries)

    def sample_swap(self, rng) -> tuple[Condition, Condition]:
        for _ in range(self.max_tries):
            anchor = int(self.rows[rng.integers(len(self.rows))])
            a_key = ("image", int(self.labels_by_view["image"][anchor]))
            b_key = ("caption", int(self.labels_by_view["caption"][anchor]))
            if a_key in self._members and b_key in self._members:
                a, b = self._condition(a_key), self._condition(b_key)
                if len(np.intersect1d(a.inside, b.inside)) >= MIN_SWAP_OVERLAP:
                    return a, b
        raise RuntimeError("ClipClusterSource found no valid swap pair in max_tries")


class CommunitySource(_PartitionSource):
    """Block 1 Stage-1 communities (labels computed on scorer-train rows; -1 elsewhere)."""

    name = "community"

    def __init__(self, community_labels, rows, min_group_rows=200, max_tries=100):
        self._setup({"community": community_labels}, rows, min_group_rows, max_tries)
