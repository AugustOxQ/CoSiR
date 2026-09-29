"""Leakage-free train/val/held splits grouped by painting and identical image vector."""

import hashlib
from dataclasses import dataclass

import numpy as np


def image_hashes(img_features: np.ndarray) -> np.ndarray:
    rows = np.ascontiguousarray(img_features, dtype=np.float32)
    return np.asarray([hashlib.md5(row.tobytes()).hexdigest() for row in rows])


def leakage_groups(paintings: np.ndarray, img_features: np.ndarray) -> np.ndarray:
    """Union rows that share a painting or an exact image vector; return dense group ids."""
    n = len(paintings)
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for keys in (np.asarray(paintings).tolist(), image_hashes(img_features).tolist()):
        first: dict = {}
        for row, key in enumerate(keys):
            if key in first:
                a, b = find(row), find(first[key])
                if a != b:
                    parent[a] = b
            else:
                first[key] = row
    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
    return np.unique(roots, return_inverse=True)[1].astype(np.int64)


@dataclass(frozen=True)
class GroupedSplit:
    train: np.ndarray
    val: np.ndarray
    held: np.ndarray


def grouped_split(groups: np.ndarray, fractions=(0.7, 0.1, 0.2), seed: int = 42) -> GroupedSplit:
    """Shuffle whole groups, then cut by cumulative row share at the fraction boundaries."""
    if len(fractions) != 3 or min(fractions) <= 0 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must be three positive numbers summing to 1")
    groups = np.asarray(groups)
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(unique))
    shares = counts[order] / counts.sum()
    start = np.cumsum(shares) - shares
    group_part = np.empty(len(unique), dtype=np.int64)
    group_part[order] = np.searchsorted(np.cumsum(fractions)[:2], start, side="right")
    row_part = group_part[inverse]
    parts = [np.flatnonzero(row_part == p).astype(np.int64) for p in range(3)]
    if any(len(p) == 0 for p in parts):
        raise ValueError("A split part is empty; too few groups for these fractions")
    return GroupedSplit(*parts)


def split_leakage(split: GroupedSplit, paintings: np.ndarray, img_features: np.ndarray) -> dict:
    """Rows of val/held whose painting or exact image vector also appears in an earlier part.

    Exact keys only: zero here means no shared painting slug and no bit-identical image vector,
    not "no near-duplicate image". Final-review measurement (seed-42 ArtELingo split):
    24 held rows (0.04%) have a train image at CLIP cosine >= 0.99 and 63 (0.10%) at >= 0.98. These
    are real WikiArt duplicates stored under different painting slugs, e.g.
    ``camille-pissarro_boulevard-montmartre-spring-rain`` vs its ``...-1897`` slug (as the reviewer reported it).
    The final review judged that they change no reported conclusion.
    """
    paintings = np.asarray(paintings)
    hashes = image_hashes(img_features)
    checks = (("val", "train", split.val, split.train),
              ("held", "train", split.held, split.train),
              ("held", "val", split.held, split.val))
    out = {}
    for part, reference_name, rows, reference in checks:
        seen_paintings, seen_hashes = set(paintings[reference]), set(hashes[reference])
        out[f"{part}_rows_painting_in_{reference_name}"] = int(sum(p in seen_paintings for p in paintings[rows]))
        out[f"{part}_rows_image_in_{reference_name}"] = int(sum(h in seen_hashes for h in hashes[rows]))
    return out
