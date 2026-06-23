"""
Synthetic tests for the full conditional-buddies init (Steps 1-6) + reorder.

Run:
    python src/test/20260609_conditional_buddy/test_compute_buddies.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import torch

from src.conditional_buddy.compute_buddies import compute_buddy_init

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = torch.cuda.is_available()


def _two_cluster_features(n_per=100, dim=64, seed=1):
    rng = np.random.default_rng(seed)
    c0 = rng.normal(0, 1, dim)
    c1 = rng.normal(5, 1, dim)
    labels = np.array([0] * n_per + [1] * n_per)
    centers = np.stack([c0, c1])
    img = centers[labels] + rng.normal(0, 0.5, (2 * n_per, dim))
    txt = centers[labels] + rng.normal(0, 0.5, (2 * n_per, dim))
    return img.astype(np.float32), txt.astype(np.float32), labels


def test_shape_and_range():
    img, txt, _ = _two_cluster_features()
    emb = compute_buddy_init(
        img, txt, n_dim=16, K=15, device=DEVICE, use_half=USE_HALF
    )
    print(f"  shape={emb.shape}, range=[{emb.min():.3f}, {emb.max():.3f}]")
    assert emb.shape == (img.shape[0], 16), f"unexpected shape {emb.shape}"
    assert emb.dtype == np.float32
    assert emb.min() >= -1.0 - 1e-5 and emb.max() <= 1.0 + 1e-5, "values out of [-1, 1]"
    print("PASS test_shape_and_range")


def test_buddies_closer_than_random():
    img, txt, labels = _two_cluster_features()
    emb = compute_buddy_init(
        img, txt, n_dim=2, K=15, device=DEVICE, use_half=USE_HALF
    )
    rng = np.random.default_rng(0)
    # "Buddies" = same-cluster pairs; "random" = arbitrary pairs.
    same_pairs = [(i, j) for i in range(0, 50) for j in range(i + 1, 50)]  # both cluster 0
    a = rng.integers(0, len(labels), 500)
    b = rng.integers(0, len(labels), 500)
    buddy_d = np.mean([np.linalg.norm(emb[i] - emb[j]) for i, j in same_pairs])
    random_d = np.mean(np.linalg.norm(emb[a] - emb[b], axis=1))
    print(f"  buddy mean dist={buddy_d:.4f}, random mean dist={random_d:.4f}")
    assert buddy_d < random_d, "buddy pairs are not closer than random pairs"
    print("PASS test_buddies_closer_than_random")


def test_reorder_correctness():
    img, txt, _ = _two_cluster_features()
    N = img.shape[0]
    ids = list(range(N))

    base = compute_buddy_init(img, txt, n_dim=16, K=15, device=DEVICE, use_half=USE_HALF)

    rng = np.random.default_rng(7)
    perm = list(rng.permutation(N))
    reordered = compute_buddy_init(
        img, txt, n_dim=16, K=15, device=DEVICE, use_half=USE_HALF,
        input_sample_ids=ids, output_sample_ids=perm,
    )
    # reordered[i] must equal base row for sample id perm[i] (== base[perm[i]] since ids==range).
    for i in range(N):
        assert np.allclose(reordered[i], base[perm[i]], atol=1e-5), f"reorder mismatch at {i}"
    print("PASS test_reorder_correctness")


if __name__ == "__main__":
    test_shape_and_range()
    test_buddies_closer_than_random()
    test_reorder_correctness()
    print("\nAll compute_buddies tests passed.")
