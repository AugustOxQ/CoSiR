"""
Synthetic tests for the conditional-buddies graph construction (Steps 1-3).

Run:
    python src/test/20260609_conditional_buddy/test_buddy_graph.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import torch

from src.conditional_buddy.buddy_graph import (
    ensure_min_degree,
    mutual_knn,
    union_graph,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = torch.cuda.is_available()


def _two_cluster_features(n_per=100, dim=64, seed=0):
    """Two well-separated clusters shared across modalities (so buddies are meaningful)."""
    rng = np.random.default_rng(seed)
    c0 = rng.normal(0, 1, dim)
    c1 = rng.normal(5, 1, dim)
    labels = np.array([0] * n_per + [1] * n_per)
    centers = np.stack([c0, c1])
    img = centers[labels] + rng.normal(0, 0.5, (2 * n_per, dim))
    txt = centers[labels] + rng.normal(0, 0.5, (2 * n_per, dim))
    return img.astype(np.float32), txt.astype(np.float32), labels


def test_mutual_symmetry_and_degree():
    img, txt, labels = _two_cluster_features()
    A_img = mutual_knn(img, K=15, device=DEVICE, use_half=USE_HALF)
    A_txt = mutual_knn(txt, K=15, device=DEVICE, use_half=USE_HALF)

    # Mutual graph must be symmetric.
    assert (A_img != A_img.T).nnz == 0, "A_img is not symmetric"
    assert (A_txt != A_txt.T).nnz == 0, "A_txt is not symmetric"

    E = union_graph(A_img, A_txt)
    avg_deg = E.nnz / E.shape[0]
    print(f"  union E avg degree = {avg_deg:.2f}")
    assert avg_deg > 2, f"avg degree {avg_deg:.2f} not > 2"

    # Edges should overwhelmingly stay within a cluster.
    coo = E.tocoo()
    same = (labels[coo.row] == labels[coo.col]).mean()
    print(f"  fraction of within-cluster edges = {same:.3f}")
    assert same > 0.9, f"only {same:.3f} edges within cluster"
    print("PASS test_mutual_symmetry_and_degree")


def test_ensure_min_degree():
    img, txt, _ = _two_cluster_features()
    # Tiny K leaves some isolated nodes after the mutual+union filter.
    A_img = mutual_knn(img, K=2, device=DEVICE, use_half=USE_HALF)
    A_txt = mutual_knn(txt, K=2, device=DEVICE, use_half=USE_HALF)
    E = union_graph(A_img, A_txt)

    E_fixed, stats = ensure_min_degree(E, img, txt, device=DEVICE, use_half=USE_HALF)
    degrees = np.diff(E_fixed.indptr)
    print(f"  isolated fixed = {stats['num_isolated']}, min degree after = {degrees.min()}")
    assert degrees.min() >= 1, "min degree still 0 after ensure_min_degree"
    assert (E_fixed != E_fixed.T).nnz == 0, "fixed graph not symmetric"
    print("PASS test_ensure_min_degree")


if __name__ == "__main__":
    test_mutual_symmetry_and_degree()
    test_ensure_min_degree()
    print("\nAll buddy_graph tests passed.")
