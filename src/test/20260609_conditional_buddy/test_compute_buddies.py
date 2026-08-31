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
from scipy.sparse.csgraph import connected_components

from src.conditional_buddy.compute_buddies import build_buddy_graphs, compute_buddy_init

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


def test_ensure_connected():
    # Two well-separated clusters → the buddy graph has 2 components. The fix should
    # bridge them into 1 with a single MST edge; off, it stays disconnected.
    img, txt, _ = _two_cluster_features()

    _, _, E_off = build_buddy_graphs(
        img, txt, K=15, device=DEVICE, use_half=USE_HALF, connect_components=False
    )
    n_off = connected_components(E_off, directed=False, return_labels=False)
    assert n_off >= 2, f"expected a disconnected graph for the assertion, got {n_off}"

    _, _, E_on = build_buddy_graphs(
        img, txt, K=15, device=DEVICE, use_half=USE_HALF, connect_components=True
    )
    n_on = connected_components(E_on, directed=False, return_labels=False)
    print(f"  components: connect_off={n_off} connect_on={n_on}")
    assert n_on == 1, f"ensure_connected left {n_on} components"
    # bridges added = components_before - 1; symmetric → 2*(C-1) directed entries.
    assert E_on.nnz == E_off.nnz + 2 * (n_off - 1), "unexpected number of bridge edges"
    print("PASS test_ensure_connected")


def test_b_weight_identity():
    # B-lean off (b_weight=1.0) must be byte-for-byte the current union-graph init.
    img, txt, _ = _two_cluster_features()
    base = compute_buddy_init(img, txt, n_dim=16, K=15, device=DEVICE, use_half=USE_HALF)
    b1 = compute_buddy_init(
        img, txt, n_dim=16, K=15, device=DEVICE, use_half=USE_HALF, b_weight=1.0
    )
    assert np.allclose(base, b1, atol=1e-6), "b_weight=1.0 changed the default init"
    print("PASS test_b_weight_identity")


def test_b_weight_tightens_buddies():
    # Upweighting strict-B affinity should pull strict-buddy pairs at least as tight.
    img, txt, _ = _two_cluster_features()
    A_img, A_txt, _ = build_buddy_graphs(
        img, txt, K=15, device=DEVICE, use_half=USE_HALF
    )
    B = A_img.multiply(A_txt).tocsr()
    B.data[:] = 1.0
    B.eliminate_zeros()
    coo = B.tocoo()
    up = coo.row < coo.col
    be = np.stack([coo.row[up], coo.col[up]])  # strict-buddy edges (i<j)
    assert be.shape[1] > 20, f"need strict buddies for the test, got {be.shape[1]}"

    def buddy_dist(beta):
        emb = compute_buddy_init(
            img, txt, n_dim=8, K=15, device=DEVICE, use_half=USE_HALF, b_weight=beta
        )
        return float(np.mean(np.linalg.norm(emb[be[0]] - emb[be[1]], axis=1)))

    d1, d8 = buddy_dist(1.0), buddy_dist(8.0)
    print(f"  strict-buddy mean emb dist: beta=1 -> {d1:.4f}, beta=8 -> {d8:.4f}")
    assert d8 <= d1 + 1e-6, "b_weight>1 did not tighten strict buddies"
    print("PASS test_b_weight_tightens_buddies")


def test_b_weight_forwarded_through_manager():
    # The training path (EmbeddingManager.initialize_embeddings_buddies) must forward
    # b_weight all the way to compute_buddy_init — else a b_weight sweep is a silent no-op.
    import tempfile
    import torch as _torch
    import src.conditional_buddy as cb
    from src.utils.embedding_manager_nocache import TrainableEmbeddingManager

    img, txt, _ = _two_cluster_features(n_per=20, dim=16)
    N, D = img.shape[0], 8
    ids = list(range(N))

    class FakeFM:
        def get_num_chunks(self):
            return 1

        def get_features_by_chunk(self, _):
            return {"img_features": _torch.from_numpy(img),
                    "txt_features": _torch.from_numpy(txt)}

        def get_all_sample_ids(self):
            return ids

    captured = {}
    orig = cb.compute_buddy_init

    def spy(*a, **kw):
        captured.update(kw)
        return np.zeros((N, D), np.float32), np.zeros((2, 1), np.int64)

    cb.compute_buddy_init = spy
    try:
        with tempfile.TemporaryDirectory() as d:
            mgr = TrainableEmbeddingManager(ids, D, d, mode="ram")
            mgr.initialize_embeddings_buddies(FakeFM(), None, "cpu", b_weight=7.0)
    finally:
        cb.compute_buddy_init = orig
    print(f"  forwarded b_weight={captured.get('b_weight')}")
    assert captured.get("b_weight") == 7.0, "b_weight not forwarded to compute_buddy_init"
    print("PASS test_b_weight_forwarded_through_manager")


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
    test_ensure_connected()
    test_b_weight_identity()
    test_b_weight_tightens_buddies()
    test_b_weight_forwarded_through_manager()
    test_reorder_correctness()
    print("\nAll compute_buddies tests passed.")
