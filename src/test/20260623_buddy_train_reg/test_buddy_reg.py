"""
Unit tests for the Family #1 buddy-graph smoothness regularizer.

Run:
    python src/test/20260623_buddy_train_reg/test_buddy_reg.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import torch

from src.metrics.regularizer import build_neighbor_csr, buddy_graph_smoothness_loss


def test_csr_symmetric():
    # one undirected edge 0-1, node 2 isolated
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    indptr, indices = build_neighbor_csr(edge_index, num_nodes=3)
    assert indptr.tolist() == [0, 1, 2, 2], indptr.tolist()
    # node 0's neighbor is 1, node 1's neighbor is 0
    assert indices[indptr[0]:indptr[1]].tolist() == [1]
    assert indices[indptr[1]:indptr[2]].tolist() == [0]
    assert indices[indptr[2]:indptr[3]].tolist() == []
    print("  test_csr_symmetric OK")


def test_loss_value_single_neighbor():
    # nodes 0,1 connected; 2,3 isolated. Each anchor has exactly one neighbor,
    # so sampling is deterministic regardless of num_samples.
    emb = torch.tensor([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]])
    indptr, indices = build_neighbor_csr(torch.tensor([[0], [1]]), num_nodes=4)
    anchors = torch.tensor([0, 1], dtype=torch.long)
    loss = buddy_graph_smoothness_loss(emb, indptr, indices, anchors, num_samples=4)
    # ||z0 - z1||^2 = 9 + 16 = 25 for both anchors
    assert abs(loss.item() - 25.0) < 1e-5, loss.item()
    print("  test_loss_value_single_neighbor OK")


def test_isolated_contributes_zero():
    emb = torch.zeros(4, 2, requires_grad=True)
    indptr, indices = build_neighbor_csr(torch.tensor([[0], [1]]), num_nodes=4)
    anchors = torch.tensor([2, 3], dtype=torch.long)  # both isolated
    loss = buddy_graph_smoothness_loss(emb, indptr, indices, anchors, num_samples=4)
    assert loss.item() == 0.0, loss.item()
    loss.backward()
    assert torch.count_nonzero(emb.grad) == 0
    print("  test_isolated_contributes_zero OK")


def test_gradient_shrinks_pair():
    emb = torch.nn.Parameter(torch.tensor([[0.0, 0.0], [3.0, 4.0]]))
    indptr, indices = build_neighbor_csr(torch.tensor([[0], [1]]), num_nodes=2)
    anchors = torch.tensor([0, 1], dtype=torch.long)
    before = (emb[0] - emb[1]).norm().item()
    opt = torch.optim.SGD([emb], lr=0.01)
    opt.zero_grad()
    loss = buddy_graph_smoothness_loss(emb, indptr, indices, anchors, num_samples=4)
    loss.backward()
    opt.step()
    after = (emb[0] - emb[1]).norm().item()
    assert after < before, (before, after)
    print("  test_gradient_shrinks_pair OK")


def test_return_edges_and_remap():
    from src.conditional_buddy.compute_buddies import compute_buddy_init

    rng = np.random.default_rng(0)
    dim = 32
    c0 = rng.normal(0, 1, dim); c1 = rng.normal(6, 1, dim)
    labels = np.array([0] * 40 + [1] * 40)
    centers = np.stack([c0, c1])
    img = (centers[labels] + rng.normal(0, 0.4, (80, dim))).astype(np.float32)
    txt = (centers[labels] + rng.normal(0, 0.4, (80, dim))).astype(np.float32)
    N = 80
    ids = list(range(N))

    # input-order edges
    _, edges0 = compute_buddy_init(
        img, txt, n_dim=16, K=10, device="cpu", use_half=False, return_edges=True,
    )
    assert edges0.shape[0] == 2 and edges0.dtype == np.int64
    assert (edges0[0] < edges0[1]).all(), "edges must be stored with i < j"

    # reordered output: output row k holds input id perm[k]
    perm = list(rng.permutation(N))
    _, edges_perm = compute_buddy_init(
        img, txt, n_dim=16, K=10, device="cpu", use_half=False, return_edges=True,
        input_sample_ids=ids, output_sample_ids=perm,
    )
    # map output positions back to input positions via reorder == perm
    reorder = np.array(perm)
    recovered = reorder[edges_perm]  # [2, M] input positions
    assert (edges_perm[0] < edges_perm[1]).all(), "remapped edges must keep i < j ordering"
    set0 = {frozenset((int(a), int(b))) for a, b in edges0.T}
    setr = {frozenset((int(a), int(b))) for a, b in recovered.T}
    assert set0 == setr, "remapped edges do not connect the same samples"
    print("  test_return_edges_and_remap OK")


def test_manager_edges_roundtrip(tmp_root=None):
    import tempfile, shutil
    from pathlib import Path
    from src.utils.embedding_manager_nocache import TrainableEmbeddingManager

    root = Path(tempfile.mkdtemp())
    try:
        exp = root / "exp" / "run0"
        emb_dir = exp / "training_embeddings"
        mgr = TrainableEmbeddingManager(
            sample_ids=list(range(6)), embedding_dim=16,
            embeddings_dir=str(emb_dir), mode="ram", initialization_strategy="zeros",
        )
        edges = np.array([[0, 2, 4], [1, 3, 5]], dtype=np.int64)
        np.save(emb_dir / "buddy_edges.npy", edges)

        # get_buddy_edges reads it back
        got = mgr.get_buddy_edges()
        assert got is not None and np.array_equal(got, edges)

        # round-trips through _copy_to / _copy_from (template persistence)
        tmpl = exp.parent / "template_embeddings"
        mgr._copy_to(tmpl)
        assert (tmpl / "buddy_edges.npy").exists(), "edges not copied into template"
        (emb_dir / "buddy_edges.npy").unlink()
        mgr._copy_from(tmpl)
        assert np.array_equal(mgr.get_buddy_edges(), edges), "edges not restored from template"
        print("  test_manager_edges_roundtrip OK")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_csr_symmetric()
    test_loss_value_single_neighbor()
    test_isolated_contributes_zero()
    test_gradient_shrinks_pair()
    test_return_edges_and_remap()
    test_manager_edges_roundtrip()
    print("ALL TASK 1 TESTS PASSED")
