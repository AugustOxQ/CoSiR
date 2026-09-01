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

from src.metrics.regularizer import (
    build_neighbor_csr,
    build_semantic_buddy_csr,
    buddy_graph_smoothness_loss,
    buddy_contrastive_loss,
    sample_buddy_neighbors,
    refresh_buddy_graph,
)
from src.conditional_buddy.buddy_graph import classify_edges
from src.conditional_buddy.compute_buddies import build_buddy_graphs


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


def test_semantic_csr_excludes_repair_and_skips_all_repair_anchor():
    # Edge 0-1 is repair-only; node 0 must become degree-zero after filtering.
    # Node 1 retains its semantic 1-2 edge so the resulting graph is non-empty.
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_types = np.array([3, 2], dtype=np.uint8)  # repair, both
    indptr, indices, edge_cdf, row_weight_sums = build_semantic_buddy_csr(
        edges, edge_types, num_nodes=3, exclude_repair=True,
    )
    assert (indptr[1] - indptr[0]).item() == 0
    assert (indptr[2] - indptr[1]).item() == 1

    comb = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    other_table = torch.eye(3, 2)
    loss, alignment = buddy_contrastive_loss(
        comb, torch.tensor([0, 1]), other_table, torch.nn.Identity(), torch.eye(2),
        indptr, indices, num_pos=2, edge_cdf=edge_cdf, row_weight_sums=row_weight_sums,
    )
    assert torch.isfinite(loss) and torch.isfinite(alignment)
    loss.backward()
    assert torch.count_nonzero(comb.grad[0]) == 0, "degree-zero repair-only anchor was not skipped"
    print("  test_semantic_csr_excludes_repair_and_skips_all_repair_anchor OK")


def test_uniform_weighted_sampling_matches_uniform_frequencies():
    edges = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
    edge_types = np.array([0, 1, 2], dtype=np.uint8)
    indptr, indices, edge_cdf, row_weight_sums = build_semantic_buddy_csr(
        edges, edge_types, num_nodes=4, type_weights={"img_only": 1.0, "txt_only": 1.0, "both": 1.0},
    )
    anchors = torch.zeros(30_000, dtype=torch.long)
    _, sampled = sample_buddy_neighbors(
        indptr, indices, anchors, num_samples=1,
        edge_cdf=edge_cdf, row_weight_sums=row_weight_sums,
        generator=torch.Generator().manual_seed(7),
    )
    sampled = sampled.flatten()
    counts = torch.bincount(sampled, minlength=4)[1:]
    expected = sampled.numel() / 3
    assert torch.all(torch.abs(counts.float() - expected) < expected * 0.04), counts.tolist()
    print("  test_uniform_weighted_sampling_matches_uniform_frequencies OK")


def test_none_type_weights_preserve_exact_uniform_sampling_path():
    edges = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
    indptr, indices, edge_cdf, row_weight_sums = build_semantic_buddy_csr(
        edges, None, num_nodes=4,
    )
    assert edge_cdf is None and row_weight_sums is None
    anchors = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    generator = torch.Generator().manual_seed(17)
    active, sampled = sample_buddy_neighbors(
        indptr, indices, anchors, num_samples=3, generator=generator,
    )
    replay = torch.Generator().manual_seed(17)
    deg = (indptr[anchors[active] + 1] - indptr[anchors[active]]).unsqueeze(1)
    offsets = torch.clamp(
        (torch.rand(active.numel(), 3, generator=replay) * deg).long(), max=deg - 1,
    )
    expected = indices[indptr[anchors[active]].unsqueeze(1) + offsets]
    assert torch.equal(sampled, expected)
    print("  test_none_type_weights_preserve_exact_uniform_sampling_path OK")


def test_skewed_type_weight_samples_both_neighbors_much_more_often():
    edges = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
    edge_types = np.array([0, 1, 2], dtype=np.uint8)  # img, txt, both
    indptr, indices, edge_cdf, row_weight_sums = build_semantic_buddy_csr(
        edges, edge_types, num_nodes=4,
        type_weights={"img_only": 1.0, "txt_only": 1.0, "both": 100.0},
    )
    anchors = torch.zeros(10_000, dtype=torch.long)
    _, sampled = sample_buddy_neighbors(
        indptr, indices, anchors, num_samples=1,
        edge_cdf=edge_cdf, row_weight_sums=row_weight_sums,
        generator=torch.Generator().manual_seed(8),
    )
    sampled = sampled.flatten()
    both_frequency = (sampled == 3).float().mean().item()
    assert abs(both_frequency - (100.0 / 102.0)) < 0.01, both_frequency
    print("  test_skewed_type_weight_samples_both_neighbors_much_more_often OK")


def test_type_aware_csr_requires_edge_provenance():
    edges = torch.tensor([[0], [1]], dtype=torch.long)
    try:
        build_semantic_buddy_csr(edges, None, num_nodes=2, exclude_repair=True)
    except ValueError as exc:
        assert "buddy_edge_types.npy" in str(exc)
    else:
        raise AssertionError("type-aware CSR accepted missing provenance")
    print("  test_type_aware_csr_requires_edge_provenance OK")


def test_weighted_smoothness_skips_zero_weight_repair_row():
    # With weighting on but repair exclusion off, the documented default repair
    # weight is zero. The weighted Family #1 path must still be grad-safe.
    edges = torch.tensor([[0], [1]], dtype=torch.long)
    indptr, indices, edge_cdf, row_weight_sums = build_semantic_buddy_csr(
        edges, np.array([3], dtype=np.uint8), num_nodes=2, type_weights={"both": 2.0},
    )
    emb = torch.randn(2, 3, requires_grad=True)
    loss = buddy_graph_smoothness_loss(
        emb, indptr, indices, torch.tensor([0]), num_samples=2,
        edge_cdf=edge_cdf, row_weight_sums=row_weight_sums,
    )
    assert loss.item() == 0.0
    loss.backward()
    assert torch.count_nonzero(emb.grad) == 0
    print("  test_weighted_smoothness_skips_zero_weight_repair_row OK")


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
    _, edges0, edge_types0 = compute_buddy_init(
        img, txt, n_dim=16, K=10, device="cpu", use_half=False, return_edges=True,
    )
    assert edges0.shape[0] == 2 and edges0.dtype == np.int64
    assert edge_types0.dtype == np.uint8 and edge_types0.shape == (edges0.shape[1],)
    assert (edges0[0] < edges0[1]).all(), "edges must be stored with i < j"

    # reordered output: output row k holds input id perm[k]
    perm = list(rng.permutation(N))
    _, edges_perm, edge_types_perm = compute_buddy_init(
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
    by_edge0 = {frozenset((int(a), int(b))): int(t) for (a, b), t in zip(edges0.T, edge_types0)}
    by_edge_perm = {
        frozenset((int(a), int(b))): int(t)
        for (a, b), t in zip(recovered.T, edge_types_perm)
    }
    assert by_edge0 == by_edge_perm, "edge types must remain aligned after edge remapping"

    A_img, A_txt, E = build_buddy_graphs(
        img, txt, K=10, device="cpu", use_half=False,
    )
    typed = classify_edges(A_img, A_txt, E, N)
    expected_counts = {
        0: int(typed["img_only"].sum()),
        1: int(typed["txt_only"].sum()),
        2: int(typed["both"].sum()),
        3: int(typed["repair"].sum()),
    }
    persisted_counts = {code: int((edge_types0 == code).sum()) for code in range(4)}
    assert persisted_counts == expected_counts, (persisted_counts, expected_counts)
    assert sum(persisted_counts.values()) == edges0.shape[1]
    print("  test_return_edges_and_remap OK")


def test_manager_edges_and_types_roundtrip(tmp_root=None):
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
        edge_types = np.array([0, 2, 3], dtype=np.uint8)
        np.save(emb_dir / "buddy_edges.npy", edges)
        np.save(emb_dir / "buddy_edge_types.npy", edge_types)

        # get_buddy_edges reads it back
        got = mgr.get_buddy_edges()
        assert got is not None and np.array_equal(got, edges)
        got_types = mgr.get_buddy_edge_types()
        assert got_types is not None and np.array_equal(got_types, edge_types)

        # round-trips through _copy_to / _copy_from (template persistence)
        tmpl = exp.parent / "template_embeddings"
        mgr._copy_to(tmpl)
        assert (tmpl / "buddy_edges.npy").exists(), "edges not copied into template"
        assert (tmpl / "buddy_edge_types.npy").exists(), "edge types not copied into template"
        (emb_dir / "buddy_edges.npy").unlink()
        (emb_dir / "buddy_edge_types.npy").unlink()
        mgr._copy_from(tmpl)
        assert np.array_equal(mgr.get_buddy_edges(), edges), "edges not restored from template"
        assert np.array_equal(mgr.get_buddy_edge_types(), edge_types), "types not restored from template"

        # A pre-provenance template must not leave stale types from a prior load.
        (tmpl / "buddy_edge_types.npy").unlink()
        mgr._copy_from(tmpl)
        assert mgr.get_buddy_edge_types() is None, "stale types survived an old-template load"
        print("  test_manager_edges_and_types_roundtrip OK")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_csr_symmetric()
    test_loss_value_single_neighbor()
    test_isolated_contributes_zero()
    test_semantic_csr_excludes_repair_and_skips_all_repair_anchor()
    test_uniform_weighted_sampling_matches_uniform_frequencies()
    test_none_type_weights_preserve_exact_uniform_sampling_path()
    test_skewed_type_weight_samples_both_neighbors_much_more_often()
    test_type_aware_csr_requires_edge_provenance()
    test_weighted_smoothness_skips_zero_weight_repair_row()
    test_gradient_shrinks_pair()
    test_return_edges_and_remap()
    test_manager_edges_and_types_roundtrip()
    print("ALL TASK 1 TESTS PASSED")
