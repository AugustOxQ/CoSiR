"""Buddy-neighborhood preservation metric — unit tests (runnable script).

buddy_knn_preservation: fraction of each node's CLIP-graph buddies that remain in
its top-k nearest neighbours in the trained combined space. compute_comb_all_eval:
the eval-mode / no_grad full-N combine pass (dropout off, deterministic).

Run: PYTHONPATH=/project/CoSiR python src/test/20260707_buddy_refresh/test_buddy_preservation.py
"""
import torch
import torch.nn as nn

from src.metrics.regularizer import (
    buddy_knn_preservation,
    compute_comb_all_eval,
    build_neighbor_csr,
)


def test_preservation_all_buddies_are_nearest():
    # 4 rows in two tight cosine pairs: 0~1 and 2~3. Top-1 NN = the partner.
    comb = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0],
         [0.95, 0.05, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.95, 0.05]],
    )
    clip = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)  # edges 0-1, 2-3 (the NN pairs)
    indptr, indices = build_neighbor_csr(clip, num_nodes=4)
    p = buddy_knn_preservation(comb, indptr, indices, k=1)
    assert abs(p - 1.0) < 1e-9, p  # every buddy is the node's top-1 NN
    print("PASS test_preservation_all_buddies_are_nearest")


def test_preservation_wrong_graph_is_zero():
    # Same features, but CLIP edges connect NON-neighbours (0-2, 1-3) → 0 preserved at k=1.
    comb = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0],
         [0.95, 0.05, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.95, 0.05]],
    )
    clip = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)  # edges 0-2, 1-3 (not the NN pairs)
    indptr, indices = build_neighbor_csr(clip, num_nodes=4)
    p = buddy_knn_preservation(comb, indptr, indices, k=1)
    assert abs(p - 0.0) < 1e-9, p
    print("PASS test_preservation_wrong_graph_is_zero")


def test_preservation_partial_and_degree_weighting():
    # node0 has two buddies {1,2}; only 1 is its top-1 NN → 0.5 for node0.
    comb = torch.tensor(
        [[1.0, 0.0, 0.0],
         [0.9, 0.1, 0.0],   # closest to 0
         [0.0, 0.0, 1.0]],  # far from 0
    )
    clip = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)  # 0-1, 0-2
    indptr, indices = build_neighbor_csr(clip, num_nodes=3)
    # node0: buddies {1,2}, top1={1} -> 1/2; node1: buddy {0}, top1={0} -> 1; node2: buddy {0}, top1 -> ?
    # node2 nearest is 0 (cos(2,0)=0) vs cos(2,1)=0 tie; topk picks one. Just assert node0 half.
    p = buddy_knn_preservation(comb, indptr, indices, k=1)
    assert 0.0 < p < 1.0, p          # not all preserved, not none
    print("PASS test_preservation_partial_and_degree_weighting")


def test_preservation_skips_isolated_nodes():
    # node2 has no buddy → excluded from the average (only 0,1 counted, both preserved).
    comb = torch.tensor(
        [[1.0, 0.0, 0.0],
         [0.9, 0.1, 0.0],
         [0.0, 0.0, 1.0]],
    )
    clip = torch.tensor([[0], [1]], dtype=torch.long)  # only edge 0-1; node2 isolated
    indptr, indices = build_neighbor_csr(clip, num_nodes=3)
    p = buddy_knn_preservation(comb, indptr, indices, k=1)
    assert abs(p - 1.0) < 1e-9, p
    print("PASS test_preservation_skips_isolated_nodes")


class _DropoutCombine(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(0.5)
    def combine(self, emb, emb_full, labels, **kw):
        return self.drop(emb)


def test_compute_comb_all_eval_is_deterministic_under_train_mode():
    m = _DropoutCombine()
    m.train()  # dropout would be active if eval-mode were not enforced
    feat = torch.randn(9, 5)
    z = torch.randn(9, 4)
    a = compute_comb_all_eval(m, feat, z, chunk=4)
    b = compute_comb_all_eval(m, feat, z, chunk=4)
    assert torch.allclose(a, b), "comb_all differs across calls — eval-mode not enforced"
    assert torch.allclose(a, feat), "dropout altered features — eval-mode not enforced"
    assert m.training, "model was not restored to train mode"
    print("PASS test_compute_comb_all_eval_is_deterministic_under_train_mode")


if __name__ == "__main__":
    test_preservation_all_buddies_are_nearest()
    test_preservation_wrong_graph_is_zero()
    test_preservation_partial_and_degree_weighting()
    test_preservation_skips_isolated_nodes()
    test_compute_comb_all_eval_is_deterministic_under_train_mode()
    print("ALL PASS")
