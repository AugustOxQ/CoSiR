"""Family #3 buddy self-refresh — unit tests (runnable script)."""
import numpy as np
import torch
from scipy.sparse import csr_matrix

import src.metrics.regularizer as reg
from src.metrics.regularizer import (
    refresh_buddy_graph,
    edge_jaccard,
    build_neighbor_csr,
)


class _IdentityCombine(torch.nn.Module):
    """Stub model whose combine() returns the combine-side feature unchanged."""
    def combine(self, emb, emb_full, labels, **kw):
        return emb


def _sorted_neighbors(indptr, indices, n):
    return [sorted(indices[indptr[i]:indptr[i + 1]].tolist()) for i in range(n)]


def _sym_csr(n, undirected_edges):
    """Build a symmetric binary csr from a list of (i, j) undirected edges."""
    rows, cols = [], []
    for i, j in undirected_edges:
        rows += [i, j]
        cols += [j, i]
    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def test_equivalence_at_blend_zero():
    # blend=0 => mutual_knn never called => CSR must equal the CLIP-only CSR.
    n = 6
    clip = torch.tensor([[0, 2, 4], [1, 3, 5]], dtype=torch.long)  # edges 0-1,2-3,4-5
    z = torch.randn(n, 4)
    feat = torch.randn(n, 4)
    indptr, indices, comb_edges, stats = refresh_buddy_graph(
        _IdentityCombine(), feat, z, clip, num_nodes=n, blend=0.0
    )
    e_ip, e_ix = build_neighbor_csr(clip, n)
    assert torch.equal(indptr, e_ip), "blend=0 indptr differs from CLIP CSR"
    assert _sorted_neighbors(indptr, indices, n) == _sorted_neighbors(e_ip, e_ix, n)
    assert comb_edges.shape[1] == 0
    assert stats["graph_n_comb_edges"] == 0.0
    print("PASS test_equivalence_at_blend_zero")


def test_union_keeps_all_clip_edges(monkeypatch=None):
    # blend=1 => every CLIP edge present, plus the comb edges from mutual_knn.
    n = 6
    clip = torch.tensor([[0], [1]], dtype=torch.long)  # single CLIP edge 0-1
    fake = _sym_csr(n, [(2, 3), (4, 5)])  # comb graph disjoint from CLIP
    reg.mutual_knn = lambda features, K, **kw: fake
    z = torch.randn(n, 4)
    feat = torch.randn(n, 4)
    indptr, indices, comb_edges, stats = refresh_buddy_graph(
        _IdentityCombine(), feat, z, clip, num_nodes=n, blend=1.0
    )
    nbrs = _sorted_neighbors(indptr, indices, n)
    assert 1 in nbrs[0] and 0 in nbrs[1], "CLIP edge 0-1 missing after union"
    assert 3 in nbrs[2] and 5 in nbrs[4], "comb edges missing after union"
    assert stats["graph_n_comb_edges"] == 2.0
    assert abs(stats["graph_new_edge_frac"] - 1.0) < 1e-9  # both comb edges are new
    print("PASS test_union_keeps_all_clip_edges")


def test_blend_fraction_is_respected():
    n = 8
    clip = torch.tensor([[0], [1]], dtype=torch.long)
    fake = _sym_csr(n, [(2, 3), (4, 5), (6, 7), (2, 5)])  # 4 undirected comb edges
    reg.mutual_knn = lambda features, K, **kw: fake
    g = torch.Generator().manual_seed(0)
    idp, idx, comb_edges, stats = refresh_buddy_graph(
        _IdentityCombine(), torch.randn(n, 4), torch.randn(n, 4),
        clip, num_nodes=n, blend=0.5, generator=g,
    )
    assert stats["graph_n_comb_edges"] == 2.0, stats  # round(0.5*4)=2
    assert comb_edges.shape[1] == 2
    print("PASS test_blend_fraction_is_respected")


def test_index_alignment_comb_edges_are_z_positions():
    # A comb edge between positions (0,3) must land at z-positions 0 and 3.
    n = 5
    clip = torch.empty(2, 0, dtype=torch.long)
    fake = _sym_csr(n, [(0, 3)])
    reg.mutual_knn = lambda features, K, **kw: fake
    idp, idx, comb_edges, stats = refresh_buddy_graph(
        _IdentityCombine(), torch.randn(n, 4), torch.randn(n, 4),
        clip, num_nodes=n, blend=1.0,
    )
    nbrs = _sorted_neighbors(idp, idx, n)
    assert nbrs[0] == [3] and nbrs[3] == [0], nbrs
    print("PASS test_index_alignment_comb_edges_are_z_positions")


def test_no_grad_safety():
    n = 4
    clip = torch.tensor([[0], [1]], dtype=torch.long)
    z = torch.randn(n, 4, requires_grad=True)
    z_before = z.detach().clone()
    idp, idx, comb_edges, stats = refresh_buddy_graph(
        _IdentityCombine(), torch.randn(n, 4), z, clip, num_nodes=n, blend=0.0
    )
    assert torch.equal(z.detach(), z_before), "z was modified by refresh"
    assert idx.grad_fn is None and not idx.requires_grad
    print("PASS test_no_grad_safety")


class _DropoutCombine(torch.nn.Module):
    """Stub model whose combine() applies real dropout (train/eval-aware)."""
    def __init__(self):
        super().__init__()
        self.drop = torch.nn.Dropout(0.5)

    def combine(self, emb, emb_full, labels, **kw):
        return self.drop(emb)


def _nearest_neighbor_knn(features, K, **kw):
    """Feature-dependent CPU fake: connect each row to its top-1 NN by L2 dist.

    Deterministic given `features`. Unlike a fixed-return fake, this is
    sensitive to whether `combine()` was called in train mode (dropout noise
    changes `features` between calls) or eval mode (identical calls).
    """
    n = features.shape[0]
    d = np.sum((features[:, None, :] - features[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = np.argmin(d, axis=1)
    edges = [(i, int(nn[i])) for i in range(n)]
    return _sym_csr(n, edges)


def test_refresh_uses_eval_mode_deterministic():
    # Without eval-mode during the combine pass, active dropout (p=0.5) makes
    # comb_all differ across two calls with the same inputs -> the
    # feature-dependent mutual_knn fake yields different edge sets -> the
    # graphs diverge. With eval-mode (FIX 1), dropout is inert -> identical.
    n = 10
    torch.manual_seed(0)
    clip = torch.empty(2, 0, dtype=torch.long)
    z = torch.randn(n, 4)
    feat = torch.randn(n, 4) * 5.0  # scale up so dropout zeroing visibly moves NN structure

    stub = _DropoutCombine()
    stub.train()  # caller (train_cosir) always has model.train() active going in
    reg.mutual_knn = _nearest_neighbor_knn

    torch.manual_seed(1)
    _, _, comb_edges_1, _ = refresh_buddy_graph(
        stub, feat, z, clip, num_nodes=n, blend=1.0,
    )
    assert stub.training, "stub must be restored to train mode after refresh"

    torch.manual_seed(2)
    _, _, comb_edges_2, _ = refresh_buddy_graph(
        stub, feat, z, clip, num_nodes=n, blend=1.0,
    )
    assert stub.training, "stub must be restored to train mode after refresh"

    set1 = reg._undirected_edge_set(comb_edges_1)
    set2 = reg._undirected_edge_set(comb_edges_2)
    assert set1 == set2, (
        f"buddy graph is non-deterministic across refreshes with the same "
        f"inputs -- combine pass ran with dropout active (not eval mode): "
        f"{set1} != {set2}"
    )
    print("PASS test_refresh_uses_eval_mode_deterministic")


def test_edge_jaccard():
    a = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)  # {0-1, 2-3}
    b = torch.tensor([[0, 4], [1, 5]], dtype=torch.long)  # {0-1, 4-5}
    assert abs(edge_jaccard(a, b) - 1.0 / 3.0) < 1e-9     # 1 shared / 3 union
    assert edge_jaccard(torch.empty(2, 0, dtype=torch.long),
                        torch.empty(2, 0, dtype=torch.long)) == 1.0
    print("PASS test_edge_jaccard")


if __name__ == "__main__":
    test_equivalence_at_blend_zero()
    test_union_keeps_all_clip_edges()
    test_blend_fraction_is_respected()
    test_index_alignment_comb_edges_are_z_positions()
    test_no_grad_safety()
    test_refresh_uses_eval_mode_deterministic()
    test_edge_jaccard()
    print("ALL PASS")
