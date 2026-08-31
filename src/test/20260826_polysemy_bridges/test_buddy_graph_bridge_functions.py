"""Tests for classify_edges/bridge_node_stats (Experiment 12,
docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md), promoted from the
one-off src/test/20260824_buddy_graph_disagreement/diagnose_disagreement.py diagnostic
into src/conditional_buddy/buddy_graph.py as reusable public functions.

Run:
    python src/test/20260826_polysemy_bridges/test_buddy_graph_bridge_functions.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
from scipy.sparse import csr_matrix

from src.conditional_buddy.buddy_graph import bridge_node_stats, classify_edges, hub_neighbor_pairs


def _csr(n, edges):
    rows, cols = [], []
    for i, j in edges:
        rows += [i, j]
        cols += [j, i]
    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def test_classify_edges_buckets_correctly():
    n = 6
    A_img = _csr(n, [(0, 1), (2, 3)])
    A_txt = _csr(n, [(0, 1), (4, 5)])
    E = _csr(n, [(0, 1), (2, 3), (4, 5), (1, 2)])

    typed = classify_edges(A_img, A_txt, E, n)
    keys = typed["keys"]
    pairs = {(int(k // n), int(k % n)) for k in keys}
    assert pairs == {(0, 1), (2, 3), (4, 5), (1, 2)}, pairs

    def _idx(pair):
        return list(keys).index(pair[0] * n + pair[1])

    assert typed["both"][_idx((0, 1))]
    assert typed["img_only"][_idx((2, 3))]
    assert typed["txt_only"][_idx((4, 5))]
    assert typed["repair"][_idx((1, 2))]
    stacked = np.stack([typed["img_only"], typed["txt_only"], typed["both"], typed["repair"]])
    assert np.all(stacked.sum(axis=0) == 1), "every edge must be classified into exactly one bucket"
    print("PASS test_classify_edges_buckets_correctly")


def test_bridge_node_detection():
    n = 5
    # Node 1 connects to 0 via img_only, and to 4 via txt_only -> node 1 is a bridge.
    A_img = _csr(n, [(0, 1), (2, 3)])
    A_txt = _csr(n, [(1, 4), (2, 3)])
    E = _csr(n, [(0, 1), (1, 4), (2, 3)])
    typed = classify_edges(A_img, A_txt, E, n)
    stats = bridge_node_stats(typed, n)
    assert stats["is_bridge"][1] == True, stats["is_bridge"]
    assert stats["is_bridge"][0] == False
    assert stats["is_bridge"][2] == False
    assert stats["n_bridge_nodes"] == 1
    print(f"PASS test_bridge_node_detection (n_bridge_nodes={stats['n_bridge_nodes']})")


def test_img_only_and_txt_only_neighbor_sets_are_disjoint():
    """A node's img-only and txt-only neighbor sets can never overlap -- each edge is
    classified into exactly one bucket, so no neighbor can appear via both bucket types
    for the same node. This is the invariant Task 4's extract_bridge_pairs relies on to
    guarantee B != C without an explicit check."""
    n = 5
    A_img = _csr(n, [(0, 1), (2, 3)])
    A_txt = _csr(n, [(1, 4), (2, 3)])
    E = _csr(n, [(0, 1), (1, 4), (2, 3)])
    typed = classify_edges(A_img, A_txt, E, n)
    stats = bridge_node_stats(typed, n)
    bridge_id = 1
    keys = typed["keys"]
    i = (keys // n).astype(np.int64)
    j = (keys % n).astype(np.int64)
    img_only_neighbors = set(j[(i == bridge_id) & typed["img_only"]]) | set(i[(j == bridge_id) & typed["img_only"]])
    txt_only_neighbors = set(j[(i == bridge_id) & typed["txt_only"]]) | set(i[(j == bridge_id) & typed["txt_only"]])
    assert img_only_neighbors.isdisjoint(txt_only_neighbors), (img_only_neighbors, txt_only_neighbors)
    print("PASS test_img_only_and_txt_only_neighbor_sets_are_disjoint")


def test_hub_neighbor_pairs_closed_vs_open():
    """A hub node (>=2 txt_only neighbors) whose neighbor-pair is ALSO connected by a
    real img_only edge is a 'closed triangle' (Experiment 14's positive-control case);
    a hub neighbor-pair with no such direct edge is 'open' (structurally identical to
    Experiment 12's B/C bridge pair, just sourced from a >=2-neighbor hub). A node with
    only 1 txt_only neighbor is a bridge but not a hub and contributes no pairs."""
    n = 8
    # Node 1: hub, txt_only neighbors 2 and 5; 2-5 is ALSO a real img_only edge -> closed.
    # Node 6: hub, txt_only neighbors 3 and 4; no edge between 3-4 -> open.
    # Node 0: bridge but NOT a hub (only 1 txt_only neighbor, to node 7) -> contributes no pairs.
    A_img = _csr(n, [(0, 1), (2, 5)])
    A_txt = _csr(n, [(1, 2), (1, 5), (6, 3), (6, 4), (7, 0)])
    E = _csr(n, [(0, 1), (2, 5), (1, 2), (1, 5), (6, 3), (6, 4), (7, 0)])

    typed = classify_edges(A_img, A_txt, E, n)
    stats = bridge_node_stats(typed, n)
    assert stats["deg_txt_only"][1] == 2
    assert stats["deg_txt_only"][6] == 2
    assert stats["deg_txt_only"][0] == 1  # bridge, but not a hub

    result = hub_neighbor_pairs(typed, stats, n)
    found = {
        (int(h), frozenset((int(c), int(d)))): bool(closed)
        for h, c, d, closed in zip(result["hub"], result["c"], result["d"], result["is_closed"])
    }
    assert found[(1, frozenset((2, 5)))] is True, found  # closed triangle
    assert found[(6, frozenset((3, 4)))] is False, found  # open pair
    assert len(found) == 2, found  # node 0 (deg_txt_only=1) contributes nothing
    print("PASS test_hub_neighbor_pairs_closed_vs_open")


if __name__ == "__main__":
    test_classify_edges_buckets_correctly()
    test_bridge_node_detection()
    test_img_only_and_txt_only_neighbor_sets_are_disjoint()
    test_hub_neighbor_pairs_closed_vs_open()
    print("ALL TESTS PASSED")
