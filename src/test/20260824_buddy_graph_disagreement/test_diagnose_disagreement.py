"""
Test: modality-disagreement diagnostic for the buddy graph — classifies every edge of
the final (repaired) union graph E by which modality(ies) actually support it
(img_only / txt_only / both / repair-added), measures how much the fixed-alpha distance
mix dilutes single-modality-only edges relative to their supporting modality alone, and
flags "bridge" nodes that connect to different neighbors via different modalities.

Run:
    python src/test/20260824_buddy_graph_disagreement/test_diagnose_disagreement.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.sparse import csr_matrix

from diagnose_disagreement import bridge_node_stats, classify_edges, diagnose, rank_normalize


def _csr(n, edges):
    """Build a symmetric binary csr_matrix from an undirected edge list."""
    rows, cols = [], []
    for i, j in edges:
        rows += [i, j]
        cols += [j, i]
    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def test_classify_edges_buckets_correctly():
    n = 6
    # A_img edges: (0,1) [also in txt -> "both"], (2,3) [img-only]
    A_img = _csr(n, [(0, 1), (2, 3)])
    # A_txt edges: (0,1) [both], (4,5) [txt-only]
    A_txt = _csr(n, [(0, 1), (4, 5)])
    # E adds one more edge (1,2) that's in NEITHER A_img nor A_txt -- a "repair" edge
    # (as ensure_min_degree/ensure_connected would add), on top of the union of the two.
    E = _csr(n, [(0, 1), (2, 3), (4, 5), (1, 2)])

    typed = classify_edges(A_img, A_txt, E, n)
    keys = typed["keys"]
    pairs = {(int(k // n), int(k % n)) for k in keys}
    assert pairs == {(0, 1), (2, 3), (4, 5), (1, 2)}, pairs

    def _mask_for(pair):
        idx = list(keys).index(pair[0] * n + pair[1])
        return idx

    assert typed["both"][_mask_for((0, 1))]
    assert typed["img_only"][_mask_for((2, 3))]
    assert typed["txt_only"][_mask_for((4, 5))]
    assert typed["repair"][_mask_for((1, 2))]
    # Every edge falls into exactly one bucket.
    stacked = np.stack([typed["img_only"], typed["txt_only"], typed["both"], typed["repair"]])
    assert np.all(stacked.sum(axis=0) == 1), "every edge must be classified into exactly one bucket"
    print("PASS test_classify_edges_buckets_correctly")


def test_rank_normalize_matches_existing_convention():
    """Must match buddy_graph.rank_normalise_sparse's semantics: smallest value -> rank
    1/n (best/closest), largest value -> rank n/n (worst/farthest)."""
    x = np.array([0.5, 0.1, 0.9, 0.3])
    r = rank_normalize(x)
    assert r[1] < r[3] < r[0] < r[2], r  # 0.1 < 0.3 < 0.5 < 0.9 in value -> same order in rank
    assert abs(r.min() - 0.25) < 1e-9 and abs(r.max() - 1.0) < 1e-9
    print("PASS test_rank_normalize_matches_existing_convention")


def test_dilution_direction_on_a_constructed_disagreement():
    """Construct exactly the scenario under discussion: node pair (0,1) is an img_only
    edge -- very close in image space, very far in text space. Its mixed rank should sit
    between its (good) image rank and its (bad) text rank, i.e. WORSE (higher) than its
    image-only rank -- this is the dilution effect, made concrete and checkable."""
    n = 4
    d = 8
    rng = np.random.default_rng(0)

    img_n = rng.normal(size=(n, d)).astype(np.float32)
    img_n /= np.linalg.norm(img_n, axis=1, keepdims=True)
    txt_n = rng.normal(size=(n, d)).astype(np.float32)
    txt_n /= np.linalg.norm(txt_n, axis=1, keepdims=True)

    # Force (0,1) to be near-identical in IMAGE space (small d_img)...
    img_n[1] = img_n[0]
    # ...and force (0,1) to be near-antipodal in TEXT space (large d_txt).
    txt_n[1] = -txt_n[0]
    # A few filler edges spanning a range of img/txt distances, so rank_normalize has
    # something to rank against besides the one edge under test.
    A_img = _csr(n, [(0, 1), (0, 2), (1, 3)])
    A_txt = _csr(n, [(0, 2), (1, 3), (2, 3)])
    E = _csr(n, [(0, 1), (0, 2), (1, 3), (2, 3)])

    typed, r_img, r_txt, r_mixed = diagnose(img_n, txt_n, A_img, A_txt, E, alpha=0.5)
    keys = typed["keys"]
    idx01 = list(keys).index(0 * n + 1)
    assert typed["img_only"][idx01], "edge (0,1) must be classified img_only"

    assert r_img[idx01] < r_mixed[idx01], (
        f"dilution check failed: expected r_img[(0,1)]={r_img[idx01]:.3f} < "
        f"r_mixed[(0,1)]={r_mixed[idx01]:.3f} (mixing in the disagreeing text distance "
        f"should make the edge look WORSE than its image-only rank, not better/equal)"
    )
    print(f"PASS test_dilution_direction_on_a_constructed_disagreement "
          f"(r_img={r_img[idx01]:.3f}, r_txt={r_txt[idx01]:.3f}, r_mixed={r_mixed[idx01]:.3f})")


def test_bridge_node_detection():
    n = 5
    # Node 1 connects to 0 via img_only, and to 4 via txt_only -> node 1 is a bridge.
    # Node 2-3 is both-only -> neither is a bridge.
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


if __name__ == "__main__":
    test_classify_edges_buckets_correctly()
    test_rank_normalize_matches_existing_convention()
    test_dilution_direction_on_a_constructed_disagreement()
    test_bridge_node_detection()
    print("ALL TESTS PASSED")
