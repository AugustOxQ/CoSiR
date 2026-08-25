"""
Test: buddy_graph.mix_distances_typed -- modality-provenance-aware distance mixing.
Uses each edge's own supporting modality's rank alone for img-only/txt-only edges;
keeps the existing fixed-alpha blend for "both" (cross-modally-confirmed) and "repair"
(neither modality) edges, where there is no disagreement to correct.

Run:
    python src/test/20260824_buddy_distance_mode/test_mix_distances_typed.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
from scipy.sparse import csr_matrix

from src.conditional_buddy.buddy_graph import mix_distances, mix_distances_typed


def test_both_and_repair_edges_keep_the_blend():
    """Edges present in both A_img and A_txt ("both"), or in neither ("repair"), must
    get the EXACT SAME value as the existing fixed-alpha mix_distances -- only
    single-modality edges should differ."""
    n = 4
    D_img = np.zeros((n, n))
    D_txt = np.zeros((n, n))
    D_img[0, 1] = D_img[1, 0] = 0.3   # (0,1): "both" edge
    D_txt[0, 1] = D_txt[1, 0] = 0.4
    D_img[2, 3] = D_img[3, 2] = 0.6   # (2,3): "repair" edge (in neither A_img nor A_txt)
    D_txt[2, 3] = D_txt[3, 2] = 0.7
    D_img_n, D_txt_n = csr_matrix(D_img), csr_matrix(D_txt)

    A_img = csr_matrix(np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    A_txt = csr_matrix(np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))

    alpha = 0.5
    mixed_typed = mix_distances_typed(D_img_n, D_txt_n, A_img, A_txt, alpha)
    mixed_blend = mix_distances(D_img_n, D_txt_n, alpha)

    assert abs(mixed_typed[0, 1] - mixed_blend[0, 1]) < 1e-9, (mixed_typed[0, 1], mixed_blend[0, 1])
    assert abs(mixed_typed[2, 3] - mixed_blend[2, 3]) < 1e-9, (mixed_typed[2, 3], mixed_blend[2, 3])
    print("PASS test_both_and_repair_edges_keep_the_blend")


def test_single_modality_edges_use_their_own_distance_alone():
    n = 4
    D_img = np.zeros((n, n))
    D_txt = np.zeros((n, n))
    D_img[0, 1] = D_img[1, 0] = 0.2   # img_only edge: good image rank...
    D_txt[0, 1] = D_txt[1, 0] = 0.9   # ...but bad (disagreeing) text rank
    D_img[2, 3] = D_img[3, 2] = 0.85  # txt_only edge: bad image rank...
    D_txt[2, 3] = D_txt[3, 2] = 0.15  # ...but good text rank
    D_img_n, D_txt_n = csr_matrix(D_img), csr_matrix(D_txt)

    A_img = csr_matrix(np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))  # only (0,1)
    A_txt = csr_matrix(np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]))  # only (2,3)

    mixed = mix_distances_typed(D_img_n, D_txt_n, A_img, A_txt, alpha=0.5)
    assert abs(mixed[0, 1] - 0.2) < 1e-9, (
        f"img_only edge should use its own (image) distance alone, got {mixed[0, 1]}"
    )
    assert abs(mixed[2, 3] - 0.15) < 1e-9, (
        f"txt_only edge should use its own (text) distance alone, got {mixed[2, 3]}"
    )
    print("PASS test_single_modality_edges_use_their_own_distance_alone")


if __name__ == "__main__":
    test_both_and_repair_edges_keep_the_blend()
    test_single_modality_edges_use_their_own_distance_alone()
    print("ALL TESTS PASSED")
