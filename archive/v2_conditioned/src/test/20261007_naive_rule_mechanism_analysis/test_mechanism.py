"""Small behavioral checks for the Task 9 scoring protocol."""

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("run_mechanism.py")
SPEC = importlib.util.spec_from_file_location("run_mechanism", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_midrank_does_not_award_all_tied_pool_first_place():
    scores = np.zeros((1, 13))
    ranks, tie_count = MODULE.midrank(scores)
    assert ranks.tolist() == [7.0]
    assert tie_count.tolist() == [12]


def test_midrank_half_credit_for_one_tied_distractor():
    scores = np.array([[2.0, 2.0, 1.0, 3.0]])
    ranks, tie_count = MODULE.midrank(scores)
    assert ranks.tolist() == [2.5]
    assert tie_count.tolist() == [1]


def test_l1_normalize_preserves_zero_rows_and_row_scale():
    weights = np.array([[0.0, 2.0, 2.0], [0.0, 0.0, 0.0]])
    actual = MODULE.l1_normalize(weights)
    np.testing.assert_array_equal(actual, [[0.0, 0.5, 0.5], [0.0, 0.0, 0.0]])


def test_role_outrank_uses_each_four_candidate_block():
    scores = np.array([[2.0, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1],
                       [2.0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    assert MODULE.role_outrank(scores) == {
        "hard_negative": 0.5, "condition_only": 0.5, "anchor_only": 0.0
    }


def test_cluster_ablation_removes_target_and_highly_correlated_copies():
    naive = np.array([[1.0, 2.0, 3.0]])
    correlation = np.array([[1, .97, .1], [.97, 1, .2], [.1, .2, 1]])
    result = MODULE.target_cluster_ablations(naive, np.array([0]), correlation, .95)
    np.testing.assert_array_equal(result["minus_cluster"], [[0, 0, 3]])
    np.testing.assert_array_equal(result["cluster_only"], [[1, 2, 0]])
