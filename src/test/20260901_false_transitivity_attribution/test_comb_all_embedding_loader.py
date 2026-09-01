"""Regression coverage for Experiment 15.2's checkpoint embedding alignment.

Run:
  python src/test/20260901_false_transitivity_attribution/test_comb_all_embedding_loader.py
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, SCRIPTS)

from analyze_polysemy_bridges import (
    _align_embeddings_to_graph_order,
    _assert_checkpoint_feature_store_matches_graph,
)


def test_aligns_snapshot_embeddings_by_sample_id_not_row_position():
    """A snapshot in another valid ID order must map back to graph node order."""
    snapshot_embeddings = np.array([[30.0], [10.0], [20.0]], dtype=np.float32)
    snapshot_ids = [30, 10, 20]
    graph_ids = [10, 20, 30]

    aligned = _align_embeddings_to_graph_order(snapshot_embeddings, snapshot_ids, graph_ids)

    np.testing.assert_array_equal(aligned[:, 0], [10.0, 20.0, 30.0])
    print("PASS test_aligns_snapshot_embeddings_by_sample_id_not_row_position")


def test_rejects_duplicate_snapshot_ids():
    embeddings = np.zeros((3, 2), dtype=np.float32)
    try:
        _align_embeddings_to_graph_order(embeddings, [10, 10, 30], [10, 20, 30])
    except AssertionError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate snapshot IDs must be rejected")
    print("PASS test_rejects_duplicate_snapshot_ids")


def test_rejects_snapshot_and_graph_id_set_mismatch():
    embeddings = np.zeros((3, 2), dtype=np.float32)
    try:
        _align_embeddings_to_graph_order(embeddings, [10, 20, 30], [10, 20, 40])
    except AssertionError as exc:
        assert "exactly the same sample IDs" in str(exc)
    else:
        raise AssertionError("missing graph sample IDs must be rejected")
    print("PASS test_rejects_snapshot_and_graph_id_set_mismatch")


def test_rejects_checkpoint_from_different_feature_store():
    try:
        _assert_checkpoint_feature_store_matches_graph(
            "/features/run_a", "/features/graph_b",
        )
    except AssertionError as exc:
        assert "feature store" in str(exc)
    else:
        raise AssertionError("checkpoint and graph stores must match")
    print("PASS test_rejects_checkpoint_from_different_feature_store")


if __name__ == "__main__":
    test_aligns_snapshot_embeddings_by_sample_id_not_row_position()
    test_rejects_duplicate_snapshot_ids()
    test_rejects_snapshot_and_graph_id_set_mismatch()
    test_rejects_checkpoint_from_different_feature_store()
    print("ALL TESTS PASSED")
