"""Unit test for score_snapshot in run_baseline_seed_snapshot.py. Synthetic
snapshot only -- no real data, GPU, or pilot modules are touched.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import run_baseline_seed_snapshot as m


def _snapshot(n_train=200, n_heldout=80):
    rng = np.random.default_rng(0)

    def split(n):
        labels = np.repeat([0, 1], n // 2).astype(np.int32)
        centers = np.array([[20.0, 0.0], [0.0, 20.0]], dtype=np.float32)
        return (centers[labels] + rng.normal(scale=0.1, size=(n, 2))).astype(np.float32), labels

    train_embedding, train_labels = split(n_train)
    heldout_embedding, heldout_labels = split(n_heldout)
    names = np.array(["awe", "fear"], dtype=object)
    genres = np.array(["landscape", "portrait"], dtype=object)
    return {
        "train_embedding_post": train_embedding, "train_community_post": train_labels,
        "heldout_embedding_post": heldout_embedding, "heldout_community_post": heldout_labels,
        "heldout_emotion": names[heldout_labels], "heldout_genre": genres[heldout_labels],
    }


def test_score_snapshot_perfect_separation():
    out = m.score_snapshot(_snapshot())

    assert out["independent"]["emotion"] == pytest.approx(1.0)
    assert out["independent"]["genre"] == pytest.approx(1.0)
    assert out["independent"]["k"] == 2 and out["independent"]["train_k"] == 2
    assert set(out["transfer"]) == {"10", "20", "30", "40"}
    for entry in out["transfer"].values():
        assert entry["emotion"] == pytest.approx(1.0)
        assert entry["genre"] == pytest.approx(1.0)


def test_score_snapshot_merge_path_runs():
    out = m.score_snapshot(_snapshot())

    merged = out["merged_transfer"]
    assert merged["k_after_merge"] == 2
    assert set(merged["by_k"]) == {"20", "40"}
    for entry in merged["by_k"].values():
        assert entry["emotion"] == pytest.approx(1.0)
