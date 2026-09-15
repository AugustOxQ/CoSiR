import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.conditional_buddy.prototype_seed import (
    coarsen_to_prototype_count,
    community_mean_features,
    detect_communities,
)


def test_detect_communities_two_disconnected_cliques():
    # nodes 0,1,2 fully connected; nodes 3,4,5 fully connected; no cross edges.
    rows = [0, 0, 1, 3, 3, 4]
    cols = [1, 2, 2, 4, 5, 5]
    all_rows = rows + cols
    all_cols = cols + rows
    E = csr_matrix((np.ones(len(all_rows)), (all_rows, all_cols)), shape=(6, 6))
    labels = detect_communities(E, seed=42)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_community_mean_features_matches_manual_average():
    labels = np.array([0, 0, 1])
    img = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    txt = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    means = community_mean_features(labels, img, txt)
    assert means.shape == (2, 2)
    np.testing.assert_allclose(means[0], [1.5, 0.0])
    np.testing.assert_allclose(means[1], [0.0, 1.0])


def test_community_mean_features_raises_on_missing_label():
    labels = np.array([0, 0, 2])  # label 1 never appears -> gap
    img = txt = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        community_mean_features(labels, img, txt)


def test_coarsen_to_prototype_count_noop_when_already_small():
    means = np.random.RandomState(42).randn(3, 4).astype(np.float32)
    out = coarsen_to_prototype_count(means, num_prototypes=8)
    np.testing.assert_array_equal(out, means)


def test_coarsen_to_prototype_count_reduces_row_count():
    means = np.random.RandomState(42).randn(20, 4).astype(np.float32)
    out = coarsen_to_prototype_count(means, num_prototypes=5, seed=42)
    assert out.shape == (5, 4)
