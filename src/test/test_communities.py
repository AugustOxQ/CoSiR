"""Community detection on the trained student's output embeddings."""

import numpy as np
from sklearn.metrics import adjusted_rand_score

from src.model.communities import community_stats, detect_communities


def test_two_separated_blobs_form_two_aligned_communities():
    rng = np.random.default_rng(42)
    centers = np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
    truth = np.repeat(np.arange(2), 24)
    embeddings = centers[truth] + 0.03 * rng.standard_normal((48, 4))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    labels = detect_communities(embeddings)

    assert labels.shape == (48,)
    assert labels.dtype == np.int64
    assert len(np.unique(labels)) == 2
    assert adjusted_rand_score(truth, labels) == 1.0


def test_several_communities_have_zero_indexed_gapless_labels():
    rng = np.random.default_rng(123)
    centers = np.eye(4)
    truth = np.repeat(np.arange(4), 8)
    embeddings = centers[truth] + 0.01 * rng.standard_normal((32, 4))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    labels = detect_communities(embeddings, k=7)

    np.testing.assert_array_equal(np.unique(labels), np.arange(4))
    assert adjusted_rand_score(truth, labels) == 1.0


def test_seed_42_repeats_labels_exactly():
    rng = np.random.default_rng(7)
    embeddings = rng.standard_normal((80, 6))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    first = detect_communities(embeddings, k=8, seed=42)
    second = detect_communities(embeddings, k=8, seed=42)

    np.testing.assert_array_equal(first, second)


def test_community_stats_reports_sizes_and_no_empty_communities():
    labels = np.array([2, 0, 2, 1, 0, 2], dtype=np.int64)

    assert community_stats(labels) == {
        "num_communities": 3,
        "sizes": [2, 1, 3],
        "min_size": 1,
        "max_size": 3,
        "empty_count": 0,
    }
