import numpy as np

from scripts.buddy_percept_sweep.targets import (
    assign_to_train_communities, build_targets, cosine_vote_fractions, one_hot,
)


def _synthetic_train():
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=0.0, scale=0.05, size=(10, 6))
    cluster_b = rng.normal(loc=3.0, scale=0.05, size=(10, 6))
    embeddings = np.concatenate([cluster_a, cluster_b]).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    labels = np.array([0] * 10 + [1] * 10)
    return embeddings, labels


def test_assign_to_train_communities_matches_nearest_cluster():
    train_embeddings, train_labels = _synthetic_train()
    query = np.tile(train_embeddings[0], (3, 1))  # near cluster 0
    assigned = assign_to_train_communities(train_embeddings, train_labels, query, k=5)
    assert (assigned == 0).all()


def test_cosine_vote_fractions_sum_to_one_per_row():
    train_embeddings, train_labels = _synthetic_train()
    fractions = cosine_vote_fractions(train_embeddings, train_labels, train_embeddings[:3], n_topics=2, k=5)
    assert fractions.shape == (3, 2)
    np.testing.assert_allclose(fractions.sum(axis=1), 1.0, atol=1e-6)


def test_build_targets_single_label_is_one_hot():
    hard_labels = np.array([0, 1, 1])
    targets = build_targets(hard_labels, n_topics=2, target_cutoff="single_label")
    expected = one_hot(hard_labels, 2)
    np.testing.assert_array_equal(targets, expected)


def test_build_targets_numeric_cutoff_can_be_multi_label():
    fractions = np.array([[0.6, 0.5], [0.9, 0.1]])
    # cutoff=0.5 -> positive if fraction > 0.5 * row max
    targets = build_targets(fractions, n_topics=2, target_cutoff=0.5)
    assert targets[0].sum() == 2  # both topics within 0.5x of the row max (0.6)
    assert targets[1].tolist() == [1, 0]


def test_build_targets_handles_two_topic_edge_case_after_aggressive_merge():
    fractions = np.array([[1.0, 0.0], [0.0, 1.0]])
    targets = build_targets(fractions, n_topics=2, target_cutoff=0.15)
    assert targets.shape == (2, 2)
    assert np.isfinite(targets).all()
