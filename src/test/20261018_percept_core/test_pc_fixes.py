"""Contract tests for the four PercepT core fixes and dimension maps."""
import numpy as np
import pytest
from sklearn.decomposition import PCA

import pc_fixes


@pytest.mark.parametrize('partition, expected', [
    ('fit', [0, 3, 4, 5, 7, 9, 10, 11]),
    ('dev', [1, 6]),
    ('eval', [2, 8]),
])
def test_pair_rows_keep_every_caption_in_frozen_order(partition, expected):
    pid = np.array([3, 0, 2, 3, 1, 4, 0, 3, 2, 4, 1, 3], dtype=np.int64)
    painting_partition = np.array(['dev', 'fit', 'eval', 'fit', 'fit'])
    rows = pc_fixes.pair_rows(pid, painting_partition, partition)
    np.testing.assert_array_equal(rows, expected)
    assert rows.dtype == np.int64
    assert len(rows) == sum(painting_partition[p] == partition for p in pid)


def test_eq2_matches_hand_worked_three_dimensional_pairs():
    image = np.array([[3, 0, 0], [0, 0, 2], [3, 0, 0]], dtype=np.float32)
    text = np.array([[0, 4, 0], [0, 0, 7], [-4, 0, 0]], dtype=np.float32)
    emotion = np.array([[0, 0, 5], [9, 0, 0], [0, 5, 0]], dtype=np.float32)
    # First C=(1/2,1/2,0), ||C||=sqrt(2)/2; second C=(0,0,1).
    expected = np.array([[1/3, 1/3, np.sqrt(2)/6], [1/3, 0, 2/3], [0, 0, 0]])
    fused = pc_fixes.fuse_eq2(image, text, emotion)
    np.testing.assert_allclose(fused, expected, rtol=1e-6, atol=1e-7)
    assert fused.shape == (3, 3) and fused.dtype == np.float32
    np.testing.assert_array_equal(image, [[3, 0, 0], [0, 0, 2], [3, 0, 0]])
    np.testing.assert_array_equal(text, [[0, 4, 0], [0, 0, 7], [-4, 0, 0]])
    np.testing.assert_array_equal(emotion, [[0, 0, 5], [9, 0, 0], [0, 5, 0]])


@pytest.mark.parametrize('n_fit, width, n_components', [(50, 9, 6), (520, 768, 512)])
def test_pca_fit_map_matches_sklearn_full_svd_on_all_captions(n_fit, width, n_components):
    rng = np.random.default_rng(17)
    fit = rng.normal(size=(n_fit, width)).astype(np.float32) + 3
    held_out = rng.normal(size=(7, width)).astype(np.float32) - 20
    all_captions = np.concatenate((fit, held_out))
    original_fit = fit.copy()
    mean, projection = pc_fixes.pca_map(fit, n_components)
    mapped = (all_captions - mean) @ projection
    reference = PCA(n_components=n_components, svd_solver='full').fit(fit.astype(np.float64))
    signs = np.sign(np.sum(projection.T * reference.components_, axis=1))
    np.testing.assert_allclose(mapped * signs, reference.transform(all_captions),
                               atol=3e-4, rtol=3e-4)
    np.testing.assert_allclose(mean, reference.mean_, atol=3e-6)
    assert mean.shape == (width,) and projection.shape == (width, n_components)
    assert mapped.shape == (n_fit + 7, n_components)
    assert mean.dtype == projection.dtype == mapped.dtype == np.float32
    np.testing.assert_array_equal(fit, original_fit)


def test_orthogonal_map_has_orthonormal_columns_and_preserves_norms():
    projection = pc_fixes.orthogonal_map(768, 2048, seed=0)
    assert projection.shape == (2048, 768) and projection.dtype == np.float32
    np.testing.assert_allclose(projection.T @ projection, np.eye(768), atol=1e-6)
    emotion = np.random.default_rng(4).normal(size=(13, 768)).astype(np.float32)
    mapped = emotion @ projection.T
    assert mapped.shape == (13, 2048) and mapped.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(mapped, axis=1),
                               np.linalg.norm(emotion, axis=1), rtol=1e-6)
    np.testing.assert_array_equal(pc_fixes.orthogonal_map(5, 9),
                                  pc_fixes.orthogonal_map(5, 9, seed=0))
    assert not np.array_equal(pc_fixes.orthogonal_map(5, 9, seed=0),
                              pc_fixes.orthogonal_map(5, 9, seed=1))


@pytest.mark.parametrize('norms, gamma, expected', [
    ([10, .3, .1, 11, .2], 2, [False, True, True, False, True]),
    ([3, 0, 2, 1], 2, [True] * 4),  # Uniform gaps: no qualifying gap.
    ([2, 2, 2, 2], 2, [True] * 4),  # Zero gaps: no qualifying gap.
    ([4, 0, 1], 1.5, [True] * 3),  # Largest gap equals gamma * mean.
    ([8, 2, 0, 4, 1, 3], 1.5, [False, True, True, True, True, True]),
    ([8, 2, 0, 4, 1, 3], 2, [False, True, True, True, True, True]),
    ([8, 2, 0, 4, 1, 3], 3, [True] * 6),
    ([8, 2, 0, 4, 1, 3], 5, [True] * 6),
    ([10, 1, 6, 0, 5], 1.5, [False, True, False, True, False]),  # First tied maximum.
])
def test_algorithm1_gap_cut_keeps_expected_original_centers(norms, gamma, expected):
    norms = np.array(norms, dtype=np.float32)
    kept = pc_fixes.kept_centers(norms, gamma=gamma)
    np.testing.assert_array_equal(kept, expected)
    assert kept.dtype == np.bool_ and kept.shape == norms.shape
    assert kept.sum() == sum(expected)
    if gamma == 2:
        np.testing.assert_array_equal(pc_fixes.kept_centers(norms), kept)


@pytest.mark.parametrize('counts,k,expected', [
    ([5] * 100, 40, list(range(40))),
    ([0] * 50 + [3, 7] + [0] * 48, 40, list(range(38)) + [50, 51]),
    ([2, 9, 5, 5, 8], 3, [1, 2, 4]),
])
def test_top_used_keeps_exactly_k_with_lower_index_ties(counts, k, expected):
    counts = np.array(counts, dtype=np.int64)
    before = counts.copy()
    kept = pc_fixes.top_used(counts) if k == 40 else pc_fixes.top_used(counts, k)
    np.testing.assert_array_equal(np.flatnonzero(kept), expected)
    assert kept.shape == counts.shape and kept.dtype == np.bool_ and kept.sum() == k
    np.testing.assert_array_equal(counts, before)


@pytest.mark.parametrize('counts,k', [
    (np.ones((2, 40), dtype=np.int64), 40),
    (np.ones(40, dtype=np.float32), 40),
    (np.array([-1] + [0] * 39), 40),
    (np.ones(39, dtype=np.int64), 40),
    (np.ones(40, dtype=np.int64), 0),
    (np.ones(40, dtype=np.int64), 1.5),
])
def test_top_used_asserts_invalid_counts_or_k(counts, k):
    with pytest.raises(AssertionError):
        pc_fixes.top_used(counts, k)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_multi_hot_matches_explicit_caption_or_in_requested_painting_order(seed):
    rng = np.random.default_rng(seed)
    pid = np.concatenate((np.arange(11), rng.integers(0, 11, size=35)))
    rng.shuffle(pid)
    topic = rng.integers(0, 6, size=len(pid), dtype=np.int16)
    paintings = np.array([7, 2, 10, 4, 0], dtype=np.int64)
    expected = np.zeros((len(paintings), 7), dtype=bool)
    for row, painting in enumerate(paintings):
        for caption_pid, caption_topic in zip(pid, topic):
            if caption_pid == painting:
                expected[row, caption_topic] |= True
    target = pc_fixes.multi_hot(pid, topic, paintings, n_topics=7)
    np.testing.assert_array_equal(target, expected)
    assert target.shape == (5, 7) and target.dtype == np.float32


def test_multi_hot_asserts_if_requested_painting_has_no_partition_caption():
    pid = np.array([4, 0, 4, 2], dtype=np.int64)
    topic = np.array([1, 0, 2, 1], dtype=np.int16)
    with pytest.raises(AssertionError, match='without captions'):
        pc_fixes.multi_hot(pid, topic, np.array([4, 7, 0]), n_topics=3)


def test_maps_have_fixed_signs():
    rng = np.random.default_rng(3)
    emotion = rng.normal(size=(300, 40)).astype(np.float32)
    _, projection = pc_fixes.pca_map(emotion, 10)
    largest = projection[np.abs(projection).argmax(axis=0), np.arange(10)]
    assert (largest > 0).all()
    matrix = np.random.default_rng(0).normal(size=(64, 16))
    q, r = np.linalg.qr(matrix, mode='reduced')
    expected = q * np.sign(np.diag(r))
    np.testing.assert_allclose(pc_fixes.orthogonal_map(16, 64, seed=0), expected, atol=1e-6)
    assert (np.diag(expected.T @ matrix) > 0).all()
