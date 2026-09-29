import numpy as np
import pytest

from src.data.splits import GroupedSplit, grouped_split, leakage_groups, split_leakage


def _features(n, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 8)).astype(np.float32)


def test_rows_sharing_an_image_vector_join_one_group_across_paintings():
    features = _features(4)
    features[3] = features[0]                      # different painting, identical image vector
    paintings = np.array(["a", "a", "b", "c"])
    groups = leakage_groups(paintings, features)
    assert groups[0] == groups[1] == groups[3] and groups[2] != groups[0]


def test_grouped_split_never_splits_a_group_and_hits_fractions():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(2000), rng.integers(1, 9, size=2000))
    split = grouped_split(groups, (0.7, 0.1, 0.2), seed=42)
    parts = [set(groups[idx]) for idx in (split.train, split.val, split.held)]
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
    assert len(split.train) + len(split.val) + len(split.held) == len(groups)
    shares = np.array([len(split.train), len(split.val), len(split.held)]) / len(groups)
    assert np.allclose(shares, (0.7, 0.1, 0.2), atol=0.01)


def test_grouped_split_is_deterministic_and_seed_sensitive():
    groups = np.repeat(np.arange(500), 3)
    a, b = grouped_split(groups, seed=42), grouped_split(groups, seed=42)
    assert np.array_equal(a.held, b.held)
    assert not np.array_equal(a.held, grouped_split(groups, seed=43).held)


def test_grouped_split_rejects_bad_fractions_and_empty_parts():
    with pytest.raises(ValueError):
        grouped_split(np.arange(10), (0.5, 0.5))
    with pytest.raises(ValueError):
        grouped_split(np.arange(10), (0.7, 0.2, 0.2))
    with pytest.raises(ValueError):
        grouped_split(np.zeros(10, dtype=int), (0.7, 0.1, 0.2))   # one group -> empty parts


def test_split_leakage_is_zero_for_grouped_split_and_nonzero_for_row_split():
    features = _features(300)
    paintings = np.array([f"p{i // 3}" for i in range(300)])
    features[1::3] = features[0::3]                # each painting's rows share one image vector
    split = grouped_split(leakage_groups(paintings, features), seed=42)
    assert all(value == 0 for value in split_leakage(split, paintings, features).values())
    perm = np.random.default_rng(0).permutation(300)
    row_split = GroupedSplit(np.sort(perm[:210]), np.sort(perm[210:240]), np.sort(perm[240:]))
    assert split_leakage(row_split, paintings, features)["held_rows_painting_in_train"] > 0
