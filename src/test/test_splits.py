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


def test_cross_painting_image_duplicates_share_a_part_and_image_leakage_is_detectable():
    """Test that image duplicates across painting boundaries are detected and grouped correctly.

    This test:
    1. Creates 100 paintings × 3 rows = 300 rows
    2. Makes pairs of different paintings share an identical image vector
    3. Asserts grouped_split keeps paired rows in the same part (zero leakage)
    4. Asserts painting-only grouping OR random splits show nonzero image leakage
    """
    # Create 100 paintings × 3 rows = 300 rows
    features = _features(300, seed=42)
    paintings = np.array([f"p{i // 3}" for i in range(300)])

    # Make pairs of consecutive paintings share image vectors:
    # p0's row 0 → p1's row 0, p2's row 0 → p3's row 0, etc.
    # This creates 50 pairs of different paintings with identical images
    for i in range(0, 300, 6):
        if i + 3 < 300:
            features[i + 3] = features[i]  # painting at i → painting at i+3

    # Test 1: grouped_split with leakage_groups gives zero leakage
    groups = leakage_groups(paintings, features)
    split = grouped_split(groups, fractions=(0.7, 0.1, 0.2), seed=42)
    leakage = split_leakage(split, paintings, features)
    assert all(value == 0 for value in leakage.values()), f"Grouped split leakage: {leakage}"

    # Test 2: Paired rows with same image land in the same part
    for i in range(0, 300, 6):
        if i + 3 < 300:
            part_i = None
            part_i3 = None
            for part_name, indices in [("train", split.train), ("val", split.val), ("held", split.held)]:
                if i in indices:
                    part_i = part_name
                if i + 3 in indices:
                    part_i3 = part_name
            assert part_i == part_i3, f"Rows {i} and {i+3} (different paintings, same image) landed in different parts"

    # Test 3: Painting-only grouping shows nonzero image leakage
    painting_groups = np.unique(paintings, return_inverse=True)[1]
    painting_split = grouped_split(painting_groups, fractions=(0.7, 0.1, 0.2), seed=42)
    painting_leakage = split_leakage(painting_split, paintings, features)
    assert painting_leakage["val_rows_image_in_train"] > 0 or painting_leakage["held_rows_image_in_train"] > 0, \
        "Painting-only split should show image leakage from cross-painting duplicates"

    # Test 4: Random row split also shows nonzero image leakage
    perm = np.random.default_rng(0).permutation(300)
    row_split = GroupedSplit(np.sort(perm[:210]), np.sort(perm[210:240]), np.sort(perm[240:]))
    row_leakage = split_leakage(row_split, paintings, features)
    assert row_leakage["val_rows_image_in_train"] > 0 or row_leakage["held_rows_image_in_train"] > 0, \
        "Random row split should show image leakage from duplicates"
