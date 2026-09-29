"""Behavior tests for mining distinct factor-conditioned episode roles."""

from itertools import combinations

import numpy as np
import pytest

from src.train.episodes import EpisodeMiningConfig, mine_episodes


def _assert_disjoint_roles(episode):
    roles = [
        {episode.anchor_idx},
        set(episode.support_idxs),
        set(episode.contrast_idxs),
        {episode.positive_idx},
        set(episode.hard_negative_idxs),
        set(episode.condition_distractor_idxs),
        set(episode.anchor_distractor_idxs),
    ]
    for role in (
        episode.support_idxs,
        episode.contrast_idxs,
        episode.hard_negative_idxs,
        episode.condition_distractor_idxs,
        episode.anchor_distractor_idxs,
    ):
        assert len(role) == len(set(role))
    for first, second in combinations(roles, 2):
        assert first.isdisjoint(second)


def _structured_codes():
    """Factor 0 has a 20/40 high/low split; other factors have thin tails."""
    codes = np.zeros((60, 3), dtype=np.float64)
    codes[:20, 0] = 10.0
    for index in range(60):
        if index % 20 < 10:
            codes[index, 1:] = (1.0 + index * 1e-5, 0.01 + index * 1e-6)
        else:
            codes[index, 1:] = (0.01 + index * 1e-6, 1.0 + index * 1e-5)
    return codes


def _cosine(left, right):
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def test_no_index_appears_in_two_roles_across_mined_episodes():
    rng = np.random.default_rng(123)
    codes = rng.random((200, 8))
    codes[:30, 0] += 5.0
    episodes = mine_episodes(
        codes, codes, EpisodeMiningConfig(min_pool_size=10), num_episodes=100
    )

    assert len(episodes) == 100
    for episode in episodes:
        _assert_disjoint_roles(episode)


def test_near_dead_factor_is_skipped_across_many_episodes():
    rng = np.random.default_rng(123)
    codes = rng.random((200, 8))
    codes[:, 0] = 0.0
    codes[0, 0] = 10.0
    episodes = mine_episodes(
        codes, codes, EpisodeMiningConfig(min_pool_size=10), num_episodes=200
    )

    assert len(episodes) == 200
    assert {episode.targeted_factor for episode in episodes}.issubset(set(range(1, 8)))
    assert all(episode.targeted_factor != 0 for episode in episodes)


def test_hard_negatives_are_low_on_condition_and_match_positive_elsewhere():
    codes = _structured_codes()
    config = EpisodeMiningConfig(
        num_support=0,
        num_contrast=1,
        num_hard_negatives=3,
        num_condition_distractors=3,
        num_anchor_distractors=3,
        min_pool_size=10,
    )
    episode = mine_episodes(codes, codes, config, num_episodes=1)[0]

    assert episode.targeted_factor == 0
    assert len(episode.hard_negative_idxs) == 3
    assert codes[episode.positive_idx, 0] == 10.0
    for index in episode.hard_negative_idxs:
        assert codes[index, 0] == 0.0
        assert _cosine(codes[index, 1:], codes[episode.positive_idx, 1:]) > 0.99


def test_condition_and_anchor_distractors_have_opposite_other_factor_matches():
    codes = _structured_codes()
    config = EpisodeMiningConfig(
        num_support=0,
        num_contrast=1,
        num_hard_negatives=3,
        num_condition_distractors=3,
        num_anchor_distractors=3,
        min_pool_size=10,
    )
    episode = mine_episodes(codes, codes, config, num_episodes=1)[0]

    assert episode.targeted_factor == 0
    assert len(episode.condition_distractor_idxs) == 3
    assert len(episode.anchor_distractor_idxs) == 3
    anchor_other = codes[episode.anchor_idx, 1:]
    for index in episode.condition_distractor_idxs:
        assert codes[index, 0] == 10.0
        assert _cosine(codes[index, 1:], anchor_other) < 0.1
    for index in episode.anchor_distractor_idxs:
        assert codes[index, 0] == 0.0
        assert _cosine(codes[index, 1:], anchor_other) > 0.99


def test_hard_negatives_match_positive_while_anchor_only_matches_anchor():
    codes = _structured_codes()
    config = EpisodeMiningConfig(
        num_support=0,
        num_contrast=1,
        num_hard_negatives=2,
        num_condition_distractors=0,
        num_anchor_distractors=2,
        min_pool_size=10,
    )
    episodes = mine_episodes(codes, codes, config, num_episodes=20)
    distinct = [
        episode for episode in episodes
        if _cosine(codes[episode.anchor_idx, 1:], codes[episode.positive_idx, 1:]) < 0.1
    ]

    assert distinct  # The two reference profiles are observably different.
    for episode in distinct:
        for index in episode.hard_negative_idxs:
            assert _cosine(codes[index, 1:], codes[episode.positive_idx, 1:]) > 0.99
            assert _cosine(codes[index, 1:], codes[episode.anchor_idx, 1:]) < 0.1
        for index in episode.anchor_distractor_idxs:
            assert _cosine(codes[index, 1:], codes[episode.anchor_idx, 1:]) > 0.99
            assert _cosine(codes[index, 1:], codes[episode.positive_idx, 1:]) < 0.1


def test_pool_shortfalls_reduce_role_counts_without_reusing_items():
    codes = np.zeros((20, 2), dtype=np.float64)
    codes[:4, 0] = 10.0
    codes[:, 1] = np.linspace(0.01, 1.0, 20)
    config = EpisodeMiningConfig(
        num_support=10,
        num_contrast=1,
        num_hard_negatives=2,
        num_condition_distractors=10,
        num_anchor_distractors=2,
        min_pool_size=3,
    )
    episode = mine_episodes(codes, codes, config, num_episodes=1)[0]

    assert episode.targeted_factor == 0
    assert len(episode.support_idxs) == 2  # Reserve anchor and positive from four highs.
    assert episode.positive_idx < 4
    assert episode.condition_distractor_idxs == []
    _assert_disjoint_roles(episode)


def test_all_near_dead_factors_raise_clear_error():
    codes = np.zeros((20, 2), dtype=np.float64)
    codes[0] = (1.0, 1.0)

    with pytest.raises(ValueError, match="valid factor|pool"):
        mine_episodes(codes, codes, EpisodeMiningConfig(min_pool_size=3), 1)
