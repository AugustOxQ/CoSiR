"""Small behavior checks for the CLIP-only evaluation row."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_clip_only_baseline import evaluate_clip_only_ranking, evaluate_clip_only_swaps  # noqa: E402
from src.train.episodes import Episode  # noqa: E402


def _fixture():
    features = np.tile(np.array([-1.0, 0.0], dtype=np.float32), (30, 1))
    features[0] = [1.0, 0.0]
    features[1] = [0.8, 0.6]
    features[2] = [1.0, 0.0]
    # Nonzero codes make an accidental factor contribution change the result.
    codes = np.arange(30 * 32, dtype=np.float32).reshape(30, 32) + 1
    episodes = [
        Episode(0, 0, [14, 15, 16, 17], [18, 19, 20, 21],
                1, [2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]),
        Episode(0, 1, [22, 23, 24, 25], [26, 27, 28, 29],
                2, [1, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]),
    ]
    return features, codes, episodes


def test_zero_weights_leave_only_clip_and_mark_beta_zero_tied():
    features, codes, episodes = _fixture()
    result = evaluate_clip_only_ranking(features, features, codes, codes, episodes)
    for direction in ("i2t", "t2i"):
        tied = result["0.0"][direction]
        assert tied["recall1"] is None and tied["recall3"] is None
        assert tied["all_tied"] is True
        for beta in ("0.03", "0.3"):
            measured = result[beta][direction]
            assert measured["ranks"] == [2, 1]
            assert measured["recall1"] == 0.5
            assert measured["recall3"] == 1.0
            assert measured["mean_abs_factor"] == 0.0


def test_condition_swap_cannot_reverse_identical_zero_weights():
    features, codes, episodes = _fixture()
    result = evaluate_clip_only_swaps(features, features, codes, codes, episodes, [(0, 1)])
    for beta in ("0.0", "0.03", "0.3"):
        for direction in ("i2t", "t2i"):
            measured = result[beta][direction]
            assert measured["appropriate_reversals"] == 0
            assert measured["reversal_rate"] == 0.0
            assert measured["any_rank_change"] == 0.0
