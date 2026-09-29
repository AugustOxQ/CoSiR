"""Small fixtures for the Task 6 factor-level diagnostic."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_diagnostic import (
    CORRELATION_METRICS, TASK5_ROWS, compute_factor_statistics,
    correlate_with_task5, factor_pools_via_mining,
)
from src.train.episodes import EpisodeMiningConfig


def test_exhaustive_mining_recovers_tied_boundary_pools():
    # At the 90th/50th percentiles, factor 0 has high indices 18-19 and
    # low indices 0-9. Factor 1 uses the same values in reverse order.
    column = np.array([0] * 10 + [1] * 8 + [2] * 2, dtype=np.float32)
    codes = np.stack((column, column[::-1]), axis=1)
    pools = factor_pools_via_mining(codes, codes, EpisodeMiningConfig(seed=42, min_pool_size=1))
    assert set(pools[0][0]) == {18, 19}
    assert set(pools[0][1]) == set(range(10))
    assert set(pools[1][0]) == {0, 1}
    assert set(pools[1][1]) == set(range(10, 20))


def test_statistics_use_four_item_mean_noise_and_pool_effect_size():
    # Population high values 2,10; low values 0,0. A four-draw high mean
    # falls below the midpoint (3) only when all four draws equal 2.
    codes = np.array([[0.0], [0.0], [2.0], [10.0]], dtype=np.float32)
    row = compute_factor_statistics(codes, [(np.array([2, 3]), np.array([0, 1]))],
                                    num_support=4, num_draws=1000, seed=42)[0]
    assert row["high_mean"] == pytest.approx(6.0)
    assert row["low_mean"] == pytest.approx(0.0)
    assert row["gap"] == pytest.approx(6.0)
    assert row["high_std"] == pytest.approx(4.0)
    assert row["low_std"] == pytest.approx(0.0)
    assert row["pooled_std"] == pytest.approx(np.sqrt(8.0))
    assert row["cohens_d"] == pytest.approx(6.0 / np.sqrt(8.0))
    assert row["snr"] == pytest.approx(3.0)
    assert 0.04 < row["ambiguous_fraction"] < 0.09


def test_task5_transcription_has_all_held_out_episodes_and_correct_predictions():
    assert len(TASK5_ROWS) == 32
    assert sum(count for count, _ in TASK5_ROWS) == 820
    assert sum(correct for _, correct in TASK5_ROWS) == 189
    assert {factor for factor, (_, correct) in enumerate(TASK5_ROWS) if correct} == {
        3, 7, 10, 12, 13, 16, 19, 21, 22, 25, 27, 28, 29, 30
    }


def test_constant_empirical_fraction_has_undefined_spearman_result():
    rows = [{metric: float(factor) for metric in CORRELATION_METRICS} for factor in range(32)]
    for row in rows:
        row["ambiguous_fraction"] = 0.0
    correlations = correlate_with_task5(rows)
    assert correlations["ambiguous_fraction"] == {"rho": None, "p_value": None}
