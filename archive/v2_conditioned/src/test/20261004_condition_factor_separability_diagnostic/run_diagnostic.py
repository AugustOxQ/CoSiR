"""Compare Task 5 recovery accuracy with real factor-code separability.

Run from the repository root in the CoSiR environment. Factor codes are
recreated with Task 3's prerequisite recipe; ConditionEncoder is not trained.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "20261001_condition_interface_validation"))

from run_validation import EXPECTED_SAMPLES, SEED, load_real_features, prepare_real_codes  # noqa: E402
from src.train.episodes import EpisodeMiningConfig, mine_episodes  # noqa: E402

# (held-out episodes, correct predictions), copied row by row from Task 5's
# 2026-10-03 report. The exact fractions, not rounded displayed percentages,
# are the correlation target. No recovery predictions are recomputed here.
TASK5_ROWS = (
    (24, 0), (32, 0), (23, 0), (21, 14), (27, 0), (29, 0), (26, 0), (28, 12),
    (25, 0), (25, 0), (21, 15), (27, 0), (19, 1), (24, 14), (24, 0), (23, 0),
    (23, 23), (24, 0), (27, 0), (19, 1), (26, 0), (27, 27), (26, 20), (24, 0),
    (29, 0), (29, 19), (26, 0), (27, 7), (24, 1), (28, 14), (25, 21), (38, 0),
)
CORRELATION_METRICS = ("gap", "cohens_d", "column_std", "snr", "ambiguous_fraction")


def factor_pools_via_mining(
    img_codes: np.ndarray, txt_codes: np.ndarray, config: EpisodeMiningConfig
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Recover full high/low pools through mine_episodes' own percentile logic.

    For each one-column factor view, request one exhaustive episode: anchor,
    support, and positive together enumerate the high pool, while contrast
    enumerates the low pool. The miner reserves one positive by construction.
    This avoids duplicating its tied-percentile boundary rule here.
    """
    if img_codes.ndim != 2 or img_codes.shape != txt_codes.shape:
        raise ValueError("Image and text codes must be matching (N, L) arrays")
    n_samples, n_factors = img_codes.shape
    exhaustive = replace(
        config, num_support=n_samples, num_contrast=n_samples,
        num_hard_negatives=0, num_condition_distractors=0,
        num_anchor_distractors=0,
    )
    pools = []
    for factor in range(n_factors):
        try:
            episode = mine_episodes(
                img_codes[:, factor:factor + 1], txt_codes[:, factor:factor + 1],
                exhaustive, num_episodes=1,
            )[0]
        except ValueError as exc:
            raise ValueError(f"Cannot recover percentile pools for factor {factor}") from exc
        high = np.sort(np.asarray(
            [episode.anchor_idx, *episode.support_idxs, episode.positive_idx], dtype=np.int64
        ))
        low = np.sort(np.asarray(episode.contrast_idxs, dtype=np.int64))
        if len(high) + len(low) > n_samples or np.intersect1d(high, low).size:
            raise AssertionError(f"Mining pools overlap for factor {factor}")
        pools.append((high, low))
    return pools


def compute_factor_statistics(
    pair_codes: np.ndarray, pools: list[tuple[np.ndarray, np.ndarray]],
    num_support: int, num_draws: int, seed: int,
) -> list[dict]:
    """Measure raw scale, normalized separation, and four-item mean noise."""
    if pair_codes.ndim != 2 or len(pools) != pair_codes.shape[1]:
        raise ValueError("Pool count must match factor-code columns")
    if num_support < 1 or num_draws < 1:
        raise ValueError("Sampling sizes must be positive")
    rng = np.random.default_rng(seed)
    rows = []
    for factor, (high_pool, low_pool) in enumerate(pools):
        column = pair_codes[:, factor].astype(np.float64)
        high_values = column[high_pool]
        low_values = column[low_pool]
        if not len(high_values) or not len(low_values):
            raise ValueError(f"Empty percentile pool for factor {factor}")
        high_mean, low_mean = float(high_values.mean()), float(low_values.mean())
        high_std, low_std = float(high_values.std()), float(low_values.std())
        gap = high_mean - low_mean
        pooled_std = float(np.sqrt((high_std ** 2 + low_std ** 2) / 2))
        if pooled_std == 0 or high_std == 0:
            raise ValueError(f"Zero within-pool variance for factor {factor}")
        support_means = rng.choice(high_values, size=(num_draws, num_support), replace=True).mean(axis=1)
        rows.append({
            "factor": factor, "high_size": len(high_pool), "low_size": len(low_pool),
            "high_mean": high_mean, "low_mean": low_mean, "gap": gap,
            "high_std": high_std, "low_std": low_std, "pooled_std": pooled_std,
            "cohens_d": gap / pooled_std,
            "column_mean": float(column.mean()), "column_std": float(column.std()),
            "column_max": float(column.max()),
            "snr": gap / (high_std / np.sqrt(num_support)),
            "ambiguous_fraction": float(np.mean(support_means < low_mean + 0.5 * gap)),
        })
    return rows


def correlate_with_task5(rows: list[dict]) -> dict[str, dict[str, float | None]]:
    """Use exact Task 5 correct/count fractions as fixed correlation targets."""
    if len(rows) != len(TASK5_ROWS):
        raise ValueError("Expected one diagnostic row for each of Task 5's 32 factors")
    accuracies = [correct / count for count, correct in TASK5_ROWS]
    result = {}
    for metric in CORRELATION_METRICS:
        values = [row[metric] for row in rows]
        if all(value == values[0] for value in values):
            result[metric] = {"rho": None, "p_value": None}
        else:
            correlation = spearmanr(values, accuracies)
            result[metric] = {"rho": float(correlation.statistic), "p_value": float(correlation.pvalue)}
    return result


def main() -> None:
    started = perf_counter()
    img_features, txt_features = load_real_features()
    img_codes, txt_codes, prerequisite = prepare_real_codes(img_features, txt_features)
    del img_features, txt_features
    if img_codes.shape != (EXPECTED_SAMPLES, len(TASK5_ROWS)):
        raise ValueError("Expected 308,723 samples and 32 factor columns")
    config = EpisodeMiningConfig(seed=SEED)
    pair_codes = 0.5 * (img_codes + txt_codes)
    pools = factor_pools_via_mining(img_codes, txt_codes, config)
    rows = compute_factor_statistics(pair_codes, pools, config.num_support, 1000, SEED)
    for row, (count, correct) in zip(rows, TASK5_ROWS):
        row.update({"held_out_episodes": count, "correct": correct, "accuracy": correct / count})
    result = {
        "samples": len(pair_codes), "factors": pair_codes.shape[1],
        "prerequisite": prerequisite, "mining_config": vars(config),
        "empirical_draws_per_factor": 1000, "empirical_sampling": "with replacement",
        "correlations": correlate_with_task5(rows),
        "rows": sorted(rows, key=lambda row: (-row["accuracy"], row["factor"])),
        "seconds": perf_counter() - started,
    }
    print("RESULT_JSON=" + json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
