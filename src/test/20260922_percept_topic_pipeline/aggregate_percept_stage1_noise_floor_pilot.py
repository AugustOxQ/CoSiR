"""Aggregate the PercepT Stage-1 seed-42 noise-floor repeats into a report.

Run once after all N invocations of run_percept_stage1_noise_floor_pilot.py
have appended their line to percept_stage1_noise_floor_results.jsonl.
"""

import json
import os
import time

import numpy as np


OUT_DIR = os.path.dirname(__file__)
REPORT_OUT_DIR = os.environ.get("PERCEPT_OUTPUT_ROOT") or OUT_DIR
RESULTS_PATH = os.path.join(
    REPORT_OUT_DIR, "percept_stage1_noise_floor_results.jsonl"
)
REPORT_PATH = os.path.join(
    REPORT_OUT_DIR, "percept_stage1_noise_floor_pilot_report.md"
)

# From the 14-seed Stage 1 extended-seed pilot
# (percept_stage1_extended_seed_pilot_report.md).
BETWEEN_SEED_EMOTION_STDEV = 0.0032
BETWEEN_SEED_GENRE_STDEV = 0.0156
# From percept_stage1_cluster_count_sweep_v2_pilot_report.md.
ESTABLISHED_EMOTION_AMI = 0.1238
ESTABLISHED_GENRE_AMI = 0.2617


def main() -> None:
    if not os.path.exists(RESULTS_PATH):
        raise RuntimeError(f"No results file found at {RESULTS_PATH}.")
    with open(RESULTS_PATH) as results_file:
        rows = [json.loads(line) for line in results_file if line.strip()]
    if len(rows) < 2:
        raise RuntimeError(
            f"Only {len(rows)} repeat(s) found in {RESULTS_PATH}; need at least 2 "
            "to measure a spread."
        )

    emotion_amis = np.array([row["emotion_ami"] for row in rows])
    genre_amis = np.array([row["genre_ami"] for row in rows])
    emotion_mean = float(np.mean(emotion_amis))
    genre_mean = float(np.mean(genre_amis))
    emotion_stdev = float(np.std(emotion_amis, ddof=1))
    genre_stdev = float(np.std(genre_amis, ddof=1))

    lines = [
        "# ArtELingo PercepT Stage 1 seed-42 noise-floor pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Purpose\n\n",
        "Measures run-to-run spread of the Stage-1 K=60/40 seed-42 refit under "
        "PyTorch's default (non-deterministic) cuDNN/cuBLAS kernels -- the same "
        "kernel behavior every prior \"established\"/\"cited\" value in this "
        "investigation was produced under. Each repeat is a fresh process "
        "invocation with identical seed 42, identical code, identical data; any "
        "spread between repeats is pure uncontrolled GPU nondeterminism, not "
        "seed sensitivity.\n\n",
        f"## Full {len(rows)}-repeat results\n\n",
        "| repeat | held-out emotion AMI | held-out genre AMI |\n",
        "|---:|---:|---:|\n",
    ]
    for row in sorted(rows, key=lambda r: r["repeat"]):
        lines.append(
            f"| {row['repeat']} | {row['emotion_ami']:.4f} | {row['genre_ami']:.4f} |\n"
        )

    lines.extend(
        [
            "\n## Spread statistics\n\n",
            "| metric | mean | min | max | sample standard deviation |\n",
            "|---|---:|---:|---:|---:|\n",
            f"| held-out emotion AMI | {emotion_mean:.4f} | "
            f"{np.min(emotion_amis):.4f} | {np.max(emotion_amis):.4f} | "
            f"{emotion_stdev:.4f} |\n",
            f"| held-out genre AMI | {genre_mean:.4f} | "
            f"{np.min(genre_amis):.4f} | {np.max(genre_amis):.4f} | "
            f"{genre_stdev:.4f} |\n\n",
        ]
    )

    if emotion_stdev > 0 and genre_stdev > 0:
        emotion_citation_z = (ESTABLISHED_EMOTION_AMI - emotion_mean) / emotion_stdev
        genre_citation_z = (ESTABLISHED_GENRE_AMI - genre_mean) / genre_stdev
        lines.append(
            "For reference, the original citation this investigation built on was "
            f"emotion={ESTABLISHED_EMOTION_AMI:.4f}, genre={ESTABLISHED_GENRE_AMI:.4f} "
            "(`percept_stage1_cluster_count_sweep_v2_pilot_report.md`); relative to "
            f"this pilot's own mean and standard deviation, that citation sits "
            f"{emotion_citation_z:+.2f} standard deviations away on emotion and "
            f"{genre_citation_z:+.2f} standard deviations away on genre.\n\n"
        )

    lines.extend(
        [
            "## Comparison against the between-seed spread\n\n",
            "The 14-seed Stage 1 extended-seed pilot "
            "(`percept_stage1_extended_seed_pilot_report.md`) measured a "
            f"between-*different*-seed standard deviation of "
            f"{BETWEEN_SEED_EMOTION_STDEV:.4f} (emotion) and "
            f"{BETWEEN_SEED_GENRE_STDEV:.4f} (genre). This pilot's within-*same*-seed "
            f"noise-floor standard deviation is {emotion_stdev:.4f} (emotion) and "
            f"{genre_stdev:.4f} (genre) across {len(rows)} repeats.\n\n",
        ]
    )

    if (
        genre_stdev >= BETWEEN_SEED_GENRE_STDEV
        or emotion_stdev >= BETWEEN_SEED_EMOTION_STDEV
    ):
        lines.append(
            "**Finding: the noise floor is comparable to or larger than the "
            "between-seed spread.** At least one axis's within-seed standard "
            "deviation from uncontrolled GPU nondeterminism meets or exceeds the "
            "between-seed standard deviation reported by the 14-seed extended-seed "
            "pilot. That pilot's \"seed-dependent result\" verdict cannot be "
            "cleanly attributed to genuine seed sensitivity alone -- uncontrolled "
            "run-to-run nondeterminism is a confound of comparable size, not a "
            "negligible side effect.\n"
        )
    else:
        lines.append(
            "**Finding: the noise floor is smaller than the between-seed spread "
            "on both axes.** Uncontrolled GPU nondeterminism, while nonzero, is "
            "not large enough on its own to explain the between-seed spread the "
            "14-seed extended-seed pilot reported; that pilot's \"seed-dependent "
            "result\" verdict is not primarily a nondeterminism artifact.\n"
        )

    os.makedirs(REPORT_OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
