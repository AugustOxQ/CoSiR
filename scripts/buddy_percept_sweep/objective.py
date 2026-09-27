"""Objective computation and Pareto-bar gating for the buddy-percept sweep.

Gate thresholds match the standing Pareto bar used throughout the
2026-09-26 investigation (emotion AMI > 0.1236, genre AMI > 0.1954).
"""

GATE_FAIL_SENTINEL = -1.0


def compute_objective(
    emotion_ami: float,
    genre_ami: float,
    stage2_macro_auc: float,
    emotion_bar: float = 0.1236,
    genre_bar: float = 0.1954,
) -> float:
    if emotion_ami > emotion_bar and genre_ami > genre_bar:
        return stage2_macro_auc
    return GATE_FAIL_SENTINEL
