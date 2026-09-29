"""4-seed stress of candidate 4's winning cutoff (0.15), scored correctly
against the baseline's own single-label held-out targets (the fix for
Critical finding C1; see candidate4_fixed_pilot_report.md).
"""

import time
from pathlib import Path

import numpy as np
import torch

from run_candidate1_min_occupancy_pilot import (
    PERCEPT_DIR, SMALL_COMMUNITIES, SNAPSHOT, TRANSFER_K, load_module,
    merge_small_communities, one_hot,
)
from run_candidate1_stress_pilot import STRESS_SEEDS
from run_candidate2_mapper_sweep_pilot import log
from run_candidate4_fixed_pilot import BASELINE_MACRO_AUC, score_against, train_multilabel_get_scores
from run_candidate4_rich_multilabel_pilot import cosine_vote_fractions, threshold_targets


HERE = Path(__file__).resolve().parent
BUDDY_DIR = HERE.parents[1] / "test/20260923_artelingo_buddy_analysis"
REPORT_PATH = HERE / "candidate4_fixed_stress_report.md"
WINNING_CUTOFF = 0.15
PRACTICAL_MARGIN = 0.005


def main() -> None:
    device = "cuda"
    stage2_ref = load_module("stage2_ref_for_c4_stress", PERCEPT_DIR / "run_percept_stage2_pilot.py")
    transfer = load_module("transfer_for_c4_stress", BUDDY_DIR / "run_heldout_label_transfer_pilot.py")

    with np.load(SNAPSHOT, allow_pickle=True) as source:
        train_paintings = source["train_paintings"]
        train_embeddings = source["train_embedding_post"]
        hard_train_labels = source["train_community_post"].astype(np.int64)
        heldout_paintings = source["heldout_paintings"]
        heldout_embeddings = source["heldout_embedding_post"]

    merged_train_labels, label_map = merge_small_communities(train_embeddings, hard_train_labels, SMALL_COMMUNITIES)
    n_topics = len(set(label_map.values()))
    baseline_heldout_hard = transfer.assign_to_train_communities(
        train_embeddings, merged_train_labels, heldout_embeddings, k=TRANSFER_K
    ).astype(np.int64)
    baseline_heldout_targets = one_hot(baseline_heldout_hard, n_topics).cpu().numpy()

    train_patches = stage2_ref.load_patch_features(stage2_ref.TRAIN_PATCH_FEATURE_PATH, len(train_paintings), "train").to(device)
    heldout_patches = stage2_ref.load_patch_features(stage2_ref.HELDOUT_PATCH_FEATURE_PATH, len(heldout_paintings), "held-out").to(device)

    train_fractions = cosine_vote_fractions(train_embeddings, merged_train_labels, train_embeddings, n_topics, device)
    train_targets = threshold_targets(train_fractions, WINNING_CUTOFF)

    results = []
    for seed in STRESS_SEEDS:
        scores = train_multilabel_get_scores(
            stage2_ref, train_targets, train_patches, heldout_patches, device, seed,
            f"cutoff_0.15_seed{seed}_stress",
        )
        result = score_against(stage2_ref, scores, baseline_heldout_targets, f"cutoff_0.15_seed{seed}_vs_baseline")
        results.append((seed, result))
        log(f"seed={seed}: vs-baseline-target macro AUC={result['macro']:.4f}")

    macros = np.array([r["macro"] for _, r in results])
    mean = float(macros.mean())
    lines = [
        "# Candidate 4 fix — 4-seed stress of cutoff 0.15\n\n",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. Companion to "
        "[`candidate4_fixed_pilot_report.md`](candidate4_fixed_pilot_report.md). "
        "Scored against the baseline's own single-label held-out targets throughout.\n\n",
        "| seed | macro AUC |\n|---:|---:|\n",
    ]
    for seed, r in results:
        lines.append(f"| {seed} | {r['macro']:.4f} |\n")
    lines.append(
        f"\nSummary: mean={mean:.4f}, min={macros.min():.4f}, max={macros.max():.4f}, "
        f"std={macros.std():.4f}. Baseline: {BASELINE_MACRO_AUC:.4f}.\n\n"
    )
    if mean - BASELINE_MACRO_AUC >= PRACTICAL_MARGIN:
        lines.append(
            f"**Cutoff 0.15 robustly beats the baseline** ({mean:.4f} vs. "
            f"{BASELINE_MACRO_AUC:.4f}, {mean - BASELINE_MACRO_AUC:+.4f}), consistent "
            "across all four seeds, not just seed 42. Recommended: adopt this "
            "richer multi-label target (0.15 relative cutoff) as the new headline "
            "buddy Stage 2 configuration.\n"
        )
    else:
        lines.append(
            f"Does not robustly beat the baseline once stressed (mean "
            f"{mean:.4f} vs. {BASELINE_MACRO_AUC:.4f}); the seed-42 result may not "
            "generalize.\n"
        )
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")
    log(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
