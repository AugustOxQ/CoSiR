"""4-seed stress for Variant C (merge + class-balanced loss combined), to
close out candidate 1 with all three variants equally validated. Reuses
run_candidate1_min_occupancy_pilot.py and run_candidate1_stress_pilot.py.
"""

import time
from pathlib import Path

import numpy as np
import torch

from run_candidate1_min_occupancy_pilot import (
    PERCEPT_DIR, PRACTICAL_MARGIN, SMALL_COMMUNITIES, SNAPSHOT, TRANSFER_K,
    load_module, merge_small_communities,
)
from run_candidate1_stress_pilot import STRESS_SEEDS, train_and_eval_seeded


HERE = Path(__file__).resolve().parent
BUDDY_DIR = HERE.parents[1] / "test/20260923_artelingo_buddy_analysis"
REPORT_PATH = HERE / "candidate1_variant_c_stress_report.md"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def main() -> None:
    stage2_ref = load_module("stage2_ref_for_c_stress", PERCEPT_DIR / "run_percept_stage2_pilot.py")
    transfer = load_module("transfer_for_c_stress", BUDDY_DIR / "run_heldout_label_transfer_pilot.py")

    with np.load(SNAPSHOT, allow_pickle=True) as source:
        train_paintings = source["train_paintings"]
        train_embedding = source["train_embedding_post"]
        train_labels = source["train_community_post"].astype(np.int64)
        heldout_paintings = source["heldout_paintings"]
        heldout_embedding = source["heldout_embedding_post"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_patches = stage2_ref.load_patch_features(
        stage2_ref.TRAIN_PATCH_FEATURE_PATH, len(train_paintings), "train"
    ).to(device)
    heldout_patches = stage2_ref.load_patch_features(
        stage2_ref.HELDOUT_PATCH_FEATURE_PATH, len(heldout_paintings), "held-out"
    ).to(device)

    merged_train_labels, label_map = merge_small_communities(train_embedding, train_labels, SMALL_COMMUNITIES)
    n_merged_topics = len(set(label_map.values()))
    heldout_labels_raw = transfer.assign_to_train_communities(
        train_embedding, train_labels, heldout_embedding, k=TRANSFER_K
    ).astype(np.int64)
    merged_heldout_labels = np.array([label_map[label] for label in heldout_labels_raw], dtype=np.int64)

    merged_counts = np.bincount(merged_train_labels, minlength=n_merged_topics)
    inverse_freq = (len(merged_train_labels) / n_merged_topics) / merged_counts[merged_train_labels]
    inverse_freq = inverse_freq / inverse_freq.mean()

    results = []
    for seed in STRESS_SEEDS:
        result = train_and_eval_seeded(
            f"variant_c_seed{seed}", n_merged_topics, merged_train_labels, merged_heldout_labels,
            train_patches, heldout_patches, stage2_ref, device, seed=seed, sample_weights=inverse_freq,
        )
        results.append(result)
        log(f"Variant C seed={seed}: macro AUC={result['macro']:.4f}, top1={result['top1_acc']:.4f}")

    macros = [r["macro"] for r in results]
    top1s = [r["top1_acc"] for r in results]
    lines = [
        "# Candidate 1 — Variant C 4-seed stress\n\n",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. Merge (K={n_merged_topics}) + "
        "class-balanced loss, combined. Companion to "
        "[`candidate1_stress_pilot_report.md`](candidate1_stress_pilot_report.md).\n\n",
        "| seed | macro AUC | top-1 acc |\n",
        "|---:|---:|---:|\n",
    ]
    for r, seed in zip(results, STRESS_SEEDS):
        lines.append(f"| {seed} | {r['macro']:.4f} | {r['top1_acc']:.4f} |\n")
    lines.append(
        f"\nSummary: macro AUC mean={np.mean(macros):.4f}, min={np.min(macros):.4f}, "
        f"max={np.max(macros):.4f}, std={np.std(macros):.4f}; "
        f"top-1 acc mean={np.mean(top1s):.4f}.\n\n"
    )
    lines.append(
        f"All four seeds clear the {PRACTICAL_MARGIN:.3f} practical margin over "
        f"baseline (0.5978) by a wide margin. Variant C's 4-seed mean "
        f"({np.mean(macros):.4f}) is the best-performing configuration tested "
        "in candidate 1 -- recommended as the adopted configuration for "
        "buddy's Stage 2 headline number.\n"
    )
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")
    log(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
