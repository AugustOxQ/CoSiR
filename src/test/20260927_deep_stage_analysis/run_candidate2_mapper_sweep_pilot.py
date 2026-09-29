"""Candidate 2: buddy Stage 2 mapper LR/epoch sweep, on top of candidate
1's adopted Variant C configuration (merge K=19->16 + class-balanced
loss). Screens at seed 42, stress-tests only configurations that beat
the 0.6334 4-seed-mean baseline by the +0.005 practical margin.

Reuses run_candidate1_min_occupancy_pilot.py and
run_candidate1_stress_pilot.py by import.
"""

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from run_candidate1_min_occupancy_pilot import (
    PERCEPT_DIR, PRACTICAL_MARGIN, SMALL_COMMUNITIES, SNAPSHOT, TRANSFER_K, load_module,
    merge_small_communities, one_hot,
)
from run_candidate1_stress_pilot import STRESS_SEEDS


HERE = Path(__file__).resolve().parent
BUDDY_DIR = HERE.parents[1] / "test/20260923_artelingo_buddy_analysis"
REPORT_PATH = HERE / "candidate2_mapper_sweep_pilot_report.md"
VARIANT_C_BASELINE_MEAN = 0.6334
LR_GRID = (3e-4, 1e-3, 3e-3, 1e-2)
EPOCH_GRID = (100, 200, 400)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def train_variant_c(
    stage2_ref, n_topics, train_labels, heldout_labels, train_patches, heldout_patches,
    device, seed, lr, epochs, sample_weights,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_targets = one_hot(train_labels, n_topics).to(device)
    heldout_targets = one_hot(heldout_labels, n_topics).cpu().numpy()
    weights_tensor = torch.as_tensor(sample_weights, dtype=torch.float32, device=device)

    mapper = stage2_ref.AttentionPoolingMapper(n_topics=n_topics).to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    mapper.train()
    for epoch in range(1, epochs + 1):
        logits = mapper(train_patches)
        per_sample_loss = bce(logits, train_targets).mean(dim=1)
        loss = (per_sample_loss * weights_tensor).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mapper.eval()
    with torch.no_grad():
        heldout_scores = torch.sigmoid(mapper(heldout_patches)).cpu().numpy()
    model_aucs, skipped = stage2_ref.evaluate_auc(heldout_scores, heldout_targets, log, f"lr={lr:g}_ep={epochs}")
    summary = stage2_ref.auc_summary(model_aucs)
    predicted = heldout_scores.argmax(axis=1)
    top1_acc = float((predicted == heldout_labels).mean())
    return {"macro": summary["macro"], "min": summary["min"], "median": summary["median"],
            "max": summary["max"], "top1_acc": top1_acc}


def main() -> None:
    stage2_ref = load_module("stage2_ref_for_candidate2", PERCEPT_DIR / "run_percept_stage2_pilot.py")
    transfer = load_module("transfer_for_candidate2", BUDDY_DIR / "run_heldout_label_transfer_pilot.py")

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
    n_topics = len(set(label_map.values()))
    heldout_labels_raw = transfer.assign_to_train_communities(
        train_embedding, train_labels, heldout_embedding, k=TRANSFER_K
    ).astype(np.int64)
    merged_heldout_labels = np.array([label_map[label] for label in heldout_labels_raw], dtype=np.int64)
    merged_counts = np.bincount(merged_train_labels, minlength=n_topics)
    inverse_freq = (len(merged_train_labels) / n_topics) / merged_counts[merged_train_labels]
    inverse_freq = inverse_freq / inverse_freq.mean()

    def run(lr, epochs, seed=42):
        return train_variant_c(
            stage2_ref, n_topics, merged_train_labels, merged_heldout_labels,
            train_patches, heldout_patches, device, seed, lr, epochs, inverse_freq,
        )

    log("=== Part A: LR sweep (seed 42, 100 epochs) ===")
    lr_results = {}
    for lr in LR_GRID:
        result = run(lr, 100)
        lr_results[lr] = result
        log(f"lr={lr:g}: macro AUC={result['macro']:.4f}, top1={result['top1_acc']:.4f}")
    best_lr = max(lr_results, key=lambda lr: lr_results[lr]["macro"])
    log(f"Best LR: {best_lr:g} (macro AUC={lr_results[best_lr]['macro']:.4f})")

    log("=== Part B: epoch sweep (seed 42, best LR) ===")
    epoch_results = {}
    for epochs in EPOCH_GRID:
        result = run(best_lr, epochs)
        epoch_results[epochs] = result
        log(f"lr={best_lr:g} epochs={epochs}: macro AUC={result['macro']:.4f}, top1={result['top1_acc']:.4f}")
    best_epochs = max(epoch_results, key=lambda e: epoch_results[e]["macro"])
    best_screen = epoch_results[best_epochs]
    log(f"Best config: lr={best_lr:g}, epochs={best_epochs}, macro AUC={best_screen['macro']:.4f}")

    beats_margin = (best_screen["macro"] - VARIANT_C_BASELINE_MEAN) >= PRACTICAL_MARGIN
    stress_results = []
    if beats_margin:
        log(f"=== Best config beats margin; 4-seed stress (lr={best_lr:g}, epochs={best_epochs}) ===")
        for seed in STRESS_SEEDS:
            result = run(best_lr, best_epochs, seed=seed)
            stress_results.append((seed, result))
            log(f"seed={seed}: macro AUC={result['macro']:.4f}")
    else:
        log("No sweep point beat the margin over 0.6334 mean; skipping stress and Part C.")

    write_report(lr_results, epoch_results, best_lr, best_epochs, beats_margin, stress_results)
    log(f"Wrote {REPORT_PATH}")


def write_report(lr_results, epoch_results, best_lr, best_epochs, beats_margin, stress_results) -> None:
    lines = [
        "# Candidate 2 — buddy Stage 2 mapper LR/epoch sweep\n\n",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. Baseline: candidate 1's "
        f"adopted Variant C (merge K=16 + class-balanced loss), 4-seed mean "
        f"**{VARIANT_C_BASELINE_MEAN:.4f}**. All sweep points below use that same "
        "merged vocabulary and loss weighting; only LR/epochs change.\n\n",
        "## Part A: learning rate sweep (seed 42, 100 epochs)\n\n",
        "| LR | macro AUC | min | median | max | top-1 acc |\n",
        "|---:|---:|---:|---:|---:|---:|\n",
    ]
    for lr, r in lr_results.items():
        lines.append(f"| {lr:g} | {r['macro']:.4f} | {r['min']:.4f} | {r['median']:.4f} | {r['max']:.4f} | {r['top1_acc']:.4f} |\n")
    lines.append(f"\nBest LR: **{best_lr:g}**.\n\n")

    lines.append("## Part B: epoch sweep (seed 42, best LR)\n\n")
    lines.append("| epochs | macro AUC | min | median | max | top-1 acc |\n")
    lines.append("|---:|---:|---:|---:|---:|---:|\n")
    for epochs, r in epoch_results.items():
        lines.append(f"| {epochs} | {r['macro']:.4f} | {r['min']:.4f} | {r['median']:.4f} | {r['max']:.4f} | {r['top1_acc']:.4f} |\n")
    lines.append(f"\nBest config: lr={best_lr:g}, epochs={best_epochs}.\n\n")

    lines.append(
        "## Part C: capacity\n\nSkipped -- see verdict below for whether Parts "
        "A/B showed enough headroom to justify it.\n\n"
    )

    lines.append("## Verdict\n\n")
    best_screen_macro = epoch_results[best_epochs]["macro"]
    if beats_margin:
        macros = [r["macro"] for _, r in stress_results]
        lines.extend([
            f"**lr={best_lr:g}, epochs={best_epochs} beat the screening margin** "
            f"(seed 42: {best_screen_macro:.4f} vs. {VARIANT_C_BASELINE_MEAN:.4f} baseline mean, "
            f"{best_screen_macro - VARIANT_C_BASELINE_MEAN:+.4f}). 4-seed stress:\n\n",
            "| seed | macro AUC |\n|---:|---:|\n",
        ])
        for seed, r in stress_results:
            lines.append(f"| {seed} | {r['macro']:.4f} |\n")
        lines.append(
            f"\nStress mean: {np.mean(macros):.4f} (min {np.min(macros):.4f}, "
            f"max {np.max(macros):.4f}, std {np.std(macros):.4f}). "
            f"{'This beats the candidate-1 baseline robustly across seeds -- adopt this LR/epoch setting.' if np.mean(macros) - VARIANT_C_BASELINE_MEAN >= PRACTICAL_MARGIN else 'This does not hold up as a robust improvement across seeds once stressed -- the seed-42 screening result was likely noise; keep candidate 1s original hyperparameters (lr=1e-3, epochs=100).'}\n"
        )
    else:
        lines.append(
            f"**No sweep point beat the {PRACTICAL_MARGIN:.3f} practical margin** "
            f"over the {VARIANT_C_BASELINE_MEAN:.4f} baseline (best screened: "
            f"{best_screen_macro:.4f}, {best_screen_macro - VARIANT_C_BASELINE_MEAN:+.4f}). "
            "Part C (capacity) is skipped per the brief's own guidance -- no "
            "evidence the mapper is under-fit rather than already well-optimized. "
            "Candidate 2 is closed as tested with a negative result; the current "
            "lr=1e-3, epochs=100 configuration from candidate 1 remains the "
            "adopted setting. Move to candidate 3 (topic-count K sweep).\n"
        )
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
