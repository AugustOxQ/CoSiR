"""Candidate 4 fix: score each multi-label-trained model against the
BASELINE's own single-label held-out targets, not each cutoff's own
(easier/harder, uncontrolled) multi-label targets.

An independent review of run_candidate4_rich_multilabel_pilot.py (Critical
finding C1) correctly caught that its negative verdict compared AUCs
computed against two different held-out target definitions -- exactly
the same class of apples-to-oranges error this investigation already
flagged in PercepT's own multi-hot-threshold numbers (master report §6b).
That is not a valid comparison regardless of which direction it cuts, so
this script re-scores properly rather than accepting the original
verdict.

Also runs the reviewer's requested W2 control: single-label targets with
the class-balanced weighting DROPPED (candidate 4's targets script drops
it for an unrelated reason -- multi-hot doesn't have a well-defined hard
class to reweight by -- but that means candidate 4's runs differ from the
0.8461 baseline in two ways at once: multi-label AND unweighted. This
control isolates the "unweighted" effect alone.)

Does not modify run_candidate4_rich_multilabel_pilot.py or any other
existing file; imports its functions.
"""

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from run_candidate1_min_occupancy_pilot import (
    PERCEPT_DIR, PRACTICAL_MARGIN, SMALL_COMMUNITIES, SNAPSHOT, TRANSFER_K,
    load_module, merge_small_communities, one_hot,
)
from run_candidate2_mapper_sweep_pilot import log
from run_candidate4_rich_multilabel_pilot import (
    BASELINE_MACRO_AUC, THRESHOLDS, cosine_vote_fractions, label_stats,
    threshold_targets,
)


HERE = Path(__file__).resolve().parent
BUDDY_DIR = HERE.parents[1] / "test/20260923_artelingo_buddy_analysis"
REPORT_PATH = HERE / "candidate4_fixed_pilot_report.md"
MAPPER_LR = 1e-2
MAPPER_EPOCHS = 400


def train_multilabel_get_scores(
    stage2_ref, train_targets: np.ndarray, train_patches: torch.Tensor,
    heldout_patches: torch.Tensor, device: str, seed: int, name: str,
    weighted: bool = False, sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Train on the given multi-hot (or one-hot) targets; return raw
    held-out sigmoid scores (not yet scored against any particular
    ground truth) so the caller can evaluate against multiple target
    definitions without retraining."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    n_topics = train_targets.shape[1]
    mapper = stage2_ref.AttentionPoolingMapper(n_topics=n_topics).to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=MAPPER_LR)
    train_tensor = torch.as_tensor(train_targets, dtype=torch.float32, device=device)
    if weighted:
        bce = nn.BCEWithLogitsLoss(reduction="none")
        weights_tensor = torch.as_tensor(sample_weights, dtype=torch.float32, device=device)
    else:
        bce = nn.BCEWithLogitsLoss()
    mapper.train()
    for epoch in range(1, MAPPER_EPOCHS + 1):
        logits = mapper(train_patches)
        if weighted:
            loss = (bce(logits, train_tensor).mean(dim=1) * weights_tensor).mean()
        else:
            loss = bce(logits, train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            log(f"{name}: epoch {epoch}/{MAPPER_EPOCHS}, BCE={loss.item():.6f}")
    mapper.eval()
    with torch.no_grad():
        heldout_scores = torch.sigmoid(mapper(heldout_patches)).cpu().numpy()
    return heldout_scores


def score_against(stage2_ref, scores: np.ndarray, targets: np.ndarray, name: str) -> dict:
    per_topic, skipped = stage2_ref.evaluate_auc(scores, targets, log, name)
    summary = stage2_ref.auc_summary(per_topic)
    return {**summary, "skipped": len(skipped)}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This pilot requires a local CUDA GPU")
    device = "cuda"
    stage2_ref = load_module("stage2_ref_for_candidate4_fixed", PERCEPT_DIR / "run_percept_stage2_pilot.py")
    transfer = load_module("transfer_for_candidate4_fixed", BUDDY_DIR / "run_heldout_label_transfer_pilot.py")

    with np.load(SNAPSHOT, allow_pickle=True) as source:
        train_paintings = source["train_paintings"]
        train_embeddings = source["train_embedding_post"]
        hard_train_labels = source["train_community_post"].astype(np.int64)
        heldout_paintings = source["heldout_paintings"]
        heldout_embeddings = source["heldout_embedding_post"]

    merged_train_labels, label_map = merge_small_communities(train_embeddings, hard_train_labels, SMALL_COMMUNITIES)
    n_topics = len(set(label_map.values()))
    if n_topics != 16:
        raise RuntimeError(f"Expected candidate 1's merged K=16, got K={n_topics}")

    # The exact single-label baseline targets from candidate 2 (K=16 merged
    # vocabulary, hard k=20 vote transfer, one-hot) -- the ground truth every
    # comparison below is scored against, for a genuine apples-to-apples
    # macro AUC alongside the 0.8461 baseline.
    baseline_heldout_hard = transfer.assign_to_train_communities(
        train_embeddings, merged_train_labels, heldout_embeddings, k=TRANSFER_K
    ).astype(np.int64)
    baseline_heldout_targets = one_hot(baseline_heldout_hard, n_topics).cpu().numpy()

    train_patches = stage2_ref.load_patch_features(
        stage2_ref.TRAIN_PATCH_FEATURE_PATH, len(train_paintings), "train",
    ).to(device)
    heldout_patches = stage2_ref.load_patch_features(
        stage2_ref.HELDOUT_PATCH_FEATURE_PATH, len(heldout_paintings), "held-out",
    ).to(device)

    log("Computing k=20 merged-community cosine vote fractions (same as candidate 4)")
    train_fractions = cosine_vote_fractions(train_embeddings, merged_train_labels, train_embeddings, n_topics, device)

    rows = []
    for cutoff in THRESHOLDS:
        train_targets = threshold_targets(train_fractions, cutoff)
        log(f"Re-scoring cutoff {cutoff:.2f} against the baseline's single-label held-out targets")
        heldout_scores = train_multilabel_get_scores(
            stage2_ref, train_targets, train_patches, heldout_patches, device, 42,
            f"cutoff_{cutoff:.2f}_seed42_refit",
        )
        own_heldout_fractions_targets = threshold_targets(
            cosine_vote_fractions(train_embeddings, merged_train_labels, heldout_embeddings, n_topics, device),
            cutoff,
        )
        own_result = score_against(stage2_ref, heldout_scores, own_heldout_fractions_targets, f"cutoff_{cutoff:.2f}_own")
        baseline_result = score_against(stage2_ref, heldout_scores, baseline_heldout_targets, f"cutoff_{cutoff:.2f}_vs_baseline")
        rows.append({
            "cutoff": cutoff, "train": label_stats(train_targets),
            "own": own_result, "vs_baseline": baseline_result,
        })
        log(f"Cutoff {cutoff:.2f}: own-target macro AUC={own_result['macro']:.4f}, "
            f"vs-baseline-target macro AUC={baseline_result['macro']:.4f}")

    # W2 control: single-label targets (same as baseline_heldout_hard/one-hot
    # for train too), but with the class-balanced weighting DROPPED --
    # isolates whether dropping the weighting alone (independent of
    # multi-labeling) explains any of the gap from 0.8461.
    log("W2 control: single-label targets, unweighted loss")
    train_targets_single = one_hot(merged_train_labels, n_topics).cpu().numpy()
    control_scores = train_multilabel_get_scores(
        stage2_ref, train_targets_single, train_patches, heldout_patches, device, 42,
        "control_single_label_unweighted", weighted=False,
    )
    control_result = score_against(stage2_ref, control_scores, baseline_heldout_targets, "control_vs_baseline")
    log(f"W2 control (single-label, unweighted): macro AUC={control_result['macro']:.4f} "
        f"(candidate-2 baseline used class-balanced weighting and got 0.8461)")

    write_report(rows, control_result)
    log(f"Wrote {REPORT_PATH}")


def write_report(rows: list[dict], control_result: dict) -> None:
    lines = [
        "# Candidate 4 fix — apples-to-apples re-scoring\n\n",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. Fixes Critical finding C1 from an "
        "independent review of `run_candidate4_rich_multilabel_pilot.py`: its negative verdict "
        "compared each cutoff's AUC against that cutoff's OWN multi-label held-out targets, not "
        "against the same single-label targets the 0.8461 baseline was scored against -- an "
        "apples-to-oranges comparison, the same class of error this investigation already flagged "
        "in PercepT's own multi-hot-threshold numbers (master report §6b). Every model below is "
        "trained exactly as in the original candidate 4 script (same multi-label targets, same "
        "unweighted loss, same lr/epochs) but is additionally re-scored against the baseline's own "
        "single-label held-out targets (candidate 2's exact k=20 hard-vote transfer on the merged "
        f"K=16 vocabulary) for a genuine like-for-like macro AUC. Baseline: {BASELINE_MACRO_AUC:.4f}.\n\n",
        "## Results\n\n",
        "| cutoff | train mean/median/max labels | own-target macro AUC | vs-baseline-target macro AUC |\n",
        "|---:|---|---:|---:|\n",
    ]
    for row in rows:
        train = row["train"]
        lines.append(
            f"| {row['cutoff']:.2f} | {train['mean']:.3f}/{train['median']:.3f}/{train['max']} | "
            f"{row['own']['macro']:.4f} | {row['vs_baseline']['macro']:.4f} |\n"
        )
    best = max(rows, key=lambda row: row["vs_baseline"]["macro"])
    lines.extend([
        f"\n## W2 control: single-label targets, unweighted loss\n\n"
        f"Macro AUC (vs. baseline single-label targets): **{control_result['macro']:.4f}**. "
        f"The candidate-2 baseline (same targets, same lr/epochs, but WITH class-balanced "
        f"weighting) got {BASELINE_MACRO_AUC:.4f}. The difference "
        f"({control_result['macro'] - BASELINE_MACRO_AUC:+.4f}) isolates the effect of dropping "
        "the class-balanced weighting alone, independent of multi-labeling.\n\n",
        "## Verdict\n\n",
        f"On the correct, like-for-like comparison (vs-baseline-target column), the best cutoff "
        f"({best['cutoff']:.2f}) scores **{best['vs_baseline']['macro']:.4f}** against the "
        f"{BASELINE_MACRO_AUC:.4f} baseline ({best['vs_baseline']['macro'] - BASELINE_MACRO_AUC:+.4f}). ",
    ])
    if best["vs_baseline"]["macro"] - BASELINE_MACRO_AUC >= PRACTICAL_MARGIN:
        lines.append(
            "**This clears the practical margin — the original negative verdict does not hold "
            "under a fair comparison.** Recommend 4-seed stressing this cutoff with the "
            "corrected evaluation before adopting.\n"
        )
    else:
        lines.append(
            "**This does not clear the practical margin — the original negative verdict for "
            "candidate 4 is confirmed under a fair comparison, for a different (and correct) "
            "reason than originally stated.** The W2 control result above additionally shows "
            f"how much of any gap is attributable to dropping the class-balanced weighting "
            "rather than to multi-labeling itself.\n"
        )
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
