"""Train a PercepT-Stage-2-comparable image-only mapper for frozen buddy topics.

The saved buddy train Leiden partition supplies one target per painting. Held-out
targets use the existing k=20 transfer onto that train vocabulary. This pilot
uses PercepT's one-hot BCE training and train-marginal AUC baseline conventions.
"""

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
PERCEPT_DIR = HERE.parent / "20260922_percept_topic_pipeline"
SNAPSHOT = HERE / "attention_h1_embedding_snapshot.npz"
REPORT = HERE / "buddy_stage2_pilot_report.md"
N_TOPICS = 19
SEED = 42
TRANSFER_K = 20
PRACTICAL_MARGIN = 0.01

# Read from percept_stage2_pilot_report.md (the completed PercepT Stage 2 run).
PERCEPT_MAPPER_SUMMARY = {
    "macro": 0.5690, "min": 0.3618, "median": 0.5536, "max": 0.8744,
}
PERCEPT_BASELINE_SUMMARY = {
    "macro": 0.5000, "min": 0.5000, "median": 0.5000, "max": 0.5000,
}
PERCEPT_SKIPPED_TOPICS = 0


def one_hot_targets(labels: np.ndarray) -> torch.Tensor:
    """Make PercepT-shaped float targets from one hard Leiden label per row."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise RuntimeError("Community targets must be a one-dimensional integer array.")
    if np.any(labels < 0) or np.any(labels >= N_TOPICS):
        raise RuntimeError(f"Community targets must be within 0..{N_TOPICS - 1}.")
    targets = torch.zeros((len(labels), N_TOPICS), dtype=torch.float32)
    targets.scatter_(1, torch.as_tensor(labels, dtype=torch.long)[:, None], 1.0)
    return targets


def write_report(
    stage2,
    train_stats: dict[str, float],
    heldout_stats: dict[str, float],
    mapper_losses: list[float],
    model_aucs: dict[int, float],
    baseline_aucs: dict[int, float],
    skipped_topics: list[int],
    transferred_labels: int,
    fallback_count: int,
) -> None:
    """Record the complete Stage 2 AUC comparison and blunt numeric verdict."""
    model_summary = stage2.auc_summary(model_aucs)
    baseline_summary = stage2.auc_summary(baseline_aucs)
    improvement = model_summary["macro"] - baseline_summary["macro"]
    percept_gap = round(model_summary["macro"], 4) - PERCEPT_MAPPER_SUMMARY["macro"]
    if percept_gap > 0:
        comparison = "better than PercepT's"
    elif percept_gap < 0:
        comparison = "worse than PercepT's"
    else:
        comparison = "equal to PercepT's at the reported four-decimal precision"

    lines = [
        "# ArtELingo buddy Stage 2 patch-attention pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Frozen Stage-1 topics and held-out transfer\n\n",
        "The frozen `attention_h1_embedding_snapshot.npz` supplies post-training "
        "32-D train and held-out embeddings and the train Leiden partition "
        f"(communities 0–{N_TOPICS - 1}). The held-out labels are assigned to "
        f"the train vocabulary with the existing cosine k-NN majority-vote transfer "
        f"at k={TRANSFER_K}; {transferred_labels}/{N_TOPICS} train communities "
        f"appear in held-out and there were {fallback_count} degenerate tie "
        "fallbacks. Both splits were checked for exactly matching unique painting-ID "
        "sets and reindexed by painting ID into the fresh "
        "`load_dedup_features()` order used by the patch caches.\n\n",
        "## Frozen multi-label target statistics\n\n",
        "| split | mean labels | median labels | max labels | fraction multi-labeled |\n",
        "|---|---:|---:|---:|---:|\n",
        f"| train | {train_stats['mean']:.3f} | {train_stats['median']:.3f} | "
        f"{train_stats['max']} | {train_stats['fraction_multi_labeled']:.2%} |\n",
        f"| held-out | {heldout_stats['mean']:.3f} | "
        f"{heldout_stats['median']:.3f} | {heldout_stats['max']} | "
        f"{heldout_stats['fraction_multi_labeled']:.2%} |\n\n",
        "Buddy has exactly one label per painting by construction: Leiden "
        "communities form a hard partition. PercepT's own Stage 2 report also "
        "records 1.000/1.000/1 labels and 0.00% multi-labeled paintings in "
        "both splits, despite its multi-hot threshold rule. Thus both realized "
        "Stage 2 target sets are single-label, making buddy's hard partition "
        "a structurally faithful comparison rather than an apples-to-oranges "
        "one.\n\n",
        "## Attention-pooling mapper training\n\n",
        "The mapper consumes only cached `[painting, 50, 512]` image patch "
        "tokens. PercepT's unchanged single-query attention-pooling architecture "
        f"has a {N_TOPICS}-topic head. It trains for "
        f"{stage2.MAPPER_EPOCHS} full-batch epochs with Adam at learning rate "
        f"{stage2.MAPPER_LEARNING_RATE:g}, using **BCEWithLogitsLoss against "
        "one-hot targets**, as in PercepT Stage 2.\n\n",
        "| epoch | full-batch BCE loss |\n",
        "|---:|---:|\n",
    ]
    lines.extend(
        f"| {epoch} | {mapper_losses[epoch - 1]:.6f} |\n"
        for epoch in range(10, stage2.MAPPER_EPOCHS + 1, 10)
    )
    lines.extend([
        "\n## Held-out per-topic AUC\n\n",
        "| topic | mapper AUC | marginal-frequency baseline AUC |\n",
        "|---:|---:|---:|\n",
    ])
    for topic in range(N_TOPICS):
        model_value = f"{model_aucs[topic]:.4f}" if topic in model_aucs else "skipped"
        baseline_value = (
            f"{baseline_aucs[topic]:.4f}" if topic in baseline_aucs else "skipped"
        )
        lines.append(f"| {topic} | {model_value} | {baseline_value} |\n")
    lines.extend([
        "\n| scorer | macro AUC | min | median | max | skipped topics |\n",
        "|---|---:|---:|---:|---:|---:|\n",
        f"| patch-attention mapper | {model_summary['macro']:.4f} | "
        f"{model_summary['min']:.4f} | {model_summary['median']:.4f} | "
        f"{model_summary['max']:.4f} | {len(skipped_topics)} |\n",
        f"| train-marginal baseline | {baseline_summary['macro']:.4f} | "
        f"{baseline_summary['min']:.4f} | {baseline_summary['median']:.4f} | "
        f"{baseline_summary['max']:.4f} | {len(skipped_topics)} |\n\n",
        "## Direct comparison with PercepT Stage 2\n\n",
        "The PercepT values below come from "
        "`../20260922_percept_topic_pipeline/percept_stage2_pilot_report.md`. "
        "Buddy has 19 topics; PercepT has 40.\n\n",
        "| system and scorer | macro AUC | min | median | max | skipped topics |\n",
        "|---|---:|---:|---:|---:|---:|\n",
        f"| buddy patch-attention mapper | {model_summary['macro']:.4f} | "
        f"{model_summary['min']:.4f} | {model_summary['median']:.4f} | "
        f"{model_summary['max']:.4f} | {len(skipped_topics)} |\n",
        f"| buddy train-marginal baseline | {baseline_summary['macro']:.4f} | "
        f"{baseline_summary['min']:.4f} | {baseline_summary['median']:.4f} | "
        f"{baseline_summary['max']:.4f} | {len(skipped_topics)} |\n",
        f"| PercepT patch-attention mapper | {PERCEPT_MAPPER_SUMMARY['macro']:.4f} | "
        f"{PERCEPT_MAPPER_SUMMARY['min']:.4f} | "
        f"{PERCEPT_MAPPER_SUMMARY['median']:.4f} | "
        f"{PERCEPT_MAPPER_SUMMARY['max']:.4f} | {PERCEPT_SKIPPED_TOPICS} |\n",
        f"| PercepT train-marginal baseline | "
        f"{PERCEPT_BASELINE_SUMMARY['macro']:.4f} | "
        f"{PERCEPT_BASELINE_SUMMARY['min']:.4f} | "
        f"{PERCEPT_BASELINE_SUMMARY['median']:.4f} | "
        f"{PERCEPT_BASELINE_SUMMARY['max']:.4f} | "
        f"{PERCEPT_SKIPPED_TOPICS} |\n\n",
        "## Conclusion\n\n",
        (
            "**Buddy's image-only Stage 2 mapper meaningfully beats its "
            "train-marginal baseline.** "
            if improvement >= PRACTICAL_MARGIN
            else "**Buddy's image-only Stage 2 mapper does not meaningfully beat "
            "its train-marginal baseline.** "
        ),
        f"Its macro AUC is {model_summary['macro']:.4f} versus "
        f"{baseline_summary['macro']:.4f}, a difference of {improvement:+.6f}; "
        f"the predeclared practical margin is {PRACTICAL_MARGIN:.2f}. "
        f"PercepT's macro AUC is {PERCEPT_MAPPER_SUMMARY['macro']:.4f}; "
        f"buddy minus PercepT is {percept_gap:+.4f}. Buddy's image-only "
        f"topic space is {comparison} by this macro-AUC "
        "comparison.\n",
    ])
    REPORT.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    """Align frozen buddy topics, fit the mapper, and score held-out AUC."""
    # Import standalone siblings without executing their main blocks.
    from importlib.util import module_from_spec, spec_from_file_location

    def load_module(name: str, path: Path):
        spec = spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import sibling module from {path}")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    stage2 = load_module(
        "percept_stage2_for_buddy_stage2", PERCEPT_DIR / "run_percept_stage2_pilot.py"
    )
    probe = load_module(
        "downstream_probe_for_buddy_stage2", HERE / "run_buddy_percept_downstream_probe_pilot.py"
    )
    audit = load_module(
        "matched_audit_for_buddy_stage2", HERE / "run_buddy_percept_matched_silhouette_audit_pilot.py"
    )
    transfer = load_module(
        "label_transfer_for_buddy_stage2", HERE / "run_heldout_label_transfer_pilot.py"
    )
    pipeline = load_module("pipeline_for_buddy_stage2_train", HERE / "run_pipeline.py")
    heldout_pipeline = load_module(
        "pipeline_for_buddy_stage2_heldout", HERE / "run_pipeline.py"
    )
    heldout_pipeline.STORAGE_DIR = stage2.base.HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = stage2.base.HELDOUT_JSON
    pipeline.assert_extraction_complete()
    heldout_pipeline.assert_extraction_complete()
    log = pipeline.log

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_paintings, _, _, _ = pipeline.load_dedup_features()
    heldout_paintings, _, _, _ = heldout_pipeline.load_dedup_features()
    if len(train_paintings) != probe.N_TRAIN or len(heldout_paintings) != probe.N_HELDOUT:
        raise RuntimeError("Fresh deduplicated painting counts differ from patch caches")
    train_patches = stage2.load_patch_features(
        stage2.TRAIN_PATCH_FEATURE_PATH, len(train_paintings), "train"
    )
    heldout_patches = stage2.load_patch_features(
        stage2.HELDOUT_PATCH_FEATURE_PATH, len(heldout_paintings), "held-out"
    )
    log(f"Patch cache directory: {stage2.PATCH_FEATURE_DIR}")

    snapshot = probe.load_snapshot(
        SNAPSHOT,
        (
            "seed", "train_paintings", "heldout_paintings",
            "train_embedding_post", "train_community_post", "heldout_embedding_post",
        ),
    )
    if int(snapshot["seed"]) != SEED:
        raise RuntimeError(f"Buddy snapshot must be a seed-{SEED} run")
    train = probe.align_snapshot(
        snapshot, train_paintings, "train",
        ("train_embedding_post", "train_community_post"), audit,
    )
    heldout = probe.align_snapshot(
        snapshot, heldout_paintings, "heldout", ("heldout_embedding_post",), audit,
    )
    train_embeddings = train["train_embedding_post"]
    heldout_embeddings = heldout["heldout_embedding_post"]
    train_labels = train["train_community_post"]
    if train_embeddings.shape != (len(train_paintings), 32) or heldout_embeddings.shape != (
        len(heldout_paintings), 32
    ):
        raise RuntimeError("Buddy post-training embeddings must be aligned 32-D arrays")
    if not np.array_equal(np.unique(train_labels), np.arange(N_TOPICS)):
        raise RuntimeError(f"Buddy train vocabulary must be contiguous 0..{N_TOPICS - 1}")
    heldout_labels = transfer.assign_to_train_communities(
        train_embeddings, train_labels, heldout_embeddings, k=TRANSFER_K
    )
    train_targets = one_hot_targets(train_labels)
    heldout_targets = one_hot_targets(heldout_labels)
    train_stats = stage2.label_statistics(train_targets)
    heldout_stats = stage2.label_statistics(heldout_targets)
    for split, stats in (("Train", train_stats), ("Held-out", heldout_stats)):
        log(
            f"{split} frozen multi-label target statistics: "
            f"mean={stats['mean']:.3f}, median={stats['median']:.3f}, "
            f"max={stats['max']}, multi={stats['fraction_multi_labeled']:.2%}."
        )
    log(
        f"k={TRANSFER_K} transfer covers {len(np.unique(heldout_labels))}/{N_TOPICS} "
        f"train communities; degenerate tie fallbacks="
        f"{transfer.assign_to_train_communities.last_fallback_count}."
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Training buddy's image-only Stage 2 mapper on {device}...")
    mapper = stage2.AttentionPoolingMapper(n_topics=N_TOPICS).to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=stage2.MAPPER_LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    train_patches = train_patches.to(device)
    train_targets = train_targets.to(device)
    mapper_losses = []
    mapper.train()
    for epoch in range(1, stage2.MAPPER_EPOCHS + 1):
        loss = loss_fn(mapper(train_patches), train_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mapper_losses.append(loss.item())
        if epoch % 10 == 0:
            log(
                f"Stage-2 epoch {epoch:03d}/{stage2.MAPPER_EPOCHS}: "
                f"mean BCE={loss.item():.6f}"
            )

    mapper.eval()
    with torch.no_grad():
        heldout_scores = torch.sigmoid(
            mapper(heldout_patches.to(device))
        ).cpu().numpy()
    heldout_targets_np = heldout_targets.cpu().numpy()
    model_aucs, skipped_topics = stage2.evaluate_auc(
        heldout_scores, heldout_targets_np, log, "patch-attention mapper"
    )
    train_marginals = train_targets.float().mean(dim=0).cpu().numpy()
    baseline_scores = np.broadcast_to(train_marginals, heldout_targets_np.shape)
    baseline_aucs, baseline_skipped_topics = stage2.evaluate_auc(
        baseline_scores, heldout_targets_np, log, "train-marginal baseline"
    )
    if skipped_topics != baseline_skipped_topics:
        raise RuntimeError("AUC validity differs between model and identical-target baseline.")
    write_report(
        stage2, train_stats, heldout_stats, mapper_losses, model_aucs,
        baseline_aucs, skipped_topics, len(np.unique(heldout_labels)),
        transfer.assign_to_train_communities.last_fallback_count,
    )
    log(f"Wrote report to {REPORT}.")


if __name__ == "__main__":
    main()
