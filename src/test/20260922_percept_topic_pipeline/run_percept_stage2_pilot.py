"""Run PercepT Stage 2 with frozen Stage-1 P-Topics and patch attention pooling.

This standalone GPU pilot first deterministically re-fits the established
K=60/40 Stage-1 configuration.  It then freezes those topics and trains the
paper-motivated image-only mapper on cached CLIP patch-token features.
"""

import importlib.util
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn


OUT_DIR = os.path.dirname(__file__)
BASE_PILOT_PATH = os.path.join(OUT_DIR, "run_percept_stage1_pilot.py")
SWEEP_PILOT_PATH = os.path.join(
    OUT_DIR, "run_percept_stage1_cluster_count_sweep_pilot.py"
)
REPORT_PATH = os.path.join(OUT_DIR, "percept_stage2_pilot_report.md")
PATCH_FEATURE_DIR = "/data/SSD2/pre_extract/artelingo_percept_patch_features"
TRAIN_PATCH_FEATURE_PATH = os.path.join(PATCH_FEATURE_DIR, "train_patch_features.pt")
HELDOUT_PATCH_FEATURE_PATH = os.path.join(
    PATCH_FEATURE_DIR, "heldout_patch_features.pt"
)

N_INITIAL_CLUSTERS = 60
N_SURVIVING_CLUSTERS = 40
LAMBDA_BALANCE = 1000
LAMBDA_RECONSTRUCTION = 1
SEED = 42
EXPECTED_HELDOUT_EMOTION_AMI = 0.1238
EXPECTED_HELDOUT_GENRE_AMI = 0.2617
REPRODUCTION_TOLERANCE = 0.002
MULTI_LABEL_THRESHOLD = 2.0 / N_SURVIVING_CLUSTERS
MAPPER_LEARNING_RATE = 1e-3
MAPPER_EPOCHS = 100


def load_module(module_name: str, path: str):
    """Load a standalone sibling script without running its main block."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("percept_stage1_base_for_stage2", BASE_PILOT_PATH)
sweep = load_module("percept_stage1_sweep_for_stage2", SWEEP_PILOT_PATH)


class AttentionPoolingMapper(nn.Module):
    """Pool 50 CLIP tokens with one learned query, then score all P-Topics."""

    def __init__(self, d_model: int = 512, n_topics: int = N_SURVIVING_CLUSTERS):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.classifier = nn.Linear(d_model, n_topics)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Return one multi-label logit per frozen topic."""
        query = self.query.expand(patch_tokens.shape[0], -1, -1)
        attn_weights = (
            query @ patch_tokens.transpose(1, 2)
        ) / (patch_tokens.shape[-1] ** 0.5)
        attn_weights = attn_weights.softmax(dim=-1)
        pooled = (attn_weights @ patch_tokens).squeeze(1)
        return self.classifier(pooled)


def load_patch_features(path: str, expected_paintings: int, split_name: str) -> torch.Tensor:
    """Load and validate the Script-1 cache aligned to deduplicated paintings."""
    if not os.path.exists(path):
        raise RuntimeError(
            f"Missing {split_name} patch-feature cache at {path}. "
            "Run run_percept_patch_feature_extraction.py first."
        )
    patch_features = torch.load(path, map_location="cpu")
    if not isinstance(patch_features, torch.Tensor):
        raise RuntimeError(f"{split_name} patch cache is not a tensor: {path}")
    expected_shape = (expected_paintings, 50, 512)
    if tuple(patch_features.shape) != expected_shape:
        raise RuntimeError(
            f"{split_name} patch cache has shape {tuple(patch_features.shape)}; "
            f"expected {expected_shape} in load_dedup_features() painting order."
        )
    if patch_features.dtype != torch.float32:
        raise RuntimeError(
            f"{split_name} patch cache has dtype {patch_features.dtype}; expected float32."
        )
    return patch_features


def multi_hot_targets(
    encoder: nn.Module,
    centers: torch.Tensor,
    inputs: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Freeze argmax-plus-threshold DEC assignments as multi-label targets."""
    encoder.eval()
    with torch.no_grad():
        q = base.soft_assignments(encoder(inputs.to(device)), centers)
        targets = torch.zeros_like(q)
        targets.scatter_(1, q.argmax(dim=1, keepdim=True), 1.0)
        targets[q > MULTI_LABEL_THRESHOLD] = 1.0
    return targets.cpu()


def label_statistics(targets: torch.Tensor) -> dict[str, float]:
    """Summarize labels per painting for the required report."""
    counts = targets.sum(dim=1).cpu().numpy()
    return {
        "mean": float(np.mean(counts)),
        "median": float(np.median(counts)),
        "max": int(np.max(counts)),
        "fraction_multi_labeled": float(np.mean(counts > 1)),
    }


def evaluate_auc(
    scores: np.ndarray,
    targets: np.ndarray,
    log,
    scorer_name: str,
) -> tuple[dict[int, float], list[int]]:
    """Score valid per-topic held-out AUCs and explicitly log invalid topics."""
    aucs = {}
    skipped = []
    for topic in range(targets.shape[1]):
        topic_targets = targets[:, topic]
        positives = int(topic_targets.sum())
        negatives = len(topic_targets) - positives
        if positives == 0 or negatives == 0:
            log(
                f"Skipping {scorer_name} AUC for topic {topic}: "
                f"{positives} positives and {negatives} negatives in held-out."
            )
            skipped.append(topic)
            continue
        aucs[topic] = float(roc_auc_score(topic_targets, scores[:, topic]))
    if not aucs:
        raise RuntimeError("No held-out topics have both positive and negative labels.")
    return aucs, skipped


def auc_summary(aucs: dict[int, float]) -> dict[str, float]:
    """Return the required macro/min/median/max aggregate statistics."""
    values = np.asarray(list(aucs.values()))
    return {
        "macro": float(np.mean(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def write_report(
    refit_metrics: dict,
    reproduced: bool,
    train_target_stats: dict[str, float] | None = None,
    heldout_target_stats: dict[str, float] | None = None,
    mapper_losses: list[float] | None = None,
    model_aucs: dict[int, float] | None = None,
    baseline_aucs: dict[int, float] | None = None,
    skipped_topics: list[int] | None = None,
) -> None:
    """Write either the reproducibility failure or the complete Stage-2 report."""
    emotion_ami = refit_metrics["emotion"]["AMI"]
    genre_ami = refit_metrics["genre"]["AMI"]
    lines = [
        "# ArtELingo PercepT Stage 2 patch-attention pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Stage-1 K=60/40 seed-42 reproduction\n\n",
        "| metric | established | re-fit | absolute difference | status |\n",
        "|---|---:|---:|---:|---|\n",
        f"| held-out emotion AMI | {EXPECTED_HELDOUT_EMOTION_AMI:.4f} | "
        f"{emotion_ami:.4f} | {abs(emotion_ami - EXPECTED_HELDOUT_EMOTION_AMI):.4f} | "
        f"{'reproduced' if reproduced else 'FAILED'} |\n",
        f"| held-out genre AMI | {EXPECTED_HELDOUT_GENRE_AMI:.4f} | "
        f"{genre_ami:.4f} | {abs(genre_ami - EXPECTED_HELDOUT_GENRE_AMI):.4f} | "
        f"{'reproduced' if reproduced else 'FAILED'} |\n\n",
    ]
    if not reproduced:
        lines.append(
            "**Reproducibility failure.** The required Stage-1 re-fit was not within "
            f"the absolute AMI tolerance of {REPRODUCTION_TOLERANCE:.3f}; Stage 2 "
            "was not trained, so it cannot silently use a different clustering.\n"
        )
        with open(REPORT_PATH, "w") as report_file:
            report_file.writelines(lines)
        return

    assert train_target_stats is not None
    assert heldout_target_stats is not None
    assert mapper_losses is not None
    assert model_aucs is not None
    assert baseline_aucs is not None
    assert skipped_topics is not None
    model_summary = auc_summary(model_aucs)
    baseline_summary = auc_summary(baseline_aucs)
    improvement = model_summary["macro"] - baseline_summary["macro"]
    meaningful = improvement >= 0.01
    lines.extend([
        "## Frozen multi-label target statistics\n\n",
        "| split | mean labels | median labels | max labels | fraction multi-labeled |\n",
        "|---|---:|---:|---:|---:|\n",
        f"| train | {train_target_stats['mean']:.3f} | {train_target_stats['median']:.3f} | "
        f"{train_target_stats['max']} | {train_target_stats['fraction_multi_labeled']:.2%} |\n",
        f"| held-out | {heldout_target_stats['mean']:.3f} | "
        f"{heldout_target_stats['median']:.3f} | {heldout_target_stats['max']} | "
        f"{heldout_target_stats['fraction_multi_labeled']:.2%} |\n\n",
        "## Attention-pooling mapper training\n\n",
        "The mapper consumes only cached `[painting, 50, 512]` patch tokens. A single "
        "learned query performs scaled dot-product attention over patches, followed by "
        "a linear 40-topic head and BCE-with-logits loss.\n\n",
        "| epoch | full-batch BCE loss |\n",
        "|---:|---:|\n",
    ])
    lines.extend(
        f"| {epoch} | {mapper_losses[epoch - 1]:.6f} |\n"
        for epoch in range(10, MAPPER_EPOCHS + 1, 10)
    )
    lines.extend([
        "\n## Held-out per-topic AUC\n\n",
        "| topic | mapper AUC | marginal-frequency baseline AUC |\n",
        "|---:|---:|---:|\n",
    ])
    lines.extend(
        f"| {topic} | {model_aucs[topic]:.4f} | {baseline_aucs[topic]:.4f} |\n"
        for topic in sorted(model_aucs)
    )
    lines.extend([
        "\n| scorer | macro AUC | min | median | max | skipped topics |\n",
        "|---|---:|---:|---:|---:|---:|\n",
        f"| patch-attention mapper | {model_summary['macro']:.4f} | "
        f"{model_summary['min']:.4f} | {model_summary['median']:.4f} | "
        f"{model_summary['max']:.4f} | {len(skipped_topics)} |\n",
        f"| train-marginal baseline | {baseline_summary['macro']:.4f} | "
        f"{baseline_summary['min']:.4f} | {baseline_summary['median']:.4f} | "
        f"{baseline_summary['max']:.4f} | {len(skipped_topics)} |\n\n",
        "## Conclusion\n\n",
        (
            "**The image-only attention-pooling mapper meaningfully beats the "
            "marginal-frequency baseline.** Its macro AUC is higher by "
            f"{improvement:.4f}, exceeding the predeclared 0.01 practical margin.\n"
            if meaningful
            else "**The image-only attention-pooling mapper does not meaningfully beat "
            "the marginal-frequency baseline.** Its macro-AUC difference is "
            f"{improvement:.4f}, below the 0.01 practical margin.\n"
        ),
    ])
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    """Re-fit frozen topics, validate reproduction, then train and evaluate Stage 2."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    pipeline = base.load_sibling_module(
        "artelingo_run_pipeline_percept_stage2_train", base.PIPELINE_PATH
    )
    affect_pilot = base.load_sibling_module(
        "artelingo_run_affect_pilot_percept_stage2", base.AFFECT_PILOT_PATH
    )
    cca_audit = base.load_sibling_module(
        "artelingo_run_cca_audit_percept_stage2", base.CCA_AUDIT_PATH
    )
    heldout_pipeline = base.load_sibling_module(
        "artelingo_run_pipeline_percept_stage2_heldout", base.PIPELINE_PATH
    )
    heldout_pipeline.STORAGE_DIR = base.HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = base.HELDOUT_JSON
    pipeline.assert_extraction_complete()
    heldout_pipeline.assert_extraction_complete()
    log = pipeline.log

    log("Loading and deduplicating train CLIP features for Stage-1 re-fit...")
    paintings, img_nodes, txt_nodes, emotion_counts = pipeline.load_dedup_features()
    train_emotions = [pipeline.majority(counts) for counts in emotion_counts]
    log("Loading and deduplicating held-out CLIP features for Stage-1 re-fit...")
    (
        heldout_paintings,
        heldout_img_nodes,
        heldout_txt_nodes,
        heldout_emotion_counts,
    ) = heldout_pipeline.load_dedup_features()
    heldout_emotions = [
        heldout_pipeline.majority(counts) for counts in heldout_emotion_counts
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {device} for Stage-1 re-fit and Stage-2 mapper training.")
    affect_train = base.extract_affect_embedding_nodes(
        pipeline.TRAIN_JSON, paintings, device, log
    )
    affect_heldout = base.extract_affect_embedding_nodes(
        base.HELDOUT_JSON, heldout_paintings, device, log
    )
    train_h = base.fused_embeddings(
        img_nodes, txt_nodes, affect_train, cca_audit, affect_pilot
    )
    heldout_h = base.fused_embeddings(
        heldout_img_nodes,
        heldout_txt_nodes,
        affect_heldout,
        cca_audit,
        affect_pilot,
    )
    if train_h.shape[1] != heldout_h.shape[1]:
        raise RuntimeError(
            f"Train/held-out fused dimensions differ: {train_h.shape[1]} vs "
            f"{heldout_h.shape[1]}."
        )

    train_inputs = torch.from_numpy(train_h)
    encoder, decoder = base.build_autoencoder(train_h.shape[1])
    encoder.to(device)
    decoder.to(device)
    base.pretrain_autoencoder(encoder, decoder, train_inputs, device, log)
    centers = sweep.initialize_cluster_centers(
        encoder, train_inputs, device, N_INITIAL_CLUSTERS, SEED
    )
    sweep.train_dec_until_stable(
        encoder,
        decoder,
        centers,
        train_inputs,
        device,
        log,
        N_INITIAL_CLUSTERS,
    )
    surviving_centers, surviving_indices = sweep.prune_centers(
        centers, N_SURVIVING_CLUSTERS
    )
    log(
        f"Pruned {N_INITIAL_CLUSTERS - N_SURVIVING_CLUSTERS} centers; retained "
        f"original indices {surviving_indices.tolist()}."
    )
    _, heldout_refit_metrics, _ = sweep.evaluate_run(
        encoder,
        centers,
        train_inputs,
        heldout_h,
        pipeline,
        heldout_pipeline,
        paintings,
        train_emotions,
        heldout_paintings,
        heldout_emotions,
        device,
        N_SURVIVING_CLUSTERS,
    )
    reproduced = (
        abs(heldout_refit_metrics["emotion"]["AMI"] - EXPECTED_HELDOUT_EMOTION_AMI)
        < REPRODUCTION_TOLERANCE
        and abs(heldout_refit_metrics["genre"]["AMI"] - EXPECTED_HELDOUT_GENRE_AMI)
        < REPRODUCTION_TOLERANCE
    )
    if not reproduced:
        write_report(heldout_refit_metrics, reproduced=False)
        log(f"Stage-1 reproducibility failure; wrote report to {REPORT_PATH}.")
        return

    train_targets = multi_hot_targets(encoder, surviving_centers, train_inputs, device)
    heldout_targets = multi_hot_targets(
        encoder, surviving_centers, torch.from_numpy(heldout_h), device
    )
    train_target_stats = label_statistics(train_targets)
    heldout_target_stats = label_statistics(heldout_targets)
    log(
        "Train label counts: "
        f"mean={train_target_stats['mean']:.3f}, "
        f"median={train_target_stats['median']:.3f}, "
        f"max={train_target_stats['max']}, "
        f"multi={train_target_stats['fraction_multi_labeled']:.2%}."
    )
    log(
        "Held-out label counts: "
        f"mean={heldout_target_stats['mean']:.3f}, "
        f"median={heldout_target_stats['median']:.3f}, "
        f"max={heldout_target_stats['max']}, "
        f"multi={heldout_target_stats['fraction_multi_labeled']:.2%}."
    )

    train_patch_features = load_patch_features(
        TRAIN_PATCH_FEATURE_PATH, len(paintings), "train"
    )
    heldout_patch_features = load_patch_features(
        HELDOUT_PATCH_FEATURE_PATH, len(heldout_paintings), "held-out"
    )
    mapper = AttentionPoolingMapper().to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=MAPPER_LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    train_patch_features = train_patch_features.to(device)
    train_targets = train_targets.to(device)
    mapper_losses = []
    mapper.train()
    for epoch in range(1, MAPPER_EPOCHS + 1):
        loss = loss_fn(mapper(train_patch_features), train_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mapper_losses.append(loss.item())
        if epoch % 10 == 0:
            log(f"Stage-2 epoch {epoch:03d}/{MAPPER_EPOCHS}: mean BCE={loss.item():.6f}")

    mapper.eval()
    with torch.no_grad():
        heldout_scores = torch.sigmoid(mapper(heldout_patch_features.to(device))).cpu().numpy()
    heldout_targets_np = heldout_targets.cpu().numpy()
    model_aucs, skipped_topics = evaluate_auc(
        heldout_scores, heldout_targets_np, log, "patch-attention mapper"
    )
    train_marginals = train_targets.float().mean(dim=0).cpu().numpy()
    baseline_scores = np.broadcast_to(train_marginals, heldout_targets_np.shape)
    baseline_aucs, baseline_skipped_topics = evaluate_auc(
        baseline_scores, heldout_targets_np, log, "train-marginal baseline"
    )
    if skipped_topics != baseline_skipped_topics:
        raise RuntimeError("AUC validity differs between model and identical-target baseline.")
    write_report(
        heldout_refit_metrics,
        reproduced=True,
        train_target_stats=train_target_stats,
        heldout_target_stats=heldout_target_stats,
        mapper_losses=mapper_losses,
        model_aucs=model_aucs,
        baseline_aucs=baseline_aucs,
        skipped_topics=skipped_topics,
    )
    log(f"Wrote report to {REPORT_PATH}.")


if __name__ == "__main__":
    main()
