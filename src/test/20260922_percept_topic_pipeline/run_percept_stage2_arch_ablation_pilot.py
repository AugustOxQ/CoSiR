"""Ablate PercepT Stage-2 patch attention with pooled CLIP embeddings on GPU.

This standalone script re-fits the fixed K=60/40 Stage-1 configuration once,
freezes its encoder and surviving centers, then trains only a linear classifier
on the already-cached global pooled CLIP image embeddings across four seeds.
"""

import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from run_percept_stage2_sweep_pilot import (
    BASE_PILOT_PATH,
    N_INITIAL_CLUSTERS,
    N_SURVIVING_CLUSTERS,
    REPRODUCTION_TOLERANCE,
    SWEEP_PILOT_PATH,
    auc_summary,
    evaluate_auc,
    load_module,
    multi_hot_targets,
    threshold_label,
)


OUT_DIR = os.path.dirname(__file__)
REPORT_PATH = os.path.join(
    OUT_DIR, "percept_stage2_arch_ablation_pilot_report.md"
)
SEED = 42
SEEDS = (42, 7, 123, 2024)
THRESHOLD_MULTIPLIER = 1.2
CLASSIFIER_LEARNING_RATE = 3e-3
CLASSIFIER_EPOCHS = 100
EXPECTED_HELDOUT_EMOTION_AMI = 0.1238
EXPECTED_HELDOUT_GENRE_AMI = 0.2617
ATTENTION_MEAN_MACRO_AUC = 0.8256
ATTENTION_MIN_MACRO_AUC = 0.8248
ATTENTION_MAX_MACRO_AUC = 0.8272


def normalize_pooled_embeddings(
    img_nodes: np.ndarray, expected_paintings: int, split_name: str
) -> torch.Tensor:
    """Validate and L2-normalize cached global CLIP image embeddings."""
    if img_nodes.shape != (expected_paintings, 512):
        raise RuntimeError(
            f"{split_name} pooled image embeddings have shape {img_nodes.shape}; "
            f"expected ({expected_paintings}, 512) in load_dedup_features() order."
        )
    embeddings = torch.from_numpy(img_nodes).float()
    return F.normalize(embeddings, p=2, dim=1)


def train_and_evaluate_linear_classifier(
    train_embeddings: torch.Tensor,
    heldout_embeddings: torch.Tensor,
    train_targets: torch.Tensor,
    heldout_targets: torch.Tensor,
    classifier_seed: int,
    device: str,
    log,
) -> dict:
    """Train one pooled-embedding linear classifier draw and score frozen targets."""
    # Re-seed immediately before construction; the shared Stage-1 fit remains fixed.
    torch.manual_seed(classifier_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(classifier_seed)
    classifier = nn.Linear(512, N_SURVIVING_CLUSTERS).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=CLASSIFIER_LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    classifier.train()
    for epoch in range(1, CLASSIFIER_EPOCHS + 1):
        loss = loss_fn(classifier(train_embeddings), train_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 10 == 0:
            log(
                f"pooled-linear seed {classifier_seed} epoch "
                f"{epoch:03d}/{CLASSIFIER_EPOCHS}: mean BCE={loss.item():.6f}"
            )

    classifier.eval()
    with torch.no_grad():
        heldout_scores = torch.sigmoid(classifier(heldout_embeddings)).cpu().numpy()
    heldout_targets_np = heldout_targets.cpu().numpy()
    model_aucs, skipped_topics = evaluate_auc(
        heldout_scores,
        heldout_targets_np,
        log,
        f"pooled-linear seed {classifier_seed}",
    )
    train_marginals = train_targets.float().mean(dim=0).cpu().numpy()
    baseline_scores = np.broadcast_to(train_marginals, heldout_targets_np.shape)
    baseline_aucs, baseline_skipped_topics = evaluate_auc(
        baseline_scores,
        heldout_targets_np,
        log,
        f"pooled-linear seed {classifier_seed} train-marginal baseline",
    )
    if skipped_topics != baseline_skipped_topics:
        raise RuntimeError("AUC validity differs between model and identical-target baseline.")
    return {
        "losses": losses,
        "aucs": model_aucs,
        "summary": auc_summary(model_aucs),
        "baseline_summary": auc_summary(baseline_aucs),
        "skipped_topics": skipped_topics,
    }


def write_report(
    refit_metrics: dict, reproduced: bool, results: dict[int, dict] | None = None
) -> None:
    """Write the Stage-1 gate result or completed architecture-ablation report."""
    emotion_ami = refit_metrics["emotion"]["AMI"]
    genre_ami = refit_metrics["genre"]["AMI"]
    lines = [
        "# ArtELingo PercepT Stage 2 architecture ablation pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Shared Stage-1 K=60/40 seed-42 reproduction\n\n",
        "Stage 1 was re-fit exactly once before the frozen `q > 1.2/40` "
        "targets and all four plain-linear runs. Every run shares that frozen "
        "encoder, 40 surviving centers, and global pooled CLIP image embeddings.\n\n",
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
            "**Reproducibility failure.** The shared Stage-1 re-fit was not within "
            f"the absolute AMI tolerance of {REPRODUCTION_TOLERANCE:.3f}; no "
            "plain-linear classifier was trained, so this ablation cannot silently "
            "use a different clustering.\n"
        )
        with open(REPORT_PATH, "w") as report_file:
            report_file.writelines(lines)
        return

    assert results is not None
    seed_macros = [results[classifier_seed]["summary"]["macro"] for classifier_seed in SEEDS]
    mean_macro = float(np.mean(seed_macros))
    min_macro = float(np.min(seed_macros))
    max_macro = float(np.max(seed_macros))
    attention_gap = ATTENTION_MEAN_MACRO_AUC - mean_macro
    ranges_overlap = not (
        ATTENTION_MIN_MACRO_AUC > max_macro
        or min_macro > ATTENTION_MAX_MACRO_AUC
    )

    lines.extend([
        "## Plain linear classifier on L2-normalized pooled CLIP embeddings\n\n",
        "Each row uses the shared frozen `q > 1.2/40` targets, a single "
        "`nn.Linear(512, 40)` on the existing L2-normalized global pooled CLIP "
        "image embeddings, `lr=3e-3`, and 100 full-batch epochs. Per-topic AUC "
        "scoring and the train-marginal baseline are imported from the Stage-2 "
        "sweep pilot.\n\n",
        "| classifier-init seed | held-out macro AUC | train-marginal macro AUC | min per-topic AUC | median per-topic AUC | max per-topic AUC | skipped topics |\n",
        "|---:|---:|---:|---:|---:|---:|---:|\n",
    ])
    lines.extend(
        f"| {classifier_seed} | {results[classifier_seed]['summary']['macro']:.4f} | "
        f"{results[classifier_seed]['baseline_summary']['macro']:.4f} | "
        f"{results[classifier_seed]['summary']['min']:.4f} | "
        f"{results[classifier_seed]['summary']['median']:.4f} | "
        f"{results[classifier_seed]['summary']['max']:.4f} | "
        f"{len(results[classifier_seed]['skipped_topics'])} |\n"
        for classifier_seed in SEEDS
    )
    lines.extend([
        "\n| seed-summary statistic | held-out macro AUC |\n",
        "|---|---:|\n",
        f"| mean | {mean_macro:.4f} |\n",
        f"| min | {min_macro:.4f} |\n",
        f"| max | {max_macro:.4f} |\n\n",
        "## Direct comparison with attention pooling\n\n",
        "The attention-pooling result is cited, not retrained here, from "
        "`percept_stage2_best_config_stress_pilot_report.md`. Both architectures "
        "use the same frozen `q > 1.2/40` targets, `lr=3e-3`, 100 epochs, and "
        "four initialization seeds.\n\n",
        "| architecture | held-out macro AUC mean | seed range |\n",
        "|---|---:|---|\n",
        f"| attention pooling over 50 patch tokens | {ATTENTION_MEAN_MACRO_AUC:.4f} | "
        f"{ATTENTION_MIN_MACRO_AUC:.4f}-{ATTENTION_MAX_MACRO_AUC:.4f} |\n",
        f"| plain linear on pooled embedding | {mean_macro:.4f} | "
        f"{min_macro:.4f}-{max_macro:.4f} |\n\n",
    ])
    if ATTENTION_MIN_MACRO_AUC > max_macro:
        lines.append(
            "**Verdict:** Patch attention pooling meaningfully outperforms the plain "
            "pooled-embedding linear baseline under the investigation's "
            "non-overlapping-range robustness standard. Its mean macro-AUC advantage "
            f"is {attention_gap:.4f}, and the ranges are cleanly separated: attention "
            f"{ATTENTION_MIN_MACRO_AUC:.4f}-{ATTENTION_MAX_MACRO_AUC:.4f} versus "
            f"plain linear {min_macro:.4f}-{max_macro:.4f}.\n"
        )
    elif min_macro > ATTENTION_MAX_MACRO_AUC:
        lines.append(
            "**Verdict:** Patch attention pooling does not outperform the plain "
            "pooled-embedding linear baseline under the investigation's "
            "non-overlapping-range robustness standard. The plain linear classifier "
            f"has a mean macro-AUC advantage of {-attention_gap:.4f}, with cleanly "
            f"separated ranges: plain linear {min_macro:.4f}-{max_macro:.4f} versus "
            f"attention {ATTENTION_MIN_MACRO_AUC:.4f}-{ATTENTION_MAX_MACRO_AUC:.4f}.\n"
        )
    else:
        lines.append(
            "**Verdict:** Patch attention pooling's mean macro-AUC gap is "
            f"{attention_gap:.4f} relative to the plain pooled-embedding linear "
            "baseline, but the seed ranges overlap "
            f"(attention {ATTENTION_MIN_MACRO_AUC:.4f}-{ATTENTION_MAX_MACRO_AUC:.4f}; "
            f"plain linear {min_macro:.4f}-{max_macro:.4f}). Under the established "
            "non-overlapping-range standard, this ablation does not establish that "
            "patch attention specifically provides a robust additional gain; the "
            "numbers may instead be largely explained by frozen topics being "
            "recoverable from the image at all.\n"
        )
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    """Re-fit Stage 1 once, then ablate Stage-2 patch attention."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    base = load_module("percept_stage1_base_for_arch_ablation", BASE_PILOT_PATH)
    cluster_sweep = load_module(
        "percept_stage1_sweep_for_arch_ablation", SWEEP_PILOT_PATH
    )
    pipeline = base.load_sibling_module(
        "artelingo_run_pipeline_percept_arch_ablation_train", base.PIPELINE_PATH
    )
    affect_pilot = base.load_sibling_module(
        "artelingo_run_affect_pilot_percept_arch_ablation", base.AFFECT_PILOT_PATH
    )
    cca_audit = base.load_sibling_module(
        "artelingo_run_cca_audit_percept_arch_ablation", base.CCA_AUDIT_PATH
    )
    heldout_pipeline = base.load_sibling_module(
        "artelingo_run_pipeline_percept_arch_ablation_heldout", base.PIPELINE_PATH
    )
    heldout_pipeline.STORAGE_DIR = base.HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = base.HELDOUT_JSON
    pipeline.assert_extraction_complete()
    heldout_pipeline.assert_extraction_complete()
    log = pipeline.log

    log("Loading and deduplicating train CLIP features for the shared Stage-1 re-fit...")
    paintings, img_nodes, txt_nodes, emotion_counts = pipeline.load_dedup_features()
    train_emotions = [pipeline.majority(counts) for counts in emotion_counts]
    log("Loading and deduplicating held-out CLIP features for the shared Stage-1 re-fit...")
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
    log(f"Using {device} for the shared Stage-1 re-fit and architecture ablation.")
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
    centers = cluster_sweep.initialize_cluster_centers(
        encoder, train_inputs, device, N_INITIAL_CLUSTERS, SEED
    )
    cluster_sweep.train_dec_until_stable(
        encoder,
        decoder,
        centers,
        train_inputs,
        device,
        log,
        N_INITIAL_CLUSTERS,
    )
    surviving_centers, surviving_indices = cluster_sweep.prune_centers(
        centers, N_SURVIVING_CLUSTERS
    )
    log(
        f"Pruned {N_INITIAL_CLUSTERS - N_SURVIVING_CLUSTERS} centers; retained "
        f"original indices {surviving_indices.tolist()}."
    )
    _, heldout_refit_metrics, _ = cluster_sweep.evaluate_run(
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

    threshold = THRESHOLD_MULTIPLIER / N_SURVIVING_CLUSTERS
    heldout_inputs = torch.from_numpy(heldout_h)
    train_targets = multi_hot_targets(
        encoder, surviving_centers, train_inputs, device, threshold
    ).to(device)
    heldout_targets = multi_hot_targets(
        encoder, surviving_centers, heldout_inputs, device, threshold
    ).to(device)
    log(f"Using frozen q > {threshold_label(THRESHOLD_MULTIPLIER)} multi-label targets.")

    train_embeddings = normalize_pooled_embeddings(
        img_nodes, len(paintings), "train"
    ).to(device)
    heldout_embeddings = normalize_pooled_embeddings(
        heldout_img_nodes, len(heldout_paintings), "held-out"
    ).to(device)
    results = {}
    for classifier_seed in SEEDS:
        results[classifier_seed] = train_and_evaluate_linear_classifier(
            train_embeddings,
            heldout_embeddings,
            train_targets,
            heldout_targets,
            classifier_seed,
            device,
            log,
        )

    write_report(heldout_refit_metrics, reproduced=True, results=results)
    log(f"Wrote report to {REPORT_PATH}.")


if __name__ == "__main__":
    main()
