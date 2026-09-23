"""Seed-stress the winning PercepT Stage-2 mapper configuration on GPU.

This standalone script re-fits the fixed K=60/40 Stage-1 configuration once,
freezes its encoder and surviving centers, and then evaluates the winning
q > 1.2/40, lr=3e-3 mapper configuration across four mapper-init seeds.
"""

import os
import time

import numpy as np
import torch

from run_percept_stage2_sweep_pilot import (
    AttentionPoolingMapper,
    BASE_PILOT_PATH,
    HELDOUT_PATCH_FEATURE_PATH,
    N_INITIAL_CLUSTERS,
    N_SURVIVING_CLUSTERS,
    PATCH_FEATURE_DIR,
    REPRODUCTION_TOLERANCE,
    SWEEP_PILOT_PATH,
    TRAIN_PATCH_FEATURE_PATH,
    load_module,
    load_patch_features,
    multi_hot_targets,
    threshold_label,
    train_and_evaluate_mapper,
)


OUT_DIR = os.path.dirname(__file__)
REPORT_PATH = os.path.join(
    OUT_DIR, "percept_stage2_best_config_stress_pilot_report.md"
)
SEED = 42
SEEDS = (42, 7, 123, 2024)
THRESHOLD_MULTIPLIER = 1.2
MAPPER_LEARNING_RATE = 3e-3
MAPPER_EPOCHS = 100
EXPECTED_HELDOUT_EMOTION_AMI = 0.1238
EXPECTED_HELDOUT_GENRE_AMI = 0.2617
BASELINE_MACRO_AUC = 0.5000
ORIGINAL_THRESHOLD_MEAN_MACRO_AUC = 0.5709
ORIGINAL_THRESHOLD_MIN_MACRO_AUC = 0.5644
ORIGINAL_THRESHOLD_MAX_MACRO_AUC = 0.5760


def write_report(refit_metrics: dict, reproduced: bool, results: dict[int, dict] | None = None) -> None:
    """Write the Stage-1 gate result or the completed four-seed stress report."""
    emotion_ami = refit_metrics["emotion"]["AMI"]
    genre_ami = refit_metrics["genre"]["AMI"]
    lines = [
        "# ArtELingo PercepT Stage 2 best-configuration seed-stress pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Shared Stage-1 K=60/40 seed-42 reproduction\n\n",
        "Stage 1 was re-fit exactly once before the frozen `q > 1.2/40` "
        "targets and all four mapper runs. Every run shares that frozen encoder, "
        "40 surviving centers, and cached patch features.\n\n",
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
            f"the absolute AMI tolerance of {REPRODUCTION_TOLERANCE:.3f}; no Stage-2 "
            "mapper was trained, so this pilot cannot silently use a different clustering.\n"
        )
        with open(REPORT_PATH, "w") as report_file:
            report_file.writelines(lines)
        return

    assert results is not None
    seed_macros = [results[mapper_seed]["summary"]["macro"] for mapper_seed in SEEDS]
    mean_macro = float(np.mean(seed_macros))
    min_macro = float(np.min(seed_macros))
    max_macro = float(np.max(seed_macros))
    spread = max_macro - min_macro
    seed_42_macro = results[42]["summary"]["macro"]
    robust_improvement = min_macro > ORIGINAL_THRESHOLD_MAX_MACRO_AUC
    seed_42_position = "above" if seed_42_macro >= mean_macro else "below"

    lines.extend([
        "## Winning configuration mapper-init seed stress test\n\n",
        "Each row uses the shared frozen `q > 1.2/40` targets, `lr=3e-3`, and "
        f"{MAPPER_EPOCHS} epochs. `AttentionPoolingMapper` and "
        "`train_and_evaluate_mapper()` are imported directly from the Stage-2 "
        "sweep pilot.\n\n",
        "| mapper-init seed | held-out macro AUC | min per-topic AUC | median per-topic AUC | max per-topic AUC | skipped topics |\n",
        "|---:|---:|---:|---:|---:|---:|\n",
    ])
    lines.extend(
        f"| {mapper_seed} | {results[mapper_seed]['summary']['macro']:.4f} | "
        f"{results[mapper_seed]['summary']['min']:.4f} | "
        f"{results[mapper_seed]['summary']['median']:.4f} | "
        f"{results[mapper_seed]['summary']['max']:.4f} | "
        f"{len(results[mapper_seed]['skipped_topics'])} |\n"
        for mapper_seed in SEEDS
    )
    lines.extend([
        "\n| seed-summary statistic | held-out macro AUC |\n",
        "|---|---:|\n",
        f"| mean | {mean_macro:.4f} |\n",
        f"| min | {min_macro:.4f} |\n",
        f"| max | {max_macro:.4f} |\n\n",
    ])
    if robust_improvement:
        lines.append(
            "**Verdict:** The richer multi-label threshold plus tuned learning rate "
            "is seed-robust relative to the original threshold: all four runs exceed "
            f"the original four-seed maximum of {ORIGINAL_THRESHOLD_MAX_MACRO_AUC:.4f}. "
            f"The actual macro-AUC spread is {min_macro:.4f}-{max_macro:.4f} "
            f"(width {spread:.4f}); seed 42's {seed_42_macro:.4f} is "
            f"{abs(seed_42_macro - mean_macro):.4f} {seed_42_position} the four-seed mean.\n\n"
        )
    else:
        lines.append(
            "**Verdict:** Seed robustness is not established: at least one rich-target "
            "run does not exceed the original threshold's four-seed maximum of "
            f"{ORIGINAL_THRESHOLD_MAX_MACRO_AUC:.4f}. The actual macro-AUC spread is "
            f"{min_macro:.4f}-{max_macro:.4f} (width {spread:.4f}); seed 42's "
            f"{seed_42_macro:.4f} is {abs(seed_42_macro - mean_macro):.4f} "
            f"{seed_42_position} the four-seed mean, so its 0.8256 sweep result must "
            "not be treated as a seed-independent estimate.\n\n"
        )
    lines.append(
        "## Comparison with the original threshold seed stress test\n\n"
        "The original `q > 2.0/40`, `lr=1e-3` configuration had four-seed held-out "
        f"macro AUC mean {ORIGINAL_THRESHOLD_MEAN_MACRO_AUC:.4f} and range "
        f"{ORIGINAL_THRESHOLD_MIN_MACRO_AUC:.4f}-{ORIGINAL_THRESHOLD_MAX_MACRO_AUC:.4f}. "
        + (
            "Because the rich-target configuration's worst seed still exceeds that "
            "original range, its improvement is robust and reproducible rather than "
            "being consumed by mapper-init variance."
            if robust_improvement
            else "Because its seed range overlaps or falls below the original range, "
            "mapper-init variance eats into the apparent gain; the seed-42 maximum "
            "alone is not enough to establish a robust improvement."
        )
        + "\n"
    )
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    """Re-fit Stage 1 once, then seed-stress its frozen best Stage-2 configuration."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    base = load_module("percept_stage1_base_for_best_config_stress", BASE_PILOT_PATH)
    cluster_sweep = load_module(
        "percept_stage1_sweep_for_best_config_stress", SWEEP_PILOT_PATH
    )
    pipeline = base.load_sibling_module(
        "artelingo_run_pipeline_percept_best_config_stress_train", base.PIPELINE_PATH
    )
    affect_pilot = base.load_sibling_module(
        "artelingo_run_affect_pilot_percept_best_config_stress", base.AFFECT_PILOT_PATH
    )
    cca_audit = base.load_sibling_module(
        "artelingo_run_cca_audit_percept_best_config_stress", base.CCA_AUDIT_PATH
    )
    heldout_pipeline = base.load_sibling_module(
        "artelingo_run_pipeline_percept_best_config_stress_heldout", base.PIPELINE_PATH
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
    log(f"Using {device} for the shared Stage-1 re-fit and Stage-2 seed stress test.")
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

    train_patch_features = load_patch_features(
        TRAIN_PATCH_FEATURE_PATH, len(paintings), "train"
    ).to(device)
    heldout_patch_features = load_patch_features(
        HELDOUT_PATCH_FEATURE_PATH, len(heldout_paintings), "held-out"
    ).to(device)

    results = {}
    for mapper_seed in SEEDS:
        results[mapper_seed] = train_and_evaluate_mapper(
            train_patch_features,
            heldout_patch_features,
            train_targets,
            heldout_targets,
            mapper_seed,
            MAPPER_LEARNING_RATE,
            device,
            log,
            f"q > {threshold_label(THRESHOLD_MULTIPLIER)} lr={MAPPER_LEARNING_RATE:.0e} seed {mapper_seed}",
        )

    write_report(heldout_refit_metrics, reproduced=True, results=results)
    log(f"Wrote report to {REPORT_PATH}.")


if __name__ == "__main__":
    main()
