"""Extend the PercepT Stage-2 winning mapper result to a 14-seed summary.

This standalone GPU script re-fits the fixed K=60/40 Stage-1 configuration
once, freezes its encoder and surviving centers, and measures ten new
mapper-initialization seeds for the established q > 1.2/40, lr=3e-3 setting.
The four prior mapper results are cited from the best-configuration stress
report rather than recomputed.
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
REPORT_OUT_DIR = os.environ.get("PERCEPT_OUTPUT_ROOT") or OUT_DIR
REPORT_PATH = os.path.join(
    REPORT_OUT_DIR, "percept_stage2_extended_seed_pilot_report.md"
)
SEED = 42
SEEDS = (1, 2, 3, 4, 5, 6, 8, 9, 10, 11)
THRESHOLD_MULTIPLIER = 1.2
MAPPER_LEARNING_RATE = 3e-3
MAPPER_EPOCHS = 100
EXPECTED_HELDOUT_EMOTION_AMI = 0.1238
EXPECTED_HELDOUT_GENRE_AMI = 0.2617
BASELINE_MACRO_AUC = 0.5000
ORIGINAL_THRESHOLD_MAX_MACRO_AUC = 0.5760
CITED_RESULTS = (
    (42, 0.8256, 0.5728, 0.8288, 0.9605),
    (7, 0.8248, 0.5366, 0.8299, 0.9595),
    (123, 0.8272, 0.5690, 0.8246, 0.9606),
    (2024, 0.8249, 0.5479, 0.8331, 0.9593),
)


def cited_result(
    macro: float, minimum: float, median: float, maximum: float
) -> dict:
    """Build the mapper-result shape used for a cited prior run."""
    return {
        "summary": {
            "macro": macro,
            "min": minimum,
            "median": median,
            "max": maximum,
        },
        "skipped_topics": [],
    }


def write_report(
    refit_metrics: dict,
    reproduced: bool,
    new_results: dict[int, dict] | None = None,
) -> None:
    """Write the Stage-1 gate result or the combined cited-and-new seed report."""
    emotion_ami = refit_metrics["emotion"]["AMI"]
    genre_ami = refit_metrics["genre"]["AMI"]
    lines = [
        "# ArtELingo PercepT Stage 2 extended seed pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Shared Stage-1 K=60/40 seed-42 reproduction\n\n",
        "Stage 1 was re-fit exactly once before the frozen `q > 1.2/40` "
        "targets and all ten newly measured mapper runs. Every mapper run shares "
        "that frozen encoder, 40 surviving centers, and cached patch features.\n\n",
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
        os.makedirs(REPORT_OUT_DIR, exist_ok=True)
        with open(REPORT_PATH, "w") as report_file:
            report_file.writelines(lines)
        return

    assert new_results is not None
    results = [
        {
            "seed": seed,
            "source": "cited from percept_stage2_best_config_stress_pilot_report.md",
            "result": cited_result(macro, minimum, median, maximum),
        }
        for seed, macro, minimum, median, maximum in CITED_RESULTS
    ]
    results.extend(
        {
            "seed": seed,
            "source": "newly measured",
            "result": new_results[seed],
        }
        for seed in SEEDS
    )
    seed_macros = [entry["result"]["summary"]["macro"] for entry in results]
    mean_macro = float(np.mean(seed_macros))
    min_macro = float(np.min(seed_macros))
    max_macro = float(np.max(seed_macros))
    stdev_macro = float(np.std(seed_macros, ddof=1))
    ci_half_width = 1.96 * stdev_macro / np.sqrt(len(seed_macros))
    ci_lower = mean_macro - ci_half_width
    ci_upper = mean_macro + ci_half_width
    all_beat_baseline = all(macro > BASELINE_MACRO_AUC for macro in seed_macros)
    all_beat_original_threshold = all(
        macro > ORIGINAL_THRESHOLD_MAX_MACRO_AUC for macro in seed_macros
    )

    lines.extend([
        "## Full 14-seed winning-configuration results\n\n",
        "Each row uses the shared frozen `q > 1.2/40` targets, `lr=3e-3`, and "
        f"{MAPPER_EPOCHS} epochs. The four cited rows reproduce the held-out macro "
        "and per-topic AUC values recorded in "
        "`percept_stage2_best_config_stress_pilot_report.md`; only the remaining "
        "ten mapper-init seeds are trained here. `AttentionPoolingMapper` and "
        "`train_and_evaluate_mapper()` are imported directly from the Stage-2 sweep "
        "pilot.\n\n",
        "| mapper-init seed | source | held-out macro AUC | min per-topic AUC | median per-topic AUC | max per-topic AUC | verdict |\n",
        "|---:|---|---:|---:|---:|---:|---|\n",
    ])
    for entry in results:
        summary = entry["result"]["summary"]
        verdict = (
            "exceeds 0.5000 baseline"
            if summary["macro"] > BASELINE_MACRO_AUC
            else "does not exceed 0.5000 baseline"
        )
        lines.append(
            f"| {entry['seed']} | {entry['source']} | {summary['macro']:.4f} | "
            f"{summary['min']:.4f} | {summary['median']:.4f} | "
            f"{summary['max']:.4f} | {verdict} |\n"
        )

    lines.extend([
        "\n## Held-out macro-AUC summary\n\n",
        "| statistic | held-out macro AUC |\n",
        "|---|---:|\n",
        f"| mean | {mean_macro:.4f} |\n",
        f"| min | {min_macro:.4f} |\n",
        f"| max | {max_macro:.4f} |\n",
        f"| sample standard deviation | {stdev_macro:.4f} |\n\n",
        "## Normal-approximation 95% confidence interval\n\n",
        f"Across all {len(seed_macros)} seeds, the held-out macro-AUC mean is "
        f"{mean_macro:.4f}. Using the requested simple normal approximation, the "
        f"95% confidence interval is {mean_macro:.4f} ± {ci_half_width:.4f} "
        f"(1.96 × {stdev_macro:.4f} / sqrt({len(seed_macros)})), or "
        f"[{ci_lower:.4f}, {ci_upper:.4f}]. ",
    ])
    if ci_lower > BASELINE_MACRO_AUC:
        lines.append(
            f"Its lower bound is {ci_lower - BASELINE_MACRO_AUC:.4f} above the "
            f"0.5000 baseline — dramatically above baseline. This pilot exists to "
            "confirm that robustness on a larger sample, not to find a new result.\n\n"
        )
    else:
        lines.append(
            "Its lower bound does not exceed the 0.5000 baseline, so the larger "
            "sample does not confirm the expected robustness.\n\n"
        )

    lines.append("## Verdict\n\n")
    if all_beat_baseline and all_beat_original_threshold:
        lines.append(
            "**Verdict:** The richer multi-label threshold plus tuned learning rate "
            "is seed-robust relative to the original threshold: all 14 runs exceed "
            f"the original four-seed maximum of {ORIGINAL_THRESHOLD_MAX_MACRO_AUC:.4f}. "
            "The substantially larger seed sample confirms the result is robust and "
            "reproducible rather than being consumed by mapper-init variance.\n"
        )
    elif all_beat_baseline:
        lines.append(
            "**Verdict:** The result remains baseline-beating across all 14 seeds, "
            "but seed robustness relative to the original threshold is not established "
            "because at least one run does not exceed its four-seed maximum.\n"
        )
    else:
        lines.append(
            "**Verdict:** Seed robustness is not established: at least one of the "
            "14 mapper initializations does not exceed the 0.5000 baseline.\n"
        )
    os.makedirs(REPORT_OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    """Re-fit Stage 1 once, then measure the ten new mapper-init seeds."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    base = load_module("percept_stage1_base_for_extended_seed", BASE_PILOT_PATH)
    cluster_sweep = load_module(
        "percept_stage1_sweep_for_extended_seed", SWEEP_PILOT_PATH
    )
    pipeline = base.load_sibling_module(
        "artelingo_run_pipeline_percept_extended_seed_train", base.PIPELINE_PATH
    )
    affect_pilot = base.load_sibling_module(
        "artelingo_run_affect_pilot_percept_extended_seed", base.AFFECT_PILOT_PATH
    )
    cca_audit = base.load_sibling_module(
        "artelingo_run_cca_audit_percept_extended_seed", base.CCA_AUDIT_PATH
    )
    heldout_pipeline = base.load_sibling_module(
        "artelingo_run_pipeline_percept_extended_seed_heldout", base.PIPELINE_PATH
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
    log(f"Using {device} for the shared Stage-1 re-fit and Stage-2 extended seed pilot.")
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

    new_results = {}
    for mapper_seed in SEEDS:
        new_results[mapper_seed] = train_and_evaluate_mapper(
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

    write_report(heldout_refit_metrics, reproduced=True, new_results=new_results)
    log(f"Wrote report to {REPORT_PATH}.")


if __name__ == "__main__":
    main()
