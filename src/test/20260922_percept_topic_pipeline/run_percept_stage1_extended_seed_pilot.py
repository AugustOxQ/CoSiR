"""Extend the PercepT Stage 1 K=60/40 seed stress to a 14-seed summary.

The four established results are cited from the V2 cluster-count sweep report.
This runner measures ten deliberately new seeds with the unchanged K=60/40
training mechanics, then writes their combined 14-seed report.
"""

import importlib.util
import os
import time

import numpy as np
import torch


OUT_DIR = os.path.dirname(__file__)
REPORT_OUT_DIR = os.environ.get("PERCEPT_OUTPUT_ROOT") or OUT_DIR
TEMPLATE_PATH = os.path.join(
    OUT_DIR, "run_percept_stage1_cluster_count_sweep_v2_pilot.py"
)
REPORT_PATH = os.path.join(REPORT_OUT_DIR, "percept_stage1_extended_seed_pilot_report.md")
N_INITIAL_CLUSTERS = 60
N_SURVIVING_CLUSTERS = 40
SEEDS = (1, 2, 3, 4, 5, 6, 8, 9, 10, 11)
EMOTION_PARETO_BAR = 0.1236
GENRE_PARETO_BAR = 0.1954
CITED_RESULTS = (
    (42, 0.1238, 0.2617),
    (7, 0.1252, 0.2507),
    (123, 0.1272, 0.2328),
    (2024, 0.1246, 0.2491),
)


def load_template():
    """Load the V2 pilot without entering its main block."""
    spec = importlib.util.spec_from_file_location(
        "percept_stage1_cluster_count_sweep_v2", TEMPLATE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import V2 pilot from {TEMPLATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


template = load_template()
cluster_sweep = template.template
base = cluster_sweep.base


def metrics(emotion_ami: float, genre_ami: float) -> dict:
    """Build the report-only metric shape for a cited held-out result.

    `base.verdict()` unconditionally reads `metrics["collapsed"]` first; all
    four cited seeds are already established non-collapsed "Real success"
    results in the source report, so this is set accordingly rather than
    omitted.
    """
    return {
        "emotion": {"AMI": emotion_ami},
        "genre": {"AMI": genre_ami},
        "collapsed": False,
    }


def clears(metrics_: dict) -> bool:
    """Apply the established strict held-out Pareto thresholds."""
    return (
        metrics_["emotion"]["AMI"] > EMOTION_PARETO_BAR
        and metrics_["genre"]["AMI"] > GENRE_PARETO_BAR
    )


def miss_description(metrics_: dict) -> str:
    """Describe every strict-bar miss without rounding it away."""
    misses = []
    emotion_margin = metrics_["emotion"]["AMI"] - EMOTION_PARETO_BAR
    genre_margin = metrics_["genre"]["AMI"] - GENRE_PARETO_BAR
    if emotion_margin <= 0:
        misses.append(f"emotion by {abs(emotion_margin):.4f}")
    if genre_margin <= 0:
        misses.append(f"genre by {abs(genre_margin):.4f}")
    return " and ".join(misses)


def summary(values: list[float]) -> str:
    """Format the requested descriptive statistics using sample standard deviation."""
    return (
        f"mean={np.mean(values):.4f}; min={np.min(values):.4f}; "
        f"max={np.max(values):.4f}; stdev={np.std(values, ddof=1):.4f}"
    )


def write_report(results: list[dict]) -> None:
    """Write the predeclared combined cited-and-new 14-seed report."""
    heldout_emotion = [result["heldout_metrics"]["emotion"]["AMI"] for result in results]
    heldout_genre = [result["heldout_metrics"]["genre"]["AMI"] for result in results]
    emotion_clearers = sum(value > EMOTION_PARETO_BAR for value in heldout_emotion)
    genre_clearers = sum(value > GENRE_PARETO_BAR for value in heldout_genre)
    both_clearers = sum(clears(result["heldout_metrics"]) for result in results)
    n_results = len(results)
    emotion_mean = float(np.mean(heldout_emotion))
    emotion_stdev = float(np.std(heldout_emotion, ddof=1))
    ci_half_width = 1.96 * emotion_stdev / np.sqrt(n_results)
    ci_lower = emotion_mean - ci_half_width
    ci_upper = emotion_mean + ci_half_width
    misses = [
        (result["seed"], miss_description(result["heldout_metrics"]))
        for result in results
        if not clears(result["heldout_metrics"])
    ]

    lines = [
        "# ArtELingo PercepT Stage 1 extended seed pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Controlled setup\n\n",
        "This pilot holds the standing K=60/40 configuration fixed: "
        "`N_INITIAL_CLUSTERS=60`, `N_SURVIVING_CLUSTERS=40`, "
        "`LAMBDA_BALANCE=1000`, and `LAMBDA_RECONSTRUCTION=1`. Each newly "
        "measured seed receives fresh Torch and CUDA seeding (when available), "
        "a fresh 100-epoch autoencoder pretrain, `KMeans(random_state=seed)`, "
        "DEC training to the established convergence criterion, and 40-of-60 "
        "center pruning. The four established seeds are cited from "
        "`percept_stage1_cluster_count_sweep_v2_pilot_report.md`, not recomputed.\n\n",
        "## Predeclared success criterion\n\n",
        "**held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, "
        "both simultaneously.**\n\n",
        "## Full 14-seed held-out results\n\n",
        "| seed | source | held-out emotion AMI | held-out genre AMI | verdict | held-out Pareto bar |\n",
        "|---:|---|---:|---:|---|---|\n",
    ]
    for result in results:
        heldout_metrics = result["heldout_metrics"]
        pareto = "clears" if clears(heldout_metrics) else "does not clear"
        lines.append(
            f"| {result['seed']} | {result['source']} | "
            f"{heldout_metrics['emotion']['AMI']:.4f} | "
            f"{heldout_metrics['genre']['AMI']:.4f} | "
            f"{base.verdict(heldout_metrics)} | {pareto} |\n"
        )

    lines.extend([
        "\n## Held-out summary statistics\n\n",
        f"- Emotion AMI across {n_results} seeds: {summary(heldout_emotion)}.\n",
        f"- Genre AMI across {n_results} seeds: {summary(heldout_genre)}.\n",
        f"- Emotion clears its individual bar in {emotion_clearers}/{n_results} "
        f"seeds ({emotion_clearers / n_results:.1%}).\n",
        f"- Genre clears its individual bar in {genre_clearers}/{n_results} seeds "
        f"({genre_clearers / n_results:.1%}).\n",
        f"- Both bars clear simultaneously in {both_clearers}/{n_results} seeds "
        f"({both_clearers / n_results:.1%}).\n\n",
        "## Emotion confidence interval\n\n",
        "The tighter-margin emotion axis is the statistical focus of this pilot. "
        f"Its {n_results}-seed mean is {emotion_mean:.4f}; using the sample "
        "standard deviation and the requested simple normal approximation, its "
        f"95% confidence interval is {emotion_mean:.4f} ± {ci_half_width:.4f}, "
        f"or [{ci_lower:.4f}, {ci_upper:.4f}]. "
        + (
            f"The lower bound clears the 0.1236 threshold by {ci_lower - EMOTION_PARETO_BAR:.4f}.\n\n"
            if ci_lower > EMOTION_PARETO_BAR
            else f"The lower bound does not clear the 0.1236 threshold; it misses by {EMOTION_PARETO_BAR - ci_lower:.4f}.\n\n"
        ),
        "## Decision\n\n",
    ])
    if both_clearers == n_results:
        lines.append(
            "**Real success: robust on a substantially larger sample.** All 14/14 "
            "seeds clear both held-out Pareto bars. This is stronger evidence than "
            "the earlier four-seed seed-robust result, while the emotion confidence "
            "interval states directly whether the narrow-margin axis clears its bar.\n"
        )
    else:
        miss_lines = "; ".join(
            f"seed {seed} misses {description}" for seed, description in misses
        )
        lines.append(
            f"**Seed-dependent result.** {both_clearers}/{n_results} seeds clear "
            f"both held-out Pareto bars; {n_results - both_clearers}/{n_results} "
            f"miss. The misses are: {miss_lines}.\n"
        )

    os.makedirs(REPORT_OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    """Measure the ten new fully rerandomized K=60/40 seed-stress runs."""
    pipeline = base.load_sibling_module(
        "artelingo_run_pipeline_percept_train", base.PIPELINE_PATH
    )
    affect_pilot = base.load_sibling_module(
        "artelingo_run_affect_pilot_percept", base.AFFECT_PILOT_PATH
    )
    cca_audit = base.load_sibling_module(
        "artelingo_run_cca_audit_percept", base.CCA_AUDIT_PATH
    )
    heldout_pipeline = base.load_sibling_module(
        "artelingo_run_pipeline_percept_heldout", base.PIPELINE_PATH
    )
    heldout_pipeline.STORAGE_DIR = base.HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = base.HELDOUT_JSON

    pipeline.assert_extraction_complete()
    heldout_pipeline.assert_extraction_complete()
    log = pipeline.log
    log("Loading and deduplicating train CLIP features...")
    paintings, img_nodes, txt_nodes, emotion_counts = pipeline.load_dedup_features()
    train_emotions = [pipeline.majority(counts) for counts in emotion_counts]
    log("Loading and deduplicating held-out CLIP features...")
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
    log(f"Using {device} for embedding extraction, pretraining, and DEC training.")
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
            f"Train/held-out fused dimensions differ: "
            f"{train_h.shape[1]} vs {heldout_h.shape[1]}."
        )
    train_inputs = torch.from_numpy(train_h)

    results = [
        {
            "seed": seed,
            "source": "cited from V2 Phase 2 table",
            "heldout_metrics": metrics(emotion_ami, genre_ami),
        }
        for seed, emotion_ami, genre_ami in CITED_RESULTS
    ]
    for seed in SEEDS:
        log(
            f"Starting fresh extended-seed run for seed={seed}, "
            f"K={N_INITIAL_CLUSTERS}/{N_SURVIVING_CLUSTERS}."
        )
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        encoder, decoder = base.build_autoencoder(train_h.shape[1])
        encoder.to(device)
        decoder.to(device)
        base.pretrain_autoencoder(encoder, decoder, train_inputs, device, log)
        centers = cluster_sweep.initialize_cluster_centers(
            encoder, train_inputs, device, N_INITIAL_CLUSTERS, seed
        )
        _, stop_reason, stop_epoch = cluster_sweep.train_dec_until_stable(
            encoder,
            decoder,
            centers,
            train_inputs,
            device,
            log,
            N_INITIAL_CLUSTERS,
        )
        train_metrics, heldout_metrics, surviving_indices = cluster_sweep.evaluate_run(
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
        log(
            f"seed={seed}: stopped at epoch {stop_epoch} via {stop_reason}; "
            f"pruned {N_INITIAL_CLUSTERS - N_SURVIVING_CLUSTERS} centers and "
            f"retained original indices {surviving_indices.tolist()}."
        )
        results.append(
            {
                "seed": seed,
                "source": "newly measured",
                "train_metrics": train_metrics,
                "heldout_metrics": heldout_metrics,
            }
        )

    write_report(results)
    log(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
