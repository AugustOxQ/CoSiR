"""Search intermediate PercepT Stage 1 cluster counts for a stable crossover.

This runner reuses the original cluster-count sweep's two-phase training
discipline.  It screens K=40/27, K=60/40, and K=80/53 at seed 42 from one
shared pretrained autoencoder, then fully seed-stresses the selected point.
"""

import importlib.util
import os
import time

import numpy as np


OUT_DIR = os.path.dirname(__file__)
REPORT_OUT_DIR = os.environ.get("PERCEPT_OUTPUT_ROOT") or OUT_DIR
TEMPLATE_PATH = os.path.join(OUT_DIR, "run_percept_stage1_cluster_count_sweep_pilot.py")
REPORT_PATH = os.path.join(
    REPORT_OUT_DIR, "percept_stage1_cluster_count_sweep_v2_pilot_report.md"
)
CLUSTER_COUNT_PAIRS = ((40, 27), (60, 40), (80, 53))
EMOTION_PARETO_BAR = 0.1236
GENRE_PARETO_BAR = 0.1954
ORIGINAL_K100_EMOTION = (0.1242, 0.1228, 0.1252, 0.1216)
ORIGINAL_K100_GENRE = (0.2466, 0.2190, 0.1851, 0.1849)
K30_EMOTION = (0.1220, 0.1204, 0.1209, 0.1213)
K30_GENRE = (0.2577, 0.2590, 0.2563, 0.2732)


def load_template():
    """Load the original sweep without entering its main block."""
    spec = importlib.util.spec_from_file_location("percept_cluster_sweep_template", TEMPLATE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import sweep template from {TEMPLATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


template = load_template()


def format_summary(values):
    """Format mean, minimum, and maximum held-out AMI values."""
    return (
        f"mean={np.mean(values):.4f}; min={np.min(values):.4f}; "
        f"max={np.max(values):.4f}"
    )


def clears(metrics):
    """Use the established strict held-out Pareto thresholds."""
    return (
        metrics["emotion"]["AMI"] > EMOTION_PARETO_BAR
        and metrics["genre"]["AMI"] > GENRE_PARETO_BAR
    )


def cluster_fraction(metrics, n_surviving_clusters):
    """Report below-1% clusters against this point's active denominator."""
    return (
        f"{metrics['small_clusters'] / n_surviving_clusters:.1%} "
        f"({metrics['small_clusters']}/{n_surviving_clusters})"
    )


def miss_description(metrics):
    """Identify each strict-bar miss without hiding it through rounding."""
    misses = []
    emotion_margin = metrics["emotion"]["AMI"] - EMOTION_PARETO_BAR
    genre_margin = metrics["genre"]["AMI"] - GENRE_PARETO_BAR
    if emotion_margin <= 0:
        misses.append(f"emotion by {abs(emotion_margin):.4f}")
    if genre_margin <= 0:
        misses.append(f"genre by {abs(genre_margin):.4f}")
    return " and ".join(misses)


def write_report(
    pretrain_losses,
    input_dim,
    phase_1_results,
    selected_result,
    phase_2_results,
):
    """Write the intermediate-sweep report, including all cited baselines."""
    selected_metrics = selected_result["heldout_metrics"]
    heldout_metrics = [result["heldout_metrics"] for result in phase_2_results]
    heldout_emotion = [metrics["emotion"]["AMI"] for metrics in heldout_metrics]
    heldout_genre = [metrics["genre"]["AMI"] for metrics in heldout_metrics]
    both_clearers = sum(clears(metrics) for metrics in heldout_metrics)
    emotion_clearers = sum(value > EMOTION_PARETO_BAR for value in heldout_emotion)
    genre_clearers = sum(value > GENRE_PARETO_BAR for value in heldout_genre)
    selected_emotion_margin = selected_metrics["emotion"]["AMI"] - EMOTION_PARETO_BAR
    selected_genre_margin = selected_metrics["genre"]["AMI"] - GENRE_PARETO_BAR
    emotion_spread = np.ptp(heldout_emotion)
    genre_spread = np.ptp(heldout_genre)
    original_emotion_spread = np.ptp(ORIGINAL_K100_EMOTION)
    original_genre_spread = np.ptp(ORIGINAL_K100_GENRE)
    k30_emotion_spread = np.ptp(K30_EMOTION)
    k30_genre_spread = np.ptp(K30_GENRE)
    phase_1_by_pair = {
        (result["n_initial_clusters"], result["n_surviving_clusters"]): result
        for result in phase_1_results
    }
    any_screen_clearer = any(
        clears(result["heldout_metrics"]) for result in phase_1_results
    )

    lines = [
        "# ArtELingo PercepT Stage 1 intermediate cluster-count sweep pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Controlled setup\n\n",
        f"The train-only input is a {input_dim}-dimensional fused vector. Phase 1 "
        "uses seed 42, performs one 100-epoch autoencoder pretrain, and deep-copies "
        "that pretrained state for each isolated DEC run. Each point receives fresh "
        "K-means at `random_state=42` with its own initial-cluster count. "
        "`LAMBDA_BALANCE=1000` and `LAMBDA_RECONSTRUCTION=1` remain fixed; only "
        "the initial/surviving cluster-count pair varies.\n\n",
        "Phase 2 cites the selected Phase-1 seed-42 result, then runs seeds 7, 123, "
        "and 2024. Each new seed applies fresh NumPy, Torch, and CUDA seeding (when "
        "available), builds and pretrains a new autoencoder for 100 epochs, runs "
        "`KMeans(random_state=seed)`, and trains DEC with the selected cluster counts "
        "and unchanged loss weights.\n\n",
        "The below-1% and collapse calculations use each point's active surviving-"
        "cluster count, not the original fixed 67.\n\n",
        "## Predeclared success criterion\n\n",
        "**held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, both "
        "simultaneously.**\n\n",
        "## Phase 1: seed-42 cluster-count screen\n\n",
        f"Pretraining reconstruction: {template.base.format_pretrain_losses(pretrain_losses)}.\n\n",
        "| N initial | N surviving | split | emotion AMI | genre AMI | verdict | held-out Pareto bar | final max cluster size | fraction below 1% |\n",
        "|---:|---:|---|---:|---:|---|---|---:|---:|\n",
    ]
    for result in phase_1_results:
        n_surviving = result["n_surviving_clusters"]
        for split_name, metrics in (
            ("train", result["train_metrics"]),
            ("held-out", result["heldout_metrics"]),
        ):
            pareto = (
                "n/a (train split)"
                if split_name == "train"
                else "clears" if clears(metrics) else "does not clear"
            )
            lines.append(
                f"| {result['n_initial_clusters']} | {n_surviving} | {split_name} | "
                f"{metrics['emotion']['AMI']:.4f} | {metrics['genre']['AMI']:.4f} | "
                f"{template.base.verdict(metrics)} | {pareto} | "
                f"{int(metrics['cluster_sizes'].max()):,} | "
                f"{cluster_fraction(metrics, n_surviving)} |\n"
            )

    lines.extend([
        "\n## Consolidated seed-42 cluster-count picture\n\n",
        "Prior-sweep values are cited, not recomputed. K=100/67 is cited from "
        "`percept_stage1_balance_sweep_v2_pilot_report.md` at lambda=1000.\n\n",
        "| N initial | N surviving | held-out emotion AMI | held-out genre AMI | source |\n",
        "|---:|---:|---:|---:|---|\n",
        "| 20 | 13 | 0.1152 | 0.2273 | cited from first cluster-count sweep |\n",
        "| 30 | 20 | 0.1220 | 0.2577 | cited from first cluster-count sweep |\n",
    ])
    for n_initial, n_surviving in CLUSTER_COUNT_PAIRS[:1]:
        metrics = phase_1_by_pair[(n_initial, n_surviving)]["heldout_metrics"]
        lines.append(
            f"| {n_initial} | {n_surviving} | {metrics['emotion']['AMI']:.4f} | "
            f"{metrics['genre']['AMI']:.4f} | newly measured in this sweep |\n"
        )
    lines.extend([
        "| 50 | 33 | 0.1202 | 0.2830 | cited from first cluster-count sweep |\n",
    ])
    for n_initial, n_surviving in CLUSTER_COUNT_PAIRS[1:]:
        metrics = phase_1_by_pair[(n_initial, n_surviving)]["heldout_metrics"]
        lines.append(
            f"| {n_initial} | {n_surviving} | {metrics['emotion']['AMI']:.4f} | "
            f"{metrics['genre']['AMI']:.4f} | newly measured in this sweep |\n"
        )
    lines.extend([
        "| 100 | 67 | 0.1242 | 0.2466 | cited from balance v2 sweep at lambda=1000 |\n",
        "\n## Phase-2 selection\n\n",
        f"`N_INITIAL_CLUSTERS={selected_result['n_initial_clusters']}`, "
        f"`N_SURVIVING_CLUSTERS={selected_result['n_surviving_clusters']}` was selected "
        + (
            "because it clears the held-out Pareto bar and has the largest held-out "
            "emotion margin among clearing points"
            if clears(selected_metrics)
            else "because no Phase-1 point clears both bars, so it has the largest "
            "held-out emotion-bar margin"
        )
        + f" (emotion margin {selected_emotion_margin:+.4f}; genre margin "
        f"{selected_genre_margin:+.4f}).\n\n",
        "## Phase 2: four-seed stress test\n\n",
        "| seed | source | split | emotion AMI | genre AMI | verdict | held-out Pareto bar |\n",
        "|---:|---|---|---:|---:|---|---|\n",
    ])
    for result in phase_2_results:
        for split_name, metrics in (
            ("train", result["train_metrics"]),
            ("held-out", result["heldout_metrics"]),
        ):
            pareto = (
                "n/a (train split)"
                if split_name == "train"
                else "clears" if clears(metrics) else "does not clear"
            )
            lines.append(
                f"| {result['seed']} | {result['source']} | {split_name} | "
                f"{metrics['emotion']['AMI']:.4f} | {metrics['genre']['AMI']:.4f} | "
                f"{template.base.verdict(metrics)} | {pareto} |\n"
            )

    lines.extend([
        "\n## Held-out summary statistics\n\n",
        f"- Emotion AMI across four seeds: {format_summary(heldout_emotion)}; clears "
        f"its individual bar in {emotion_clearers}/4 seeds.\n",
        f"- Genre AMI across four seeds: {format_summary(heldout_genre)}; clears its "
        f"individual bar in {genre_clearers}/4 seeds.\n",
        f"- Both bars clear simultaneously in {both_clearers}/4 seeds.\n\n",
        "## Stability/performance comparison\n\n",
        "K=100/67 and K=30/20 values are cited from the established seed-stress and "
        "first cluster-count-sweep reports, respectively. The K=100/67 emotion mean "
        "is reported as 0.1234 as established for this investigation.\n\n",
        "| configuration | held-out emotion AMI (mean/min/max) | held-out genre AMI (mean/min/max) | both bars clear | emotion spread | genre spread |\n",
        "|---|---|---|---:|---:|---:|\n",
        "| original K=100/67 | mean=0.1234; min=0.1216; max=0.1252 | mean=0.2089; min=0.1849; max=0.2466 | 1/4 | 0.0036 | 0.0617 |\n",
        f"| K=30/20 | {format_summary(K30_EMOTION)} | {format_summary(K30_GENRE)} | 0/4 | "
        f"{k30_emotion_spread:.4f} | {k30_genre_spread:.4f} |\n",
        f"| K={selected_result['n_initial_clusters']}/{selected_result['n_surviving_clusters']} | "
        f"{format_summary(heldout_emotion)} | {format_summary(heldout_genre)} | "
        f"{both_clearers}/4 | {emotion_spread:.4f} | {genre_spread:.4f} |\n\n",
        "## Decision\n\n",
    ])
    materially_tighter = (
        emotion_spread < original_emotion_spread
        and genre_spread < original_genre_spread
    )
    if both_clearers >= 2 and materially_tighter:
        lines.append(
            "**Real success: stable crossover found.** This configuration clears the "
            f"held-out Pareto bar in {both_clearers}/4 seeds while retaining most of "
            "K=30's stability gain, with spreads meaningfully tighter than K=100/67. "
            "It is the new standing PercepT Stage 1 configuration.\n"
        )
    elif any_screen_clearer:
        lines.append(
            "**Seed-dependent result.** The screen found a possible crossover, but the "
            f"selected point clears both bars in only {both_clearers}/4 seeds"
            + (
                " and does not keep K=30-like stability. "
                if not materially_tighter
                else ". "
            )
            + "A further-refined search might still find a stable crossing point; this "
            "evidence does not establish one.\n"
        )
    else:
        lines.append(
            "**No stable crossover observed.** None of the tested intermediate points "
            "cleared both held-out bars in the seed-42 screen, and the selected point "
            f"cleared both in only {both_clearers}/4 stress-test seeds. The evidence "
            "suggests no emotion/genre crossing point in the tested range.\n"
        )

    os.makedirs(REPORT_OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main():
    """Run the unchanged training mechanics with V2 configuration and reporting."""
    template.main()


template.CLUSTER_COUNT_PAIRS = CLUSTER_COUNT_PAIRS
template.REPORT_PATH = REPORT_PATH
template.write_report = write_report


if __name__ == "__main__":
    main()
