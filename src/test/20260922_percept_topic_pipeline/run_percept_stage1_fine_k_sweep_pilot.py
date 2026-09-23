"""Fine-screen PercepT Stage 1 cluster counts around the K=60/40 crossover.

Phase 1 cites the already measured K=50/33 seed-42 result and measures the
three intervening configurations from one shared pretrained autoencoder.
Phase 2 seed-stresses the selected Phase-1 point using the established seeds.
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
REPORT_PATH = os.path.join(
    REPORT_OUT_DIR, "percept_stage1_fine_k_sweep_pilot_report.md"
)
NEW_CLUSTER_COUNT_PAIRS = ((55, 37), (65, 43), (70, 47))
SEED = 42
SEEDS = (7, 123, 2024)
K60_EMOTION_MEAN = 0.1252
K60_GENRE_MEAN = 0.2486


def load_template():
    """Load the v2 sweep and its training helpers without running its main block."""
    spec = importlib.util.spec_from_file_location("percept_fine_k_sweep_template", TEMPLATE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import sweep template from {TEMPLATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = load_template()
helpers = sweep.template


def cited_k50_result():
    """Return the first-sweep K=50/33 seed-42 metrics without retraining it."""
    return {
        "n_initial_clusters": 50,
        "n_surviving_clusters": 33,
        "dec_losses": None,
        "stop_reason": "cited from first cluster-count sweep",
        "stop_epoch": None,
        "source": "cited from first cluster-count sweep",
        "train_metrics": {
            "emotion": {"AMI": 0.1453},
            "genre": {"AMI": 0.3141},
            "cluster_sizes": np.array([7790]),
            "small_clusters": 3,
            "collapsed": False,
        },
        "heldout_metrics": {
            "emotion": {"AMI": 0.1202},
            "genre": {"AMI": 0.2830},
            "cluster_sizes": np.array([930]),
            "small_clusters": 3,
            "collapsed": False,
        },
    }


def format_phase_1_rows(results):
    """Format cited and newly measured screen points with the standard columns."""
    lines = []
    for result in results:
        n_surviving = result["n_surviving_clusters"]
        source = result.get("source", "newly measured in this sweep")
        for split_name, metrics in (
            ("train", result["train_metrics"]),
            ("held-out", result["heldout_metrics"]),
        ):
            pareto = (
                "n/a (train split)"
                if split_name == "train"
                else "clears" if sweep.clears(metrics) else "does not clear"
            )
            lines.append(
                f"| {result['n_initial_clusters']} | {n_surviving} | {split_name} | "
                f"{metrics['emotion']['AMI']:.4f} | {metrics['genre']['AMI']:.4f} | "
                f"{helpers.base.verdict(metrics)} | {pareto} | "
                f"{int(metrics['cluster_sizes'].max()):,} | "
                f"{sweep.cluster_fraction(metrics, n_surviving)} | {source} |\n"
            )
    return lines


def write_report(pretrain_losses, input_dim, phase_1_results, selected_result, phase_2_results):
    """Write the fine-sweep report with cited baselines and honest K=60 comparison."""
    selected_metrics = selected_result["heldout_metrics"]
    heldout_metrics = [result["heldout_metrics"] for result in phase_2_results]
    heldout_emotion = [metrics["emotion"]["AMI"] for metrics in heldout_metrics]
    heldout_genre = [metrics["genre"]["AMI"] for metrics in heldout_metrics]
    both_clearers = sum(sweep.clears(metrics) for metrics in heldout_metrics)
    emotion_clearers = sum(value > sweep.EMOTION_PARETO_BAR for value in heldout_emotion)
    genre_clearers = sum(value > sweep.GENRE_PARETO_BAR for value in heldout_genre)
    emotion_mean = np.mean(heldout_emotion)
    genre_mean = np.mean(heldout_genre)
    selected_emotion_margin = selected_metrics["emotion"]["AMI"] - sweep.EMOTION_PARETO_BAR
    selected_genre_margin = selected_metrics["genre"]["AMI"] - sweep.GENRE_PARETO_BAR
    phase_1_by_pair = {
        (result["n_initial_clusters"], result["n_surviving_clusters"]): result
        for result in phase_1_results
    }
    any_screen_clearer = any(sweep.clears(result["heldout_metrics"]) for result in phase_1_results)
    beats_k60 = emotion_mean > K60_EMOTION_MEAN and genre_mean > K60_GENRE_MEAN

    lines = [
        "# ArtELingo PercepT Stage 1 fine K-sweep pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Controlled setup\n\n",
        f"The train-only input is a {input_dim}-dimensional fused vector. Phase 1 uses "
        "seed 42, performs one 100-epoch autoencoder pretrain, and deep-copies that "
        "pretrained state for each isolated new DEC run. Each new point receives fresh "
        "K-means at `random_state=42`. `LAMBDA_BALANCE=1000` and "
        "`LAMBDA_RECONSTRUCTION=1` remain fixed; only the initial/surviving "
        "cluster-count pair varies. K=50/33 is cited from the first cluster-count "
        "sweep rather than retrained.\n\n",
        "Phase 2 cites the selected Phase-1 seed-42 result, then runs seeds 7, 123, "
        "and 2024. Each new seed rebuilds and pretrains an autoencoder for 100 epochs, "
        "runs `KMeans(random_state=seed)`, and trains DEC with unchanged loss weights.\n\n",
        "The below-1% and collapse calculations use each point's active surviving-cluster "
        "count, not the original fixed 67.\n\n",
        "## Predeclared success criterion\n\n",
        "**held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, both simultaneously.**\n\n",
        "## Phase 1: seed-42 cluster-count screen\n\n",
        f"Pretraining reconstruction: {helpers.base.format_pretrain_losses(pretrain_losses)}.\n\n",
        "| N initial | N surviving | split | emotion AMI | genre AMI | verdict | held-out Pareto bar | final max cluster size | fraction below 1% | source |\n",
        "|---:|---:|---|---:|---:|---|---|---:|---:|---|\n",
    ]
    lines.extend(format_phase_1_rows(phase_1_results))
    lines.extend([
        "\n## Consolidated cluster-count picture\n\n",
        "Prior values are cited, not recomputed. K=50/33 is cited from "
        "`percept_stage1_cluster_count_sweep_pilot_report.md`; K=60/40 is cited "
        "from `percept_stage1_cluster_count_sweep_v2_pilot_report.md`; K=100/67 "
        "is cited from the lambda=1000 balance-v2 sweep.\n\n",
        "| N initial | N surviving | held-out emotion AMI | held-out genre AMI | source |\n",
        "|---:|---:|---:|---:|---|\n",
        "| 20 | 13 | 0.1152 | 0.2273 | cited from first cluster-count sweep |\n",
        "| 30 | 20 | 0.1220 | 0.2577 | cited from first cluster-count sweep |\n",
        "| 40 | 27 | 0.1220 | 0.2736 | cited from v2 cluster-count sweep |\n",
        "| 50 | 33 | 0.1202 | 0.2830 | cited from first cluster-count sweep |\n",
    ])
    for n_initial, n_surviving in NEW_CLUSTER_COUNT_PAIRS:
        metrics = phase_1_by_pair[(n_initial, n_surviving)]["heldout_metrics"]
        lines.append(
            f"| {n_initial} | {n_surviving} | {metrics['emotion']['AMI']:.4f} | "
            f"{metrics['genre']['AMI']:.4f} | newly measured in this sweep |\n"
        )
    lines.extend([
        "| 60 | 40 | 0.1238 | 0.2617 | cited from v2 cluster-count sweep |\n",
        "| 80 | 53 | 0.1220 | 0.2396 | cited from v2 cluster-count sweep |\n",
        "| 100 | 67 | 0.1242 | 0.2466 | cited from balance v2 sweep at lambda=1000 |\n",
        "\n## Phase-2 selection\n\n",
        f"`N_INITIAL_CLUSTERS={selected_result['n_initial_clusters']}`, "
        f"`N_SURVIVING_CLUSTERS={selected_result['n_surviving_clusters']}` was selected "
        + (
            "because it clears the held-out Pareto bar and has the largest held-out emotion margin among clearing points"
            if sweep.clears(selected_metrics)
            else "because no Phase-1 point clears both bars, so it has the largest held-out emotion-bar margin"
        )
        + f" (emotion margin {selected_emotion_margin:+.4f}; genre margin {selected_genre_margin:+.4f}).\n\n",
        "## Phase 2: four-seed stress test\n\n",
        "| seed | source | split | emotion AMI | genre AMI | verdict | held-out Pareto bar |\n",
        "|---:|---|---|---:|---:|---|---|\n",
    ])
    for result in phase_2_results:
        for split_name, metrics in (("train", result["train_metrics"]), ("held-out", result["heldout_metrics"])):
            pareto = "n/a (train split)" if split_name == "train" else ("clears" if sweep.clears(metrics) else "does not clear")
            lines.append(
                f"| {result['seed']} | {result['source']} | {split_name} | "
                f"{metrics['emotion']['AMI']:.4f} | {metrics['genre']['AMI']:.4f} | "
                f"{helpers.base.verdict(metrics)} | {pareto} |\n"
            )
    lines.extend([
        "\n## Held-out summary statistics\n\n",
        f"- Emotion AMI across four seeds: {sweep.format_summary(heldout_emotion)}; clears its individual bar in {emotion_clearers}/4 seeds.\n",
        f"- Genre AMI across four seeds: {sweep.format_summary(heldout_genre)}; clears its individual bar in {genre_clearers}/4 seeds.\n",
        f"- Both bars clear simultaneously in {both_clearers}/4 seeds.\n\n",
        "## Comparison with K=60/40\n\n",
        "K=60/40's established four-seed mean is emotion 0.1252 and genre 0.2486. "
        "The comparison is like-for-like because both use seeds 42, 7, 123, and 2024.\n\n",
        "| configuration | held-out emotion AMI (mean/min/max) | held-out genre AMI (mean/min/max) | both bars clear |\n",
        "|---|---|---|---:|\n",
        "| K=60/40 | mean=0.1252 | mean=0.2486 | 4/4 |\n",
        f"| K={selected_result['n_initial_clusters']}/{selected_result['n_surviving_clusters']} | "
        f"{sweep.format_summary(heldout_emotion)} | {sweep.format_summary(heldout_genre)} | {both_clearers}/4 |\n\n",
        "## Decision\n\n",
    ])
    if beats_k60:
        lines.append(
            "**Fine-sweep improvement observed.** The selected point exceeds K=60/40's "
            "four-seed mean on both held-out AMIs.\n"
        )
    elif any_screen_clearer:
        lines.append(
            "**K=60/40 remains the standing result.** A fine-sweep point showed possible "
            "single-seed crossover evidence, but its four-seed mean does not exceed "
            "K=60/40's emotion 0.1252 and genre 0.2486 simultaneously. Do not force a "
            "new winner narrative from a weaker or partial result.\n"
        )
    else:
        lines.append(
            "**K=60/40 remains the standing result.** None of the fine-screen points cleared "
            "the held-out Pareto bar at seed 42, and the selected stress-tested point does "
            "not exceed K=60/40's four-seed mean on both metrics.\n"
        )

    os.makedirs(REPORT_OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main():
    """Run the cited-plus-new screen, then the established Phase-2 seed stress."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    base = helpers.base
    pipeline = base.load_sibling_module("artelingo_run_pipeline_percept_train", base.PIPELINE_PATH)
    affect_pilot = base.load_sibling_module("artelingo_run_affect_pilot_percept", base.AFFECT_PILOT_PATH)
    cca_audit = base.load_sibling_module("artelingo_run_cca_audit_percept", base.CCA_AUDIT_PATH)
    heldout_pipeline = base.load_sibling_module("artelingo_run_pipeline_percept_heldout", base.PIPELINE_PATH)
    heldout_pipeline.STORAGE_DIR = base.HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = base.HELDOUT_JSON
    pipeline.assert_extraction_complete()
    heldout_pipeline.assert_extraction_complete()
    log = pipeline.log
    log("Loading and deduplicating train CLIP features...")
    paintings, img_nodes, txt_nodes, emotion_counts = pipeline.load_dedup_features()
    train_emotions = [pipeline.majority(counts) for counts in emotion_counts]
    log("Loading and deduplicating held-out CLIP features...")
    heldout_paintings, heldout_img_nodes, heldout_txt_nodes, heldout_emotion_counts = heldout_pipeline.load_dedup_features()
    heldout_emotions = [heldout_pipeline.majority(counts) for counts in heldout_emotion_counts]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {device} for embedding extraction, pretraining, and DEC training.")
    affect_train = base.extract_affect_embedding_nodes(pipeline.TRAIN_JSON, paintings, device, log)
    affect_heldout = base.extract_affect_embedding_nodes(base.HELDOUT_JSON, heldout_paintings, device, log)
    train_h = base.fused_embeddings(img_nodes, txt_nodes, affect_train, cca_audit, affect_pilot)
    heldout_h = base.fused_embeddings(heldout_img_nodes, heldout_txt_nodes, affect_heldout, cca_audit, affect_pilot)
    if train_h.shape[1] != heldout_h.shape[1]:
        raise RuntimeError(f"Train/held-out fused dimensions differ: {train_h.shape[1]} vs {heldout_h.shape[1]}.")
    train_inputs = torch.from_numpy(train_h)
    pretrained_encoder, pretrained_decoder = base.build_autoencoder(train_h.shape[1])
    pretrained_encoder.to(device)
    pretrained_decoder.to(device)
    pretrain_losses = base.pretrain_autoencoder(pretrained_encoder, pretrained_decoder, train_inputs, device, log)
    pretrained_encoder_state = helpers.copy.deepcopy(pretrained_encoder.state_dict())
    pretrained_decoder_state = helpers.copy.deepcopy(pretrained_decoder.state_dict())
    del pretrained_encoder, pretrained_decoder

    phase_1_results = [cited_k50_result()]
    for n_initial, n_surviving in NEW_CLUSTER_COUNT_PAIRS:
        log(f"Starting isolated seed-42 DEC sweep point K={n_initial}/{n_surviving} from the shared pretrained state.")
        encoder, decoder = base.build_autoencoder(train_h.shape[1])
        encoder.load_state_dict(pretrained_encoder_state)
        decoder.load_state_dict(pretrained_decoder_state)
        encoder.to(device)
        decoder.to(device)
        centers = helpers.initialize_cluster_centers(encoder, train_inputs, device, n_initial, SEED)
        dec_losses, stop_reason, stop_epoch = helpers.train_dec_until_stable(encoder, decoder, centers, train_inputs, device, log, n_initial)
        train_metrics, heldout_metrics, surviving_indices = helpers.evaluate_run(encoder, centers, train_inputs, heldout_h, pipeline, heldout_pipeline, paintings, train_emotions, heldout_paintings, heldout_emotions, device, n_surviving)
        log(f"K={n_initial}/{n_surviving}: stopped at epoch {stop_epoch} via {stop_reason}; pruned {n_initial - n_surviving} centers and retained original indices {surviving_indices.tolist()}.")
        phase_1_results.append({"n_initial_clusters": n_initial, "n_surviving_clusters": n_surviving, "dec_losses": dec_losses, "stop_reason": stop_reason, "stop_epoch": stop_epoch, "source": "newly measured in this sweep", "train_metrics": train_metrics, "heldout_metrics": heldout_metrics})

    selected_result = helpers.select_phase_2_result(phase_1_results)
    selected_initial = selected_result["n_initial_clusters"]
    selected_surviving = selected_result["n_surviving_clusters"]
    log(f"Selected K={selected_initial}/{selected_surviving} for Phase 2 seed stress.")
    phase_2_results = [helpers.cited_seed_42_result(selected_result)]
    for seed in SEEDS:
        log(f"Starting fresh Phase-2 full seed-stress run for seed={seed}, K={selected_initial}/{selected_surviving}.")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        encoder, decoder = base.build_autoencoder(train_h.shape[1])
        encoder.to(device)
        decoder.to(device)
        base.pretrain_autoencoder(encoder, decoder, train_inputs, device, log)
        centers = helpers.initialize_cluster_centers(encoder, train_inputs, device, selected_initial, seed)
        _, stop_reason, stop_epoch = helpers.train_dec_until_stable(encoder, decoder, centers, train_inputs, device, log, selected_initial)
        train_metrics, heldout_metrics, surviving_indices = helpers.evaluate_run(encoder, centers, train_inputs, heldout_h, pipeline, heldout_pipeline, paintings, train_emotions, heldout_paintings, heldout_emotions, device, selected_surviving)
        log(f"seed={seed}: stopped at epoch {stop_epoch} via {stop_reason}; pruned {selected_initial - selected_surviving} centers and retained original indices {surviving_indices.tolist()}.")
        phase_2_results.append({"seed": seed, "source": "newly measured", "n_initial_clusters": selected_initial, "n_surviving_clusters": selected_surviving, "train_metrics": train_metrics, "heldout_metrics": heldout_metrics})
    write_report(pretrain_losses, train_h.shape[1], phase_1_results, selected_result, phase_2_results)
    log(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
