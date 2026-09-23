"""Stress-test PercepT Stage 1's lambda=1000 result across fresh seeds.

Feature extraction and fusion are deterministic and shared once.  Each new
seed instead receives a fresh autoencoder initialization, 100-epoch
pretraining run, seeded K-means initialization, and DEC training run.
"""

import importlib.util
import os
import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn


OUT_DIR = os.path.dirname(__file__)
REPORT_OUT_DIR = os.environ.get("PERCEPT_OUTPUT_ROOT") or OUT_DIR
BASE_PILOT_PATH = os.path.join(OUT_DIR, "run_percept_stage1_pilot.py")
BALANCE_SWEEP_PATH = os.path.join(
    OUT_DIR, "run_percept_stage1_balance_sweep_v2_pilot.py"
)
REPORT_PATH = os.path.join(REPORT_OUT_DIR, "percept_stage1_seed_stress_pilot_report.md")
SEEDS = (7, 123, 2024)
LAMBDA_BALANCE = 1000
EMOTION_PARETO_BAR = 0.1236
GENRE_PARETO_BAR = 0.1954


def load_module(module_name: str, path: str):
    """Load a standalone sibling script without running its main block."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("percept_stage1_base_pilot", BASE_PILOT_PATH)
balance_sweep = load_module("percept_stage1_balance_sweep_v2", BALANCE_SWEEP_PATH)

N_INITIAL_CLUSTERS = base.N_INITIAL_CLUSTERS
N_SURVIVING_CLUSTERS = base.N_SURVIVING_CLUSTERS


def initialize_cluster_centers(
    encoder: nn.Sequential, inputs: torch.Tensor, device: str, seed: int
) -> nn.Parameter:
    """Initialize DEC centers with K-means randomized for this stress-test seed."""
    encoder.eval()
    with torch.no_grad():
        latent = encoder(inputs.to(device)).cpu().numpy()
    kmeans = KMeans(
        n_clusters=N_INITIAL_CLUSTERS,
        n_init=10,
        random_state=seed,
    )
    kmeans.fit(latent)
    return nn.Parameter(
        torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(device)
    )


def clears_heldout_pareto_bar(metrics: dict) -> bool:
    """Return whether both held-out AMI thresholds clear simultaneously."""
    return (
        metrics["emotion"]["AMI"] > EMOTION_PARETO_BAR
        and metrics["genre"]["AMI"] > GENRE_PARETO_BAR
    )


def cited_seed_42_result() -> dict:
    """Return lambda=1000's established v2-report result without recomputing it."""
    return {
        "seed": 42,
        "source": "cited from v2 sweep report",
        "train_metrics": {
            "emotion": {"AMI": 0.1444},
            "genre": {"AMI": 0.3193},
            "collapsed": False,
        },
        "heldout_metrics": {
            "emotion": {"AMI": 0.1242},
            "genre": {"AMI": 0.2466},
            "collapsed": False,
        },
    }


def format_summary(values: list[float]) -> str:
    """Format the required held-out summary statistics."""
    return (
        f"mean={np.mean(values):.4f}; min={np.min(values):.4f}; "
        f"max={np.max(values):.4f}"
    )


def miss_description(metrics: dict) -> str:
    """State every held-out bar miss and its margin without rounding it away."""
    misses = []
    emotion_margin = metrics["emotion"]["AMI"] - EMOTION_PARETO_BAR
    genre_margin = metrics["genre"]["AMI"] - GENRE_PARETO_BAR
    if emotion_margin <= 0:
        misses.append(f"emotion by {abs(emotion_margin):.4f}")
    if genre_margin <= 0:
        misses.append(f"genre by {abs(genre_margin):.4f}")
    return " and ".join(misses)


def write_report(results: list[dict], input_dim: int) -> None:
    """Write the four-seed report after the three new GPU runs finish."""
    heldout_metrics = [result["heldout_metrics"] for result in results]
    heldout_emotion = [metrics["emotion"]["AMI"] for metrics in heldout_metrics]
    heldout_genre = [metrics["genre"]["AMI"] for metrics in heldout_metrics]
    emotion_clearers = sum(value > EMOTION_PARETO_BAR for value in heldout_emotion)
    genre_clearers = sum(value > GENRE_PARETO_BAR for value in heldout_genre)
    both_clearers = sum(clears_heldout_pareto_bar(metrics) for metrics in heldout_metrics)
    misses = [
        (result["seed"], miss_description(result["heldout_metrics"]))
        for result in results
        if not clears_heldout_pareto_bar(result["heldout_metrics"])
    ]

    lines = [
        "# ArtELingo PercepT Stage 1 seed stress pilot at lambda=1000\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Controlled setup\n\n",
        f"The train-only input is a {input_dim}-dimensional fused vector. "
        "GoEmotions-RoBERTa affect extraction and fused-embedding construction "
        "are deterministic and were computed once, then reused across all new "
        "seeds. For each new seed, `torch.manual_seed(seed)` and CUDA seeding "
        "(when available) occurred immediately before a fresh autoencoder was "
        "built. That autoencoder was pretrained for 100 epochs, K-means used "
        "`random_state=seed`, and DEC ran with lambda=1000 to the established "
        "convergence criterion before 67-of-100 centroid pruning.\n\n",
        "Seed 42 is cited from `percept_stage1_balance_sweep_v2_pilot_report.md` "
        "at lambda=1000; it is not recomputed here.\n\n",
        "## Predeclared success criterion\n\n",
        "**held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, "
        "both simultaneously.**\n\n",
        "## Results\n\n",
        "| seed | source | split | emotion AMI | genre AMI | verdict | held-out Pareto bar |\n",
        "|---:|---|---|---:|---:|---|---|\n",
    ]
    for result in results:
        for split_name, metrics in (
            ("train", result["train_metrics"]),
            ("held-out", result["heldout_metrics"]),
        ):
            pareto_clearance = (
                "n/a (train split)"
                if split_name == "train"
                else ("clears" if clears_heldout_pareto_bar(metrics) else "does not clear")
            )
            lines.append(
                f"| {result['seed']} | {result['source']} | {split_name} | "
                f"{metrics['emotion']['AMI']:.4f} | {metrics['genre']['AMI']:.4f} | "
                f"{base.verdict(metrics)} | {pareto_clearance} |\n"
            )

    lines.extend([
        "\n## Held-out summary statistics\n\n",
        f"- Emotion AMI across four seeds: {format_summary(heldout_emotion)}; "
        f"clears its individual bar in {emotion_clearers}/4 seeds.\n",
        f"- Genre AMI across four seeds: {format_summary(heldout_genre)}; "
        f"clears its individual bar in {genre_clearers}/4 seeds.\n",
        f"- Both bars clear simultaneously in {both_clearers}/4 seeds.\n\n",
        "## Decision\n\n",
    ])
    if both_clearers == len(results):
        lines.append(
            "**Real success: robust result.** All 4/4 seeds clear the held-out "
            "Pareto bar, so lambda=1000 should be treated as the reliable standing "
            "PercepT Stage 1 configuration going into Stage 2.\n"
        )
    else:
        miss_lines = "; ".join(
            f"seed {seed} misses {description}" for seed, description in misses
        )
        lines.append(
            f"**Seed-dependent result.** Only {both_clearers}/4 seeds clear the "
            f"held-out Pareto bar. The misses are: {miss_lines}. Lambda=1000 should "
            "be treated as fragile/seed-dependent rather than as the reliable standing "
            "PercepT Stage 1 configuration going into Stage 2.\n"
        )

    os.makedirs(REPORT_OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    """Extract deterministic inputs once, then run fresh full pipelines per seed."""
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
    heldout_paintings, heldout_img_nodes, heldout_txt_nodes, heldout_emotion_counts = (
        heldout_pipeline.load_dedup_features()
    )
    heldout_emotions = [
        heldout_pipeline.majority(counts) for counts in heldout_emotion_counts
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {device} for one-time deterministic embedding extraction.")
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
        heldout_img_nodes, heldout_txt_nodes, affect_heldout, cca_audit, affect_pilot
    )
    if train_h.shape[1] != heldout_h.shape[1]:
        raise RuntimeError(
            f"Train/held-out fused dimensions differ: "
            f"{train_h.shape[1]} vs {heldout_h.shape[1]}."
        )

    train_inputs = torch.from_numpy(train_h)
    results = [cited_seed_42_result()]
    for seed in SEEDS:
        log(f"Starting fresh full seed-stress run for seed={seed}.")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        encoder, decoder = base.build_autoencoder(train_h.shape[1])
        encoder.to(device)
        decoder.to(device)
        base.pretrain_autoencoder(encoder, decoder, train_inputs, device, log)
        centers = initialize_cluster_centers(encoder, train_inputs, device, seed)
        trajectory = []
        _, stop_reason, stop_epoch = balance_sweep.train_dec_until_stable(
            encoder,
            decoder,
            centers,
            train_inputs,
            device,
            log,
            LAMBDA_BALANCE,
            trajectory,
        )
        surviving_centers, surviving_indices = base.prune_centers(centers)
        log(
            f"seed={seed}: stopped at epoch {stop_epoch} via {stop_reason}; "
            f"pruned {N_INITIAL_CLUSTERS - N_SURVIVING_CLUSTERS} centers and retained "
            f"original indices {surviving_indices.tolist()}."
        )
        encoder.eval()
        with torch.no_grad():
            train_latent = encoder(train_inputs.to(device))
            train_assignments = (
                base.soft_assignments(train_latent, surviving_centers)
                .argmax(dim=1)
                .cpu()
                .numpy()
            )
            heldout_latent = encoder(torch.from_numpy(heldout_h).to(device))
            heldout_assignments = (
                base.soft_assignments(heldout_latent, surviving_centers)
                .argmax(dim=1)
                .cpu()
                .numpy()
            )
        results.append({
            "seed": seed,
            "source": "newly measured",
            "train_metrics": base.evaluate_assignments(
                pipeline,
                paintings,
                train_emotions,
                train_latent.cpu().numpy(),
                train_assignments,
            ),
            "heldout_metrics": base.evaluate_assignments(
                heldout_pipeline,
                heldout_paintings,
                heldout_emotions,
                heldout_latent.cpu().numpy(),
                heldout_assignments,
            ),
        })

    write_report(results, train_h.shape[1])
    log(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
