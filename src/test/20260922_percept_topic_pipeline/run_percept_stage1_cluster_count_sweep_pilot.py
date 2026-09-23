"""Sweep PercepT Stage 1 cluster counts, then seed-stress the best point.

Phase 1 shares one seed-42 pretrained autoencoder across three cluster-count
pairs. Phase 2 cites the selected Phase-1 result and reruns that configuration
from scratch at the three established stress-test seeds.
"""

import copy
import importlib.util
import os
import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F


OUT_DIR = os.path.dirname(__file__)
BASE_PILOT_PATH = os.path.join(OUT_DIR, "run_percept_stage1_pilot.py")
REPORT_PATH = os.path.join(
    OUT_DIR, "percept_stage1_cluster_count_sweep_pilot_report.md"
)
CLUSTER_COUNT_PAIRS = ((20, 13), (30, 20), (50, 33))
LAMBDA_BALANCE = 1000
LAMBDA_RECONSTRUCTION = 1
SEED = 42
SEEDS = (7, 123, 2024)
EMOTION_PARETO_BAR = 0.1236
GENRE_PARETO_BAR = 0.1954
BASELINE_HELDOUT_EMOTION = (0.1242, 0.1228, 0.1252, 0.1216)
BASELINE_HELDOUT_GENRE = (0.2466, 0.2190, 0.1851, 0.1849)
BASELINE_BOTH_CLEARERS = 1


def load_module(module_name: str, path: str):
    """Load a standalone sibling script without running its main block."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("percept_stage1_base_pilot", BASE_PILOT_PATH)

MAX_DEC_EPOCHS = base.MAX_DEC_EPOCHS
STABILITY_THRESHOLD = base.STABILITY_THRESHOLD
DEC_LEARNING_RATE = base.DEC_LEARNING_RATE


def initialize_cluster_centers(
    encoder: nn.Sequential,
    inputs: torch.Tensor,
    device: str,
    n_initial_clusters: int,
    seed: int,
) -> nn.Parameter:
    """Initialize the requested number of DEC centers with fresh K-means."""
    encoder.eval()
    with torch.no_grad():
        latent = encoder(inputs.to(device)).cpu().numpy()
    kmeans = KMeans(n_clusters=n_initial_clusters, n_init=10, random_state=seed)
    kmeans.fit(latent)
    return nn.Parameter(
        torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(device)
    )


def train_dec_until_stable(
    encoder: nn.Sequential,
    decoder: nn.Sequential,
    centers: nn.Parameter,
    inputs: torch.Tensor,
    device: str,
    log,
    n_initial_clusters: int,
) -> tuple[list[dict[str, float]], str, int]:
    """Train balanced DEC with the original reconstruction weight."""
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + [centers],
        lr=DEC_LEARNING_RATE,
    )
    inputs = inputs.to(device)
    encoder.eval()
    with torch.no_grad():
        previous_assignments = base.soft_assignments(encoder(inputs), centers).argmax(
            dim=1
        )
    log(
        f"Starting K={n_initial_clusters} joint DEC training "
        f"(up to {MAX_DEC_EPOCHS} full-batch epochs, lr={DEC_LEARNING_RATE:g}, "
        f"stability threshold={STABILITY_THRESHOLD:.3f}, "
        f"balance lambda={LAMBDA_BALANCE:g})..."
    )
    losses = []
    encoder.train()
    decoder.train()
    for epoch in range(1, MAX_DEC_EPOCHS + 1):
        latent = encoder(inputs)
        reconstruction = decoder(latent)
        q = base.soft_assignments(latent, centers)
        p = base.target_distribution(q)
        kl_loss = F.kl_div(q.log(), p, reduction="batchmean", log_target=False)
        reconstruction_loss = F.mse_loss(reconstruction, inputs)
        mean_q = q.mean(dim=0)
        uniform = torch.full_like(mean_q, 1.0 / mean_q.shape[0])
        balance_loss = F.kl_div(
            mean_q.clamp_min(1e-8).log(), uniform, reduction="sum"
        )
        total_loss = (
            kl_loss
            + LAMBDA_RECONSTRUCTION * reconstruction_loss
            + LAMBDA_BALANCE * balance_loss
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        encoder.eval()
        with torch.no_grad():
            current_assignments = base.soft_assignments(
                encoder(inputs), centers
            ).argmax(dim=1)
        num_changed = (current_assignments != previous_assignments).sum().item()
        fraction_changed = num_changed / len(inputs)
        previous_assignments = current_assignments
        encoder.train()
        checkpoint = {
            "epoch": float(epoch),
            "total": total_loss.item(),
            "kl": kl_loss.item(),
            "reconstruction": reconstruction_loss.item(),
            "balance": balance_loss.item(),
            "fraction_changed": fraction_changed,
        }
        losses.append(checkpoint)
        stopped_for_stability = fraction_changed < STABILITY_THRESHOLD
        reached_epoch_ceiling = epoch == MAX_DEC_EPOCHS
        stopping = stopped_for_stability or reached_epoch_ceiling
        if epoch % 25 == 0 or stopping:
            log(
                f"K={n_initial_clusters} DEC epoch {epoch:03d}/{MAX_DEC_EPOCHS}: "
                f"total={checkpoint['total']:.6f}, KL={checkpoint['kl']:.6f}, "
                f"reconstruction={checkpoint['reconstruction']:.6f}, "
                f"balance={checkpoint['balance']:.6f}"
            )
        if epoch % 10 == 0 or stopping:
            log(
                f"K={n_initial_clusters} DEC epoch {epoch:03d}/{MAX_DEC_EPOCHS}: "
                f"fraction_changed={fraction_changed:.6f} ({num_changed:,} nodes)"
            )
        if stopping:
            stop_reason = (
                "stability criterion" if stopped_for_stability else "epoch ceiling"
            )
            log(f"Stopping K={n_initial_clusters} DEC at epoch {epoch}: {stop_reason}.")
            return losses, stop_reason, epoch
    raise RuntimeError("DEC training exited without a stopping condition.")


def prune_centers(
    centers: torch.Tensor, n_surviving_clusters: int
) -> tuple[torch.Tensor, np.ndarray]:
    """Keep the requested number of highest-norm final centers."""
    norms = torch.linalg.vector_norm(centers.detach(), dim=1)
    surviving = torch.argsort(norms, descending=True)[:n_surviving_clusters]
    return centers.detach()[surviving], surviving.cpu().numpy()


def evaluate_assignments(
    pipeline,
    paintings: list[str],
    majority_emotion: list[str],
    latent: np.ndarray,
    assignments: np.ndarray,
    n_surviving_clusters: int,
) -> dict:
    """Score labels and collapse against this point's surviving-cluster count."""
    emotion_metrics = pipeline.external_metrics(assignments, majority_emotion)
    genre_map = pipeline.load_genre_map()
    genre_indices = [
        index for index, painting in enumerate(paintings) if painting in genre_map
    ]
    if not genre_indices:
        raise RuntimeError("No genre-labelled paintings overlap this split.")
    genre_metrics = pipeline.external_metrics(
        [assignments[index] for index in genre_indices],
        [genre_map[paintings[index]] for index in genre_indices],
    )
    cluster_sizes = np.bincount(assignments, minlength=n_surviving_clusters)
    small_clusters = int(np.sum(cluster_sizes < 0.01 * len(assignments)))
    return {
        "emotion": emotion_metrics,
        "genre": genre_metrics,
        "genre_count": len(genre_indices),
        "cluster_sizes": cluster_sizes,
        "small_clusters": small_clusters,
        "collapsed": small_clusters > n_surviving_clusters / 2,
    }


def clears_heldout_pareto_bar(metrics: dict) -> bool:
    """Return whether both held-out AMI thresholds clear simultaneously."""
    return (
        metrics["emotion"]["AMI"] > EMOTION_PARETO_BAR
        and metrics["genre"]["AMI"] > GENRE_PARETO_BAR
    )


def select_phase_2_result(results: list[dict]) -> dict:
    """Prefer Pareto clearance, then the greatest held-out emotion-bar margin."""
    return max(
        results,
        key=lambda result: (
            clears_heldout_pareto_bar(result["heldout_metrics"]),
            result["heldout_metrics"]["emotion"]["AMI"] - EMOTION_PARETO_BAR,
            result["heldout_metrics"]["genre"]["AMI"] - GENRE_PARETO_BAR,
        ),
    )


def cited_seed_42_result(phase_1_result: dict) -> dict:
    """Cite the selected Phase-1 seed-42 result without recomputing it."""
    return {
        "seed": SEED,
        "source": "cited from Phase 1",
        "n_initial_clusters": phase_1_result["n_initial_clusters"],
        "n_surviving_clusters": phase_1_result["n_surviving_clusters"],
        "train_metrics": phase_1_result["train_metrics"],
        "heldout_metrics": phase_1_result["heldout_metrics"],
    }


def format_summary(values: list[float] | tuple[float, ...]) -> str:
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


def format_cluster_fraction(metrics: dict, n_surviving_clusters: int) -> str:
    """Format the point-specific share of surviving clusters below 1% of N."""
    return (
        f"{metrics['small_clusters'] / n_surviving_clusters:.1%} "
        f"({metrics['small_clusters']}/{n_surviving_clusters})"
    )


def write_report(
    pretrain_losses: list[float],
    input_dim: int,
    phase_1_results: list[dict],
    selected_result: dict,
    phase_2_results: list[dict],
) -> None:
    """Write the predeclared screening and four-seed stability report."""
    selected_heldout = selected_result["heldout_metrics"]
    heldout_metrics = [result["heldout_metrics"] for result in phase_2_results]
    heldout_emotion = [metrics["emotion"]["AMI"] for metrics in heldout_metrics]
    heldout_genre = [metrics["genre"]["AMI"] for metrics in heldout_metrics]
    emotion_clearers = sum(value > EMOTION_PARETO_BAR for value in heldout_emotion)
    genre_clearers = sum(value > GENRE_PARETO_BAR for value in heldout_genre)
    both_clearers = sum(clears_heldout_pareto_bar(metrics) for metrics in heldout_metrics)
    misses = [
        (result["seed"], miss_description(result["heldout_metrics"]))
        for result in phase_2_results
        if not clears_heldout_pareto_bar(result["heldout_metrics"])
    ]
    selected_emotion_margin = selected_heldout["emotion"]["AMI"] - EMOTION_PARETO_BAR
    selected_genre_margin = selected_heldout["genre"]["AMI"] - GENRE_PARETO_BAR
    baseline_emotion_spread = np.ptp(BASELINE_HELDOUT_EMOTION)
    baseline_genre_spread = np.ptp(BASELINE_HELDOUT_GENRE)
    emotion_spread = np.ptp(heldout_emotion)
    genre_spread = np.ptp(heldout_genre)
    clearly_improved = both_clearers > BASELINE_BOTH_CLEARERS

    lines = [
        "# ArtELingo PercepT Stage 1 cluster-count sweep pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Controlled setup\n\n",
        f"The train-only input is a {input_dim}-dimensional fused vector. "
        "Phase 1 uses seed 42, performs one 100-epoch autoencoder pretrain, "
        "and deep-copies that pretrained state for each isolated DEC run. Each "
        "point receives fresh K-means at `random_state=42` with its own "
        "initial-cluster count. `LAMBDA_BALANCE=1000` and "
        "`LAMBDA_RECONSTRUCTION=1` remain fixed; only the initial/surviving "
        "cluster-count pair varies.\n\n",
        "Phase 2 cites the selected Phase-1 seed-42 result, then runs seeds 7, "
        "123, and 2024. Each new seed applies fresh Torch and CUDA seeding "
        "(when available), builds and pretrains a new autoencoder for 100 "
        "epochs, runs `KMeans(random_state=seed)`, and trains DEC with the "
        "selected cluster counts and the unchanged loss weights.\n\n",
        "The below-1% and collapse calculations use each sweep point's active "
        "surviving-cluster count, rather than the original fixed 67.\n\n",
        "## Predeclared success criterion\n\n",
        "**held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, "
        "both simultaneously.**\n\n",
        "## Phase 1: seed-42 cluster-count screen\n\n",
        f"Pretraining reconstruction: {base.format_pretrain_losses(pretrain_losses)}.\n\n",
        "| N initial | N surviving | split | emotion AMI | genre AMI | verdict | held-out Pareto bar | final max cluster size | fraction below 1% |\n",
        "|---:|---:|---|---:|---:|---|---|---:|---:|\n",
    ]
    for result in phase_1_results:
        n_surviving_clusters = result["n_surviving_clusters"]
        for split_name, metrics in (
            ("train", result["train_metrics"]),
            ("held-out", result["heldout_metrics"]),
        ):
            pareto_clearance = (
                "n/a (train split)"
                if split_name == "train"
                else "clears" if clears_heldout_pareto_bar(metrics) else "does not clear"
            )
            lines.append(
                f"| {result['n_initial_clusters']} | {n_surviving_clusters} | "
                f"{split_name} | {metrics['emotion']['AMI']:.4f} | "
                f"{metrics['genre']['AMI']:.4f} | {base.verdict(metrics)} | "
                f"{pareto_clearance} | {int(metrics['cluster_sizes'].max()):,} | "
                f"{format_cluster_fraction(metrics, n_surviving_clusters)} |\n"
            )

    selection_reason = (
        "it clears the held-out Pareto bar and has the largest held-out emotion "
        "margin among the clearing points"
        if clears_heldout_pareto_bar(selected_heldout)
        else "no Phase-1 point clears both bars, so it has the largest held-out "
        "emotion-bar margin"
    )
    lines.extend([
        "\n## Phase-2 selection\n\n",
        f"`N_INITIAL_CLUSTERS={selected_result['n_initial_clusters']}`, "
        f"`N_SURVIVING_CLUSTERS={selected_result['n_surviving_clusters']}` was "
        f"selected because {selection_reason} (emotion margin "
        f"{selected_emotion_margin:+.4f}; genre margin "
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
            pareto_clearance = (
                "n/a (train split)"
                if split_name == "train"
                else "clears" if clears_heldout_pareto_bar(metrics) else "does not clear"
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
        "## Before/after comparison with original K=100/67\n\n",
        "The baseline values below are the original K=100/67 four-seed results "
        "from `percept_stage1_seed_stress_pilot_report.md`. This pilot uses the "
        "same seed set, so it is a like-for-like stability comparison.\n\n",
        "| configuration | held-out emotion AMI (mean/min/max) | held-out genre AMI (mean/min/max) | both bars clear |\n",
        "|---|---|---|---:|\n",
        f"| original K=100/67 | {format_summary(BASELINE_HELDOUT_EMOTION)} | "
        f"{format_summary(BASELINE_HELDOUT_GENRE)} | {BASELINE_BOTH_CLEARERS}/4 |\n",
        f"| K={selected_result['n_initial_clusters']}/"
        f"{selected_result['n_surviving_clusters']} | {format_summary(heldout_emotion)} | "
        f"{format_summary(heldout_genre)} | {both_clearers}/4 |\n\n",
        f"The emotion range is {emotion_spread:.4f} versus the original "
        f"{baseline_emotion_spread:.4f}; the genre range is {genre_spread:.4f} "
        f"versus the original {baseline_genre_spread:.4f}.\n\n",
        "## Decision\n\n",
    ])
    if both_clearers == len(phase_2_results):
        lines.append(
            "**Real success: robust result.** All 4/4 seeds clear the held-out "
            "Pareto bar, so this coarser-cluster configuration is robust under "
            "the established seed-stress criterion.\n"
        )
    else:
        miss_lines = "; ".join(
            f"seed {seed} misses {description}" for seed, description in misses
        )
        improvement = (
            "It is a clear improvement over the original 1/4 both-clear result, "
            "but it does not meet the 4/4 robustness bar."
            if clearly_improved
            else "Do not treat a marginal or partial change as a solved stability problem."
        )
        lines.append(
            f"**Seed-dependent result.** Only {both_clearers}/4 seeds clear the "
            f"held-out Pareto bar. The misses are: {miss_lines}. {improvement}\n"
        )

    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def evaluate_run(
    encoder: nn.Sequential,
    centers: nn.Parameter,
    train_inputs: torch.Tensor,
    heldout_h: np.ndarray,
    pipeline,
    heldout_pipeline,
    paintings: list[str],
    train_emotions: list[str],
    heldout_paintings: list[str],
    heldout_emotions: list[str],
    device: str,
    n_surviving_clusters: int,
) -> tuple[dict, dict, np.ndarray]:
    """Prune and evaluate using the active surviving-cluster count."""
    surviving_centers, surviving_indices = prune_centers(
        centers, n_surviving_clusters
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
    train_metrics = evaluate_assignments(
        pipeline,
        paintings,
        train_emotions,
        train_latent.cpu().numpy(),
        train_assignments,
        n_surviving_clusters,
    )
    heldout_metrics = evaluate_assignments(
        heldout_pipeline,
        heldout_paintings,
        heldout_emotions,
        heldout_latent.cpu().numpy(),
        heldout_assignments,
        n_surviving_clusters,
    )
    return train_metrics, heldout_metrics, surviving_indices


def main() -> None:
    """Run the seed-42 screen, then fully rerandomized Phase-2 seed stress."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

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
    pretrained_encoder, pretrained_decoder = base.build_autoencoder(train_h.shape[1])
    pretrained_encoder.to(device)
    pretrained_decoder.to(device)
    pretrain_losses = base.pretrain_autoencoder(
        pretrained_encoder, pretrained_decoder, train_inputs, device, log
    )
    pretrained_encoder_state = copy.deepcopy(pretrained_encoder.state_dict())
    pretrained_decoder_state = copy.deepcopy(pretrained_decoder.state_dict())
    del pretrained_encoder, pretrained_decoder

    phase_1_results = []
    for n_initial_clusters, n_surviving_clusters in CLUSTER_COUNT_PAIRS:
        log(
            "Starting isolated seed-42 DEC sweep point "
            f"K={n_initial_clusters}/{n_surviving_clusters} from the shared "
            "pretrained state."
        )
        encoder, decoder = base.build_autoencoder(train_h.shape[1])
        encoder.load_state_dict(pretrained_encoder_state)
        decoder.load_state_dict(pretrained_decoder_state)
        encoder.to(device)
        decoder.to(device)
        centers = initialize_cluster_centers(
            encoder, train_inputs, device, n_initial_clusters, SEED
        )
        dec_losses, stop_reason, stop_epoch = train_dec_until_stable(
            encoder,
            decoder,
            centers,
            train_inputs,
            device,
            log,
            n_initial_clusters,
        )
        train_metrics, heldout_metrics, surviving_indices = evaluate_run(
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
            n_surviving_clusters,
        )
        log(
            f"K={n_initial_clusters}/{n_surviving_clusters}: stopped at epoch "
            f"{stop_epoch} via {stop_reason}; pruned "
            f"{n_initial_clusters - n_surviving_clusters} centers and retained "
            f"original indices {surviving_indices.tolist()}."
        )
        phase_1_results.append({
            "n_initial_clusters": n_initial_clusters,
            "n_surviving_clusters": n_surviving_clusters,
            "dec_losses": dec_losses,
            "stop_reason": stop_reason,
            "stop_epoch": stop_epoch,
            "train_metrics": train_metrics,
            "heldout_metrics": heldout_metrics,
        })

    selected_result = select_phase_2_result(phase_1_results)
    selected_initial = selected_result["n_initial_clusters"]
    selected_surviving = selected_result["n_surviving_clusters"]
    log(f"Selected K={selected_initial}/{selected_surviving} for Phase 2 seed stress.")
    phase_2_results = [cited_seed_42_result(selected_result)]
    for seed in SEEDS:
        log(
            f"Starting fresh Phase-2 full seed-stress run for seed={seed}, "
            f"K={selected_initial}/{selected_surviving}."
        )
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        encoder, decoder = base.build_autoencoder(train_h.shape[1])
        encoder.to(device)
        decoder.to(device)
        base.pretrain_autoencoder(encoder, decoder, train_inputs, device, log)
        centers = initialize_cluster_centers(
            encoder, train_inputs, device, selected_initial, seed
        )
        _, stop_reason, stop_epoch = train_dec_until_stable(
            encoder,
            decoder,
            centers,
            train_inputs,
            device,
            log,
            selected_initial,
        )
        train_metrics, heldout_metrics, surviving_indices = evaluate_run(
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
            selected_surviving,
        )
        log(
            f"seed={seed}: stopped at epoch {stop_epoch} via {stop_reason}; "
            f"pruned {selected_initial - selected_surviving} centers and retained "
            f"original indices {surviving_indices.tolist()}."
        )
        phase_2_results.append({
            "seed": seed,
            "source": "newly measured",
            "n_initial_clusters": selected_initial,
            "n_surviving_clusters": selected_surviving,
            "train_metrics": train_metrics,
            "heldout_metrics": heldout_metrics,
        })

    write_report(
        pretrain_losses,
        train_h.shape[1],
        phase_1_results,
        selected_result,
        phase_2_results,
    )
    log(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
