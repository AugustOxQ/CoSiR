"""Run a faithful IDEC alternative for ArtELingo PercepT Stage 1.

This standalone GPU pilot keeps reconstruction active during joint DEC training
and scales only the clustering KL loss by IDEC's published gamma=0.1 default.
It intentionally contains no balanced-assignment regularizer.
"""

import importlib.util
import os
import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F


OUT_DIR = os.path.dirname(__file__)
REPORT_OUT_DIR = os.environ.get("PERCEPT_OUTPUT_ROOT") or OUT_DIR
BASE_PILOT_PATH = os.path.join(OUT_DIR, "run_percept_stage1_pilot.py")
REPORT_PATH = os.path.join(REPORT_OUT_DIR, "percept_stage1_idec_faithful_pilot_report.md")
GAMMA = 0.1
SEEDS = (7, 123, 2024)
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

N_INITIAL_CLUSTERS = base.N_INITIAL_CLUSTERS
N_SURVIVING_CLUSTERS = base.N_SURVIVING_CLUSTERS
MAX_DEC_EPOCHS = base.MAX_DEC_EPOCHS
STABILITY_THRESHOLD = base.STABILITY_THRESHOLD
DEC_LEARNING_RATE = base.DEC_LEARNING_RATE


def initialize_cluster_centers(
    encoder: nn.Sequential, inputs: torch.Tensor, device: str, seed: int
) -> nn.Parameter:
    """Initialize DEC centers with K-means randomized for this run's seed."""
    encoder.eval()
    with torch.no_grad():
        latent = encoder(inputs.to(device)).cpu().numpy()
    kmeans = KMeans(n_clusters=N_INITIAL_CLUSTERS, n_init=10, random_state=seed)
    kmeans.fit(latent)
    return nn.Parameter(
        torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(device)
    )


def cluster_size_diagnostic(q: torch.Tensor) -> dict[str, float]:
    """Summarize hard assignments across all 100 pre-pruning DEC centers."""
    assignments = q.argmax(dim=1)
    sizes = torch.bincount(assignments, minlength=N_INITIAL_CLUSTERS)
    return {
        "min": float(sizes.min().item()),
        "max": float(sizes.max().item()),
        "median": float(sizes.float().median().item()),
        "below_one_percent": float((sizes < 0.01 * len(assignments)).sum().item()),
    }


def log_cluster_size_diagnostic(
    epoch: int, q: torch.Tensor, log, trajectory: list[dict[str, float]], seed: int
) -> None:
    """Record and log an all-center hard-assignment distribution checkpoint."""
    diagnostic = {"epoch": float(epoch), **cluster_size_diagnostic(q)}
    trajectory.append(diagnostic)
    log(
        f"seed={seed} IDEC cluster sizes epoch {epoch:03d}/{MAX_DEC_EPOCHS}: "
        f"min={int(diagnostic['min']):,}, max={int(diagnostic['max']):,}, "
        f"median={diagnostic['median']:.1f}, "
        f"below_1pct={int(diagnostic['below_one_percent'])}/{N_INITIAL_CLUSTERS}"
    )


def train_idec_until_stable(
    encoder: nn.Sequential,
    decoder: nn.Sequential,
    centers: nn.Parameter,
    inputs: torch.Tensor,
    device: str,
    log,
    seed: int,
) -> tuple[list[dict[str, float]], list[dict[str, float]], str, int]:
    """Train faithful IDEC: reconstruction plus gamma-scaled DEC KL only."""
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + [centers],
        lr=DEC_LEARNING_RATE,
    )
    inputs = inputs.to(device)
    trajectory = []
    encoder.eval()
    with torch.no_grad():
        initial_q = base.soft_assignments(encoder(inputs), centers)
        previous_assignments = initial_q.argmax(dim=1)
        log_cluster_size_diagnostic(0, initial_q, log, trajectory, seed)
    log(
        f"Starting seed={seed} faithful IDEC joint training "
        f"(up to {MAX_DEC_EPOCHS} full-batch epochs, lr={DEC_LEARNING_RATE:g}, "
        f"gamma={GAMMA:g}, stability threshold={STABILITY_THRESHOLD:.3f})..."
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
        total_loss = reconstruction_loss + GAMMA * kl_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        encoder.eval()
        with torch.no_grad():
            current_q = base.soft_assignments(encoder(inputs), centers)
            current_assignments = current_q.argmax(dim=1)
        num_changed = (current_assignments != previous_assignments).sum().item()
        fraction_changed = num_changed / len(inputs)
        previous_assignments = current_assignments
        encoder.train()
        checkpoint = {
            "epoch": float(epoch),
            "total": total_loss.item(),
            "kl": kl_loss.item(),
            "reconstruction": reconstruction_loss.item(),
            "fraction_changed": fraction_changed,
        }
        losses.append(checkpoint)
        stopped_for_stability = fraction_changed < STABILITY_THRESHOLD
        reached_epoch_ceiling = epoch == MAX_DEC_EPOCHS
        stopping = stopped_for_stability or reached_epoch_ceiling
        if epoch % 25 == 0 or stopping:
            log(
                f"seed={seed} IDEC epoch {epoch:03d}/{MAX_DEC_EPOCHS}: "
                f"total={checkpoint['total']:.6f}, KL={checkpoint['kl']:.6f}, "
                f"reconstruction={checkpoint['reconstruction']:.6f}"
            )
            log_cluster_size_diagnostic(epoch, current_q, log, trajectory, seed)
        if epoch % 10 == 0 or stopping:
            log(
                f"seed={seed} IDEC epoch {epoch:03d}/{MAX_DEC_EPOCHS}: "
                f"fraction_changed={fraction_changed:.6f} ({num_changed:,} nodes)"
            )
        if stopping:
            stop_reason = "stability criterion" if stopped_for_stability else "epoch ceiling"
            log(f"Stopping seed={seed} IDEC at epoch {epoch}: {stop_reason}.")
            return losses, trajectory, stop_reason, epoch
    raise RuntimeError("IDEC training exited without a stopping condition.")


def clears_heldout_pareto_bar(metrics: dict) -> bool:
    """Return whether both held-out AMI thresholds clear simultaneously."""
    return (
        metrics["emotion"]["AMI"] > EMOTION_PARETO_BAR
        and metrics["genre"]["AMI"] > GENRE_PARETO_BAR
    )


def format_dec_losses(losses: list[dict[str, float]]) -> str:
    """Format the standard IDEC loss and assignment-change checkpoints."""
    by_epoch = {int(loss["epoch"]): loss for loss in losses}
    checkpoints = sorted({1, len(losses), *range(25, len(losses) + 1, 50)})
    return "; ".join(
        f"epoch {epoch}: total={by_epoch[epoch]['total']:.6f}, "
        f"KL={by_epoch[epoch]['kl']:.6f}, "
        f"recon={by_epoch[epoch]['reconstruction']:.6f}, "
        f"fraction_changed={by_epoch[epoch]['fraction_changed']:.6f}"
        for epoch in checkpoints
    )


def format_trajectory(trajectory: list[dict[str, float]]) -> str:
    """Format each all-100-center cluster-size diagnostic checkpoint."""
    return "; ".join(
        f"epoch {int(point['epoch'])}: min={int(point['min']):,}, "
        f"max={int(point['max']):,}, median={point['median']:.1f}, "
        f"below_1pct={int(point['below_one_percent'])}/{N_INITIAL_CLUSTERS}"
        for point in trajectory
    )


def format_summary(values: list[float]) -> str:
    """Format the required held-out summary statistics."""
    return (
        f"mean={np.mean(values):.4f}; min={np.min(values):.4f}; "
        f"max={np.max(values):.4f}"
    )


def run_seed(
    seed: int,
    train_h: np.ndarray,
    heldout_h: np.ndarray,
    train_inputs: torch.Tensor,
    device: str,
    log,
    pipeline,
    paintings: list[str],
    train_emotions: list[str],
    heldout_pipeline,
    heldout_paintings: list[str],
    heldout_emotions: list[str],
) -> dict:
    """Run a fresh, fully seeded IDEC pipeline and evaluate both splits."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    encoder, decoder = base.build_autoencoder(train_h.shape[1])
    encoder.to(device)
    decoder.to(device)
    pretrain_losses = base.pretrain_autoencoder(encoder, decoder, train_inputs, device, log)
    centers = initialize_cluster_centers(encoder, train_inputs, device, seed)
    dec_losses, trajectory, stop_reason, stop_epoch = train_idec_until_stable(
        encoder, decoder, centers, train_inputs, device, log, seed
    )
    surviving_centers, surviving_indices = base.prune_centers(centers)
    log(
        f"seed={seed}: stopped at epoch {stop_epoch} via {stop_reason}; pruned "
        f"{N_INITIAL_CLUSTERS - N_SURVIVING_CLUSTERS} centers and retained "
        f"original indices {surviving_indices.tolist()}."
    )
    encoder.eval()
    with torch.no_grad():
        train_latent = encoder(train_inputs.to(device))
        train_assignments = (
            base.soft_assignments(train_latent, surviving_centers).argmax(dim=1).cpu().numpy()
        )
        heldout_latent = encoder(torch.from_numpy(heldout_h).to(device))
        heldout_assignments = (
            base.soft_assignments(heldout_latent, surviving_centers).argmax(dim=1).cpu().numpy()
        )
    return {
        "seed": seed,
        "pretrain_losses": pretrain_losses,
        "dec_losses": dec_losses,
        "trajectory": trajectory,
        "stop_reason": stop_reason,
        "stop_epoch": stop_epoch,
        "train_metrics": base.evaluate_assignments(
            pipeline, paintings, train_emotions, train_latent.cpu().numpy(), train_assignments
        ),
        "heldout_metrics": base.evaluate_assignments(
            heldout_pipeline, heldout_paintings, heldout_emotions,
            heldout_latent.cpu().numpy(), heldout_assignments
        ),
    }


def append_phase_one_results(lines: list[str], result: dict) -> None:
    """Append the required seed-42 trajectory, diagnostics, and split results."""
    heldout_clearance = clears_heldout_pareto_bar(result["heldout_metrics"])
    lines.extend([
        "## Phase 1: seed 42\n\n",
        f"- Pretraining reconstruction: {base.format_pretrain_losses(result['pretrain_losses'])}.\n",
        f"- IDEC stopped via the **{result['stop_reason']}** at epoch **{result['stop_epoch']}** "
        f"(threshold `fraction_changed < {STABILITY_THRESHOLD:.3f}`, ceiling {MAX_DEC_EPOCHS}).\n",
        f"- Joint IDEC: {format_dec_losses(result['dec_losses'])}.\n",
        f"- All-100-center diagnostics (including epoch-0 K-means initialization): "
        f"{format_trajectory(result['trajectory'])}.\n\n",
        "| seed | split | emotion AMI | genre AMI | verdict | held-out Pareto bar | "
        "cluster min | cluster max | cluster median | fraction below 1% |\n",
        "|---:|---|---:|---:|---|---|---:|---:|---:|---:|\n",
    ])
    for split_name, metrics in (("train", result["train_metrics"]), ("held-out", result["heldout_metrics"])):
        sizes = metrics["cluster_sizes"]
        clearance = "n/a (train split)" if split_name == "train" else (
            "clears" if heldout_clearance else "does not clear"
        )
        lines.append(
            f"| 42 | {split_name} | {metrics['emotion']['AMI']:.4f} | "
            f"{metrics['genre']['AMI']:.4f} | {base.verdict(metrics)} | {clearance} | "
            f"{int(sizes.min()):,} | {int(sizes.max()):,} | {float(np.median(sizes)):.1f} | "
            f"{metrics['small_clusters'] / N_SURVIVING_CLUSTERS:.1%} "
            f"({metrics['small_clusters']}/{N_SURVIVING_CLUSTERS}) |\n"
        )
    lines.append("\n")


def append_phase_two_results(lines: list[str], results: list[dict]) -> None:
    """Append the required four-seed IDEC and balance-fix comparison."""
    heldout = [result["heldout_metrics"] for result in results]
    emotion = [metrics["emotion"]["AMI"] for metrics in heldout]
    genre = [metrics["genre"]["AMI"] for metrics in heldout]
    clearers = sum(clears_heldout_pareto_bar(metrics) for metrics in heldout)
    balance_emotion = [0.1242, 0.1228, 0.1252, 0.1216]
    balance_genre = [0.2466, 0.2190, 0.1851, 0.1849]
    lines.extend([
        "## Phase 2: conditional seed stress\n\n",
        "Phase 1 cleared the held-out Pareto bar, so seeds 7, 123, and 2024 were "
        "run with fresh seeded initialization, fresh 100-epoch pretraining, seeded "
        "K-means, and faithful IDEC training.\n\n",
        "| seed | split | emotion AMI | genre AMI | verdict | held-out Pareto bar |\n",
        "|---:|---|---:|---:|---|---|\n",
    ])
    for result in results:
        for split_name, metrics in (("train", result["train_metrics"]), ("held-out", result["heldout_metrics"])):
            clearance = "n/a (train split)" if split_name == "train" else (
                "clears" if clears_heldout_pareto_bar(metrics) else "does not clear"
            )
            lines.append(
                f"| {result['seed']} | {split_name} | {metrics['emotion']['AMI']:.4f} | "
                f"{metrics['genre']['AMI']:.4f} | {base.verdict(metrics)} | {clearance} |\n"
            )
    lines.extend([
        "\n## Held-out stability comparison\n\n",
        f"- Faithful IDEC (`gamma={GAMMA:g}`): emotion AMI {format_summary(emotion)}; "
        f"genre AMI {format_summary(genre)}; both bars clear in {clearers}/4 seeds.\n",
        "- Balance fix (`LAMBDA_RECONSTRUCTION=1`, `LAMBDA_BALANCE=1000`, cited from "
        "`percept_stage1_seed_stress_pilot_report.md`): emotion AMI "
        f"{format_summary(balance_emotion)}; genre AMI {format_summary(balance_genre)}; "
        "both bars clear in 1/4 seeds.\n",
    ])
    if clearers > 1:
        lines.append(
            "- IDEC is more stable than the balance-term fix by the predeclared "
            "fraction-of-seeds-clearing-both-bars criterion.\n\n"
        )
    elif clearers < 1:
        lines.append(
            "- IDEC is less stable than the balance-term fix by the predeclared "
            "fraction-of-seeds-clearing-both-bars criterion.\n\n"
        )
    else:
        lines.append(
            "- IDEC matches the balance-term fix's 1/4 clearance rate; compare the "
            "reported mean/min/max AMI values for any secondary difference.\n\n"
        )


def write_report(input_dim: int, phase_one: dict, phase_two: list[dict] | None) -> None:
    """Write the conditional IDEC pilot report after its GPU execution."""
    phase_one_clearance = clears_heldout_pareto_bar(phase_one["heldout_metrics"])
    lines = [
        "# ArtELingo PercepT Stage 1 faithful IDEC pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Controlled setup\n\n",
        f"The train-only input is a {input_dim}-dimensional fused vector built with "
        "the base pilot's unchanged embedding construction. Each run uses the unchanged "
        "128-dimensional autoencoder, 100-epoch reconstruction-only pretraining, "
        "100-center K-means initialization, Student's-t assignments, self-sharpened "
        "target, convergence threshold, and 67-of-100 norm pruning.\n\n",
        f"This is IDEC in isolation: `total_loss = reconstruction_loss + {GAMMA:g} * "
        "kl_loss`. There is no balance term and no reconstruction down-weighting.\n\n",
        "## Predeclared success criterion\n\n",
        "**held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, "
        "both simultaneously.**\n\n",
    ]
    append_phase_one_results(lines, phase_one)
    if phase_two is not None:
        append_phase_two_results(lines, phase_two)
    else:
        lines.extend([
            "## Phase 2\n\n",
            "Skipped because Phase 1 did not clear the held-out Pareto bar; additional "
            "GPU seed stress would not be informative for a configuration that already "
            "missed its first test.\n\n",
        ])
    lines.extend([
        "## Interpretation\n\n",
        "The all-100-center trajectory checks conventional collapse before pruning. It "
        "also checks IDEC's distinct alternate failure mode: if reconstruction dominates, "
        "the fraction changed can quickly become negligible and the diagnostics can remain "
        "close to the epoch-0 K-means snapshot while external AMI stays weak. That outcome "
        "is not a successful non-collapse; it means clustering barely moved and did not "
        "produce informative assignments.\n\n",
        "## Decision\n\n",
    ])
    if not phase_one_clearance:
        lines.append(
            "**Merely a compromise.** IDEC's literature-standard fix did not outperform "
            "this project's own balance approach on this task at `gamma=0.1`; no further "
            "gamma sweep is in scope.\n"
        )
    elif phase_two is None:
        raise RuntimeError("A Phase-1 success must trigger Phase 2 before reporting.")
    else:
        clearers = sum(
            clears_heldout_pareto_bar(result["heldout_metrics"]) for result in phase_two
        )
        if clearers == len(phase_two):
            lines.append(
                "**Real success: robust result.** All four seeds clear the held-out "
                "Pareto bar.\n"
            )
        else:
            lines.append(
                f"**Seed-dependent result.** Only {clearers}/4 seeds clear the held-out "
                "Pareto bar, so faithful IDEC is not a reliable standing configuration.\n"
            )
    os.makedirs(REPORT_OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    """Run seed 42 first, then conditionally execute the prescribed seed stress."""
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
    log(f"Using {device} for deterministic embedding extraction and IDEC training.")
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
            f"Train/held-out fused dimensions differ: {train_h.shape[1]} vs "
            f"{heldout_h.shape[1]}."
        )
    train_inputs = torch.from_numpy(train_h)
    phase_one = run_seed(
        base.SEED, train_h, heldout_h, train_inputs, device, log, pipeline, paintings,
        train_emotions, heldout_pipeline, heldout_paintings, heldout_emotions,
    )
    phase_two = None
    if clears_heldout_pareto_bar(phase_one["heldout_metrics"]):
        log("Phase 1 clears the held-out Pareto bar; starting the prescribed seed stress.")
        phase_two = [phase_one]
        for seed in SEEDS:
            phase_two.append(run_seed(
                seed, train_h, heldout_h, train_inputs, device, log, pipeline, paintings,
                train_emotions, heldout_pipeline, heldout_paintings, heldout_emotions,
            ))
    else:
        log("Phase 1 misses the held-out Pareto bar; skipping Phase 2 seed stress.")
    write_report(train_h.shape[1], phase_one, phase_two)
    log(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
