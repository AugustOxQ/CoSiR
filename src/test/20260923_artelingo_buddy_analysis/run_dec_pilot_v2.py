"""Run DEC with assignment-stability convergence control on ArtELingo affect nodes.

Run manually in a GPU-capable environment.  This v2 experiment keeps the
GoEmotions-only input and model from ``run_dec_pilot.py`` while replacing its
fixed-length DEC stage with the stopping rule used by DEC.
"""

import importlib.util
import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


OUT_DIR = os.path.dirname(__file__)
DEC_PILOT_PATH = os.path.join(OUT_DIR, "run_dec_pilot.py")
REPORT_PATH = os.path.join(OUT_DIR, "dec_pilot_v2_report.md")

MAX_DEC_EPOCHS = 500
STABILITY_THRESHOLD = 0.001
DEC_LEARNING_RATE = 1e-4
NUM_NODES = 61_402


def _load_dec_pilot_module():
    """Load v1 as a library so its shared experiment helpers stay canonical."""
    spec = importlib.util.spec_from_file_location("artelingo_run_dec_pilot", DEC_PILOT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helpers from {DEC_PILOT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dec_pilot = _load_dec_pilot_module()

# Keep every v1 helper that is unchanged in this controlled comparison.
load_sibling_module = dec_pilot.load_sibling_module
log = dec_pilot.log
build_autoencoder = dec_pilot.build_autoencoder
pretrain_autoencoder = dec_pilot.pretrain_autoencoder
soft_assignments = dec_pilot.soft_assignments
target_distribution = dec_pilot.target_distribution
initialize_cluster_centers = dec_pilot.initialize_cluster_centers
evaluate_clusters = dec_pilot.evaluate_clusters
final_silhouette = dec_pilot.final_silhouette

PIPELINE_PATH = dec_pilot.PIPELINE_PATH
AFFECT_PILOT_PATH = dec_pilot.AFFECT_PILOT_PATH
INPUT_DIM = dec_pilot.INPUT_DIM
N_CLUSTERS = dec_pilot.N_CLUSTERS
LAMBDA_RECONSTRUCTION = dec_pilot.LAMBDA_RECONSTRUCTION
SEED = dec_pilot.SEED


def train_dec_until_stable(
    encoder: nn.Sequential,
    decoder: nn.Sequential,
    centers: nn.Parameter,
    inputs: torch.Tensor,
    device: str,
) -> tuple[list[dict[str, float]], str, int]:
    """Jointly train DEC until hard assignments become stable or hit the cap."""
    if len(inputs) != NUM_NODES:
        raise RuntimeError(f"Expected {NUM_NODES:,} nodes, got {len(inputs):,}.")

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + [centers],
        lr=DEC_LEARNING_RATE,
    )
    inputs = inputs.to(device)
    losses = []

    encoder.eval()
    with torch.no_grad():
        previous_assignments = soft_assignments(encoder(inputs), centers).argmax(dim=1)

    log(
        "Starting joint DEC training "
        f"(up to {MAX_DEC_EPOCHS} full-batch epochs, lr={DEC_LEARNING_RATE:g}, "
        f"stability threshold={STABILITY_THRESHOLD:.3f})..."
    )
    encoder.train()
    decoder.train()
    for epoch in range(1, MAX_DEC_EPOCHS + 1):
        latent = encoder(inputs)
        reconstruction = decoder(latent)
        q = soft_assignments(latent, centers)
        p = target_distribution(q)
        kl_loss = F.kl_div(q.log(), p, reduction="batchmean", log_target=False)
        reconstruction_loss = F.mse_loss(reconstruction, inputs)
        total_loss = kl_loss + LAMBDA_RECONSTRUCTION * reconstruction_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        encoder.eval()
        with torch.no_grad():
            current_assignments = soft_assignments(encoder(inputs), centers).argmax(dim=1)
        num_changed = (current_assignments != previous_assignments).sum().item()
        fraction_changed = num_changed / NUM_NODES
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
                f"DEC epoch {epoch:03d}/{MAX_DEC_EPOCHS}: total={checkpoint['total']:.6f}, "
                f"KL={checkpoint['kl']:.6f}, reconstruction={checkpoint['reconstruction']:.6f}"
            )
        if epoch % 10 == 0 or stopping:
            log(
                f"DEC epoch {epoch:03d}/{MAX_DEC_EPOCHS}: "
                f"fraction_changed={fraction_changed:.6f} "
                f"({num_changed:,} nodes)"
            )
        if stopping:
            stop_reason = "stability criterion" if stopped_for_stability else "epoch ceiling"
            log(f"Stopping DEC at epoch {epoch}: {stop_reason}.")
            return losses, stop_reason, epoch

    raise RuntimeError("DEC training exited without reaching a stopping condition.")


def _checkpoint_epochs(losses: list[dict[str, float]]) -> list[int]:
    """Select a small, representative set of completed DEC epochs for reports."""
    final_epoch = int(losses[-1]["epoch"])
    checkpoints = {1, final_epoch}
    checkpoints.update(range(25, final_epoch + 1, 50))
    return sorted(checkpoints)


def format_dec_loss_checkpoints(losses: list[dict[str, float]]) -> str:
    """Format representative total, KL, and reconstruction loss values."""
    by_epoch = {int(checkpoint["epoch"]): checkpoint for checkpoint in losses}
    return "; ".join(
        f"epoch {epoch}: total={by_epoch[epoch]['total']:.6f}, "
        f"KL={by_epoch[epoch]['kl']:.6f}, "
        f"recon={by_epoch[epoch]['reconstruction']:.6f}"
        for epoch in _checkpoint_epochs(losses)
    )


def format_stability_checkpoints(losses: list[dict[str, float]]) -> str:
    """Format representative hard-assignment changes, including the final epoch."""
    by_epoch = {int(checkpoint["epoch"]): checkpoint for checkpoint in losses}
    checkpoints = {1, int(losses[-1]["epoch"])}
    checkpoints.update(range(10, int(losses[-1]["epoch"]) + 1, 50))
    return "; ".join(
        f"epoch {epoch}: {by_epoch[epoch]['fraction_changed']:.6f}"
        for epoch in sorted(checkpoints)
    )


def write_report(
    pretrain_losses: list[float],
    dec_losses: list[dict[str, float]],
    stop_reason: str,
    stop_epoch: int,
    cluster_sizes: np.ndarray,
    emotion_metrics: dict,
    genre_metrics: dict,
    genre_count: int,
    silhouette: float,
    silhouette_population: str,
) -> None:
    """Write the v2 convergence evidence and comparison with fixed baselines."""
    small_clusters = int(np.sum(cluster_sizes < 614))
    collapsed = small_clusters >= N_CLUSTERS / 2
    cleared_ami_bar = emotion_metrics["AMI"] > 0.177
    converged = stop_reason == "stability criterion"

    lines = [
        "# ArtELingo DEC pilot v2 — convergence-controlled\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "**Setup:** The same 61,402 x 28 mean-pooled GoEmotions sigmoid-probability "
        "nodes used by the GoEmotions-only Leiden and DEC-v1 pilots were clustered "
        "with the same 28→64→32→16→32→64→28 autoencoder and K=28 DEC centers. "
        f"Genre metrics use the {genre_count}-painting genre-labelled overlap.\n\n",
        "## Convergence and training trajectories\n\n",
        f"DEC stopped via the **{stop_reason}** at epoch **{stop_epoch}** "
        f"(threshold: fraction_changed < {STABILITY_THRESHOLD:.3f}; ceiling: {MAX_DEC_EPOCHS} epochs).\n\n",
        f"- Pretraining mean reconstruction loss: {dec_pilot.format_pretrain_checkpoints(pretrain_losses)}.\n",
        f"- Hard-assignment fraction_changed: {format_stability_checkpoints(dec_losses)}.\n",
        f"- Joint DEC losses: {format_dec_loss_checkpoints(dec_losses)}.\n\n",
        "## Cluster-size collapse detection\n\n",
        f"**Prominent collapse check:** cluster sizes range from {int(cluster_sizes.min()):,} "
        f"to {int(cluster_sizes.max()):,} nodes (median {float(np.median(cluster_sizes)):.1f}); "
        f"{small_clusters} of {N_CLUSTERS} clusters contain <1% of nodes (<614). "
        f"This {'IS' if collapsed else 'is not'} a collapse under the predeclared rule "
        "(fewer than half of clusters must be below that threshold).\n\n",
        f"Final latent silhouette score ({silhouette_population}): {silhouette:.4f}.\n\n",
        "## Comparison\n\n",
        "| method | emotion AMI | emotion V-measure | genre AMI | genre V-measure |\n",
        "|---|---:|---:|---:|---:|\n",
        "| Leiden on GoEmotions-only | 0.1180 | — | 0.0396 | — |\n",
        "| DEC v1 (100 fixed epochs, lr=1e-3, non-converged) | 0.1258 | — | 0.0365 | — |\n",
        f"| DEC v2 ({'converged' if converged else 'epoch-capped'}, lr=1e-4) | "
        f"{emotion_metrics['AMI']:.4f} | {emotion_metrics['V_measure']:.4f} | "
        f"{genre_metrics['AMI']:.4f} | {genre_metrics['V_measure']:.4f} |\n\n",
        "## Decision\n\n",
    ]
    convergence_statement = (
        "Convergence did occur: the stability criterion fired before the epoch ceiling."
        if converged
        else "Convergence did not occur before the safety ceiling: the run was epoch-capped."
    )
    ami_statement = (
        f"The result clears the predeclared emotion AMI > 0.177 bar (actual "
        f"AMI={emotion_metrics['AMI']:.4f})."
        if cleared_ami_bar
        else f"The result does not clear the predeclared emotion AMI > 0.177 bar "
        f"(actual AMI={emotion_metrics['AMI']:.4f})."
    )
    conclusion = (
        "v2 changes the v1 conclusion."
        if cleared_ami_bar and not collapsed
        else "v2 confirms that the v1 conclusion holds even with proper convergence control."
    )
    lines.append(f"{convergence_statement} {ami_statement} {conclusion}\n")

    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    pipeline = load_sibling_module("artelingo_run_pipeline", PIPELINE_PATH)
    affect_pilot = load_sibling_module("artelingo_run_affect_pilot", AFFECT_PILOT_PATH)

    pipeline.assert_extraction_complete()
    log("Loading and deduplicating cached CLIP features...")
    paintings, _, _, emotion_counts = pipeline.load_dedup_features()
    majority_emotion = [pipeline.majority(counts) for counts in emotion_counts]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {device} for GoEmotions extraction and DEC training.")

    log("Extracting GoEmotions affect-only nodes...")
    affect_nodes = affect_pilot.extract_affect_nodes(pipeline.TRAIN_JSON, paintings, device)
    if affect_nodes.shape != (len(paintings), INPUT_DIM):
        raise RuntimeError(
            f"Expected GoEmotions nodes shaped ({len(paintings)}, {INPUT_DIM}), got {affect_nodes.shape}."
        )
    affect_nodes = affect_pilot.l2_normalize(affect_nodes).astype(np.float32, copy=False)
    inputs = torch.from_numpy(affect_nodes.astype(np.float32, copy=False))

    encoder, decoder = build_autoencoder()
    encoder.to(device)
    decoder.to(device)
    pretrain_losses = pretrain_autoencoder(encoder, decoder, inputs, device)
    centers = initialize_cluster_centers(encoder, inputs, device)
    dec_losses, stop_reason, stop_epoch = train_dec_until_stable(
        encoder, decoder, centers, inputs, device
    )

    log("Evaluating final DEC assignments and latent-space separation...")
    encoder.eval()
    with torch.no_grad():
        final_latent_tensor = encoder(inputs.to(device))
        final_q = soft_assignments(final_latent_tensor, centers)
        assignments = final_q.argmax(dim=1).cpu().numpy()
        final_latent = final_latent_tensor.cpu().numpy()
    cluster_sizes = np.bincount(assignments, minlength=N_CLUSTERS)
    small_clusters = int(np.sum(cluster_sizes < 614))
    log(
        "COLLAPSE CHECK: "
        f"min={cluster_sizes.min()}, max={cluster_sizes.max()}, median={np.median(cluster_sizes):.1f}, "
        f"clusters below 1%={small_clusters}/{N_CLUSTERS}."
    )

    emotion_metrics, genre_metrics, genre_count = evaluate_clusters(
        pipeline, assignments, paintings, majority_emotion
    )
    silhouette, silhouette_population = final_silhouette(final_latent, assignments)
    log(
        f"Final metrics: emotion AMI={emotion_metrics['AMI']:.4f}, "
        f"genre AMI={genre_metrics['AMI']:.4f}, silhouette={silhouette:.4f} ({silhouette_population})."
    )
    write_report(
        pretrain_losses,
        dec_losses,
        stop_reason,
        stop_epoch,
        cluster_sizes,
        emotion_metrics,
        genre_metrics,
        genre_count,
        silhouette,
        silhouette_population,
    )
    log(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
