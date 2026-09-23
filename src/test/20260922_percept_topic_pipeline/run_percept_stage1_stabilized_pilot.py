"""Run the stabilized ArtELingo PercepT Stage 1 P-Topic Formation pilot.

This controlled follow-up keeps the original pilot pipeline unchanged while
adding full-batch assignment balancing and diagnostics for DEC collapse.
"""

import importlib.util
import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


OUT_DIR = os.path.dirname(__file__)
BASE_PILOT_PATH = os.path.join(OUT_DIR, "run_percept_stage1_pilot.py")
REPORT_PATH = os.path.join(OUT_DIR, "percept_stage1_stabilized_pilot_report.md")
LAMBDA_BALANCE = 1.0


def load_base_pilot():
    """Load the original standalone pilot so all non-stabilization behavior matches."""
    spec = importlib.util.spec_from_file_location("percept_stage1_base_pilot", BASE_PILOT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import base pilot from {BASE_PILOT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_pilot()

# Keep these module-level constants explicit so this pilot's controlled change
# is visible without inspecting the imported baseline.
N_INITIAL_CLUSTERS = base.N_INITIAL_CLUSTERS
N_SURVIVING_CLUSTERS = base.N_SURVIVING_CLUSTERS
MAX_DEC_EPOCHS = base.MAX_DEC_EPOCHS
STABILITY_THRESHOLD = base.STABILITY_THRESHOLD
DEC_LEARNING_RATE = base.DEC_LEARNING_RATE
LAMBDA_RECONSTRUCTION = base.LAMBDA_RECONSTRUCTION

CLUSTER_SIZE_TRAJECTORY: list[dict[str, float]] = []


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


def log_cluster_size_diagnostic(epoch: int, q: torch.Tensor, log) -> dict[str, float]:
    """Record and log an all-center hard-assignment distribution checkpoint."""
    diagnostic = {"epoch": float(epoch), **cluster_size_diagnostic(q)}
    CLUSTER_SIZE_TRAJECTORY.append(diagnostic)
    log(
        f"DEC cluster sizes epoch {epoch:03d}/{MAX_DEC_EPOCHS}: "
        f"min={int(diagnostic['min']):,}, max={int(diagnostic['max']):,}, "
        f"median={diagnostic['median']:.1f}, "
        f"below_1pct={int(diagnostic['below_one_percent'])}/{N_INITIAL_CLUSTERS}"
    )
    return diagnostic


def train_dec_until_stable(
    encoder: nn.Sequential,
    decoder: nn.Sequential,
    centers: nn.Parameter,
    inputs: torch.Tensor,
    device: str,
    log,
) -> tuple[list[dict[str, float]], str, int]:
    """Train the baseline DEC objective plus global assignment balancing."""
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + [centers],
        lr=DEC_LEARNING_RATE,
    )
    inputs = inputs.to(device)
    encoder.eval()
    with torch.no_grad():
        initial_q = base.soft_assignments(encoder(inputs), centers)
        previous_assignments = initial_q.argmax(dim=1)
        log_cluster_size_diagnostic(0, initial_q, log)
    log(
        "Starting joint DEC training "
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
            "balance": balance_loss.item(),
            "fraction_changed": fraction_changed,
        }
        losses.append(checkpoint)
        stopped_for_stability = fraction_changed < STABILITY_THRESHOLD
        reached_epoch_ceiling = epoch == MAX_DEC_EPOCHS
        stopping = stopped_for_stability or reached_epoch_ceiling
        if epoch % 25 == 0 or stopping:
            log(
                f"DEC epoch {epoch:03d}/{MAX_DEC_EPOCHS}: "
                f"total={checkpoint['total']:.6f}, KL={checkpoint['kl']:.6f}, "
                f"reconstruction={checkpoint['reconstruction']:.6f}, "
                f"balance={checkpoint['balance']:.6f}"
            )
            log_cluster_size_diagnostic(epoch, current_q, log)
        if epoch % 10 == 0 or stopping:
            log(
                f"DEC epoch {epoch:03d}/{MAX_DEC_EPOCHS}: "
                f"fraction_changed={fraction_changed:.6f} ({num_changed:,} nodes)"
            )
        if stopping:
            stop_reason = "stability criterion" if stopped_for_stability else "epoch ceiling"
            log(f"Stopping DEC at epoch {epoch}: {stop_reason}.")
            return losses, stop_reason, epoch
    raise RuntimeError("DEC training exited without a stopping condition.")


def format_dec_losses(losses: list[dict[str, float]]) -> str:
    by_epoch = {int(loss["epoch"]): loss for loss in losses}
    checkpoints = sorted({1, len(losses), *range(25, len(losses) + 1, 50)})
    return "; ".join(
        f"epoch {epoch}: total={by_epoch[epoch]['total']:.6f}, "
        f"KL={by_epoch[epoch]['kl']:.6f}, "
        f"recon={by_epoch[epoch]['reconstruction']:.6f}, "
        f"balance={by_epoch[epoch]['balance']:.6f}, "
        f"fraction_changed={by_epoch[epoch]['fraction_changed']:.6f}"
        for epoch in checkpoints
    )


def format_cluster_size_trajectory() -> str:
    return "\n".join(
        f"- Epoch {int(point['epoch'])}: min {int(point['min']):,}; "
        f"max {int(point['max']):,}; median {point['median']:.1f}; "
        f"below 1%: {int(point['below_one_percent'])}/{N_INITIAL_CLUSTERS}.\n"
        for point in CLUSTER_SIZE_TRAJECTORY
    )


def write_report(
    pretrain_losses: list[float], dec_losses: list[dict[str, float]],
    stop_reason: str, stop_epoch: int, input_dim: int,
    train_metrics: dict, heldout_metrics: dict,
) -> None:
    """Write the baseline report plus stabilization-specific trajectories."""
    final_verdict = base.verdict(heldout_metrics)
    lines = [
        "# ArtELingo PercepT Stage 1 stabilized P-Topic Formation pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Setup and documented deviations\n\n",
        f"The train-only input is a {input_dim}-dimensional fused vector. Content is the existing independently normalized CLIP image/text concatenation, normalized again as a whole; affect is a 768-dimensional embedding from masked-token mean pooling of `SamLowe/roberta-base-go_emotions`, then caption-mean pooling by painting. This is an embedding-level affect signal, a methodological upgrade over all earlier investigation pilots' 28-dimensional label probabilities. RoBERTa substitutes for the paper's ModernBERT-family GoEmotions encoder because this cached, validated project encoder shares the fine-tuning objective but not the backbone family.\n\n",
        "Because this repository's ViT-B/32 CLIP content vector and the 768-dimensional affect vector have unequal dimensions, literal Eq. 2 summation is impossible without an unvalidated projection. The documented substitute is `L2_normalize(concat([h_C', h_C', h_E]))`: repeated content preserves the paper's 2:1 content:affect norm-budget weighting.\n\n",
        "DEC uses convergence-controlled full-batch training (lr=1e-4; stop at `fraction_changed < 0.001`, ceiling 500) and retains the 67 highest-L2-norm centroids out of 100, exactly as in the collapsed base pilot. The sole stabilization intervention is a full-batch balanced-assignment penalty, `LAMBDA_BALANCE = 1.0`, on the KL divergence between the global mean soft assignment and uniform. `LAMBDA_RECONSTRUCTION` remains 1.0.\n\n",
        "## Predeclared success criterion\n\n",
        "**held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, both simultaneously.**\n\n",
        "## Training trajectories\n\n",
        f"- Pretraining reconstruction: {base.format_pretrain_losses(pretrain_losses)}.\n",
        f"- DEC stopped via the **{stop_reason}** at epoch **{stop_epoch}** (threshold `fraction_changed < {STABILITY_THRESHOLD:.3f}`, ceiling {MAX_DEC_EPOCHS}).\n",
        f"- Joint DEC: {format_dec_losses(dec_losses)}.\n\n",
        "## Cluster-size distribution over training\n\n",
        "Hard assignments are measured over all 100 pre-pruning DEC centers. Epoch 0 is the K-means initialization snapshot.\n\n",
        format_cluster_size_trajectory(),
        "\n## Cluster-size collapse detection\n\n",
    ]
    for split_name, metrics in (("Train", train_metrics), ("Held-out", heldout_metrics)):
        sizes = metrics["cluster_sizes"]
        lines.append(
            f"- {split_name}: sizes range from {int(sizes.min()):,} to {int(sizes.max()):,} "
            f"(median {float(np.median(sizes)):.1f}); {metrics['small_clusters']}/{N_SURVIVING_CLUSTERS} clusters are below 1% of assigned nodes. This {'is' if metrics['collapsed'] else 'is not'} collapsed under the predeclared rule (>50% of clusters below 1%).\n"
        )
    lines.extend([
        "\n## Results\n\n",
        "| architecture | split | emotion AMI | emotion V-measure | genre AMI | genre V-measure | collapse verdict | silhouette |\n",
        "|---|---|---:|---:|---:|---:|---|---:|\n",
    ])
    for split_name, metrics in (("train", train_metrics), ("held-out", heldout_metrics)):
        lines.append(
            f"| PercepT Stage 1 fused AE+DEC + balance (67 surviving topics) | {split_name} | {metrics['emotion']['AMI']:.4f} | {metrics['emotion']['V_measure']:.4f} | {metrics['genre']['AMI']:.4f} | {metrics['genre']['V_measure']:.4f} | {'Collapsed' if metrics['collapsed'] else 'not collapsed'} | {metrics['silhouette']:.4f} |\n"
        )
    lines.extend([
        "\n## Before/after comparison with collapsed base pilot\n\n",
        "| pilot | split | emotion AMI | genre AMI | collapse verdict |\n",
        "|---|---|---:|---:|---|\n",
        "| Collapsed base | train | 0.0478 | 0.3281 | Collapsed |\n",
        "| Collapsed base | held-out | 0.0363 | 0.3081 | Collapsed |\n",
        f"| Stabilized balance | train | {train_metrics['emotion']['AMI']:.4f} | {train_metrics['genre']['AMI']:.4f} | {'Collapsed' if train_metrics['collapsed'] else 'not collapsed'} |\n",
        f"| Stabilized balance | held-out | {heldout_metrics['emotion']['AMI']:.4f} | {heldout_metrics['genre']['AMI']:.4f} | {'Collapsed' if heldout_metrics['collapsed'] else 'not collapsed'} |\n",
        "\nGenre metrics use the genre-labelled overlap for each split (train n="
        f"{train_metrics['genre_count']:,}; held-out n={heldout_metrics['genre_count']:,}).\n\n",
        "The paper's reported 0.97 silhouette was measured on its own held-out fused-embedding input directly. This pilot instead reports a 128-dimensional DEC latent alongside label-based external metrics, so a large silhouette gap is not itself a failure signal; AMI against real labels is the primary criterion.\n\n",
        "## Decision\n\n",
        f"**{final_verdict}.** " + (
            "The cluster-size collapse rule fired.\n" if final_verdict == "Collapsed" else
            "The held-out Pareto bar clears both thresholds.\n" if final_verdict == "Real success" else
            "The pilot did not simultaneously clear both held-out AMI thresholds.\n"
        ),
    ])
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    """Run the exact baseline pipeline with only the functions above replaced."""
    CLUSTER_SIZE_TRAJECTORY.clear()
    base.REPORT_PATH = REPORT_PATH
    base.train_dec_until_stable = train_dec_until_stable
    base.write_report = write_report
    base.main()


if __name__ == "__main__":
    main()
