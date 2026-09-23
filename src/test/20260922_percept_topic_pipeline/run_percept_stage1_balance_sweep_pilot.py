"""Sweep balanced-assignment strengths for ArtELingo PercepT Stage 1."""

import copy
import importlib.util
import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


OUT_DIR = os.path.dirname(__file__)
REPORT_OUT_DIR = os.environ.get("PERCEPT_OUTPUT_ROOT") or OUT_DIR
BASE_PILOT_PATH = os.path.join(OUT_DIR, "run_percept_stage1_pilot.py")
REPORT_PATH = os.path.join(REPORT_OUT_DIR, "percept_stage1_balance_sweep_pilot_report.md")
LAMBDA_BALANCE_VALUES = (10, 50, 100, 500)


def load_base_pilot():
    """Load baseline helpers without running its standalone main block."""
    spec = importlib.util.spec_from_file_location("percept_stage1_base_pilot", BASE_PILOT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import base pilot from {BASE_PILOT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_pilot()

N_INITIAL_CLUSTERS = base.N_INITIAL_CLUSTERS
N_SURVIVING_CLUSTERS = base.N_SURVIVING_CLUSTERS
MAX_DEC_EPOCHS = base.MAX_DEC_EPOCHS
STABILITY_THRESHOLD = base.STABILITY_THRESHOLD
DEC_LEARNING_RATE = base.DEC_LEARNING_RATE
LAMBDA_RECONSTRUCTION = base.LAMBDA_RECONSTRUCTION


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
    epoch: int, q: torch.Tensor, log, trajectory: list[dict[str, float]], lambda_balance: int
) -> dict[str, float]:
    """Record and log an all-center hard-assignment distribution checkpoint."""
    diagnostic = {"epoch": float(epoch), **cluster_size_diagnostic(q)}
    trajectory.append(diagnostic)
    log(
        f"lambda={lambda_balance:g} DEC cluster sizes epoch {epoch:03d}/{MAX_DEC_EPOCHS}: "
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
    lambda_balance: int,
    trajectory: list[dict[str, float]],
) -> tuple[list[dict[str, float]], str, int]:
    """Train DEC plus global assignment balancing for one sweep value."""
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + [centers],
        lr=DEC_LEARNING_RATE,
    )
    inputs = inputs.to(device)
    encoder.eval()
    with torch.no_grad():
        initial_q = base.soft_assignments(encoder(inputs), centers)
        previous_assignments = initial_q.argmax(dim=1)
        log_cluster_size_diagnostic(0, initial_q, log, trajectory, lambda_balance)
    log(
        f"Starting lambda={lambda_balance:g} joint DEC training "
        f"(up to {MAX_DEC_EPOCHS} full-batch epochs, lr={DEC_LEARNING_RATE:g}, "
        f"stability threshold={STABILITY_THRESHOLD:.3f}, balance lambda={lambda_balance:g})..."
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
        balance_loss = F.kl_div(mean_q.clamp_min(1e-8).log(), uniform, reduction="sum")
        total_loss = (
            kl_loss
            + LAMBDA_RECONSTRUCTION * reconstruction_loss
            + lambda_balance * balance_loss
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
                f"lambda={lambda_balance:g} DEC epoch {epoch:03d}/{MAX_DEC_EPOCHS}: "
                f"total={checkpoint['total']:.6f}, KL={checkpoint['kl']:.6f}, "
                f"reconstruction={checkpoint['reconstruction']:.6f}, "
                f"balance={checkpoint['balance']:.6f}"
            )
            log_cluster_size_diagnostic(epoch, current_q, log, trajectory, lambda_balance)
        if epoch % 10 == 0 or stopping:
            log(
                f"lambda={lambda_balance:g} DEC epoch {epoch:03d}/{MAX_DEC_EPOCHS}: "
                f"fraction_changed={fraction_changed:.6f} ({num_changed:,} nodes)"
            )
        if stopping:
            stop_reason = "stability criterion" if stopped_for_stability else "epoch ceiling"
            log(f"Stopping lambda={lambda_balance:g} DEC at epoch {epoch}: {stop_reason}.")
            return losses, stop_reason, epoch
    raise RuntimeError("DEC training exited without a stopping condition.")


def trajectory_summary(trajectory: list[dict[str, float]]) -> str:
    """Describe the recorded checkpoints without inferring behavior from only the final state."""
    initial = trajectory[0]
    final = trajectory[-1]
    below_counts = [point["below_one_percent"] for point in trajectory]
    max_sizes = [point["max"] for point in trajectory]
    return (
        f"Across its logged checkpoints, below-1% centers moved from "
        f"{int(initial['below_one_percent'])}/{N_INITIAL_CLUSTERS} to "
        f"{int(final['below_one_percent'])}/{N_INITIAL_CLUSTERS} "
        f"(range {int(min(below_counts))}-{int(max(below_counts))}); the largest center moved "
        f"from {int(initial['max']):,} to {int(final['max']):,} "
        f"(range {int(min(max_sizes)):,}-{int(max(max_sizes)):,})."
    )


def collapse_trend(results: list[dict]) -> str:
    """State whether held-out surviving-topic collapse improves across the ordered grid."""
    fractions = [result["heldout_metrics"]["small_clusters"] / N_SURVIVING_CLUSTERS for result in results]
    if all(later <= earlier for earlier, later in zip(fractions, fractions[1:])):
        return "Higher lambda monotonically reduces (or leaves unchanged) the held-out collapse fraction."
    if all(later >= earlier for earlier, later in zip(fractions, fractions[1:])):
        return "Higher lambda monotonically worsens (or leaves unchanged) the held-out collapse fraction."
    return "The held-out collapse fraction is non-monotonic across the lambda grid, indicating a possible intermediate sweet spot."


def refinement_recommendation(results: list[dict]) -> str:
    """Turn the observed held-out collapse trend into the required next-step recommendation."""
    fractions = [result["heldout_metrics"]["small_clusters"] / N_SURVIVING_CLUSTERS for result in results]
    if all(later <= earlier for earlier, later in zip(fractions, fractions[1:])):
        return "Because the tested range still monotonically improves or preserves the collapse fraction, a refined higher-lambda sweep is worth trying, while checking carefully for uniform-noise flattening."
    if all(later >= earlier for earlier, later in zip(fractions, fractions[1:])):
        return "Because higher lambda only worsens or preserves collapse, this balancing approach appears to have hit a wall in the tested direction."
    return "Because the result is non-monotonic, a refined sweep around the least-collapsed setting is worth trying rather than treating the approach as exhausted."


def write_report(pretrain_losses: list[float], input_dim: int, results: list[dict]) -> None:
    """Write one compact, cross-lambda report after all isolated DEC runs finish."""
    heldout_noncollapsed = [
        result for result in results if not result["heldout_metrics"]["collapsed"]
    ]
    successes = [
        result for result in results
        if base.verdict(result["heldout_metrics"]) == "Real success"
    ]
    lines = [
        "# ArtELingo PercepT Stage 1 LAMBDA_BALANCE sweep pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Controlled setup\n\n",
        f"The train-only input is a {input_dim}-dimensional fused vector built with the base pilot's embedding extraction and fusion helpers. GoEmotions-RoBERTa extraction for both splits and 100-epoch autoencoder pretraining run exactly once before the sweep. Each lambda then receives newly constructed encoder/decoder modules loaded from a deep-copied pretrained state and a fresh deterministic K-means initialization (`SEED={base.SEED}`), so no DEC-trained weights leak between sweep points.\n\n",
        "DEC retains the stabilized pilot's Student's-t assignment, self-sharpened target, reconstruction term, convergence criterion, logging cadence, and 67-of-100 center pruning. Only `LAMBDA_BALANCE` varies over 10, 50, 100, and 500.\n\n",
        "## Predeclared success criterion\n\n",
        "**held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, both simultaneously.**\n\n",
        "## Results\n\n",
        "| lambda | split | emotion AMI | genre AMI | verdict | final max cluster size (67 surviving) | clusters below 1% | fraction below 1% |\n",
        "|---:|---|---:|---:|---|---:|---:|---:|\n",
    ]
    for result in results:
        for split_name, metrics in (("train", result["train_metrics"]), ("held-out", result["heldout_metrics"])):
            sizes = metrics["cluster_sizes"]
            lines.append(
                f"| {result['lambda_balance']:g} | {split_name} | "
                f"{metrics['emotion']['AMI']:.4f} | {metrics['genre']['AMI']:.4f} | "
                f"{base.verdict(metrics)} | {int(sizes.max()):,} | "
                f"{metrics['small_clusters']}/{N_SURVIVING_CLUSTERS} | "
                f"{metrics['small_clusters'] / N_SURVIVING_CLUSTERS:.1%} |\n"
            )
    lines.extend(["\n## Final DEC cluster-size diagnostics\n\n", "Hard assignments here cover all 100 pre-pruning DEC centers; epoch 0 is the shared K-means initialization snapshot.\n\n"])
    for result in results:
        final = result["trajectory"][-1]
        lines.append(
            f"- **lambda={result['lambda_balance']:g}:** stopped at epoch {result['stop_epoch']} via {result['stop_reason']}; "
            f"final diagnostic min {int(final['min']):,}, max {int(final['max']):,}, "
            f"median {final['median']:.1f}, below 1% {int(final['below_one_percent'])}/{N_INITIAL_CLUSTERS}. "
            f"{trajectory_summary(result['trajectory'])}\n"
        )
    lines.append(
        "\nA lower collapse fraction alone is not success: at the high end, a nearly uniform hard-assignment diagnostic paired with weak external AMI is the opposite failure mode—balance-forced uniform noise rather than meaningful topic structure. The verdicts above therefore require both non-collapse and the held-out Pareto bar.\n"
    )
    noncollapsed_lambdas = ", ".join(f"{result['lambda_balance']:g}" for result in heldout_noncollapsed) or "none"
    success_lambdas = ", ".join(f"{result['lambda_balance']:g}" for result in successes) or "none"
    lines.extend([
        "\n## Decision\n\n",
        f"Under the established held-out decision convention, lambda values escaping collapse: **{noncollapsed_lambdas}**. Lambda values clearing the held-out Pareto bar: **{success_lambdas}**.\n\n",
    ])
    if successes:
        lines.append(
            f"**Real success.** Lambda {success_lambdas} is the new standing PercepT-pipeline result because it is non-collapsed and clears both held-out AMI thresholds.\n"
        )
    elif not heldout_noncollapsed:
        lines.append(
            f"**Collapsed.** No tested lambda escaped held-out collapse. {collapse_trend(results)} {refinement_recommendation(results)} This sweep therefore does not establish a new standing PercepT-pipeline result.\n"
        )
    else:
        lines.append(
            f"**Merely a compromise.** Some lambda values escape collapse, but none clear both held-out AMI thresholds. {collapse_trend(results)} {refinement_recommendation(results)} This sweep does not establish a new standing PercepT-pipeline result.\n"
        )
    os.makedirs(REPORT_OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    """Extract and pretrain once, then run four isolated DEC balance settings."""
    np.random.seed(base.SEED)
    torch.manual_seed(base.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base.SEED)

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
    log(f"Using {device} for one-time embedding extraction, pretraining, and DEC sweep training.")
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
    pretrain_losses = base.pretrain_autoencoder(
        pretrained_encoder, pretrained_decoder, train_inputs, device, log
    )
    pretrained_encoder_state = copy.deepcopy(pretrained_encoder.state_dict())
    pretrained_decoder_state = copy.deepcopy(pretrained_decoder.state_dict())
    del pretrained_encoder, pretrained_decoder

    results = []
    for lambda_balance in LAMBDA_BALANCE_VALUES:
        log(f"Starting isolated DEC sweep point lambda={lambda_balance:g} from the shared pretrained state.")
        encoder, decoder = base.build_autoencoder(train_h.shape[1])
        encoder.load_state_dict(pretrained_encoder_state)
        decoder.load_state_dict(pretrained_decoder_state)
        encoder.to(device)
        decoder.to(device)
        centers = base.initialize_cluster_centers(encoder, train_inputs, device)
        trajectory = []
        dec_losses, stop_reason, stop_epoch = train_dec_until_stable(
            encoder, decoder, centers, train_inputs, device, log, lambda_balance, trajectory
        )
        surviving_centers, surviving_indices = base.prune_centers(centers)
        log(
            f"lambda={lambda_balance:g}: pruned {N_INITIAL_CLUSTERS - N_SURVIVING_CLUSTERS} lowest-norm centers; "
            f"retained original center indices {surviving_indices.tolist()}."
        )
        encoder.eval()
        with torch.no_grad():
            train_latent = encoder(train_inputs.to(device))
            train_assignments = base.soft_assignments(train_latent, surviving_centers).argmax(dim=1).cpu().numpy()
            heldout_latent = encoder(torch.from_numpy(heldout_h).to(device))
            heldout_assignments = base.soft_assignments(heldout_latent, surviving_centers).argmax(dim=1).cpu().numpy()
        results.append({
            "lambda_balance": lambda_balance,
            "dec_losses": dec_losses,
            "stop_reason": stop_reason,
            "stop_epoch": stop_epoch,
            "trajectory": trajectory,
            "train_metrics": base.evaluate_assignments(
                pipeline, paintings, train_emotions, train_latent.cpu().numpy(), train_assignments
            ),
            "heldout_metrics": base.evaluate_assignments(
                heldout_pipeline, heldout_paintings, heldout_emotions,
                heldout_latent.cpu().numpy(), heldout_assignments
            ),
        })

    write_report(pretrain_losses, train_h.shape[1], results)
    log(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
