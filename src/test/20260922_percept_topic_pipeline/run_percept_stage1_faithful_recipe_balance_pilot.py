"""Screen balance on the faithful PercepT Stage 1 recipe, then stress one winner.

Only Variant A is used: latent Gaussian noise during pretraining, clean latent
vectors throughout joint DEC. Run this script separately on a GPU.
"""

import copy
import os
import time

import numpy as np
import torch
from torch.nn import functional as F


OUT_DIR = os.path.dirname(__file__)
REPORT_OUT_DIR = os.environ.get("PERCEPT_OUTPUT_ROOT") or OUT_DIR
FAITHFUL_PATH = os.path.join(OUT_DIR, "run_percept_stage1_faithful_recipe_pilot.py")
REPORT_PATH = os.path.join(
    REPORT_OUT_DIR, "percept_stage1_faithful_recipe_balance_pilot_report.md"
)

SEED = 42
SEEDS = (7, 123, 2024)
# Faithful DEC converged at KL ~= 0.39-0.40; the earlier balance sweep's KL
# was ~= 0.09. These are of similar order, so this bracket is a reasonable
# starting screen rather than a range transplanted from a differently scaled loss.
LAMBDA_BALANCE_VALUES = (100, 300, 1000)
LAMBDA_RECONSTRUCTION = 1.0


def load_module(module_name: str, path: str):
    """Use the same standalone sibling-import convention as the faithful pilot."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


faithful = load_module("percept_stage1_faithful_recipe", FAITHFUL_PATH)
base = faithful.base


def log_cluster_size_diagnostic(epoch, q, log, trajectory, seed, lambda_balance):
    diagnostic = {"epoch": epoch, **faithful.cluster_size_diagnostic(q)}
    trajectory.append(diagnostic)
    log(
        f"seed={seed} lambda={lambda_balance:g} DEC cluster sizes epoch "
        f"{epoch:03d}/{faithful.MAX_DEC_EPOCHS}: min={diagnostic['min']:,}, "
        f"max={diagnostic['max']:,}, median={diagnostic['median']:.1f}, "
        f"below_1pct={diagnostic['below_one_percent']}/{faithful.N_INITIAL_CLUSTERS}"
    )


def train_dec_until_stable(encoder, decoder, centers, inputs, device, log,
                           seed, lambda_balance):
    """Faithful Variant-A DEC with the prior sweep's exact balance loss."""
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + [centers],
        lr=faithful.DEC_LEARNING_RATE,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=faithful.DEC_COSINE_T_MAX, eta_min=faithful.LR_FLOOR
    )
    inputs = inputs.to(device)
    trajectory = []
    encoder.eval()
    with torch.no_grad():
        initial_q = base.soft_assignments(encoder(inputs), centers)
        previous_assignments = initial_q.argmax(dim=1)
        log_cluster_size_diagnostic(0, initial_q, log, trajectory, seed, lambda_balance)
    log(
        f"Starting seed={seed} lambda={lambda_balance:g} joint DEC training "
        f"(up to {faithful.MAX_DEC_EPOCHS} full-batch epochs, "
        f"lr={faithful.DEC_LEARNING_RATE:g}, cosine T_max={faithful.DEC_COSINE_T_MAX}, "
        f"eta_min={faithful.LR_FLOOR:g}, "
        f"stability threshold={faithful.STABILITY_THRESHOLD:.3f})..."
    )
    losses = []
    encoder.train()
    decoder.train()
    for epoch in range(1, faithful.MAX_DEC_EPOCHS + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        z = encoder(inputs)
        reconstruction = decoder(z)  # Variant A: clean z in joint DEC.
        q = base.soft_assignments(z, centers)
        p = base.target_distribution(q)
        kl_loss = F.kl_div(q.log(), p, reduction="batchmean", log_target=False)
        reconstruction_loss = F.mse_loss(reconstruction, inputs)
        mean_q = q.mean(dim=0)
        uniform = torch.full_like(mean_q, 1.0 / mean_q.shape[0])
        balance_loss = F.kl_div(mean_q.clamp_min(1e-8).log(), uniform, reduction="sum")
        total_loss = (
            kl_loss + LAMBDA_RECONSTRUCTION * reconstruction_loss
            + lambda_balance * balance_loss
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # Keep the faithful schedule at its floor after T_max.
        if epoch <= faithful.DEC_COSINE_T_MAX:
            scheduler.step()

        encoder.eval()
        with torch.no_grad():
            current_q = base.soft_assignments(encoder(inputs), centers)
            current_assignments = current_q.argmax(dim=1)
        num_changed = (current_assignments != previous_assignments).sum().item()
        fraction_changed = num_changed / len(inputs)
        previous_assignments = current_assignments
        encoder.train()
        checkpoint = {
            "epoch": epoch,
            "total": total_loss.item(),
            "kl": kl_loss.item(),
            "reconstruction": reconstruction_loss.item(),
            "balance": balance_loss.item(),
            "fraction_changed": fraction_changed,
            "lr": learning_rate,
        }
        losses.append(checkpoint)
        stopped_for_stability = fraction_changed < faithful.STABILITY_THRESHOLD
        stopping = stopped_for_stability or epoch == faithful.MAX_DEC_EPOCHS
        if epoch % 25 == 0 or stopping:
            log(
                f"seed={seed} lambda={lambda_balance:g} DEC epoch "
                f"{epoch:03d}/{faithful.MAX_DEC_EPOCHS}: "
                f"total={checkpoint['total']:.6f}, KL={checkpoint['kl']:.6f}, "
                f"reconstruction={checkpoint['reconstruction']:.6f}, "
                f"balance={checkpoint['balance']:.6f}, lr={learning_rate:.8g}"
            )
            log_cluster_size_diagnostic(
                epoch, current_q, log, trajectory, seed, lambda_balance
            )
        if epoch % 10 == 0 or stopping:
            log(
                f"seed={seed} lambda={lambda_balance:g} DEC epoch "
                f"{epoch:03d}/{faithful.MAX_DEC_EPOCHS}: "
                f"fraction_changed={fraction_changed:.6f} ({num_changed:,} nodes), "
                f"lr={learning_rate:.8g}"
            )
        if stopping:
            reason = "stability criterion" if stopped_for_stability else "epoch ceiling"
            log(f"Stopping seed={seed} lambda={lambda_balance:g} DEC at epoch {epoch}: {reason}.")
            return losses, trajectory, reason, epoch
    raise RuntimeError("DEC training exited without a stopping condition.")


def run_seed(seed, lambda_values, train_h, heldout_h, train_inputs, device, log,
             pipeline, paintings, train_emotions, heldout_pipeline,
             heldout_paintings, heldout_emotions):
    """Pretrain afresh per seed; isolate each requested DEC setting."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pretrained_encoder, pretrained_decoder = base.build_autoencoder(train_h.shape[1])
    pretrained_encoder.to(device)
    pretrained_decoder.to(device)
    pretrain_losses = faithful.pretrain_autoencoder(
        pretrained_encoder, pretrained_decoder, train_inputs, device, log, seed
    )
    encoder_state = copy.deepcopy(pretrained_encoder.state_dict())
    decoder_state = copy.deepcopy(pretrained_decoder.state_dict())
    del pretrained_encoder, pretrained_decoder

    results = []
    for lambda_balance in lambda_values:
        log(f"Starting seed={seed} lambda={lambda_balance:g} from pretrained state.")
        encoder, decoder = base.build_autoencoder(train_h.shape[1])
        encoder.load_state_dict(encoder_state)
        decoder.load_state_dict(decoder_state)
        encoder.to(device)
        decoder.to(device)
        centers = faithful.initialize_cluster_centers(
            encoder, train_inputs, device, seed
        )
        dec_losses, trajectory, stop_reason, stop_epoch = train_dec_until_stable(
            encoder, decoder, centers, train_inputs, device, log, seed, lambda_balance
        )
        surviving_centers, surviving_indices = base.prune_centers(centers)
        log(
            f"seed={seed} lambda={lambda_balance:g}: stopped at epoch "
            f"{stop_epoch} via {stop_reason}; pruned "
            f"{faithful.N_INITIAL_CLUSTERS - faithful.N_SURVIVING_CLUSTERS} "
            f"centers and retained original indices {surviving_indices.tolist()}."
        )
        encoder.eval()
        with torch.no_grad():
            train_latent = encoder(train_inputs.to(device))
            train_assignments = (
                base.soft_assignments(train_latent, surviving_centers)
                .argmax(dim=1).cpu().numpy()
            )
            heldout_latent = encoder(torch.from_numpy(heldout_h).to(device))
            heldout_assignments = (
                base.soft_assignments(heldout_latent, surviving_centers)
                .argmax(dim=1).cpu().numpy()
            )
        # Faithful evaluator computes silhouette on final 128-D Z for each split,
        # including its one-cluster fallback when silhouette is undefined.
        train_metrics = faithful.evaluate_assignments(
            pipeline, paintings, train_emotions,
            train_latent.cpu().numpy(), train_assignments,
        )
        heldout_metrics = faithful.evaluate_assignments(
            heldout_pipeline, heldout_paintings, heldout_emotions,
            heldout_latent.cpu().numpy(), heldout_assignments,
        )
        results.append({
            "seed": seed,
            "lambda_balance": lambda_balance,
            "pretrain_losses": pretrain_losses,
            "dec_losses": dec_losses,
            "trajectory": trajectory,
            "stop_reason": stop_reason,
            "stop_epoch": stop_epoch,
            "train_metrics": train_metrics,
            "heldout_metrics": heldout_metrics,
        })
        del encoder, decoder, centers
    return results


def format_dec_losses(losses):
    by_epoch = {point["epoch"]: point for point in losses}
    checkpoints = sorted({
        1, len(losses), *range(10, len(losses) + 1, 10),
        *range(25, len(losses) + 1, 25),
    })
    return "; ".join(
        f"epoch {epoch}: total={by_epoch[epoch]['total']:.6f}, "
        f"KL={by_epoch[epoch]['kl']:.6f}, "
        f"recon={by_epoch[epoch]['reconstruction']:.6f}, "
        f"balance={by_epoch[epoch]['balance']:.6f}, "
        f"fraction_changed={by_epoch[epoch]['fraction_changed']:.6f}, "
        f"lr={by_epoch[epoch]['lr']:.8g}"
        for epoch in checkpoints
    )


def append_trajectories(lines, results):
    for result in results:
        lines.extend([
            f"### Lambda {result['lambda_balance']:g}, seed {result['seed']}\n\n",
            f"- Pretraining reconstruction and LR: "
            f"{faithful.format_pretrain_losses(result['pretrain_losses'])}.\n",
            f"- DEC stopped via **{result['stop_reason']}** at epoch "
            f"**{result['stop_epoch']}** (threshold "
            f"`fraction_changed < {faithful.STABILITY_THRESHOLD:.3f}`, "
            f"ceiling {faithful.MAX_DEC_EPOCHS}).\n",
            f"- Joint DEC losses and LR: {format_dec_losses(result['dec_losses'])}.\n",
            "- All-100-center diagnostics (epoch 0 and every 25 epochs or stop): "
            f"{faithful.format_trajectory(result['trajectory'])}.\n\n",
        ])


def append_results_table(lines, results, include_source=False):
    lines.append(
        "| lambda | seed | "
        + ("source | " if include_source else "")
        + "split | emotion AMI | genre AMI | silhouette (128-D Z) | "
        "verdict | held-out Pareto bar | surviving min | surviving max | "
        "surviving median | surviving below 1% | all-100 min | all-100 max | "
        "all-100 median | all-100 below 1% |\n"
    )
    lines.append(
        "|---:|---:|" + ("---|" if include_source else "")
        + "---|---:|---:|---:|---|---|---:|---:|---:|---:|"
        "---:|---:|---:|---:|\n"
    )
    for result in results:
        final_all = result["trajectory"][-1]
        for split_name, metrics in (
            ("train", result["train_metrics"]),
            ("held-out", result["heldout_metrics"]),
        ):
            sizes = metrics["cluster_sizes"]
            silhouette = (
                f"{metrics['silhouette']:.4f}"
                if np.isfinite(metrics["silhouette"]) else "n/a (one cluster)"
            )
            clearance = (
                "n/a (train split)" if split_name == "train" else
                "clears" if faithful.clears_heldout_pareto_bar(metrics)
                else "does not clear"
            )
            source = "reused Stage 1" if result["seed"] == SEED else "newly measured"
            source_column = f"{source} | " if include_source else ""
            lines.append(
                f"| {result['lambda_balance']:g} | {result['seed']} | "
                f"{source_column}{split_name} | "
                f"{metrics['emotion']['AMI']:.4f} | {metrics['genre']['AMI']:.4f} | "
                f"{silhouette} | {base.verdict(metrics)} | {clearance} | "
                f"{int(sizes.min()):,} | {int(sizes.max()):,} | "
                f"{float(np.median(sizes)):.1f} | "
                f"{metrics['small_clusters']}/{faithful.N_SURVIVING_CLUSTERS} | "
                f"{final_all['min']:,} | {final_all['max']:,} | "
                f"{final_all['median']:.1f} | "
                f"{final_all['below_one_percent']}/{faithful.N_INITIAL_CLUSTERS} |\n"
            )


def write_report(input_dim, stage_one, winner, stage_two):
    """Write the complete conditional report after separate GPU execution."""
    lines = [
        "# ArtELingo PercepT Stage 1 faithful recipe + balance pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Controlled setup\n\n",
        f"The train-only input is the base pilot's {input_dim}-dimensional fused "
        "vector: independently normalized CLIP content and GoEmotions-RoBERTa "
        "affect embeddings, with repeated content for a 2:1 norm budget. "
        "The 128-D autoencoder, Student's-t assignment, self-sharpened target, "
        "100-center seeded K-means, and 67 highest-norm center pruning follow "
        "the faithful-recipe pilot.\n\n",
        "Only Variant A is used: Gaussian latent noise with standard deviation "
        "0.1 during the 100-epoch pretrain, and clean z for reconstruction, KL, "
        "and balance during joint DEC. Both phases use Adam at initial LR 1e-3 "
        f"and cosine annealing to {faithful.LR_FLOOR:g}; pretrain T_max=100, "
        "DEC T_max=200 and held at the floor thereafter. DEC keeps "
        f"`fraction_changed < {faithful.STABILITY_THRESHOLD:.3f}` and the "
        f"{faithful.MAX_DEC_EPOCHS}-epoch ceiling. Joint loss is KL + "
        "1.0 * MSE reconstruction + lambda_balance * "
        "KL(uniform || mean soft assignment).\n\n",
        "Stage 1 isolates lambda_balance in {100, 300, 1000} at seed 42 "
        "after a shared fresh pretrain. Stage 2, when gated in, uses fresh "
        "autoencoders, noisy pretraining, and seeded K-means at seeds 7, 123, "
        "and 2024 for only the chosen lambda.\n\n",
        "## Predeclared success criterion\n\n",
        "**Held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > "
        "0.1954, both simultaneously.** Real success additionally requires "
        "non-collapse under the surviving-center rule. Silhouette uses final "
        "128-D Z and surviving-center assignments separately on both splits; "
        "it is n/a if only one cluster is assigned.\n\n",
        "## Stage 1: seed 42 balance screen\n\n",
        "### Training and cluster-size trajectories\n\n",
    ]
    append_trajectories(lines, stage_one)
    lines.append("### Results on both splits\n\n")
    append_results_table(lines, stage_one)
    lines.append(
        "\nSurviving-center diagnostics are split-specific. All-100 diagnostics "
        "repeat the final train checkpoint before pruning.\n\n"
    )

    if winner is None:
        lines.extend([
            "## Stage 2\n\n",
            "Skipped: none of the three screened values cleared both held-out "
            "Pareto bars at seed 42. All three are plain misses. The lambda "
            "range was not widened.\n\n",
            "## Decision\n\n",
            "**No held-out Pareto success.** All three prescribed lambda values "
            "missed at least one bar; no Stage-2 seed stress was run. "
            "This pilot does not establish a new standing PercepT Stage 1 "
            "configuration.\n",
        ])
    else:
        winning_lambda = winner["lambda_balance"]
        all_results = [winner, *stage_two]
        heldout = [result["heldout_metrics"] for result in all_results]
        emotion = [metrics["emotion"]["AMI"] for metrics in heldout]
        genre = [metrics["genre"]["AMI"] for metrics in heldout]
        emotion_clearers = sum(value > faithful.EMOTION_PARETO_BAR for value in emotion)
        genre_clearers = sum(value > faithful.GENRE_PARETO_BAR for value in genre)
        both_clearers = sum(faithful.clears_heldout_pareto_bar(m) for m in heldout)
        real_successes = sum(base.verdict(m) == "Real success" for m in heldout)
        misses = [
            (result["seed"], faithful.EMOTION_PARETO_BAR - result["heldout_metrics"]["emotion"]["AMI"],
             faithful.GENRE_PARETO_BAR - result["heldout_metrics"]["genre"]["AMI"])
            for result in all_results
            if not faithful.clears_heldout_pareto_bar(result["heldout_metrics"])
        ]
        lines.extend([
            "## Stage 2: winner-only seed stress\n\n",
            f"Lambda {winning_lambda:g} was selected among Stage-1 clearers by "
            "the point with the largest emotion margin. Seed 42 is reused "
            "from Stage 1; seeds 7, 123, and 2024 are newly measured.\n\n",
            "### Four-seed results\n\n",
        ])
        append_results_table(lines, all_results, include_source=True)
        lines.append("\n### Additional-seed trajectories\n\n")
        append_trajectories(lines, stage_two)
        lines.extend([
            "### Held-out summary statistics\n\n",
            f"- Emotion AMI across four seeds: {faithful.format_summary(emotion)}; "
            f"clears its individual bar in {emotion_clearers}/4 seeds.\n",
            f"- Genre AMI across four seeds: {faithful.format_summary(genre)}; "
            f"clears its individual bar in {genre_clearers}/4 seeds.\n",
            f"- Both bars clear simultaneously in {both_clearers}/4 seeds; "
            f"{real_successes}/4 also satisfy non-collapse.\n\n",
            "### Three-way mechanism comparison (held-out)\n\n",
            "| configuration | emotion AMI | genre AMI | silhouette | "
            "both-bar-clearing seeds |\n",
            "|---|---:|---:|---:|---:|\n",
            f"| Faithful recipe + balance, lambda={winning_lambda:g} (four-seed "
            f"means) | {np.mean(emotion):.4f} | {np.mean(genre):.4f} | "
            "see four-seed table | "
            f"{both_clearers}/4 |\n",
            "| Standing K=60/40, LAMBDA_BALANCE=1000 (four-seed means) | "
            "0.1252 | 0.2486 | n/a | 4/4 |\n",
            "| Faithful recipe alone, Variant A (seed 42) | "
            "0.1092 | 0.3288 | 0.5120 | 0/1 |\n\n",
        ])
        standing_ami_better = (
            np.mean(emotion) > faithful.STANDING_EMOTION_MEAN
            and np.mean(genre) > faithful.STANDING_GENRE_MEAN
        )
        faithful_ami_better = np.mean(emotion) > 0.1092 and np.mean(genre) > 0.3288
        lines.append(
            f"Combined mechanism {'beats' if standing_ami_better else 'does not beat'} "
            "the standing configuration on both mean AMIs; it "
            f"{'matches' if both_clearers == 4 else 'falls below'} its 4/4 "
            "both-bar-clearing seed count. It "
            f"{'beats' if faithful_ami_better else 'does not beat'} the faithful "
            "recipe alone on both reported AMIs and "
            f"{'beats' if both_clearers > 0 else 'matches'} its 0/1 both-bar "
            "seed count. "
            "The faithful-recipe comparison uses one seed; the other rows "
            "use four, so this is descriptive rather than a paired estimate.\n\n"
        )
        lines.append("## Decision\n\n")
        if real_successes == 4:
            lines.append(
                "**Real success: robust result.** All 4/4 seeds are "
                "non-collapsed and clear both held-out bars.\n"
            )
        else:
            miss_text = "; ".join(
                f"seed {seed} misses " + " and ".join(
                    part for part in (
                        f"emotion by {emotion_miss:.4f}" if emotion_miss >= 0 else "",
                        f"genre by {genre_miss:.4f}" if genre_miss >= 0 else "",
                    ) if part
                )
                for seed, emotion_miss, genre_miss in misses
            ) or "no AMI misses; the remaining seeds failed the non-collapse rule"
            lines.append(
                f"**Seed-dependent result.** {both_clearers}/4 seeds clear "
                f"both held-out bars and {real_successes}/4 are also "
                f"non-collapsed. {miss_text}. This pilot does not establish "
                "a robust new standing configuration.\n"
            )
    os.makedirs(REPORT_OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main():
    """Extract once, screen exactly three values, gate one winner's seed stress."""
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
    heldout_paintings, heldout_img_nodes, heldout_txt_nodes, heldout_counts = (
        heldout_pipeline.load_dedup_features()
    )
    heldout_emotions = [heldout_pipeline.majority(counts) for counts in heldout_counts]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {device} for deterministic embedding extraction and faithful DEC training.")
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
    args = (
        train_h, heldout_h, train_inputs, device, log, pipeline, paintings,
        train_emotions, heldout_pipeline, heldout_paintings, heldout_emotions,
    )
    stage_one = run_seed(SEED, LAMBDA_BALANCE_VALUES, *args)
    clearers = [
        result for result in stage_one
        if faithful.clears_heldout_pareto_bar(result["heldout_metrics"])
    ]
    winner = None
    stage_two = []
    if clearers:
        # Established rule: "the point with the largest emotion margin".
        # Python's max retains the first screened value for an exact margin tie.
        winner = max(
            clearers,
            key=lambda result: (
                result["heldout_metrics"]["emotion"]["AMI"]
                - faithful.EMOTION_PARETO_BAR
            ),
        )
        winning_lambda = winner["lambda_balance"]
        log(
            f"Stage 1 selected lambda={winning_lambda:g} by largest emotion "
            "margin; starting prescribed fresh-seed stress at 7, 123, and 2024."
        )
        for seed in SEEDS:
            stage_two.extend(run_seed(seed, (winning_lambda,), *args))
    else:
        log("No Stage-1 lambda clears the held-out Pareto bar; skipping Stage 2.")
    write_report(train_h.shape[1], stage_one, winner, stage_two)
    log(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
