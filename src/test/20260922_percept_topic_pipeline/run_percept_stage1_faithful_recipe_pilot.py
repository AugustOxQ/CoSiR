"""Test the paper's Stage 1 noise and cosine schedule at native K=100/67.

Run both readings of whether latent reconstruction noise persists into DEC.
Only variants clearing the seed-42 held-out Pareto bar receive seed stress.
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
REPORT_OUT_DIR = os.environ.get("PERCEPT_OUTPUT_ROOT") or OUT_DIR
BASE_PILOT_PATH = os.path.join(OUT_DIR, "run_percept_stage1_pilot.py")
REPORT_PATH = os.path.join(REPORT_OUT_DIR, "percept_stage1_faithful_recipe_pilot_report.md")

SEED = 42
SEEDS = (7, 123, 2024)
VARIANTS = {"A": "pretrain-only noise", "B": "noise persists"}
NOISE_STD = 0.1
PRETRAIN_LEARNING_RATE = 1e-3
DEC_LEARNING_RATE = 1e-3
LR_FLOOR = 1e-5
PRETRAIN_EPOCHS = 100
DEC_COSINE_T_MAX = 200
N_INITIAL_CLUSTERS = 100
N_SURVIVING_CLUSTERS = 67
MAX_DEC_EPOCHS = 500
STABILITY_THRESHOLD = 0.001
LAMBDA_RECONSTRUCTION = 1.0
EMOTION_PARETO_BAR = 0.1236
GENRE_PARETO_BAR = 0.1954
STANDING_EMOTION_MEAN = 0.1252
STANDING_GENRE_MEAN = 0.2486


def load_module(module_name: str, path: str):
    """Load a standalone sibling script without running its main block."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("percept_stage1_base_pilot", BASE_PILOT_PATH)


def pretrain_autoencoder(encoder, decoder, inputs, device, log, seed):
    """Reconstruct clean fused embeddings from noisy 128-D latent vectors."""
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=PRETRAIN_LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PRETRAIN_EPOCHS, eta_min=LR_FLOOR
    )
    losses = []
    encoder.train()
    decoder.train()
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        permutation = torch.randperm(len(inputs))
        total_loss = 0.0
        total_nodes = 0
        for start in range(0, len(inputs), base.PRETRAIN_BATCH_SIZE):
            batch = inputs[permutation[start:start + base.PRETRAIN_BATCH_SIZE]].to(device)
            z = encoder(batch)
            z_hat = z + NOISE_STD * torch.randn_like(z)
            # Appendix E calls for MSE; the displayed Section 4.1 equation uses unsquared L2.
            reconstruction_loss = F.mse_loss(decoder(z_hat), batch)
            optimizer.zero_grad()
            reconstruction_loss.backward()
            optimizer.step()
            total_loss += reconstruction_loss.item() * len(batch)
            total_nodes += len(batch)
        mean_loss = total_loss / total_nodes
        losses.append({"epoch": epoch, "reconstruction": mean_loss, "lr": learning_rate})
        scheduler.step()
        if epoch % 10 == 0:
            log(
                f"seed={seed} pretrain epoch {epoch:03d}/{PRETRAIN_EPOCHS}: "
                f"mean reconstruction={mean_loss:.6f}, lr={learning_rate:.8g}"
            )
    return losses


def initialize_cluster_centers(encoder, inputs, device, seed):
    """Fit 100 K-means centers to clean pretrained train latents."""
    encoder.eval()
    with torch.no_grad():
        latent = encoder(inputs.to(device)).cpu().numpy()
    kmeans = KMeans(n_clusters=N_INITIAL_CLUSTERS, n_init=10, random_state=seed)
    kmeans.fit(latent)
    return nn.Parameter(
        torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(device)
    )


def cluster_size_diagnostic(q):
    """Count hard assignments across all 100 centers before pruning."""
    assignments = q.argmax(dim=1)
    sizes = torch.bincount(assignments, minlength=N_INITIAL_CLUSTERS)
    return {
        "min": int(sizes.min().item()),
        "max": int(sizes.max().item()),
        "median": float(np.median(sizes.cpu().numpy())),
        "below_one_percent": int((sizes < 0.01 * len(assignments)).sum().item()),
    }


def log_cluster_size_diagnostic(epoch, q, log, trajectory, seed, variant):
    diagnostic = {"epoch": epoch, **cluster_size_diagnostic(q)}
    trajectory.append(diagnostic)
    log(
        f"seed={seed} variant={variant} DEC cluster sizes epoch "
        f"{epoch:03d}/{MAX_DEC_EPOCHS}: min={diagnostic['min']:,}, "
        f"max={diagnostic['max']:,}, median={diagnostic['median']:.1f}, "
        f"below_1pct={diagnostic['below_one_percent']}/{N_INITIAL_CLUSTERS}"
    )


def train_dec_until_stable(encoder, decoder, centers, inputs, device, log, seed, variant):
    """Keep baseline full-batch DEC and convergence behavior with cosine LR."""
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + [centers],
        lr=DEC_LEARNING_RATE,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=DEC_COSINE_T_MAX, eta_min=LR_FLOOR
    )
    inputs = inputs.to(device)
    trajectory = []
    encoder.eval()
    with torch.no_grad():
        initial_q = base.soft_assignments(encoder(inputs), centers)
        previous_assignments = initial_q.argmax(dim=1)
        log_cluster_size_diagnostic(0, initial_q, log, trajectory, seed, variant)
    log(
        f"Starting seed={seed} variant={variant} joint DEC training "
        f"(up to {MAX_DEC_EPOCHS} full-batch epochs, lr={DEC_LEARNING_RATE:g}, "
        f"cosine T_max={DEC_COSINE_T_MAX}, eta_min={LR_FLOOR:g}, "
        f"stability threshold={STABILITY_THRESHOLD:.3f})..."
    )
    losses = []
    encoder.train()
    decoder.train()
    for epoch in range(1, MAX_DEC_EPOCHS + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        z = encoder(inputs)
        z_for_reconstruction = (
            z + NOISE_STD * torch.randn_like(z) if variant == "B" else z
        )
        reconstruction = decoder(z_for_reconstruction)
        q = base.soft_assignments(z, centers)  # Clustering always uses clean z.
        p = base.target_distribution(q)
        kl_loss = F.kl_div(q.log(), p, reduction="batchmean", log_target=False)
        reconstruction_loss = F.mse_loss(reconstruction, inputs)
        total_loss = kl_loss + LAMBDA_RECONSTRUCTION * reconstruction_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # Never step beyond T_max: CosineAnnealingLR would otherwise rise again.
        if epoch <= DEC_COSINE_T_MAX:
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
            "fraction_changed": fraction_changed,
            "lr": learning_rate,
        }
        losses.append(checkpoint)
        stopped_for_stability = fraction_changed < STABILITY_THRESHOLD
        reached_epoch_ceiling = epoch == MAX_DEC_EPOCHS
        stopping = stopped_for_stability or reached_epoch_ceiling
        if epoch % 25 == 0 or stopping:
            log(
                f"seed={seed} variant={variant} DEC epoch {epoch:03d}/{MAX_DEC_EPOCHS}: "
                f"total={checkpoint['total']:.6f}, KL={checkpoint['kl']:.6f}, "
                f"reconstruction={checkpoint['reconstruction']:.6f}, "
                f"lr={learning_rate:.8g}"
            )
            log_cluster_size_diagnostic(epoch, current_q, log, trajectory, seed, variant)
        if epoch % 10 == 0 or stopping:
            log(
                f"seed={seed} variant={variant} DEC epoch {epoch:03d}/{MAX_DEC_EPOCHS}: "
                f"fraction_changed={fraction_changed:.6f} ({num_changed:,} nodes), "
                f"lr={learning_rate:.8g}"
            )
        if stopping:
            stop_reason = "stability criterion" if stopped_for_stability else "epoch ceiling"
            log(f"Stopping seed={seed} variant={variant} DEC at epoch {epoch}: {stop_reason}.")
            return losses, trajectory, stop_reason, epoch
    raise RuntimeError("DEC training exited without a stopping condition.")


def clears_heldout_pareto_bar(metrics):
    return (
        metrics["emotion"]["AMI"] > EMOTION_PARETO_BAR
        and metrics["genre"]["AMI"] > GENRE_PARETO_BAR
    )


def evaluate_assignments(pipeline, paintings, emotions, latent, assignments):
    """Use the baseline evaluator, preserving a report when SI is undefined."""
    if 1 < len(np.unique(assignments)) < len(assignments):
        return base.evaluate_assignments(
            pipeline, paintings, emotions, latent, assignments
        )
    # sklearn requires at least two assigned clusters. A one-cluster collapse
    # still needs its AMI and size diagnostic reported rather than aborting.
    emotion_metrics = pipeline.external_metrics(assignments, emotions)
    genre_map = pipeline.load_genre_map()
    genre_indices = [index for index, painting in enumerate(paintings) if painting in genre_map]
    if not genre_indices:
        raise RuntimeError("No genre-labelled paintings overlap this split.")
    genre_metrics = pipeline.external_metrics(
        [assignments[index] for index in genre_indices],
        [genre_map[paintings[index]] for index in genre_indices],
    )
    cluster_sizes = np.bincount(assignments, minlength=N_SURVIVING_CLUSTERS)
    small_clusters = int(np.sum(cluster_sizes < 0.01 * len(assignments)))
    return {
        "emotion": emotion_metrics,
        "genre": genre_metrics,
        "genre_count": len(genre_indices),
        "cluster_sizes": cluster_sizes,
        "small_clusters": small_clusters,
        "collapsed": small_clusters > N_SURVIVING_CLUSTERS / 2,
        "silhouette": float("nan"),
    }


def run_seed(seed, variants, train_h, heldout_h, train_inputs, device, log,
             pipeline, paintings, train_emotions, heldout_pipeline,
             heldout_paintings, heldout_emotions):
    """Fresh seed, autoencoder and pretrain; isolate each requested DEC variant."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pretrained_encoder, pretrained_decoder = base.build_autoencoder(train_h.shape[1])
    pretrained_encoder.to(device)
    pretrained_decoder.to(device)
    pretrain_losses = pretrain_autoencoder(
        pretrained_encoder, pretrained_decoder, train_inputs, device, log, seed
    )
    encoder_state = copy.deepcopy(pretrained_encoder.state_dict())
    decoder_state = copy.deepcopy(pretrained_decoder.state_dict())
    del pretrained_encoder, pretrained_decoder

    results = []
    for variant in variants:
        log(f"Starting seed={seed} variant={variant} ({VARIANTS[variant]}) from pretrained state.")
        encoder, decoder = base.build_autoencoder(train_h.shape[1])
        encoder.load_state_dict(encoder_state)
        decoder.load_state_dict(decoder_state)
        encoder.to(device)
        decoder.to(device)
        centers = initialize_cluster_centers(encoder, train_inputs, device, seed)
        dec_losses, trajectory, stop_reason, stop_epoch = train_dec_until_stable(
            encoder, decoder, centers, train_inputs, device, log, seed, variant
        )
        surviving_centers, surviving_indices = base.prune_centers(centers)
        log(
            f"seed={seed} variant={variant}: stopped at epoch {stop_epoch} via "
            f"{stop_reason}; pruned {N_INITIAL_CLUSTERS - N_SURVIVING_CLUSTERS} "
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
        # The base evaluator computes silhouette on the same 128-D Z and
        # surviving-center assignments used for both external AMI metrics.
        train_metrics = evaluate_assignments(
            pipeline, paintings, train_emotions,
            train_latent.cpu().numpy(), train_assignments,
        )
        heldout_metrics = evaluate_assignments(
            heldout_pipeline, heldout_paintings, heldout_emotions,
            heldout_latent.cpu().numpy(), heldout_assignments,
        )
        results.append({
            "seed": seed,
            "variant": variant,
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


def format_pretrain_losses(losses):
    return "; ".join(
        f"epoch {point['epoch']}: recon={point['reconstruction']:.6f}, "
        f"lr={point['lr']:.8g}"
        for point in losses if point["epoch"] % 10 == 0
    )


def format_dec_losses(losses):
    by_epoch = {point["epoch"]: point for point in losses}
    checkpoints = sorted({
        1, len(losses),
        *range(10, len(losses) + 1, 10),
        *range(25, len(losses) + 1, 25),
    })
    return "; ".join(
        f"epoch {epoch}: total={by_epoch[epoch]['total']:.6f}, "
        f"KL={by_epoch[epoch]['kl']:.6f}, "
        f"recon={by_epoch[epoch]['reconstruction']:.6f}, "
        f"fraction_changed={by_epoch[epoch]['fraction_changed']:.6f}, "
        f"lr={by_epoch[epoch]['lr']:.8g}"
        for epoch in checkpoints
    )


def format_trajectory(trajectory):
    return "; ".join(
        f"epoch {point['epoch']}: min={point['min']:,}, max={point['max']:,}, "
        f"median={point['median']:.1f}, "
        f"below_1pct={point['below_one_percent']}/{N_INITIAL_CLUSTERS}"
        for point in trajectory
    )


def format_summary(values):
    return f"mean={np.mean(values):.4f}; min={np.min(values):.4f}; max={np.max(values):.4f}"


def append_results_table(lines, results, include_collapse):
    header = (
        "| variant | seed | split | emotion AMI | genre AMI | silhouette (128-D Z) | "
        "verdict | held-out Pareto bar | surviving min | surviving max | "
        "surviving median | surviving below 1% | all-100 min | all-100 max | "
        "all-100 median | all-100 below 1% |\n"
    )
    lines.append(header)
    lines.append("|---|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for result in results:
        final_all_centers = result["trajectory"][-1]
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
                "clears" if clears_heldout_pareto_bar(metrics) else "does not clear"
            )
            # The all-100 diagnostic is on train, before pruning, for both rows.
            all_center_text = (
                f"{final_all_centers['min']:,} | {final_all_centers['max']:,} | "
                f"{final_all_centers['median']:.1f} | "
                f"{final_all_centers['below_one_percent']}/{N_INITIAL_CLUSTERS}"
            ) if include_collapse else "n/a | n/a | n/a | n/a"
            lines.append(
                f"| {result['variant']} ({VARIANTS[result['variant']]}) | "
                f"{result['seed']} | {split_name} | "
                f"{metrics['emotion']['AMI']:.4f} | {metrics['genre']['AMI']:.4f} | "
                f"{silhouette} | {base.verdict(metrics)} | {clearance} | "
                f"{int(sizes.min()):,} | {int(sizes.max()):,} | "
                f"{float(np.median(sizes)):.1f} | "
                f"{metrics['small_clusters']}/{N_SURVIVING_CLUSTERS} | "
                f"{all_center_text} |\n"
            )


def append_trajectories(lines, results):
    for result in results:
        lines.extend([
            f"### Variant {result['variant']} ({VARIANTS[result['variant']]}), "
            f"seed {result['seed']}\n\n",
            f"- Pretraining reconstruction and LR: "
            f"{format_pretrain_losses(result['pretrain_losses'])}.\n",
            f"- DEC stopped via **{result['stop_reason']}** at epoch "
            f"**{result['stop_epoch']}** (threshold "
            f"`fraction_changed < {STABILITY_THRESHOLD:.3f}`, ceiling "
            f"{MAX_DEC_EPOCHS}).\n",
            f"- Joint DEC losses and LR: {format_dec_losses(result['dec_losses'])}.\n",
            f"- All-100-center diagnostics (epoch 0 and every 25 epochs or stop): "
            f"{format_trajectory(result['trajectory'])}.\n\n",
        ])


def write_report(input_dim, phase_one, phase_two):
    """Write the complete conditional report after GPU execution."""
    lines = [
        "# ArtELingo PercepT Stage 1 faithful recipe pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Controlled setup\n\n",
        f"Train-only input is the base pilot's {input_dim}-dimensional fused vector: "
        "independently normalized CLIP content and GoEmotions-RoBERTa affect "
        "embeddings, with repeated content to preserve a 2:1 norm budget. "
        "The base 128-D autoencoder, Student's-t assignments, self-sharpened "
        "target, and 67 highest-norm of 100-center pruning are unchanged. "
        "RoBERTa and concatenation remain the documented adaptations to the "
        "paper's ModernBERT-family encoder and elementwise fusion.\n\n",
        "Both variants reconstruct clean h from a latent z corrupted with "
        "Gaussian standard deviation 0.1 during pretraining. Variant A uses "
        "clean z for joint reconstruction; variant B continues the same "
        "corruption for joint reconstruction only. KL always uses clean z. "
        "Appendix E calls the reconstruction objective MSE, whereas the "
        "displayed Section 4.1 equation uses an unsquared L2 norm; this "
        "pilot follows Appendix E's MSE convention. No balance term is used.\n\n",
        f"Both phases use Adam at initial LR 1e-3 with cosine annealing to "
        f"{LR_FLOOR:g}: pretraining T_max={PRETRAIN_EPOCHS} for 100 epochs, "
        f"joint DEC T_max={DEC_COSINE_T_MAX}, held at the floor after epoch "
        f"{DEC_COSINE_T_MAX}. Joint DEC retains the project's "
        f"`fraction_changed < {STABILITY_THRESHOLD:.3f}` stop rule and "
        f"{MAX_DEC_EPOCHS}-epoch ceiling rather than a fixed 200 epochs. "
        f"Reconstruction weight is {LAMBDA_RECONSTRUCTION:g}.\n\n",
        "## Predeclared success criterion\n\n",
        "**Held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > "
        "0.1954, both simultaneously.** A Real success also requires "
        "non-collapse under the established surviving-center rule.\n\n",
        "Silhouette is computed on the final 128-D latent Z using the same "
        "surviving-center hard assignments as AMI, separately for train and "
        "held-out. The paper reports 0.97 for its full method; this pilot "
        "reports its own measured values without treating that reference "
        "number as the held-out decision rule. If all samples share one "
        "cluster, silhouette is undefined and reported as n/a.\n\n",
        "## Phase 1: seed 42, both variants\n\n",
        "### Training and cluster-size trajectories\n\n",
    ]
    append_trajectories(lines, phase_one)
    lines.append("### Final results\n\n")
    append_results_table(lines, phase_one, include_collapse=True)
    lines.append(
        "\nSurviving-center collapse numbers are split-specific. The all-100 "
        "numbers repeat the final train checkpoint before pruning.\n\n"
    )
    if phase_two:
        lines.extend([
            "## Phase 2: conditional seed stress\n\n",
            "Only Phase-1 variants clearing both held-out AMI bars received "
            "fresh runs at seeds 7, 123, and 2024. Each seed uses a newly "
            "initialized autoencoder, fresh 100-epoch pretraining, and "
            "K-means with `random_state=seed`.\n\n",
            "### Four-seed results\n\n",
        ])
        all_results = phase_one + phase_two
        stressed_variants = [
            result["variant"] for result in phase_one
            if clears_heldout_pareto_bar(result["heldout_metrics"])
        ]
        stressed_results = [
            result for result in all_results if result["variant"] in stressed_variants
        ]
        append_results_table(lines, stressed_results, include_collapse=True)
        lines.append("\n### Additional-seed trajectories\n\n")
        append_trajectories(lines, phase_two)
        lines.append("### Held-out summary statistics and standing-result comparison\n\n")
        for variant in stressed_variants:
            results = [result for result in stressed_results if result["variant"] == variant]
            heldout = [result["heldout_metrics"] for result in results]
            emotion = [metrics["emotion"]["AMI"] for metrics in heldout]
            genre = [metrics["genre"]["AMI"] for metrics in heldout]
            emotion_clearers = sum(value > EMOTION_PARETO_BAR for value in emotion)
            genre_clearers = sum(value > GENRE_PARETO_BAR for value in genre)
            both_clearers = sum(clears_heldout_pareto_bar(metrics) for metrics in heldout)
            lines.extend([
                f"- Variant {variant} emotion AMI across four seeds: "
                f"{format_summary(emotion)}; individual bar clear in "
                f"{emotion_clearers}/4 seeds.\n",
                f"- Variant {variant} genre AMI across four seeds: "
                f"{format_summary(genre)}; individual bar clear in "
                f"{genre_clearers}/4 seeds. Both bars clear together in "
                f"{both_clearers}/4 seeds.\n",
            ])
            matches_standing = (
                np.mean(emotion) >= STANDING_EMOTION_MEAN
                and np.mean(genre) >= STANDING_GENRE_MEAN
                and both_clearers >= 4
            )
            lines.append(
                f"- Variant {variant} {'matches or beats' if matches_standing else 'does not match or beat'} "
                "the standing K=60/40, `LAMBDA_BALANCE=1000` result on "
                "both held-out mean AMIs and the both-bar seed count. "
                f"Standing: emotion mean {STANDING_EMOTION_MEAN:.4f}, genre "
                f"mean {STANDING_GENRE_MEAN:.4f}, 4/4 seeds clearing both. "
                "This pilot uses K=100/67 and no balance term.\n\n"
            )
    else:
        lines.extend([
            "## Phase 2\n\n",
            "Skipped: neither variant cleared the held-out Pareto bar in "
            "Phase 1. Both are plain misses at seed 42; no other noise "
            "strength or LR schedule was tested.\n\n",
        ])
    lines.append("## Decision\n\n")
    for result in phase_one:
        variant = result["variant"]
        metrics = result["heldout_metrics"]
        phase_one_verdict = base.verdict(metrics)
        if not clears_heldout_pareto_bar(metrics):
            lines.append(
                f"- **Variant {variant}: {phase_one_verdict}.** Phase 1 missed the "
                "held-out Pareto bar; Phase 2 was not run for this variant.\n"
            )
            continue
        results = [run for run in phase_one + phase_two if run["variant"] == variant]
        clearers = sum(clears_heldout_pareto_bar(run["heldout_metrics"]) for run in results)
        noncollapsed_clearers = sum(
            base.verdict(run["heldout_metrics"]) == "Real success" for run in results
        )
        if noncollapsed_clearers == 4:
            lines.append(
                f"- **Variant {variant}: Real success, robust result.** "
                "All 4/4 seeds are non-collapsed and clear both held-out bars.\n"
            )
        else:
            lines.append(
                f"- **Variant {variant}: seed-dependent result.** "
                f"{clearers}/4 seeds clear both held-out bars and "
                f"{noncollapsed_clearers}/4 also pass the non-collapse rule.\n"
            )
    os.makedirs(REPORT_OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main():
    """Extract once, run both seed-42 variants, then gate three fresh seeds."""
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
    heldout_emotions = [heldout_pipeline.majority(counts) for counts in heldout_emotion_counts]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {device} for deterministic embedding extraction and faithful DEC training.")
    affect_train = base.extract_affect_embedding_nodes(
        pipeline.TRAIN_JSON, paintings, device, log
    )
    affect_heldout = base.extract_affect_embedding_nodes(
        base.HELDOUT_JSON, heldout_paintings, device, log
    )
    train_h = base.fused_embeddings(img_nodes, txt_nodes, affect_train, cca_audit, affect_pilot)
    heldout_h = base.fused_embeddings(
        heldout_img_nodes, heldout_txt_nodes, affect_heldout, cca_audit, affect_pilot
    )
    if train_h.shape[1] != heldout_h.shape[1]:
        raise RuntimeError(
            f"Train/held-out fused dimensions differ: {train_h.shape[1]} "
            f"vs {heldout_h.shape[1]}."
        )
    train_inputs = torch.from_numpy(train_h)
    args = (
        train_h, heldout_h, train_inputs, device, log, pipeline, paintings,
        train_emotions, heldout_pipeline, heldout_paintings, heldout_emotions,
    )
    phase_one = run_seed(SEED, tuple(VARIANTS), *args)
    qualifying_variants = tuple(
        result["variant"] for result in phase_one
        if clears_heldout_pareto_bar(result["heldout_metrics"])
    )
    phase_two = []
    if qualifying_variants:
        log(
            f"Phase 1 qualifying variants {qualifying_variants}; starting prescribed "
            "fresh-seed stress at 7, 123, and 2024."
        )
        for seed in SEEDS:
            phase_two.extend(run_seed(seed, qualifying_variants, *args))
    else:
        log("Neither Phase 1 variant clears the held-out Pareto bar; skipping Phase 2.")
    write_report(train_h.shape[1], phase_one, phase_two)
    log(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
