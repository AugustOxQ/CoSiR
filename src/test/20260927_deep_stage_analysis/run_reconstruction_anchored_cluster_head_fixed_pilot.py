"""Gap 3: re-run the reconstruction-anchored clustering-head pilot (the
6th/7th and most-recent DEC-hybrid attempt, §4/REFERENCES in the original
report) with BOTH bugs fixed that an independent review found in PercepT's
own code and that this pilot copied "verbatim":

1. `prune_centers` kept the highest-norm centers; the paper's rule is to
   keep the LOWEST-norm (best-populated) ones. Fixed here as
   `prune_centers_fixed` (ascending sort).
2. `F.mse_loss(reconstruction, inputs)` used the default 'mean' reduction,
   which divides by the 32-D feature dimension on top of the batch mean --
   the same class of bug as PercepT's 2,816x-too-weak reconstruction loss,
   just smaller here (32x). Fixed here as `summed_mse_loss` (sum over
   features, mean over batch), matching the paper's L_R = ||h - h_hat||^2.

This is the only one of the ~6 DEC-hybrid attempts where a reconstruction
loss coexists with the prune_centers bug in a form cheap to re-run: the
buddy trunk is frozen (loads attention_h1_embedding_snapshot.npz directly),
so no InfoNCE retraining is needed. Everything else (architecture, LAMBDA_DEC
grid, warm-up schedule, seeds, silhouette/AMI evaluation) is reused from the
original module unchanged, via import.
"""

import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
BUDDY_DIR = HERE.parents[1] / "test/20260923_artelingo_buddy_analysis"
ORIGINAL_PATH = BUDDY_DIR / "run_reconstruction_anchored_cluster_head_pilot.py"
REPORT_PATH = HERE / "reconstruction_anchored_cluster_head_fixed_pilot_report.md"
ORIGINAL_FUSED_SILHOUETTE_MEAN = -0.0793
ORIGINAL_EMOTION_AMI_MEAN = 0.1122
ORIGINAL_GENRE_AMI_MEAN = 0.1099


def load_module(module_name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


orig = load_module("reconstruction_anchored_original_for_fix", ORIGINAL_PATH)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def prune_centers_fixed(centers: torch.Tensor):
    """Paper-faithful direction: keep the N LOWEST-norm centers, not highest."""
    norms = torch.linalg.vector_norm(centers.detach(), dim=1)
    surviving = torch.argsort(norms, descending=False)[: orig.N_SURVIVING_CLUSTERS]
    return centers.detach()[surviving], surviving.cpu().numpy()


def summed_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Paper-matching L_R = ||h - h_hat||^2: sum over feature dim, mean over batch."""
    return F.mse_loss(pred, target, reduction="none").sum(dim=1).mean()


def pretrain_fixed(encoder, decoder, inputs, device, seed, log_fn):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=orig.PRETRAIN_LEARNING_RATE
    )
    losses = []
    encoder.train()
    decoder.train()
    for epoch in range(1, orig.PRETRAIN_EPOCHS + 1):
        permutation = torch.randperm(len(inputs))
        total_loss, total_nodes = 0.0, 0
        for start in range(0, len(inputs), orig.PRETRAIN_BATCH_SIZE):
            batch = inputs[permutation[start:start + orig.PRETRAIN_BATCH_SIZE]].to(device)
            loss = summed_mse_loss(decoder(encoder(batch)), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
            total_nodes += len(batch)
        mean_loss = total_loss / total_nodes
        losses.append(mean_loss)
        if epoch % 20 == 0 or epoch == orig.PRETRAIN_EPOCHS:
            log_fn(f"seed={seed} pretrain epoch {epoch:03d}/{orig.PRETRAIN_EPOCHS}: reconstruction={mean_loss:.6f}")
    return losses


def run_seed_fixed(seed, lambda_dec, data, device, log_fn):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    encoder, decoder = orig.build_autoencoder()
    encoder.to(device)
    decoder.to(device)
    train_inputs = data["train_inputs"]
    pretrain_losses = pretrain_fixed(encoder, decoder, train_inputs, device, seed, log_fn)

    encoder.eval()
    with torch.no_grad():
        clean_train_latent = encoder(train_inputs.to(device))
    kmeans = KMeans(n_clusters=orig.N_INITIAL_CLUSTERS, n_init=10, random_state=seed)
    kmeans.fit(clean_train_latent.cpu().numpy())
    centers = nn.Parameter(torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(device))
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + [centers],
        lr=orig.PRETRAIN_LEARNING_RATE,
    )
    log_fn(f"Starting joint DEC phase (fixed): seed={seed}, LAMBDA_DEC={lambda_dec:g}.")
    train_inputs_device = train_inputs.to(device)
    encoder.train()
    decoder.train()
    for epoch in range(1, orig.MAX_EPOCHS + 1):
        dec_weight = lambda_dec * min(1.0, epoch / orig.DEC_WARMUP_EPOCHS)
        z = encoder(train_inputs_device)
        reconstruction = decoder(z)
        q = orig.soft_assignments(z, centers)
        p = orig.target_distribution(q)
        kl_loss = F.kl_div(q.log(), p, reduction="batchmean")
        reconstruction_loss = summed_mse_loss(reconstruction, train_inputs_device)
        total_loss = dec_weight * kl_loss + orig.LAMBDA_RECONSTRUCTION * reconstruction_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if epoch % 25 == 0 or epoch == orig.MAX_EPOCHS:
            with torch.no_grad():
                sizes = orig.cluster_size_diagnostic(q.detach())
            log_fn(
                f"seed={seed} LAMBDA_DEC={lambda_dec:g} epoch={epoch} "
                f"KL={kl_loss.item():.6f} recon={reconstruction_loss.item():.6f} "
                f"dec_weight={dec_weight:.4f} below_1%={sizes['below_one_percent']}/{orig.N_INITIAL_CLUSTERS}"
            )
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        final_train_latent = encoder(train_inputs_device)
        surviving_centers, surviving_indices = prune_centers_fixed(centers)
        train_assignments = orig.soft_assignments(final_train_latent, surviving_centers).argmax(dim=1).cpu().numpy()
        heldout_latent = encoder(data["heldout_inputs"].to(device))
        heldout_assignments = orig.soft_assignments(heldout_latent, surviving_centers).argmax(dim=1).cpu().numpy()
    if len(surviving_indices) != orig.N_SURVIVING_CLUSTERS:
        raise RuntimeError("Pruning did not return the expected number of centers.")
    train_cluster_latent = final_train_latent.cpu().numpy()
    heldout_cluster_latent = heldout_latent.cpu().numpy()
    train_metrics = orig.split_metrics(
        data["train_fused"], train_cluster_latent, train_assignments,
        data["train_emotion"], data["train_genre_indices"], data["train_genre"],
    )
    heldout_metrics = orig.split_metrics(
        data["heldout_fused"], heldout_cluster_latent, heldout_assignments,
        data["heldout_emotion"], data["heldout_genre_indices"], data["heldout_genre"],
    )
    result = {
        "seed": seed, "lambda_dec": lambda_dec,
        "train": train_metrics, "heldout": heldout_metrics,
        "train_surviving": orig.surviving_cluster_diagnostic(train_assignments),
        "heldout_surviving": orig.surviving_cluster_diagnostic(heldout_assignments),
        "pretrain_final": pretrain_losses[-1],
    }
    log_fn(
        f"seed={seed} LAMBDA_DEC={lambda_dec:g} final held-out emotion "
        f"AMI={heldout_metrics['emotion_ami']:.4f} genre AMI={heldout_metrics['genre_ami']:.4f} "
        f"fused silhouette={heldout_metrics['fused_silhouette']:.4f} "
        f"cluster silhouette={heldout_metrics['cluster_silhouette']:.4f}"
    )
    return result


def write_report(screen, winner, validated_winner, stress) -> None:
    four_seeds = [winner, *stress]
    clear_count = sum(orig.clears_pareto(r) for r in four_seeds)
    lines = [
        "# Gap 3 — reconstruction-anchored cluster head, BOTH known bugs fixed\n\n",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. Companion/fixed re-run of "
        "[`reconstruction_anchored_cluster_head_pilot_report.md`]"
        "(../../test/20260923_artelingo_buddy_analysis/reconstruction_anchored_cluster_head_pilot_report.md). "
        "Fixes: `prune_centers` now keeps the lowest-norm (paper-faithful) centers; "
        "the reconstruction loss now sums over the feature dimension before "
        "averaging over the batch, matching the paper's L_R = ||h - h_hat||^2 "
        "instead of PyTorch's default mean-over-all-elements reduction. "
        "Original (buggy) file and report are left unmodified for provenance.\n\n",
        "## Single-seed screen (seed 42)\n\n",
        "| LAMBDA_DEC | split | emotion AMI | genre AMI | cluster silhouette | fused silhouette | clusters | held-out Pareto bar |\n",
        "|---:|---|---:|---:|---:|---:|---:|---|\n",
    ]
    for result in screen:
        for split, key in (("train", "train"), ("held-out", "heldout")):
            metrics = result[key]
            verdict = ("clears" if orig.clears_pareto(result) else "does not clear") if split == "held-out" else "n/a"
            lines.append(
                f"| {result['lambda_dec']:g} | {split} | {metrics['emotion_ami']:.4f} | "
                f"{metrics['genre_ami']:.4f} | {orig.format_metric(metrics['cluster_silhouette'])} | "
                f"{orig.format_metric(metrics['fused_silhouette'])} | {metrics['communities']} | {verdict} |\n"
            )
    lines.append("\n## Winner selection\n\n")
    selected = winner["heldout"]
    if validated_winner:
        clear_values = ", ".join(f"{r['lambda_dec']:g}" for r in screen if orig.clears_pareto(r))
        lines.append(
            f"Pareto-bar clearers: {clear_values}. Selected LAMBDA_DEC={winner['lambda_dec']:g}, "
            f"highest held-out fused silhouette among clearers ({orig.format_metric(selected['fused_silhouette'])}).\n\n"
        )
    else:
        lines.append(
            f"No screened value clears the held-out Pareto bar. Selected LAMBDA_DEC="
            f"{winner['lambda_dec']:g} for highest held-out fused silhouette among all "
            f"three ({orig.format_metric(selected['fused_silhouette'])}): best available, "
            "does not clear the Pareto bar.\n\n"
        )
    lines.extend([
        "## Four-seed stress of the selected LAMBDA_DEC (fixed)\n\n",
        "| seed | held-out emotion AMI | held-out genre AMI | held-out cluster silhouette | held-out fused silhouette | held-out clusters | Pareto bar |\n",
        "|---:|---:|---:|---:|---:|---:|---|\n",
    ])
    for result in four_seeds:
        heldout = result["heldout"]
        lines.append(
            f"| {result['seed']} | {heldout['emotion_ami']:.4f} | {heldout['genre_ami']:.4f} | "
            f"{orig.format_metric(heldout['cluster_silhouette'])} | {orig.format_metric(heldout['fused_silhouette'])} | "
            f"{heldout['communities']} | {'clears' if orig.clears_pareto(result) else 'does not clear'} |\n"
        )
    fused_values = [r["heldout"]["fused_silhouette"] for r in four_seeds]
    finite_fused = np.asarray(fused_values, dtype=np.float64)
    finite_fused = finite_fused[np.isfinite(finite_fused)]
    mean_fused = float(np.mean(finite_fused)) if len(finite_fused) else float("nan")
    mean_emotion = float(np.mean([r["heldout"]["emotion_ami"] for r in four_seeds]))
    mean_genre = float(np.mean([r["heldout"]["genre_ami"] for r in four_seeds]))
    lines.append(f"\n- Both held-out Pareto bars clear in {clear_count}/4 seeds.\n\n")
    lines.append("## Comparison against the original (buggy) run\n\n")
    lines.append("| | held-out emotion AMI (4-seed mean) | held-out genre AMI (4-seed mean) | held-out fused silhouette (4-seed mean) |\n")
    lines.append("|---|---:|---:|---:|\n")
    lines.append(f"| Original (buggy prune_centers + loss scale) | {ORIGINAL_EMOTION_AMI_MEAN:.4f} | {ORIGINAL_GENRE_AMI_MEAN:.4f} | {ORIGINAL_FUSED_SILHOUETTE_MEAN:.4f} |\n")
    lines.append(f"| **Fixed (this run)** | **{mean_emotion:.4f}** | **{mean_genre:.4f}** | **{orig.format_metric(mean_fused)}** |\n\n")
    lines.append("## Verdict\n\n")
    changed_materially = (
        clear_count >= 3
        or (np.isfinite(mean_fused) and mean_fused > 0 and mean_fused - ORIGINAL_FUSED_SILHOUETTE_MEAN > 0.05)
    )
    if changed_materially:
        lines.append(
            "**The two bug fixes materially change this attempt's conclusion.** "
            f"Fixed fused silhouette {orig.format_metric(mean_fused)} vs. original "
            f"{ORIGINAL_FUSED_SILHOUETTE_MEAN:.4f}, Pareto bar clears in {clear_count}/4 "
            "seeds. This DEC-hybrid direction needs re-opening -- the §4 negative "
            "conclusion for this attempt no longer holds and the other DEC-hybrid "
            "attempts (which share the same prune_centers bug) should now be "
            "considered untested rather than settled negatives.\n"
        )
    else:
        lines.append(
            f"**The two bug fixes do not materially change this attempt's "
            f"conclusion.** Fixed fused silhouette {orig.format_metric(mean_fused)} "
            f"vs. original {ORIGINAL_FUSED_SILHOUETTE_MEAN:.4f} "
            f"({mean_fused - ORIGINAL_FUSED_SILHOUETTE_MEAN:+.4f} if both finite), "
            f"Pareto bar still clears in only {clear_count}/4 seeds. This is "
            "consistent with the §4 diagnosis that the failure mode here is the "
            "embedding geometry / DEC self-sharpening dynamics, not the specific "
            "center-selection or loss-scale bugs -- those bugs affect *which* "
            "centers survive pruning and how strongly reconstruction anchors the "
            "latent, but not the underlying separability the training dynamics "
            "produce. The §4/§6b negative conclusion for the DEC-hybrid direction "
            "is not overturned by this fix.\n"
        )
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    torch.manual_seed(orig.SEED)
    np.random.seed(orig.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(orig.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using {device} for the fixed reconstruction-anchored cluster head re-run.")

    with np.load(orig.BUDDY_SNAPSHOT, allow_pickle=True) as snapshot:
        train_fused = np.asarray(snapshot["train_embedding_post"], dtype=np.float32)
        heldout_fused = np.asarray(snapshot["heldout_embedding_post"], dtype=np.float32)
        train_emotion = np.asarray(snapshot["train_emotion"])
        heldout_emotion = np.asarray(snapshot["heldout_emotion"])
        train_genre_all = np.asarray(snapshot["train_genre"])
        heldout_genre_all = np.asarray(snapshot["heldout_genre"])
    train_genre_indices = np.flatnonzero(train_genre_all != "")
    heldout_genre_indices = np.flatnonzero(heldout_genre_all != "")
    data = {
        "train_inputs": torch.from_numpy(train_fused),
        "heldout_inputs": torch.from_numpy(heldout_fused),
        "train_fused": train_fused, "heldout_fused": heldout_fused,
        "train_emotion": train_emotion, "heldout_emotion": heldout_emotion,
        "train_genre_indices": train_genre_indices, "heldout_genre_indices": heldout_genre_indices,
        "train_genre": train_genre_all[train_genre_indices],
        "heldout_genre": heldout_genre_all[heldout_genre_indices],
    }
    log(f"Loaded frozen buddy embeddings: train={train_fused.shape}, held-out={heldout_fused.shape}.")

    screen = [run_seed_fixed(orig.SEED, value, data, device, log) for value in orig.LAMBDA_DEC_VALUES]
    winner, validated_winner = orig.choose_winner(screen)
    log(
        f"Selected LAMBDA_DEC={winner['lambda_dec']:g}: "
        f"{'Pareto-clear winner' if validated_winner else 'best available, does not clear the Pareto bar'}."
    )
    stress = [run_seed_fixed(seed, winner["lambda_dec"], data, device, log) for seed in orig.STRESS_SEEDS]
    write_report(screen, winner, validated_winner, stress)
    log(f"Wrote {REPORT_PATH}.")


if __name__ == "__main__":
    main()
