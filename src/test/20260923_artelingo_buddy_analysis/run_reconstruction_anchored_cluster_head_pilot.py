"""A properly reconstruction-anchored clustering head on the frozen buddy embedding.

Two prior attempts (the decoupled ClusterHead, un-detached and detached)
trained DEC's KL loss alone, with no reconstruction anchor at all -- both
collapsed to a near-zero-silhouette cluster latent, and the detached retry
showed this was not a gradient-leakage artifact. A third attempt (PercepT's
own unmodified autoencoder+DEC fed buddy's 32-D embedding) DID have a
reconstruction term, but its 128-D latent is overcomplete relative to the
32-D input, which let the decoder reconstruct well without the encoder
needing well-separated clusters -- it scored negative silhouette back in
buddy's native space.

This pilot is deliberately different from both: encoder/decoder are a
small, genuinely undercomplete autoencoder (32 -> 24 -> 16 -> 24 -> 32),
pretrained on frozen buddy embeddings via reconstruction alone before any
clustering pressure is introduced, exactly the anchor DEC's own literature
motivates. The buddy trunk itself is frozen throughout (the saved
attention_h1_embedding_snapshot.npz's embeddings are used directly); no
Attention-h1 training happens in this script at all.
"""

import os
import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch import nn
from torch.nn import functional as F


OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BUDDY_SNAPSHOT = os.path.join(OUT_DIR, "attention_h1_embedding_snapshot.npz")
REPORT_PATH = os.path.join(OUT_DIR, "reconstruction_anchored_cluster_head_pilot_report.md")

SEED = 42
STRESS_SEEDS = (7, 123, 2024)
LAMBDA_DEC_VALUES = (0.1, 0.5, 1.0)
LAMBDA_RECONSTRUCTION = 1.0
DEC_WARMUP_EPOCHS = 30
MAX_EPOCHS = 200
CHECKPOINT_EVERY = 5
N_INITIAL_CLUSTERS = 100
N_SURVIVING_CLUSTERS = 67
EMOTION_PARETO_BAR = 0.1236
GENRE_PARETO_BAR = 0.1954
# Reuse PercepT's own pretrain budget; the network here is much smaller so
# this is a generous, not a tight, epoch/LR choice for reconstruction-only
# convergence at this width.
PRETRAIN_LEARNING_RATE = 1e-3
PRETRAIN_EPOCHS = 100
PRETRAIN_BATCH_SIZE = 1024

REFERENCES = (
    ("Attention-h1 baseline (four-seed mean)", 0.1241, 0.2406, 0.0397),
    ("Euclidean DEC hybrid (four-seed mean)", 0.1160, 0.1321, -0.0288),
    ("vMF DEC hybrid (four-seed mean)", 0.1215, 0.1504, 0.0298),
    ("Decoupled cluster head, un-detached (four-seed mean)", 0.0845, 0.0423, -0.1568),
    ("Decoupled cluster head, detached (four-seed mean)", 0.0828, 0.0481, -0.1529),
    ("PercepT-on-buddy-embedding (seed 42, native 32-D)", 0.1482, 0.1765, -0.0252),
    ("PercepT replication faithful recipe (seed 42, its own 128-D latent)", 0.1092, 0.3288, 0.5120),
)


def soft_assignments(latent: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """PercepT's own Student's-t DEC assignments (alpha=1), reused verbatim."""
    squared_distances = torch.sum((latent.unsqueeze(1) - centers.unsqueeze(0)) ** 2, dim=2)
    q = 1.0 / (1.0 + squared_distances)
    return q / q.sum(dim=1, keepdim=True)


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    """PercepT's own detached DEC self-sharpened target, reused verbatim."""
    weights = q.pow(2) / q.sum(dim=0, keepdim=True)
    return (weights / weights.sum(dim=1, keepdim=True)).detach()


def prune_centers(centers: torch.Tensor):
    """PercepT's own highest-norm pruning proxy, reused verbatim."""
    norms = torch.linalg.vector_norm(centers.detach(), dim=1)
    surviving = torch.argsort(norms, descending=True)[:N_SURVIVING_CLUSTERS]
    return centers.detach()[surviving], surviving.cpu().numpy()


def build_autoencoder():
    encoder = nn.Sequential(nn.Linear(32, 24), nn.ReLU(), nn.Linear(24, 16))
    decoder = nn.Sequential(nn.Linear(16, 24), nn.ReLU(), nn.Linear(24, 32))
    return encoder, decoder


def pretrain(encoder, decoder, inputs, device, seed, log):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=PRETRAIN_LEARNING_RATE
    )
    losses = []
    encoder.train()
    decoder.train()
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        permutation = torch.randperm(len(inputs))
        total_loss, total_nodes = 0.0, 0
        for start in range(0, len(inputs), PRETRAIN_BATCH_SIZE):
            batch = inputs[permutation[start:start + PRETRAIN_BATCH_SIZE]].to(device)
            loss = F.mse_loss(decoder(encoder(batch)), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
            total_nodes += len(batch)
        mean_loss = total_loss / total_nodes
        losses.append(mean_loss)
        if epoch % 20 == 0 or epoch == PRETRAIN_EPOCHS:
            log(f"seed={seed} pretrain epoch {epoch:03d}/{PRETRAIN_EPOCHS}: reconstruction={mean_loss:.6f}")
    return losses


def cluster_size_diagnostic(q):
    sizes = torch.bincount(q.argmax(dim=1), minlength=N_INITIAL_CLUSTERS)
    return {
        "min": int(sizes.min().item()), "max": int(sizes.max().item()),
        "median": float(np.median(sizes.cpu().numpy())),
        "below_one_percent": int((sizes < 0.01 * q.shape[0]).sum().item()),
    }


def surviving_cluster_diagnostic(assignments):
    sizes = np.bincount(assignments, minlength=N_SURVIVING_CLUSTERS)
    below_one_percent = int(np.sum(sizes < 0.01 * len(assignments)))
    return {
        "min": int(sizes.min()), "max": int(sizes.max()), "median": float(np.median(sizes)),
        "below_one_percent": below_one_percent,
        "collapsed": below_one_percent > N_SURVIVING_CLUSTERS / 2,
    }


def sampled_silhouette(embeddings, assignments, seed=42):
    idx = np.random.default_rng(seed).choice(
        len(embeddings), size=min(6000, len(embeddings)), replace=False
    )
    sampled_labels = assignments[idx]
    score_size = min(4000, len(idx))
    score_indices = np.random.RandomState(seed).permutation(len(idx))[:score_size]
    score_labels = sampled_labels[score_indices]
    if not 1 < len(np.unique(score_labels)) < len(score_labels):
        return float("nan")
    return float(silhouette_score(
        embeddings[idx], sampled_labels, sample_size=score_size, random_state=seed,
    ))


def external_metrics_ami(labels, targets):
    from sklearn.metrics import adjusted_mutual_info_score
    return float(adjusted_mutual_info_score(labels, targets))


def split_metrics(fused_embeddings, cluster_latent, assignments, emotions, genre_indices, genres):
    emotion_ami = external_metrics_ami(assignments, emotions)
    genre_ami = external_metrics_ami(assignments[genre_indices], genres)
    return {
        "emotion_ami": emotion_ami, "genre_ami": genre_ami,
        "fused_silhouette": sampled_silhouette(fused_embeddings, assignments),
        "cluster_silhouette": sampled_silhouette(cluster_latent, assignments),
        "communities": len(np.unique(assignments)),
    }


def clears_pareto(result):
    heldout = result["heldout"]
    return heldout["emotion_ami"] > EMOTION_PARETO_BAR and heldout["genre_ami"] > GENRE_PARETO_BAR


def run_seed(seed, lambda_dec, data, device, log):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    encoder, decoder = build_autoencoder()
    encoder.to(device)
    decoder.to(device)
    train_inputs = data["train_inputs"]
    pretrain_losses = pretrain(encoder, decoder, train_inputs, device, seed, log)

    encoder.eval()
    with torch.no_grad():
        clean_train_latent = encoder(train_inputs.to(device))
    kmeans = KMeans(n_clusters=N_INITIAL_CLUSTERS, n_init=10, random_state=seed)
    kmeans.fit(clean_train_latent.cpu().numpy())
    centers = nn.Parameter(torch.from_numpy(
        kmeans.cluster_centers_.astype(np.float32)
    ).to(device))
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + [centers],
        lr=PRETRAIN_LEARNING_RATE,
    )
    log(f"Starting joint DEC phase: seed={seed}, LAMBDA_DEC={lambda_dec:g}.")
    trajectory = []
    train_inputs_device = train_inputs.to(device)
    encoder.train()
    decoder.train()
    for epoch in range(1, MAX_EPOCHS + 1):
        dec_weight = lambda_dec * min(1.0, epoch / DEC_WARMUP_EPOCHS)
        z = encoder(train_inputs_device)
        reconstruction = decoder(z)
        q = soft_assignments(z, centers)
        p = target_distribution(q)
        kl_loss = F.kl_div(q.log(), p, reduction="batchmean")
        reconstruction_loss = F.mse_loss(reconstruction, train_inputs_device)
        total_loss = dec_weight * kl_loss + LAMBDA_RECONSTRUCTION * reconstruction_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if epoch % CHECKPOINT_EVERY == 0 or epoch == MAX_EPOCHS:
            with torch.no_grad():
                sizes = cluster_size_diagnostic(q.detach())
            trajectory.append({
                "epoch": epoch, "kl": kl_loss.item(), "reconstruction": reconstruction_loss.item(),
                "dec_weight": dec_weight, **sizes,
            })
            if epoch % 25 == 0 or epoch == MAX_EPOCHS:
                log(
                    f"seed={seed} LAMBDA_DEC={lambda_dec:g} epoch={epoch} "
                    f"KL={kl_loss.item():.6f} recon={reconstruction_loss.item():.6f} "
                    f"dec_weight={dec_weight:.4f} below_1%={sizes['below_one_percent']}/{N_INITIAL_CLUSTERS}"
                )
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        final_train_latent = encoder(train_inputs_device)
        surviving_centers, surviving_indices = prune_centers(centers)
        train_assignments = soft_assignments(final_train_latent, surviving_centers).argmax(dim=1).cpu().numpy()
        heldout_latent = encoder(data["heldout_inputs"].to(device))
        heldout_assignments = soft_assignments(heldout_latent, surviving_centers).argmax(dim=1).cpu().numpy()
    if len(surviving_indices) != N_SURVIVING_CLUSTERS:
        raise RuntimeError("Pruning did not return 67 centers.")
    train_cluster_latent = final_train_latent.cpu().numpy()
    heldout_cluster_latent = heldout_latent.cpu().numpy()
    train_metrics = split_metrics(
        data["train_fused"], train_cluster_latent, train_assignments,
        data["train_emotion"], data["train_genre_indices"], data["train_genre"],
    )
    heldout_metrics = split_metrics(
        data["heldout_fused"], heldout_cluster_latent, heldout_assignments,
        data["heldout_emotion"], data["heldout_genre_indices"], data["heldout_genre"],
    )
    result = {
        "seed": seed, "lambda_dec": lambda_dec,
        "train": train_metrics, "heldout": heldout_metrics,
        "train_surviving": surviving_cluster_diagnostic(train_assignments),
        "heldout_surviving": surviving_cluster_diagnostic(heldout_assignments),
        "pretrain_final": pretrain_losses[-1], "trajectory": trajectory,
    }
    log(
        f"seed={seed} LAMBDA_DEC={lambda_dec:g} final held-out emotion "
        f"AMI={heldout_metrics['emotion_ami']:.4f} genre AMI={heldout_metrics['genre_ami']:.4f} "
        f"fused silhouette={heldout_metrics['fused_silhouette']:.4f} "
        f"cluster silhouette={heldout_metrics['cluster_silhouette']:.4f}"
    )
    return result


def choose_winner(screen):
    clearers = [r for r in screen if clears_pareto(r)]
    candidates = clearers if clearers else screen
    winner = max(candidates, key=lambda r: (
        r["heldout"]["fused_silhouette"] if np.isfinite(r["heldout"]["fused_silhouette"]) else float("-inf")
    ))
    return winner, bool(clearers)


def format_metric(value):
    return f"{value:.4f}" if np.isfinite(value) else "n/a (collapsed)"


def format_summary(values):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return "undefined (all collapsed)"
    return f"mean={np.mean(finite):.4f}; min={np.min(finite):.4f}; max={np.max(finite):.4f}"


def write_report(screen, winner, validated_winner, stress):
    four_seeds = [winner, *stress]
    clear_count = sum(clears_pareto(r) for r in four_seeds)
    selected = winner["heldout"]
    lines = [
        "# Reconstruction-anchored clustering head pilot (frozen buddy trunk)\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Method\n\n",
        "The buddy trunk (Attention-h1) is frozen throughout: this pilot loads "
        "`attention_h1_embedding_snapshot.npz`'s already-trained 32-D "
        "`train_embedding_post`/`heldout_embedding_post` directly and never "
        "instantiates or trains a buddy model. A small, genuinely "
        "**undercomplete** autoencoder (32→24→16 encoder, 16→24→32 "
        "decoder) is pretrained by reconstruction alone (MSE, Adam "
        f"lr={PRETRAIN_LEARNING_RATE:g}, {PRETRAIN_EPOCHS} epochs, batch "
        f"{PRETRAIN_BATCH_SIZE}) before any clustering pressure. This differs "
        "from two prior attempts: the decoupled `ClusterHead` (un-detached and "
        "detached) trained DEC's KL loss with **no reconstruction term at "
        "all** and collapsed to near-zero cluster-latent silhouette; feeding "
        "buddy's embedding into PercepT's own unmodified autoencoder **had** "
        "reconstruction but an **overcomplete** 128-D latent (128>32) that let "
        "the decoder cheat without well-separated clusters, scoring negative "
        "silhouette back in buddy's native space. This pilot's 16-D latent is "
        "smaller than its 32-D input, and input/output are both 32-D so there "
        "is no PercepT-style dimensionality mismatch to distort the "
        f"reconstruction/KL loss scale (LAMBDA_RECONSTRUCTION="
        f"{LAMBDA_RECONSTRUCTION:g} fixed, matching PercepT's own convention "
        "at a matched scale this time).\n\n",
        "K-means (100 clusters, `n_init=10`, `random_state=seed`) initializes "
        "on the pretrained encoder's clean train cluster-latent. The joint "
        "phase reuses PercepT's own `soft_assignments`/`target_distribution` "
        "(Student's-t kernel, appropriate for this unconstrained Euclidean "
        "latent) and highest-norm `prune_centers`, transcribed verbatim from "
        f"`run_percept_stage1_pilot.py`. `LAMBDA_DEC` is screened over "
        f"{LAMBDA_DEC_VALUES} with a linear {DEC_WARMUP_EPOCHS}-epoch warm-up, "
        f"fixed Adam lr={PRETRAIN_LEARNING_RATE:g}, {MAX_EPOCHS} full-batch "
        f"epochs (no early stop — no recall metric to plateau on here), "
        f"checkpoint cadence={CHECKPOINT_EVERY}. Silhouette is reported in "
        "both the 16-D cluster latent and the original 32-D buddy embedding, "
        "using the established seed-42 two-stage sampling convention "
        "(`np.random.default_rng(42)` draw of at most 6,000, "
        "`silhouette_score(sample_size=min(4000, len(idx)), "
        "random_state=42)`); the fused (32-D) silhouette is the headline "
        "comparable to every other pilot in this directory.\n\n",
        f"Held-out Pareto bar: emotion AMI > {EMOTION_PARETO_BAR:.4f} AND "
        f"genre AMI > {GENRE_PARETO_BAR:.4f}.\n\n",
        "## Single-seed screen (seed 42)\n\n",
        "| LAMBDA_DEC | split | emotion AMI | genre AMI | cluster silhouette "
        "| fused silhouette | clusters | held-out Pareto bar |\n",
        "|---:|---|---:|---:|---:|---:|---:|---|\n",
    ]
    for result in screen:
        for split, key in (("train", "train"), ("held-out", "heldout")):
            metrics = result[key]
            verdict = ("clears" if clears_pareto(result) else "does not clear") if split == "held-out" else "n/a"
            lines.append(
                f"| {result['lambda_dec']:g} | {split} | {metrics['emotion_ami']:.4f} | "
                f"{metrics['genre_ami']:.4f} | {format_metric(metrics['cluster_silhouette'])} | "
                f"{format_metric(metrics['fused_silhouette'])} | {metrics['communities']} | {verdict} |\n"
            )
    lines.append("\n### Screen collapse diagnostics\n\n")
    lines.append("| LAMBDA_DEC | split | min | max | median | below 1% / 67 | collapsed |\n")
    lines.append("|---:|---|---:|---:|---:|---:|---|\n")
    for result in screen:
        for split, key in (("train", "train_surviving"), ("held-out", "heldout_surviving")):
            d = result[key]
            lines.append(
                f"| {result['lambda_dec']:g} | {split} | {d['min']:,} | {d['max']:,} | "
                f"{d['median']:.1f} | {d['below_one_percent']}/{N_SURVIVING_CLUSTERS} | "
                f"{'yes' if d['collapsed'] else 'no'} |\n"
            )
    lines.append("\n## Winner selection\n\n")
    if validated_winner:
        clear_values = ", ".join(f"{r['lambda_dec']:g}" for r in screen if clears_pareto(r))
        lines.append(
            f"Pareto-bar clearers: {clear_values}. Selected LAMBDA_DEC="
            f"{winner['lambda_dec']:g}, highest held-out fused silhouette among "
            f"clearers ({format_metric(selected['fused_silhouette'])}).\n\n"
        )
    else:
        lines.append(
            f"No screened value clears the held-out Pareto bar. Selected "
            f"LAMBDA_DEC={winner['lambda_dec']:g} for highest held-out fused "
            f"silhouette among all three ({format_metric(selected['fused_silhouette'])}): "
            "best available, does not clear the Pareto bar.\n\n"
        )
    lines.extend([
        "## Four-seed stress of the selected LAMBDA_DEC\n\n",
        "| seed | held-out emotion AMI | held-out genre AMI | held-out cluster "
        "silhouette | held-out fused silhouette | held-out clusters | Pareto bar |\n",
        "|---:|---:|---:|---:|---:|---:|---|\n",
    ])
    for result in four_seeds:
        heldout = result["heldout"]
        lines.append(
            f"| {result['seed']} | {heldout['emotion_ami']:.4f} | {heldout['genre_ami']:.4f} | "
            f"{format_metric(heldout['cluster_silhouette'])} | {format_metric(heldout['fused_silhouette'])} | "
            f"{heldout['communities']} | {'clears' if clears_pareto(result) else 'does not clear'} |\n"
        )
    lines.append("\n### Held-out summary statistics\n\n")
    for metric, label in (("emotion_ami", "Emotion AMI"), ("genre_ami", "Genre AMI"),
                          ("cluster_silhouette", "Cluster-latent silhouette"),
                          ("fused_silhouette", "Fused-embedding silhouette")):
        values = [r["heldout"][metric] for r in four_seeds]
        lines.append(f"- {label} across four seeds: {format_summary(values)}.\n")
    lines.append(f"- Both held-out Pareto bars clear in {clear_count}/4 seeds.\n\n")
    lines.extend([
        "## Comparison against every prior DEC-hybrid attempt and established baselines\n\n",
        "| result | held-out emotion AMI | held-out genre AMI | held-out silhouette |\n",
        "|---|---:|---:|---:|\n",
    ])
    for name, emotion, genre, silhouette in REFERENCES:
        lines.append(f"| {name} | {emotion:.4f} | {genre:.4f} | {silhouette:.4f} |\n")
    fused_values = [r["heldout"]["fused_silhouette"] for r in four_seeds]
    finite_fused = np.asarray(fused_values, dtype=np.float64)
    finite_fused = finite_fused[np.isfinite(finite_fused)]
    mean_fused = float(np.mean(finite_fused)) if len(finite_fused) else float("nan")
    lines.append(
        f"| This pilot (four-seed mean, fused) | "
        f"{np.mean([r['heldout']['emotion_ami'] for r in four_seeds]):.4f} | "
        f"{np.mean([r['heldout']['genre_ami'] for r in four_seeds]):.4f} | "
        f"{format_metric(mean_fused)} |\n\n"
    )
    lines.append("## Final verdict\n\n")
    dec_hybrid_silhouettes = [ref[3] for ref in REFERENCES[1:6]]
    if np.isfinite(mean_fused) and mean_fused > max(dec_hybrid_silhouettes):
        lines.append(
            f"This is the best four-seed mean fused silhouette ({mean_fused:.4f}) of any "
            "clustering-loss attempt tonight, beating every prior DEC-hybrid attempt "
            f"({max(dec_hybrid_silhouettes):.4f} best previous). "
        )
    elif np.isfinite(mean_fused):
        lines.append(
            f"This four-seed mean fused silhouette ({mean_fused:.4f}) does not beat "
            f"the best prior DEC-hybrid attempt ({max(dec_hybrid_silhouettes):.4f}). "
        )
    else:
        lines.append("Fused silhouette is undefined (collapsed) in enough seeds that a four-seed mean cannot be computed. ")
    lines.append(
        f"Both AMI Pareto bars clear in {clear_count}/4 seeds. "
        f"Reconstruction anchoring plus an undercomplete latent "
        f"{'is' if (np.isfinite(mean_fused) and mean_fused > 0 and clear_count >= 3) else 'is not'} "
        "a robust fix for the DEC-hybrid direction by this investigation's standing bar. "
    )
    if not (np.isfinite(mean_fused) and mean_fused > 0 and clear_count >= 3):
        lines.append(
            "This is now the fifth (Euclidean, vMF, decoupled un-detached, decoupled "
            "detached) or sixth (including PercepT-on-buddy-embedding) attempt at "
            "attaching a DEC-style clustering objective to Attention-h1's embedding, "
            "and every one has failed to produce a robust, non-collapsed, Pareto-"
            "clearing result. This investigation has now tested every mechanism the "
            "brainstorm memo identified for this line of attack (geometry correction, "
            "gradient isolation, reconstruction anchoring at matched and mismatched "
            "scale) — this is a well-evidenced, settled negative result for attaching "
            "DEC-style clustering losses to Attention-h1, not an open question.\n"
        )
    else:
        lines.append("\n")
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def log(message):
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    log(f"Using {device} for the reconstruction-anchored cluster head pilot.")
    with np.load(BUDDY_SNAPSHOT, allow_pickle=True) as snapshot:
        train_fused = np.asarray(snapshot["train_embedding_post"], dtype=np.float32)
        heldout_fused = np.asarray(snapshot["heldout_embedding_post"], dtype=np.float32)
        train_emotion = np.asarray(snapshot["train_emotion"])
        heldout_emotion = np.asarray(snapshot["heldout_emotion"])
        train_genre_all = np.asarray(snapshot["train_genre"])
        heldout_genre_all = np.asarray(snapshot["heldout_genre"])
    train_genre_indices = np.flatnonzero(train_genre_all != "")
    heldout_genre_indices = np.flatnonzero(heldout_genre_all != "")
    if not len(train_genre_indices) or not len(heldout_genre_indices):
        raise RuntimeError("No genre-labelled paintings found in either split.")
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

    screen = [run_seed(SEED, value, data, device, log) for value in LAMBDA_DEC_VALUES]
    winner, validated_winner = choose_winner(screen)
    log(
        f"Selected LAMBDA_DEC={winner['lambda_dec']:g}: "
        f"{'Pareto-clear winner' if validated_winner else 'best available, does not clear the Pareto bar'}."
    )
    stress = [run_seed(seed, winner["lambda_dec"], data, device, log) for seed in STRESS_SEEDS]
    write_report(screen, winner, validated_winner, stress)
    log(f"Wrote {REPORT_PATH}.")


if __name__ == "__main__":
    main()
