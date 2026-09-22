"""Compare DEC with Leiden on the same GoEmotions-affect-only ArtELingo nodes.

Run manually in a GPU-capable environment.  This script deliberately uses the
same deduplicated painting order and mean-pooled GoEmotions probabilities as
``run_single_modality_pilot.py`` so that the only changed variable is the
clustering method.
"""

import importlib.util
import os
import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


OUT_DIR = os.path.dirname(__file__)
PIPELINE_PATH = os.path.join(OUT_DIR, "run_pipeline.py")
AFFECT_PILOT_PATH = os.path.join(OUT_DIR, "run_affect_pilot.py")
REPORT_PATH = os.path.join(OUT_DIR, "dec_pilot_report.md")

INPUT_DIM = 28
LATENT_DIM = 16
N_CLUSTERS = 28
PRETRAIN_EPOCHS = 50
DEC_EPOCHS = 100
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
LAMBDA_RECONSTRUCTION = 1.0
SEED = 42
SILHOUETTE_SUBSAMPLE_SIZE = 10_000


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_sibling_module(module_name: str, path: str):
    """Import a sibling standalone script without executing its ``main`` block."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_autoencoder() -> tuple[nn.Sequential, nn.Sequential]:
    """Build the deliberately small DEC autoencoder for 28-dimensional input."""
    encoder = nn.Sequential(
        nn.Linear(INPUT_DIM, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, LATENT_DIM),
    )
    decoder = nn.Sequential(
        nn.Linear(LATENT_DIM, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, INPUT_DIM),
    )
    return encoder, decoder


def pretrain_autoencoder(
    encoder: nn.Sequential,
    decoder: nn.Sequential,
    inputs: torch.Tensor,
    device: str,
) -> list[float]:
    """Pretrain with shuffled mini-batch reconstruction loss."""
    loader = DataLoader(TensorDataset(inputs), batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE
    )
    losses = []

    log(f"Starting autoencoder pretraining ({PRETRAIN_EPOCHS} epochs, batch_size={BATCH_SIZE})...")
    encoder.train()
    decoder.train()
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        total_loss = 0.0
        total_examples = 0
        for (batch,) in loader:
            batch = batch.to(device)
            reconstruction = decoder(encoder(batch))
            loss = F.mse_loss(reconstruction, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
            total_examples += len(batch)

        mean_loss = total_loss / total_examples
        losses.append(mean_loss)
        if epoch % 10 == 0:
            log(f"Pretrain epoch {epoch:03d}/{PRETRAIN_EPOCHS}: mean reconstruction loss={mean_loss:.6f}")
    return losses


def soft_assignments(latent: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Compute standard DEC Student's-t assignments with one degree of freedom."""
    squared_distances = torch.cdist(latent, centers).pow(2)
    q = (1.0 + squared_distances).reciprocal()
    return q / q.sum(dim=1, keepdim=True)


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    """Compute DEC's detached self-sharpening target distribution."""
    p = q.pow(2) / q.sum(dim=0, keepdim=True)
    p = p / p.sum(dim=1, keepdim=True)
    return p.detach()


def initialize_cluster_centers(encoder: nn.Sequential, inputs: torch.Tensor, device: str) -> nn.Parameter:
    """Fit KMeans on all pretrained latent nodes and return trainable centers."""
    encoder.eval()
    with torch.no_grad():
        latent = encoder(inputs.to(device)).cpu().numpy()
    log(f"Initializing {N_CLUSTERS} DEC cluster centers with KMeans (n_init=10, seed={SEED})...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED)
    kmeans.fit(latent)
    return nn.Parameter(torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(device))


def train_dec(
    encoder: nn.Sequential,
    decoder: nn.Sequential,
    centers: nn.Parameter,
    inputs: torch.Tensor,
    device: str,
) -> list[dict[str, float]]:
    """Jointly optimize encoder, decoder, and DEC centers over full batches."""
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + [centers],
        lr=LEARNING_RATE,
    )
    inputs = inputs.to(device)
    losses = []

    log(f"Starting joint DEC training ({DEC_EPOCHS} full-batch epochs)...")
    encoder.train()
    decoder.train()
    for epoch in range(1, DEC_EPOCHS + 1):
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

        checkpoint = {
            "total": total_loss.item(),
            "kl": kl_loss.item(),
            "reconstruction": reconstruction_loss.item(),
        }
        losses.append(checkpoint)
        if epoch % 10 == 0:
            log(
                f"DEC epoch {epoch:03d}/{DEC_EPOCHS}: total={checkpoint['total']:.6f}, "
                f"KL={checkpoint['kl']:.6f}, reconstruction={checkpoint['reconstruction']:.6f}"
            )
    return losses


def evaluate_clusters(
    pipeline,
    assignments: np.ndarray,
    paintings: list[str],
    majority_emotion: list[str],
) -> tuple[dict, dict, int]:
    """Evaluate full emotion labels and the genre-labelled painting subset."""
    emotion_metrics = pipeline.external_metrics(assignments, majority_emotion)
    genre_map = pipeline.load_genre_map()
    genre_indices = [i for i, painting in enumerate(paintings) if painting in genre_map]
    if not genre_indices:
        raise RuntimeError("No genre-labelled paintings overlap the deduplicated node set.")
    genre_metrics = pipeline.external_metrics(
        assignments[genre_indices],
        [genre_map[paintings[i]] for i in genre_indices],
    )
    return emotion_metrics, genre_metrics, len(genre_indices)


def final_silhouette(latent: np.ndarray, assignments: np.ndarray) -> tuple[float, str]:
    """Prefer the requested full silhouette calculation, with a 10k fallback."""
    try:
        return float(silhouette_score(latent, assignments)), f"full {len(latent):,} nodes"
    except (MemoryError, ValueError) as exc:
        log(f"Full silhouette score unavailable ({exc}); using a {SILHOUETTE_SUBSAMPLE_SIZE:,}-node subsample.")
        rng = np.random.default_rng(SEED)
        indices = rng.choice(len(latent), size=SILHOUETTE_SUBSAMPLE_SIZE, replace=False)
        return (
            float(silhouette_score(latent[indices], assignments[indices])),
            f"random {SILHOUETTE_SUBSAMPLE_SIZE:,}-node subsample after full-score fallback",
        )


def format_pretrain_checkpoints(losses: list[float]) -> str:
    checkpoints = [1, 10, 20, 30, 40, PRETRAIN_EPOCHS]
    return ", ".join(f"epoch {epoch}: {losses[epoch - 1]:.6f}" for epoch in checkpoints)


def format_dec_checkpoints(losses: list[dict[str, float]]) -> str:
    checkpoints = [1, 10, 20, 40, 60, 80, DEC_EPOCHS]
    return "; ".join(
        f"epoch {epoch}: total={losses[epoch - 1]['total']:.6f}, "
        f"KL={losses[epoch - 1]['kl']:.6f}, recon={losses[epoch - 1]['reconstruction']:.6f}"
        for epoch in checkpoints
    )


def write_report(
    pretrain_losses: list[float],
    dec_losses: list[dict[str, float]],
    cluster_sizes: np.ndarray,
    emotion_metrics: dict,
    genre_metrics: dict,
    genre_count: int,
    silhouette: float,
    silhouette_population: str,
) -> None:
    """Write the DEC-vs-Leiden result and the predeclared win decision."""
    small_clusters = int(np.sum(cluster_sizes < 614))
    collapsed = small_clusters >= N_CLUSTERS / 2
    cleared_ami_bar = emotion_metrics["AMI"] > 0.177
    real_win = cleared_ami_bar and not collapsed

    lines = [
        "# ArtELingo DEC-vs-Leiden pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "**Setup:** The same 61,402 x 28 mean-pooled GoEmotions sigmoid-probability "
        "nodes used by the GoEmotions-only Leiden pilot were clustered with a small "
        "28→64→32→16→32→64→28 autoencoder and DEC (K=28). Genre metrics use the "
        f"{genre_count}-painting genre-labelled overlap.\n\n",
        "## Training trajectories\n\n",
        f"- Pretraining mean reconstruction loss: {format_pretrain_checkpoints(pretrain_losses)}.\n",
        f"- Joint DEC losses: {format_dec_checkpoints(dec_losses)}.\n\n",
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
        "| Leiden on GoEmotions-only (reference) | 0.1180 | — | 0.0396 | — |\n",
        f"| DEC on GoEmotions-only (this run) | {emotion_metrics['AMI']:.4f} | "
        f"{emotion_metrics['V_measure']:.4f} | {genre_metrics['AMI']:.4f} | "
        f"{genre_metrics['V_measure']:.4f} |\n\n",
        "## Decision\n\n",
    ]
    if real_win:
        lines.append(
            f"DEC is a real win: emotion AMI={emotion_metrics['AMI']:.4f} clears the "
            "predeclared AMI > 0.177 bar (a 50% relative gain over 0.1180), and the "
            "cluster-size check did not detect collapse.\n"
        )
    else:
        unmet = []
        if not cleared_ami_bar:
            unmet.append(f"emotion AMI={emotion_metrics['AMI']:.4f} did not clear AMI > 0.177")
        if collapsed:
            unmet.append(f"{small_clusters} of {N_CLUSTERS} clusters fell below 1% of nodes")
        lines.append(
            "DEC is not a real win under the predeclared criteria: "
            + "; and ".join(unmet)
            + ". Both AMI improvement and non-collapse were required.\n"
        )

    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
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
    # Match the affect-only Leiden pilot, whose single-modality graph builder
    # L2-normalizes these same mean-pooled GoEmotions vectors before clustering.
    affect_nodes = affect_pilot.l2_normalize(affect_nodes).astype(np.float32, copy=False)
    inputs = torch.from_numpy(affect_nodes.astype(np.float32, copy=False))

    encoder, decoder = build_autoencoder()
    encoder.to(device)
    decoder.to(device)
    pretrain_losses = pretrain_autoencoder(encoder, decoder, inputs, device)
    centers = initialize_cluster_centers(encoder, inputs, device)
    dec_losses = train_dec(encoder, decoder, centers, inputs, device)

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
