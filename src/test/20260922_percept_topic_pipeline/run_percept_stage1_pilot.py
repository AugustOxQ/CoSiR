"""Run the ArtELingo PercepT Stage 1 P-Topic Formation pilot.

This standalone GPU pilot fits a reconstruction-grounded, DEC-sharpened topic
space on train paintings only, then evaluates frozen surviving centroids on
the genuinely held-out val+test split.  It deliberately does not implement
PercepT Stage 2's supervised image-only topic mapper.
"""

import importlib.util
import json
import os
import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


OUT_DIR = os.path.dirname(__file__)
BUDDY_DIR = os.path.abspath(os.path.join(OUT_DIR, "..", "20260923_artelingo_buddy_analysis"))
PIPELINE_PATH = os.path.join(BUDDY_DIR, "run_pipeline.py")
AFFECT_PILOT_PATH = os.path.join(BUDDY_DIR, "run_affect_pilot.py")
CCA_AUDIT_PATH = os.path.join(BUDDY_DIR, "run_cca_audit_pilot.py")
REPORT_PATH = os.path.join(OUT_DIR, "percept_stage1_pilot_report.md")

HELDOUT_STORAGE_DIR = "/data/SSD2/pre_extract/artelingo_heldout/features"
HELDOUT_JSON = "/data/PDD/artelingo/artelingo_val_test.json"
MODEL_NAME = "SamLowe/roberta-base-go_emotions"
BATCH_SIZE = 256
MAX_LENGTH = 64
PRETRAIN_BATCH_SIZE = 1024
PRETRAIN_EPOCHS = 100
N_INITIAL_CLUSTERS = 100
N_SURVIVING_CLUSTERS = 67
MAX_DEC_EPOCHS = 500
STABILITY_THRESHOLD = 0.001
PRETRAIN_LEARNING_RATE = 1e-3
DEC_LEARNING_RATE = 1e-4
LAMBDA_RECONSTRUCTION = 1.0
SEED = 42


def load_sibling_module(module_name: str, path: str):
    """Load a standalone sibling script without running its main block."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def caption_text(record: dict) -> str:
    """Match ArtELingo's string-or-singleton-list caption convention."""
    caption = record["caption"]
    return caption if isinstance(caption, str) else caption[0]


def extract_affect_embedding_nodes(train_json: str, paintings: list[str], device: str, log) -> np.ndarray:
    """Mean-pool masked RoBERTa token embeddings, then captions by painting."""
    log(f"Loading embedding-level GoEmotions encoder ({MODEL_NAME}) on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    with open(train_json) as train_file:
        train = json.load(train_file)
    painting_to_idx = {painting: index for index, painting in enumerate(paintings)}
    row_node_indices = []
    captions = []
    for record in train:
        if record.get("language", "english").lower() != "english":
            continue
        try:
            row_node_indices.append(painting_to_idx[record["painting"]])
        except KeyError as exc:
            raise RuntimeError(
                "Caption row references a painting absent from deduplicated features: "
                f"{record['painting']}"
            ) from exc
        captions.append(caption_text(record))
    if not captions:
        raise RuntimeError("No English caption rows found in TRAIN_JSON.")

    embedding_sums = None
    embedding_counts = np.zeros(len(paintings), dtype=np.int32)
    total_batches = (len(captions) + BATCH_SIZE - 1) // BATCH_SIZE
    log(
        f"Extracting embeddings for {len(captions):,} English caption rows in "
        f"{total_batches} batches (batch_size={BATCH_SIZE}, max_length={MAX_LENGTH})..."
    )
    with torch.no_grad():
        for start in range(0, len(captions), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(captions))
            encoded = tokenizer(
                captions[start:end], padding=True, truncation=True,
                max_length=MAX_LENGTH, return_tensors="pt",
            ).to(device)
            outputs = model(**encoded).last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            caption_embeddings = (
                (outputs * attention_mask).sum(dim=1)
                / attention_mask.sum(dim=1).clamp_min(1)
            ).cpu().numpy().astype(np.float32, copy=False)
            if embedding_sums is None:
                embedding_sums = np.zeros(
                    (len(paintings), caption_embeddings.shape[1]), dtype=np.float32
                )
            node_indices = np.asarray(row_node_indices[start:end], dtype=np.intp)
            np.add.at(embedding_sums, node_indices, caption_embeddings)
            np.add.at(embedding_counts, node_indices, 1)
            if (start // BATCH_SIZE + 1) % 100 == 0 or end == len(captions):
                log(f"Affect encoder progress: {end:,}/{len(captions):,} caption rows.")

    missing = np.flatnonzero(embedding_counts == 0)
    if len(missing):
        raise RuntimeError(f"{len(missing)} deduplicated paintings have no English affect embeddings.")
    return embedding_sums / embedding_counts[:, None]


def fused_embeddings(img_nodes: np.ndarray, txt_nodes: np.ndarray, affect_embeddings: np.ndarray, cca_audit, affect_pilot) -> np.ndarray:
    """Build the documented 2:1 content:affect concat-with-repetition input h."""
    content = cca_audit.content_features(img_nodes, txt_nodes, affect_pilot)
    content = affect_pilot.l2_normalize(content)
    affect = affect_pilot.l2_normalize(affect_embeddings)
    return affect_pilot.l2_normalize(
        np.concatenate((content, content, affect), axis=1)
    ).astype(np.float32, copy=False)


def build_autoencoder(input_dim: int) -> tuple[nn.Sequential, nn.Sequential]:
    """Build the paper-specified 128-dimensional PercepT Stage 1 autoencoder."""
    encoder = nn.Sequential(
        nn.Linear(input_dim, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(),
        nn.Linear(500, 2000), nn.ReLU(), nn.Linear(2000, 128),
    )
    decoder = nn.Sequential(
        nn.Linear(128, 2000), nn.ReLU(), nn.Linear(2000, 500), nn.ReLU(),
        nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, input_dim),
    )
    return encoder, decoder


def pretrain_autoencoder(encoder: nn.Sequential, decoder: nn.Sequential, inputs: torch.Tensor, device: str, log) -> list[float]:
    """Pretrain reconstruction with shuffled 1,024-node batches for 100 epochs."""
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=PRETRAIN_LEARNING_RATE)
    losses = []
    encoder.train()
    decoder.train()
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        permutation = torch.randperm(len(inputs))
        total_loss = 0.0
        total_nodes = 0
        for start in range(0, len(inputs), PRETRAIN_BATCH_SIZE):
            batch = inputs[permutation[start:start + PRETRAIN_BATCH_SIZE]].to(device)
            reconstruction_loss = F.mse_loss(decoder(encoder(batch)), batch)
            optimizer.zero_grad()
            reconstruction_loss.backward()
            optimizer.step()
            total_loss += reconstruction_loss.item() * len(batch)
            total_nodes += len(batch)
        mean_loss = total_loss / total_nodes
        losses.append(mean_loss)
        if epoch % 10 == 0:
            log(f"Pretrain epoch {epoch:03d}/{PRETRAIN_EPOCHS}: mean reconstruction={mean_loss:.6f}")
    return losses


def soft_assignments(latent: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Student's-t DEC assignments (alpha=1), normalized per node."""
    squared_distances = torch.sum((latent.unsqueeze(1) - centers.unsqueeze(0)) ** 2, dim=2)
    q = 1.0 / (1.0 + squared_distances)
    return q / q.sum(dim=1, keepdim=True)


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    """Return the detached DEC self-sharpened target distribution P."""
    weights = q.pow(2) / q.sum(dim=0, keepdim=True)
    return (weights / weights.sum(dim=1, keepdim=True)).detach()


def initialize_cluster_centers(encoder: nn.Sequential, inputs: torch.Tensor, device: str) -> nn.Parameter:
    """Initialize 100 DEC centers with K-means on pretrained train latents."""
    encoder.eval()
    with torch.no_grad():
        latent = encoder(inputs.to(device)).cpu().numpy()
    kmeans = KMeans(n_clusters=N_INITIAL_CLUSTERS, n_init=10, random_state=SEED)
    kmeans.fit(latent)
    return nn.Parameter(torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(device))


def train_dec_until_stable(encoder: nn.Sequential, decoder: nn.Sequential, centers: nn.Parameter, inputs: torch.Tensor, device: str, log) -> tuple[list[dict[str, float]], str, int]:
    """Reuse the v2 convergence-controlled full-batch DEC mechanics exactly."""
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + [centers], lr=DEC_LEARNING_RATE
    )
    inputs = inputs.to(device)
    encoder.eval()
    with torch.no_grad():
        previous_assignments = soft_assignments(encoder(inputs), centers).argmax(dim=1)
    log(
        "Starting joint DEC training "
        f"(up to {MAX_DEC_EPOCHS} full-batch epochs, lr={DEC_LEARNING_RATE:g}, "
        f"stability threshold={STABILITY_THRESHOLD:.3f})..."
    )
    losses = []
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
        fraction_changed = num_changed / len(inputs)
        previous_assignments = current_assignments
        encoder.train()
        checkpoint = {
            "epoch": float(epoch), "total": total_loss.item(), "kl": kl_loss.item(),
            "reconstruction": reconstruction_loss.item(), "fraction_changed": fraction_changed,
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
            log(f"DEC epoch {epoch:03d}/{MAX_DEC_EPOCHS}: fraction_changed={fraction_changed:.6f} ({num_changed:,} nodes)")
        if stopping:
            stop_reason = "stability criterion" if stopped_for_stability else "epoch ceiling"
            log(f"Stopping DEC at epoch {epoch}: {stop_reason}.")
            return losses, stop_reason, epoch
    raise RuntimeError("DEC training exited without a stopping condition.")


def prune_centers(centers: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
    """Keep the 67 highest-norm final centers as the documented pruning proxy."""
    norms = torch.linalg.vector_norm(centers.detach(), dim=1)
    surviving = torch.argsort(norms, descending=True)[:N_SURVIVING_CLUSTERS]
    return centers.detach()[surviving], surviving.cpu().numpy()


def evaluate_assignments(pipeline, paintings: list[str], majority_emotion: list[str], latent: np.ndarray, assignments: np.ndarray) -> dict:
    """Score labels, cluster-size collapse, and all-node latent silhouette."""
    emotion_metrics = pipeline.external_metrics(assignments, majority_emotion)
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
    collapsed = small_clusters > N_SURVIVING_CLUSTERS / 2
    silhouette = float(silhouette_score(latent, assignments))
    return {
        "emotion": emotion_metrics, "genre": genre_metrics, "genre_count": len(genre_indices),
        "cluster_sizes": cluster_sizes, "small_clusters": small_clusters,
        "collapsed": collapsed, "silhouette": silhouette,
    }


def format_pretrain_losses(losses: list[float]) -> str:
    return "; ".join(f"epoch {epoch}: {losses[epoch - 1]:.6f}" for epoch in range(10, PRETRAIN_EPOCHS + 1, 10))


def format_dec_losses(losses: list[dict[str, float]]) -> str:
    by_epoch = {int(loss["epoch"]): loss for loss in losses}
    checkpoints = sorted({1, len(losses), *range(25, len(losses) + 1, 50)})
    return "; ".join(
        f"epoch {epoch}: total={by_epoch[epoch]['total']:.6f}, KL={by_epoch[epoch]['kl']:.6f}, "
        f"recon={by_epoch[epoch]['reconstruction']:.6f}, fraction_changed={by_epoch[epoch]['fraction_changed']:.6f}"
        for epoch in checkpoints
    )


def verdict(metrics: dict) -> str:
    if metrics["collapsed"]:
        return "Collapsed"
    if metrics["emotion"]["AMI"] > 0.1236 and metrics["genre"]["AMI"] > 0.1954:
        return "Real success"
    return "Merely a compromise"


def write_report(pretrain_losses: list[float], dec_losses: list[dict[str, float]], stop_reason: str, stop_epoch: int, input_dim: int, train_metrics: dict, heldout_metrics: dict) -> None:
    """Write the predeclared comparison report after GPU execution."""
    final_verdict = verdict(heldout_metrics)
    lines = [
        "# ArtELingo PercepT Stage 1 P-Topic Formation pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Setup and documented deviations\n\n",
        f"The train-only input is a {input_dim}-dimensional fused vector. Content is the existing independently normalized CLIP image/text concatenation, normalized again as a whole; affect is a 768-dimensional embedding from masked-token mean pooling of `SamLowe/roberta-base-go_emotions`, then caption-mean pooling by painting. This is an embedding-level affect signal, a methodological upgrade over all earlier investigation pilots' 28-dimensional label probabilities. RoBERTa substitutes for the paper's ModernBERT-family GoEmotions encoder because this cached, validated project encoder shares the fine-tuning objective but not the backbone family.\n\n",
        "Because this repository's ViT-B/32 CLIP content vector and the 768-dimensional affect vector have unequal dimensions, literal Eq. 2 summation is impossible without an unvalidated projection. The documented substitute is `L2_normalize(concat([h_C', h_C', h_E]))`: repeated content preserves the paper's 2:1 content:affect norm-budget weighting.\n\n",
        "DEC uses convergence-controlled full-batch training (lr=1e-4; stop at `fraction_changed < 0.001`, ceiling 500) rather than the paper's fixed 200 epochs. This is the deliberate, previously validated project deviation. Norm-threshold pruning is approximated by retaining the 67 highest-L2-norm centroids out of 100, matching the paper's reported retention rate because its exact threshold rule is unstated.\n\n",
        "## Predeclared success criterion\n\n",
        "**held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, both simultaneously.**\n\n",
        "## Training trajectories\n\n",
        f"- Pretraining reconstruction: {format_pretrain_losses(pretrain_losses)}.\n",
        f"- DEC stopped via the **{stop_reason}** at epoch **{stop_epoch}** (threshold `fraction_changed < {STABILITY_THRESHOLD:.3f}`, ceiling {MAX_DEC_EPOCHS}).\n",
        f"- Joint DEC: {format_dec_losses(dec_losses)}.\n\n",
        "## Cluster-size collapse detection\n\n",
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
            f"| PercepT Stage 1 fused AE+DEC (67 surviving topics) | {split_name} | "
            f"{metrics['emotion']['AMI']:.4f} | {metrics['emotion']['V_measure']:.4f} | "
            f"{metrics['genre']['AMI']:.4f} | {metrics['genre']['V_measure']:.4f} | "
            f"{'Collapsed' if metrics['collapsed'] else 'not collapsed'} | {metrics['silhouette']:.4f} |\n"
        )
    lines.extend([
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
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    pipeline = load_sibling_module("artelingo_run_pipeline_percept_train", PIPELINE_PATH)
    affect_pilot = load_sibling_module("artelingo_run_affect_pilot_percept", AFFECT_PILOT_PATH)
    cca_audit = load_sibling_module("artelingo_run_cca_audit_percept", CCA_AUDIT_PATH)
    heldout_pipeline = load_sibling_module("artelingo_run_pipeline_percept_heldout", PIPELINE_PATH)
    heldout_pipeline.STORAGE_DIR = HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = HELDOUT_JSON

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
    log(f"Using {device} for embedding extraction and DEC training.")
    affect_train = extract_affect_embedding_nodes(pipeline.TRAIN_JSON, paintings, device, log)
    affect_heldout = extract_affect_embedding_nodes(HELDOUT_JSON, heldout_paintings, device, log)
    train_h = fused_embeddings(img_nodes, txt_nodes, affect_train, cca_audit, affect_pilot)
    heldout_h = fused_embeddings(heldout_img_nodes, heldout_txt_nodes, affect_heldout, cca_audit, affect_pilot)
    if train_h.shape[1] != heldout_h.shape[1]:
        raise RuntimeError(f"Train/held-out fused dimensions differ: {train_h.shape[1]} vs {heldout_h.shape[1]}.")

    train_inputs = torch.from_numpy(train_h)
    encoder, decoder = build_autoencoder(train_h.shape[1])
    encoder.to(device)
    decoder.to(device)
    pretrain_losses = pretrain_autoencoder(encoder, decoder, train_inputs, device, log)
    centers = initialize_cluster_centers(encoder, train_inputs, device)
    dec_losses, stop_reason, stop_epoch = train_dec_until_stable(encoder, decoder, centers, train_inputs, device, log)
    surviving_centers, surviving_indices = prune_centers(centers)
    log(f"Pruned {N_INITIAL_CLUSTERS - N_SURVIVING_CLUSTERS} lowest-norm centers; retained original center indices {surviving_indices.tolist()}.")

    encoder.eval()
    with torch.no_grad():
        train_latent = encoder(train_inputs.to(device))
        train_assignments = soft_assignments(train_latent, surviving_centers).argmax(dim=1).cpu().numpy()
        heldout_latent = encoder(torch.from_numpy(heldout_h).to(device))
        heldout_assignments = soft_assignments(heldout_latent, surviving_centers).argmax(dim=1).cpu().numpy()
    train_metrics = evaluate_assignments(pipeline, paintings, train_emotions, train_latent.cpu().numpy(), train_assignments)
    heldout_metrics = evaluate_assignments(heldout_pipeline, heldout_paintings, heldout_emotions, heldout_latent.cpu().numpy(), heldout_assignments)
    write_report(pretrain_losses, dec_losses, stop_reason, stop_epoch, train_h.shape[1], train_metrics, heldout_metrics)
    log(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
