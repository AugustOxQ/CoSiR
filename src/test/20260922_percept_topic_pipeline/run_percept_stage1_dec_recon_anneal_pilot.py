"""Re-fit deterministic PercepT Stage 1 with annealed DEC reconstruction loss."""

import importlib.util
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.metrics import silhouette_score
from torch.nn import functional as F


OUT_DIR = os.path.dirname(__file__)
BASE_PILOT_PATH = os.path.join(OUT_DIR, "run_percept_stage1_pilot.py")
SWEEP_PILOT_PATH = os.path.join(OUT_DIR, "run_percept_stage1_cluster_count_sweep_pilot.py")
NPZ_PATH = os.path.join(OUT_DIR, "percept_stage1_dec_recon_anneal_snapshot.npz")
REPORT_PATH = os.path.join(OUT_DIR, "percept_stage1_dec_recon_anneal_pilot_report.md")

N_INITIAL_CLUSTERS = 60
N_SURVIVING_CLUSTERS = 40
SEED = 42
# First-pass, untuned schedule choice.
ANNEAL_EPOCHS = 200
RECON_FLOOR = 0.1
NEAREST_CENTER_CHUNK = 4096
# Prior run_percept_stage1_embedding_snapshot_pilot.py fixed point and an ad hoc
# silhouette check of its snapshot; these are reference values, not gates.
BASELINE_HELDOUT_EMOTION_AMI = 0.1225
BASELINE_HELDOUT_GENRE_AMI = 0.2274
BASELINE_TRAIN_SILHOUETTE_PRE = 0.0489
BASELINE_TRAIN_SILHOUETTE_POST = 0.0402
BASELINE_TOPICS_BELOW_ONE_PERCENT = 0


def load_module(module_name: str, path: str):
    """Load a standalone sibling script without running its main block."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nearest_center_labels(latent: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Hard-assign each row of `latent` to its nearest row of `centers`."""
    labels = np.empty(latent.shape[0], dtype=np.int32)
    for start in range(0, latent.shape[0], NEAREST_CENTER_CHUNK):
        block = latent[start : start + NEAREST_CENTER_CHUNK]
        dists = ((block[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels[start : start + NEAREST_CENTER_CHUNK] = dists.argmin(axis=1)
    return labels


def reconstruction_weight(epoch: int, anneal_epochs: int, recon_floor: float) -> float:
    """Linearly anneal the DEC reconstruction coefficient, then hold its floor."""
    return 1.0 - (1.0 - recon_floor) * min(epoch, anneal_epochs) / anneal_epochs


def train_dec_until_stable_annealed(
    encoder, decoder, centers, inputs, device, log, n_initial_clusters,
    anneal_epochs=ANNEAL_EPOCHS, recon_floor=RECON_FLOOR,
):
    """Train balanced DEC with only the reconstruction coefficient annealed."""
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + [centers],
        lr=cluster_sweep.DEC_LEARNING_RATE,
    )
    inputs = inputs.to(device)
    encoder.eval()
    with torch.no_grad():
        previous_assignments = base.soft_assignments(encoder(inputs), centers).argmax(
            dim=1
        )
    log(
        f"Starting K={n_initial_clusters} joint DEC training "
        f"(up to {cluster_sweep.MAX_DEC_EPOCHS} full-batch epochs, "
        f"lr={cluster_sweep.DEC_LEARNING_RATE:g}, "
        f"stability threshold={cluster_sweep.STABILITY_THRESHOLD:.3f}, "
        f"balance lambda={cluster_sweep.LAMBDA_BALANCE:g})..."
    )
    losses = []
    encoder.train()
    decoder.train()
    for epoch in range(1, cluster_sweep.MAX_DEC_EPOCHS + 1):
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
        lambda_reconstruction = reconstruction_weight(epoch, anneal_epochs, recon_floor)
        total_loss = (
            kl_loss
            + lambda_reconstruction * reconstruction_loss
            + cluster_sweep.LAMBDA_BALANCE * balance_loss
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        encoder.eval()
        with torch.no_grad():
            current_assignments = base.soft_assignments(
                encoder(inputs), centers
            ).argmax(dim=1)
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
            "lambda_reconstruction": lambda_reconstruction,
        }
        losses.append(checkpoint)
        stopped_for_stability = fraction_changed < cluster_sweep.STABILITY_THRESHOLD
        reached_epoch_ceiling = epoch == cluster_sweep.MAX_DEC_EPOCHS
        stopping = stopped_for_stability or reached_epoch_ceiling
        if epoch % 25 == 0 or stopping:
            log(
                f"K={n_initial_clusters} DEC epoch {epoch:03d}/{cluster_sweep.MAX_DEC_EPOCHS}: "
                f"total={checkpoint['total']:.6f}, KL={checkpoint['kl']:.6f}, "
                f"reconstruction={checkpoint['reconstruction']:.6f}, "
                f"balance={checkpoint['balance']:.6f}, "
                f"lambda_reconstruction={lambda_reconstruction:.6f}"
            )
        if epoch % 10 == 0 or stopping:
            log(
                f"K={n_initial_clusters} DEC epoch {epoch:03d}/{cluster_sweep.MAX_DEC_EPOCHS}: "
                f"fraction_changed={fraction_changed:.6f} ({num_changed:,} nodes), "
                f"lambda_reconstruction={lambda_reconstruction:.6f}"
            )
        if stopping:
            stop_reason = (
                "stability criterion" if stopped_for_stability else "epoch ceiling"
            )
            log(f"Stopping K={n_initial_clusters} DEC at epoch {epoch}: {stop_reason}.")
            return losses, stop_reason, epoch
    raise RuntimeError("DEC training exited without a stopping condition.")


def sampled_silhouette(latent: np.ndarray, labels: np.ndarray) -> float:
    """Apply the reference's seeded two-stage silhouette sampling."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(latent), size=min(6000, len(latent)), replace=False)
    return float(
        silhouette_score(
            latent[idx], labels[idx], sample_size=min(4000, len(idx)),
            random_state=SEED,
        )
    )


def write_report(
    heldout_metrics: dict, train_silhouette_pre: float,
    train_silhouette_post: float, topic_sizes: np.ndarray, train_n: int,
    heldout_n: int, stop_reason: str, epochs: int, elapsed_seconds: float,
) -> str:
    """Write the comparison, collapse diagnostic, and predeclared verdict."""
    emotion_ami = heldout_metrics["emotion"]["AMI"]
    genre_ami = heldout_metrics["genre"]["AMI"]
    below_one_percent = int(np.sum(topic_sizes < 0.01 * train_n))
    non_collapsed = below_one_percent <= BASELINE_TOPICS_BELOW_ONE_PERCENT
    if not non_collapsed:
        verdict = "Collapsed"
    elif (
        emotion_ami > cluster_sweep.EMOTION_PARETO_BAR
        and genre_ami > cluster_sweep.GENRE_PARETO_BAR
    ):
        verdict = "Real success"
    else:
        verdict = "Merely a compromise"

    rows = (
        ("Held-out emotion AMI", BASELINE_HELDOUT_EMOTION_AMI, emotion_ami),
        ("Held-out genre AMI", BASELINE_HELDOUT_GENRE_AMI, genre_ami),
        ("Train silhouette pre-DEC", BASELINE_TRAIN_SILHOUETTE_PRE, train_silhouette_pre),
        ("Train silhouette post-DEC", BASELINE_TRAIN_SILHOUETTE_POST, train_silhouette_post),
    )
    lines = [
        "# PercepT Stage 1 DEC reconstruction anneal pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Controlled change\n\n",
        "K=60/40, seed 42, the same forced-deterministic fit, and "
        "`LAMBDA_BALANCE=1000` are retained. The only training-objective change "
        "is the reconstruction coefficient: `1 - (1 - 0.1) * min(epoch, 200) / 200` "
        "for 1-indexed DEC epochs, then 0.1. The 200-epoch duration and 0.1 "
        "floor are first-pass, untuned choices. The hypothesis is that "
        "reconstruction weight 1 throughout the run opposes KL clustering and "
        "leaves the post-DEC latent geometry loose. The balance regularizer "
        "remains at 1000 because earlier unbalanced DEC collapsed.\n\n",
        f"DEC stopped at epoch {epochs} by {stop_reason}; whole-script wall time "
        f"{elapsed_seconds:.1f} seconds. Train N={train_n:,}; held-out N={heldout_n:,}.\n\n",
        "## Comparison with established no-anneal baseline\n\n",
        "The fixed-weight-1.0 AMI baseline comes from the prior "
        "`run_percept_stage1_embedding_snapshot_pilot.py` run. Its silhouette "
        "values come from an ad hoc check of that snapshot. The pre-DEC "
        "silhouette is a correctness check on the unchanged pretraining path, "
        "not a new result or a gate. Both silhouette calculations here use "
        "a seeded sample of at most 6,000 rows, followed by "
        "`silhouette_score(..., sample_size=min(4000, N), random_state=42)`.\n\n",
        "| Metric | No anneal baseline | This annealed run | Absolute difference |\n",
        "|---|---:|---:|---:|\n",
    ]
    for name, baseline, observed in rows:
        lines.append(
            f"| {name} | {baseline:.6f} | {observed:.6f} | "
            f"{abs(observed - baseline):.6f} |\n"
        )
    improvement = train_silhouette_post - BASELINE_TRAIN_SILHOUETTE_POST
    direction = "improved" if improvement > 0 else "declined" if improvement < 0 else "was unchanged"
    lines.extend([
        "\n",
        f"Train post-DEC silhouette {direction} versus the 0.0402 no-anneal "
        f"baseline by {abs(improvement):.6f}.\n\n",
        "## Surviving-topic size diagnostic (train hard assignments)\n\n",
        "| Statistic | This run | Known K=60/40 baseline |\n",
        "|---|---:|---:|\n",
        f"| Minimum topic size | {int(topic_sizes.min()):,} | not recorded |\n",
        f"| Maximum topic size | {int(topic_sizes.max()):,} | not recorded |\n",
        f"| Median topic size | {float(np.median(topic_sizes)):.1f} | not recorded |\n",
        f"| Topics below 1% of train N | {below_one_percent}/{N_SURVIVING_CLUSTERS} "
        f"| {BASELINE_TOPICS_BELOW_ONE_PERCENT}/{N_SURVIVING_CLUSTERS} |\n\n",
        "## Verdict\n\n",
        f"**{verdict}.** Non-collapsed means no more than "
        f"{BASELINE_TOPICS_BELOW_ONE_PERCENT} of 40 surviving train topics below "
        f"1% of train N. The held-out Pareto bars are emotion AMI > "
        f"{cluster_sweep.EMOTION_PARETO_BAR:.4f} and genre AMI > "
        f"{cluster_sweep.GENRE_PARETO_BAR:.4f}; both must clear for Real success.\n",
    ])
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)
    return verdict


def main() -> None:
    """Re-fit the standing configuration once with annealed reconstruction."""
    start_time = time.perf_counter()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    pipeline = base.load_sibling_module(
        "artelingo_run_pipeline_percept_anneal_train", base.PIPELINE_PATH
    )
    affect_pilot = base.load_sibling_module(
        "artelingo_run_affect_pilot_percept_anneal", base.AFFECT_PILOT_PATH
    )
    cca_audit = base.load_sibling_module(
        "artelingo_run_cca_audit_percept_anneal", base.CCA_AUDIT_PATH
    )
    heldout_pipeline = base.load_sibling_module(
        "artelingo_run_pipeline_percept_anneal_heldout", base.PIPELINE_PATH
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
    (
        heldout_paintings, heldout_img_nodes, heldout_txt_nodes,
        heldout_emotion_counts,
    ) = heldout_pipeline.load_dedup_features()
    heldout_emotions = [
        heldout_pipeline.majority(counts) for counts in heldout_emotion_counts
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {device} for the annealed Stage-1 re-fit and embedding snapshot.")
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
    heldout_inputs = torch.from_numpy(heldout_h)
    encoder, decoder = base.build_autoencoder(train_h.shape[1])
    encoder.to(device)
    decoder.to(device)
    base.pretrain_autoencoder(encoder, decoder, train_inputs, device, log)

    encoder.eval()
    with torch.no_grad():
        train_latent_pre = encoder(train_inputs.to(device)).cpu().numpy()
        heldout_latent_pre = encoder(heldout_inputs.to(device)).cpu().numpy()
    log("Captured pre-DEC encoder latents for train and held-out.")

    centers = cluster_sweep.initialize_cluster_centers(
        encoder, train_inputs, device, N_INITIAL_CLUSTERS, SEED
    )
    initial_centers_snapshot = centers.detach().clone().cpu().numpy()
    train_initial_label = nearest_center_labels(train_latent_pre, initial_centers_snapshot)

    losses, stop_reason, epochs = train_dec_until_stable_annealed(
        encoder, decoder, centers, train_inputs, device, log, N_INITIAL_CLUSTERS
    )
    surviving_centers, surviving_indices = cluster_sweep.prune_centers(
        centers, N_SURVIVING_CLUSTERS
    )
    log(
        f"Pruned {N_INITIAL_CLUSTERS - N_SURVIVING_CLUSTERS} centers; retained "
        f"original indices {surviving_indices.tolist()}."
    )

    encoder.eval()
    with torch.no_grad():
        train_latent_post = encoder(train_inputs.to(device)).cpu().numpy()
        heldout_latent_post = encoder(heldout_inputs.to(device)).cpu().numpy()
        train_topic = (
            base.soft_assignments(
                torch.from_numpy(train_latent_post).to(device), surviving_centers
            ).argmax(dim=1).cpu().numpy()
        )
        heldout_topic = (
            base.soft_assignments(
                torch.from_numpy(heldout_latent_post).to(device), surviving_centers
            ).argmax(dim=1).cpu().numpy()
        )
    log("Captured post-DEC encoder latents and surviving-topic assignments.")

    _, heldout_metrics, _ = cluster_sweep.evaluate_run(
        encoder, centers, train_inputs, heldout_h, pipeline, heldout_pipeline,
        paintings, train_emotions, heldout_paintings, heldout_emotions, device,
        N_SURVIVING_CLUSTERS,
    )
    log("Calculating seeded train silhouettes...")
    train_silhouette_pre = sampled_silhouette(train_latent_pre, train_initial_label)
    train_silhouette_post = sampled_silhouette(train_latent_post, train_topic)
    topic_sizes = np.bincount(train_topic, minlength=N_SURVIVING_CLUSTERS).astype(np.int64)
    genre_map = pipeline.load_genre_map()
    train_genre = np.array([genre_map.get(p, "") for p in paintings], dtype=object)
    heldout_genre = np.array(
        [genre_map.get(p, "") for p in heldout_paintings], dtype=object
    )

    np.savez_compressed(
        NPZ_PATH,
        train_paintings=np.array(paintings, dtype=object),
        train_latent_pre=train_latent_pre.astype(np.float32),
        train_latent_post=train_latent_post.astype(np.float32),
        train_initial_label=train_initial_label.astype(np.int16),
        train_topic=train_topic.astype(np.int16),
        train_emotion=np.array(train_emotions, dtype=object),
        train_genre=train_genre,
        heldout_paintings=np.array(heldout_paintings, dtype=object),
        heldout_latent_pre=heldout_latent_pre.astype(np.float32),
        heldout_latent_post=heldout_latent_post.astype(np.float32),
        heldout_topic=heldout_topic.astype(np.int16),
        heldout_emotion=np.array(heldout_emotions, dtype=object),
        heldout_genre=heldout_genre,
        surviving_indices=surviving_indices.astype(np.int16),
        n_initial_clusters=N_INITIAL_CLUSTERS,
        n_surviving_clusters=N_SURVIVING_CLUSTERS,
        seed=SEED,
        train_silhouette_pre=train_silhouette_pre,
        train_silhouette_post=train_silhouette_post,
        topic_sizes=topic_sizes,
    )
    log(f"Wrote embedding snapshot to {NPZ_PATH}.")
    elapsed_seconds = time.perf_counter() - start_time
    verdict = write_report(
        heldout_metrics, train_silhouette_pre, train_silhouette_post,
        topic_sizes, len(paintings), len(heldout_paintings), stop_reason, epochs,
        elapsed_seconds,
    )
    log(f"Wrote report to {REPORT_PATH}.")
    log(
        f"Final: pre silhouette={train_silhouette_pre:.6f}, "
        f"post silhouette={train_silhouette_post:.6f}, held-out emotion "
        f"AMI={heldout_metrics['emotion']['AMI']:.6f}, genre "
        f"AMI={heldout_metrics['genre']['AMI']:.6f}, verdict={verdict}; "
        f"wall time={elapsed_seconds:.1f}s."
    )


base = load_module("percept_stage1_base_for_anneal", BASE_PILOT_PATH)
cluster_sweep = load_module("percept_stage1_sweep_for_anneal", SWEEP_PILOT_PATH)


if __name__ == "__main__":
    main()
