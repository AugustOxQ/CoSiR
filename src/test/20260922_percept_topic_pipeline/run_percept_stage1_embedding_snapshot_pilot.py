"""Snapshot PercepT Stage 1 K=60/40 embeddings before/after DEC, with labels.

This read-only GPU diagnostic re-fits the standing deterministic K=60/40
seed-42 Stage-1 configuration -- the exact recipe and forced-deterministic
cuDNN/cuBLAS settings used in `run_percept_stage2_extended_seed_pilot.py`,
rebased onto that pipeline's own observed fixed point (0.1225 emotion AMI /
0.2274 genre AMI on held-out; see that module's docstring for why the
original 0.1238/0.2617 citation is not the reproduction target under forced
determinism) -- then dumps the encoder's 128-D latent space immediately
after autoencoder pretraining ("pre-DEC") and again after DEC converges
("post-DEC"), for both train and held-out paintings, together with each
painting's majority emotion label and (where available) genre label.

No new training objective or Stage-2 mapper is introduced here; this is
purely additional read-only instrumentation on top of the existing
reproducible fit, intended to feed a separate, GPU-free plotting script that
builds a before/after embedding-space figure and a per-cluster genre/emotion
composition figure.
"""

import importlib.util
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


OUT_DIR = os.path.dirname(__file__)
REPORT_OUT_DIR = os.environ.get("PERCEPT_OUTPUT_ROOT") or OUT_DIR
BASE_PILOT_PATH = os.path.join(OUT_DIR, "run_percept_stage1_pilot.py")
SWEEP_PILOT_PATH = os.path.join(
    OUT_DIR, "run_percept_stage1_cluster_count_sweep_pilot.py"
)
REPORT_PATH = os.path.join(
    REPORT_OUT_DIR, "percept_stage1_embedding_snapshot_pilot_report.md"
)
NPZ_PATH = os.path.join(REPORT_OUT_DIR, "percept_stage1_embedding_snapshot.npz")

N_INITIAL_CLUSTERS = 60
N_SURVIVING_CLUSTERS = 40
SEED = 42
# Rebased onto the determinism-patched pipeline's own observed fixed point
# (see run_percept_stage2_extended_seed_pilot.py's module docstring), not the
# original non-deterministic citation of 0.1238/0.2617.
EXPECTED_HELDOUT_EMOTION_AMI = 0.1225
EXPECTED_HELDOUT_GENRE_AMI = 0.2274
REPRODUCTION_TOLERANCE = 0.002
NEAREST_CENTER_CHUNK = 4096


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


def write_failure_report(refit_metrics: dict) -> None:
    emotion_ami = refit_metrics["emotion"]["AMI"]
    genre_ami = refit_metrics["genre"]["AMI"]
    lines = [
        "# PercepT Stage 1 embedding snapshot pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Stage-1 K=60/40 seed-42 reproduction\n\n",
        "| metric | expected (rebased fixed point) | re-fit | absolute difference | status |\n",
        "|---|---:|---:|---:|---|\n",
        f"| held-out emotion AMI | {EXPECTED_HELDOUT_EMOTION_AMI:.4f} | "
        f"{emotion_ami:.4f} | "
        f"{abs(emotion_ami - EXPECTED_HELDOUT_EMOTION_AMI):.4f} | FAILED |\n",
        f"| held-out genre AMI | {EXPECTED_HELDOUT_GENRE_AMI:.4f} | "
        f"{genre_ami:.4f} | "
        f"{abs(genre_ami - EXPECTED_HELDOUT_GENRE_AMI):.4f} | FAILED |\n\n",
        "**Reproducibility failure.** No embedding snapshot was written.\n",
    ]
    os.makedirs(REPORT_OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def write_success_report(
    refit_metrics: dict, shapes: dict[str, int], genre_counts: dict[str, int]
) -> None:
    emotion_ami = refit_metrics["emotion"]["AMI"]
    genre_ami = refit_metrics["genre"]["AMI"]
    lines = [
        "# PercepT Stage 1 embedding snapshot pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "This read-only diagnostic re-fits the standing deterministic K=60/40 "
        "seed-42 Stage-1 configuration and dumps the encoder's 128-D latent "
        "space immediately after autoencoder pretraining (\"pre-DEC\") and "
        "again after DEC converges (\"post-DEC\"), for both train and "
        "held-out paintings, together with each painting's majority emotion "
        "label and (where available) genre label. No new training objective "
        "or Stage-2 mapper is introduced; this is purely additional "
        "instrumentation on top of the existing reproducible fit.\n\n",
        "## Stage-1 K=60/40 seed-42 reproduction\n\n",
        "| metric | expected (rebased fixed point) | re-fit | absolute difference | status |\n",
        "|---|---:|---:|---:|---|\n",
        f"| held-out emotion AMI | {EXPECTED_HELDOUT_EMOTION_AMI:.4f} | "
        f"{emotion_ami:.4f} | "
        f"{abs(emotion_ami - EXPECTED_HELDOUT_EMOTION_AMI):.4f} | reproduced |\n",
        f"| held-out genre AMI | {EXPECTED_HELDOUT_GENRE_AMI:.4f} | "
        f"{genre_ami:.4f} | "
        f"{abs(genre_ami - EXPECTED_HELDOUT_GENRE_AMI):.4f} | reproduced |\n\n",
        "## Snapshot contents\n\n",
        f"Written to `{os.path.basename(NPZ_PATH)}`.\n\n",
        "| array | shape | meaning |\n",
        "|---|---|---|\n",
        f"| train_paintings | ({shapes['train_n']:,},) | painting ids, train split |\n",
        f"| train_latent_pre | ({shapes['train_n']:,}, 128) | encoder output right after autoencoder pretrain, before any DEC step |\n",
        f"| train_latent_post | ({shapes['train_n']:,}, 128) | encoder output after DEC converged |\n",
        f"| train_initial_label | ({shapes['train_n']:,},) | nearest of the 60 initial K-means centers, evaluated on train_latent_pre |\n",
        f"| train_topic | ({shapes['train_n']:,},) | primary (argmax) of the 40 surviving DEC topics, evaluated on train_latent_post |\n",
        f"| train_emotion | ({shapes['train_n']:,},) | majority caption emotion label (full coverage) |\n",
        f"| train_genre | ({shapes['train_n']:,},) | genre label, or \"\" where unavailable "
        f"({genre_counts['train']:,}/{shapes['train_n']:,} covered) |\n",
        f"| heldout_paintings | ({shapes['heldout_n']:,},) | painting ids, held-out (val+test) split |\n",
        f"| heldout_latent_pre | ({shapes['heldout_n']:,}, 128) | same pre-DEC encoder applied out-of-sample |\n",
        f"| heldout_latent_post | ({shapes['heldout_n']:,}, 128) | same post-DEC encoder applied out-of-sample |\n",
        f"| heldout_topic | ({shapes['heldout_n']:,},) | primary of the 40 surviving DEC topics, evaluated on heldout_latent_post |\n",
        f"| heldout_emotion | ({shapes['heldout_n']:,},) | majority caption emotion label (full coverage) |\n",
        f"| heldout_genre | ({shapes['heldout_n']:,},) | genre label, or \"\" where unavailable "
        f"({genre_counts['heldout']:,}/{shapes['heldout_n']:,} covered) |\n",
        "| surviving_indices | (40,) | which of the 60 initial cluster indices survived pruning |\n\n",
        "Genre coverage is sparse by construction (ArtELingo's genre "
        "annotation does not cover every painting): a per-cluster genre "
        "figure built from this snapshot will have much smaller per-cluster "
        "counts than the emotion figure and should be read accordingly.\n",
    ]
    os.makedirs(REPORT_OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    """Re-fit K=60/40 seed 42 once, then dump the before/after snapshot."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    base = load_module("percept_stage1_base_for_snapshot", BASE_PILOT_PATH)
    cluster_sweep = load_module(
        "percept_stage1_sweep_for_snapshot", SWEEP_PILOT_PATH
    )
    pipeline = base.load_sibling_module(
        "artelingo_run_pipeline_percept_snapshot_train", base.PIPELINE_PATH
    )
    affect_pilot = base.load_sibling_module(
        "artelingo_run_affect_pilot_percept_snapshot", base.AFFECT_PILOT_PATH
    )
    cca_audit = base.load_sibling_module(
        "artelingo_run_cca_audit_percept_snapshot", base.CCA_AUDIT_PATH
    )
    heldout_pipeline = base.load_sibling_module(
        "artelingo_run_pipeline_percept_snapshot_heldout", base.PIPELINE_PATH
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
        heldout_paintings,
        heldout_img_nodes,
        heldout_txt_nodes,
        heldout_emotion_counts,
    ) = heldout_pipeline.load_dedup_features()
    heldout_emotions = [
        heldout_pipeline.majority(counts) for counts in heldout_emotion_counts
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {device} for the shared Stage-1 re-fit and embedding snapshot.")
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
    log("Captured pre-DEC (pretrain-only) encoder latents for train and held-out.")

    centers = cluster_sweep.initialize_cluster_centers(
        encoder, train_inputs, device, N_INITIAL_CLUSTERS, SEED
    )
    initial_centers_snapshot = centers.detach().clone().cpu().numpy()
    train_initial_label = nearest_center_labels(train_latent_pre, initial_centers_snapshot)

    cluster_sweep.train_dec_until_stable(
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
            )
            .argmax(dim=1)
            .cpu()
            .numpy()
        )
        heldout_topic = (
            base.soft_assignments(
                torch.from_numpy(heldout_latent_post).to(device), surviving_centers
            )
            .argmax(dim=1)
            .cpu()
            .numpy()
        )
    log("Captured post-DEC encoder latents and surviving-topic assignments.")

    _, heldout_refit_metrics, _ = cluster_sweep.evaluate_run(
        encoder,
        centers,
        train_inputs,
        heldout_h,
        pipeline,
        heldout_pipeline,
        paintings,
        train_emotions,
        heldout_paintings,
        heldout_emotions,
        device,
        N_SURVIVING_CLUSTERS,
    )
    reproduced = (
        abs(heldout_refit_metrics["emotion"]["AMI"] - EXPECTED_HELDOUT_EMOTION_AMI)
        < REPRODUCTION_TOLERANCE
        and abs(heldout_refit_metrics["genre"]["AMI"] - EXPECTED_HELDOUT_GENRE_AMI)
        < REPRODUCTION_TOLERANCE
    )
    if not reproduced:
        write_failure_report(heldout_refit_metrics)
        log(f"Stage-1 reproducibility failure; wrote report to {REPORT_PATH}.")
        return

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
    )
    log(f"Wrote embedding snapshot to {NPZ_PATH}.")

    shapes = {"train_n": len(paintings), "heldout_n": len(heldout_paintings)}
    genre_counts = {
        "train": int(sum(1 for value in train_genre if value)),
        "heldout": int(sum(1 for value in heldout_genre if value)),
    }
    write_success_report(heldout_refit_metrics, shapes, genre_counts)
    log(f"Wrote report to {REPORT_PATH}.")


if __name__ == "__main__":
    main()
