"""Measure the PercepT Stage-1 seed-42 noise floor under default GPU kernels.

Two determinism-patched runs of the Stage-2 extended-seed pilot showed the
Stage-1 K=60/40 seed-42 refit is fully self-consistent once cuDNN/cuBLAS are
forced deterministic, but that deterministic fixed point (0.1225/0.2274)
differs from the original non-deterministic citation (0.1238/0.2617) by
0.0013 emotion / 0.0343 genre AMI. That genre gap (0.0343) is already larger
than the between-*different*-seed genre AMI standard deviation measured by
the 14-seed Stage-1 extended-seed pilot (0.0156) -- meaning uncontrolled
run-to-run GPU nondeterminism could be as large as, or larger than, the
"seed sensitivity" that pilot reported.

This script isolates that question directly: it repeats ONLY the shared
Stage-1 K=60/40 seed-42 refit, deliberately under PyTorch's DEFAULT
(non-deterministic) cuDNN/cuBLAS kernels -- no determinism forcing, matching
every prior "established" citation in this investigation. No Stage-2 mapper
training happens here.

Run this script N times as N SEPARATE process invocations (a shell loop, not
an in-process loop -- cuDNN's algorithm cache/selection can persist within
one process and would understate real cross-launch variance). Each
invocation appends one {"repeat", "emotion_ami", "genre_ami", "timestamp"}
line to a shared JSONL file. After all N repeats, run
aggregate_percept_stage1_noise_floor_pilot.py once to produce the report.
"""

import json
import os
import time

import numpy as np
import torch

from run_percept_stage2_sweep_pilot import (
    BASE_PILOT_PATH,
    N_INITIAL_CLUSTERS,
    N_SURVIVING_CLUSTERS,
    SWEEP_PILOT_PATH,
    load_module,
)


OUT_DIR = os.path.dirname(__file__)
REPORT_OUT_DIR = os.environ.get("PERCEPT_OUTPUT_ROOT") or OUT_DIR
RESULTS_PATH = os.path.join(
    REPORT_OUT_DIR, "percept_stage1_noise_floor_results.jsonl"
)
SEED = 42
REPEAT_INDEX = int(os.environ.get("PERCEPT_NOISE_FLOOR_REPEAT", "0"))


def main() -> None:
    """Run exactly one fresh, non-deterministic Stage-1 K=60/40 seed-42 refit."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    # Deliberately no cudnn.deterministic / use_deterministic_algorithms here:
    # this pilot exists to measure the noise those settings would suppress.

    base = load_module("percept_stage1_base_for_noise_floor", BASE_PILOT_PATH)
    cluster_sweep = load_module(
        "percept_stage1_sweep_for_noise_floor", SWEEP_PILOT_PATH
    )
    pipeline = base.load_sibling_module(
        "artelingo_run_pipeline_percept_noise_floor_train", base.PIPELINE_PATH
    )
    affect_pilot = base.load_sibling_module(
        "artelingo_run_affect_pilot_percept_noise_floor", base.AFFECT_PILOT_PATH
    )
    cca_audit = base.load_sibling_module(
        "artelingo_run_cca_audit_percept_noise_floor", base.CCA_AUDIT_PATH
    )
    heldout_pipeline = base.load_sibling_module(
        "artelingo_run_pipeline_percept_noise_floor_heldout", base.PIPELINE_PATH
    )
    heldout_pipeline.STORAGE_DIR = base.HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = base.HELDOUT_JSON
    pipeline.assert_extraction_complete()
    heldout_pipeline.assert_extraction_complete()
    log = pipeline.log

    log(f"[repeat {REPEAT_INDEX}] Loading and deduplicating train CLIP features...")
    paintings, img_nodes, txt_nodes, emotion_counts = pipeline.load_dedup_features()
    train_emotions = [pipeline.majority(counts) for counts in emotion_counts]
    log(f"[repeat {REPEAT_INDEX}] Loading and deduplicating held-out CLIP features...")
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
    log(f"[repeat {REPEAT_INDEX}] Using {device} for this noise-floor refit.")
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
    encoder, decoder = base.build_autoencoder(train_h.shape[1])
    encoder.to(device)
    decoder.to(device)
    base.pretrain_autoencoder(encoder, decoder, train_inputs, device, log)
    centers = cluster_sweep.initialize_cluster_centers(
        encoder, train_inputs, device, N_INITIAL_CLUSTERS, SEED
    )
    cluster_sweep.train_dec_until_stable(
        encoder, decoder, centers, train_inputs, device, log, N_INITIAL_CLUSTERS
    )
    surviving_centers, surviving_indices = cluster_sweep.prune_centers(
        centers, N_SURVIVING_CLUSTERS
    )
    log(
        f"[repeat {REPEAT_INDEX}] Pruned {N_INITIAL_CLUSTERS - N_SURVIVING_CLUSTERS} "
        f"centers; retained original indices {surviving_indices.tolist()}."
    )
    _, heldout_metrics, _ = cluster_sweep.evaluate_run(
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
    emotion_ami = heldout_metrics["emotion"]["AMI"]
    genre_ami = heldout_metrics["genre"]["AMI"]
    log(
        f"[repeat {REPEAT_INDEX}] held-out emotion AMI={emotion_ami:.4f}, "
        f"genre AMI={genre_ami:.4f}."
    )

    os.makedirs(REPORT_OUT_DIR, exist_ok=True)
    with open(RESULTS_PATH, "a") as results_file:
        results_file.write(
            json.dumps(
                {
                    "repeat": REPEAT_INDEX,
                    "emotion_ami": emotion_ami,
                    "genre_ami": genre_ami,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            + "\n"
        )
    log(f"[repeat {REPEAT_INDEX}] Appended result to {RESULTS_PATH}.")


if __name__ == "__main__":
    main()
