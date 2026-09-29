"""Instrument the bug-fixed PercepT K=60/40 Stage 1+2 pipeline to save a
snapshot for deep analysis, without changing any of its logic.

Reuses `run_percept_stage2_fixed_pilot.py` (the script that produced the
verified 0.5925 macro AUC number) by importing it as a module and calling
its functions directly, replicating its own `main()` flow exactly but
additionally capturing embeddings/assignments/predictions to an .npz
snapshot, mirroring the field names buddy's own
`attention_h1_embedding_snapshot.npz` uses where sensible. Does not modify
`run_percept_stage2_fixed_pilot.py` or any other existing file.
"""

from pathlib import Path
import sys
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PERCEPT_DIR = HERE.parents[1] / "test/20260922_percept_topic_pipeline"
FIXED_STAGE2_PATH = PERCEPT_DIR / "run_percept_stage2_fixed_pilot.py"
SNAPSHOT_PATH = HERE / "percept_fixed_snapshot.npz"
REPORT_PATH = HERE / "percept_fixed_snapshot_pilot_report.md"
EXPECTED_MACRO_AUC = 0.5925
TOLERANCE = 0.001

for directory in (str(PERCEPT_DIR),):
    if directory not in sys.path:
        sys.path.insert(0, directory)


def load_module(module_name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def genre_lookup(genre_map: dict, paintings: list[str]) -> np.ndarray:
    return np.array([genre_map.get(painting, "") for painting in paintings], dtype=object)


def main() -> None:
    log(f"Loading fixed Stage-2 module from {FIXED_STAGE2_PATH}")
    s2 = load_module("percept_stage2_fixed_for_snapshot", FIXED_STAGE2_PATH)

    np.random.seed(s2.SEED)
    torch.manual_seed(s2.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s2.SEED)

    pipeline = s2.base.load_sibling_module("artelingo_pipeline_snapshot_train", s2.base.PIPELINE_PATH)
    affect_pilot = s2.base.load_sibling_module("artelingo_affect_snapshot", s2.base.AFFECT_PILOT_PATH)
    cca_audit = s2.base.load_sibling_module("artelingo_cca_snapshot", s2.base.CCA_AUDIT_PATH)
    heldout_pipeline = s2.base.load_sibling_module("artelingo_pipeline_snapshot_heldout", s2.base.PIPELINE_PATH)
    heldout_pipeline.STORAGE_DIR = s2.base.HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = s2.base.HELDOUT_JSON
    pipeline.assert_extraction_complete()
    heldout_pipeline.assert_extraction_complete()
    log_fn = pipeline.log

    log_fn("Loading and deduplicating train CLIP features...")
    paintings, img_nodes, txt_nodes, emotion_counts = pipeline.load_dedup_features()
    train_emotions = [pipeline.majority(counts) for counts in emotion_counts]
    log_fn("Loading and deduplicating held-out CLIP features...")
    heldout_paintings, heldout_img_nodes, heldout_txt_nodes, heldout_emotion_counts = (
        heldout_pipeline.load_dedup_features()
    )
    heldout_emotions = [heldout_pipeline.majority(counts) for counts in heldout_emotion_counts]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_fn(f"Using {device}.")
    affect_train = s2.base.extract_affect_embedding_nodes(pipeline.TRAIN_JSON, paintings, device, log_fn)
    affect_heldout = s2.base.extract_affect_embedding_nodes(s2.base.HELDOUT_JSON, heldout_paintings, device, log_fn)
    train_h = s2.base.fused_embeddings(img_nodes, txt_nodes, affect_train, cca_audit, affect_pilot)
    heldout_h = s2.base.fused_embeddings(heldout_img_nodes, heldout_txt_nodes, affect_heldout, cca_audit, affect_pilot)

    train_inputs = torch.from_numpy(train_h)
    encoder, decoder = s2.base.build_autoencoder(train_h.shape[1])
    encoder.to(device)
    decoder.to(device)
    s2.base.pretrain_autoencoder(encoder, decoder, train_inputs, device, log_fn)
    centers = s2.sweep.initialize_cluster_centers(encoder, train_inputs, device, s2.N_INITIAL_CLUSTERS, s2.SEED)
    s2.train_dec_until_stable_fixed(encoder, decoder, centers, train_inputs, device, log_fn, s2.N_INITIAL_CLUSTERS)
    surviving_centers, surviving_indices = s2.prune_centers_fixed(centers, s2.N_SURVIVING_CLUSTERS)
    log_fn(f"Pruned to {len(surviving_indices)} surviving centers.")

    _, heldout_refit_metrics, _ = s2.sweep.evaluate_run(
        encoder, centers, train_inputs, heldout_h, pipeline, heldout_pipeline,
        paintings, train_emotions, heldout_paintings, heldout_emotions, device,
        s2.N_SURVIVING_CLUSTERS,
    )
    log_fn(
        f"Stage-1 re-fit: held-out emotion AMI={heldout_refit_metrics['emotion']['AMI']:.4f}, "
        f"genre AMI={heldout_refit_metrics['genre']['AMI']:.4f}."
    )

    encoder.eval()
    with torch.no_grad():
        train_latent = encoder(train_inputs.to(device))
        train_topic = s2.base.soft_assignments(train_latent, surviving_centers).argmax(dim=1).cpu().numpy()
        heldout_latent = encoder(torch.from_numpy(heldout_h).to(device))
        heldout_topic = s2.base.soft_assignments(heldout_latent, surviving_centers).argmax(dim=1).cpu().numpy()

    train_targets = s2.multi_hot_targets(encoder, surviving_centers, train_inputs, device)
    heldout_targets = s2.multi_hot_targets(encoder, surviving_centers, torch.from_numpy(heldout_h), device)

    train_patch_features = s2.load_patch_features(s2.TRAIN_PATCH_FEATURE_PATH, len(paintings), "train")
    heldout_patch_features = s2.load_patch_features(s2.HELDOUT_PATCH_FEATURE_PATH, len(heldout_paintings), "held-out")
    mapper = s2.AttentionPoolingMapper().to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=s2.MAPPER_LEARNING_RATE)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    train_patch_features_dev = train_patch_features.to(device)
    train_targets_dev = train_targets.to(device)
    mapper.train()
    for epoch in range(1, s2.MAPPER_EPOCHS + 1):
        loss = loss_fn(mapper(train_patch_features_dev), train_targets_dev)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            log_fn(f"Stage-2 epoch {epoch:03d}/{s2.MAPPER_EPOCHS}: BCE={loss.item():.6f}")

    mapper.eval()
    with torch.no_grad():
        heldout_scores = torch.sigmoid(mapper(heldout_patch_features.to(device))).cpu().numpy()
    heldout_targets_np = heldout_targets.cpu().numpy()
    model_aucs, skipped_topics = s2.evaluate_auc(heldout_scores, heldout_targets_np, log_fn, "patch-attention mapper")
    macro_auc = s2.auc_summary(model_aucs)["macro"]
    log_fn(f"Re-derived macro AUC: {macro_auc:.4f} (expected {EXPECTED_MACRO_AUC:.4f} +/- {TOLERANCE})")

    sanity_ok = abs(macro_auc - EXPECTED_MACRO_AUC) <= TOLERANCE
    if not sanity_ok:
        REPORT_PATH.write_text(
            f"# PercepT fixed-snapshot sanity check FAILED\n\n"
            f"Re-derived macro AUC {macro_auc:.4f} does not match the expected "
            f"{EXPECTED_MACRO_AUC:.4f} within tolerance {TOLERANCE}. Stopping "
            "before saving a snapshot or proceeding to deep analysis.\n",
            encoding="utf-8",
        )
        log_fn(f"SANITY CHECK FAILED. Wrote {REPORT_PATH}. Not saving snapshot.")
        raise SystemExit(1)

    genre_map = pipeline.load_genre_map()
    train_genre = genre_lookup(genre_map, paintings)
    heldout_genre = genre_lookup(genre_map, heldout_paintings)

    per_topic_auc_array = np.full(s2.N_SURVIVING_CLUSTERS, np.nan, dtype=np.float64)
    for topic, auc in model_aucs.items():
        per_topic_auc_array[topic] = auc

    np.savez(
        SNAPSHOT_PATH,
        train_paintings=np.array(paintings, dtype=object),
        train_embedding=train_latent.detach().cpu().numpy().astype(np.float32),
        train_topic=train_topic.astype(np.int32),
        train_emotion=np.array(train_emotions, dtype=object),
        train_genre=train_genre,
        heldout_paintings=np.array(heldout_paintings, dtype=object),
        heldout_embedding=heldout_latent.detach().cpu().numpy().astype(np.float32),
        heldout_topic=heldout_topic.astype(np.int32),
        heldout_emotion=np.array(heldout_emotions, dtype=object),
        heldout_genre=heldout_genre,
        heldout_stage2_scores=heldout_scores.astype(np.float32),
        heldout_stage2_targets=heldout_targets_np.astype(np.float32),
        per_topic_auc=per_topic_auc_array,
        skipped_topics=np.array(skipped_topics, dtype=np.int64),
        seed=np.int64(s2.SEED),
    )
    log_fn(f"Wrote snapshot to {SNAPSHOT_PATH}")

    REPORT_PATH.write_text(
        "# PercepT fixed-snapshot instrumentation pass\n\n"
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n"
        f"Sanity check passed: re-derived macro AUC {macro_auc:.4f} matches the "
        f"expected {EXPECTED_MACRO_AUC:.4f} within {TOLERANCE}.\n\n"
        f"Held-out emotion AMI {heldout_refit_metrics['emotion']['AMI']:.4f}, "
        f"genre AMI {heldout_refit_metrics['genre']['AMI']:.4f}. "
        f"Surviving centers: {len(surviving_indices)}.\n\n"
        f"Snapshot written to `{SNAPSHOT_PATH.name}` with fields: "
        "train_paintings, train_embedding, train_topic, train_emotion, "
        "train_genre, heldout_paintings, heldout_embedding, heldout_topic, "
        "heldout_emotion, heldout_genre, heldout_stage2_scores, "
        "heldout_stage2_targets, per_topic_auc, skipped_topics, seed.\n",
        encoding="utf-8",
    )
    log_fn(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
