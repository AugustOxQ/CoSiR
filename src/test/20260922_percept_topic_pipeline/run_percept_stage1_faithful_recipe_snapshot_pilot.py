"""Re-run faithful-recipe Variant A at seed 42 and save its final arrays.

Training and evaluation are delegated to the original faithful-recipe helpers;
only the single-variant run body is repeated to expose its arrays.
"""

import copy
import importlib.util
import os

import numpy as np
import torch


OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FAITHFUL_PATH = os.path.join(OUT_DIR, "run_percept_stage1_faithful_recipe_pilot.py")
NPZ_PATH = os.path.join(OUT_DIR, "percept_stage1_faithful_recipe_snapshot.npz")
BUDDY_SNAPSHOT_PATH = os.path.abspath(os.path.join(
    OUT_DIR, "..", "20260923_artelingo_buddy_analysis",
    "attention_h1_embedding_snapshot.npz",
))


def load_module(module_name, path):
    """Use the same sibling-script import convention as the original pilot."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


faithful = load_module("percept_stage1_faithful_recipe_for_snapshot", FAITHFUL_PATH)
base = faithful.base
SEED = faithful.SEED
VARIANTS = faithful.VARIANTS
N_INITIAL_CLUSTERS = faithful.N_INITIAL_CLUSTERS
N_SURVIVING_CLUSTERS = faithful.N_SURVIVING_CLUSTERS
EMOTION_PARETO_BAR = faithful.EMOTION_PARETO_BAR
GENRE_PARETO_BAR = faithful.GENRE_PARETO_BAR
pretrain_autoencoder = faithful.pretrain_autoencoder
initialize_cluster_centers = faithful.initialize_cluster_centers
train_dec_until_stable = faithful.train_dec_until_stable
evaluate_assignments = faithful.evaluate_assignments


def assert_matching_paintings(actual, expected, split, expected_count):
    """Require a unique, identical painting population before saving."""
    actual_set = set(np.asarray(actual).tolist())
    expected_set = set(np.asarray(expected).tolist())
    if len(actual) != expected_count or len(expected) != expected_count:
        raise RuntimeError(
            f"{split} painting count mismatch: PercepT={len(actual)}, "
            f"buddy={len(expected)}, expected={expected_count}."
        )
    if len(actual_set) != expected_count or len(expected_set) != expected_count:
        raise RuntimeError(f"{split} painting ids are not unique in both sources.")
    if actual_set != expected_set:
        raise RuntimeError(
            f"{split} painting sets differ: "
            f"PercepT-only={len(actual_set - expected_set)}, "
            f"buddy-only={len(expected_set - actual_set)}."
        )


def run_seed_with_snapshot(seed, train_h, heldout_h, train_inputs, device, log,
                           pipeline, paintings, train_emotions, heldout_pipeline,
                           heldout_paintings, heldout_emotions):
    """Duplicate run_seed's Variant-A path, returning its post-prune arrays."""
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

    variant = "A"
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
    # Preserve run_seed's full-split evaluation, including its silhouette.
    train_metrics = evaluate_assignments(
        pipeline, paintings, train_emotions,
        train_latent.cpu().numpy(), train_assignments,
    )
    heldout_metrics = evaluate_assignments(
        heldout_pipeline, heldout_paintings, heldout_emotions,
        heldout_latent.cpu().numpy(), heldout_assignments,
    )
    result = {
        "seed": seed,
        "variant": variant,
        "pretrain_losses": pretrain_losses,
        "dec_losses": dec_losses,
        "trajectory": trajectory,
        "stop_reason": stop_reason,
        "stop_epoch": stop_epoch,
        "train_metrics": train_metrics,
        "heldout_metrics": heldout_metrics,
        "train_latent": train_latent.cpu().numpy(),
        "train_assignments": train_assignments,
        "heldout_latent": heldout_latent.cpu().numpy(),
        "heldout_assignments": heldout_assignments,
        "surviving_indices": surviving_indices,
    }
    del encoder, decoder, centers
    return result


def main():
    """Load the original features, run Variant A once, and save its snapshot."""
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
    with np.load(BUDDY_SNAPSHOT_PATH, allow_pickle=True) as buddy:
        assert_matching_paintings(paintings, buddy["train_paintings"], "train", 61402)
        assert_matching_paintings(
            heldout_paintings, buddy["heldout_paintings"], "held-out", 9365
        )
    log("Painting-set identity confirmed: 61,402 train and 9,365 held-out.")
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
    result = run_seed_with_snapshot(
        SEED, train_h, heldout_h, train_inputs, device, log, pipeline, paintings,
        train_emotions, heldout_pipeline, heldout_paintings, heldout_emotions,
    )
    metrics = result["heldout_metrics"]
    emotion_ami = metrics["emotion"]["AMI"]
    genre_ami = metrics["genre"]["AMI"]
    silhouette = metrics["silhouette"]
    log(
        f"Variant A seed {SEED} held-out: emotion AMI={emotion_ami:.4f}, "
        f"genre AMI={genre_ami:.4f}, silhouette={silhouette:.4f}."
    )
    published = {"emotion AMI": (emotion_ami, 0.1092),
                 "genre AMI": (genre_ami, 0.3288),
                 "silhouette": (silhouette, 0.5120)}
    for name, (measured, cited) in published.items():
        if not np.isfinite(measured) or abs(measured - cited) > 0.00005:
            log(
                f"WARNING: Variant A {name}={measured:.8f} differs from "
                f"published {cited:.4f} by more than rounding tolerance; "
                "inspect before trusting this snapshot."
            )
    genre_map = pipeline.load_genre_map()
    np.savez_compressed(
        NPZ_PATH,
        train_paintings=np.array(paintings, dtype=object),
        train_latent=result["train_latent"],
        train_topic=result["train_assignments"],
        train_emotion=np.array(train_emotions, dtype=object),
        train_genre=np.array([genre_map.get(p, "") for p in paintings], dtype=object),
        heldout_paintings=np.array(heldout_paintings, dtype=object),
        heldout_latent=result["heldout_latent"],
        heldout_topic=result["heldout_assignments"],
        heldout_emotion=np.array(heldout_emotions, dtype=object),
        heldout_genre=np.array(
            [genre_map.get(p, "") for p in heldout_paintings], dtype=object
        ),
        surviving_indices=result["surviving_indices"],
        n_initial_clusters=N_INITIAL_CLUSTERS,
        n_surviving_clusters=N_SURVIVING_CLUSTERS,
        seed=SEED,
        held_out_silhouette=silhouette,
        stop_epoch=result["stop_epoch"],
        stop_reason=result["stop_reason"],
    )
    log(f"Wrote faithful-recipe Variant-A snapshot to {NPZ_PATH}.")


if __name__ == "__main__":
    main()
