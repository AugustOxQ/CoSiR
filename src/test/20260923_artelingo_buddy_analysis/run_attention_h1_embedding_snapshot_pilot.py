"""Snapshot the buddy-graph Attention-h1 learned-student embedding before/after training.

This is the "our own method" counterpart to the PercepT Stage-1 embedding
snapshot pilot: it re-fits the standing Attention-h1 two-teacher contrastive
student (the architecture-sweep winner in
`run_learned_student_arch_sweep_pilot.py`, held-out emotion AMI=0.1249, genre
AMI=0.2404) exactly as that script does for the "attn1" configuration, and
dumps the 32-D joint embedding both at initialization (epoch 0, before any
InfoNCE training step) and after training converges, for both train and
held-out paintings, together with a genuine Leiden community assignment at
BOTH points and each painting's emotion/genre label.

The original architecture-sweep script only ever runs Leiden once, on the
FINAL trained embedding -- there is no "before" cluster assignment to compare
against. This script adds that missing epoch-0 Leiden pass (using the exact
same graph-building and community-detection helpers the original script uses
for its final evaluation), so a downstream plotting script can build a
before/after figure in the same style as the PercepT one, for this project's
own fusion method rather than the replicated DEC-based PercepT method.

Read-only in spirit: no change to the established architecture-sweep script,
no new training objective, and this pilot exists purely for visualization
and reporting. It does duplicate real GPU training work (buddy-graph
Attention-h1 is genuinely retrained here, from the fixed seed 42, no cached
checkpoint exists), since a live model is needed to also compute the epoch-0
Leiden pass that was never previously computed or saved.
"""

import importlib.util
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.decomposition import PCA


OUT_DIR = os.path.dirname(os.path.abspath(__file__))
# `run_learned_student_arch_sweep_pilot.py` (loaded below as a sibling module)
# resolves its own `src.conditional_buddy.prototype_seed` import via
# `if REPO_ROOT not in sys.path: sys.path.insert(0, REPO_ROOT)`. On node404,
# `/local/wding/CoSiR` is already present in `sys.path` via this conda env's
# PYTHONPATH, positioned AFTER site-packages -- so that guard is truthy and
# skips the insert, and that later-position entry then fails to resolve
# `src.conditional_buddy` as a script-mode `__main__` (confirmed empirically:
# an unconditional insert at position 0 works; the guarded version, and a
# guarded pre-import here, both reproduce the identical failure). Force it to
# position 0 UNCONDITIONALLY here, before loading the sibling module, so the
# resulting cached `sys.modules` entries make its own guarded insert's
# (skipped) no-op irrelevant.
_REPO_ROOT = os.path.abspath(os.path.join(OUT_DIR, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
import src.conditional_buddy.prototype_seed  # noqa: F401  (see comment above)
REPORT_OUT_DIR = os.environ.get("PERCEPT_OUTPUT_ROOT") or OUT_DIR
ARCH_SWEEP_PATH = os.path.join(OUT_DIR, "run_learned_student_arch_sweep_pilot.py")
REPORT_PATH = os.path.join(
    REPORT_OUT_DIR, "attention_h1_embedding_snapshot_pilot_report.md"
)
NPZ_PATH = os.path.join(REPORT_OUT_DIR, "attention_h1_embedding_snapshot.npz")

SEED = 42
# Citation from learned_student_arch_sweep_pilot_report.md (Attention-h1 row).
CITED_TRAIN_EMOTION_AMI = 0.1351
CITED_TRAIN_GENRE_AMI = 0.2397
CITED_HELDOUT_EMOTION_AMI = 0.1249
CITED_HELDOUT_GENRE_AMI = 0.2404


def load_module(module_name: str, path: str):
    """Load a standalone sibling script without running its main block."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def community_and_metrics(
    arch,
    embeddings_np: np.ndarray,
    label: str,
    pipeline_module,
    affect_pilot,
    device: str,
    expected_nodes: int,
    single_modality,
):
    """Build a mutual-kNN graph on `embeddings_np` and run Leiden on it."""
    graph = single_modality.build_single_modality_graph(
        label, embeddings_np, pipeline_module, affect_pilot, device,
        expected_nodes=expected_nodes,
    )
    communities = arch.detect_communities(graph, seed=SEED)
    return communities


def write_report(
    train_metrics_pre: dict,
    train_metrics_post: dict,
    heldout_metrics_pre: dict,
    heldout_metrics_post: dict,
    shapes: dict,
    genre_counts: dict,
) -> None:
    lines = [
        "# Attention-h1 embedding snapshot pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "This is the buddy-graph \"our own method\" counterpart to the PercepT "
        "Stage-1 embedding snapshot pilot. It re-fits the standing Attention-h1 "
        "two-teacher contrastive student and dumps its 32-D joint embedding at "
        "initialization (epoch 0, before any InfoNCE step) and after training "
        "converges, with a genuine Leiden community assignment computed at "
        "both points (the original architecture-sweep script only ever runs "
        "Leiden once, on the final embedding).\n\n",
        "## Reproduction check against the cited Attention-h1 result\n\n",
        "| split | metric | cited | this re-fit | absolute difference |\n",
        "|---|---|---:|---:|---:|\n",
        f"| train | emotion AMI | {CITED_TRAIN_EMOTION_AMI:.4f} | "
        f"{train_metrics_post['emotion']['AMI']:.4f} | "
        f"{abs(train_metrics_post['emotion']['AMI'] - CITED_TRAIN_EMOTION_AMI):.4f} |\n",
        f"| train | genre AMI | {CITED_TRAIN_GENRE_AMI:.4f} | "
        f"{train_metrics_post['genre']['AMI']:.4f} | "
        f"{abs(train_metrics_post['genre']['AMI'] - CITED_TRAIN_GENRE_AMI):.4f} |\n",
        f"| held-out | emotion AMI | {CITED_HELDOUT_EMOTION_AMI:.4f} | "
        f"{heldout_metrics_post['emotion']['AMI']:.4f} | "
        f"{abs(heldout_metrics_post['emotion']['AMI'] - CITED_HELDOUT_EMOTION_AMI):.4f} |\n",
        f"| held-out | genre AMI | {CITED_HELDOUT_GENRE_AMI:.4f} | "
        f"{heldout_metrics_post['genre']['AMI']:.4f} | "
        f"{abs(heldout_metrics_post['genre']['AMI'] - CITED_HELDOUT_GENRE_AMI):.4f} |\n\n",
        "This is reported for transparency, not gated: unlike the PercepT branch, "
        "this specific script has no prior determinism investigation, so a small "
        "difference from the citation is expected and does not invalidate the "
        "snapshot below.\n\n",
        "## Epoch-0 (pre-training) diagnostic AMI, newly computed\n\n",
        "No prior run of this architecture ever computed these -- the original "
        "architecture-sweep script only evaluates recall/gate diagnostics at "
        "epoch 0, never Leiden AMI.\n\n",
        "| split | metric | value |\n",
        "|---|---|---:|\n",
        f"| train | emotion AMI | {train_metrics_pre['emotion']['AMI']:.4f} |\n",
        f"| train | genre AMI | {train_metrics_pre['genre']['AMI']:.4f} |\n",
        f"| held-out | emotion AMI | {heldout_metrics_pre['emotion']['AMI']:.4f} |\n",
        f"| held-out | genre AMI | {heldout_metrics_pre['genre']['AMI']:.4f} |\n\n",
        "## Snapshot contents\n\n",
        f"Written to `{os.path.basename(NPZ_PATH)}`.\n\n",
        "| array | shape | meaning |\n",
        "|---|---|---|\n",
        f"| train_paintings | ({shapes['train_n']:,},) | painting ids, train split |\n",
        f"| train_embedding_pre | ({shapes['train_n']:,}, 32) | joint embedding at initialization (epoch 0) |\n",
        f"| train_embedding_post | ({shapes['train_n']:,}, 32) | joint embedding after training converged |\n",
        f"| train_community_pre | ({shapes['train_n']:,},) | Leiden community id on the epoch-0 embedding ({shapes['train_communities_pre']} communities found) |\n",
        f"| train_community_post | ({shapes['train_n']:,},) | Leiden community id on the final embedding ({shapes['train_communities_post']} communities found) |\n",
        f"| train_emotion | ({shapes['train_n']:,},) | majority caption emotion label (full coverage) |\n",
        f"| train_genre | ({shapes['train_n']:,},) | genre label, or \"\" where unavailable ({genre_counts['train']:,}/{shapes['train_n']:,} covered) |\n",
        f"| heldout_paintings | ({shapes['heldout_n']:,},) | painting ids, held-out (val+test) split |\n",
        f"| heldout_embedding_pre | ({shapes['heldout_n']:,}, 32) | same epoch-0 model applied out-of-sample |\n",
        f"| heldout_embedding_post | ({shapes['heldout_n']:,}, 32) | same final model applied out-of-sample |\n",
        f"| heldout_community_pre | ({shapes['heldout_n']:,},) | Leiden community id on the epoch-0 held-out embedding ({shapes['heldout_communities_pre']} communities found) |\n",
        f"| heldout_community_post | ({shapes['heldout_n']:,},) | Leiden community id on the final held-out embedding ({shapes['heldout_communities_post']} communities found) |\n",
        f"| heldout_emotion | ({shapes['heldout_n']:,},) | majority caption emotion label (full coverage) |\n",
        f"| heldout_genre | ({shapes['heldout_n']:,},) | genre label, or \"\" where unavailable ({genre_counts['heldout']:,}/{shapes['heldout_n']:,} covered) |\n\n",
        "Unlike PercepT's fixed K=60/40, Leiden chooses its own community count "
        "each time it runs -- the pre- and post-training community counts above "
        "are not necessarily equal, and that difference is itself a real, "
        "reportable fact about this method rather than a bug.\n",
    ]
    os.makedirs(REPORT_OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    arch = load_module("attention_h1_snapshot_arch_sweep", ARCH_SWEEP_PATH)
    pipeline = arch.load_sibling_module(
        "artelingo_run_pipeline_attn_snapshot", arch.PIPELINE_PATH
    )
    affect_pilot = arch.load_sibling_module(
        "artelingo_run_affect_pilot_attn_snapshot", arch.AFFECT_PILOT_PATH
    )
    single_modality = arch.load_sibling_module(
        "artelingo_run_single_modality_attn_snapshot", arch.SINGLE_MODALITY_PATH
    )
    cca_audit = arch.load_sibling_module(
        "artelingo_run_cca_audit_attn_snapshot", arch.CCA_AUDIT_PATH
    )
    heldout_pipeline = arch.load_sibling_module(
        "artelingo_run_pipeline_attn_snapshot_heldout", arch.PIPELINE_PATH
    )
    heldout_pipeline.STORAGE_DIR = arch.HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = arch.HELDOUT_JSON
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = arch.log
    log(f"Using {device} for the Attention-h1 embedding snapshot.")

    log("Verifying and loading train CLIP features...")
    pipeline.assert_extraction_complete()
    paintings, img_nodes, txt_nodes, emotion_counts = pipeline.load_dedup_features()
    majority_emotion = [pipeline.majority(counts) for counts in emotion_counts]
    log("Extracting train GoEmotions probabilities...")
    affect_train = np.asarray(
        affect_pilot.extract_affect_nodes(pipeline.TRAIN_JSON, paintings, str(device)),
        dtype=np.float64,
    )
    log("Verifying and loading held-out CLIP features...")
    heldout_pipeline.assert_extraction_complete()
    (
        heldout_paintings,
        heldout_img,
        heldout_txt,
        heldout_emotion_counts,
    ) = heldout_pipeline.load_dedup_features()
    heldout_majority_emotion = [
        heldout_pipeline.majority(counts) for counts in heldout_emotion_counts
    ]
    log("Extracting held-out GoEmotions probabilities...")
    affect_heldout = np.asarray(
        affect_pilot.extract_affect_nodes(arch.HELDOUT_JSON, heldout_paintings, str(device)),
        dtype=np.float64,
    )

    content_train = cca_audit.content_features(img_nodes, txt_nodes, affect_pilot)
    content_heldout = cca_audit.content_features(heldout_img, heldout_txt, affect_pilot)
    log(f"Fitting train-only content PCA ({arch.CONTENT_PCA_DIM} components)...")
    pca = PCA(n_components=arch.CONTENT_PCA_DIM, random_state=SEED)
    content_train = pca.fit_transform(content_train).astype(np.float32)
    content_heldout = pca.transform(content_heldout).astype(np.float32)

    log("Building train content and affect teacher graphs...")
    _img_graph, _txt_graph, content_teacher_graph = pipeline.build_buddy_graphs(
        img_nodes, txt_nodes, K=pipeline.K, alpha=pipeline.ALPHA, device=str(device),
        connect_components=True,
    )
    affect_teacher_graph = single_modality.build_single_modality_graph(
        "train-affect-teacher", affect_train, pipeline, affect_pilot, str(device),
        expected_nodes=len(paintings),
    )
    content_edges = arch.upper_triangle_edges(content_teacher_graph)
    affect_edges = arch.upper_triangle_edges(affect_teacher_graph)
    log(f"Teacher edge lists: content={len(content_edges):,}, affect={len(affect_edges):,}.")

    train_content_t = torch.as_tensor(content_train, dtype=torch.float32, device=device)
    train_affect_t = torch.as_tensor(affect_train, dtype=torch.float32, device=device)
    heldout_content_t = torch.as_tensor(content_heldout, dtype=torch.float32, device=device)
    heldout_affect_t = torch.as_tensor(affect_heldout, dtype=torch.float32, device=device)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = arch.LearnedStudent("attn1").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arch.LEARNING_RATE)

    log("Capturing epoch-0 (pre-training) embeddings and running Leiden...")
    model.eval()
    with torch.no_grad():
        train_embedding_pre, _ = model(train_content_t, train_affect_t)
        heldout_embedding_pre, _ = model(heldout_content_t, heldout_affect_t)
    train_embedding_pre_np = train_embedding_pre.cpu().numpy().astype(np.float32, copy=False)
    heldout_embedding_pre_np = heldout_embedding_pre.cpu().numpy().astype(np.float32, copy=False)
    train_community_pre = community_and_metrics(
        arch, train_embedding_pre_np, "epoch0-train-attn1", pipeline, affect_pilot,
        str(device), len(paintings), single_modality,
    )
    heldout_community_pre = community_and_metrics(
        arch, heldout_embedding_pre_np, "epoch0-heldout-attn1", heldout_pipeline,
        affect_pilot, str(device), len(heldout_paintings), single_modality,
    )
    train_metrics_pre = {
        "emotion": pipeline.external_metrics(train_community_pre, majority_emotion),
    }
    genre_map = pipeline.load_genre_map()
    genre_indices = [
        index for index, painting in enumerate(paintings) if painting in genre_map
    ]
    train_metrics_pre["genre"] = pipeline.external_metrics(
        [train_community_pre[index] for index in genre_indices],
        [genre_map[paintings[index]] for index in genre_indices],
    )
    heldout_metrics_pre = {
        "emotion": heldout_pipeline.external_metrics(
            heldout_community_pre, heldout_majority_emotion
        ),
    }
    heldout_genre_indices = [
        index for index, painting in enumerate(heldout_paintings) if painting in genre_map
    ]
    heldout_metrics_pre["genre"] = pipeline.external_metrics(
        [heldout_community_pre[index] for index in heldout_genre_indices],
        [genre_map[heldout_paintings[index]] for index in heldout_genre_indices],
    )
    log(
        f"epoch=0 train emotion AMI={train_metrics_pre['emotion']['AMI']:.4f} "
        f"genre AMI={train_metrics_pre['genre']['AMI']:.4f}; held-out emotion "
        f"AMI={heldout_metrics_pre['emotion']['AMI']:.4f} genre "
        f"AMI={heldout_metrics_pre['genre']['AMI']:.4f}."
    )

    log("Training Attention-h1 to convergence (this reruns the established recipe)...")
    trajectory = [{"epoch": 0, "content_recall": float("nan"), "affect_recall": float("nan")}]
    plateau_count, stop_reason = 0, f"reached MAX_EPOCHS={arch.MAX_EPOCHS}"
    diagnostic_rng = np.random.default_rng(SEED)
    sampled_nodes = diagnostic_rng.choice(
        len(heldout_paintings), size=arch.EDGE_SAMPLE_SIZE, replace=False
    )
    rank_nodes = diagnostic_rng.choice(
        len(heldout_paintings), size=arch.EFFECTIVE_RANK_SAMPLE_SIZE, replace=False
    )
    heldout_content_graph = single_modality.build_single_modality_graph(
        "held-out-content-reference", content_heldout, heldout_pipeline, affect_pilot,
        str(device), expected_nodes=len(heldout_paintings),
    )
    heldout_affect_graph = single_modality.build_single_modality_graph(
        "held-out-affect-reference", affect_heldout, heldout_pipeline, affect_pilot,
        str(device), expected_nodes=len(heldout_paintings),
    )
    epoch_0_diag = arch.evaluate_checkpoint(
        model, heldout_content_t, heldout_affect_t, heldout_content_graph,
        heldout_affect_graph, sampled_nodes, rank_nodes, single_modality,
        heldout_pipeline, affect_pilot, str(device),
    )
    trajectory[0].update(epoch_0_diag)
    for epoch in range(1, arch.MAX_EPOCHS + 1):
        model.train()
        epoch_rng = np.random.default_rng(SEED + epoch)
        content_pairs = arch.sample_positive_pairs(content_edges, epoch_rng)
        affect_pairs = arch.sample_positive_pairs(affect_edges, epoch_rng)
        content_embeddings, remapped_content_pairs = arch.content_batch_embeddings(
            model, train_content_t, train_affect_t, content_pairs, device
        )
        content_loss = arch.symmetric_infonce(content_embeddings, remapped_content_pairs, device)
        affect_embeddings, _mixing_weights = model(train_content_t, train_affect_t)
        affect_loss = arch.symmetric_infonce(affect_embeddings, affect_pairs, device)
        total_loss = content_loss + affect_loss
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        if epoch % arch.CHECKPOINT_EVERY:
            continue
        diagnostics = arch.evaluate_checkpoint(
            model, heldout_content_t, heldout_affect_t, heldout_content_graph,
            heldout_affect_graph, sampled_nodes, rank_nodes, single_modality,
            heldout_pipeline, affect_pilot, str(device),
        )
        checkpoint = {"epoch": epoch, **diagnostics}
        trajectory.append(checkpoint)
        log(
            f"epoch={epoch} content_recall={checkpoint['content_recall']:.4f} "
            f"affect_recall={checkpoint['affect_recall']:.4f}"
        )
        previous = trajectory[-2]
        content_plateau = (
            arch.relative_improvement(checkpoint["content_recall"], previous["content_recall"])
            < arch.PLATEAU_REL_IMPROVEMENT
        )
        affect_plateau = (
            arch.relative_improvement(checkpoint["affect_recall"], previous["affect_recall"])
            < arch.PLATEAU_REL_IMPROVEMENT
        )
        plateau_count = plateau_count + 1 if content_plateau and affect_plateau else 0
        if plateau_count >= arch.PLATEAU_WINDOW:
            stop_reason = f"both recalls plateaued for {arch.PLATEAU_WINDOW} consecutive checkpoints"
            log(f"Stopping: {stop_reason} at epoch {epoch}.")
            break
    log(f"Training stopped: {stop_reason}.")

    log("Capturing final (post-training) embeddings and running Leiden...")
    model.eval()
    with torch.no_grad():
        train_embedding_post, _ = model(train_content_t, train_affect_t)
        heldout_embedding_post, _ = model(heldout_content_t, heldout_affect_t)
    train_embedding_post_np = train_embedding_post.cpu().numpy().astype(np.float32, copy=False)
    heldout_embedding_post_np = heldout_embedding_post.cpu().numpy().astype(np.float32, copy=False)
    train_community_post = community_and_metrics(
        arch, train_embedding_post_np, "final-train-attn1", pipeline, affect_pilot,
        str(device), len(paintings), single_modality,
    )
    heldout_community_post = community_and_metrics(
        arch, heldout_embedding_post_np, "final-heldout-attn1", heldout_pipeline,
        affect_pilot, str(device), len(heldout_paintings), single_modality,
    )
    train_metrics_post = {
        "emotion": pipeline.external_metrics(train_community_post, majority_emotion),
        "genre": pipeline.external_metrics(
            [train_community_post[index] for index in genre_indices],
            [genre_map[paintings[index]] for index in genre_indices],
        ),
    }
    heldout_metrics_post = {
        "emotion": heldout_pipeline.external_metrics(
            heldout_community_post, heldout_majority_emotion
        ),
        "genre": pipeline.external_metrics(
            [heldout_community_post[index] for index in heldout_genre_indices],
            [genre_map[heldout_paintings[index]] for index in heldout_genre_indices],
        ),
    }
    log(
        f"final train emotion AMI={train_metrics_post['emotion']['AMI']:.4f} "
        f"genre AMI={train_metrics_post['genre']['AMI']:.4f}; held-out emotion "
        f"AMI={heldout_metrics_post['emotion']['AMI']:.4f} genre "
        f"AMI={heldout_metrics_post['genre']['AMI']:.4f}."
    )

    train_genre = np.array([genre_map.get(p, "") for p in paintings], dtype=object)
    heldout_genre = np.array([genre_map.get(p, "") for p in heldout_paintings], dtype=object)

    np.savez_compressed(
        NPZ_PATH,
        train_paintings=np.array(paintings, dtype=object),
        train_embedding_pre=train_embedding_pre_np,
        train_embedding_post=train_embedding_post_np,
        train_community_pre=train_community_pre.astype(np.int32),
        train_community_post=train_community_post.astype(np.int32),
        train_emotion=np.array(majority_emotion, dtype=object),
        train_genre=train_genre,
        heldout_paintings=np.array(heldout_paintings, dtype=object),
        heldout_embedding_pre=heldout_embedding_pre_np,
        heldout_embedding_post=heldout_embedding_post_np,
        heldout_community_pre=heldout_community_pre.astype(np.int32),
        heldout_community_post=heldout_community_post.astype(np.int32),
        heldout_emotion=np.array(heldout_majority_emotion, dtype=object),
        heldout_genre=heldout_genre,
        seed=SEED,
    )
    log(f"Wrote embedding snapshot to {NPZ_PATH}.")

    shapes = {
        "train_n": len(paintings),
        "heldout_n": len(heldout_paintings),
        "train_communities_pre": int(train_community_pre.max() + 1),
        "train_communities_post": int(train_community_post.max() + 1),
        "heldout_communities_pre": int(heldout_community_pre.max() + 1),
        "heldout_communities_post": int(heldout_community_post.max() + 1),
    }
    genre_counts = {
        "train": int(sum(1 for value in train_genre if value)),
        "heldout": int(sum(1 for value in heldout_genre if value)),
    }
    write_report(
        train_metrics_pre, train_metrics_post, heldout_metrics_pre, heldout_metrics_post,
        shapes, genre_counts,
    )
    log(f"Wrote report to {REPORT_PATH}.")


if __name__ == "__main__":
    main()
