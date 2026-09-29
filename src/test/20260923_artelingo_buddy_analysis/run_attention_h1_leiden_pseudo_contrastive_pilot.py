"""Train Attention-h1 with periodic Leiden pseudo-label contrastive regularization."""

import importlib.util
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(OUT_DIR, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
import src.conditional_buddy.prototype_seed  # noqa: F401; preload before arch's guarded import

ARCH_SWEEP_PATH = os.path.join(OUT_DIR, "run_learned_student_arch_sweep_pilot.py")
NPZ_PATH = os.path.join(OUT_DIR, "attention_h1_leiden_pseudo_contrastive_snapshot.npz")
REPORT_PATH = os.path.join(OUT_DIR, "attention_h1_leiden_pseudo_contrastive_pilot_report.md")
RECLUSTER_EVERY = 20
LAMBDA_CLUSTER_MAX = 1.0
CLUSTER_WARMUP_EPOCHS = 50
DOMINANT_FRACTION_GUARD = 0.80
SEED = 42

# Literal baseline values from the prior run_attention_h1_embedding_snapshot_pilot.py run.
BASELINE = {
    "train emotion AMI": 0.1351,
    "train genre AMI": 0.2397,
    "held-out emotion AMI": 0.1249,
    "held-out genre AMI": 0.2404,
    "train silhouette pre": -0.0024,
    "train silhouette post": 0.0253,
    "held-out silhouette pre": 0.0302,
    "held-out silhouette post": 0.0392,
}


def load_module(module_name: str, path: str):
    """Load a sibling standalone script without executing its main block."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_same_community_pairs(
    communities: np.ndarray, rng: np.random.Generator, batch_size: int
) -> np.ndarray:
    """Sample distinct, same-community node pairs from at least two eligible groups."""
    labels, inverse, counts = np.unique(communities, return_inverse=True, return_counts=True)
    eligible = np.flatnonzero(counts >= 2)
    if len(eligible) < 2:
        raise RuntimeError(
            "Cannot sample same-community pairs: fewer than 2 communities have >=2 members."
        )
    members = [np.flatnonzero(inverse == group) for group in eligible]
    weights = counts[eligible].astype(np.float64)
    choices = rng.choice(len(eligible), size=batch_size, p=weights / weights.sum())
    pairs = np.empty((batch_size, 2), dtype=np.int64)
    for row, choice in enumerate(choices):
        pairs[row] = rng.choice(members[choice], size=2, replace=False)
    return pairs


def recompute_pseudo_communities(
    model, train_content, train_affect, pipeline, affect_pilot, single_modality,
    device, expected_nodes, log,
) -> tuple[np.ndarray, float]:
    """Refresh train Leiden labels on the current full student embedding."""
    model.eval()
    with torch.no_grad():
        embedding, _ = model(train_content, train_affect)
    embedding_np = embedding.cpu().numpy().astype(np.float32, copy=False)
    graph = single_modality.build_single_modality_graph(
        "pseudo-train-learned-student", embedding_np, pipeline, affect_pilot,
        str(device), expected_nodes=expected_nodes,
    )
    communities = np.asarray(arch.detect_communities(graph, seed=SEED))
    if len(communities) != expected_nodes:
        raise RuntimeError(
            f"Pseudo-Leiden returned {len(communities)} labels for {expected_nodes} nodes."
        )
    _, counts = np.unique(communities, return_counts=True)
    dominant_fraction = float(counts.max() / expected_nodes)
    log(f"Pseudo-recluster: communities={len(counts)}, dominant_fraction={dominant_fraction:.4f}.")
    return communities, dominant_fraction


def graph_communities(
    embeddings_np, label, pipeline_module, affect_pilot, single_modality,
    device, expected_nodes,
):
    graph = single_modality.build_single_modality_graph(
        label, embeddings_np, pipeline_module, affect_pilot, str(device),
        expected_nodes=expected_nodes,
    )
    return np.asarray(arch.detect_communities(graph, seed=SEED))


def capture_embeddings(model, content, affect):
    model.eval()
    with torch.no_grad():
        embeddings, _ = model(content, affect)
    return embeddings.cpu().numpy().astype(np.float32, copy=False)


def sampled_silhouette(embeddings, communities):
    idx = np.random.default_rng(SEED).choice(
        len(embeddings), size=min(6000, len(embeddings)), replace=False
    )
    return float(silhouette_score(
        embeddings[idx], communities[idx],
        sample_size=min(4000, len(idx)), random_state=SEED,
    ))


def run_pseudo_contrastive_training(
    model, train_content, train_affect, heldout_content, heldout_affect,
    content_edges, affect_edges, heldout_content_graph, heldout_affect_graph,
    sampled_nodes, rank_nodes, pipeline, heldout_pipeline, affect_pilot,
    single_modality, device, expected_nodes, log,
):
    """Mirror arch's optimization and stopping loop, adding recycled Leiden positives."""
    optimizer = torch.optim.Adam(model.parameters(), lr=arch.LEARNING_RATE)
    log("Evaluating epoch-0 checkpoint before training...")
    epoch_0 = arch.evaluate_checkpoint(
        model, heldout_content, heldout_affect, heldout_content_graph,
        heldout_affect_graph, sampled_nodes, rank_nodes, single_modality,
        heldout_pipeline, affect_pilot, str(device),
    )
    trajectory = [{"epoch": 0, **epoch_0}]
    log(
        f"epoch=0 content_recall={epoch_0['content_recall']:.4f} "
        f"affect_recall={epoch_0['affect_recall']:.4f} "
        f"top_eigen_fraction={epoch_0['top_eigen_fraction']:.4f} "
        f"effective_rank_95={epoch_0['effective_rank_95']} "
        f"gate_mean={epoch_0['gate_mean']:.4f} gate_std={epoch_0['gate_std']:.4f} "
        f"gate_saturated={epoch_0['gate_saturated_fraction']:.4f}"
    )
    current_communities = None
    current_count = 0
    current_dominant = float("nan")
    recluster_passes = []
    skipped = 0
    plateau_count, stop_reason = 0, f"reached MAX_EPOCHS={arch.MAX_EPOCHS}"
    for epoch in range(1, arch.MAX_EPOCHS + 1):
        if (epoch - 1) % RECLUSTER_EVERY == 0:
            proposed, fraction = recompute_pseudo_communities(
                model, train_content, train_affect, pipeline, affect_pilot,
                single_modality, device, expected_nodes, log,
            )
            count = len(np.unique(proposed))
            if fraction > DOMINANT_FRACTION_GUARD:
                skipped += 1
                log(
                    f"WARNING: pseudo-clustering degenerate, dominant community covers "
                    f"{fraction:.2%} of nodes -- skipping this recluster pass (epoch {epoch})."
                )
            else:
                # The sampler needs two communities with at least two members each.
                _, counts = np.unique(proposed, return_counts=True)
                if np.count_nonzero(counts >= 2) < 2:
                    skipped += 1
                    log(
                        f"WARNING: pseudo-clustering has fewer than two pair-eligible "
                        f"communities -- skipping this recluster pass (epoch {epoch})."
                    )
                else:
                    current_communities = proposed
                    current_count = count
                    current_dominant = fraction
                    recluster_passes.append((epoch, count, fraction))
        model.train()
        epoch_rng = np.random.default_rng(arch.SEED + epoch)
        content_pairs = arch.sample_positive_pairs(content_edges, epoch_rng)
        affect_pairs = arch.sample_positive_pairs(affect_edges, epoch_rng)
        content_embeddings, remapped_content_pairs = arch.content_batch_embeddings(
            model, train_content, train_affect, content_pairs, device
        )
        content_loss = arch.symmetric_infonce(content_embeddings, remapped_content_pairs, device)
        affect_embeddings, _mixing_weights = model(train_content, train_affect)
        affect_loss = arch.symmetric_infonce(affect_embeddings, affect_pairs, device)
        lambda_cluster = LAMBDA_CLUSTER_MAX * min(1.0, epoch / CLUSTER_WARMUP_EPOCHS)
        cluster_loss_value = float("nan")
        if current_communities is None:
            total_loss = content_loss + affect_loss
        else:
            cluster_pairs = sample_same_community_pairs(
                current_communities, epoch_rng, arch.BATCH_SIZE
            )
            cluster_embeddings, remapped = arch.content_batch_embeddings(
                model, train_content, train_affect, cluster_pairs, device
            )
            cluster_loss = arch.symmetric_infonce(cluster_embeddings, remapped, device)
            cluster_loss_value = float(cluster_loss.detach().item())
            total_loss = content_loss + affect_loss + lambda_cluster * cluster_loss
        gradient_share = None
        if epoch % arch.CHECKPOINT_EVERY == 0:
            optimizer.zero_grad(set_to_none=True)
            content_loss.backward(retain_graph=True)
            content_grad_norm = arch.selected_parameter_gradient_norm(model)
            optimizer.zero_grad(set_to_none=True)
            affect_loss.backward(retain_graph=True)
            affect_grad_norm = arch.selected_parameter_gradient_norm(model)
            gradient_share = content_grad_norm / max(content_grad_norm + affect_grad_norm, 1e-12)
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        if epoch % arch.CHECKPOINT_EVERY:
            continue
        diagnostics = arch.evaluate_checkpoint(
            model, heldout_content, heldout_affect, heldout_content_graph,
            heldout_affect_graph, sampled_nodes, rank_nodes, single_modality,
            heldout_pipeline, affect_pilot, str(device),
        )
        checkpoint = {
            "epoch": epoch,
            "content_loss": float(content_loss.detach().item()),
            "affect_loss": float(affect_loss.detach().item()),
            "cluster_loss": cluster_loss_value,
            "lambda_cluster": lambda_cluster,
            "pseudo_community_count": current_count,
            "pseudo_dominant_fraction": current_dominant,
            "content_gradient_share": float(gradient_share),
            **diagnostics,
        }
        trajectory.append(checkpoint)
        log(
            f"epoch={epoch} content_recall={checkpoint['content_recall']:.4f} "
            f"affect_recall={checkpoint['affect_recall']:.4f} "
            f"content_loss={checkpoint['content_loss']:.4f} "
            f"affect_loss={checkpoint['affect_loss']:.4f} "
            f"cluster_loss={cluster_loss_value:.4f} lambda_cluster={lambda_cluster:.4f} "
            f"pseudo_communities={current_count} dominant_fraction={current_dominant:.4f} "
            f"content_grad_share={gradient_share:.4f} "
            f"top_eigen_fraction={checkpoint['top_eigen_fraction']:.4f} "
            f"effective_rank_95={checkpoint['effective_rank_95']} "
            f"gate_mean={checkpoint['gate_mean']:.4f} gate_std={checkpoint['gate_std']:.4f} "
            f"gate_saturated={checkpoint['gate_saturated_fraction']:.4f}"
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
    return trajectory, recluster_passes, skipped, stop_reason


def external_amis(communities, paintings, majority_emotion, genre_map, pipeline_module):
    genre_indices = [i for i, painting in enumerate(paintings) if painting in genre_map]
    if not genre_indices:
        raise RuntimeError("No genre-labelled paintings overlap this node set.")
    return {
        "emotion": pipeline_module.external_metrics(communities, majority_emotion)["AMI"],
        "genre": pipeline_module.external_metrics(
            communities[genre_indices],
            [genre_map[paintings[i]] for i in genre_indices],
        )["AMI"],
    }


def write_report(results, passes, skipped, stop_reason, elapsed, counts):
    lines = [
        "# Attention-h1 Leiden pseudo-contrastive pilot\n\n",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Method\n\n",
        "The original Attention-h1 student learned from CLIP-content and GoEmotions-affect "
        "teacher graph InfoNCE losses, then used Leiden only after training. This pilot tests "
        "whether recycled Leiden pseudo-label positives tighten and separate the learned "
        "embedding by adding a third, same-community symmetric InfoNCE loss. The original "
        "two losses and recall-based plateau rule remain in place.\n\n",
        f"Schedule: recluster every {RECLUSTER_EVERY} epochs starting at epoch 1; "
        f"LAMBDA_CLUSTER_MAX={LAMBDA_CLUSTER_MAX}; "
        f"CLUSTER_WARMUP_EPOCHS={CLUSTER_WARMUP_EPOCHS}; "
        f"DOMINANT_FRACTION_GUARD={DOMINANT_FRACTION_GUARD}; "
        f"batch={arch.BATCH_SIZE}; temperature={arch.TEMPERATURE}; seed={SEED}; "
        f"MAX_EPOCHS={arch.MAX_EPOCHS}; CHECKPOINT_EVERY={arch.CHECKPOINT_EVERY}; "
        f"learning_rate={arch.LEARNING_RATE}.\n\n",
        f"Stopped: {stop_reason}. Wall-clock time: {elapsed:.1f} seconds.\n\n",
        "## Comparison with established Attention-h1 baseline\n\n",
        "Baseline numbers are literal constants from the prior "
        "`run_attention_h1_embedding_snapshot_pilot.py` run. Silhouette uses a "
        "seed-42 6,000-point draw followed by `silhouette_score` with "
        "`sample_size=min(4000, len(idx))`, `random_state=42`.\n\n",
        "| metric | baseline | this run | absolute difference |\n",
        "|---|---:|---:|---:|\n",
    ]
    for key, baseline in BASELINE.items():
        value = results[key]
        lines.append(f"| {key} | {baseline:.4f} | {value:.4f} | {abs(value - baseline):.4f} |\n")
    lines.extend([
        "\n## Community counts and reclustering\n\n",
        f"Train Leiden communities: pre={counts['train_pre']}, post={counts['train_post']}. "
        f"Held-out Leiden communities: pre={counts['heldout_pre']}, post={counts['heldout_post']}.\n\n",
        f"Degenerate recluster passes skipped: {skipped}. Successful recluster trajectory:\n\n",
        "| epoch | communities | dominant fraction |\n",
        "|---:|---:|---:|\n",
    ])
    for epoch, count, fraction in passes:
        lines.append(f"| {epoch} | {count} | {fraction:.4f} |\n")
    lines.append("\n")
    for split in ("train", "held-out"):
        difference = results[f"{split} silhouette post"] - BASELINE[f"{split} silhouette post"]
        verb = "improved" if difference > 0 else "did not improve"
        lines.append(
            f"{split.capitalize()} post-training silhouette {verb} versus the baseline "
            f"({results[f'{split} silhouette post']:.4f} vs "
            f"{BASELINE[f'{split} silhouette post']:.4f}; signed change {difference:+.4f}).\n"
        )
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main():
    started = time.perf_counter()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    global arch
    arch = load_module("attention_h1_pseudo_arch_sweep", ARCH_SWEEP_PATH)
    pipeline = arch.load_sibling_module("artelingo_run_pipeline_attn_pseudo", arch.PIPELINE_PATH)
    affect_pilot = arch.load_sibling_module("artelingo_run_affect_pilot_attn_pseudo", arch.AFFECT_PILOT_PATH)
    single_modality = arch.load_sibling_module("artelingo_run_single_modality_attn_pseudo", arch.SINGLE_MODALITY_PATH)
    cca_audit = arch.load_sibling_module("artelingo_run_cca_audit_attn_pseudo", arch.CCA_AUDIT_PATH)
    arch.cca_audit = cca_audit
    heldout_pipeline = arch.load_sibling_module("artelingo_run_pipeline_attn_pseudo_heldout", arch.PIPELINE_PATH)
    heldout_pipeline.STORAGE_DIR = arch.HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = arch.HELDOUT_JSON
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = arch.log
    log(f"Using {device} for Attention-h1 Leiden pseudo-contrastive pilot.")

    log("Verifying and loading train CLIP features...")
    pipeline.assert_extraction_complete()
    paintings, img_nodes, txt_nodes, emotion_counts = pipeline.load_dedup_features()
    majority_emotion = [pipeline.majority(counts) for counts in emotion_counts]
    log("Extracting train GoEmotions probabilities...")
    affect_train = np.asarray(
        affect_pilot.extract_affect_nodes(pipeline.TRAIN_JSON, paintings, str(device)), dtype=np.float64
    )
    if affect_train.shape != (len(paintings), 28):
        raise RuntimeError(f"Expected train affect features ({len(paintings)}, 28), got {affect_train.shape}.")
    log("Verifying and loading held-out CLIP features...")
    heldout_pipeline.assert_extraction_complete()
    heldout_paintings, heldout_img, heldout_txt, heldout_emotion_counts = heldout_pipeline.load_dedup_features()
    heldout_majority_emotion = [heldout_pipeline.majority(counts) for counts in heldout_emotion_counts]
    if len(heldout_paintings) != arch.HELDOUT_PAINTINGS:
        raise RuntimeError(f"Expected {arch.HELDOUT_PAINTINGS:,} held-out paintings, got {len(heldout_paintings):,}.")
    log("Extracting held-out GoEmotions probabilities...")
    affect_heldout = np.asarray(
        affect_pilot.extract_affect_nodes(arch.HELDOUT_JSON, heldout_paintings, str(device)), dtype=np.float64
    )
    if affect_heldout.shape != (len(heldout_paintings), 28):
        raise RuntimeError(f"Expected held-out affect features ({len(heldout_paintings)}, 28), got {affect_heldout.shape}.")

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
    log("Building held-out content and affect reference graphs...")
    heldout_content_graph = single_modality.build_single_modality_graph(
        "held-out-content-reference", content_heldout, heldout_pipeline, affect_pilot,
        str(device), expected_nodes=len(heldout_paintings),
    )
    heldout_affect_graph = single_modality.build_single_modality_graph(
        "held-out-affect-reference", affect_heldout, heldout_pipeline, affect_pilot,
        str(device), expected_nodes=len(heldout_paintings),
    )
    diagnostic_rng = np.random.default_rng(SEED)
    sampled_nodes = diagnostic_rng.choice(len(heldout_paintings), size=arch.EDGE_SAMPLE_SIZE, replace=False)
    rank_nodes = diagnostic_rng.choice(len(heldout_paintings), size=arch.EFFECTIVE_RANK_SAMPLE_SIZE, replace=False)
    train_content_t = torch.as_tensor(content_train, dtype=torch.float32, device=device)
    train_affect_t = torch.as_tensor(affect_train, dtype=torch.float32, device=device)
    heldout_content_t = torch.as_tensor(content_heldout, dtype=torch.float32, device=device)
    heldout_affect_t = torch.as_tensor(affect_heldout, dtype=torch.float32, device=device)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = arch.LearnedStudent("attn1").to(device)

    log("Capturing independent epoch-0 train and held-out Leiden snapshots...")
    train_pre = capture_embeddings(model, train_content_t, train_affect_t)
    heldout_pre = capture_embeddings(model, heldout_content_t, heldout_affect_t)
    train_community_pre = graph_communities(train_pre, "epoch0-train-attn1-pseudo", pipeline, affect_pilot, single_modality, device, len(paintings))
    heldout_community_pre = graph_communities(heldout_pre, "epoch0-heldout-attn1-pseudo", heldout_pipeline, affect_pilot, single_modality, device, len(heldout_paintings))

    log("Training Attention-h1 with periodic Leiden pseudo-label contrastive term...")
    trajectory, passes, skipped, stop_reason = run_pseudo_contrastive_training(
        model, train_content_t, train_affect_t, heldout_content_t, heldout_affect_t,
        content_edges, affect_edges, heldout_content_graph, heldout_affect_graph,
        sampled_nodes, rank_nodes, pipeline, heldout_pipeline, affect_pilot,
        single_modality, device, len(paintings), log,
    )

    log("Capturing independent final train and held-out Leiden snapshots...")
    train_post = capture_embeddings(model, train_content_t, train_affect_t)
    heldout_post = capture_embeddings(model, heldout_content_t, heldout_affect_t)
    train_community_post = graph_communities(train_post, "final-train-attn1-pseudo", pipeline, affect_pilot, single_modality, device, len(paintings))
    heldout_community_post = graph_communities(heldout_post, "final-heldout-attn1-pseudo", heldout_pipeline, affect_pilot, single_modality, device, len(heldout_paintings))
    genre_map = pipeline.load_genre_map()
    train_amis = external_amis(train_community_post, paintings, majority_emotion, genre_map, pipeline)
    heldout_amis = external_amis(heldout_community_post, heldout_paintings, heldout_majority_emotion, genre_map, heldout_pipeline)
    log("Computing four sampled silhouettes...")
    results = {
        "train emotion AMI": train_amis["emotion"],
        "train genre AMI": train_amis["genre"],
        "held-out emotion AMI": heldout_amis["emotion"],
        "held-out genre AMI": heldout_amis["genre"],
        "train silhouette pre": sampled_silhouette(train_pre, train_community_pre),
        "train silhouette post": sampled_silhouette(train_post, train_community_post),
        "held-out silhouette pre": sampled_silhouette(heldout_pre, heldout_community_pre),
        "held-out silhouette post": sampled_silhouette(heldout_post, heldout_community_post),
    }
    np.savez_compressed(
        NPZ_PATH,
        train_paintings=np.array(paintings, dtype=object),
        train_embedding_pre=train_pre, train_embedding_post=train_post,
        train_community_pre=train_community_pre.astype(np.int32),
        train_community_post=train_community_post.astype(np.int32),
        train_emotion=np.array(majority_emotion, dtype=object),
        train_genre=np.array([genre_map.get(p, "") for p in paintings], dtype=object),
        heldout_paintings=np.array(heldout_paintings, dtype=object),
        heldout_embedding_pre=heldout_pre, heldout_embedding_post=heldout_post,
        heldout_community_pre=heldout_community_pre.astype(np.int32),
        heldout_community_post=heldout_community_post.astype(np.int32),
        heldout_emotion=np.array(heldout_majority_emotion, dtype=object),
        heldout_genre=np.array([genre_map.get(p, "") for p in heldout_paintings], dtype=object),
        seed=SEED,
        train_silhouette_pre=results["train silhouette pre"],
        train_silhouette_post=results["train silhouette post"],
        heldout_silhouette_pre=results["held-out silhouette pre"],
        heldout_silhouette_post=results["held-out silhouette post"],
        recluster_every=RECLUSTER_EVERY, lambda_cluster_max=LAMBDA_CLUSTER_MAX,
        cluster_warmup_epochs=CLUSTER_WARMUP_EPOCHS,
    )
    counts = {
        "train_pre": len(np.unique(train_community_pre)),
        "train_post": len(np.unique(train_community_post)),
        "heldout_pre": len(np.unique(heldout_community_pre)),
        "heldout_post": len(np.unique(heldout_community_post)),
    }
    elapsed = time.perf_counter() - started
    write_report(results, passes, skipped, stop_reason, elapsed, counts)
    log(f"Results: {results}; community counts={counts}; skipped={skipped}; elapsed={elapsed:.1f}s.")
    log(f"Wrote {NPZ_PATH} and {REPORT_PATH}.")


if __name__ == "__main__":
    main()
