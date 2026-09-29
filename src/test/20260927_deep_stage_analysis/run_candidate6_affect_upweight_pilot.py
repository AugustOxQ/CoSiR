"""Screen affect InfoNCE weights for the no-noise Attention-h1 student."""

import importlib.util
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


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
REPORT_OUT_DIR = OUT_DIR
ARCH_SWEEP_PATH = os.path.join(OUT_DIR, "..", "20260923_artelingo_buddy_analysis", "run_learned_student_arch_sweep_pilot.py")
REPORT_PATH = os.path.join(REPORT_OUT_DIR, "candidate6_affect_upweight_pilot_report.md")

SEED = 42
SEEDS = (7, 123, 2024)
LAMBDA_AFFECTS = (1.0, 2.0, 4.0, 8.0)
NOISE_STD = 0.0
BASELINE_METRICS = {"emotion_ami": 0.1306, "genre_ami": 0.1973, "silhouette": 0.0488}
BASELINE_TOLERANCE = 0.001
PRACTICAL_MARGIN = 0.005
EMOTION_PARETO_BAR = 0.1236
GENRE_PARETO_BAR = 0.1954
LR_START = 1e-3
LR_FLOOR = 1e-5
def load_module(module_name: str, path: str):
    """Load a standalone sibling script without running its main block."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def noisy_unit_embedding(embedding: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std == 0.0:
        return embedding
    noised = embedding + noise_std * torch.randn_like(embedding)
    return torch.nn.functional.normalize(noised, dim=-1)


def capture_embeddings(model, content, affect) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        embeddings, _ = model(content, affect)
    return embeddings.cpu().numpy().astype(np.float32, copy=False)


def communities_for(arch, embeddings, label, pipeline_module, affect_pilot,
                    single_modality, device, expected_nodes):
    graph = single_modality.build_single_modality_graph(
        label, embeddings, pipeline_module, affect_pilot, str(device),
        expected_nodes=expected_nodes,
    )
    return np.asarray(arch.detect_communities(graph, seed=SEED))


def sampled_silhouette(embeddings, communities):
    # Same two-stage sampling convention as the Leiden pseudo-contrastive pilot.
    idx = np.random.default_rng(SEED).choice(
        len(embeddings), size=min(6000, len(embeddings)), replace=False
    )
    return float(silhouette_score(
        embeddings[idx], communities[idx],
        sample_size=min(4000, len(idx)), random_state=SEED,
    ))


def split_metrics(embeddings, communities, paintings, emotions, genre_map, pipeline_module):
    genre_indices = [i for i, painting in enumerate(paintings) if painting in genre_map]
    if not genre_indices:
        raise RuntimeError("No genre-labelled paintings overlap this node set.")
    return {
        "emotion_ami": pipeline_module.external_metrics(communities, emotions)["AMI"],
        "genre_ami": pipeline_module.external_metrics(
            communities[genre_indices],
            [genre_map[paintings[i]] for i in genre_indices],
        )["AMI"],
        "silhouette": sampled_silhouette(embeddings, communities),
        "communities": len(np.unique(communities)),
    }


def clears_pareto(result):
    heldout = result["heldout_post"]
    return (heldout["emotion_ami"] > EMOTION_PARETO_BAR
            and heldout["genre_ami"] > GENRE_PARETO_BAR)


def run_seed(seed, noise_std, context, lambda_affect):
    arch = context["arch"]
    pipeline = context["pipeline"]
    heldout_pipeline = context["heldout_pipeline"]
    affect_pilot = context["affect_pilot"]
    single_modality = context["single_modality"]
    device = context["device"]
    log = arch.log

    # Every screen and stress run starts with fresh random state and weights.
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = arch.LearnedStudent("attn1").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_START)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=arch.MAX_EPOCHS, eta_min=LR_FLOOR
    )
    log(f"Starting Attention-h1 run: seed={seed}, noise_std={noise_std:g}.")

    train_pre = capture_embeddings(model, context["train_content_t"], context["train_affect_t"])
    heldout_pre = capture_embeddings(model, context["heldout_content_t"], context["heldout_affect_t"])
    train_community_pre = communities_for(
        arch, train_pre, f"epoch0-train-attn1-noise-{seed}-{noise_std:g}", pipeline,
        affect_pilot, single_modality, device, len(context["paintings"]),
    )
    heldout_community_pre = communities_for(
        arch, heldout_pre, f"epoch0-heldout-attn1-noise-{seed}-{noise_std:g}",
        heldout_pipeline, affect_pilot, single_modality, device,
        len(context["heldout_paintings"]),
    )
    train_pre_metrics = split_metrics(
        train_pre, train_community_pre, context["paintings"], context["majority_emotion"],
        context["genre_map"], pipeline,
    )
    heldout_pre_metrics = split_metrics(
        heldout_pre, heldout_community_pre, context["heldout_paintings"],
        context["heldout_majority_emotion"], context["genre_map"], heldout_pipeline,
    )
    epoch_0_diag = arch.evaluate_checkpoint(
        model, context["heldout_content_t"], context["heldout_affect_t"],
        context["heldout_content_graph"], context["heldout_affect_graph"],
        context["sampled_nodes"], context["rank_nodes"], single_modality,
        heldout_pipeline, affect_pilot, str(device),
    )
    trajectory = [{"epoch": 0, "lr": LR_START, **epoch_0_diag}]
    log(
        f"seed={seed} noise_std={noise_std:g} epoch=0 lr={LR_START:.8f} "
        f"content_recall={epoch_0_diag['content_recall']:.4f} "
        f"affect_recall={epoch_0_diag['affect_recall']:.4f}"
    )
    plateau_count, stop_reason = 0, f"reached MAX_EPOCHS={arch.MAX_EPOCHS}"
    for epoch in range(1, arch.MAX_EPOCHS + 1):
        model.train()
        epoch_rng = np.random.default_rng(seed + epoch)
        content_pairs = arch.sample_positive_pairs(context["content_edges"], epoch_rng)
        affect_pairs = arch.sample_positive_pairs(context["affect_edges"], epoch_rng)
        content_embeddings, remapped_content_pairs = arch.content_batch_embeddings(
            model, context["train_content_t"], context["train_affect_t"],
            content_pairs, device,
        )
        content_loss = arch.symmetric_infonce(
            noisy_unit_embedding(content_embeddings, noise_std),
            remapped_content_pairs, device,
        )
        affect_embeddings, _mixing_weights = model(
            context["train_content_t"], context["train_affect_t"]
        )
        affect_loss = arch.symmetric_infonce(
            noisy_unit_embedding(affect_embeddings, noise_std), affect_pairs, device,
        )
        total_loss = content_loss + lambda_affect * affect_loss
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        learning_rate = optimizer.param_groups[0]["lr"]
        optimizer.step()
        # Never step beyond T_max: CosineAnnealingLR would otherwise rise again.
        if epoch <= arch.MAX_EPOCHS:
            scheduler.step()
        if epoch % arch.CHECKPOINT_EVERY:
            continue
        diagnostics = arch.evaluate_checkpoint(
            model, context["heldout_content_t"], context["heldout_affect_t"],
            context["heldout_content_graph"], context["heldout_affect_graph"],
            context["sampled_nodes"], context["rank_nodes"], single_modality,
            heldout_pipeline, affect_pilot, str(device),
        )
        checkpoint = {"epoch": epoch, "lr": learning_rate, **diagnostics}
        trajectory.append(checkpoint)
        log(
            f"seed={seed} noise_std={noise_std:g} epoch={epoch} "
            f"lr={learning_rate:.8f} "
            f"content_recall={checkpoint['content_recall']:.4f} "
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
            stop_reason = (
                f"both recalls plateaued for {arch.PLATEAU_WINDOW} "
                "consecutive checkpoints"
            )
            log(f"Stopping: {stop_reason} at epoch {epoch}.")
            break
    log(f"Training stopped: {stop_reason}.")

    train_post = capture_embeddings(model, context["train_content_t"], context["train_affect_t"])
    heldout_post = capture_embeddings(
        model, context["heldout_content_t"], context["heldout_affect_t"]
    )
    train_community_post = communities_for(
        arch, train_post, f"final-train-attn1-noise-{seed}-{noise_std:g}", pipeline,
        affect_pilot, single_modality, device, len(context["paintings"]),
    )
    heldout_community_post = communities_for(
        arch, heldout_post, f"final-heldout-attn1-noise-{seed}-{noise_std:g}",
        heldout_pipeline, affect_pilot, single_modality, device,
        len(context["heldout_paintings"]),
    )
    result = {
        "seed": seed,
        "noise_std": noise_std,
        "train_pre": train_pre_metrics,
        "heldout_pre": heldout_pre_metrics,
        "train_post": split_metrics(
            train_post, train_community_post, context["paintings"],
            context["majority_emotion"], context["genre_map"], pipeline,
        ),
        "heldout_post": split_metrics(
            heldout_post, heldout_community_post, context["heldout_paintings"],
            context["heldout_majority_emotion"], context["genre_map"], heldout_pipeline,
        ),
        "trajectory": trajectory,
        "stop_reason": stop_reason,
    }
    log(
        f"seed={seed} noise_std={noise_std:g} final train emotion "
        f"AMI={result['train_post']['emotion_ami']:.4f} genre "
        f"AMI={result['train_post']['genre_ami']:.4f} silhouette="
        f"{result['train_post']['silhouette']:.4f}; held-out emotion "
        f"AMI={result['heldout_post']['emotion_ami']:.4f} genre "
        f"AMI={result['heldout_post']['genre_ami']:.4f} silhouette="
        f"{result['heldout_post']['silhouette']:.4f}."
    )
    return result


def run_weight(seed, lambda_affect, context):
    result = run_seed(seed, NOISE_STD, context, lambda_affect)
    result["lambda_affect"] = lambda_affect
    return result


def baseline_discrepancies(result):
    heldout = result["heldout_post"]
    return {
        key: (heldout[key], expected)
        for key, expected in BASELINE_METRICS.items()
        if abs(heldout[key] - expected) > BASELINE_TOLERANCE
    }


def choose_winner(screen):
    qualifying = [
        result for result in screen
        if result["lambda_affect"] > 1.0
        and result["heldout_post"]["emotion_ami"] > BASELINE_METRICS["emotion_ami"] + PRACTICAL_MARGIN
        and clears_pareto(result)
    ]
    return max(qualifying, key=lambda result: result["heldout_post"]["emotion_ami"], default=None)


def write_report(screen, winner, stress, discrepancies, arch):
    lines = [
        "# Candidate 6 — affect InfoNCE upweighting pilot\n\n",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. Local CUDA Stage-1 experiment; no DAS6.\n\n",
        "## Method\n\n",
        "Copied the established Attention-h1 noise/schedule pilot's data setup, teacher graphs, "
        "training loop, cosine LR schedule, plateau stopping, and clean-embedding post-hoc "
        "Leiden/AMI/silhouette evaluation. Set noise_std=0.0; the only training change is "
        "`total_loss = content_loss + LAMBDA_AFFECT * affect_loss`. "
        f"Adam LR={LR_START:g}, CosineAnnealingLR(T_max={arch.MAX_EPOCHS}, eta_min={LR_FLOOR:g}). "
        "The held-out Pareto bar is emotion AMI > 0.1236 AND genre AMI > 0.1954. "
        "A winner also needs emotion AMI > 0.1356 (strictly more than +0.005 over 0.1306).\n\n",
        "## Seed-42 screen\n\n",
        "| LAMBDA_AFFECT | held-out emotion AMI | held-out genre AMI | held-out silhouette | Pareto bar | practical margin |\n",
        "|---:|---:|---:|---:|---|---|\n",
    ]
    for result in screen:
        h = result["heldout_post"]
        margin = h["emotion_ami"] > BASELINE_METRICS["emotion_ami"] + PRACTICAL_MARGIN
        lines.append(
            f"| {result['lambda_affect']:g} | {h['emotion_ami']:.4f} | "
            f"{h['genre_ami']:.4f} | {h['silhouette']:.4f} | "
            f"{'clears' if clears_pareto(result) else 'does not clear'} | "
            f"{'clears' if margin else 'does not clear'} |\n"
        )
    baseline = screen[0]["heldout_post"]
    lines.append(
        "\nBaseline sanity check: expected 0.1306 / 0.1973 / 0.0488 "
        "(emotion AMI / genre AMI / silhouette), "
        f"observed {baseline['emotion_ami']:.4f} / {baseline['genre_ami']:.4f} / "
        f"{baseline['silhouette']:.4f}; tolerance ±{BASELINE_TOLERANCE:g} per metric. "
        f"{'FAILED' if discrepancies else 'Passed'}.\n\n"
    )
    if discrepancies:
        lines.append(f"Stopped before testing other weights because baseline reproduction failed: {discrepancies}.\n\n")
        lines.append("## Verdict\n\nThe baseline did not reproduce, so no inference about affect weighting is reliable.\n")
    elif winner is None:
        lines.append("## Winner selection\n\nNo upweighted point both beats the +0.005 emotion AMI margin and clears both Pareto bars. No non-qualifying point was stress-tested.\n\n")
        best = max(screen[1:], key=lambda result: result["heldout_post"]["emotion_ami"])
        h = best["heldout_post"]
        direction = "higher" if h["emotion_ami"] > baseline["emotion_ami"] else "lower or unchanged"
        lines.append("## Verdict\n\n")
        lines.append(
            f"Upweighting the affect loss produced {direction} held-out emotion AMI at best "
            f"({h['emotion_ami']:.4f} at LAMBDA_AFFECT={best['lambda_affect']:g}), "
            "but no setting delivered a meaningful emotion gain while preserving the "
            "genre Pareto bar. This screen does not support the flat equal-weight loss "
            "as the bottleneck behind Finding D; the weak emotion separation likely "
            "depends on the teacher signal, representation, or clustering as well.\n"
        )
    else:
        h = winner["heldout_post"]
        four = [winner, *stress]
        clear_count = sum(clears_pareto(result) for result in four)
        lines.append(
            f"## Winner selection\n\nLAMBDA_AFFECT={winner['lambda_affect']:g} has the highest "
            f"qualifying seed-42 emotion AMI ({h['emotion_ami']:.4f}); genre AMI="
            f"{h['genre_ami']:.4f}.\n\n"
            "## Four-seed stress\n\nSeed 42 is reused from the screen; additional seeds are 7, 123, and 2024.\n\n"
            "| seed | emotion AMI | genre AMI | silhouette | Pareto bar |\n"
            "|---:|---:|---:|---:|---|\n"
        )
        for result in four:
            m = result["heldout_post"]
            lines.append(
                f"| {result['seed']} | {m['emotion_ami']:.4f} | {m['genre_ami']:.4f} | "
                f"{m['silhouette']:.4f} | {'clears' if clears_pareto(result) else 'does not clear'} |\n"
            )
        lines.append("\n| held-out metric | mean | min | max | std (population) |\n|---|---:|---:|---:|---:|\n")
        for key, label in (("emotion_ami", "Emotion AMI"), ("genre_ami", "Genre AMI"), ("silhouette", "Silhouette")):
            values = np.array([result["heldout_post"][key] for result in four])
            lines.append(f"| {label} | {values.mean():.4f} | {values.min():.4f} | {values.max():.4f} | {values.std():.4f} |\n")
        lines.append(f"\nBoth held-out Pareto bars clear in {clear_count}/4 seeds.\n\n## Verdict\n\n")
        if clear_count == 4:
            lines.append("Upweighting affect materially improves seed-42 emotion separation while preserving the genre bar across all four seeds. This supports the flat equal-weight loss as one bottleneck behind Finding D. Re-running Stage 2 (Leiden re-clustering plus the candidate-1/2/4 pipeline) on the new embedding is a follow-up decision to test whether the 0.8534 macro AUC headline changes.\n")
        else:
            lines.append("Upweighting affect improves seed-42 emotion separation, but the four-seed stress does not consistently preserve both Pareto bars. The flat equal-weight loss may contribute to Finding D, but this result does not validate it as a robust bottleneck.\n")
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

    arch = load_module("attention_h1_noise_arch_sweep", ARCH_SWEEP_PATH)
    pipeline = arch.load_sibling_module("artelingo_run_pipeline_attn_noise", arch.PIPELINE_PATH)
    affect_pilot = arch.load_sibling_module("artelingo_run_affect_pilot_attn_noise", arch.AFFECT_PILOT_PATH)
    single_modality = arch.load_sibling_module(
        "artelingo_run_single_modality_attn_noise", arch.SINGLE_MODALITY_PATH
    )
    cca_audit = arch.load_sibling_module("artelingo_run_cca_audit_attn_noise", arch.CCA_AUDIT_PATH)
    # evaluate_checkpoint refers to this module-level global in the sibling.
    arch.cca_audit = cca_audit
    heldout_pipeline = arch.load_sibling_module(
        "artelingo_run_pipeline_attn_noise_heldout", arch.PIPELINE_PATH
    )
    heldout_pipeline.STORAGE_DIR = arch.HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = arch.HELDOUT_JSON
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = arch.log
    log(f"Using {device} for the Attention-h1 noise/schedule pilot.")

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
    heldout_paintings, heldout_img, heldout_txt, heldout_emotion_counts = (
        heldout_pipeline.load_dedup_features()
    )
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
    sampled_nodes = diagnostic_rng.choice(
        len(heldout_paintings), size=arch.EDGE_SAMPLE_SIZE, replace=False
    )
    rank_nodes = diagnostic_rng.choice(
        len(heldout_paintings), size=arch.EFFECTIVE_RANK_SAMPLE_SIZE, replace=False
    )
    context = {
        "arch": arch, "pipeline": pipeline, "heldout_pipeline": heldout_pipeline,
        "affect_pilot": affect_pilot, "single_modality": single_modality,
        "device": device, "paintings": paintings,
        "majority_emotion": majority_emotion,
        "heldout_paintings": heldout_paintings,
        "heldout_majority_emotion": heldout_majority_emotion,
        "genre_map": pipeline.load_genre_map(),
        "content_edges": content_edges, "affect_edges": affect_edges,
        "heldout_content_graph": heldout_content_graph,
        "heldout_affect_graph": heldout_affect_graph,
        "sampled_nodes": sampled_nodes, "rank_nodes": rank_nodes,
        "train_content_t": torch.as_tensor(content_train, dtype=torch.float32, device=device),
        "train_affect_t": torch.as_tensor(affect_train, dtype=torch.float32, device=device),
        "heldout_content_t": torch.as_tensor(content_heldout, dtype=torch.float32, device=device),
        "heldout_affect_t": torch.as_tensor(affect_heldout, dtype=torch.float32, device=device),
    }
    if not torch.cuda.is_available():
        raise RuntimeError("This pilot requires the local CUDA GPU.")
    screen = [run_weight(SEED, LAMBDA_AFFECTS[0], context)]
    discrepancies = baseline_discrepancies(screen[0])
    if discrepancies:
        write_report(screen, None, [], discrepancies, arch)
        log(f"Baseline sanity check failed: {discrepancies}. Wrote {REPORT_PATH}.")
        return
    for lambda_affect in LAMBDA_AFFECTS[1:]:
        screen.append(run_weight(SEED, lambda_affect, context))
    winner = choose_winner(screen)
    stress = [run_weight(seed, winner["lambda_affect"], context) for seed in SEEDS] if winner else []
    write_report(screen, winner, stress, {}, arch)
    log(f"Wrote report to {REPORT_PATH}.")


if __name__ == "__main__":
    main()
