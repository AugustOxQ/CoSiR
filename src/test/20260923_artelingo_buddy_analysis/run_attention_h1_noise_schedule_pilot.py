"""Screen embedding noise and cosine LR on the standing buddy-graph Attention-h1 student.

This adapts the PercepT noise/schedule idea, not its autoencoder recipe:
Attention-h1 has no reconstruction decoder or clean reconstruction target, and its
32-D LayerNorm/L2 embedding lives on the unit hypersphere rather than in an
unconstrained 128-D latent space. Gaussian noise is added only immediately
before each training InfoNCE loss, then the perturbed embedding is returned
to unit L2 norm so the InfoNCE temperature retains its meaning. Checkpoint
diagnostics and epoch-0/final Leiden, AMI, and silhouette use clean model
forward passes throughout.
"""

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
REPORT_OUT_DIR = os.environ.get("PERCEPT_OUTPUT_ROOT") or OUT_DIR
ARCH_SWEEP_PATH = os.path.join(OUT_DIR, "run_learned_student_arch_sweep_pilot.py")
REPORT_PATH = os.path.join(REPORT_OUT_DIR, "attention_h1_noise_schedule_pilot_report.md")

SEED = 42
SEEDS = (7, 123, 2024)
NOISE_STDS = (0.0, 0.02, 0.05, 0.1)
EMOTION_PARETO_BAR = 0.1236
GENRE_PARETO_BAR = 0.1954
LR_START = 1e-3
LR_FLOOR = 1e-5
REFERENCES = (
    ("Attention-h1 baseline", 0.1249, 0.2404, 0.0392),
    ("PercepT replication standing balance-hack", 0.1252, 0.2486, None),
    ("PercepT replication faithful recipe", 0.1092, 0.3288, 0.5120),
)


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


def run_seed(seed, noise_std, context):
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
        total_loss = content_loss + affect_loss
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


def choose_winner(screen):
    clearers = [result for result in screen if clears_pareto(result)]
    candidates = clearers if clearers else screen
    winner = max(candidates, key=lambda result: result["heldout_post"]["silhouette"])
    return winner, bool(clearers)


def format_summary(values):
    return f"mean={np.mean(values):.4f}; min={np.min(values):.4f}; max={np.max(values):.4f}"


def write_report(screen, winner, validated_winner, stress, arch):
    four_seeds = [winner, *stress]
    clear_count = sum(clears_pareto(result) for result in four_seeds)
    lines = [
        "# Attention-h1 embedding noise and cosine LR schedule pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Method\n\n",
        "This adapts the PercepT noise/schedule idea to the standing buddy-graph "
        "Attention-h1 two-teacher contrastive student. Attention-h1 has no "
        "autoencoder, reconstruction decoder, or clean reconstruction target; "
        "its LayerNorm/L2-normalized 32-D embedding lives on the unit hypersphere, "
        "unlike PercepT's unconstrained 128-D latent. Gaussian noise is added "
        "immediately before each of the two training InfoNCE losses, followed "
        "by L2 renormalization to preserve unit norm and the temperature's "
        "effective meaning. Epoch-0/final embeddings, checkpoint diagnostics, "
        "Leiden, AMI, and silhouette all use clean forward passes. "
        "Teacher graphs, Attention-h1 architecture, pair sampling, InfoNCE, "
        "and recall plateau stopping follow the embedding snapshot pilot.\n\n",
        f"Adam starts at {LR_START:g}; CosineAnnealingLR has T_max={arch.MAX_EPOCHS} "
        f"and eta_min={LR_FLOOR:g}; it steps once after each training epoch, "
        "never beyond T_max. The learning rate in each trajectory row is the "
        "rate used to update weights in that epoch. "
        f"CHECKPOINT_EVERY={arch.CHECKPOINT_EVERY}; seed-42 screen noise_std="
        f"{NOISE_STDS}; additional stress seeds={SEEDS}.\n\n",
        "Silhouette follows the Leiden pseudo-contrastive pilot: a seed-42 "
        "draw of at most 6,000 embeddings, then `silhouette_score` on the "
        "embedding and its Leiden labels with `sample_size=min(4000, len(idx))` "
        "and `random_state=42`.\n\n",
        "## Single-seed screen (seed 42)\n\n",
        f"Held-out Pareto bar: emotion AMI > {EMOTION_PARETO_BAR:.4f} AND "
        f"genre AMI > {GENRE_PARETO_BAR:.4f}.\n\n",
        "| noise_std | split | pre emotion AMI | pre genre AMI | pre silhouette | "
        "post emotion AMI | post genre AMI | post silhouette | pre Leiden communities | "
        "post Leiden communities | held-out Pareto bar |\n",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n",
    ]
    for result in screen:
        for split, pre_key, post_key in (
            ("train", "train_pre", "train_post"),
            ("held-out", "heldout_pre", "heldout_post"),
        ):
            pre, post = result[pre_key], result[post_key]
            verdict = ("clears" if clears_pareto(result) else "does not clear") if split == "held-out" else "n/a"
            lines.append(
                f"| {result['noise_std']:g} | {split} | {pre['emotion_ami']:.4f} | "
                f"{pre['genre_ami']:.4f} | {pre['silhouette']:.4f} | "
                f"{post['emotion_ami']:.4f} | {post['genre_ami']:.4f} | "
                f"{post['silhouette']:.4f} | {pre['communities']} | "
                f"{post['communities']} | {verdict} |\n"
            )
    lines.append("\n## Screen training and LR trajectories\n\n")
    for result in screen:
        lines.extend([
            f"### noise_std={result['noise_std']:g}\n\n",
            f"Stopped: {result['stop_reason']}.\n\n",
            "| epoch | LR used | content held-out recall | affect held-out recall | "
            "top eigenvalue fraction | effective rank at 95% | attention mean | "
            "attention std | attention saturated fraction |\n",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        ])
        for row in result["trajectory"]:
            lines.append(
                f"| {row['epoch']} | {row['lr']:.8f} | "
                f"{row['content_recall']:.4f} | {row['affect_recall']:.4f} | "
                f"{row['top_eigen_fraction']:.4f} | {row['effective_rank_95']} | "
                f"{row['gate_mean']:.4f} | {row['gate_std']:.4f} | "
                f"{row['gate_saturated_fraction']:.4f} |\n"
            )
        lines.append("\n")
    lines.append("## Winner selection\n\n")
    if validated_winner:
        clear_values = ", ".join(
            f"{result['noise_std']:g}" for result in screen if clears_pareto(result)
        )
        lines.append(
            f"Pareto-bar clearers: {clear_values}. Selected noise_std="
            f"{winner['noise_std']:g}, with the highest held-out silhouette among "
            f"clearers ({winner['heldout_post']['silhouette']:.4f}). This is the "
            "seed-42 winner; robustness is decided by the stress seeds below.\n\n"
        )
    else:
        lines.append(
            f"No screened value clears the held-out Pareto bar. Selected noise_std="
            f"{winner['noise_std']:g} for highest held-out silhouette among all "
            f"four ({winner['heldout_post']['silhouette']:.4f}): best available, "
            "does not clear the Pareto bar. This is not a validated winner. "
            "Only this selected value receives seed stress.\n\n"
        )
    lines.extend([
        "## Four-seed stress of the selected noise_std\n\n",
        f"Only noise_std={winner['noise_std']:g} was run at seeds 7, 123, and "
        "2024; seed 42 is the screen result, not a repeat fit.\n\n",
        "| seed | train emotion AMI | train genre AMI | train silhouette | "
        "held-out emotion AMI | held-out genre AMI | held-out silhouette | "
        "held-out Pareto bar |\n",
        "|---:|---:|---:|---:|---:|---:|---:|---|\n",
    ])
    for result in four_seeds:
        train, heldout = result["train_post"], result["heldout_post"]
        lines.append(
            f"| {result['seed']} | {train['emotion_ami']:.4f} | "
            f"{train['genre_ami']:.4f} | {train['silhouette']:.4f} | "
            f"{heldout['emotion_ami']:.4f} | {heldout['genre_ami']:.4f} | "
            f"{heldout['silhouette']:.4f} | "
            f"{'clears' if clears_pareto(result) else 'does not clear'} |\n"
        )
    lines.append("\n### Held-out summary statistics\n\n")
    for metric, label in (
        ("emotion_ami", "Emotion AMI"),
        ("genre_ami", "Genre AMI"),
        ("silhouette", "Silhouette"),
    ):
        values = [result["heldout_post"][metric] for result in four_seeds]
        lines.append(f"- {label} across four seeds: {format_summary(values)}.\n")
    lines.append(f"- Both held-out Pareto bars clear simultaneously in {clear_count}/4 seeds.\n\n")
    lines.extend([
        "## Established reference points\n\n",
        "| result | held-out emotion AMI | held-out genre AMI | held-out silhouette |\n",
        "|---|---:|---:|---:|\n",
    ])
    for name, emotion, genre, silhouette in REFERENCES:
        silhouette_text = "never measured" if silhouette is None else f"{silhouette:.4f}"
        lines.append(f"| {name} | {emotion:.4f} | {genre:.4f} | {silhouette_text} |\n")
    lines.append("\n## Final numeric verdict\n\n")
    selected = winner["heldout_post"]
    baseline = REFERENCES[0]
    lines.append(
        f"Seed-42 selected noise_std={winner['noise_std']:g}: held-out "
        f"emotion AMI={selected['emotion_ami']:.4f}, genre "
        f"AMI={selected['genre_ami']:.4f}, silhouette={selected['silhouette']:.4f}. "
        f"It {'beats' if selected['silhouette'] > baseline[3] else 'does not beat'} "
        f"the Attention-h1 baseline silhouette ({baseline[3]:.4f}); signed "
        f"change={selected['silhouette'] - baseline[3]:+.4f}.\n\n"
    )
    for name, emotion, genre, silhouette in REFERENCES[1:]:
        wins = []
        if selected["emotion_ami"] > emotion:
            wins.append("emotion AMI")
        if selected["genre_ami"] > genre:
            wins.append("genre AMI")
        if silhouette is not None and selected["silhouette"] > silhouette:
            wins.append("silhouette")
        comparison = ", ".join(wins) if wins else "no measured axis"
        extra = " Silhouette was never measured for this reference." if silhouette is None else ""
        lines.append(f"Against {name}, the seed-42 selection beats {comparison}.{extra}\n\n")
    robust_baseline = all(result["heldout_post"]["silhouette"] > baseline[3] for result in four_seeds)
    lines.append(
        f"Attention-h1 silhouette improvement is {'robust (4/4 seeds)' if robust_baseline else 'not robust (fewer than 4/4 seeds)'} "
        f"versus {baseline[3]:.4f}; {sum(result['heldout_post']['silhouette'] > baseline[3] for result in four_seeds)}/4 "
        f"seeds beat it. Both Pareto bars clear in {clear_count}/4 seeds. "
        f"The selected value {'is' if validated_winner else 'is not'} a seed-42 "
        "Pareto-bar winner.\n"
    )
    for name, emotion, genre, silhouette in REFERENCES[1:]:
        axis_counts = {
            "emotion AMI": sum(result["heldout_post"]["emotion_ami"] > emotion for result in four_seeds),
            "genre AMI": sum(result["heldout_post"]["genre_ami"] > genre for result in four_seeds),
        }
        if silhouette is not None:
            axis_counts["silhouette"] = sum(
                result["heldout_post"]["silhouette"] > silhouette for result in four_seeds
            )
        counts_text = "; ".join(f"{axis}: {count}/4" for axis, count in axis_counts.items())
        lines.append(f"Against {name}, four-seed axis wins are {counts_text}.\n")
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
    screen = [run_seed(SEED, noise_std, context) for noise_std in NOISE_STDS]
    winner, validated_winner = choose_winner(screen)
    log(
        f"Selected noise_std={winner['noise_std']:g} from seed-42 screen: "
        f"{'Pareto-clear winner' if validated_winner else 'best available, does not clear the Pareto bar'}."
    )
    stress = [run_seed(seed, winner["noise_std"], context) for seed in SEEDS]
    write_report(screen, winner, validated_winner, stress, arch)
    log(f"Wrote report to {REPORT_PATH}.")


if __name__ == "__main__":
    main()
