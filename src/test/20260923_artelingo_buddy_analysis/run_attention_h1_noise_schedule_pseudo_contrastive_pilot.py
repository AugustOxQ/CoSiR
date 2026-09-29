"""Combine embedding noise, cosine LR, and Leiden pseudo-contrastive training.

This adapts the PercepT noise/schedule idea, not its autoencoder recipe:
Attention-h1 has no reconstruction decoder or clean reconstruction target, and its
32-D LayerNorm/L2 embedding lives on the unit hypersphere rather than in an
unconstrained 128-D latent space. Each of the three training InfoNCE losses,
including the new same-community loss, receives its embedding through
noisy_unit_embedding immediately before InfoNCE. For nonzero noise it adds
Gaussian noise and returns the embedding to unit L2 norm, preserving the
temperature's meaning; the reported winner NOISE_STD=0 makes it an identity.
Checkpoint diagnostics and epoch-0/final Leiden, AMI, and silhouette use clean model
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
REPORT_PATH = os.path.join(
    REPORT_OUT_DIR, "attention_h1_noise_schedule_pseudo_contrastive_pilot_report.md"
)

SEED = 42
SEEDS = (7, 123, 2024)
# Four-seed stress winner in attention_h1_noise_schedule_pilot_report.md.
NOISE_STD = 0.0
EMOTION_PARETO_BAR = 0.1236
GENRE_PARETO_BAR = 0.1954
LR_START = 1e-3
LR_FLOOR = 1e-5
RECLUSTER_EVERY = 20
LAMBDA_CLUSTER_MAX = 1.0
CLUSTER_WARMUP_EPOCHS = 50
DOMINANT_FRACTION_GUARD = 0.80
REFERENCES = (
    ("Attention-h1 baseline", 0.1249, 0.2404, 0.0392),
    ("Leiden pseudo-contrastive alone", 0.1165, 0.2351, 0.0877),
    # The noise+schedule row is inserted from literal values in its report below.
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
    arch, model, train_content, train_affect, pipeline, affect_pilot, single_modality,
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
    trajectory = [{
        "epoch": 0, "lr": LR_START, "content_loss": float("nan"),
        "affect_loss": float("nan"), "cluster_loss": float("nan"),
        "lambda_cluster": 0.0, "pseudo_community_count": 0,
        "pseudo_dominant_fraction": float("nan"), **epoch_0_diag,
    }]
    log(
        f"seed={seed} noise_std={noise_std:g} epoch=0 lr={LR_START:.8f} "
        f"content_recall={epoch_0_diag['content_recall']:.4f} "
        f"affect_recall={epoch_0_diag['affect_recall']:.4f}"
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
                arch, model, context["train_content_t"], context["train_affect_t"],
                pipeline, affect_pilot, single_modality, device,
                len(context["paintings"]), log,
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
        lambda_cluster = LAMBDA_CLUSTER_MAX * min(1.0, epoch / CLUSTER_WARMUP_EPOCHS)
        cluster_loss_value = float("nan")
        if current_communities is None:
            total_loss = content_loss + affect_loss
        else:
            cluster_pairs = sample_same_community_pairs(
                current_communities, epoch_rng, arch.BATCH_SIZE
            )
            cluster_embeddings, remapped = arch.content_batch_embeddings(
                model, context["train_content_t"], context["train_affect_t"],
                cluster_pairs, device,
            )
            cluster_loss = arch.symmetric_infonce(
                noisy_unit_embedding(cluster_embeddings, noise_std), remapped, device
            )
            cluster_loss_value = float(cluster_loss.detach().item())
            total_loss = content_loss + affect_loss + lambda_cluster * cluster_loss
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
        checkpoint = {
            "epoch": epoch, "lr": learning_rate,
            "content_loss": float(content_loss.detach().item()),
            "affect_loss": float(affect_loss.detach().item()),
            "cluster_loss": cluster_loss_value,
            "lambda_cluster": lambda_cluster,
            "pseudo_community_count": current_count,
            "pseudo_dominant_fraction": current_dominant,
            **diagnostics,
        }
        trajectory.append(checkpoint)
        log(
            f"seed={seed} noise_std={noise_std:g} epoch={epoch} "
            f"lr={learning_rate:.8f} "
            f"content_recall={checkpoint['content_recall']:.4f} "
            f"affect_recall={checkpoint['affect_recall']:.4f} "
            f"content_loss={checkpoint['content_loss']:.4f} "
            f"affect_loss={checkpoint['affect_loss']:.4f} "
            f"cluster_loss={cluster_loss_value:.4f} lambda_cluster={lambda_cluster:.4f} "
            f"pseudo_communities={current_count} dominant_fraction={current_dominant:.4f}"
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
        "recluster_passes": recluster_passes,
        "skipped_reclusters": skipped,
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


def format_summary(values):
    return f"mean={np.mean(values):.4f}; min={np.min(values):.4f}; max={np.max(values):.4f}"


def noise_sibling_results():
    """Literal winner values from attention_h1_noise_schedule_pilot_report.md."""
    return (0.1306, 0.1973, 0.0488), [
        (42, 0.1306, 0.1973, 0.0488),
        (7, 0.1244, 0.2583, 0.0438),
        (123, 0.1334, 0.2452, 0.0487),
        (2024, 0.1222, 0.2576, 0.0451),
    ]


def comparison_text(value, reference):
    if reference is None:
        return "not measured"
    return f"{'beats' if value > reference else 'does not beat'} {reference:.4f} ({value - reference:+.4f})"


def write_report(seed42, stress, arch):
    all_results = [seed42, *stress]
    sibling_seed42, sibling_stress = noise_sibling_results()
    references = [*REFERENCES[:2]]
    references.append((
        "Noise + schedule alone (winning seed 42)",
        *(sibling_seed42 if sibling_seed42 is not None else (None, None, None)),
    ))
    references.extend(REFERENCES[2:])
    lines = [
        "# Attention-h1 noise, cosine schedule, and Leiden pseudo-contrastive pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Method\n\n",
        "The Attention-h1 student retains both teacher InfoNCE losses and adds the "
        "same-community symmetric InfoNCE term from the Leiden pseudo-contrastive "
        "pilot. All three training losses pass their embeddings through the "
        "noise/renormalization helper immediately before InfoNCE. The selected "
        "noise_std=0 makes that helper an identity in this run. Epoch-0, "
        "checkpoint diagnostics, reclustering, final Leiden, AMI, and silhouette "
        "use clean model forward passes. No noise value or cluster hyperparameter "
        "is screened in this pilot.\n\n",
        f"noise_std={NOISE_STD:g}; recluster every {RECLUSTER_EVERY} epochs "
        f"starting at 1; LAMBDA_CLUSTER_MAX={LAMBDA_CLUSTER_MAX}; "
        f"CLUSTER_WARMUP_EPOCHS={CLUSTER_WARMUP_EPOCHS}; "
        f"DOMINANT_FRACTION_GUARD={DOMINANT_FRACTION_GUARD}; "
        f"batch={arch.BATCH_SIZE}; temperature={arch.TEMPERATURE}. "
        f"Adam starts at {LR_START:g}; CosineAnnealingLR T_max={arch.MAX_EPOCHS}, "
        f"eta_min={LR_FLOOR:g}; one step after each training epoch, never beyond "
        "T_max. LR rows show the rate used for that epoch's update. "
        f"CHECKPOINT_EVERY={arch.CHECKPOINT_EVERY}. Recall-based plateau stopping "
        "matches the sibling pilots. Silhouette uses the pseudo pilot's seed-42 "
        "6,000-point draw and 4,000-point score sample.\n\n",
        "## Seed 42 result\n\n",
        f"Held-out Pareto bar: emotion AMI > {EMOTION_PARETO_BAR:.4f} AND "
        f"genre AMI > {GENRE_PARETO_BAR:.4f}. Seed 42 "
        f"{'clears' if clears_pareto(seed42) else 'does not clear'} it.\n\n",
        "| split | pre emotion AMI | pre genre AMI | pre silhouette | "
        "post emotion AMI | post genre AMI | post silhouette | "
        "pre Leiden communities | post Leiden communities | held-out Pareto bar |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n",
    ]
    for split, pre_key, post_key in (
        ("train", "train_pre", "train_post"),
        ("held-out", "heldout_pre", "heldout_post"),
    ):
        pre, post = seed42[pre_key], seed42[post_key]
        verdict = ("clears" if clears_pareto(seed42) else "does not clear") if split == "held-out" else "n/a"
        lines.append(
            f"| {split} | {pre['emotion_ami']:.4f} | {pre['genre_ami']:.4f} | "
            f"{pre['silhouette']:.4f} | {post['emotion_ami']:.4f} | "
            f"{post['genre_ami']:.4f} | {post['silhouette']:.4f} | "
            f"{pre['communities']} | {post['communities']} | {verdict} |\n"
        )
    lines.append("\n## Full training, LR, and loss trajectories\n\n")
    for result in all_results:
        lines.extend([
            f"### Seed {result['seed']}\n\n",
            f"Stopped: {result['stop_reason']}.\n\n",
            "| epoch | LR used | content loss | affect loss | cluster loss | "
            "cluster weight | pseudo communities | dominant fraction | "
            "content recall | affect recall | top eigen fraction | "
            "effective rank 95% | gate mean | gate std | gate saturated |\n",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        ])
        for row in result["trajectory"]:
            lines.append(
                f"| {row['epoch']} | {row['lr']:.8f} | "
                f"{row['content_loss']:.4f} | {row['affect_loss']:.4f} | "
                f"{row['cluster_loss']:.4f} | {row['lambda_cluster']:.4f} | "
                f"{row['pseudo_community_count']} | {row['pseudo_dominant_fraction']:.4f} | "
                f"{row['content_recall']:.4f} | {row['affect_recall']:.4f} | "
                f"{row['top_eigen_fraction']:.4f} | {row['effective_rank_95']} | "
                f"{row['gate_mean']:.4f} | {row['gate_std']:.4f} | "
                f"{row['gate_saturated_fraction']:.4f} |\n"
            )
        lines.extend([
            "\nSuccessful recluster trajectory (rejected passes retain the prior labels): "
            f"{result['skipped_reclusters']} skipped.\n\n",
            "| epoch | communities | dominant fraction |\n",
            "|---:|---:|---:|\n",
        ])
        for epoch, count, fraction in result["recluster_passes"]:
            lines.append(f"| {epoch} | {count} | {fraction:.4f} |\n")
        lines.append("\n")
    if stress:
        lines.extend([
            "## Four-seed stress\n\n",
            "Seed 42 cleared the bar, so seeds 7, 123, and 2024 were run with "
            "the same fixed noise value and training recipe.\n\n",
            "| seed | train emotion AMI | train genre AMI | train silhouette | "
            "held-out emotion AMI | held-out genre AMI | held-out silhouette | "
            "held-out communities | Pareto bar |\n",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|\n",
        ])
        for result in all_results:
            train, heldout = result["train_post"], result["heldout_post"]
            lines.append(
                f"| {result['seed']} | {train['emotion_ami']:.4f} | "
                f"{train['genre_ami']:.4f} | {train['silhouette']:.4f} | "
                f"{heldout['emotion_ami']:.4f} | {heldout['genre_ami']:.4f} | "
                f"{heldout['silhouette']:.4f} | {heldout['communities']} | "
                f"{'clears' if clears_pareto(result) else 'does not clear'} |\n"
            )
        lines.append("\n")
        for metric, label in (
            ("emotion_ami", "Emotion AMI"),
            ("genre_ami", "Genre AMI"),
            ("silhouette", "Silhouette"),
        ):
            values = [result["heldout_post"][metric] for result in all_results]
            lines.append(f"- {label} across four seeds: {format_summary(values)}.\n")
        clear_count = sum(clears_pareto(result) for result in all_results)
        lines.append(f"- Both held-out Pareto bars clear in {clear_count}/4 seeds.\n\n")
    else:
        lines.append(
            "## Stress decision\n\nSeed 42 missed the held-out Pareto bar. "
            "Stress was skipped; this is a plain miss. No cluster parameters "
            "were retuned.\n\n"
        )
    lines.extend([
        "## Reference comparison (seed 42)\n\n",
        "| result | held-out emotion AMI | held-out genre AMI | held-out silhouette |\n",
        "|---|---:|---:|---:|\n",
    ])
    selected = seed42["heldout_post"]
    for name, emotion, genre, silhouette in references:
        fields = ["not available" if value is None else f"{value:.4f}" for value in (emotion, genre, silhouette)]
        if name == "PercepT replication standing balance-hack":
            fields[2] = "never measured"
        lines.append(f"| {name} | {' | '.join(fields)} |\n")
    lines.append(
        f"| This combination | {selected['emotion_ami']:.4f} | "
        f"{selected['genre_ami']:.4f} | {selected['silhouette']:.4f} |\n\n"
    )
    if sibling_stress:
        lines.extend([
            "Noise + schedule alone, four-seed held-out results from its report:\n\n",
            "| seed | emotion AMI | genre AMI | silhouette |\n",
            "|---:|---:|---:|---:|\n",
        ])
        for seed, emotion, genre, silhouette in sibling_stress:
            lines.append(f"| {seed} | {emotion:.4f} | {genre:.4f} | {silhouette:.4f} |\n")
        lines.append("\n")
    lines.append("## Final numeric verdict\n\n")
    for name, emotion, genre, silhouette in references:
        lines.append(
            f"Against {name}: emotion AMI {comparison_text(selected['emotion_ami'], emotion)}; "
            f"genre AMI {comparison_text(selected['genre_ami'], genre)}; "
            f"silhouette {comparison_text(selected['silhouette'], silhouette)}.\n\n"
        )
    individual = [references[1][3], references[2][3]]
    if all(value is not None for value in individual):
        beats_both = selected["silhouette"] > max(individual)
        below_both = selected["silhouette"] <= min(individual)
        if beats_both:
            lines.append("Combining beats both individual buddy mechanisms on held-out silhouette. ")
        elif below_both:
            lines.append(
                "Combining fails to beat either individual buddy mechanism on held-out "
                "silhouette. The combination underperforms both; this is a negative result. "
            )
        else:
            lines.append("Combining beats only one individual buddy mechanism on held-out silhouette. ")
    else:
        lines.append("A two-way silhouette comparison with both individual buddy mechanisms is unavailable. ")
    lines.append(
        "The axis-by-axis comparisons above show AMI retention or loss against "
        "each individual buddy mechanism and whether either PercepT reference "
        "is beaten on a measured axis.\n"
    )
    os.makedirs(REPORT_OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    if NOISE_STD is None:
        raise RuntimeError(
            "NOISE_STD is None: fill the NOISE_STD constant at the top of this "
            "script with the four-seed stress winner from "
            "attention_h1_noise_schedule_pilot_report.md before running."
        )
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
    log(f"Using {device} for the Attention-h1 noise/schedule/pseudo-contrastive pilot.")

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
    seed42 = run_seed(SEED, NOISE_STD, context)
    if clears_pareto(seed42):
        log("Seed 42 clears both held-out Pareto bars; running three stress seeds.")
        stress = [run_seed(seed, NOISE_STD, context) for seed in SEEDS]
    else:
        log("Seed 42 misses the held-out Pareto bar; skipping stress without retuning.")
        stress = []
    write_report(seed42, stress, arch)
    log(f"Wrote report to {REPORT_PATH}.")


if __name__ == "__main__":
    main()
