"""Train Attention-h1 with its two teacher losses and PercepT's DEC loss.

The seed-42 screen selects one DEC weight for three fresh-seed stress runs.
The report is written only when this GPU pilot is executed.
"""

import importlib.util
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from torch import nn
from torch.nn import functional as F


OUT_DIR = os.path.dirname(os.path.abspath(__file__))
# The architecture sibling's guarded root insert can leave an existing PYTHONPATH
# entry behind site-packages. Preload this package from the root at position zero.
_REPO_ROOT = os.path.abspath(os.path.join(OUT_DIR, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
import src.conditional_buddy.prototype_seed  # noqa: F401

REPORT_OUT_DIR = os.environ.get("PERCEPT_OUTPUT_ROOT") or OUT_DIR
ARCH_SWEEP_PATH = os.path.join(OUT_DIR, "run_learned_student_arch_sweep_pilot.py")
PERCEPT_BASE_PATH = os.path.abspath(os.path.join(
    OUT_DIR, "..", "20260922_percept_topic_pipeline", "run_percept_stage1_pilot.py"
))
REPORT_PATH = os.path.join(REPORT_OUT_DIR, "attention_h1_dec_hybrid_pilot_report.md")

SEED = 42
SEEDS = (7, 123, 2024)
LAMBDA_DEC_VALUES = (0.1, 0.5, 1.0)
DEC_WARMUP_EPOCHS = 30
N_INITIAL_CLUSTERS = 100
N_SURVIVING_CLUSTERS = 67
EMOTION_PARETO_BAR = 0.1236
GENRE_PARETO_BAR = 0.1954
LR_START = 1e-3
LR_FLOOR = 1e-5
REFERENCES = (
    ("Attention-h1 baseline", 0.1249, 0.2404, 0.0392),
    ("Leiden pseudo-contrastive alone", 0.1165, 0.2351, 0.0877),
    ("Noise + schedule alone (winning seed 42)", 0.1306, 0.1973, 0.0488),
    ("Noise + schedule + pseudo-contrastive", 0.1210, 0.2623, 0.0789),
    ("PercepT replication standing balance-hack", 0.1252, 0.2486, None),
    ("PercepT replication faithful recipe", 0.1092, 0.3288, 0.5120),
)
NOISE_SCHEDULE_FOUR_SEED = (
    (42, 0.1306, 0.1973, 0.0488),
    (7, 0.1244, 0.2583, 0.0438),
    (123, 0.1334, 0.2452, 0.0487),
    (2024, 0.1222, 0.2576, 0.0451),
)


def load_module(module_name: str, path: str):
    """Load a standalone sibling pilot without executing its main block."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Exactly the sibling-import pattern used by the PercepT faithful-recipe pilot.
base = load_module("percept_stage1_base_pilot_for_attention_dec", PERCEPT_BASE_PATH)


def capture_embeddings(model, content, affect):
    model.eval()
    with torch.no_grad():
        embeddings, _ = model(content, affect)
    return embeddings.cpu().numpy().astype(np.float32, copy=False)


def hard_assignments(embeddings, centers, device):
    """Use PercepT's Student-t rule on clean embeddings and shared centers."""
    with torch.no_grad():
        latent = torch.as_tensor(embeddings, dtype=centers.dtype, device=device)
        return base.soft_assignments(latent, centers).argmax(dim=1).cpu().numpy()


def cluster_size_diagnostic(q):
    """Count hard train assignments across all 100 centers, before pruning."""
    sizes = torch.bincount(q.argmax(dim=1), minlength=N_INITIAL_CLUSTERS)
    return {
        "min": int(sizes.min().item()),
        "max": int(sizes.max().item()),
        "median": float(np.median(sizes.cpu().numpy())),
        "below_one_percent": int((sizes < 0.01 * q.shape[0]).sum().item()),
    }


def surviving_cluster_diagnostic(assignments):
    """Measure empty and tiny hard-label topics after the 67-center prune."""
    sizes = np.bincount(assignments, minlength=N_SURVIVING_CLUSTERS)
    below_one_percent = int(np.sum(sizes < 0.01 * len(assignments)))
    return {
        "min": int(sizes.min()), "max": int(sizes.max()),
        "median": float(np.median(sizes)),
        "below_one_percent": below_one_percent,
        "collapsed": below_one_percent > N_SURVIVING_CLUSTERS / 2,
    }


def sampled_silhouette(embeddings, assignments):
    # Keep the other buddy pilots' fixed seed-42 two-stage sample convention.
    idx = np.random.default_rng(SEED).choice(
        len(embeddings), size=min(6000, len(embeddings)), replace=False
    )
    sampled_labels = assignments[idx]
    score_size = min(4000, len(idx))
    # Match sklearn's seeded second-stage draw before testing the label count.
    # A rare cluster can occur in the 6,000-point draw but vanish from the
    # 4,000-point score sample, making silhouette_score raise ValueError.
    score_indices = np.random.RandomState(SEED).permutation(len(idx))[:score_size]
    score_labels = sampled_labels[score_indices]
    if not 1 < len(np.unique(score_labels)) < len(score_labels):
        return float("nan")
    return float(silhouette_score(
        embeddings[idx], sampled_labels,
        sample_size=score_size, random_state=SEED,
    ))


def split_metrics(embeddings, assignments, paintings, emotions, genre_map, pipeline):
    genre_indices = [i for i, painting in enumerate(paintings) if painting in genre_map]
    if not genre_indices:
        raise RuntimeError("No genre-labelled paintings overlap this node set.")
    return {
        "emotion_ami": pipeline.external_metrics(assignments, emotions)["AMI"],
        "genre_ami": pipeline.external_metrics(
            assignments[genre_indices],
            [genre_map[paintings[i]] for i in genre_indices],
        )["AMI"],
        "silhouette": sampled_silhouette(embeddings, assignments),
        "communities": len(np.unique(assignments)),
    }


def clears_pareto(result):
    heldout = result["heldout_post"]
    return (heldout["emotion_ami"] > EMOTION_PARETO_BAR
            and heldout["genre_ami"] > GENRE_PARETO_BAR)


def run_seed(seed, lambda_dec, context):
    arch = context["arch"]
    device = context["device"]
    log = arch.log

    # Every screen and stress run has fresh model weights and random state.
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = arch.LearnedStudent("attn1").to(device)
    train_pre = capture_embeddings(model, context["train_content_t"], context["train_affect_t"])
    heldout_pre = capture_embeddings(
        model, context["heldout_content_t"], context["heldout_affect_t"]
    )
    # K-means follows PercepT's clean-latent init, with a per-run seed. Its
    # float32 centers live on the model device and are trained by the same Adam.
    kmeans = KMeans(n_clusters=N_INITIAL_CLUSTERS, n_init=10, random_state=seed)
    kmeans.fit(train_pre)
    centers = nn.Parameter(torch.from_numpy(
        kmeans.cluster_centers_.astype(np.float32)
    ).to(device))
    optimizer = torch.optim.Adam(list(model.parameters()) + [centers], lr=LR_START)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=arch.MAX_EPOCHS, eta_min=LR_FLOOR
    )
    log(f"Starting Attention-h1 DEC hybrid: seed={seed}, LAMBDA_DEC={lambda_dec:g}.")

    train_pre_labels = hard_assignments(train_pre, centers, device)
    heldout_pre_labels = hard_assignments(heldout_pre, centers, device)
    train_pre_metrics = split_metrics(
        train_pre, train_pre_labels, context["paintings"], context["majority_emotion"],
        context["genre_map"], context["pipeline"],
    )
    heldout_pre_metrics = split_metrics(
        heldout_pre, heldout_pre_labels, context["heldout_paintings"],
        context["heldout_majority_emotion"], context["genre_map"],
        context["heldout_pipeline"],
    )
    epoch_0_diag = arch.evaluate_checkpoint(
        model, context["heldout_content_t"], context["heldout_affect_t"],
        context["heldout_content_graph"], context["heldout_affect_graph"],
        context["sampled_nodes"], context["rank_nodes"], context["single_modality"],
        context["heldout_pipeline"], context["affect_pilot"], str(device),
    )
    with torch.no_grad():
        initial_q = base.soft_assignments(
            torch.as_tensor(train_pre, dtype=centers.dtype, device=device), centers
        )
        initial_sizes = cluster_size_diagnostic(initial_q)
    trajectory = [{
        "epoch": 0, "lr": LR_START, "content_loss": float("nan"),
        "affect_loss": float("nan"), "dec_loss": float("nan"),
        "lambda_dec": 0.0, **initial_sizes, **epoch_0_diag,
    }]
    log(
        f"seed={seed} LAMBDA_DEC={lambda_dec:g} epoch=0 "
        f"content_recall={epoch_0_diag['content_recall']:.4f} "
        f"affect_recall={epoch_0_diag['affect_recall']:.4f} "
        f"cluster_min={initial_sizes['min']} cluster_max={initial_sizes['max']} "
        f"cluster_median={initial_sizes['median']:.1f} "
        f"below_one_percent={initial_sizes['below_one_percent']}/{N_INITIAL_CLUSTERS}"
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
            content_embeddings, remapped_content_pairs, device
        )
        affect_embeddings, _mixing_weights = model(
            context["train_content_t"], context["train_affect_t"]
        )
        affect_loss = arch.symmetric_infonce(affect_embeddings, affect_pairs, device)
        # DEC's self-sharpening can degenerate if it dominates before the
        # two teacher losses establish structure; ramp its weight over 30 epochs.
        dec_weight = lambda_dec * min(1.0, epoch / DEC_WARMUP_EPOCHS)
        q = base.soft_assignments(affect_embeddings, centers)
        p = base.target_distribution(q)
        dec_loss = F.kl_div(q.log(), p, reduction="batchmean")
        total_loss = content_loss + affect_loss + dec_weight * dec_loss
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        learning_rate = optimizer.param_groups[0]["lr"]
        optimizer.step()
        # Never advance beyond T_max, where CosineAnnealingLR would rise again.
        if epoch <= arch.MAX_EPOCHS:
            scheduler.step()
        if epoch % arch.CHECKPOINT_EVERY:
            continue
        with torch.no_grad():
            # q is the clean training forward from this epoch, before the
            # optimizer step. No third train forward is needed for diagnostics.
            sizes = cluster_size_diagnostic(q.detach())
        diagnostics = arch.evaluate_checkpoint(
            model, context["heldout_content_t"], context["heldout_affect_t"],
            context["heldout_content_graph"], context["heldout_affect_graph"],
            context["sampled_nodes"], context["rank_nodes"], context["single_modality"],
            context["heldout_pipeline"], context["affect_pilot"], str(device),
        )
        checkpoint = {
            "epoch": epoch, "lr": learning_rate,
            "content_loss": float(content_loss.detach().item()),
            "affect_loss": float(affect_loss.detach().item()),
            "dec_loss": float(dec_loss.detach().item()),
            "lambda_dec": dec_weight, **sizes, **diagnostics,
        }
        trajectory.append(checkpoint)
        log(
            f"seed={seed} LAMBDA_DEC={lambda_dec:g} epoch={epoch} "
            f"lr={learning_rate:.8f} dec_loss={checkpoint['dec_loss']:.6f} "
            f"lambda_dec={dec_weight:.4f} "
            f"content_recall={checkpoint['content_recall']:.4f} "
            f"affect_recall={checkpoint['affect_recall']:.4f} "
            f"cluster_min={sizes['min']} cluster_max={sizes['max']} "
            f"cluster_median={sizes['median']:.1f} "
            f"below_one_percent={sizes['below_one_percent']}/{N_INITIAL_CLUSTERS}"
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

    # DEC's centers define the topics. Prune exactly once and reuse those
    # surviving centers on both splits; no Leiden graph or pass is involved.
    surviving_centers, surviving_indices = base.prune_centers(centers)
    if len(surviving_indices) != N_SURVIVING_CLUSTERS:
        raise RuntimeError("PercepT pruning did not return 67 centers.")
    log(
        f"seed={seed}: retained 67 highest-norm DEC centers "
        f"(original indices {surviving_indices.tolist()})."
    )
    train_post = capture_embeddings(model, context["train_content_t"], context["train_affect_t"])
    heldout_post = capture_embeddings(
        model, context["heldout_content_t"], context["heldout_affect_t"]
    )
    train_post_labels = hard_assignments(train_post, surviving_centers, device)
    heldout_post_labels = hard_assignments(heldout_post, surviving_centers, device)
    result = {
        "seed": seed, "lambda_dec": lambda_dec,
        "train_pre": train_pre_metrics, "heldout_pre": heldout_pre_metrics,
        "train_post": split_metrics(
            train_post, train_post_labels, context["paintings"],
            context["majority_emotion"], context["genre_map"], context["pipeline"],
        ),
        "heldout_post": split_metrics(
            heldout_post, heldout_post_labels, context["heldout_paintings"],
            context["heldout_majority_emotion"], context["genre_map"],
            context["heldout_pipeline"],
        ),
        "train_surviving_sizes": surviving_cluster_diagnostic(train_post_labels),
        "heldout_surviving_sizes": surviving_cluster_diagnostic(heldout_post_labels),
        "trajectory": trajectory, "stop_reason": stop_reason,
    }
    log(
        f"seed={seed} LAMBDA_DEC={lambda_dec:g} final train emotion "
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
    # Undefined silhouette after complete collapse cannot win over a measured one.
    winner = max(candidates, key=lambda result: (
        result["heldout_post"]["silhouette"]
        if np.isfinite(result["heldout_post"]["silhouette"]) else float("-inf")
    ))
    return winner, bool(clearers)


def format_summary(values):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return "undefined (all assignments collapsed)"
    return f"mean={np.mean(finite):.4f}; min={np.min(finite):.4f}; max={np.max(finite):.4f}"


def format_metric(value):
    return f"{value:.4f}" if np.isfinite(value) else "n/a (collapsed)"


def write_report(screen, winner, validated_winner, stress, arch):
    """Write the predeclared comparison report after GPU execution."""
    four_seeds = [winner, *stress]
    clear_count = sum(clears_pareto(result) for result in four_seeds)
    selected = winner["heldout_post"]
    lines = [
        "# Attention-h1 + PercepT DEC-loss hybrid pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Method\n\n",
        "Attention-h1 retains the content and affect teacher-graph symmetric InfoNCE "
        "losses. PercepT's unchanged Student-t soft assignments, detached sharpened "
        "target, and highest-norm-center pruning supply the DEC mechanism. "
        "The full clean train embedding supplies the DEC loss each epoch; the "
        "centers and student weights share one optimizer. There is no embedding "
        "noise, discrete reclustering, or Leiden pass.\n\n",
        f"Each run initializes {N_INITIAL_CLUSTERS} centers by KMeans(n_init=10, "
        "random_state=seed) on its own clean epoch-0 train embedding, then keeps "
        f"{N_SURVIVING_CLUSTERS} highest-norm trained centers. LAMBDA_DEC is "
        f"screened over {LAMBDA_DEC_VALUES}, linearly warmed up over "
        f"{DEC_WARMUP_EPOCHS} epochs to avoid premature self-sharpening collapse. "
        f"Adam starts at {LR_START:g}; cosine T_max={arch.MAX_EPOCHS}, "
        f"eta_min={LR_FLOOR:g}; checkpoint cadence={arch.CHECKPOINT_EVERY}. "
        "The plateau stopping rule follows the noise-schedule sibling. "
        "Checkpoint all-100 cluster sizes come from that epoch's clean train "
        "forward before the optimizer step; checkpoint recalls evaluate the "
        "updated model. An early recall plateau may stop before the 30-epoch "
        "DEC warmup completes. "
        "Silhouette uses a seed-42 draw of at most 6,000 embeddings and "
        "silhouette_score with a 4,000-point sample and random_state=42. "
        "Undefined silhouette from one-cluster collapse is reported as n/a.\n\n",
        "## Single-seed screen (seed 42)\n\n",
        f"Held-out Pareto bar: emotion AMI > {EMOTION_PARETO_BAR:.4f} AND "
        f"genre AMI > {GENRE_PARETO_BAR:.4f}.\n\n",
        "| LAMBDA_DEC | split | pre emotion AMI | pre genre AMI | pre silhouette | "
        "post emotion AMI | post genre AMI | post silhouette | pre DEC clusters | "
        "post DEC clusters | held-out Pareto bar |\n",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n",
    ]
    for result in screen:
        for split, pre_key, post_key in (
            ("train", "train_pre", "train_post"),
            ("held-out", "heldout_pre", "heldout_post"),
        ):
            pre, post = result[pre_key], result[post_key]
            verdict = (
                "clears" if clears_pareto(result) else "does not clear"
            ) if split == "held-out" else "n/a"
            lines.append(
                f"| {result['lambda_dec']:g} | {split} | "
                f"{pre['emotion_ami']:.4f} | {pre['genre_ami']:.4f} | "
                f"{format_metric(pre['silhouette'])} | "
                f"{post['emotion_ami']:.4f} | {post['genre_ami']:.4f} | "
                f"{format_metric(post['silhouette'])} | {pre['communities']} | "
                f"{post['communities']} | {verdict} |\n"
            )
    lines.append("\n## Full training, collapse, and recall trajectories\n\n")
    for result in [*screen, *stress]:
        lines.extend([
            f"### LAMBDA_DEC={result['lambda_dec']:g}, seed {result['seed']}\n\n",
            f"Stopped: {result['stop_reason']}. All-100-center sizes use clean "
            "train embeddings before pruning.\n\n",
            "| epoch | LR used | content loss | affect loss | raw DEC KL | DEC weight | "
            "content recall | affect recall | all-100 min | all-100 max | "
            "all-100 median | all-100 below 1% | top eigen fraction | "
            "effective rank 95% | gate mean | gate std | gate saturated |\n",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        ])
        for row in result["trajectory"]:
            lines.append(
                f"| {row['epoch']} | {row['lr']:.8f} | "
                f"{row['content_loss']:.4f} | {row['affect_loss']:.4f} | "
                f"{row['dec_loss']:.6f} | {row['lambda_dec']:.4f} | "
                f"{row['content_recall']:.4f} | {row['affect_recall']:.4f} | "
                f"{row['min']:,} | {row['max']:,} | {row['median']:.1f} | "
                f"{row['below_one_percent']}/{N_INITIAL_CLUSTERS} | "
                f"{row['top_eigen_fraction']:.4f} | {row['effective_rank_95']} | "
                f"{row['gate_mean']:.4f} | {row['gate_std']:.4f} | "
                f"{row['gate_saturated_fraction']:.4f} |\n"
            )
        lines.append("\n")
    lines.append("## Final surviving-center collapse diagnostics\n\n")
    lines.append(
        "These counts use hard labels after pruning to 67 centers. Collapse means "
        "more than half of centers have below 1% of assigned nodes.\n\n"
    )
    lines.append(
        "| LAMBDA_DEC | seed | split | min | max | median | below 1% / 67 | collapsed |\n"
    )
    lines.append("|---:|---:|---|---:|---:|---:|---:|---|\n")
    for result in [*screen, *stress]:
        for split, key in (("train", "train_surviving_sizes"),
                           ("held-out", "heldout_surviving_sizes")):
            diagnostic = result[key]
            lines.append(
                f"| {result['lambda_dec']:g} | {result['seed']} | {split} | "
                f"{diagnostic['min']:,} | {diagnostic['max']:,} | "
                f"{diagnostic['median']:.1f} | "
                f"{diagnostic['below_one_percent']}/{N_SURVIVING_CLUSTERS} | "
                f"{'yes' if diagnostic['collapsed'] else 'no'} |\n"
            )
    lines.append("\n")
    lines.append("## Winner selection\n\n")
    if validated_winner:
        clear_values = ", ".join(
            f"{result['lambda_dec']:g}" for result in screen if clears_pareto(result)
        )
        lines.append(
            f"Pareto-bar clearers: {clear_values}. Selected LAMBDA_DEC="
            f"{winner['lambda_dec']:g}, with the highest held-out silhouette among "
            f"clearers ({format_metric(selected['silhouette'])}). This is the "
            "seed-42 winner; robustness is decided by the stress seeds below.\n\n"
        )
    else:
        if any(np.isfinite(result["heldout_post"]["silhouette"]) for result in screen):
            reason = (
                f"highest held-out silhouette among all three "
                f"({format_metric(selected['silhouette'])})"
            )
        else:
            reason = (
                "all three held-out silhouettes are undefined after collapse; "
                "the first sweep value is selected by the declared order"
            )
        lines.append(
            "No screened value clears the held-out Pareto bar. Selected LAMBDA_DEC="
            f"{winner['lambda_dec']:g} for {reason}: best available, "
            "does not clear the Pareto bar. This is not a validated winner. "
            "Only this selected value receives seed stress.\n\n"
        )
    lines.extend([
        "## Four-seed stress of the selected LAMBDA_DEC\n\n",
        f"Only LAMBDA_DEC={winner['lambda_dec']:g} was run at seeds 7, 123, "
        "and 2024; seed 42 is the screen result, not a repeat fit.\n\n",
        "| seed | train emotion AMI | train genre AMI | train silhouette | "
        "held-out emotion AMI | held-out genre AMI | held-out silhouette | "
        "held-out DEC clusters | held-out Pareto bar |\n",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|\n",
    ])
    for result in four_seeds:
        train, heldout = result["train_post"], result["heldout_post"]
        lines.append(
            f"| {result['seed']} | {train['emotion_ami']:.4f} | "
            f"{train['genre_ami']:.4f} | {format_metric(train['silhouette'])} | "
            f"{heldout['emotion_ami']:.4f} | {heldout['genre_ami']:.4f} | "
            f"{format_metric(heldout['silhouette'])} | {heldout['communities']} | "
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
    lines.append(
        f"| This DEC hybrid (seed 42) | {selected['emotion_ami']:.4f} | "
        f"{selected['genre_ami']:.4f} | {format_metric(selected['silhouette'])} |\n\n"
    )
    lines.extend([
        "Noise + schedule alone, four-seed held-out results from its report:\n\n",
        "| seed | emotion AMI | genre AMI | silhouette |\n",
        "|---:|---:|---:|---:|\n",
    ])
    for seed, emotion, genre, silhouette in NOISE_SCHEDULE_FOUR_SEED:
        lines.append(f"| {seed} | {emotion:.4f} | {genre:.4f} | {silhouette:.4f} |\n")
    lines.append("\n## Final numeric verdict\n\n")
    for name, emotion, genre, silhouette in REFERENCES:
        comparisons = [
            f"emotion AMI {'beats' if selected['emotion_ami'] > emotion else 'does not beat'} "
            f"{emotion:.4f} ({selected['emotion_ami'] - emotion:+.4f})",
            f"genre AMI {'beats' if selected['genre_ami'] > genre else 'does not beat'} "
            f"{genre:.4f} ({selected['genre_ami'] - genre:+.4f})",
        ]
        if silhouette is None:
            comparisons.append("silhouette never measured")
        elif np.isfinite(selected["silhouette"]):
            comparisons.append(
                f"silhouette {'beats' if selected['silhouette'] > silhouette else 'does not beat'} "
                f"{silhouette:.4f} ({selected['silhouette'] - silhouette:+.4f})"
            )
        else:
            comparisons.append("silhouette undefined (collapsed)")
        lines.append(f"Against {name}: {'; '.join(comparisons)}.\n\n")
    faithful_silhouette = REFERENCES[-1][3]
    baseline_silhouette = REFERENCES[0][3]
    gap = faithful_silhouette - baseline_silhouette
    # The brief asks for a plain meaningful-gap verdict but gives no cutoff.
    # Declare a 10% gap-closure marker so the wording is reproducible.
    meaningful_fraction = 0.10
    if np.isfinite(selected["silhouette"]):
        fraction = (selected["silhouette"] - baseline_silhouette) / gap
        lines.append(
            f"Seed-42 selected hybrid closes {fraction:.1%} of the held-out silhouette "
            f"gap from Attention-h1 ({baseline_silhouette:.4f}) to faithful PercepT "
            f"({faithful_silhouette:.4f}); signed difference to faithful PercepT="
            f"{selected['silhouette'] - faithful_silhouette:+.4f}. "
            f"With a declared 10% gap-closure marker, this "
            f"{'is' if fraction >= meaningful_fraction else 'is not'} a meaningful "
            "fraction of the gap. "
        )
        lines.append(
            f"It {'exceeds' if selected['silhouette'] > max(REFERENCES[1][3], REFERENCES[2][3], REFERENCES[3][3]) else 'does not exceed'} "
            "every previously measured buddy-variant silhouette. "
        )
    else:
        lines.append("Seed-42 selected hybrid has undefined silhouette due to collapse. ")
    measured_buddy_silhouettes = [value for _, _, _, value in REFERENCES[:4]]
    if not np.isfinite(selected["silhouette"]):
        lines.append("The selected DEC assignment collapsed to an undefined silhouette. ")
    elif winner["heldout_surviving_sizes"]["collapsed"]:
        lines.append(
            "The selected held-out 67-center assignment meets the cluster-size "
            "collapse rule despite a finite silhouette; this is a negative result. "
        )
    elif selected["silhouette"] <= min(measured_buddy_silhouettes):
        lines.append(
            "The selected DEC hybrid underperforms every measured buddy variant "
            "on held-out silhouette; this is a negative result. "
        )
    robust_count = sum(
        clears_pareto(result)
        and not result["heldout_surviving_sizes"]["collapsed"]
        and np.isfinite(result["heldout_post"]["silhouette"])
        and (result["heldout_post"]["silhouette"] - baseline_silhouette) / gap
        >= meaningful_fraction
        for result in four_seeds
    )
    lines.append(
        f"It {'clears' if validated_winner else 'does not clear'} the seed-42 "
        f"held-out AMI Pareto bar; {clear_count}/4 seeds clear both bars. "
        f"{sum(np.isfinite(result['heldout_post']['silhouette']) and result['heldout_post']['silhouette'] > baseline_silhouette for result in four_seeds)}/4 "
        "seeds beat the Attention-h1 baseline silhouette. "
        f"{sum(np.isfinite(result['heldout_post']['silhouette']) and result['heldout_post']['silhouette'] > REFERENCES[1][3] for result in four_seeds)}/4 "
        "seeds beat the strongest measured buddy variant (Leiden pseudo-contrastive). "
        f"Meaningful gap closure, both AMI bars, and noncollapse hold together in {robust_count}/4 "
        f"seeds: {'robust across all four seeds' if robust_count == 4 else 'not robust across four seeds; any seed-42 success is seed-dependent'}.\n"
    )
    os.makedirs(REPORT_OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    arch = load_module("attention_h1_dec_arch_sweep", ARCH_SWEEP_PATH)
    pipeline = arch.load_sibling_module("artelingo_run_pipeline_attn_dec", arch.PIPELINE_PATH)
    affect_pilot = arch.load_sibling_module("artelingo_run_affect_pilot_attn_dec", arch.AFFECT_PILOT_PATH)
    single_modality = arch.load_sibling_module(
        "artelingo_run_single_modality_attn_dec", arch.SINGLE_MODALITY_PATH
    )
    cca_audit = arch.load_sibling_module("artelingo_run_cca_audit_attn_dec", arch.CCA_AUDIT_PATH)
    # evaluate_checkpoint refers to this module-level global in the arch sibling.
    arch.cca_audit = cca_audit
    heldout_pipeline = arch.load_sibling_module(
        "artelingo_run_pipeline_attn_dec_heldout", arch.PIPELINE_PATH
    )
    heldout_pipeline.STORAGE_DIR = arch.HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = arch.HELDOUT_JSON
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = arch.log
    log(f"Using {device} for the Attention-h1 DEC hybrid pilot.")

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
    screen = [run_seed(SEED, value, context) for value in LAMBDA_DEC_VALUES]
    winner, validated_winner = choose_winner(screen)
    log(
        f"Selected LAMBDA_DEC={winner['lambda_dec']:g} from seed-42 screen: "
        f"{'Pareto-clear winner' if validated_winner else 'best available, does not clear the Pareto bar'}."
    )
    # As in the noise-schedule screen, stress only the selected value even if
    # no screened value clears the Pareto bar; label that fallback explicitly.
    stress = [run_seed(seed, winner["lambda_dec"], context) for seed in SEEDS]
    write_report(screen, winner, validated_winner, stress, arch)
    log(f"Wrote report to {REPORT_PATH}.")


if __name__ == "__main__":
    main()
