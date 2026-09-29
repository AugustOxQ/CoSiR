"""Screen wider Attention-h1 shared embeddings, then stress one clear winner.

Run this GPU pilot explicitly. It preserves the embedding-snapshot pilot's
two-teacher training recipe and generates its Markdown report at run time.
"""

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
REPO_ROOT = os.path.abspath(os.path.join(OUT_DIR, "..", "..", ".."))
# Match the snapshot pilot's script-mode import fix before loading arch.
sys.path.insert(0, REPO_ROOT)
import src.conditional_buddy.prototype_seed  # noqa: F401

ARCH_SWEEP_PATH = os.path.join(OUT_DIR, "run_learned_student_arch_sweep_pilot.py")
BASELINE_SNAPSHOT_PATH = os.path.join(OUT_DIR, "attention_h1_embedding_snapshot.npz")
REPORT_OUT_DIR = os.environ.get("PERCEPT_OUTPUT_ROOT") or OUT_DIR
REPORT_PATH = os.path.join(REPORT_OUT_DIR, "dshared_capacity_sweep_pilot_report.md")

D_SHARED_VALUES = (64, 128)
SCREEN_SEED = 42
STRESS_SEEDS = (7, 123, 2024)
EMOTION_BAR = 0.1236
GENRE_BAR = 0.1954
BASELINE_SILHOUETTE = 0.0377
# Larger than the baseline's observed four-seed range (0.0063) and the
# schedule-only seed-42 increase (0.0111); fixed before inspecting this run.
MIN_SILHOUETTE_GAIN = 0.0200
PERCEPT_SILHOUETTE = 0.5120  # Different latent/label protocol; context only.


def load_module(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load sibling module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def occupancy(labels: np.ndarray) -> dict:
    """Summarize observed Leiden communities on the full split."""
    _, sizes = np.unique(labels, return_counts=True)
    return {
        "count": len(sizes),
        "min": int(sizes.min()),
        "max": int(sizes.max()),
        "median": float(np.median(sizes)),
        "below_1pct": int(np.sum(sizes < 0.01 * len(labels))),
    }


def sampled_silhouette(embeddings: np.ndarray, communities: np.ndarray) -> float:
    """Use the baseline stress pilot's seed-42, 6000-then-4000 convention."""
    idx = np.random.default_rng(42).choice(
        len(embeddings), size=min(6000, len(embeddings)), replace=False
    )
    if len(np.unique(communities[idx])) < 2:
        return float("nan")
    try:
        return float(silhouette_score(
            embeddings[idx], communities[idx],
            sample_size=min(4000, len(idx)), random_state=42,
        ))
    except ValueError as exc:
        # The inner sample can lose a rare second community.
        if "Number of labels" not in str(exc):
            raise
        return float("nan")


def communities_for(arch, embeddings, label, pipeline_module, context, seed):
    graph = context["single_modality"].build_single_modality_graph(
        label, embeddings, pipeline_module, context["affect_pilot"],
        str(context["device"]), expected_nodes=len(embeddings),
    )
    return np.asarray(arch.detect_communities(graph, seed=seed))


def split_amis(communities, paintings, emotions, genre_map, pipeline_module):
    genre_indices = [i for i, painting in enumerate(paintings) if painting in genre_map]
    if not genre_indices:
        raise RuntimeError("No genre-labelled paintings overlap this split.")
    return {
        "emotion": float(pipeline_module.external_metrics(communities, emotions)["AMI"]),
        "genre": float(pipeline_module.external_metrics(
            communities[genre_indices],
            [genre_map[paintings[i]] for i in genre_indices],
        )["AMI"]),
    }


def clear_pareto(row: dict) -> bool:
    return row["emotion"] > EMOTION_BAR and row["genre"] > GENRE_BAR


def screen_winner(rows: list[dict]):
    eligible = [
        row for row in rows
        if clear_pareto(row)
        and np.isfinite(row["silhouette"])
        and row["silhouette"] - BASELINE_SILHOUETTE >= MIN_SILHOUETTE_GAIN
    ]
    return max(eligible, key=lambda row: (row["silhouette"], -row["width"]), default=None)


class SavedEmbeddingModel:
    """Present the saved baseline embedding to arch.evaluate_checkpoint.

    The helper's attention summary is discarded; its graph recalls and
    covariance diagnostics depend only on the supplied embedding.
    """

    is_attention = True

    def __init__(self, embedding: np.ndarray):
        self.embedding = torch.as_tensor(embedding)

    def eval(self):
        return self

    def __call__(self, _content, _affect):
        weights = torch.zeros((len(self.embedding), 2, 2), dtype=torch.float32)
        return self.embedding, weights


def evaluate_diagnostics(arch, model, context, sampled_nodes, rank_nodes):
    return arch.evaluate_checkpoint(
        model, context["heldout_content_t"], context["heldout_affect_t"],
        context["heldout_content_graph"], context["heldout_affect_graph"],
        sampled_nodes, rank_nodes, context["single_modality"],
        context["heldout_pipeline"], context["affect_pilot"], str(context["device"]),
    )


def diagnostic_nodes(arch, context, seed):
    rng = np.random.default_rng(seed)
    n = len(context["heldout_paintings"])
    return (
        rng.choice(n, size=arch.EDGE_SAMPLE_SIZE, replace=False),
        rng.choice(n, size=arch.EFFECTIVE_RANK_SAMPLE_SIZE, replace=False),
    )


def cited_baseline(arch, context):
    """Use cited scores and recover missing diagnostics from the saved fit."""
    if not os.path.isfile(BASELINE_SNAPSHOT_PATH):
        raise FileNotFoundError(
            f"The cited seed-42 snapshot is required for baseline diagnostics: "
            f"{BASELINE_SNAPSHOT_PATH}"
        )
    with np.load(BASELINE_SNAPSHOT_PATH, allow_pickle=True) as snapshot:
        if int(snapshot["seed"]) != SCREEN_SEED:
            raise RuntimeError("The saved baseline snapshot is not seed 42.")
        if not np.array_equal(snapshot["heldout_paintings"], context["heldout_paintings"]):
            raise RuntimeError("Saved baseline held-out painting order does not match.")
        if not np.array_equal(snapshot["train_paintings"], context["paintings"]):
            raise RuntimeError("Saved baseline train painting order does not match.")
        baseline_embedding = np.asarray(snapshot["heldout_embedding_post"])
        train_occupancy = occupancy(snapshot["train_community_post"])
        heldout_occupancy = occupancy(snapshot["heldout_community_post"])
    sampled_nodes, rank_nodes = diagnostic_nodes(arch, context, SCREEN_SEED)
    diagnostics = evaluate_diagnostics(
        arch, SavedEmbeddingModel(baseline_embedding), context, sampled_nodes, rank_nodes
    )
    return {
        "width": 32, "seed": SCREEN_SEED, "emotion": 0.1249, "genre": 0.2404,
        "silhouette": BASELINE_SILHOUETTE, "train_occupancy": train_occupancy,
        "heldout_occupancy": heldout_occupancy, "diagnostics": diagnostics,
        "stop_reason": "cited baseline; no retraining",
    }


def prepare_context(arch, seed):
    pipeline = arch.load_sibling_module("dshared_train_pipeline", arch.PIPELINE_PATH)
    affect_pilot = arch.load_sibling_module("dshared_affect_pilot", arch.AFFECT_PILOT_PATH)
    single_modality = arch.load_sibling_module(
        "dshared_single_modality", arch.SINGLE_MODALITY_PATH
    )
    cca_audit = arch.load_sibling_module("dshared_cca_audit", arch.CCA_AUDIT_PATH)
    arch.cca_audit = cca_audit  # evaluate_checkpoint resolves this module global.
    heldout_pipeline = arch.load_sibling_module(
        "dshared_heldout_pipeline", arch.PIPELINE_PATH
    )
    heldout_pipeline.STORAGE_DIR = arch.HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = arch.HELDOUT_JSON
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch.log(f"Using {device} for D_SHARED capacity sweep.")

    pipeline.assert_extraction_complete()
    paintings, img_nodes, txt_nodes, emotion_counts = pipeline.load_dedup_features()
    majority_emotion = [pipeline.majority(counts) for counts in emotion_counts]
    affect_train = np.asarray(
        affect_pilot.extract_affect_nodes(pipeline.TRAIN_JSON, paintings, str(device)),
        dtype=np.float64,
    )
    heldout_pipeline.assert_extraction_complete()
    heldout_paintings, heldout_img, heldout_txt, heldout_emotion_counts = (
        heldout_pipeline.load_dedup_features()
    )
    heldout_majority_emotion = [
        heldout_pipeline.majority(counts) for counts in heldout_emotion_counts
    ]
    affect_heldout = np.asarray(
        affect_pilot.extract_affect_nodes(arch.HELDOUT_JSON, heldout_paintings, str(device)),
        dtype=np.float64,
    )
    content_train = cca_audit.content_features(img_nodes, txt_nodes, affect_pilot)
    content_heldout = cca_audit.content_features(heldout_img, heldout_txt, affect_pilot)
    pca = PCA(n_components=arch.CONTENT_PCA_DIM, random_state=seed)
    content_train = pca.fit_transform(content_train).astype(np.float32)
    content_heldout = pca.transform(content_heldout).astype(np.float32)

    _img_graph, _txt_graph, content_teacher_graph = pipeline.build_buddy_graphs(
        img_nodes, txt_nodes, K=pipeline.K, alpha=pipeline.ALPHA,
        device=str(device), connect_components=True,
    )
    affect_teacher_graph = single_modality.build_single_modality_graph(
        "train-affect-teacher", affect_train, pipeline, affect_pilot, str(device),
        expected_nodes=len(paintings),
    )
    heldout_content_graph = single_modality.build_single_modality_graph(
        "held-out-content-reference", content_heldout, heldout_pipeline,
        affect_pilot, str(device), expected_nodes=len(heldout_paintings),
    )
    heldout_affect_graph = single_modality.build_single_modality_graph(
        "held-out-affect-reference", affect_heldout, heldout_pipeline,
        affect_pilot, str(device), expected_nodes=len(heldout_paintings),
    )
    return {
        "pipeline": pipeline, "heldout_pipeline": heldout_pipeline,
        "affect_pilot": affect_pilot, "single_modality": single_modality,
        "device": device, "paintings": paintings, "heldout_paintings": heldout_paintings,
        "majority_emotion": majority_emotion,
        "heldout_majority_emotion": heldout_majority_emotion,
        "genre_map": pipeline.load_genre_map(),
        "content_edges": arch.upper_triangle_edges(content_teacher_graph),
        "affect_edges": arch.upper_triangle_edges(affect_teacher_graph),
        "heldout_content_graph": heldout_content_graph,
        "heldout_affect_graph": heldout_affect_graph,
        "train_content_t": torch.as_tensor(content_train, dtype=torch.float32, device=device),
        "train_affect_t": torch.as_tensor(affect_train, dtype=torch.float32, device=device),
        "heldout_content_t": torch.as_tensor(content_heldout, dtype=torch.float32, device=device),
        "heldout_affect_t": torch.as_tensor(affect_heldout, dtype=torch.float32, device=device),
    }


def run_width(arch, context, width, seed):
    arch.CURRENT_ARCHITECTURE = f"D_SHARED={width}, seed={seed}"
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # D_SHARED is read only in the architecture constructors. No source edit.
    arch.D_SHARED = width
    model = arch.LearnedStudent("attn1").to(context["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=arch.LEARNING_RATE)
    sampled_nodes, rank_nodes = diagnostic_nodes(arch, context, seed)
    previous = evaluate_diagnostics(arch, model, context, sampled_nodes, rank_nodes)
    plateau_count = 0
    stop_reason = f"reached MAX_EPOCHS={arch.MAX_EPOCHS}"
    final_diagnostics = previous
    for epoch in range(1, arch.MAX_EPOCHS + 1):
        model.train()
        epoch_rng = np.random.default_rng(seed + epoch)
        content_pairs = arch.sample_positive_pairs(context["content_edges"], epoch_rng)
        affect_pairs = arch.sample_positive_pairs(context["affect_edges"], epoch_rng)
        content_embeddings, remapped_pairs = arch.content_batch_embeddings(
            model, context["train_content_t"], context["train_affect_t"],
            content_pairs, context["device"],
        )
        content_loss = arch.symmetric_infonce(
            content_embeddings, remapped_pairs, context["device"]
        )
        affect_embeddings, _ = model(context["train_content_t"], context["train_affect_t"])
        affect_loss = arch.symmetric_infonce(affect_embeddings, affect_pairs, context["device"])
        optimizer.zero_grad(set_to_none=True)
        (content_loss + affect_loss).backward()
        optimizer.step()
        if epoch % arch.CHECKPOINT_EVERY:
            continue
        final_diagnostics = evaluate_diagnostics(arch, model, context, sampled_nodes, rank_nodes)
        arch.log(
            f"epoch={epoch} content_recall={final_diagnostics['content_recall']:.4f} "
            f"affect_recall={final_diagnostics['affect_recall']:.4f}"
        )
        content_plateau = (
            arch.relative_improvement(
                final_diagnostics["content_recall"], previous["content_recall"]
            ) < arch.PLATEAU_REL_IMPROVEMENT
        )
        affect_plateau = (
            arch.relative_improvement(
                final_diagnostics["affect_recall"], previous["affect_recall"]
            ) < arch.PLATEAU_REL_IMPROVEMENT
        )
        plateau_count = plateau_count + 1 if content_plateau and affect_plateau else 0
        previous = final_diagnostics
        if plateau_count >= arch.PLATEAU_WINDOW:
            stop_reason = (
                f"both recalls plateaued for {arch.PLATEAU_WINDOW} consecutive checkpoints"
            )
            arch.log(f"Stopping at epoch {epoch}: {stop_reason}.")
            break

    model.eval()
    with torch.no_grad():
        train_embedding, _ = model(context["train_content_t"], context["train_affect_t"])
        heldout_embedding, _ = model(context["heldout_content_t"], context["heldout_affect_t"])
    train_embedding = train_embedding.cpu().numpy().astype(np.float32, copy=False)
    heldout_embedding = heldout_embedding.cpu().numpy().astype(np.float32, copy=False)
    train_communities = communities_for(
        arch, train_embedding, "final-train-attn1", context["pipeline"], context, seed
    )
    heldout_communities = communities_for(
        arch, heldout_embedding, "final-heldout-attn1",
        context["heldout_pipeline"], context, seed,
    )
    heldout_amis = split_amis(
        heldout_communities, context["heldout_paintings"],
        context["heldout_majority_emotion"], context["genre_map"],
        context["heldout_pipeline"],
    )
    row = {
        "width": width, "seed": seed, **heldout_amis,
        "silhouette": sampled_silhouette(heldout_embedding, heldout_communities),
        "train_occupancy": occupancy(train_communities),
        "heldout_occupancy": occupancy(heldout_communities),
        "diagnostics": final_diagnostics, "stop_reason": stop_reason,
    }
    arch.log(
        f"Completed: emotion AMI={row['emotion']:.4f}, genre AMI={row['genre']:.4f}, "
        f"silhouette={row['silhouette']:.4f}, Pareto={clear_pareto(row)}."
    )
    return row


def occupancy_text(value):
    return (
        f"{value['count']} communities; {value['min']}/{value['median']:.1f}/"
        f"{value['max']} min/median/max; {value['below_1pct']}/{value['count']} below 1%"
    )


def metric_text(value):
    return f"{value:.4f}" if np.isfinite(value) else "n/a"


def report_row(row):
    diagnostics = row["diagnostics"]
    return (
        f"| {row['width']} | {row['seed']} | {diagnostics['effective_rank_95']} | "
        f"{diagnostics['top_eigen_fraction']:.4f} | "
        f"{diagnostics['content_recall']:.4f} | {diagnostics['affect_recall']:.4f} | "
        f"{row['emotion']:.4f} | {row['genre']:.4f} | "
        f"{occupancy_text(row['train_occupancy'])} | "
        f"{occupancy_text(row['heldout_occupancy'])} | "
        f"{metric_text(row['silhouette'])} | "
        f"{'clears' if clear_pareto(row) else 'misses'} |\n"
    )


def write_report(baseline, screen, winner=None, stress=None):
    stress = stress or []
    lines = [
        "# D_SHARED capacity sweep pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Method\n\n",
        "The shared architecture file was not modified. The pilot sets "
        "`arch.D_SHARED` on the imported sibling module immediately before "
        "constructing each `arch.LearnedStudent(\"attn1\")`. Widths 64 and 128 "
        "use the unchanged snapshot pilot's train-only 50-D content PCA, "
        "28-D affect probabilities, content and affect teacher graphs, "
        "symmetric two-teacher InfoNCE, temperature, fixed Adam learning "
        "rate, pair sampling, checkpoint cadence, recall plateau stopping, "
        "and independent train/held-out Leiden passes. No noise, cosine "
        "schedule, or clustering loss was added.\n\n",
        "The D_SHARED=32 seed-42 AMIs and silhouette are cited from "
        "`attention_h1_embedding_snapshot_pilot_report.md` and "
        "`attention_h1_baseline_seed_stress_pilot_report.md`; it was not "
        "retrained. Its occupancy and embedding-only diagnostics are recovered "
        "from `attention_h1_embedding_snapshot.npz`, using the same "
        "`arch.evaluate_checkpoint` helper and seed-42 diagnostic draws as "
        "the new widths. The helper computes held-out content/affect "
        "teacher-graph recall, effective rank at 95% variance, and top "
        "covariance eigenvalue fraction. The cited AMIs use all held-out "
        "paintings for emotion and the genre-labelled subset for genre.\n\n",
        "Held-out silhouette uses a fresh `np.random.default_rng(42)` draw "
        "of at most 6,000 points followed by `silhouette_score` with "
        "`sample_size=min(4000, len(idx))`, `random_state=42`, default "
        "Euclidean distance, and independent held-out Leiden labels. "
        "Occupancy counts all observed Leiden communities on each full "
        "split; below 1% means strictly fewer than 1% of that split's "
        "paintings. The held-out Pareto bar is emotion AMI > 0.1236 "
        "and genre AMI > 0.1954 (both strict).\n\n",
        f"Screen-to-stress gate: both AMI bars and held-out silhouette "
        f"at least {MIN_SILHOUETTE_GAIN:.4f} above the cited seed-42 "
        f"D_SHARED=32 value {BASELINE_SILHOUETTE:.4f}, i.e. at least "
        f"{BASELINE_SILHOUETTE + MIN_SILHOUETTE_GAIN:.4f}. "
        "If both widths qualify, only the higher-silhouette width is "
        "stressed (smaller width breaks an exact tie). This 0.0200 "
        "absolute bar exceeds the baseline's four-seed silhouette "
        "range (0.0063) and the schedule-only seed-42 increase "
        "(0.0111).\n\n",
        "## Seed-42 capacity screen\n\n",
        "| D_SHARED | seed | held-out effective rank 95% | top eigen fraction "
        "| content recall | affect recall | held-out emotion AMI | "
        "held-out genre AMI | train Leiden occupancy | held-out Leiden occupancy "
        "| held-out silhouette | Pareto bar |\n",
        "|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|\n",
    ]
    lines.extend(report_row(row) for row in [baseline, *screen])
    lines.append("\n")
    if winner is None:
        lines.append(
            "## Stress decision\n\nNo width cleared both AMI bars and the "
            "+0.0200 held-out silhouette gate, so no additional seed "
            "was trained.\n\n"
        )
    else:
        lines.append(
            f"## Stress decision\n\nD_SHARED={winner['width']} cleared the "
            f"seed-42 screen with silhouette gain "
            f"{winner['silhouette'] - BASELINE_SILHOUETTE:+.4f}; "
            "stress seeds 7, 123, and 2024 were selected.\n\n"
        )
        if len(stress) == len(STRESS_SEEDS):
            four = [winner, *stress]
            mean_silhouette = float(np.mean([row["silhouette"] for row in four]))
            clear_count = sum(clear_pareto(row) for row in four)
            lines.extend([
                "## Four-seed stress of the selected width\n\n",
                "| D_SHARED | seed | held-out effective rank 95% | top eigen fraction "
                "| content recall | affect recall | held-out emotion AMI | "
                "held-out genre AMI | train Leiden occupancy | held-out Leiden occupancy "
                "| held-out silhouette | Pareto bar |\n",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|\n",
            ])
            lines.extend(report_row(row) for row in four)
            lines.append(
                f"\nFour-seed mean held-out silhouette: {mean_silhouette:.4f} "
                f"versus cited D_SHARED=32 mean 0.0397 "
                f"({mean_silhouette - 0.0397:+.4f}); Pareto clearance: "
                f"{clear_count}/4 versus baseline 3/4.\n\n"
            )
    lines.append("## Final verdict\n\n")
    if winner is None:
        best = max(screen, key=lambda row: row["silhouette"] if np.isfinite(row["silhouette"]) else -np.inf)
        lines.append(
            f"Capacity alone did not yield a qualifying improvement: "
            f"best seed-42 widened silhouette was {metric_text(best['silhouette'])} "
            f"at D_SHARED={best['width']} versus 0.0377 at 32, with "
            f"Pareto bar {'cleared' if clear_pareto(best) else 'missed'}. "
            "Under the unchanged two-InfoNCE objective, widening does not "
            "explain the reported silhouette gap to PercepT.\n"
        )
    elif len(stress) < len(STRESS_SEEDS):
        lines.append(
            "The screen has a qualifying winner; the four-seed result is "
            "pending. This is not yet a robust capacity verdict.\n"
        )
    else:
        four = [winner, *stress]
        mean_silhouette = float(np.mean([row["silhouette"] for row in four]))
        clear_count = sum(clear_pareto(row) for row in four)
        robust = (
            clear_count >= 3
            and np.isfinite(mean_silhouette)
            and mean_silhouette - 0.0397 >= MIN_SILHOUETTE_GAIN
        )
        if robust:
            lines.append(
                f"Widening produced a repeatable native-space gain: "
                f"four-seed mean silhouette {mean_silhouette:.4f} versus "
                f"0.0397 for D_SHARED=32, with {clear_count}/4 seeds "
                f"clearing both AMI bars. "
            )
        else:
            lines.append(
                f"The seed-42 gain did not survive the four-seed gate: "
                f"mean silhouette {metric_text(mean_silhouette)} versus "
                f"0.0397 for D_SHARED=32, with {clear_count}/4 seeds "
                f"clearing both AMI bars. "
            )
        lines.append(
            f"PercepT's reported silhouette is {PERCEPT_SILHOUETTE:.4f} "
            "in a different 128-D latent with different labels and a "
            "full-split scoring protocol, so its numerical gap is not a "
            "matched causal estimate. These results test capacity under "
            "the unchanged buddy objective only.\n"
        )
    os.makedirs(REPORT_OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as report:
        report.writelines(lines)


def main():
    torch.manual_seed(SCREEN_SEED)
    np.random.seed(SCREEN_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SCREEN_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    arch = load_module("dshared_arch_sweep", ARCH_SWEEP_PATH)
    context = prepare_context(arch, SCREEN_SEED)
    baseline = cited_baseline(arch, context)
    screen = []
    for width in D_SHARED_VALUES:
        screen.append(run_width(arch, context, width, SCREEN_SEED))
    winner = screen_winner(screen)
    write_report(baseline, screen, winner)
    arch.log(f"Wrote screen report to {REPORT_PATH}.")
    if winner is not None:
        stress = []
        for seed in STRESS_SEEDS:
            # The baseline stress pilot reruns train PCA and reference graphs
            # at each seed; rebuild them here with the same seed as well.
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            seed_context = prepare_context(arch, seed)
            stress.append(run_width(arch, seed_context, winner["width"], seed))
        write_report(baseline, screen, winner, stress)
        arch.log(f"Wrote four-seed stress report to {REPORT_PATH}.")
    arch.CURRENT_ARCHITECTURE = None


if __name__ == "__main__":
    main()
