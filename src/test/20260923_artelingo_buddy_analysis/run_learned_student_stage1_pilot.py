"""Run the Stage 1 learned two-teacher ArtELingo student pilot on a GPU.

This is deliberately a standalone, manually launched experiment.  It fits all
learned components on the train split only, uses val+test only for checkpoint
diagnostics, and writes its report beside this script.
"""

import importlib.util
import os
import sys
import time

import numpy as np
import torch
from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F


OUT_DIR = os.path.dirname(__file__)
PIPELINE_PATH = os.path.join(OUT_DIR, "run_pipeline.py")
AFFECT_PILOT_PATH = os.path.join(OUT_DIR, "run_affect_pilot.py")
SINGLE_MODALITY_PATH = os.path.join(OUT_DIR, "run_single_modality_pilot.py")
CCA_AUDIT_PATH = os.path.join(OUT_DIR, "run_cca_audit_pilot.py")
REPORT_PATH = os.path.join(OUT_DIR, "learned_student_stage1_pilot_report.md")

HELDOUT_STORAGE_DIR = "/data/SSD2/pre_extract/artelingo_heldout/features"
HELDOUT_JSON = "/data/PDD/artelingo/artelingo_val_test.json"
HELDOUT_PAINTINGS = 9_365

D_SHARED = 32
CONTENT_PCA_DIM = 50
BATCH_SIZE = 1024
TEMPERATURE = 0.1
LEARNING_RATE = 1e-3
MAX_EPOCHS = 200
PLATEAU_WINDOW = 5
PLATEAU_REL_IMPROVEMENT = 0.01
CHECKPOINT_EVERY = 5
EDGE_SAMPLE_SIZE = 2_000
EFFECTIVE_RANK_SAMPLE_SIZE = 5_000
SEED = 42

REPO_ROOT = os.path.abspath(os.path.join(OUT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.conditional_buddy.prototype_seed import detect_communities


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_sibling_module(module_name: str, path: str):
    """Import a sibling standalone script without running its main block."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LearnedStudent(nn.Module):
    """The predeclared linear two-view student with a scalar fusion gate."""

    def __init__(self) -> None:
        super().__init__()
        self.proj_content = nn.Linear(CONTENT_PCA_DIM, D_SHARED)
        self.proj_affect = nn.Linear(28, D_SHARED)
        self.gate = nn.Sequential(
            nn.Linear(2 * D_SHARED, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, content: torch.Tensor, affect: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        content_proj = F.normalize(self.proj_content(content), dim=1)
        affect_proj = F.normalize(self.proj_affect(affect), dim=1)
        gate = self.gate(torch.cat((content_proj, affect_proj), dim=1))
        student = F.normalize(gate * content_proj + (1.0 - gate) * affect_proj, dim=1)
        return student, gate


def upper_triangle_edges(graph) -> np.ndarray:
    """Extract each undirected graph edge once as node-index endpoint pairs."""
    graph = coo_matrix(graph)
    keep = graph.row < graph.col
    edges = np.column_stack((graph.row[keep], graph.col[keep])).astype(np.int64, copy=False)
    if len(edges) == 0:
        raise RuntimeError("Teacher graph contains no upper-triangle edges.")
    return edges


def sample_positive_pairs(edges: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample exactly BATCH_SIZE positive pairs, replacing when required."""
    indices = rng.choice(len(edges), size=BATCH_SIZE, replace=len(edges) < BATCH_SIZE)
    return edges[indices]


def symmetric_infonce(embeddings: torch.Tensor, pairs: np.ndarray, device: torch.device) -> torch.Tensor:
    """Compute symmetric in-batch InfoNCE over sampled positive endpoints."""
    pair_tensor = torch.as_tensor(pairs, dtype=torch.long, device=device)
    anchors = embeddings[pair_tensor[:, 0]]
    positives = embeddings[pair_tensor[:, 1]]
    targets = torch.arange(BATCH_SIZE, device=device)
    forward_loss = F.cross_entropy(anchors @ positives.T / TEMPERATURE, targets)
    reverse_loss = F.cross_entropy(positives @ anchors.T / TEMPERATURE, targets)
    return 0.5 * (forward_loss + reverse_loss)


def content_batch_embeddings(
    model: LearnedStudent,
    content: torch.Tensor,
    affect: torch.Tensor,
    pairs: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """Embed every unique content-teacher endpoint once, then remap its pairs."""
    node_ids, inverse = np.unique(pairs.reshape(-1), return_inverse=True)
    node_tensor = torch.as_tensor(node_ids, dtype=torch.long, device=device)
    embeddings, _gate = model(content[node_tensor], affect[node_tensor])
    return embeddings, inverse.reshape(-1, 2)


def selected_parameter_gradient_norm(model: LearnedStudent) -> float:
    """Return the L2 norm in the two components monitored for teacher balance."""
    squared_norm = 0.0
    for module in (model.proj_content, model.gate):
        for parameter in module.parameters():
            if parameter.grad is not None:
                squared_norm += float(parameter.grad.detach().pow(2).sum().item())
    return squared_norm ** 0.5


def relative_improvement(current: float, previous: float) -> float:
    """Compute checkpoint-over-checkpoint relative improvement safely near zero."""
    return (current - previous) / max(abs(previous), 1e-12)


def evaluate_checkpoint(
    model: LearnedStudent,
    heldout_content: torch.Tensor,
    heldout_affect: torch.Tensor,
    content_graph,
    affect_graph,
    sampled_nodes: np.ndarray,
    effective_rank_nodes: np.ndarray,
    single_modality,
    heldout_pipeline,
    affect_pilot,
    device: str,
) -> dict:
    """Measure held-out retrieval, rank, and gate behavior without gradients."""
    model.eval()
    with torch.no_grad():
        embeddings, gate = model(heldout_content, heldout_affect)
    embeddings_np = embeddings.cpu().numpy().astype(np.float32, copy=False)
    gate_np = gate.squeeze(1).cpu().numpy()
    student_graph = single_modality.build_single_modality_graph(
        "held-out-learned-student",
        embeddings_np,
        heldout_pipeline,
        affect_pilot,
        device,
        expected_nodes=len(embeddings_np),
    )
    content_recall = cca_audit.graph_overlap_fraction(content_graph, student_graph, sampled_nodes)
    affect_recall = cca_audit.graph_overlap_fraction(affect_graph, student_graph, sampled_nodes)

    rank_embeddings = embeddings_np[effective_rank_nodes]
    covariance = np.cov(rank_embeddings, rowvar=False)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)[::-1]
    total_variance = float(eigenvalues.sum())
    if total_variance <= 0.0:
        top_eigen_fraction = 1.0
        effective_rank_95 = 1
    else:
        top_eigen_fraction = float(eigenvalues[0] / total_variance)
        effective_rank_95 = int(np.searchsorted(np.cumsum(eigenvalues) / total_variance, 0.95) + 1)
    saturated_fraction = float(np.mean((gate_np < 0.1) | (gate_np > 0.9)))
    return {
        "content_recall": content_recall,
        "affect_recall": affect_recall,
        "top_eigen_fraction": top_eigen_fraction,
        "effective_rank_95": effective_rank_95,
        "gate_mean": float(gate_np.mean()),
        "gate_std": float(gate_np.std()),
        "gate_saturated_fraction": saturated_fraction,
    }


def write_report(
    trajectory: list[dict],
    stop_reason: str,
    final_metrics: dict,
    heldout_metrics: dict,
    verdict: str,
    pareto_verdict: str,
) -> None:
    """Write the predeclared results table and evidence-based conclusion."""
    first = trajectory[0]
    final = trajectory[-1]
    lines = [
        "# ArtELingo learned two-teacher student — Stage 1\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Architecture and training\n\n",
        "The student projects train-only-PCA-reduced, independently L2-normalized CLIP "
        "image/text content (50 dimensions) and raw 28-dimensional GoEmotions probabilities "
        "into a shared 32-dimensional space. A small scalar gate combines the two normalized "
        "projections for every node. It is trained with equally weighted symmetric in-batch "
        "InfoNCE losses from content and affect buddy-graph edges; neither teacher is tuned "
        "or reweighted. Held-out retrieval uses a fixed 2,000-node sample, and all PCA fitting "
        "is restricted to train paintings.\n\n",
        "## Checkpoint trajectory\n\n",
        "| epoch | content held-out recall | affect held-out recall | content loss | affect loss | content gradient share | top-eigenvalue variance fraction | effective rank at 95% | gate mean | gate std | gate saturated fraction |\n",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in trajectory:
        content_loss = "—" if np.isnan(row["content_loss"]) else f"{row['content_loss']:.4f}"
        affect_loss = "—" if np.isnan(row["affect_loss"]) else f"{row['affect_loss']:.4f}"
        gradient_share = (
            "—" if np.isnan(row["content_gradient_share"]) else f"{row['content_gradient_share']:.4f}"
        )
        lines.append(
            f"| {row['epoch']} | {row['content_recall']:.4f} | {row['affect_recall']:.4f} | "
            f"{content_loss} | {affect_loss} | {gradient_share} | {row['top_eigen_fraction']:.4f} | "
            f"{row['effective_rank_95']} | {row['gate_mean']:.4f} | {row['gate_std']:.4f} | "
            f"{row['gate_saturated_fraction']:.4f} |\n"
        )
    lines.extend([
        "\n## Stopping and collapse determination\n\n",
        f"Training stopped at epoch {final['epoch']} because **{stop_reason}**.\n\n",
        f"**{verdict}.** Here, “first” means epoch 0 (pre-training, before any gradient steps). "
        f"First/final held-out content recall: {first['content_recall']:.4f} / "
        f"{final['content_recall']:.4f}; affect recall: {first['affect_recall']:.4f} / "
        f"{final['affect_recall']:.4f}; final content gradient share: "
        f"{final['content_gradient_share']:.4f}; final gate saturated fraction: "
        f"{final['gate_saturated_fraction']:.4f}.\n\n",
        "## Final comparison\n\n",
        "| signal | emotion AMI | emotion V-measure | genre AMI | genre V-measure |\n",
        "|---|---:|---:|---:|---:|\n",
        "| Content-only (reference) | 0.0593 | 0.0600 | 0.4384 | 0.4572 |\n",
        "| GoEmotions-affect-only (reference) | 0.1180 | 0.1189 | 0.0396 | 0.0937 |\n",
        "| Late fusion — union (reference) | 0.1236 | 0.1241 | 0.1394 | 0.1677 |\n",
        "| Hierarchical refinement (reference) | 0.1072 | 0.1141 | 0.1954 | 0.3901 |\n",
        f"| Learned student, Stage 1 (this run) | {final_metrics['emotion']['AMI']:.4f} | "
        f"{final_metrics['emotion']['V_measure']:.4f} | {final_metrics['genre']['AMI']:.4f} | "
        f"{final_metrics['genre']['V_measure']:.4f} |\n",
        "\n## Held-out generalization\n\n",
        "| signal | emotion AMI | emotion V-measure | genre AMI | genre V-measure |\n",
        "|---|---:|---:|---:|---:|\n",
        f"| Learned student — TRAIN | {final_metrics['emotion']['AMI']:.4f} | "
        f"{final_metrics['emotion']['V_measure']:.4f} | {final_metrics['genre']['AMI']:.4f} | "
        f"{final_metrics['genre']['V_measure']:.4f} |\n",
        f"| Learned student — HELD-OUT | {heldout_metrics['emotion']['AMI']:.4f} | "
        f"{heldout_metrics['emotion']['V_measure']:.4f} | {heldout_metrics['genre']['AMI']:.4f} | "
        f"{heldout_metrics['genre']['V_measure']:.4f} |\n",
        "\n",
        f"The absolute train-to-held-out drop is {abs(final_metrics['emotion']['AMI'] - heldout_metrics['emotion']['AMI']):.4f} "
        f"points for emotion AMI and {abs(final_metrics['genre']['AMI'] - heldout_metrics['genre']['AMI']):.4f} "
        "points for genre AMI. ",
        "\n## Pareto-bar verdict\n\n",
        f"**{pareto_verdict}.** ",
    ])
    heldout_pareto_cleared = heldout_metrics["emotion"]["AMI"] > 0.1236 and heldout_metrics["genre"]["AMI"] > 0.1954
    if heldout_pareto_cleared:
        lines.append("Held-out results also clear the same Pareto bar.\n\n")
    else:
        lines.append("Held-out results do not clear the same Pareto bar. ")
        if pareto_verdict == "Cleared both":
            lines.append(
                "Because train results did clear it, this is evidence of the same train/held-out "
                "generalization gap seen in the BERT pilot, not a new failure mode.\n\n"
            )
        else:
            lines.append("\n\n")
    if verdict == "Real success":
        lines.append("The Stage 1 baseline is a reportable win; a richer architecture is not warranted yet.\n")
    elif verdict == "Merely a compromise":
        lines.append(
            "Both teachers retained signal, but the partition did not clear both AMI targets; "
            "this indicates incompatible teacher constraints rather than insufficient capacity, "
            "so do not escalate architecture on this evidence alone.\n"
        )
    else:
        lines.append(
            "At least one teacher failed the predeclared development/balance/collapse criterion; "
            "inspect the trajectory before considering a richer architecture.\n"
        )
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    global cca_audit
    pipeline = load_sibling_module("artelingo_run_pipeline_stage1", PIPELINE_PATH)
    affect_pilot = load_sibling_module("artelingo_run_affect_pilot_stage1", AFFECT_PILOT_PATH)
    single_modality = load_sibling_module("artelingo_run_single_modality_stage1", SINGLE_MODALITY_PATH)
    cca_audit = load_sibling_module("artelingo_run_cca_audit_stage1", CCA_AUDIT_PATH)
    heldout_pipeline = load_sibling_module("artelingo_run_pipeline_stage1_heldout", PIPELINE_PATH)
    heldout_pipeline.STORAGE_DIR = HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = HELDOUT_JSON

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using {device} for learned-student training and graph construction.")

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
    if len(heldout_paintings) != HELDOUT_PAINTINGS:
        raise RuntimeError(f"Expected {HELDOUT_PAINTINGS:,} held-out paintings, got {len(heldout_paintings):,}.")
    log("Extracting held-out GoEmotions probabilities...")
    affect_heldout = np.asarray(
        affect_pilot.extract_affect_nodes(HELDOUT_JSON, heldout_paintings, str(device)), dtype=np.float64
    )
    if affect_heldout.shape != (len(heldout_paintings), 28):
        raise RuntimeError(f"Expected held-out affect features ({len(heldout_paintings)}, 28), got {affect_heldout.shape}.")

    content_train = cca_audit.content_features(img_nodes, txt_nodes, affect_pilot)
    content_heldout = cca_audit.content_features(heldout_img, heldout_txt, affect_pilot)
    log(f"Fitting train-only content PCA ({CONTENT_PCA_DIM} components)...")
    pca = PCA(n_components=CONTENT_PCA_DIM, random_state=SEED)
    content_train = pca.fit_transform(content_train).astype(np.float32)
    content_heldout = pca.transform(content_heldout).astype(np.float32)

    log("Building train content and affect teacher graphs...")
    _img_graph, _txt_graph, content_teacher_graph = pipeline.build_buddy_graphs(
        img_nodes, txt_nodes, K=pipeline.K, alpha=pipeline.ALPHA, device=str(device), connect_components=True
    )
    affect_teacher_graph = single_modality.build_single_modality_graph(
        "train-affect-teacher", affect_train, pipeline, affect_pilot, str(device), expected_nodes=len(paintings)
    )
    content_edges = upper_triangle_edges(content_teacher_graph)
    affect_edges = upper_triangle_edges(affect_teacher_graph)
    log(f"Teacher edge lists: content={len(content_edges):,}, affect={len(affect_edges):,}.")

    log("Building held-out content and affect reference graphs...")
    heldout_content_graph = single_modality.build_single_modality_graph(
        "held-out-content-reference", content_heldout, heldout_pipeline, affect_pilot, str(device),
        expected_nodes=len(heldout_paintings),
    )
    heldout_affect_graph = single_modality.build_single_modality_graph(
        "held-out-affect-reference", affect_heldout, heldout_pipeline, affect_pilot, str(device),
        expected_nodes=len(heldout_paintings),
    )
    diagnostic_rng = np.random.default_rng(SEED)
    sampled_nodes = diagnostic_rng.choice(len(heldout_paintings), size=EDGE_SAMPLE_SIZE, replace=False)
    rank_nodes = diagnostic_rng.choice(len(heldout_paintings), size=EFFECTIVE_RANK_SAMPLE_SIZE, replace=False)

    train_content = torch.as_tensor(content_train, dtype=torch.float32, device=device)
    train_affect = torch.as_tensor(affect_train, dtype=torch.float32, device=device)
    heldout_content = torch.as_tensor(content_heldout, dtype=torch.float32, device=device)
    heldout_affect = torch.as_tensor(affect_heldout, dtype=torch.float32, device=device)
    model = LearnedStudent().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    log("Evaluating epoch-0 checkpoint before training...")
    epoch_0_diagnostics = evaluate_checkpoint(
        model, heldout_content, heldout_affect, heldout_content_graph, heldout_affect_graph,
        sampled_nodes, rank_nodes, single_modality, heldout_pipeline, affect_pilot, str(device),
    )
    epoch_0_checkpoint = {
        "epoch": 0,
        "content_loss": float("nan"),
        "affect_loss": float("nan"),
        "content_gradient_share": float("nan"),
        **epoch_0_diagnostics,
    }
    trajectory = [epoch_0_checkpoint]
    log(
        f"epoch=0 content_recall={epoch_0_checkpoint['content_recall']:.4f} "
        f"affect_recall={epoch_0_checkpoint['affect_recall']:.4f} content_loss=— "
        "affect_loss=— content_grad_share=— "
        f"top_eigen_fraction={epoch_0_checkpoint['top_eigen_fraction']:.4f} "
        f"effective_rank_95={epoch_0_checkpoint['effective_rank_95']} "
        f"gate_mean={epoch_0_checkpoint['gate_mean']:.4f} gate_std={epoch_0_checkpoint['gate_std']:.4f} "
        f"gate_saturated={epoch_0_checkpoint['gate_saturated_fraction']:.4f}"
    )
    plateau_count = 0
    stop_reason = f"reached MAX_EPOCHS={MAX_EPOCHS}"

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_rng = np.random.default_rng(SEED + epoch)
        content_pairs = sample_positive_pairs(content_edges, epoch_rng)
        affect_pairs = sample_positive_pairs(affect_edges, epoch_rng)

        content_embeddings, remapped_content_pairs = content_batch_embeddings(
            model, train_content, train_affect, content_pairs, device
        )
        content_loss = symmetric_infonce(content_embeddings, remapped_content_pairs, device)
        # The affect teacher deliberately runs the full inexpensive node matrix, with both heads and gate.
        affect_embeddings, _affect_gate = model(train_content, train_affect)
        affect_loss = symmetric_infonce(affect_embeddings, affect_pairs, device)
        total_loss = content_loss + affect_loss

        gradient_share = None
        if epoch % CHECKPOINT_EVERY == 0:
            optimizer.zero_grad(set_to_none=True)
            content_loss.backward(retain_graph=True)
            content_grad_norm = selected_parameter_gradient_norm(model)
            optimizer.zero_grad(set_to_none=True)
            affect_loss.backward(retain_graph=True)
            affect_grad_norm = selected_parameter_gradient_norm(model)
            gradient_share = content_grad_norm / max(content_grad_norm + affect_grad_norm, 1e-12)
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if epoch % CHECKPOINT_EVERY != 0:
            continue
        diagnostics = evaluate_checkpoint(
            model, heldout_content, heldout_affect, heldout_content_graph, heldout_affect_graph,
            sampled_nodes, rank_nodes, single_modality, heldout_pipeline, affect_pilot, str(device),
        )
        checkpoint = {
            "epoch": epoch,
            "content_loss": float(content_loss.detach().item()),
            "affect_loss": float(affect_loss.detach().item()),
            "content_gradient_share": float(gradient_share),
            **diagnostics,
        }
        trajectory.append(checkpoint)
        log(
            f"epoch={epoch} content_recall={checkpoint['content_recall']:.4f} "
            f"affect_recall={checkpoint['affect_recall']:.4f} content_loss={checkpoint['content_loss']:.4f} "
            f"affect_loss={checkpoint['affect_loss']:.4f} content_grad_share={gradient_share:.4f} "
            f"top_eigen_fraction={checkpoint['top_eigen_fraction']:.4f} "
            f"effective_rank_95={checkpoint['effective_rank_95']} gate_mean={checkpoint['gate_mean']:.4f} "
            f"gate_std={checkpoint['gate_std']:.4f} gate_saturated={checkpoint['gate_saturated_fraction']:.4f}"
        )
        if len(trajectory) > 1:
            previous = trajectory[-2]
            content_plateau = relative_improvement(checkpoint["content_recall"], previous["content_recall"]) < PLATEAU_REL_IMPROVEMENT
            affect_plateau = relative_improvement(checkpoint["affect_recall"], previous["affect_recall"]) < PLATEAU_REL_IMPROVEMENT
            plateau_count = plateau_count + 1 if content_plateau and affect_plateau else 0
            if plateau_count >= PLATEAU_WINDOW:
                stop_reason = f"both recalls plateaued for {PLATEAU_WINDOW} consecutive checkpoints"
                log(f"Stopping: {stop_reason} at epoch {epoch}.")
                break

    final = trajectory[-1]
    model.eval()
    with torch.no_grad():
        train_embeddings, _train_gate = model(train_content, train_affect)
    final_graph = single_modality.build_single_modality_graph(
        "final-train-learned-student", train_embeddings.cpu().numpy(), pipeline, affect_pilot, str(device),
        expected_nodes=len(paintings),
    )
    log(f"Final learned student: running Leiden (seed={SEED})...")
    communities = detect_communities(final_graph, seed=SEED)
    emotion_metrics = pipeline.external_metrics(communities, majority_emotion)
    genre_map = pipeline.load_genre_map()
    genre_indices = [index for index, painting in enumerate(paintings) if painting in genre_map]
    if not genre_indices:
        raise RuntimeError("No genre-labelled paintings overlap the train node set.")
    genre_metrics = pipeline.external_metrics(
        [communities[index] for index in genre_indices], [genre_map[paintings[index]] for index in genre_indices]
    )

    log(f"Final held-out learned student: building graph and running Leiden (seed={SEED})...")
    model.eval()
    with torch.no_grad():
        heldout_embeddings, _heldout_gate = model(heldout_content, heldout_affect)
    heldout_final_graph = single_modality.build_single_modality_graph(
        "final-heldout-learned-student", heldout_embeddings.cpu().numpy(), heldout_pipeline, affect_pilot, str(device),
        expected_nodes=len(heldout_paintings),
    )
    heldout_communities = detect_communities(heldout_final_graph, seed=SEED)
    heldout_emotion_metrics = heldout_pipeline.external_metrics(heldout_communities, heldout_majority_emotion)
    heldout_genre_indices = [
        index for index, painting in enumerate(heldout_paintings) if painting in genre_map
    ]
    if not heldout_genre_indices:
        raise RuntimeError("No genre-labelled paintings overlap the held-out node set.")
    heldout_genre_metrics = pipeline.external_metrics(
        [heldout_communities[index] for index in heldout_genre_indices],
        [genre_map[heldout_paintings[index]] for index in heldout_genre_indices],
    )

    content_developed = relative_improvement(final["content_recall"], trajectory[0]["content_recall"]) >= 0.20
    affect_developed = relative_improvement(final["affect_recall"], trajectory[0]["affect_recall"]) >= 0.20
    gradient_collapsed = final["content_gradient_share"] < 0.05 or final["content_gradient_share"] > 0.95
    gate_collapsed = final["gate_saturated_fraction"] > 0.90
    collapsed = not content_developed or not affect_developed or gradient_collapsed or gate_collapsed
    pareto_cleared = emotion_metrics["AMI"] > 0.1236 and genre_metrics["AMI"] > 0.1954
    verdict = "Collapsed" if collapsed else ("Real success" if pareto_cleared else "Merely a compromise")
    cleared_count = int(emotion_metrics["AMI"] > 0.1236) + int(genre_metrics["AMI"] > 0.1954)
    pareto_verdict = ("Cleared both" if cleared_count == 2 else "Cleared one" if cleared_count == 1 else "Cleared neither")
    write_report(
        trajectory,
        stop_reason,
        {"emotion": emotion_metrics, "genre": genre_metrics},
        {"emotion": heldout_emotion_metrics, "genre": heldout_genre_metrics},
        verdict,
        pareto_verdict,
    )
    log(
        f"Wrote report to {REPORT_PATH}; verdict={verdict}; {pareto_verdict.lower()} Pareto targets "
        f"(emotion AMI={emotion_metrics['AMI']:.4f}, genre AMI={genre_metrics['AMI']:.4f})."
    )


if __name__ == "__main__":
    main()
