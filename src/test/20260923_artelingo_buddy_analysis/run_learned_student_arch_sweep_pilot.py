"""Run the learned-student architecture sweep on a GPU.

This standalone experiment fits learned components on the train split only.
It reuses the committed Stage 1 and Stage 2 results and runs only MLP-128,
Attention-h1, and Attention-h4.
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
REPORT_PATH = os.path.join(OUT_DIR, "learned_student_arch_sweep_pilot_report.md")

HELDOUT_STORAGE_DIR = "/data/SSD2/pre_extract/artelingo_heldout/features"
HELDOUT_JSON = "/data/PDD/artelingo/artelingo_val_test.json"
HELDOUT_PAINTINGS = 9_365

D_SHARED = 32
CONTENT_PCA_DIM = 50
HIDDEN_DIM = 128
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
ARCHITECTURES = (
    ("MLP-128", "mlp128"),
    ("Attention-h1", "attn1"),
    ("Attention-h4", "attn4"),
)
CURRENT_ARCHITECTURE: str | None = None

# These fixed rows are the final values from the committed Stage 1/Stage 2 reports.
REFERENCE_RESULTS = (
    {
        "label": "Linear (reference, not rerun)",
        "final_metrics": {"emotion": {"AMI": 0.1284}, "genre": {"AMI": 0.2799}},
        "heldout_metrics": {"emotion": {"AMI": 0.1095}, "genre": {"AMI": 0.2901}},
        "verdict": "Collapsed",
        "gradient_share": 0.4988,
        "gate_mean": 0.4926,
        "gate_saturated_fraction": 0.0,
    },
    {
        "label": "MLP-64 (reference, not rerun)",
        "final_metrics": {"emotion": {"AMI": 0.1230}, "genre": {"AMI": 0.2319}},
        "heldout_metrics": {"emotion": {"AMI": 0.1046}, "genre": {"AMI": 0.2087}},
        "verdict": "Merely a compromise",
        "gradient_share": 0.4823,
        "gate_mean": 0.4865,
        "gate_saturated_fraction": 0.0,
    },
)

REPO_ROOT = os.path.abspath(os.path.join(OUT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.conditional_buddy.prototype_seed import detect_communities


def log(message: str) -> None:
    prefix = "" if CURRENT_ARCHITECTURE is None else f"[{CURRENT_ARCHITECTURE}] "
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {prefix}{message}", flush=True)


def load_sibling_module(module_name: str, path: str):
    """Import a sibling standalone script without running its main block."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AttentionFusion(nn.Module):
    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=D_SHARED, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(D_SHARED)

    def forward(self, content_proj: torch.Tensor, affect_proj: torch.Tensor):
        tokens = torch.stack((content_proj, affect_proj), dim=1)
        attended, attn_weights = self.attn(
            tokens, tokens, tokens, need_weights=True, average_attn_weights=True
        )
        pooled = attended.mean(dim=1)
        student = F.normalize(self.norm(pooled), dim=1)
        return student, attn_weights


class LearnedStudent(nn.Module):
    """A configured MLP-gate or linear-head attention two-view student."""

    def __init__(self, heads: str) -> None:
        super().__init__()
        self.heads = heads
        self.is_attention = heads in {"attn1", "attn4"}
        if heads == "mlp128":
            self.proj_content = nn.Sequential(
                nn.Linear(CONTENT_PCA_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, D_SHARED)
            )
            self.proj_affect = nn.Sequential(
                nn.Linear(28, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, D_SHARED)
            )
            self.gate = nn.Sequential(
                nn.Linear(2 * D_SHARED, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
            )
        elif heads in {"attn1", "attn4"}:
            self.proj_content = nn.Linear(CONTENT_PCA_DIM, D_SHARED)
            self.proj_affect = nn.Linear(28, D_SHARED)
            self.fusion = AttentionFusion(1 if heads == "attn1" else 4)
        else:
            raise ValueError(f"Unknown architecture config: {heads}")

    def forward(self, content: torch.Tensor, affect: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        content_proj = F.normalize(self.proj_content(content), dim=1)
        affect_proj = F.normalize(self.proj_affect(affect), dim=1)
        if self.is_attention:
            return self.fusion(content_proj, affect_proj)
        gate = self.gate(torch.cat((content_proj, affect_proj), dim=1))
        student = F.normalize(gate * content_proj + (1.0 - gate) * affect_proj, dim=1)
        return student, gate


def upper_triangle_edges(graph) -> np.ndarray:
    graph = coo_matrix(graph)
    keep = graph.row < graph.col
    edges = np.column_stack((graph.row[keep], graph.col[keep])).astype(np.int64, copy=False)
    if len(edges) == 0:
        raise RuntimeError("Teacher graph contains no upper-triangle edges.")
    return edges


def sample_positive_pairs(edges: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return edges[rng.choice(len(edges), size=BATCH_SIZE, replace=len(edges) < BATCH_SIZE)]


def symmetric_infonce(embeddings: torch.Tensor, pairs: np.ndarray, device: torch.device) -> torch.Tensor:
    pair_tensor = torch.as_tensor(pairs, dtype=torch.long, device=device)
    anchors = embeddings[pair_tensor[:, 0]]
    positives = embeddings[pair_tensor[:, 1]]
    targets = torch.arange(BATCH_SIZE, device=device)
    forward_loss = F.cross_entropy(anchors @ positives.T / TEMPERATURE, targets)
    reverse_loss = F.cross_entropy(positives @ anchors.T / TEMPERATURE, targets)
    return 0.5 * (forward_loss + reverse_loss)


def content_batch_embeddings(model, content, affect, pairs, device) -> tuple[torch.Tensor, np.ndarray]:
    node_ids, inverse = np.unique(pairs.reshape(-1), return_inverse=True)
    node_tensor = torch.as_tensor(node_ids, dtype=torch.long, device=device)
    embeddings, _mixing_weights = model(content[node_tensor], affect[node_tensor])
    return embeddings, inverse.reshape(-1, 2)


def selected_parameter_gradient_norm(model: LearnedStudent) -> float:
    """Measure the content projection plus shared mixing component gradient norm."""
    squared_norm = 0.0
    mixing_module = model.fusion if model.is_attention else model.gate
    for module in (model.proj_content, mixing_module):
        for parameter in module.parameters():
            if parameter.grad is not None:
                squared_norm += float(parameter.grad.detach().pow(2).sum().item())
    return squared_norm ** 0.5


def relative_improvement(current: float, previous: float) -> float:
    return (current - previous) / max(abs(previous), 1e-12)


def gate_values(model: LearnedStudent, mixing_weights: torch.Tensor) -> np.ndarray:
    """Return gate values; for attention these are per-node self-attention weights."""
    if model.is_attention:
        # "gate" means mean diagonal self-attention weight for attention variants.
        values = 0.5 * (mixing_weights[:, 0, 0] + mixing_weights[:, 1, 1])
    else:
        values = mixing_weights.squeeze(1)
    return values.cpu().numpy()


def evaluate_checkpoint(
    model, heldout_content, heldout_affect, content_graph, affect_graph, sampled_nodes,
    effective_rank_nodes, single_modality, heldout_pipeline, affect_pilot, device: str,
) -> dict:
    model.eval()
    with torch.no_grad():
        embeddings, mixing_weights = model(heldout_content, heldout_affect)
    embeddings_np = embeddings.cpu().numpy().astype(np.float32, copy=False)
    gate_np = gate_values(model, mixing_weights)
    student_graph = single_modality.build_single_modality_graph(
        "held-out-learned-student", embeddings_np, heldout_pipeline, affect_pilot, device,
        expected_nodes=len(embeddings_np),
    )
    content_recall = cca_audit.graph_overlap_fraction(content_graph, student_graph, sampled_nodes)
    affect_recall = cca_audit.graph_overlap_fraction(affect_graph, student_graph, sampled_nodes)
    covariance = np.cov(embeddings_np[effective_rank_nodes], rowvar=False)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)[::-1]
    total_variance = float(eigenvalues.sum())
    if total_variance <= 0.0:
        top_eigen_fraction, effective_rank_95 = 1.0, 1
    else:
        top_eigen_fraction = float(eigenvalues[0] / total_variance)
        effective_rank_95 = int(np.searchsorted(np.cumsum(eigenvalues) / total_variance, 0.95) + 1)
    return {
        "content_recall": content_recall,
        "affect_recall": affect_recall,
        "top_eigen_fraction": top_eigen_fraction,
        "effective_rank_95": effective_rank_95,
        "gate_mean": float(gate_np.mean()),
        "gate_std": float(gate_np.std()),
        "gate_saturated_fraction": float(np.mean((gate_np < 0.1) | (gate_np > 0.9))),
    }


def write_report(results: list[dict]) -> None:
    """Write the predeclared five-architecture comparison after all new runs."""
    all_results = [*REFERENCE_RESULTS, *results]
    heldout_successes = [
        result for result in results
        if result["heldout_metrics"]["emotion"]["AMI"] > 0.1236
        and result["heldout_metrics"]["genre"]["AMI"] > 0.1954
    ]
    best = max(
        all_results,
        key=lambda result: (
            result["heldout_metrics"]["emotion"]["AMI"] + result["heldout_metrics"]["genre"]["AMI"],
            result["heldout_metrics"]["emotion"]["AMI"],
            result["heldout_metrics"]["genre"]["AMI"],
        ),
    )
    stage1_heldout = REFERENCE_RESULTS[0]["heldout_metrics"]
    new_best_beats_stage1 = [
        result for result in results
        if result["heldout_metrics"]["emotion"]["AMI"] > stage1_heldout["emotion"]["AMI"]
        and result["heldout_metrics"]["genre"]["AMI"] > stage1_heldout["genre"]["AMI"]
    ]
    by_label = {result["label"]: result for result in all_results}
    lines = [
        "# ArtELingo learned two-teacher student — architecture sweep\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Context\n\n",
        "This sweep asks whether self-attention can beat Stage 1's scalar gate while holding "
        "linear projection heads fixed, and whether Stage 2's MLP-underperforms finding remains "
        "true with 128 hidden units instead of 64. All new runs retain Stage 1's data, PCA, "
        "loss, sampling, learning rate, training budget, diagnostics, and stopping rule.\n\n",
        "## Results\n\n",
        "| architecture | split | emotion AMI | genre AMI | collapse verdict | final content gradient share | final gate/attention-weight mean | final gate/attention-weight saturated fraction |\n",
        "|---|---|---:|---:|---|---:|---:|---:|\n",
    ]
    for result in all_results:
        if "trajectory" in result:
            final = result["trajectory"][-1]
            gradient_share = final["content_gradient_share"]
            gate_mean = final["gate_mean"]
            saturated_fraction = final["gate_saturated_fraction"]
        else:
            gradient_share = result["gradient_share"]
            gate_mean = result["gate_mean"]
            saturated_fraction = result["gate_saturated_fraction"]
        for split, metrics in (("train", result["final_metrics"]), ("held-out", result["heldout_metrics"])):
            lines.append(
                f"| {result['label']} | {split} | {metrics['emotion']['AMI']:.4f} | "
                f"{metrics['genre']['AMI']:.4f} | {result['verdict']} | {gradient_share:.4f} | "
                f"{gate_mean:.4f} | {saturated_fraction:.4f} |\n"
            )
    lines.extend([
        "\n## Held-out Pareto verdict\n\n",
        "The held-out Pareto bar is emotion AMI > 0.1236 and genre AMI > 0.1954. ",
    ])
    if heldout_successes:
        lines.append("New architecture(s) clearing it: " + ", ".join(result["label"] for result in heldout_successes) + ". ")
    else:
        lines.append("No new architecture clears it. ")
    if new_best_beats_stage1:
        lines.append("New architecture(s) beating Stage 1 on both held-out axes: " + ", ".join(result["label"] for result in new_best_beats_stage1) + ".\n\n")
    else:
        lines.append("No new architecture beats Stage 1 on both held-out axes simultaneously.\n\n")
    attention_h1, attention_h4 = by_label["Attention-h1"], by_label["Attention-h4"]
    h1_sum = sum(metric["AMI"] for metric in attention_h1["heldout_metrics"].values())
    h4_sum = sum(metric["AMI"] for metric in attention_h4["heldout_metrics"].values())
    mlp64, mlp128 = by_label["MLP-64 (reference, not rerun)"], by_label["MLP-128"]
    mlp64_sum = sum(metric["AMI"] for metric in mlp64["heldout_metrics"].values())
    mlp128_sum = sum(metric["AMI"] for metric in mlp128["heldout_metrics"].values())
    attention_comparison = (
        "more heads help" if h4_sum > h1_sum else "more heads hurt" if h4_sum < h1_sum
        else "more heads make no material difference"
    )
    mlp_comparison = (
        "More MLP capacity improves the result." if mlp128_sum > mlp64_sum
        else "More MLP capacity does not improve the result."
    )
    lines.extend([
        "## Attention-head comparison\n\n",
        f"Attention-h1 held-out summed AMI is {h1_sum:.4f}; Attention-h4 is {h4_sum:.4f}. "
        f"On this measure, {attention_comparison}.\n\n",
        "## MLP-capacity comparison\n\n",
        f"MLP-64 held-out summed AMI is {mlp64_sum:.4f}; MLP-128 is {mlp128_sum:.4f}. "
        f"{mlp_comparison}\n\n",
        "## Conclusion\n\n",
        f"The best configuration by held-out summed AMI is {best['label']} at "
        f"{best['heldout_metrics']['emotion']['AMI'] + best['heldout_metrics']['genre']['AMI']:.4f}. ",
    ])
    if best["label"].startswith("Linear"):
        lines.append("Stage 1 / Linear remains the best mechanism found.\n")
    else:
        lines.append("This changes the investigation's standing conclusion: Stage 1 / Linear is no longer the best mechanism found.\n")
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def run_architecture(
    label, heads, train_content, train_affect, heldout_content, heldout_affect, content_edges,
    affect_edges, heldout_content_graph, heldout_affect_graph, sampled_nodes, rank_nodes,
    single_modality, heldout_pipeline, affect_pilot, pipeline, paintings, majority_emotion,
    heldout_paintings, heldout_majority_emotion, device,
) -> dict:
    global CURRENT_ARCHITECTURE
    CURRENT_ARCHITECTURE = label
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = LearnedStudent(heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    log("Evaluating epoch-0 checkpoint before training...")
    epoch_0_diagnostics = evaluate_checkpoint(model, heldout_content, heldout_affect, heldout_content_graph,
        heldout_affect_graph, sampled_nodes, rank_nodes, single_modality, heldout_pipeline, affect_pilot, str(device))
    trajectory = [{"epoch": 0, "content_loss": float("nan"), "affect_loss": float("nan"),
                   "content_gradient_share": float("nan"), **epoch_0_diagnostics}]
    log(f"epoch=0 content_recall={trajectory[-1]['content_recall']:.4f} affect_recall={trajectory[-1]['affect_recall']:.4f} content_loss=— affect_loss=— content_grad_share=— top_eigen_fraction={trajectory[-1]['top_eigen_fraction']:.4f} effective_rank_95={trajectory[-1]['effective_rank_95']} gate_mean={trajectory[-1]['gate_mean']:.4f} gate_std={trajectory[-1]['gate_std']:.4f} gate_saturated={trajectory[-1]['gate_saturated_fraction']:.4f}")
    plateau_count, stop_reason = 0, f"reached MAX_EPOCHS={MAX_EPOCHS}"
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_rng = np.random.default_rng(SEED + epoch)
        content_pairs = sample_positive_pairs(content_edges, epoch_rng)
        affect_pairs = sample_positive_pairs(affect_edges, epoch_rng)
        content_embeddings, remapped_content_pairs = content_batch_embeddings(model, train_content, train_affect, content_pairs, device)
        content_loss = symmetric_infonce(content_embeddings, remapped_content_pairs, device)
        affect_embeddings, _mixing_weights = model(train_content, train_affect)
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
        if epoch % CHECKPOINT_EVERY:
            continue
        diagnostics = evaluate_checkpoint(model, heldout_content, heldout_affect, heldout_content_graph,
            heldout_affect_graph, sampled_nodes, rank_nodes, single_modality, heldout_pipeline, affect_pilot, str(device))
        checkpoint = {"epoch": epoch, "content_loss": float(content_loss.detach().item()),
                      "affect_loss": float(affect_loss.detach().item()), "content_gradient_share": float(gradient_share), **diagnostics}
        trajectory.append(checkpoint)
        log(f"epoch={epoch} content_recall={checkpoint['content_recall']:.4f} affect_recall={checkpoint['affect_recall']:.4f} content_loss={checkpoint['content_loss']:.4f} affect_loss={checkpoint['affect_loss']:.4f} content_grad_share={gradient_share:.4f} top_eigen_fraction={checkpoint['top_eigen_fraction']:.4f} effective_rank_95={checkpoint['effective_rank_95']} gate_mean={checkpoint['gate_mean']:.4f} gate_std={checkpoint['gate_std']:.4f} gate_saturated={checkpoint['gate_saturated_fraction']:.4f}")
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
        train_embeddings, _mixing_weights = model(train_content, train_affect)
        heldout_embeddings, _mixing_weights = model(heldout_content, heldout_affect)
    final_graph = single_modality.build_single_modality_graph("final-train-learned-student", train_embeddings.cpu().numpy(), pipeline, affect_pilot, str(device), expected_nodes=len(paintings))
    log(f"Final learned student: running Leiden (seed={SEED})...")
    communities = detect_communities(final_graph, seed=SEED)
    emotion_metrics = pipeline.external_metrics(communities, majority_emotion)
    genre_map = pipeline.load_genre_map()
    genre_indices = [index for index, painting in enumerate(paintings) if painting in genre_map]
    if not genre_indices:
        raise RuntimeError("No genre-labelled paintings overlap the train node set.")
    genre_metrics = pipeline.external_metrics([communities[index] for index in genre_indices], [genre_map[paintings[index]] for index in genre_indices])
    log(f"Final held-out learned student: building graph and running Leiden (seed={SEED})...")
    heldout_final_graph = single_modality.build_single_modality_graph("final-heldout-learned-student", heldout_embeddings.cpu().numpy(), heldout_pipeline, affect_pilot, str(device), expected_nodes=len(heldout_paintings))
    heldout_communities = detect_communities(heldout_final_graph, seed=SEED)
    heldout_emotion_metrics = heldout_pipeline.external_metrics(heldout_communities, heldout_majority_emotion)
    heldout_genre_indices = [index for index, painting in enumerate(heldout_paintings) if painting in genre_map]
    if not heldout_genre_indices:
        raise RuntimeError("No genre-labelled paintings overlap the held-out node set.")
    heldout_genre_metrics = pipeline.external_metrics([heldout_communities[index] for index in heldout_genre_indices], [genre_map[heldout_paintings[index]] for index in heldout_genre_indices])
    content_developed = relative_improvement(final["content_recall"], trajectory[0]["content_recall"]) >= 0.20
    affect_developed = relative_improvement(final["affect_recall"], trajectory[0]["affect_recall"]) >= 0.20
    gradient_collapsed = final["content_gradient_share"] < 0.05 or final["content_gradient_share"] > 0.95
    gate_collapsed = final["gate_saturated_fraction"] > 0.90
    collapsed = not content_developed or not affect_developed or gradient_collapsed or gate_collapsed
    pareto_cleared = emotion_metrics["AMI"] > 0.1236 and genre_metrics["AMI"] > 0.1954
    verdict = "Collapsed" if collapsed else ("Real success" if pareto_cleared else "Merely a compromise")
    result = {"label": label, "trajectory": trajectory, "stop_reason": stop_reason,
              "final_metrics": {"emotion": emotion_metrics, "genre": genre_metrics},
              "heldout_metrics": {"emotion": heldout_emotion_metrics, "genre": heldout_genre_metrics}, "verdict": verdict}
    log(f"Completed run; verdict={verdict}; train emotion AMI={emotion_metrics['AMI']:.4f}, genre AMI={genre_metrics['AMI']:.4f}; held-out emotion AMI={heldout_emotion_metrics['AMI']:.4f}, genre AMI={heldout_genre_metrics['AMI']:.4f}.")
    return result


def main() -> None:
    global cca_audit, CURRENT_ARCHITECTURE
    pipeline = load_sibling_module("artelingo_run_pipeline_arch_sweep", PIPELINE_PATH)
    affect_pilot = load_sibling_module("artelingo_run_affect_pilot_arch_sweep", AFFECT_PILOT_PATH)
    single_modality = load_sibling_module("artelingo_run_single_modality_arch_sweep", SINGLE_MODALITY_PATH)
    cca_audit = load_sibling_module("artelingo_run_cca_audit_arch_sweep", CCA_AUDIT_PATH)
    heldout_pipeline = load_sibling_module("artelingo_run_pipeline_arch_sweep_heldout", PIPELINE_PATH)
    heldout_pipeline.STORAGE_DIR, heldout_pipeline.TRAIN_JSON = HELDOUT_STORAGE_DIR, HELDOUT_JSON
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using {device} for learned-student training and graph construction.")
    log("Verifying and loading train CLIP features...")
    pipeline.assert_extraction_complete()
    paintings, img_nodes, txt_nodes, emotion_counts = pipeline.load_dedup_features()
    majority_emotion = [pipeline.majority(counts) for counts in emotion_counts]
    log("Extracting train GoEmotions probabilities...")
    affect_train = np.asarray(affect_pilot.extract_affect_nodes(pipeline.TRAIN_JSON, paintings, str(device)), dtype=np.float64)
    if affect_train.shape != (len(paintings), 28):
        raise RuntimeError(f"Expected train affect features ({len(paintings)}, 28), got {affect_train.shape}.")
    log("Verifying and loading held-out CLIP features...")
    heldout_pipeline.assert_extraction_complete()
    heldout_paintings, heldout_img, heldout_txt, heldout_emotion_counts = heldout_pipeline.load_dedup_features()
    heldout_majority_emotion = [heldout_pipeline.majority(counts) for counts in heldout_emotion_counts]
    if len(heldout_paintings) != HELDOUT_PAINTINGS:
        raise RuntimeError(f"Expected {HELDOUT_PAINTINGS:,} held-out paintings, got {len(heldout_paintings):,}.")
    log("Extracting held-out GoEmotions probabilities...")
    affect_heldout = np.asarray(affect_pilot.extract_affect_nodes(HELDOUT_JSON, heldout_paintings, str(device)), dtype=np.float64)
    if affect_heldout.shape != (len(heldout_paintings), 28):
        raise RuntimeError(f"Expected held-out affect features ({len(heldout_paintings)}, 28), got {affect_heldout.shape}.")
    content_train = cca_audit.content_features(img_nodes, txt_nodes, affect_pilot)
    content_heldout = cca_audit.content_features(heldout_img, heldout_txt, affect_pilot)
    log(f"Fitting train-only content PCA ({CONTENT_PCA_DIM} components)...")
    pca = PCA(n_components=CONTENT_PCA_DIM, random_state=SEED)
    content_train, content_heldout = pca.fit_transform(content_train).astype(np.float32), pca.transform(content_heldout).astype(np.float32)
    log("Building train content and affect teacher graphs...")
    _img_graph, _txt_graph, content_teacher_graph = pipeline.build_buddy_graphs(img_nodes, txt_nodes, K=pipeline.K, alpha=pipeline.ALPHA, device=str(device), connect_components=True)
    affect_teacher_graph = single_modality.build_single_modality_graph("train-affect-teacher", affect_train, pipeline, affect_pilot, str(device), expected_nodes=len(paintings))
    content_edges, affect_edges = upper_triangle_edges(content_teacher_graph), upper_triangle_edges(affect_teacher_graph)
    log(f"Teacher edge lists: content={len(content_edges):,}, affect={len(affect_edges):,}.")
    log("Building held-out content and affect reference graphs...")
    heldout_content_graph = single_modality.build_single_modality_graph("held-out-content-reference", content_heldout, heldout_pipeline, affect_pilot, str(device), expected_nodes=len(heldout_paintings))
    heldout_affect_graph = single_modality.build_single_modality_graph("held-out-affect-reference", affect_heldout, heldout_pipeline, affect_pilot, str(device), expected_nodes=len(heldout_paintings))
    diagnostic_rng = np.random.default_rng(SEED)
    sampled_nodes = diagnostic_rng.choice(len(heldout_paintings), size=EDGE_SAMPLE_SIZE, replace=False)
    rank_nodes = diagnostic_rng.choice(len(heldout_paintings), size=EFFECTIVE_RANK_SAMPLE_SIZE, replace=False)
    train_content, train_affect = torch.as_tensor(content_train, dtype=torch.float32, device=device), torch.as_tensor(affect_train, dtype=torch.float32, device=device)
    heldout_content, heldout_affect = torch.as_tensor(content_heldout, dtype=torch.float32, device=device), torch.as_tensor(affect_heldout, dtype=torch.float32, device=device)
    results = [run_architecture(label, heads, train_content, train_affect, heldout_content, heldout_affect, content_edges, affect_edges, heldout_content_graph, heldout_affect_graph, sampled_nodes, rank_nodes, single_modality, heldout_pipeline, affect_pilot, pipeline, paintings, majority_emotion, heldout_paintings, heldout_majority_emotion, device) for label, heads in ARCHITECTURES]
    CURRENT_ARCHITECTURE = None
    write_report(results)
    log(f"Wrote architecture-sweep report to {REPORT_PATH}.")


if __name__ == "__main__":
    main()
