"""Run the seed-42 RedCaps 150k single-teacher Stage 1 pilot on a local GPU.

The final test indices are saved but never evaluated. This script produces the
split artifact and Markdown report only when its main function is run.
"""

from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
ART_DIR = REPO_ROOT / "src/test/20260923_artelingo_buddy_analysis"
REDCAPS_DIR = REPO_ROOT / "src/test/20260623_redcaps_buddy"
for directory in (REPO_ROOT, ART_DIR, REDCAPS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from redcaps_buddy import ANNOT, STORAGE, Data, build_graphs, edges, load_data, subreddit_lift
from src.conditional_buddy.buddy_graph import ensure_min_degree, mutual_knn
from run_learned_student_arch_sweep_pilot import (
    BATCH_SIZE, CHECKPOINT_EVERY, MAX_EPOCHS, PLATEAU_REL_IMPROVEMENT,
    PLATEAU_WINDOW, detect_communities, relative_improvement,
    sample_positive_pairs, symmetric_infonce, upper_triangle_edges,
)
from run_heldout_label_transfer_pilot import assign_to_train_communities


SEED = 42
K = 30
TRANSFER_K = 20
LEARNING_RATE = 1e-3
EMBED_BATCH_SIZE = 4096
RECALL_SAMPLE_SIZE = 2000
COMMUNITY_PAIR_NODE_CAP = 2000
EXPERIMENT_16_LIFT = 22.788  # 150k / K=30, repaired union, Stage A table.
SANITY_REL_TOLERANCE = 0.25  # "Same ballpark" gate for the 120k train subsample.
REPORT_PATH = HERE / "b1_redcaps_single_teacher_pilot_report.md"
SPLIT_PATH = HERE / "b1_redcaps_single_teacher_pilot_split.npz"
EXPERIMENT_16_REPORT = "docs/reports/2026-09-01_buddy_k_scaling_stage_a.md"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def restrict_data(data: Data, indices: np.ndarray) -> Data:
    """Keep global subreddit IDs/names aligned while selecting feature rows."""
    return Data(
        img=np.ascontiguousarray(data.img[indices]),
        txt=np.ascontiguousarray(data.txt[indices]),
        sample_ids=[data.sample_ids[int(i)] for i in indices],
        sub_id=data.sub_id[indices],
        sub_names=data.sub_names,
        records=[data.records[int(i)] for i in indices],
    )


class RedCapsStudent(nn.Module):
    """Two CLIP tokens, one-head self-attention, one normalized embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.proj_img = nn.Linear(512, 32)
        self.proj_txt = nn.Linear(512, 32)
        # Faithful two-token copy of arch-sweep AttentionFusion, not its art class.
        self.attn = nn.MultiheadAttention(embed_dim=32, num_heads=1, batch_first=True)
        self.norm = nn.LayerNorm(32)

    def forward(self, img_features: torch.Tensor, txt_features: torch.Tensor) -> torch.Tensor:
        img_token = F.normalize(self.proj_img(img_features), dim=1)
        txt_token = F.normalize(self.proj_txt(txt_features), dim=1)
        tokens = torch.stack((img_token, txt_token), dim=1)
        attended, _ = self.attn(
            tokens, tokens, tokens, need_weights=True, average_attn_weights=True
        )
        return F.normalize(self.norm(attended.mean(dim=1)), dim=1)


class RedCapsMeanStudent(nn.Module):
    """Same input projections and output width, with plain token mean pooling."""

    def __init__(self) -> None:
        super().__init__()
        self.proj_img = nn.Linear(512, 32)
        self.proj_txt = nn.Linear(512, 32)
        self.norm = nn.LayerNorm(32)

    def forward(self, img_features: torch.Tensor, txt_features: torch.Tensor) -> torch.Tensor:
        img_token = F.normalize(self.proj_img(img_features), dim=1)
        txt_token = F.normalize(self.proj_txt(txt_features), dim=1)
        return F.normalize(self.norm((img_token + txt_token) * 0.5), dim=1)


def embedding_graph(embeddings: np.ndarray, device: str):
    """One space has one mutual-kNN graph; no two-modality union is needed."""
    return mutual_knn(embeddings, K=K, device=device)


@torch.no_grad()
def encode(model: nn.Module, data: Data, device: torch.device) -> np.ndarray:
    model.eval()
    result = np.empty((data.n, 32), dtype=np.float32)
    for start in range(0, data.n, EMBED_BATCH_SIZE):
        stop = min(start + EMBED_BATCH_SIZE, data.n)
        img = torch.from_numpy(data.img[start:stop]).to(device)
        txt = torch.from_numpy(data.txt[start:stop]).to(device)
        result[start:stop] = model(img, txt).cpu().numpy()
    return result


def sampled_neighbor_recall(reference, comparison, sampled_nodes: np.ndarray) -> float:
    """Mean per-node recall, matching the ArtELingo checkpoint convention."""
    scores = []
    for node in sampled_nodes:
        a = reference.getrow(int(node)).indices
        if len(a):
            b = comparison.getrow(int(node)).indices
            scores.append(float(np.isin(a, b).mean()))
    if not scores:
        raise RuntimeError("No sampled train nodes have teacher neighbors.")
    return float(np.mean(scores))


def train_student(name: str, model_type: type[nn.Module], train: Data, teacher,
                  teacher_edges: np.ndarray, device: torch.device) -> tuple[nn.Module, dict]:
    """Use sampled nodes for every loss; full embeddings only at checkpoints."""
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model = model_type().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    recall_nodes = np.random.default_rng(SEED).choice(
        train.n, size=min(RECALL_SAMPLE_SIZE, train.n), replace=False
    )
    train_img = torch.from_numpy(train.img)
    train_txt = torch.from_numpy(train.txt)

    def checkpoint_recall() -> float:
        embedding = encode(model, train, device)
        graph = embedding_graph(embedding, str(device))
        return sampled_neighbor_recall(teacher, graph, recall_nodes)

    previous_recall = checkpoint_recall()
    trajectory = [{"epoch": 0, "loss": None, "teacher_recall": previous_recall}]
    plateau_count = 0
    stop_reason = f"reached MAX_EPOCHS={MAX_EPOCHS}"
    log(f"{name}: epoch=0 train teacher recall={previous_recall:.4f}")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        pair_rng = np.random.default_rng(SEED + epoch)
        pairs = sample_positive_pairs(teacher_edges, pair_rng)
        node_ids, inverse = np.unique(pairs.reshape(-1), return_inverse=True)
        img = train_img[node_ids].to(device)
        txt = train_txt[node_ids].to(device)
        remapped_pairs = inverse.reshape(-1, 2)
        optimizer.zero_grad(set_to_none=True)
        loss = symmetric_infonce(model(img, txt), remapped_pairs, device)
        loss.backward()
        optimizer.step()
        if epoch % CHECKPOINT_EVERY:
            continue
        recall = checkpoint_recall()
        trajectory.append({
            "epoch": epoch, "loss": float(loss.detach().item()), "teacher_recall": recall
        })
        log(f"{name}: epoch={epoch} loss={loss.item():.4f} train teacher recall={recall:.4f}")
        improved = relative_improvement(recall, previous_recall)
        plateau_count = plateau_count + 1 if improved < PLATEAU_REL_IMPROVEMENT else 0
        previous_recall = recall
        if plateau_count >= PLATEAU_WINDOW:
            stop_reason = f"recall plateaued for {PLATEAU_WINDOW} consecutive checkpoints"
            log(f"{name}: {stop_reason}; stopping at epoch {epoch}")
            break
    return model, {"trajectory": trajectory, "stop_reason": stop_reason,
                   "parameters": sum(parameter.numel() for parameter in model.parameters())}


def occupancy(labels: np.ndarray) -> dict:
    _, counts = np.unique(labels, return_counts=True)
    return {
        "count": int(len(counts)),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "median": float(np.median(counts)),
        "below_one_percent": int(np.count_nonzero(counts < 0.01 * len(labels))),
    }


def lift_result(data: Data, pairs: np.ndarray) -> dict:
    if len(pairs) == 0:
        return {"overall_lift": float("nan"), "obs_same_frac": float("nan"),
                "exp_same_frac": float("nan"), "n_qualifying": 0, "n_edges": 0}
    result = subreddit_lift(data, pairs, top_k=None)
    return {
        "overall_lift": result["overall_lift"],
        "obs_same_frac": result["obs_same_frac"],
        "exp_same_frac": result["exp_same_frac"],
        "n_qualifying": len(result["top_enriched"]),
        "n_edges": int(len(pairs)),
    }


def community_pairs(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Bound large communities by sampling 2,000 members before pair creation."""
    rng = np.random.default_rng(SEED)
    groups = []
    subsampled = 0
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label).astype(np.int32)
        if len(members) > COMMUNITY_PAIR_NODE_CAP:
            members = rng.choice(members, size=COMMUNITY_PAIR_NODE_CAP, replace=False)
            subsampled += 1
        if len(members) < 2:
            continue
        left, right = np.triu_indices(len(members), k=1)
        groups.append(np.column_stack((members[left], members[right])))
    if not groups:
        return np.empty((0, 2), dtype=np.int32), subsampled
    return np.concatenate(groups), subsampled


def raw_concat(data: Data) -> np.ndarray:
    """Equal-weight cosine distance from both normalized raw CLIP modalities."""
    joined = np.concatenate((data.img, data.txt), axis=1)
    norms = np.linalg.norm(joined, axis=1, keepdims=True)
    return np.ascontiguousarray(joined / np.maximum(norms, 1e-12))


def evaluate_partition(name: str, train_embedding: np.ndarray, val_embedding: np.ndarray,
                       train_labels: np.ndarray, validation: Data, device: str,
                       val_graph=None) -> dict:
    transferred = assign_to_train_communities(
        train_embedding, train_labels, val_embedding, k=TRANSFER_K
    )
    coverage = len(np.unique(transferred))
    fallback_count = assign_to_train_communities.last_fallback_count
    if val_graph is None:
        val_graph = embedding_graph(val_embedding, device)
    graph_lift = lift_result(validation, edges(val_graph))
    pairs, subsampled = community_pairs(transferred)
    community_lift = lift_result(validation, pairs)
    log(f"{name}: validation graph lift={graph_lift['overall_lift']:.3f}x; "
        f"community lift={community_lift['overall_lift']:.3f}x; "
        f"coverage={coverage}/{len(np.unique(train_labels))}; fallbacks={fallback_count}")
    return {
        "name": name, "graph": graph_lift, "community": community_lift,
        "occupancy": occupancy(train_labels), "coverage": coverage,
        "fallbacks": fallback_count, "subsampled_communities": subsampled,
    }


def fmt_lift(result: dict) -> str:
    return f"{result['overall_lift']:.3f}× ({result['obs_same_frac']:.4f}/{result['exp_same_frac']:.4f})"


def report(split_sizes: tuple[int, int, int], teacher_raw: dict, teacher_repaired: dict,
           edge_types: dict, repair_stats: dict, results: list[dict], training: dict | None,
           sanity_ok: bool) -> None:
    reference = f"[{EXPERIMENT_16_REPORT}]({REPO_ROOT / EXPERIMENT_16_REPORT})"
    lines = [
        "# B1 — RedCaps single-teacher Stage 1 pilot\n\n",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}; seed {SEED}; ",
        f"150k source `STORAGE={STORAGE}` and `ANNOT={ANNOT}`.\n\n",
        "## Split and teacher sanity check\n\n",
        f"Train/validation/test sizes: **{split_sizes[0]:,}/{split_sizes[1]:,}/{split_sizes[2]:,}**. ",
        f"The shuffled index arrays are saved at `{SPLIT_PATH}`. ",
        "The split uses `default_rng(42).shuffle(arange(N))`, with no subreddit lookup; ",
        "test rows are saved but never evaluated by this pilot.\n\n",
        f"Train raw union: **{fmt_lift(teacher_raw)}**, {teacher_raw['n_edges']:,} edges, ",
        f"{teacher_raw['n_qualifying']} qualifying subreddits. ",
        f"Train degree-repaired union: **{fmt_lift(teacher_repaired)}**, ",
        f"{teacher_repaired['n_edges']:,} edges, {teacher_repaired['n_qualifying']} qualifying subreddits. ",
        f"Repair fixed {repair_stats['num_isolated']:,} isolated nodes and added ",
        f"{edge_types['repair']:,} undirected edges. Edge types: ",
        f"image-only {edge_types['img_only']:,}, text-only {edge_types['txt_only']:,}, ",
        f"both {edge_types['both']:,}, repair {edge_types['repair']:,}.\n\n",
        f"Experiment 16 Stage A reports **{EXPERIMENT_16_LIFT:.3f}×** for the **full 150k, ",
        f"K=30 repaired union** ({reference}). The train-only repaired union differs by ",
        f"{100 * (teacher_repaired['overall_lift'] / EXPERIMENT_16_LIFT - 1):+.1f}% and ",
        f"the raw union by {100 * (teacher_raw['overall_lift'] / EXPERIMENT_16_LIFT - 1):+.1f}%. ",
        "The repaired figure is the primary comparison because the cited protocol includes repair. ",
        ("This is within the declared 25% same-ballpark gate.\n\n" if sanity_ok else
         "This is outside the declared 25% same-ballpark gate. Check data loading/split; training was stopped.\n\n"),
    ]
    if not sanity_ok:
        REPORT_PATH.write_text("".join(lines), encoding="utf-8")
        return

    lines.extend([
        "## Validation results\n\n",
        "Lift entries show overall lift × (observed/expected same-subreddit fraction); ",
        "expectation uses the edge-endpoint subreddit marginal.\n\n",
        "| Method | Validation embedding-graph lift | Community-level lift | ",
        "Train Leiden occupancy (C; min/median/max; below 1%) | Transfer coverage | Fallbacks |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ])
    for item in results:
        occ = item.get("occupancy")
        occ_str = (f"{occ['count']}; {occ['min']}/{occ['median']:.1f}/{occ['max']}; "
                   f"{occ['below_one_percent']}" if occ else "n/a")
        coverage = f"{item['coverage']}/{occ['count']}" if occ else "n/a"
        fallbacks = str(item["fallbacks"]) if occ else "n/a"
        community = fmt_lift(item["community"]) if item.get("community") else "n/a"
        lines.append(f"| {item['name']} | {fmt_lift(item['graph'])} | {community} | "
                     f"{occ_str} | {coverage} | {fallbacks} |\n")

    lines.append("\n### Lift support and sampling\n\n")
    for item in results:
        graph = item["graph"]
        lines.append(f"- {item['name']}: graph {graph['n_edges']:,} edges, "
                     f"{graph['n_qualifying']} qualifying subreddits")
        if item.get("community"):
            community = item["community"]
            lines.append(f"; same-community {community['n_edges']:,} sampled pairs, "
                         f"{community['n_qualifying']} qualifying subreddits, "
                         f"{item['subsampled_communities']} communities above 2,000 points subsampled")
        lines.append(".\n")
    lines.extend([
        "\nFor any transferred community above 2,000 validation points, a seed-42 random ",
        "sample of 2,000 members is selected before pairs are enumerated. Thus no large ",
        "community creates its full dense pair set (maximum 1,999,000 pairs per community).\n\n",
        "Training positive pairs and the graph-only Leiden partition use the raw train ",
        "teacher union `E`; the repaired graph is reported for the Experiment 16 sanity ",
        "comparison only. Graph-only cosine k=20 transfer uses unit-normalized, ",
        "concatenated raw image+text CLIP features; its validation embedding graph ",
        "uses the same features at K=30. ",
        "Both modalities therefore contribute ",
        "equally in this no-training control. Raw image/text controls have no Leiden labels.\n\n",
        "## Training\n\n",
        f"Both students use seed {SEED}, Adam at {LEARNING_RATE:g}, {BATCH_SIZE} sampled "
        f"positive pairs per epoch, K={K}, a {MAX_EPOCHS}-epoch ceiling, and a ",
        f"{CHECKPOINT_EVERY}-epoch recall checkpoint with {PLATEAU_WINDOW} consecutive "
        f"relative gains below {PLATEAU_REL_IMPROVEMENT:.2f} as the stopping rule. ",
        f"Recall is mean per-node raw-train-teacher-neighbor recall on a fixed seed-42 ",
        f"sample of {RECALL_SAMPLE_SIZE:,} train nodes.\n\n",
    ])
    for name, detail in training.items():
        last = detail["trajectory"][-1]
        lines.append(f"- {name}: {detail['parameters']:,} trainable parameters; "
                     f"stopped at epoch {last['epoch']} ({detail['stop_reason']}); "
                     f"final sampled train teacher recall {last['teacher_recall']:.4f}.\n")
    lines.append("Both controls have the same two 512→32 projections, 32-D output, and "
                 "LayerNorm. Attention adds its own Q/K/V/output weights; the mean-pool "
                 "control has no learned attention. Parameter counts are therefore shown "
                 "rather than claimed to be identical.\n")

    by_name = {item["name"]: item for item in results}
    attn = by_name["Attention student"]
    mean = by_name["Mean-pool control"]
    controls = [by_name[name] for name in
                ("Mean-pool control", "Raw image control", "Raw text control", "Graph-only baseline")]
    attn_lift = attn["graph"]["overall_lift"]
    attention_beats_all = bool(np.isfinite(attn_lift) and all(
        attn_lift >= item["graph"]["overall_lift"] for item in controls
    ))
    attn_better = bool(attn_lift > mean["graph"]["overall_lift"])
    external_controls = [by_name[name] for name in
                         ("Raw image control", "Raw text control", "Graph-only baseline")]
    b2_candidates = [item for item in (attn, mean) if
                     item["occupancy"]["count"] > 1 and item["coverage"] > 1 and
                     np.isfinite(item["graph"]["overall_lift"]) and all(
                         item["graph"]["overall_lift"] >= control["graph"]["overall_lift"]
                         for control in external_controls)]
    proceed = bool(b2_candidates)
    community_comparison = (attn["community"]["overall_lift"] >=
                            mean["community"]["overall_lift"] and
                            attn["community"]["overall_lift"] >=
                            by_name["Graph-only baseline"]["community"]["overall_lift"])
    lines.extend([
        "\n## Verdict\n\n",
        ("The attention student preserves or improves validation embedding-graph "
         "subreddit lift relative to every listed control. " if attention_beats_all else
         "The attention student does not match every listed validation embedding-graph "
         "lift control. "),
        ("Attention improves over mean pooling on this lift measure. " if attn_better else
         "Attention does not improve over mean pooling on this lift measure. "),
        ("Attention also matches or improves both community-level lift controls. "
         if community_comparison else
         "Attention does not match both community-level lift controls. "),
        (f"B2 candidate(s): {', '.join(item['name'] for item in b2_candidates)}. "
         "Each has a populated partition and transfer and matches all raw-feature "
         "and graph-only embedding-graph lift controls. Proceed to a paired multi-seed "
         "check and B2 patch-feature Stage 2 at 150k; reserve 300k/500k DAS6 runs "
         "until B2's downstream probe justifies them.\n" if proceed else
         "Stop before B2 patch extraction or 300k/500k DAS6 training and diagnose "
         "the teacher, student geometry, and transfer coverage.\n"),
        "This one-seed validation gate is not a sealed-test or downstream-topic claim.\n",
    ])
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Loading 150k RedCaps features using source defaults; device={device}.")
    data = load_data()
    if data.img.shape[1] != 512 or data.txt.shape[1] != 512:
        raise RuntimeError("Expected 512-D image and text CLIP features.")
    indices = np.arange(data.n, dtype=np.int64)
    np.random.default_rng(SEED).shuffle(indices)
    n_train = round(0.8 * data.n)
    n_val = round(0.1 * data.n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    np.savez(SPLIT_PATH, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    train = restrict_data(data, train_idx)
    validation = restrict_data(data, val_idx)
    del data
    log(f"Split: train={train.n:,}, validation={validation.n:,}, test={len(test_idx):,} saved to {SPLIT_PATH}.")

    graphs = build_graphs(train, K=K, device=str(device))
    raw_teacher = graphs["E"]
    repaired_teacher, repair_stats = ensure_min_degree(
        raw_teacher, train.img, train.txt, device=str(device)
    )
    raw_edges = edges(raw_teacher)
    repaired_edges = edges(repaired_teacher)
    both_edges = len(edges(graphs["B"]))
    edge_types = {
        "img_only": len(edges(graphs["A_img"])) - both_edges,
        "txt_only": len(edges(graphs["A_txt"])) - both_edges,
        "both": both_edges,
        "repair": len(repaired_edges) - len(raw_edges),
    }
    if sum(edge_types.values()) != len(repaired_edges):
        raise RuntimeError("Teacher edge-type counts do not sum to the repaired graph.")
    teacher_raw_lift = lift_result(train, raw_edges)
    teacher_repaired_lift = lift_result(train, repaired_edges)
    sanity_ok = all(
        np.isfinite(result["overall_lift"])
        and abs(result["overall_lift"] / EXPERIMENT_16_LIFT - 1) <= SANITY_REL_TOLERANCE
        for result in (teacher_raw_lift, teacher_repaired_lift)
    )
    log(f"Train teacher: raw={teacher_raw_lift['overall_lift']:.3f}x, "
        f"repaired={teacher_repaired_lift['overall_lift']:.3f}x; "
        f"Experiment 16 full-150k repaired={EXPERIMENT_16_LIFT:.3f}x.")
    if not sanity_ok:
        report((train.n, validation.n, len(test_idx)), teacher_raw_lift,
               teacher_repaired_lift, edge_types, repair_stats, [], None, False)
        raise RuntimeError("Train teacher lift is outside the 25% Experiment 16 sanity gate; see report.")

    teacher_pairs = upper_triangle_edges(raw_teacher)
    raw_train = raw_concat(train)
    raw_val = raw_concat(validation)
    log("Partitioning the raw train teacher union for the graph-only baseline.")
    graph_only_labels = detect_communities(raw_teacher, seed=SEED)

    results = []
    training = {}
    for name, model_type in (("Attention student", RedCapsStudent),
                             ("Mean-pool control", RedCapsMeanStudent)):
        model, detail = train_student(name, model_type, train, raw_teacher,
                                      teacher_pairs, device)
        training[name] = detail
        train_embedding = encode(model, train, device)
        val_embedding = encode(model, validation, device)
        train_graph = embedding_graph(train_embedding, str(device))
        train_labels = detect_communities(train_graph, seed=SEED)
        results.append(evaluate_partition(name, train_embedding, val_embedding,
                                          train_labels, validation, str(device)))
        del model, train_embedding, val_embedding, train_graph
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for name, embedding in (("Raw image control", validation.img),
                            ("Raw text control", validation.txt)):
        graph = embedding_graph(embedding, str(device))
        results.append({"name": name, "graph": lift_result(validation, edges(graph))})

    results.append(evaluate_partition("Graph-only baseline", raw_train, raw_val,
                                      graph_only_labels, validation, str(device)))
    report((train.n, validation.n, len(test_idx)), teacher_raw_lift,
           teacher_repaired_lift, edge_types, repair_stats, results, training, True)
    log(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
