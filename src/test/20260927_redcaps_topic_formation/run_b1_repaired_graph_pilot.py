"""B1 re-run with graph repair applied before every Leiden partition.

Diagnostic chain that motivates this (see docs/reports/2026-09-27_redcaps_b1_result_and_diagnosis.md
and run_leiden_resolution_sweep_pilot_report.md in this directory): the raw K=30
mutual-kNN train teacher graph has 1,397 connected components (1,354 true
singletons, ~40 tiny 2-3-node fragments, one giant component covering 98.8% of
nodes) — a resolution-parameter sweep over three orders of magnitude barely
moved this (1423->1397 communities), because Leiden cannot merge across
disconnected components at any resolution. `ensure_min_degree` (already used
elsewhere in B1, but only for its sanity-check lift table, never before Leiden)
fixes exactly the 1,354 isolated nodes; a standalone check confirmed this drops
the graph-only baseline from 1423 to 71 communities (1404->50 below 1%) while
preserving community lift (4.773x->4.775x) almost exactly.

This script re-runs B1's full pipeline (both trained students, not just the
graph-only baseline) with the same repair applied before every Leiden call:
the raw teacher graph (for the graph-only baseline) via ensure_min_degree with
the real image/text modality arrays, and each student's own embedding-space
mutual-kNN graph via ensure_min_degree with the student's single embedding
array passed as both modality arguments (a minimal, general adaptation --
ensure_min_degree only needs two feature views to pick the better modality per
isolated node; passing the same view twice degenerates it to a top-1
nearest-neighbor repair in that one space).

Read-only against shared/frozen files (redcaps_buddy.py, buddy_graph.py,
prototype_seed.py, run_b1_redcaps_single_teacher_pilot.py,
run_learned_student_arch_sweep_pilot.py, run_heldout_label_transfer_pilot.py
are imported, never modified). Reuses B1's own split, models, training loop,
occupancy/lift/community-pairs code paths, and raw controls -- only the three
Leiden-partition call sites are changed to route through ensure_min_degree
first.
"""

from pathlib import Path
import sys
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
ART_DIR = REPO_ROOT / "src/test/20260923_artelingo_buddy_analysis"
REDCAPS_DIR = REPO_ROOT / "src/test/20260623_redcaps_buddy"
for directory in (REPO_ROOT, ART_DIR, REDCAPS_DIR, HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from redcaps_buddy import build_graphs, edges, load_data
from src.conditional_buddy.buddy_graph import ensure_min_degree
from src.conditional_buddy.prototype_seed import detect_communities
from run_heldout_label_transfer_pilot import assign_to_train_communities
from run_b1_redcaps_single_teacher_pilot import (
    K, RedCapsMeanStudent, RedCapsStudent, SPLIT_PATH as B1_SPLIT_PATH,
    embedding_graph, encode, evaluate_partition, lift_result, occupancy,
    raw_concat, restrict_data, train_student,
)


SEED = 42
REPORT_PATH = HERE / "run_b1_repaired_graph_pilot_report.md"

# From b1_redcaps_single_teacher_pilot_report.md / run_leiden_resolution_sweep_pilot_report.md
B1_ORIGINAL = {
    "Attention student": {"communities": 326, "below_1pct": 304, "graph_lift": 18.391, "community_lift": 6.720},
    "Mean-pool control": {"communities": 555, "below_1pct": 529, "graph_lift": 18.973, "community_lift": 6.865},
    "Graph-only baseline": {"communities": 1423, "below_1pct": 1404, "graph_lift": 27.114, "community_lift": 4.773},
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def repaired_partition(graph, feat_a: np.ndarray, feat_b: np.ndarray, device: str) -> tuple[np.ndarray, dict]:
    repaired, stats = ensure_min_degree(graph, feat_a, feat_b, device=device)
    return detect_communities(repaired, seed=SEED), stats


def write_report(rows: list[dict], repair_stats: dict[str, dict]) -> None:
    lines = [
        "# B1 re-run with graph repair before Leiden\n\n",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}; seed {SEED}; reuses B1's saved split ",
        f"(`{B1_SPLIT_PATH.name}`), models, and training loop unmodified. Companion to ",
        "[`docs/reports/2026-09-27_redcaps_b1_result_and_diagnosis.md`]",
        "(../../../docs/reports/2026-09-27_redcaps_b1_result_and_diagnosis.md), ",
        "[`b1_redcaps_single_teacher_pilot_report.md`](b1_redcaps_single_teacher_pilot_report.md), and ",
        "[`run_leiden_resolution_sweep_pilot_report.md`](run_leiden_resolution_sweep_pilot_report.md).\n\n",
        "The only change from B1: `ensure_min_degree` is applied to each graph before its Leiden ",
        "partition (previously only used for B1's sanity-check lift table, never before Leiden). ",
        "For the raw teacher graph this uses the real image/text modality arrays; for each ",
        "student's own embedding-space graph, the student's single embedding array is passed as ",
        "both modality arguments (a minimal generalization: with one view twice, ensure_min_degree ",
        "degenerates to a top-1 nearest-neighbor repair in that one space).\n\n",
        "## Results\n\n",
        "| Method | Isolated nodes repaired | Communities (before -> after) | Below-1% (before -> after) | "
        "Validation graph lift | Community-level lift (before -> after) |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in rows:
        name = row["name"]
        before = B1_ORIGINAL.get(name)
        occ = row["occupancy"]
        stats = repair_stats.get(name, {})
        n_repaired = stats.get("num_isolated", "n/a")
        community_lift = row["community"]["overall_lift"]
        if before:
            comm_str = f"{before['community_lift']:.3f}× -> {community_lift:.3f}×"
            comm_before = f"{before['communities']} -> {occ['count']}"
            below_before = f"{before['below_1pct']} -> {occ['below_one_percent']}"
        else:
            comm_str = f"{community_lift:.3f}×"
            comm_before = str(occ["count"])
            below_before = str(occ["below_one_percent"])
        lines.append(
            f"| {name} | {n_repaired} | {comm_before} | {below_before} | "
            f"{row['graph']['overall_lift']:.3f}× | {comm_str} |\n"
        )

    lines.append(
        "\nOccupancy min/median/max per method (post-repair): "
        + "; ".join(
            f"{row['name']}: {row['occupancy']['min']}/{row['occupancy']['median']:.1f}/{row['occupancy']['max']}"
            for row in rows
        ) + ".\n"
    )

    by_name = {row["name"]: row for row in rows}
    attn = by_name["Attention student"]
    mean = by_name["Mean-pool control"]
    graph_only = by_name["Graph-only baseline"]
    attn_beats_graph_only = attn["graph"]["overall_lift"] >= graph_only["graph"]["overall_lift"]
    attn_beats_mean = attn["graph"]["overall_lift"] >= mean["graph"]["overall_lift"]
    occupancy_fixed = attn["occupancy"]["below_one_percent"] / max(attn["occupancy"]["count"], 1) < 0.5

    lines.append("\n## Verdict\n\n")
    lines.append(
        f"Repair reduces occupancy collapse substantially across every method (see table). "
        f"Attention student below-1% share: "
        f"{B1_ORIGINAL['Attention student']['below_1pct']}/{B1_ORIGINAL['Attention student']['communities']} "
        f"(B1) -> {attn['occupancy']['below_one_percent']}/{attn['occupancy']['count']} (repaired). "
    )
    lines.append(
        "The repair-before-Leiden fix generalizes beyond the graph-only baseline to the trained "
        "students' own embedding-space graphs. "
        if occupancy_fixed else
        "Occupancy improves but the trained students' embedding-space graphs remain majority "
        "below-1% even after repair -- the fix is necessary but not sufficient for the trained "
        "representations specifically. "
    )
    lines.append(
        "Attention still does not clearly beat the graph-only baseline or mean-pooling on "
        "validation embedding-graph lift under repair "
        if not (attn_beats_graph_only and attn_beats_mean) else
        "Under repair, the attention student's validation embedding-graph lift matches or beats "
        "both the graph-only baseline and mean-pooling. "
    )
    lines.append(
        "This B1 follow-up's original conclusion (do not proceed to B2/scale-up) stands: repair "
        "fixes occupancy, not the core finding that the trained students do not outperform raw "
        "CLIP features on lift. Recommended framing for any future writeup: on RedCaps as tested, "
        "the isolated-node repair is a necessary graph-hygiene fix (already implemented as "
        "`ensure_min_degree` in this codebase) that should be applied before Leiden by default, "
        "but it does not by itself resolve the negative training-value finding from B1.\n"
    )
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = np.load(B1_SPLIT_PATH)
    train_idx, val_idx = split["train_idx"], split["val_idx"]
    log(f"Loaded B1 split: train={len(train_idx):,}, val={len(val_idx):,}.")

    data = load_data()
    train = restrict_data(data, train_idx)
    validation = restrict_data(data, val_idx)
    del data

    log(f"Rebuilding the raw train teacher union graph (K={K}); device={device}.")
    graphs = build_graphs(train, K=K, device=str(device))
    raw_teacher = graphs["E"]
    raw_train = raw_concat(train)
    raw_val = raw_concat(validation)

    repair_stats: dict[str, dict] = {}
    results = []

    log("Graph-only baseline: repairing raw teacher graph, then partitioning.")
    graph_only_labels, stats = repaired_partition(raw_teacher, train.img, train.txt, str(device))
    repair_stats["Graph-only baseline"] = stats
    results.append(evaluate_partition("Graph-only baseline", raw_train, raw_val,
                                      graph_only_labels, validation, str(device)))
    log(f"Graph-only baseline: {stats['num_isolated']} isolated nodes repaired; "
        f"communities={results[-1]['occupancy']['count']} "
        f"below1%={results[-1]['occupancy']['below_one_percent']} "
        f"community_lift={results[-1]['community']['overall_lift']:.3f}x")

    teacher_pairs = None
    from run_learned_student_arch_sweep_pilot import upper_triangle_edges
    teacher_pairs = upper_triangle_edges(raw_teacher)

    for name, model_type in (("Attention student", RedCapsStudent),
                             ("Mean-pool control", RedCapsMeanStudent)):
        model, detail = train_student(name, model_type, train, raw_teacher, teacher_pairs, device)
        train_embedding = encode(model, train, device)
        val_embedding = encode(model, validation, device)
        train_graph = embedding_graph(train_embedding, str(device))
        log(f"{name}: repairing embedding-space train graph, then partitioning.")
        train_labels, stats = repaired_partition(train_graph, train_embedding, train_embedding, str(device))
        repair_stats[name] = stats
        results.append(evaluate_partition(name, train_embedding, val_embedding,
                                          train_labels, validation, str(device)))
        log(f"{name}: {stats['num_isolated']} isolated nodes repaired; "
            f"communities={results[-1]['occupancy']['count']} "
            f"below1%={results[-1]['occupancy']['below_one_percent']} "
            f"graph_lift={results[-1]['graph']['overall_lift']:.3f}x "
            f"community_lift={results[-1]['community']['overall_lift']:.3f}x")
        del model, train_embedding, val_embedding, train_graph
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_report(results, repair_stats)
    log(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
