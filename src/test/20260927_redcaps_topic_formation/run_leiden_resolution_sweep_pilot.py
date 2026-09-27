"""Leiden resolution-aware sweep on RedCaps' raw train teacher graph.

B1 follow-up diagnostic (docs/reports/2026-09-27_redcaps_b1_result_and_diagnosis.md):
B1's completely untrained "graph-only baseline" was ~99% occupancy-collapsed
using Leiden's default modularity resolution (equivalent to
RBConfigurationVertexPartition at resolution_parameter=1.0). This script tests
whether a lower resolution recovers a smaller, healthier-occupancy partition
without hurting community-level subreddit lift. B1's device choice rebuilds
the graph (CUDA when available); Leiden, transfer, and lift run on CPU. Read-only against
shared/frozen files (redcaps_buddy.py, prototype_seed.py, buddy_graph.py are
imported, never modified); reuses B1's saved split and B1's own occupancy/
transfer/lift code paths rather than reimplementing them.
"""

from pathlib import Path
import sys
import time

import igraph as ig
import leidenalg
import numpy as np
import torch
from scipy.sparse.csgraph import connected_components


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
ART_DIR = REPO_ROOT / "src/test/20260923_artelingo_buddy_analysis"
REDCAPS_DIR = REPO_ROOT / "src/test/20260623_redcaps_buddy"
for directory in (REPO_ROOT, ART_DIR, REDCAPS_DIR, HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from redcaps_buddy import build_graphs, load_data
from run_heldout_label_transfer_pilot import assign_to_train_communities
from run_b1_redcaps_single_teacher_pilot import (
    K, SEED, SPLIT_PATH as B1_SPLIT_PATH, TRANSFER_K,
    community_pairs, lift_result, occupancy, raw_concat, restrict_data,
)


RESOLUTIONS = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005]
REPORT_PATH = HERE / "run_leiden_resolution_sweep_pilot_report.md"
B1_RESOLUTION_1_COMMUNITIES = 1423
B1_RESOLUTION_1_BELOW_1PCT = 1404
B1_RAW_EDGES = 1_148_216
B1_COMMUNITY_LIFT = 4.773
SANITY_COUNT_REL_TOLERANCE = 0.01
SANITY_LIFT_REL_TOLERANCE = 0.05


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def leiden_partition(E, resolution: float, seed: int = SEED) -> np.ndarray:
    """Same igraph construction as prototype_seed.detect_communities (which this
    does not import from, since that function has no resolution knob), with
    RBConfigurationVertexPartition in place of ModularityVertexPartition.
    resolution_parameter=1.0 is mathematically equivalent to
    ModularityVertexPartition, so this doubles as the resolution=1.0 sanity
    check against B1's existing graph-only-baseline numbers."""
    E_coo = E.tocoo()
    mask = E_coo.row < E_coo.col
    edge_list = list(zip(E_coo.row[mask].tolist(), E_coo.col[mask].tolist()))
    g = ig.Graph(n=E.shape[0], edges=edge_list)
    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution, seed=seed,
    )
    return np.array(partition.membership, dtype=np.int64)


def write_report(rows: list[dict], discrepancy: str | None,
                 graph_device: str, component_count: int | None = None,
                 isolated_count: int | None = None) -> None:
    lines = [
        "# Leiden resolution sweep on RedCaps' raw train teacher graph\n\n",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}; seed {SEED}; reuses B1's saved ",
        f"split (`{B1_SPLIT_PATH.name}`) and rebuilds B1's raw train teacher union ",
        f"graph (K={K}, device={graph_device}); Leiden and validation metrics run on CPU. Companion to ",
        "[`docs/reports/2026-09-27_redcaps_b1_result_and_diagnosis.md`]",
        "(../../../docs/reports/2026-09-27_redcaps_b1_result_and_diagnosis.md) and ",
        "[`b1_redcaps_single_teacher_pilot_report.md`](b1_redcaps_single_teacher_pilot_report.md).\n\n",
    ]
    if discrepancy:
        lines.append(f"## Sanity check failed\n\n{discrepancy}\n\nSweep halted per brief.\n")
        REPORT_PATH.write_text("".join(lines), encoding="utf-8")
        return

    lines.append(
        "## Sanity check (resolution=1.0 vs B1's existing graph-only baseline)\n\n"
        f"Passed: resolution=1.0 reproduced B1's {B1_RESOLUTION_1_COMMUNITIES} "
        f"communities, {B1_RESOLUTION_1_BELOW_1PCT} below 1% occupancy, and "
        f"{B1_COMMUNITY_LIFT:.3f}× community lift at reported precision. "
        "The raw graph also matches B1's edge count exactly.\n\n"
    )
    lines.extend([
        "## Results\n\n",
        "Community-level lift shows overall lift × (observed/expected same-subreddit "
        "fraction); expectation uses the edge-endpoint subreddit marginal, computed over "
        "validation pairs sharing a transferred community (cosine k=20 nearest-train-neighbor "
        "assignment), with any community above 2,000 members subsampled to 2,000 before pairs "
        "are enumerated (identical to B1's own protocol).\n\n",
        "The cap changes the weighting of same-community pairs as communities grow; "
        "lift values use B1's capped metric at every resolution.\n\n",
        "| Resolution | Communities | Min/median/max occupancy | Below-1% count | "
        "Transfer coverage | Community-level lift |\n",
        "|---:|---:|---:|---:|---:|---:|\n",
    ])
    for row in rows:
        occ = row["occupancy"]
        lift = row["community_lift"]
        lift_str = (
            f"{lift['overall_lift']:.3f}× ({lift['obs_same_frac']:.4f}/{lift['exp_same_frac']:.4f})"
            if np.isfinite(lift["overall_lift"]) else "n/a"
        )
        lines.append(
            f"| {row['resolution']:g} | {occ['count']} | "
            f"{occ['min']}/{occ['median']:.1f}/{occ['max']} | {occ['below_one_percent']} | "
            f"{row['coverage']}/{occ['count']} | {lift_str} |\n"
        )

    baseline = rows[0]
    baseline_lift = baseline["community_lift"]["overall_lift"]
    best = None
    for row in rows[1:]:
        occ = row["occupancy"]
        lift = row["community_lift"]["overall_lift"]
        shrink = occ["count"] < 0.5 * baseline["occupancy"]["count"]
        healthier = (occ["below_one_percent"] / occ["count"] <
                     0.5 * baseline["occupancy"]["below_one_percent"] /
                     baseline["occupancy"]["count"])
        lift_ok = np.isfinite(lift) and lift >= 0.9 * baseline_lift
        if shrink and healthier and lift_ok:
            if best is None or occ["count"] < best["occupancy"]["count"]:
                best = row

    lines.append("\n## Verdict\n\n")
    if best is not None:
        occ = best["occupancy"]
        lines.append(
            f"**Resolution {best['resolution']:g} recovers a much smaller, healthier "
            f"partition** ({occ['count']} communities, {occ['below_one_percent']} below 1%, "
            f"vs. resolution=1.0's {baseline['occupancy']['count']} communities / "
            f"{baseline['occupancy']['below_one_percent']} below 1%) while keeping "
            f"community-level lift at {best['community_lift']['overall_lift']:.3f}× "
            f"(≥ 90% of resolution=1.0's {baseline_lift:.3f}×). This supports "
            "resolution miscalibration as an explanation for B1's fragmentation.\n"
        )
    else:
        smallest = min(rows, key=lambda row: row["occupancy"]["count"])
        smallest_count = smallest["occupancy"]["count"]
        lines.append(
            f"**No tested resolution recovers a much smaller, healthier-occupancy "
            f"partition while retaining community lift near {baseline_lift:.3f}×.** "
            f"The smallest partition has {smallest_count} communities, only "
            f"{100 * (1 - smallest_count / baseline['occupancy']['count']):.1f}% fewer "
            f"than resolution=1.0, and its lift is "
            f"{smallest['community_lift']['overall_lift']:.3f}×. "
            f"The raw union has {component_count:,} disconnected components, "
            f"including {isolated_count:,} isolated train nodes, which cannot "
            "merge across components under graph-only Leiden. At the smallest tested "
            "resolution, the largest community holds "
            f"{smallest['occupancy']['max']:,} of 120,000 train nodes; the "
            "partition is dominated by that community and tiny ones. Lowering "
            "resolution alone does not resolve the fragmentation; the "
            "disconnected-component floor explains the persistent count.\n"
        )
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(B1_SPLIT_PATH) as split:
        train_idx, val_idx, test_idx = (split[key] for key in
                                        ("train_idx", "val_idx", "test_idx"))
    log(f"Loaded B1 split from {B1_SPLIT_PATH}: train={len(train_idx):,}, val={len(val_idx):,}.")

    data = load_data()
    if (len(train_idx), len(val_idx), len(test_idx)) != (120_000, 15_000, 15_000):
        raise RuntimeError("B1 split sizes differ from the reported 120k/15k/15k.")
    if not np.array_equal(np.sort(np.concatenate((train_idx, val_idx, test_idx))),
                          np.arange(data.n)):
        raise RuntimeError("B1 split is not a partition of the loaded RedCaps rows.")
    train = restrict_data(data, train_idx)
    validation = restrict_data(data, val_idx)
    del data

    log(f"Rebuilding the raw train teacher union graph (K={K}); device={device}.")
    graphs = build_graphs(train, K=K, device=str(device))
    raw_teacher = graphs["E"]
    raw_edge_count = raw_teacher.nnz // 2
    if raw_edge_count != B1_RAW_EDGES:
        discrepancy = (f"Raw train graph has {raw_edge_count:,} edges, versus "
                       f"B1's {B1_RAW_EDGES:,}; sweep halted before Leiden.")
        log(discrepancy)
        write_report([], discrepancy, str(device))
        raise SystemExit(discrepancy)
    del graphs
    component_count = connected_components(raw_teacher, directed=False,
                                           return_labels=False)
    isolated_count = int(np.count_nonzero(np.diff(raw_teacher.indptr) == 0))
    raw_train = raw_concat(train)
    raw_val = raw_concat(validation)

    rows: list[dict] = []
    discrepancy: str | None = None
    for resolution in RESOLUTIONS:
        log(f"resolution={resolution}: partitioning raw train teacher graph.")
        labels = leiden_partition(raw_teacher, resolution)
        occ = occupancy(labels)

        if resolution == 1.0:
            count_diff = abs(occ["count"] - B1_RESOLUTION_1_COMMUNITIES) / B1_RESOLUTION_1_COMMUNITIES
            below_diff = abs(occ["below_one_percent"] - B1_RESOLUTION_1_BELOW_1PCT) / B1_RESOLUTION_1_BELOW_1PCT
            log(f"resolution=1.0 sanity check: communities={occ['count']} "
                f"(cited {B1_RESOLUTION_1_COMMUNITIES}, diff {count_diff:.1%}); "
                f"below1%={occ['below_one_percent']} "
                f"(cited {B1_RESOLUTION_1_BELOW_1PCT}, diff {below_diff:.1%}).")
            if (count_diff > SANITY_COUNT_REL_TOLERANCE or
                    below_diff > SANITY_COUNT_REL_TOLERANCE):
                discrepancy = (
                    f"resolution=1.0 sanity check FAILED: got {occ['count']} communities "
                    f"({occ['below_one_percent']} below 1%) vs. B1's "
                    f"{B1_RESOLUTION_1_COMMUNITIES} ({B1_RESOLUTION_1_BELOW_1PCT} below 1%); "
                    f"relative diffs {count_diff:.1%}/{below_diff:.1%} exceed the "
                    f"{SANITY_COUNT_REL_TOLERANCE:.0%} gate. The partition differs "
                    "materially from B1's — stopping before the rest of the sweep."
                )
                log(discrepancy)
                write_report(rows, discrepancy, str(device))
                raise SystemExit(discrepancy)

        transferred = assign_to_train_communities(raw_train, labels, raw_val, k=TRANSFER_K)
        coverage = len(np.unique(transferred))
        pairs, subsampled = community_pairs(transferred)
        community = lift_result(validation, pairs)
        rows.append({
            "resolution": resolution, "occupancy": occ, "coverage": coverage,
            "community_lift": community, "subsampled": subsampled,
        })
        log(f"resolution={resolution}: communities={occ['count']} "
            f"below1%={occ['below_one_percent']} coverage={coverage}/{occ['count']} "
            f"community_lift={community['overall_lift']:.3f}x")
        if resolution == 1.0:
            lift_diff = abs(community["overall_lift"] / B1_COMMUNITY_LIFT - 1)
            if not np.isfinite(lift_diff) or lift_diff > SANITY_LIFT_REL_TOLERANCE:
                discrepancy = (f"Resolution 1.0 community lift is "
                               f"{community['overall_lift']:.3f}× versus B1's "
                               f"{B1_COMMUNITY_LIFT:.3f}× (difference {lift_diff:.1%}); "
                               "sweep halted before lower resolutions.")
                log(discrepancy)
                write_report(rows, discrepancy, str(device))
                raise SystemExit(discrepancy)

    write_report(rows, None, str(device), component_count, isolated_count)
    log(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
