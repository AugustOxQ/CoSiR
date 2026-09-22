"""Compare late edge-level fusion of ArtELingo content and affect buddy graphs.

Each modality first builds its own mutual-kNN graph in its native feature
space.  The fully repaired graphs are then combined at the edge level, rather
than concatenating features before neighbour search.  Run this manually in a
GPU-capable environment; it recomputes GoEmotions features.
"""

import importlib.util
import os
import sys
import time

import numpy as np
import torch
from scipy.sparse import csgraph


OUT_DIR = os.path.dirname(__file__)
PIPELINE_PATH = os.path.join(OUT_DIR, "run_pipeline.py")
AFFECT_PILOT_PATH = os.path.join(OUT_DIR, "run_affect_pilot.py")
SINGLE_MODALITY_PILOT_PATH = os.path.join(OUT_DIR, "run_single_modality_pilot.py")
REPORT_PATH = os.path.join(OUT_DIR, "late_fusion_pilot_report.md")
SEED = 42

CONTENT_EMOTION_AMI = 0.0593
CONTENT_GENRE_AMI = 0.4384
AFFECT_EMOTION_AMI = 0.1180
EARLY_FUSION_GENRE_AMI = 0.0867

# Match run_pipeline.py's repository-root import convention when this script is
# launched directly from outside the repository root.
REPO_ROOT = os.path.abspath(os.path.join(OUT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.conditional_buddy.buddy_graph import (  # noqa: E402
    ensure_connected,
    ensure_min_degree,
    union_graph,
)
from src.conditional_buddy.compute_buddies import build_buddy_graphs  # noqa: E402
from src.conditional_buddy.prototype_seed import detect_communities  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_sibling_module(module_name: str, path: str):
    """Import a sibling standalone script without executing its ``main`` block."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def repair_intersection(graph, repair_features: np.ndarray, device: str, pipeline):
    """Apply the standard minimum-degree and connectivity repairs to an edge intersection."""
    graph, degree_stats = ensure_min_degree(
        graph,
        repair_features,
        repair_features,
        device=device,
        use_half=True,
    )
    components_before = csgraph.connected_components(
        graph, directed=False, return_labels=False
    )
    log(
        "Late intersection: "
        f"isolated fixed={degree_stats['num_isolated']}; "
        f"connected components after minimum-degree repair={components_before}."
    )
    graph, connectivity_stats = ensure_connected(
        graph,
        repair_features,
        repair_features,
        alpha=pipeline.ALPHA,
        device=device,
        use_half=True,
    )
    components_after = csgraph.connected_components(
        graph, directed=False, return_labels=False
    )
    log(
        "Late intersection: "
        f"added {connectivity_stats['bridges_added']} bridge(s) over "
        f"{connectivity_stats['n_components']} components; "
        f"final connected components={components_after}."
    )
    if components_after != 1:
        raise RuntimeError(
            f"Late intersection repair failed; {components_after} components remain."
        )
    return graph


def write_report(
    content_nnz: int,
    affect_nnz: int,
    union_nnz: int,
    intersection_isolated_fraction: float,
    union_result: dict,
    intersection_result: dict,
    genre_count: int,
    k: int,
) -> None:
    """Write the requested fixed-reference comparison and data-dependent conclusions."""
    union_emotion = union_result["emotion"]
    union_genre = union_result["genre"]
    intersection_emotion = intersection_result["emotion"]
    intersection_genre = intersection_result["genre"]

    affect_pilot_bar = (
        union_emotion["AMI"] > 0.09
        and union_genre["AMI"] >= CONTENT_GENRE_AMI * 0.8
    )
    dec_pilot_bar = union_emotion["AMI"] > 0.177
    better_than_early_genre = union_genre["AMI"] > EARLY_FUSION_GENRE_AMI
    intersection_informative = (
        intersection_emotion["AMI"] > union_emotion["AMI"]
        and intersection_genre["AMI"] >= union_genre["AMI"]
    )

    lines = [
        "# ArtELingo late-fusion buddy-graph pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "**Late fusion in plain language:** the content view and the affect view each "
        "build a buddy graph independently in their own native feature space. Their "
        "graphs are then combined by their edges, rather than by combining feature "
        "vectors before neighbour search. This differs from the earlier **early "
        "fusion** pilot, which concatenated GoEmotions and CLIP-text features into one "
        "vector and built one mutual-kNN graph in that shared space.\n\n",
        f"**Setup:** K={k}, alpha=0.5, Leiden seed={SEED}; genre metrics use the "
        f"{genre_count}-painting genre-labelled overlap. The content graph has "
        f"`E_content.nnz={content_nnz:,}`, the GoEmotions affect graph has "
        f"`E_affect.nnz={affect_nnz:,}`, and their late union has "
        f"`E_late_union.nnz={union_nnz:,}`. The raw late intersection left "
        f"{intersection_isolated_fraction:.2%} of nodes isolated before any repair.\n\n",
        "| signal | emotion AMI | emotion V-measure | genre AMI | genre V-measure |\n",
        "|---|---:|---:|---:|---:|\n",
        "| Content-only (CLIP img+txt union, reference) | 0.0593 | 0.0600 | 0.4384 | 0.4572 |\n",
        "| GoEmotions-affect-only (reference, single-modality ceiling) | 0.1180 | 0.1189 | 0.0396 | 0.0937 |\n",
        "| Early fusion, best point (weight=4.0, reference) | 0.1160 | 0.1166 | 0.0867 | 0.1237 |\n",
        f"| Late fusion — union (this run) | {union_emotion['AMI']:.4f} | "
        f"{union_emotion['V_measure']:.4f} | {union_genre['AMI']:.4f} | "
        f"{union_genre['V_measure']:.4f} |\n",
        f"| Late fusion — intersection (this run) | {intersection_emotion['AMI']:.4f} | "
        f"{intersection_emotion['V_measure']:.4f} | {intersection_genre['AMI']:.4f} | "
        f"{intersection_genre['V_measure']:.4f} |\n",
        "\n## Conclusion\n\n",
        "**(a) Original affect-pilot bar:** this requires emotion AMI > 0.09 (at least "
        "a 50% relative improvement over the 0.0593 content-only baseline) while "
        "retaining at least 80% of the 0.4384 genre baseline (genre AMI >= 0.3507). "
        f"Late union {'clears' if affect_pilot_bar else 'does not clear'} this bar: "
        f"emotion AMI={union_emotion['AMI']:.4f}, genre AMI={union_genre['AMI']:.4f}.\n\n",
        "**(b) Stricter DEC-pilot bar:** this requires emotion AMI > 0.177, a 50% "
        "relative improvement over GoEmotions' 0.1180 single-modality ceiling. "
        f"Late union {'clears' if dec_pilot_bar else 'does not clear'} this bar: "
        f"emotion AMI={union_emotion['AMI']:.4f}.\n\n",
        f"Late fusion {'preserved genre structure better' if better_than_early_genre else 'did not preserve genre structure better'} "
        f"than early fusion: late-union genre AMI={union_genre['AMI']:.4f} versus "
        f"the early-fusion value of {EARLY_FUSION_GENRE_AMI:.4f}.\n\n",
        "## Intersection diagnostic\n\n",
        f"The raw edge intersection isolated {intersection_isolated_fraction:.2%} of nodes "
        "before repair. After minimum-degree and connectivity repair, its emotion/genre "
        f"AMI values were {intersection_emotion['AMI']:.4f}/{intersection_genre['AMI']:.4f}, "
        f"versus {union_emotion['AMI']:.4f}/{union_genre['AMI']:.4f} for union. "
        f"This {'suggests the intersection retains an informative shared signal' if intersection_informative else 'suggests the intersection is degenerate or noise-dominated after heavy repair'}, "
        "with the raw isolated-node fraction providing the necessary context for that interpretation.\n",
    ]
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    pipeline = load_sibling_module("artelingo_run_pipeline", PIPELINE_PATH)
    affect_pilot = load_sibling_module("artelingo_run_affect_pilot", AFFECT_PILOT_PATH)
    single_modality_pilot = load_sibling_module(
        "artelingo_run_single_modality_pilot", SINGLE_MODALITY_PILOT_PATH
    )

    pipeline.assert_extraction_complete()
    log("Loading and deduplicating cached CLIP features...")
    paintings, img_nodes, txt_nodes, emotion_counts = pipeline.load_dedup_features()
    majority_emotion = [pipeline.majority(counts) for counts in emotion_counts]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {device} for affect extraction and late graph construction.")

    log("Building the repaired CLIP content graph...")
    A_img, A_txt, E_content = build_buddy_graphs(
        img_nodes,
        txt_nodes,
        K=pipeline.K,
        alpha=pipeline.ALPHA,
        device=device,
        connect_components=True,
    )
    log(f"Content graph: E_content.nnz={E_content.nnz:,}.")

    log("Extracting GoEmotions affect features...")
    affect_nodes = affect_pilot.extract_affect_nodes(
        pipeline.TRAIN_JSON, paintings, device
    )
    E_affect = single_modality_pilot.build_single_modality_graph(
        "GoEmotions-affect-only",
        affect_nodes,
        pipeline,
        affect_pilot,
        device,
        expected_nodes=len(paintings),
    )
    log(f"Affect graph: E_affect.nnz={E_affect.nnz:,}.")

    E_late_union = union_graph(E_content, E_affect)
    log(
        f"Late union: E_late_union.nnz={E_late_union.nnz:,} "
        f"(content={E_content.nnz:,}, affect={E_affect.nnz:,})."
    )

    E_late_intersection = E_content.multiply(E_affect).tocsr()
    E_late_intersection.data[:] = 1.0
    raw_isolated_fraction = float(
        np.mean(np.diff(E_late_intersection.indptr) == 0)
    )
    log(
        f"Raw late intersection: nnz={E_late_intersection.nnz:,}; "
        f"isolated-node fraction={raw_isolated_fraction:.2%} before repair."
    )

    repair_features = affect_pilot.l2_normalize(img_nodes).astype(np.float32, copy=False)
    E_late_intersection = repair_intersection(
        E_late_intersection, repair_features, device, pipeline
    )

    log(f"Late union: running Leiden community detection (seed={SEED})...")
    union_community = detect_communities(E_late_union, seed=SEED)
    log(f"Late intersection: running Leiden community detection (seed={SEED})...")
    intersection_community = detect_communities(E_late_intersection, seed=SEED)

    union_emotion, union_genre, genre_count = single_modality_pilot.evaluate_graph(
        pipeline, union_community, paintings, majority_emotion
    )
    intersection_emotion, intersection_genre, _ = single_modality_pilot.evaluate_graph(
        pipeline, intersection_community, paintings, majority_emotion
    )
    union_result = {"emotion": union_emotion, "genre": union_genre}
    intersection_result = {"emotion": intersection_emotion, "genre": intersection_genre}
    log(
        f"Late union: emotion AMI={union_emotion['AMI']:.4f}, "
        f"genre AMI={union_genre['AMI']:.4f}."
    )
    log(
        f"Late intersection: emotion AMI={intersection_emotion['AMI']:.4f}, "
        f"genre AMI={intersection_genre['AMI']:.4f}."
    )

    write_report(
        E_content.nnz,
        E_affect.nnz,
        E_late_union.nnz,
        raw_isolated_fraction,
        union_result,
        intersection_result,
        genre_count,
        pipeline.K,
    )
    log(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
