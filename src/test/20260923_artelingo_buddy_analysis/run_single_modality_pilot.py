"""Measure ArtELingo buddy-community label ceilings for individual modalities.

This is deliberately a single-modality experiment: each graph starts from one
mutual-kNN adjacency, rather than the image/text union used by run_pipeline.py.
Run manually in a GPU-capable environment; it recomputes GoEmotions features.
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
REPORT_PATH = os.path.join(OUT_DIR, "single_modality_pilot_report.md")
SEED = 42

# Make this standalone script importable from any working directory, matching
# run_pipeline.py's repository-root import convention.
REPO_ROOT = os.path.abspath(os.path.join(OUT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.conditional_buddy.buddy_graph import (
    ensure_connected,
    ensure_min_degree,
    mutual_knn,
)
from src.conditional_buddy.prototype_seed import detect_communities


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


def build_single_modality_graph(
    name: str,
    nodes: np.ndarray,
    pipeline,
    affect_pilot,
    device: str,
    expected_nodes: int,
):
    """Build a fully repaired graph from exactly one node-feature matrix.

    The repair helpers have image/text arguments because the usual graph is a
    union. Passing this one normalized matrix for both arguments retains their
    exact degree/connectivity guarantees without introducing another modality.
    """
    nodes = np.asarray(nodes, dtype=np.float32)
    if nodes.ndim != 2 or nodes.shape[0] != expected_nodes:
        raise ValueError(
            f"{name}: expected ({expected_nodes}, D) node features, got {nodes.shape}."
        )
    normalized_nodes = affect_pilot.l2_normalize(nodes).astype(np.float32, copy=False)

    log(f"{name}: building mutual-kNN graph (K={pipeline.K}, device={device})...")
    adjacency = mutual_knn(
        normalized_nodes,
        K=pipeline.K,
        device=device,
        use_half=True,
    )
    graph, degree_stats = ensure_min_degree(
        adjacency,
        normalized_nodes,
        normalized_nodes,
        device=device,
        use_half=True,
    )
    components_before = csgraph.connected_components(
        graph, directed=False, return_labels=False
    )
    log(
        f"{name}: mutual graph nnz={adjacency.nnz:,}; "
        f"isolated fixed={degree_stats['num_isolated']}; "
        f"connected components={components_before}."
    )
    graph, connectivity_stats = ensure_connected(
        graph,
        normalized_nodes,
        normalized_nodes,
        alpha=pipeline.ALPHA,
        device=device,
        use_half=True,
    )
    components_after = csgraph.connected_components(
        graph, directed=False, return_labels=False
    )
    log(
        f"{name}: added {connectivity_stats['bridges_added']} bridge(s) over "
        f"{connectivity_stats['n_components']} components; "
        f"final connected components={components_after}."
    )
    if components_after != 1:
        raise RuntimeError(f"{name}: graph repair failed; {components_after} components remain.")
    return graph


def evaluate_graph(pipeline, community: np.ndarray, paintings: list[str], majority_emotion: list[str]):
    """Evaluate full-graph emotion and genre-overlap labels using pipeline helpers."""
    emotion_metrics = pipeline.external_metrics(community, majority_emotion)
    genre_map = pipeline.load_genre_map()
    genre_indices = [i for i, painting in enumerate(paintings) if painting in genre_map]
    if not genre_indices:
        raise RuntimeError("No genre-labelled paintings overlap the deduplicated node set.")
    genre_metrics = pipeline.external_metrics(
        [community[i] for i in genre_indices],
        [genre_map[paintings[i]] for i in genre_indices],
    )
    return emotion_metrics, genre_metrics, len(genre_indices)


def write_report(results: list[dict], genre_count: int, k: int) -> None:
    """Write the four-way comparison and a data-dependent ceiling interpretation."""
    affect_result = next(result for result in results if result["name"] == "GoEmotions-affect-only")
    affect_emotion_ami = affect_result["emotion"]["AMI"]
    affect_genre_ami = affect_result["genre"]["AMI"]
    genre_near_zero = abs(affect_genre_ami) < 0.02
    genre_statement = "is near zero" if genre_near_zero else "is not near zero"

    lines = [
        "# ArtELingo single-modality buddy-graph pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        f"**Setup:** Three independent mutual-kNN graphs (K={k}) are built from CLIP "
        "image features, CLIP text features, and mean-pooled GoEmotions sigmoid "
        "probabilities respectively. Each graph receives the same minimum-degree and "
        "connectivity repairs as the normal buddy graph, then Leiden uses seed=42. "
        f"Genre metrics use the {genre_count}-painting genre-labelled overlap.\n\n",
        "| graph | emotion_AMI | emotion_Vmeasure | genre_AMI | genre_Vmeasure |\n",
        "|---|---:|---:|---:|---:|\n",
        "| Existing CLIP img+txt UNION baseline (reference) | 0.0593 | — | 0.4384 | — |\n",
    ]
    for result in results:
        lines.append(
            f"| {result['name']} | {result['emotion']['AMI']:.4f} | "
            f"{result['emotion']['V_measure']:.4f} | {result['genre']['AMI']:.4f} | "
            f"{result['genre']['V_measure']:.4f} |\n"
        )

    lines.extend([
        "\n## Interpretation\n\n",
        f"The GoEmotions-only graph's emotion AMI ceiling is {affect_emotion_ami:.4f}. "
        "Compare this directly with the affect-fusion sweep to determine whether the "
        "shared graph diluted a substantially stronger affect structure or was already "
        f"approaching the single-signal ceiling. Its genre AMI is {affect_genre_ami:.4f}, "
        f"which {genre_statement}; this indicates whether affect and genre specialize "
        "in different structure rather than the affect signal being merely noise.\n",
    ])
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    pipeline = load_sibling_module("artelingo_run_pipeline", PIPELINE_PATH)
    affect_pilot = load_sibling_module("artelingo_run_affect_pilot", AFFECT_PILOT_PATH)

    pipeline.assert_extraction_complete()
    log("Loading and deduplicating cached CLIP features...")
    paintings, img_nodes, txt_nodes, emotion_counts = pipeline.load_dedup_features()
    majority_emotion = [pipeline.majority(counts) for counts in emotion_counts]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {device} for affect extraction and single-modality graph construction.")

    log("Extracting GoEmotions affect features...")
    affect_nodes = affect_pilot.extract_affect_nodes(pipeline.TRAIN_JSON, paintings, device)

    modalities = (
        ("CLIP-image-only", img_nodes),
        ("CLIP-text-only", txt_nodes),
        ("GoEmotions-affect-only", affect_nodes),
    )
    results = []
    genre_count = 0
    for name, nodes in modalities:
        graph = build_single_modality_graph(
            name, nodes, pipeline, affect_pilot, device, expected_nodes=len(paintings)
        )
        log(f"{name}: running Leiden community detection (seed={SEED})...")
        community = detect_communities(graph, seed=SEED)
        emotion_metrics, genre_metrics, genre_count = evaluate_graph(
            pipeline, community, paintings, majority_emotion
        )
        results.append({"name": name, "emotion": emotion_metrics, "genre": genre_metrics})
        log(
            f"{name}: emotion AMI={emotion_metrics['AMI']:.4f}, "
            f"genre AMI={genre_metrics['AMI']:.4f}."
        )

    write_report(results, genre_count, pipeline.K)
    log(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
