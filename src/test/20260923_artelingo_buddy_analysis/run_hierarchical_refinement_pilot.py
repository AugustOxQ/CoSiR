"""Content-first, affect-within-content hierarchical Leiden pilot for ArtELingo.

Run this manually in the GPU environment.  It deliberately does not execute
any experiment at import time.
"""

import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score


OUT_DIR = os.path.dirname(__file__)
PIPELINE_PATH = os.path.join(OUT_DIR, "run_pipeline.py")
AFFECT_PILOT_PATH = os.path.join(OUT_DIR, "run_affect_pilot.py")
SINGLE_MODALITY_PATH = os.path.join(OUT_DIR, "run_single_modality_pilot.py")
REPORT_PATH = os.path.join(OUT_DIR, "hierarchical_refinement_pilot_report.md")

MIN_PARENT_SIZE = 50
STABILITY_SEED_B = 43
STABILITY_ARI_THRESHOLD = 0.5
SUCCESS_MARGIN_AMI = 0.02
GENRE_RETENTION_FLOOR = 0.3507
SEED = 42


REPO_ROOT = os.path.abspath(os.path.join(OUT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.conditional_buddy.buddy_graph import ensure_connected, ensure_min_degree, mutual_knn
from src.conditional_buddy.compute_buddies import build_buddy_graphs
from src.conditional_buddy.prototype_seed import detect_communities


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_sibling_module(module_name: str, path: str):
    """Import a sibling standalone script without executing its main block."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def composite_labels(parent_labels: np.ndarray, child_labels: np.ndarray) -> np.ndarray:
    """Factorize ``parent_child`` keys into contiguous integer community labels."""
    keys = np.asarray(
        [f"{parent_id}_{child_id}" for parent_id, child_id in zip(parent_labels, child_labels)],
        dtype=str,
    )
    _, labels = np.unique(keys, return_inverse=True)
    return labels.astype(np.int64, copy=False)


def build_affect_subgraph(
    parent_id: int,
    affect_rows: np.ndarray,
    pipeline,
    affect_pilot,
    device: str,
):
    """Build a repaired mutual-kNN graph only among one content parent's nodes."""
    n_parent = len(affect_rows)
    k_sub = min(pipeline.K, n_parent - 1)
    if k_sub < 1:
        raise ValueError(f"Parent {parent_id}: cannot build a graph for {n_parent} node(s).")

    # This is the same single-modality repair pattern as build_single_modality_graph:
    # the affect array fills both feature slots of the repair functions.
    affect_rows = affect_pilot.l2_normalize(
        np.asarray(affect_rows, dtype=np.float32)
    ).astype(np.float32, copy=False)
    adjacency = mutual_knn(affect_rows, K=k_sub, device=device, use_half=True)
    graph, degree_stats = ensure_min_degree(
        adjacency, affect_rows, affect_rows, device=device, use_half=True
    )
    graph, connectivity_stats = ensure_connected(
        graph,
        affect_rows,
        affect_rows,
        alpha=pipeline.ALPHA,
        device=device,
        use_half=True,
    )
    log(
        f"Parent {parent_id}: affect graph K_sub={k_sub}, nnz={adjacency.nnz:,}, "
        f"isolated fixed={degree_stats['num_isolated']}, "
        f"bridges added={connectivity_stats['bridges_added']}."
    )
    return graph


def write_report(
    parent_metrics: tuple[dict, dict],
    results: dict[str, tuple[dict, dict]],
    genre_count: int,
    parent_diagnostics: list[dict],
    total_nodes: int,
) -> None:
    """Write the control-centered hierarchical-refinement report."""
    eligible = [item for item in parent_diagnostics if item["eligible"]]
    stable = [item for item in eligible if item["stable"]]
    genuinely_split = [item for item in parent_diagnostics if item["genuinely_split"]]
    split_nodes = sum(item["size"] for item in genuinely_split)
    child_sizes = [size for item in genuinely_split for size in item["child_sizes"]]
    content_count_deltas = [
        abs(item["content_children"] - item["final_children"])
        for item in genuinely_split
    ]

    hierarchical_emotion_ami = results["hierarchical"][0]["AMI"]
    control_a_emotion_ami = results["control_a"][0]["AMI"]
    hierarchical_genre_ami = results["hierarchical"][1]["AMI"]
    real_signal = hierarchical_emotion_ami - control_a_emotion_ami >= SUCCESS_MARGIN_AMI
    genre_retained = hierarchical_genre_ami >= GENRE_RETENTION_FLOOR

    lines = [
        "# ArtELingo content-first hierarchical refinement pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "This pilot first fixes a CLIP content-parent Leiden partition, then permits a "
        "GoEmotions-only Leiden split *inside* each parent. It never merges nodes "
        "across content parents, so affect cannot create the cross-genre merges seen "
        "in flat fusion.\n\n",
        "**AMI caveat:** AMI is not monotone under refinement. Splitting a community can "
        "raise or lower chance-corrected AMI even when the split has no real affect "
        "information. Therefore the raw hierarchical values alone cannot answer the "
        "question; only its comparison with the exactly size-matched random-split "
        "Control A can.\n\n",
        f"**Setup:** {total_nodes:,} deduplicated paintings; content graph K={20}, "
        "alpha=0.5; affect subgraphs use K_sub=min(20, N_parent - 1); Leiden seeds "
        f"{SEED} and {STABILITY_SEED_B}; genre overlap n={genre_count}.\n\n",
        "## Content-only reproduction check\n\n",
        "The parent labels are the untouched content-only graph partition. Compare this "
        "run's computed values to the reference emotion AMI=0.0593 / V-measure=0.0600 "
        "and genre AMI=0.4384 / V-measure=0.4572 before interpreting refinements.\n\n",
        "## Split diagnostics\n\n",
        f"There were {len(parent_diagnostics):,} content parents. {len(eligible):,} were "
        f"eligible (N >= {MIN_PARENT_SIZE}); {len(stable):,} eligible parents met the "
        f"stability threshold (ARI >= {STABILITY_ARI_THRESHOLD:.1f}); and "
        f"{len(genuinely_split):,} were stable and genuinely split (>=2 children). "
        f"Those genuinely-split parents contain {split_nodes:,}/{total_nodes:,} nodes "
        f"({split_nodes / total_nodes:.1%}).\n\n",
    ]
    if child_sizes:
        lines.append(
            "Across all children in genuinely-split parents, child size was "
            f"min/median/max = {min(child_sizes)}/{np.median(child_sizes):.1f}/{max(child_sizes)}.\n\n"
        )
    else:
        lines.append("No parent produced a stable genuine split, so no child-size distribution exists.\n\n")

    lines.extend([
        "| signal | emotion AMI | emotion V-measure | genre AMI | genre V-measure |\n",
        "|---|---:|---:|---:|---:|\n",
        "| Content-only (reference) | 0.0593 | 0.0600 | 0.4384 | 0.4572 |\n",
        f"| Content-parent partition (this run's reproduction check) | {parent_metrics[0]['AMI']:.4f} | {parent_metrics[0]['V_measure']:.4f} | {parent_metrics[1]['AMI']:.4f} | {parent_metrics[1]['V_measure']:.4f} |\n",
        f"| Hierarchical (content-parent + affect-child) | {results['hierarchical'][0]['AMI']:.4f} | {results['hierarchical'][0]['V_measure']:.4f} | {results['hierarchical'][1]['AMI']:.4f} | {results['hierarchical'][1]['V_measure']:.4f} |\n",
        f"| Control A — random split (matched sizes) | {results['control_a'][0]['AMI']:.4f} | {results['control_a'][0]['V_measure']:.4f} | {results['control_a'][1]['AMI']:.4f} | {results['control_a'][1]['V_measure']:.4f} |\n",
        f"| Control B — content re-split (not size-matched) | {results['control_b'][0]['AMI']:.4f} | {results['control_b'][0]['V_measure']:.4f} | {results['control_b'][1]['AMI']:.4f} | {results['control_b'][1]['V_measure']:.4f} |\n",
        "\n## Decision-rule conclusion\n\n",
    ])
    if real_signal and genre_retained:
        outcome = "Real emotion signal above Control A, with genre retained."
    elif real_signal:
        outcome = "Real emotion signal above Control A, but genre retention was lost."
    elif genre_retained:
        outcome = "No real emotion signal above Control A, while genre was retained."
    else:
        outcome = "No real emotion signal above Control A, and genre retention was lost."
    lines.append(
        f"Decision rule: real, non-granularity-driven emotion signal requires hierarchical "
        f"emotion AMI - Control A emotion AMI >= {SUCCESS_MARGIN_AMI:.2f}. Here the "
        f"difference is {hierarchical_emotion_ami - control_a_emotion_ami:.4f}; {outcome} "
        f"Separately, hierarchical genre AMI={hierarchical_genre_ami:.4f} is "
        f"{'at or above' if genre_retained else 'below'} the {GENRE_RETENTION_FLOOR:.4f} "
        "80%-retention floor.\n\n"
    )
    lines.append("## Control B child-count caveat\n\n")
    if content_count_deltas:
        lines.append(
            "Control B is not size-matched. Across the genuinely-split parents, the "
            "mean absolute difference between its natural content-child count and the "
            f"real affect-child count was {np.mean(content_count_deltas):.2f} "
            f"(min/median/max {min(content_count_deltas)}/"
            f"{np.median(content_count_deltas):.1f}/{max(content_count_deltas)}). "
            "Per-parent counts were logged during the run.\n"
        )
    else:
        lines.append("No genuine affect splits occurred, so Control B had no eligible parents to re-split.\n")

    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    pipeline = load_sibling_module("artelingo_run_pipeline", PIPELINE_PATH)
    affect_pilot = load_sibling_module("artelingo_run_affect_pilot", AFFECT_PILOT_PATH)
    single_modality = load_sibling_module("artelingo_run_single_modality_pilot", SINGLE_MODALITY_PATH)

    pipeline.assert_extraction_complete()
    log("Loading and deduplicating cached CLIP features...")
    paintings, img_nodes, txt_nodes, emotion_counts = pipeline.load_dedup_features()
    majority_emotion = [pipeline.majority(counts) for counts in emotion_counts]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {device} for GoEmotions extraction and hierarchical graph construction.")

    log("Extracting and L2-normalizing GoEmotions affect features...")
    affect_nodes = affect_pilot.l2_normalize(
        affect_pilot.extract_affect_nodes(pipeline.TRAIN_JSON, paintings, device)
    ).astype(np.float32, copy=False)

    log("Building the content-only parent graph...")
    _, _, content_graph = build_buddy_graphs(
        img_nodes,
        txt_nodes,
        K=pipeline.K,
        alpha=pipeline.ALPHA,
        device=device,
        connect_components=True,
    )
    log("Running Leiden for the content-parent partition (seed=42)...")
    parent_labels = detect_communities(content_graph, seed=SEED)
    parent_metrics = single_modality.evaluate_graph(
        pipeline, parent_labels, paintings, majority_emotion
    )
    log(
        f"Content reproduction: emotion AMI={parent_metrics[0]['AMI']:.4f}, "
        f"genre AMI={parent_metrics[1]['AMI']:.4f}; "
        "references are 0.0593 and 0.4384 respectively."
    )

    n_nodes = len(paintings)
    hierarchical_children = np.zeros(n_nodes, dtype=np.int64)
    control_a_children = np.zeros(n_nodes, dtype=np.int64)
    control_b_children = np.zeros(n_nodes, dtype=np.int64)
    rng = np.random.default_rng(SEED)
    diagnostics = []

    unique_parents = np.unique(parent_labels)
    log(f"Processing {len(unique_parents):,} content parents for affect-only sub-clustering...")
    for number, parent_id in enumerate(unique_parents, start=1):
        indices = np.flatnonzero(parent_labels == parent_id)
        n_parent = len(indices)
        record = {
            "parent_id": int(parent_id),
            "size": n_parent,
            "eligible": n_parent >= MIN_PARENT_SIZE,
            "stable": False,
            "genuinely_split": False,
            "final_children": 1,
            "child_sizes": [n_parent],
            "content_children": 1,
        }
        if record["eligible"]:
            sub_graph = build_affect_subgraph(
                int(parent_id), affect_nodes[indices], pipeline, affect_pilot, device
            )
            affect_child_seed42 = detect_communities(sub_graph, seed=SEED)
            affect_child_seed43 = detect_communities(sub_graph, seed=STABILITY_SEED_B)
            stability_ari = adjusted_rand_score(affect_child_seed42, affect_child_seed43)
            record["stability_ari"] = float(stability_ari)
            if stability_ari >= STABILITY_ARI_THRESHOLD:
                n_children = len(np.unique(affect_child_seed42))
                record["stable"] = True
                record["final_children"] = n_children
                record["child_sizes"] = np.bincount(affect_child_seed42).tolist()
                if n_children >= 2:
                    record["genuinely_split"] = True
                    hierarchical_children[indices] = affect_child_seed42
                    # Permuting labels, rather than generating assignments, preserves the
                    # exact local multiset of child sizes for Control A.
                    control_a_children[indices] = rng.permutation(affect_child_seed42)

                    k_sub = min(pipeline.K, n_parent - 1)
                    _, _, content_subgraph = build_buddy_graphs(
                        img_nodes[indices],
                        txt_nodes[indices],
                        K=k_sub,
                        alpha=pipeline.ALPHA,
                        device=device,
                        connect_components=True,
                    )
                    content_children = detect_communities(content_subgraph, seed=SEED)
                    control_b_children[indices] = content_children
                    record["content_children"] = len(np.unique(content_children))
                    log(
                        f"Parent {parent_id}: stable affect split {n_children} child(ren), "
                        f"Control B content split {record['content_children']} child(ren)."
                    )
                else:
                    log(f"Parent {parent_id}: stable but unsplit (1 affect child).")
            else:
                log(
                    f"Parent {parent_id}: unstable affect split (ARI={stability_ari:.4f} < "
                    f"{STABILITY_ARI_THRESHOLD:.1f}); reverted to child 0."
                )
        diagnostics.append(record)
        if number % 20 == 0 or number == len(unique_parents):
            eligible_count = sum(item["eligible"] for item in diagnostics)
            stable_count = sum(item["stable"] for item in diagnostics)
            split_count = sum(item["genuinely_split"] for item in diagnostics)
            log(
                f"Parents processed {number:,}/{len(unique_parents):,}: eligible={eligible_count:,}, "
                f"stable={stable_count:,}, genuinely split={split_count:,}."
            )

    hierarchical_labels = composite_labels(parent_labels, hierarchical_children)
    control_a_labels = composite_labels(parent_labels, control_a_children)
    control_b_labels = composite_labels(parent_labels, control_b_children)
    results = {}
    for name, labels in (
        ("hierarchical", hierarchical_labels),
        ("control_a", control_a_labels),
        ("control_b", control_b_labels),
    ):
        results[name] = single_modality.evaluate_graph(
            pipeline, labels, paintings, majority_emotion
        )[:2]
        log(
            f"{name}: emotion AMI={results[name][0]['AMI']:.4f}, "
            f"genre AMI={results[name][1]['AMI']:.4f}."
        )

    genre_count = single_modality.evaluate_graph(
        pipeline, parent_labels, paintings, majority_emotion
    )[2]
    write_report(parent_metrics, results, genre_count, diagnostics, n_nodes)
    log(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
