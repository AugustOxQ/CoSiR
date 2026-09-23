"""Run the ArtELingo linear CCA and conditional-residual go/no-go audit.

This diagnostic is deliberately not a fusion method.  It measures whether
CLIP content and GoEmotions affect features share a stable linear subspace on
unseen paintings, then tests whether affect left unexplained by content still
forms emotion-informative communities.  Run it manually in a GPU-capable
environment; it writes its report beside this script.
"""

import importlib.util
import os
import sys
import time

import numpy as np
import torch
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


OUT_DIR = os.path.dirname(__file__)
REPORT_OUT_DIR = os.environ.get("PERCEPT_OUTPUT_ROOT") or OUT_DIR
PIPELINE_PATH = os.path.join(OUT_DIR, "run_pipeline.py")
AFFECT_PILOT_PATH = os.path.join(OUT_DIR, "run_affect_pilot.py")
SINGLE_MODALITY_PATH = os.path.join(OUT_DIR, "run_single_modality_pilot.py")
REPORT_PATH = os.path.join(REPORT_OUT_DIR, "cca_audit_pilot_report.md")

PERCEPT_FEATURE_ROOT = os.environ.get("PERCEPT_FEATURE_ROOT", "/data/SSD2/pre_extract")
PERCEPT_RAW_JSON_ROOT = os.environ.get("PERCEPT_RAW_JSON_ROOT", "/data/PDD/artelingo")
HELDOUT_STORAGE_DIR = f"{PERCEPT_FEATURE_ROOT}/artelingo_heldout/features"
HELDOUT_JSON = f"{PERCEPT_RAW_JSON_ROOT}/artelingo_val_test.json"
HELDOUT_PAINTINGS = 9_365
CONTENT_PCA_DIM = 50
CCA_N_COMPONENTS = 10
N_PERMUTATIONS = 20
REAL_SIGNAL_CORR_THRESHOLD = 0.15
RESIDUAL_EMOTION_AMI_THRESHOLD = 0.0944
RAW_AFFECT_EMOTION_AMI = 0.1180
RAW_AFFECT_GENRE_AMI = 0.0396
EDGE_SAMPLE_SIZE = 2_000
SEED = 42

REPO_ROOT = os.path.abspath(os.path.join(OUT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.conditional_buddy.buddy_graph import mutual_knn
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


def content_features(img_nodes: np.ndarray, txt_nodes: np.ndarray, affect_pilot) -> np.ndarray:
    """Concatenate independently L2-normalized CLIP image and text features."""
    return np.concatenate(
        (affect_pilot.l2_normalize(img_nodes), affect_pilot.l2_normalize(txt_nodes)), axis=1
    ).astype(np.float64, copy=False)


def component_correlations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return one held-out Pearson correlation for every canonical pair."""
    return np.asarray(
        [np.corrcoef(left[:, component], right[:, component])[0, 1]
         for component in range(left.shape[1])],
        dtype=np.float64,
    )


def graph_overlap_fraction(reference_graph, comparison_graph, sampled_nodes: np.ndarray) -> float:
    """Mean per-node recall of reference neighbors in the comparison graph."""
    overlaps = []
    for node in sampled_nodes:
        reference_neighbors = reference_graph[int(node)].indices
        if len(reference_neighbors) == 0:
            continue
        comparison_neighbors = comparison_graph[int(node)].indices
        overlaps.append(np.isin(reference_neighbors, comparison_neighbors).mean())
    if not overlaps:
        raise RuntimeError("No sampled nodes had graph neighbors for edge-retrieval scoring.")
    return float(np.mean(overlaps))


def random_neighbor_overlap_fraction(
    reference_graph,
    comparison_graph,
    sampled_nodes: np.ndarray,
    n_nodes: int,
    rng: np.random.Generator,
) -> float:
    """Mean overlap from random neighbor sets matching each reference degree."""
    overlaps = []
    all_nodes = np.arange(n_nodes)
    for node in sampled_nodes:
        node = int(node)
        degree = len(reference_graph[node].indices)
        if degree == 0:
            continue
        candidates = np.delete(all_nodes, node)
        random_neighbors = rng.choice(candidates, size=degree, replace=False)
        comparison_neighbors = comparison_graph[node].indices
        overlaps.append(np.isin(random_neighbors, comparison_neighbors).mean())
    if not overlaps:
        raise RuntimeError("No sampled nodes had graph neighbors for random edge scoring.")
    return float(np.mean(overlaps))


def evaluate_residual_graph(
    pipeline,
    single_modality,
    affect_pilot,
    residual_affect_train: np.ndarray,
    paintings: list[str],
    majority_emotion: list[str],
    device: str,
) -> tuple[dict, dict, int]:
    """Cluster residual affect and score it against emotion and genre labels."""
    graph = single_modality.build_single_modality_graph(
        "residual-GoEmotions-affect-only",
        residual_affect_train,
        pipeline,
        affect_pilot,
        device,
        expected_nodes=len(paintings),
    )
    log(f"Residual affect: running Leiden community detection (seed={SEED})...")
    communities = detect_communities(graph, seed=SEED)
    emotion_metrics = pipeline.external_metrics(communities, majority_emotion)
    genre_map = pipeline.load_genre_map()
    genre_indices = [index for index, painting in enumerate(paintings) if painting in genre_map]
    if not genre_indices:
        raise RuntimeError("No genre-labelled paintings overlap the train node set.")
    genre_metrics = pipeline.external_metrics(
        [communities[index] for index in genre_indices],
        [genre_map[paintings[index]] for index in genre_indices],
    )
    return emotion_metrics, genre_metrics, len(genre_indices)


def write_report(
    real_correlations: np.ndarray,
    null_correlations: np.ndarray,
    content_recall: float,
    affect_recall: float,
    random_content_recall: float,
    random_affect_recall: float,
    heldout_r2: float,
    residual_emotion_metrics: dict,
    residual_genre_metrics: dict,
    genre_count: int,
) -> None:
    """Write the predeclared audit decisions and their supporting diagnostics."""
    null_means = null_correlations.mean(axis=0)
    null_95ths = np.percentile(null_correlations, 95, axis=0)
    threshold_passes = real_correlations > REAL_SIGNAL_CORR_THRESHOLD
    null_passes = real_correlations > null_95ths
    stable_shared_signal = bool(threshold_passes[0] and null_passes[0])
    informative_residual = residual_emotion_metrics["AMI"] >= RESIDUAL_EMOTION_AMI_THRESHOLD

    if stable_shared_signal:
        synthesis = (
            "Shared signal found: a small learned-student pilot is licensed, consistent "
            "with the second brainstorm's ranking."
        )
    elif informative_residual:
        synthesis = (
            "No shared signal but an informative residual: affect is real but orthogonal "
            "to content, supporting conditional/two-output representations rather than "
            "a single joint graph."
        )
    elif not stable_shared_signal and not informative_residual:
        synthesis = (
            "No shared signal and no informative residual: this supports abandoning "
            "joint-fusion methods entirely."
        )
    else:
        synthesis = "Mixed/inconclusive: the two diagnostics do not support a clean narrative."

    lines = [
        "# ArtELingo linear CCA + conditional-residual audit\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## What this audit tests\n\n",
        "CCA asks whether matched CLIP content and GoEmotions affect features contain a "
        "linear subspace that stays correlated on paintings the fit never saw. The residual "
        "test removes the part of affect linearly predicted by content, then asks whether "
        "the leftover affect still carries emotion structure. Together these are a go/no-go "
        "gate for the learned two-teacher student, SNF, and co-regularized spectral "
        "clustering ideas: they distinguish no useful signal, a shared signal, and affect "
        "that is useful but orthogonal to content.\n\n",
        "## Held-out CCA\n\n",
        f"CCA fit only on train paintings after a train-only {CONTENT_PCA_DIM}-component "
        "content PCA. Each null fit scrambles train content/affect correspondence, but "
        "always evaluates on the same correctly paired held-out paintings.\n\n",
        "| component | held-out correlation | null mean | null 95th percentile | > 0.15 | > null 95th |\n",
        "|---:|---:|---:|---:|---|---|\n",
    ]
    for component in range(CCA_N_COMPONENTS):
        lines.append(
            f"| {component + 1} | {real_correlations[component]:.4f} | "
            f"{null_means[component]:.4f} | {null_95ths[component]:.4f} | "
            f"{'pass' if threshold_passes[component] else 'fail'} | "
            f"{'pass' if null_passes[component] else 'fail'} |\n"
        )
    lines.extend([
        "\n**Stable shared signal found: " + ("yes" if stable_shared_signal else "no") + ".** "
        f"The predeclared rule is that the first component must exceed both {REAL_SIGNAL_CORR_THRESHOLD:.2f} "
        "and its own permutation-null 95th percentile.\n\n",
        "## Held-out edge retrieval\n\n",
        "For the same 2,000 held-out paintings, this measures how much each true content "
        "or affect graph's neighborhood is retained in the joint CCA-space mutual-kNN graph. "
        "The chance floor replaces each true neighbor set with a random set of equal size.\n\n",
        "| reference graph | joint-CCA neighbor recall | random-neighbor chance floor |\n",
        "|---|---:|---:|\n",
        f"| content | {content_recall:.4f} | {random_content_recall:.4f} |\n",
        f"| affect | {affect_recall:.4f} | {random_affect_recall:.4f} |\n",
        "\n## Conditional residual\n\n",
        f"Held-out R² for linear content → affect prediction: **{heldout_r2:.4f}**. "
        "This is the fraction of held-out affect variance explained by content.\n\n",
        "| signal | emotion AMI | emotion V-measure | genre AMI | genre V-measure |\n",
        "|---|---:|---:|---:|---:|\n",
        f"| Raw GoEmotions-affect-only reference | {RAW_AFFECT_EMOTION_AMI:.4f} | — | "
        f"{RAW_AFFECT_GENRE_AMI:.4f} | — |\n",
        f"| Residual-affect-only (genre n={genre_count:,}) | {residual_emotion_metrics['AMI']:.4f} | "
        f"{residual_emotion_metrics['V_measure']:.4f} | {residual_genre_metrics['AMI']:.4f} | "
        f"{residual_genre_metrics['V_measure']:.4f} |\n",
        "\n**Informative residual: " + ("yes" if informative_residual else "no") + ".** "
        f"The predeclared rule is residual emotion AMI ≥ {RESIDUAL_EMOTION_AMI_THRESHOLD:.4f} "
        f"(80% of the raw GoEmotions reference, {RAW_AFFECT_EMOTION_AMI:.4f}).\n\n",
        "## Final synthesis\n\n",
        synthesis + "\n",
    ])
    os.makedirs(REPORT_OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    pipeline = load_sibling_module("artelingo_run_pipeline_train", PIPELINE_PATH)
    affect_pilot = load_sibling_module("artelingo_run_affect_pilot_cca", AFFECT_PILOT_PATH)
    single_modality = load_sibling_module(
        "artelingo_run_single_modality_pilot_cca", SINGLE_MODALITY_PATH
    )
    heldout_pipeline = load_sibling_module("artelingo_run_pipeline_cca_heldout", PIPELINE_PATH)
    heldout_pipeline.STORAGE_DIR = HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = HELDOUT_JSON

    log("Verifying train feature extraction is complete...")
    pipeline.assert_extraction_complete()
    log("Loading and deduplicating train CLIP features...")
    paintings, img_nodes, txt_nodes, emotion_counts = pipeline.load_dedup_features()
    majority_emotion = [pipeline.majority(counts) for counts in emotion_counts]

    log("Verifying held-out feature extraction is complete...")
    heldout_pipeline.assert_extraction_complete()
    log("Loading and deduplicating held-out CLIP features...")
    heldout_paintings, heldout_img_nodes, heldout_txt_nodes, heldout_emotion_counts = (
        heldout_pipeline.load_dedup_features()
    )
    if len(heldout_paintings) != HELDOUT_PAINTINGS:
        raise RuntimeError(
            f"Expected {HELDOUT_PAINTINGS:,} held-out paintings, got {len(heldout_paintings):,}."
        )
    heldout_majority_emotion = [heldout_pipeline.majority(counts) for counts in heldout_emotion_counts]
    del heldout_majority_emotion  # Labels are loaded for split parity; CCA itself is label-free.

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {device} for GoEmotions extraction and graph construction.")
    log("Extracting train GoEmotions affect nodes...")
    affect_train = affect_pilot.extract_affect_nodes(pipeline.TRAIN_JSON, paintings, device)
    log("Extracting held-out GoEmotions affect nodes...")
    affect_heldout = affect_pilot.extract_affect_nodes(HELDOUT_JSON, heldout_paintings, device)

    content_train = content_features(img_nodes, txt_nodes, affect_pilot)
    content_heldout = content_features(heldout_img_nodes, heldout_txt_nodes, affect_pilot)
    affect_train = np.asarray(affect_train, dtype=np.float64)
    affect_heldout = np.asarray(affect_heldout, dtype=np.float64)

    log(f"Fitting train-only content PCA ({CONTENT_PCA_DIM} components)...")
    pca = PCA(n_components=CONTENT_PCA_DIM, random_state=SEED)
    content_train_pca = pca.fit_transform(content_train)
    content_heldout_pca = pca.transform(content_heldout)

    log(f"Fitting train-only linear CCA ({CCA_N_COMPONENTS} components)...")
    cca = CCA(n_components=CCA_N_COMPONENTS)
    cca.fit(content_train_pca, affect_train)
    content_heldout_cca, affect_heldout_cca = cca.transform(content_heldout_pca, affect_heldout)
    real_correlations = component_correlations(content_heldout_cca, affect_heldout_cca)

    log(f"Building {N_PERMUTATIONS} train-permutation CCA null fits...")
    null_correlations = np.zeros((N_PERMUTATIONS, CCA_N_COMPONENTS), dtype=np.float64)
    for permutation_seed in range(N_PERMUTATIONS):
        permutation = np.random.default_rng(permutation_seed).permutation(len(affect_train))
        null_cca = CCA(n_components=CCA_N_COMPONENTS)
        null_cca.fit(content_train_pca, affect_train[permutation])
        null_content, null_affect = null_cca.transform(content_heldout_pca, affect_heldout)
        null_correlations[permutation_seed] = component_correlations(null_content, null_affect)
        log(f"CCA permutation {permutation_seed + 1}/{N_PERMUTATIONS} complete.")

    log("Building held-out content and affect reference graphs...")
    content_graph = single_modality.build_single_modality_graph(
        "held-out-content-reference",
        content_heldout,
        heldout_pipeline,
        affect_pilot,
        device,
        expected_nodes=len(heldout_paintings),
    )
    affect_graph = single_modality.build_single_modality_graph(
        "held-out-affect-reference",
        affect_heldout,
        heldout_pipeline,
        affect_pilot,
        device,
        expected_nodes=len(heldout_paintings),
    )
    joint_cca_nodes = np.concatenate((content_heldout_cca, affect_heldout_cca), axis=1)
    joint_cca_nodes = affect_pilot.l2_normalize(joint_cca_nodes).astype(np.float32, copy=False)
    log(f"Building held-out joint-CCA mutual-kNN graph (K={heldout_pipeline.K}, device={device})...")
    joint_cca_graph = mutual_knn(joint_cca_nodes, K=heldout_pipeline.K, device=device, use_half=True)

    sampled_nodes = np.random.default_rng(SEED).choice(
        len(heldout_paintings), size=EDGE_SAMPLE_SIZE, replace=False
    )
    content_recall = graph_overlap_fraction(content_graph, joint_cca_graph, sampled_nodes)
    affect_recall = graph_overlap_fraction(affect_graph, joint_cca_graph, sampled_nodes)
    random_rng = np.random.default_rng(SEED)
    random_content_recall = random_neighbor_overlap_fraction(
        content_graph, joint_cca_graph, sampled_nodes, len(heldout_paintings), random_rng
    )
    random_affect_recall = random_neighbor_overlap_fraction(
        affect_graph, joint_cca_graph, sampled_nodes, len(heldout_paintings), random_rng
    )

    log("Fitting train-only linear content-to-affect regression...")
    regression = LinearRegression()
    regression.fit(content_train_pca, affect_train)
    heldout_r2 = float(regression.score(content_heldout_pca, affect_heldout))
    residual_affect_train = affect_train - regression.predict(content_train_pca)
    residual_affect_heldout = affect_heldout - regression.predict(content_heldout_pca)
    del residual_affect_heldout  # Computed explicitly for the conditional-residual audit.

    log("Building and evaluating train residual-affect graph...")
    residual_emotion_metrics, residual_genre_metrics, genre_count = evaluate_residual_graph(
        pipeline,
        single_modality,
        affect_pilot,
        residual_affect_train,
        paintings,
        majority_emotion,
        device,
    )
    write_report(
        real_correlations,
        null_correlations,
        content_recall,
        affect_recall,
        random_content_recall,
        random_affect_recall,
        heldout_r2,
        residual_emotion_metrics,
        residual_genre_metrics,
        genre_count,
    )
    log(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
