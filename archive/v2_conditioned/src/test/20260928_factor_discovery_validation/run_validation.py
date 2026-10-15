"""One default-configuration ArtELingo factor-discovery validation run."""

import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.model.communities import community_stats, detect_communities
from src.model.graph import GraphConfig, build_content_graph
from src.train.stage1 import Stage1Config, train_stage1
from src.train.train_factors import FactorTrainingConfig, train_factors
from src.utils import FeatureManager


FEATURE_DIR = "/data/SSD2/pre_extract/artelingo/features"
ANNOTATIONS_PATH = Path("/data/PDD/artelingo/artelingo_train.json")
EXPECTED_SAMPLES = 308_723
NEAR_ZERO_MEAN = 1e-4
PRIVATE_RATIO = 0.05
SINGLE_COMMUNITY_MASS = 0.50


def reconstruction_error(model, img_features, txt_features, device):
    """Mean per-row relative L2 error for each modality, over all rows."""
    totals = np.zeros(2, dtype=np.float64)
    count = 0
    with torch.no_grad():
        for start in range(0, len(img_features), 8192):
            batches = (
                torch.as_tensor(img_features[start:start + 8192], dtype=torch.float32, device=device),
                torch.as_tensor(txt_features[start:start + 8192], dtype=torch.float32, device=device),
            )
            reconstructed = (
                model.reconstruct_image(model.encode_image(batches[0])),
                model.reconstruct_text(model.encode_text(batches[1])),
            )
            for modality in range(2):
                relative = torch.linalg.vector_norm(reconstructed[modality] - batches[modality], dim=1)
                relative /= torch.linalg.vector_norm(batches[modality], dim=1).clamp_min(1e-12)
                totals[modality] += relative.sum().item()
            count += len(batches[0])
    return (totals / count).tolist()


def cosine_agreement(img_codes, txt_codes, seed):
    """Compare matched factors with one fixed shuffled-text control."""
    shuffled = np.random.default_rng(seed).permutation(len(txt_codes))
    img_norm = np.linalg.norm(img_codes, axis=1)
    txt_norm = np.linalg.norm(txt_codes, axis=1)
    matched = np.sum(img_codes * txt_codes, axis=1) / np.maximum(img_norm * txt_norm, 1e-8)
    control = np.sum(img_codes * txt_codes[shuffled], axis=1) / np.maximum(
        img_norm * txt_norm[shuffled], 1e-8
    )
    return float(matched.mean()), float(control.mean())


def factor_diagnostics(img_codes, txt_codes, labels):
    """Per-factor modality means and community activation mass."""
    img_mean = img_codes.mean(axis=0, dtype=np.float64)
    txt_mean = txt_codes.mean(axis=0, dtype=np.float64)
    maximum = np.maximum(img_mean, txt_mean)
    minimum = np.minimum(img_mean, txt_mean)
    dead = maximum < NEAR_ZERO_MEAN
    private = ~dead & (minimum / np.maximum(maximum, 1e-12) <= PRIVATE_RATIO)

    combined = 0.5 * (img_codes + txt_codes)
    largest_share = np.zeros(combined.shape[1], dtype=np.float64)
    n_communities = int(labels.max()) + 1
    for factor in range(combined.shape[1]):
        mass = np.bincount(labels, weights=combined[:, factor], minlength=n_communities)
        if mass.sum() > 0:
            largest_share[factor] = mass.max() / mass.sum()
    topic_like = ~dead & (largest_share >= SINGLE_COMMUNITY_MASS)
    spanning = ~dead & ~topic_like
    return {
        "image_mean": img_mean.tolist(),
        "text_mean": txt_mean.tolist(),
        "modality_private_indices": np.flatnonzero(private).tolist(),
        "dead_indices": np.flatnonzero(dead).tolist(),
        "largest_community_mass_share": largest_share.tolist(),
        "topic_like_indices": np.flatnonzero(topic_like).tolist(),
        "spanning_indices": np.flatnonzero(spanning).tolist(),
    }


def main() -> None:
    started = perf_counter()
    manager = FeatureManager(storage_dir=FEATURE_DIR)
    with ANNOTATIONS_PATH.open() as file:
        annotations = json.load(file)
    if not (len(annotations) == manager.total_samples == EXPECTED_SAMPLES):
        raise ValueError("Annotation/feature count mismatch")
    features = manager.load_all_to_ram(["img_features", "txt_features"])
    sample_ids = np.asarray(manager.get_all_sample_ids(), dtype=np.int64)
    img_features = features["img_features"].numpy()
    txt_features = features["txt_features"].numpy()
    if not (len(sample_ids) == len(img_features) == len(txt_features) == EXPECTED_SAMPLES):
        raise ValueError("Feature rows and sample IDs do not match metadata")
    if (sample_ids.min() < 0 or sample_ids.max() >= len(annotations)
            or len(np.unique(sample_ids)) != len(sample_ids)):
        raise ValueError("Sample IDs must uniquely index the annotation list")
    emotions = np.asarray([annotations[int(index)]["emotion"] for index in sample_ids])
    print(f"Loaded {len(sample_ids):,} rows, {len(np.unique(emotions))} emotions; "
          f"seconds={perf_counter() - started:.3f}", flush=True)

    stage_started = perf_counter()
    graph = build_content_graph(img_features, txt_features, GraphConfig())
    graph_seconds = perf_counter() - stage_started
    print(f"Graph: {graph.nnz // 2:,} edges; seconds={graph_seconds:.3f}", flush=True)

    stage_started = perf_counter()
    _, embeddings = train_stage1(img_features, txt_features, graph, Stage1Config())
    stage1_seconds = perf_counter() - stage_started
    if not np.isfinite(embeddings).all():
        raise ValueError("Stage 1 produced non-finite embeddings")
    print(f"Stage 1: seconds={stage1_seconds:.3f}", flush=True)

    stage_started = perf_counter()
    labels = detect_communities(embeddings)
    community_seconds = perf_counter() - stage_started
    stats = community_stats(labels)
    print(f"Communities: {stats['num_communities']}; seconds={community_seconds:.3f}", flush=True)

    stage_started = perf_counter()
    config = FactorTrainingConfig()
    log = io.StringIO()
    with redirect_stdout(log):
        model, img_codes, txt_codes = train_factors(img_features, txt_features, graph, config)
    factor_seconds = perf_counter() - stage_started
    losses = np.asarray([float(value) for value in re.findall(
        r"factor epoch=\d+ loss=([\d.]+)", log.getvalue()
    )])
    if len(losses) != config.epochs or not np.isfinite(losses).all():
        raise ValueError("Factor training loss trace is incomplete or non-finite")
    if not (np.isfinite(img_codes).all() and np.isfinite(txt_codes).all()):
        raise ValueError("Factor training produced non-finite codes")
    print(f"Factors: epochs={config.epochs}; seconds={factor_seconds:.3f}; "
          f"first5={losses[:5].mean():.6f}; last5={losses[-5:].mean():.6f}", flush=True)

    selected_device = next(model.parameters()).device
    result = {
        "samples": len(img_features),
        "edges": graph.nnz // 2,
        "communities": stats["num_communities"],
        "config": vars(config),
        "thresholds": {
            "near_zero_mean": NEAR_ZERO_MEAN,
            "private_min_over_max": PRIVATE_RATIO,
            "single_community_mass": SINGLE_COMMUNITY_MASS,
        },
        "loss_first5_mean": float(losses[:5].mean()),
        "loss_last5_mean": float(losses[-5:].mean()),
        "loss_min": float(losses.min()),
        "loss_max": float(losses.max()),
        "relative_reconstruction_l2": reconstruction_error(
            model, img_features, txt_features, selected_device
        ),
        "paired_cosine_matched_and_shuffled": cosine_agreement(
            img_codes, txt_codes, config.seed
        ),
        "timings_seconds": {
            "graph": graph_seconds,
            "stage1": stage1_seconds,
            "communities": community_seconds,
            "factors": factor_seconds,
            "total": perf_counter() - started,
        },
    }
    result.update(factor_diagnostics(img_codes, txt_codes, labels))
    print("RESULT_JSON=" + json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
