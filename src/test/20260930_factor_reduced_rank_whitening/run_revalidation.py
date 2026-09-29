"""Revalidate ArtELingo CLIP factors at three reduced PCA whitening ranks."""

import io
import json
import re
import sys
import gc
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.model.communities import community_stats, detect_communities
from src.model.graph import GraphConfig, build_content_graph
from src.model.whitening import pca_whiten, select_pca_rank
from src.train.stage1 import Stage1Config, train_stage1
from src.train.train_factors import FactorTrainingConfig, train_factors
from src.utils import FeatureManager


FEATURE_DIR = "/data/SSD2/pre_extract/artelingo/features"
ANNOTATIONS_PATH = Path("/data/PDD/artelingo/artelingo_train.json")
EXPECTED_SAMPLES = 308_723
NEAR_ZERO_MEAN = 1e-4
PRIVATE_RATIO = 0.05
SINGLE_COMMUNITY_MASS = 0.50
VARIANCE_THRESHOLDS = (0.99, 0.95, 0.999)


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
    full_img, image_pca = pca_whiten(img_features)
    full_txt, text_pca = pca_whiten(txt_features)
    whitening_seconds = perf_counter() - stage_started
    if not (np.isfinite(full_img).all() and np.isfinite(full_txt).all()):
        raise ValueError("PCA whitening produced non-finite features")
    if image_pca.n_components_ != img_features.shape[1] or text_pca.n_components_ != txt_features.shape[1]:
        raise ValueError("PCA whitening discarded CLIP components")
    print(f"Full-rank whitening fit: seconds={whitening_seconds:.3f}", flush=True)

    config = FactorTrainingConfig(lambda_usage_balance=0.1)
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
        "timings_seconds": {
            "graph": graph_seconds,
            "stage1": stage1_seconds,
            "communities": community_seconds,
            "whitening": whitening_seconds,
        },
        "runs": {},
    }

    for threshold in VARIANCE_THRESHOLDS:
        # The factor trainer requires matching modality widths. The smallest
        # common K meeting both independent PCA thresholds is max(K_img, K_txt).
        image_min_k = select_pca_rank(image_pca, threshold)
        text_min_k = select_pca_rank(text_pca, threshold)
        shared_k = max(image_min_k, text_min_k)
        whitened_img = full_img[:, :shared_k]
        whitened_txt = full_txt[:, :shared_k]
        whitened_variance = [np.var(array, axis=0) for array in (whitened_img, whitened_txt)]
        if any(variance.min() < 0.8 or variance.max() > 1.2 for variance in whitened_variance):
            raise ValueError("PCA whitening failed to give near-unit component variance")

        stage_started = perf_counter()
        log = io.StringIO()
        with redirect_stdout(log):
            model, img_codes, txt_codes = train_factors(whitened_img, whitened_txt, graph, config)
        factor_seconds = perf_counter() - stage_started
        losses = np.asarray([float(value) for value in re.findall(
            r"factor epoch=\d+ loss=([\d.]+)", log.getvalue()
        )])
        if len(losses) != config.epochs or not np.isfinite(losses).all():
            raise ValueError("Factor training loss trace is incomplete or non-finite")
        if not (np.isfinite(img_codes).all() and np.isfinite(txt_codes).all()):
            raise ValueError("Factor training produced non-finite codes")

        run = {
            "variance_threshold": threshold,
            "image_min_k": image_min_k,
            "text_min_k": text_min_k,
            "shared_k": shared_k,
            "variance_covered_at_shared_k": [
                float(np.sum(pca.explained_variance_ratio_[:shared_k]))
                for pca in (image_pca, text_pca)
            ],
            "whitened_variance_ranges": [
                [float(variance.min()), float(variance.max())]
                for variance in whitened_variance
            ],
            "loss_first5_mean": float(losses[:5].mean()),
            "loss_last5_mean": float(losses[-5:].mean()),
            "loss_min": float(losses.min()),
            "loss_max": float(losses.max()),
            "relative_reconstruction_l2_whitened": reconstruction_error(
                model, whitened_img, whitened_txt, next(model.parameters()).device
            ),
            "paired_cosine_matched_and_shuffled": cosine_agreement(
                img_codes, txt_codes, config.seed
            ),
            "factor_seconds": factor_seconds,
        }
        run.update(factor_diagnostics(img_codes, txt_codes, labels))
        mass = 0.5 * (np.asarray(run["image_mean"]) + np.asarray(run["text_mean"]))
        if mass.sum() <= 0:
            raise ValueError("Factor dictionary has no activation mass")
        ranked = np.argsort(mass)[::-1]
        run["max_factor_mass_share"] = float(mass[ranked[0]] / mass.sum())
        run["top_two_factor_mass_share"] = float(mass[ranked[:2]].sum() / mass.sum())
        run["top_two_factor_indices"] = ranked[:2].tolist()
        result["runs"][str(threshold)] = run
        print(f"threshold={threshold} K={shared_k} (image min={image_min_k}, text min={text_min_k}) "
              f"top2={run['top_two_factor_mass_share']:.4%} "
              f"reconstruction={run['relative_reconstruction_l2_whitened']} "
              f"factor_seconds={factor_seconds:.3f}", flush=True)
        del model, img_codes, txt_codes, run, log, losses
        gc.collect()

    result["timings_seconds"]["total"] = perf_counter() - started
    print("RESULT_JSON=" + json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
