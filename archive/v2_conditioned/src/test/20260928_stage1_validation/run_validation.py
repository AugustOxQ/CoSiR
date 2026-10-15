"""One-off, full-train ArtELingo validation of content-only Stage 1."""

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.model.communities import community_stats, detect_communities
from src.model.graph import GraphConfig, build_content_graph
from src.train.stage1 import Stage1Config, train_stage1
from src.utils import FeatureManager


FEATURE_DIR = "/data/SSD2/pre_extract/artelingo/features"
ANNOTATIONS_PATH = Path("/data/PDD/artelingo/artelingo_train.json")
EXPECTED_SAMPLES = 308_723


def main() -> None:
    started = perf_counter()
    manager = FeatureManager(storage_dir=FEATURE_DIR)
    with ANNOTATIONS_PATH.open() as file:
        annotations = json.load(file)
    if not (len(annotations) == manager.total_samples == EXPECTED_SAMPLES):
        raise ValueError(
            "Annotation/feature count mismatch: "
            f"annotations={len(annotations)}, features={manager.total_samples}, "
            f"expected={EXPECTED_SAMPLES}"
        )

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
    print(
        f"Loaded {len(sample_ids):,} samples; {len(np.unique(emotions))} emotion labels; "
        f"load_seconds={perf_counter() - started:.3f}",
        flush=True,
    )

    stage_started = perf_counter()
    graph = build_content_graph(img_features, txt_features, GraphConfig())
    graph_seconds = perf_counter() - stage_started
    print(f"Graph: nodes={graph.shape[0]:,}, edges={graph.nnz // 2:,}, "
          f"seconds={graph_seconds:.3f}", flush=True)

    stage_started = perf_counter()
    _, embeddings = train_stage1(
        img_features, txt_features, graph, Stage1Config()
    )
    training_seconds = perf_counter() - stage_started
    if not np.isfinite(embeddings).all():
        raise ValueError("Stage 1 produced non-finite embeddings")
    print(f"Training: embeddings={embeddings.shape}, "
          f"seconds={training_seconds:.3f}", flush=True)

    stage_started = perf_counter()
    labels = detect_communities(embeddings)
    detection_seconds = perf_counter() - stage_started
    stats = community_stats(labels)
    median_size = float(np.median(stats["sizes"]))
    ami = adjusted_mutual_info_score(emotions, labels)

    print("\nArtELingo content-only Stage 1 — train-split in-sample validation", flush=True)
    print(f"Samples: {len(sample_ids):,}")
    print(f"Communities: {stats['num_communities']:,}")
    print(f"Community sizes (min / median / max): "
          f"{stats['min_size']:,} / {median_size:g} / {stats['max_size']:,}")
    print(f"Empty communities: {stats['empty_count']}")
    print(f"Emotion AMI (in-sample): {ami:.6f}")
    print(f"Wall-clock graph construction: {graph_seconds:.3f} s")
    print(f"Wall-clock Stage 1 training: {training_seconds:.3f} s")
    print(f"Wall-clock community detection: {detection_seconds:.3f} s")
    print(f"Wall-clock total (including load and AMI): {perf_counter() - started:.3f} s",
          flush=True)


if __name__ == "__main__":
    main()
