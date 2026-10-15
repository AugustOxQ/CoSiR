"""One-off raw-CLIP baseline for the full-train Stage 1 emotion AMI."""

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.model.communities import community_stats, detect_communities
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

    # Unit-normalize each modality separately so neither feature norm sets its weight.
    img_unit = img_features / np.linalg.norm(img_features, axis=1, keepdims=True)
    txt_unit = txt_features / np.linalg.norm(txt_features, axis=1, keepdims=True)
    embeddings = np.concatenate((img_unit, txt_unit), axis=1)
    if not np.isfinite(embeddings).all():
        raise ValueError("Raw-CLIP baseline produced non-finite embeddings")
    print(f"Raw-CLIP embeddings: {embeddings.shape}", flush=True)

    detection_started = perf_counter()
    labels = detect_communities(embeddings)
    detection_seconds = perf_counter() - detection_started
    stats = community_stats(labels)
    ami = adjusted_mutual_info_score(emotions, labels)

    print("\nArtELingo raw-CLIP baseline — train-split in-sample validation")
    print(f"Samples: {len(sample_ids):,}")
    print(f"Communities: {stats['num_communities']:,}")
    print(f"Community sizes (min / median / max): "
          f"{stats['min_size']:,} / {np.median(stats['sizes']):g} / "
          f"{stats['max_size']:,}")
    print(f"Community sizes (all): {stats['sizes']}")
    print(f"Empty communities: {stats['empty_count']}")
    print(f"Emotion AMI (in-sample): {ami:.6f}")
    print(f"Wall-clock community detection: {detection_seconds:.3f} s")
    print(f"Wall-clock total (including load and AMI): {perf_counter() - started:.3f} s",
          flush=True)


if __name__ == "__main__":
    main()
