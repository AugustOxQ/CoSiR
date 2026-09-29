"""Load cached ArtELingo CLIP features with their positional annotation join."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils import FeatureManager

FEATURE_DIR = "/data/SSD2/pre_extract/artelingo/features"
ANNOTATIONS_PATH = Path("/data/PDD/artelingo/artelingo_train.json")


@dataclass(frozen=True)
class ArtelingoData:
    img_features: np.ndarray
    txt_features: np.ndarray
    sample_ids: np.ndarray
    emotions: np.ndarray
    paintings: np.ndarray


def join_annotations(sample_ids: np.ndarray, annotations: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """annotations[sample_id] per feature row; ids must be unique and in range."""
    ids = np.asarray(sample_ids, dtype=np.int64)
    if len(ids) == 0 or ids.min() < 0 or ids.max() >= len(annotations) or len(np.unique(ids)) != len(ids):
        raise ValueError("Sample IDs must uniquely index the annotation list")
    rows = [annotations[int(i)] for i in ids]
    if any(not row.get("emotion") or not row.get("painting") for row in rows):
        raise ValueError("Every ArtELingo row needs non-empty emotion and painting fields")
    return (np.asarray([row["emotion"] for row in rows]),
            np.asarray([row["painting"] for row in rows]))


def load_artelingo(feature_dir: str = FEATURE_DIR, annotations_path: Path = ANNOTATIONS_PATH,
                   expected_samples: int = 308_723) -> ArtelingoData:
    manager = FeatureManager(storage_dir=feature_dir)
    with Path(annotations_path).open() as file:
        annotations = json.load(file)
    if not (len(annotations) == manager.total_samples == expected_samples):
        raise ValueError("Annotation/feature count mismatch")
    features = manager.load_all_to_ram(["img_features", "txt_features"])
    sample_ids = np.asarray(manager.get_all_sample_ids(), dtype=np.int64)
    img = features["img_features"].numpy().astype(np.float32, copy=False)
    txt = features["txt_features"].numpy().astype(np.float32, copy=False)
    if not (len(sample_ids) == len(img) == len(txt) == expected_samples):
        raise ValueError("Feature rows and sample IDs do not match metadata")
    emotions, paintings = join_annotations(sample_ids, annotations)
    return ArtelingoData(img, txt, sample_ids, emotions, paintings)
