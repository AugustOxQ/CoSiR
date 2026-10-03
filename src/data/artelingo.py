"""Load cached ArtELingo CLIP features with their positional annotation join."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils import FeatureManager

# Local defaults; a cluster job overrides them through the environment (e.g. scripts/run_mllm_probe_8b.sh).
FEATURE_DIR = os.environ.get("COSIR_ARTELINGO_FEATURES") or "/data/SSD2/pre_extract/artelingo/features"
ANNOTATIONS_PATH = Path(os.environ.get("COSIR_ARTELINGO_ANNOTATIONS") or "/data/PDD/artelingo/artelingo_train.json")


@dataclass(frozen=True)
class ArtelingoData:
    img_features: np.ndarray
    txt_features: np.ndarray
    sample_ids: np.ndarray
    emotions: np.ndarray          # per annotation row
    paintings: np.ndarray
    art_styles: np.ndarray        # str, one per row; one style per painting (asserted at load)


def _rows_by_sample_id(sample_ids: np.ndarray, annotations: list[dict]) -> list[dict]:
    """annotations[sample_id] per feature row (the positional join); ids must be unique and in range."""
    ids = np.asarray(sample_ids, dtype=np.int64)
    if len(ids) == 0 or ids.min() < 0 or ids.max() >= len(annotations) or len(np.unique(ids)) != len(ids):
        raise ValueError("Sample IDs must uniquely index the annotation list")
    return [annotations[int(i)] for i in ids]


def join_annotations(sample_ids: np.ndarray, annotations: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """annotations[sample_id] per feature row; ids must be unique and in range."""
    rows = _rows_by_sample_id(sample_ids, annotations)
    if any(not row.get("emotion") or not row.get("painting") for row in rows):
        raise ValueError("Every ArtELingo row needs non-empty emotion and painting fields")
    return (np.asarray([row["emotion"] for row in rows]),
            np.asarray([row["painting"] for row in rows]))


def join_art_styles(sample_ids: np.ndarray, annotations: list[dict]) -> np.ndarray:
    """annotations[sample_id]["art_style"] per feature row (same positional join as join_annotations).

    Art style is a per-painting label (Ruling 12): every painting must map to exactly one style.
    """
    rows = _rows_by_sample_id(sample_ids, annotations)
    if any(not row.get("art_style") or not row.get("painting") for row in rows):
        raise ValueError("Every ArtELingo row needs non-empty art_style and painting fields")
    styles = np.asarray([row["art_style"] for row in rows])
    by_painting: dict = {}
    for row, style in zip(rows, styles.tolist()):
        if by_painting.setdefault(row["painting"], style) != style:
            raise ValueError(f"Painting {row['painting']!r} maps to more than one art_style")
    return styles


def join_captions(sample_ids: np.ndarray, annotations: list[dict]) -> np.ndarray:
    """annotations[sample_id]["caption"] per feature row (the same positional join as join_annotations)."""
    rows = _rows_by_sample_id(sample_ids, annotations)
    if any(not isinstance(row.get("caption"), str) or not row["caption"].strip() for row in rows):
        raise ValueError("Every ArtELingo row needs a non-empty string caption")
    return np.asarray([row["caption"] for row in rows], dtype=object)


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
    art_styles = join_art_styles(sample_ids, annotations)
    return ArtelingoData(img, txt, sample_ids, emotions, paintings, art_styles)
