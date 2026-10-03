"""The stage (d) ArtELingo split and per-row aspect label codes (CVPR plan spec §2.3, §5)."""

from dataclasses import dataclass

import numpy as np

from src.data.splits import grouped_split, grouped_subsplit, leakage_groups
from src.data.wikiart_genre import load_wikiart_genre

SPLIT_SEED = 42
SELECTION_FRACTION = 0.15
EXPECTED_SIZES = {"train": 216_107, "val": 30_872, "held": 61_744, "scorer_train": 183_694, "selection": 32_413}
EMOTION_CATCH_ALL = "something else"


@dataclass(frozen=True)
class ArtelingoSplits:
    groups: np.ndarray
    scorer_train: np.ndarray
    selection: np.ndarray
    val: np.ndarray
    held: np.ndarray


def artelingo_splits(data) -> ArtelingoSplits:
    """Recompute grouped_split(leakage_groups(...), 42) and the 15% selection sub-split; assert the known sizes."""
    groups = leakage_groups(data.paintings, data.img_features)
    split = grouped_split(groups, seed=SPLIT_SEED)
    scorer_train, selection = grouped_subsplit(groups, split.train, SELECTION_FRACTION, seed=SPLIT_SEED)
    got = {"train": len(split.train), "val": len(split.val), "held": len(split.held),
           "scorer_train": len(scorer_train), "selection": len(selection)}
    if got != EXPECTED_SIZES:
        raise AssertionError(f"split sizes {got} differ from the stage (d) split {EXPECTED_SIZES}")
    return ArtelingoSplits(groups, scorer_train, selection, np.sort(split.val), np.sort(split.held))


def encode_labels(values, missing=("", None), exclude=()) -> tuple[np.ndarray, list]:
    """Integer codes in sorted name order; -1 for missing or excluded values."""
    values = np.asarray(values, dtype=object)
    bad = set(missing) | set(exclude)
    names = sorted({v for v in values.tolist() if v not in bad})
    lookup = {name: i for i, name in enumerate(names)}
    return np.asarray([lookup.get(v, -1) for v in values.tolist()], dtype=np.int64), names


def artelingo_aspect_labels(data) -> dict[str, np.ndarray]:
    """Per-row codes for the three ArtELingo aspects; emotion excludes the catch-all, genre is -1 where unlabelled."""
    emotion, _ = encode_labels(data.emotions, exclude=(EMOTION_CATCH_ALL,))
    style, _ = encode_labels(data.art_styles)
    genre = load_wikiart_genre(np.asarray(data.paintings, dtype=object))
    return {"emotion": emotion, "style": style, "genre": genre}
