"""CUB-200-2011 with Reed et al. captions, per-image attribute groups and the xlsa17 zero-shot species split."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

CUB_ROOT = Path("/data/SSD/cub")
MIN_CERTAINTY = 3                                   # CUB certainty 3 = "probably", 4 = "definitely"


@dataclass(frozen=True)
class CubData:
    image_ids: np.ndarray
    paths: np.ndarray
    species: np.ndarray
    captions: list
    attributes: dict
    attribute_values: dict


def _pairs(path):
    return [line.split(maxsplit=1) for line in Path(path).read_text().splitlines() if line.strip()]


def load_cub(root=CUB_ROOT) -> CubData:
    root = Path(root)
    base = root / "CUB_200_2011"
    images = _pairs(base / "images.txt")
    image_ids = np.array([int(i) for i, _ in images])
    paths = np.array([p for _, p in images], dtype=object)
    species = np.array([int(c) for _, c in _pairs(base / "image_class_labels.txt")])
    attr_file = root / "attributes.txt" if (root / "attributes.txt").exists() else base / "attributes" / "attributes.txt"
    group_of, value_of, values = {}, {}, {}
    for aid, name in _pairs(attr_file):
        group, value = name.split("::")
        group_of[int(aid)] = group
        values.setdefault(group, []).append(value)
        value_of[int(aid)] = values[group].index(value)
    present = {g: [set() for _ in image_ids] for g in values}
    pos = {int(i): k for k, i in enumerate(image_ids)}
    for line in (base / "attributes" / "image_attribute_labels.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        img, aid, is_present, certainty = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        if is_present == 1 and certainty >= MIN_CERTAINTY:
            present[group_of[aid]][pos[img]].add(value_of[aid])
    attributes = {g: np.array([next(iter(s)) if len(s) == 1 else -1 for s in sets], dtype=np.int64)
                  for g, sets in present.items()}
    captions = []
    for p in paths:
        cls, name = Path(p).parent.name, Path(p).stem
        lines = (root / "captions" / "extracted" / "text_c10" / cls / f"{name}.txt").read_text().splitlines()
        captions.append([l.strip() for l in lines if l.strip()])
    return CubData(image_ids, paths, species, captions, attributes, values)


def zero_shot_split(species: np.ndarray, class_list_dir, cub_base=CUB_ROOT / "CUB_200_2011"):
    """Indices of images whose species is in xlsa17 trainvalclasses.txt / testclasses.txt."""
    name_to_id = {name: int(i) for i, name in _pairs(Path(cub_base) / "classes.txt")}
    train = {name_to_id[n.strip()] for n in Path(class_list_dir, "trainvalclasses.txt").read_text().split()}
    test = {name_to_id[n.strip()] for n in Path(class_list_dir, "testclasses.txt").read_text().split()}
    if train & test:
        raise ValueError("zero-shot train and test species overlap")
    return np.flatnonzero(np.isin(species, list(train))), np.flatnonzero(np.isin(species, list(test)))


def dev_species(train_species_ids, n_dev: int = 30, seed: int = 42) -> np.ndarray:
    ids = np.unique(np.asarray(train_species_ids))
    return np.sort(np.random.default_rng(seed).choice(ids, n_dev, replace=False))
