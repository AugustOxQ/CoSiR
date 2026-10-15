"""ArtELingo in CLIP row order; every backbone uses the frozen CLIP split."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np

from r9_backbone import BACKBONES, DIMENSIONS, STORE_PATHS, TEST
from src.data.artelingo import ArtelingoData, load_artelingo
from src.data.artelingo_splits import EXPECTED_SIZES, SPLIT_SEED, artelingo_splits
from src.data.splits import grouped_split

N_ROWS = 308_723
SPLIT_PATHS = {
    "prepare": TEST / "20261013_stage_d_selection/cache/prepare.npz",
    "held": TEST / "20261014_stage_d_final/cache/held_codes.npz",
}
# Round 6's whole-file pins; only index keys are read, never held feature codes.
SPLIT_SHA256 = {
    "prepare": "d30f0281bb521d18c7a4d4adca1a7938689e2789db7fd528ce0322b90ac44c8b",
    "held": "2901f875e7ad59f9f54509b61352a9cecb0eba310b31357f872219f25e2a12e9",
}


def _assert_features(features, shape, dtype, name):
    assert features.shape == shape, f"{name} shape {features.shape} != {shape}"
    assert features.dtype == dtype, f"{name} dtype {features.dtype} != {dtype}"
    for start in range(0, len(features), 4096):
        assert np.isfinite(features[start:start + 4096]).all(), f"{name} has non-finite features"


def load_data(backbone) -> ArtelingoData:
    """Read caption i and its indexed image, retaining CLIP's labels and row IDs."""
    assert backbone in BACKBONES, f"unknown backbone: {backbone}"
    data = load_artelingo()
    assert all(getattr(data, field).shape == (N_ROWS,) for field in (
        "sample_ids", "emotions", "paintings", "art_styles")), "CLIP row count"
    assert data.sample_ids.dtype == np.int64, "CLIP sample_id dtype"
    if backbone == "clip_b32":
        for name in ("img_features", "txt_features"):
            _assert_features(getattr(data, name), (N_ROWS, DIMENSIONS[backbone]), np.float32, name)
        return data
    path = STORE_PATHS[backbone]
    with (path / "index.json").open() as file:
        index = json.load(file)
    required = {"dataset", "backbone", "dim", "dtype", "n_images", "n_texts", "images", "texts"}
    assert required <= index.keys(), f"{path}/index.json missing columns"
    assert (index["dataset"], index["backbone"], index["dim"], index["dtype"]) == (
        "artelingo_full", backbone, DIMENSIONS[backbone], "float16"), "feature store metadata"
    assert len(index["texts"]) == len(data.sample_ids) == len(data.paintings) == N_ROWS, "caption row count"
    assert index["n_texts"] == N_ROWS, "caption metadata count"
    assert index["n_images"] == len(index["images"]) == len(set(data.paintings)), "image metadata count"
    assert all("painting" in row for row in index["images"]), "image metadata columns"
    assert len({row["painting"] for row in index["images"]}) == index["n_images"], "duplicate image painting"
    for row, sample_id, painting in zip(index["texts"], data.sample_ids, data.paintings):
        assert {"sample_id", "painting", "img"} <= row.keys(), "caption metadata columns"
        assert type(row["sample_id"]) is int, f"sample_id is not an integer: {row}"
        assert row["sample_id"] == sample_id, f"sample_id differs from CLIP: {row}"
        assert row["painting"] == painting, f"painting differs from CLIP: {row}"
        img_row = row["img"]
        assert type(img_row) is int and 0 <= img_row < len(index["images"]), f"image join out of range: {row}"
        assert index["images"][img_row]["painting"] == painting, f"image join painting differs: {row}"
    images = np.load(path / "img.npy", mmap_mode="r", allow_pickle=False)
    texts = np.load(path / "txt.npy", mmap_mode="r", allow_pickle=False)
    _assert_features(images, (index["n_images"], DIMENSIONS[backbone]), np.float16, "img.npy")
    _assert_features(texts, (N_ROWS, DIMENSIONS[backbone]), np.float16, "txt.npy")
    image_rows = np.asarray([row["img"] for row in index["texts"]], dtype=np.int64)
    # Cast the per-painting image matrix first, avoiding a large float16 row expansion.
    img = images.astype(np.float32)[image_rows]
    txt = texts.astype(np.float32)
    return ArtelingoData(img, txt, data.sample_ids, data.emotions, data.paintings, data.art_styles)


def load_split(data) -> SimpleNamespace:
    """Read pinned CLIP indices, with the round-6 recomputation check on CLIP only.

    ArtelingoData has no backbone field: equality with the canonical CLIP image
    features identifies CLIP. New features never form groups.
    Validation rows are the complement of the stored train and held arrays.
    """
    assert all(getattr(data, field).shape == (N_ROWS,) for field in (
        "sample_ids", "paintings", "emotions", "art_styles")), "data row count differs from ArtELingo"
    assert data.img_features.ndim == 2 and data.img_features.shape[1] in DIMENSIONS.values(), "unknown feature dimension"
    dim = data.img_features.shape[1]
    assert data.img_features.shape == data.txt_features.shape == (N_ROWS, dim), "feature row alignment"
    assert data.img_features.dtype == data.txt_features.dtype == np.float32, "feature dtype must be float32"
    reference = load_artelingo()
    assert np.array_equal(data.sample_ids, reference.sample_ids), "sample_id row order differs from CLIP"
    assert np.array_equal(data.paintings, reference.paintings), "painting row order differs from CLIP"
    is_clip = np.array_equal(data.img_features, reference.img_features)
    del reference
    for role, path in SPLIT_PATHS.items():
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1 << 22), b""):
                digest.update(block)
        assert digest.hexdigest() == SPLIT_SHA256[role], f"{path}: SHA-256 differs from the CLIP pin"
    with np.load(SPLIT_PATHS["prepare"], allow_pickle=False) as prepared:
        assert {"groups", "split_train", "scorer_train", "selection"} <= set(prepared.files), "prepare.npz columns"
        groups, train, scorer_train, selection = (
            prepared[key] for key in ("groups", "split_train", "scorer_train", "selection"))
    with np.load(SPLIT_PATHS["held"], allow_pickle=False) as held_file:
        assert "held_rows" in held_file.files, "held_codes.npz columns"
        held = held_file["held_rows"]
    for name, rows in (("groups", groups), ("train", train), ("held", held),
                       ("scorer_train", scorer_train), ("selection", selection)):
        size = N_ROWS if name == "groups" else EXPECTED_SIZES[name]
        assert rows.shape == (size,), f"{name} shape differs from CLIP's size {size}"
        assert rows.dtype == np.int64, f"{name} dtype must be int64"
        assert ((rows >= 0) & (rows < N_ROWS)).all(), f"{name} indices out of range"
        if name != "groups":
            assert (np.diff(rows) > 0).all(), f"{name} rows must be unique and sorted"
    val = np.setdiff1d(np.arange(N_ROWS, dtype=np.int64), np.union1d(train, held))
    assert len(val) == EXPECTED_SIZES["val"], "val size differs from CLIP"
    assert np.intersect1d(train, held).size == 0, "held overlaps train"
    assert np.intersect1d(scorer_train, selection).size == 0 and np.array_equal(
        np.union1d(scorer_train, selection), train), "scorer_train and selection do not partition train"
    for left, right in ((train, val), (train, held), (val, held), (scorer_train, selection)):
        assert np.intersect1d(groups[left], groups[right]).size == 0, "a leakage group spans split parts"
    if is_clip:
        recomputed = artelingo_splits(data)
        gs = grouped_split(recomputed.groups, seed=SPLIT_SEED)
        for name, mine, frozen in (
            ("groups", recomputed.groups, groups), ("split_train", gs.train, train),
            ("scorer_train", recomputed.scorer_train, scorer_train),
            ("selection", recomputed.selection, selection), ("val", recomputed.val, val),
            ("held_rows", recomputed.held, held),
        ):
            assert np.array_equal(mine, frozen), f"recomputed {name} differs from CLIP's pinned split"
        assert np.array_equal(gs.val, val) and np.array_equal(gs.held, held), "grouped_split disagrees on val or held"
    return SimpleNamespace(groups=groups, train=train, val=val, held=held,
                           scorer_train=scorer_train, selection=selection)
