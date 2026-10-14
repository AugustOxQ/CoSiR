"""Synthetic file-boundary checks for the ArtELingo joins and frozen split."""

from dataclasses import replace
from functools import partial
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9_data as R
from src.data import artelingo_splits as S
from src.data.artelingo import ArtelingoData, load_artelingo


@pytest.fixture
def store(tmp_path, monkeypatch):
    # A real tiny CLIP shard store and annotations, with deliberately nontrivial IDs.
    clip = tmp_path / "clip"
    shard = clip / "shards/shard_00000"
    shard.mkdir(parents=True)
    ids = np.array([2, 0, 1], dtype=np.int64)
    np.save(shard / "sample_ids.npy", ids)
    for modality in ("img", "txt"):
        np.save(shard / f"{modality}_features.npy", np.zeros((3, 512), dtype=np.float32))
    (clip / "metadata.json").write_text(json.dumps({
        "total_samples": 3, "num_shards": 1, "shard_size": 3,
        "available_features": ["img_features", "txt_features"],
        "feature_dims": {"img_features": [512], "txt_features": [512]},
    }))
    ann = tmp_path / "annotations.json"
    ann.write_text(json.dumps([
        {"painting": "p0", "emotion": "awe", "art_style": "Baroque"},
        {"painting": "p1", "emotion": "fear", "art_style": "Cubism"},
        {"painting": "p0", "emotion": "sadness", "art_style": "Baroque"},
    ]))
    monkeypatch.setattr(R, "N_ROWS", 3)
    monkeypatch.setattr(R, "load_artelingo", partial(
        load_artelingo, feature_dir=str(clip), annotations_path=ann, expected_samples=3))
    paths = {}
    for backbone, dim in (("qwen3vl_emb_2b", 2048), ("siglip2_so400m", 1152)):
        path = tmp_path / backbone
        path.mkdir()
        # Image order differs from painting order: p0 is image 1.
        np.save(path / "img.npy", np.full((2, dim), [[7], [3]], dtype=np.float16))
        np.save(path / "txt.npy", np.full((3, dim), [[2], [4], [6]], dtype=np.float16))
        (path / "index.json").write_text(json.dumps({
            "dataset": "artelingo_full", "backbone": backbone, "dim": dim,
            "dtype": "float16", "n_images": 2, "n_texts": 3,
            "images": [{"painting": "p1"}, {"painting": "p0"}],
            "texts": [{"sample_id": 2, "painting": "p0", "img": 1},
                      {"sample_id": 0, "painting": "p0", "img": 1},
                      {"sample_id": 1, "painting": "p1", "img": 0}],
        }))
        paths[backbone] = path
        monkeypatch.setitem(R.STORE_PATHS, backbone, path)
    return paths


@pytest.mark.parametrize("backbone,dim", [("qwen3vl_emb_2b", 2048), ("siglip2_so400m", 1152)])
def test_caption_rows_join_their_images_and_keep_clip_labels(store, backbone, dim):
    data = R.load_data(backbone)
    assert data.sample_ids.tolist() == [2, 0, 1]
    assert data.paintings.tolist() == ["p0", "p0", "p1"]
    assert data.emotions.tolist() == ["sadness", "awe", "fear"]
    assert data.art_styles.tolist() == ["Baroque", "Baroque", "Cubism"]
    assert data.img_features.shape == data.txt_features.shape == (3, dim)
    assert data.img_features.dtype == data.txt_features.dtype == np.float32
    assert data.img_features[:, 0].tolist() == [3, 3, 7]
    assert data.txt_features[:, 0].tolist() == [2, 4, 6]


@pytest.mark.parametrize("backbone", ["qwen3vl_emb_2b", "siglip2_so400m"])
@pytest.mark.parametrize("change", ["permuted", "same_painting", "painting"])
def test_store_rows_must_match_clip_sample_ids_and_paintings(store, backbone, change):
    path = store[backbone] / "index.json"
    index = json.loads(path.read_text())
    if change == "permuted":
        index["texts"][1], index["texts"][2] = index["texts"][2], index["texts"][1]
    elif change == "same_painting":
        index["texts"][0], index["texts"][1] = index["texts"][1], index["texts"][0]
    else:
        index["texts"][0]["painting"] = "p1"
    path.write_text(json.dumps(index))
    message = "sample_id" if change == "same_painting" else "sample_id|painting"
    with pytest.raises(AssertionError, match=message):
        R.load_data(backbone)


@pytest.mark.parametrize("backbone", ["qwen3vl_emb_2b", "siglip2_so400m"])
@pytest.mark.parametrize("wrong_image", [0, -1, 2])
def test_wrong_image_join_is_refused(store, backbone, wrong_image):
    path = store[backbone] / "index.json"
    index = json.loads(path.read_text())
    index["texts"][0]["img"] = wrong_image
    path.write_text(json.dumps(index))
    with pytest.raises(AssertionError, match="image join"):
        R.load_data(backbone)


@pytest.mark.parametrize("backbone", ["qwen3vl_emb_2b", "siglip2_so400m"])
@pytest.mark.parametrize("damage", ["nan_image", "inf_text", "cut_text", "cut_image", "dim", "metadata", "columns"])
def test_feature_file_boundaries_are_asserted(store, backbone, damage):
    path = store[backbone]
    if damage in ("metadata", "columns"):
        index_path = path / "index.json"
        index = json.loads(index_path.read_text())
        if damage == "metadata":
            index["backbone"] = "wrong_backbone"
        else:
            del index["texts"][0]["sample_id"]
        index_path.write_text(json.dumps(index))
    else:
        name = "img.npy" if damage in ("nan_image", "cut_image") else "txt.npy"
        features = np.load(path / name)
        if damage.startswith("cut"):
            features = features[:-1]
        elif damage == "dim":
            features = features[:, :-1]
        else:
            features[0, 0] = np.nan if damage == "nan_image" else np.inf
        np.save(path / name, features)
    with pytest.raises(AssertionError):
        R.load_data(backbone)


def test_clip_keeps_the_original_loader_result(store):
    data = R.load_data("clip_b32")
    reference = R.load_artelingo()
    for field in ("img_features", "txt_features", "sample_ids", "paintings", "emotions", "art_styles"):
        assert np.array_equal(getattr(data, field), getattr(reference, field))


@pytest.fixture
def frozen_split(tmp_path, monkeypatch):
    # Literal known stage-(d) row positions in a 20-row dataset.
    arrays = {
        "groups": np.arange(20, dtype=np.int64),
        "split_train": np.array([0, 3, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19]),
        "scorer_train": np.array([0, 5, 6, 7, 9, 10, 11, 14, 15, 16, 18, 19]),
        "selection": np.array([3, 12]),
    }
    held = np.array([1, 8, 13, 17])
    paths = {"prepare": tmp_path / "prepare.npz", "held": tmp_path / "held_codes.npz"}
    # Object payloads must remain unread (allow_pickle=False would refuse them).
    np.savez(paths["prepare"], **arrays, img_codes=np.array([{"unread": True}], dtype=object))
    np.savez(paths["held"], held_rows=held, img_codes=np.array([{"unread": True}], dtype=object))
    for role, path in paths.items():
        monkeypatch.setitem(R.SPLIT_PATHS, role, path)
        monkeypatch.setitem(R.SPLIT_SHA256, role, hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(R, "N_ROWS", 20)
    monkeypatch.setattr(R, "EXPECTED_SIZES", {
        "train": 14, "val": 2, "held": 4, "scorer_train": 12, "selection": 2,
    })
    monkeypatch.setattr(S, "EXPECTED_SIZES", R.EXPECTED_SIZES)
    features = np.zeros((20, 512), dtype=np.float32)
    features[:, 0] = np.arange(20)
    reference = ArtelingoData(
        features, features.copy(), np.arange(19, -1, -1, dtype=np.int64),
        np.full(20, "awe"), np.array([f"p{i}" for i in range(20)]), np.full(20, "Baroque"),
    )
    # Input boundary: this is the same canonical CLIP dataset for every backbone.
    monkeypatch.setattr(R, "load_artelingo", lambda: reference)
    return reference, paths, arrays, held


@pytest.mark.parametrize("dim", [512, 2048, 1152])
def test_every_backbone_uses_the_frozen_clip_split(frozen_split, dim):
    reference, _, arrays, held = frozen_split
    data = reference if dim == 512 else replace(
        reference, img_features=np.ones((20, dim), dtype=np.float32),
        txt_features=np.ones((20, dim), dtype=np.float32),
    )
    # Recomputing groups on either new backbone would collapse all 20 rows.
    split = R.load_split(data)
    assert split.groups.tolist() == list(range(20))
    assert split.train.tolist() == arrays["split_train"].tolist()
    assert split.scorer_train.tolist() == arrays["scorer_train"].tolist()
    assert split.selection.tolist() == [3, 12]
    assert split.val.tolist() == [2, 4]
    assert split.held.tolist() == held.tolist()
    assert all(getattr(split, name).dtype == np.int64 for name in (
        "groups", "train", "scorer_train", "selection", "val", "held"))


@pytest.mark.parametrize("key", ["groups", "split_train", "scorer_train", "selection", "held_rows"])
def test_cut_split_array_is_refused_even_with_a_matching_hash(frozen_split, monkeypatch, key):
    reference, paths, arrays, held = frozen_split
    role = "held" if key == "held_rows" else "prepare"
    payload = {"held_rows": held} if role == "held" else dict(arrays)
    payload[key] = payload[key][:-1]
    np.savez(paths[role], **payload)
    monkeypatch.setitem(R.SPLIT_SHA256, role, hashlib.sha256(paths[role].read_bytes()).hexdigest())
    data = replace(reference, img_features=np.ones((20, 2048), dtype=np.float32),
                   txt_features=np.ones((20, 2048), dtype=np.float32))
    with pytest.raises(AssertionError, match="size|shape"):
        R.load_split(data)


@pytest.mark.parametrize("role", ["prepare", "held"])
def test_split_file_hash_must_match_the_pin(frozen_split, role):
    reference, paths, _, _ = frozen_split
    with paths[role].open("ab") as file:
        file.write(b"hash mismatch; arrays are unchanged")
    with pytest.raises(AssertionError, match="SHA-256"):
        R.load_split(reference)


def test_clip_recomputation_must_match_the_frozen_groups(frozen_split, monkeypatch):
    reference, _, _, _ = frozen_split
    features = reference.img_features.copy()
    features[1] = features[0]  # Same image vector now links two distinct paintings.
    changed_clip = replace(reference, img_features=features)
    monkeypatch.setattr(R, "load_artelingo", lambda: changed_clip)
    with pytest.raises(AssertionError):
        R.load_split(changed_clip)


def test_features_different_from_clip_do_not_recompute_its_split(frozen_split):
    reference, _, arrays, held = frozen_split
    # Equal dimensions do not establish that these are the canonical CLIP features.
    data = replace(reference, img_features=np.ones((20, 512), dtype=np.float32))
    split = R.load_split(data)
    assert np.array_equal(split.groups, arrays["groups"])
    assert np.array_equal(split.train, arrays["split_train"])
    assert np.array_equal(split.scorer_train, arrays["scorer_train"])
    assert np.array_equal(split.selection, arrays["selection"])
    assert split.val.tolist() == [2, 4]
    assert np.array_equal(split.held, held)


def test_clip_recomputation_checks_train_subsplit_when_val_and_held_match(frozen_split, monkeypatch):
    reference, paths, arrays, held = frozen_split
    payload = {key: rows.copy() for key, rows in arrays.items()}
    # Swap one train row between the two training parts; keep train, val and held.
    payload["scorer_train"] = np.array([3, 5, 6, 7, 9, 10, 11, 14, 15, 16, 18, 19])
    payload["selection"] = np.array([0, 12])
    np.savez(paths["prepare"], **payload)
    monkeypatch.setitem(R.SPLIT_SHA256, "prepare", hashlib.sha256(paths["prepare"].read_bytes()).hexdigest())
    recomputed = S.artelingo_splits(reference)
    assert recomputed.val.tolist() == [2, 4]
    assert np.array_equal(recomputed.held, held)
    with pytest.raises(AssertionError, match="recomputed scorer_train"):
        R.load_split(reference)


@pytest.mark.parametrize("change", ["ids", "paintings", "cut"])
def test_split_requires_canonical_row_alignment(frozen_split, change):
    reference, _, _, _ = frozen_split
    if change == "ids":
        bad = replace(reference, sample_ids=reference.sample_ids[::-1])
    elif change == "paintings":
        bad = replace(reference, paintings=reference.paintings[::-1])
    else:
        bad = replace(reference, sample_ids=reference.sample_ids[:-1])
    with pytest.raises(AssertionError, match="row|sample_id|painting"):
        R.load_split(bad)


@pytest.mark.parametrize("damage", ["columns", "dtype", "duplicate", "overlap", "group_leakage"])
def test_split_index_file_boundaries_are_asserted(frozen_split, monkeypatch, damage):
    reference, paths, arrays, _ = frozen_split
    payload = {k: v.copy() for k, v in arrays.items()}
    if damage == "columns":
        del payload["selection"]
    elif damage == "dtype":
        payload["selection"] = payload["selection"].astype(np.float32)
    elif damage == "duplicate":
        payload["selection"][:] = 3
    elif damage == "overlap":
        payload["selection"] = np.array([0, 3])
    else:
        payload["groups"][1] = 0  # Held row 1 and train row 0 share a leakage group.
    np.savez(paths["prepare"], **payload)
    monkeypatch.setitem(R.SPLIT_SHA256, "prepare", hashlib.sha256(paths["prepare"].read_bytes()).hexdigest())
    data = replace(reference, img_features=np.ones((20, 2048), dtype=np.float32),
                   txt_features=np.ones((20, 2048), dtype=np.float32))
    with pytest.raises(AssertionError):
        R.load_split(data)
