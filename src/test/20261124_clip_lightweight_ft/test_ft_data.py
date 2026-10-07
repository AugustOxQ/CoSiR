"""Unit tests for ft_data on toy data and tiny synthetic JPEGs (no real dataset is read)."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_data  # noqa: E402


def _toy(tmp_path):
    """6 feature rows over 3 paintings; sample_ids are shuffled (not 0..5)."""
    wiki = tmp_path / "wikiart"
    (wiki / "A").mkdir(parents=True)
    colours = {"p0": (255, 0, 0), "p1": (0, 255, 0), "p2": (0, 0, 255)}
    ann = []
    for k, (p, c) in enumerate(colours.items()):
        Image.new("RGB", (40 + 10 * k, 30), c).save(wiki / "A" / f"{p}.jpg", quality=95)
    for i in range(6):  # annotation i: painting p(i//2)
        p = f"p{i // 2}"
        ann.append({"image": f"A/{p}.jpg", "caption": f"cap{i}", "painting": p, "emotion": "x"})
    sample_ids = np.array([5, 0, 3, 1, 4, 2])
    paintings = np.array([ann[i]["painting"] for i in sample_ids])
    data = SimpleNamespace(sample_ids=sample_ids, paintings=paintings)
    # feature rows: 0..5 -> painting p2,p0,p1,p0,p2,p1
    splits = SimpleNamespace(scorer_train=np.array([0, 4, 1, 3]), val=np.array([2, 5]),
                             selection=np.array([], dtype=int), held=np.array([], dtype=int))
    return data, ann, wiki, splits


def test_split_index_excludes_held(tmp_path):
    data, ann, wiki, _ = _toy(tmp_path)
    splits = SimpleNamespace(scorer_train=np.array([0, 1]), val=np.array([2]), selection=np.array([3]),
                             held=np.array([4, 5]))
    idx = ft_data.split_index(data, splits=splits)
    assert set(idx) == {"scorer_train", "val", "selection"}
    assert not (set(np.concatenate(list(idx.values())).tolist()) & {4, 5})
    assert idx["val"].tolist() == [2]


def test_split_index_rejects_overlap(tmp_path):
    data, ann, wiki, _ = _toy(tmp_path)
    splits = SimpleNamespace(scorer_train=np.array([0, 1]), val=np.array([1]), selection=np.array([3]),
                             held=np.array([4, 5]))
    with pytest.raises(ValueError):
        ft_data.split_index(data, splits=splits)


def test_captions_follow_sample_ids(tmp_path):
    data, ann, wiki, _ = _toy(tmp_path)
    assert ft_data.captions(data.sample_ids, ann) == ["cap5", "cap0", "cap3", "cap1", "cap4", "cap2"]
    assert ft_data.captions(data.sample_ids[[2, 0]], ann) == ["cap3", "cap5"]


def test_painting_table(tmp_path):
    data, ann, wiki, _ = _toy(tmp_path)
    uniq, pos = ft_data.painting_table(data, np.array([0, 1, 2, 3, 4, 5]))
    assert uniq.tolist() == ["p0", "p1", "p2"]
    assert uniq[pos].tolist() == data.paintings.tolist()
    uniq, pos = ft_data.painting_table(data, np.array([4, 2]))
    assert uniq.tolist() == ["p1", "p2"] and pos.tolist() == [1, 0]


def test_cache_build_and_load(tmp_path):
    data, ann, wiki, splits = _toy(tmp_path)
    out = tmp_path / "cache"
    rec = ft_data.build_image_cache(out, data, ann, wiki, splits=splits, workers=2, verbose=False)
    arr = np.load(out / "images_uint8.npy", mmap_mode="r")
    assert arr.shape == (3, 224, 224, 3) and arr.dtype == np.uint8
    idx = json.loads((out / "paintings.json").read_text())
    assert [idx["paintings"][p]["index"] for p in ("p0", "p1", "p2")] == [0, 1, 2]
    assert idx["paintings"]["p1"]["image"] == "A/p1.jpg"
    # colours land in the right slot (jpeg tolerance)
    assert arr[0, 112, 112, 0] > 200 and arr[1, 112, 112, 1] > 200 and arr[2, 112, 112, 2] > 200
    assert rec["n_images"] == 3 and rec["normalisation_check"]["max_abs_diff"] < 1e-4
    images, index, record = ft_data.load_image_cache(out)
    assert images.shape == (3, 224, 224, 3) and index["p2"] == 2
    assert record["sha256"]["images_uint8.npy"] == hashlib.sha256((out / "images_uint8.npy").read_bytes()).hexdigest()


def test_refuse_overwrite(tmp_path):
    data, ann, wiki, splits = _toy(tmp_path)
    out = tmp_path / "cache"
    ft_data.build_image_cache(out, data, ann, wiki, splits=splits, workers=1, verbose=False)
    with pytest.raises(FileExistsError):
        ft_data.build_image_cache(out, data, ann, wiki, splits=splits, workers=1, verbose=False)


def test_held_row_raises(tmp_path):
    data, ann, wiki, _ = _toy(tmp_path)
    bad = SimpleNamespace(scorer_train=np.array([0, 1]), val=np.array([2]), selection=np.array([], dtype=int),
                          held=np.array([1, 5]))
    with pytest.raises(ValueError, match="held"):
        ft_data.build_image_cache(tmp_path / "c", data, ann, wiki, splits=bad, workers=1, verbose=False)
    assert not (tmp_path / "c" / "images_uint8.npy").exists()


def test_painting_shared_with_held_raises(tmp_path):
    data, ann, wiki, _ = _toy(tmp_path)  # rows 0,4 are p2; make row 4 held while row 0 is used
    bad = SimpleNamespace(scorer_train=np.array([0]), val=np.array([2]), selection=np.array([], dtype=int),
                          held=np.array([4]))
    with pytest.raises(ValueError, match="held"):
        ft_data.build_image_cache(tmp_path / "c", data, ann, wiki, splits=bad, workers=1, verbose=False)


def test_load_detects_corruption(tmp_path):
    data, ann, wiki, splits = _toy(tmp_path)
    out = tmp_path / "cache"
    ft_data.build_image_cache(out, data, ann, wiki, splits=splits, workers=1, verbose=False)
    (out / "paintings.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA"):
        ft_data.load_image_cache(out)
