"""Adapter listing, scrubbing, GeneCIS crop and resume behaviour of scripts/extract_features.py (tiny fixtures)."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts import extract_features as ef
from scripts.extract_features import scrub_semart


def test_scrub_semart_removes_author_title_years():
    out = scrub_semart("Holbein painted the Darmstadt Madonna in 1526, late in his career.",
                       "HOLBEIN, Hans the Younger", "Darmstadt Madonna", "1526-28")
    assert "Holbein" not in out and "Darmstadt Madonna" not in out and "1526" not in out


def test_scrub_semart_possessive_particles_and_year_lengths():
    out = scrub_semart("Holbein's portrait of the king, c. 150 and 1499, not 12345.", "HOLBEIN, Hans the Younger",
                       "Nothing", "1499")
    assert "Holbein" not in out and "'s" not in out and "150" not in out and "1499" not in out
    assert "the king" in out and "12345" in out          # particles stay; five-digit numbers are not years


def _img(path, size=(40, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (200, 10, 10)).save(path)


def test_semart_adapter(tmp_path):
    hdr = "IMAGE_FILE\tDESCRIPTION\tAUTHOR\tTITLE\tTECHNIQUE\tDATE\tTYPE\tSCHOOL\tTIMEFRAME\n"
    for s in ("train", "val", "test"):
        (tmp_path / f"semart_{s}.csv").write_text(
            hdr + f"{s}.jpg\tGuardi painted Lagoon View in 1770.\tGUARDI, Francesco\tLagoon View\toil\t1770\t"
                  f"landscape\tItalian\t1751-1800\n", encoding="latin-1")
        _img(tmp_path / "Images" / f"{s}.jpg")
    spec = ef.adapter_semart(semart_dir=tmp_path)
    assert [m["split"] for m in spec.images] == ["train", "val", "test"]
    assert set(spec.images[0]) == {"image_file", "split", "type", "school", "timeframe"}
    assert all("Guardi" not in t and "1770" not in t and "Lagoon View" not in t for t in spec.texts)
    assert spec.load_image(0).size == (40, 30) and spec.extra["n_descriptions_changed_by_scrub"] == 3


def test_coco_adapter_groups_captions_and_checks_path(tmp_path):
    ann = {"images": [{"id": 2, "file_name": "b.jpg"}, {"id": 1, "file_name": "a.jpg"}],
           "annotations": [{"image_id": 2, "id": 20, "caption": " two "}, {"image_id": 1, "id": 11, "caption": "x"},
                           {"image_id": 1, "id": 10, "caption": "y"}]}
    (tmp_path / "c.json").write_text(json.dumps(ann))
    for n in "ab":
        _img(tmp_path / "imgs" / f"{n}.jpg")
    spec = ef.adapter_coco_train2014(image_dir=tmp_path / "imgs", captions_path=tmp_path / "c.json")
    assert [m["image_id"] for m in spec.images] == [1, 2]
    assert spec.texts == ["y", "x", "two"] and [m["img"] for m in spec.text_meta] == [0, 0, 1]
    with pytest.raises(SystemExit):
        ef.adapter_coco_train2014(image_dir=tmp_path / "missing", captions_path=tmp_path / "c.json")


def test_genecis_coco_adapter_matches_store(tmp_path):
    rows = [{"image": "val2014/a.jpg", "caption": "c0", "image_id": "coco_1", "caption_id": 5},
            {"image": "val2014/a.jpg", "caption": "c1", "image_id": "coco_1", "caption_id": 6},
            {"image": "val2014/b.jpg", "caption": "c2", "image_id": "coco_2", "caption_id": 7}]
    (tmp_path / "g.json").write_text(json.dumps(rows))
    shard = tmp_path / "store" / "shards" / "shard_00000"
    shard.mkdir(parents=True)
    np.save(shard / "sample_ids.npy", np.arange(3))
    for n in "ab":
        _img(tmp_path / "images" / "val2014" / f"{n}.jpg")
    spec = ef.adapter_genecis_coco(json_path=tmp_path / "g.json", image_root=tmp_path / "images",
                                   store=tmp_path / "store")
    assert [m["img"] for m in spec.text_meta] == [0, 0, 1] and [m["sample_id"] for m in spec.text_meta] == [0, 1, 2]
    np.save(shard / "sample_ids.npy", np.arange(3)[::-1].copy())
    with pytest.raises(ValueError):
        ef.adapter_genecis_coco(json_path=tmp_path / "g.json", image_root=tmp_path / "images", store=tmp_path / "store")


def test_artelingo_adapter_follows_sample_id_order(tmp_path):
    ann = [{"image": f"S/{p}.jpg", "caption": f"cap{i}", "painting": p} for i, p in enumerate("abab")]
    (tmp_path / "ann.json").write_text(json.dumps(ann))
    for p in "ab":
        _img(tmp_path / "S" / f"{p}.jpg")
    spec = ef.adapter_artelingo_full(annotations_path=tmp_path / "ann.json", image_dir=tmp_path,
                                     sample_ids=[3, 0, 2, 1], expected=(2, 4))
    assert spec.texts == ["cap3", "cap0", "cap2", "cap1"]
    assert [m["painting"] for m in spec.images] == ["b", "a"] and [m["img"] for m in spec.text_meta] == [0, 1, 1, 0]
    with pytest.raises(ValueError):
        ef.adapter_artelingo_full(annotations_path=tmp_path / "ann.json", image_dir=tmp_path, sample_ids=[0, 1],
                                  expected=(2, 4))


def test_vg_crops_union_sorted_without_roles(tmp_path):
    def role(i, bb):
        return {"image_id": str(i), "instance_bbox": bb}
    items = [{"condition": "SECRET red", "reference": role(10, [5, 5, 20, 10]), "target": role(2, [0, 0, 8, 8]),
              "gallery": [role(10, [5, 5, 20, 10]), role(2, [1, 1, 8, 8])]},
             {"condition": "SECRET blue", "reference": role(2, [0, 0, 8, 8]), "target": role(10, [0, 0, 4, 4]),
              "gallery": []}]
    (tmp_path / "f.json").write_text(json.dumps(items))
    for i in (2, 10):
        _img(tmp_path / "vg" / f"{i}.jpg", (100, 60))
    spec = ef.adapter_genecis_vg_crops(focus_path=tmp_path / "f.json", image_dir=tmp_path / "vg")
    assert [(m["image_id"], m["bbox"]) for m in spec.images] == [
        ("2", [0, 0, 8, 8]), ("2", [1, 1, 8, 8]), ("10", [0, 0, 4, 4]), ("10", [5, 5, 20, 10])]
    assert all(set(m) == {"image_id", "bbox"} for m in spec.images) and spec.texts == []
    assert "SECRET" not in json.dumps(spec.images)
    assert all(spec.load_image(i).size[0] == spec.load_image(i).size[1] for i in range(4))   # padded to a square


def test_crop_matches_genecis_arithmetic(tmp_path):
    _img(tmp_path / "7.jpg", (100, 60))
    im = ef.load_cropped_image(tmp_path, "7", [50, 30, 20, 10])
    # left=max(0,50-14)=36 top=max(0,30-7)=23 right=min(100,36+34)=70 bottom=min(60,23+17)=40 -> 34x17, padded to 34
    assert im.size == (34, 34)


class FakeEnc:
    name, dim = ef.QWEN, 8

    def __init__(self):
        self.p = SimpleNamespace(tokenizer=lambda ts, **k: {"input_ids": [list(range(len(t))) for t in ts]})
        self.calls = 0

    def _prompt(self, content):
        return content[0]["text"]

    def encode_images(self, images, bs):
        self.calls += 1
        return np.tile(np.eye(8, dtype=np.float32)[0], (len(images), 1))

    def encode_texts(self, texts, bs):
        self.calls += 1
        return np.tile(np.eye(8, dtype=np.float32)[1], (len(texts), 1))


def _spec(tmp_path, n=5):
    for i in range(n):
        _img(tmp_path / f"{i}.jpg")
    images = [{"id": i} for i in range(n)]
    texts = ["a" * (600 if i == 0 else 3) for i in range(n)]
    return ef.Spec(images, lambda i: ef._open_rgb(tmp_path / f"{i}.jpg"), texts, [{"img": i} for i in range(n)])


def test_extract_writes_float16_index_counts_truncation_and_refuses_overwrite(tmp_path):
    out = tmp_path / "out"
    prog = ef.extract(_spec(tmp_path), FakeEnc(), out, dataset="t", log=lambda s: None)
    assert np.load(out / "img.npy").dtype == np.float16 and np.load(out / "img.npy").shape == (5, 8)
    idx = json.load(open(out / "index.json"))
    assert idx["n_texts_truncated"] == 1 and prog["n_truncated"] == 1 and len(idx["images"]) == 5
    with pytest.raises(SystemExit):
        ef.extract(_spec(tmp_path), FakeEnc(), out, dataset="t", log=lambda s: None)
    ef.extract(_spec(tmp_path), FakeEnc(), out, overwrite=True, dataset="t", log=lambda s: None)


def test_extract_resumes_from_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(ef, "IMG_CHUNK", 2)
    monkeypatch.setattr(ef, "TXT_CHUNK", 2)
    out = tmp_path / "out"

    class CrashOnSecondImageChunk(FakeEnc):
        def encode_images(self, images, bs):
            if self.calls == 1:
                raise RuntimeError("crash")
            return super().encode_images(images, bs)
    with pytest.raises(RuntimeError):
        ef.extract(_spec(tmp_path), CrashOnSecondImageChunk(), out, dataset="t", log=lambda s: None)
    assert json.load(open(out / "progress.json"))["img_done"] == 2 and not (out / "index.json").exists()
    again = FakeEnc()
    ef.extract(_spec(tmp_path), again, out, dataset="t", log=lambda s: None)
    assert (np.load(out / "img.npy")[:, 0] == 1).all() and (np.load(out / "txt.npy")[:, 1] == 1).all()
    assert again.calls == 2 + 3          # image chunks 2..4 and 4..5, then three text chunks of 2
    assert json.load(open(out / "progress.json"))["img_done"] == 5
