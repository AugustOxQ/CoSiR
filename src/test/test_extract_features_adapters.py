"""Adapter listing, scrubbing, GeneCIS crop and resume behaviour of scripts/extract_features.py (tiny fixtures)."""
import csv
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


def test_impressions_adapter(tmp_path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    train = [{"image_id": 42, "ImgId": "b_cap2_2", "image": "b_cap2_2.png",
              "caption": " first ", "caption_type": "aesthetic"},
             {"image_id": 17, "ImgId": "a", "image": "a.png", "caption": "second",
              "caption_type": "description"},
             {"image_id": 81, "ImgId": "b_cap2_2", "image": "b_cap2_2.png", "caption": "third",
              "caption_type": "impression"},
             {"image_id": 90, "ImgId": "a", "image": "a.png", "caption": "",
              "caption_type": "caption"}]
    test = [{"image_id": 23, "ImgId": "c", "image": "c.png", "caption": ["fourth", "", "fifth"],
             "caption_type": ["impression", "caption", "aesthetic"]},
            {"image_id": 51, "ImgId": "a", "image": "a.png", "caption": ["sixth", "   "]}]
    for split, rows in (("train", train), ("test", test)):
        (metadata / f"impressions_{split}.json").write_text(json.dumps(rows))
    (metadata / "mean_impact_scores.csv").write_text(
        ",ImgId,Impact_Mean,Impact_std,Impact_var\n0,b,1.0,0,0\n1,b_cap2_2,4.5,0,0\n2,a,2.0,0,0\n")
    for filename in ("a.png", "b_cap2_2.png", "c.png"):
        _img(tmp_path / "media" / filename)
    assert ef.ALLOWED["impressions"] == [ef.CLIP, ef.QWEN, ef.SIGLIP2]
    spec = ef.ADAPTERS["impressions"](impressions_dir=tmp_path)
    assert spec.images == [
        {"image": "b_cap2_2.png", "ImgId": "b_cap2_2", "uuid": "b", "has_overlay": True,
         "splits": ["train"], "impact_mean": 4.5},
        {"image": "a.png", "ImgId": "a", "uuid": "a", "has_overlay": False,
         "splits": ["test", "train"], "impact_mean": 2.0},
        {"image": "c.png", "ImgId": "c", "uuid": "c", "has_overlay": False,
         "splits": ["test"], "impact_mean": None}]
    assert spec.texts == [" first ", "second", "third", "fourth", "fifth", "sixth"]
    assert spec.text_meta == [
        {"img": 0, "split": "train", "caption_type": "aesthetic", "image_id": 42, "elem": 0},
        {"img": 1, "split": "train", "caption_type": "description", "image_id": 17, "elem": 0},
        {"img": 0, "split": "train", "caption_type": "impression", "image_id": 81, "elem": 0},
        {"img": 2, "split": "test", "caption_type": "impression", "image_id": 23, "elem": 0},
        {"img": 2, "split": "test", "caption_type": "aesthetic", "image_id": 23, "elem": 2},
        {"img": 1, "split": "test", "caption_type": None, "image_id": 51, "elem": 0}]
    assert spec.extra == {"n_train_texts": 3, "n_test_texts": 3, "n_images": 3, "n_overlay_images": 1,
                          "n_impact_missing": 1, "n_empty_captions_skipped": 3}
    with spec.load_image(0) as image:
        assert image.size == (40, 30) and image.mode == "RGB"


def test_impressions_adapter_missing_image(tmp_path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "mean_impact_scores.csv").write_text("ImgId,Impact_Mean\n")
    (metadata / "impressions_train.json").write_text(json.dumps([
        {"image_id": 75, "ImgId": "missing", "image": "missing.png", "caption": "text"}]))
    (metadata / "impressions_test.json").write_text("[]")
    with pytest.raises(SystemExit, match="missing.png"):
        ef.adapter_impressions(impressions_dir=tmp_path)


def _affection_fixture(tmp_path):
    roots = {"coco_affective": {s: tmp_path / s for s in ("train2014", "val2014")},
             "visual_genome_affective": tmp_path / "vg", "flickr30k_affective": tmp_path / "flickr"}
    rows = [
        ["w0", "shared.jpg", "visual_genome_affective", " first ", "awe", "train"],
        ["w1", "COCO_val2014_000000000078.jpg", "coco_affective", "second", "joy", "val"],
        ["w2", "absent.jpg", "you_aaai_affective", "drop", "fear", "test"],
        ["w3", "shared.jpg", "flickr30k_affective", "third", "sadness", "test"],
        ["w4", "shared.jpg", "visual_genome_affective", "fourth", "awe", "test"],
        ["w5", "absent.jpg", "emotional_machines_affective", "drop too", "anger", "train"],
        ["w6", "shared.jpg", "visual_genome_affective", "", "something-else", "ignored"],
        ["w7", "COCO_train2014_000000000078.jpg", "coco_affective", "fifth", "joy", "train"],
        ["w8", "empty.jpg", "flickr30k_affective", "NaN", "sadness", "ignored"],
        ["w9", "shared.jpg", "flickr30k_affective", "   ", "sadness", "val"],
    ]
    for version in ("raw", "preprocessed"):
        with (tmp_path / f"affection_{version}_public_version_0.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["workerid", "image_name", "dataset", "utterance", "emotion", "split",
                             "utterance_spelled", "tokens"])
            writer.writerows(r + ["DO NOT USE", "unused"] for r in rows)
    for _, name, source, *_ in rows:
        if source in roots:
            root = roots[source]
            if source == "coco_affective":
                root = root[name.split("_")[1]]
            _img(root / name)
    return roots


def test_affection_local_adapter(tmp_path):
    roots = _affection_fixture(tmp_path)
    assert ef.ALLOWED["affection_local"] == [ef.CLIP, ef.QWEN, ef.SIGLIP2]
    spec = ef.ADAPTERS["affection_local"](affection_dir=tmp_path, image_roots=roots)
    assert spec.images == [
        {"source": "visual_genome_affective", "image_name": "shared.jpg",
         "splits": ["ignored", "test", "train"], "n_texts": 2},
        {"source": "coco_affective", "image_name": "COCO_val2014_000000000078.jpg",
         "splits": ["val"], "n_texts": 1},
        {"source": "flickr30k_affective", "image_name": "shared.jpg",
         "splits": ["test", "val"], "n_texts": 1},
        {"source": "coco_affective", "image_name": "COCO_train2014_000000000078.jpg",
         "splits": ["train"], "n_texts": 1},
        {"source": "flickr30k_affective", "image_name": "empty.jpg",
         "splits": ["ignored"], "n_texts": 0}]
    assert spec.texts == [" first ", "second", "third", "fourth", "fifth"]
    assert spec.text_meta == [
        {"img": 0, "csv_row": 0, "emotion": "awe", "workerid": "w0", "split": "train"},
        {"img": 1, "csv_row": 1, "emotion": "joy", "workerid": "w1", "split": "val"},
        {"img": 2, "csv_row": 3, "emotion": "sadness", "workerid": "w3", "split": "test"},
        {"img": 0, "csv_row": 4, "emotion": "awe", "workerid": "w4", "split": "test"},
        {"img": 3, "csv_row": 7, "emotion": "joy", "workerid": "w7", "split": "train"}]
    assert spec.extra == {
        "sources": ["coco_affective", "visual_genome_affective", "flickr30k_affective"],
        "n_images_per_source": {"coco_affective": 2, "visual_genome_affective": 1, "flickr30k_affective": 2},
        "n_texts_per_source": {"coco_affective": 2, "visual_genome_affective": 2, "flickr30k_affective": 1},
        "n_empty_skipped": 3, "n_rows_dropped_other_sources": 2}
    for i in range(len(spec.images)):
        with spec.load_image(i) as image:
            assert image.size == (40, 30) and image.mode == "RGB"


@pytest.mark.parametrize("column", ["workerid", "image_name", "dataset", "length"])
def test_affection_local_adapter_misaligned_csv(tmp_path, column):
    roots = _affection_fixture(tmp_path)
    path = tmp_path / "affection_preprocessed_public_version_0.csv"
    with path.open() as f:
        rows = list(csv.reader(f))
    if column == "length":
        rows = rows[:-1]
        mismatch = 9
    else:
        rows[3][rows[0].index(column)] = "wrong"
        mismatch = 2  # Even a dropped source must match before filtering.
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    with pytest.raises(SystemExit, match=f"mismatch at row {mismatch} "):
        ef.adapter_affection_local(affection_dir=tmp_path, image_roots=roots)


def test_affection_local_adapter_missing_image(tmp_path):
    roots = _affection_fixture(tmp_path)
    roots["visual_genome_affective"] = tmp_path / "missing"
    with pytest.raises(SystemExit, match="missing/shared.jpg"):
        ef.adapter_affection_local(affection_dir=tmp_path, image_roots=roots)


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


@pytest.mark.parametrize("text,gone", [("built in the 1660s by him", "1660"), ("the 1770's view", "1770"),
                                       ("dated 1553-54 or so", "54"), ("c. 1730 approx", "1730")])
def test_scrub_semart_decades_ranges(text, gone):
    out = scrub_semart(text, "X, Y", "Z", "")
    assert gone not in out and "-" not in out and "'s" not in out


def test_scrub_semart_accent_fold_and_title_word_guard():
    out = scrub_semart("Velazquez and Velázquez, Madonnas and Madonna", "VELÁZQUEZ, Diego", "Madonna", "")
    assert "elazquez" not in out.replace("á", "a") and "Madonnas" in out and out.endswith("and")


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


def test_gate_refuses_nan_and_verify_reads_sealed(tmp_path):
    class NaNEnc(FakeEnc):
        def encode_texts(self, texts, bs):
            x = super().encode_texts(texts, bs)
            x[0] = np.nan
            return x
    out = tmp_path / "bad"
    with pytest.raises(SystemExit):
        ef.extract(_spec(tmp_path), NaNEnc(), out, dataset="t", log=lambda s: None)
    assert not (out / "index.json").exists() and not json.load(open(out / "progress.json"))["gate"]["ok"]
    good = tmp_path / "root" / "d" / ef.QWEN
    ef.extract(_spec(tmp_path), FakeEnc(), good, dataset="d", log=lambda s: None)
    assert ef.verify("d", ef.QWEN, root=tmp_path / "root")["ok"]
    t = np.load(good / "txt.npy"); t[1] *= 2; np.save(good / "txt.npy", t)
    assert not ef.verify("d", ef.QWEN, root=tmp_path / "root")["ok"]
