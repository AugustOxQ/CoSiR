"""Frozen-backbone feature extraction for the later CVPR experiments (plan Task 16, E4).

    python scripts/extract_features.py --dataset NAME --backbone {clip_b32,qwen3vl_emb_2b} [--limit N]

Writes /data/SSD2/pre_extract/<dataset>/<backbone>/{img.npy, txt.npy, index.json} (float16, L2-normalised) with a
progress.json resume file. With --limit the output goes to /data/SSD2/pre_extract/_smoke/... instead, so a smoke run
can never be mistaken for a real one. A finished output (index.json present) is never overwritten without
--overwrite; an unfinished one (progress.json only) is resumed.

Layout: img.npy has one row per image, txt.npy one row per text; index.json["images"][i] and ["texts"][j] describe
the rows, and every text carries "img" (the img.npy row of its image, or -1).

GeneCIS access disclosure (controller ruling): the genecis_vg_crops adapter opens
/project/genecis/genecis/focus_attribute.json ONLY to build the union of (image_id, bbox) crops over all roles. It
does not store, print or score any role, condition text, target identity or gallery membership. index.json keeps only
(image_id, bbox) per row, sorted deterministically, so the crop list cannot reveal targets. Nothing is scored here,
so this is a disclosed access, not a held read.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PRE_EXTRACT = Path("/data/SSD2/pre_extract")
CLIP, QWEN = "clip_b32", "qwen3vl_emb_2b"
ARTELINGO_ANN = Path("/data/PDD/artelingo/artelingo_train.json")
ARTELINGO_FEATURES = "/data/SSD2/pre_extract/artelingo/features"
ARTELINGO_IMAGES = Path("/data/PDD/wikiart_proj/wikiart")
SEMART_DIR = Path("/data/SSD/semart/SemArt")
COCO_ROOT = Path("/data/SSD/coco")
COCO_TRAIN_IMAGES = COCO_ROOT / "images" / "train2014"
COCO_TRAIN_CAPTIONS = COCO_ROOT / "annotations" / "captions_train2014.json"
GENECIS_COCO_JSON = Path("/data/PDD/genecis/genecis_coco.json")
GENECIS_COCO_STORE = Path("/data/SSD2/pre_extract/genecis_coco/features")
GENECIS_FOCUS_ATTRIBUTE = Path("/project/genecis/genecis/focus_attribute.json")
VG_IMAGES = Path("/data/SSD/visual_genome/VG_100K_all")

DILATION = 0.7        # GeneCIS datasets/vaw_dataset.py
PAD_CROP = True

ALLOWED = {
    "artelingo_full": [QWEN],
    "semart": [CLIP, QWEN],
    "coco_train2014": [CLIP, QWEN],
    "genecis_vg_crops": [CLIP, QWEN],
    "genecis_coco": [QWEN],
}
# batch sizes chosen to fit well under ~20 GB on an otherwise idle 24 GB GPU
BATCH = {CLIP: dict(img=256, txt=512), QWEN: dict(img=32, txt=128)}
IMG_CHUNK, TXT_CHUNK = 256, 4096


@dataclass
class Spec:
    """One dataset: n images (lazily loaded), a text list, and the row metadata for index.json."""
    images: list                                   # per-image metadata dicts
    load_image: Callable[[int], "object"]          # row -> RGB PIL image
    texts: list = field(default_factory=list)
    text_meta: list = field(default_factory=list)  # per-text dicts, each with "img"
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- SemArt scrubbing
_PARTICLES = {"the", "of", "de", "di", "da", "del", "della", "van", "von", "der", "den", "la", "le", "el", "and",
              "dei", "des", "du", "ten", "ter"}


def scrub_semart(description: str, author: str, title: str, date: str) -> str:
    """Remove the title string, the author's name tokens and every 3- or 4-digit year from a description.

    Name tokens are the author's words of three or more letters that are not particles (the, of, de, van, ...);
    a trailing possessive is removed with the token. Matching is case-insensitive on word boundaries. `date` is
    scrubbed through the year rule (its digits are years); centuries written in words are left alone.
    """
    out = description
    title = (title or "").strip()
    if title:
        out = re.sub(re.escape(title), " ", out, flags=re.IGNORECASE)
    for tok in re.findall(r"[^\W\d_]+", author or "", flags=re.UNICODE):
        if len(tok) >= 3 and tok.lower() not in _PARTICLES:
            out = re.sub(rf"\b{re.escape(tok)}(?:['’]s)?\b", " ", out, flags=re.IGNORECASE)
    out = re.sub(r"\b\d{3,4}\b", " ", out)
    return re.sub(r"\s+", " ", out).strip()


# ---------------------------------------------------------------- adapters
def _open_rgb(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")


def adapter_artelingo_full(annotations_path=ARTELINGO_ANN, image_dir=ARTELINGO_IMAGES, sample_ids=None,
                           expected=(61_402, 308_723), **_) -> Spec:
    """One image per painting; one caption per row in load_artelingo() order (so rows align with the CLIP cache)."""
    ann = json.load(open(annotations_path))
    if sample_ids is None:
        from src.utils import FeatureManager
        sample_ids = np.asarray(FeatureManager(storage_dir=ARTELINGO_FEATURES).get_all_sample_ids())
    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    if expected is not None and len(sample_ids) != expected[1]:
        raise ValueError(f"expected {expected[1]} captions, got {len(sample_ids)}")
    img_row: dict = {}
    images, texts, tmeta = [], [], []
    for sid in sample_ids.tolist():
        r = ann[sid]
        p = r["painting"]
        if p not in img_row:
            img_row[p] = len(images)
            images.append({"painting": p, "image": r["image"]})
        texts.append(r["caption"])
        tmeta.append({"sample_id": sid, "painting": p, "img": img_row[p]})
    if expected is not None and len(images) != expected[0]:
        raise ValueError(f"expected {expected[0]} paintings, got {len(images)}")
    image_dir = Path(image_dir)
    return Spec(images, lambda i: _open_rgb(image_dir / images[i]["image"]), texts, tmeta)


def adapter_semart(semart_dir=SEMART_DIR, **_) -> Spec:
    import pandas as pd
    semart_dir = Path(semart_dir)
    images, texts, tmeta, n_changed = [], [], [], 0
    for split in ("train", "val", "test"):
        df = pd.read_csv(semart_dir / f"semart_{split}.csv", sep="\t", encoding="latin-1", dtype=str,
                         keep_default_na=False)
        for _, r in df.iterrows():
            desc = scrub_semart(r["DESCRIPTION"], r["AUTHOR"], r["TITLE"], r["DATE"])
            n_changed += int(desc != re.sub(r"\s+", " ", r["DESCRIPTION"]).strip())
            images.append({"image_file": r["IMAGE_FILE"], "split": split, "type": r["TYPE"],
                           "school": r["SCHOOL"], "timeframe": r["TIMEFRAME"]})
            texts.append(desc)
            tmeta.append({"img": len(images) - 1})
    return Spec(images, lambda i: _open_rgb(semart_dir / "Images" / images[i]["image_file"]), texts, tmeta,
                {"n_descriptions_changed_by_scrub": n_changed})


def adapter_coco_train2014(image_dir=COCO_TRAIN_IMAGES, captions_path=COCO_TRAIN_CAPTIONS, **_) -> Spec:
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise SystemExit(f"COCO train2014 images not found at {image_dir}; check the path before extracting")
    d = json.load(open(captions_path))
    files = {im["id"]: im["file_name"] for im in d["images"]}
    caps: dict = {}
    for a in sorted(d["annotations"], key=lambda a: (a["image_id"], a["id"])):
        caps.setdefault(a["image_id"], []).append((a["id"], a["caption"].strip()))
    ids = sorted(files)
    images, texts, tmeta = [], [], []
    for iid in ids:
        images.append({"image_id": iid, "file_name": files[iid]})
        for cid, c in caps.get(iid, []):
            texts.append(c)
            tmeta.append({"caption_id": cid, "img": len(images) - 1})
    return Spec(images, lambda i: _open_rgb(image_dir / images[i]["file_name"]), texts, tmeta)


def adapter_genecis_coco(json_path=GENECIS_COCO_JSON, image_root=COCO_ROOT / "images", store=GENECIS_COCO_STORE,
                         **_) -> Spec:
    """Same item list as the B/32 GeneCIS COCO store: text row j is genecis_coco.json[j] (sample_id j)."""
    rows = json.load(open(json_path))
    if store is not None:
        ids = np.load(Path(store) / "shards" / "shard_00000" / "sample_ids.npy")
        if len(ids) != len(rows) or not (ids == np.arange(len(rows))).all():
            raise ValueError("genecis_coco.json does not line up with the existing feature store")
    img_row: dict = {}
    images, texts, tmeta = [], [], []
    for j, r in enumerate(rows):
        if r["image"] not in img_row:
            img_row[r["image"]] = len(images)
            images.append({"image": r["image"], "image_id": r["image_id"]})
        texts.append(r["caption"])
        tmeta.append({"sample_id": j, "caption_id": r["caption_id"], "img": img_row[r["image"]]})
    image_root = Path(image_root)
    return Spec(images, lambda i: _open_rgb(image_root / images[i]["image"]), texts, tmeta)


def _expand2square(pil_img, background_color):
    """Copied from /project/genecis/datasets/vaw_dataset.py."""
    from PIL import Image
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


def load_cropped_image(image_dir, image_id, bbox, dilate=DILATION, pad_crop=PAD_CROP):
    """GeneCIS VAWDataset.load_cropped_image (/project/genecis/datasets/vaw_dataset.py), same arithmetic:
    dilated crop (left/top shifted by dilate*size, clipped to the image), padded to a square with black."""
    from PIL import Image
    im = Image.open(os.path.join(image_dir, f"{image_id}.jpg"))
    im_width, im_height = im.size
    width, height = bbox[2], bbox[3]
    if dilate:
        orig_left, orig_top = bbox[0], bbox[1]
        left, top = max(0, orig_left - dilate * width), max(0, orig_top - dilate * height)
        right, bottom = min(im_width, left + (1 + dilate) * width), min(im_height, top + (1 + dilate) * height)
    else:
        left, top = bbox[0], bbox[1]
        right, bottom = bbox[0] + width, bbox[1] + height
    im = im.crop((left, top, right, bottom))
    if pad_crop:
        im = _expand2square(im, (0,) if im.mode == "L" else (0, 0, 0))
    return im.convert("RGB")


def _crop_sort_key(c):
    iid, bbox = c
    return (0, int(iid), bbox) if str(iid).isdigit() else (1, str(iid), bbox)


def adapter_genecis_vg_crops(focus_path=GENECIS_FOCUS_ATTRIBUTE, image_dir=VG_IMAGES, **_) -> Spec:
    """Union of (image_id, bbox) crops over every role; only these two fields are kept (see module docstring)."""
    data = json.load(open(focus_path))
    crops = set()
    for item in data:
        for role in (item["reference"], item["target"], *item["gallery"]):
            crops.add((str(role["image_id"]), tuple(role["instance_bbox"])))
    del data
    ordered = sorted(crops, key=_crop_sort_key)
    images = [{"image_id": iid, "bbox": list(bb)} for iid, bb in ordered]
    return Spec(images, lambda i: load_cropped_image(image_dir, images[i]["image_id"], images[i]["bbox"]))


ADAPTERS = {"artelingo_full": adapter_artelingo_full, "semart": adapter_semart,
            "coco_train2014": adapter_coco_train2014, "genecis_coco": adapter_genecis_coco,
            "genecis_vg_crops": adapter_genecis_vg_crops}


def apply_limit(spec: Spec, limit: Optional[int]) -> Spec:
    """First `limit` images and the texts that belong to them (for artelingo, texts of the first images)."""
    if limit is None:
        return spec
    keep = [j for j, m in enumerate(spec.text_meta) if 0 <= m["img"] < limit]
    return Spec(spec.images[:limit], spec.load_image, [spec.texts[j] for j in keep],
                [spec.text_meta[j] for j in keep], spec.extra)


# ---------------------------------------------------------------- extraction
def count_truncated(enc, texts) -> int:
    """Texts that the encoder's tokenizer would cut: more than 77 CLIP tokens, or more than 512 tokens of the full
    Qwen prompt (chat template included, so the generation suffix is lost for these)."""
    if enc.name == CLIP:
        tok, limit, wrap = enc.p.tokenizer, 77, (lambda t: t)
    else:
        tok, limit, wrap = enc.p.tokenizer, 512, (lambda t: enc._prompt([{"type": "text", "text": t}]))
    n = 0
    for i in range(0, len(texts), 2048):
        ids = tok([wrap(t) for t in texts[i:i + 2048]], truncation=False, add_special_tokens=True)["input_ids"]
        n += sum(len(x) > limit for x in ids)
    return n


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def out_dir_for(dataset, backbone, smoke, root=PRE_EXTRACT):
    return Path(root) / ("_smoke" if smoke else "") / dataset / backbone


def extract(spec: Spec, enc, out: Path, overwrite=False, smoke=False, dataset="", log=print) -> dict:
    out = Path(out)
    index_path, prog_path = out / "index.json", out / "progress.json"
    if index_path.exists() and not (overwrite or smoke):
        raise SystemExit(f"{index_path} exists (finished output); pass --overwrite to redo it")
    if out.exists() and (overwrite or smoke):
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    n_img, n_txt, dim = len(spec.images), len(spec.texts), enc.dim
    from numpy.lib.format import open_memmap
    resume = prog_path.exists() and (out / "img.npy").exists() and (out / "txt.npy").exists()
    prog = json.load(open(prog_path)) if resume else dict(img_done=0, txt_done=0, n_truncated=0, img_seconds=0.0,
                                                          txt_seconds=0.0)
    if resume and (prog["n_img"], prog["n_txt"]) != (n_img, n_txt):
        raise SystemExit("progress.json belongs to a different item list; use --overwrite")
    prog.update(n_img=n_img, n_txt=n_txt)
    mode = "r+" if resume else "w+"
    img = open_memmap(out / "img.npy", mode=mode, dtype=np.float16, shape=(n_img, dim))
    txt = open_memmap(out / "txt.npy", mode=mode, dtype=np.float16, shape=(n_txt, dim))
    bs = BATCH[enc.name]

    def save():
        img.flush(); txt.flush()
        tmp = prog_path.with_suffix(".tmp")
        json.dump(prog, open(tmp, "w")); os.replace(tmp, prog_path)

    save()
    pool = ThreadPoolExecutor(8)

    def load_chunk(a, b):
        return list(pool.map(spec.load_image, range(a, b)))

    starts = list(range(prog["img_done"], n_img, IMG_CHUNK))
    fut = pool.submit(load_chunk, starts[0], min(starts[0] + IMG_CHUNK, n_img)) if starts else None
    for k, a in enumerate(starts):
        b = min(a + IMG_CHUNK, n_img)
        ims = fut.result()
        if k + 1 < len(starts):
            fut = pool.submit(load_chunk, starts[k + 1], min(starts[k + 1] + IMG_CHUNK, n_img))
        t = time.time()
        img[a:b] = enc.encode_images(ims, bs["img"]).astype(np.float16)
        prog["img_seconds"] += time.time() - t
        prog["img_done"] = b
        save()
        log(f"[{dataset}/{enc.name}] images {b}/{n_img}  {prog['img_done'] / max(prog['img_seconds'], 1e-9):.1f}/s")
    for a in range(prog["txt_done"], n_txt, TXT_CHUNK):
        b = min(a + TXT_CHUNK, n_txt)
        chunk = spec.texts[a:b]
        prog["n_truncated"] += count_truncated(enc, chunk)
        t = time.time()
        txt[a:b] = enc.encode_texts(chunk, bs["txt"]).astype(np.float16)
        prog["txt_seconds"] += time.time() - t
        prog["txt_done"] = b
        save()
        log(f"[{dataset}/{enc.name}] texts {b}/{n_txt}  {prog['txt_done'] / max(prog['txt_seconds'], 1e-9):.1f}/s")
    pool.shutdown()
    save()
    del img, txt
    index = {"dataset": dataset, "backbone": enc.name, "smoke": smoke, "dtype": "float16", "dim": dim,
             "n_images": n_img, "n_texts": n_txt, "n_texts_truncated": prog["n_truncated"],
             "truncation_limit": 77 if enc.name == CLIP else 512, "extra": spec.extra,
             "images": spec.images, "texts": spec.text_meta}
    with open(index_path, "w") as f:
        json.dump(index, f)
    prog["index_sha256"] = _sha256(index_path)
    json.dump(prog, open(prog_path, "w"))
    return prog


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--backbone", required=True, choices=[CLIP, QWEN])
    ap.add_argument("--limit", type=int, default=None, help="first N images (smoke; writes under _smoke/)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if a.backbone not in ALLOWED[a.dataset]:
        raise SystemExit(f"{a.dataset} is extracted with {ALLOWED[a.dataset]} only")
    spec = apply_limit(ADAPTERS[a.dataset](), a.limit)
    print(f"{a.dataset}: {len(spec.images)} images, {len(spec.texts)} texts", flush=True)
    from src.data.feature_extract import load_encoder
    enc = load_encoder(a.backbone, a.device)
    out = out_dir_for(a.dataset, a.backbone, smoke=a.limit is not None)
    t = time.time()
    prog = extract(spec, enc, out, overwrite=a.overwrite, smoke=a.limit is not None, dataset=a.dataset,
                   log=lambda s: print(s, flush=True))
    print(json.dumps({k: prog[k] for k in ("n_img", "n_txt", "n_truncated", "img_seconds", "txt_seconds",
                                           "index_sha256")}), flush=True)
    print(f"done {out} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
