"""
Preprocess the COCO half of GeneCIS (tasks focus_object, change_object) into CoSiR's
paired image-caption format.

Builds
  1. /data/PDD/genecis/genecis_coco.json   positional annotation list, one row per
     (image, caption); FeatureManager sample_id == row index.
  2. /data/PDD/genecis/templates/*.json    byte-for-byte copies of the four GeneCIS templates
     (the attribute ones need Visual Genome 1.2 images, which are not available locally).
  3. /data/PDD/genecis/manifest.json       provenance, hashes, counts, gallery-slot semantics.
  4. CLIP ViT-B/32 FeatureManager store (img_features / txt_features, unnormalised), extracted
     exactly like the historical redcaps extract_features.py (raw HF submodules, no src.model).
  5. Post-extraction checks (asserted and printed).

Run:
  /root/miniconda3/envs/CoSiR/bin/python scripts/preprocess_genecis.py
Extraction is skipped if the store already has a metadata.json.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset.cosir_datamodule import FeatureExtractionDataset  # noqa: E402
from src.utils import FeatureManager  # noqa: E402

BACKBONE = "openai/clip-vit-base-patch32"
TASKS_ALL = ["focus_object", "change_object", "focus_attribute", "change_attribute"]
TASKS_COCO = ["focus_object", "change_object"]


def encode_img(model, images):
    out = model.vision_model(**images)
    return model.visual_projection(out.pooler_output)


def encode_txt(model, texts):
    out = model.text_model(**texts)
    return model.text_projection(out.pooler_output)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def build_rows(args):
    templates = {t: json.load(open(os.path.join(args.genecis_dir, f"{t}.json"))) for t in TASKS_COCO}
    ids = set()
    for tpl in templates.values():
        for e in tpl:
            ids.add(e["reference"]["val_image_id"])
            ids.add(e["target"]["val_image_id"])
            ids.update(g["val_image_id"] for g in e["gallery"])
    assert len(ids) == 3033, len(ids)

    caps = collections.defaultdict(list)
    for a in json.load(open(args.coco_captions))["annotations"]:
        if a["image_id"] in ids:
            caps[a["image_id"]].append(a)

    rows = []
    for cid in sorted(ids):
        rel = "val2014/COCO_val2014_%012d.jpg" % cid
        assert os.path.exists(os.path.join(args.img_root, rel)), rel
        cl = sorted(caps[cid], key=lambda a: a["id"])
        assert len(cl) >= 5, (cid, len(cl))
        for i, a in enumerate(cl):
            rows.append({"image": rel, "caption": a["caption"].strip(), "image_id": f"coco_{cid}",
                         "coco_id": cid, "caption_index": i, "caption_id": a["id"]})
    return rows, templates


def write_artifacts(args, rows, templates):
    os.makedirs(os.path.dirname(args.annot), exist_ok=True)
    with open(args.annot, "w") as f:
        json.dump(rows, f)
    tdir = os.path.join(os.path.dirname(args.annot), "templates")
    os.makedirs(tdir, exist_ok=True)
    hashes = {}
    for t in TASKS_ALL:
        dst = os.path.join(tdir, f"{t}.json")
        shutil.copyfile(os.path.join(args.genecis_dir, f"{t}.json"), dst)
        hashes[f"{t}.json"] = sha256(dst)
    commit = subprocess.check_output(["git", "-C", os.path.dirname(args.genecis_dir.rstrip("/")),
                                      "rev-parse", "HEAD"], text=True).strip()
    per_img = collections.Counter(r["coco_id"] for r in rows)
    manifest = {
        "source_repo": "https://github.com/facebookresearch/genecis",
        "source_commit": commit,
        "template_sha256": hashes,
        "n_rows": len(rows),
        "n_unique_images": len(per_img),
        "template_counts": {t: len(templates[t]) for t in TASKS_COCO},
        "captions_per_image_histogram": dict(sorted(collections.Counter(per_img.values()).items())),
        "image_root": args.img_root,
        "annotation_path": args.annot,
        "feature_store": args.storage,
        "backbone": BACKBONE,
        "gallery_slot_semantics": {
            "object_tasks": "gallery slots 0-8 = similar scene WITHOUT the condition object; "
                            "slots 9-13 = dissimilar scene WITH the condition object. The target is "
                            "separate and conventionally ranked first. Source: GeneCIS paper section 4 "
                            "and a COCO-instance check (condition present in 100% of slots 9-13 vs "
                            "10-20% of slots 0-8 for thing-class conditions).",
            "attribute_tasks": "focus_attribute / change_attribute need Visual Genome 1.2 images, "
                               "not available locally; templates copied for completeness only.",
        },
    }
    with open(os.path.join(os.path.dirname(args.annot), "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("Rows:", len(rows), "unique images:", len(per_img),
          "captions/image hist:", manifest["captions_per_image_histogram"])


def extract(args, device):
    if os.path.exists(os.path.join(args.storage, "metadata.json")):
        print(f"Store already exists at {args.storage}, skipping extraction.")
        return
    model = AutoModel.from_pretrained(BACKBONE).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(BACKBONE, use_fast=False)
    dataset = FeatureExtractionDataset(annotation_path=args.annot, image_path=args.img_root,
                                       processor=processor, ratio=1)
    print(f"Dataset: {len(dataset):,} samples")
    fm = FeatureManager(args.storage, shard_size=100_000, hdf5_compression=True, hdf5_compression_level=4)

    probe = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    img_in, txt_in, _ = next(iter(probe))
    img_in = img_in.to(device)
    txt_in = {k: v.to(device) for k, v in txt_in.items()}
    with torch.no_grad():
        img_e = encode_img(model, img_in)
        txt_e = encode_txt(model, txt_in)
    dims = {"img_features": tuple(img_e.shape[1:]), "txt_features": tuple(txt_e.shape[1:])}
    print("Feature dims:", dims)
    del img_in, txt_in, img_e, txt_e, probe

    fm.open_for_writing(len(dataset), dims, backbone_model=BACKBONE)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
    with torch.no_grad():
        for image_inputs, text_inputs, sample_ids in tqdm(loader, desc="Extracting"):
            sample_ids = [int(s) for s in sample_ids]
            image_inputs = image_inputs.to(device)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            fm.write_batch(encode_img(model, image_inputs), encode_txt(model, text_inputs),
                           sample_ids, img_full=None, txt_full=None)
            torch.cuda.empty_cache()
    fm.finalize_writing()


def check(args, rows, device):
    fm = FeatureManager(args.storage, shard_size=100_000, hdf5_compression=True, hdf5_compression_level=4)
    meta = json.load(open(os.path.join(args.storage, "metadata.json")))
    assert meta["total_samples"] == len(rows), (meta["total_samples"], len(rows))
    print("CHECK total_samples == len(rows):", meta["total_samples"])
    sids = fm.get_all_sample_ids()
    assert len(sids) == len(rows) and set(sids) == set(range(len(rows)))
    print("CHECK sample_ids == range(n): OK")

    d = fm.load_all_to_ram()
    ids = d["sample_ids"].numpy().astype(int)
    pos = {int(s): i for i, s in enumerate(ids)}
    img, txt = d["img_features"].float(), d["txt_features"].float()
    assert torch.isfinite(img).all() and torch.isfinite(txt).all()
    print("CHECK finite: OK", tuple(img.shape), tuple(txt.shape))

    rng = random.Random(0)
    by_img = collections.defaultdict(list)
    for sid, r in enumerate(rows):
        by_img[r["coco_id"]].append(sid)
    for cid in rng.sample(sorted(by_img), 3):
        f = torch.stack([img[pos[s]] for s in by_img[cid]])
        diff = (f - f[0]).abs().max().item()
        assert diff < 1e-4, (cid, diff)
        print(f"CHECK identical image feats coco {cid} ({len(f)} rows): max abs diff {diff:.2e}")

    model = AutoModel.from_pretrained(BACKBONE).to(device).eval()
    processor = AutoProcessor.from_pretrained(BACKBONE, use_fast=False)
    cos_all = []
    for sid in rng.sample(range(len(rows)), 8):
        r = rows[sid]
        im = Image.open(os.path.join(args.img_root, r["image"])).convert("RGB")
        ii = processor(images=im, return_tensors="pt").to(device)
        tt = processor(text=r["caption"], return_tensors="pt", padding="max_length", truncation=True).to(device)
        with torch.no_grad():
            ie = encode_img(model, dict(ii)).cpu().float()[0]
            te = encode_txt(model, dict(tt)).cpu().float()[0]
        ci = torch.cosine_similarity(ie, img[pos[sid]], dim=0).item()
        ct = torch.cosine_similarity(te, txt[pos[sid]], dim=0).item()
        assert ci > 0.9999 and ct > 0.9999, (sid, ci, ct)
        cos_all.append((sid, ci, ct))
        print(f"CHECK re-encode sample {sid}: img cos {ci:.6f} txt cos {ct:.6f}")
    print("metadata.json:", json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genecis-dir", default="/project/genecis/genecis")
    ap.add_argument("--coco-captions", default="/data/SSD/coco/annotations/captions_val2014.json")
    ap.add_argument("--img-root", default="/data/SSD/coco/images")
    ap.add_argument("--annot", default="/data/PDD/genecis/genecis_coco.json")
    ap.add_argument("--storage", default="/data/SSD2/pre_extract/genecis_coco/features")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows, templates = build_rows(args)
    write_artifacts(args, rows, templates)
    extract(args, device)
    check(args, rows, device)


if __name__ == "__main__":
    main()
