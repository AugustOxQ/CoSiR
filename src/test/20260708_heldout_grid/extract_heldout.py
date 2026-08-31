"""
Extract held-out encoder features for the buddy generalization grid.

Reuses each dataset's existing `load_data()` (row order + positional join already
solved) so features align to the buddy-graph row order. Images or captions are
embedded in row order and cached per model. Existing DINOv2 caches are reused.

Usage:
  python extract_heldout.py --dataset impressions --model siglip_v
  python extract_heldout.py --dataset redcaps --model e5 --smoke 64

Design: docs/superpowers/specs/2026-07-08-heldout-grid-design.md
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src", "test", "20260622_buddy_analysis"))
sys.path.insert(0, os.path.join(ROOT, "src", "test", "20260623_redcaps_buddy"))

from heldout_models import MODELS, HeldoutEncoder

# dataset -> (module_name, existing DINOv2 cache to reuse for the full run)
DATASETS = {
    "impressions": ("buddy_analysis",
                    os.path.join(ROOT, "src/test/20260622_buddy_analysis/dino_feats.npy")),
    "redcaps": ("redcaps_buddy",
                os.path.join(ROOT, "src/test/20260623_redcaps_buddy/dino_feats.npy")),
}

CACHE_DIR = os.path.join(HERE, "heldout_feats")


def _mod(dataset):
    return __import__(DATASETS[dataset][0])


def cache_path(dataset, model, smoke):
    d = os.path.join(CACHE_DIR, dataset)
    os.makedirs(d, exist_ok=True)
    stem = f"smoke{smoke}_{model}" if smoke else model
    return os.path.join(d, f"{stem}.npy")


def _embed_images(enc, paths, batch):
    n = len(paths)
    feats = np.zeros((n, enc.dim), dtype=np.float32)
    missing = 0
    for start in range(0, n, batch):
        end = min(start + batch, n)
        imgs, idx = [], []
        for i in range(start, end):
            try:
                imgs.append(Image.open(paths[i]).convert("RGB"))
                idx.append(i)
            except (FileNotFoundError, OSError):
                missing += 1
        if imgs:
            feats[idx] = enc.encode_images(imgs)
        if start % (batch * 20) == 0:
            print(f"  {end}/{n}", flush=True)
    return feats, missing


def _embed_texts(enc, texts, batch):
    n = len(texts)
    feats = np.zeros((n, enc.dim), dtype=np.float32)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        feats[start:end] = enc.encode_texts(texts[start:end])
        if start % (batch * 40) == 0:
            print(f"  {end}/{n}", flush=True)
    return feats, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--smoke", type=int, default=0, help="embed only first N rows")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = cache_path(args.dataset, args.model, args.smoke)
    if os.path.exists(out) and not args.force:
        f = np.load(out)
        print(f"cache exists: {out}  shape={f.shape} (use --force to redo)")
        return

    mod = _mod(args.dataset)
    data = mod.load_data()
    records = data.records
    if args.smoke:
        records = records[: args.smoke]
    cfg = MODELS[args.model]

    # DINOv2 full-run reuse: slice the existing cache instead of re-extracting.
    dino_cache = DATASETS[args.dataset][1]
    if args.model == "dinov2" and not args.smoke and os.path.exists(dino_cache):
        f = np.load(dino_cache)
        assert f.shape[0] == data.n, f"dino cache {f.shape} != n={data.n}"
        np.save(out, f.astype(np.float32))
        print(f"reused DINOv2 cache -> {out}  shape={f.shape}")
        return

    enc = HeldoutEncoder(args.model)
    if cfg["modality"] == "image":
        paths = [os.path.join(mod.IMG_ROOT, r["image"]) for r in records]
        feats, missing = _embed_images(enc, paths, args.batch)
    else:
        texts = [r["caption"] for r in records]
        feats, missing = _embed_texts(enc, texts, args.batch)

    np.save(out, feats)
    zero = int((np.abs(feats).sum(1) == 0).sum())
    print(f"saved {out}  shape={feats.shape}  missing_files={missing}  zero_rows={zero}")


if __name__ == "__main__":
    main()
