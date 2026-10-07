"""Data for the lightweight CLIP fine-tuning comparator: split rows, captions and the uint8 image cache.

Held rows are never read: the cache covers the paintings of scorer_train, val and selection only.
Cache order: paintings sorted by painting id (string sort); cache row k is the k-th sorted painting.
CLI: python ft_data.py build-cache --out <dir> [--limit N] [--workers 8]
"""
import argparse
import multiprocessing as mp
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CLIP_NAME = "openai/clip-vit-base-patch32"
WIKIART_DIR = "/data/PDD/wikiart_proj/wikiart"
DEFAULT_CACHE_DIR = "/data/SSD2/pre_extract/artelingo_clip224"
IMAGES_FILE, PAINTINGS_FILE, RECORD_FILE = "images_uint8.npy", "paintings.json", "cache_record.json"
SPLIT_NAMES = ("scorer_train", "val", "selection")
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def split_index(data, splits=None) -> dict:
    """Feature-row indices of the three non-held splits.

    Returns {"scorer_train", "val", "selection": np.ndarray[int64] of feature rows (sorted)}. Held is excluded.
    `splits` (an object with scorer_train/val/selection/held arrays) defaults to
    `artelingo_splits(data)`, which asserts the known split sizes. Raises ValueError if the three splits
    overlap each other or the held rows.
    """
    if splits is None:
        from src.data.artelingo_splits import artelingo_splits
        splits = artelingo_splits(data)
    out = {k: np.sort(np.asarray(getattr(splits, k), dtype=np.int64)) for k in SPLIT_NAMES}
    held = np.asarray(splits.held, dtype=np.int64)
    allrows = np.concatenate(list(out.values()))
    if len(np.unique(allrows)) != len(allrows):
        raise ValueError("scorer_train, val and selection overlap")
    if np.intersect1d(allrows, held).size:
        raise ValueError("a non-held split contains held rows")
    return out


def captions(sample_ids, annotations) -> list:
    """Captions in feature-row order: `annotations[sample_ids[i]]["caption"]`.

    Returns list[str], one per entry of `sample_ids` (pass `data.sample_ids[rows]` for a row subset).
    """
    return [annotations[int(i)]["caption"] for i in np.asarray(sample_ids)]


def painting_table(data, rows):
    """The painting of each row.

    Returns (unique_paintings: np.ndarray of str, sorted; row_to_pos: np.ndarray[int64] with
    unique_paintings[row_to_pos[j]] == data.paintings[rows[j]]).
    """
    p = np.asarray(data.paintings)[np.asarray(rows, dtype=np.int64)]
    uniq, pos = np.unique(p, return_inverse=True)
    return uniq, pos.astype(np.int64)


def _sha256(path, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


_PROC = None


def _processor():
    from transformers import CLIPImageProcessor
    return CLIPImageProcessor.from_pretrained(CLIP_NAME, do_normalize=False, do_rescale=False)


def _decode(path) -> np.ndarray:
    """One image -> (224, 224, 3) uint8 via the CLIP processor's resize and centre crop (no normalisation)."""
    global _PROC
    from PIL import Image
    if _PROC is None:
        _PROC = _processor()
    with Image.open(path) as im:
        px = _PROC(images=im.convert("RGB"), return_tensors="np")["pixel_values"][0]  # (3,224,224) 0..255
    return np.clip(np.rint(px), 0, 255).astype(np.uint8).transpose(1, 2, 0)


def _decode_job(job):
    k, path = job
    return k, _decode(path)


def _normalisation_check(path) -> dict:
    """Applying CLIP's own rescale+normalise to the cached uint8 must reproduce the stock processor."""
    from PIL import Image
    from transformers import CLIPImageProcessor
    stock = CLIPImageProcessor.from_pretrained(CLIP_NAME)
    with Image.open(path) as im:
        ref = stock(images=im.convert("RGB"), return_tensors="np")["pixel_values"][0]
    u8 = _decode(path).astype(np.float32).transpose(2, 0, 1) / 255.0
    mean = np.array(CLIP_MEAN, dtype=np.float32)[:, None, None]
    std = np.array(CLIP_STD, dtype=np.float32)[:, None, None]
    got = (u8 - mean) / std
    return {"image": str(path), "max_abs_diff": float(np.abs(got - ref).max()),
            "mean_mean": list(map(float, stock.image_mean)), "mean_std": list(map(float, stock.image_std))}


def build_image_cache(out_dir, data, annotations, wikiart_dir=WIKIART_DIR, splits=None, workers=8,
                      limit=None, verbose=True) -> dict:
    """Write the uint8 image cache for the paintings of scorer_train, val and selection.

    Files in `out_dir`: images_uint8.npy (N, 224, 224, 3) uint8, paintings.json
    ({"order": ..., "paintings": {painting: {"index": cache row, "image": annotation image path}}}),
    cache_record.json (SHA-256s, counts, processor settings, normalisation check).
    Cache row k is the k-th painting in sorted painting-id order. `limit` keeps only the first N paintings
    (dry runs). Raises FileExistsError if the cache exists, ValueError if a used painting also has a held row
    or its rows disagree on the image. Returns the record dict.
    """
    out_dir = Path(out_dir)
    if (out_dir / IMAGES_FILE).exists() or (out_dir / RECORD_FILE).exists():
        raise FileExistsError(f"cache already exists in {out_dir}")
    if splits is None:
        from src.data.artelingo_splits import artelingo_splits
        splits = artelingo_splits(data)
    idx = split_index(data, splits=splits)
    rows = np.concatenate([idx[k] for k in SPLIT_NAMES])
    all_p = np.asarray(data.paintings)
    held_p = set(all_p[np.asarray(splits.held, dtype=np.int64)].tolist())
    uniq, _ = painting_table(data, rows)
    if held_p & set(uniq.tolist()):
        raise ValueError("a painting of the cache also has held rows")
    image_of = {}
    sids = np.asarray(data.sample_ids)
    for r in rows:
        p, im = str(all_p[r]), annotations[int(sids[r])]["image"]
        if image_of.setdefault(p, im) != im:
            raise ValueError(f"painting {p!r} has more than one image")
    order = [str(p) for p in uniq]
    if limit is not None:
        order = order[:int(limit)]
    n = len(order)
    paths = [Path(wikiart_dir) / image_of[p] for p in order]
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / (IMAGES_FILE + ".part")
    from numpy.lib.format import open_memmap
    mm = open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n, 224, 224, 3))
    t0, done = time.time(), 0
    jobs = list(enumerate(paths))
    if workers <= 1:
        results = map(_decode_job, jobs)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=min(int(workers), 8),
                                   mp_context=mp.get_context("spawn"))  # fork deadlocks after the big parent load
        results = pool.map(_decode_job, jobs, chunksize=16)
    try:
        for k, arr in results:
            mm[k] = arr
            done += 1
            if verbose and (done % 500 == 0 or done == n):
                print(f"[cache] {done}/{n} images, {time.time() - t0:.0f}s", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    mm.flush()
    del mm
    tmp.rename(out_dir / IMAGES_FILE)
    (out_dir / PAINTINGS_FILE).write_text(json.dumps(
        {"order": "painting id, sorted", "paintings": {p: {"index": k, "image": image_of[p]}
                                                       for k, p in enumerate(order)}}))
    chk = _normalisation_check(paths[0]) if n else None
    record = {"n_images": n, "shape": [n, 224, 224, 3], "dtype": "uint8", "limit": limit,
              "split_rows": {k: int(len(idx[k])) for k in SPLIT_NAMES}, "n_held_rows_excluded": int(len(splits.held)),
              "processor": {"name": CLIP_NAME, "do_normalize": False, "do_rescale": False,
                            "resize_and_center_crop": 224},
              "normalisation_check": chk, "wikiart_dir": str(wikiart_dir),
              "sha256": {IMAGES_FILE: _sha256(out_dir / IMAGES_FILE),
                         PAINTINGS_FILE: _sha256(out_dir / PAINTINGS_FILE)}}
    (out_dir / RECORD_FILE).write_text(json.dumps(record, indent=1))
    return record


def load_image_cache(cache_dir, verify=True):
    """Open the cache.

    Returns (images: np.memmap (N, 224, 224, 3) uint8 read-only, painting_index: dict painting -> cache row,
    record: dict). With verify=True the SHA-256s in cache_record.json are checked (raises ValueError on a
    mismatch); this reads the whole file once.
    """
    d = Path(cache_dir)
    record = json.loads((d / RECORD_FILE).read_text())
    if verify:
        for name, sha in record["sha256"].items():
            if _sha256(d / name) != sha:
                raise ValueError(f"SHA-256 mismatch for {name} in {d}")
    images = np.load(d / IMAGES_FILE, mmap_mode="r")
    index = {p: v["index"] for p, v in json.loads((d / PAINTINGS_FILE).read_text())["paintings"].items()}
    if images.shape[0] != len(index) or images.shape[0] != record["n_images"]:
        raise ValueError("cache index and image array disagree")
    return images, index, record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-cache")
    b.add_argument("--out", default=DEFAULT_CACHE_DIR)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--wikiart-dir", default=WIKIART_DIR)
    a = ap.parse_args(argv)
    from src.data.artelingo import ANNOTATIONS_PATH, load_artelingo
    print("[cache] loading features and annotations", flush=True)
    data = load_artelingo()
    annotations = json.loads(Path(ANNOTATIONS_PATH).read_text())
    rec = build_image_cache(a.out, data, annotations, a.wikiart_dir, workers=a.workers, limit=a.limit)
    print(json.dumps({k: rec[k] for k in ("n_images", "split_rows", "normalisation_check")}), flush=True)


if __name__ == "__main__":
    main()
