"""Round 6 uint8 image cache for the fine-tuned CLIP baselines (DECISION_RULE.md of this folder: section 2 FT-LB and
FT-LoRA, section 8 item 3; contracts section 9): the held paintings' images, decoded exactly as
src/test/20261124_clip_lightweight_ft/ft_data._decode does for the existing artelingo_clip224 cache. That cache excludes
held rows by design and its builder refuses them, so this is a small copy of the decode and the build loop; the original
stays unchanged.

The cache is built on the CPU from a job folder's rows_manifest.npz: one entry per distinct neutral image name of the
chosen rows, read from <image dir>/<name> (the staging folder of r6_gpu_inputs.py holds symlinks to the WikiArt files;
on a node the folder is the synced images). Nothing here carries a WikiArt path, a style folder or a label.

Layout (<out>):
- images_uint8.npy   (N, 224, 224, 3) uint8; entry k is the k-th neutral name in sorted order.
- paintings.json     {"order": "neutral image name, sorted", "images": {name: k}}. (The existing cache is keyed by
  painting id and sorted by it; the manifest holds neither, so this one is keyed and sorted by the neutral name.)
- cache_record.json  SHA-256 of both files, counts, the processor settings, the normalisation check.

Run:  python r6_ft_cache.py --job-dir <job folder> --out <cache dir> --image-dir <staging> [--workers 8] [--limit N]
Prints counts and paths; no metric. Guards carry a `# guard:<name>` marker; the tests delete each on a copy.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_gpu_common as G  # noqa: E402  (imports neither src nor r6_common)
import r6_gpu_t12 as T  # noqa: E402

CLIP_NAME = "openai/clip-vit-base-patch32"
IMAGES_FILE, PAINTINGS_FILE, RECORD_FILE = "images_uint8.npy", "paintings.json", "cache_record.json"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_PROC = None


def sha256_file(path, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _processor():
    from transformers import CLIPImageProcessor
    return CLIPImageProcessor.from_pretrained(CLIP_NAME, do_normalize=False, do_rescale=False)


def decode(path) -> np.ndarray:
    """One image -> (224, 224, 3) uint8 via the CLIP processor's resize and centre crop (no normalisation); the same
    statements as ft_data._decode."""
    global _PROC
    from PIL import Image
    if _PROC is None:
        _PROC = _processor()
    with Image.open(path) as im:
        px = _PROC(images=im.convert("RGB"), return_tensors="np")["pixel_values"][0]  # (3,224,224) 0..255
    return np.clip(np.rint(px), 0, 255).astype(np.uint8).transpose(1, 2, 0)


def _decode_job(job):
    k, path = job
    return k, decode(path)


def normalisation_check(path) -> dict:
    """CLIP's own rescale and normalise applied to the cached uint8 reproduces the stock processor."""
    from PIL import Image
    from transformers import CLIPImageProcessor
    stock = CLIPImageProcessor.from_pretrained(CLIP_NAME)
    with Image.open(path) as im:
        ref = stock(images=im.convert("RGB"), return_tensors="np")["pixel_values"][0]
    u8 = decode(path).astype(np.float32).transpose(2, 0, 1) / 255.0
    mean = np.array(CLIP_MEAN, dtype=np.float32)[:, None, None]
    std = np.array(CLIP_STD, dtype=np.float32)[:, None, None]
    return {"image_name": Path(path).name, "max_abs_diff": float(np.abs((u8 - mean) / std - ref).max())}


def image_names(manifest, rows=None) -> list:
    """Sorted distinct neutral image names of ``rows`` (default: every manifest row)."""
    names = manifest.image_name if rows is None else manifest.image_name[manifest.index(rows)]
    return sorted(set(names.tolist()))


def build_cache(out_dir, names, image_dir, workers=8, verbose=True) -> dict:
    """Write the cache of the images <image_dir>/<name> for the sorted distinct ``names``; -> cache_record.
    Refuses an existing cache; the files appear only when whole (images via .part, then the json files)."""
    out_dir, image_dir = Path(out_dir), Path(image_dir)
    names = list(names)
    G._require(names == sorted(set(names)) and names, "names must be sorted, distinct and not empty")
    G._require(all(G.IMAGE_NAME.fullmatch(x) for x in names), "image names must be neutral (<20 hex>.<ext>)")  # guard:neutral
    G._require(not any((out_dir / f).exists() for f in (IMAGES_FILE, PAINTINGS_FILE, RECORD_FILE)),
               f"a cache already exists in {out_dir}")  # guard:no_overwrite
    missing = [x for x in names if not (image_dir / x).is_file()]
    G._require(not missing, f"{len(missing)} images missing under {image_dir}, e.g. {missing[:3]}")
    paths = [image_dir / x for x in names]
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(names)
    tmp = out_dir / (IMAGES_FILE + ".part")
    from numpy.lib.format import open_memmap
    mm = open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n, 224, 224, 3))
    t0, done = time.time(), 0
    jobs = list(enumerate(paths))
    if workers <= 1:
        results, pool = map(_decode_job, jobs), None
    else:
        pool = ProcessPoolExecutor(max_workers=min(int(workers), 8), mp_context=mp.get_context("spawn"))
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
    G.write_json(out_dir / PAINTINGS_FILE, {"order": "neutral image name, sorted",
                                            "images": {x: k for k, x in enumerate(names)}})
    record = {"n_images": n, "shape": [n, 224, 224, 3], "dtype": "uint8",
              "processor": {"name": CLIP_NAME, "do_normalize": False, "do_rescale": False,
                            "resize_and_center_crop": 224},
              "normalisation_check": normalisation_check(paths[0]), "built": G.amsterdam_now(),
              "sha256": {IMAGES_FILE: sha256_file(out_dir / IMAGES_FILE),
                         PAINTINGS_FILE: sha256_file(out_dir / PAINTINGS_FILE)}}
    G.write_json(out_dir / RECORD_FILE, record)
    return record


def load_cache(cache_dir, verify=True):
    """-> (images memmap (N, 224, 224, 3) uint8, {name: cache row}, record). verify checks both SHA-256s."""
    d = Path(cache_dir)
    record = json.loads((d / RECORD_FILE).read_text())
    if verify:
        for name, sha in record["sha256"].items():
            G._require(sha256_file(d / name) == sha, f"SHA-256 mismatch for {name} in {d}")  # guard:verify
    images = np.load(d / IMAGES_FILE, mmap_mode="r")
    index = json.loads((d / PAINTINGS_FILE).read_text())["images"]
    G._require(images.shape[0] == len(index) == record["n_images"], "cache index and image array disagree")
    return images, index, record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Round 6 uint8 image cache of a job folder's images (CPU); no metric")
    ap.add_argument("--job-dir", required=True, help="a job folder with rows_manifest.npz (and ft_rows.npz)")
    ap.add_argument("--out", required=True, help="the cache folder to create")
    ap.add_argument("--image-dir", required=True, help="folder of the images under their neutral names")
    ap.add_argument("--workers", type=int, default=8, help="decode processes (at most 8)")
    ap.add_argument("--limit", type=int, default=None, help="keep only the first N names (dry runs)")
    args = ap.parse_args(argv)
    job = Path(args.job_dir)
    manifest = G.Manifest(job / "rows_manifest.npz")
    rows = T.load_ft_rows(job / "ft_rows.npz", manifest) if (job / "ft_rows.npz").is_file() else None
    names = image_names(manifest, rows)
    if args.limit is not None:
        names = names[:args.limit]
    rec = build_cache(args.out, names, args.image_dir, args.workers)
    print(f"cache {args.out}: {rec['n_images']} images, sha256 {rec['sha256'][IMAGES_FILE][:12]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
