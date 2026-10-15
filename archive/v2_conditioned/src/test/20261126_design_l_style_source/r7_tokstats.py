"""CLIP token statistics (DECISION_RULE L5, L1/C6 and section 10; spec D12, D19).

Every invocation runs under flock -n -o -E 75 /tmp/gpu0.lock. The mandatory
256-image cache check precedes any output. Full extraction is for the controller:
  python r7_tokstats.py --check-only
  python r7_tokstats.py --limit 512 --out /path/to/trial
No labels, groupings or episode scores are computed here.
"""
import r7_common as C

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

STYLE = Path("/data/SSD2/pre_extract/artelingo/style_csd_vitl")
CACHE = Path("/data/SSD2/pre_extract/artelingo/features")
OUTPUT = Path("/data/SSD2/pre_extract/artelingo/clip_b32_tokstats")
MODEL_ID = "openai/clip-vit-base-patch32"
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
MODEL_PATH = Path("/data/SSD2/HF_home/hub/models--openai--clip-vit-base-patch32/snapshots") / REVISION
BLOCKS = (4, 6, 8)
N_IMAGES, N_ROWS, DIM = 61402, 308723, 4608
HOOK = C.MAIN / ".history/src/hook/train_cosir_20260421002916.py"


def sha_array(a):
    """run_step1.real_style_features convention: contiguous array bytes, not .npy."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read_inputs():
    meta_path = STYLE / "meta.json"
    assert C.sha_file(meta_path) == C.INPUTS[str(meta_path)], "C6 meta SHA mismatch"
    meta = json.loads(meta_path.read_text())
    assert meta["mode"] == "full", "C6 needs the full image list"
    assert meta["plan"]["sha256"] == "b00a4a7aa796957addc232dd92e1b1579306f8b312748a36ac1479331f8b0520", "C6 plan SHA"
    images = meta["images"]
    paths = np.asarray(images["image_paths"], dtype=object)
    assert len(paths) == len(set(paths)) == images["n_images_total"] == images["n_images_in_this_file"] == N_IMAGES, "image count/order"
    annotations_path = Path(images["annotations"])
    assert C.sha_file(annotations_path) == images["annotations_sha256"], "annotation SHA mismatch"
    ann = json.loads(annotations_path.read_text())
    assert isinstance(ann, list) and len(ann) == N_ROWS, "annotation rows"
    r2i = np.load(STYLE / "row_to_image.npy", allow_pickle=False)
    r2ia = np.load(STYLE / "row_to_image_annotation_order.npy", allow_pickle=False)
    for a, key in ((r2i, "row_to_image_sha256"), (r2ia, "row_to_image_annotation_order_sha256")):
        assert a.shape == (N_ROWS,) and a.dtype == np.int32, f"{key}: shape/dtype"
        assert a.min() == 0 and a.max() == N_IMAGES - 1, f"{key}: image indices"
        assert sha_array(a) == meta["row_mapping"][key], f"{key}: array SHA"
    ann_paths = np.asarray([a["image"] for a in ann], dtype=object)
    assert np.array_equal(paths[r2ia], ann_paths), "annotation image join"
    assert np.array_equal(paths, list(dict.fromkeys(ann_paths))), "first-appearance image order"
    cache_meta = json.loads((CACHE / "metadata.json").read_text())
    assert cache_meta["backbone_model"] == MODEL_ID, "cached backbone"
    assert cache_meta["total_samples"] == N_ROWS and cache_meta["feature_dims"]["img_features"] == [512], "cache dimensions"
    ids, features = [], []
    for k in range(cache_meta["num_shards"]):
        shard = CACHE / "shards" / f"shard_{k:05d}"
        sid = np.load(shard / "sample_ids.npy", mmap_mode="r", allow_pickle=False)
        img = np.load(shard / "img_features.npy", mmap_mode="r", allow_pickle=False)
        n = min(cache_meta["shard_size"], N_ROWS - k * cache_meta["shard_size"])
        assert sid.shape == (n,) and sid.dtype == np.int64, "shard sample IDs"
        assert img.shape == (n, 512) and img.dtype == np.float32, "shard image features"
        ids.append(sid)
        features.append(img)
    sids = np.concatenate(ids)
    assert sids.shape == (N_ROWS,) and np.array_equal(np.sort(sids), np.arange(N_ROWS)), "sample ID coverage"
    assert sha_array(sids) == meta["row_mapping"]["checks"]["sample_ids_sha256"], "sample ID SHA"
    assert np.array_equal(r2i, r2ia[sids]), "cache row to annotation join"
    assert np.array_equal(paths[r2i], ann_paths[sids]), "cache row to image path join"
    unique_images, first = np.unique(r2i, return_index=True)
    assert len(unique_images) == N_IMAGES, "every image used"
    _, paintings = np.unique(np.asarray([a["painting"] for a in ann], dtype=object), return_inverse=True)
    pairs = np.unique(np.stack((paintings[sids], r2i), axis=1), axis=0)
    assert len(pairs) == len(np.unique(pairs[:, 0])) == len(np.unique(pairs[:, 1])) == N_IMAGES, "painting/image bijection"
    # np.unique sorts image IDs: first[j] is the first cached row with r2i == j.
    reference = np.empty((256, 512), dtype=np.float32)
    offset = 0
    for img in features:
        selected = (first[:256] >= offset) & (first[:256] < offset + len(img))
        reference[selected] = img[first[:256][selected] - offset]
        offset += len(img)
    assert np.isfinite(reference).all() and (np.linalg.norm(reference, axis=1) > 0).all(), "cache reference values"
    path_sha = hashlib.sha256(json.dumps(paths.tolist(), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    joins = {
        "source_meta": str(meta_path), "source_meta_sha256": C.sha_file(meta_path),
        "row_to_image": str(STYLE / "row_to_image.npy"),
        "row_to_image_annotation_order": str(STYLE / "row_to_image_annotation_order.npy"),
        "row_to_image_sha256": sha_array(r2i),
        "row_to_image_annotation_order_sha256": sha_array(r2ia),
        "sample_ids_sha256": sha_array(sids), "image_paths_sha256": path_sha,
        "image_paths_sha256_encoding": "UTF-8 JSON, ensure_ascii=False, separators=(',', ':')",
        "cache_metadata_sha256": C.sha_file(CACHE / "metadata.json"),
        "checks": {"every_image_used": True, "painting_to_image_one_to_one": True,
                   "row_to_image_equals_annotation_map_at_sample_ids": True},
    }
    return images, reference, joins


def load_model():
    import torch
    from transformers import CLIPImageProcessorPil, CLIPModel

    assert torch.cuda.is_available(), "GPU required"
    # The old hook's CUDA float32 cache uses TF32 kernels. Disabling these
    # misses L5's cosine gate on image 250; enabling them matches every image.
    # Inputs, weights, activations and saved statistics remain torch.float32.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Transformers 5's CLIPImageProcessor uses torchvision. Select PIL explicitly
    # to preserve AutoProcessor(..., use_fast=False) from the old training hook.
    processor = CLIPImageProcessorPil.from_pretrained(str(MODEL_PATH), local_files_only=True)
    model = CLIPModel.from_pretrained(str(MODEL_PATH), local_files_only=True,
                                      dtype=torch.float32, attn_implementation="eager").eval().to("cuda:0")
    cfg = model.config.vision_config
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.patch_size, cfg.image_size) == (12, 768, 32, 224), "CLIP B/32 architecture"
    assert all(p.dtype == torch.float32 for p in model.parameters()), "float32 weights"
    return model, processor


def batches(images, n, batch_size, processor):
    import torch
    from PIL import Image

    root = Path(images["root"])
    for start in range(0, n, batch_size):
        pixels = []
        for path in images["image_paths"][start:start + batch_size]:
            with Image.open(root / path) as raw:
                pixels.append(processor(images=raw.convert("RGB"), return_tensors="pt")["pixel_values"][0])
        px = torch.stack(pixels).to(device="cuda:0", dtype=torch.float32)
        assert px.shape == (len(pixels), 3, 224, 224) and torch.isfinite(px).all(), "preprocessed pixels"
        yield start, px


def check_cache(model, processor, images, reference, batch_size):
    import torch

    encoded = []
    with torch.inference_mode():
        for _, px in batches(images, 256, batch_size, processor):
            vision = model.vision_model(pixel_values=px)
            encoded.append(model.visual_projection(vision.pooler_output).cpu().numpy())
    fresh = np.concatenate(encoded).astype(np.float64)
    cached = reference.astype(np.float64)
    cos = np.sum(fresh * cached, axis=1) / (np.linalg.norm(fresh, axis=1) * np.linalg.norm(cached, axis=1))
    rec = {"n_images": 256, "threshold": 0.9999, "min_cosine": float(cos.min()),
           "median_cosine": float(np.median(cos)), "cosines": cos.tolist(),
           "cache_row": "first row with row_to_image == image index"}
    print(f"cache check: min={rec['min_cosine']:.10f}, median={rec['median_cosine']:.10f}", flush=True)
    if not (np.isfinite(cos).all() and (cos >= 0.9999).all()):
        raise SystemExit("L5 cache check failed; writing nothing")
    return rec


def token_stats(hidden_states):
    import torch

    assert len(hidden_states) == 13, "embedding plus 12 encoder blocks"
    parts, dropped = [], []
    for block in BLOCKS:
        tokens = hidden_states[block][:, 1:, :]
        assert tokens.shape[1:] == (49, 768) and tokens.dtype == torch.float32, "49 patch tokens, float32"
        assert torch.isfinite(tokens).all(), "finite hidden states"
        norms = torch.linalg.vector_norm(tokens, dim=-1)
        keep = norms <= 3 * norms.median(dim=1, keepdim=True).values
        count = keep.sum(dim=1, keepdim=True)
        assert (count > 0).all(), "at least one retained patch"
        mask = keep.unsqueeze(-1)
        mean = (tokens * mask).sum(dim=1) / count
        variance = ((tokens - mean[:, None, :]).square() * mask).sum(dim=1) / count
        parts.extend((mean, variance.sqrt()))
        dropped.append(int((~keep).sum().item()))
    result = torch.cat(parts, dim=1).cpu().numpy()
    assert result.shape[1] == DIM and result.dtype == np.float32 and np.isfinite(result).all(), "token statistics output"
    return result, np.asarray(dropped, dtype=np.int64)


def extract(model, processor, images, joins, check, n, out, batch_size):
    import torch
    import transformers

    # Exclusive creation of the directory reserves this write-once pair. A failed
    # extraction remains visibly incomplete (no meta.json), never overwritten.
    out.mkdir(parents=True, exist_ok=False)
    target = out / "tokstats.npy"
    arr = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32, shape=(n, DIM))
    dropped = np.zeros(3, dtype=np.int64)
    with torch.inference_mode():
        for start, px in batches(images, n, batch_size, processor):
            vision = model.vision_model(pixel_values=px, output_hidden_states=True)
            values, counts = token_stats(vision.hidden_states)
            arr[start:start + len(values)] = values
            dropped += counts
            if start % (batch_size * 16) == 0:
                print(f"extracted {start + len(values)}/{n}", flush=True)
    arr.flush()
    del arr
    rec = {
        "mode": "full" if n == N_IMAGES else "trial", "kind": "CLIP B/32 token statistics",
        "images": {**images, "n_images_in_this_file": n, "image_paths": images["image_paths"][:n]},
        "row_mapping": joins, "model": {"id": MODEL_ID, "revision": REVISION, "path": str(MODEL_PATH),
                                         "weights_sha256": C.sha_file(MODEL_PATH / "pytorch_model.bin")},
        "preprocessing": {"PIL": "convert('RGB')", "processor": "CLIPImageProcessorPil (legacy use_fast=False)",
                          "config": processor.to_dict(), "precision": "float32, no autocast, TF32 kernels enabled (cache-compatible)",
                          "training_hook": str(HOOK), "training_hook_sha256": C.sha_file(HOOK)},
        "script": {"path": str(Path(__file__).resolve()), "sha256": C.sha_file(__file__)},
        "cache_check": check, "blocks": list(BLOCKS), "patch_tokens": 49,
        "statistics": "drop norm > 3 * median; per-channel mean then population std (ddof=0)",
        "dropped_tokens_per_block": dropped.tolist(), "dropped_share_per_block": (dropped / (n * 49)).tolist(),
        "dropped_share_n_images": n, "batch_size": batch_size,
        "environment": {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__},
        "output": {"tokstats.npy": {"shape": [n, DIM], "dtype": "float32", "sha256": C.sha_file(target)}},
    }
    C.write_json_once(out / "meta.json", rec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--_locked", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args._locked:
        raise SystemExit(subprocess.run(["flock", "-n", "-o", "-E", "75", "/tmp/gpu0.lock",
                                        sys.executable, str(Path(__file__).resolve()), "--_locked", *sys.argv[1:]]).returncode)
    C.assert_rule()
    C.assert_threads()
    assert 1 <= args.batch_size <= 64, "batch size must fit 24 GB (1..64)"
    if args.limit is not None:
        assert 1 <= args.limit < N_IMAGES and args.out is not None, "trial requires --limit 1..61401 --out DIR"
        assert args.out.resolve() != OUTPUT.resolve(), "trial must not use the full output directory"
    out = args.out or OUTPUT
    if not args.check_only:
        assert not out.exists(), f"{out} exists; use a fresh directory (write once)"
    images, reference, joins = read_inputs()
    model, processor = load_model()
    check = check_cache(model, processor, images, reference, args.batch_size)
    if not args.check_only:
        extract(model, processor, images, joins, check, args.limit or N_IMAGES, out, args.batch_size)


if __name__ == "__main__":
    main()
