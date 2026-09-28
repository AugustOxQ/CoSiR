"""Extract row-aligned SigLIP image features for ArtELingo train."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor


ANNOTATIONS = Path("/data/PDD/artelingo/artelingo_train.json")
IMAGE_ROOT = Path("/data/PDD/wikiart_proj/wikiart")
FEATURE_DIR = Path(__file__).resolve().parent / "features"
EXPECTED_ROWS = 308_723
EXPECTED_UNIQUE_IMAGES = 61_402
IMAGE_MODEL_ID = "google/siglip-base-patch16-224"
FEATURE_DIM = 768


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class SiglipEncoder:
    def __init__(self, device: torch.device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(IMAGE_MODEL_ID)
        self.model = AutoModel.from_pretrained(IMAGE_MODEL_ID).to(device).eval()

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        if not torch.is_tensor(features):
            features = features.pooler_output
        return F.normalize(features.float(), dim=1).cpu().numpy().astype(np.float32)


def extract_images(
    rows: list[dict], encoder: SiglipEncoder, output: np.ndarray,
    batch_size: int, workers: int, expected_unique: int | None = None,
) -> None:
    paths = [row["image"] for row in rows]
    unique_paths = list(dict.fromkeys(paths))
    if expected_unique is not None and len(unique_paths) != expected_unique:
        raise ValueError(f"Expected {expected_unique:,} unique images, got {len(unique_paths):,}")
    path_indices = {path: index for index, path in enumerate(unique_paths)}
    inverse = np.fromiter((path_indices[path] for path in paths), dtype=np.int32, count=len(paths))
    unique_features = np.empty((len(unique_paths), FEATURE_DIM), dtype=np.float32)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(unique_paths), batch_size):
            end = min(start + batch_size, len(unique_paths))
            images = list(pool.map(open_rgb, [IMAGE_ROOT / path for path in unique_paths[start:end]]))
            try:
                unique_features[start:end] = encoder.encode(images)
            finally:
                for image in images:
                    image.close()
            if end % 4096 < batch_size or end == len(unique_paths):
                print(f"Images: {end:,}/{len(unique_paths):,} unique", flush=True)

    for start in range(0, len(rows), 8192):
        end = min(start + 8192, len(rows))
        output[start:end] = unique_features[inverse[start:end]]


def summarize(name: str, features: np.ndarray, expected_shape: tuple[int, int]) -> str:
    if features.shape != expected_shape or features.dtype != np.float32:
        raise ValueError(f"{name}: expected float32 {expected_shape}, got {features.shape} {features.dtype}")
    norm_min, norm_max, norm_sum, count = float("inf"), float("-inf"), 0.0, 0
    value_min, value_max = float("inf"), float("-inf")
    for start in range(0, len(features), 8192):
        batch = np.asarray(features[start : start + 8192])
        if not np.isfinite(batch).all():
            raise ValueError(f"{name}: non-finite value at or after row {start}")
        norms = np.linalg.norm(batch, axis=1)
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
        norm_sum += float(norms.sum(dtype=np.float64))
        count += len(norms)
        value_min = min(value_min, float(batch.min()))
        value_max = max(value_max, float(batch.max()))
    if norm_min < 0.9999 or norm_max > 1.0001:
        raise ValueError(f"{name}: row norms outside [0.9999, 1.0001]: {norm_min}, {norm_max}")
    return (
        f"{name}: shape={features.shape}, dtype=float32, finite=yes, "
        f"norm min/mean/max={norm_min:.7f}/{norm_sum / count:.7f}/{norm_max:.7f}, "
        f"value min/max={value_min:.7f}/{value_max:.7f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-rows", type=int, default=256)
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--image-workers", type=int, default=8)
    args = parser.parse_args()
    if min(args.smoke_rows, args.image_batch_size, args.image_workers) < 1:
        parser.error("row count, batch size, and worker count must all be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Local CUDA GPU is required for this full-scale extraction")
    with ANNOTATIONS.open() as file:
        rows = json.load(file)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} annotation rows, got {len(rows):,}")
    encoder = SiglipEncoder(torch.device("cuda"))

    smoke_rows = rows[:min(args.smoke_rows, len(rows))]
    smoke = np.empty((len(smoke_rows), FEATURE_DIM), dtype=np.float32)
    print(f"Running smoke test on {len(smoke_rows):,} rows", flush=True)
    extract_images(smoke_rows, encoder, smoke, args.image_batch_size, args.image_workers)
    print("SMOKE PASS: " + summarize("SigLIP", smoke, (len(smoke_rows), FEATURE_DIM)), flush=True)
    if args.smoke_only:
        return

    print(f"Smoke gate passed; extracting all {len(rows):,} rows", flush=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = FEATURE_DIR / "siglip_v_img.tmp.npy"
    started = perf_counter()
    features = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(len(rows), FEATURE_DIM)
    )
    extract_images(rows, encoder, features, args.image_batch_size, args.image_workers,
                   expected_unique=EXPECTED_UNIQUE_IMAGES)
    features.flush()
    print("FULL PASS: " + summarize("SigLIP", features, (len(rows), FEATURE_DIM)), flush=True)
    del features
    temporary.replace(FEATURE_DIR / "siglip_v_img.npy")
    print(f"Full extraction wall-clock: {perf_counter() - started:.3f} s", flush=True)


if __name__ == "__main__":
    main()
