"""Extract row-aligned DINOv2 image and e5 caption features for ArtELingo train."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer


ANNOTATIONS = Path("/data/PDD/artelingo/artelingo_train.json")
IMAGE_ROOT = Path("/data/PDD/wikiart_proj/wikiart")
FEATURE_DIR = Path(__file__).resolve().parent / "features"
EXPECTED_ROWS = 308_723
IMAGE_MODEL_ID = "facebook/dinov2-small"
TEXT_MODEL_ID = "intfloat/e5-base-v2"


def normalize(features: torch.Tensor) -> np.ndarray:
    """Match the reference encoder's float32 L2 normalization convention."""
    return F.normalize(features.float(), dim=1).cpu().numpy().astype(np.float32)


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class DinoEncoder:
    def __init__(self, device: torch.device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(IMAGE_MODEL_ID)
        self.model = AutoModel.from_pretrained(IMAGE_MODEL_ID).to(device).eval()

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return normalize(outputs.last_hidden_state[:, 0])


class E5Encoder:
    def __init__(self, device: torch.device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
        self.model = AutoModel.from_pretrained(TEXT_MODEL_ID).to(device).eval()

    @torch.inference_mode()
    def encode(self, captions: list[str]) -> np.ndarray:
        inputs = self.tokenizer(
            ["query: " + caption for caption in captions],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)
        hidden = self.model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return normalize(pooled)


def extract_images(
    rows: list[dict], encoder: DinoEncoder, output: np.ndarray, batch_size: int, workers: int
) -> None:
    paths = [row["image"] for row in rows]
    unique_paths = list(dict.fromkeys(paths))
    path_indices = {path: index for index, path in enumerate(unique_paths)}
    inverse = np.fromiter((path_indices[path] for path in paths), dtype=np.int32, count=len(paths))
    unique_features = np.empty((len(unique_paths), 384), dtype=np.float32)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(unique_paths), batch_size):
            end = min(start + batch_size, len(unique_paths))
            batch_paths = [IMAGE_ROOT / path for path in unique_paths[start:end]]
            images = list(pool.map(open_rgb, batch_paths))
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


def extract_texts(rows: list[dict], encoder: E5Encoder, output: np.ndarray, batch_size: int) -> None:
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        output[start:end] = encoder.encode([row["caption"] for row in rows[start:end]])
        if end % 8192 < batch_size or end == len(rows):
            print(f"Captions: {end:,}/{len(rows):,}", flush=True)


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


def run_smoke(rows: list[dict], image_encoder: DinoEncoder, text_encoder: E5Encoder,
              image_batch_size: int, text_batch_size: int, workers: int) -> None:
    image_features = np.empty((len(rows), 384), dtype=np.float32)
    text_features = np.empty((len(rows), 768), dtype=np.float32)
    extract_images(rows, image_encoder, image_features, image_batch_size, workers)
    extract_texts(rows, text_encoder, text_features, text_batch_size)
    print("SMOKE PASS: " + summarize("DINOv2", image_features, (len(rows), 384)), flush=True)
    print("SMOKE PASS: " + summarize("e5", text_features, (len(rows), 768)), flush=True)


def run_full(rows: list[dict], image_encoder: DinoEncoder, text_encoder: E5Encoder,
             image_batch_size: int, text_batch_size: int, workers: int) -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    image_temp = FEATURE_DIR / "dinov2_img.tmp.npy"
    text_temp = FEATURE_DIR / "e5_txt.tmp.npy"
    started = perf_counter()
    image_features = np.lib.format.open_memmap(image_temp, mode="w+", dtype=np.float32,
                                               shape=(len(rows), 384))
    text_features = np.lib.format.open_memmap(text_temp, mode="w+", dtype=np.float32,
                                              shape=(len(rows), 768))
    extract_images(rows, image_encoder, image_features, image_batch_size, workers)
    extract_texts(rows, text_encoder, text_features, text_batch_size)
    image_features.flush()
    text_features.flush()
    print("FULL PASS: " + summarize("DINOv2", image_features, (len(rows), 384)), flush=True)
    print("FULL PASS: " + summarize("e5", text_features, (len(rows), 768)), flush=True)
    del image_features, text_features
    image_temp.replace(FEATURE_DIR / "dinov2_img.npy")
    text_temp.replace(FEATURE_DIR / "e5_txt.npy")
    print(f"Full extraction wall-clock: {perf_counter() - started:.3f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-rows", type=int, default=256)
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--image-workers", type=int, default=8)
    args = parser.parse_args()
    if min(args.smoke_rows, args.image_batch_size, args.text_batch_size, args.image_workers) < 1:
        parser.error("row count, batch sizes, and worker count must all be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Local CUDA GPU is required for this full-scale extraction")
    with ANNOTATIONS.open() as file:
        rows = json.load(file)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} annotation rows, got {len(rows):,}")
    device = torch.device("cuda")
    image_encoder = DinoEncoder(device)
    text_encoder = E5Encoder(device)
    smoke_rows = rows[:min(args.smoke_rows, len(rows))]
    print(f"Running smoke test on {len(smoke_rows):,} rows", flush=True)
    run_smoke(smoke_rows, image_encoder, text_encoder,
              args.image_batch_size, args.text_batch_size, args.image_workers)
    if args.smoke_only:
        return
    print(f"Smoke gate passed; extracting all {len(rows):,} rows", flush=True)
    run_full(rows, image_encoder, text_encoder,
             args.image_batch_size, args.text_batch_size, args.image_workers)


if __name__ == "__main__":
    main()
