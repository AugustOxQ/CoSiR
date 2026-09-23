"""Cache CLIP ViT-B/32 projected patch tokens in Stage 1 painting order."""

import importlib.util
import json
import os

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


OUT_DIR = os.path.dirname(__file__)
BASE_PILOT_PATH = os.path.join(OUT_DIR, "run_percept_stage1_pilot.py")
TRAIN_JSON = "/data/PDD/artelingo/artelingo_train.json"
HELDOUT_JSON = "/data/PDD/artelingo/artelingo_val_test.json"
IMAGE_ROOT = "/data/PDD/wikiart_proj/wikiart"
CACHE_DIR = "/data/SSD2/pre_extract/artelingo_percept_patch_features"
TRAIN_CACHE_PATH = os.path.join(CACHE_DIR, "train_patch_features.pt")
HELDOUT_CACHE_PATH = os.path.join(CACHE_DIR, "heldout_patch_features.pt")
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 64


def load_module(module_name: str, path: str):
    """Load a standalone sibling script without entering its main block."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("percept_stage1_base_for_patch_extraction", BASE_PILOT_PATH)


def image_records_in_painting_order(json_path: str) -> tuple[list[str], list[str]]:
    """Return one image path per sorted painting, rejecting inconsistent rows."""
    with open(json_path) as source:
        records = json.load(source)
    by_painting = {}
    inconsistent = []
    for record in records:
        painting = record["painting"]
        image = record["image"]
        prior = by_painting.setdefault(painting, image)
        if prior != image:
            inconsistent.append(painting)
    if inconsistent:
        raise RuntimeError(
            f"{len(inconsistent)} paintings have inconsistent image fields in {json_path}: "
            f"{sorted(set(inconsistent))[:10]}"
        )
    paintings = sorted(by_painting)
    return paintings, [os.path.join(IMAGE_ROOT, by_painting[painting]) for painting in paintings]


def ensure_stage1_alignment(paintings: list[str], pipeline, split_name: str) -> None:
    """Assert that JSON ordering exactly matches cached Stage 1 feature ordering."""
    stage1_paintings, _, _, _ = pipeline.load_dedup_features()
    if paintings != stage1_paintings:
        raise RuntimeError(f"{split_name} JSON painting order does not match load_dedup_features() order.")


def cache_is_usable(cache_path: str, expected_count: int) -> bool:
    """Return whether a completed cache has the expected number of paintings."""
    if not os.path.exists(cache_path):
        return False
    cached = torch.load(cache_path, map_location="cpu")
    return isinstance(cached, torch.Tensor) and cached.shape[0] == expected_count


def validate_images(paintings: list[str], image_paths: list[str], log) -> None:
    """Find every unreadable image before extraction can create a partial cache."""
    failures = []
    for painting, image_path in zip(paintings, image_paths):
        try:
            with Image.open(image_path) as image:
                image.convert("RGB")
        except Exception as exc:
            log(f"Image load failure for painting={painting!r}, path={image_path}: {exc}")
            failures.append(f"{painting!r} ({image_path}): {exc}")
    if failures:
        raise RuntimeError(
            f"Refusing to write a partial patch cache; {len(failures)} image load failures:\n"
            + "\n".join(failures)
        )


def extract_split(split_name, json_path, cache_path, pipeline, device, processor, vision_model, visual_projection, log) -> None:
    """Extract one aligned split, skipping only a complete existing cache."""
    paintings, image_paths = image_records_in_painting_order(json_path)
    ensure_stage1_alignment(paintings, pipeline, split_name)
    if cache_is_usable(cache_path, len(paintings)):
        log(f"{split_name}: existing patch cache has {len(paintings):,} paintings; skipping re-extraction.")
        return
    log(f"{split_name}: validating {len(paintings):,} images before patch extraction (batch_size={BATCH_SIZE}).")
    validate_images(paintings, image_paths, log)
    total_batches = (len(paintings) + BATCH_SIZE - 1) // BATCH_SIZE
    features = []
    with torch.inference_mode():
        for start in range(0, len(paintings), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(paintings))
            images = []
            for image_path in image_paths[start:end]:
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt").to(device)
            projected_tokens = visual_projection(vision_model(**inputs).last_hidden_state)
            features.append(projected_tokens.cpu().to(torch.float32))
            batch_number = start // BATCH_SIZE + 1
            if batch_number % 20 == 0 or end == len(paintings):
                log(f"{split_name} patch extraction: batch {batch_number}/{total_batches}; {end:,}/{len(paintings):,} paintings (batch_size={BATCH_SIZE}).")
    patch_features = torch.cat(features, dim=0)
    if patch_features.shape[0] != len(paintings):
        raise RuntimeError(f"{split_name} patch feature count {patch_features.shape[0]} does not match the {len(paintings)} Stage 1 paintings.")
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(patch_features, cache_path)
    log(f"{split_name}: wrote float32 patch cache {tuple(patch_features.shape)} to {cache_path}.")


def main() -> None:
    pipeline = base.load_sibling_module("artelingo_run_pipeline_patch_train", base.PIPELINE_PATH)
    heldout_pipeline = base.load_sibling_module("artelingo_run_pipeline_patch_heldout", base.PIPELINE_PATH)
    heldout_pipeline.STORAGE_DIR = base.HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = HELDOUT_JSON
    pipeline.assert_extraction_complete()
    heldout_pipeline.assert_extraction_complete()
    log = pipeline.log
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Loading {MODEL_NAME} raw vision modules on {device}; extraction batch_size={BATCH_SIZE}.")
    backbone = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    vision_model = backbone.vision_model.to(device).eval()
    visual_projection = backbone.visual_projection.to(device).eval()
    extract_split("train", TRAIN_JSON, TRAIN_CACHE_PATH, pipeline, device, processor, vision_model, visual_projection, log)
    extract_split("held-out", HELDOUT_JSON, HELDOUT_CACHE_PATH, heldout_pipeline, device, processor, vision_model, visual_projection, log)


if __name__ == "__main__":
    main()
