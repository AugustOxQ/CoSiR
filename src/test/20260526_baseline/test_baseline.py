import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import json
from pathlib import Path
import torch
import torch.nn.functional as F
import pytest
from PIL import Image

from src.baseline.train_baseline import CLIPBaseline, AnnotationDataset

CLIP_MODEL = "openai/clip-vit-base-patch32"


def _trainable_names(model):
    return {n for n, p in model.named_parameters() if p.requires_grad}


def test_full_mode_all_params_trainable():
    model = CLIPBaseline(CLIP_MODEL, mode="full")
    total = sum(1 for _ in model.parameters())
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert total == trainable
    assert model.trainable_param_count() > 0


def test_linear_mode_only_projections_trainable():
    model = CLIPBaseline(CLIP_MODEL, mode="linear")
    names = _trainable_names(model)
    assert all("projection" in n for n in names), f"Unexpected trainable params: {names}"
    assert model.trainable_param_count() > 0


def test_last_blocks_mode_trainable_count_between_linear_and_full():
    linear = CLIPBaseline(CLIP_MODEL, mode="linear").trainable_param_count()
    last2 = CLIPBaseline(CLIP_MODEL, mode="last_blocks", num_blocks=2).trainable_param_count()
    full = CLIPBaseline(CLIP_MODEL, mode="full").trainable_param_count()
    assert linear < last2 < full


def test_last_blocks_num_blocks_respected():
    m1 = CLIPBaseline(CLIP_MODEL, mode="last_blocks", num_blocks=1).trainable_param_count()
    m2 = CLIPBaseline(CLIP_MODEL, mode="last_blocks", num_blocks=2).trainable_param_count()
    assert m1 < m2


def test_lora_mode_has_lora_params():
    model = CLIPBaseline(CLIP_MODEL, mode="lora", lora_rank=4, lora_alpha=8)
    names = _trainable_names(model)
    assert any("lora_" in n for n in names), f"No lora params found. Trainable: {names}"


def test_lora_mode_projections_are_trainable():
    model = CLIPBaseline(CLIP_MODEL, mode="lora", lora_rank=4, lora_alpha=8)
    names = _trainable_names(model)
    assert any("visual_projection" in n for n in names), f"visual_projection not trainable in lora mode. Got: {names}"
    assert any("text_projection" in n for n in names), f"text_projection not trainable in lora mode. Got: {names}"


def test_forward_returns_correct_shapes():
    from transformers import AutoProcessor
    import numpy as np
    processor = AutoProcessor.from_pretrained(CLIP_MODEL, use_fast=False)
    model = CLIPBaseline(CLIP_MODEL, mode="linear")
    model.eval()

    inputs = processor(
        images=[Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))] * 2,
        text=["hello", "world"],
        return_tensors="pt", padding="max_length", truncation=True,
    )
    with torch.no_grad():
        img_emb, txt_emb = model(inputs)
    assert img_emb.shape == (2, 512)
    assert txt_emb.shape == (2, 512)


# ── AnnotationDataset Tests ───────────────────────────────────────────────────
def _make_fake_dataset(tmp_path, captions_per_image=1):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    annotations = []
    for i in range(4):
        img_path = img_dir / f"img{i}.jpg"
        Image.new("RGB", (32, 32), color=(255, 255, 255)).save(str(img_path))
        cap = f"caption {i}" if captions_per_image == 1 else [f"caption {i}_{j}" for j in range(captions_per_image)]
        annotations.append({"image": f"img{i}.jpg", "caption": cap})
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(json.dumps(annotations))
    return str(ann_path), str(img_dir)


def test_annotation_dataset_single_caption(tmp_path):
    ann_path, img_dir = _make_fake_dataset(tmp_path, captions_per_image=1)
    ds = AnnotationDataset(ann_path, img_dir)
    assert len(ds) == 4
    assert ds.captions_per_image == 1
    img, cap = ds[0]
    assert hasattr(img, "mode")  # PIL Image
    assert isinstance(cap, str)


def test_annotation_dataset_multi_caption(tmp_path):
    ann_path, img_dir = _make_fake_dataset(tmp_path, captions_per_image=5)
    ds = AnnotationDataset(ann_path, img_dir)
    assert ds.captions_per_image == 5
    img, cap = ds[0]
    assert isinstance(cap, str)  # always returns one string (first caption)


def test_annotation_dataset_ratio(tmp_path):
    ann_path, img_dir = _make_fake_dataset(tmp_path, captions_per_image=1)
    ds = AnnotationDataset(ann_path, img_dir, ratio=0.5)
    assert len(ds) == 2


# ── clip_loss Tests ───────────────────────────────────────────────────────────
from src.baseline.train_baseline import clip_loss


def test_clip_loss_perfect_alignment_lower_than_random():
    B = 4
    emb = F.normalize(torch.randn(B, 512), dim=-1)
    loss_perfect = clip_loss(emb, emb)
    loss_random = clip_loss(emb, F.normalize(torch.randn(B, 512), dim=-1))
    assert loss_perfect < loss_random


def test_clip_loss_is_symmetric():
    img = F.normalize(torch.randn(4, 512), dim=-1)
    txt = F.normalize(torch.randn(4, 512), dim=-1)
    loss_forward = clip_loss(img, txt)
    loss_backward = clip_loss(txt, img)
    assert abs(loss_forward.item() - loss_backward.item()) < 1e-5


def test_clip_loss_returns_scalar():
    img = torch.randn(8, 512)
    txt = torch.randn(8, 512)
    loss = clip_loss(img, txt)
    assert loss.shape == ()
