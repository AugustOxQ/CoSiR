# CLIP Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained CLIP fine-tuning baseline script (`src/baseline/train_baseline.py`) supporting four training modes on three datasets, logging the same 10 retrieval metrics as CoSiR to wandb.

**Architecture:** A single Python file under `src/baseline/` with four components — `CLIPBaseline` model, `AnnotationDataset`, `clip_loss`, and `run_eval`. It imports only three symbols from the existing codebase (`RecallMetrics`, `EvaluationConfig`, `CoSiRValidationDataset`) and is otherwise independent of CoSiR's training machinery.

**Tech Stack:** PyTorch, HuggingFace `transformers` (AutoModel/AutoProcessor), `peft==0.19.1` (LoRA), `wandb`, existing `src.eval.metrics.RecallMetrics`.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/baseline/__init__.py` | Package marker (empty) |
| Create | `src/baseline/train_baseline.py` | Everything: model, dataset, loss, eval, train loop, argparse |
| Create | `src/test/20260526_baseline/test_baseline.py` | Unit tests for all components |

---

## Task 1: File Structure

**Files:**
- Create: `src/baseline/__init__.py`
- Create: `src/baseline/train_baseline.py` (skeleton)
- Create: `src/test/20260526_baseline/test_baseline.py` (skeleton)

- [ ] **Step 1: Create the package and skeleton files**

```bash
mkdir -p src/baseline src/test/20260526_baseline
touch src/baseline/__init__.py
```

- [ ] **Step 2: Write `train_baseline.py` skeleton (imports + stubs)**

Create `src/baseline/train_baseline.py` with:

```python
import argparse
import math
import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm
import wandb

from src.eval.metrics import RecallMetrics
from src.eval.config import EvaluationConfig
from src.dataset import CoSiRValidationDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Model ─────────────────────────────────────────────────────────────────────
class CLIPBaseline(nn.Module):
    pass  # Task 2


# ── Dataset ───────────────────────────────────────────────────────────────────
class AnnotationDataset(Dataset):
    pass  # Task 3


# ── Loss ──────────────────────────────────────────────────────────────────────
def clip_loss(img_emb: torch.Tensor, txt_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    pass  # Task 4


# ── Evaluation ────────────────────────────────────────────────────────────────
def run_eval(model, processor, val_loader, device, k_vals=(1, 5, 10)):
    pass  # Task 4


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    pass  # Task 5


def parse_args():
    pass  # Task 5


if __name__ == "__main__":
    train(parse_args())
```

- [ ] **Step 3: Write `test_baseline.py` skeleton**

Create `src/test/20260526_baseline/test_baseline.py` with:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import torch
import torch.nn.functional as F
import pytest
from PIL import Image
```

---

## Task 2: CLIPBaseline Model

**Files:**
- Modify: `src/baseline/train_baseline.py` — replace `CLIPBaseline` stub
- Modify: `src/test/20260526_baseline/test_baseline.py`

- [ ] **Step 1: Write the failing tests**

Add to `src/test/20260526_baseline/test_baseline.py`:

```python
from src.baseline.train_baseline import CLIPBaseline

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
    # Only visual_projection and text_projection weights/biases should be trainable
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


def test_forward_returns_correct_shapes():
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(CLIP_MODEL, use_fast=False)
    model = CLIPBaseline(CLIP_MODEL, mode="linear")
    model.eval()

    img = torch.zeros(2, 3, 224, 224)
    inputs = processor(
        images=[Image.fromarray(torch.zeros(224, 224, 3).byte().numpy())] * 2,
        text=["hello", "world"],
        return_tensors="pt", padding="max_length", truncation=True,
    )
    with torch.no_grad():
        img_emb, txt_emb = model(inputs)
    assert img_emb.shape == (2, 512)
    assert txt_emb.shape == (2, 512)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && \
  python -m pytest src/test/20260526_baseline/test_baseline.py -v 2>&1 | head -40
```

Expected: all tests fail with `TypeError` or `AttributeError` on the stub.

- [ ] **Step 3: Implement `CLIPBaseline` in `train_baseline.py`**

Replace the `CLIPBaseline` stub with:

```python
class CLIPBaseline(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        mode: str = "linear",
        num_blocks: int = 2,
        lora_rank: int = 16,
        lora_alpha: int = 32,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(clip_model_name)

        if mode == "full":
            pass  # all params trainable by default

        elif mode == "last_blocks":
            for p in self.backbone.parameters():
                p.requires_grad = False
            for layer in self.backbone.vision_model.encoder.layers[-num_blocks:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for layer in self.backbone.text_model.encoder.layers[-num_blocks:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self.backbone.visual_projection.parameters():
                p.requires_grad = True
            for p in self.backbone.text_projection.parameters():
                p.requires_grad = True

        elif mode == "linear":
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.backbone.visual_projection.parameters():
                p.requires_grad = True
            for p in self.backbone.text_projection.parameters():
                p.requires_grad = True

        elif mode == "lora":
            from peft import get_peft_model, LoraConfig
            for p in self.backbone.parameters():
                p.requires_grad = False
            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_cfg)

        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from full|last_blocks|linear|lora")

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone.vision_model(pixel_values=pixel_values)
        return self.backbone.visual_projection(out.pooler_output)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return self.backbone.text_projection(out.pooler_output)

    def forward(self, inputs: dict) -> tuple:
        img_emb = self.encode_image(inputs["pixel_values"])
        txt_emb = self.encode_text(inputs["input_ids"], inputs["attention_mask"])
        return img_emb, txt_emb

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && \
  python -m pytest src/test/20260526_baseline/test_baseline.py -v -k "model or forward or trainable or linear or lora or last_blocks or full" 2>&1 | tail -20
```

Expected: all 6 model tests pass.

---

## Task 3: AnnotationDataset

**Files:**
- Modify: `src/baseline/train_baseline.py` — replace `AnnotationDataset` stub
- Modify: `src/test/20260526_baseline/test_baseline.py`

- [ ] **Step 1: Write the failing tests**

Add to test file:

```python
import json
import tempfile
from pathlib import Path
from src.baseline.train_baseline import AnnotationDataset


def _make_fake_dataset(tmp_path, captions_per_image=1):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    annotations = []
    for i in range(4):
        img_path = img_dir / f"img{i}.jpg"
        # Create a tiny white JPEG
        from PIL import Image as PILImage
        PILImage.new("RGB", (32, 32), color=(255, 255, 255)).save(str(img_path))
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
    assert isinstance(cap, str)  # always returns one string


def test_annotation_dataset_ratio(tmp_path):
    ann_path, img_dir = _make_fake_dataset(tmp_path, captions_per_image=1)
    ds = AnnotationDataset(ann_path, img_dir, ratio=0.5)
    assert len(ds) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && \
  python -m pytest src/test/20260526_baseline/test_baseline.py -v -k "dataset" 2>&1 | tail -15
```

Expected: 3 failures on the stub.

- [ ] **Step 3: Implement `AnnotationDataset`**

Replace the stub:

```python
class AnnotationDataset(Dataset):
    def __init__(self, annotation_path: str, image_path: str, ratio: float = 1.0):
        with open(annotation_path) as f:
            annotations = json.load(f)
        n = max(1, int(len(annotations) * ratio))
        self.annotations = annotations[:n]
        self.image_path = image_path

        cap0 = self.annotations[0]["caption"]
        self.captions_per_image = 1 if isinstance(cap0, str) else len(cap0)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        ann = self.annotations[idx]
        img = Image.open(os.path.join(self.image_path, ann["image"])).convert("RGB")
        cap = ann["caption"] if isinstance(ann["caption"], str) else ann["caption"][0]
        return img, cap
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && \
  python -m pytest src/test/20260526_baseline/test_baseline.py -v -k "dataset" 2>&1 | tail -10
```

Expected: 3 dataset tests pass.

---

## Task 4: `clip_loss` and `run_eval`

**Files:**
- Modify: `src/baseline/train_baseline.py` — replace `clip_loss` and `run_eval` stubs
- Modify: `src/test/20260526_baseline/test_baseline.py`

- [ ] **Step 1: Write failing tests**

Add to test file:

```python
from src.baseline.train_baseline import clip_loss


def test_clip_loss_diagonal_lowest():
    """Loss should be near zero when embeddings are perfectly aligned."""
    B = 4
    emb = F.normalize(torch.randn(B, 512), dim=-1)
    # Perfect alignment: img == txt
    loss_perfect = clip_loss(emb, emb)
    # Random alignment
    loss_random = clip_loss(emb, F.normalize(torch.randn(B, 512), dim=-1))
    assert loss_perfect < loss_random


def test_clip_loss_is_symmetric():
    img = F.normalize(torch.randn(4, 512), dim=-1)
    txt = F.normalize(torch.randn(4, 512), dim=-1)
    loss_forward = clip_loss(img, txt)
    loss_backward = clip_loss(txt, img)
    # Symmetric loss (both directions averaged) should be equal
    assert abs(loss_forward.item() - loss_backward.item()) < 1e-5


def test_clip_loss_shape():
    img = torch.randn(8, 512)
    txt = torch.randn(8, 512)
    loss = clip_loss(img, txt)
    assert loss.shape == ()  # scalar
```

- [ ] **Step 2: Run to verify failures**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && \
  python -m pytest src/test/20260526_baseline/test_baseline.py -v -k "loss" 2>&1 | tail -15
```

- [ ] **Step 3: Implement `clip_loss`**

Replace the stub:

```python
def clip_loss(img_emb: torch.Tensor, txt_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    img = F.normalize(img_emb, dim=-1)
    txt = F.normalize(txt_emb, dim=-1)
    logits = img @ txt.T / temperature
    labels = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
```

- [ ] **Step 4: Run loss tests to verify they pass**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && \
  python -m pytest src/test/20260526_baseline/test_baseline.py -v -k "loss" 2>&1 | tail -10
```

Expected: 3 loss tests pass.

- [ ] **Step 5: Implement `run_eval`**

Replace the stub. This function iterates over a `CoSiRValidationDataset` dataloader, collects embeddings, builds the retrieval maps, and delegates metric computation to `RecallMetrics`.

```python
def run_eval(model, processor, val_loader, device, k_vals=(1, 5, 10)):
    """Compute I2T/T2I recall@K, mAP, meanR for the validation set."""
    model.eval()
    all_img, all_txt = [], []
    t2i_map_list, i2t_map_list = [], []
    text_idx = img_idx = 0
    cpi = getattr(val_loader.dataset, "captions_per_image", 1)

    with torch.no_grad():
        for images, raw_texts in tqdm(val_loader, desc="Eval"):
            B = images["pixel_values"].shape[0]

            # CoSiRValidationDataset collation: when cpi>1, raw_texts is a list of
            # cpi lists each of length B (default_collate transposes the list-of-lists)
            if cpi == 1:
                flat_texts = list(raw_texts)
            else:
                flat_texts = [raw_texts[i][b] for b in range(B) for i in range(cpi)]

            text_inputs = processor(
                text=flat_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            ).to(device)

            img_emb = model.encode_image(images["pixel_values"].to(device))
            txt_emb = model.encode_text(
                text_inputs["input_ids"], text_inputs["attention_mask"]
            )

            all_img.append(img_emb.cpu())
            all_txt.append(txt_emb.cpu())

            for _ in range(B):
                i2t_map_list.append(list(range(text_idx, text_idx + cpi)))
                t2i_map_list.extend([img_idx] * cpi)
                text_idx += cpi
                img_idx += 1

            del img_emb, txt_emb
            torch.cuda.empty_cache()

    img_emb = torch.cat(all_img)
    txt_emb = torch.cat(all_txt)
    t2i_map = torch.LongTensor(t2i_map_list)
    i2t_map = torch.LongTensor(i2t_map_list)

    eval_cfg = EvaluationConfig(device=device, k_vals=list(k_vals), print_metrics=True)
    return RecallMetrics(eval_cfg).compute_all_recalls(img_emb, txt_emb, t2i_map, i2t_map)
```

---

## Task 5: Training Loop, argparse, and `main()`

**Files:**
- Modify: `src/baseline/train_baseline.py` — replace `train` and `parse_args` stubs

- [ ] **Step 1: Implement `parse_args()`**

Replace the stub:

```python
def parse_args():
    p = argparse.ArgumentParser(description="CLIP baseline fine-tuning")

    # Required
    p.add_argument("--dataset", required=True, choices=["coco", "redcaps", "impressions"])
    p.add_argument("--mode", required=True, choices=["full", "last_blocks", "linear", "lora"])
    p.add_argument("--train_annotation_path", required=True)
    p.add_argument("--train_image_path", required=True)
    p.add_argument("--test_annotation_path", required=True)
    p.add_argument("--test_image_path", required=True)

    # Model
    p.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--ratio", type=float, default=1.0, help="Fraction of training data to use")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_interval", type=int, default=5)
    p.add_argument("--k_vals", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", default="res/baseline")

    # WandB
    p.add_argument("--wandb_project", default="cosir_image")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_name", default=None)
    p.add_argument("--wandb_tags", nargs="*", default=["baseline"])

    return p.parse_args()
```

- [ ] **Step 2: Implement `train()`**

Replace the stub:

```python
def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(args.clip_model, use_fast=False)

    model = CLIPBaseline(
        args.clip_model, args.mode, args.num_blocks, args.lora_rank, args.lora_alpha
    ).to(device)
    print(f"[{args.mode}] Trainable params: {model.trainable_param_count():,}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = AnnotationDataset(
        args.train_annotation_path, args.train_image_path, ratio=args.ratio
    )

    def collate_fn(batch):
        images, texts = zip(*batch)
        return processor(
            images=list(images),
            text=list(texts),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_dataset = CoSiRValidationDataset(
        args.test_annotation_path, args.test_image_path, processor, ratio=1.0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=256, shuffle=False, num_workers=args.num_workers
    )

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.05,
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── WandB ─────────────────────────────────────────────────────────────────
    run_name = args.wandb_name or f"{args.mode}_{args.dataset}_seed{args.seed}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        tags=args.wandb_tags,
        config=vars(args),
    )

    # ── Train Loop ────────────────────────────────────────────────────────────
    step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            img_emb, txt_emb = model(batch)
            loss = clip_loss(img_emb, txt_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            scheduler.step()

            wandb.log({
                "train/loss": loss.item(),
                "train/lr": scheduler.get_last_lr()[0],
                "epoch": epoch,
                "step": step,
            })
            epoch_loss += loss.item()
            step += 1

        print(f"Epoch {epoch + 1}: avg loss = {epoch_loss / len(train_loader):.4f}")

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            metrics = run_eval(model, processor, val_loader, device, args.k_vals)
            wandb.log({"epoch": epoch, **{f"eval/{k}": v for k, v in metrics.items()}})

    # ── Save Checkpoint ───────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"{run_name}.pt")
    # For LoRA models, save the peft adapter weights separately
    if args.mode == "lora":
        lora_dir = os.path.join(args.output_dir, run_name + "_lora")
        model.backbone.save_pretrained(lora_dir)
        print(f"Saved LoRA adapters to {lora_dir}")
    else:
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    wandb.finish()
```

- [ ] **Step 3: Run a smoke test (2 epochs, tiny subset)**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && \
  python src/baseline/train_baseline.py \
    --dataset impressions \
    --mode linear \
    --train_annotation_path /project/Impressions/metadata/impressions_train.json \
    --train_image_path /project/Impressions/media \
    --test_annotation_path /project/Impressions/metadata/impressions_test.json \
    --test_image_path /project/Impressions/media \
    --epochs 2 --ratio 0.01 --batch_size 32 --eval_interval 1 \
    --wandb_project cosir_image --wandb_name smoke_test_linear
```

Expected: runs 2 epochs, logs to wandb, prints eval metrics at epoch 2.

- [ ] **Step 4: Run full test suite**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && \
  python -m pytest src/test/20260526_baseline/test_baseline.py -v 2>&1 | tail -20
```

Expected: all tests pass.

---

## Final: How to run 3 modes × Impressions

```bash
for mode in last_blocks linear lora; do
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && \
  python src/baseline/train_baseline.py \
    --dataset impressions \
    --mode $mode \
    --train_annotation_path /project/Impressions/metadata/impressions_train.json \
    --train_image_path /project/Impressions/media \
    --test_annotation_path /project/Impressions/metadata/impressions_test.json \
    --test_image_path /project/Impressions/media \
    --wandb_project cosir_image \
    --wandb_entity augustoxq
done
```

To run with multiple seeds and different GPUs in parallel:
```bash
for mode in last_blocks linear lora; do
  for seed in 42 123 456; do
    CUDA_VISIBLE_DEVICES=0 python src/baseline/train_baseline.py \
      --dataset impressions --mode $mode --seed $seed \
      --train_annotation_path /project/Impressions/metadata/impressions_train.json \
      --train_image_path /project/Impressions/media \
      --test_annotation_path /project/Impressions/metadata/impressions_test.json \
      --test_image_path /project/Impressions/media \
      --wandb_project cosir_image --wandb_entity augustoxq &
  done
done
wait
```
