# CLIP Baselines Design

**Date:** 2026-05-26  
**Status:** Approved  

## Goal

Add a self-contained baseline script for comparing CoSiR against standard CLIP fine-tuning strategies. Supports four training modes across three datasets, logs the same 10 retrieval metrics as CoSiR to wandb, and is designed for multi-seed runs with minimal hyperparameter variation.

---

## File Structure

```
src/baseline/
  __init__.py          # empty
  train_baseline.py    # all logic lives here
```

No other files. All components (model, datasets, training loop, eval) are in `train_baseline.py`.

---

## CLI (argparse)

| Argument | Default | Notes |
|---|---|---|
| `--dataset` | required | `coco \| redcaps \| impressions` |
| `--mode` | required | `full \| last_blocks \| linear \| lora` |
| `--clip_model` | `openai/clip-vit-base-patch32` | Any HF CLIP-compatible model |
| `--num_blocks` | `2` | Number of last transformer layers to unfreeze (`last_blocks` mode) |
| `--lora_rank` | `16` | LoRA rank (`lora` mode) |
| `--lora_alpha` | `32` | LoRA alpha (`lora` mode) |
| `--epochs` | `30` | |
| `--batch_size` | `256` | |
| `--lr` | `1e-5` | |
| `--seed` | `42` | Run multiple seeds, report mean ± std |
| `--train_annotation_path` | required | |
| `--train_image_path` | required | |
| `--test_annotation_path` | required | |
| `--test_image_path` | required | |
| `--eval_interval` | `5` | Epochs between full eval runs |
| `--wandb_project` | `cosir_image` | |
| `--wandb_entity` | — | Optional |
| `--wandb_name` | — | Auto-generated from mode+dataset+seed if empty |
| `--output_dir` | `res/baseline/` | Checkpoint save dir |

---

## Model: `CLIPBaseline`

Single `nn.Module` wrapping HuggingFace `AutoModel`. Constructor takes `mode`, `num_blocks`, `lora_rank`, `lora_alpha` and configures which parameters are trainable.

### Trainable parameters per mode

| Mode | Trainable params |
|---|---|
| `full` | All backbone params |
| `last_blocks` | Last `num_blocks` layers of `vision_model.encoder.layers` + last `num_blocks` of `text_model.encoder.layers` + `visual_projection` + `text_projection` |
| `linear` | `visual_projection` + `text_projection` only; entire backbone frozen |
| `lora` | LoRA adapters (via `peft.get_peft_model`) on `q_proj`/`v_proj` in attention layers; rest frozen |

### Interface

```python
def encode_image(self, image_inputs) -> Tensor   # [B, D], unnormalized
def encode_text(self, text_inputs) -> Tensor     # [B, D], unnormalized
def forward(self, image_inputs, text_inputs) -> tuple[Tensor, Tensor]
```

No combiner, no label embeddings. Normalization happens inside the loss function.

---

## Datasets

### `AnnotationDataset` (training — COCO, RedCaps, Impressions)

Loads a JSON annotation file where each entry has `"image"` (filename) and `"caption"` (string or list of strings).

- `captions_per_image`: auto-detected from the first annotation — `1` if caption is a string, `len(caption)` if a list. Readable as `dataloader.dataset.captions_per_image`.
- `__getitem__` returns `(PIL image, caption_str)` where caption is always a single string (first element if list). Training uses one positive pair per image.
- A `collate_fn` runs `processor(images=..., text=..., padding="max_length", truncation=True, return_tensors="pt")` and returns a batch dict.

### Eval dataset

Import `CoSiRValidationDataset` directly from `src.dataset`. No duplication. It already handles multi-caption images and exposes `captions_per_image`.

---

## Loss

Symmetric InfoNCE (identical temperature to CoSiR):

```python
def clip_loss(img_emb, txt_emb, temperature=0.07):
    img = F.normalize(img_emb, dim=-1)
    txt = F.normalize(txt_emb, dim=-1)
    logits = img @ txt.T / temperature
    labels = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
```

---

## Training Loop

1. Single `AdamW` optimizer over all trainable params
2. Cosine LR schedule with 5% linear warmup (by total gradient steps)
3. Log `train/loss` and `train/lr` to wandb every step
4. Run `run_eval()` every `--eval_interval` epochs and log all metrics

---

## Evaluation: `run_eval()`

Local function (not imported from `src/`):

1. Set model to `eval()`, iterate over `CoSiRValidationDataset` dataloader
2. Collect `img_emb` and `txt_emb` tensors + build `text_to_image_map` / `image_to_text_map`
3. Call `RecallMetrics(EvaluationConfig(device=device, k_vals=[1,5,10])).compute_all_recalls(img_emb, txt_emb, t2i_map, i2t_map)`
4. Log the returned dict to wandb

**Imports from `src/`:**
- `from src.eval.metrics import RecallMetrics`
- `from src.eval.config import EvaluationConfig`
- `from src.dataset import CoSiRValidationDataset`

No oracle metrics, no condition predictor — raw retrieval recall only.

---

## What this does NOT include

- Feature caching / FeatureManager
- TrainableEmbeddingManager or label embeddings
- ExperimentManager
- Visualization (UMAP, clustering)
- Oracle or condition-conditioned metrics

These are CoSiR-specific. The baseline measures only standard I2T/T2I retrieval quality.
