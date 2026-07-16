# Held-out encoder grid — do buddies generalize across vision *and* language models?

**Date:** 2026-07-08
**Datasets:** Impressions (12,123) and RedCaps-150k
**Builds on:** `docs/reports/2026-06-22_buddy_analysis.md` (Impressions) and
`docs/reports/2026-06-23_redcaps_buddy.md` (RedCaps), which each ran a *single*
held-out encoder (DINOv2-small, vision only).

## Motivation

The buddy graphs B (= A_img ∩ A_txt) and E (= A_img ∪ A_txt) are built from CLIP
ViT-B/32 features (both towers). The standing worry is that buddies are a **CLIP
artifact**. The existing held-out test refutes this on the vision side with one
encoder (DINOv2): cross-content buddy pairs are closer than random pairs in a
representation the graph never saw.

This spec **widens that test into a 2-axis grid** — 3 independent vision encoders
(image space) and 3 independent language encoders (text space), on both datasets,
for both graphs. If buddy < random distance holds across *every* held-out encoder,
none of which built the graph, buddy meaningfulness generalizes beyond CLIP and
beyond a single modality.

## Test design (decided)

**Within-modality, two independent axes.** Each held-out model scores buddies in
its *own* modality — vision models embed images, language models embed captions —
and we compare mean cosine **distance** of buddy edges vs random pairs. This is a
direct generalization of the DINOv2 test, not a cross-modal alignment test (image
and text embeddings from unpaired models are not comparable). Any model qualifies;
towers need not be paired.

The scoring reuses each dataset's existing `heldout_distance_test`:
- **Impressions** keeps the within-photo / cross-photo split (source_id), since 814
  source photos back the 12,123 records. The honest number is **cross-photo**.
- **RedCaps** is genuinely 1:1, so every buddy edge is cross-content; a single
  buddy-vs-random number per model.

## Model slate

All load via plain `transformers` `AutoModel` / `AutoProcessor` / `AutoTokenizer`
(no `sentence-transformers` dependency, which is not installed). Chosen to span
distinct training paradigms so agreement is not a within-family artifact.

### Vision axis (image → embedding), held out from CLIP ViT-B/32

| key | HF id | paradigm | pooling | dim |
|---|---|---|---|---|
| `dinov2` | facebook/dinov2-small | self-supervised | CLS token | 384 |
| `siglip_v` | google/siglip-base-patch16-224 | sigmoid VLM | `get_image_features` | 768 |
| `vit_sup` | google/vit-base-patch16-224 | ImageNet supervised | CLS (pooler) | 768 |

### Language axis (caption → embedding), held out from CLIP text tower

| key | HF id | pooling | prefix | dim |
|---|---|---|---|---|
| `minilm` | sentence-transformers/all-MiniLM-L6-v2 | mean | — | 384 |
| `bge` | BAAI/bge-base-en-v1.5 | CLS | — | 768 |
| `e5` | intfloat/e5-base-v2 | mean | `"query: "` both sides | 768 |

All outputs are L2-normalized before scoring.

## Components (`src/test/20260708_heldout_grid/`)

1. **`heldout_models.py`** — a `MODELS` registry mapping key → spec (hf id,
   modality, loader, pooling, optional prefix). One `HeldoutEncoder` class exposing
   `encode_images(list[PIL]) -> (n,d)` **or** `encode_texts(list[str]) -> (n,d)`
   (whichever matches its modality), batched, `torch.no_grad`, returns L2-normalized
   `float32` numpy. Pooling/prefix differences live entirely in the registry.

2. **`extract_heldout.py --dataset {impressions,redcaps} --model KEY [--smoke N]
   [--batch B]`** — imports the **existing** `load_data()` from the dataset's buddy
   module (row order + positional join already solved) to get records in feature-row
   order. For a vision model, opens `IMG_ROOT/record["image"]`; for a text model,
   reads `record["caption"]`. Embeds in row order and caches to
   `heldout_feats/<dataset>/<model>.npy` (shape `(N,d)`, zero rows for missing
   images). **Skips extraction if the cache already exists.** DINOv2 caches already
   exist for both datasets and are reused verbatim (symlinked/copied into the cache
   dir on first run). `--smoke N` embeds only the first N rows and writes to
   `heldout_feats/<dataset>/smoke_<model>.npy`.

3. **`score_grid.py --dataset {impressions,redcaps} [--smoke N]`** — builds B and E
   once via the existing `build_graphs`, loads every cached held-out matrix for that
   dataset, and runs the dataset's own `heldout_distance_test` per (graph, model).
   Emits `docs/reports/assets/heldout_grid/<dataset>_grid.json` and a heatmap figure
   `<dataset>_grid.png` (rows = models grouped by modality; columns = B / E; cell =
   buddy_cross vs random distance and their ratio). `--smoke N` scores the
   `smoke_<model>.npy` caches on the first-N subgraph (pipeline sanity only).

4. **`docs/reports/2026-07-08_heldout_grid.md`** — the maintained report: motivation,
   model slate, method, a **Smoke** section (filled now), and per-dataset result
   tables + verdict (filled when the full extraction runs after the GPU frees up).

## Data flow

```
load_data() [existing, per dataset]  ->  records in feature-row order
        |                                     |
        v (images)                            v (captions)
  IMG_ROOT/record["image"]             record["caption"]
        \                                     /
         ->  HeldoutEncoder.encode_*  ->  heldout_feats/<dataset>/<model>.npy
                                              |
build_graphs() [existing] -> B, E             |
        \                                     /
         ->  heldout_distance_test  ->  <dataset>_grid.{json,png}  ->  report
```

## Compute & sequencing

- Heavy step = images × new vision models. RedCaps is 150k × **2** new vision passes
  (siglip_v, vit_sup; DINOv2 reused). Text is cheap (150k captions, seconds each).
  Impressions is trivial. Extraction is the only GPU-bound step; scoring is CPU.
- The **GPU is busy shortly**, so: build all code now, run a CPU-friendly
  `--smoke 64` on both datasets to prove the path end-to-end, and defer the full
  150k image passes. The report carries the smoke result and result-table
  placeholders until then.

## Smoke test (acceptance for this session)

`--smoke 64` on both datasets, one vision + one text model each:
- extraction writes `smoke_<model>.npy` with shape `(64, d)`, no all-zero matrix;
- scoring runs `heldout_distance_test` on the 64-row subgraph and returns finite
  numbers with buddy ≤ random distance (sign check; magnitudes are noisy at N=64
  and are **not** interpreted).

Smoke proves the registry, both encoder paths, the row-order join, caching, and the
scorer wiring. Conclusions wait for the full run.

## Out of scope

- Cross-modal (image-vs-caption) alignment scoring — explicitly not done; unpaired
  towers are not comparable.
- Retraining, retrieval evaluation, or changing how conditions are initialized.
- No git commits (per standing user preference: commits are done manually).
