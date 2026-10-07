# Lightweight CLIP fine-tuning on ArtELingo as an extra condition-free comparator (design)

**Date:** 2026-10-07 (Amsterdam). **Status:** design approved by the user in chat ("yes"), after the CVPR readiness
memo (`docs/reports/stage/2026-10-07_cvpr_readiness.md` §2). This file is the written spec. It fixes the training,
selection and evaluation protocol before any training run. There is no separate decision rule: these numbers decide
nothing by themselves; how the comparator enters the ArtELingo held test is the user's call afterwards.

## 1. Why

AFF beats every condition-free comparator on fresh ArtELingo episodes, but by +0.59 R@1 over the strongest one
(round 3), and its whole gain is on emotion. A CLIP adapted to ArtELingo's emotion-laden captions could raise the
condition-free floor exactly there. A reviewer will ask; we measure it before the held test.

**User decisions (2026-10-07):** the paper's framing stays on hold; the comparator is a lightweight fine-tune in three
variants (a linear probe, the last block, LoRA); **no full fine-tuning**; the runs may use the DAS6 nodes 404, 405 and
411, up to 9 GPUs. The linear probe is a linear map trained with the contrastive loss, not a classifier on labels
(confirmed with the design).

## 2. Data

| Split (rows, paintings) | Use |
|---|---|
| scorer-train (183,694 rows, 36,518 paintings) | training pairs |
| val (30,872 rows, 6,152 paintings) | model selection by image–caption retrieval |
| selection (32,413 rows, 6,451 paintings) | features for the aspect episodes (evaluation) |
| held (61,744 rows, 12,281 paintings) | **not touched**: no image loaded, no feature extracted |

- A row is one caption with its painting's image (`src/data/artelingo.py::load_artelingo`, splits from
  `src/data/artelingo_splits.py`). Features and rows follow the feature-row order (`data.sample_ids`), never the
  annotation order; a caption is `annotations[sample_ids[i]]["caption"]`.
- **Image cache.** The 49,121 images of scorer-train, val and selection are preprocessed once, locally, with the CLIP
  processor of `openai/clip-vit-base-patch32` (resize, centre crop 224, before normalisation), into a uint8 array of
  shape (49,121, 224, 224, 3) (about 7.4 GB) with a painting index, under `/data/SSD2/pre_extract/artelingo_clip224/`
  (this project's cache folder; SHA-256s recorded). The nodes receive it through the cluster CLI's data-sync functions.
  No augmentation.
- No evaluation label (emotion, style, genre) is read by training or selection.

## 3. Model, objective and variants

- **Backbone:** CLIP ViT-B/32 (`openai/clip-vit-base-patch32`), features = the projection-layer outputs as in the
  existing cache (`visual_projection(vision_model(...).pooler_output)` and the text equivalent), L2-normalised for the
  loss and for scoring.
- **Objective:** the symmetric CLIP InfoNCE loss with a learnable temperature initialised from CLIP's `logit_scale`.
- **Batches:** each epoch draws one caption per painting (36,518 pairs), shuffled; a batch never holds two captions of
  the same painting, so no same-painting caption is a negative. Batch size 256.

| Variant | Trained parameters | Inputs |
|---|---|---|
| **LP**, linear probe | one 512 × 512 linear map per modality (initialised to the identity), on the frozen projection features | the cached frozen features (no images) |
| **LB**, last block | the last transformer layer, the final layer norm and the projection of each encoder | images (cache) and captions |
| **LoRA** | rank-16 adapters (α 32, dropout 0.05) on the query, key, value and output projections of every attention layer of both encoders | images (cache) and captions |

- **Optimiser:** AdamW, linear warm-up over the first 5% of steps then cosine decay, mixed precision; at most 10 epochs;
  one training seed (0). The exact weight decay per variant is fixed in the plan before any run.
- **Learning-rate grid** (three values per variant, so 9 runs): LP {1e-4, 3e-4, 1e-3}; LB {3e-6, 1e-5, 3e-5}; LoRA
  {3e-5, 1e-4, 3e-4}.

## 4. Selection (fixed now, before any run)

- After every epoch each run computes **val retrieval**: image→caption R@1 (an image query is correct if its top
  caption among all val captions belongs to the same painting) and caption→image R@1 (the top image among the val
  paintings' images is the caption's own), and their mean, **the selection metric**.
- Per variant, the selected model is the (learning rate, epoch) with the highest selection metric; ties go to the
  smaller learning rate, then the earlier epoch. The aspect episodes are never used for selection or early stopping.
- Plain CLIP's val retrieval is reported as the reference.

## 5. Evaluation (descriptive)

For each variant's selected model:

1. Features of the selection rows, in feature-row order (NaN elsewhere, as `EvalContext` masks them).
2. A cosine baseline on the existing aspect episodes of seed 42 (development) and seeds 49, 50 and 51 (round 3's test
   seeds, now descriptive), with `src.eval.aspect_scorers.cosine_scores` and `src.eval.aspect_metrics.per_anchor`. Its
   condition gain is 0 by construction.
3. Beside it: plain CLIP cosine, B, B′(A0) and AFF from round 3's stored per-anchor arrays (seeds 49 to 51) and the
   seed-42 records (B′(A1) on seed 42 only). Reported per seed, pooled over 49 to 51, and per aspect pair: R@1,
   either rate, and the paired differences AFF minus fine-tuned cosine, fine-tuned cosine minus plain cosine, and
   fine-tuned cosine minus B′(A0), each with a 95% painting-bootstrap interval (5,000 resamples, seed 42, pooled with
   one cluster per painting across seeds).
4. Val retrieval of every run and epoch, and the selected models.

No other scorer is rebuilt on the fine-tuned features (B on fine-tuned features, or AFF on top, would belong with the
second-backbone check, K4).

## 6. Infrastructure and process

- **Code:** `src/test/20261124_clip_lightweight_ft/` (folder date is a sequence number; 20261123 is left to the idea-3
  tab), modules prefixed `ft_`; a launcher `scripts/run_clipft.sh <variant> <lr>` with `--check-only`; a data-sync
  script on the pattern of `scripts/das6_sync_mllm_probe_8b.py` (run with the system Python).
- **Placement:** the three LP runs on node404 (features only), the three LB runs on node405, the three LoRA runs on
  node411, one GPU each, launched with the cluster CLI (`cluster launch --node <node>`). A real in-job data check runs
  on each node before training (the CLI's pre-check is unreliable for new paths).
- **Outputs per run:** per-epoch val metrics, the selected checkpoint's trained parameters only, and the selection-row
  and val-row features of its best epoch; pulled back with `cluster pull`.
- **Steps:** the plan; implementation by subagents with task reviews (data cache, trainer with unit tests on tiny
  synthetic inputs, cluster scripts, evaluation); a local smoke run on a tiny subset; one cluster smoke job; the nine
  runs (the main session launches them); an independent agent recomputes the evaluation numbers from the pulled
  features with its own code; the report `docs/reports/auto/v2/2026-11-24_clip_lightweight_ft.md` with its
  `reports_sum.md` row, after a final review.
- Times in Amsterdam local time; commits to `main` by explicit path, never pushed; storage: the image cache (7.4 GB) in
  the project's pre_extract folder, run features in the results folder, smoke files deleted.
