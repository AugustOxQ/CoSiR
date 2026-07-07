# Buddy Analysis — Design

**Date:** 2026-06-22
**Status:** Approved (proceed to implementation)
**Branch:** `experiment/conditional_buddy`

## Question

Are "condition buddies" (cross-modal mutual nearest neighbours used to initialize
condition vectors) *meaningful* — i.e. do they connect samples that genuinely share
content — or are they a graph artifact? The originally-requested probe is a
type confusion matrix; this design strengthens it and adds type-free evidence.

## Key data fact (reshapes the analysis)

Impressions has **814 unique source images** behind its **12,123** records. Each
photo is reused 7–56× (median 14) with different captions, and every source image
spans multiple `caption_type`s.

Consequences:
1. The buddy graph is built on CLIP **image** features, so same-source-image
   siblings (near-identical image vectors) will dominate image-side edges. Those
   siblings carry *different* caption types → a raw type confusion matrix looks
   off-diagonal **because they are the same photo in a different caption style**,
   not because buddies are meaningless. The confound must be controlled.
2. This gives a **free, finer-than-type ground truth: source-image identity (814
   classes)**, parsed from `ImgId` (`{hash}_cap{N}_{M}` → `{hash}`).

`image_id` ≠ list position in `impressions_train.json`, so types/captions/images
must be looked up via an explicit `image_id → record` dict (CLAUDE.md sample-ID
consistency rule), never by list index.

## Scope

- Dataset: Impressions, all 12,123 samples.
- Both graphs: strict **B = A_img ∩ A_txt** and union **E = A_img ∪ A_txt**
  (reuse `src/conditional_buddy/buddy_graph.py:mutual_knn`, K=30).
- Code/artifacts under `src/test/20260622_buddy_analysis/`.

## Inputs / loading

- Features: `FeatureManager("/data/SSD2/pre_extract/impressions/features")`,
  `load_all_to_ram(["img_features","txt_features"])` → img/txt feats + `sample_ids`.
  L2-normalize before KNN (match init pipeline).
- Metadata: `/project/Impressions/metadata/impressions_train.json` (list of
  `{image_id, ImgId, image, caption, caption_type}`). Build `image_id → record`.
- Type map: `{caption:0, description:1, impression:2, aesthetic:3}`.
- Image root for VLM phase: Impressions image dir (resolve from `image` filename).

## Analysis steps

**1. Source-image identity (backbone).** For B and E: edge counts, avg degree,
fraction of edges within-source-image vs cross-image, buddy-by-photo concentration.
Split all downstream edge metrics into **within-image** and **cross-image**.

**2. Type confusion matrix (corrected).** `n_type × n_type` over buddy edges as
**lift over type base rate** (observed/expected under random pairing) + χ² test.
Computed over (a) all edges and (b) **cross-image-only** edges. Symmetrize edges.

**3. Independent-signal NN test (type-free).** Are buddies closer than random in a
representation NOT used to build the graph? Primary: **DINOv2 image features**
(self-supervised, different from CLIP). Report buddy-vs-random mean distance ratio
separately for within- and **cross-image** edges (cross-image is the real test).
Fallbacks if DINOv2 weights don't download offline: sentence-transformer caption
embeddings, then caption noun-overlap.

**4. VLM pairwise judgement (Phase 2, gated on Qwen vLLM server).** Sample
cross-image buddy pairs vs random cross-image pairs; reuse `QwenAnnotator`
cross-relevance (image_i vs caption_j and image_j vs caption_i) → "same content?"
score. Runs only when the server is up.

## Deliverable

Notebook in `src/test/20260622_buddy_analysis/` + short report in `docs/reports/`
with matrices/plots and a verdict: are buddies meaningful, and at what granularity
(same-photo vs same-scene vs same-type)?

## Phasing

- Phase 1 (offline, self-contained): steps 1–3.
- Phase 2 (needs server): step 4.
