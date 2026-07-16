# Do buddies generalize across vision *and* language models? — Held-out encoder grid

**Date:** 2026-07-08
**Datasets:** Impressions (12,123) · RedCaps-150k
**Design:** `docs/superpowers/specs/2026-07-08-heldout-grid-design.md`
**Code:** `src/test/20260708_heldout_grid/`
**Status:** ✅ complete — full run 2026-07-09; all 24 cells (6 encoders × 2 graphs × 2 datasets) show buddy < random

## Question

The buddy graphs **B** (= A_img ∩ A_txt) and **E** (= A_img ∪ A_txt) are built from
CLIP ViT-B/32 (both towers). Prior work refuted the "buddies are a CLIP artifact"
worry with **one** held-out encoder — DINOv2, vision only
(`2026-06-22_buddy_analysis.md`, `2026-06-23_redcaps_buddy.md`). This report widens
that into a **2-axis grid**: 3 independent vision encoders (image space) + 3
independent language encoders (text space), on both datasets, for both graphs. If
cross-content buddy pairs stay closer than random pairs across *every* held-out
encoder — spanning self-supervised, VLM, supervised vision and three unrelated text
embedders — buddy meaningfulness generalizes beyond CLIP and beyond one modality.

## Method

**Within-modality, two independent axes.** Each held-out model scores buddies in its
own modality (vision → images, language → captions) by mean cosine **distance** of
buddy edges vs random pairs. Lower buddy-vs-random ratio = buddies are closer =
meaningful in a representation the graph never saw. This is a direct generalization
of the DINOv2 test; image and text embeddings from unpaired models are not
comparable, so no cross-modal cell is scored. Scoring reuses each dataset's existing
`heldout_distance_test`, so **Impressions keeps the within/cross-photo split** (the
honest number is *cross-photo*, given 814 source photos back 12,123 records) and
**RedCaps is a single number** (genuinely 1:1, every edge cross-content).

### Model slate (all via plain `transformers`, no `sentence-transformers`)

| axis | key | HF id | paradigm | pool | dim |
|---|---|---|---|---|---|
| vision | `dinov2` | facebook/dinov2-small | self-supervised | CLS | 384 |
| vision | `siglip_v` | google/siglip-base-patch16-224 | sigmoid VLM | pooled image feats | 768 |
| vision | `vit_sup` | google/vit-base-patch16-224 | ImageNet supervised | CLS | 768 |
| language | `minilm` | sentence-transformers/all-MiniLM-L6-v2 | — | mean | 384 |
| language | `bge` | BAAI/bge-base-en-v1.5 | — | CLS | 768 |
| language | `e5` | intfloat/e5-base-v2 | — | mean, `"query: "` prefix | 768 |

DINOv2 features already exist for both datasets and are reused verbatim; only the
five new encoders are extracted.

## Smoke test — pipeline validated ✅

`--smoke 64` (first 64 rows), run 2026-07-08. Purpose: exercise every code path
(both image paths — backbone-CLS and `get_image_features`; both text paths — mean and
CLS pooling, with/without prefix; row-order join; caching; scorer). Magnitudes at
N=64 are noisy and **not** interpreted — only shapes and the buddy < random *sign*.

**Impressions smoke** — all six encoders extracted cleanly (correct dims, 0 missing,
0 zero-rows). The first-64 subgraph contained **6 E-edges** (B is sparse → 0 edges in
64 nodes), so the sign check fired on E for all six:

| encoder | modality | E buddy dist | E random dist | ratio (buddy/random) |
|---|---|---|---|---|
| dinov2 | vision | 0.437 | 0.934 | **0.47** |
| siglip_v | vision | 0.269 | 0.525 | **0.51** |
| vit_sup | vision | 0.498 | 0.907 | **0.55** |
| minilm | text | 0.604 | 0.863 | **0.70** |
| bge | text | 0.370 | 0.541 | **0.68** |
| e5 | text | 0.197 | 0.263 | **0.75** |

Every encoder — three vision paradigms and three text embedders — puts buddies
closer than random (ratio < 1), even on 6 edges. Vision separates harder (0.47–0.55)
than text (0.68–0.75), consistent with the DINOv2-only finding.

**RedCaps smoke** — `siglip_v` + `e5` extracted cleanly (`/data/PDD` image root +
caption field wired correctly); the full **150k buddy graph builds** and the scorer
runs end-to-end (exit 0, JSON written). The 64-node window contained 0 buddy edges
(expected — RedCaps buddies are globally sparse in a 1:1 dataset), so the RedCaps
sign check is **deferred to the full run**.

## Full-run results — Impressions (cross-photo)

Ratio = cross-photo buddy dist / random cross-photo dist (< 1 = buddies closer =
meaningful). Figure: `assets/heldout_grid/impressions_grid.png`.

| encoder | modality | B ratio | E ratio |
|---|---|---|---|
| dinov2 | vision | **0.41** | 0.69 |
| siglip_v | vision | **0.31** | 0.60 |
| vit_sup | vision | **0.44** | 0.70 |
| minilm | text | **0.48** | 0.72 |
| bge | text | **0.53** | 0.72 |
| e5 | text | **0.53** | 0.74 |
| *mean vision* | | *0.38* | *0.66* |
| *mean text* | | *0.51* | *0.73* |

## Full-run results — RedCaps-150k

Ratio = buddy dist / random dist (all edges cross-content; no same-photo confound
to control for). Figure: `assets/heldout_grid/redcaps_grid.png`.

| encoder | modality | B ratio | E ratio |
|---|---|---|---|
| dinov2 | vision | **0.41** | 0.61 |
| siglip_v | vision | **0.37** | 0.52 |
| vit_sup | vision | **0.44** | 0.62 |
| minilm | text | **0.46** | 0.70 |
| bge | text | **0.49** | 0.70 |
| e5 | text | **0.48** | 0.70 |
| *mean vision* | | *0.40* | *0.58* |
| *mean text* | | *0.48* | *0.70* |

## Verdict

**Buddies generalize completely — across encoder, across modality, across dataset.**
Every one of the 24 cells (6 held-out encoders × {B, E} × {Impressions, RedCaps}) puts
buddy pairs closer than random, and the four structural facts line up cleanly:

1. **Not a CLIP artifact, not a single-encoder artifact.** Three vision paradigms
   (self-supervised DINOv2, sigmoid-VLM SigLIP, ImageNet-supervised ViT) *and* three
   unrelated text embedders (MiniLM, BGE, E5) — none of which built the graph — all
   agree. The signal survives every representation we throw at it.

2. **Not a modality artifact.** The vision axis alone (the old test) could have been a
   quirk of image features. Adding the text axis closes that: independent *language*
   models confirm buddy captions are closer than random captions too (mean E ratio
   ~0.70). Vision separates a bit harder than text (mean B 0.38–0.40 vs 0.48–0.51),
   but both are unambiguous.

3. **Not a same-photo artifact — the key result.** The standing caveat on Impressions
   was its 814 source photos behind 12,123 records. RedCaps is genuinely 1:1 (every
   image unique), so its buddy edges *cannot* lean on near-duplicates — and the grid
   is if anything **tighter** there (mean vision E 0.58 vs 0.66 on Impressions). Buddy
   meaningfulness is a real cross-content property, not near-duplicate bookkeeping.

4. **B ≪ E, everywhere — the recurring theme.** The strict intersection is markedly
   tighter than the union in all 24 cells (mean B ~0.38–0.51 vs E ~0.58–0.73).
   Initialization uses **E** for full connectivity, yet every quality probe — type
   lift, DINOv2, the VLM judge, and now this six-encoder grid — prefers **B**. This
   sharpens the earlier recommendation: test an init that leans on B where it exists
   and falls back to E only for B-isolated nodes.

This is the strongest form of the original question ("are buddies reasonable?") we can
answer without training: yes, robustly, and the strict graph B is consistently the
cleaner signal.

## Reproduce

```bash
conda activate CoSiR
cd src/test/20260708_heldout_grid

# Smoke (done): first 64 rows, one path per encoder type
python extract_heldout.py --dataset impressions --model siglip_v --smoke 64
python score_grid.py --dataset impressions --smoke 64

# Full run (~35 min; DINOv2 reused): one command does all extraction + scoring.
bash run_full.sh
```
