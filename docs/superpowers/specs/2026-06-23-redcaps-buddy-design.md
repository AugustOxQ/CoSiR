# RedCaps Buddy Run — Validate Generalization + Explore Init Structure

**Date:** 2026-06-23
**Status:** Approved (no commit per user)
**Branch:** `experiment/conditional_buddy`
**Precursors:** `docs/reports/2026-06-22_buddy_analysis.md` (Impressions),
`docs/superpowers/specs/2026-06-09-conditional-buddies-init-design.md`

## Why RedCaps

Impressions' buddy signal was dominated by a near-duplicate quirk: 814 source
photos behind 12,123 records, so 80% of strict-buddy (`B`) edges literally
connect the *same photo*. The honest open question was: does the buddy signal
survive on a dataset with **no near-dup degeneracy**?

RedCaps is that test — genuinely **1 image : 1 caption** (every `image_id`
unique). Every buddy edge is **cross-content by construction**; there is no
same-photo scaffolding to prop up the signal. RedCaps also hands us a far richer
free ground truth than Impressions' 4 caption-types: **350 subreddits**, parsed
from the image path (`redcaps/imagesYYYY/<subreddit>/id.jpg`).

## Goal

1. **Validate** that condition buddies capture real content on clean 1:1 data
   (not a near-dup artifact), repeating yesterday's probes adapted to RedCaps.
2. **Explore** the structure of the buddy-init condition space (subreddit
   clustering, coarse→fine hierarchy) — the "bigger picture" of what the space
   means.

## Decisions (from brainstorming)

1. **Scale:** uniform-random **150K** subsample of the 1.55M records (seed 42),
   preserving natural subreddit frequencies. Big enough for structure across 350
   subreddits and solid subreddit-lift stats; brute-force mutual-KNN stays exact
   and fast. No ANN backend, no full-dataset run.
2. **Subsample mechanics:** pre-build a shuffled subsample JSON. The current
   `FeatureExtractionDataset` only takes a *prefix* (`ratio`), and the file is
   grouped by subreddit — a prefix would be badly biased.
3. **Fresh feature store:** extract into `/data/SSD2/pre_extract/redcaps_150k/features`
   (the stale 6 GB `redcaps/features` store is old-format and untouched).
4. **VLM probe deferred** to a Phase 2 (needs the Qwen2.5-VL vLLM server).
5. **No photo-identity probe** — N/A by design (no near-dups). That absence is
   the point.

## Data & Extraction

- `build_subsample.py`: `redcaps_medium.json` (1,553,447) → `redcaps_150k.json`
  (150,000), `random.seed(42)`. Records keep `{image, caption, image_id}`.
  Path: `/data/PDD/redcaps/redcaps_plus/redcaps_150k.json`.
- `extract_features.py`: standalone, replicates `_extract_or_load_features`
  extraction loop. Builds `CoSiRModel(backbone="openai/clip-vit-base-patch32")`
  + `AutoProcessor`, `FeatureExtractionDataset(redcaps_150k.json, ratio=1)`,
  writes `img_features [512]`, `txt_features [512]`, `sample_ids` (shuffled) via
  `FeatureManager` to the fresh store, current shard format.

## Phase 1 — offline probes (B vs E throughout)

`B = A_img ∩ A_txt` (strict), `E = A_img ∪ A_txt` (union, used for init), K=30.

1. **Subreddit lift** (replaces type-confusion): `P(buddy same-subreddit) /
   P(random same-subreddit)`. The honest headline — no photo-identity split
   needed (no near-dups). Report overall lift + χ², and the strongest/weakest
   subreddits. χ² over the 350-way contingency (or top-K subreddits + "other").
2. **Held-out DINOv2** (cleanest, encoder-independent): `extract_dino.py`
   (DINOv2-small) for the 150K, mean buddy vs random image cosine distance. All
   pairs are cross-content here.

## Phase 1 — init-structure exploration

Run `compute_buddy_init(img, txt, n_dim=16, K=30, alpha=0.5, method="spectral",
normalize_method="rank")` → `[150K, 16]`, then:

- **Subreddit silhouette** in the 16-d init space (Impressions caption-type
  silhouette was ≈0.019; does a 350-way label show real structure?).
- **2D projection colored by subreddit** — do topical clusters emerge?
- **Coarse→fine hierarchy test:** mutual information of each of the 16 dims with
  subreddit; K-means at low-K (meta-topics) vs high-K (subreddits) to test
  whether the space is hierarchical (hypothesis: dim-index ≈ granularity).

## Outputs & Layout

- Code: `src/test/20260623_redcaps_buddy/` (CLAUDE.md dated-folder convention):
  `build_subsample.py`, `extract_features.py`, `redcaps_buddy.py` (RedCaps
  `load_data` with subreddit-from-path; reuses `src/conditional_buddy/`
  primitives `mutual_knn`, `union_graph`, `compute_buddy_init`),
  `extract_dino.py`, `run_phase1.py`.
- Report: `docs/reports/2026-06-23_redcaps_buddy.md` + `assets/redcaps_buddy/`.

## Phase 2 (deferred)

VLM pairwise judge (Qwen2.5-VL): does a buddy's caption describe the anchor
image vs a random caption? Even cleaner than on Impressions — no same-style or
same-photo confound. Needs the vLLM server; stubbed for follow-up.

## Not Doing

- Full 1.55M run / ANN backend.
- Any change to the buddy-init algorithm or model.
- Photo-identity probe (no near-dups).
