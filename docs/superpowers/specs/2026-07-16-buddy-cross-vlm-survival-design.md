# Cross-VLM Buddy Survival — Design

**Date:** 2026-07-16
**Status:** Design approved, pending implementation plan
**Dataset:** RedCaps (150k subsample, the existing buddy-analysis store)

## Question

Do conditional buddies **survive when the vision–language model is changed**? Concretely:
when we rebuild the cross-modal buddy graph using a *different* vision encoder (for the
image side) crossed with a *different* text encoder (for the caption side), do the same
buddy pairs recur — and is there a stable "core" of buddies independent of encoder choice?

This is **not** the held-out grid (`2026-07-08-heldout-grid-design.md`). There, the buddy
graph is fixed at CLIP∪CLIP and six single-modality encoders only *score* whether buddy
pairs are closer than random in their own space — the graph is never rebuilt. Here we
**rebuild the cross-modal buddy graph per (vision × text) encoder pair** and measure how
much the resulting buddy edge sets agree.

## Key idea: buddies are a graph, not a shared space

The buddy graph combines *edges*, not feature vectors:

```
A_img = mutual_knn(vision_feats, K)   # mutual-NN graph in the vision encoder's own space
A_txt = mutual_knn(text_feats,   K)   # mutual-NN graph in the text encoder's own space
B = A_img ∩ A_txt                      # strict buddies (mutual-NN in BOTH modalities)
E = A_img ∪ A_txt                      # union buddies (what seeds condition vectors)
```

Because only the per-modality edge sets are combined (never the raw vectors), the vision
and text encoders can be heterogeneous — different architectures, different dims, no shared
embedding space. A cell in the grid is just one (vision, text) pairing.

## Scope

- **Comparison:** full pairwise agreement across the grid, then a consensus "core" of
  recurring buddies. CLIP×CLIP is one cell, so the reference-anchored view (does the CLIP
  buddy set survive?) is recoverable for free as that cell's row/column.
- **Grid (4 × 4 = 16 cells):**
  - Vision axis `V = {clip_img, dinov2, siglip_v, vit_sup}`
  - Text axis `T = {clip_txt, minilm, bge, e5}`
  - `clip_img` / `clip_txt` from `redcaps_buddy.load_data()`; the other 6 from the held-out
    grid feature caches (`heldout_feats/redcaps/<model>.npy`, aligned to `data.sample_ids`).
- **Buddy definitions:** report **both** `B` (strict) and `E` (union) equally, side by side.
- **Dataset:** RedCaps only (clean 1:1 image–caption; every edge is cross-content, so
  agreement is a pure "do the same distinct pairs recur" signal).

**Explicit non-goals (YAGNI):** no spectral embedding, no training, no Impressions, no new
model loaders. Embedding-level agreement (running the full spectral init per cell and
comparing condition embeddings) is a deliberate future extension, not part of this work.

## Architecture

New dated folder `src/test/20260716_buddy_cross_vlm/`. It **imports**, does not fork:

- `src/test/20260623_redcaps_buddy/redcaps_buddy.py` → `load_data()` (CLIP features +
  records, row order + positional join already solved), `subreddit_lift()`.
- `src/test/20260708_heldout_grid/extract_heldout.py` → `cache_path()`; `heldout_models.py`
  → `MODELS` / `HeldoutEncoder` (the six loaders, for extracting any missing caches).
- `src/conditional_buddy/buddy_graph.py` → `mutual_knn`, `union_graph`.

### Modules

- **`cross_vlm_buddy.py`** — pure-function library over in-memory arrays (no I/O), so each
  piece is unit-testable:
  - `load_grid_features()` → the 8 feature matrices in `data.sample_ids` row order **plus a
    single global valid-row mask**.
  - `build_cell_graphs()` → 8 `mutual_knn` graphs (one per distinct feature matrix: 4 vision
    + 4 text), then the 16 cells' `B` and `E` edge sets by set algebra.
  - `agreement_matrix()` → 16×16 Jaccard + overlap coefficient, with the permutation null.
  - `consensus_core()` → co-occurrence graph, survival curve `n_core(t)`.
  - `core_subreddit_lift()` → subreddit lift as a function of consensus level `t`.
- **`run_grid.py`** — CLI driver: load → build → metrics → write artifacts. Flags:
  `--smoke N` (first-N rows), `--K 30`, `--n_perm 200`, `--seed 42`.
- **`test_cross_vlm_buddy.py`** — unit tests (see Testing).
- **`20260716_buddy_cross_vlm_log.md`** — results/debugging log per repo convention.

### Data flow (single pass)

1. Load all 8 feature matrices in `data.sample_ids` row order.
2. **Common node set (linchpin):** one global valid-row mask = rows where *every* vision
   encoder has a nonzero feature (missing RedCaps images produce zero rows; text is always
   present). Slice all 8 matrices to those rows **once**, so all 16 cells share identical
   nodes — the precondition for comparing edge sets.
3. Per distinct feature matrix: `A = mutual_knn(feats, K=30)`. Eight builds total (4 vision
   + 4 text; `clip_img` and `clip_txt` are distinct matrices, one per side). Each graph is
   built once and reused across every cell that references it.
4. Per cell: `B = A_img ∩ A_txt`, `E = A_img ∪ A_txt`; extract undirected (i<j) edge sets.
5. Feed edge sets to the metrics stage.

**Efficiency:** the mutual-kNN graphs are built once and reused; a cell is cheap set
algebra. The grid costs ~8 kNN builds + 16 intersections/unions, not 32 kNN builds.

## Metrics

Computed separately for `B` and `E`.

### (a) Pairwise agreement — 16×16 matrices

For every pair of cells: Jaccard `|E₁∩E₂| / |E₁∪E₂|` and overlap coefficient
`|E₁∩E₂| / min(|E₁|,|E₂|)`. Headline number = median off-diagonal **lift** (see below).
The CLIP×CLIP row/column is the reference-anchored slice.

### (b) Chance correction

Raw Jaccard is inflated by density and luck. Each observed value gets a null: **relabel one
cell's node identities by a random permutation** (preserves that cell's exact degree
sequence, destroys alignment), recompute Jaccard, repeat `n_perm ≈ 200` times. Report
observed, null mean, and **lift = observed / null-mean** (plus percentile of observed in the
null). Agreement is only credited when it clears the null.

### (c) Consensus core

Build a co-occurrence graph over the common node set: each undirected edge weighted by how
many of the 16 cells contain it (0–16).
- **Survival curve** `n_core(t)` = number of edges present in ≥ `t` cells, for `t = 1…16`,
  reported as a count and as a fraction of the all-cells union.
- **Core set** at majority threshold `t ≥ 8` = "buddies that survive VL model choice."

### (d) Core validation via subreddit lift (independent ground truth)

DINOv2 is now inside the grid, so it can no longer serve as the independent validator.
RedCaps' **subreddit** label (parsed from the image path, used by nothing in the pipeline)
is fully encoder-independent. Reuse `redcaps_buddy.subreddit_lift`: compute same-subreddit
lift as a function of consensus level `t`. If the core is real, lift **rises with `t`**, and
the `t ≥ 8` core beats both the typical single cell and random pairs — i.e. the surviving
buddies are semantically coherent, not mutually-agreed noise.

## Artifacts

Written to `docs/reports/assets/buddy_cross_vlm/`:

- `grid_agreement.json` — 16×16 Jaccard / overlap / lift for `B` and `E`, plus headline
  off-diagonal median lift.
- `agreement_B.png`, `agreement_E.png` — heatmaps, cells labeled `Vᵢ×Tⱼ`.
- `survival_curves.png` — `n_core(t)` vs `t` for `B` and `E`.
- `core_lift.png` — subreddit lift vs consensus level `t`, with random-pair and typical
  single-cell reference lines.
- `core_edges_B.npy`, `core_edges_E.npy` — the `t ≥ 8` surviving edge lists.

## Testing

- **Unit (`test_cross_vlm_buddy.py`)** on synthetic tiny graphs with hand-computable answers:
  1. Two identical edge sets → Jaccard 1.0, lift > 1.
  2. Two disjoint edge sets → Jaccard 0.
  3. Permutation null of a graph vs an independent random graph → lift ≈ 1.
  4. Consensus: an edge present in all cells reaches `t = 16`; an edge in one cell → `t = 1`.
  5. Common-node mask drops exactly the zero-feature rows and nothing else.
- **Smoke:** `run_grid.py --smoke 512` — full pipeline on 512 rows; asserts shapes and
  finite values, no interpretation of magnitudes (mirrors the held-out grid convention).
- **Real run:** full RedCaps; eyeball heatmaps + curves.

## Prerequisite check

The six held-out feature caches for RedCaps (`heldout_feats/redcaps/{dinov2,siglip_v,
vit_sup,minilm,bge,e5}.npy`) must exist; any missing one is produced by
`extract_heldout.py --dataset redcaps --model <name>`. CLIP features come from the existing
RedCaps FeatureManager store via `load_data()`.
