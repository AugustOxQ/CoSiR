# Conditional Buddies Init — Session Log (2026-06-09)

## Problem
Implement a new condition-initialization strategy from the cross-modal
mutual-nearest-neighbour ("buddy") graph, per `.claude/conditional_buddies_init.md`,
fitting the existing CoSiR framework as a sibling to the `imgtxt` strategy. Before
committing to the full N_DIM=16 init, run a 2D version and visually verify that
buddies end up as neighbours.

## What was built
`src/conditional_buddy/`: `buddy_graph.py` (mutual_knn via GPU brute-force topk,
union graph, isolated-node fix, sparse cosine distance, rank normalise, mix),
`embedding_methods.py` (spectral embedding + rank/zscore normalization),
`compute_buddies.py` (Steps 1–6 orchestration + reorder), `visualize.py`,
`init_conditions.py` (hydra runner + 2D sanity mode). Wired into
`embedding_manager_nocache.py` (`buddies` strategy) and `train_cosir.py`.

## Investigation: 2D embedding collapse
First 2D run on Impressions (12,123 samples) reported a great buddy/random distance
ratio (0.046) but the plot showed ~all points collapsed at the origin with a few
spikes at the corners.

- Checked connectivity: graph was **fully connected** (1 component, 0 isolated),
  so this was **not** a disconnection artifact.
- Per-dimension spread of the raw spectral embedding: p1..p99 spanned ~1e-5..1e-4
  while max reached 0.18; **99.8%** of points fell within 1e-3 of the median.

### Root cause
Laplacian-Eigenmaps **eigenvector localization**: on graphs with heterogeneous
degree, the low eigenvectors concentrate their mass on a few hub nodes. The original
z-score normalization then divides by an outlier-dominated std, squashing the bulk
to a single point. The ratio metric looked good only because *everything* (buddies
and random alike) had collapsed.

### Solution
Replaced the default `normalise_embedding` with **per-dimension rank → uniform
[-1, 1]**. This preserves neighbourhood ordering (buddies stay close) while
guaranteeing spread. After the fix on Impressions: random mean dist ≈ 1.03 (well
spread over [-1,1]²), buddy mean dist ≈ 0.064, ratio ≈ 0.062; the 2D plot fills the
square with visible local structure. z-score retained as a non-default option.

Separately, SMACOF (source spec's Method 1) was dropped: sklearn 1.8 removed the
`smacof(weight=...)` argument its missing-edge-aware variant relied on, and spectral
+ rank normalization covers the use case and scales to MS-COCO / RedCaps.

## Verification
- Unit tests (`test_buddy_graph.py`, `test_compute_buddies.py`): all pass —
  E avg degree > 2, mutual-graph symmetry, isolated-node fix (min degree ≥ 1),
  output shape (N,16) ∈ [-1,1], buddies closer than random, reorder correctness.
- Integration through the real training path: `TrainableEmbeddingManager
  .initialize_embeddings_buddies` on Impressions → (12123, 16) ∈ [-1,1], id→row
  mapping correct, template store/load roundtrip OK, `extra`-dict guard correctly
  rejects a changed `k`.
- 2D sanity artifact: `/data/SSD2/pre_extract/impressions/buddies_debug/debug_init_2d.png`.

## Notes / follow-ups
- KNN is exact GPU brute-force (O(N²)); fine for Impressions (instant) and COCO
  (~minutes). RedCaps (1.55M) is a one-time cached template (~1–2 h). `mutual_knn`
  has a `backend` seam for a cuvs ANN swap if RedCaps needs to be fast.
- Full COCO/RedCaps init runs were left to the user (size/runtime).
