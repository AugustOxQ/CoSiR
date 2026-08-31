# Conditional Buddies Initialization — Design

**Date:** 2026-06-09
**Status:** Approved
**Source idea:** `.claude/conditional_buddies_init.md`

## Goal

Provide a new condition-vector **initialization strategy** for the CoSiR /
Combiner framework. Instead of the current `imgtxt` init (`(img_feat − txt_feat)`
→ PCA → normalize), initialize per-sample condition vectors from the cross-modal
mutual-nearest-neighbour ("buddy") graph, so samples sharing multimodal context
start close in condition space.

The new strategy plugs into the existing `TrainableEmbeddingManager` exactly like
`imgtxt`: it produces a `[N, D]` float32 array ordered by `sorted(sample_ids)`,
written to `embeddings.npy`, and reuses the existing `template_embeddings` caching.

## Non-goals

- The **buddy loss** (training-time use of the strict intersection graph `B =
  A_img ∩ A_txt`) is out of scope; this task is initialization only. The union
  graph `E = A_img ∪ A_txt` is used for init, per the source spec.
- No change to model architecture, loss, or the training loop beyond wiring the
  new strategy name.

## Decisions (from brainstorming)

1. **Integration**: a new `initialization_strategy: buddies`, fitting the current
   framework (sibling to `imgtxt`). Heavy logic lives in `src/conditional_buddy/`.
2. **KNN backend**: torch GPU brute-force `topk` over L2-normalized features.
   Exact (identical to the spec's `faiss.IndexFlatIP`), no new dependency. Kept
   behind a clean `mutual_knn` interface so a cuvs ANN backend can be swapped in
   later for million-scale datasets without touching callers.
3. **Features**: assume pre-extracted. Load via `FeatureManager` from
   `cfg.featuremanager.storage_dir`; error clearly if the store is missing.
4. **`N_DIM`** = `cfg.model.embedding_dim` (16 for clip_base), never hardcoded.
5. **Embedding method**: spectral (Laplacian Eigenmaps), spectral-only. SMACOF was
   dropped after the sanity check (see Findings) — sklearn 1.8 removed the `weight=`
   argument its missing-edge-aware variant required, and spectral covers the use case.
6. **Normalization**: per-dimension **rank → [-1, 1]** (default), with z-score kept
   as a non-default option. Decided after the 2D sanity check exposed eigenvector
   localization (see Findings).

## Module layout (`src/conditional_buddy/`)

| File | Responsibility |
|------|----------------|
| `buddy_graph.py` | `mutual_knn(features, K, device, batch_size, backend="torch")`, `union_graph(A_img, A_txt)`, `sparse_cosine_distance(feats, E)`, `rank_normalise_sparse(D)`, `mix_distances(D_img_n, D_txt_n, alpha)`, `ensure_min_degree(E, feats_img, feats_txt)` |
| `embedding_methods.py` | `spectral_embedding(D_mixed, n_dim, seed)`, `normalise_embedding(emb, method="rank"|"zscore")` |
| `compute_buddies.py` | `compute_buddy_init(img_feats, txt_feats, sample_ids, n_dim, method, K, alpha, device, ...) -> np.ndarray[N, n_dim]`. Orchestrates Steps 1–6, prints per-step sanity checks, reorders to `sample_ids`. |
| `visualize.py` | `plot_2d_buddies(emb2d, A_img, A_txt, out_path)` + buddy-vs-random distance ratio report. |
| `init_conditions.py` | Hydra/CLI entry: load features for a dataset, run pipeline, write output. `--n-dim 2 --visualize` runs the 2D sanity check → `debug_init_2d.png`. |
| `__init__.py` | Re-exports `compute_buddy_init`. |

## Data flow

```
features (FeatureManager, L2-normalized)
  → A_img, A_txt          # mutual KNN per modality (torch GPU topk)
  → E = union(A_img, A_txt) + ensure_min_degree   # broad, connected
  → D_img, D_txt          # sparse cosine distance on E's edges only
  → rank_normalise_sparse # global rank / nnz ∈ (0,1]
  → D_mixed = α·D_img_n + (1−α)·D_txt_n
  → spectral_embedding (default) | smacof_embedding (small N)
  → normalise_embedding   # z-score, clip ±3, /3 → ~[−1, 1]
  → reorder rows to sample_ids order
  → [N, n_dim] float32 → embeddings.npy
```

## Integration points (minimal edits to existing files)

- **`src/utils/embedding_manager_nocache.py`**
  - Add `"buddies"` branch to `initialize()`. It loads `img_features`/`txt_features`
    shard-by-shard from `feature_manager`, calls `compute_buddy_init`, and reorders
    rows to `self.sample_ids` order.
  - Add `initialize_embeddings_buddies(...)` wrapper (mirrors `initialize_embeddings_imgtxt`).
  - Extend `store_imgtxt_template` / `load_imgtxt_template` with a backward-compatible
    `extra: dict | None = None` argument recording `{K, alpha, method, n_dim}`; the
    load-time mismatch check includes these so a stale buddies template is rejected.
- **`src/hook/train_cosir.py::_init_embedding_manager`**
  - Add `"buddies"` to the strategy gate and `_init_fn_map`. Buddy hyperparameters
    are read from `cfg.train.buddies.*` and threaded through.
- **`configs/train/default.yaml`**
  - Add a `buddies:` block: `k: 30`, `alpha: 0.5`, `method: spectral`,
    `knn_batch_size: 1024`, `normalize_method: rank`. Activate via
    `train.initialization_strategy: buddies`.

## Robustness

- **Isolated nodes**: union of mutual-KNN may leave zero-degree samples; every
  sample needs a vector, so `ensure_min_degree` connects each isolated node to its
  top-1 NN (directed → symmetrized), guaranteeing degree ≥ 1. Report avg degree and
  `n_connected_components`.
- **SMACOF memory guard**: dense `N×N` only built when `N ≤ smacof_max_n`; otherwise
  raise with a pointer to spectral.
- **Determinism**: seed 42 throughout.
- **Sample-ID consistency**: rows reordered to `sample_ids` order before writing
  (CLAUDE.md critical pattern).
- **Dataset-agnostic**: coco / impressions / redcaps differ only by `storage_dir`.
- **Memory**: never materialize full N×N similarity; `mutual_knn` batches over query
  rows. Sparse matrices used throughout the spectral path.

## 2D sanity check

`init_conditions.py --n-dim 2 --visualize` runs the full pipeline at `N_DIM=2`
(optionally on a subsample), saves `debug_init_2d.png` (scatter with mutual-buddy
edges overdrawn), and prints mean embedded distance of buddy pairs vs. random pairs
(buddies should be markedly closer).

## Testing

Synthetic test under `src/test/20260609_conditional_buddy/`: two Gaussian clusters
in img + txt space, asserting:
- (a) `E` average degree > 2,
- (b) buddy pairs closer than random pairs in the 2D embedding,
- (c) output shape `[N, 16]`, values within `[−1, 1]`,
- (d) sample-ID reorder correctness (shuffled `sample_ids` → correct row order).

## Findings from the 2D sanity check (Impressions, 12,123 samples)

- Union graph `E`: avg degree **23.78**, **1** connected component, 0 isolated nodes.
- **Eigenvector localization**: with the original z-score normalization, the
  Laplacian-Eigenmaps low eigenvectors concentrated on a few hub nodes — 99.8% of
  samples collapsed within 1e-3 of the median while a few spiked to ±1. As an init
  this would start ~all conditions at the same point.
- **Fix**: per-dimension rank → [-1, 1] normalization. Result spreads dims evenly
  (per-dim std ≈ 0.577) while buddies stay clearly closer than random
  (buddy/random distance ratio ≈ 0.06). This is now the default `normalise_embedding`.
- **SMACOF**: dropped. sklearn 1.8's `smacof` no longer exposes `weight=`, so the
  missing-edge-aware variant the source spec used is unavailable; spectral + rank
  covers the need and scales to MS-COCO / RedCaps.

## Acceptance criteria (from source spec)

- `E` has average degree > 2 per node on Impressions.
- 2D Laplacian Eigenmaps projection inspectable via `debug_init_2d.png`.
- `init_conditions` shape `(N, 16)`, values in `[−1, 1]`.
- No dense N×N matrix allocated on the spectral path.
