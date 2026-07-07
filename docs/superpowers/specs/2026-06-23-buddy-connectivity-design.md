# Buddy-graph Connectivity Fix — Design

**Date:** 2026-06-23
**Status:** Approved
**Branch:** `experiment/conditional_buddy`
**Context:** `docs/reports/2026-06-23_redcaps_buddy.md` (post-fix update),
`.claude/20260623_log.md` (eigensolver fix)

## Problem

On diverse 1:1 data the buddy union graph `E` fragments (54 components on
RedCaps-150K). A disconnected graph's Laplacian has one near-zero eigenvalue per
component, so a 16-dim spectral init is consumed by **component-indicator**
vectors and carries almost no topical structure (ARI vs subreddit 0.02, per-dim MI
~0.15 — vs 0.17 / ~1.3 on a forced-connected proxy). `ensure_min_degree`
guarantees degree ≥ 1 but **not connectedness**.

## Goal

Make `E` a single connected component with minimal, content-aware bridge edges, so
the existing spectral path yields an init whose 16 dims carry within-graph smooth
structure — letting the validated per-edge buddy signal (lift 20×, DINO 0.39) reach
the init. No change to the spectral solver or normalization.

## Component: `ensure_connected`

New `ensure_connected(E, img_feats, txt_feats, alpha=0.5, device="cuda",
use_half=True) -> (csr_matrix, dict)` in `src/conditional_buddy/buddy_graph.py`,
called in `build_buddy_graphs` immediately **after** `ensure_min_degree`.

1. Label components (`scipy.sparse.csgraph.connected_components`). If 1 → return
   `E` unchanged with `{"n_components": 1, "bridges_added": 0}`.
2. Per component, pick a **medoid** = the node whose mix-weighted concat feature
   `[√α·img_n, √(1−α)·txt_n]` has the highest cosine to the component centroid.
   Cosine on this concat equals `α·cos_img + (1−α)·cos_txt`, matching the
   pipeline's mixed similarity.
3. Build a **minimum spanning tree over the medoids** (dense `C×C`, C = #components
   ≤ tens, trivial) using `1 − concat_cosine` as distance.
4. Add the `C−1` MST edges to `E` as **binary, symmetric** entries. The downstream
   `sparse_cosine_distance(feats, E)` assigns each bridge its true per-modality
   cosine distance, so bridges are naturally weak (cross-component pairs are far).

Result: `E` becomes 1 component → Laplacian null space dim 1 → all 16 dims carry
smooth structure. Spectral + normalize steps unchanged.

## Interfaces & backward compatibility

- `connect_components: bool = True` threaded through `build_buddy_graphs` and
  `compute_buddy_init` (so it can be disabled for ablation).
- **Default-on is safe**: a no-op when `E` is already connected, so Impressions /
  COCO and every prior validated run are unchanged. Only fragmented graphs get
  bridges.

## Testing

Extend `src/test/20260609_conditional_buddy/test_compute_buddies.py`: the existing
synthetic two-Gaussian case yields `components=2`. Assert `ensure_connected` →
1 component with `bridges_added == 1`, and that buddies stay closer than random.

## Validation

Re-run the real 150K structure probe with connectivity on. Report a **3-way
comparison** in the RedCaps report: baseline proxy / post-fix no-connectivity /
**post-fix with connectivity**. Success = ARI + per-dim MI recover. Add a
`--no-connect` flag to `run_structure.py` for the ablation row.

## Not doing

Changing the spectral solver or normalization; virtual nodes; per-component
embedding; touching `ensure_min_degree`.
