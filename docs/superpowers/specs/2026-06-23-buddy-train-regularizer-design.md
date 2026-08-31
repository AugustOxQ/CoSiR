# Buddy-Graph Smoothness Regularizer (Family #1) — Design

**Date:** 2026-06-23
**Status:** Approved (no commit by Claude per user; user handles commits)
**Branch:** `experiment/conditional_buddy_train`
**Precursors:** `docs/reports/2026-06-23_redcaps_buddy.md` (buddy signal validated: subreddit
lift ~20×, held-out DINOv2 0.39, VLM GOOD-rate 81%), `src/conditional_buddy/`,
`src/metrics/loss.py`, `src/metrics/regularizer.py`.

## Context

Today the conditional-buddy signal is used **only to initialize** the per-sample trainable
label embeddings `z` (`embedding_manager.initialize_embeddings_buddies` →
`compute_buddy_init`). After init, `z` trains freely under `LabelContrastiveLoss_enhance`
and the buddy geometry is free to wash out. This is the first of three staged experiments
("use the buddy signal during training, beyond init"); it is the easiest and most
self-contained. Families #2 (buddies as contrastive supervision) and #3 (self-refreshing
buddies / co-training) are deferred to their own specs.

## Goal

Keep the validated buddy geometry alive during training by adding a Laplacian smoothness
term on `z` along the union graph `E` — the **same energy that produced the init**. Laplacian
Eigenmaps minimizes `Σ_{(i,j)∈E} w_ij·‖z_i − z_j‖²`; we keep minimizing exactly this during
training so the init becomes a regularization target rather than a starting point we drift
from. Success = the term trains stably, buddy pairs stay close in `z` through training, and
downstream CoSiR metrics are at least preserved (ideally improved) versus `lambda_buddy=0`.

## The term

```
L_buddy = (1 / |S|) · Σ_{(i,j) ∈ S} ‖z_i − z_j‖²
```

- `z` = the trainable label embeddings (`embedding_manager.embeddings`, the differentiable
  leaf `[N, D]`, `D=16`). Raw `z`, **no re-normalization** — the term then matches the init
  energy exactly (decision: "Plain L2, freeze none").
- Unit edge weights (E is structural; distance-weighting is a deferred ablation).
- `S` = a per-step sample of edges from `E` (see Scope). The total objective adds
  `lambda_buddy · L_buddy`.

## Scope — global table gather (not in-batch)

Random batches almost never contain buddy pairs, so an in-batch-only term would rarely fire.
Instead, each step:

1. Take the batch's sample-ids → positions `p` into the `[N, D]` table.
2. For each anchor position `i` in the batch, look up its E-neighbors and sample up to
   `s = buddy_reg_samples` of them (fewer if degree < s; skip anchors with no neighbors).
3. Gather the anchors' and sampled buddies' current `z` rows directly from the full table
   (a differentiable leaf — gradients flow to **both** anchor and buddy rows, including buddy
   rows not present in the batch).
4. Accumulate `‖z_i − z_j‖²` over the sampled edges and divide by the edge count.

Cost: `batch × s` gathers of 16-d vectors per step — negligible. Isolated anchors contribute
0 (after `ensure_min_degree` + `connect_components`, near-isolated nodes are rare).

## Persisting E

`compute_buddy_init` builds `E` internally and discards it. Changes:

- `compute_buddy_init(..., return_edges: bool = False)`: when `True`, also return `E`'s edge
  list as a COO `int64 [2, M]` array, **remapped through the same `input→output` sample-id
  reorder** already applied to the init rows (so node indices are table positions in
  `output_sample_ids` order). Undirected edges stored once (i < j); self-loops dropped.
- `embedding_manager_nocache` buddy-init path: when it computes the init, also request the
  edges and save `buddy_edges.npy` next to the init template (same directory/template key).
  The template-reuse path loads `buddy_edges.npy` alongside the cached init; if the file is
  missing (older template), warn and fall back to `lambda_buddy = 0` for the run rather than
  recomputing.

## Wiring

- New standalone function in `src/metrics/regularizer.py`:
  `buddy_graph_smoothness_loss(embeddings_table, edge_index, batch_positions, num_samples, generator=None) -> Tensor`.
  Pure function of the table + graph + batch; no model forward; unit-testable in isolation.
  Returns a scalar (0.0 tensor when no eligible edges).
- Invoked in the **training loop** in `src/hook/train_cosir.py` (where the full table, batch
  sample-ids, and `edge_index` all live), added to the total loss as
  `lambda_buddy · L_buddy`, and logged to wandb like the other terms (`loss_buddy`).
- **Not** threaded through `LabelContrastiveLoss_enhance.forward` — that loss only sees the
  batch slice; passing the full table + graph through it would muddy its interface. Keeping
  `L_buddy` in the train loop keeps the term isolated.
- `edge_index` is loaded once at train start (from the embedding manager / template) into a
  CUDA `int64 [2, M]` tensor, plus a CSR-style neighbor index for O(1) per-anchor sampling.

## Config (cfg.loss)

| key | default | meaning |
|-----|---------|---------|
| `lambda_buddy` | `0.0` | weight; **0 → term off, fully backward-compatible** |
| `buddy_reg_samples` | `4` | buddies sampled per anchor per step |
| `buddy_reg_graph` | `"E"` | graph source (only `E` wired now; `B` is a later ablation) |

With `lambda_buddy = 0.0` (default), behavior is byte-for-byte the existing pipeline: no
edge load, no term. Existing experiments and configs are unaffected.

## Testing

Unit (`src/test/<date>_buddy_train_reg/`):
- `L_buddy` on a tiny synthetic table + hand-specified edges equals the hand-computed
  `mean ‖z_i − z_j‖²` (deterministic sampling via a seeded generator, or s ≥ max degree).
- One optimizer step with only `L_buddy` provably **shrinks** a buddy pair's distance.
- An isolated node (no edges) contributes exactly 0 and produces no gradient.
- Edge remap: a permuted `input→output` sample-id map yields edges whose endpoints still
  connect the correct samples.

Smoke:
- Training runs end-to-end with `lambda_buddy > 0`; `buddy_edges.npy` is written, reloaded
  from a template, and round-trips; `loss_buddy` appears in logs and is finite.

## Out of scope (deferred ablations / later families)

Weight schedules / decay of `lambda_buddy`; normalized-`z` or distance-weighted edges;
delta- or combined-feature targets; the `B` graph; Family #2 (contrastive supervision) and
Family #3 (co-training). Each is a separate follow-up.
