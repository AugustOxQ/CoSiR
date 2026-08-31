# B-leaning condition initialization (β affinity reweight)

**Date:** 2026-07-08
**Motivates from:** `docs/reports/2026-07-08_heldout_grid.md` — across 6 held-out
encoders × 2 graphs × 2 datasets, the strict intersection **B** is consistently a
much cleaner buddy signal than the union **E** (mean B ratio ~0.38–0.51 vs E
~0.58–0.73). Yet init uses **E**. This adds an *option* to lean the init on B.

## Decision (approved)

- **Mechanism: affinity reweight.** Keep E as the graph (connectivity), but multiply
  the spectral **affinity** of edges that are also in B by a factor **β ≥ 1**. β=1 is
  byte-for-byte today's behavior (zero regression risk); higher β pulls strict
  buddies tighter in the Laplacian-Eigenmaps geometry. B ⊆ E, so B-isolated nodes
  (36% on Impressions) keep their E edges and stay connected — no new isolation.
- **Scope: init geometry only.** `buddy_edges.npy` (the training-time smoothness
  regularizer, `cfg.loss.lambda_buddy`) stays = E, unchanged. One variable moves at a
  time, so any retrieval delta is attributable to the init.

## Where affinity is formed

`embedding_methods.spectral_embedding` builds `affinity = 1 − rank_norm_mixed_dist`
on E's edges, symmetrizes, and runs `SpectralEmbedding(affinity="precomputed")`. The
reweight is a one-line boost right after symmetrization:

```
A = (A + Aᵀ)/2                          # affinity on E edges, in [0,1]
if b_edges is not None and β != 1.0:
    A = A + (β − 1.0) * A.multiply(Bm)  # affinities on B edges scaled by β
```

`A.multiply(Bm)` keeps affinity only on B edges (Bm = symmetric binary B); adding
`(β−1)×` those makes them `β×`. Affinities may exceed 1 — fine for a precomputed
affinity (weights need not be ≤1). β=1 ⇒ the boost term is 0 ⇒ identical output.

## Changes (4 files + tests)

1. **`embedding_methods.py`** — `spectral_embedding(..., b_edges: csr_matrix|None =
   None, b_weight: float = 1.0)`; apply the boost above. Defaults = no-op.
2. **`compute_buddies.py`** — `compute_buddy_init(..., b_weight: float = 1.0)`.
   Capture `A_img, A_txt` (already returned by `build_buddy_graphs`), form
   `B = binarize(A_img ∩ A_txt)` when β≠1, pass `b_edges=B, b_weight=β` to
   `spectral_embedding`. B is in the same pre-reorder row order as E — correct.
3. **`init_conditions.py`** — read `b_weight` in `_buddy_cfg` (default 1.0); pass to
   `compute_buddy_init` (full path) and to `spectral_embedding` in the visualize path
   (compute B before the embed call there). Record `b_weight` in `template_config`.
4. **`configs/train/default.yaml`** — add `b_weight: 1.0` under `train.buddies` with a
   one-line comment (B-lean off by default).

## Tests (TDD, added to `test_compute_buddies.py`)

- `test_b_weight_identity` — β=1.0 output is `allclose` to a default call (no-op
  guarantee; the regression gate).
- `test_b_weight_tightens_buddies` — on the two-cluster synthetic, take the actual B
  edges from `build_buddy_graphs`; mean emb distance over B edges at **β=8 ≤ β=1**
  (strict buddies pulled at least as tight, in practice tighter). Sign check, not a
  magnitude target.

## Out of scope

- Hybrid B-skeleton graph and changing the regularizer edges (considered, not chosen).
- Running a training sweep over β — that's the follow-up experiment once the option
  lands and the identity/tightening tests pass. Suggested grid: β ∈ {1,2,4,8}.
- No git commits (standing preference).
