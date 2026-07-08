# Buddy Self-Refresh / Co-Training (Family #3) — Design

**Date:** 2026-07-07
**Status:** Approved, ready for implementation plan
**Branch:** `experiment/conditional_buddy_train`

## Context

This is the **third and last** of three families for using the validated
conditional-buddy signal during CoSiR training (beyond seeding the per-sample
label embeddings at init).

- **Family #1** (buddy-graph smoothness regularizer) — keeps the init geometry
  smooth on `z`. Implemented, shipped on this branch.
- **Family #2** (buddies as contrastive supervision) — uses buddies as extra
  retrieval positives in combined/retrieval space. Implemented, shipped.
- **Family #3** (this design) — the buddy **graph** itself stops being frozen at
  init and is periodically **recomputed from the evolving representations**, so
  the supervision co-trains with the model instead of being pinned to the
  init-time CLIP neighborhood.

Both #1 and #2 reuse a single **frozen** graph: the init-time cross-modal CLIP
mutual-KNN (`E`, persisted as `buddy_edges.npy` in z-table order). Family #3
replaces that static graph, for the #2 term only, with one that is refreshed on
a schedule from the model's current combined features — while keeping the
validated CLIP graph as a permanent anchor.

## Goal

Let the buddy graph feeding Family #2's contrastive term **evolve with the
model**: on a warm-up-then-periodic schedule, recompute a mutual-KNN graph from
the current `comb_emb`, union it with the frozen CLIP graph, and rebuild the CSR
that `buddy_contrastive_loss` reads — all gated default-off, with a blend knob
whose zero setting reproduces Family #2 exactly.

## Key design decisions (locked during brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Refresh source | **Combined / retrieval space** (`comb_emb`) | The space the R1 metric and Family #2 both live in; buddies evolve toward what actually helps retrieval. The only *evolving* representations are `z` and `comb_emb`; `comb_emb` is the one #2 supervises. |
| Consumer | **Family #2's contrastive term** (swap its CSR) | Same space as the refresh source. #3 becomes "#2 with a live graph", isolating "static vs refreshed graph" as the only change vs #2. **No new loss term.** |
| Collapse guard | **Blend with frozen CLIP graph** (`blend=0` ⇒ exactly #2) | Never abandons the validated init edges — refresh only *adds* edges. Plus a structural guard inherited from #2: the pull *target* stays the buddy's **frozen** other-side CLIP feature, so the loss never pulls `comb→comb`. |
| Schedule | **Warm-up `W`, then every `R` epochs** | Refreshing at epoch 0 is a no-op (`z≈init` ⇒ `comb_emb`≈frozen-CLIP-fused ⇒ reproduces the CLIP graph). Warm-up lets the model depart the init first. |
| Gating | Master switch `buddy_refresh`; requires `lambda_buddy_con>0`; guard on `embeddings.requires_grad` | Independently on/off; recommended with Family #1 held off. |

## Why this is a genuinely new hypothesis (vs #1 / #2)

Families #1 and #2 both assert: *the init-time CLIP neighborhood is worth
preserving/supervising throughout training.* Family #3 tests the opposite-leaning
hypothesis: *once the model has learned, its own combined-space neighborhood is a
better (or complementary) source of positives than the frozen CLIP one.* The
blend knob makes this a continuum rather than a binary: `blend=0` trusts CLIP
entirely (= #2), `blend=1` unions in everything the model currently believes,
while **always** retaining the CLIP edges as an anchor.

## Structural anti-collapse guard (inherited from #2, important)

`buddy_contrastive_loss` pulls each anchor's `comb_emb` toward its buddies'
**frozen other-side CLIP feature** through `project_other` — *not* toward another
`comb_emb`. So even when the graph is recomputed from `comb_emb`, the loss target
is anchored to frozen CLIP features. Family #3 makes only the **selection of who
your buddies are** dynamic; the **pull target** stays frozen. This is what keeps
the co-training loop from degenerating into pure `comb→comb` self-reinforcement.
The CLIP-graph blend then further bounds "rich-get-richer" drift in the
selection.

## Architecture & data flow

```
SETUP (once, when buddy_refresh on):
  buddy_edges.npy (CLIP E, z-order) ─► CLIP_edge_index [2, M0]      [reused from #1/#2]
  combine-side pooled feats ────────► combine_feat_table [N, Dfeat] [NEW, sibling of #2's
                                        (z-table order, on device)   other_feat_table]

EACH SCHEDULED REFRESH (epoch >= W and (epoch-W) % R == 0), under no_grad:
  comb_all = model.combine(combine_feat_table, None, z_table)         [N, Dp]  (chunked)
  A_comb   = mutual_knn(comb_all.cpu().numpy(), K=refresh_k)          scipy csr (single space)
  comb_ei  = subsample(upper_coo(A_comb), frac=refresh_blend)         [2, Mc]
  edge_ei  = concat(CLIP_edge_index, comb_ei)                         [2, M0+Mc]  (union; always keeps CLIP)
  buddy_indptr, buddy_indices = build_neighbor_csr(edge_ei, N)        rebind in place

PER BATCH (unchanged from #2):
  buddy_contrastive_loss(comb_emb, anchor_pos, other_feat_table, project_other,
                         other_emb, buddy_indptr, buddy_indices, ...)
```

Index space stays z-table position order end to end: `comb_all` is built in
`embedding_manager.sample_ids` order, so `mutual_knn` row/col indices *are*
z-positions; the CLIP edges are already z-order; the union and the rebuilt CSR
are therefore z-order, matching `anchor_pos = batch_indices` and
`other_feat_table` (the classic index-consistency requirement).

## Components

### 1. Setup (in `train_cosir`, extending #2's buddy block)
- When `buddy_refresh` on (and `lambda_buddy_con>0` and edges present and feature
  store fits in RAM): build **`combine_feat_table`** — the combine-side pooled
  feature for all N in z-table order (`img_features` if `combine_side=="img"`,
  else `txt_features`), via `reorder_features_to_z`, on device, frozen.
  (This is the sibling of #2's `other_feat_table`, which is the *other* side.)
- Keep `CLIP_edge_index` (the z-order edge tensor already loaded at setup) around
  so refresh can union against it.
- Read the config knobs (below) via `getattr`.

### 2. New refresh function (`src/metrics/regularizer.py`, beside #1/#2's)
```
refresh_buddy_graph(
    model,                 # for model.combine + model.project_other side is irrelevant here
    combine_feat_table,    # [N, Dfeat] frozen, z-order (combine-side pooled feature)
    z_table,               # [N, D] current embedding_manager.embeddings (detached inside)
    clip_edge_index,       # [2, M0] frozen CLIP E edges, z-order
    num_nodes,             # N
    k=30,                  # mutual-KNN K on comb space
    blend=1.0,             # fraction of comb edges to union in (0 => CLIP only)
    chunk=4096,            # combine() batch size for the full-N pass
    generator=None,        # for reproducible blend subsample
) -> (indptr, indices, comb_edges, stats)
```
Behavior (all under `torch.no_grad()`):
- Compute `comb_all` by chunked `model.combine(combine_feat_table[s:e], None, z_table[s:e])`.
- `A_comb = mutual_knn(comb_all.float().cpu().numpy(), K=k)`.
- Extract one direction of `A_comb` as an edge list (upper triangle to avoid
  double count; `build_neighbor_csr` re-symmetrizes).
- If `blend < 1`, keep a random `blend` fraction of comb edges (via `generator`);
  if `blend == 0`, keep none. **CLIP edges are always kept in full.**
- `edge_index = cat([clip_edge_index, comb_edges], dim=1)`.
- `indptr, indices = build_neighbor_csr(edge_index, num_nodes)`.
- `stats = {new_edge_frac vs CLIP, n_comb_edges, avg_degree}` (all
  self-contained; `graph_churn` is cross-call so it is computed by the **caller**,
  which retains the previous refreshed edge set — see §3).
- Returns `(indptr, indices, comb_edges, stats)` — `comb_edges` is returned so the
  caller can diff successive refreshes for churn. Grad-safe: `z_table` is detached;
  no autograd tape.

Note: `refresh_buddy_graph` does not need `project_other` or the other-side
table — it only builds the *graph*. The loss target machinery is untouched.

### 3. Wiring (train loop, epoch boundary, before the batch loop)
- Guard: `buddy_refresh` **and** `_lambda_buddy_con>0` **and**
  `other_feat_table is not None` **and** `embedding_manager.embeddings.requires_grad`
  **and** `epoch >= W` **and** `(epoch - W) % R == 0`.
- Call `refresh_buddy_graph(...)`; rebind `buddy_indptr, buddy_indices`.
- Retain the returned `comb_edges` across calls; compute `graph_churn` (Jaccard)
  against the previous refresh's `comb_edges` (first refresh: churn undefined/0).
- Log `stats` + `graph_churn` under a `buddy_refresh` section on the eval/graph cadence.
- The per-batch #2 term (lines ~1571–1587) is **unchanged** — it just reads the
  now-rebound CSR.

### 4. Config (read via `getattr`, not in YAML → add with `+` on the CLI)
| key | default | meaning |
|---|---|---|
| `buddy_refresh` | `False` | master switch for Family #3 |
| `buddy_refresh_warmup` | `50` | first refresh epoch `W` |
| `buddy_refresh_period` | `50` | refresh every `R` epochs |
| `buddy_refresh_blend` | `1.0` | fraction of comb edges unioned (`0` = static #2) |
| `buddy_refresh_k` | `30` | mutual-KNN K for the comb graph (matches init K) |

### 5. Diagnostics
- `graph_churn`: Jaccard between successive refreshed edge sets (thrashing vs
  stabilizing over training).
- `graph_new_edge_frac`: fraction of refreshed edges NOT in the CLIP graph (how
  much the model's neighborhood disagrees with CLIP).
- `graph_avg_degree`: sanity on graph density after union.
- Keep #2's `buddy_con_alignment`.

## Error handling / guards
- `buddy_refresh` on but `lambda_buddy_con == 0` → nothing consumes the graph;
  print a `[buddy-refresh]` warning and disable refresh (no-op).
- No `buddy_edges.npy` / no CLIP edges → #2 already disables; refresh disables
  with it.
- Streaming feature store (no RAM table) → #2's existing guard disables the term;
  refresh cannot build `combine_feat_table` either → disabled. Out of scope.
- `buddy_refresh_blend == 0` (or `buddy_refresh` off) → CSR is the CLIP graph;
  behavior is **byte-for-byte Family #2**.
- Refresh runs under `no_grad`; `z_table` detached ⇒ cannot perturb `z` or leak
  gradient into the optimizer step.

## Testing
New tests under `src/test/20260707_buddy_refresh/`:

1. **Equivalence at blend=0** — `refresh_buddy_graph(..., blend=0)` yields a CSR
   identical (indptr + sorted neighbor sets) to `build_neighbor_csr(clip_edge_index)`
   ⇒ Family #3 at blend=0 ≡ Family #2.
2. **Index alignment** — with a permuted z-order, `comb_all` row `p` corresponds
   to `sample_ids[p]`; the mutual-KNN edges index into z-positions (hand-built
   `combine_feat_table` where the argmax neighbor is known).
3. **Union correctness** — after refresh at `blend=1`, every CLIP edge is present;
   at `blend=0.5` roughly half the comb edges survive (seeded generator, exact
   count check).
4. **No-grad safety** — `z_table.requires_grad` unaffected; a refresh call leaves
   `embedding_manager.embeddings` numerically unchanged and creates no grad_fn.
5. **Schedule gating** — a small epoch-loop harness fires refresh at exactly
   `{W, W+R, W+2R, …}` and never before `W`.
6. **Diagnostics** — `graph_churn` (Jaccard) and `graph_new_edge_frac` compute
   correctly against hand-built previous/CLIP edge sets.

Run: `python src/test/20260707_buddy_refresh/test_buddy_refresh.py`
(needs `PYTHONPATH=/project/CoSiR`).

## Backward compatibility & ablation
- Default-off (`buddy_refresh=False`) ⇒ identical to current behavior.
- `buddy_refresh_blend=0` ⇒ identical to Family #2 (static CLIP graph).
- Neither `buddy_refresh*` key is part of the buddy init **template key**, so all
  arms reuse the **same buddy init template** — every refresh arm and the static
  baseline share one init; only the training-time graph differs.
- Runner: `scripts/run_buddyrefresh_full.sh` (Hydra `-m`), mirroring
  `run_buddycon_full.sh` — holds `lambda_buddy=0`, fixes `lambda_buddy_con=0.3`
  (term on), sweeps `+loss.buddy_refresh=true` with
  `+loss.buddy_refresh_blend ∈ {0, 1.0}` (0 = static #2 baseline, 1.0 = full
  refresh), isolating "static vs refreshed graph."

## Out of scope (explicit)
- Refresh feeding **Family #1** (they share the CSR; refresh would silently change
  #1 too). Recommended config holds #1 off (`lambda_buddy=0`), like
  `run_buddycon_full.sh`.
- Strict-buddy (B / intersection) refresh; only the union-with-CLIP graph is wired.
- EMA/momentum graph (a considered alternative guard) — the CLIP-blend guard was
  chosen instead; EMA is a possible follow-up.
- Accumulating graphs across refreshes — each refresh recomputes from
  `CLIP ∪ comb_t`, not `CLIP ∪ graph_{t-1} ∪ …`.
- Streaming-dataset support (no RAM feature table).
- cuVS/ANN mutual-KNN for ≫150K N (existing seam in `mutual_knn`).
- Refreshing `z` space or `project_other` output as the graph source (combined
  space was the chosen fork).
