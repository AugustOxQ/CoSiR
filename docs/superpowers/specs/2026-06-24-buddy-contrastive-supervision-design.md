# Buddy Contrastive Supervision (Family #2) — Design

**Date:** 2026-06-24
**Status:** Approved, ready for implementation plan
**Branch:** `experiment/conditional_buddy_train`

## Context

This is the **second** of three families for using the validated conditional-buddy
signal during CoSiR training (beyond seeding the per-sample label embeddings at
init). Family #1 (buddy-graph smoothness regularizer) is implemented and shipped on
this branch. Family #2 escalates from "keep the buddy geometry smooth" to "use
buddies as actual retrieval supervision." Family #3 (self-refreshing buddies /
co-training) is a separate follow-up.

**Why combined space, not condition (z) space.** Family #1 already pulls the z-space
lever (smoothness on the per-sample label embeddings along the buddy graph). A
contrastive loss on z would just be a stronger version of the same lever and could
not test a genuinely new hypothesis. Family #2 instead acts in **combined /
retrieval space** — where the evaluation metric (t2i / i2t R1) actually lives — so
it can move the metric in a way #1 may not. This was the explicit design fork chosen
during brainstorming.

## Goal

Add an independently-gated InfoNCE term so that each anchor's fused feature
(`comb_emb`) retrieves not just its own image but its **buddies' images** too —
turning the buddy graph into extra positives for the retrieval objective.

## One-line mechanism

For each batch anchor, pull its `comb_emb` toward its buddies' **projected
other-side features** (the exact space the main retrieval InfoNCE uses), using the
in-batch other-side features as negatives, with a temperature-scaled multi-positive
(SupCon) softmax.

## Key design decisions (locked during brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Space | **Combined / retrieval space** (not z) | Acts where the metric lives; tests a new hypothesis vs #1 |
| Positive target | **Buddy's other-side feature → `project_other`** | Same space as the retrieval positive `other_emb`; frozen underlying feature ⇒ stable, no collapse risk |
| Loss form | **Multi-positive InfoNCE (SupCon)** | Canonical contrastive supervision; temperature + many negatives tolerate looser positives |
| Negatives | **In-batch `other_emb`** (anchor's own row masked) | Free (already computed by the main path); matches the main loss's negative distribution |
| Graph | **E (union)** — reuse #1's `buddy_edges.npy` | No new persistence; holding the graph constant isolates "smoothness vs contrastive" as the only difference between #1 and #2 |
| Gradient flow | Updates `z_i`, combiner, `project_other`; buddy `z_j` untouched | Targets are frozen features through a near-identity-init linear ⇒ stable supervision |
| Gating | Separate `lambda_buddy_con`; guard on `embeddings.requires_grad` | Independently ablatable / combinable with #1; parallel to #1 for a clean comparison |

## How the retrieval loss is actually wired (grounding)

The main loss (`LabelContrastiveLoss_enhance.forward`) computes
`cos_pos = sim(combined_features, image_features)` where, in the train loop, the
`image_features` argument is `other_emb = model.project_other(loss_img_target)` and
`loss_img_target` is the **non-combine-side** pooled feature:

- `combine_side == "img"` → combined built from image; retrieval target = `project_other(txt_features)`; buddy positive = `project_other(txt_j)`.
- `combine_side == "txt"` → combined built from text; retrieval target = `project_other(img_features)`; buddy positive = `project_other(img_j)`.

So the buddy positive target is the buddy's **other-side** pooled feature passed
through `project_other`. The underlying pooled feature is frozen; `project_other` is
a trainable (identity-init) linear layer.

## Feasibility (verified against the codebase)

- **Combiner callable standalone:** `model.combine(emb, emb_full, labels, …)` already
  exists and is used standalone in the train loop. (Not even needed for buddies under
  the chosen positive target — see below — but confirms the architecture is separable.)
- **`CombinerGated.forward` ignores `text_full`** (`# unused`) — only pooled feature +
  `z` matter. The chosen positive target needs **no combiner forward on buddies at
  all**: the buddy target is a frozen pooled feature through `project_other` (a pure
  gather + light linear).
- **Pooled features gatherable by sample id in RAM:** `CoSiRShardDataset` loads the
  full feature set into RAM (`feature_manager.load_all_to_ram(...)` → `self._data`).
  Building `other_feat_table` is a one-time gather. The streaming dataset
  (`CoSiRShardStreamDataset`) has no full RAM table → out-of-scope (warn + disable).

## Architecture & data flow

```
buddy_edges.npy (E, z-order) ──► CSR indptr/indices                 [reused from #1]
other-side pooled feats ───────► other_feat_table [N, Dfeat]
                                  (z-table order, on device, ~24MB for impressions)  [new, once]

per batch:
  comb_emb [B, Dp], other_emb [B, Dp]            (already computed by the main path)
  buddy_pos = sample ≤K buddies per anchor from CSR
  pos_targets = normalize(project_other(other_feat_table[buddy_pos]))   [B, K, Dp]
  L_buddy_con = SupCon(comb_emb ; pos_targets ; other_emb negatives)    scalar
  loss += lambda_buddy_con * L_buddy_con
```

Index space is consistent with #1: edges/CSR are in z-table position order
(`embedding_manager.sample_ids`), `anchor_pos = batch_indices` (already z-order), and
`other_feat_table` is built in z-order by gathering from the RAM feature table via
`sample_ids`.

## Components

### 1. Setup (in `train_cosir`, refactoring #1's buddy block)
- Load edges + build CSR when **either** `lambda_buddy > 0` **or**
  `lambda_buddy_con > 0` (today it triggers on #1 only).
- Build `other_feat_table`: gather the non-combine-side pooled feature for all N
  samples into a `[N, Dfeat]` device tensor in z-table order. Built only when
  `lambda_buddy_con > 0`. Frozen (no grad).

### 2. New loss function (`src/metrics/regularizer.py`, beside #1's)
```
buddy_contrastive_loss(
    comb_emb,            # [B, Dp]  anchor combined features (grad)
    anchor_pos,          # [B]      z-table positions of the batch
    other_feat_table,    # [N, Dfeat] frozen, z-order
    project_other,       # callable: [*, Dfeat] -> [*, Dp]
    other_emb_neg,       # [B, Dp]  in-batch negatives (the batch's other_emb)
    indptr, indices,     # CSR over E (z-order)
    num_pos=4,
    temperature=0.07,
    generator=None,
) -> scalar
```
- Degree-mask anchors with no buddy (skip); if no active anchor, return
  `comb_emb.sum() * 0.0` (grad-safe zero, same idiom as #1).
- Sample ≤`num_pos` buddy positions per active anchor from CSR (uniform, same
  clamp-by-degree sampling as #1's smoothness loss).
- Positives `= normalize(project_other(other_feat_table[buddy_pos]))`.
- Negatives `= normalize(other_emb_neg)` with **the anchor's own batch row masked**
  (its own retrieval positive belongs to the main loss, not here).
- Multi-positive SupCon:
  `L_i = −(1/K) Σ_k log[ exp(s_ik/τ) / (Σ_k exp(s_ik/τ) + Σ_neg exp(s_in/τ)) ]`,
  mean over active anchors. `comb_emb` is L2-normalized before the dot products.

### 3. Wiring (train loop, after the main loss, mirroring #1's term)
- Guard: `lambda_buddy_con > 0` **and** CSR present **and**
  `embedding_manager.embeddings.requires_grad`.
- `loss = loss + lambda_buddy_con * buddy_con_loss`; `loss_dict["loss_buddy_con"] = buddy_con_loss.detach()`.

### 4. Config (read via `getattr`, not in YAML → add with `+` on the CLI)
- `lambda_buddy_con` (default `0.0`)
- `buddy_con_samples` (default `4`)
- `buddy_con_temperature` (default `0.07`)

### 5. Diagnostic
- Log `buddy_con_alignment` (mean cosine between each anchor's `comb_emb` and its
  buddy positives) on the eval cadence — the #2 analogue of #1's `drift_from_init`,
  so the post-sweep analysis can separate "active" from "inert."

## Error handling / guards

- `lambda_buddy_con > 0` but no `buddy_edges.npy` → print a `[buddy-con]` warning and
  disable the term (mirror #1's fallback).
- No-buddy anchors → masked to zero contribution.
- Streaming dataset (no RAM feature table) → warn + disable; document as out-of-scope.
- `lambda_buddy_con = 0` (default) → no `other_feat_table` build, no term, byte-for-byte
  unchanged pipeline. Fully independent of #1.

## Testing

New tests under `src/test/20260624_buddy_contrastive/`:

1. **Positive-gather correctness** — `other_feat_table[buddy_pos]` returns the
   features at the right z-positions for a hand-built CSR.
2. **Gradient direction** — one optimizer step on a toy batch raises mean
   anchor↔buddy similarity relative to anchor↔negative similarity.
3. **Isolated anchor** — an anchor with no buddy contributes exactly zero.
4. **Self-masking** — the anchor's own row is excluded from its negative set.
5. **Index alignment** — `other_feat_table[p]` corresponds to `sample_ids[p]` after a
   permuted build.
6. **Shape / temperature sanity** — output is a finite scalar; temperature scales
   logits as expected.

Run: `python src/test/20260624_buddy_contrastive/test_buddy_contrastive.py`.

## Backward compatibility & ablation

- Default-off (`lambda_buddy_con = 0`) ⇒ identical to current behavior.
- Independent of `lambda_buddy` ⇒ supports a clean 2×2 (or sweep) ablation: #1 alone,
  #2 alone, both, neither — with the **same E graph and same buddy init template**
  (neither lambda is part of the template key), so every arm reuses one init.
- Sweep axis mirrors the v4 pattern: add `+loss.lambda_buddy_con` (and optionally
  `+loss.lambda_buddy`) to the existing `run_sweep_agent.py` / `sweep_config` machinery.

## Out of scope (explicit)

- Condition-space (z) contrastive — superseded by the combined-space choice.
- Strict-buddy (B / intersection) graph and its persistence.
- Buddy-combined-feature positives (the moving-target SupCon variant).
- Streaming-dataset support for `other_feat_table`.
- Family #3 (self-refreshing buddies / co-training) — separate follow-up.
