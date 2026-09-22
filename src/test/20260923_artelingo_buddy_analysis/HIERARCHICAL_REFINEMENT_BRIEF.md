# Brief: content-first, affect-within-content hierarchical refinement

Write a new script
`src/test/20260923_artelingo_buddy_analysis/run_hierarchical_refinement_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context and why this run matters

Every fusion attempt so far (early/feature fusion, late/union fusion, late
intersection) forces ONE flat Leiden partition to serve two nearly-
orthogonal ground truths (genre AMI on GoEmotions-only is 0.0396, barely
above chance) — so every result so far is a point on one trade-off frontier,
not an escape from it (`affect_pilot_report.md`, `late_fusion_pilot_report.md`).

This pilot tests a genuinely different mechanism, from a two-way brainstorm
with an independent Codex pass
(`fusion_brainstorm_codex_findings.md`): **content-first hierarchical
refinement**. Run Leiden on the content graph alone first (this exactly
reproduces the 0.4384 genre-AMI reference partition, untouched). Then,
*within* each resulting community (never merging across communities), run a
second Leiden pass using only the affect signal restricted to that
community's own nodes. The final label is a composite
`(content_parent_id, affect_child_id)`. This structurally forbids the
specific failure mode seen in both fusion pilots: an affect edge merging two
different-genre communities together.

**Important, Codex-caught correction — do not skip the controls below.** AMI
is NOT monotone under refinement: splitting a cluster into smaller pieces
can lower or raise its chance-corrected AMI even with zero real signal in
the split. So a genre-AMI change or an emotion-AMI change after refinement
does NOT by itself prove affect added real information — it could just be a
granularity artifact. This pilot MUST report two controls at exactly
matched split sizes (see below), and the report's conclusion must be framed
around the *comparison to those controls*, not around the raw hierarchical
numbers in isolation.

Read `run_pipeline.py`, `run_affect_pilot.py`, and `run_single_modality_pilot.py`
in full first, for conventions and reuse. Match their `log()` timestamped
progress-printing convention throughout.

## Implementation

1. Import `run_pipeline.py` and `run_affect_pilot.py` as sibling library
   modules via the established `load_sibling_module()` pattern. Import
   `build_buddy_graphs` from `src.conditional_buddy.compute_buddies` and
   `detect_communities` from `src.conditional_buddy.prototype_seed`, and
   `ensure_min_degree`/`ensure_connected`/`mutual_knn` from
   `src.conditional_buddy.buddy_graph`.

2. Load and dedup features via `pipeline.assert_extraction_complete()` /
   `pipeline.load_dedup_features()`. Get `majority_emotion`. Extract
   GoEmotions nodes via `affect_pilot.extract_affect_nodes(pipeline.TRAIN_JSON,
   paintings, device)`, then L2-normalize via `affect_pilot.l2_normalize()`.

3. **Content-parent partition**: `A_img, A_txt, E_content = build_buddy_graphs(
   img_nodes, txt_nodes, K=pipeline.K, alpha=pipeline.ALPHA, device=device,
   connect_components=True)`, then `parent_labels = detect_communities(
   E_content, seed=42)`. This is the reference content-only partition
   (should reproduce emotion AMI=0.0593 / genre AMI=0.4384 if computed
   directly on `parent_labels` alone — compute and log this as a
   reproduction check before doing anything else, same convention as
   `run_affect_pilot.py`'s CLIP-only reproduction check).

4. **Constants** (predeclare, do not tune after seeing results):
   ```python
   MIN_PARENT_SIZE = 50
   STABILITY_SEED_B = 43
   STABILITY_ARI_THRESHOLD = 0.5
   SUCCESS_MARGIN_AMI = 0.02       # hierarchical must beat Control A by this much
   GENRE_RETENTION_FLOOR = 0.3507  # 80% of the 0.4384 content-only genre baseline
   ```

5. **Per-parent affect sub-clustering**, for each unique value of
   `parent_labels`:
   - Let `N_p` = number of nodes in this parent. If `N_p < MIN_PARENT_SIZE`,
     this parent is INELIGIBLE: every member gets child id `0` (unsplit),
     skip to the next parent.
   - Otherwise build a sub-graph from this parent's own affect feature rows
     only: `K_sub = min(pipeline.K, N_p - 1)`, `mutual_knn(affect_rows,
     K=K_sub, device=device)`, then repair with `ensure_min_degree` /
     `ensure_connected` using the same "pass the affect array into both the
     img_feats and txt_feats argument slots" trick already used by
     `build_single_modality_graph` (read `run_single_modality_pilot.py`'s
     `build_single_modality_graph` for the exact reference pattern — you may
     import and call it directly instead of re-deriving the repair calls, by
     passing this parent's affect-row subset as `nodes` and this parent's
     `N_p` as `expected_nodes`).
   - Run `detect_communities(sub_graph, seed=42)` → `child_labels_seed42`
     (local ids, 0-indexed within this parent).
   - **Stability check**: also run `detect_communities(sub_graph, seed=
     STABILITY_SEED_B)` → `child_labels_seed43`. Compute
     `sklearn.metrics.adjusted_rand_score(child_labels_seed42,
     child_labels_seed43)`. If this ARI is `< STABILITY_ARI_THRESHOLD`, this
     parent's split does NOT stick: revert every member of this parent to
     child id `0` (unsplit) — treat it the same as an ineligible parent from
     this point forward (do not use `child_labels_seed42` for it in any
     variant below).
   - If stable, this parent's real affect split is `child_labels_seed42`,
     with however many distinct child ids Leiden found (could be 1, meaning
     it didn't actually split even though it was eligible — that's fine, log
     it, it still counts as "eligible" for control purposes only if it
     produced >=2 children; if it produced exactly 1 child, treat as
     unsplit/ineligible for the controls below too, since there's nothing to
     compare against).
   - Track, per parent: eligible (Y/N), stable (Y/N, only meaningful if
     eligible), final number of children, child size list.

6. **Control A — random split, exactly size-matched**: for every parent that
   ended up genuinely split (eligible, stable, >=2 children) in step 5, take
   its real affect child-label array (local to that parent's node order) and
   apply `numpy.random.default_rng(42).permutation()` to shuffle which node
   gets which child id, preserving the exact multiset of child sizes exactly
   (permuting the label array itself accomplishes this trivially — do not
   regenerate sizes independently). Every other parent (ineligible or
   reverted-unstable) keeps child id `0`, exactly as in the real hierarchical
   result.

7. **Control B — content re-split (not size-matched, report as such)**: for
   the same set of genuinely-split parents, build a sub-graph from this
   parent's own CLIP content features instead (reuse both `img_nodes` and
   `txt_nodes` restricted to the parent, through `build_buddy_graphs` again
   at `K_sub` on just this subset), run `detect_communities(seed=42)` on it,
   and use however many children it naturally finds — log this count next to
   the real affect split's count for each such parent so the size mismatch
   is visible, don't force them to match.

8. Build three final composite integer label arrays over all 61,402 nodes —
   hierarchical (real affect split), Control A (random, matched), Control B
   (content re-split) — each via `f"{parent_id}_{child_id}"` string keys
   factorized to integers (e.g. `numpy.unique(..., return_inverse=True)`).
   Ineligible/reverted-unstable parents contribute the same `f"{parent_id}_0"`
   key in all three variants.

9. Compute emotion/genre AMI + V-measure for `parent_labels` alone (reference
   check), and for all three composite variants, via `pipeline.
   external_metrics()` and `pipeline.load_genre_map()` (reuse
   `run_single_modality_pilot.py`'s `evaluate_graph()` directly for the
   composite variants).

## Report

Write
`src/test/20260923_artelingo_buddy_analysis/hierarchical_refinement_pilot_report.md`:

- Explain the mechanism in plain language, and explicitly restate the
  AMI-non-monotonicity caveat up front — say plainly that raw hierarchical
  numbers alone cannot answer the question, only the comparison to Control A
  can.
- Report the content-only reproduction check (parent_labels alone vs. the
  0.4384/0.0593 references).
- Report split diagnostics: total number of content parents, how many were
  eligible (>= MIN_PARENT_SIZE), how many of those were stable (ARI >=
  0.5) and genuinely split (>=2 children), the fraction of the 61,402 total
  nodes that ended up in a genuinely-split parent, and the child-size
  distribution (min/median/max) across genuinely-split parents.
- A comparison table:

  | signal | emotion AMI | emotion V-measure | genre AMI | genre V-measure |
  |---|---:|---:|---:|---:|
  | Content-only (reference) | 0.0593 | 0.0600 | 0.4384 | 0.4572 |
  | Content-parent partition (this run's reproduction check) | *computed* | *computed* | *computed* | *computed* |
  | Hierarchical (content-parent + affect-child) | *computed* | *computed* | *computed* | *computed* |
  | Control A — random split (matched sizes) | *computed* | *computed* | *computed* | *computed* |
  | Control B — content re-split (not size-matched) | *computed* | *computed* | *computed* | *computed* |

- A conclusion paragraph structured around the predeclared decision rule,
  stated explicitly: hierarchical refinement shows **real, not granularity-
  driven, emotion signal** only if `hierarchical_emotion_AMI - control_A_emotion_AMI
  >= 0.02`. Separately state whether `hierarchical_genre_AMI >= 0.3507` (the
  80%-retention floor). Report all four possible outcomes plainly (real
  signal + genre retained / real signal but genre lost / no real signal
  above Control A / etc.) — do not force a "yes/no" framing if the truth is
  mixed.
- A short paragraph comparing Control B's natural child counts to the real
  affect split's child counts per parent (summary statistic, e.g. mean
  absolute difference), to make the "not size-matched" caveat concrete
  rather than just asserted.

Print clear timestamped progress logs matching the other scripts' `log()`
format throughout, including per-parent eligible/stable/split counts as they
accumulate (a running tally every ~20 parents processed is fine, matching
the batching-progress-log convention already used elsewhere).
