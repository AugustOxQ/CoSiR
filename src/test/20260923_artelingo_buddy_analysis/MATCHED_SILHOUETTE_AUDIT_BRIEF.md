# Brief: matched-protocol silhouette/occupancy audit, buddy vs. PercepT (candidate 1 from the brainstorm memo)

This is a two-script task. Write both; do NOT run either — execution
happens separately, on GPU, outside this task.

## Context

Read, in full, before writing any code:
- `docs/reports/2026-09-26_artelingo_buddy_vs_percept_stage1_report.md`
  (the consolidated investigation report — especially §2's reference table
  and the new note at the top of §6 pointing at the brainstorm memo)
- `docs/reports/2026-09-26_buddy_silhouette_gap_brainstorm.md` (candidate 1,
  "Decide on the Stage 2 objective with a matched audit...")
- `../20260922_percept_topic_pipeline/percept_stage1_faithful_recipe_pilot_report.md`
  — read its full results table. Note precisely: Variant A held-out is
  **emotion AMI 0.1092, genre AMI 0.3288, silhouette 0.5120, verdict
  "Collapsed"** — 50/67 surviving held-out centers received fewer than 1%
  of held-out points, and the held-out **minimum surviving-center count is
  0** (many centers get zero held-out points at all). This run misses the
  project's own Pareto bar *and* fails its own non-collapse criterion. Both
  facts must appear plainly in the new audit's report, not just the raw
  silhouette number.
- `../20260922_percept_topic_pipeline/run_percept_stage1_faithful_recipe_pilot.py`
  in full — specifically `run_seed`, `evaluate_assignments`,
  `initialize_cluster_centers`, and how it imports `base`
  (`run_percept_stage1_pilot.py`) for `build_autoencoder`,
  `soft_assignments`, `prune_centers`. This pilot does **not** currently
  save an npz snapshot of embeddings/assignments — script 1 below fixes
  that without altering its existing behavior or reported numbers.
- `run_attention_h1_baseline_seed_stress_pilot_report.md` and
  `run_attention_h1_embedding_snapshot_pilot.py` (the buddy baseline that
  produces `attention_h1_embedding_snapshot.npz` — confirm its schema by
  reading the script, not by guessing).
- `run_heldout_label_transfer_pilot.py` — reuse `assign_to_train_communities`
  unmodified; this is buddy's validated, deployable held-out label source
  (k=20 majority vote onto the 19 frozen train Leiden communities).
- `../20260922_percept_topic_pipeline/percept_stage1_dec_recon_anneal_snapshot.npz`
  — inspect its keys (a **different, unrelated PercepT variant** — the
  standing K=60/40 balance-hack, not the faithful recipe) purely as a
  schema example for what an npz snapshot of this kind conventionally
  contains (train/heldout paintings, latents, labels, emotion/genre,
  surviving_indices). Do not use this file's data in the audit itself —
  it is the wrong PercepT variant for this comparison.

## Script 1: snapshot the faithful recipe's Variant A (seed 42)

Write
`../20260922_percept_topic_pipeline/run_percept_stage1_faithful_recipe_snapshot_pilot.py`.

Reuse `run_percept_stage1_faithful_recipe_pilot.py`'s module-level setup
(imports, `base` sibling-import, constants, `EMOTION_PARETO_BAR`/
`GENRE_PARETO_BAR`, data loading calls it makes in its own `main()`) and its
`pretrain_autoencoder`, `initialize_cluster_centers`,
`train_dec_until_stable`, `evaluate_assignments`, `base.build_autoencoder`,
`base.soft_assignments`, `base.prune_centers` functions, all unmodified.
`run_seed` itself does not expose the raw latent/assignment arrays it
computes internally (only summary metrics) — write a new function
`run_seed_with_snapshot` that duplicates `run_seed`'s body for a **single
variant, "A" only** (drop the multi-variant loop and the `variants`
parameter), and additionally returns `train_latent`, `train_assignments`,
`heldout_latent`, `heldout_assignments`, and `surviving_indices` as
plain numpy arrays alongside the same `train_metrics`/`heldout_metrics`
dicts `run_seed` already returns. Do not change any hyperparameter,
schedule, or stopping rule — the numeric results must reproduce Variant A's
already-published numbers (emotion AMI 0.1092/genre AMI 0.3288/silhouette
0.5120 held-out at seed 42) as a correctness check.

Save an npz to
`../20260922_percept_topic_pipeline/percept_stage1_faithful_recipe_snapshot.npz`
with these keys (match the existing `dec_recon_anneal` snapshot's naming
convention where the concept overlaps, so a future reader recognizes the
schema): `train_paintings`, `train_latent` (128-D, post-DEC), `train_topic`
(final hard assignments after 67-center pruning), `train_emotion`,
`train_genre`, `heldout_paintings`, `heldout_latent`, `heldout_topic`,
`heldout_emotion`, `heldout_genre`, `surviving_indices`,
`n_initial_clusters`, `n_surviving_clusters`, `seed`, `held_out_silhouette`
(the value reported in the existing report, as a correctness cross-check),
`stop_epoch`, `stop_reason`.

The script's `main()` should: build the identical Variant-A-only run at
seed 42, call `run_seed_with_snapshot`, print the resulting held-out
emotion AMI, genre AMI, and silhouette so they can be checked by eye
against the published 0.1092/0.3288/0.5120 before trusting the snapshot,
then write the npz. If any of those three printed numbers differ from the
published ones by more than a rounding-level tolerance, print a loud
warning (do not raise — the run should still complete and save) so the
discrepancy is visible when the log is read, not silently swallowed.

## Script 2: the matched audit

Write
`src/test/20260923_artelingo_buddy_analysis/run_buddy_percept_matched_silhouette_audit_pilot.py`.

Load both snapshots:
- Buddy: `attention_h1_embedding_snapshot.npz` (train and held-out
  embeddings/paintings/emotion/genre, plus `train_community_post` as the
  frozen train vocabulary).
- PercepT: `../20260922_percept_topic_pipeline/percept_stage1_faithful_recipe_snapshot.npz`
  (script 1's output).

**Confirm, and assert, that both snapshots' `heldout_paintings` arrays are
identical as sets** (same 9,365 held-out paintings) before comparing
anything — if they differ, raise a clear error rather than silently
comparing mismatched populations. Same check for `train_paintings` (61,402).

For **buddy**, compute deployable held-out hard labels via
`assign_to_train_communities(train_embeddings=snapshot["train_embedding_post"],
train_community=snapshot["train_community_post"],
query_embeddings=snapshot["heldout_embedding_post"], k=20)`, reusing the
function unmodified (import it from `run_heldout_label_transfer_pilot.py`
via the standard `load_module` sibling pattern). This is buddy's one
deployable, Stage-2-compatible held-out label source — do **not** also use
`heldout_community_post` (the independent re-clustering) as an alternative
in this audit; the memo is explicit that k=20 transfer is the one to use.

For **PercepT**, use `heldout_topic` directly from its snapshot — it is
already the deployable, train-fitted-center-based held-out label source
(no additional computation needed).

**Compute, for both systems, under an identical protocol:**
1. Held-out emotion AMI and genre AMI (reuse `pipeline.external_metrics`
   and the genre-overlap-index pattern every other pilot in this directory
   uses — import a `pipeline` sibling module the standard way for this).
2. Held-out silhouette, using the **same fixed-seed two-stage sampling**
   convention used throughout this whole investigation: draw
   `min(6000, N)` indices via `np.random.default_rng(42)`, score with
   `silhouette_score(..., sample_size=min(4000, len(idx)), random_state=42)`.
   Apply this **within each system's own embedding space** (32-D buddy
   fused embedding, 128-D PercepT latent Z) — there is no shared space to
   project into, so this cannot be a single number computed once; state
   this limitation plainly in the report rather than implying it is a
   perfectly controlled metric.
3. Cluster occupancy: for buddy, over its 19 train communities; for
   PercepT, over its 67 surviving centers. Report min/max/median count and
   the count of labels with **zero** held-out points assigned, not only the
   below-1% collapse rule (a center can have zero points and still not
   trip a naive "below 1%" check if computed as a fraction including itself
   — count zero-occupancy labels explicitly and separately).
4. Report **both** systems' outcome against this project's own predeclared
   collapse rule (`below_one_percent > N_labels / 2`) side by side with
   both AMI Pareto-bar checks, in one combined table, so a reader sees all
   four judgments (AMI bar, collapse rule, silhouette, occupancy) for both
   systems at a glance rather than needing to cross-reference two report
   files.

## Report

Write
`src/test/20260923_artelingo_buddy_analysis/buddy_percept_matched_silhouette_audit_pilot_report.md`
with: the painting-set-identity check result, the combined comparison
table (both systems × all four judgments), full occupancy histograms
(min/max/median/zero-count/below-1%) for both, and a final verdict section
that states plainly: given matched sampling and side-by-side occupancy,
does PercepT's held-out silhouette advantage still look like a meaningful
difference in topic quality, or does the occupancy picture (PercepT: many
zero-occupancy surviving centers; buddy: reused, previously-reported
occupancy numbers from `heldout_label_transfer_pilot_report.md`) suggest
part of that gap reflects degenerate over-confident clustering rather than
better topics. State explicitly that this audit does **not** attempt to
answer whether either system's labels are more *useful downstream*
(candidate 1's second half, the shared mapper probe, is deliberately out of
scope for this pilot and left for a follow-up decision).

Do not touch git, do not modify any other file in the repository.
