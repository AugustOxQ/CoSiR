# Brief: candidate 1 — minimum-occupancy handling for buddy's Stage 2 targets

## Context

`deep_stage_analysis_report.md` (this directory) found buddy's Stage 2
macro AUC is significantly correlated with per-topic occupancy (Pearson
r=0.531, p=0.019, n=19) — buddy's three smallest held-out topics (indices
16/17/18, sizes 87/69/65 of 9,365) likely drag the macro-AUC average down.
This is the top-priority candidate from that analysis's updated list. Goal:
test two independent, cheap fixes and see whether either meaningfully
raises buddy's Stage 2 macro AUC (currently 0.5978, single seed 42) above
its current level, screening at seed 42 only first (this project's
established discipline: screen cheap before stress-testing further).

## What to build

New script `src/test/20260927_deep_stage_analysis/run_candidate1_min_occupancy_pilot.py`.
Reuse buddy's existing frozen embedding snapshot
(`src/test/20260923_artelingo_buddy_analysis/attention_h1_embedding_snapshot.npz`,
fields: `train_paintings, train_embedding_post, train_community_post,
heldout_paintings, heldout_embedding_post`, `seed`) and reuse
`run_heldout_label_transfer_pilot.py::assign_to_train_communities` for the
k=20 held-out label transfer, and `run_percept_stage2_pilot.py`'s
`AttentionPoolingMapper`, `load_patch_features`, `evaluate_auc`,
`auc_summary` (unmodified, image-only patch-attention mapper — same
classes/functions `run_buddy_stage2_pilot.py` and this project's whole
Stage 2 investigation already use) — do not reimplement any of this, only
import. Read `run_buddy_stage2_pilot.py` in full first for the exact
existing baseline pipeline (targets, training loop, evaluation) to match
exactly except for the two variants below.

Baseline to reproduce first, as a sanity check: buddy's existing Stage 2
run gets macro AUC 0.5978 (seed 42, `BCEWithLogitsLoss` against one-hot
19-way targets). Confirm this reproduces within 0.001 before testing the
two variants; if it doesn't, stop and report the discrepancy.

### Variant A — class-balanced (inverse-frequency) loss weighting

Same 19 topics, same one-hot targets, no merging. Compute each topic's
train frequency from `train_community_post`, set per-topic BCE positive
weight inversely proportional to frequency (normalized so the mean weight
across topics is 1, to keep the loss scale comparable to the baseline —
e.g. `weight[t] = (N_train / 19) / count[t]`, then optionally renormalize).
Use `BCEWithLogitsLoss(pos_weight=...)` or an equivalent manual per-topic
weighting — check which is correct given this is a one-hot (not general
multi-label) target in practice, since `pos_weight` only reweights the
positive-class term; if that's not the right lever for a near-one-hot
setup, weight each sample's total loss by its target topic's inverse
frequency instead (a per-sample weight, not per-class `pos_weight`) — use
judgement here and state clearly in the report which mechanism was used
and why.

### Variant B — merge small train communities before freezing targets

Compute train community sizes from `train_community_post` (19 communities,
61,402 train paintings). Identify communities below 1% of train (614
paintings) — this project's own established "below 1%" occupancy
convention, used throughout this investigation's occupancy tables. For
each such small community, compute its centroid (mean of its members'
`train_embedding_post` vectors, L2-renormalized) and merge it into the
single nearest OTHER (non-small) community by cosine centroid similarity —
i.e. relabel every painting in the small community to its nearest larger
neighbor's label. Repeat until no communities remain below the 1%
threshold (a small community's members could end up merged into another
originally-small one if that's genuinely closest; recompute centroids
after each merge round, or do a stable single-pass since 19 is small — use
judgement, but state the exact procedure used). This produces a new,
smaller vocabulary (K < 19) for BOTH train and held-out (transfer held-out
labels via k=20 exactly as before, but onto this new, merged train
vocabulary). Retrain the same mapper architecture with a head sized to the
new K.

## Evaluation

For baseline, Variant A, and Variant B, report: macro AUC, min/median/max
per-topic AUC, and overall top-1 accuracy (predicted argmax topic ==
true hard topic) — reuse the exact same top-1 accuracy definition
`deep_stage_analysis_report.md`'s script used (argmax of mapper score
vector, compared to the frozen hard topic label). Also recompute the
AUC-vs-occupancy Pearson/Spearman correlation for each variant, to check
whether the fix actually reduced the size-dependence, not just shifted
the average.

## Output

`run_candidate1_min_occupancy_pilot.py` and
`candidate1_min_occupancy_pilot_report.md` (this directory), with:
- Baseline reproduction check
- A results table: variant, K (topic count), macro AUC, min/median/max
  AUC, top-1 accuracy, AUC-occupancy Spearman r
- A plain verdict: does either variant meaningfully beat the 0.5978
  baseline (predeclared practical margin: +0.005, matching this
  investigation's established 0.01-ish margins scaled to this smaller
  expected effect size — state this margin explicitly and use it
  consistently)? If yes, name which variant and by how much. If no, say so
  plainly and note this closes candidate 1 as tested (do not silently try
  more variants beyond A and B).

## Constraints

- Local GPU only, single seed 42, this is a cheap screening pilot — no
  DAS6, no multi-seed stress here (seed-stress is a separate, later
  candidate in the list if this one shows promise).
- Do not modify any existing file (buddy's snapshot, `run_buddy_stage2_pilot.py`,
  `run_percept_stage2_pilot.py`, `run_heldout_label_transfer_pilot.py`) —
  read-only imports only.
- Do not touch PercepT's side of anything — this candidate is entirely
  about improving buddy's own Stage 2 mapper.
