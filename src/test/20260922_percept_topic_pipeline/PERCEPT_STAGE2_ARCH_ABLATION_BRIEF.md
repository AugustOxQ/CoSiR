# Brief: PercepT Stage 2 — does patch attention specifically matter?

Write a new script
`src/test/20260922_percept_topic_pipeline/run_percept_stage2_arch_ablation_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context: the open question this closes

Read `run_percept_stage2_sweep_pilot.py`,
`run_percept_stage2_best_config_stress_pilot.py`, and
`percept_stage2_best_config_stress_pilot_report.md` in full first (all in
this directory).

The winning Stage-2 configuration (`q > 1.2/40` targets, `lr=3e-3`, 100
epochs) reached held-out macro AUC 0.8256 (4-seed mean, range
0.8248-0.8272) using `AttentionPoolingMapper` — one learned query doing
attention over 50 CLIP patch tokens, then a linear multi-label head. It is
not yet established whether the PATCH-LEVEL ATTENTION specifically drives
this result, or whether any image-only classifier on this frozen target set
would do about as well (i.e., is the win from "the image contains real
signal for these frozen topics," which a much simpler classifier would
also capture, or specifically from attention over patches).

## What to build

A single linear classifier — `nn.Linear(512, 40)` — applied directly to the
EXISTING, ALREADY-CACHED global pooled CLIP image embedding for each
painting (reuse `load_dedup_features()`'s own `img_nodes` array from the
base pilot module — the SAME array Stage 1's own `content_features()`
already uses — L2-normalize it, matching that established convention). No
patch tokens, no attention, no new feature extraction needed — this
reuses infrastructure that already exists in `run_percept_stage1_pilot.py`.

## Sweep structure

Reuse the exact Stage-1 re-fit code (K=60/40, seed 42, same reproduction
check against 0.1238/0.2617 held-out emotion/genre AMI) — fit ONCE, shared
across both this ablation and (implicitly) comparable to the already-
established attention-pooling result. Derive the SAME `q > 1.2/40`
multi-label targets from this frozen fit (identical threshold, so the
comparison is apples-to-apples on targets).

Train the plain linear classifier with the SAME training recipe as the
winning attention-pooling configuration — `lr=3e-3`, 100 epochs, full-batch,
`BCEWithLogitsLoss`, `Adam` — across the SAME 4 seeds used throughout this
investigation (`SEEDS = (42, 7, 123, 2024)`, re-seeding
`torch.manual_seed`/CUDA immediately before constructing the linear
classifier each time). Evaluate held-out macro AUC identically (reuse
`evaluate_auc()`/`auc_summary()` from `run_percept_stage2_sweep_pilot.py`
via import, do not reimplement) plus the same train-marginal baseline for
context.

## Report

Write
`src/test/20260922_percept_topic_pipeline/percept_stage2_arch_ablation_pilot_report.md`
with:
- The shared Stage-1 reproduction check.
- A 4-seed results table for the plain-linear-on-pooled-embedding
  classifier (held-out macro AUC, min/median/max per-topic AUC, skipped
  topics) plus mean/min/max summary.
- A direct side-by-side comparison table against the attention-pooling
  result already established (cite `percept_stage2_best_config_stress_pilot_report.md`'s
  4-seed mean/range: 0.8256 mean, 0.8248-0.8272 range — do not retrain
  that configuration here).
- An honest, explicit verdict: does patch attention pooling meaningfully
  outperform the plain pooled-embedding linear baseline (state the actual
  gap, and whether the two seed ranges overlap or are cleanly separated —
  use the same non-overlapping-range robustness standard already
  established in this investigation), or is the earlier result mostly
  explained by "the frozen topics are recoverable from the image at all,"
  with attention pooling contributing a smaller or negligible additional
  gain? Do not default to assuming the fancier architecture must be
  responsible — let the numbers decide, and say so plainly either way.
