# Brief: PercepT Stage 2 — extended seed count for a real confidence interval

Write a new script
`src/test/20260922_percept_topic_pipeline/run_percept_stage2_extended_seed_pilot.py`.
Do NOT run it — this will be launched separately on a DAS6 cluster node.

## Context

Read `run_percept_stage2_best_config_stress_pilot.py` and
`percept_stage2_best_config_stress_pilot_report.md` in full first (both in
this directory), plus `docs/reports/2026-09-23_artelingo_percept_stage2_report.md`.

Stage 2's winning configuration (`q > 1.2/40` multi-label targets, `lr=3e-3`,
100 epochs) reached held-out macro AUC 0.8256 (4-seed range 0.8248-0.8272:
seeds 42, 7, 123, 2024). Mirror what the Stage 1 extended-seed pilot is
doing for Stage 1: extend to a much larger seed sample for a real
confidence interval, not just a 4-seed anecdote.

## What to do

Reuse `run_percept_stage2_best_config_stress_pilot.py`'s exact structure:
ONE shared Stage-1 K=60/40 seed-42 re-fit (with the same reproduction check
against 0.1238/0.2617 — stop if it fails), ONE frozen `q > 1.2/40`
multi-label target set derived from it, cached patch features loaded once.
Import and reuse `AttentionPoolingMapper`/`train_and_evaluate_mapper()`
from `run_percept_stage2_sweep_pilot.py` exactly as the best-config-stress
script already does — do not reimplement.

Run **10 NEW mapper-init seeds**: `SEEDS = (1, 2, 3, 4, 5, 6, 8, 9, 10, 11)`
(same set used by the Stage 1 extended-seed pilot, for consistency across
this investigation — skip 7 since it's already covered). Combine with the
4 already-established seeds (42, 7, 123, 2024 — CITE their held-out macro
AUC values from `percept_stage2_best_config_stress_pilot_report.md`, do not
recompute) for a full **14-seed** picture. Keep `lr=3e-3`, `q > 1.2/40`,
100 epochs fixed throughout — only mapper-init seed varies.

## Report

Write
`src/test/20260922_percept_topic_pipeline/percept_stage2_extended_seed_pilot_report.md`
with:
- The shared Stage-1 reproduction check.
- A full 14-seed results table (source: cited vs newly measured, held-out
  macro AUC, min/median/max per-topic AUC, verdict).
- Summary statistics across all 14: mean, min, max, standard deviation for
  held-out macro AUC.
- A simple normal-approximation 95% confidence interval (mean ± 1.96 *
  stdev/sqrt(14)) for held-out macro AUC, and an explicit comparison
  against the 0.5000 baseline (the interval's lower bound should be
  dramatically above baseline given the 4-seed range was already
  0.8248-0.8272 — state this plainly, this pilot exists to confirm that
  robustness on a larger sample, not to find a new result).
- A final verdict using this investigation's established language.
