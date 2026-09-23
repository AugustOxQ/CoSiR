# Brief: PercepT Stage 2 — mapper-seed stress test + hyperparameter sweep

Write a new script
`src/test/20260922_percept_topic_pipeline/run_percept_stage2_sweep_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context: what the smoke test found, and what this sweep must answer

Read `src/test/20260922_percept_topic_pipeline/run_percept_stage2_pilot.py` and
`percept_stage2_pilot_report.md` in full first (both in this directory).

The smoke test reproduced Stage 1's K=60/40 seed-42 result exactly (0.0000
difference on both held-out AMIs), trained cleanly, and beat the train-
marginal baseline (macro AUC 0.5690 vs 0.5000, held-out). Two open questions
before trusting this as a real result:

1. **Mapper-seed robustness.** The smoke test used one mapper weight
   initialization. Does the +0.069 macro-AUC improvement hold across
   different mapper-init seeds, holding the frozen Stage-1 targets and patch
   features fixed, or was it a lucky init — the exact same question this
   investigation already asked (and answered "no" or "yes" depending on the
   config) for Stage 1 itself?
2. **The multi-label threshold never fired.** `q > 2.0/40` produced 0%
   multi-labeled paintings — Stage 2 trained as an effectively single-label
   classifier, not the genuinely multi-label setup PercepT intends. Test
   whether a lower, still-explainable threshold (`q > 1.5/40` and
   `q > 1.2/40` — 1.5x and 1.2x uniform probability, versus the original
   2x) produces real multi-label targets, and whether that changes results.

## Efficiency: re-fit Stage 1 exactly once, reuse for every Stage-2 variant

Stage 1 (K=60/40, seed 42) is now fixed and frozen — it does not depend on
any Stage-2 hyperparameter. Refit it ONCE at the start of this script (reuse
`run_percept_stage2_pilot.py`'s exact re-fit code and its reproduction check
against the established 0.1238/0.2617 numbers — stop if it does not
reproduce, exactly as that script already does), obtaining one frozen
encoder + 40 surviving centers + patch feature tensors. Compute multi-label
targets for each of the 3 thresholds (2.0/40, 1.5/40, 1.2/40) from this ONE
frozen fit — do not refit Stage 1 per threshold or per Stage-2 sweep point;
only the target-derivation threshold and the mapper's own training
hyperparameters vary from here on.

## Sweep structure

**Part A — mapper-seed stress test at the ORIGINAL threshold (2.0/40,
lr=1e-3, 100 epochs, matching the smoke test exactly):** train the
`AttentionPoolingMapper` (reuse the exact class from
`run_percept_stage2_pilot.py`) at 4 seeds — `torch.manual_seed` set
immediately before constructing the mapper, same seeds used throughout this
investigation: `SEEDS = (42, 7, 123, 2024)` (42 here is a NEW mapper-init
draw, not literally the smoke test's exact run — note this explicitly, do
not claim it's a citation of the smoke test's number). Report held-out
macro AUC per seed, plus mean/min/max — same rigor as every other seed
stress test in this directory: state plainly whether the improvement over
baseline (0.5000) holds in all 4 seeds or is seed-dependent.

**Part B — multi-label threshold comparison at seed 42 (mapper init),
lr=1e-3, 100 epochs:** train and evaluate at thresholds 1.5/40 and 1.2/40
(2.0/40 is already covered by Part A's seed-42 point — cite it, do not
retrain). For each, report the same multi-label target statistics table
already established (mean/median/max labels, fraction multi-labeled) so the
reader can see whether these thresholds actually produce multi-label data,
and the held-out macro AUC. State plainly whether a lower threshold changes
the result meaningfully or not.

**Part C — learning-rate mini-sweep at the threshold selected from Part B
(prefer whichever threshold produces genuine multi-label targets AND the
best or comparable macro AUC; if none differ meaningfully from 2.0/40,
keep 2.0/40), seed 42 (mapper init), 100 epochs:** train at
`lr ∈ {3e-4, 1e-3 (cite from Part B/A, do not retrain), 3e-3}`. Report held-
out macro AUC for each.

## Report

Write
`src/test/20260922_percept_topic_pipeline/percept_stage2_sweep_pilot_report.md`
with:
- Confirmation of the single shared Stage-1 re-fit's reproduction check.
- Part A results table (4 seeds, held-out macro AUC, min/median/max
  per-topic AUC) and summary statistics, with an explicit verdict on
  whether the baseline-beating result is seed-robust.
- Part B results table (threshold, multi-label target stats for both
  splits, held-out macro AUC) with an explicit statement of which
  threshold(s) actually produce multi-label targets.
- Part C results table (learning rate, held-out macro AUC).
- A closing recommendation: the single best, most defensible Stage-2
  configuration found across this sweep (seed/threshold/lr), and whether
  its macro-AUC improvement over the 0.5000 baseline should be considered
  robust or still provisional, using the same non-inflated, margin-stating
  language this investigation has used throughout (e.g. do not call a 0.01-
  0.02 macro-AUC difference "clearly better" without saying so is a thin
  margin).
