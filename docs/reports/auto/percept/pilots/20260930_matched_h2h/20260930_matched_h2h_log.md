# 2026-09-30 matched PercepT-vs-buddy head-to-head (H2H)

## Problem

Master report §6g ("PercepT 0.9226 vs buddy 0.8534 on Stage 2") and §6i
(buddy `m8x7ifx4` at 0.9355) were not comparable: different topic counts
(K), different held-out labels, different Stage 1 implementations and very
unequal tuning budgets. The question was which system gives better Stage 2
macro AUC, and how their topics compare on emotion and genre AMI, when both
run through one harness at matched K, with the same labels, evaluation and
tuning budget.

Design: `docs/superpowers/specs/2026-09-30-matched-percept-buddy-h2h-design.md`.
Plan: `docs/superpowers/plans/2026-09-30-matched-percept-buddy-h2h.md`.
SDD ledger: `.superpowers/sdd/2026-09-30-matched-percept-buddy-h2h/progress.md`.

## Investigation

- **Validation (V1 to V3).** `early_v2_check.py` (buddy pilot Stage 1 port,
  bit-exact vs the snapshot pilot at seeds 42/7/123/2024; logs
  `logs/ev2-s*.log`), `early_v3_check.py` (PercepT port at §6g settings;
  `logs/ev3-s42.log`), `validate_v1_stage2.py` + `test_validate_v1.py`
  (shared Stage 2 on the pilots' saved topics, local RTX 3090;
  `v1_results.json`, `logs/v1_local.log`). `smoke_trial.py` timed one trial
  per preset (`logs/smoke-*.log`).
- **Sweeps.** Four W&B Bayesian sweeps in `polysemic/CoSiR-h2h`
  (buddy K16 `l0hg1hb7`, buddy K40 `0o1hc9gm`, PercepT K16 `dzzmonjy`,
  PercepT K40 `7un6c1ak`), 300 trials each, 9 DAS6 GPUs, about 16 hours.
  `analyze_h2h_sweeps.py dump` froze every run to `sweep_runs.json`.
- **Selection.** Primary: `h2h_select.py select` froze each cell's top 5 by
  val AUC (`finalists_primary_*.json`). Secondary, added after the interim
  leaderboard: `analyze_h2h_sweeps.py constrained` froze the top 5 above a
  val emotion floor of 0.11167 (`finalists_constrained_*.json`).
- **Stress and test.** Each finalist ran at 4 stress seeds on val
  (`logs/h2h-stress-*.log`), each cell winner at 5 test seeds on test, plus
  the two references (`logs/h2h-test-*.log`). `h2h_select.py summarize`
  writes `stress_summary.md` and `test_summary.md` from these logs.

## Finding

Test half, 5 seeds, PercepT as the baseline (full numbers in the report):
- Plain AUC (primary): buddy 0.9931 / 0.9937 vs PercepT 0.9664 / 0.9604
  (K = 16 / 40), with independent emotion AMI 0.060 vs 0.079 / 0.110.
  Replicated on the val stress runs.
- Emotion-constrained (secondary): 0.9462 vs 0.9435 and 0.9550 vs 0.9598;
  no difference detected on either half. Buddy's constrained winners showed
  no less emotion than PercepT's; a test-half genre gap did not replicate
  on val.

The final whole-branch review found that every number reproduced, and
that several report claims were over- or under-stated. Fixes (same day):
`h2h_trial.py` now treats a PercepT seed with fewer train topics than K as
a K miss (5 sweep trials affected, no finalist, winner or reported number
changed); `h2h_select.py` reports the sample SD and recomputes summaries
from `H2H_SEED` lines, and both summary files were regenerated (SD column
only). Three buddy sweep trials (`2kymzq7f`, `2w0gxbfm`, `reb4kvk5`)
ended with an empty W&B summary; the cause is unknown.

## Solution / follow-up

Report: `docs/reports/auto/percept/2026-09-30_matched_percept_buddy_h2h.md`
(figures built by `docs/reports/assets/build_2026-09-30_h2h_figures.py`);
master report §6k. Open follow-ups: which comparison leads (the user's
choice), a pilot-only buddy search, and a search aimed at buddy's
high-emotion region.
