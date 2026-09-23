# Brief: PercepT Stage 1 — extended seed count for a real confidence interval

Write a new script
`src/test/20260922_percept_topic_pipeline/run_percept_stage1_extended_seed_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context

Read `run_percept_stage1_cluster_count_sweep_v2_pilot.py` and
`percept_stage1_cluster_count_sweep_v2_pilot_report.md` in full first (both
in this directory), plus
`docs/reports/2026-09-23_artelingo_percept_stage1_report.md` section V.

K=60/40 cleared the held-out Pareto bar (emotion AMI > 0.1236, genre AMI >
0.1954) in 4/4 seeds (42, 7, 123, 2024): mean emotion AMI 0.1252, min 0.1238
— clearing the emotion bar by only 0.0002 in the worst case. Four seeds is
a small sample for a margin that thin. This pilot extends the seed count to
build an actual confidence interval instead of a small-sample anecdote.

## What to do

Reuse the exact K=60/40 training mechanics from
`run_percept_stage1_cluster_count_sweep_v2_pilot.py` (full pipeline
re-randomization per seed: fresh `torch.manual_seed`/CUDA seeding, fresh
100-epoch autoencoder pretrain, `KMeans(random_state=seed)`, DEC to
convergence, 60-of-100... wait, 40-of-60 pruning, same
`LAMBDA_BALANCE=1000`, `LAMBDA_RECONSTRUCTION=1`) — import and reuse its
helper functions directly (`initialize_cluster_centers`,
`train_dec_until_stable`, `prune_centers`, `evaluate_run`, or equivalent
names — read the file to get the exact names), do not reimplement.

Run **10 NEW seeds**: `SEEDS = (1, 2, 3, 4, 5, 6, 8, 9, 10, 11)` (deliberately
skipping 7 since that one is already covered by the existing 4-seed
result — avoids any accidental duplicate-seed confusion). Combine these 10
new results with the 4 already-established seeds (42, 7, 123, 2024 — CITE
their held-out emotion/genre AMI values from
`percept_stage1_cluster_count_sweep_v2_pilot_report.md`'s Phase 2 table,
do not recompute them) for a full **14-seed** picture.

## Report

Write
`src/test/20260922_percept_topic_pipeline/percept_stage1_extended_seed_pilot_report.md`
with:
- A full 14-seed results table (source: cited vs newly measured, held-out
  emotion AMI, held-out genre AMI, verdict, held-out Pareto bar clearance).
- Summary statistics across all 14: mean, min, max, standard deviation for
  both held-out emotion AMI and held-out genre AMI, and the count/fraction
  of seeds clearing each bar individually and both simultaneously.
- An honest statistical framing: with 14 seeds, report the mean and a
  simple normal-approximation 95% confidence interval (mean ± 1.96 *
  stdev/sqrt(14)) for held-out emotion AMI specifically (the tighter-margin
  axis), and state plainly whether that interval's lower bound clears the
  0.1236 threshold or not — this is the actual question this pilot exists
  to answer, do not bury it.
- A final verdict using this investigation's established language, updated
  for the larger sample: if all 14 seeds clear both bars, say the result is
  now robust on a substantially larger sample, not just "seed-robust" per
  the earlier 4-seed standard. If any of the 14 miss, report the exact
  count and margins, the same honest way every miss has been reported
  throughout this investigation.
