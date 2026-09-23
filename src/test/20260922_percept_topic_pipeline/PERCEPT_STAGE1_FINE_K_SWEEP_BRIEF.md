# Brief: PercepT Stage 1 — fine K-sweep around the K=60 crossover

Write a new script
`src/test/20260922_percept_topic_pipeline/run_percept_stage1_fine_k_sweep_pilot.py`.
Do NOT run it — this will be launched separately on a DAS6 cluster node.

## Context

Read `run_percept_stage1_cluster_count_sweep_v2_pilot.py` and
`percept_stage1_cluster_count_sweep_v2_pilot_report.md` in full first (both
in this directory), plus `docs/reports/2026-09-23_artelingo_percept_stage1_report.md`
section IV's consolidated cluster-count table.

Only 40/27, 60/40, and 80/53 were tested in the region around the K=60/40
crossover (the only config to clear the held-out Pareto bar in 4/4 seeds).
This pilot fills in the gap with a finer screen to check whether an even
better nearby point exists, or whether 60/40 already sits near a local
optimum.

## What to do

Reuse `run_percept_stage1_cluster_count_sweep_v2_pilot.py`'s exact
structure and mechanics (shared pretrained autoencoder at seed 42,
`LAMBDA_BALANCE=1000`, `LAMBDA_RECONSTRUCTION=1`, isolated K-means+DEC per
sweep point, same ~67% retention ratio for surviving cluster count) —
import and reuse its helper functions directly, do not reimplement.

**Phase 1 — seed-42 screen** at `(N_INITIAL, N_SURVIVING) ∈ {(50, 33),
(55, 37), (65, 43), (70, 47)}` (50/33 was already tested as a single-seed
point in the FIRST cluster-count sweep — cite its held-out result from
`percept_stage1_cluster_count_sweep_pilot_report.md` rather than
retraining it; the other three are new).

**Phase 2 — seed-stress the single best Phase-1 point** (same selection
rule as prior sweeps: prefer clearing both held-out bars with the largest
emotion margin; if none clear, largest emotion margin regardless) across
`SEEDS = (7, 123, 2024)` plus citing the Phase-1 seed-42 result — same
4-seed methodology as every other stress test in this directory.

## Report

Write
`src/test/20260922_percept_topic_pipeline/percept_stage1_fine_k_sweep_pilot_report.md`
with the same structure as `percept_stage1_cluster_count_sweep_v2_pilot_report.md`:
Phase 1 table, a consolidated table across ALL tested K values so far
(cite 20/13, 30/20, 40/27, 50/33 [cited from sweep 1], 60/40, 80/53, 100/67
from prior reports; include this pilot's new 55/37, 65/43, 70/47 points),
Phase-2 selection reasoning, Phase 2 four-seed table and summary stats, and
a final honest verdict on whether anything beats K=60/40's 4-seed mean
(emotion 0.1252, genre 0.2486) — state plainly if 60/40 remains the best
point rather than forcing a new "winner" narrative if the fine sweep
doesn't actually improve on it.
