# 20261016 factor-learning grid: script, prepare, timing smoke

## Problem
Task 3 of the v2 factor-learning plan: build `run_grid.py` for the 2x2 grid (C0, A, S, AS) on scorer-train rows,
run its prepare phase, and measure run times to decide local RTX 3090 versus a DAS6 node.

## Steps
1. Wrote `run_grid.py` (constants, `cell_config`, `prepare`, `run_cell`, `smoke`, `main`). `--evaluate` and
   `--tables` raise `NotImplementedError("Task 4")`.
2. `--prepare`: 183,694 scorer-train rows, 32,413 selection rows, 36,518 paintings, 1,873,347 graph edges
   (5.0 s), 64 CLIP image groups, R0 SHA-256 verified, R0 readout reference [0.4930, 0.4691]; 15.9 s total.
3. `--smoke`: each cell at 10 and 60 steps; per-step time is the slope between them.

## Smoke table (local RTX 3090, seed 42)
| cell | s/step | projected full run (min) | peak GiB |
|---|---|---|---|
| C0 | 0.248 | 8.3 | 3.86 |
| A  | 0.247 | 8.2 | 3.13 |
| S  | 0.277 | 9.2 | 3.86 |
| AS | 0.259 | 8.6 | 3.14 |

Projected grid 2,061 s (34.4 min); replication 2,100 s (35.0 min); total sequential 4,162 s (1.16 h);
peak 3.86 GiB. Decision: `run_locally` (thresholds: 45 min per run, 3 h total, 20 GiB).

## What was verified
- CUDA present; prepare numbers match expectations (183,694 rows, 36,518 paintings, 64 groups, finite reference).
- All four cells train 10 and 60 steps with finite codes; checkpoints and histories written.
- Smoke checkpoints (`*_smoke*.pt`) are discarded, never evaluated. Cache, checkpoints, results and logs are gitignored.
