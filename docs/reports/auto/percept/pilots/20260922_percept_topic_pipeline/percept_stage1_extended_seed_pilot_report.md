# ArtELingo PercepT Stage 1 extended seed pilot

Generated automatically, 2026-09-23 08:22:15.

## Controlled setup

This pilot holds the standing K=60/40 configuration fixed: `N_INITIAL_CLUSTERS=60`, `N_SURVIVING_CLUSTERS=40`, `LAMBDA_BALANCE=1000`, and `LAMBDA_RECONSTRUCTION=1`. Each newly measured seed receives fresh Torch and CUDA seeding (when available), a fresh 100-epoch autoencoder pretrain, `KMeans(random_state=seed)`, DEC training to the established convergence criterion, and 40-of-60 center pruning. The four established seeds are cited from `percept_stage1_cluster_count_sweep_v2_pilot_report.md`, not recomputed.

## Predeclared success criterion

**held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, both simultaneously.**

## Full 14-seed held-out results

| seed | source | held-out emotion AMI | held-out genre AMI | verdict | held-out Pareto bar |
|---:|---|---:|---:|---|---|
| 42 | cited from V2 Phase 2 table | 0.1238 | 0.2617 | Real success | clears |
| 7 | cited from V2 Phase 2 table | 0.1252 | 0.2507 | Real success | clears |
| 123 | cited from V2 Phase 2 table | 0.1272 | 0.2328 | Real success | clears |
| 2024 | cited from V2 Phase 2 table | 0.1246 | 0.2491 | Real success | clears |
| 1 | newly measured | 0.1265 | 0.2750 | Real success | clears |
| 2 | newly measured | 0.1226 | 0.2691 | Merely a compromise | does not clear |
| 3 | newly measured | 0.1157 | 0.2441 | Merely a compromise | does not clear |
| 4 | newly measured | 0.1247 | 0.2675 | Real success | clears |
| 5 | newly measured | 0.1243 | 0.2537 | Real success | clears |
| 6 | newly measured | 0.1258 | 0.2617 | Real success | clears |
| 8 | newly measured | 0.1235 | 0.2399 | Merely a compromise | does not clear |
| 9 | newly measured | 0.1206 | 0.2192 | Merely a compromise | does not clear |
| 10 | newly measured | 0.1289 | 0.2606 | Real success | clears |
| 11 | newly measured | 0.1253 | 0.2381 | Real success | clears |

## Held-out summary statistics

- Emotion AMI across 14 seeds: mean=0.1242; min=0.1157; max=0.1289; stdev=0.0032.
- Genre AMI across 14 seeds: mean=0.2517; min=0.2192; max=0.2750; stdev=0.0156.
- Emotion clears its individual bar in 10/14 seeds (71.4%).
- Genre clears its individual bar in 14/14 seeds (100.0%).
- Both bars clear simultaneously in 10/14 seeds (71.4%).

## Emotion confidence interval

The tighter-margin emotion axis is the statistical focus of this pilot. Its 14-seed mean is 0.1242; using the sample standard deviation and the requested simple normal approximation, its 95% confidence interval is 0.1242 ± 0.0017, or [0.1225, 0.1259]. The lower bound does not clear the 0.1236 threshold; it misses by 0.0011.

## Decision

**Seed-dependent result.** 10/14 seeds clear both held-out Pareto bars; 4/14 miss. The misses are: seed 2 misses emotion by 0.0010; seed 3 misses emotion by 0.0079; seed 8 misses emotion by 0.0001; seed 9 misses emotion by 0.0030.
