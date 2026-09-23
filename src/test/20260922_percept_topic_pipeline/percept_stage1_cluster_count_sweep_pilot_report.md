# ArtELingo PercepT Stage 1 cluster-count sweep pilot

Generated automatically, 2026-09-23 02:03:09.

## Controlled setup

The train-only input is a 2816-dimensional fused vector. Phase 1 uses seed 42, performs one 100-epoch autoencoder pretrain, and deep-copies that pretrained state for each isolated DEC run. Each point receives fresh K-means at `random_state=42` with its own initial-cluster count. `LAMBDA_BALANCE=1000` and `LAMBDA_RECONSTRUCTION=1` remain fixed; only the initial/surviving cluster-count pair varies.

Phase 2 cites the selected Phase-1 seed-42 result, then runs seeds 7, 123, and 2024. Each new seed applies fresh Torch and CUDA seeding (when available), builds and pretrains a new autoencoder for 100 epochs, runs `KMeans(random_state=seed)`, and trains DEC with the selected cluster counts and the unchanged loss weights.

The below-1% and collapse calculations use each sweep point's active surviving-cluster count, rather than the original fixed 67.

## Predeclared success criterion

**held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, both simultaneously.**

## Phase 1: seed-42 cluster-count screen

Pretraining reconstruction: epoch 10: 0.000077; epoch 20: 0.000056; epoch 30: 0.000046; epoch 40: 0.000040; epoch 50: 0.000036; epoch 60: 0.000034; epoch 70: 0.000031; epoch 80: 0.000030; epoch 90: 0.000029; epoch 100: 0.000028.

| N initial | N surviving | split | emotion AMI | genre AMI | verdict | held-out Pareto bar | final max cluster size | fraction below 1% |
|---:|---:|---|---:|---:|---|---|---:|---:|
| 20 | 13 | train | 0.1406 | 0.2825 | Real success | n/a (train split) | 9,249 | 0.0% (0/13) |
| 20 | 13 | held-out | 0.1152 | 0.2273 | Merely a compromise | does not clear | 1,270 | 0.0% (0/13) |
| 30 | 20 | train | 0.1440 | 0.3340 | Real success | n/a (train split) | 10,160 | 5.0% (1/20) |
| 30 | 20 | held-out | 0.1220 | 0.2577 | Merely a compromise | does not clear | 1,203 | 10.0% (2/20) |
| 50 | 33 | train | 0.1453 | 0.3141 | Real success | n/a (train split) | 7,790 | 9.1% (3/33) |
| 50 | 33 | held-out | 0.1202 | 0.2830 | Merely a compromise | does not clear | 930 | 9.1% (3/33) |

## Phase-2 selection

`N_INITIAL_CLUSTERS=30`, `N_SURVIVING_CLUSTERS=20` was selected because no Phase-1 point clears both bars, so it has the largest held-out emotion-bar margin (emotion margin -0.0016; genre margin +0.0623).

## Phase 2: four-seed stress test

| seed | source | split | emotion AMI | genre AMI | verdict | held-out Pareto bar |
|---:|---|---|---:|---:|---|---|
| 42 | cited from Phase 1 | train | 0.1440 | 0.3340 | Real success | n/a (train split) |
| 42 | cited from Phase 1 | held-out | 0.1220 | 0.2577 | Merely a compromise | does not clear |
| 7 | newly measured | train | 0.1439 | 0.3248 | Real success | n/a (train split) |
| 7 | newly measured | held-out | 0.1204 | 0.2590 | Merely a compromise | does not clear |
| 123 | newly measured | train | 0.1462 | 0.3172 | Real success | n/a (train split) |
| 123 | newly measured | held-out | 0.1209 | 0.2563 | Merely a compromise | does not clear |
| 2024 | newly measured | train | 0.1467 | 0.3324 | Real success | n/a (train split) |
| 2024 | newly measured | held-out | 0.1213 | 0.2732 | Merely a compromise | does not clear |

## Held-out summary statistics

- Emotion AMI across four seeds: mean=0.1212; min=0.1204; max=0.1220; clears its individual bar in 0/4 seeds.
- Genre AMI across four seeds: mean=0.2616; min=0.2563; max=0.2732; clears its individual bar in 4/4 seeds.
- Both bars clear simultaneously in 0/4 seeds.

## Before/after comparison with original K=100/67

The baseline values below are the original K=100/67 four-seed results from `percept_stage1_seed_stress_pilot_report.md`. This pilot uses the same seed set, so it is a like-for-like stability comparison.

| configuration | held-out emotion AMI (mean/min/max) | held-out genre AMI (mean/min/max) | both bars clear |
|---|---|---|---:|
| original K=100/67 | mean=0.1234; min=0.1216; max=0.1252 | mean=0.2089; min=0.1849; max=0.2466 | 1/4 |
| K=30/20 | mean=0.1212; min=0.1204; max=0.1220 | mean=0.2616; min=0.2563; max=0.2732 | 0/4 |

The emotion range is 0.0016 versus the original 0.0036; the genre range is 0.0169 versus the original 0.0617.

## Decision

**Seed-dependent result.** Only 0/4 seeds clear the held-out Pareto bar. The misses are: seed 42 misses emotion by 0.0016; seed 7 misses emotion by 0.0032; seed 123 misses emotion by 0.0027; seed 2024 misses emotion by 0.0023. Do not treat a marginal or partial change as a solved stability problem.
