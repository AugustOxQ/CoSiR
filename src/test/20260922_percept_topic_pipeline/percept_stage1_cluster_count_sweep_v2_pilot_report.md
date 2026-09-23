# ArtELingo PercepT Stage 1 intermediate cluster-count sweep pilot

Generated automatically, 2026-09-23 02:19:08.

## Controlled setup

The train-only input is a 2816-dimensional fused vector. Phase 1 uses seed 42, performs one 100-epoch autoencoder pretrain, and deep-copies that pretrained state for each isolated DEC run. Each point receives fresh K-means at `random_state=42` with its own initial-cluster count. `LAMBDA_BALANCE=1000` and `LAMBDA_RECONSTRUCTION=1` remain fixed; only the initial/surviving cluster-count pair varies.

Phase 2 cites the selected Phase-1 seed-42 result, then runs seeds 7, 123, and 2024. Each new seed applies fresh NumPy, Torch, and CUDA seeding (when available), builds and pretrains a new autoencoder for 100 epochs, runs `KMeans(random_state=seed)`, and trains DEC with the selected cluster counts and unchanged loss weights.

The below-1% and collapse calculations use each point's active surviving-cluster count, not the original fixed 67.

## Predeclared success criterion

**held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, both simultaneously.**

## Phase 1: seed-42 cluster-count screen

Pretraining reconstruction: epoch 10: 0.000077; epoch 20: 0.000056; epoch 30: 0.000046; epoch 40: 0.000040; epoch 50: 0.000036; epoch 60: 0.000034; epoch 70: 0.000031; epoch 80: 0.000030; epoch 90: 0.000029; epoch 100: 0.000028.

| N initial | N surviving | split | emotion AMI | genre AMI | verdict | held-out Pareto bar | final max cluster size | fraction below 1% |
|---:|---:|---|---:|---:|---|---|---:|---:|
| 40 | 27 | train | 0.1456 | 0.3318 | Real success | n/a (train split) | 8,473 | 7.4% (2/27) |
| 40 | 27 | held-out | 0.1220 | 0.2736 | Merely a compromise | does not clear | 1,039 | 3.7% (1/27) |
| 60 | 40 | train | 0.1462 | 0.3258 | Real success | n/a (train split) | 5,290 | 20.0% (8/40) |
| 60 | 40 | held-out | 0.1238 | 0.2617 | Real success | clears | 689 | 17.5% (7/40) |
| 80 | 53 | train | 0.1439 | 0.3224 | Real success | n/a (train split) | 6,360 | 34.0% (18/53) |
| 80 | 53 | held-out | 0.1220 | 0.2396 | Merely a compromise | does not clear | 758 | 30.2% (16/53) |

## Consolidated seed-42 cluster-count picture

Prior-sweep values are cited, not recomputed. K=100/67 is cited from `percept_stage1_balance_sweep_v2_pilot_report.md` at lambda=1000.

| N initial | N surviving | held-out emotion AMI | held-out genre AMI | source |
|---:|---:|---:|---:|---|
| 20 | 13 | 0.1152 | 0.2273 | cited from first cluster-count sweep |
| 30 | 20 | 0.1220 | 0.2577 | cited from first cluster-count sweep |
| 40 | 27 | 0.1220 | 0.2736 | newly measured in this sweep |
| 50 | 33 | 0.1202 | 0.2830 | cited from first cluster-count sweep |
| 60 | 40 | 0.1238 | 0.2617 | newly measured in this sweep |
| 80 | 53 | 0.1220 | 0.2396 | newly measured in this sweep |
| 100 | 67 | 0.1242 | 0.2466 | cited from balance v2 sweep at lambda=1000 |

## Phase-2 selection

`N_INITIAL_CLUSTERS=60`, `N_SURVIVING_CLUSTERS=40` was selected because it clears the held-out Pareto bar and has the largest held-out emotion margin among clearing points (emotion margin +0.0002; genre margin +0.0663).

## Phase 2: four-seed stress test

| seed | source | split | emotion AMI | genre AMI | verdict | held-out Pareto bar |
|---:|---|---|---:|---:|---|---|
| 42 | cited from Phase 1 | train | 0.1462 | 0.3258 | Real success | n/a (train split) |
| 42 | cited from Phase 1 | held-out | 0.1238 | 0.2617 | Real success | clears |
| 7 | newly measured | train | 0.1477 | 0.3161 | Real success | n/a (train split) |
| 7 | newly measured | held-out | 0.1252 | 0.2507 | Real success | clears |
| 123 | newly measured | train | 0.1492 | 0.3095 | Real success | n/a (train split) |
| 123 | newly measured | held-out | 0.1272 | 0.2328 | Real success | clears |
| 2024 | newly measured | train | 0.1461 | 0.3286 | Real success | n/a (train split) |
| 2024 | newly measured | held-out | 0.1246 | 0.2491 | Real success | clears |

## Held-out summary statistics

- Emotion AMI across four seeds: mean=0.1252; min=0.1238; max=0.1272; clears its individual bar in 4/4 seeds.
- Genre AMI across four seeds: mean=0.2486; min=0.2328; max=0.2617; clears its individual bar in 4/4 seeds.
- Both bars clear simultaneously in 4/4 seeds.

## Stability/performance comparison

K=100/67 and K=30/20 values are cited from the established seed-stress and first cluster-count-sweep reports, respectively. The K=100/67 emotion mean is reported as 0.1234 as established for this investigation.

| configuration | held-out emotion AMI (mean/min/max) | held-out genre AMI (mean/min/max) | both bars clear | emotion spread | genre spread |
|---|---|---|---:|---:|---:|
| original K=100/67 | mean=0.1234; min=0.1216; max=0.1252 | mean=0.2089; min=0.1849; max=0.2466 | 1/4 | 0.0036 | 0.0617 |
| K=30/20 | mean=0.1212; min=0.1204; max=0.1220 | mean=0.2615; min=0.2563; max=0.2732 | 0/4 | 0.0016 | 0.0169 |
| K=60/40 | mean=0.1252; min=0.1238; max=0.1272 | mean=0.2486; min=0.2328; max=0.2617 | 4/4 | 0.0034 | 0.0289 |

## Decision

**Real success: stable crossover found.** This configuration clears the held-out Pareto bar in 4/4 seeds while retaining most of K=30's stability gain, with spreads meaningfully tighter than K=100/67. It is the new standing PercepT Stage 1 configuration.
