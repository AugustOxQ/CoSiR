# ArtELingo PercepT Stage 1 fine K-sweep pilot

Generated automatically, 2026-09-23 08:32:32.

## Controlled setup

The train-only input is a 2816-dimensional fused vector. Phase 1 uses seed 42, performs one 100-epoch autoencoder pretrain, and deep-copies that pretrained state for each isolated new DEC run. Each new point receives fresh K-means at `random_state=42`. `LAMBDA_BALANCE=1000` and `LAMBDA_RECONSTRUCTION=1` remain fixed; only the initial/surviving cluster-count pair varies. K=50/33 is cited from the first cluster-count sweep rather than retrained.

Phase 2 cites the selected Phase-1 seed-42 result, then runs seeds 7, 123, and 2024. Each new seed rebuilds and pretrains an autoencoder for 100 epochs, runs `KMeans(random_state=seed)`, and trains DEC with unchanged loss weights.

The below-1% and collapse calculations use each point's active surviving-cluster count, not the original fixed 67.

## Predeclared success criterion

**held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, both simultaneously.**

## Phase 1: seed-42 cluster-count screen

Pretraining reconstruction: epoch 10: 0.000078; epoch 20: 0.000056; epoch 30: 0.000046; epoch 40: 0.000041; epoch 50: 0.000037; epoch 60: 0.000034; epoch 70: 0.000032; epoch 80: 0.000030; epoch 90: 0.000029; epoch 100: 0.000028.

| N initial | N surviving | split | emotion AMI | genre AMI | verdict | held-out Pareto bar | final max cluster size | fraction below 1% | source |
|---:|---:|---|---:|---:|---|---|---:|---:|---|
| 50 | 33 | train | 0.1453 | 0.3141 | Real success | n/a (train split) | 7,790 | 9.1% (3/33) | cited from first cluster-count sweep |
| 50 | 33 | held-out | 0.1202 | 0.2830 | Merely a compromise | does not clear | 930 | 9.1% (3/33) | cited from first cluster-count sweep |
| 55 | 37 | train | 0.1412 | 0.3286 | Real success | n/a (train split) | 7,188 | 13.5% (5/37) | newly measured in this sweep |
| 55 | 37 | held-out | 0.1208 | 0.2622 | Merely a compromise | does not clear | 854 | 13.5% (5/37) | newly measured in this sweep |
| 65 | 43 | train | 0.1442 | 0.3246 | Real success | n/a (train split) | 8,388 | 27.9% (12/43) | newly measured in this sweep |
| 65 | 43 | held-out | 0.1238 | 0.2376 | Real success | clears | 971 | 23.3% (10/43) | newly measured in this sweep |
| 70 | 47 | train | 0.1430 | 0.3083 | Real success | n/a (train split) | 6,541 | 29.8% (14/47) | newly measured in this sweep |
| 70 | 47 | held-out | 0.1237 | 0.2357 | Real success | clears | 779 | 27.7% (13/47) | newly measured in this sweep |

## Consolidated cluster-count picture

Prior values are cited, not recomputed. K=50/33 is cited from `percept_stage1_cluster_count_sweep_pilot_report.md`; K=60/40 is cited from `percept_stage1_cluster_count_sweep_v2_pilot_report.md`; K=100/67 is cited from the lambda=1000 balance-v2 sweep.

| N initial | N surviving | held-out emotion AMI | held-out genre AMI | source |
|---:|---:|---:|---:|---|
| 20 | 13 | 0.1152 | 0.2273 | cited from first cluster-count sweep |
| 30 | 20 | 0.1220 | 0.2577 | cited from first cluster-count sweep |
| 40 | 27 | 0.1220 | 0.2736 | cited from v2 cluster-count sweep |
| 50 | 33 | 0.1202 | 0.2830 | cited from first cluster-count sweep |
| 55 | 37 | 0.1208 | 0.2622 | newly measured in this sweep |
| 65 | 43 | 0.1238 | 0.2376 | newly measured in this sweep |
| 70 | 47 | 0.1237 | 0.2357 | newly measured in this sweep |
| 60 | 40 | 0.1238 | 0.2617 | cited from v2 cluster-count sweep |
| 80 | 53 | 0.1220 | 0.2396 | cited from v2 cluster-count sweep |
| 100 | 67 | 0.1242 | 0.2466 | cited from balance v2 sweep at lambda=1000 |

## Phase-2 selection

`N_INITIAL_CLUSTERS=65`, `N_SURVIVING_CLUSTERS=43` was selected because it clears the held-out Pareto bar and has the largest held-out emotion margin among clearing points (emotion margin +0.0002; genre margin +0.0422).

## Phase 2: four-seed stress test

| seed | source | split | emotion AMI | genre AMI | verdict | held-out Pareto bar |
|---:|---|---|---:|---:|---|---|
| 42 | cited from Phase 1 | train | 0.1442 | 0.3246 | Real success | n/a (train split) |
| 42 | cited from Phase 1 | held-out | 0.1238 | 0.2376 | Real success | clears |
| 7 | newly measured | train | 0.1416 | 0.3097 | Real success | n/a (train split) |
| 7 | newly measured | held-out | 0.1196 | 0.2294 | Merely a compromise | does not clear |
| 123 | newly measured | train | 0.1470 | 0.3121 | Real success | n/a (train split) |
| 123 | newly measured | held-out | 0.1253 | 0.2355 | Real success | clears |
| 2024 | newly measured | train | 0.1457 | 0.3188 | Real success | n/a (train split) |
| 2024 | newly measured | held-out | 0.1247 | 0.2718 | Real success | clears |

## Held-out summary statistics

- Emotion AMI across four seeds: mean=0.1234; min=0.1196; max=0.1253; clears its individual bar in 3/4 seeds.
- Genre AMI across four seeds: mean=0.2435; min=0.2294; max=0.2718; clears its individual bar in 4/4 seeds.
- Both bars clear simultaneously in 3/4 seeds.

## Comparison with K=60/40

K=60/40's established four-seed mean is emotion 0.1252 and genre 0.2486. The comparison is like-for-like because both use seeds 42, 7, 123, and 2024.

| configuration | held-out emotion AMI (mean/min/max) | held-out genre AMI (mean/min/max) | both bars clear |
|---|---|---|---:|
| K=60/40 | mean=0.1252 | mean=0.2486 | 4/4 |
| K=65/43 | mean=0.1234; min=0.1196; max=0.1253 | mean=0.2435; min=0.2294; max=0.2718 | 3/4 |

## Decision

**K=60/40 remains the standing result.** A fine-sweep point showed possible single-seed crossover evidence, but its four-seed mean does not exceed K=60/40's emotion 0.1252 and genre 0.2486 simultaneously. Do not force a new winner narrative from a weaker or partial result.
