# ArtELingo PercepT Stage 1 reconstruction-weight sweep pilot

Generated automatically, 2026-09-23 01:43:17.

## Controlled setup

The train-only input is a 2816-dimensional fused vector. GoEmotions-RoBERTa extraction and fused-embedding construction are deterministic and occur once. Phase 1 uses seed 42, performs one 100-epoch autoencoder pretrain, and deep-copies that pretrained state for each isolated DEC run. Every Phase-1 point receives a fresh K-means initialization at `random_state=42`. `LAMBDA_BALANCE=1000` is fixed; only `LAMBDA_RECONSTRUCTION` varies.

Phase 2 cites the selected Phase-1 seed-42 result rather than recomputing it, then runs seeds 7, 123, and 2024. For each new seed, `torch.manual_seed(seed)` and CUDA seeding (when available) occur immediately before a fresh autoencoder build, followed by a fresh 100-epoch pretrain, `KMeans(random_state=seed)`, and DEC with the selected reconstruction weight and `LAMBDA_BALANCE=1000`.

## Predeclared success criterion

**held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, both simultaneously.**

## Phase 1: seed-42 reconstruction-weight sweep

Pretraining reconstruction: epoch 10: 0.000077; epoch 20: 0.000056; epoch 30: 0.000046; epoch 40: 0.000040; epoch 50: 0.000036; epoch 60: 0.000034; epoch 70: 0.000031; epoch 80: 0.000030; epoch 90: 0.000029; epoch 100: 0.000028.

| lambda reconstruction | split | emotion AMI | genre AMI | verdict | held-out Pareto bar |
|---:|---|---:|---:|---|---|
| 100 | train | 0.1435 | 0.3116 | Real success | n/a (train split) |
| 100 | held-out | 0.1236 | 0.1853 | Merely a compromise | does not clear |
| 300 | train | 0.1436 | 0.3137 | Real success | n/a (train split) |
| 300 | held-out | 0.1245 | 0.1916 | Merely a compromise | does not clear |
| 1000 | train | 0.1436 | 0.3129 | Real success | n/a (train split) |
| 1000 | held-out | 0.1237 | 0.1900 | Merely a compromise | does not clear |

## Phase-2 selection

`LAMBDA_RECONSTRUCTION=300` was selected because no Phase-1 value clears both bars, so it has the largest held-out emotion-bar margin (emotion margin +0.0009; genre margin -0.0038).

## Phase 2: four-seed stress test

| seed | source | split | emotion AMI | genre AMI | verdict | held-out Pareto bar |
|---:|---|---|---:|---:|---|---|
| 42 | cited from Phase 1 | train | 0.1436 | 0.3137 | Real success | n/a (train split) |
| 42 | cited from Phase 1 | held-out | 0.1245 | 0.1916 | Merely a compromise | does not clear |
| 7 | newly measured | train | 0.1455 | 0.3105 | Real success | n/a (train split) |
| 7 | newly measured | held-out | 0.1226 | 0.2177 | Merely a compromise | does not clear |
| 123 | newly measured | train | 0.1456 | 0.3026 | Real success | n/a (train split) |
| 123 | newly measured | held-out | 0.1252 | 0.1922 | Merely a compromise | does not clear |
| 2024 | newly measured | train | 0.1425 | 0.3045 | Real success | n/a (train split) |
| 2024 | newly measured | held-out | 0.1215 | 0.1849 | Merely a compromise | does not clear |

## Held-out summary statistics

- Emotion AMI across four seeds: mean=0.1234; min=0.1215; max=0.1252; clears its individual bar in 2/4 seeds.
- Genre AMI across four seeds: mean=0.1966; min=0.1849; max=0.2177; clears its individual bar in 1/4 seeds.
- Both bars clear simultaneously in 0/4 seeds.

## Before/after comparison with `LAMBDA_RECONSTRUCTION=1`

The baseline values below are the same four seeds from `percept_stage1_seed_stress_pilot_report.md`; the new result uses the same seed set, so this is a point-for-point stability comparison.

| configuration | held-out emotion AMI (mean/min/max) | held-out genre AMI (mean/min/max) | both bars clear |
|---|---|---|---:|
| reconstruction=1 | mean=0.1234; min=0.1216; max=0.1252 | mean=0.2089; min=0.1849; max=0.2466 | 1/4 |
| reconstruction=300 | mean=0.1234; min=0.1215; max=0.1252 | mean=0.1966; min=0.1849; max=0.2177 | 0/4 |

The emotion range is 0.0036 versus the baseline 0.0036; the genre range is 0.0328 versus the baseline 0.0617. This is not established as measurably more stable: the required combination of tighter emotion and genre spreads plus more both-bar clearers is not present.

## Decision

**Seed-dependent result.** Only 0/4 seeds clear the held-out Pareto bar. The misses are: seed 42 misses genre by 0.0038; seed 7 misses emotion by 0.0010; seed 123 misses genre by 0.0032; seed 2024 misses emotion by 0.0021 and genre by 0.0105. Do not treat partial improvement as a solved stability problem.
