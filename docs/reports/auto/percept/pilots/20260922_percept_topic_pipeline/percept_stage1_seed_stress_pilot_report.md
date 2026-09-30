# ArtELingo PercepT Stage 1 seed stress pilot at lambda=1000

Generated automatically, 2026-09-23 01:01:49.

## Controlled setup

The train-only input is a 2816-dimensional fused vector. GoEmotions-RoBERTa affect extraction and fused-embedding construction are deterministic and were computed once, then reused across all new seeds. For each new seed, `torch.manual_seed(seed)` and CUDA seeding (when available) occurred immediately before a fresh autoencoder was built. That autoencoder was pretrained for 100 epochs, K-means used `random_state=seed`, and DEC ran with lambda=1000 to the established convergence criterion before 67-of-100 centroid pruning.

Seed 42 is cited from `percept_stage1_balance_sweep_v2_pilot_report.md` at lambda=1000; it is not recomputed here.

## Predeclared success criterion

**held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, both simultaneously.**

## Results

| seed | source | split | emotion AMI | genre AMI | verdict | held-out Pareto bar |
|---:|---|---|---:|---:|---|---|
| 42 | cited from v2 sweep report | train | 0.1444 | 0.3193 | Real success | n/a (train split) |
| 42 | cited from v2 sweep report | held-out | 0.1242 | 0.2466 | Real success | clears |
| 7 | newly measured | train | 0.1456 | 0.3086 | Real success | n/a (train split) |
| 7 | newly measured | held-out | 0.1228 | 0.2190 | Merely a compromise | does not clear |
| 123 | newly measured | train | 0.1457 | 0.3022 | Real success | n/a (train split) |
| 123 | newly measured | held-out | 0.1252 | 0.1851 | Merely a compromise | does not clear |
| 2024 | newly measured | train | 0.1425 | 0.3042 | Real success | n/a (train split) |
| 2024 | newly measured | held-out | 0.1216 | 0.1849 | Merely a compromise | does not clear |

## Held-out summary statistics

- Emotion AMI across four seeds: mean=0.1235; min=0.1216; max=0.1252; clears its individual bar in 2/4 seeds.
- Genre AMI across four seeds: mean=0.2089; min=0.1849; max=0.2466; clears its individual bar in 2/4 seeds.
- Both bars clear simultaneously in 1/4 seeds.

## Decision

**Seed-dependent result.** Only 1/4 seeds clear the held-out Pareto bar. The misses are: seed 7 misses emotion by 0.0008; seed 123 misses genre by 0.0103; seed 2024 misses emotion by 0.0020 and genre by 0.0105. Lambda=1000 should be treated as fragile/seed-dependent rather than as the reliable standing PercepT Stage 1 configuration going into Stage 2.
