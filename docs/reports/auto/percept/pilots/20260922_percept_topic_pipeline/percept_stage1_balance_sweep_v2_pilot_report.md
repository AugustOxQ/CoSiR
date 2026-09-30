# ArtELingo PercepT Stage 1 LAMBDA_BALANCE sweep pilot, round 2 (higher lambda)

Generated automatically, 2026-09-23 00:48:48.

## Controlled setup

This is a follow-up to `percept_stage1_balance_sweep_pilot_report.md` (lambda in {10, 50, 100, 500}), where every value still collapsed on held-out, but lambda=500 was the best point on every axis (lowest collapse fraction, highest held-out emotion AMI at 0.1142, closest yet to the 0.1236 bar) and still trending in the right direction at the top of that grid. This round extends the grid upward to see whether the trend continues far enough to cross the bar.

The train-only input is a 2816-dimensional fused vector built with the base pilot's embedding extraction and fusion helpers. GoEmotions-RoBERTa extraction for both splits and 100-epoch autoencoder pretraining run exactly once before the sweep. Each lambda then receives newly constructed encoder/decoder modules loaded from a deep-copied pretrained state and a fresh deterministic K-means initialization (`SEED=42`), so no DEC-trained weights leak between sweep points.

DEC retains the stabilized pilot's Student's-t assignment, self-sharpened target, reconstruction term, convergence criterion, logging cadence, and 67-of-100 center pruning. Only `LAMBDA_BALANCE` varies over 1000, 2500, and 5000.

## Predeclared success criterion

**held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, both simultaneously.**

## Results

| lambda | split | emotion AMI | genre AMI | verdict | final max cluster size (67 surviving) | clusters below 1% | fraction below 1% |
|---:|---|---:|---:|---|---:|---:|---:|
| 1000 | train | 0.1444 | 0.3193 | Real success | 4,485 | 31/67 | 46.3% |
| 1000 | held-out | 0.1242 | 0.2466 | Real success | 550 | 29/67 | 43.3% |
| 2500 | train | 0.1432 | 0.3124 | Real success | 4,984 | 32/67 | 47.8% |
| 2500 | held-out | 0.1242 | 0.1970 | Real success | 621 | 27/67 | 40.3% |
| 5000 | train | 0.1450 | 0.3077 | Collapsed | 7,342 | 34/67 | 50.7% |
| 5000 | held-out | 0.1242 | 0.2208 | Real success | 847 | 31/67 | 46.3% |

## Final DEC cluster-size diagnostics

Hard assignments here cover all 100 pre-pruning DEC centers; epoch 0 is the shared K-means initialization snapshot.

- **lambda=1000:** stopped at epoch 89 via stability criterion; final diagnostic min 21, max 2,447, median 457.0, below 1% 64/100. Across its logged checkpoints, below-1% centers moved from 61/100 to 64/100 (range 61-65); the largest center moved from 1,185 to 2,447 (range 1,185-2,679).
- **lambda=2500:** stopped at epoch 81 via stability criterion; final diagnostic min 41, max 3,540, median 460.0, below 1% 63/100. Across its logged checkpoints, below-1% centers moved from 55/100 to 63/100 (range 55-64); the largest center moved from 943 to 3,540 (range 943-3,750).
- **lambda=5000:** stopped at epoch 83 via stability criterion; final diagnostic min 50, max 3,328, median 425.0, below 1% 65/100. Across its logged checkpoints, below-1% centers moved from 57/100 to 65/100 (range 57-66); the largest center moved from 949 to 3,328 (range 949-3,608).

A lower collapse fraction alone is not success: at the high end, a nearly uniform hard-assignment diagnostic paired with weak external AMI is the opposite failure mode—balance-forced uniform noise rather than meaningful topic structure. The verdicts above therefore require both non-collapse and the held-out Pareto bar.

## Decision

Under the established held-out decision convention, lambda values escaping collapse: **1000, 2500, 5000**. Lambda values clearing the held-out Pareto bar: **1000, 2500, 5000**.

**Real success.** Lambda 1000, 2500, 5000 is the new standing PercepT-pipeline result because it is non-collapsed and clears both held-out AMI thresholds.
