# ArtELingo PercepT Stage 1 LAMBDA_BALANCE sweep pilot

Generated automatically, 2026-09-23 00:34:01.

## Controlled setup

The train-only input is a 2816-dimensional fused vector built with the base pilot's embedding extraction and fusion helpers. GoEmotions-RoBERTa extraction for both splits and 100-epoch autoencoder pretraining run exactly once before the sweep. Each lambda then receives newly constructed encoder/decoder modules loaded from a deep-copied pretrained state and a fresh deterministic K-means initialization (`SEED=42`), so no DEC-trained weights leak between sweep points.

DEC retains the stabilized pilot's Student's-t assignment, self-sharpened target, reconstruction term, convergence criterion, logging cadence, and 67-of-100 center pruning. Only `LAMBDA_BALANCE` varies over 10, 50, 100, and 500.

## Predeclared success criterion

**held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, both simultaneously.**

## Results

| lambda | split | emotion AMI | genre AMI | verdict | final max cluster size (67 surviving) | clusters below 1% | fraction below 1% |
|---:|---|---:|---:|---|---:|---:|---:|
| 10 | train | 0.1217 | 0.3243 | Collapsed | 18,591 | 61/67 | 91.0% |
| 10 | held-out | 0.0952 | 0.3108 | Collapsed | 2,414 | 61/67 | 91.0% |
| 50 | train | 0.1119 | 0.2736 | Collapsed | 10,194 | 49/67 | 73.1% |
| 50 | held-out | 0.0899 | 0.2332 | Collapsed | 1,655 | 49/67 | 73.1% |
| 100 | train | 0.1176 | 0.2977 | Collapsed | 10,188 | 52/67 | 77.6% |
| 100 | held-out | 0.0942 | 0.2502 | Collapsed | 1,454 | 53/67 | 79.1% |
| 500 | train | 0.1360 | 0.2745 | Collapsed | 5,175 | 46/67 | 68.7% |
| 500 | held-out | 0.1142 | 0.2116 | Collapsed | 777 | 44/67 | 65.7% |

## Final DEC cluster-size diagnostics

Hard assignments here cover all 100 pre-pruning DEC centers; epoch 0 is the shared K-means initialization snapshot.

- **lambda=10:** stopped at epoch 435 via stability criterion; final diagnostic min 0, max 18,448, median 0.0, below 1% 94/100. Across its logged checkpoints, below-1% centers moved from 61/100 to 94/100 (range 61-94); the largest center moved from 1,168 to 18,448 (range 1,168-18,448).
- **lambda=50:** stopped at epoch 500 via epoch ceiling; final diagnostic min 0, max 10,194, median 10.0, below 1% 82/100. Across its logged checkpoints, below-1% centers moved from 61/100 to 82/100 (range 61-82); the largest center moved from 1,168 to 10,194 (range 1,168-11,602).
- **lambda=100:** stopped at epoch 500 via epoch ceiling; final diagnostic min 0, max 10,158, median 25.0, below 1% 85/100. Across its logged checkpoints, below-1% centers moved from 55/100 to 85/100 (range 55-85); the largest center moved from 943 to 10,158 (range 943-10,158).
- **lambda=500:** stopped at epoch 500 via epoch ceiling; final diagnostic min 21, max 4,494, median 205.0, below 1% 76/100. Across its logged checkpoints, below-1% centers moved from 55/100 to 76/100 (range 55-76); the largest center moved from 943 to 4,494 (range 943-4,494).

A lower collapse fraction alone is not success: at the high end, a nearly uniform hard-assignment diagnostic paired with weak external AMI is the opposite failure mode—balance-forced uniform noise rather than meaningful topic structure. The verdicts above therefore require both non-collapse and the held-out Pareto bar.

## Decision

Under the established held-out decision convention, lambda values escaping collapse: **none**. Lambda values clearing the held-out Pareto bar: **none**.

**Collapsed.** No tested lambda escaped held-out collapse. The held-out collapse fraction is non-monotonic across the lambda grid, indicating a possible intermediate sweet spot. Because the result is non-monotonic, a refined sweep around the least-collapsed setting is worth trying rather than treating the approach as exhausted. This sweep therefore does not establish a new standing PercepT-pipeline result.
