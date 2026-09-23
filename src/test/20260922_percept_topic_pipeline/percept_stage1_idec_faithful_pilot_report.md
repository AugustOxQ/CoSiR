# ArtELingo PercepT Stage 1 faithful IDEC pilot

Generated automatically, 2026-09-23 01:48:58.

## Controlled setup

The train-only input is a 2816-dimensional fused vector built with the base pilot's unchanged embedding construction. Each run uses the unchanged 128-dimensional autoencoder, 100-epoch reconstruction-only pretraining, 100-center K-means initialization, Student's-t assignments, self-sharpened target, convergence threshold, and 67-of-100 norm pruning.

This is IDEC in isolation: `total_loss = reconstruction_loss + 0.1 * kl_loss`. There is no balance term and no reconstruction down-weighting.

## Predeclared success criterion

**held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, both simultaneously.**

## Phase 1: seed 42

- Pretraining reconstruction: epoch 10: 0.000075; epoch 20: 0.000055; epoch 30: 0.000046; epoch 40: 0.000041; epoch 50: 0.000037; epoch 60: 0.000034; epoch 70: 0.000032; epoch 80: 0.000031; epoch 90: 0.000029; epoch 100: 0.000028.
- IDEC stopped via the **stability criterion** at epoch **377** (threshold `fraction_changed < 0.001`, ceiling 500).
- Joint IDEC: epoch 1: total=0.000738, KL=0.007097, recon=0.000028, fraction_changed=0.078255; epoch 25: total=0.002058, KL=0.020183, recon=0.000040, fraction_changed=0.023077; epoch 75: total=0.003125, KL=0.030714, recon=0.000054, fraction_changed=0.014658; epoch 125: total=0.004170, KL=0.041083, recon=0.000061, fraction_changed=0.012850; epoch 175: total=0.005103, KL=0.050359, recon=0.000067, fraction_changed=0.011693; epoch 225: total=0.005962, KL=0.058951, recon=0.000067, fraction_changed=0.014299; epoch 275: total=0.006690, KL=0.066258, recon=0.000064, fraction_changed=0.039999; epoch 325: total=0.007256, KL=0.071943, recon=0.000062, fraction_changed=0.009251; epoch 375: total=0.007677, KL=0.076169, recon=0.000060, fraction_changed=0.004006; epoch 377: total=0.007691, KL=0.076311, recon=0.000060, fraction_changed=0.000407.
- All-100-center diagnostics (including epoch-0 K-means initialization): epoch 0: min=354, max=934, median=600.0, below_1pct=56/100; epoch 25: min=13, max=9,085, median=159.0, below_1pct=79/100; epoch 50: min=0, max=11,553, median=49.0, below_1pct=80/100; epoch 75: min=0, max=13,663, median=23.0, below_1pct=82/100; epoch 100: min=0, max=15,973, median=12.0, below_1pct=85/100; epoch 125: min=0, max=19,750, median=5.0, below_1pct=88/100; epoch 150: min=0, max=22,419, median=3.0, below_1pct=89/100; epoch 175: min=0, max=22,703, median=1.0, below_1pct=90/100; epoch 200: min=0, max=18,796, median=0.0, below_1pct=91/100; epoch 225: min=0, max=16,159, median=0.0, below_1pct=92/100; epoch 250: min=0, max=21,370, median=0.0, below_1pct=92/100; epoch 275: min=0, max=24,497, median=0.0, below_1pct=93/100; epoch 300: min=0, max=28,280, median=0.0, below_1pct=94/100; epoch 325: min=0, max=28,687, median=0.0, below_1pct=95/100; epoch 350: min=0, max=29,101, median=0.0, below_1pct=96/100; epoch 375: min=0, max=29,937, median=0.0, below_1pct=97/100; epoch 377: min=0, max=29,891, median=0.0, below_1pct=97/100.

| seed | split | emotion AMI | genre AMI | verdict | held-out Pareto bar | cluster min | cluster max | cluster median | fraction below 1% |
|---:|---|---:|---:|---|---|---:|---:|---:|---:|
| 42 | train | 0.0564 | 0.2853 | Collapsed | n/a (train split) | 0 | 29,890 | 0.0 | 95.5% (64/67) |
| 42 | held-out | 0.0444 | 0.2718 | Collapsed | does not clear | 0 | 5,023 | 0.0 | 95.5% (64/67) |

## Phase 2

Skipped because Phase 1 did not clear the held-out Pareto bar; additional GPU seed stress would not be informative for a configuration that already missed its first test.

## Interpretation

The all-100-center trajectory checks conventional collapse before pruning. It also checks IDEC's distinct alternate failure mode: if reconstruction dominates, the fraction changed can quickly become negligible and the diagnostics can remain close to the epoch-0 K-means snapshot while external AMI stays weak. That outcome is not a successful non-collapse; it means clustering barely moved and did not produce informative assignments.

## Decision

**Merely a compromise.** IDEC's literature-standard fix did not outperform this project's own balance approach on this task at `gamma=0.1`; no further gamma sweep is in scope.
