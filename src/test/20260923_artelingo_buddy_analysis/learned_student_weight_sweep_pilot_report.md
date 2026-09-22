# ArtELingo learned two-teacher student — content-loss-weight sweep

Generated automatically, 2026-09-22 20:50:57.

## Context

Stage 1's equally weighted learned student was the strongest result so far, but its held-out emotion AMI missed the target. Its checkpoint trajectory showed the content gradient share at 0.7356 at epoch 5 before it settled toward about 0.50 by epoch 200, motivating this predeclared sweep of a higher content-loss weight while holding the architecture, data, diagnostics, and stopping rule fixed.

## Sanity check

The `content_weight=1.0` in-sweep sanity check reproduces Stage 1 within 0.005 AMI: train emotion/genre AMI=0.1284/0.2799 (Stage 1: 0.1284/0.2799) and held-out emotion/genre AMI=0.1095/0.2901 (Stage 1: 0.1095/0.2901).

## Results

| content weight | split | emotion AMI | genre AMI | collapse verdict | final content gradient share | final gate mean | final gate saturated fraction |
|---:|---|---:|---:|---|---:|---:|---:|
| 1.0 | train | 0.1284 | 0.2799 | Collapsed | 0.4988 | 0.4926 | 0.0000 |
| 1.0 | held-out | 0.1095 | 0.2901 | Collapsed | 0.4988 | 0.4926 | 0.0000 |
| 1.5 | train | 0.0865 | 0.3775 | Merely a compromise | 0.5024 | 0.5342 | 0.0000 |
| 1.5 | held-out | 0.0912 | 0.3796 | Merely a compromise | 0.5024 | 0.5342 | 0.0000 |
| 2.0 | train | 0.0847 | 0.3971 | Collapsed | 0.5126 | 0.5606 | 0.0000 |
| 2.0 | held-out | 0.0848 | 0.4072 | Collapsed | 0.5126 | 0.5606 | 0.0000 |
| 3.0 | train | 0.0734 | 0.4426 | Collapsed | 0.5290 | 0.5847 | 0.0000 |
| 3.0 | held-out | 0.0631 | 0.4246 | Collapsed | 0.5290 | 0.5847 | 0.0000 |

## Held-out Pareto verdict

The held-out Pareto bar is emotion AMI > 0.1236 and genre AMI > 0.1954. No weight clears it. The best point by summed held-out AMI is content_weight=2.0, with emotion/genre AMI=0.0848/0.4072.

## Held-out trend

Held-out emotion AMI across content weights is 1.0: 0.1095, 1.5: 0.0912, 2.0: 0.0848, 3.0: 0.0631; it falls monotonically. The corresponding held-out genre AMIs are 1.0: 0.2901, 1.5: 0.3796, 2.0: 0.4072, 3.0: 0.4246.

## Conclusion

No weight clears the held-out Pareto bar, and the sweep makes no material improvement over Stage 1's unweighted held-out emotion-AMI shortfall.
