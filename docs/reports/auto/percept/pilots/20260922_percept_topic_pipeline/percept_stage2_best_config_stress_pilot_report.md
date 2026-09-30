# ArtELingo PercepT Stage 2 best-configuration seed-stress pilot

Generated automatically, 2026-09-23 03:11:33.

## Shared Stage-1 K=60/40 seed-42 reproduction

Stage 1 was re-fit exactly once before the frozen `q > 1.2/40` targets and all four mapper runs. Every run shares that frozen encoder, 40 surviving centers, and cached patch features.

| metric | established | re-fit | absolute difference | status |
|---|---:|---:|---:|---|
| held-out emotion AMI | 0.1238 | 0.1238 | 0.0000 | reproduced |
| held-out genre AMI | 0.2617 | 0.2617 | 0.0000 | reproduced |

## Winning configuration mapper-init seed stress test

Each row uses the shared frozen `q > 1.2/40` targets, `lr=3e-3`, and 100 epochs. `AttentionPoolingMapper` and `train_and_evaluate_mapper()` are imported directly from the Stage-2 sweep pilot.

| mapper-init seed | held-out macro AUC | min per-topic AUC | median per-topic AUC | max per-topic AUC | skipped topics |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.8256 | 0.5728 | 0.8288 | 0.9605 | 0 |
| 7 | 0.8248 | 0.5366 | 0.8299 | 0.9595 | 0 |
| 123 | 0.8272 | 0.5690 | 0.8246 | 0.9606 | 0 |
| 2024 | 0.8249 | 0.5479 | 0.8331 | 0.9593 | 0 |

| seed-summary statistic | held-out macro AUC |
|---|---:|
| mean | 0.8256 |
| min | 0.8248 |
| max | 0.8272 |

**Verdict:** The richer multi-label threshold plus tuned learning rate is seed-robust relative to the original threshold: all four runs exceed the original four-seed maximum of 0.5760. The actual macro-AUC spread is 0.8248-0.8272 (width 0.0025); seed 42's 0.8256 is 0.0000 below the four-seed mean.

## Comparison with the original threshold seed stress test

The original `q > 2.0/40`, `lr=1e-3` configuration had four-seed held-out macro AUC mean 0.5709 and range 0.5644-0.5760. Because the rich-target configuration's worst seed still exceeds that original range, its improvement is robust and reproducible rather than being consumed by mapper-init variance.
