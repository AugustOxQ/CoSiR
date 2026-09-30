# ArtELingo PercepT Stage 2 sweep pilot

Generated automatically, 2026-09-23 03:03:43.

## Shared Stage-1 K=60/40 seed-42 reproduction

Stage 1 was re-fit exactly once, before every frozen target and mapper variant below. All Stage-2 runs share that one frozen encoder, 40 surviving centers, and cached patch features.

| metric | established | re-fit | absolute difference | status |
|---|---:|---:|---:|---|
| held-out emotion AMI | 0.1238 | 0.1238 | 0.0000 | reproduced |
| held-out genre AMI | 0.2617 | 0.2617 | 0.0000 | reproduced |

## Part A — mapper-init seed stress test

Each row uses the original `q > 2.0/40` target threshold, `lr=1e-3`, and 100 epochs. Seed 42 is a new mapper-initialization draw, not a citation of the earlier smoke-test number.

| mapper-init seed | held-out macro AUC | min per-topic AUC | median per-topic AUC | max per-topic AUC | skipped topics |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.5704 | 0.3845 | 0.5317 | 0.8790 | 0 |
| 7 | 0.5727 | 0.3600 | 0.5464 | 0.8753 | 0 |
| 123 | 0.5760 | 0.3518 | 0.5490 | 0.8812 | 0 |
| 2024 | 0.5644 | 0.3384 | 0.5271 | 0.8791 | 0 |

| seed-summary statistic | held-out macro AUC |
|---|---:|
| mean | 0.5709 |
| min | 0.5644 |
| max | 0.5760 |

**Verdict:** The baseline-beating result is seed-robust across these four mapper initializations: every held-out macro AUC exceeds the 0.5000 train-marginal baseline.

## Part B — multi-label threshold comparison

All targets below were derived from the single shared frozen Stage-1 fit. The `2.0/40` AUC cites Part A's seed-42 run and was not retrained.

| threshold | train mean / median / max labels | train fraction multi-labeled | held-out mean / median / max labels | held-out fraction multi-labeled | held-out macro AUC |
|---|---|---:|---|---:|---:|
| q > 2.0/40 | 1.000 / 1.000 / 1 | 0.00% | 1.000 / 1.000 / 1 | 0.00% | 0.5704 |
| q > 1.5/40 | 1.000 / 1.000 / 2 | 0.02% | 1.000 / 1.000 / 2 | 0.02% | 0.5705 |
| q > 1.2/40 | 2.905 / 3.000 / 8 | 76.89% | 2.820 / 3.000 / 8 | 75.00% | 0.6894 |

**Multi-label result:** Genuine multi-label targets occur at 1.5/40, 1.2/40 (at least one split has nonzero multi-labeled paintings). The table shows whether their macro-AUC change is material relative to the original threshold.

## Part C — learning-rate mini-sweep

The selected threshold is `q > 1.2/40`. Selection prefers genuine multi-label targets whose macro AUC is within the predeclared 0.01 practical margin of the original threshold; otherwise it retains `q > 2.0/40`. The `1e-3` result is cited from Part A or Part B and was not retrained.

| learning rate | held-out macro AUC | source |
|---:|---:|---|
| 3e-04 | 0.5673 | new Part C run |
| 1e-03 | 0.6894 | Part B |
| 3e-03 | 0.8256 | new Part C run |

## Recommendation

The single best observed Stage-2 configuration is mapper-init seed 42, `q > 1.2/40`, `lr=3e-03`, and 100 epochs, with held-out macro AUC 0.8256. Its improvement over the 0.5000 train-marginal baseline is 0.3256. This is True; the selected maximum should nevertheless be interpreted as a sweep result rather than as an independently replicated estimate.
