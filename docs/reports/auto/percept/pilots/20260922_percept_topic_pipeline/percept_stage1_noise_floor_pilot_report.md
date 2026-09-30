# ArtELingo PercepT Stage 1 seed-42 noise-floor pilot

Generated automatically, 2026-09-23 09:41:22.

## Purpose

Measures run-to-run spread of the Stage-1 K=60/40 seed-42 refit under PyTorch's default (non-deterministic) cuDNN/cuBLAS kernels -- the same kernel behavior every prior "established"/"cited" value in this investigation was produced under. Each repeat is a fresh process invocation with identical seed 42, identical code, identical data; any spread between repeats is pure uncontrolled GPU nondeterminism, not seed sensitivity.

## Full 5-repeat results

| repeat | held-out emotion AMI | held-out genre AMI |
|---:|---:|---:|
| 1 | 0.1225 | 0.2274 |
| 2 | 0.1225 | 0.2274 |
| 3 | 0.1225 | 0.2274 |
| 4 | 0.1225 | 0.2274 |
| 5 | 0.1225 | 0.2274 |

## Spread statistics

| metric | mean | min | max | sample standard deviation |
|---|---:|---:|---:|---:|
| held-out emotion AMI | 0.1225 | 0.1225 | 0.1225 | 0.0000 |
| held-out genre AMI | 0.2274 | 0.2274 | 0.2274 | 0.0000 |

## Comparison against the between-seed spread

The 14-seed Stage 1 extended-seed pilot (`percept_stage1_extended_seed_pilot_report.md`) measured a between-*different*-seed standard deviation of 0.0032 (emotion) and 0.0156 (genre). This pilot's within-*same*-seed noise-floor standard deviation is 0.0000 (emotion) and 0.0000 (genre) across 5 repeats.

**Finding: the noise floor is smaller than the between-seed spread on both axes.** Uncontrolled GPU nondeterminism, while nonzero, is not large enough on its own to explain the between-seed spread the 14-seed extended-seed pilot reported; that pilot's "seed-dependent result" verdict is not primarily a nondeterminism artifact.
