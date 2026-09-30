# ArtELingo PercepT Stage 2 patch-attention pilot (BUG-FIXED re-run)

Generated automatically, 2026-09-27 15:47:13.

Fixes the same two bugs as `run_percept_stage1_faithful_recipe_fixed_pilot.py` (center-pruning direction, reconstruction-loss scale), applied here to the K=60/40 Stage-1 configuration this Stage-2 pilot actually depends on -- NOT the K=100/67 faithful-recipe configuration, which is a separate, unrelated Stage-1 fit. See [`docs/reports/auto/percept/2026-09-27_agy_independent_percept_review.md`](../../2026-09-27_agy_independent_percept_review.md) for the full derivation. The LAMBDA_BALANCE=1000 term is left unchanged from the original, to isolate the effect of the two verified bugs alone. Original (buggy) file and report are left unmodified for provenance: `run_percept_stage2_pilot.py`, `percept_stage2_pilot_report.md`.

## Stage-1 K=60/40 seed-42 re-fit (fixed pruning + loss scale)

Held-out emotion AMI: **0.1094** (original buggy citation: 0.1238). Held-out genre AMI: **0.2798** (original buggy citation: 0.2617). This does not attempt to reproduce the original numbers -- they came from the buggy pruning direction and loss scale, so a different result is expected and is the point of this re-run.

Surviving-center occupancy (of 40): min=13, max=860, median=162.5, below 1%=11/40.

## Frozen multi-label target statistics

| split | mean labels | median labels | max labels | fraction multi-labeled |
|---|---:|---:|---:|---:|
| train | 1.000 | 1.000 | 1 | 0.00% |
| held-out | 1.000 | 1.000 | 1 | 0.00% |

## Held-out macro AUC

| scorer | macro AUC | min | median | max | skipped topics |
|---|---:|---:|---:|---:|---:|
| patch-attention mapper (fixed Stage 1) | 0.5925 | 0.2756 | 0.5894 | 0.9143 | 0 |
| train-marginal baseline | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0 |

## Comparison

- Original buggy PercepT Stage 2 macro AUC: 0.5690
- Fixed PercepT Stage 2 macro AUC: **0.5925**
- Buddy's own Stage 2 macro AUC: 0.5978
- **Buddy (0.5978) still beats fixed PercepT (0.5925)**
**The image-only attention-pooling mapper meaningfully beats the marginal-frequency baseline.** Its macro AUC is higher by 0.0925, exceeding the predeclared 0.01 practical margin.
