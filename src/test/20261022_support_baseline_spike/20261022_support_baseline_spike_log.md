# 20261022 support-baseline spike log

## Question
Do simple support-set baselines on raw CLIP features (prototype, per-episode logistic probe, direction) match the learned factor model SE on the same 4,096+4,096 selection episodes? Throwaway; nothing committed; selection rows only (all other feature rows NaN, asserted).

## What was run
`run_spike.py` (CPU, 22 s). Checks: (1) episode SHA-256 equals stored for both labels; (2) CLIP only reproduces stored `clip_only__-__0.3` ranks exactly; (3) SE and C0 naive beta 0.3 reproduce stored ranks exactly. All three passed (exact array equality, both labels, both directions).
Scorers: z(cos)+lam*z(T) with lam in {0,.25,.5,1,2,4,inf}; terms proto, proto_pos, probe, probe_x, dir, SE_term, C0_term; raw_naive on beta grid {0,.03,.1,.3,1}. Cross-fit by episode-index parity (lam tuned on pooled mean-of-directions R@1 of one parity half over both labels, applied to the other half; first grid value wins ties).
Probe: batched torch Newton solver in the dual form of sklearn's objective (C=1, L2, free intercept). Within-episode ordering equals sklearn (tol 1e-10) on 100% of 50 random episodes per setting; vs sklearn default tol 1e-4, 96% on emotion i2t (lbfgs stopping noise, not the objective).

## R@1 % (mean of directions), cross-fitted

| scorer | pooled | emotion | style | lam/beta picked (fold0, fold1) |
|---|---|---|---|---|
| proto_cv | 23.40 | 15.77 | 31.03 | lam4, lam4 |
| proto_pos_cv | 20.81 | 12.76 | 28.87 | laminf, laminf |
| probe_cv | 23.57 | 15.89 | 31.24 | lam4, lam4 |
| probe_x_cv | 24.10 | 16.56 | 31.63 | lam4, lam4 |
| dir_cv | 18.73 | 11.76 | 25.71 | lam4, lam4 |
| SE_term_cv | 21.91 | 17.46 | 26.35 | laminf, laminf |
| C0_term_cv | 20.45 | 15.83 | 25.07 | laminf, laminf |
| raw_naive_cv | 13.46 | 10.88 | 16.05 | beta1, beta1 |
| SE (stored naive b0.3 / clip only) | 21.22 | 16.43 | 26.01 | |
| C0 (stored naive b0.3 / clip only) | 20.01 | 15.12 | 24.89 | |
| R3 (stored naive b0.3 / clip only) | 19.79 | 15.76 | 23.82 | |
| CLIP_only (stored naive b0.3 / clip only) | 13.45 | 10.83 | 16.06 | |

## Paired difference vs SE (pp R@1, mean of directions, bootstrap 5000 seed 42)

| scorer | pooled | emotion | style |
|---|---|---|---|
| proto_cv | +2.18 [+1.43, +2.96] | -0.66 [-1.68, +0.35] | +5.02 [+3.87, +6.16] |
| proto_pos_cv | -0.41 [-1.20, +0.42] | -3.67 [-4.71, -2.61] | +2.86 [+1.66, +4.04] |
| probe_cv | +2.34 [+1.61, +3.13] | -0.54 [-1.58, +0.49] | +5.22 [+4.08, +6.37] |
| probe_x_cv | +2.87 [+2.14, +3.64] | +0.13 [-0.89, +1.15] | +5.62 [+4.48, +6.71] |
| dir_cv | -2.49 [-3.23, -1.76] | -4.68 [-5.68, -3.65] | -0.31 [-1.42, +0.82] |
| SE_term_cv | +0.68 [+0.33, +1.04] | +1.03 [+0.51, +1.54] | +0.34 [-0.15, +0.84] |
| C0_term_cv | -0.77 [-1.39, -0.18] | -0.60 [-1.44, +0.23] | -0.94 [-1.81, -0.05] |
| raw_naive_cv | -7.76 [-8.47, -7.06] | -5.55 [-6.47, -4.65] | -9.96 [-10.99, -8.94] |

## Paired difference vs C0 (pp R@1, mean of directions, bootstrap 5000 seed 42)

| scorer | pooled | emotion | style |
|---|---|---|---|
| proto_cv | +3.39 [+2.65, +4.17] | +0.65 [-0.39, +1.68] | +6.14 [+5.00, +7.29] |
| proto_pos_cv | +0.81 [+0.04, +1.63] | -2.37 [-3.44, -1.31] | +3.98 [+2.77, +5.19] |
| probe_cv | +3.56 [+2.81, +4.35] | +0.77 [-0.31, +1.83] | +6.35 [+5.21, +7.52] |
| probe_x_cv | +4.09 [+3.36, +4.86] | +1.44 [+0.35, +2.48] | +6.74 [+5.60, +7.85] |
| dir_cv | -1.28 [-2.00, -0.54] | -3.37 [-4.36, -2.39] | +0.82 [-0.28, +1.93] |
| SE_term_cv | +1.90 [+1.29, +2.50] | +2.33 [+1.49, +3.17] | +1.46 [+0.62, +2.28] |
| C0_term_cv | +0.45 [+0.09, +0.80] | +0.71 [+0.21, +1.23] | +0.18 [-0.29, +0.66] |
| raw_naive_cv | -6.54 [-7.21, -5.88] | -4.25 [-5.09, -3.38] | -8.84 [-9.89, -7.79] |

## Surprises / caveats
- lam picks sit at the grid edge (lam=4 for proto/probe/probe_x/dir, inf for proto_pos and the factor terms): the grid is truncated for the prototype and probe terms, so their numbers are, if anything, conservative.
- SE_term_cv (factor term alone, tuned) beats stored SE (naive beta 0.3, untuned) by +0.68 pp pooled, so SE's headline sits slightly below its own tuned optimum; the fair SE comparator is SE_term_cv.
- raw_naive is useless (13.46 vs CLIP only 13.45): beta 1 was picked at the grid edge. Raw unit-norm CLIP coordinates are ~1/sqrt(512) in size, so the factor term is ~1e-3 against cos in [-1,1]; beta does not transfer to raw coordinates. Not a fair test of the naive rule on raw features.
- Emotion: probe_x and proto are within the noise of SE (+0.13 and -0.66 pp, CIs include 0). Style: all support-set prototype/probe baselines beat SE by about +5 pp. dir (query-inclusive) loses to SE on emotion.
- Chance R@1 is 7.7% (13 candidates). Tuning uses only selection episodes, so cross-fit halves share paintings; the parity split does not separate paintings.
