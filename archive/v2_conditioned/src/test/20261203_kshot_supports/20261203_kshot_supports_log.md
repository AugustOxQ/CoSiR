# K-shot supports: exploratory seed-42 check

Purpose: describe frozen AFF and comparator sensitivity to fewer example pairs. These development numbers decide nothing.

Design: the exact seed-42 12,288 development episodes (4,096 per pair), checked against round 6’s per-pair SHA-256s. Keep first k support and first k contrast pairs; queries, 13 candidates and targets stay identical. Condition b swaps the sets. Frozen tune-half h picks score parity 1−h; AFF cells 39/119 and CF cells 149/10. Frozen A0 readers, heads and A3 checkpoint; all four thresholds unchanged. R@1 averages both conditions and directions; condition gain is R@1 minus other-aspect retrieval, in percentage points.

At k=1, inside each half-reader call, replace the six spread features with that half-reader StandardScaler means so they scale to exactly zero; all other features stay unchanged.

Run from `/project/CoSiR`:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= \
/root/miniconda3/envs/CoSiR/bin/python src/test/20261203_kshot_supports/run_ks.py \
>> src/test/20261203_kshot_supports/results/ks_run.log 2>&1
```

C1 PASS: all six pooled k=4 R@1 values equal recorded round 6 values exactly; AFF/CF/cosine/RCA per-episode metrics and original reader features match exactly. B/B0 pass the original `r6_picks.frozen_equals_crossfit` check; verification picks equal the frozen picks.
C2 PASS: cosine’s four raw condition/direction score arrays are identical at all k.
C3 PASS: each episode and reader agreement input has exactly k pairs in each set; factor centring uses 2k items and RCA projection statistics use k pairs. Nested prefixes and unchanged targets/candidates are asserted.
C4 PASS: at k=1 all six spread columns scale to exactly zero for each half-reader and each condition; at k=2,3,4 features and reader probabilities equal the original path exactly.
C5 PASS: every k=2,3,4 result, including every interval, equals the committed JSON exactly (`/tmp/claude-0/-project-CoSiR/f8e5aee2-8d30-4a4a-9b25-7044a814a226/scratchpad/k1/kshot_seed42_k234_committed.json`).

Pooled points (R@1 / condition gain, percentage points):

| Scorer | k=1 | k=2 | k=3 | k=4 |
|---|---:|---:|---:|---:|
| cosine | 12.962 / 0.000 | 12.962 / 0.000 | 12.962 / 0.000 | 12.962 / 0.000 |
| RCA | 13.287 / -0.006 | 13.350 / 0.028 | 13.399 / 0.130 | 13.381 / 0.104 |
| B | 17.861 / 0.000 | 18.119 / 0.000 | 18.294 / 0.000 | 18.341 / 0.000 |
| B0 | 18.182 / 0.000 | 18.392 / 0.000 | 18.406 / 0.000 | 18.437 / 0.000 |
| AFF | 16.457 / 1.990 | 18.579 / 2.580 | 19.041 / 3.099 | 19.137 / 3.111 |
| CF | 17.710 / 0.000 | 18.221 / 0.000 | 18.335 / 0.000 | 18.396 / 0.000 |

Requested pooled differences; point [95% CI], percentage points. 5,000 paired anchor-painting cluster resamples, seed 42, round 6 `r6_stats.bootstrap_draws`; 4602 clusters across all three pairs, shared draws for every difference. No decision test.

| k | AFF−B0 R@1 | AFF−CF R@1 | AFF−CF gain |
|---|---:|---:|---:|
| 1 | -1.725 [-2.051, -1.405] | -1.253 [-1.546, -0.962] | +1.990 [+1.664, +2.314] |
| 2 | +0.187 [-0.065, +0.440] | +0.358 [+0.124, +0.590] | +2.580 [+2.243, +2.922] |
| 3 | +0.635 [+0.397, +0.873] | +0.706 [+0.483, +0.924] | +3.099 [+2.771, +3.428] |
| 4 | +0.700 [+0.460, +0.937] | +0.741 [+0.520, +0.960] | +3.111 [+2.780, +3.456] |

| k−4 | AFF R@1 | AFF gain | B0 R@1 | B0 gain |
|---|---:|---:|---:|---:|
| 1−4 | -2.679 [-2.980, -2.372] | -1.121 [-1.520, -0.736] | -0.254 [-0.415, -0.091] | +0.000 [+0.000, +0.000] |
| 2−4 | -0.557 [-0.773, -0.349] | -0.531 [-0.855, -0.212] | -0.045 [-0.179, +0.085] | +0.000 [+0.000, +0.000] |
| 3−4 | -0.096 [-0.261, +0.069] | -0.012 [-0.264, +0.235] | -0.031 [-0.131, +0.068] | +0.000 [+0.000, +0.000] |
| 4−4 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

AFF gate-open share (fractions):

| k | τ0 a | τ0 b | τ2 a | τ2 b |
|---|---:|---:|---:|---:|
| 1 | 0.835612 | 0.619059 | 0.799723 | 0.571452 |
| 2 | 0.723470 | 0.329671 | 0.562174 | 0.183594 |
| 3 | 0.763184 | 0.299805 | 0.580811 | 0.137044 |
| 4 | 0.809001 | 0.295166 | 0.596191 | 0.111328 |

Aspect pair `emotion__style`; points only (R@1 / condition gain):

| Scorer | k=1 | k=2 | k=3 | k=4 |
|---|---:|---:|---:|---:|
| cosine | 10.059 / 0.000 | 10.059 / 0.000 | 10.059 / 0.000 | 10.059 / 0.000 |
| RCA | 10.150 / 0.006 | 10.193 / -0.061 | 10.144 / 0.018 | 10.175 / -0.043 |
| B | 12.305 / 0.000 | 12.305 / 0.000 | 12.262 / 0.000 | 12.317 / 0.000 |
| B0 | 12.671 / 0.000 | 12.677 / 0.000 | 12.555 / 0.000 | 12.549 / 0.000 |
| AFF | 13.599 / 0.629 | 13.501 / 0.787 | 13.641 / 1.093 | 13.525 / 0.879 |
| CF | 12.970 / 0.000 | 12.897 / 0.000 | 12.720 / 0.000 | 12.769 / 0.000 |

Aspect pair `emotion__genre`; points only (R@1 / condition gain):

| Scorer | k=1 | k=2 | k=3 | k=4 |
|---|---:|---:|---:|---:|
| cosine | 14.594 / 0.000 | 14.594 / 0.000 | 14.594 / 0.000 | 14.594 / 0.000 |
| RCA | 15.021 / -0.122 | 15.137 / 0.049 | 15.216 / 0.159 | 15.234 / 0.244 |
| B | 20.819 / 0.000 | 21.222 / 0.000 | 21.570 / 0.000 | 21.570 / 0.000 |
| B0 | 21.088 / 0.000 | 21.472 / 0.000 | 21.631 / 0.000 | 21.692 / 0.000 |
| AFF | 19.305 / 3.595 | 22.314 / 4.767 | 23.096 / 5.811 | 23.145 / 5.780 |
| CF | 20.483 / 0.000 | 21.234 / 0.000 | 21.509 / 0.000 | 21.515 / 0.000 |

Aspect pair `style__genre`; points only (R@1 / condition gain):

| Scorer | k=1 | k=2 | k=3 | k=4 |
|---|---:|---:|---:|---:|
| cosine | 14.233 / 0.000 | 14.233 / 0.000 | 14.233 / 0.000 | 14.233 / 0.000 |
| RCA | 14.691 / 0.098 | 14.722 / 0.098 | 14.838 / 0.214 | 14.734 / 0.110 |
| B | 20.459 / 0.000 | 20.831 / 0.000 | 21.051 / 0.000 | 21.136 / 0.000 |
| B0 | 20.789 / 0.000 | 21.027 / 0.000 | 21.033 / 0.000 | 21.069 / 0.000 |
| AFF | 16.467 / 1.746 | 19.922 / 2.185 | 20.386 / 2.393 | 20.740 / 2.673 |
| CF | 19.678 / 0.000 | 20.532 / 0.000 | 20.776 / 0.000 | 20.905 / 0.000 |

Runtime: 42.9 seconds (CPU, eight threads), including input loading, k=4 validation, all k and bootstraps.

Adapters: `ks_core.prefix` replaces the four-pair episode validator with exact-prefix/shape validation of the hash-verified, previously validated full episodes; `ks_core.features` implements the original 18-feature arithmetic with a k-aware guard. Factor `_code_sides`/`centered_term` and RCA `_proj_pairs`/`_cov` already derive the count from array shapes and are imported unchanged. The bundle/scorer adapters follow the relevant round 6 calls and omit unrequested scorers.

Deviation/interpretation: there is no saved RCA PCA basis. Its original deterministic PCA calculation is reconstructed once from the exact frozen 60,000-row scorer-train draw, unit normalisation and seed; the resulting basis is fixed across k. No head/reader/checkpoint/threshold or episode-based parameter is fitted or picked. The later saved affect classifier is checked against round 6’s coefficient hashes; other heads use the saved round 6 verified E2 posteriors. The k=4 grid run is solely the explicitly requested reproduction check. No held features or annotations were read; only train/selection feature values are selected from read-only memmaps. Git-dependent import path discovery is replaced in memory with `/project/CoSiR`.

Terminology: the brief describes examples sharing the query’s value. The actual frozen episodes use value-disjoint examples that share the aspect with one another and exclude the anchor’s value; the required existing episodes are preserved.

Previous completed-run runtime: 32.5 seconds.

Implementation note: the first execution stopped after 13.8 seconds at a wrong module reference in the crossfit verification call (AttributeError, after the k=4 numeric comparisons). The reference was corrected and the complete run restarted. No required numerical check failed. Completed-run runtime is 32.5 seconds; both executions total 46.3 seconds. The raw log retains both attempts.

## 2026-10-14 follow-up: split by the aspect the pairs show

The user asked for results within each aspect. `ks_by_condition.py` re-scores the same frozen path (CPU, under a
minute) and splits R@1 by condition: emotion, style, genre told, each pooled over its two aspect pairs. k = 4
reproduces the pooled AFF R@1 (19.14). AFF − B′(A0): emotion +2.88 [+2.40, +3.38] at k = 4, +4.41 at k = 1; style
−0.09 at k = 4, −1.15 at k = 1; genre −0.69 [−1.07, −0.31] at k = 4, −8.44 [−9.11, −7.73] at k = 1. Table in
`results/ks_by_condition.txt` and the report. The user's reading is in the report and the direction log.
