# Candidate 4 fix — 4-seed stress of cutoff 0.15

Generated 2026-09-27 21:15:36. Companion to [`candidate4_fixed_pilot_report.md`](candidate4_fixed_pilot_report.md). Scored against the baseline's own single-label held-out targets throughout.

| seed | macro AUC |
|---:|---:|
| 42 | 0.8533 |
| 7 | 0.8535 |
| 123 | 0.8535 |
| 2024 | 0.8534 |

Summary: mean=0.8534, min=0.8533, max=0.8535, std=0.0001. Baseline: 0.8461.

**Cutoff 0.15 robustly beats the baseline** (0.8534 vs. 0.8461, +0.0073), consistent across all four seeds, not just seed 42. Recommended: adopt this richer multi-label target (0.15 relative cutoff) as the new headline buddy Stage 2 configuration.
