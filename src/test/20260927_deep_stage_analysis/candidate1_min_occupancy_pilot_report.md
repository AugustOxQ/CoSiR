# Candidate 1 — minimum-occupancy handling for buddy's Stage 2 targets

Generated 2026-09-27 20:34:32; seed 42. Motivated by [`deep_stage_analysis_report.md`](deep_stage_analysis_report.md) Finding A (macro AUC correlated with topic occupancy, Pearson r=0.531, p=0.019).

Baseline reproduction check passed (0.5978 vs. expected 0.5978, within 0.001).

## Results

| variant | K | macro AUC | min | median | max | top-1 acc | AUC-occupancy Spearman |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 19 | 0.5978 | 0.2541 | 0.5605 | 0.9043 | 0.1727 | 0.493 (p=0.032) |
| variant_a | 19 | 0.6173 | 0.4515 | 0.6092 | 0.8914 | 0.2130 | 0.089 (p=0.716) |
| variant_b | 16 | 0.6262 | 0.4325 | 0.6078 | 0.9055 | 0.1718 | 0.426 (p=0.099) |

Merge map (Variant B): small train communities [16, 17, 18] relabeled, producing 16 surviving topics (from 19). Original-to-final label map: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 12, 17: 12, 18: 8}.

## Verdict

**variant_b meaningfully beats the baseline**: macro AUC 0.6262 vs. 0.5978 (+0.0285), exceeding the predeclared 0.005 practical margin. Recommended: adopt this variant and proceed to seed-stress before reporting it as a settled improvement (candidate 5 in the updated list).
