# CoSiR v2 Candidate A: factor separability and condition recovery

**Verdict: the cross-factor scale hypothesis has limited support; the proposed four-item support noise floor does not explain the dead/live split.** Recovery accuracy rises modestly with the *raw* high/low gap (Spearman ρ = 0.384, p = 0.030), while scale-normalized Cohen's d has essentially no association (ρ = −0.014, p = 0.939). Dataset-wide column standard deviation has the same positive direction but weaker evidence (ρ = 0.310, p = 0.084). The analytical four-item SNR has a significant association in the *opposite* direction from the small-sample-noise prediction (ρ = −0.550, p = 0.0011): factors with larger gap relative to high-pool mean noise tended to recover *less* often. Every factor's empirical ambiguous-mean fraction was 0/1,000, leaving its correlation undefined. Raw activation scale is therefore a plausible contributor to the tied-head collapse, though these 32 observational rows do not establish a normalization fix or a complete cause.

## Five comparisons with Task 5 recovery

The target is Task 5's exact `correct / held-out episodes` for each factor, transcribed from its 32-row table. Spearman coefficients use all 32 factors, including the 18 tied 0%-accuracy values. Reported p-values are SciPy's nominal, two-sided values; they are not adjusted for five comparisons.

| Factor statistic | Spearman ρ vs. accuracy | p-value |
|---|---:|---:|
| Absolute high/low gap | +0.3839 | 0.0301 |
| Cohen's d | −0.0142 | 0.9387 |
| Dataset-wide column standard deviation | +0.3103 | 0.0839 |
| Four-item support-mean SNR | −0.5505 | 0.0011 |
| Empirical ambiguous-mean fraction | Undefined | Undefined |

The last coefficient and p-value are mathematically undefined because the measured fraction is exactly zero for every factor; assigning it a numeric correlation would conceal that lack of variation. The SNR sign matters: the noise hypothesis predicts *higher* SNR for better-recovered factors, not the observed negative association. As a descriptive check, live factors' median gap was 0.20763 versus 0.18873 for dead factors, and median column standard deviation was 0.08305 versus 0.07691. Median SNR was 24.17 for live factors versus 28.62 for dead factors. The raw-gap p-value is nominally below 0.05 but would not pass a Bonferroni threshold of 0.01 across five comparisons, so the scale verdict is deliberately limited.

## Per-factor comparison, sorted by recovery accuracy

| Factor | Task 5 accuracy | Raw gap | Cohen's d | Column SD | Four-item SNR | Ambiguous fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 23/23 (100.00%) | 0.32731 | 9.535 | 0.10621 | 13.497 | 0.000 |
| 21 | 27/27 (100.00%) | 0.16344 | 6.785 | 0.06130 | 14.709 | 0.000 |
| 30 | 21/25 (84.00%) | 0.26256 | 9.136 | 0.10561 | 27.260 | 0.000 |
| 22 | 20/26 (76.92%) | 0.20207 | 7.548 | 0.08111 | 23.345 | 0.000 |
| 10 | 15/21 (71.43%) | 0.21425 | 8.636 | 0.08397 | 24.915 | 0.000 |
| 3 | 14/21 (66.67%) | 0.24827 | 7.717 | 0.10087 | 21.639 | 0.000 |
| 25 | 19/29 (65.52%) | 0.26070 | 8.406 | 0.10730 | 28.684 | 0.000 |
| 13 | 14/24 (58.33%) | 0.19444 | 7.245 | 0.07892 | 21.737 | 0.000 |
| 29 | 14/28 (50.00%) | 0.21257 | 8.103 | 0.08440 | 23.434 | 0.000 |
| 7 | 12/28 (42.86%) | 0.20270 | 8.778 | 0.08213 | 30.547 | 0.000 |
| 27 | 7/27 (25.93%) | 0.19544 | 7.673 | 0.08041 | 27.652 | 0.000 |
| 12 | 1/19 (5.26%) | 0.23433 | 9.027 | 0.09584 | 30.421 | 0.000 |
| 19 | 1/19 (5.26%) | 0.18268 | 5.913 | 0.07403 | 18.282 | 0.000 |
| 28 | 1/24 (4.17%) | 0.17904 | 6.441 | 0.07449 | 25.860 | 0.000 |
| 0 | 0/24 (0.00%) | 0.20342 | 8.479 | 0.07997 | 23.815 | 0.000 |
| 1 | 0/32 (0.00%) | 0.18495 | 8.669 | 0.07574 | 29.431 | 0.000 |
| 2 | 0/23 (0.00%) | 0.19250 | 7.304 | 0.08163 | 31.593 | 0.000 |
| 4 | 0/27 (0.00%) | 0.21080 | 8.784 | 0.08653 | 29.896 | 0.000 |
| 5 | 0/29 (0.00%) | 0.18309 | 8.284 | 0.07507 | 27.919 | 0.000 |
| 6 | 0/26 (0.00%) | 0.18150 | 8.340 | 0.07527 | 30.412 | 0.000 |
| 8 | 0/25 (0.00%) | 0.17866 | 8.180 | 0.07373 | 28.265 | 0.000 |
| 9 | 0/25 (0.00%) | 0.23765 | 7.631 | 0.09849 | 27.211 | 0.000 |
| 11 | 0/27 (0.00%) | 0.17971 | 7.048 | 0.07494 | 25.874 | 0.000 |
| 14 | 0/24 (0.00%) | 0.17488 | 7.240 | 0.07301 | 27.182 | 0.000 |
| 15 | 0/23 (0.00%) | 0.20844 | 8.562 | 0.08575 | 30.082 | 0.000 |
| 17 | 0/24 (0.00%) | 0.17745 | 8.177 | 0.07278 | 26.485 | 0.000 |
| 18 | 0/27 (0.00%) | 0.20593 | 8.507 | 0.08519 | 30.546 | 0.000 |
| 20 | 0/26 (0.00%) | 0.20802 | 8.551 | 0.08560 | 30.213 | 0.000 |
| 23 | 0/24 (0.00%) | 0.19283 | 8.796 | 0.07807 | 28.996 | 0.000 |
| 24 | 0/29 (0.00%) | 0.22424 | 8.593 | 0.09185 | 28.218 | 0.000 |
| 26 | 0/26 (0.00%) | 0.17802 | 8.025 | 0.07367 | 28.576 | 0.000 |
| 31 | 0/38 (0.00%) | 0.18245 | 8.176 | 0.07523 | 28.657 | 0.000 |

## Reproduction and measurement

Run `python src/test/20261004_condition_factor_separability_diagnostic/run_diagnostic.py` from the repository root in the CoSiR environment. The script imports Task 3's `load_real_features` and `prepare_real_codes`, reproducing the positional join of 308,723 ArtELingo CLIP image/text pairs, the raw-feature content graph (3,130,544 edges), Stage 1 and 21 communities, and `SharedFactorEncoder` factor training with 32 columns, `lambda_usage_balance=0.1`, no whitening, and seed 42. This regenerates the prerequisite image/text factor codes; it does **not** train or evaluate `ConditionEncoder`. Pair codes are exactly `0.5 * (img_codes + txt_codes)`.

For pool membership, the script calls `mine_episodes` itself once per one-column factor view with the default 90th/50th percentile settings and seed 42. It requests enough support and contrast items to exhaust both pools, then reconstructs the high pool from anchor + support + positive and the low pool from contrast. All other candidate roles are disabled for these measurement calls. Thus the miner's actual disjoint/tied-cutoff behavior supplies membership without a second percentile implementation. Across factors, high pools held 30,873–30,874 rows and low pools held 154,362–154,364 rows; all passed the miner's default minimum pool size of 50.

For each factor, the script records high/low means and population standard deviations (`ddof=0`), `gap = high_mean − low_mean`, `pooled_std = sqrt((high_std² + low_std²)/2)`, and `Cohen's d = gap/pooled_std`. It also records the entire pair-code column's mean, standard deviation, and maximum. Column means ranged from 0.06854 to 0.13180; maxima ranged from 0.26801 to 0.54064. The analytical SNR is `gap / (high_std / sqrt(4))`, matching the default `num_support=4`. A separate seed-42 generator drew 1,000 four-item high-pool samples with replacement per factor. A draw counted as ambiguous when its mean fell *below* `low_mean + 0.5 * gap`; none did, out of 32,000 draws. The script's `RESULT_JSON` includes every per-factor raw mean, standard deviation, pool size, dataset-wide mean/maximum, five comparison statistics, and exact Task 5 counts.

This fixed-seed analysis is observational. A tied head may depend on combinations of support and contrast codes, factor competition, or other features not summarized by a univariate gap. The 1,000-draw midpoint test cannot estimate extremely rare ambiguous means; its observed zero rate is evidence only at this sampling resolution. The nominal Spearman p-values also do not account for the five comparisons or uncertainty in each factor's Task 5 accuracy estimate.
