# Reader fix round 2: independent re-derivation, phase 2 (comparison with the main run)

Written 2026-10-06 16:57 (Amsterdam) by the independent re-derivation agent under `../DECISION_RULE.md` (SHA-256
368bec11363b222d348622a37a6e3aedcfe795d772781d7a255d79b60fab265c). Phase 2 started after the controller's message that
the main run's regression check had passed (`results/regression_check.json`, at HEAD cdf9b4d); the marker
`out/PHASE2_AUTHORISED` was created then. Only the main run's result files (`results/*.json`, `*.npz`, `*.txt`) were
read; the implementation's `.py` files were never opened. Our candidate numbers (`rd2_phase2.py`, 59 s) were computed
before any stored candidate number was read; the comparison is `rd2_compare.py`.

## 1. Verdict of the comparison

**Overall agreement: yes.** We compared 480 quantities. The 164 that rule §7 lists as decision numbers are all
identical, and so are the other 316, which fall outside §7. Every float difference is exactly 0: bar margins, gain
statistics, bounds, τ, μ42, σ42, π̂, D(k), the 72 SMDs and the CV log losses. Every per-anchor array, probability
array, T, margin, pick and gate array is bit-identical. There is no disagreement to trace.

| Candidate | Bar margin [95% interval] (comparator) | Gain statistic [95% interval] | Clauses 1 / 2 / 3 | Clears bar |
|---|---|---|---|---|
| R1 / A0 | +0.4720052083333333 [0.24030286496949704, 0.7028163072942406] (counterpart) | 2.667236328125 [2.325087836946873, 3.012361650695922] | no / yes / yes | **no** |
| R2 / A0 | +0.115966796875 [−0.07289118013657525, 0.30106240852848065] (B′) | 0.9175618489583334 [0.6617526421912794, 1.1691833632202837] | no / no / yes | **no** |
| R3 / A0 | +0.07731119791666666 [−0.11125362620492793, 0.2709495572650762] (B′) | 0.9541829427083334 [0.7021970530564084, 1.1988815133025827] | no / no / yes | **no** |

So no candidate clears the development bar by our own computation either. The largest bar margin belongs to R1, which
by §4.8 would be the "best development candidate, not carried". Applying the rule is the controller's step; this
comparison only confirms the inputs it will use.

Chosen cells, by both implementations (identical):

| Candidate | fused, tune half 0 | fused, tune half 1 | counterpart, half 0 | counterpart, half 1 | σ* (h0, h1) | τ_0..τ_3 |
|---|---|---|---|---|---|---|
| R1 | 116 (k 13, τ_2, 0, 2) | 119 (k 13, τ_2, 0, 16) | 58 (k 13, τ_1, 0, 0.5) | 571 (k 3, τ_2, 0.5, 1) | 0, 0 | 3.8684538364530674e-05, 0.21702129490553143, 0.47973989883399526, 0.7502585816077211 |
| R2 | 67 (k 13, τ_1, 0.5, 1) | 615 (k 3, τ_2, 16, 16) | 278 (k 5, τ_0, 16, 8) | 155 (k 13, τ_2, 8, 1) | 0, 0 | 3.771746264230602e-05, 0.2907081713292494, 0.5994124414690893, 0.861841000813007 |
| R3 | 55 (k 13, τ_0, 16, 16) | 574 (k 3, τ_2, 0.5, 8) | 157 (k 13, τ_2, 8, 4) | 156 (k 13, τ_2, 8, 2) | 0, 0 | 3.399954748600775e-05, 0.12456908995572917, 0.2854599358396749, 0.4974364340057035 |

Ties at the chosen maximum (resolved to the lowest cell by both implementations): R1 fused half 0 (4 cells), R1
counterpart half 0 (4) and half 1 (2), R2 fused half 0 (2), R2 counterpart half 1 (2), R3 fused half 1 (2), R3
counterpart half 0 (4) and half 1 (2). The integer criteria (ρ, γ, ρ_ctrl) of each pick also agree (rows outside §7).

## 2. What was compared

- **Rule §7 decision numbers (164 rows, table in §4):** the regression check; for each candidate τ_0..τ_3, the fused
  and counterpart cells per tune half with their settings, σ* per half, the bar comparator, the bar margin and its bounds,
  the gain statistic and its bounds, the three D12 clauses and clears_bar, and the per-anchor fused/counterpart r1 and
  gain arrays with bar_v; for R2 μ42 (18), σ42 (18, none zero), π̂ (3), the iteration count (stored `n_iter` 39 = our 39
  updates; stop at s = 38), cap not reached, code check; for R3 D(1..4), k* = 2 (r3_k_A0.json and probs_R3_A0.json),
  the 14 SHA-256s (order, img, cap and the purity-1..4 feature matrices per half; both sides hash the stacked
  (2N, 18) float64 matrix, C order) and the chosen C per half (100 and 10).
- **Outside §7 (316 rows, all identical; full table `out/rd2_compare_table.md`):** mean R@1 of fused, counterpart, B,
  B′; the margin, fused − B, fused − B′ and counterpart − B blocks (R@1, gain, either; point and bounds); bar margin per
  aspect pair; pick accuracy; the per-anchor other/swap/strict arrays; probs, picks, margins, T and gates in each
  `cand_*_A0.npz` against ours; anchor_group, pair_index, parity; R2's P′ (`probs_R2_A0.npz`) and last EM change; R3's
  probabilities (`probs_R3_A0.npz`, bit-identical), the 72 SMDs, the 10 CV mean log losses, OOF accuracies and the
  redraw counts.

Tolerances applied (§7): discrete identical; μ42, σ42, π̂, τ, D(k) within 1e-12 relative (absolute 1e-12 for the exactly
zero Δ entries of μ42); margins, gain statistics and bounds within 1e-9 pp. None was needed: all differences are 0.

## 3. Paired differences of §5 item 2 (separate step; decides nothing)

Our own numbers, for checking `r2_apply_rule.py`'s output later. Paired per anchor, painting bootstrap (5,000 resamples,
seed 42), pp. Round-1 R-c's arrays are `fused__r1`, `cf__r1`, `bar_v` of `cand_Rc_Rb_expected_A0.npz`; "margin" is the
fused R@1 minus the matched counterpart's R@1.

| Difference | Fused R@1 | Margin (R@1) | Bar margin (R@1) |
|---|---|---|---|
| R1 − round-1 R-c (effect of the top-k restriction) | 0.0 [0.0, 0.0] | +0.028483072916666668 [−0.002015970518766697, 0.05859603924379825] | +0.028483072916666668 [−0.002015970518766697, 0.05859603924379825] |
| R2 − R1 | −0.3662109375 [−0.5810134324794499, −0.14946765617208524] | −0.20955403645833334 [−0.4497728303413582, 0.03220754789030175] | −0.35603841145833337 [−0.6121769045475611, −0.10586364786038095] |
| R3 − R1 | −0.4048665364583333 [−0.6242973676640935, −0.18615867040603357] | −0.384521484375 [−0.6330981481079135, −0.12273462219873763] | −0.3946940104166667 [−0.6408379764487693, −0.1461624808083971] |

Reading (ours, descriptive). For R1 the top-k restriction changed nothing on the fused side: both fused picks stay at
k_top = 13 (cells 116 and 119), so its fused R@1 is round-1 R-c's exactly (18.918863932291664). The only change is the
counterpart's half-1 pick. It moved to cell 571 (k_top 3), which scores 4,509 hits on its tune half against 4,502 for
round 1's cell 123. Applied to the other half, it does worse: counterpart R@1 is 18.447 against round 1's 18.475. The
bar margin rose by 0.028 (from 0.4435 to 0.4720) because the control's extra freedom chose worse out of sample. The
reader did not improve. Both reader fixes lower the fused R@1 by about 0.37 to 0.40 against R1, and their gain
statistics (0.92, 0.95) are about a third of R1's (2.67).

Pick accuracy (D13, diagnostic): R1 51.261393229166664 [50.66, 51.84], R2 47.24527994791667 [46.68, 47.83], R3
55.692545572916664 [55.09, 56.30] (chance 33.3), all identical between the two implementations.

## 4. Rule §7 comparison table (164 rows)

| group | quantity | stored | re-derived | abs diff | agree |
|---|---|---|---|---|---|
| regression | passed | True | True |  | yes |
| regression | all 34 comparisons ok | True | True |  | yes |
| R1/A0 | tau_0 (tau_R1_A0.json) | 3.8684538364530674e-05 | 3.8684538364530674e-05 | 0 | yes |
| R1/A0 | tau_1 (tau_R1_A0.json) | 0.21702129490553143 | 0.21702129490553143 | 0 | yes |
| R1/A0 | tau_2 (tau_R1_A0.json) | 0.47973989883399526 | 0.47973989883399526 | 0 | yes |
| R1/A0 | tau_3 (tau_R1_A0.json) | 0.7502585816077211 | 0.7502585816077211 | 0 | yes |
| R1/A0 | tau (cand json) equal tau json | True | True |  | yes |
| R1/A0 | fused cell, tune half 0 | 116 | 116 |  | yes |
| R1/A0 | fused cell settings, half 0 (k_top, tau idx, lam_u, lam_a) | [13, 2, 0.0, 2.0] | [13, 2, 0.0, 2.0] |  | yes |
| R1/A0 | counterpart cell, tune half 0 | 58 | 58 |  | yes |
| R1/A0 | counterpart cell settings, half 0 | [13, 1, 0.0, 0.5] | [13, 1, 0.0, 0.5] |  | yes |
| R1/A0 | control sigma*, half 0 | 0.0 | 0.0 |  | yes |
| R1/A0 | fused cell, tune half 1 | 119 | 119 |  | yes |
| R1/A0 | fused cell settings, half 1 (k_top, tau idx, lam_u, lam_a) | [13, 2, 0.0, 16.0] | [13, 2, 0.0, 16.0] |  | yes |
| R1/A0 | counterpart cell, tune half 1 | 571 | 571 |  | yes |
| R1/A0 | counterpart cell settings, half 1 | [3, 2, 0.5, 1.0] | [3, 2, 0.5, 1.0] |  | yes |
| R1/A0 | control sigma*, half 1 | 0.0 | 0.0 |  | yes |
| R1/A0 | npz fused_cells / cf_cells / ctrl_sigma | [[116, 119], [58, 571], [0.0, 0.0]] | [[116, 119], [58, 571], [0.0, 0.0]] |  | yes |
| R1/A0 | bar comparator | counterpart | counterpart |  | yes |
| R1/A0 | bar margin point | 0.4720052083333333 | 0.4720052083333333 | 0 | yes |
| R1/A0 | bar margin lower | 0.24030286496949704 | 0.24030286496949704 | 0 | yes |
| R1/A0 | bar margin upper | 0.7028163072942406 | 0.7028163072942406 | 0 | yes |
| R1/A0 | gain statistic point | 2.667236328125 | 2.667236328125 | 0 | yes |
| R1/A0 | gain statistic lower | 2.325087836946873 | 2.325087836946873 | 0 | yes |
| R1/A0 | gain statistic upper | 3.012361650695922 | 3.012361650695922 | 0 | yes |
| R1/A0 | clause 1 (bar point >= 0.5) | False | False |  | yes |
| R1/A0 | clause 2 (bar lower > 0) | True | True |  | yes |
| R1/A0 | clause 3 (gain lower > 0) | True | True |  | yes |
| R1/A0 | clears_bar | False | False |  | yes |
| R1/A0 | per-anchor fused__r1 | array (12288,) float64 | array (12288,) float64 | 0 | yes |
| R1/A0 | per-anchor fused__gain | array (12288,) float64 | array (12288,) float64 | 0 | yes |
| R1/A0 | per-anchor cf__r1 | array (12288,) float64 | array (12288,) float64 | 0 | yes |
| R1/A0 | per-anchor cf__gain | array (12288,) float64 | array (12288,) float64 | 0 | yes |
| R1/A0 | per-anchor bar_v | array (12288,) float64 | array (12288,) float64 | 0 | yes |
| R2/A0 | tau_0 (tau_R2_A0.json) | 3.771746264230602e-05 | 3.771746264230602e-05 | 0 | yes |
| R2/A0 | tau_1 (tau_R2_A0.json) | 0.2907081713292494 | 0.2907081713292494 | 0 | yes |
| R2/A0 | tau_2 (tau_R2_A0.json) | 0.5994124414690893 | 0.5994124414690893 | 0 | yes |
| R2/A0 | tau_3 (tau_R2_A0.json) | 0.861841000813007 | 0.861841000813007 | 0 | yes |
| R2/A0 | tau (cand json) equal tau json | True | True |  | yes |
| R2/A0 | fused cell, tune half 0 | 67 | 67 |  | yes |
| R2/A0 | fused cell settings, half 0 (k_top, tau idx, lam_u, lam_a) | [13, 1, 0.5, 1.0] | [13, 1, 0.5, 1.0] |  | yes |
| R2/A0 | counterpart cell, tune half 0 | 278 | 278 |  | yes |
| R2/A0 | counterpart cell settings, half 0 | [5, 0, 16.0, 8.0] | [5, 0, 16.0, 8.0] |  | yes |
| R2/A0 | control sigma*, half 0 | 0.0 | 0.0 |  | yes |
| R2/A0 | fused cell, tune half 1 | 615 | 615 |  | yes |
| R2/A0 | fused cell settings, half 1 (k_top, tau idx, lam_u, lam_a) | [3, 2, 16.0, 16.0] | [3, 2, 16.0, 16.0] |  | yes |
| R2/A0 | counterpart cell, tune half 1 | 155 | 155 |  | yes |
| R2/A0 | counterpart cell settings, half 1 | [13, 2, 8.0, 1.0] | [13, 2, 8.0, 1.0] |  | yes |
| R2/A0 | control sigma*, half 1 | 0.0 | 0.0 |  | yes |
| R2/A0 | npz fused_cells / cf_cells / ctrl_sigma | [[67, 615], [278, 155], [0.0, 0.0]] | [[67, 615], [278, 155], [0.0, 0.0]] |  | yes |
| R2/A0 | bar comparator | B_prime | B_prime |  | yes |
| R2/A0 | bar margin point | 0.115966796875 | 0.115966796875 | 0 | yes |
| R2/A0 | bar margin lower | -0.07289118013657525 | -0.07289118013657525 | 0 | yes |
| R2/A0 | bar margin upper | 0.30106240852848065 | 0.30106240852848065 | 0 | yes |
| R2/A0 | gain statistic point | 0.9175618489583334 | 0.9175618489583334 | 0 | yes |
| R2/A0 | gain statistic lower | 0.6617526421912794 | 0.6617526421912794 | 0 | yes |
| R2/A0 | gain statistic upper | 1.1691833632202837 | 1.1691833632202837 | 0 | yes |
| R2/A0 | clause 1 (bar point >= 0.5) | False | False |  | yes |
| R2/A0 | clause 2 (bar lower > 0) | False | False |  | yes |
| R2/A0 | clause 3 (gain lower > 0) | True | True |  | yes |
| R2/A0 | clears_bar | False | False |  | yes |
| R2/A0 | per-anchor fused__r1 | array (12288,) float64 | array (12288,) float64 | 0 | yes |
| R2/A0 | per-anchor fused__gain | array (12288,) float64 | array (12288,) float64 | 0 | yes |
| R2/A0 | per-anchor cf__r1 | array (12288,) float64 | array (12288,) float64 | 0 | yes |
| R2/A0 | per-anchor cf__gain | array (12288,) float64 | array (12288,) float64 | 0 | yes |
| R2/A0 | per-anchor bar_v | array (12288,) float64 | array (12288,) float64 | 0 | yes |
| R3/A0 | tau_0 (tau_R3_A0.json) | 3.399954748600775e-05 | 3.399954748600775e-05 | 0 | yes |
| R3/A0 | tau_1 (tau_R3_A0.json) | 0.12456908995572917 | 0.12456908995572917 | 0 | yes |
| R3/A0 | tau_2 (tau_R3_A0.json) | 0.2854599358396749 | 0.2854599358396749 | 0 | yes |
| R3/A0 | tau_3 (tau_R3_A0.json) | 0.4974364340057035 | 0.4974364340057035 | 0 | yes |
| R3/A0 | tau (cand json) equal tau json | True | True |  | yes |
| R3/A0 | fused cell, tune half 0 | 55 | 55 |  | yes |
| R3/A0 | fused cell settings, half 0 (k_top, tau idx, lam_u, lam_a) | [13, 0, 16.0, 16.0] | [13, 0, 16.0, 16.0] |  | yes |
| R3/A0 | counterpart cell, tune half 0 | 157 | 157 |  | yes |
| R3/A0 | counterpart cell settings, half 0 | [13, 2, 8.0, 4.0] | [13, 2, 8.0, 4.0] |  | yes |
| R3/A0 | control sigma*, half 0 | 0.0 | 0.0 |  | yes |
| R3/A0 | fused cell, tune half 1 | 574 | 574 |  | yes |
| R3/A0 | fused cell settings, half 1 (k_top, tau idx, lam_u, lam_a) | [3, 2, 0.5, 8.0] | [3, 2, 0.5, 8.0] |  | yes |
| R3/A0 | counterpart cell, tune half 1 | 156 | 156 |  | yes |
| R3/A0 | counterpart cell settings, half 1 | [13, 2, 8.0, 2.0] | [13, 2, 8.0, 2.0] |  | yes |
| R3/A0 | control sigma*, half 1 | 0.0 | 0.0 |  | yes |
| R3/A0 | npz fused_cells / cf_cells / ctrl_sigma | [[55, 574], [157, 156], [0.0, 0.0]] | [[55, 574], [157, 156], [0.0, 0.0]] |  | yes |
| R3/A0 | bar comparator | B_prime | B_prime |  | yes |
| R3/A0 | bar margin point | 0.07731119791666666 | 0.07731119791666666 | 0 | yes |
| R3/A0 | bar margin lower | -0.11125362620492793 | -0.11125362620492793 | 0 | yes |
| R3/A0 | bar margin upper | 0.2709495572650762 | 0.2709495572650762 | 0 | yes |
| R3/A0 | gain statistic point | 0.9541829427083334 | 0.9541829427083334 | 0 | yes |
| R3/A0 | gain statistic lower | 0.7021970530564084 | 0.7021970530564084 | 0 | yes |
| R3/A0 | gain statistic upper | 1.1988815133025827 | 1.1988815133025827 | 0 | yes |
| R3/A0 | clause 1 (bar point >= 0.5) | False | False |  | yes |
| R3/A0 | clause 2 (bar lower > 0) | False | False |  | yes |
| R3/A0 | clause 3 (gain lower > 0) | True | True |  | yes |
| R3/A0 | clears_bar | False | False |  | yes |
| R3/A0 | per-anchor fused__r1 | array (12288,) float64 | array (12288,) float64 | 0 | yes |
| R3/A0 | per-anchor fused__gain | array (12288,) float64 | array (12288,) float64 | 0 | yes |
| R3/A0 | per-anchor cf__r1 | array (12288,) float64 | array (12288,) float64 | 0 | yes |
| R3/A0 | per-anchor cf__gain | array (12288,) float64 | array (12288,) float64 | 0 | yes |
| R3/A0 | per-anchor bar_v | array (12288,) float64 | array (12288,) float64 | 0 | yes |
| R2 mu42 | affect__S | 0.03750044020747131 | 0.03750044020747131 | 0 | yes |
| R2 mu42 | affect__C | 0.03750044020747131 | 0.03750044020747131 | 0 | yes |
| R2 mu42 | affect__Delta | 0.0 | 0.0 | 0 | yes |
| R2 mu42 | affect__sd_support | 0.012186092079906357 | 0.012186092079906357 | 0 | yes |
| R2 mu42 | affect__sd_contrast | 0.012186092079906256 | 0.012186092079906256 | 0 | yes |
| R2 mu42 | affect__argmax_match | 0.09319051106770833 | 0.09319051106770833 | 0 | yes |
| R2 mu42 | image__S | 0.03357454000131099 | 0.03357454000131099 | 0 | yes |
| R2 mu42 | image__C | 0.03357454000131099 | 0.03357454000131099 | 0 | yes |
| R2 mu42 | image__Delta | 0.0 | 0.0 | 0 | yes |
| R2 mu42 | image__sd_support | 0.03537311514088052 | 0.03537311514088052 | 0 | yes |
| R2 mu42 | image__sd_contrast | 0.03537311514088042 | 0.03537311514088042 | 0 | yes |
| R2 mu42 | image__argmax_match | 0.062255859375 | 0.062255859375 | 0 | yes |
| R2 mu42 | caption__S | 0.029346030526260353 | 0.029346030526260353 | 0 | yes |
| R2 mu42 | caption__C | 0.029346030526260353 | 0.029346030526260353 | 0 | yes |
| R2 mu42 | caption__Delta | 0.0 | 0.0 | 0 | yes |
| R2 mu42 | caption__sd_support | 0.026659937123360792 | 0.026659937123360792 | 0 | yes |
| R2 mu42 | caption__sd_contrast | 0.02665993712336069 | 0.02665993712336069 | 0 | yes |
| R2 mu42 | caption__argmax_match | 0.04376220703125 | 0.04376220703125 | 0 | yes |
| R2 sigma42 | affect__S | 0.007849502536376857 | 0.007849502536376857 | 0 | yes |
| R2 sigma42 | affect__C | 0.007849502536376863 | 0.007849502536376863 | 0 | yes |
| R2 sigma42 | affect__Delta | 0.011123818700800807 | 0.011123818700800807 | 0 | yes |
| R2 sigma42 | affect__sd_support | 0.00984874049618295 | 0.00984874049618295 | 0 | yes |
| R2 sigma42 | affect__sd_contrast | 0.009848740496182914 | 0.009848740496182914 | 0 | yes |
| R2 sigma42 | affect__argmax_match | 0.14525199401530955 | 0.14525199401530955 | 0 | yes |
| R2 sigma42 | image__S | 0.031862780677996125 | 0.031862780677996125 | 0 | yes |
| R2 sigma42 | image__C | 0.03186278067799615 | 0.03186278067799615 | 0 | yes |
| R2 sigma42 | image__Delta | 0.048657327089135255 | 0.048657327089135255 | 0 | yes |
| R2 sigma42 | image__sd_support | 0.04698367349249152 | 0.04698367349249152 | 0 | yes |
| R2 sigma42 | image__sd_contrast | 0.04698367349249159 | 0.04698367349249159 | 0 | yes |
| R2 sigma42 | image__argmax_match | 0.12397833376970978 | 0.12397833376970978 | 0 | yes |
| R2 sigma42 | caption__S | 0.020214559591368704 | 0.020214559591368704 | 0 | yes |
| R2 sigma42 | caption__C | 0.020214559591368718 | 0.020214559591368718 | 0 | yes |
| R2 sigma42 | caption__Delta | 0.030127717583177564 | 0.030127717583177564 | 0 | yes |
| R2 sigma42 | caption__sd_support | 0.027450448085455683 | 0.027450448085455683 | 0 | yes |
| R2 sigma42 | caption__sd_contrast | 0.027450448085455614 | 0.027450448085455614 | 0 | yes |
| R2 sigma42 | caption__argmax_match | 0.10264521961959072 | 0.10264521961959072 | 0 | yes |
| R2 sigma42 | zero entries replaced | [] | [] |  | yes |
| R2 EM | pi_hat[affect] | 0.33954936858584167 | 0.33954936858584167 | 0 | yes |
| R2 EM | pi_hat[image] | 0.31214248949301643 | 0.31214248949301643 | 0 | yes |
| R2 EM | pi_hat[caption] | 0.348308141921142 | 0.348308141921142 | 0 | yes |
| R2 EM | iterations (stored n_iter vs our updates computed) | 39 | 39 |  | yes |
| R2 EM | cap reached | False | False |  | yes |
| R2 code check | passed | True | True |  | yes |
| R3 D(k) | D(1) | 0.10945540348228223 | 0.10945540348228223 | 0 | yes |
| R3 D(k) | D(2) | 0.08964658887594207 | 0.08964658887594207 | 0 | yes |
| R3 D(k) | D(3) | 0.18506833918606608 | 0.18506833918606608 | 0 | yes |
| R3 D(k) | D(4) | 0.27487353947498383 | 0.27487353947498383 | 0 | yes |
| R3 | k* | 2 | 2 |  | yes |
| R3 | k* (probs_R3_A0.json) | 2 | 2 |  | yes |
| R3 SHA-256 | half 0 order | 8ef89646cb27fe53a07ba210eed5afcc581000fce74d8e980fe81a171597a9db | 8ef89646cb27fe53a07ba210eed5afcc581000fce74d8e980fe81a171597a9db |  | yes |
| R3 SHA-256 | half 0 img | 60418d2fe5350e53d117de1137833673af516066d4ff817a3e79858cea5f4a6c | 60418d2fe5350e53d117de1137833673af516066d4ff817a3e79858cea5f4a6c |  | yes |
| R3 SHA-256 | half 0 cap | afcb783e03d791a0a7b892a2a0d1253456f1ea01dc5acbc485f21172d081f695 | afcb783e03d791a0a7b892a2a0d1253456f1ea01dc5acbc485f21172d081f695 |  | yes |
| R3 SHA-256 | half 0 features purity 1 | 1da07a4ec2a4f130005042caca49561f5f91fba8cb19b5303de89614c328bd5a | 1da07a4ec2a4f130005042caca49561f5f91fba8cb19b5303de89614c328bd5a |  | yes |
| R3 SHA-256 | half 0 features purity 2 | 48692d40d8bd37b59239c29f4626e4085c77346e7f4873ebc4f24cc36ce07f49 | 48692d40d8bd37b59239c29f4626e4085c77346e7f4873ebc4f24cc36ce07f49 |  | yes |
| R3 SHA-256 | half 0 features purity 3 | c296740a785e9d259946eedcdf0c7c1fa6f6b94a8a7f0bd2f74746242d48460a | c296740a785e9d259946eedcdf0c7c1fa6f6b94a8a7f0bd2f74746242d48460a |  | yes |
| R3 SHA-256 | half 0 features purity 4 | fc0fa4c9e022f7688f159c1da326afc5fad879ec2b14c953b5190e26e794d7a8 | fc0fa4c9e022f7688f159c1da326afc5fad879ec2b14c953b5190e26e794d7a8 |  | yes |
| R3 SHA-256 | half 1 order | 43a0f2f8d92fcef6b08d5741714f3fa700c590d4d86541b61a7c72c3014eeb6e | 43a0f2f8d92fcef6b08d5741714f3fa700c590d4d86541b61a7c72c3014eeb6e |  | yes |
| R3 SHA-256 | half 1 img | 7947a3fafedc170666eee83a0ba86cfedf13ff3fc2d2f0c3c181d1efb751be05 | 7947a3fafedc170666eee83a0ba86cfedf13ff3fc2d2f0c3c181d1efb751be05 |  | yes |
| R3 SHA-256 | half 1 cap | f9501f8372c17aee3825e856fe68ece4828eed6d76ca6e6c382f1c5dd33900c8 | f9501f8372c17aee3825e856fe68ece4828eed6d76ca6e6c382f1c5dd33900c8 |  | yes |
| R3 SHA-256 | half 1 features purity 1 | fae2f096f7e4d642546671bc1e140c3784283b819e9495e771efdd0077957c2c | fae2f096f7e4d642546671bc1e140c3784283b819e9495e771efdd0077957c2c |  | yes |
| R3 SHA-256 | half 1 features purity 2 | 9b3fa584df66109b3c7eda07f9883642dcd8eb9f018e8d535d20a08b998975bb | 9b3fa584df66109b3c7eda07f9883642dcd8eb9f018e8d535d20a08b998975bb |  | yes |
| R3 SHA-256 | half 1 features purity 3 | c2752dab47159f49eeb4eeed262c708bf463ef274898bf52363746ee382bfac0 | c2752dab47159f49eeb4eeed262c708bf463ef274898bf52363746ee382bfac0 |  | yes |
| R3 SHA-256 | half 1 features purity 4 | dc2c411dfa649b04cd9763c1077d49aeec8b52d89fe293d038157c2427702688 | dc2c411dfa649b04cd9763c1077d49aeec8b52d89fe293d038157c2427702688 |  | yes |
| R3 training | chosen C, half 0 | 100.0 | 100.0 |  | yes |
| R3 training | chosen C, half 1 | 10.0 | 10.0 |  | yes |
| rule | candidates clearing the bar | [] | [] |  | yes |

## 5. Files and storage

- Scripts: `rd2_phase2.py` (candidate numbers, 59 s, CPU, 8 threads), `rd2_compare.py` (comparison). Outputs in
  `rederive/out/`: `rd2_phase2.{json,npz,log}`, `rd2_compare.json` (all 480 rows), `rd2_compare_table.md` (the full
  table).
- After the comparison we deleted the bank-feature caches `out/rd2_r3_A0_bankfeat.npz` (113 MB) and
  `out/rd2_r3_A1_bankfeat.npz` (302 MB). Their SHA-256s are recorded in `rd2_r3_A{0,1}.json`, and `rd2_r3.py` rebuilds
  them in about 2 minutes. `rederive/out/` now holds 117 MB (gitignored): the seed-42 cache (18 MB), R3's draws, OOF
  arrays and probabilities (`rd2_r3_A0.npz` 27 MB, `rd2_r3_A1.npz` 56 MB) and smaller files. The A1 files stay for the
  ablation that runs after the rule is applied (`rd2_phase2.py --a1 <reader>`).
