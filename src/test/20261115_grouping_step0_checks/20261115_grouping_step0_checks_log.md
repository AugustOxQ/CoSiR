# 2026-11-15 grouping redesign, step 0: three label-light checks (EXPLORATORY: decides nothing; seed 42 development episodes only)

## Problem

Before redesigning the grouping component (synthesis `src/test/20261114_grouping_research/synthesis.md` §6 to §8), three
cheap checks on the existing groupings were meant to steer the effort. (0a) Can an image tell which affect group a
caption falls in at all, given that a painting's rows share one image but each row's affect group comes from its own
caption? (0b) Does placeability, a label-free score, rank groupings the way their told margins do? (0c) Does
sibling-aware agreement p_imgᵀ S p_txt help the told and reader margins without blurring same-row against random-pair
agreement? Nothing was re-clustered; every grouping, head draw and episode set is one that earlier runs already
produced and checked.

## What was run

`PLAN.md` (SHA-256 `b26d98c5f40e8ff95d79f8031e288f7444a009f6e3e0e992c7e51a2e457e7fac`, written
2026-10-05T17:54:55+02:00, before any number of this folder existed) fixed the measures and the readings.
`run_checks.py` (SHA-256 `663a57342dd4…`, the same version for all three full stages) imports
`run_told_oracle.py` (`fit_one_head`, `evaluate_arm`, `pairs2`/`dpairs`/`mix`, `margin_arrays`) and `run_sweep.py`
(`setup`, which rebuilds B and R0 and asserts the stored numbers) and changes nothing outside this folder. CPU only,
8 threads per process. Each stage was smoke tested first (`--smoke`: 2 sweep cells, 2 permutations, heads on 3,000
rows; outputs in `results/smoke/`), then the three full stages ran in parallel from 2026-10-05T18:07:56+02:00 to
18:16:23+02:00: stage a 70 s, stage b 508 s, stage c 242 s.

- **0a.** On the 183,694 scorer-train rows (36,518 paintings): exact pair counts by unique codes for p_same (unordered
  pairs within a painting in the same group) and p_diff (unordered pairs from different paintings in the same group),
  R = p_same / p_diff, and the leave-one-out painting-majority accuracy A_loo (vectorised; ties among t groups that
  contain the true group count 1/t). Run for R0 (E2 affect k-means 64), L (Leiden default, 41 groups), the 17 sweep
  cells, and the E2 image and caption partitions; control: 20 size-keeping random relabellings (seeds 0 to 19).
- **0b.** Affect heads of the 9 Leiden cells and 8 k-means controls refit with `fit_one_head` (60,000 draw rows,
  10,000 check rows); R0, image and caption from the stored N6 posteriors. On the 32,413 selection rows: P_ami (AMI of
  the argmax image head against the argmax caption head) and P_lift (same-row agreement over the exact mean over ordered
  different-painting pairs). The sweep's told margins were read once from `sweep.json`, to validate the criterion
  (disclosed in PLAN.md).
- **0c.** The told-oracle arm L setting: affect = L with refit heads, image and caption = stored N6 posteriors, B = the
  stored C2. Every caption posterior q is replaced by S q, so the told term, N6's reader (Δ included) and their
  condition-free counterparts all use p_imgᵀ S p_txt. Arms I (identity), F1-all (primary), F1-affect, F3-all and
  F2-all (descriptive). Label-free gate: AUC of same-row agreement (all selection rows) against 200,000 random
  ordered different-painting pairs (seed 0). Margins on B with 95% intervals from 5,000 painting resamples; paired
  per-anchor differences against arm I with the told oracle's helper (`run_checks.point_ci`).

### Checks (all passed)

| Check | Result |
|---|---|
| PLAN.md SHA-256 equals the dispatched version (all stages) | yes |
| E2 partitions SHA-256 and painting alignment (`run_n6.partition_labels`) | yes |
| Every sweep cell's stored partition matches its JSON `partition_sha256`; cells written by `run_sweep.py` `13314f6f17…` | yes |
| Told oracle's `partition_L` equals the sweep's `leiden_k20_r1.0` partition | exact |
| Stored A_head equals PLAN.md (R0 13.47, L 9.81) | yes |
| 0a: vectorised A_loo and pair counts against brute force (first 600 paintings for R0, L and a permuted R0; a synthetic labelling with many ties) | per row identical; p_same, p_diff within rtol 1e−12 |
| 0b: all 17 refits reproduce the stored held-out accuracies, draw SHA-256 and class counts | exact, 17 of 17; 0 convergence warnings |
| 0b: P_lift formula against brute force on 2,000 selection rows (R0) | 1.2173370611506151 both |
| 0c: `run_sweep.setup`: B equals stored C2; R0 told +1.14 [0.90, 1.41] and reader +0.14 [−0.04, 0.32] reproduce `diagnose_counterparts.json`, `diagnose_fixes.json` and `told_oracle.json`; stored L and K arrays reproduce `told_oracle.json` | exact |
| 0c: refit L head equals the told oracle's arm L head (9.81 / 35.72, draw SHA-256) | exact |
| 0c: the check-row variant of `fit_one_head` gives selection posteriors bit identical to `fit_one_head` (affect L) and to the stored N6 posteriors (image, caption); held-out accuracies equal | bit identical |
| 0c: applying S = identity through the transform leaves the posteriors bit identical | yes |
| 0c: gate AUC (sklearn) equals Mann–Whitney U / (n₁n₂) on the affect identity scores | rtol 1e−9 |
| 0c: **arm I reproduces told-oracle arm L**: told, reader and pick blocks, per-anchor arrays, B′, and the stated told +1.64 [1.37, 1.92] and reader +0.35 [0.15, 0.57] | exact |

After the runs, the implementer re-derived A_loo and p_same for R0 and L by a plain Python loop over all 183,692 rows
(7.3309% and 6.3877%, p_same 0.0623 and 0.057934, equal to `step0a.json`). It also re-derived the I and F1-all margins
and the F1-all − I paired differences from `step0c_per_anchor.npz` with its own bootstrap (told +0.02 [−0.26, 0.30],
reader −0.01 [−0.25, 0.25], equal to `step0c.json`).

**Controller review (2026-10-05, evening).** The main session re-derived, with its own code from the stored inputs
and outputs (`rederive_step0.py` in the session scratchpad): 0a p_same, p_diff, R and A_loo for R0 and L with a
Counter-based count (all equal to `step0a.json` to the printed precision); 0b P_ami and P_lift for R0, image and caption
from the stored N6 posteriors (equal); the 0c gate AUC for image and caption under I and F1 with its own 200,000 random
pairs (seed 7: image 0.8973 and 0.8496, caption 0.8710 and 0.8190, against 0.8968 / 0.8487 and 0.8705 / 0.8182 here,
the same drops within sampling noise); the image F1 matrix rebuilt from CLIP features (maximum absolute difference 0.0);
and the F1-all − I paired margins with its own painting bootstrap (told +0.02 [−0.26, +0.30], reader −0.01
[−0.26, +0.26]). The controller also judges the 0a A_loo clause to be a design error of PLAN.md (first caveat below), so
the 0a "No room" reading is reported as written but not used as a decision.

## Results

### 0a. Same-painting ceiling (scorer-train rows; baseline: 20 size-keeping random relabellings, mean [min, max])

| Grouping | Groups | p_same | p_diff | R | A_loo | Majority share | Control R | Control A_loo |
|---|---|---|---|---|---|---|---|---|
| R0 (k-means 64) | 64 | 0.0623 | 0.0404 | **1.541** | **7.33%** | 11.49% | 1.001 [0.987, 1.014] | 4.51% [4.44, 4.60] |
| L (Leiden default) | 41 | 0.0579 | 0.0351 | **1.652** | **6.39%** | 6.29% | 1.000 [0.989, 1.021] | 3.60% [3.52, 3.71] |
| image (reference) | 64 | 1.0000 | 0.0164 | 60.852 | 100.00% | 2.70% | 1.005 [0.968, 1.030] | 1.66% [1.61, 1.71] |
| caption (reference) | 64 | 0.1338 | 0.0192 | 6.955 | 16.50% | 4.05% | 1.002 [0.984, 1.025] | 1.96% [1.91, 2.03] |

A_loo used 183,692 rows (two rows sit alone in their painting). The image partition behaves as it must (every pair
within a painting shares the group), which checks the counting. For both affect groupings, row pairs from one painting
share a group about 1.5 to 1.7 times as often as pairs from different paintings, against 1.00 for the random control:
viewers of one painting do agree on affect beyond chance. The agreement is weak in absolute terms, though. Only 6% of
same-painting pairs share an L group, and predicting a caption's group from the other captions of its painting (about
four of them) reaches 7.33% (R0) and 6.39% (L). That is above the random control (4.51%, 3.60%) but below R0's
majority share (11.49%) and level with L's (6.29%).

Sweep cells against their matched k-means (R | A_loo | majority share):

| Count | Leiden cell | k-means |
|---|---|---|
| 14 | k20_r0.25: 1.350 / 16.17% / 18.71% | 1.339 / 18.12% / 24.03% |
| 15 | k10_r0.25: 1.470 / 14.61% / 12.76%; k40_r0.25: 1.417 / 15.97% / 15.38% | 1.356 / 17.12% / 20.14% |
| 31 | k40_r1.0: 1.619 / 7.48% / 6.30% | 1.473 / 10.43% / 13.99% |
| 41 | k20_r1.0: 1.652 / 6.39% / 6.29% | 1.471 / 9.87% / 14.60% |
| 44 | k10_r1.0: 1.689 / 6.11% / 6.36% | 1.487 / 8.68% / 12.41% |
| 88 | k40_r4.0: 1.944 / 2.86% / 2.65% | 1.597 / 5.55% / 8.93% |
| 95 | k20_r4.0: 1.965 / 2.71% / 2.32% | 1.621 / 5.54% / 8.95% |
| 118 | k10_r4.0: 2.049 / 2.29% / 1.96% | 1.659 / 4.52% / 8.36% |

R is higher for Leiden than for k-means at every matched count and rises with the group count in both families (Leiden
1.350 at 14 groups to 2.049 at 118). A_loo runs the other way, because it rewards a dominant group: k-means has the
larger majority share at every count, and A_loo falls as groups multiply. A_loo therefore tracks group-size balance as
much as same-painting agreement.

### 0b. Placeability (selection rows; baseline: k-means at the matched count)

| Leiden cell (groups) | P_ami Leiden | P_ami k-means | P_lift Leiden | P_lift k-means | Told margin, Leiden − k-means (sweep) |
|---|---|---|---|---|---|
| k10_r0.25 (15) | 0.0432 | 0.0384 | 1.2293 | 1.1707 | +0.71 [0.45, 0.97] |
| k10_r1.0 (44) | 0.0493 | 0.0385 | 1.2861 | 1.2109 | +0.66 [0.40, 0.91] |
| k10_r4.0 (118) | 0.0726 | 0.0418 | 1.3270 | 1.2476 | +0.88 [0.63, 1.14] |
| k20_r0.25 (14) | 0.0384 | 0.0346 | 1.1721 | 1.1571 | +0.66 [0.40, 0.92] |
| k20_r1.0 (41) | 0.0496 | 0.0377 | 1.2715 | 1.2061 | +0.54 [0.31, 0.78] |
| k20_r4.0 (95) | 0.0660 | 0.0431 | 1.3251 | 1.2494 | +0.71 [0.47, 0.96] |
| k40_r0.25 (15) | 0.0430 | 0.0384 | 1.2086 | 1.1707 | +0.64 [0.40, 0.89] |
| k40_r1.0 (31) | 0.0480 | 0.0381 | 1.2663 | 1.2065 | +0.48 [0.24, 0.72] |
| k40_r4.0 (88) | 0.0616 | 0.0425 | 1.3112 | 1.2397 | +0.67 [0.42, 0.92] |

References (stored N6 posteriors, 64 groups): R0 P_ami 0.0400, P_lift 1.2280 (held-out 13.47 / 34.63); image 0.2416,
5.8961; caption 0.1959, 4.3924. Spearman over the 17 cells (descriptive): P_ami against the told margin ρ = +0.503
(p 0.040); P_lift ρ = +0.267 (p 0.300).

Leiden's P_ami exceeded its matched k-means in all nine pairs (Leiden 0.0384 to 0.0726, k-means 0.0346 to 0.0431), and
so did P_lift. The criterion separates the two method families the way the told margins did. It does not order
settings inside a family: P_ami rises with the group count (Leiden 0.0384 at 14 groups to 0.0726 at 118), the
inflation with many fine groups that synthesis §6 warned about, while the sweep's nine Leiden told margins all lie
inside each other's intervals (+1.64 to +1.95). The 17-cell Spearman of +0.50 mostly reflects the family split, as
§6 anticipated. Raw argmax agreement runs the other way (k-means 26.23% to 35.66% of rows, Leiden 11.90% to 26.60%):
concentrated predictions, as k-means' dominant clusters produce, raise chance agreement, which AMI subtracts.

### 0c. Sibling-aware agreement (baseline: arm I, which equals told-oracle arm L)

S per grouping (off-diagonal mean | share of off-diagonal entries above 0 | smallest eigenvalue):

| Grouping | F1 (centred-centroid soft cosine) | F3 (hierarchy, 4 cuts) | F2 (co-membership, check rows) |
|---|---|---|---|
| affect L (41) | 0.1097 / 0.465 / −0.130 | 0.0854 / 0.187 / +0.250 (cuts 41, 21, 11, 6) | 0.7623 / 1.000 / −0.013 |
| image (64) | 0.1173 / 0.427 / −0.228 | 0.0661 / 0.149 / +0.250 (cuts 64, 32, 16, 8) | 0.2069 / 1.000 / +0.029 |
| caption (64) | 0.0832 / 0.398 / −0.139 | 0.0723 / 0.162 / +0.250 (cuts 64, 32, 16, 8) | 0.2445 / 1.000 / −0.065 |

Every F3 cut produced exactly the requested number of clusters.

**Gate** (AUC of same-row against random different-painting agreement; passes if AUC_S ≥ AUC_I − 0.005; ratio of means
in parentheses):

| Grouping | I | F1 | F3 | F2 |
|---|---|---|---|---|
| affect | 0.6853 (1.272) | 0.6145, −0.0707 (1.128) | 0.6246, −0.0606 (1.133) | 0.5898, −0.0955 (1.018) |
| image | 0.8968 (5.911) | 0.8487, −0.0482 (2.345) | 0.8334, −0.0634 (2.790) | 0.8505, −0.0463 (1.775) |
| caption | 0.8705 (4.394) | 0.8182, −0.0523 (2.161) | 0.8039, −0.0666 (2.314) | 0.7725, −0.0980 (1.448) |

The gate failed for every arm, F1-affect included (affect only: −0.0707). Each S lowered the rank separation between
same-row and random pairs by 0.046 to 0.098 for every grouping, far beyond the 0.005 tolerance. Smoothing also credits
random pairs whose groups are siblings, and the ratio of means fell with it (affect 1.272 to 1.128 under F1, image
5.911 to 2.345).

**Margins on B** (R@1 in pp; B = C2, R@1 18.34 [17.97, 18.70]):

| Arm | Told margin | Reader margin | Reader gain | Reader either | Pick accuracy | Told − I (paired) | Reader − I (paired) |
|---|---|---|---|---|---|---|---|
| I | +1.64 [1.37, 1.92] | +0.35 [0.15, 0.57] | +1.34 [1.03, 1.65] | −0.63 [−0.92, −0.35] | 54.7 [54.1, 55.3] | (baseline) | (baseline) |
| **F1-all** | +1.66 [1.41, 1.92] | +0.35 [0.14, 0.56] | +2.25 [1.92, 2.60] | −1.55 [−1.85, −1.26] | 58.9 [58.3, 59.5] | **+0.02 [−0.26, 0.30]** | **−0.01 [−0.25, 0.25]** |
| F1-affect | +1.46 [1.17, 1.76] | +0.21 [0.04, 0.39] | +0.63 [0.38, 0.88] | −0.20 [−0.45, 0.04] | 50.4 [49.8, 51.1] | −0.18 [−0.40, 0.04] | −0.14 [−0.34, 0.07] |
| F3-all | +1.32 [1.05, 1.58] | +0.30 [0.09, 0.51] | +1.95 [1.62, 2.28] | −1.35 [−1.65, −1.06] | 55.9 [55.2, 56.5] | −0.33 [−0.61, −0.03] | −0.05 [−0.31, 0.21] |
| F2-all | +1.32 [1.06, 1.59] | +0.20 [−0.02, 0.42] | +1.58 [1.27, 1.89] | −1.19 [−1.52, −0.86] | 49.2 [48.6, 49.8] | −0.32 [−0.63, −0.02] | −0.16 [−0.41, 0.10] |

For F1-all against I, the reader's paired gain rose +0.91 [0.57, 1.27] and its paired either fell −0.93 [−1.28, −0.56],
so the R@1 margin did not move. The told term's gain and either did not move either (+0.03 [−0.30, 0.38],
+0.01 [−0.44, 0.45]). Per aspect pair, F1-all's told margin fell on emotion × style and rose on the other two (e×s
+1.91 against I's +2.50; e×g +3.17 against +3.03; s×g −0.09 against −0.61); its reader margin rose only on emotion ×
genre (e×s +0.37 against +0.49; e×g +1.04 against +0.67; s×g −0.37 against −0.10). The reader picked the told partition
more often under F1-all (58.9% against 54.7%), but the better picks did not reach R@1. F3-all lowered the told margin
(−0.33 [−0.61, −0.03]) and F1-affect's point fell too (−0.18 [−0.40, 0.04]). F2, whose affect S links every pair of
groups (off-diagonal mean 0.76, all entries above 0), lowered the told margin (−0.32 [−0.63, −0.02]), the reader's
point (−0.16 [−0.41, 0.10]) and pick accuracy (49.2%).

**Controller correction (2026-10-05, evening, before the addendum) to the reading of the paragraph above.** The "either" in the margin
columns is the fused term's either minus its counterpart's, not an absolute loss of aspect-finding. Measured against B
(`step0c.json`, `fusedT_vs_B` and `fusedTcf_vs_B`), F1-all improved the reader on B from +0.41 to +0.89 R@1 (gain
+1.34 to +2.25, either −0.52 to −0.46, i.e. unchanged) and improved its condition-free counterpart on B by the same
amount, from +0.05 to +0.55 R@1 (either +0.11 to +1.09). The told term moved the same way (on B +2.35 to +2.65,
counterpart +0.70 to +0.99). So S made the head agreement a better similarity in both its conditioned and its
condition-free use, by about half a point each, and the margin from reading the condition stayed at +0.35. The better
picks did reach R@1 on B; the counterpart rose with them. Against the development bar's max(B′, counterpart), the
reader's R@1 is 19.23 − 19.06 = +0.17 under F1-all against 18.75 − 18.44 = +0.31 under I. The gate measured
instance-level separation (a row's own image–caption pair against random pairs), which smoothing over sibling groups
lowers by design, so it was not the right test of whether S inflates unrelated pairs; the "Do not adopt" reading
rests on the margin clause, which fails on its own.

Information only, B′ (B rebuilt with the arm's averaged-heads term): I 18.44 [18.07, 18.80] (B′ − B +0.10
[−0.06, 0.25]); F1-all 19.06 [18.67, 19.42] (+0.71 [0.50, 0.94]); F1-affect 18.45 (+0.11 [−0.03, 0.27]); F3-all 18.65
(+0.31 [0.10, 0.52]); F2-all 18.32 (−0.02 [−0.20, 0.16]). The F1 smoothing helps the condition-free use of the heads
more than the conditioned one; every margin above is still on the unchanged B.

## Readings of PLAN.md, applied literally

| Check | Rule | Numbers | Reading |
|---|---|---|---|
| 0a, R0 | Room if A_loo ≥ 1.5 × A_head and R ≥ 1.5; No room if A_loo < 1.2 × A_head or R < 1.2 | A_loo 7.33% vs A_head 13.47% (ratio 0.544); R 1.541 | **No room** |
| 0a, L | same | A_loo 6.39% vs A_head 9.81% (ratio 0.651); R 1.652 | **No room** |
| 0b | adopted if P_ami(Leiden) > P_ami(matched k-means) in at least 8 of 9 pairs | 9 of 9 (P_lift 9 of 9) | **placeability adopted** as the within-source criterion |
| 0c, F1-all | Adopt if the gate passes and one paired lower bound > 0 with the other point ≥ 0; Do not adopt if the gate fails or both lower bounds ≤ 0 | gate fails (AUC −0.0707, −0.0482, −0.0523); told − I lower bound −0.26, reader − I lower bound −0.25 | **Do not adopt** (both clauses) |

Under the same rule, applied descriptively and not as a PLAN reading, F1-affect, F3-all and F2-all also come out
"Do not adopt".

Both 0a readings come from the A_loo clause alone; R by itself (1.541, 1.652) is in the Room range. See the first
caveat.

## Choices not fixed by PLAN.md

- Random relabellings in 0a: `np.random.default_rng(seed).permutation(labels)` over all scorer-train rows, seeds 0 to
  19, the same seeds for every grouping. Majority share over all scorer-train rows (the share over the 183,692 A_loo
  rows is also in the JSON). A_loo ties: t counts every group whose other-row count equals the maximum, the true group
  included.
- A_head read from `n6_seed42.json` (R0) and `told_oracle.json` arm L (L) and asserted equal to PLAN.md's rounded
  values; the reading uses the stored full-precision values.
- 0b: AMI with sklearn defaults (arithmetic normalisation); argmax ties go to the first class (numpy). P_lift in float64.
  "Higher" means strictly greater. The told margin per cell for the Spearman is the cell's own told margin R@1 point
  from `sweep.json`; the Spearman of P_lift is added beside it.
- 0c, S construction: F1 in float64 on scorer-train rows, μ̄ = mean over all scorer-train rows; affect centroids on
  the raw 28 GoEmotions probabilities, the image grouping on unit-normalised CLIP image features, the caption grouping
  on unit-normalised CLIP text features (the modality each E2 partition was clustered on). F1's matrix product was
  symmetrised by averaging with its transpose (rounding only). F2: C symmetrised as (C + Cᵀ)/2, check-row posteriors in
  float64 from `predict_proba`, diagonal set to 1. F3: S = mean over the 4 cuts of the co-membership indicator.
- 0c, application: S q computed in float64 and stored as float32 like every other posterior; arm I passes the
  posteriors untouched (an identity S through the same code was checked bit identical).
- 0c gate pairs: (i, j) drawn uniformly with replacement from the selection rows, same-painting pairs rejected, the
  first 200,000 accepted (batches of 200,000 draws); the same pairs for every grouping and arm. Same-row and pair
  scores use each arm's transformed float32 posteriors, summed in float64.
- 0c reading uses the R@1 paired differences (the margin used throughout this line); gain and either are reported
  beside them. Per-aspect-pair paired R@1 differences against I were added to the JSON.
- B′ needs the A3 condition-free term T_N1u, which `run_sweep.setup` discards; it was recomputed with
  `run_checks.model_inputs` and `centered_term`, and checked through arm I's B′, which equals the told oracle's
  stored B′ for L.

## Caveats

- **0a compares two different predictors.** A_loo predicts a caption's group from about four other captions of its
  painting and ignores everything else; the image head is fit on 60,000 rows and can lean on global group priors.
  Always guessing R0's largest group would score 11.49%, more than R0's A_loo (7.33%), so A_loo is not an upper bound
  on what an image-side head can reach, and A_loo < A_head does not by itself show that image-side affect work has no
  room. PLAN.md's reading was applied as written. R (1.54, 1.65 against 1.00 for random relabellings) shows real but
  weak same-painting agreement. PLAN.md's own caveat also holds: A_head's check rows were drawn by row, so some of their
  paintings are in the fitting draw.
- 0b read the sweep's told margins once, to validate the criterion; no grouping was chosen with them. One Leiden seed
  and one head draw per cell. P_ami grows with the group count, so "adopted" holds for comparisons at a matched count
  within one source, the use PLAN.md names; it does not rank settings of different granularity.
- 0c's margins read evaluation labels through the seed 42 development episodes, which earlier runs have looked at many
  times. S's formulas, γ = 1 and the gate were fixed in PLAN.md before any number. The gate is label free but runs on
  the selection rows the episodes draw from. F1 is not positive semidefinite (smallest eigenvalues −0.130 to −0.228);
  F3 is. F2 is descriptive only and partly relearns placement, as synthesis §7 warned.
- B's cross-fit picks were tuned on the same parity halves that the fusion and counterpart cross-fits reuse, a small
  second-order leak shared by every arm.
- `results/smoke/` holds the smoke runs (3,000-row heads, 2 permutations, 2 cells); their numbers are not results.

## Files

- `PLAN.md`: the fixed design (unchanged).
- `run_checks.py`: stages `a`, `b`, `c` and `--smoke`; asserts PLAN.md's SHA-256 and refuses to overwrite non-smoke
  outputs.
- `results/step0a.json`, `results/step0a.txt`: 0a measures per grouping, per-seed controls, self-test, reading.
- `results/step0b.json`, `results/step0b.txt`: refit head provenance, P_ami and P_lift per cell, references, matched
  pairs, Spearman, reading.
- `results/step0c.json`, `results/step0c.txt`: checks, heads, S statistics, gate, per-arm evaluation (overall and per
  pair), paired differences against I, B′, reading.
- `results/step0c_per_anchor.npz` (1.8 MB): per-anchor fused and counterpart arrays for every arm and metric, B and
  B′ arrays, every S matrix, the gate pairs and `partition_L`, so every margin can be re-derived.
- `results/run_{a,b,c}.log`: run logs. Every output file is gitignored, and nothing over 100 MB was written (the
  whole `results/` folder is 3.8 MB).

## Addendum 0a: share of the painting-level ceiling reached by the image head

Exploratory, decides nothing. Design: `ADDENDUM_0a.md` (SHA-256 `56ba5b5f27e399f99c8e1722168d67d46123b6f20f7d7eae9208bc18cadc110d`,
written 2026-10-05T19:01:16+02:00 before any number), applied as written. Script `run_addendum_0a.py`, CPU only; run
2026-10-05T19:03:09+02:00 to 19:03:42+02:00 (Amsterdam), 34 s. Outputs `results/step0a_addendum.{json,txt}` (provenance with
the addendum, script and input SHA-256s is in the JSON). A smoke run (3,000-row heads, 2 permutations, 50 resamples) is in
`results/smoke/`; its numbers are not results.

**Checks (all passed).** Addendum SHA asserted. R0 and L partitions as in `run_checks.py` stage a. The refit image heads
(local copy of `fit_one_head`'s image branch that keeps the classifier; same 60,000-row draw, `LogisticRegression(C=1,
max_iter=300)`, unit-normalised CLIP features) reproduce the stored held-out accuracies exactly (R0 13.47, L 9.81; 0
convergence warnings), and R0's selection-row image posteriors equal the stored `n6_posteriors.npz` `affect__img` bit for
bit. H and R_u from per-painting sums agree with a brute-force loop on the first 300 unseen paintings (1,471 rows) to
1e-12 relative (R0: H 1.320463 both ways, R_u 1.663724; L: H 1.370574, R_u 1.798920); R_u's pair counts also equal
`run_checks.pair_counts`. Rows of one painting do not always share one image feature vector: 309 of 36,518 scorer-train
paintings have rows whose features differ, by at most 1.05e-5 in any component (float noise; the posteriors are
effectively per-painting).

**Results** (unseen paintings; 95% percentile intervals from 1,000 painting resamples, seed 0, information only; control =
20 relabellings, seeds 0 to 19, same head, mean [min, max]).

| | R0 (64 groups) | L (41 groups) |
|---|---|---|
| Unseen paintings / rows | 5,231 of 36,518 / 25,062 of 183,694 | same rows |
| Image head held-out accuracy | 13.47% | 9.81% |
| R_u (ceiling) | 1.543 [1.483, 1.600] | 1.652 [1.585, 1.722] |
| H (head) | 1.272 [1.259, 1.285] | 1.327 [1.314, 1.342] |
| F = (H - 1) / (R_u - 1) | 0.501 [0.455, 0.558] | 0.502 [0.454, 0.555] |
| Random control H | 0.9992 [0.9946, 1.0078] | 1.0002 [0.9911, 1.0117] |
| Random control R_u | 1.0020 [0.9637, 1.0307] | 0.9953 [0.9711, 1.0553] |
| Reading | Limited room (0.40 < F < 0.75) | Limited room (0.40 < F < 0.75) |

Both groupings sit at about half of the calibrated ceiling, with the interval above 0.40 and below 0.75 for each, so
neither "No room" nor "Room" is reached. R_u (1.54 and 1.65 on unseen rows; 1.54 and 1.65 on all scorer-train rows in
step 0a) bounds even full room, and holds for a calibrated predictor only. Both controls sit at 1, as expected.

**Choices not fixed by the addendum.**
- "Unseen" is computed on painting ids of scorer-train rows (`np.unique` over `groups[scorer_train]`), against the paintings
  of the 60,000 drawn rows; the draw is the one `fit_one_head` makes (`default_rng(rc.PROBE_SEED).choice`).
- The painting set is small (5,231 paintings, 25,062 rows) because the 60,000-row draw touches 31,287 of the 36,518
  paintings. Intervals are wide for that reason.
- The relabellings permute the grouping over all 183,694 scorer-train rows (`default_rng(seed).permutation`), then the unseen
  rows are scored.
- Bootstrap pair counts: resampled copies of a painting are distinct paintings, so a cross-copy pair counts as a
  different-painting pair (H's denominator uses the total group counts including all copies; R_u likewise).
- The resampling intervals are percentile intervals on H, R_u and F computed for each resample.
- The "identical feature" test is exact equality of float32 values against the painting's first row.

**Caveats.** The reading is a threshold on one number from one head draw on one seed's partitions (R0 and L are fixed
earlier partitions; no re-clustering). Intervals resample paintings only, not the head fit. F compares a calibrated
predictor's ceiling with a logistic head that is not calibrated for this purpose, so F near 0.5 does not say how far a
better head would go in practice. The 5,231 unseen paintings are not a random sample of all paintings: they are those that
a 60,000-row row-level draw missed. Their R_u (1.543, 1.652) is close to the all-row values of step 0a (1.541, 1.652),
so no shift is visible.

**Controller review (2026-10-05T19:05+02:00).** The main session re-derived the addendum with its own code
(`rederive_addendum.py` in the session scratchpad): its own draw of the 60,000 rows and logistic fit, H by an explicit
loop over ordered same-painting pairs, R_u by Counter-based pair counts. It found 5,231 unseen paintings and 25,062 rows,
held-out image accuracy 13.47 (R0) and 9.81 (L), and R0: R_u 1.5426, H 1.2716, F 0.5006; L: R_u 1.6521, H 1.3271,
F 0.5016, equal to `step0a_addendum.json`. One more caveat: paintings with fewer rows are more likely to be missed by a
row-level draw, so the unseen set leans toward paintings with few rows.
