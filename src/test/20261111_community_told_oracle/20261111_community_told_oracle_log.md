# 2026-11-11 told-partition oracle with community and coarser affect partitions (EXPLORATORY: decides nothing; seed 42 development episodes only)

## Problem

On the current partitions the told-partition oracle (emotion read from the affect partition, style and genre from the
image partition) beat its matched condition-free counterpart by +1.14 [0.90, 1.41] R@1 on B (= C2, R@1 18.34), and
N6's label-free reader by only +0.14 [−0.04, 0.32] (`src/test/20261109_fix_diagnostics/results/diagnose_counterparts.txt`).
The partition profile (`src/test/20261110_partition_profile/`) found that the affect partition (E2's k-means, 64
clusters on the 28 GoEmotions probabilities) carries emotion as groups (pair lift 2.17) but that little of it survives
the cross-modal heads (ratio 1.11). `PLAN.md` (SHA-256 `3b8f90bf…7a9`, written before any number here) asked whether a
coarser affect partition, built as Leiden communities the way the buddy pipeline builds them, raises the told oracle.
Only the affect partition changes between arms; image and caption partitions, their heads and posteriors, the told
mapping and B stay as they are.

## What was run

`run_told_oracle.py` (CPU, 227 s, deterministic; outputs in `results/`, gitignored). It imports the reviewed
diagnostics code and changes nothing outside this folder.

1. **Context and B.** Seed 42 `EvalContext`; B rebuilt with `crossfit_condition_free(cos, T_N1u, T_6u)` from the stored
   N6 posteriors and checked equal to the stored C2 arrays (`matched__*`) exactly.
2. **Arm R0.** Told term (`T6oracle`) and N6's hard reader (`T6`) fused on B with `crossfit_nested(B, B, T)`; matched
   counterpart `T_cf = 0.5(T_a + T_b)` fused with `crossfit_condition_free(B, B, T_cf)`; margin = fused T minus fused
   T_cf, 95% intervals from 5,000 painting resamples. Done twice: from the stored posteriors and from a fresh
   `run_n6.fit_heads` refit of all three partitions.
3. **Arm L.** `detect_communities(affect_probs)` with its defaults (kNN union graph, k = 20, Euclidean, modularity Leiden,
   seed 42) on the 183,694 scorer-train rows; communities under 200 rows merged into the nearest community (centroid
   distance) holding at least 200 rows.
4. **Arm K.** `MiniBatchKMeans` with E2's settings (`random_state 42, n_init 3, batch_size 4096`) and k = 41, L's count
   after merging. The same settings with k = 64 reproduce E2's affect partition exactly in this environment.
5. **L and K measurements.** Affect heads refit with the new labels (same 60,000 draw rows and 10,000 check rows as
   N6); image and caption posteriors are R0's. For each arm: group sizes, the profile's pair statistics for affect ×
   emotion, held-out affect head accuracy, the told and reader terms on B against their counterparts (overall and per
   aspect pair), the reader's pick accuracy under the unchanged told mapping, the paired per-anchor difference of each
   margin against R0's, and (information only) B′, which is B rebuilt with the arm's averaged-heads term.

### Checks (all passed)

| Check | Result |
|---|---|
| B equals the stored C2 per-anchor arrays | exact |
| R0 from stored posteriors: told and reader blocks (fused T − B, fused T_cf − B, margin, picks) equal `diagnose_counterparts.json` | exact |
| R0 pick accuracy equals `diagnose_fixes.json` (52.40 [51.79, 53.03], both correct 28.1%) | exact |
| Fresh `fit_heads` refit equals the stored posteriors (affect, image, caption; both modalities) | bit identical, max abs difference 0.0 |
| Fresh refit held-out accuracies equal `n6_seed42.json` (affect 13.47 / 34.63) | exact |
| One-partition copy of `fit_heads` (`fit_one_head`) equals `run_n6.fit_heads` on R0's affect labels | bit identical |
| Copied pair statistics reproduce `profile.json` for R0 (15 quantities, rtol 1e−9) | yes |
| `affect_prepare.npz` SHA-256 and `affect_probs` SHA-256 match `affect_prepare.json`; `affect_local` equals E2's affect partition | yes |

## Results

Baselines throughout: B = C2, R@1 18.34 [17.97, 18.70]; R0 told margin +1.14 [0.90, 1.41]; R0 reader margin +0.14
[−0.04, 0.32]. All values in pp; intervals are 95% painting-cluster bootstrap intervals (5,000 resamples).

### 1. Partitions

| Arm | Groups | Min | Median | Max | Under 200 | Effective number | Largest share |
|---|---|---|---|---|---|---|---|
| R0 (E2 k-means, 64) | 64 | 244 | 1,690 | 21,098 | 0 | 39.5 | 11.5% |
| L (Leiden, merged) | 41 | 204 | 3,875 | 11,551 | 0 | 32.8 | 6.3% |
| K (k-means, 41) | 41 | 706 | 2,858 | 26,822 | 0 | 26.1 | 14.6% |

Leiden found 42 communities in 34 s (kNN plus Leiden). One of them (172 rows) was under 200 rows and was merged into
community 11 (centroid distance 0.64 in the 28-d probability space), which left 41. K at k = 41 needed no merging
(smallest cluster 706 rows). Leiden's groups are the most balanced (largest 6.3% of rows); K concentrates more rows
in its largest cluster than R0 does.

### 2. Affect × emotion pair statistics

Groups: scorer-train rows, different-painting pairs. Lift = P(same group | same emotion) / P(same group | different
emotion). Contrast ratio = P(same group | same A, different B) / P(same group | same B, different A). Heads: dot product
of the image-head posterior of row i and the caption-head posterior of row j on selection rows, same over different.

| Arm | Lift on groups | e×s ratio (groups) | e×g ratio (groups) | Ratio through heads | e×s ratio (heads) | e×g ratio (heads) |
|---|---|---|---|---|---|---|
| R0 | 2.174 | 2.038 | 2.019 | 1.109 | 1.042 | 1.028 |
| L | 2.711 | 2.546 | 2.508 | 1.145 | 1.071 | 1.038 |
| K | 2.044 | 1.925 | 1.917 | 1.120 | 1.051 | 1.041 |

Leiden's groups carry more emotion than either k-means partition (lift 2.71 against 2.17 and 2.04), so the gain is not
coarseness alone: K has the same number of groups and a lower lift than R0. Through the heads all three arms keep only
a small part of it (ratios 1.11 to 1.15), with L highest.

### 3. Held-out affect head accuracy

| Arm | Image head | Caption head | Majority class share | 1/k |
|---|---|---|---|---|
| R0 | 13.5 | 34.6 | 10.9 | 1.56 |
| L | 9.8 | 35.7 | 6.4 | 2.44 |
| K | 16.0 | 41.4 | 14.0 | 2.44 |

Raw accuracy is not comparable across class counts and balances, so the majority share is given beside it. The image
head stays close to the majority rate in every arm (R0 1.24 times it, L 1.53 times, K 1.14 times). K's higher raw
accuracy comes mostly from its larger dominant cluster.

### 4. Told term on B (emotion to affect, style and genre to image)

| Arm | Fused T − B | Fused T_cf − B | Margin R@1 | Margin gain | Margin either | Paired margin − R0's (R@1) |
|---|---|---|---|---|---|---|
| R0 | +1.50 [1.22, 1.79] | +0.35 [0.16, 0.55] | **+1.14 [0.90, 1.41]** | +4.56 | −2.27 | (baseline) |
| L | +2.35 [2.03, 2.68] | +0.70 [0.48, 0.93] | **+1.64 [1.37, 1.92]** | +6.14 | −2.86 | **+0.50 [0.25, 0.74]** |
| K | +1.46 [1.16, 1.78] | +0.36 [0.17, 0.56] | **+1.10 [0.81, 1.40]** | +5.83 | −3.63 | **−0.04 [−0.22, 0.13]** |

Per aspect pair, margin R@1:

| Arm | emotion × style | emotion × genre | style × genre |
|---|---|---|---|
| R0 | +1.14 [0.69, 1.59] | +2.62 [2.11, 3.12] | −0.33 [−0.65, −0.02] |
| L | +2.50 [1.97, 3.03] | +3.03 [2.48, 3.58] | −0.61 [−0.89, −0.35] |
| K | +1.48 [0.96, 1.99] | +2.48 [1.90, 3.05] | −0.65 [−1.02, −0.30] |

L's told margin rose by half a point over R0, and the rise held against a counterpart that was itself stronger (fused
T_cf − B +0.70 against +0.35). It came mostly from emotion × style (+2.50 against +1.14). On style × genre the told
term is the image dot under both conditions, so it equals its counterpart there. The negative margins on that pair
(−0.33, −0.61, −0.65) are only the cost of fusion weights picked on the pooled pairs. K raised the condition gain
(+5.83) but lost more either (−3.63) and left the R@1 margin where R0 had it.

### 5. N6 reader term on B

| Arm | Margin R@1 | Gain | Either | Pick accuracy | Both correct | Paired reader margin − R0's |
|---|---|---|---|---|---|---|
| R0 | +0.14 [−0.04, 0.32] | +0.79 | −0.51 | 52.4 [51.8, 53.0] | 28.1% | (baseline) |
| L | +0.35 [0.15, 0.57] | +1.34 | −0.63 | 54.7 [54.1, 55.3] | 29.8% | +0.21 [0.01, 0.41] |
| K | +0.16 [0.00, 0.33] | +0.63 | −0.31 | 51.3 [50.7, 51.9] | 27.6% | +0.02 [−0.09, 0.14] |

Per pair, the reader margin for L was +0.49 (e×s), +0.67 (e×g) and −0.10 (s×g), against +0.19, +0.31 and −0.08 for
R0. L's reader picked the affect partition more often under emotion (e×g condition a: 72.5% against 67.9%) and the
image partition more often under genre (64.8% against 61.9%). Both of L's affect heads have 41 classes, and so do K's,
so the different class count cannot by itself explain L's higher pick accuracy: K lowered it to 51.3.

### 6. Information only: B′

B rebuilt with the arm's averaged-heads term: L 18.44 [18.07, 18.80] (B′ − B +0.10 [−0.06, 0.25]); K 18.40 [18.04,
18.76] (+0.06 [−0.05, 0.17]). Every margin above is on the unchanged B.

## Reading of PLAN.md, applied literally

| Arm | Told margin ≥ +2.0 | Paired vs R0 lower bound > 0 | Reader margin moves toward +0.5 | Reading |
|---|---|---|---|---|
| L | no (+1.64) | yes (+0.25) | yes (+0.35 against R0's +0.14) | **promising** (second clause) |
| K | no (+1.10) | no (−0.22) | barely (+0.16) | **not better** |

L against K: neither told margin lies inside the other's interval (L +1.64 [1.37, 1.92], K +1.10 [0.81, 1.40]), so by
PLAN.md's rule the change does not come from granularity alone. The paired L − K told margin is +0.54 [0.31, 0.78].
With the number of groups held at 41, Leiden's communities raised the told oracle and k-means did not.

What this does and does not say: L meets the "promising" bar through its second clause, not through the +2.0 bar. Its
told margin (+1.64) stays below that bar, and the reader margin it allowed (+0.35) stays below the +0.5 that PLAN.md
names. As PLAN.md says, a promising arm still needs a decision rule and the fresh-seed test before it counts.

## Caveats

- Exploratory: it decides nothing about the GO, and the seed 42 development episodes have been looked at many times
  before (this is a reused development seed).
- The told mapping (emotion to affect, style and genre to image) was chosen with labels, so the told term is an
  oracle, not a method.
- B's cross-fit picks were tuned on the same parity halves that the fusion and counterpart cross-fits reuse, a small
  second-order leak shared by every arm.
- Only the affect heads were refit; L and K are compared with R0 on otherwise identical posteriors, and the paired
  differences use R0's stored arrays, which the fresh refit reproduced bit for bit.
- The pair statistics read evaluation labels for description only; no partition setting was chosen with them.

## Choices not fixed by PLAN.md or the brief

- Merge rule for L: one pass from the unmerged centroids (mean of the 28 probabilities per community, float64), with
  Euclidean distance; only communities of at least 200 rows can receive; labels re-compacted afterwards. Only one
  community was affected.
- K was not merged (E2 did not merge); its smallest cluster has 706 rows, so this did not matter.
- "Reader margin moves toward +0.5" was read as the arm's reader margin point exceeding R0's +0.14. L also passes a
  stricter reading (paired reader difference lower bound +0.01 > 0); K passes only the point reading.
- "Within each other's intervals" was read as each arm's told margin point lying inside the other's 95% interval; the
  paired L − K difference is reported in addition.
- `run_n6.fit_heads` loops over E2's three partitions read from E2's file, so a one-partition copy (`fit_one_head`)
  was written and checked bit identical to the original on R0's labels. No code on the path assumes 64 classes;
  only `profile_partitions.py`'s checks 1 and 2 (a 64-row table) and its printed 1/64 baseline do, and those are not
  used here. `profile_partitions.py` runs on import, so its pair-statistics helpers were copied (checks 3 and 3b only)
  and checked against `profile.json` for R0.
- Added beside the PLAN.md measurements: majority-class share and 1/k beside head accuracy, per-pair fused T − B,
  and the picked-partition distribution.

## Files

- `run_told_oracle.py`: the script (refuses to overwrite its outputs).
- `results/told_oracle.json`, `results/told_oracle.txt`: full numbers and summary.
- `results/per_anchor_told_oracle.npz` (1 MB): per-anchor arrays for B and each arm's fused and counterpart scores,
  plus the R0, raw L, merged L and K partitions, for re-deriving every margin.
- `results/run.log`: the run log. Nothing over 100 MB was written.
