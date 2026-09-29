# CoSiR v2 Candidate A: exact CLIP-only ranking baseline

The new condition weight is `w(c) = zeros(32)` for every episode. The factor term in
`s(I,T|c) = β·cos(CLIP_I(I), CLIP_T(T)) + Σ_l w_l(c)·a_I,l(I)·a_T,l(T)` is exactly zero
for every candidate in both retrieval directions. This closes the gap left by the
`ones(32)/32` uniform control, whose constant factor term is small but nonzero.

## Reproduction and method

Run `/root/miniconda3/envs/CoSiR/bin/python src/test/20261006_condition_clip_only_baseline/run_clip_only_baseline.py`
from the repository root. The script imports Task 7's feature loader, item-disjoint
split, train-only factor preparation, role validation, swap-pair selection, and
`score_pool` unchanged. It copies Task 7's inline mining and split-local-to-global
episode remap in the same order, with the same seed-42 config and single-threaded
BLAS during mining. It does not train the condition head, which is irrelevant to a
zero-vector weight. Only the new row is scored; the five other rows below are
transcribed from [Task 7's report](2026-10-05_cosir_v2_candidate_a_condition_ranking_evaluation.md).

The split is 246,978 train items and 61,745 unseen held-out items out of 308,723,
with 4,096 train and 1,024 held-out mined episodes. Each held-out episode has one
positive in candidate slot zero and 12 distractors. The task's original
strict-greater rank rule is used: rank is one plus the number of candidates
scoring strictly above the positive. At `β=0`, all 13 candidate scores are exactly
zero, so this rule would label every positive rank one; this is an optimistic
tie artifact, and CLIP-only Recall is reported as **N/A (all tied)** instead.
The five transcribed `β=0` rows retain Task 7's tie rule; their individual
episodes were not audited for ties in this rerun.

The rerun reproduced Task 7's 2,463,679 train-only graph edges, 21 Stage 1
communities, target-gap rank bins **245 / 122 / 89 / 414 / 154** for ranks
1 / 2 / 3 / 4–10 / 11–32, and **61** swap pairs with **30** distinct B episodes.
Task 7 did not publish episode IDs or saved factor codes, so direct item-by-item
comparison with that run is unavailable; using its unchanged seeded code path
and matching these independent diagnostics is the reproduction check. For
future checks, this run's SHA-256 over the ordered held-out episode records is
`cb50ab7f026678d317dddde274d779674ec5e039921612c999cd19751b74410f`.
Episodes 0, 1, and 2 have `(anchor, positive, target factor)` of
`(241678, 70328, 2)`, `(173548, 91950, 4)`, and `(74915, 137378, 28)`.

## Six-row comparison

All figures are Recall@1 / Recall@3; each percentage has denominator 1,024.

| β | Condition weights | i2t R@1 / R@3 | t2i R@1 / R@3 |
|---:|---|---:|---:|
| 0 | Trained | 49.51% / 75.49% | 42.58% / 66.11% |
| 0 | Naive | 52.83% / 77.15% | 46.78% / 68.75% |
| 0 | Uniform | 49.32% / 75.29% | 42.38% / 66.02% |
| 0 | Shuffled-condition | 49.41% / 75.29% | 42.29% / 65.92% |
| 0 | Oracle | 42.87% / 73.63% | 38.18% / 64.65% |
| 0 | **CLIP-only** | **N/A (all tied)** | **N/A (all tied)** |
| 0.03 | Trained | 49.51% / 75.49% | 42.58% / 66.11% |
| 0.03 | Naive | 53.03% / 77.93% | 47.07% / 69.04% |
| 0.03 | Uniform | 52.93% / 78.81% | 45.31% / 68.85% |
| 0.03 | Shuffled-condition | 49.41% / 75.29% | 42.29% / 65.92% |
| 0.03 | Oracle | 46.78% / 76.46% | 39.45% / 68.46% |
| 0.03 | **CLIP-only** | **13.67% / 36.72%** | **21.68% / 45.90%** |
| 0.3 | Trained | 49.51% / 75.59% | 42.58% / 66.11% |
| 0.3 | Naive | 57.62% / 82.62% | 48.73% / 73.34% |
| 0.3 | Uniform | 40.23% / 71.09% | 39.06% / 64.75% |
| 0.3 | Shuffled-condition | 49.51% / 75.29% | 42.38% / 65.92% |
| 0.3 | Oracle | 54.69% / 81.93% | 48.44% / 73.34% |
| 0.3 | **CLIP-only** | **13.67% / 36.72%** | **21.68% / 45.90%** |

At either positive `β`, CLIP-only succeeds on **140/1,024** i2t Recall@1,
**376/1,024** i2t Recall@3, **222/1,024** t2i Recall@1, and **470/1,024**
t2i Recall@3. The per-episode ranks are identical at `β=0.03` and `0.3`:
positive rescaling of the same CLIP cosines cannot change their order. The
measured mean absolute factor contribution is exactly **0** in both directions
at every `β`.

## Condition swap and interpretation

The CLIP-only score is identical under either condition in each same-anchor
swap pair. Appropriate positive-order reversal was **0/61 (0%)** in both
directions at `β=0`, `0.03`, and `0.3`; any full-pool rank-order change was
also **0/61** in every case. The `β=0` scores are degenerate. At positive
`β`, a nonzero reversal would indicate a bug in pair scoring or indexing.

The percentage-point gaps below compare Task 7's published rows with this
rerun. They assume the seeded episode regeneration reproduced the same pools;
the matching diagnostics support that assumption, but no saved Task 7 episode
IDs exist to verify it item by item. They are therefore cross-run differences,
not newly measured paired-episode effects.

At Task 7's scale-matched `β=0.3`, naive minus CLIP-only is **+43.95/+45.90
percentage points** in i2t Recall@1/@3 and **+27.05/+27.44 points** in t2i.
Naive minus uniform was only **+17.38/+11.52** and **+9.67/+8.59** points,
respectively. Thus **the naive lift more than survives** the true baseline:
it is substantially **larger** than the naive-over-uniform lift in every
metric. Uniform itself exceeds CLIP-only by **+26.56/+34.38** i2t points and
**+17.38/+18.85** t2i points. Its mean factor term may be small compared
with its mean CLIP term, but it materially changes rankings in these mined
candidate pools. Thus the plan's inference from average score magnitudes
that uniform is effectively CLIP-only does not hold for ranking. The larger
naive-over-CLIP gap measures the value of factor-based scoring on pools mined
from those same factors; **naive-over-uniform remains the tighter comparison
for condition-specific benefit**. At `β=0.03`, naive and uniform are close in Task 7's table,
yet naive remains far above CLIP-only; the same qualitative comparison holds.

The trained head also has a **larger** margin over CLIP-only than over uniform
at `β=0.3`: **+35.84/+38.87** i2t and **+20.90/+20.21** t2i points over
CLIP-only, versus **+9.28/+4.49** and **+3.52/+1.37** over uniform.
This does not rescue its condition sensitivity: Task 7 found it nearly
indistinguishable from a shuffled condition and rarely reversing the
condition-specific positives. All comparisons remain limited to the same
factor-defined, automatically mined candidates rather than human relevance.
