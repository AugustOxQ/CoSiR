# Addendum to PLAN.md, check 0a: how much of the painting-level ceiling does the image head reach?

Written 2026-10-05, before any number of this addendum exists. Exploratory; decides nothing about the GO. PLAN.md and
its results are unchanged. Reason: PLAN.md's 0a reading compared the image head's accuracy with a leave-one-out
painting-majority accuracy (A_loo). A_loo ignores how common each group is (it scores below always guessing R0's largest
group, 7.33% against 11.49%), so it is not an upper bound and the "No room" reading is not usable. This addendum asks the
same question with two quantities on one scale.

## Quantities

Groupings: R0 (E2 affect k-means 64) and L (Leiden default, 41 groups), as in PLAN.md.

**Rows.** Scorer-train rows of *unseen paintings*: paintings none of whose rows is in the 60,000-row fitting draw of
`run_told_oracle.fit_one_head` (so the head never saw the painting's image). Their number of paintings and rows is
reported. All pairs below use only these rows.

**Ceiling R_u** (a calibrated predictor that knows each painting's true distribution of affect groups, estimated from
the painting's other rows): p_same / p_diff on the unseen rows, as PLAN.md defines them (unordered pairs within one
painting against pairs from different paintings, exact counts). If a predictor's posterior equals the painting's
distribution, its agreement ratio below equals R_u.

**Head H** (the actual image head): the image head's posterior p_i (refit with `fit_one_head`, image modality, applied to
the unseen rows' unit-normalised CLIP image features) scored against the affect group g_j of another row:

H = [mean over ordered pairs (i, j), i ≠ j, from the same painting, of p_i(g_j)] /
    [mean over ordered pairs (i, j) from different paintings, of p_i(g_j)],

computed exactly from per-painting sums (Σ_i p_i and group counts per painting). H = 1 means the image tells nothing about
the painting's affect groups; H = R_u means it reaches the calibrated ceiling.

**Fraction F = (H − 1) / (R_u − 1)**, the share of the ceiling the head reaches.

**Controls and checks.** The refit image heads must reproduce the stored held-out accuracies (R0 13.47, L 9.81) and, for
R0, the stored N6 selection posteriors bit for bit. Rows of one painting are expected to share one image feature; whether
they do is reported. Random control: 20 size-keeping relabellings of the grouping (seeds 0 to 19) with the same fitted
head, scored the same way (H and R_u should be near 1). Intervals: 1,000 painting resamples (seed 0), resampled copies
counted as distinct paintings; information only.

## Reading (applied to R0 and to L separately)

- **No room**: F ≥ 0.75. The image head already reaches most of what painting-level knowledge allows; image-side head
  work for affect (synthesis §8 step 3, arm b) is skipped.
- **Room**: F ≤ 0.40. Arm (b) for affect is worthwhile.
- **Limited room**: otherwise.

R_u itself is reported beside the reading: even full room is bounded by R_u (about 1.5 to 1.7 on all scorer-train rows,
PLAN.md 0a), and R_u holds for a calibrated predictor (a sharper predictor changes both the numerator and the
denominator).

## Execution

`run_addendum_0a.py` in this folder, CPU only (`CUDA_VISIBLE_DEVICES=`, `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`),
asserts this file's SHA-256, refuses to overwrite; outputs `results/step0a_addendum.json` and `.txt`; results appended to
`20261115_grouping_step0_checks_log.md` under a new heading.
