# Addendum 3: N6c, N6's reader on the centered factor base (committed before its gate or test is scored)

**Date:** 2026-10-04 (overnight run). **Status of the evidence so far:** exploratory only.

## 1. How N6c came about (disclosed)

N6 failed its pre-registered seed-42 pass by a hair (`ADDENDUM_2_N6.md`; R@1 minus its control +0.23 [−0.01, 0.47];
the lower bound stayed at or below 0 in all of bootstrap seeds 0 to 99). An exploratory analysis on seed 42
(`explore_n6.py`, 48b30b4, which decides nothing) then placed N6's hard-reader term on the strongest condition-free
base measured in the project, A3's centered uniform factor term (`ADDENDUM_1.md`): R@1 18.49 against its nested control
17.94, +0.55 [0.31, 0.79], gain +1.19 [0.88, 1.50]. Because this configuration was found after looking at seed-42
results, its seed-42 numbers are not evidence. `ADDENDUM_2_N6.md` §4 said no further method would be built overnight;
the controller overrides that line for this one configuration, so that the user finds a confirmatory answer instead of
an exploratory hint (ruling recorded in the run ledger). Nothing else is added.

## 2. N6c

- **Score:** the nested score z(cos) + λ_u·z(T_N1u) + λ_a·z(T_6), with A′'s grid and min-margin parity cross-fit
  (`crossfit_nested(cos, T_N1u, T_6, parity)`), rerun on every episode set's own halves.
- **T_N1u:** A3's centered uniform term, `centered_term(A3 inputs, ep, uniform=True)`; A3 checkpoint SHA-256
  dadfef1bed95bbd6… (E3's pick).
- **T_6:** N6's hard-reader term (`ADDENDUM_2_N6.md` §1) from the frozen heads of N6's seed-42 run, whose selection-row
  posteriors are `results/n6_posteriors.npz` (SHA-256 2ad75026e5869c46…); no head is refitted.
- **Control C1:** the nested uniform control returned by `crossfit_nested`, z(cos) + σ·z(T_N1u).
- **Matched control C2** (the lesson of `ADDENDUM_1.md`): N6c with the condition removed, i.e. T_6 replaced by its
  condition-free version T_6u (`uniform_probe_scores` over the three partition heads): the 56-cell family
  z(cos) + λ_u·z(T_N1u) + λ_a·z(T_6u), max-R@1 cross-fit (`crossfit_condition_free(cos, T_N1u, T_6u, parity)`).

## 3. Gate on seed 42 (not evidence, a stop rule)

N6c is tested only if, on the seed-42 episodes, its paired R@1 and condition-gain differences against **both** C1 and
C2 have 95% lower bounds above 0 (painting-clustered bootstrap, 5,000 resamples, seed 42). The comparison against C2
has not been computed before this commit.

## 4. Test

Seeds 45, 47 and 48 (the user's named test seeds; they were used to test N1-nested-A3, and A3's T_N1u-only matched
control was scored on them in `ADDENDUM_1.md` R3, but nothing about N6c was picked on them). **GO** = on the three
seeds pooled (one cluster per painting across seeds), the paired R@1 and gain differences against cosine, RCA, C1 and
C2 all have 95% lower bounds above 0. Each seed is reported on its own. N6c is the second configuration tested on
these seeds (after N1-nested-A3); the report says so.

## 5. What follows

GO or NO-GO, the result goes to the user; nothing further is built overnight. A GO would make N6c the candidate
method to discuss (development of its two components on seed 42, a confirmatory test on three fresh seeds), not a
finished claim: its pieces (label-free partitions shaped like the evaluation aspects, the centered factor term) and
the multiplicity of tonight's candidates are disclosed.
