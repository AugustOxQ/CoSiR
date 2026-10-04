# Addendum 1 to DECISION_RULE.md: N1's matched condition-free control (committed before the test results are read)

**Date:** 2026-10-04, written by the controller during the overnight run the user asked to be fully automated.
**State when committed:** the seed-42 checks ran at bc36e72 (decision row 1, N1-nested-A3). The fresh-seed test of
`TEST_CONFIG.md` ran at 04:11 under §6 as committed, before the finding below arrived; its outputs
(`results/test_seeds.*`) exist and have **not been read**. Nothing in `DECISION_RULE.md` or `TEST_CONFIG.md` is edited;
this file adds to them, and where they conflict this file governs.

## 1. The finding (final whole-branch review, numbers re-derived from the stored seed-42 arrays)

N1-nested-A3 is z(cos) + λ_u·z(T_u) + λ_a·z(T_N1), with T_u A3's **uncentered** uniform term and T_N1 the **centered**
agreement term. Its declared control, z(cos) + σ·z(T_u), removes two things at once: the condition weights and N1's
centering of the query on the episode's 8 example items. The spec's own definition of a condition-free control is
"same factors, condition removed" (spec §2), and the ARS synthesis (§4.5, item 4) named N1's own control as "its
centered score with uniform weights". `DECISION_RULE.md` §3 left that control out. With the condition-removed term
T_N1u = `centered_term(..., uniform=True)` (term only: R@1 17.95, the strongest condition-free scorer measured in the
project), the reviewer found on seed 42:

- the same cell (8, 2) with T_N1 replaced by T_N1u: N1-nested-A3 minus it, R@1 −0.07 [−0.22, 0.07], either
  −0.41 [−0.62, −0.20], gain +0.26: the condition costs more either rate than it adds gain, the A′ pattern again;
- z(cos) + λ_u·z(T_N1u) + λ_a·z(T_N1) against z(cos) + σ·z(T_N1u), A′'s cross-fit: R@1 −0.03 [−0.07, 0.00],
  gain −0.00 [−0.05, 0.05];
- the best condition-free combination of cos, T_u and T_N1u reaches R@1 17.94; N1-nested-A3 (16.79) is 1.15
  [0.92, 1.38] below it.

So the seed-42 pass came from centering, which needs no condition, and not from reading the condition.

## 2. Rulings

**R1. N1's own condition-free control** (replaces the control of the N1 rows of `DECISION_RULE.md` §3): for a
checkpoint X, the *matched control* is the 56-cell nested family z(cos) + λ_u·z(T_u) + λ_a·z(T_N1u) with X's T_u and
T_N1u, λ_u ∈ {0, 0.5, 1, 2, 4, 8, 16}, λ_a ∈ {0, 0.25, 0.5, 1, 2, 4, 8, 16}, cross-fitted on the episode-index parity
halves: on each tuning half the cell with the highest R@1 is picked (ties to the first cell in row-major order, λ_u
outer), and each half's pick scores the other half. It holds the configuration's own terms and weight budget with only
the condition removed; its gain is 0 by construction. The N2 rows keep their control (the unreordered ranking).

**R2. Seed-42 table re-applied.** `DECISION_RULE.md` §4 is applied again on the seed-42 arrays with R1's control for
N1-nested-{A3, C0, SE} (paired R@1 and gain against it, lower bounds above 0, same bootstrap), computed by a reviewed
script before any further step. The D0 reading is unchanged.

**R3. The fresh-seed test.** The matched control of A3 (R1, cross-fitted on each test seed's own halves) is added as a
fourth comparator: GO requires the pooled paired R@1 and gain differences against cosine, RCA, the declared control
and the matched control all to have 95% lower bounds above 0. The verdict under §6 as committed is reported as well,
but this addendum's verdict governs.

**R4. What follows** (not written down before; `DECISION_RULE.md` §4 and §6 named no step after a failed test):

| R2 outcome | R3 outcome | Next step |
|---|---|---|
| some N1 configuration passes against the matched control | GO | report; N1-nested-A3 stands |
| any | NO-GO, or R2 finds no passing configuration | the re-applied table governs: with no N1 or N2 configuration passing and D0 "close", **row 2: build N6** |
| (D0 "far", not the case on seed 42) | | row 3 |

**R5. N6** gets its own pre-specification, committed before N6 is scored on seed 42: the same checks (term alone and
nested against cosine, RCA and its own matched condition-free control, built as in R1 with N6's uniform head weights),
the same pass rule on seed 42, and, if it passes, a test on seeds 45, 47 and 48 (the user's named test seeds; they were
not used to develop or pick anything for N6) with the GO rule of R3.

## 3. Disclosures carried into the report (from the same review)

- On seed 42, N1-nested-A3's gain over RCA was +0.16 [−0.13, 0.44]: it did not clear the §6 bar on its own
  development seed; its R@1 margins over cosine (+3.82) and RCA (+3.41) come from the condition-free control.
- The gain of 0.26 is about 1.2% of Told's gain (21.06) and sits mostly in emotion × genre.
- N1's stop rule (a) passed on a point estimate only (+0.25 [−0.27, 0.78]): N1 is not shown to read the condition
  better than the current rule on A3.
- D0 "close" is pooled; per pair, Inferred keeps 31% of Told's gain on emotion × style (hard-pick accuracy 38%, chance
  33%), 74% on emotion × genre and 67% on style × genre.
- The both-in-top-k share does not cap N2's gain (half of N2-3-agree's gain comes from rankings without both aspect
  candidates in the top 3); spec §4.3 and `DECISION_RULE.md` §5 say otherwise and are wrong on this point.
- Probe sizes: emotion 53,395, style 60,000, genre 48,779 rows; "anchor-parity halves" are episode-index parity halves.
