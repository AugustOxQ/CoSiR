# 2026-10-04: quick checks D0, N1, N2 (seed-42 development) and the fresh-seed test of N1-nested-A3

## Problem

After the A′ repair stopped at branch 3 and the 8B MLLM probe did not select the aspect, the user wanted to try more
methods before taking the branch decision. Spec `docs/superpowers/specs/2026-10-04-new-method-quick-checks-design.md`
(approved 2026-10-04) named three cheap CPU checks on the seed-42 development episodes: D0 (can the aspect be read
from the 4+4 example pairs when items are represented by label probes), N1 (a centered agreement rule on the existing
factor codes) and N2 (rank by the condition-free score, rerank its top k by the condition term).

## Steps

1. `DECISION_RULE.md` committed first (7e50f18), before any code: spec §5 with the D0 threshold decided with the user
   (Inferred keeps at least half of Told's condition gain), numeric stop rules and the GO rule for fresh seeds.
2. Plan `docs/superpowers/plans/2026-10-04-new-method-quick-checks.md` (c29f238), executed with subagent-driven
   development: Task 1 scorers (6aa7709), Task 2 decision functions (f6c49f2), Task 3 runner (bc36e72); each task
   reviewed clean (the Task 3 reviewer re-derived the Told mapping, the cascade and all config margins on the smoke run).
3. Real run on CPU: `run_checks.py` at bc36e72, about 3 minutes; every in-script assertion passed, including the exact
   reproduction of A′'s stored nested and control arrays for A3, C0 and SE and E1's cosine arrays.

## Results (seed 42, 12,288 pooled episodes, 4,602 anchor paintings)

Comparators: cosine R@1 12.96 (either 25.92), RCA R@1 13.38, gain 0.10 [−0.08, 0.29].

**D0 (label probes; diagnostic).** Probes fitted on the 60,000-row scorer-train draw minus unlabelled rows
(emotion 53,395 rows, style 60,000, genre 48,779). Selection accuracy image/caption: emotion 38.9/60.7, style
60.9/25.4, genre 81.7/58.3.

| Scorer | R@1 | Gain | Either |
|---|---|---|---|
| Told | 30.66 [30.16, 31.16] | 21.06 [20.48, 21.61] | 40.27 |
| Inferred, hard | 25.02 [24.58, 25.48] | 13.33 [12.78, 13.91] | 36.70 |
| Inferred, soft | 24.97 [24.52, 25.41] | 9.62 [9.12, 10.11] | 40.31 |

Hard-pick accuracy 59.3% (emotion × style 38.2, emotion × genre 77.0, style × genre 62.6); soft fallback 18.2% of
rankings. Decision variant hard, gain 13.33 ≥ 0.5 × 21.06 = 10.53: **close**. Inferred keeps 63% of Told's gain.

**N1 (centered rule).** Term only, A3: R@1 14.08, gain 1.25, either 26.91 (current rule 11.02, 0.99, 21.04; cosine
either 25.92). N1 minus current term-only gain: A3 +0.25 [−0.27, 0.77], C0 +0.64 [0.12, 1.16], SE +0.58 [0.04, 1.11].
N1 stops on C0 and SE (either 25.79 and 25.66 < 25.92), not on A3.
Nested: A3 R@1 16.79 vs control 16.55, **m_R +0.24 [0.09, 0.39], m_g +0.26 [0.05, 0.47]**, both halves picked
(λ_u, λ_a) = (8, 2). C0 and SE nested do not pass. Diagonal KISSME on codes: term-only either 15 to 16, nested gain ≈ 0.
Label-trained diagnostics: L3 nested N1 m_R +0.70, gain 1.59; LT +0.45, 0.65.

**N2 (cascade on A3).** Both p_a and p_b in the control's top k: 4.74% (k = 2), 10.85% (k = 3), 25.84% (k = 5).
Every cascade loses R@1 to the control (agree: −2.29, −3.44, −4.58; N1 reorder: −0.46, −1.23, −1.91) while adding
gain 0.36 to 0.72. No N2 configuration passes.

**Decision (mechanical): row 1, N1-nested-A3.** Against the GO comparators on seed 42 (descriptive): vs cosine R@1
+3.82 [3.49, 4.16], gain +0.26 [0.05, 0.47]; vs RCA R@1 +3.41 [3.05, 3.75], gain +0.16 [−0.13, 0.44].

## Fresh-seed test

`TEST_CONFIG.md` (00f2b32) fixed N1-nested-A3 before the seeds were built; seeds 45, 47, 48 built with E1's runner
(about 200 s each; 15 distinct episode SHA-256s over seeds 42, 43, 45, 47, 48, each file labelled with its own seed);
`test_seeds.py` (dbe517f, reviewed: its real path reproduced the seed-42 arrays bit for bit). Run at 04:11.

| Pooled (36,864 episodes) | R@1 | Gain | Either |
|---|---|---|---|
| N1-nested-A3 | 16.57 [16.33, 16.80] | 0.15 [0.03, 0.27] | 32.98 |
| declared control | 16.50 [16.27, 16.73] | 0 | 33.00 |
| cosine | 13.15 [12.97, 13.33] | 0 | 26.30 |
| RCA | 13.20 [13.02, 13.39] | 0.01 [−0.08, 0.11] | 26.39 |

Config minus control: R@1 +0.07 [−0.04, 0.17], gain +0.15 [0.03, 0.27]; minus RCA: gain +0.14 [−0.02, 0.29].
**NO-GO under §6 as committed** (failed: rca/gain, control/r1). Per seed, config minus control R@1 was +0.10, +0.09,
+0.01 and gain +0.22, +0.08, +0.15: the seed-42 gain (0.26) roughly halved.

## Final review and Addendum 1 (matched control)

The final whole-branch review (Opus) re-derived every seed-42 number bit for bit and found the critical flaw: the
declared control (uncentered T_u) removes both the condition and N1's centering. `ADDENDUM_1.md` (812cab2, committed
at 04:13 before the test outputs written at 04:11 were read) made N1's matched control (same nested family with
T_N1 replaced by the condition-free T_N1u, max-R@1 cross-fit) the governing control, re-applied the table and added
the matched control to the GO comparators. `matched_controls.py` (58778cd, reviewed) results:

| | Matched control R@1 | N1-nested minus matched: R@1 | gain | pass |
|---|---|---|---|---|
| A3, seed 42 | 17.94 [17.57, 18.30] | −1.15 [−1.38, −0.92] | +0.26 [0.05, 0.47] | no |
| C0, seed 42 | 17.47 | −1.24 | +0.07 | no |
| SE, seed 42 | 17.78 | −1.60 | −0.14 | no |
| A3, test seeds pooled | 17.75 [17.52, 17.98] | −1.19 [−1.33, −1.05] | +0.15 [0.03, 0.27] | NO-GO |

Re-applied seed-42 table: **row 2** (no N1 or N2 configuration passes; D0 close). The condition-free centered score
(mostly the centered uniform term, picks λ_u 0, λ_a 8 to 16) is the strongest condition-free score in the project
(17.94 on seed 42 against E3's uncentered control 16.55). It centres the query on the episode's 8 example items, which
are identical under both conditions, so it uses the examples but not the condition.

## N6 (Addendum 2)

`ADDENDUM_2_N6.md` (223e04b) pre-specified N6 before any N6 code; `run_n6.py` (808ab47, reviewed: the reviewer
refitted all six heads by an independent route and reproduced the posteriors to 3e-8 and every per-anchor array).
Dev stage run once (83 s).

Heads, held-out accuracy on 10,000 other scorer-train rows (image / caption head): affect 13.5 / 34.6, image
92.7 / 22.9, caption 21.6 / 89.7 (64 clusters each, chance 1.6).

| Seed 42 | R@1 | Gain | Either |
|---|---|---|---|
| T_6 term only (hard reader) | 16.51 [16.15, 16.87] | **4.41 [3.96, 4.87]** | 28.61 |
| T_6soft term only | 17.34 [16.98, 17.72] | 3.36 [2.97, 3.76] | 31.33 |
| T_6u term only (condition-free) | 16.96 [16.60, 17.30] | 0 | 33.91 |
| N6-nested | 17.30 [16.94, 17.66] | 1.69 [1.35, 2.02] | 32.91 |
| its control z(cos) + σ·z(T_6u) | 17.07 [16.73, 17.42] | 0 | 34.15 |

N6-nested minus control: R@1 +0.23 [−0.01, 0.47], gain +1.69 [1.35, 2.02]: **does not pass** (R@1 lower bound
−0.01). Against cosine +4.34 R@1 and +1.69 gain; against RCA +3.92 and +1.59; against A3's matched control
(descriptive) R@1 −0.63 [−0.96, −0.31]. Picks: (8, 8) on one half, (1, 1) on the other. Per pair, nested gain
emotion × style 0.76, emotion × genre 3.18, style × genre 1.14. The hard reader picks the affect partition in 59 and 68%
of emotion-conditioned rankings (emotion × style, emotion × genre) and the image partition in 54 to 62% of
style-conditioned rankings in emotion × style and genre-conditioned rankings, but under the style condition of
style × genre it picks affect in 58% and image in only 17%: the image partition carries genre more than style (E2
AMI 0.397 against 0.318), so there the contrast pairs (which share genre) agree on image clusters more than the
support pairs do, the style × genre weakness the ARS synthesis predicted. Soft fallback 18.8%.

By `ADDENDUM_2_N6.md` §4, N6 failing at seed 42 ends method work under this spec: no test stage, no new method
overnight. The term-only gain of 4.41 is the largest label-free condition gain measured in the project (N1 on A3 1.25, the agreement rule
0.99; label-trained L3 3.20 with N1's rule, 1.85 with the agreement rule; Qwen3-VL-8B 0.21), so N6's reader does read the condition; under the nested score it
loses either rate (32.91 against 34.15) about as fast as it gains selection.

## Exploratory analysis of N6 (decides nothing) and Addendum 3 (N6c)

`explore_n6.py` (48b30b4), seed 42 only:

- Bootstrap seeds 0 to 99: N6-nested minus control R@1 lower bound > 0 in 0% (min −0.023, median −0.012, max 0.000);
  gain lower bound > 0 in 100% (min 1.35). The N6 fail is a boundary case that never flips.
- N6's hard term on A3's centered uniform base (`crossfit_nested(cos, T_N1u, T_6)`): R@1 18.49 [18.12, 18.88] against
  its control 17.94; +0.55 [0.31, 0.79] R@1 and +1.19 [0.88, 1.50] gain; picks (16, 8) on both halves. With the soft
  term: +0.63 [0.41, 0.86] and +0.84 [0.58, 1.11].
- Per pair (N6-nested minus control): emotion × style R@1 +0.02, gain +0.76; emotion × genre +0.89, +3.18;
  style × genre −0.23, +1.14.

`ADDENDUM_3_N6C.md` (8ff2fc5) pre-specified this combination as N6c before any gate or test: gate on seed 42 against
its nested control C1 and its matched control C2 (z(cos) + λ_u·z(T_N1u) + λ_a·z(T_6u), max-R@1 cross-fit; never
computed before the commit), then GO on seeds 45, 47, 48 pooled against cosine, RCA, C1 and C2. The controller
overrode Addendum 2's "no further method overnight" for this one configuration (ruling in the run ledger).

### N6c gate (seed 42; `run_n6c.py`, bd50874, reviewed)

| | R@1 | Gain | Either |
|---|---|---|---|
| N6c | 18.49 [18.12, 18.88] | 1.19 [0.88, 1.50] | 35.80 |
| C1: z(cos) + σ·z(T_N1u) | 17.94 [17.58, 18.30] | 0 | 35.89 |
| C2: N6c with the condition removed (T_6u for T_6) | 18.34 [17.97, 18.70] | 0 | 36.68 |

N6c minus C1: R@1 +0.55 [0.31, 0.79]; minus C2: R@1 **+0.15 [−0.06, 0.38]**; gain +1.19 [0.88, 1.50] (the same
number against cosine, C1 and C2, whose gains are all 0). The config and C1 numbers equal the exploratory run exactly.
**Gate not passed** (R@1 against C2): most of the exploratory R@1 lift over C1 came from adding the condition-free head
term T_6u, not from reading the condition. N6c gives up 0.88 either rate to gain 1.19 selection (R@1 = (either +
gain) / 2: +0.15). Per pair, against C2: emotion × style +0.16, emotion × genre +0.51 [0.12, 0.91], style × genre
−0.21. By `ADDENDUM_3_N6C.md` §5: no test; method work stops for the night.

## Root cause in one line

Every reader that selects the conditioned aspect (now including N6, term-only gain 4.41) still pays for it in either
rate once fused with the strongest condition-free score; the margin left (+0.15 R@1 for N6c) is below what the pass
rules can confirm. The night's new condition-free scores (centered factor term 17.94; plus the head term 18.34) raised
the bar the conditioned scorers must clear.
