# Reader fix, round 2: follow-ups of the confidence-gated reader (design)

**Date:** 2026-10-06 (Amsterdam). **Status:** design approved section by section in chat by the user; this file is the
written spec. The binding rule will be `src/test/20261118_reader_fix_round2/DECISION_RULE.md`, written from this spec,
checked by a fresh Opus reviewer and committed before any code. Where the rule and this spec differ, the rule governs.

## 1. Why this round

Round 1 (rule `src/test/20261117_reader_fix_csd/DECISION_RULE.md`, report
`docs/reports/auto/v2/2026-11-18_reader_fix_csd.md`) tested seven candidates on episode seed 42. None cleared the
development bar (bar margin ≥ +0.5 R@1 points, lower bound above 0, gain lower bound above 0). The best was the
confidence-gated reader on the learned reader's weighted (expected) term, on the configuration without the style
grouping (A0): bar margin +0.444 [+0.216, +0.674] against its matched counterpart. Two weak points remained:

- **Picks.** The learned reader under the gate picks the told grouping in 51.3% of conditions on A0 (79% on its own
  practice episodes). The told ceiling on A0 is +1.64 over the matched counterpart, against the reader's +0.444.
- **Promoted low-ranked candidates.** R@1 = (either + gain) / 2. Demoting the other aspect's candidate below a
  negative is R@1-neutral; what costs R@1 is failing to put the right candidate first, or promoting a candidate that
  the condition-free scorer B ranked low over the right one.

User decisions (2026-10-06): test both improvement families with the gate built into every candidate; develop
without the style grouping and keep it as a descriptive ablation; a fresh Opus check of the rule instead of an ARS
round; design L later; results as early as possible (today if the runs allow).

## 2. Candidates

Configuration A0 (affect Leiden 41 groups, image k-means 64, caption k-means 64), the standard heads of round 1,
fused on B exactly as in round 1. Every candidate uses the **weighted** scoring T^c = Σ_h P^c(h)·s_h (top-pick
scoring dropped: on A0 it gave +0.144 against the weighted +0.313).

Three readers, differing only in P^c(h):

1. **Current learned reader (R1).** Round 1's A0 half-readers (`results/rb_reader_A0.pkl` of round 1), frozen,
   probabilities averaged over the two halves.
2. **Adapted learned reader (R2).** The same trained half-readers, adapted to real episodes without labels:
   (a) each half-reader's input standardisation is replaced by the mean and standard deviation of the seed-42
   features (both conditions); (b) the averaged probabilities get an EM class-prior correction (Saerens, Latinne and
   Decaestecker, 2002) estimated on the unlabelled seed-42 (episode, condition) values, starting from the bank's
   uniform prior. Both the statistics and the prior are frozen from seed 42 for the test.
3. **Realistic-practice learned reader (R3).** Retrained with round 1's recipe on round 1's A0 banks made impure:
   in every bank episode only k of the 4 support pairs and k of the 4 contrast pairs keep their shared group; the
   other 4 − k are replaced by random image-caption pairs from the same half (seeded). k ∈ {1, 2, 3, 4} is chosen
   without labels: the k whose bank features best match the seed-42 features (smallest mean absolute standardised
   mean difference over the reader's inputs). k = 4 is round 1's reader.

**One fusion family for all three**, containing round 1's gated fusion:

| Part | Values |
|---|---|
| Weights (λ_u, λ_a) | the 56 cells of round 1 |
| Confidence gate threshold τ | 0th, 25th, 50th, 75th percentile of the reader's seed-42 top-two margin (0th = no gate) |
| Top-k restriction (new) | k_top ∈ {13 (none), 5, 3, 2}: first place must come from B's top k_top candidates; below it, B's order |

896 cells, ordered k_top (13, 5, 3, 2), then τ, then λ_u, then λ_a, ties to the first (so no restriction and no gate
win ties). Min-margin cross-fit on the parity halves as in round 1. Matched counterpart: the same family with the
gated term replaced by its two-condition mean G_cf, max-R@1 cross-fit; the top-k set depends only on B, so the
restriction is condition-free.

**Regression check.** R1 restricted to the k_top = 13 cells must reproduce round 1's confidence-gated reader
(`cand_Rc_Rb_expected_A0`) exactly, fused and counterpart arrays, bar margin +0.4435. Otherwise the run stops.
R1's full result minus +0.444 is then the effect of top-k alone; R2 and R3 against R1 are the effects of the reader
fixes.

## 3. Decision rule (inherited from round 1 unless marked)

- Comparators: B, B′ (B rebuilt on A0's groupings, 18.437 on seed 42), the matched counterpart; the bar comparator is
  the strongest of the three.
- Development bar: unchanged (+0.5, lower bound above 0, gain lower bound above 0). Kept although heuristic.
- Carry: the largest bar margin among candidates that clear; ties within 0.05 go to R1, then R2, then R3 (new order).
- No candidate clears: no test; the choice goes to the user.
- Frozen from seed 42 (new items): each candidate's τ values, R2's adaptation statistics and prior, R3's k.
- Fresh-seed test: unchanged (seeds 49, 50, 51 built once, hash check, GO = pooled lower bounds above 0 for R@1 against
  cosine, RCA, B, B′ and the matched counterpart, the gain statistic, and the gain against RCA; sensitivity projection
  before the seeds; only GO quantities before the verdict; the NO-GO reading; claim licensed: new episodes on the same
  6,451 paintings). Per-seed cross-fitting now chooses among 896 cells; the counterpart has the same freedom.
- Diagnostics, deciding nothing: pick accuracy, chosen k_top and τ, gate-open shares, the shift report for R2 and R3,
  the k chosen for R3; the carried candidate on A1 (with the style grouping) on seed 42 as an ablation.
- Disclosure (new): these candidates were designed after seeing seed-42 results of seven earlier candidates; selection
  inflation is larger than in round 1. The fresh-seed test is the protection; a descriptive line with seed-42 cells
  frozen accompanies it.

## 4. Process and timeline

Rule written from this spec, fresh Opus check, fixes, commit before code, sent to the user. Implementation by two
subagents in parallel (fusion stream: top-k and the 896-cell cross-fit with its counterpart, extending round 1's
`rc_core.py` by import; reader stream: R2 and R3), reusing round 1's verified code by import without modifying it.
The main session launches the runs (CPU only); an independent agent re-derives every decision number before the rule
is applied; if a candidate clears, the sensitivity projection, seeds 49 to 51, an independent re-derivation of the test
numbers and a whole-branch final review follow, then the full report. Target: results on Tuesday 6 October; cutoff for
development numbers Thursday 8 October 12:00; time-box ends Tuesday 13 October.
