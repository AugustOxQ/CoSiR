contract_role: methodology

## Dimension Scores

### D1: methodology_rigor
score: block
trigger: "At least one candidate that the decision rule can carry forward would be uninterpretable as planned"
block_class: repairable

### D2: writing_and_structure
score: not_assessed

## Review Body

### Reviewer Identity

Peer Reviewer 1 (methodology), configured by Card #2: a statistician in machine-learning evaluation methodology (selective inference after selection on reused development data, select-then-test protocols, cross-fitting at the level of the independent unit, cluster bootstrap when items recur across episodes and seeds, intersection-union GO tests), with hands-on experience of episodic few-shot evaluation, of meta-learners trained on clustering-constructed pseudo-tasks, and of matched negative controls in score fusion. The object of review is Appendix A §5 (5.1 to 5.6), with §5.4 as the rule to be committed. Items the user fixed (memo §2, Appendix A §3) are treated as constraints: I say where they limit what a GO can license and do not ask to reopen them.

### Seat Recommendation

Revise §5.4 before any code (Major Revision on the template scale). This is the methodology seat's view only; the panel decision belongs to the synthesizer. Every fix below is a rewrite of §5 or §5.4 that fits Tuesday 6 October and needs no new data, seeds or runs beyond those the plan already schedules.

### Confidence Score

4. Core expertise for selection, control design, cluster bootstrap and pseudo-task transfer. I checked the plan and the appendices' arithmetic from the text; I could not read the stored arrays or the code.

### Calibration Status

`NOT_CALIBRATED`

### Summary Assessment

The plan replaces the label-free reader of CoSiR v2 with three candidates (R-a scaled Δ, R-b a learned reader trained on pseudo-aspect bank episodes, R-c a confidence gate), selects one on the seed-42 development episodes against a draft rule (§5.4), and tests it once, pooled over fresh episode seeds 49 to 51, against cosine, RCA, B′ and its matched counterpart. The test architecture is sound: one carried configuration, an intersection-union GO over every comparator, gain counted once, and a painting-clustered bootstrap that respects paintings shared across seeds. The matched counterparts of R-a and both R-b variants remove only the condition, and no evaluation row can reach R-b's training. The draft rule is not yet mechanical. R-c, which the rule can carry, has a counterpart that is not condition-free as written and would fail the code's own identity check, so its control would be improvised after the other numbers exist; this alone gives a repairable block. The kill rules read the told mapping and leave outcome patterns without a stated action, R-a's spread estimator mixes signal into its noise scale and partly re-creates failure 1, and R-b leaves free choices that must be fixed before any seed-42 number. Smaller gaps: two definitions of B′, no stated test sensitivity, an unscoped GO claim, A0's double role, a partial tie order, test-time cross-fits, and no cutoff in the timeline. Each is a one-sentence to one-paragraph rewrite of §5.

### Score Rationale for D1

D1 is block (repairable) on the trigger quoted above, carried by W1: item 1 makes R-c a candidate and item 4 can carry it (its tie-break ranks "no gate before gate"; it does not exclude the gate), yet R-c's declared control is not a condition-free score. W2 falls under the kill-rule clause of the same Phase 1 trigger, but its effect is confined to development selection, which the fresh-seed test still guards, so I band it Major. W3 to W11 are warn-level or below. I considered the fatal trigger because the fresh seeds reuse the 6,451 selection paintings that development used. It is not met: the test draws new episodes, the seed-42 interval widths are consistent with episode-level ranking noise (W6), and nothing selected on seed 42 is fitted per painting (one of seven configurations, a few fusion weights, one spread per grouping, a reader trained on scorer-train rows). The shared paintings therefore narrow what a GO licenses (W7) without leaving the test unable to discriminate. That reuse is fixed by the user (memo §2) and is treated as a scope limit.

### Criterion-Bound Judgements

| Dimension / criterion | Criterion source | Judgement | Evidence anchors | Rationale | Uncertainty or scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Candidate specification (frozen, runnable) | Contract D1; Phase 1 look-for | PARTLY_MEETS | §5.2 rows; S3, W1, W4 | R-a is fully specified; R-c's gate and R-b's settings are open | text only, no code read | yes: W1 carries the block |
| Matched controls and comparators | Contract D1; Appendix F.1 lesson | PARTLY_MEETS | §5.4 item 2; S2, W1, W5 | R-a and both R-b variants matched; R-c not; B′ defined twice | none identified | yes |
| Development bar | Contract D1 | PARTLY_MEETS | §1, §5.4 item 3; W5, W6, W9 | metric, comparator, threshold and direction fixed before results; calibration rests on two observed halvings; two clauses never bind | halving prior from two cases | yes |
| Kill and carry rules | Contract D1; Phase 1 block trigger | DOES_NOT_MEET | §5.4 items 4 and 5, §5.2 domain check; W2, W9 | outcome patterns without an action; told mapping inside rules | none identified | yes |
| Independence of the confirmatory test | Contract D1; Phase 1 fatal trigger | MEETS within the fixed seed policy | §5.4 item 6; S1, W7, W10 | new episodes; same paintings | scope of the claim | yes, as a scope note |
| Multiplicity and selection | Contract D1 | MEETS | S1, W9 | select on seed 42, one carried configuration, intersection-union GO | painting-level carry-over not removable under the seed policy | no |
| Uncertainty and power | Contract D1 | PARTLY_MEETS | §3 intervals; W6, W10 | painting bootstrap is right; test sensitivity unstated | variance split between paintings and episodes unknown | yes, for reading a NO-GO |
| Leakage in learned components | Contract D1 | MEETS | S3, W4 | no evaluation row enters R-b; cross-fitted heads for training | head sharpness mismatch | no |
| Reproducibility | Contract D1 | PARTLY_MEETS | §6; W4, W1 | paths, commands, seeds and data given; R-b and R-c parameters missing | none identified | yes |
| Time budget | Contract D1 | PARTLY_MEETS | §5.5; W11 | fits if R-b does not slip; no cutoff | run times as stated by the author | no |

### S1: The confirmatory test is a clean select-then-test with an intersection-union GO

One configuration is carried from seed 42, the test seeds are built once with a hash check against all earlier episodes, and the GO requires lower bounds above 0 against each of cosine, RCA, B′ and the matched counterpart. An intersection-union test keeps its level without a multiplicity correction, and the plan correctly counts the shared gain number once. Pooling with one cluster per painting across seeds is the right unit, because the same paintings recur across seeds. Selection noise on seed 42 does not bias this test at the episode level, since its episodes are new draws.
**Evidence Anchor**: text: Appendix A §5.4 item 6 "pooled over the three seeds (one cluster per painting across seeds)"

### S2: The two-condition mean defines matched counterparts that remove only the condition

For R-a and R-b arg-max the counterpart is the mean of the picked-grouping terms under the two conditions; for R-b's expected term it is Σ_h P̄(h)·s_h with P̄ the two-condition mean of the reader's probabilities, which keeps the example-adaptive weighting and removes only the condition. The counterpart gets the same 56 cells with the max-R@1 rule, the most favourable rule for a control, and the bar margin takes the larger of B′ and the counterpart. This is exactly the guard the N1 false pass called for (Appendix F.1).
**Evidence Anchor**: text: Appendix A §5.4 item 2 "the same fused score with the reader term replaced by its two-condition"

### S3: R-b's training cannot see an evaluation row

The bank uses scorer-train rows only, the heads that build R-b's training features are cross-fitted on painting halves so the reader learns from out-of-sample posteriors, and evaluation reads selection rows that no head or grouping was fit on. The groupings themselves were built on scorer-train paintings (Appendix C: the CSD partition labels the 36,518 scorer-train paintings, and a painting's vector is that of its first scorer-train row). I found no path by which an evaluation row or label enters R-b.
**Evidence Anchor**: text: Appendix A §5.2 R-b row "on pseudo-aspect bank episodes built from the four groupings (scorer-train rows only)"

### S4: The rule is reviewed and committed before the numbers it governs

The order of work puts this review and the commit of the rule before any code, and §7 lists the failures already paid for (the N1 control, the gate after z-scoring, the antisymmetry of Δ, the shared parity halves). The earlier design errors of step 0 are disclosed rather than hidden. This makes the remaining gaps fixable on paper.
**Evidence Anchor**: text: memo §0 "and its §5.4 rule are fixed before any code" and "before any number it governs exists"

### S5: The numbers the bar and the references rest on hold together

From the text I re-derived: R@1 = (either + gain) / 2 for every row of Appendix D §1 and of Appendix E Table 10 (for example (36.27 + 0.79) / 2 = 18.53, and (4.56 − 2.27) / 2 = 1.145 for the told margin); for all seven step-1 arms, reader margin = (Reader − B) − (T_cf − B) and bar margin = (Reader − B) − max(B′ − B, T_cf − B) within rounding, with the comparator the table names; 1,576 + 1,883 = 3,459 episodes, 28.1% of 12,288; and F = (H − 1)/(R_u − 1) = 0.50 for both groupings in Appendix B §8. The references in items 4 and 5 can be trusted as stated.
**Evidence Anchor**: table: Appendix C Arms table, columns Reader − B (T_cf − B), B′ (B′ − B) and Bar margin

### W1: R-c's matched counterpart is not condition-free as written, and its gate is unspecified

**Problem**: R-c scores s = z(B) + λ·g·z(T) with g "from the reader's top-two margin". That margin differs between the two conditions. For R-a, condition a's margin is the gap between the two largest scaled Δ values and condition b's is the gap between the two smallest (Δ under b is −Δ under a); for R-b, condition b's probabilities come from swapped inputs. So g_a ≠ g_b in general, and a counterpart that "applies the same gate to T_cf", read as g_c·z(T_cf), changes with the condition: its weight on T_cf follows the condition, its gain need not be 0, and `crossfit_condition_free` raises an error on it (memo §3). The form of g (hard threshold or ramp), the threshold grid, and whether the counterpart searches the same λ-by-threshold grid are also unstated.
**Evidence Anchor**: text: Appendix A §5.2 R-c row "the counterpart applies the same gate to T_cf"
**Why it matters**: R-c can be carried (items 1 and 4), and the GO compares the carried configuration with its matched counterpart on the test seeds. As written, the implementer must invent R-c's control on Wednesday, after R-a and R-b numbers exist. A control that removes more or less than the condition is the failure Appendix F.1 records: it can fake a pass or a fail.
**Suggestion** (fix to §5.2 R-c and §5.4 item 2): Define (i) the gate as g_c = 1 when the reader's top-two margin under condition c is at least τ, else 0, with τ on a fixed label-free grid: the 0th, 25th, 50th and 75th percentiles of the parent reader's seed-42 margin, both conditions pooled (the 0th percentile gives g ≡ 1, which is the parent itself). (ii) The fused score as (1 + λ_u)·z(B) + λ_a·g_c·z(T_c) on the existing 56 cells times the 4 thresholds, min-margin cross-fit. (iii) The counterpart term as G_cf = (g_a·z(T_a) + g_b·z(T_b)) / 2, fused as (1 + λ_u)·z(B) + λ_a·G_cf on the same 224 cells with the max-R@1 rule. G_cf is identical under both conditions, so it passes `crossfit_condition_free`'s check unchanged. I recommend G_cf over ḡ·z(T_cf) with ḡ the two-condition mean gate. Both are condition-free, but G_cf applies the operator every other counterpart in the plan uses (the two-condition mean of the complete conditioned term, as for T_cf and for R-b's expected term), whereas ḡ·z(T_cf) is a product of means and drops the covariance between gate and term, so it removes more than the condition. As a sanity check, report the 0th-percentile cell, where R-c and its counterpart should reproduce the parent and its counterpart up to z-scoring before rather than after the average. Cost: minutes.
**Severity**: Critical
**Confidence**: 4 (core expertise: matched negative controls in score fusion)

### W2: The kill rules are not mechanical and read the told mapping

**Problem**: (a) Item 5 kills when no candidate raises pick accuracy by 10 points "or reaches the bar". A candidate that gains 10 points of pick accuracy but misses the bar is neither killed (item 5) nor carried (item 4). §5.5 then sends "nothing cleared the bar" to the user, so the pick clause has no decision effect except to declare the reader route alive, which invites a post hoc extension. (b) R-b's domain-shift kill has no numbers ("high", "does not move") and no stated precedence over the bar: if R-b clears the bar while its pick accuracy does not move, the rule does not say whether R-b is killed. (c) Both kills read pick accuracy under the told mapping, which memo §2 calls a diagnostic only, and the same 10 points mean different things on A1 (chance 25%, injective mapping, ceiling 100%) and A0 (chance 33%, ceiling 83.3%). Appendix D's instruction never to compute pick accuracy on test seeds before the verdict is not restated.
**Evidence Anchor**: text: Appendix A §5.4 item 5 "if no candidate raises pick accuracy by at least 10 points over its configuration's arg-max reader" and §5.2 R-b requirements "first is high and the second does not move, R-b is killed"
**Why it matters**: Pattern (b) is plausible. Appendix C notes that pick accuracy understates how usable CSD is for genre, because a CSD pick under a genre condition counts as wrong; a reader that works through CSD can clear the bar without moving pick accuracy. A label-informed kill could then remove the one candidate that works, or the action is decided case by case after the numbers.
**Suggestion** (rewrite item 5 and the R-b check; I recommend option A). Option A: "5. Kill: if no candidate clears the development bar (item 3), no test is built and the result goes to the user for the Friday choice. Pick accuracy under the told mapping and R-b's held-out bank accuracy are seed-42 diagnostics, reported for every candidate, and enter no rule; pick accuracy is not computed on test seeds until the GO verdict is recorded." Replace R-b's pick-accuracy comparison by a label-free shift report: the standardized mean difference of each R-b input feature, and the distribution of R-b's top probability, on held-out bank episodes against seed-42 episodes (reported, not a kill). Option B, only if the user wants pick accuracy to stop work early: keep it as a stop for building R-c, with numbers scaled to each configuration's headroom (10 points on A1 is 18% of the distance to its ceiling, which is about 5 points on A0), and state that it never carries or kills a candidate the bar decides. Option A is cleaner: the bar and the matched counterpart already measure what reading adds, and it keeps the told mapping out of every rule.
**Severity**: Major
**Confidence**: 4 (core expertise: pre-registered decision rules)

### W3: R-a's spread mixes signal into the noise scale

**Problem**: R-a divides Δ_h by the root mean square of Δ_h over all seed-42 episodes, both conditions pooled. Because Δ_b = −Δ_a, that RMS equals the square root of noise variance plus mean squared signal. The standardization therefore fixes each grouping's mean squared scaled Δ at 1 over all episodes, so a grouping that carries a strong signal in many episodes is held near |scaled Δ| ≈ 1 exactly where it is right, while an uninformative grouping still exceeds 1 in about a third of episodes (P(|Z| above 1) ≈ 0.32). With the image grouping's mean Δ magnitudes from Appendix B §3 (0.0055, 0.0236 and 0.0205 across the three pairs) and the per-episode spread of about 0.017 reported there, its RMS would be about 0.025. In emotion × genre its scaled Δ would then be about 0.94 instead of about 1.39 under a noise-only scale, and the chance that one uninformative grouping outranks it rises from about 16% to about 22% (normal approximation). The 0.017 spread is reported for the affect grouping, so this is an order-of-magnitude check.
**Evidence Anchor**: text: Appendix A §5.2 R-a row "Spread = root mean square of Δ_h over the seed-42 development episodes, both conditions pooled"
**Why it matters**: R-a is the cheapest candidate and the one Appendix B §10 expects to fix failure 1 (coarse groupings winning by noise). An estimator that compresses the advantage of strongly informative groupings partly re-creates failure 1, so a miss by R-a would not show that noise scaling fails, and a Friday NO-GO could send the work to design L for an estimator reason.
**Suggestion** (fix to the R-a row; answers memo Q4): keep the development episodes as the source (label-free, same distribution as the test) and estimate noise only: σ_h = sqrt(mean over seed-42 episodes of (s²_S,h + s²_C,h) / 4), where s²_S,h and s²_C,h are the sample variances of the four support-pair and the four contrast-pair agreements on grouping h. This is the standard error of Δ_h implied by pair-to-pair scatter, pooled over episodes so it is stable, identical under both conditions, and built from the per-pair spread R-b already uses as a feature. Freeze it from seed 42 as planned; re-estimating per test seed is legitimate but unnecessary. Do not use bank episodes: the image heads' 60,000-row draw already touches 31,287 of the 36,518 scorer-train paintings (86%), so bank posteriors are largely in-sample and sharper, and bank supports share a group exactly. Replace the RMS rather than adding a second R-a, which would add a candidate. Cost: minutes.
**Severity**: Major
**Confidence**: 3 (core expertise in standardized test statistics; the size of the effect is estimated from appendix summaries)

### W4: R-b is not yet a frozen configuration

**Problem**: Four choices that define R-b are open. (i) The cross-fitted heads' training size and recipe: the standard heads use a 60,000-row draw, and heads refit "on half of the scorer-train paintings" with another row count would differ in sharpness, so the features R-b learns from (S_h, C_h, Δ_h and their spread all scale with posterior sharpness) would not match those it reads at evaluation from the standard heads. (ii) Regularisation, feature scaling, bank size and how the two half-readers are combined. (iii) The A0 reader needs its own three-grouping bank, in which the third grouping is controlled on the candidates, so it is a separate model. (iv) Nothing forbids revising these after seeing seed-42 pick accuracy or bar margins. The bank's label space also limits what R-b can learn: its classes are the groupings with a uniform prior and exact group sharing, while real conditions never call for the caption grouping under the told mapping and genre is not a class. "CSD and image agreeing together means genre" can therefore be learned only as an agreement pattern bank episodes happen to contain, at feature magnitudes larger than real episodes reach.
**Evidence Anchor**: absence: Appendix A §5.2 R-b row and R-b requirements — expected the cross-fitted heads' training-row count and recipe, R-b's regularisation, feature scaling, bank size and the A0 bank, fixed before any seed-42 number; checked §5.2, §5.4, §5.5, §6, memo §3 and Appendix D Step 1
**Why it matters**: A sharpness mismatch can sink R-b for an engineering reason, and unrecorded tuning on seed 42 either reads the told mapping (pick accuracy) or adds hidden selection beyond the seven declared configurations. R-b also feeds R-c, so its freedom propagates.
**Suggestion** (add an "R-b specification" paragraph to §5.2, committed with the rule): cross-fitted heads with `fit_one_head`'s recipe and a 60,000-row draw within each half (each half holds about 92,000 rows), with each refit head's held-out accuracy reported beside the §3 values before R-b is trained; a fixed bank size per half (for example 65,536 episodes, about 200 s each); features standardized on bank training episodes; C chosen by five-fold cross-validation on bank episodes only; the two half-readers' probabilities averaged; the same recipe for A0 on a three-grouping bank. State that no R-b setting changes after any seed-42 number is read, and that a change becomes a new candidate reported as such. Optional, only if declared now: a label-free correction of R-b's class prior by EM on unlabelled seed-42 episodes (Saerens and colleagues, 2002), which addresses the uniform bank prior without reading labels.
**Severity**: Major
**Confidence**: 4 (core expertise: meta-learning on clustering-constructed pseudo-tasks)

### W5: B′ has two definitions, B is missing from the comparators, and the fusion base is not declared

**Problem**: Item 2 defines B′ as B plus a term, while memo §3 says B′ is B rebuilt (`crossfit_condition_free(cos, T_N1u, T_6u over the configuration's own groupings, parity)`), which produced the stored 18.44 and 18.80. The two give different numbers, and B′ was the binding bar comparator for A0 and A1 on seed 42. A rebuilt B′ can fall below B (A2s: 18.24 against 18.34), yet B is not in the GO list. The rule also does not say what the reader is fused on: memo §3 fuses on B, Appendix D Step 4 fused on B′.
**Evidence Anchor**: text: Appendix A §5.4 item 2 "B′ = B plus the configuration's averaged-heads term" and memo §3 "so B′ is B rebuilt, not B plus a term"
**Why it matters**: The bar margin is the selection statistic, and its binding comparator should not depend on which sentence the implementer reads. Fusing on B while comparing with B′ also makes the bar margin conservative: on step-1 numbers it sits 0.04 to 0.05 below the reader margin (A0 +0.31 against +0.35, A1 +0.01 against +0.06).
**Suggestion**: In item 2 write "B′ = `crossfit_condition_free(cos, T_N1u, T_6u over the configuration's own groupings, parity)`, as in step 1". Make the bar comparator the largest R@1 of B, B′ and the counterpart, and add B to item 6's GO list; B is computed on every seed anyway, so this costs nothing. State the base: "fused on B, `crossfit_nested(B, B, T, parity)`, as in step 1". I recommend keeping B over switching to B′, because the references in items 4 and 5 are on B and the conservative bias is small; fusing on B′ would require recomputing them.
**Severity**: Minor
**Confidence**: 4 (cross-section consistency check from the text)

### W6: The test's sensitivity is not stated, so a NO-GO has no defined reading

**Problem**: The development bar rests on two observed halvings, but the plan never states how small a margin the pooled three-seed GO can detect. From the seed-42 bar-margin intervals (half-widths 0.215 for A0 and 0.235 for A1, so a standard error of about 0.11 to 0.12), the pooled half-width for seeds 49 to 51 would be about 0.12 to 0.14 if episode-level ranking noise dominates (three times the episodes) and about 0.18 to 0.20 if painting-level variation dominates (at most 6,451 painting clusters against 4,602 now). If the carried margin halves to +0.25, the chance that one comparison's lower bound clears 0 is about 0.96 in the first case and 0.73 in the second; at +0.15 it is about 0.62 and 0.34. The GO needs both the B′ and the counterpart comparisons, so joint power is a little lower.
**Evidence Anchor**: absence: Appendix A §5.4 item 6 and §5.5 — expected a projected pooled half-width or smallest detectable bar margin for seeds 49 to 51, and a stated reading of a NO-GO whose pooled point estimate is above 0; checked memo §1, §5.3, §5.4, §5.6, Appendix D §5 and Appendix F.2
**Why it matters**: On Friday a NO-GO with a positive point estimate can mean that the reader does not work or that the test could not see a margin of that size. The choice between continuing the reader and design L depends on which.
**Suggestion** (fits the user's seed policy; no extra seeds, no change to the GO): on Thursday, before building the seeds, split the carried configuration's seed-42 per-episode margin variance into a between-painting and a within-painting part (a one-way decomposition by anchor painting, minutes on CPU), project the pooled three-seed half-width from it, and write that number in the log. Add to item 6: "A NO-GO whose pooled point estimate is above 0 is reported as inconclusive at a detectable margin of x, not as evidence that the reader fails."
**Severity**: Minor
**Confidence**: 3 (core expertise in cluster-bootstrap inference; the variance split is not recoverable from the text)

### W7: What a GO licenses is not stated in the rule

**Problem**: Fresh seeds re-draw episodes on the same 6,451 selection paintings, of which seed 42 already anchors 4,602 (71%), and every episode draws its 16 example rows and 13 candidates from the same pool. The grouping set was settled with labelled development episodes in view: CSD was kept over Gram after its told margins were seen (Appendix B §10). The GO is pooled over aspect pairs, while A1's reader margin on style × genre was −0.73 on seed 42.
**Evidence Anchor**: text: Appendix E §2.2 "Different episode seeds reuse the same 6,451 selection paintings"
**Why it matters**: The Friday decision feeds a CVPR plan. A GO shows that the configuration generalizes to new episodes on known paintings with a pooled margin; it does not show transfer to new paintings or reading on each pair. These are limits of user-fixed items (seed policy, grouping choice), stated here as a scope note.
**Suggestion**: Add a "Claim licensed" line to item 6: "A GO shows that the carried configuration beats each comparator, pooled over the three aspect pairs, on new episodes drawn from the same 6,451 selection paintings. It does not show transfer to new paintings (the held split, 12,281 paintings, stays reserved for the paper test) or a margin on each pair (per-pair results are reported, not tested)." On the label policy, the paper may describe the groupings as built without evaluation labels and may cite the label-free diagnostics that also rank CSD above Gram (placeability P_ami 0.208 against 0.150, Leiden-seed stability 0.807 against 0.673, Appendix C), and should disclose that the decision to continue with CSD came after its told margins were read.
**Severity**: Minor
**Confidence**: 4 (core expertise: select-then-test scope)

### W8: A0 is both reference and candidate, and the comparison that answers its purpose is not measured

**Problem**: §5.1 runs every reader on A0 to show whether CSD helps once the reader works, but §5.3 lists paired differences only against each configuration's current arg-max reader, not A1 against A0 under the same reader. Item 1 makes the A0 runs carry-eligible and item 4 prefers A0 within 0.05, so plan (a), framed around CSD, can end in a GO that says nothing about CSD.
**Evidence Anchor**: text: Appendix A §5.1 "to show whether the CSD grouping helps once the reader works"
**Why it matters**: The stated purpose of the A0 runs has no measure, and the tie-break can turn a near-tie into a GO on the configuration without CSD.
**Suggestion**: Add to §5.3 "paired A1 − A0 difference in bar margin under each reader (seed 42, descriptive)". In item 4, replace the A0-before-A1 tie-break by a priority: "carry the best A1 candidate that clears the bar; carry an A0 candidate only if no A1 candidate clears it". I recommend this over making A0 reference-only: it keeps the user's chance of a GO while making the CSD question answerable and the priority explicit before the numbers.
**Severity**: Minor
**Confidence**: 4 (rule-structure check from the text)

### W9: The selection rule has a partial order and clauses that never bind

**Problem**: Item 4's simplicity order does not rank R-a on A1 against R-b on A0, does not rank R-b arg-max against R-b expected (§5.2 calls them primary and secondary, item 1 treats them alike), and does not say what "within 0.05" is measured from. Item 1 builds R-c on the best candidate by bar margin without saying whether that candidate must clear the bar. Item 3's lower-bound clause is implied by the +0.5 point at seed-42 half-widths of 0.21 to 0.25, and its gain clause was met by every step-1 arm (gain margins +0.28 to +1.57), so neither filters anything. Item 6 does not say that per-seed results are descriptive.
**Evidence Anchor**: text: Appendix A §5.4 item 4 "R-a before R-b, no gate before gate, A0 before A1"
**Why it matters**: A partial order leaves the carried configuration open in a tie, and an unstated per-seed role invites a per-seed veto after the pooled verdict.
**Suggestion**: Write a total order: R-a, R-b arg-max, R-b expected, R-c, within A1 first and then A0 if W8's priority is adopted (A0 before A1 otherwise); "ties: candidates within 0.05 R@1 of the largest bar margin go to the earliest in this order"; "R-c is built on the candidate with the largest bar margin, whether or not it clears the bar"; keep item 3's clauses, noted as kept for continuity; add "per-seed results are reported and do not change the pooled verdict". The 0.05 window is far below the noise of paired differences between candidates, so the rule in effect carries the maximum. That is acceptable here because the fresh-seed test decides, and the expected optimism of a maximum over seven configurations (about 1.35 standard errors, near 0.15 R@1 if they were independent, less since they share episodes) lies within the halving the +0.5 bar already allows.
**Severity**: Minor
**Confidence**: 4 (core expertise: winner's curse after selection)

### W10: Test-time cross-fits read the test labels on parity halves that share paintings

**Problem**: Item 6 reruns every cross-fit on each test seed's own halves, so the fusion cells (and R-c's threshold) are chosen with the R@1 and gain of labelled test episodes, one half at a time. The halves are episode-index parity, so the same anchor paintings sit in both halves, and the bootstrap holds the cross-fit picks fixed, so the intervals omit the variance of the cell choice.
**Evidence Anchor**: text: Appendix A §5.4 item 6 "rerun every cross-fit on each" and memo §3 "two cross-fit halves are the episodes' index parity"
**Why it matters**: The leak is small (a few scalar weights from at most a few hundred cells, with the same freedom for the configuration and its counterpart), and the procedure matches the earlier fresh-seed tests whose halvings set the bar. The GO then certifies "reader plus per-seed cross-fit" rather than a frozen scorer, and the paper must say so.
**Suggestion**: Minimal: state in item 6 that per-seed cross-fitting is part of the method's definition and that the intervals hold the picks fixed. Optional and free: report beside the GO the same comparisons with the seed-42 cells frozen, as a descriptive line and not a second verdict. Painting-level halves would be cleaner but are not needed for a decision of this size.
**Severity**: Minor
**Confidence**: 4 (core expertise: cross-fitting at the level of the independent unit)

### W11: The timeline has no cutoff, and the test numbers are not reviewed before the decision

**Problem**: R-b is the long pole (cross-fitted heads for four groupings, two modalities and two halves, two banks, separate A1 and A0 readers), R-c depends on knowing the best of R-a and R-b, and Thursday holds both the rule application and the test. The timeline schedules the controller's re-derivation of development numbers on Wednesday but not of the test numbers, and it does not schedule the whole-branch final review that Appendix A §6 requires, so the Friday decision would rest on unreviewed test numbers.
**Evidence Anchor**: table: Appendix A §5.5 Timeline, rows Wed 7 Oct and Thu 8 Oct
**Why it matters**: If R-b slips, the rule does not say whether to wait (pushing the test into Friday) or proceed without it, a choice that would be made with R-a's numbers known. The project's own record (Appendix F.1) shows a final review catching a false pass.
**Suggestion**: Add to §5.4: "Candidates without development numbers by Thu 8 Oct 12:00 are dropped from this round, and R-c is built on the best completed candidate." Add to §5.5: Thursday evening, the controller re-derives the test numbers; Friday morning, the whole-branch final review runs before the GO or NO-GO is reported. Fallback order if time runs short: R-a and R-c on R-a first, R-b arg-max on A1 next, the remaining configurations last.
**Severity**: Minor
**Confidence**: 3 (process judgment from the text; run times as stated by the author)

### Answers to the Six Review Questions

1. Matched counterparts and B′: R-a and both R-b variants are matched (S2). R-c is not, as written (W1). B′ needs one definition, and B should join the comparators (W5).
2. Leakage in R-b: no evaluation row or label can enter R-b, and cross-fitted heads keep training posteriors out of sample (S3). The remaining risks are a head-sharpness mismatch between training and evaluation and unrecorded tuning on seed 42 (W4).
3. Selection on a reused seed 42: choosing the maximum of seven configurations inflates the development margin by roughly 0.1 to 0.15 R@1, which the +0.5 bar absorbs, and the fresh-seed test keeps its level because its episode noise is new and the GO is an intersection-union test (S1, W9). What the test cannot remove under the fixed seed policy is carry-over through shared paintings (W7), and its sensitivity is unstated (W6).
4. R-a's spread: development episodes are an acceptable source; the RMS estimator is the problem; use the pooled within-episode standard error and freeze it (W3).
5. Thresholds and label policy: the kill rules are incoherent with item 4 and read the told mapping (W2); the selection details need a total order (W9); the label policy limits the grouping claim as noted in W7.
6. Timeline: feasible if R-b does not slip; R-b slips first, then R-c; add a cutoff and schedule the test re-derivation and final review before Friday's decision (W11).

### Questions for Authors

1. Is R-c's top-two margin meant per condition, as the antisymmetry of Δ implies, or per episode, and should g be a hard threshold or a ramp?
2. Was item 5's pick-accuracy clause meant as a route to the test without the bar, or as a reason to keep developing the reader?
3. Will a GO on an A0 configuration count as a GO for plan (a) in the user's Friday decision?
4. Were the two earlier halvings (0.52 to 0.26, 0.26 to 0.15) measured on R@1 margins against a matched control or on another quantity? The bar's calibration rests on them.

### Minor Issues

- Item 3: name the gain statistic as step 1 did (`fusedT_vs_fusedTcf.gain`) and say that "≥ +0.5" applies to the full-precision point, as Appendix C's readings did.
- Item 5's reference pick accuracies (A1 43.2%, A0 54.7%) are seed-42 values of the arg-max reader fused on B; if W2's option A is adopted they move to §5.3 as diagnostics.

## Arithmetic Receipts

### AR1
procedure_id: grim
evidence_anchor: text: Appendix D §3 "under both conditions in 28.1% of episodes"
reported_inputs: reported share 28.1% of episodes at one decimal; N = 12,288 seed-42 episodes (memo §3, Appendix E §2.1); per-episode indicator 0 or 1; reported count 3,459 episodes (emotion × style 1,576, emotion × genre 1,883, style × genre 0)
assumptions: unweighted share over episodes as the text states; analytic N = 12,288 as stated for seed 42; rounding rule not stated, so half-up, half-even and truncation are each checked
derivation: 1,576 + 1,883 + 0 = 3,459; 3,459 / 12,288 = 0.2814941 = 28.149%; granularity 1/12,288 = 0.0081 percentage points, finer than the 0.1-point precision; 28.149 rounds to 28.1 under half-up, half-even and truncation
derived_value_or_range: 28.149% from the stated count
rounding_interval: [28.05%, 28.15%) under half-up or half-even; [28.1%, 28.2%) under truncation
nearest_achievable: 3,452/12,288 = 28.092% and 3,453/12,288 = 28.101% straddle 28.1%; the stated count gives 28.149%, inside both intervals
comparison_rule: consistent if the stated count's share lies in the rounding interval of 28.1 under every candidate rounding rule
status: consistent

### AR2
procedure_id: grim
evidence_anchor: table: Appendix B Table 2, row affect k-means 64, image head 13.5%
reported_inputs: held-out accuracy 13.5% at one decimal; N = 10,000 held-out scorer-train rows (Table 2 caption); per-row outcome correct or incorrect
assumptions: unweighted share of the 10,000 held-out rows as the caption states; rounding rule not stated, so half-up, half-even and truncation are each checked
derivation: granularity 1/10,000 = 0.01 percentage points; 1,350/10,000 = 13.50% exactly, which rounds to 13.5 under all three rules
derived_value_or_range: 13.50% attainable with k = 1,350
rounding_interval: [13.45%, 13.55%) under half-up or half-even; [13.5%, 13.6%) under truncation
nearest_achievable: 1,349/10,000 = 13.49% and 1,351/10,000 = 13.51% straddle 13.5%; 1,350/10,000 = 13.50% equals it
comparison_rule: consistent if some k/10,000 lies in the rounding interval under every candidate rounding rule
status: consistent

### AR3
procedure_id: grim
evidence_anchor: table: Appendix B Table 2, row affect k-means 64, caption head 34.6%
reported_inputs: held-out accuracy 34.6% at one decimal; N = 10,000 held-out scorer-train rows (Table 2 caption); per-row outcome correct or incorrect
assumptions: unweighted share of the 10,000 held-out rows as the caption states; rounding rule not stated, so half-up, half-even and truncation are each checked
derivation: granularity 1/10,000 = 0.01 percentage points; 3,460/10,000 = 34.60% exactly, which rounds to 34.6 under all three rules
derived_value_or_range: 34.60% attainable with k = 3,460
rounding_interval: [34.55%, 34.65%) under half-up or half-even; [34.6%, 34.7%) under truncation
nearest_achievable: 3,459/10,000 = 34.59% and 3,461/10,000 = 34.61% straddle 34.6%; 3,460/10,000 = 34.60% equals it
comparison_rule: consistent if some k/10,000 lies in the rounding interval under every candidate rounding rule
status: consistent

### AR4
procedure_id: grim
evidence_anchor: table: Appendix B Table 2, row image k-means 64, image head 92.7%
reported_inputs: held-out accuracy 92.7% at one decimal; N = 10,000 held-out scorer-train rows (Table 2 caption); per-row outcome correct or incorrect
assumptions: unweighted share of the 10,000 held-out rows as the caption states; rounding rule not stated, so half-up, half-even and truncation are each checked
derivation: granularity 1/10,000 = 0.01 percentage points; 9,270/10,000 = 92.70% exactly, which rounds to 92.7 under all three rules
derived_value_or_range: 92.70% attainable with k = 9,270
rounding_interval: [92.65%, 92.75%) under half-up or half-even; [92.7%, 92.8%) under truncation
nearest_achievable: 9,269/10,000 = 92.69% and 9,271/10,000 = 92.71% straddle 92.7%; 9,270/10,000 = 92.70% equals it
comparison_rule: consistent if some k/10,000 lies in the rounding interval under every candidate rounding rule
status: consistent

### AR5
procedure_id: grim
evidence_anchor: table: Appendix B Table 2, row image k-means 64, caption head 22.9%
reported_inputs: held-out accuracy 22.9% at one decimal; N = 10,000 held-out scorer-train rows (Table 2 caption); per-row outcome correct or incorrect
assumptions: unweighted share of the 10,000 held-out rows as the caption states; rounding rule not stated, so half-up, half-even and truncation are each checked
derivation: granularity 1/10,000 = 0.01 percentage points; 2,290/10,000 = 22.90% exactly, which rounds to 22.9 under all three rules
derived_value_or_range: 22.90% attainable with k = 2,290
rounding_interval: [22.85%, 22.95%) under half-up or half-even; [22.9%, 23.0%) under truncation
nearest_achievable: 2,289/10,000 = 22.89% and 2,291/10,000 = 22.91% straddle 22.9%; 2,290/10,000 = 22.90% equals it
comparison_rule: consistent if some k/10,000 lies in the rounding interval under every candidate rounding rule
status: consistent

### AR6
procedure_id: grim
evidence_anchor: table: Appendix B Table 2, row caption k-means 64, image head 21.6%
reported_inputs: held-out accuracy 21.6% at one decimal; N = 10,000 held-out scorer-train rows (Table 2 caption); per-row outcome correct or incorrect
assumptions: unweighted share of the 10,000 held-out rows as the caption states; rounding rule not stated, so half-up, half-even and truncation are each checked
derivation: granularity 1/10,000 = 0.01 percentage points; 2,160/10,000 = 21.60% exactly, which rounds to 21.6 under all three rules
derived_value_or_range: 21.60% attainable with k = 2,160
rounding_interval: [21.55%, 21.65%) under half-up or half-even; [21.6%, 21.7%) under truncation
nearest_achievable: 2,159/10,000 = 21.59% and 2,161/10,000 = 21.61% straddle 21.6%; 2,160/10,000 = 21.60% equals it
comparison_rule: consistent if some k/10,000 lies in the rounding interval under every candidate rounding rule
status: consistent

### AR7
procedure_id: grim
evidence_anchor: table: Appendix B Table 2, row caption k-means 64, caption head 89.7%
reported_inputs: held-out accuracy 89.7% at one decimal; N = 10,000 held-out scorer-train rows (Table 2 caption); per-row outcome correct or incorrect
assumptions: unweighted share of the 10,000 held-out rows as the caption states; rounding rule not stated, so half-up, half-even and truncation are each checked
derivation: granularity 1/10,000 = 0.01 percentage points; 8,970/10,000 = 89.70% exactly, which rounds to 89.7 under all three rules
derived_value_or_range: 89.70% attainable with k = 8,970
rounding_interval: [89.65%, 89.75%) under half-up or half-even; [89.7%, 89.8%) under truncation
nearest_achievable: 8,969/10,000 = 89.69% and 8,971/10,000 = 89.71% straddle 89.7%; 8,970/10,000 = 89.70% equals it
comparison_rule: consistent if some k/10,000 lies in the rounding interval under every candidate rounding rule
status: consistent
