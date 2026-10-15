contract_role: methodology
## Dimension Scores

### D1: methodology_rigor
score: block
trigger: "At least one proposed step could not discriminate the readings it claims to test as designed"
block_class: repairable

The block rests on W1, because H3's rules send the outcome that E3 makes most likely to two different next steps. W2, W4, W6 and W7 add to it. Each of these is a rule change, or a run that fits inside the same 10-minute GPU lock, so the design can be repaired before anything runs. Nothing in the memo shows the seed-45 episodes or the held rows spent, so the fatal trigger does not apply.

### D2: writing_and_structure
score: not_assessed

## Review Body

**Assessment in brief.** The memo asks a question it can answer: which repair step to run first, and under which rules, within about six working days. It also gets the hard constraints right. Seed 45 and the held rows are reserved and untouched, the pre-registration comes before any seed-45 score, H3 is kept off the method as a diagnostic on scorer-train rows, and §3 identifies the comparison that binds a GO. The weaknesses are in the decision rules.

- H3's fit rule is anchored at zero, and its scorer is unspecified, so an E3-like weak fit can come out as either "no fit" or "fits, weak transfer" (W1).
- The H1 pilot keeps a pick criterion that trades R@1 for gain at a rate the GO's R@1 comparison does not allow (W2), and it calls a result "promising" at a bar that predicts about a coin flip or worse on seed 45 (W3).
- "Falls short" is undefined, which leaves open a second GO test (W4).
- One H3 branch would carry evaluation-label information into the method's settings (W6).
- The 3.0 transfer bar counts the winner's-curse discount where it does not apply (W7).

None of these needs more than a day of the window, and most cost no compute. I score D1 block (repairable) and recommend a modified O3.

**Numbers I re-derived** (prose checks outside the four bounded receipt procedures; the receipts are in the last section).
- GO arithmetic (§3): 2 × 16.72 − 27.25 = 6.19 and 2 × 16.72 − 21.37 = 12.07, matching 6.2 and 12.1. The identity mean(R@1, gain) = either/4 + 3·gain/4 follows from R@1 = (either + gain)/2. With Appendix A §6.1's rounded seed-43 inputs it gives 7.095, 7.053, 6.818, 6.070 and 6.763 at λ = 0.5, 0.25, 1, ∞ and for the cosine, against the reported 7.09, 7.05, 6.82, 6.07 and 6.76. That is within 0.01, which rounding the inputs explains.
- §3's "roughly 0.7 to 1.0". Against a control with the same either rate, the R@1 margin equals gain/2. With E3's paired R@1 half-width of 0.33 (standard error 0.168), the expected lower bound reaches 0 at a gain of 0.66, which gives about a 50% chance of clearing. A gain of 0.94 gives about 80% (normal approximation, 2 × 2.80 × 0.168). The gain comparison itself needs only 0.43 to 0.63 for 80% at half-widths of 0.30 to 0.44. So the R@1 comparison against the nested control binds, and §3's range runs from a coin flip to 80% power. The memo should say which end it means.
- The 3.0 bar is 3 × 0.99 = 2.97. Appendix A §9.2's 80%-power figure reproduces: 1.17/1.96 × 2.80 = 1.67.
- The nested control's 56 cells collapse to 30 distinct λ sums (0 to 32, with no ∞). E3's uniform grid had 9 values, including ∞, with a per-half extension to 32 and 64.
- The memo's §2 table matches Appendix A wherever both report a value.
- No statistic in the manuscript is a t, z, F or chi-square value with df. Every interval is a painting-clustered percentile bootstrap, so p_from_test_statistic and n_from_df have nothing to act on. GRIM is informative only where few rankings stand behind a rate. That holds for the MLLM probe (3,600 pooled and 1,200 per pair). On the GO and development draws, at least 16,384 rankings stand behind every reported rate, so every two-decimal value is reachable. AR13 shows that argument on a value the memo quotes.

**Q1. Which order serves the three aims.** I recommend a modified O3. Commit the revised rules first. Then run the H1 pilot and H3 on day 1, and score H3's transfer on day 2, once the nested scorer exists.
1. Before any run, one hashed commit that holds the revised H3 and H1 rules below, the joint decision table, and the single-look rule for seed 45 (W4).
2. Day 1: write and test the 2-D cross-fit (H1). Build LAB and a matched-k bank, and launch L3, L5, a fixed-τ run and the matched-k run in one GPU lock (E3 ran three at a time, about 10 minutes each, 3.9 GiB each). Run the H1 pilot with A3 as the pre-specified primary model and the other checkpoints as descriptive rows.
3. Day 2 morning: score the LAB runs in-distribution and on seed 42 under both the term-only score and the nested score, each paired against A3 and C0 on identical episodes. Apply the joint table. If both readings are negative, the repair stops on day 2.

The reasons are as follows.
- H1 lies on every path after H3. Under "fits, weak transfer" the next step is H1 on the existing checkpoints. Under "fits and transfers" it is H2 scored with H1. Any A′ will be scored with the nested score. So the pilot is needed whatever H3 says. It is also the cheapest step (half a day, minutes of CPU), the most direct test of the A′ candidate, and the fastest way to close the shortest path if it fails.
- H3 answers a different question: is retraining worth the remaining days? That matters most when H1 is marginal, and the memo itself says H3's transfer read "is more informative when the nested scorer of H1 already exists". That argues against O1.
- The cost the memo lists for O3 (two development looks on seed 42 at once) is not extra. O1 makes the same two looks a day apart. Only O2, with H3 skipped, makes fewer, and it then risks spending the single seed-45 look on a thin margin without knowing whether retraining could widen it.
- O4 is the least diagnostic step, as the memo says.

If only one code stream can be reviewed per day, run H1 first and H3 on day 2, never O1. On the user's leaning: H3 is worth running, because it is the only step that can separate "the partitions are the limit" from "the architecture or loss is the limit". As designed, though, it should not run first and alone. It delays the cheapest decisive step by a day, and W1 leaves its most likely outcome without a clear next step.

**Q2. Can H3 discriminate its readings, and what sharpens it cheaply.** As designed, H3 separates "no fit" from a large, clear label-trained success. It does not handle the most likely outcome, a weak fit at E3's level. Applied to E3's own fresh pseudo-aspect results (Appendix A §6.4), the "no fit" rule fires for A3 under the cross-fitted agreement rule, 0.36 [−0.12, 0.86]. It does not fire under the training score, 0.95 [0.28, 1.62]. A4's training-score bound is −0.004. These are the changes, in order of value per cost.
- (a) **Needed:** a real baseline on identical episodes. Score A3 and C0 on the same fresh label episodes, and on the same seed-42 episodes, as each LAB run, and decide on the paired difference (LAB run minus A3, painting-clustered as in E3). "Fit" then means that labels help this architecture beyond what pseudo-training gave, and that is the quantity that separates the readings. This costs CPU minutes.
- (b) **Needed:** fix the decision scorer and the episode count. The primary scorer should be the term-only agreement rule (λ = ∞), whose gain replicated across draws (0.99 and 0.97). The cross-fitted agreement rule and the training score are reported as secondary scorers and do not decide. Use 4,096 fresh episodes per pair at a stated seed.
- (c) **Needed if "no fit" is to stop the repair:** make L5 mandatory and add one fixed-τ run. τ is the leading H2 suspect (Appendix A §6.3). "No fit" then requires every LAB run to fail. The memo's own §8 item 3 says that otherwise "no fit" may reflect A3's settings. All of these runs share one lock.
- (d) **Desirable, and doubly useful:** the matched-k control (k = 8, 23 and 10 on E2's affect, image and caption spaces). Its reading matters only on the "fits" path (granularity against feature space). It uses no labels, so it is also a legitimate A′ candidate. The cost is one bank (about 4 minutes of CPU) and one run in the same lock.
- (e) **Conditional:** a second model seed, used only when a deciding statistic lands within one half-width of its threshold.

**Q3. H3's rules and thresholds, and committing them first.** Yes, commit them in a hashed commit before the bank is built, as E3 did. The rules themselves should change in four ways (W1, W7).
- The 5% loss bar should be descriptive only. A3 ended 1.1% below its constant-score value and still had the largest term-only gain, so the loss level is not calibrated to gain.
- The fit reading should use the paired comparison of (a).
- The transfer bar should be stated in GO units. Leave out the halving, which does not apply to an unselected measurement. The memo's own chain then gives about 1.5. Better still, decide on the nested score directly: a label-trained ceiling below the gain a GO needs rules out any proxy-trained A′.
- The readings must cover every outcome. At present, a term-only gain of at least 3.0 with a lower bound at or below 0 falls under none of them.

Proposed H3 rules:
- **Fit, per LAB run (L3, L5, the fixed-τ run).** The run fits if the paired term-only gain over A3 on fresh label episodes has a lower bound above 0. It shows no fit if that point estimate is at or below 0. Anything in between is inconclusive and gets a second model seed (10 minutes).
- **H3 "no fit".** No LAB run fits. Reading: within the tried settings, labels do not improve this architecture and loss beyond pseudo-training, so retraining is unlikely to pay off inside the window. LAB may still answer yes or no for further H2 settings, under W6's restriction.
- **Ceiling, for the best fitting run.** Take its cross-fitted nested-score gain on seed 42 against its nested control. The ceiling is too low if that gain is below g* = max(5.6 × SE_R, 2.8 × SE_g). SE_R and SE_g are the paired standard errors of A3's nested score against its nested control in the pilot, so the formula is fixed before the run even though the number is not. With E3's half-widths, g* ≈ 0.94. Otherwise the ceiling is sufficient. Report the term-only seed-42 gain beside it.
- **Matched-k, on the fits path.** Take the matched-k run's seed-42 term-only gain, paired against A3. A lower bound above 0 means granularity is a lever, and H2 starts from the matched-k banks.

**Q4. Threats to the A′ GO test and to what the paper may claim.**
- **H3 does not threaten the GO test's validity.** It reads no selection, val or held row in training, and it touches seed 42 only. That holds under three conditions. The LAB checkpoints are kept out of the A′ candidate set by a hash list. No LAB-chosen setting passes into A′ (W6). The paper reports H3 as an upper bound trained on the evaluation aspects and values, which bounds selection among trained aspects and says nothing about C2 transfer.
- **The H1 pilot does not threaten the GO test's validity either,** under three conditions of its own. The whole of A′ is fixed in a commit before any seed-45 file exists: model, grid, pick rule, control, comparators, episode count and bootstrap. Seed 45 is scored once, and a failure ends the repair (W4). The ledger counts every seed-42 look.
- **Both steps do narrow what the paper may claim.** H1 came from a post-hoc profile that included the spent seed-43 test draw. Seed 45 draws its episodes from the same 6,451 selection paintings as seeds 42 and 43, so a GO is a fresh-episode result, not a fresh-painting one (W11). The paper must therefore report E3's NO-GO beside A′, the provenance of H1, the number of looks, and that dependence.
- **An existing conflict:** Appendix A §11 suggests seed 45 for a possible 8B probe (W9).

**Q5. What the H1 pilot design is missing.** The control is the right counterfactual (S2). Its set of sums differs from E3's grid (30 distinct sums, no ∞), which looks harmless here because A3's uniform term reached 16.89 at λ = 8 against 16.74 at ∞. Even so, the control should be computed over the distinct sums with a stated tie rule (W10). The grid is adequate, and including λ_a = 0 lets the nested score fall back to an unconditioned fusion. What is missing:
- a pick rule aligned with the binding comparison (W2);
- a pre-specified primary model, or a model choice cross-fitted together with λ (W3);
- a "promising" bar tied to seed-45 power, together with "not promising" and "inconclusive" branches (W3);
- a paired A3-minus-C0 comparison under the nested score, so that a pass can be credited to aspect training rather than to the factor basis;
- a precision step before the pre-registration. Use the pilot's paired standard errors to predict seed-45 power at half the observed margins. If that power falls below 50%, consider pre-registering a larger seed-45 episode count (scoring takes minutes of CPU). Check the gain on seed 42 first by subsampling, since the 4,575 anchor paintings cap what more episodes can buy.

Proposed H1 pilot rules:
- **Primary model:** A3 (E3's pick), with A1, A2, A4, A5, A6, C0 and SE as descriptive rows.
- **Pick, per parity half:** the nested control maximises R@1 (its gain is 0). The nested score maximises min(R@1 minus the R@1 of the control tuned on the same half, gain), with the first cell in a stated order winning ties. The same rule is reused unchanged for the A′ pick.
- **Reading:**
  - Promising: both point margins (R@1 against the nested control, and gain) are at least 2.80 times their paired standard errors, which puts both lower bounds above 0.
  - Not promising: either point margin is at or below 0.
  - Inconclusive: anything else.

**Joint decision table** (commit it with the rules):

| H1 pilot on A3 | H3 | Next step |
|---|---|---|
| promising | any | Pre-register A′ as the nested score on A3. H4 replaces it only if H3's ceiling is sufficient and an H2 model passes its pseudo-bank fit gate and beats A3 in the pilot by Oct 9, decided before the pre-registration commit. |
| inconclusive or not promising | ceiling sufficient | Run the pre-registered H2 grid on pseudo banks behind the training-fit gate (10 runs at most), then re-pilot the nested score on seed 42. Pre-register by Oct 9 if promising, otherwise branch 3. |
| inconclusive or not promising | no fit, or ceiling too low | Stop and go to branch 3 on day 2. |
| any path | any | Seed 45 is scored once, by Oct 12 at the latest. A failure ends the repair. |

**Needed versus merely desirable.**
- Needed, because without them a reading is uninterpretable or the GO test is weakened:
  - the paired A3 and C0 baselines and a fixed scorer for H3 (W1);
  - outcome sets that cover every outcome, with an inconclusive band, for both steps (W1, W3);
  - the aligned pick rule (W2);
  - a pre-specified primary model for the pilot (W3);
  - a definition of "falls short" and the single-look rule (W4);
  - LAB used as a yes or no gate only (W6);
  - L5 and a fixed-τ run, if "no fit" is to stop the repair (W8).
- Merely desirable:
  - the matched-k run (cheap, and a label-free candidate);
  - a second model seed near a threshold;
  - a seed-45 episode count chosen for power;
  - more bootstrap resamples, to shrink Monte Carlo error at boundary cases (three of E3's six bounds were boundary cases). Any change here must be pre-registered, with the seed fixed.

### S1: A data-use ledger that protects the GO test
**Evidence Anchor**: text: §7 "Seed 45 is reserved for the A′ GO test and is untouched (no episode or result file uses"

§7 names the use of every episode seed, keeps the held budget at 0 of 2, fixes RCA as the GO bar without re-picking it, and bans decision rules added after results. That is the right skeleton for a single, unbiased confirmatory look.

### S2: The nested uniform control is the right counterfactual
**Evidence Anchor**: text: §5.2 "which reduces to z(cos) + (λ_u + λ_a)·z(T_u)"

Replacing T_a with T_u keeps the fusion family, the code and the cross-fitting fixed, and removes only the condition. GO against this control therefore isolates what the conditioned term adds. The unit test that the 2-D function equals the 1-D one at λ_u = 0 guards the implementation.

### S3: H3 is fenced off as an upper-bound diagnostic
**Evidence Anchor**: text: §4 "It is an upper-bound diagnostic, labelled as such in the log and report, and never reported as the method."

The memo states that LAB uses scorer-train rows only and that no selection, val or held row enters the bank. It also states that H3 bounds selection among trained aspects, not transfer to an unseen aspect (§8 item 2).

### S4: The GO arithmetic identifies the binding constraint
**Evidence Anchor**: text: §3 "a method needs a gain above 2 × 16.72 − 27.25 ≈ 6.2 points"

The derivation from R@1 = (either + gain)/2 is correct (I re-derived 6.19 and 12.07). It shows why a fit repair alone needs 6 to 12 points, and it explains why the nested score is the most direct lead.

### S5: The forking-paths risk is disclosed before any result
**Evidence Anchor**: text: §4 "The 2-D grid adds pick freedom on seed 42. Changing the test-time score after a failed test"

The memo names the pick freedom, the change of score after a failed test, and the mitigations (a fresh seed, pre-registration, reporting E3's NO-GO). §8 adds the repeated seed-42 looks and the single draws.

### S6: Reproducibility affordances are concrete
**Evidence Anchor**: text: §5.2 "A1 to A6 (existing checkpoints, SHA-256 checked against E3's records)"

The memo names the existing checkpoints with hash checks, the builder and diagnostic scripts, unit tests for non-finite scores, and the E1 runner for the seed-45 comparators. A third party could rerun each step from the memo and Appendix B.

### W1: H3's fit rule is anchored at zero and its scorer is unspecified, so its most likely outcome has no single reading
**Severity**: Major
**Evidence Anchor**: text: §5.1 "the gain on fresh label episodes over scorer-train rows (a new episode seed), with the"
**Confidence**: 5 (read `train_fit_diagnostic.py`; applied the rule to the memo's own E3 numbers)

"No fit" requires two things: the loss ends less than 5% below its constant-score value, and the fresh label-episode gain has a lower bound at or below 0. The gain is measured "with the procedure of `train_fit_diagnostic.py`", and that script reports two gains, the cross-fitted agreement rule and the β = 0.3 training score. On E3's fresh pseudo-aspect episodes, A3 is "no fit" under the first (0.36 [−0.12, 0.86]) and "fits" under the second (0.95 [0.28, 1.62]). A5 "fits" under the first (0.61 [0.07, 1.18]), and A4's bound under the second is −0.004.

A LAB run that fits as weakly as E3 did is the outcome the memo's "architecture or loss" reading predicts. For that run, the choice of scorer and the bootstrap noise decide the classification, and the two classifications lead to different next steps: H2 on LAB, or H1 only and then branch 3. For LAB, the in-distribution and transfer measurements are the same labelled task on different rows, so "fits, weak transfer" mostly means a weak fit, which is the "no fit" reading under another name. A zero bar cannot tell "labels help" from "labels reproduce E3's level". The readings also do not cover every outcome. The author could decide on paired differences against A3 and C0 on identical episodes, with a fixed scorer and episode count, readings that cover every outcome and an inconclusive band (Q2 and Q3).

### W2: The pick criterion works against the comparison that binds a GO
**Severity**: Major
**Evidence Anchor**: text: §5.2 "parity halves, mean(R@1, gain) per half, as in E3"
**Confidence**: 5 (algebra from the memo's own identity)

The criterion equals either/4 + 3·gain/4, and the nested control, whose gain is 0, is in effect tuned on R@1 alone. Along a line of equal criterion, the either rate changes by −3 times the change in gain, so R@1 changes by minus the change in gain. The criterion strictly prefers a cell that adds 1.0 point of gain and loses 2.9 points of either rate (criterion +0.025), although that cell's R@1 falls 0.95 below the control's. The R@1 comparison against the nested control is the one that binds (see the re-derived numbers). This is the mechanism by which E3's cross-fit kept λ at 0.25 to 0.5 and lost 2.96 R@1 to its control.

Left as it is, the pilot can read "not promising" when a GO-aligned pick would pass, and the pass probability on seed 45 drops. The author could pre-register a criterion whose trade-off matches the GO, such as the min-margin rule in Q5, and use it unchanged in both the pilot and the A′ pick.

### W3: The pilot's "promising" bar is lenient and has no negative branch
**Severity**: Major
**Evidence Anchor**: text: §5.2 "control on seed 42 on both R@1 and gain (lower bounds above 0)"
**Confidence**: 4 (selection-inference reasoning from the memo's own winner's-curse figures)

"The best run" is a maximum over six or more models, scored on the draw where A3 was already picked, and E3's picked gain halved on the fresh draw (0.52 to 0.26). By §3's own arithmetic, margins that just clear 0 correspond to about a 50% chance of clearing on a fresh draw before any shrinkage, and to much less after it. Since seed 45 is a single look, this bar decides whether the repair's one test is spent on a likely failure. The pilot reading also has no "not promising" or "inconclusive" outcome. The author could pre-specify A3, or cross-fit the model choice together with λ, and require point margins of at least 2.80 paired standard errors on both comparisons. The author could add the negative and inconclusive branches and predict seed-45 power from the pilot's standard errors before the pre-registration.

### W4: "Falls short" is undefined, and seed 45 is not committed as a single look
**Severity**: Major
**Evidence Anchor**: text: §4 "the H1 score on an H2-repaired model, if H1 alone falls short"
**Confidence**: 4 (reading of §4, §5.3, §6 and §7 together)

If "falls short" can mean a failed seed-45 test, H4 would need a second GO test on another fresh draw of the same rows. Each extra attempt adds its own chance of a false GO, so two attempts roughly double it, and the memo states no correction. §7 reserves seed 45 "for the A′ GO test" but does not say that A′ is tested once or what follows a failure. The author could state three rules. First, "falls short" refers to the development pilot only. Second, seed 45 is scored once, for one pre-registered A′. Third, a seed-45 failure ends the repair. Any further fresh seed would need an error split committed before the first test.

### W5: No joint decision table and no dated stop
**Severity**: Minor
**Evidence Anchor**: absence: memo §5 to §6 — expected a joint table that maps each H1-pilot outcome and each H3 outcome to one next action with a dated stop; checked §0 aims, §5.1 outcome rules, §5.2 pilot reading, §5.3, §6 table and its notes, §7, §8 item 7
**Confidence**: 4 (the memo's aims set against its rules)

Aim 3, stopping early on a negative, depends on combinations: H1 negative with H3 negative should stop the repair on day 2. As written, the rules cover H3's three outcomes, the H1 reading has no negative case, and no date caps the H2 or H4 path. The table in the Review Body closes this at no cost.

### W6: One H3 branch would put evaluation-label information into A′'s settings
**Severity**: Major
**Evidence Anchor**: text: §5.1 "LAB, a fast testbed whose target is known to be learnable, before any pseudo-bank retraining"
**Confidence**: 4 (§7 and spec §4 C2 read together with the "no fit" branch)

On the "no fit" path, the next step is to tune H2's settings on label episodes of the evaluation aspects and then retrain on pseudo banks. The chosen τ, β or schedule would then have been selected with evaluation labels. §7 rules that out for the method, and it would weaken the paper's label-free claim on the most plausible path. The author could pre-register the H2 grid before H3 runs and let LAB answer only one question: does any setting in that fixed family fit labels? The whole grid would then train on pseudo banks, and the A′ cell would be picked by the pseudo-bank fit gate and the seed-42 criterion. The author could also exclude LAB checkpoints by hash and disclose LAB's role.

### W7: The 3.0 transfer bar applies the winner's-curse discount where it does not belong
**Severity**: Major
**Evidence Anchor**: text: §5.1 "gains halved on the fresh draw in E3, fusion keeps only part of a term's gain"
**Confidence**: 4 (re-derived from Appendix A §5 and §6.1)

The halving belongs to a picked, cross-fitted value: A3's gain of 0.52 against 0.26. H3's seed-42 term-only gain is an unselected, fixed-λ measurement, and A3's own term-only gain did not shrink between draws (0.99 against 0.97). The fusion discount has not been measured for the nested score. In E3's 1-D fixed-λ profile the gain survived at high λ (1.14 at λ = 8 against 0.97 term-only) and was lost to the cross-fitted pick. Without the halving, the memo's own chain gives a bar of about 1.5, half of 3.0. As written, the bar can classify a sufficient ceiling as "weak transfer" and stop H2 when it should not. The author could state the ceiling rule in the nested score's units, as the g* formula in Q3 does.

### W8: H3's settings confound is left optional
**Severity**: Minor
**Evidence Anchor**: text: §5.1 "Optionally L5 = A5's settings (β = 0), the one E3 run whose fresh-episode"
**Confidence**: 4 (cost and design read from §5.1, §6 and Appendix A §2)

The memo itself warns (§8 item 3) that "no fit" may reflect settings tuned for pseudo banks. Running L5 and a fixed-τ run in the same lock costs no extra calendar time, and it folds the memo's own "no fit" next step (H2 settings on LAB) into the first step. With them, "no fit" can stop the repair. The matched-k run and a conditional second model seed are desirable rather than needed (Q2).

### W9: Seed 45 is also proposed for the 8B probe
**Severity**: Minor
**Evidence Anchor**: text: Appendix A §11 "on a fresh episode seed (for example 45; seed 44 has been scored by both 2B runs)"
**Confidence**: 5 (direct comparison with §7)

The appended E3 report suggests seed 45 for an 8B probe, and the memo reserves the same seed for A′. If the probe is ever run before the A′ test, it would create seed-45 files, and the ledger's claim would no longer hold. The author could reassign the probe to a later seed in the ledger now.

### W10: Several reproducibility parameters are unstated
**Severity**: Minor
**Evidence Anchor**: absence: memo §5.1 Bank LAB and Measurements, and §5.2 Cross-fitting — expected the bank seed, the count and seed of the fresh label episodes, and a tie rule for the 56 cells that collapse to 30 control scores; checked §5.1, §5.2, the §6 cost note, §7, Appendix B Evaluation rules and Code entry points
**Confidence**: 4 (read `crossfit_lambda` in `src/eval/aspect_scorers.py`)

The memo gives "a new bank seed" and "a new episode seed" but no values and no episode count. `crossfit_lambda` breaks ties by list order, so with duplicate control cells the order of the cell list defines the pick. The author could state these values in the commit that fixes the rules.

### W11: Seed 45 shares paintings with every earlier look
**Severity**: Minor
**Evidence Anchor**: text: §8 "only the seed-45 test controls that, and only for the final claim"
**Confidence**: 4 (row scope in §1 and Appendix A §10 item 12)

Seed 45 removes episode-level selection, as E3's halving showed. It does not remove painting-level dependence, because all seeds draw from the same 32,413 selection rows (6,451 paintings), while the painting-clustered bootstrap treats paintings as the sampling unit. H1 was also generated from a profile that included the spent seed-43 draw. Neither can be removed in the window without reading val rows, which the plan rightly keeps unread. The author could disclose both and keep the look count in the ledger.

## Arithmetic Receipts

### AR1
procedure_id: grim
evidence_anchor: table: Appendix A §9.2 v2 table, pooled (900) row, MLLM v2 R@1 column
reported_inputs: MLLM v2 pooled R@1 13.50 percent at two decimals; 900 pooled episodes (300 per aspect pair); four rankings per episode (two conditions, two directions), 3,600 rankings; per-ranking outcome 1 if the target ranks strictly first, else 0
assumptions: R@1 is the unweighted share of the 3,600 rankings, as licensed by its definition and by the count of 3,600 rankings in §9.2; the paper states no rounding rule, so half-up, half-even and truncation are each checked and a verdict is given only if all three agree
derivation: 13.50 / 100 × 3,600 = 486.0, and 486 / 3,600 = 0.135000 exactly, which prints as 13.50 under all three rules
derived_value_or_range: 486 / 3,600 = 13.5000 percent
comparison_rule: the reported value must equal some attainable k / 3,600 at two decimals under each of half-up, half-even and truncation
rounding_interval: [13.495, 13.505) for half-up and half-even, which times 36 is [485.82, 486.18); [13.50, 13.51) for truncation, which times 36 is [486.00, 486.36)
nearest_achievable: 485/3,600 = 13.4722 and 487/3,600 = 13.5278 on either side of the attained 486/3,600 = 13.5000
status: consistent

### AR2
procedure_id: grim
evidence_anchor: table: Appendix A §9.2 v2 table, emotion × style row, MLLM v2 R@1 column
reported_inputs: MLLM v2 emotion × style R@1 11.33 percent at two decimals; 300 episodes, four rankings each, 1,200 rankings; per-ranking outcome 0 or 1
assumptions: unweighted share of the 1,200 rankings of this pair (300 episodes per pair, four prompts each, as stated in §9); rounding rule unstated, so half-up, half-even and truncation are each checked
derivation: half-up or half-even interval times 12 is [135.90, 136.02) and contains k = 136; truncation interval times 12 is [135.96, 136.08) and contains k = 136; 136 / 1,200 = 11.3333
derived_value_or_range: 136 / 1,200 = 11.3333 percent
comparison_rule: the reported value must equal some attainable k / 1,200 at two decimals under each of the three rules
rounding_interval: [11.325, 11.335) for half-up and half-even; [11.33, 11.34) for truncation
nearest_achievable: 135/1,200 = 11.2500 and 136/1,200 = 11.3333, straddling 11.33 from below and above
status: consistent

### AR3
procedure_id: grim
evidence_anchor: table: Appendix A §9.2 v2 table, emotion × genre row, MLLM v2 R@1 column
reported_inputs: MLLM v2 emotion × genre R@1 14.33 percent at two decimals; 1,200 rankings (300 episodes, four each); per-ranking outcome 0 or 1
assumptions: unweighted share of 1,200 rankings, as in AR2; rounding rule unstated, so all three rules are checked
derivation: half-up or half-even interval times 12 is [171.90, 172.02) and contains k = 172; truncation interval times 12 is [171.96, 172.08) and contains k = 172; 172 / 1,200 = 14.3333
derived_value_or_range: 172 / 1,200 = 14.3333 percent
comparison_rule: the reported value must equal some attainable k / 1,200 at two decimals under each of the three rules
rounding_interval: [14.325, 14.335) for half-up and half-even; [14.33, 14.34) for truncation
nearest_achievable: 171/1,200 = 14.2500 and 172/1,200 = 14.3333, straddling 14.33
status: consistent

### AR4
procedure_id: grim
evidence_anchor: table: Appendix A §9.2 v2 table, style × genre row, MLLM v2 R@1 column
reported_inputs: MLLM v2 style × genre R@1 14.83 percent at two decimals; 1,200 rankings (300 episodes, four each); per-ranking outcome 0 or 1
assumptions: unweighted share of 1,200 rankings, as in AR2; rounding rule unstated, so all three rules are checked
derivation: half-up or half-even interval times 12 is [177.90, 178.02) and contains k = 178; truncation interval times 12 is [177.96, 178.08) and contains k = 178; 178 / 1,200 = 14.8333
derived_value_or_range: 178 / 1,200 = 14.8333 percent
comparison_rule: the reported value must equal some attainable k / 1,200 at two decimals under each of the three rules
rounding_interval: [14.825, 14.835) for half-up and half-even; [14.83, 14.84) for truncation
nearest_achievable: 177/1,200 = 14.7500 and 178/1,200 = 14.8333, straddling 14.83
status: consistent

### AR5
procedure_id: grim
evidence_anchor: table: Appendix A §9.2 v2 table, emotion × style row, MLLM v2 gain column
reported_inputs: MLLM v2 emotion × style condition gain 0.58 points at two decimals; 1,200 rankings; per-ranking value +1 if the target ranks first, −1 if the other aspect's candidate ranks first, 0 otherwise
assumptions: gain is R@1 minus the other-aspect rate over the same 1,200 rankings (definition in §1 and Appendix A §3), so it is the unweighted mean of a per-ranking value in {−1, 0, +1}; rounding rule unstated, so all three rules are checked
derivation: half-up or half-even interval times 12 is [6.90, 7.02) and contains k = 7; truncation interval times 12 is [6.96, 7.08) and contains k = 7; 7 / 1,200 = 0.5833
derived_value_or_range: 7 / 1,200 = 0.5833 points
comparison_rule: the reported value must equal some attainable k / 1,200 at two decimals under each of the three rules
rounding_interval: [0.575, 0.585) for half-up and half-even; [0.58, 0.59) for truncation
nearest_achievable: 6/1,200 = 0.5000 and 7/1,200 = 0.5833, straddling 0.58
status: consistent

### AR6
procedure_id: grim
evidence_anchor: table: Appendix A §9.2 v2 table, emotion × genre row, MLLM v2 gain column
reported_inputs: MLLM v2 emotion × genre condition gain −2.25 points at two decimals; 1,200 rankings; per-ranking value in {−1, 0, +1}
assumptions: gain is the unweighted mean of the per-ranking value over 1,200 rankings, as in AR5; rounding rule unstated, so all three rules are checked
derivation: −2.25 / 100 × 1,200 = −27.0, and −27 / 1,200 = −0.0225 exactly, which prints as −2.25 under all three rules
derived_value_or_range: −27 / 1,200 = −2.2500 points
comparison_rule: the reported value must equal some attainable k / 1,200 at two decimals under each of the three rules
rounding_interval: [−2.255, −2.245) for half-up and half-even, which times 12 is [−27.06, −26.94); (−2.26, −2.25] for truncation toward zero, which times 12 is (−27.12, −27.00]
nearest_achievable: −28/1,200 = −2.3333 and −26/1,200 = −2.1667 on either side of the attained −27/1,200 = −2.2500
status: consistent

### AR7
procedure_id: grim
evidence_anchor: table: Appendix A §9.2 v2 table, style × genre row, MLLM v2 gain column
reported_inputs: MLLM v2 style × genre condition gain 0.08 points at two decimals; 1,200 rankings; per-ranking value in {−1, 0, +1}
assumptions: gain is the unweighted mean of the per-ranking value over 1,200 rankings, as in AR5; rounding rule unstated, so all three rules are checked
derivation: half-up or half-even interval times 12 is [0.90, 1.02) and contains k = 1; truncation interval times 12 is [0.96, 1.08) and contains k = 1; 1 / 1,200 = 0.0833
derived_value_or_range: 1 / 1,200 = 0.0833 points
comparison_rule: the reported value must equal some attainable k / 1,200 at two decimals under each of the three rules
rounding_interval: [0.075, 0.085) for half-up and half-even; [0.08, 0.09) for truncation
nearest_achievable: 0/1,200 = 0.0000 and 1/1,200 = 0.0833, straddling 0.08
status: consistent

### AR8
procedure_id: grim
evidence_anchor: table: Appendix A §9.2 v2 table, emotion × genre row, Cosine R@1 column
reported_inputs: seed-44 cosine emotion × genre R@1 14.83 percent at two decimals; 1,200 rankings (the same 300 episodes, four rankings each); per-ranking outcome 0 or 1
assumptions: unweighted share of 1,200 rankings, as in AR2; rounding rule unstated, so all three rules are checked
derivation: half-up or half-even interval times 12 is [177.90, 178.02) and contains k = 178; truncation interval times 12 is [177.96, 178.08) and contains k = 178; 178 / 1,200 = 14.8333
derived_value_or_range: 178 / 1,200 = 14.8333 percent
comparison_rule: the reported value must equal some attainable k / 1,200 at two decimals under each of the three rules
rounding_interval: [14.825, 14.835) for half-up and half-even; [14.83, 14.84) for truncation
nearest_achievable: 177/1,200 = 14.7500 and 178/1,200 = 14.8333, straddling 14.83 for the cosine column
status: consistent

### AR9
procedure_id: grim
evidence_anchor: table: Appendix A §9.2 v2 table, style × genre row, Cosine R@1 column
reported_inputs: seed-44 cosine style × genre R@1 14.58 percent at two decimals; 1,200 rankings; per-ranking outcome 0 or 1
assumptions: unweighted share of 1,200 rankings, as in AR2; rounding rule unstated, so all three rules are checked
derivation: half-up or half-even interval times 12 is [174.90, 175.02) and contains k = 175; truncation interval times 12 is [174.96, 175.08) and contains k = 175; 175 / 1,200 = 14.5833
derived_value_or_range: 175 / 1,200 = 14.5833 percent
comparison_rule: the reported value must equal some attainable k / 1,200 at two decimals under each of the three rules
rounding_interval: [14.575, 14.585) for half-up and half-even; [14.58, 14.59) for truncation
nearest_achievable: 174/1,200 = 14.5000 and 175/1,200 = 14.5833, straddling 14.58
status: consistent

### AR10
procedure_id: grim
evidence_anchor: table: Appendix A §9.1 v1 table, emotion × style row, MLLM v1 R@1 column
reported_inputs: MLLM v1 emotion × style R@1 11.83 percent at two decimals; 1,200 rankings (300 episodes, four each); per-ranking outcome 0 or 1
assumptions: unweighted share of 1,200 rankings, as in AR2; rounding rule unstated, so all three rules are checked
derivation: half-up or half-even interval times 12 is [141.90, 142.02) and contains k = 142; truncation interval times 12 is [141.96, 142.08) and contains k = 142; 142 / 1,200 = 11.8333
derived_value_or_range: 142 / 1,200 = 11.8333 percent
comparison_rule: the reported value must equal some attainable k / 1,200 at two decimals under each of the three rules
rounding_interval: [11.825, 11.835) for half-up and half-even; [11.83, 11.84) for truncation
nearest_achievable: 141/1,200 = 11.7500 and 142/1,200 = 11.8333, straddling 11.83
status: consistent

### AR11
procedure_id: grim
evidence_anchor: table: Appendix A §9.1 v1 table, emotion × style row, MLLM v1 gain column
reported_inputs: MLLM v1 emotion × style condition gain 1.00 points at two decimals; 1,200 rankings; per-ranking value in {−1, 0, +1}
assumptions: gain is the unweighted mean of the per-ranking value over 1,200 rankings, as in AR5; rounding rule unstated, so all three rules are checked
derivation: 1.00 / 100 × 1,200 = 12.0, and 12 / 1,200 = 0.0100 exactly, which prints as 1.00 under all three rules
derived_value_or_range: 12 / 1,200 = 1.0000 points
comparison_rule: the reported value must equal some attainable k / 1,200 at two decimals under each of the three rules
rounding_interval: [0.995, 1.005) for half-up and half-even, which times 12 is [11.94, 12.06); [1.00, 1.01) for truncation, which times 12 is [12.00, 12.12)
nearest_achievable: 11/1,200 = 0.9167 and 13/1,200 = 1.0833 on either side of the attained 12/1,200 = 1.0000
status: consistent

### AR12
procedure_id: grim
evidence_anchor: table: Appendix A §9.1 v1 table, pooled (900) row, MLLM v1 R@1 column
reported_inputs: MLLM v1 pooled R@1 13.56 percent at two decimals; 3,600 rankings (900 episodes, four each); per-ranking outcome 0 or 1
assumptions: unweighted share of 3,600 rankings, as in AR1; the paper states no rounding rule
derivation: under half-up or half-even the interval [13.555, 13.565) times 36 is [487.98, 488.34) and contains k = 488 (13.5556); under truncation the interval [13.56, 13.57) times 36 is [488.16, 488.52) and contains no integer
derived_value_or_range: 488 / 3,600 = 13.5556 percent rounds to 13.56, but no attainable k / 3,600 truncates to 13.56
comparison_rule: reachability at two decimals under each candidate rounding rule, with a verdict only if the rules agree; they do not agree here
status: not_computable
not_computable_reason: rounding_rule_ambiguous

### AR13
procedure_id: grim
evidence_anchor: table: memo §2 table, row Picked A3 (λ_aspect 3), cross-fitted, R@1 / gain
reported_inputs: A3 cross-fitted R@1 13.76 percent at two decimals on the seed-43 test episodes; 12,288 pooled episodes (4,096 per pair); four rankings per episode, 49,152 rankings; per-ranking outcome 0 or 1
assumptions: unweighted share of the rankings (R@1 definition in memo §1); rounding rule unstated, so all three rules are checked; the verdict also holds if each episode counted once (12,288), because the granularity 100/12,288 = 0.0081 is below the 0.01 width of every rounding interval
derivation: half-up or half-even interval times 491.52 is [6760.86, 6765.77) and contains k = 6761 to 6765; truncation interval times 491.52 is [6763.32, 6768.23) and contains k = 6764 to 6768
derived_value_or_range: for example 6764 / 49,152 = 13.7614 percent, which rounds and truncates to 13.76
comparison_rule: the reported value must equal some attainable k / 49,152 at two decimals under each of the three rules
rounding_interval: [13.755, 13.765) for half-up and half-even; [13.76, 13.77) for truncation
nearest_achievable: 6763/49,152 = 13.7594 and 6764/49,152 = 13.7614, straddling 13.76
status: consistent
