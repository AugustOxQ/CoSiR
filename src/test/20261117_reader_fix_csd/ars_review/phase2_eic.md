contract_role: eic
## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: writing_and_structure
score: block
trigger: "a kill or carry rule is stated in more than one place with conflicting thresholds, metrics, seed sets or ordering, or rests on a load-bearing term that is undefined or used in two incompatible senses"

## Review Body

### Reviewer Identity
Journal-Fit Reviewer per Reviewer Configuration Card #1: a research lead in vision-language representation learning and few-shot retrieval with long area-chair experience (field-general, no venue bound), who chairs go/no-go reviews and reads registered-report style protocols. Here "journal fit" means fitness for purpose: can someone holding only the cover memo (§0 to §4) and Appendix A §5 write `DECISION_RULE.md` and apply it, without asking, to every development and test outcome.

### Overall Recommendation
Minor Revision (seat signal on D2 only; this seat does not assess D1). The draft rule needs a rewrite of §5.4 before it is committed. No new analysis or data is needed.

### Confidence Score
4. Decision-document structure and rule wording are core competence; whether the thresholds and controls are statistically sound is outside this seat.

### Calibration Status
`NOT_CALIBRATED`

### Criterion-Bound Judgements
| Dimension / criterion | Criterion source | Judgement | Evidence anchors | Rationale | Uncertainty or scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| D2 writing_and_structure | contract D2; my Phase 1 scoring plan | DOES_NOT_MEET | anchors of W1, W2, W3 (supported by W4 to W10) | the draft rule cannot be applied unambiguously to several plausible outcomes, and no precedence rule names a binding version among the places the rule is stated | text-level judgement only; soundness of thresholds is not judged here | yes: sets the D2 score |
| D1 methodology_rigor | contract D1 | NOT_ASSESSED | none | not eligible for this seat | none | no |
| Card #1 focus 1: outcome-to-action completeness | Reviewer Configuration Card #1 | PARTLY_MEETS | anchors of W1, W3, W4, W5 | carry, bar and GO are concrete; the kill, the R-b kill, the tie-break and the NO-GO path are not | none identified | yes: feeds D2 |
| Card #1 focus 2: terms and names | Reviewer Configuration Card #1 | PARTLY_MEETS | anchors of W2, W7, W8 | B′ has two incompatible definitions; A′ is never defined; several codes and names collide | weighed by whether the rule file would be misread | yes: W2 feeds the block |
| Card #1 focus 3: memo fidelity and self-containment | Reviewer Configuration Card #1 | PARTLY_MEETS | anchors of S1, S3, S5, W10 | §4 quotes §5.6 exactly; §1 restates the bar incompletely; the definitions the rule needs sit in memo §3, not in §5 | figures do not render; checked that claims resting on them are restated in text | no: correctable wording |
| Card #1 focus 4: framing and demarcation | Reviewer Configuration Card #1 | PARTLY_MEETS | anchors of S2, W6, W9, W10 | fixed and open items are cleanly fenced; A0's role, the plan's own prior and the process gates are not visible where the decision is framed | none identified | no: correctable wording |

### Summary Assessment
The document asks whether Appendix A §5 can be committed as `DECISION_RULE.md` and applied by someone holding only the memo and §5. As a plan it is well organised. The memo frames six questions that match §5.6 word for word, the user-fixed constraints are fenced off from what the review may change, memo §3 names the exact functions, grids and selection rules the rule depends on, and the GO rule is concrete about pooling, clustering and not counting gain twice. The draft rule is not yet committable. Item 5's kill reads two ways, and under either reading one plausible development outcome gets no action or contradicts the timeline. Item 2 defines B′ as "B plus a term", which memo §3 explicitly says it is not, and §5 never states what the reader is fused on. R-b carries a second kill rule outside §5.4 with no numbers. The tie-break is a partial order, A0 is both reference and candidate with no stated reading, R-c leans on an undefined "A′", and there is no single outcome table, slip rule or slot for the final review. None of this needs new analysis. Rewriting §5.4 as a self-contained rule with a definitions block, an outcome table and one precedence line fixes all of it in a few hours. This card judges text and structure; the soundness of the thresholds and controls belongs to the methodology seat.

### Score rationale for writing and structure
I score D2 block under my Phase 1 block trigger. The kill rule is stated in §5.4 item 5, §5.2 (R-b's domain-shift kill) and §5.5 (Thursday and Friday rows), and under one reading of item 5 those statements conflict (W1). R-b's kill rests on the undefined terms "high" and "does not move" (W3). B′, a comparator in both the bar and the GO, is used in two incompatible senses (W2). The document names §5.4 as the draft to be committed but gives no precedence rule between §5.4, §5.2, §5.5, memo §1 and §3, and the Appendix D template the plan adapts, so for these items there is no binding version. Two careful readers can therefore reach different verdicts on outcomes the evidence makes plausible: a candidate that clears the bar without a 10-point pick gain (in the Leiden sweep the reader margin reached +0.50 while pick accuracy stayed at 54.1 to 55.9%, Appendix B §5), or R-b clearing the bar with a modest pick gain. The block is about committability, not the plan's design. Fixing W1 to W3 and adding a precedence line would bring D2 to warn; adding the outcome table (W4) as well would bring it to pass. Where these ambiguities also change what is computed they bear on D1. I score them here only as text a reader cannot apply, and do not judge their methodological soundness.

### S1: The memo quotes the review targets exactly and frames them as answerable questions
I compared memo §4 with Appendix A §5.6 word by word: they are identical. The six questions in §0 cover the same items in the same order. A reader can trust the memo's account of what is under review.
**Evidence Anchor**: text: memo §4 heading "What the review should look at (Appendix A §5.6, verbatim)"

### S2: Fixed and open items are fenced clearly
Memo §2 and Appendix A §3 list what the user fixed (plan (a), the four groupings, the label policy, matched controls, seed handling), and §5.4 says the review may change any of its items. Reviewers and implementers know which levers are open.
**Evidence Anchor**: text: Appendix A §3 heading "Decided and not to be reopened" and §5.4 "The review may change any item."

### S3: The implementation facts make most rule quantities computable
Memo §3 gives the fusion grid (7 × 8, 56 cells), the selection rule of each cross-fit, the counterpart construction and its identical-under-both-conditions check, the bar-margin comparator choice, the resampling unit and the antisymmetry of Δ. This is what a rule file needs. The weakness (W2) is only that §5 does not carry it.
**Evidence Anchor**: text: memo §3 "Read from the code on 2026-10-06; stated here because the plan refers to these functions by name."

### S4: The GO rule is concrete
Item 6 names the seeds, the build command, the hash check, the pooling unit, the four comparators, the instruction not to count gain several times, per-seed reporting and the ledger update. Apart from the gaps in W4, a reader can apply it as written.
**Evidence Anchor**: text: Appendix A §5.4 item 6 "pooled over the three seeds (one cluster per painting across seeds)" and "do not count it several times"

### S5: Claims that cite figures are restated in numbers
The figures do not render from this folder. Appendix A §4's diagnosis of the two failures still gives the pick shares in the text, and they match Appendix C's pick tables (70.5%, 15.8%, 55.0%), so nothing in the plan's motivation depends on an image.
**Evidence Anchor**: text: Appendix A §4 "picked in 70.5% of emotion × genre genre conditions (image, correct, 15.8%)"

### S6: Each candidate is tied to the failure it targets and to its cost
The candidate table orders the readers cheapest first and says which diagnosed failure each one addresses, which lets the reader of the rule see why each candidate exists and what a kill would give up.
**Evidence Anchor**: table: Appendix A §5.2 candidate table, Targets and Cost columns (R-a failure 1, minutes; R-b failures 1 and 2, hours; R-c remaining wrong picks, minutes)

### S7: Earlier design errors are carried forward as reasons for the process
§7 lists the pitfalls already paid for and ties the order of work (review before code) to two earlier rule errors. That is the right instinct for a pre-registration and explains the document's existence to a stranger.
**Evidence Anchor**: text: Appendix A §7 "before running, which is why the ARS review comes first"

### W1: The kill rule (§5.4 item 5) has two readings, and neither fits the rest of the rule
**Problem**: "If no candidate raises pick accuracy by at least 10 points ... or reaches the bar" can mean (a) kill only when no candidate does either, or (b) kill when no candidate gains 10 points, or when none reaches the bar. Under (a), a candidate that gains 10 points but misses the bar is not killed, yet item 4 has nothing to carry, so the pick-accuracy clause has no effect. Under (b), a candidate that clears the bar without a 10-point pick gain is killed, while §5.5 Thursday says "if a candidate clears the bar, build seeds 49 to 51". In Appendix D this clause stopped a development step; here it gates the test, a job the bar already does. It also makes pick accuracy, which §5.3 lists as a diagnostic, decide a kill, and "pick" is undefined for R-c (whose picks are its base reader's) and for R-b scored by the expected term.
**Evidence Anchor**: text: Appendix A §5.4 item 5 "raises pick accuracy by at least 10 points over its configuration's arg-max reader" and "or reaches the bar, no test is built"
**Why it matters**: Both outcomes are plausible. In the Leiden sweep the reader margin reached +0.50 while pick accuracy stayed at 54.1 to 55.9% (Appendix B §5). Two implementers would build or not build the test on the same numbers.
**Suggestion**: Replace item 5 with one of these and align §5.5. (i) "Kill: if no candidate clears the development bar (item 3), no test is built; report to the user. Pick accuracy is reported as a diagnostic and decides nothing." (ii) If pick accuracy must gate, move it into eligibility: "A candidate can be carried only if it clears the bar and its pick accuracy is at least 10 points above its configuration's arg-max reader (A1 43.2%, A0 54.7%). R-c inherits its base reader's pick accuracy; R-b expected is scored by the arg-max of P(h)."
**Severity**: Major
**Confidence**: 5 (core competence: reading decision rules and protocols)

### W2: B′ is defined in two incompatible ways, and §5 never defines the fused score
**Problem**: §5.4 item 2 says B′ is B plus the configuration's averaged-heads term. Memo §3 says B′ is B rebuilt with T_6u averaged over the configuration's own groupings, "not B plus a term", and Appendix A §4 and Appendix B §2 agree with the memo. Appendix D Step 4, the template the plan adapts, says "B' = B plus any new condition-free ingredient" and fuses the reader on B′. §5 does not say what the reader is fused on (memo §3: `crossfit_nested(B, B, T, parity)`, on B). Item 3's "the reader's gain over its counterpart" already needed an interpretation note in step 1 (Appendix C, "Choices not fixed by PLAN.md").
**Evidence Anchor**: text: Appendix A §5.4 item 2 "B′ = B plus the configuration's averaged-heads term" against memo §3 "so B′ is B rebuilt, not B plus a term"
**Why it matters**: B′ enters the bar margin (item 3) and the GO (item 6). A rule file copied from item 2 states the wrong comparator, and an implementer following Appendix D would fuse on B′. For A1, B′ sat +0.46 above B on seed 42, so the choice of definition is of the same size as the bar.
**Suggestion**: Open `DECISION_RULE.md` with a definitions block copied from memo §3 and point every item at it. Fused reader: `crossfit_nested(B, B, T, parity)`, 56 cells, the cell maximising min(R@1 − R@1 of B, gain) on one parity half applied to the other. Matched counterpart: `crossfit_condition_free(B, B, T_cf, parity)`, T_cf = (T under a + T under b) / 2, same cells, max-R@1 rule. B′: `crossfit_condition_free(cos, T_N1u, T_6u, parity)` with T_6u averaged over the configuration's own groupings (B rebuilt; A0 18.44, A1 18.80 on seed 42). Gain clause: gain of the fused reader minus gain of the fused counterpart (the latter 0 by construction). Bar-margin comparator: whichever of B′ and the counterpart has the larger mean R@1, ties to B′. Then add one line: "Where this file differs from §5.2, §5.5, the memo or Appendices B to F, this file governs."
**Severity**: Major
**Confidence**: 4 (the conflict is textual and certain; whether the two definitions give different numbers is a methodology question)

### W3: R-b's domain-shift kill sits outside the rule, with no numbers and no place in the order
**Problem**: §5.2 kills R-b if its held-out bank accuracy "is high" and its seed-42 pick accuracy "does not move". Neither term has a threshold, the check is absent from §5.4, and the text does not say whether it runs before the bar, overrides a bar pass, or what happens to R-c when R-c was built on R-b.
**Evidence Anchor**: text: Appendix A §5.2 R-b requirements "first is high and the second does not move, R-b is killed"
**Why it matters**: R-b is the candidate built for failure 2, the failure the CSD grouping introduced. If R-b clears the bar with high bank accuracy and a few points of pick gain, one reader keeps it and another kills it, which changes the carried configuration or whether a test runs.
**Suggestion**: Move the check into §5.4 as its own item, with numbers, for example: "R-b domain-shift kill: if R-b's accuracy on held-out bank episodes is at least X% and its seed-42 pick accuracy is less than Y points above its configuration's arg-max reader, both R-b scorings and any R-c built on R-b leave the candidate list before item 4; R-c is then built on the best remaining candidate." State X and Y (Y could reuse W1's 10 points if a pick clause survives) and state that the kill applies whatever the bar margin.
**Severity**: Major
**Confidence**: 4 (core competence for the wording; the right values of X and Y are for the methodology seat)

### W4: No single outcome table, and several outcomes have no written action
**Problem**: The actions are spread over §5.2, §5.4 items 3 to 6 and §5.5, and some outcomes have none. A test NO-GO: Friday says only that the user decides, and options are listed only for "nothing cleared the bar". A configuration that passes some GO comparators but not others: implied NO-GO by "each of", not stated. What is computed on test seeds before the verdict: Appendix D said never to compute pick accuracy there before the verdict, and §5 does not restate it. Per-seed and per-pair test results: reported, but not said to decide nothing. §5 also never says that items 1 to 5 are development selection and item 6 is the one confirmatory test.
**Evidence Anchor**: absence: Appendix A §5.4 — expected one outcome-to-action table naming the action for every development and test outcome, including a test NO-GO, a partial pass, R-b killed after clearing the bar, and what is computed on test seeds before the verdict; checked memo §0 to §4, Appendix A §5.1 to §5.6, §6 and §7
**Why it matters**: The rule's users are the Friday decision maker and the implementing agents. A table makes the mapping checkable at a glance, and an explicit test-seed order stops label-reading diagnostics from being computed before the verdict by default.
**Suggestion**: Add a table to `DECISION_RULE.md`: (1) no candidate eligible: no test, Friday chooses between named options; (2) one eligible: carry it; (3) several within 0.05: tie-break (W5); (4) R-b killed by the domain-shift check: per W3; (5) test, all six interval checks pass (R@1 against cosine, RCA, B′ and the counterpart; gain above 0; gain against RCA): GO; (6) any check fails: NO-GO, with the options named; (7) on test seeds, compute only the GO quantities until the verdict is written down, then per-seed, per-pair and diagnostic numbers as description. Add one status line: "Items 1 to 5 are development selection on seed 42; item 6 is the only confirmatory test."
**Severity**: Minor
**Confidence**: 4 (core competence: decision documents)

### W5: The tie-break is a partial order
**Problem**: Item 4's order does not rank R-a on A1 against R-b on A0 (its first and third criteria disagree), does not rank R-b arg-max against R-b expected, and does not say whether "within 0.05" is measured from the largest bar margin or pairwise, which allows chains.
**Evidence Anchor**: text: Appendix A §5.4 item 4 "simpler (R-a before R-b, no gate before gate, A0 before A1)"
**Why it matters**: With seven configurations and development intervals about ±0.25 wide, near-ties are plausible, and the tie-break chooses the one configuration that gets the only test.
**Suggestion**: "Every candidate whose bar margin is within 0.05 of the largest is tied. Among tied candidates apply in order, and the first criterion that separates them decides: (1) no gate before gate; (2) R-a before R-b; (3) R-b arg-max before R-b expected; (4) A0 before A1, or the reverse, per W6."
**Severity**: Minor
**Confidence**: 4 (core competence: rule wording)

### W6: A0 is both reference and candidate, and its stated purpose has no measure
**Problem**: §5.1 runs every reader on A0 to show whether the CSD grouping helps, but §5.3 lists paired differences only against each configuration's own arg-max reader, not A1 against A0 under the same reader, and §5.4 does not say how that comparison is read. Item 1 makes A0 configurations candidates and item 4 prefers A0 on ties, so the one tested configuration may lack the CSD grouping, while the title and memo §0 frame the work as a reader fix with CSD in the set.
**Evidence Anchor**: text: Appendix A §5.1 "to show whether the CSD grouping helps once the reader works"
**Why it matters**: If an A0 configuration is carried, the Friday GO says nothing about CSD, and any later statement that CSD helps would rest on a comparison that was never written down before the numbers.
**Suggestion**: Add to §5.3: "Per reader, A1 minus A0 under the same reader (paired; R@1 and bar margin), descriptive, seed 42." Add to §5.4 one sentence on what each outcome means for CSD, for example: "If an A0 configuration is carried, a GO supports the reader fix without the CSD grouping, and the CSD question stays open." Say in item 1 whether A0 is a candidate or only a reference.
**Severity**: Minor
**Confidence**: 3 (depends on the author's intent for A0)

### W7: R-c cannot be built from the text
**Problem**: R-c's row chooses λ and the threshold by "A′'s min-margin cross-fit", and "A′" is defined nowhere in the document (Appendices D and E use it without a definition). The gate's form (hard threshold or a ramp into [0, 1]), the margin it reads for each base reader (scaled Δ for R-a, probabilities for R-b), the λ and threshold grids and the counterpart's formula are not stated. R-c's score z(B) + λ·g·z(T) also differs from the (1 + λ_u)·z(B) + λ_a·z(T) form in memo §3 without comment.
**Evidence Anchor**: text: Appendix A §5.2 R-c row "λ and threshold by A′'s min-margin cross-fit; the counterpart applies the same gate to T_cf"
**Why it matters**: R-c can be carried. Whatever the text leaves open will be settled on seed 42 during implementation, after the rule is committed.
**Suggestion**: Replace "A′'s min-margin cross-fit" with the rule itself (the cell maximising min(R@1 − R@1 of B, gain) on one parity half, applied to the other) and state in the R-c row the gate form, the margin per base reader, both grids and the counterpart formula. Whether that counterpart removes only the condition is outside this card.
**Severity**: Minor
**Confidence**: 3 (wording is core; the gate design is adjacent methodology)

### W8: Internal codes and colliding names in the text a fresh reader needs
**Problem**: The memo and §5 use codes defined nowhere in the document or only deep in appendices: C2, E2 and its "AIC bank", N4, N6, scan T6. Names collide. "A3" is a step-1 arm in Appendices B and C and also the method-A checkpoint whose term T_N1u sits inside B. R1 to R3 (step-1 readings), R0 (an arm), R-a to R-c (readers) and R@1 (the metric) share a prefix. Conditions a and b in the plan meet A and B in Appendix E, where A and B also name the aspects. Folder and report names carry sequence dates (20261117, "2026-11-12") on a document dated 2026-10-06, and three rule-file paths appear (20261110 in Appendix D, 20261113 and 20261117 in §5.4).
**Evidence Anchor**: text: memo §3 "with T_N1u the centered factor term of A3" and Appendix A §4 table header "B = C2 with R@1 18.34"
**Why it matters**: The project's own readers will mostly not stumble, but `DECISION_RULE.md` will be read later, and a reader meeting "A3" or "R1" can misidentify the object. The sequence-date convention is never explained.
**Suggestion**: In the rule file, give a one-line glossary of the codes it keeps (B, B′, T_cf, T_N1u, T_6u, A0, A1, R-a to R-c); drop or gloss C2, E2, N4, N6 and T6; write "the method-A checkpoint" instead of "A3" in B's definition; add "Folder and report dates are sequence numbers, not calendar dates."
**Severity**: Minor
**Confidence**: 3 (weighed by whether the rule file or a fresh implementer would misread it)

### W9: The timeline omits process gates the plan requires and has no slip rule
**Problem**: Appendix A §6 requires the controller to re-derive load-bearing numbers and a whole-branch final review before the work is called done. §5.5 schedules re-derivation only for Wednesday's development numbers, with no slot for re-deriving the test numbers or for the final review before Friday's decision. Tuesday says "commit the rule" while §6 says to commit only when the user asks. Nothing says what happens if R-b or its domain-shift check is not finished by Wednesday night.
**Evidence Anchor**: absence: Appendix A §5.5 timeline — expected scheduled slots for the user's go-ahead to commit the rule, the re-derivation of the test numbers and the whole-branch final review, plus a rule for a slipped day; checked §5.5, §6 process and git bullets, §7 and memo §0
**Why it matters**: The project's record shows the final review catching a false pass (N1, Appendix F.1). If Thursday's test goes to Friday's decision without it, the decision rests on numbers nobody has re-derived.
**Suggestion**: Add to §5.5: Tue "user approves the rule commit"; Thu evening "controller re-derives the GO quantities"; Fri morning "whole-branch final review of the test, then the user decides". Add a slip rule, for example: "If R-b is not evaluated by Wed 23:00, the rule is applied to the candidates finished by then and R-b is not added later" (or the reverse, as the user prefers).
**Severity**: Minor
**Confidence**: 4 (core competence: decision documents; timing realism itself is not judged here)

### W10: The memo's summary of the bars drifts from §5.4 and leaves out the plan's own prior
**Problem**: Memo §1 states the development bar without item 3's gain clause, lists B among the comparators although the GO uses B′, and calls RCA "the GO bar" although the GO has four comparators. Appendix B §2 states the bar as a reader margin and the GO against B. Appendix B §10's estimate that the chance is "moderate at best" appears nowhere in the memo or Appendix A.
**Evidence Anchor**: text: memo §1 "bar margin ≥ +0.5 with a 95% lower bound above 0" and "RCA (13.38, the GO bar)"
**Why it matters**: The memo is what the decision maker reads. A summary that differs from the rule invites checking the wrong thing, and the prior tells the Friday reader how to weigh a NO-GO.
**Suggestion**: In memo §1 replace the bar line with "Development bar and GO: exactly as Appendix A §5.4 items 3 and 6." Rename RCA's label to "the strongest raw pair metric". Add to §0: "Our own estimate of the chance of a GO is moderate at best (Appendix B §10): fresh-seed tests have so far roughly halved development margins."
**Severity**: Minor
**Confidence**: 4 (checked against §5.4 and the appendices)

### Detailed Comments

#### Journal Fit (fitness for purpose)
- No venue is bound (`criteria_binding_unavailable`), and this card makes no venue-fit claim. Judged as a decision document, the memo serves the Friday decision maker well, and §5 serves implementers well for R-a and R-b and less well for R-c (W7). §5.4 is not yet a rule file (W1 to W3).
- The object of review is about 950 of about 19,100 words. That is acceptable for a memo with evidence appended, provided memo plus §5 is self-sufficient for applying the rule; with W2's definitions block and W4's table it would be.

#### Originality
- Not assessed. Novelty of a learned grouping reader against the conditional-similarity and few-shot literature is outside this reduced panel's scope.

#### Significance
- The decision is consequential: it decides whether plan (a) goes forward toward the CVPR abstract or the project turns to design L. That raises the bar for an unambiguous rule.

#### Structural Coherence
- The order (memo, Appendix A §5, evidence) is sound, and the memo's questions match §5.6. The weak point is that the rule is split across §5.2, §5.4, §5.5 and memo §3, with no statement of which text governs (W2 to W4).

#### Title & Abstract
- The title says "pre-results plan for review" and the status line says nothing has run, which is honest. §0 works as the abstract but omits the plan's own prior (W10).

#### Conclusion
- There is no conclusion by design. Its equivalent is the Friday row of §5.5, which names an action only for "nothing cleared the bar" (W4).

### Questions for Authors
1. Item 5: which reading is intended, and should pick accuracy decide anything at all?
2. R-b: what numbers define "high" and "does not move", and does that check apply whatever the bar margin?
3. Is A0 a candidate or only a reference? If an A0 configuration is carried, what does a GO mean for the CSD grouping?
4. After a test NO-GO, which options will the user choose between on Friday?
5. On test seeds, which quantities are computed before the verdict is written down?
6. Is R-c's gate hard or soft, and which margin does it read for each base reader?

### Minor Issues
- None listed separately. Text-level issues are covered as W8 and W10.
