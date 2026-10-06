# Editorial Decision

## Calibration Resolution

`calibration_status: NOT_CALIBRATED`

Current runtime boundary: this package is not upgraded from a candidate or prose-named profile. `PROFILE_MEASURED` stays unavailable until a closed profile artifact and replay validator bind the exact target fields to a completed panel's execution topology.

## Manuscript Information

- **Title**: Fixing the aspect reader with a CSD style grouping in the set: a pre-results plan for review (CoSiR v2, plan (a)), with the 2026-10-06 handoff, the grouping report, the step-1 log, the 2026-10-04 reader handoff and stage-report sections 2 and 14 appended
- **Manuscript ID**: none assigned (internal pre-results plan; review folder `src/test/20261117_reader_fix_csd/ars_review/`)
- **Submission Date**: 2026-10-06 (plan date)
- **Decision Date**: 2026-10-06 01:13
- **Review Round**: 1
- **Mode and contract**: `reviewer_methodology_focus`, contract `reviewer/reviewer_methodology_focus/v2`, baseline v3.20.0, `panel_size` 2, contract generated 2026-10-06 00:34.
- **Object of review**: Appendix A §5 (5.1 to 5.6), including the draft decision rule §5.4 that the author will rewrite and commit as `src/test/20261117_reader_fix_csd/DECISION_RULE.md`. The cover memo §0 to §4 frames six questions; Appendices B to F are evidence and were not re-reviewed (Phase 0 strategy).
- **Contract stage note**: "Missing results are not a block; flaws that leave a candidate uninterpretable or weaken the test are."
- **Target binding**: `criteria_binding_unavailable`, disclosed in both Phase 1 cards. No venue criteria were applied and this letter makes no venue-fit claim. CVPR is author-stated context only.

## Review Panel Composition and Provenance (plain text)

This mode is not `reviewer_full`, so the typed `review-panel-provenance/1.0` artifact and its six-axis block do not apply and none was built. The panel is described here in plain text.

- **Seats.** Two scoring seats: the Journal-Fit Reviewer (`contract_role: eic`, owner of D2 `writing_and_structure`, priority normal) and Peer Reviewer 1 (`contract_role: methodology`, owner of D1 `methodology_rigor`, mandatory). This mode has no domain seat, no perspective seat and no Devil's Advocate seat.
- **Execution.** Both seats ran on the same model family (Claude, provider Anthropic), each in its own fresh subagent context. Each Phase 1 was paper-content-blind (contract and metadata only). Neither seat saw the other's output before committing its card. No human reviewer sat on the panel. This synthesis was also written by a Claude model.
- **Conformance.** Per the orchestrator, both Phase 2 cards passed `check_phase_conformance.py` and `check_panel_synthesis.py --layer1-only` before synthesis. Usable cards: 2 of `panel_size` 2, so no `[PANEL-SHRUNK]`.
- **Correlated-error caveat.** Role separation is not independence. Two seats from one model family can share blind spots and make the same mistake, so agreement between them is weaker evidence than agreement between independent reviewers, and an error both seats share would not show up as a disagreement here. No binary or numeric independence claim is made. Context separation holds within this panel attempt only.
- **Evidence access.** Both seats worked from the text. The methodology seat states that it "could not read the stored arrays or the code" and checked arithmetic from the text (Meth S5, AR1 to AR7). The memo preamble's claim that controller reviews re-derived every quoted number from stored arrays was treated as an unverified author statement (Phase 0, item 4).
- **Manuscript text aimed at the panel.** None asks for leniency or a verdict (Phase 0 report). The seed-handling preference (Appendix F.2) could steer attention away from multiplicity; the methodology seat addressed multiplicity as the memo asks (Meth S1, W9, answer to Q3) and kept its proposals within that preference. Agent-directed text in Appendices A and D was read as evidence of the planned workflow.

Seat labels used below: "EIC" is the Journal-Fit Reviewer; "Meth" is Peer Reviewer 1 (methodology; seat `R1` in the roadmap schema's seat enum). Transport refs R1 to R5 and S1 to S17 are roadmap positions, not seats and not a work rank. They are unrelated to the step-1 readings R1 to R3 and to the readers R-a to R-c.

---

## Decision

### Major Revision

The draft rule in §5.4 must be rewritten before it is committed and before any code. Nothing fatal was found: the methodology seat considered its fatal trigger and found it not met. Both seats judge every change to be a rewrite of §5 and §5.4. The methodology seat states that each fix "fits Tuesday 6 October and needs no new data, seeds or runs beyond those the plan already schedules"; the Journal-Fit seat estimates "a few hours" for its items.

---

## Mechanical Synthesis (sprint contract, three steps)

**Step 1: role-scoped scoring matrix.** Only assessed scores from eligible seats count. Ineligible `not_assessed` values are excluded from numerator and denominator.

| Dimension | Priority | Eligible roles | eic seat | methodology seat | Assessed eligible seats | Verdict |
|---|---|---|---|---|---|---|
| D1 methodology_rigor | mandatory | methodology | not_assessed (ineligible, excluded) | block, `block_class: repairable` | 1 | block |
| D2 writing_and_structure | normal | eic | block | not_assessed (ineligible, excluded) | 1 | block |

Both dimensions have an assessed eligible seat, so there is no `[DIMENSION-UNASSESSED]`. Criterion judgements on the sprint scale (`judgement_scale: sprint_contract`, values copied unchanged): D1 = block, repairable (methodology seat); D2 = block (eic seat). The seats' narrative criterion rows (for example EIC's D2 `DOES_NOT_MEET`, Meth's "Kill and carry rules" `DOES_NOT_MEET`) are seat-internal and were not translated into the sprint scale.

**Step 2: failure conditions.** Every expression parsed against the recognised vocabulary (§9). The cross-reviewer quantifier is applied per dimension over its assessed eligible seats (n = 1 for each dimension here).

| Condition | Severity | Quantifier | Expression (pattern) | Evaluation | Fired | Action |
|---|---|---|---|---|---|---|
| F1 | 95 | any | `D1 has a fatal block` (6) | D1: 0 of 1 seats fatal (block is repairable) | no | reject |
| F2 | 90 | any | `D1 scores 'block'` (4) | D1: 1 of 1 seats score block | yes | major_revision |
| F3 | 70 | any | `D1 scores 'warn'` (4, exact match) | D1: 0 of 1 seats score exactly warn (the score is block) | no | major_revision |
| F4 | 40 | any | `D2 scores 'warn' or worse` (8) | D2: 1 of 1 seats at warn or worse (block) | yes | minor_revision |
| F0 | 10 | all | `every dimension scores 'pass'` (9) | D1: 0 of 1 pass; D2: 0 of 1 pass | no | accept |

**Step 3: precedence.** Fired: F2 (severity 90) and F4 (severity 40). F2 has the highest severity, so its action applies. There is no DA seat, so there are no DA CRITICAL IDs to adjudicate; the decision is not accept, so no DA-versus-accept marker applies. The fired action is not softened.

dimension_verdicts: [D1=block, D2=block]

fired_conditions: [F2, F4]

da_critical_adjudications: []

editorial_decision=major_revision

`ARS_CROSS_MODEL` is not set, so no cross-model blind decision check was run.

---

## Blocking Issues (3, immutable source order)

The methodology seat's D1 block is "carried by W1", with W2 falling "under the kill-rule clause of the same Phase 1 trigger". The Journal-Fit seat's D2 block rests on its W1, W2 and W3 and on the missing precedence rule ("Fixing W1 to W3 and adding a precedence line would bring D2 to warn"). These are grouped by the rule element they affect.

| Transport ref | Blocking issue | Source reviewer(s) | Evidence anchor | Resolving roadmap item |
|---|---|---|---|---|
| R1, R4 | The kill rules are not mechanical: item 5 reads two ways and leaves an outcome with no action; R-b's domain-shift kill has no numbers and no precedence over the bar; both read the told mapping | EIC W1, W3; Meth W2 | EIC W1: text: Appendix A §5.4 item 5 "raises pick accuracy by at least 10 points over its configuration's arg-max reader" and "or reaches the bar, no test is built". EIC W3: text: Appendix A §5.2 R-b requirements "first is high and the second does not move, R-b is killed" | REV-1, REV-5 |
| R2, R3 | B′ has two incompatible definitions, §5 never states what the reader is fused on, and no text is named as binding | EIC W2; Meth W5 | text: Appendix A §5.4 item 2 "B′ = B plus the configuration's averaged-heads term" against memo §3 "so B′ is B rebuilt, not B plus a term" | REV-3, REV-4 |
| R5 | R-c, which the rule can carry, has a declared counterpart that is not condition-free, and its gate is unspecified, so its control would be improvised after the other numbers exist | Meth W1; EIC W7 | Meth W1: text: Appendix A §5.2 R-c row "the counterpart applies the same gate to T_cf". EIC W7: text: Appendix A §5.2 R-c row "λ and threshold by A′'s min-margin cross-fit; the counterpart applies the same gate to T_cf" | REV-10 |

---

## Reviewer Summary

| Seat | Role | Dimension scored | Recommendation (seat signal) | Confidence | Findings |
|---|---|---|---|---|---|
| Journal-Fit Reviewer (`eic`) | Research lead in vision-language representation learning and few-shot retrieval who chairs go/no-go reviews of registered-report style protocols (Phase 0 Card #1) | D2 block | Minor Revision, "seat signal on D2 only" | 4 report level; 3 to 5 per finding | 7 strengths; 10 weaknesses (3 Major, 7 Minor); 6 questions |
| Peer Reviewer 1 (`methodology`) | Statistician in ML evaluation methodology: selective inference, cross-fitting, cluster bootstrap, meta-learning on pseudo-tasks, matched controls (Phase 0 Card #2) | D1 block, repairable | Major Revision on the template scale, "the methodology seat's view only" | 4 report level; 3 to 4 per finding | 5 strengths; 11 weaknesses (1 Critical, 3 Major, 7 Minor); 4 questions; 2 minor issues; 7 arithmetic receipts, all consistent |
| Peer Reviewer 2 (domain), Peer Reviewer 3 (perspective), Devil's Advocate | Not in this mode's panel | n/a | n/a | n/a | n/a |

The two seat signals differ in scale, not in substance: each speaks for its own dimension, and under the contract D2 alone can only reach minor revision (F4). The decision is the mechanical result above. Confidence values are self-reported and were not used to weigh anything.

---

## Rewrite Map: where each item lands

The author will rewrite §5 and §5.4 once and commit the result as `DECISION_RULE.md`. This map regroups every roadmap item by the part of that file (or the memo) it changes. It is a presentation aid only; the roadmap below keeps immutable source order.

| Part of the rewrite | Must fix | Should fix |
|---|---|---|
| Head of the file: status line, precedence, glossary | R3 | S2 (status line), S6 |
| Definitions block (new, at the head) | R2 | S11 (B in the bar comparator), S16 (full-precision point) |
| §5.2 candidates | R4 (R-b check), R5 (R-c) | S9 (R-a spread), S10 (R-b specification) |
| §5.3 measures | none | S1 (pick for R-c and R-b expected), S5 (A1 minus A0, paired), S16 (reference pick accuracies) |
| §5.4 items 1 to 5 (development selection) | R1 (kill), R4 | S4 (tie order), S5 (A1 priority), S14 (R-c base; item 3 clauses), S17 (basis of the halvings) |
| §5.4 item 6 (the test) | R1 (test-seed computation order) | S3, S11 (B in the GO list), S12, S13, S15 |
| Outcome-to-action table (new) | none | S2 |
| §5.5 timeline | R1 (Thursday row aligned) | S7, S12 (Thursday variance split) |
| Memo §0 and §1 | none | S8 |

**What a re-review will look for.** The must-fix group carries both blocks. For D2 to reach pass, the Journal-Fit seat also needs the outcome table (S2): "adding the outcome table (W4) as well would bring it to pass". For D1, the methodology seat bands its W3 to W11 "warn-level or below" without saying which would hold D1 at warn on its own; under F3 a D1 warn alone still returns major_revision. Two of those carry Major severity: S9 (R-a's spread, Meth W3) and S10 (R-b's frozen specification, Meth W4). Every should-fix item the methodology seat proposes is costed in minutes or as text, and none asks for extra seeds, painting-level splits or a reopened fixed item.

---

## The Panel's Answers to the Six Questions (memo §0)

1. **Matched counterparts and B′.** R-a and both R-b variants have counterparts that remove only the condition (Meth S2). R-c's does not, as written (Meth W1; R5). B′ needs one definition (EIC W2, Meth W5; R2), and the methodology seat adds B to the comparators (Meth W5; S11).
2. **Leakage in R-b.** No evaluation row or label can reach R-b, and cross-fitted heads keep training posteriors out of sample (Meth S3). The remaining risks are a head-sharpness mismatch between training and evaluation and unrecorded tuning on seed 42 (Meth W4; S10). The Journal-Fit seat did not assess this.
3. **Selection on a reused seed 42.** The methodology seat finds the test keeps its level: one configuration is carried, the test draws new episodes, and the GO is an intersection-union test (Meth S1). It estimates that the maximum of seven configurations inflates the development margin by roughly 0.1 to 0.15 R@1, within the halving the +0.5 bar allows (Meth W9). What the test cannot remove under the fixed seed policy is carry-over through shared paintings (Meth W7; S13), and its sensitivity is unstated (Meth W6; S12). The Journal-Fit seat asks only that the tie order be total (EIC W5; S4).
4. **R-a's spread.** Development episodes are an acceptable source; the root-mean-square estimator is the problem. Use the pooled within-episode standard error, frozen from seed 42, not bank episodes (Meth W3; S9).
5. **Thresholds and label policy.** Both seats find the kill rules incoherent with item 4 and reading the told mapping (EIC W1, W3; Meth W2; R1, R4). The label policy limits what the paper may claim about the grouping choice (Meth W7; S13).
6. **Timeline.** The methodology seat finds it feasible if R-b does not slip; R-b slips first, then R-c (Meth W11). Both seats ask for a cutoff and for slots for the test re-derivation and the whole-branch final review before Friday's decision (EIC W9, Meth W11; S7). The Journal-Fit seat does not judge timing realism.

---

## Consensus Analysis

**Counting in a two-seat panel.** CONSENSUS-4 and CONSENSUS-3 are defined over four non-DA reviewers and cannot occur here. The labels below are the role file's lower-count labels over this panel's two seats: a sub-claim both seats raised with no conflict is a *corroborated finding (2 of 2 seats)*; one seat's sub-claim is a *single-seat finding*. Silence is neither agreement nor dissent. The seats score disjoint dimensions, and Phase 0 routed rule-changing ambiguities to the methodology seat and reader-facing ones to the Journal-Fit seat, so overlap is expected only where a defect is both. The full sub-claim inventory is in the appendix.

### Points of Agreement

**Corroborated findings (2 of 2 seats, no conflict).**
1. **SC-1, SC-2. Item 5's kill rule.** EIC W1 (Major): the clause "can mean (a) kill only when no candidate does either, or (b) kill when no candidate gains 10 points, or when none reaches the bar"; under (a) the pick clause has no effect, under (b) a candidate that clears the bar is killed while §5.5 Thursday builds the test. Meth W2(a) (Major): "A candidate that gains 10 points of pick accuracy but misses the bar is neither killed (item 5) nor carried (item 4)." Both also find that pick accuracy, a diagnostic under the label policy, decides a kill (EIC W1; Meth W2(c)). Same severity; the Journal-Fit seat's first option (i) and the methodology seat's recommended option A are the same rule. Roadmap R1.
2. **SC-14. Per-seed and per-pair results.** EIC W4 and Meth W9 (both Minor): the rule does not say that per-seed results decide nothing. Same fix. S3.
3. **SC-16. The tie-break is a partial order.** EIC W5 and Meth W9 (both Minor). The two cards write the same order (no gate before gate, R-a before R-b, R-b arg-max before R-b expected), measured from the largest bar margin. S4.
4. **SC-17, SC-18. A0 is both reference and candidate, and its stated purpose has no measure.** EIC W6 and Meth W8 (both Minor). Both ask for a paired A1 minus A0 comparison in §5.3. The Journal-Fit seat asks the author to say whether A0 is a candidate; the methodology seat recommends an A1 priority over making A0 reference-only. Compatible: the methodology seat's option answers the Journal-Fit seat's question. S5.
5. **SC-25, SC-27. Process gates and a cutoff are missing from the timeline.** EIC W9 and Meth W11 (both Minor). Both ask for a Thursday re-derivation of the test numbers, a Friday-morning whole-branch final review before the decision, and a cutoff for unfinished candidates. The cutoff time differs only as an example: EIC W9 gives "for example ... Wed 23:00 ... (or the reverse, as the user prefers)", Meth W11 proposes Thu 8 Oct 12:00. Both require the rule to state one time; the user chooses it. S7.

**Corroborated sub-claims with a severity or remedy conflict** (SPLIT; arbitrated under Points of Disagreement): SC-4, SC-5, SC-7 (B′, fusion base, gain statistic; R2); SC-13 (test-seed computation order; R1); SC-20 (R-c's unspecified gate; R5); SC-8, SC-9 (R-b's domain-shift kill; R4).

**Corroborated strengths (2 of 2 seats).**
- The confirmatory test is concrete and well built: EIC S4 (item 6 names the seeds, command, hash check, pooling unit, four comparators and the single gain count); Meth S1 (one carried configuration, an intersection-union GO, clustering by painting across seeds).
- The rule is reviewed and committed before the numbers it governs, and earlier errors are carried forward as reasons for that order: EIC S7; Meth S4.

**Single-seat findings, Journal-Fit seat** (all D2; none disputed by the methodology seat).
- SC-3 (EIC W1, Major): "pick" is undefined for R-c and for R-b scored by the expected term. S1.
- SC-6 (EIC W2, Major): no precedence rule names a binding version among §5.4, §5.2, §5.5, memo §1 and §3 and Appendix D. R3.
- SC-10, SC-11, SC-12, SC-15 (EIC W4, Minor): no outcome table; a test NO-GO has no named options; a partial GO pass is only implied to be a NO-GO; no line says items 1 to 5 are development selection and item 6 the one confirmatory test. S2.
- SC-19, SC-21 (EIC W7, Minor): "A′" is defined nowhere; R-c's score form differs from memo §3 without comment. R5.
- SC-22, SC-23, SC-24 (EIC W8, Minor): undefined codes, colliding names, unexplained sequence dates and three rule-file paths. S6.
- SC-26 (EIC W9, Minor): Tuesday "commit the rule" meets §6 "commit only when the user asks". S7.
- SC-28, SC-29 (EIC W10, Minor): memo §1 drifts from §5.4; the plan's own prior ("moderate at best") is absent. S8.
- Strengths: the memo quotes §5.6 exactly (S1); fixed and open items are fenced (S2); memo §3 makes most rule quantities computable (S3); figure-based claims are restated in numbers (S5); each candidate is tied to its failure and cost (S6).

**Single-seat findings, methodology seat** (all D1; none disputed by the Journal-Fit seat, which stated where it did not judge methodology).
- SC-30 (Meth W1, Critical): R-c's counterpart g_c·z(T_cf) is not condition-free, because the top-two margin differs between conditions, and `crossfit_condition_free` would raise an error on it. The Journal-Fit seat wrote that "whether that counterpart removes only the condition is outside this card". Carries the D1 block. R5.
- SC-31 (Meth W2, Major): the same 10 points mean different things on A1 (chance 25%, ceiling 100%) and A0 (chance 33%, ceiling 83.3%). Moot under option A; applies to the fallback in R1.
- SC-32 (Meth W3, Major): R-a's root mean square folds mean squared signal into the noise scale and partly re-creates failure 1. S9.
- SC-33 to SC-37 (Meth W4, Major): R-b is not a frozen configuration (head size and recipe, regularisation and bank size, the A0 bank, no ban on revising after seed-42 numbers) and the bank's label space limits what it can learn. S10.
- SC-38 (Meth W5, Minor): a rebuilt B′ can fall below B (A2s: 18.24 against 18.34), yet B is not a comparator. S11.
- SC-39 (Meth W6, Minor): the test's sensitivity is unstated, so a NO-GO with a positive point estimate has no reading. S12.
- SC-40, SC-41 (Meth W7, Minor): what a GO licenses, and the label-policy disclosure. S13.
- SC-42, SC-43 (Meth W9, Minor): whether R-c's base must clear the bar; item 3's two clauses never bind. S14.
- SC-44 (Meth W10, Minor): test-time cross-fits read test labels; intervals hold picks fixed. S15.
- SC-45 (Meth Minor Issues): the full-precision point; where the reference pick accuracies go. S16.
- SC-46 (Meth Q4): whether the two halvings behind the +0.5 bar were measured on R@1 margins against a matched control. S17.
- Strengths: the two-condition mean defines matched counterparts for R-a and both R-b variants (S2); no evaluation row reaches R-b (S3); the numbers the bar rests on hold together (S5).
- Arithmetic: the methodology seat's re-derivations (R@1 = (either + gain) / 2 across Appendix D §1 and Appendix E Table 10, the seven step-1 bar margins, 3,459 of 12,288 = 28.1%, F = 0.50) and its seven receipts (AR1 to AR7) found no mismatch. Receipts attest auditability, not correctness.

### Points of Disagreement

In this two-seat panel the Journal-Fit seat is itself a party to each split, so the synthesizer arbitrates under Step 3b (evidence first, then expertise). Where a card states its own deference, that statement is part of the evidence.

**Disagreement 1: R-b's domain-shift check (SC-8, SC-9)**
- **EIC W3 view** (Major): the check sits outside §5.4 with undefined terms ("high", "does not move") and no place in the order. Move it into §5.4 as a numbered kill: "if R-b's accuracy on held-out bank episodes is at least X% and its seed-42 pick accuracy is less than Y points above its configuration's arg-max reader, both R-b scorings and any R-c built on R-b leave the candidate list before item 4; R-c is then built on the best remaining candidate", applied "whatever the bar margin". The card adds: "the right values of X and Y are for the methodology seat".
- **Meth W2 view** (Major): the check reads the told mapping, which memo §2 calls a diagnostic only, and pick accuracy understates CSD's usability for genre (Appendix C: "pick accuracy understates how usable the grouping is for genre"), so "a reader that works through CSD can clear the bar without moving pick accuracy" and "a label-informed kill could then remove the one candidate that works". Recommended option A: R-b's held-out bank accuracy and pick accuracy are seed-42 diagnostics that "enter no rule"; replace the pick comparison by a label-free shift report, "reported, not a kill". Even its option B "never carries or kills a candidate the bar decides".
- **Disagreement type**: direction (incompatible remedies). Both seats agree the present text cannot be applied.
- **Editor's Resolution**: adopt Meth W2 option A. R-b survives or falls on the development bar like every other candidate. §5.4 states that R-b's held-out bank accuracy, its seed-42 pick accuracy and the label-free shift report are diagnostics that decide nothing. The kill sentence is removed from §5.2. Roadmap R4.
- **Resolution Rationale**: (1) Expertise: whether a label-reading statistic may gate a candidate is a D1 question, and the Journal-Fit card itself assigns the thresholds to the methodology seat. (2) Evidence: the methodology seat's argument is anchored in a fixed item (memo §2, "the told mapping is a diagnostic only") and in an Appendix C passage that the synthesizer confirmed in the manuscript. (3) The Journal-Fit seat's own concern is met: the check moves into §5.4, no undefined threshold remains, and its relation to the bar is stated. (4) Consistency: the Journal-Fit seat's own option (i) for item 5 (SC-1) says pick accuracy "decides nothing"; keeping a pick-based kill for R-b would bring the told mapping back into a rule. **Residual**: if the user nonetheless wants R-b to be killable, EIC W3's form is the minimum (numeric X and Y and the stated precedence, all in the commit before any seed-42 number), and the methodology seat's objection then stands as recorded dissent. This letter does not adopt that path.

**Disagreement 2: severity differences on shared sub-claims with the same fix (SC-4, SC-5, SC-7, SC-13, SC-20)**
- **SC-4, SC-5** (B′ defined twice; fusion base unstated): EIC W2 Major ("the choice of definition is of the same size as the bar", B′ − B = +0.46 for A1); Meth W5 Minor. Same fix: one definition, B rebuilt, fused on B.
- **SC-7** (item 3's gain clause needs an exact statistic): EIC W2 Major (it "already needed an interpretation note in step 1"); Meth Minor Issues (no per-finding tag, minor by section). Same fix: name the statistic as step 1 did.
- **SC-13** (what is computed on test seeds before the verdict): EIC W4 Minor; Meth W2 Major. Compatible fixes; the Journal-Fit wording is broader and contains the methodology seat's.
- **SC-20** (R-c's gate form, grids and counterpart formula unstated): EIC W7 Minor; Meth W1 Critical. The Journal-Fit seat scored only the textual gap and wrote that the counterpart's validity "is outside this card"; the Critical severity attaches to the consequence only the methodology seat assessed.
- **Disagreement type**: severity (perspective difference: text committability in D2 against methodological consequence in D1).
- **Editor's Resolution**: both severities are transported unchanged on each roadmap row. No action changes. Each of these sub-claims sits in a must-fix item (R2, R1, R5) because it belongs to a block-carrying finding of at least one seat.
- **Resolution Rationale**: the remedies agree in every case, and each card states its own scope, so the difference reflects which dimension each seat scores, not a dispute about the defect or the fix.

**Checked and found not to conflict.**
- Bar-margin comparator: EIC W2's definitions block transcribes memo §3 (B′ or the counterpart); Meth W5 proposes adding B. The Journal-Fit seat is silent on adding B, so this is a single-seat methodology item (S11), and R2's block should carry whichever comparator set the author adopts.
- Tie order (SC-16) and A0's role (SC-17, SC-18): compatible, as noted above.
- Cutoff time (SC-27): both cards leave the time to the user.
- Seat recommendations (Minor against Major): different scales for different dimensions, not a conflict.

### Not Assessed by This Panel

The reduced panel has no domain seat and no Devil's Advocate. These points were not scored, and nothing in this letter should be read as a judgement on them (Phase 0, coverage gaps; EIC "Originality: Not assessed"):
- Whether CSD and the CLIP heads can plausibly carry style and genre, and whether R-b's hand-built features suit the vision problem.
- Novelty of a learned grouping reader, and positioning against the conditional-similarity and few-shot retrieval literature (Conditional Similarity Networks, GeneCIS and others).
- An adversarial case for design L, or for stopping now, instead of plan (a).
- Venue fit (`criteria_binding_unavailable`).
- Data-level re-derivation from stored per-anchor arrays, posteriors, banks or code.

### Scope Limits from User-Fixed Items (memo §2)

No required or suggested item reopens a fixed item. Where a fixed item limits a claim, the cards record a scope note:
- **Seed policy.** Fresh seeds 49 to 51 re-draw episodes on the same 6,451 selection paintings (seed 42 already anchors 4,602). The methodology seat considered this under its fatal trigger and found the trigger not met; a GO licenses new episodes on known paintings, not transfer to new paintings (Meth W7; S13). S12's sensitivity projection uses seed-42 data only and adds no seed.
- **Groupings.** The CSD grouping was kept after its told margins were read; the paper should disclose this and may cite the label-free diagnostics that also rank CSD above Gram (Meth W7; S13). No grouping is re-chosen.
- **Label policy.** R1 and R4 bring the rule into line with "the told mapping is a diagnostic only".
- **Plan (a).** No seat argued for design L or for stopping now; this panel has no Devil's Advocate.

---

## Decision Rationale

The decision follows mechanically from the contract. The methodology seat, the only seat eligible for D1, scored methodology_rigor block with block_class repairable. Its block is carried by Meth W1: R-c, which items 1 and 4 can carry, has a declared counterpart (the same gate applied to T_cf) that changes with the condition, because the gate reads a top-two margin that differs between conditions. Its control would therefore be invented after the R-a and R-b numbers exist, the failure Appendix F.1 records. Meth W2 falls under the same trigger's kill-rule clause. The Journal-Fit seat, the only seat eligible for D2, scored writing_and_structure block: item 5 reads two ways (EIC W1), B′ has two incompatible definitions and §5 never states the fusion base (EIC W2), R-b's kill sits outside the rule with no numbers (EIC W3), and no text is named as binding. F2 (severity 90) and F4 (severity 40) fired; F2 takes precedence.

Why not reject: F1 needs a fatal D1 block. The methodology seat tested its fatal trigger against the shared selection paintings and found it not met, because the test draws new episodes and nothing selected on seed 42 is fitted per painting. Both seats credit the test architecture (EIC S4, Meth S1). Why not minor revision: the D1 block leaves a carry-eligible candidate uninterpretable, which the contract's stage note names as block-worthy, and a fired action may not be softened.

Here major revision means one rewrite of §5 and §5.4 before any code, not new experiments. The seats agree on the kill rule, B′, A0's role, the tie order, per-seed reporting and the timeline gates. They disagree on one remedy, R-b's domain-shift check, which is arbitrated in favour of the methodology seat's option A.

---

## Required Revisions (Must Fix)

These five items carry the D1 and D2 blocks. Severity, evidence anchor and confidence are transported unchanged from the cards. Consequence codes come from the roadmap schema's closed set.

| Transport ref | Revision Item | Sub-Claim(s) | Severity | Evidence Anchor | Confidence | Source Reviewer | Obligation class | Cost scope | Bounded consequence |
|---|---|---|---|---|---|---|---|---|---|
| R1 | Kill decided by the development bar alone; the told mapping kept out of every rule; test-seed computation order stated | SC-1, SC-2, SC-13, SC-31 | major (EIC W1); major (Meth W2); minor (EIC W4, SC-13 only) | text: Appendix A §5.4 item 5 "raises pick accuracy by at least 10 points over its configuration's arg-max reader" and "or reaches the bar, no test is built" | 5 (EIC W1); 4 (Meth W2); 4 (EIC W4) | EIC W1, W4; Meth W2(a), W2(c) | must_fix | sentence: §5.4 items 5 and 6; §5.5 Thursday row | interpretive_ambiguity_remains; section §5.4 item 5 |
| R2 | One definitions block: fusion base, matched counterpart, B′ rebuilt, gain statistic, bar-margin comparator | SC-4, SC-5, SC-7 | major (EIC W2); minor (Meth W5); minor [SEVERITY-SOURCE: letter-fallback] (Meth Minor Issues, SC-7 only) | text: Appendix A §5.4 item 2 "B′ = B plus the configuration's averaged-heads term" against memo §3 "so B′ is B rebuilt, not B plus a term" | 4 (EIC W2); 4 (Meth W5); 4 [CONFIDENCE-SOURCE: report-level] (Meth Minor Issues) | EIC W2; Meth W5, Minor Issues | must_fix | section: new definitions block; §5.4 items 2 and 3 | interpretive_ambiguity_remains; section §5.4 items 2 and 3 |
| R3 | `DECISION_RULE.md` names itself as the binding text and is self-contained | SC-6 | major (EIC W2) | text: Appendix A §5.4 item 2 "B′ = B plus the configuration's averaged-heads term" against memo §3 "so B′ is B rebuilt, not B plus a term" | 4 (EIC W2) | EIC W2 (Suggestion; Score rationale) | must_fix | sentence: head of `DECISION_RULE.md` | interpretive_ambiguity_remains; manuscript (Appendix A §5 and memo §0 to §4) |
| R4 | R-b's domain-shift check stated in §5.4 as a diagnostic that decides nothing, with a label-free shift report (arbitrated) | SC-8, SC-9 | major (EIC W3); major (Meth W2) | text: Appendix A §5.2 R-b requirements "first is high and the second does not move, R-b is killed" | 4 (EIC W3); 4 (Meth W2) | EIC W3; Meth W2(b) | must_fix | section: §5.2 R-b requirements; §5.4 | interpretive_ambiguity_remains; section §5.2 and §5.4 |
| R5 | R-c fully specified, with a condition-free counterpart | SC-19, SC-20, SC-21, SC-30 | critical (Meth W1); minor (EIC W7) | text: Appendix A §5.2 R-c row "λ and threshold by A′'s min-margin cross-fit; the counterpart applies the same gate to T_cf" | 4 (Meth W1); 3 (EIC W7) | Meth W1; EIC W7 | must_fix | section: §5.2 R-c row; §5.4 item 2 | interpretive_ambiguity_remains; claim: R-c's margin against its matched counterpart (bar and GO) |

### Required Item Details

**R1: Kill decided by the development bar alone; told mapping out of every rule; test-seed computation order**
- **Problem**: Item 5 reads two ways. Under one reading a candidate that gains 10 points of pick accuracy but misses the bar is neither killed nor carried; under the other a candidate that clears the bar is killed while §5.5 Thursday builds the test (EIC W1; Meth W2(a)). The Leiden sweep shows the second pattern is plausible: reader margin up to +0.50 with pick accuracy at 54.1 to 55.9% (Appendix B §5). Pick accuracy reads the told mapping, a diagnostic under memo §2 (EIC W1; Meth W2(c)). Appendix D's instruction never to compute pick accuracy on test seeds before the verdict is not restated (Meth W2(c); EIC W4).
- **Source**: EIC W1 ("Two implementers would build or not build the test on the same numbers"), EIC W4 item (7), EIC Q1 and Q5; Meth W2(a), W2(c), Q2.
- **Requirement**: Replace item 5 with the methodology seat's option A, which is the Journal-Fit seat's option (i): "5. Kill: if no candidate clears the development bar (item 3), no test is built and the result goes to the user for the Friday choice. Pick accuracy under the told mapping and R-b's held-out bank accuracy are seed-42 diagnostics, reported for every candidate, and enter no rule; pick accuracy is not computed on test seeds until the GO verdict is recorded." Widen the test-seed clause in item 6 to the Journal-Fit wording: "on test seeds, compute only the GO quantities until the verdict is written down", then per-seed, per-pair and diagnostic numbers as description. Align §5.5 Thursday with item 5. Fallback, only if the user wants pick accuracy to stop work early (Meth W2 option B, SC-31): keep it only as a stop for building R-c, with thresholds scaled to each configuration's headroom (10 points on A1 is 18% of the distance to its ceiling, about 5 points on A0), and state that it never carries or kills a candidate the bar decides.
- **Acceptance criteria**: Item 5 makes the development bar the only quantity that decides whether a test is built, no rule reads the told mapping, item 6 states what may be computed on test seeds before the verdict, and §5.5 Thursday matches item 5.

**R2: One definitions block**
- **Problem**: Item 2 defines B′ as "B plus the configuration's averaged-heads term"; memo §3 says B′ is "B rebuilt, not B plus a term", which produced the stored 18.44 (A0) and 18.80 (A1); Appendix D Step 4 fuses on B′, while memo §3 fuses on B, and §5 states neither (EIC W2; Meth W5). The two definitions give different numbers, and B′ was the binding bar comparator for A0 and A1 on seed 42 (Meth W5). Item 3's gain clause needed an interpretation note in step 1 (EIC W2).
- **Source**: EIC W2; Meth W5; Meth Minor Issues (gain statistic).
- **Requirement**: Open `DECISION_RULE.md` with a definitions block copied from memo §3 and point every item at it (EIC W2). Fused reader: `crossfit_nested(B, B, T, parity)`, 56 cells, the cell maximising min(R@1 − R@1 of B, gain) on one parity half applied to the other, fused on B "as in step 1" (Meth W5 recommends keeping B over switching to B′, because the references in items 4 and 5 are on B and the conservative bias is small, 0.04 to 0.05 on step-1 numbers). Matched counterpart: `crossfit_condition_free(B, B, T_cf, parity)`, T_cf = (T under a + T under b) / 2, same cells, max-R@1 rule; for R-b expected, Σ_h P̄(h)·s_h with P̄ the two-condition mean of the reader's probabilities (Meth S2); for R-c, per R5. B′: `crossfit_condition_free(cos, T_N1u, T_6u, parity)` with T_6u averaged over the configuration's own groupings (B rebuilt; A0 18.44, A1 18.80 on seed 42). Gain clause: gain of the fused reader minus gain of the fused counterpart (the latter 0 by construction), named as step 1 named it (`fusedT_vs_fusedTcf.gain`). Bar-margin comparator: whichever named comparator has the larger mean R@1, ties to B′ (B′ and the counterpart as now; add B if S11 is adopted).
- **Acceptance criteria**: The file defines the fusion base, the matched counterpart (per reader), B′, the gain statistic and the bar-margin comparator once, item 2 no longer says "B plus a term", and every rule item refers to these definitions.

**R3: `DECISION_RULE.md` names itself as binding and is self-contained**
- **Problem**: The rule is spread over §5.2, §5.4, §5.5 and memo §3, and the text names no binding version among them and Appendix D, so two careful readers can reach different verdicts on plausible outcomes (EIC W2; EIC Score rationale).
- **Source**: EIC W2 Suggestion ("Then add one line ... this file governs"); EIC Score rationale ("Fixing W1 to W3 and adding a precedence line would bring D2 to warn").
- **Requirement**: Add at the head of the file: "Where this file differs from §5.2, §5.5, the memo or Appendices B to F, this file governs." Carry into the file every definition and rule a reader needs, so that someone holding only the memo and the file can apply it (EIC Card #1 focus).
- **Acceptance criteria**: The file states that it governs, and no quantity, threshold or action needed to apply the rule exists only in §5.2, §5.5, memo §3 or an appendix.

**R4: R-b's domain-shift check as a stated diagnostic (arbitrated)**
- **Problem**: The kill in §5.2 rests on "high" and "does not move", which have no numbers, sits outside §5.4, and has no stated precedence over the bar or effect on an R-c built on R-b (EIC W3; Meth W2(b)). It reads the told mapping (Meth W2(c)).
- **Source**: EIC W3, Q2; Meth W2(b). Arbitrated under Disagreement 1.
- **Requirement**: Delete the kill sentence from §5.2. Add to §5.4: "R-b's accuracy on held-out bank episodes and its seed-42 pick accuracy are diagnostics, reported for every R-b configuration, and enter no rule; R-b survives or falls on the development bar like every other candidate." Add the methodology seat's label-free shift report: the standardized mean difference of each R-b input feature, and the distribution of R-b's top probability, on held-out bank episodes against seed-42 episodes, reported and not a kill. If the user instead wants a kill, use EIC W3's form with numeric X and Y and the stated precedence (applies whatever the bar margin; both R-b scorings and any R-c built on R-b leave before item 4; R-c is then built on the best remaining candidate), committed before any seed-42 number, with the methodology seat's objection recorded.
- **Acceptance criteria**: No undefined threshold remains for R-b, §5.4 states how the check relates to the bar, and no told-mapping quantity enters a rule unless the user chose the kill path, in which case X, Y and the precedence are in the commit.

**R5: R-c fully specified, with a condition-free counterpart**
- **Problem**: R-c's gate g comes from the reader's top-two margin, which differs between conditions (for R-a, the gap between the two largest scaled Δ under a and between the two smallest under b; for R-b, swapped inputs). A counterpart that "applies the same gate to T_cf", read as g_c·z(T_cf), changes with the condition, its gain need not be 0, and `crossfit_condition_free` raises an error on it (Meth W1). "A′'s min-margin cross-fit" refers to an A′ defined nowhere; the gate form, the margin each base reader supplies, the λ and threshold grids and the counterpart formula are unstated; R-c's score z(B) + λ·g·z(T) differs from memo §3's (1 + λ_u)·z(B) + λ_a·z(T) without comment (EIC W7).
- **Source**: Meth W1, Q1; EIC W7, Q6.
- **Requirement**: Write the methodology seat's specification into the R-c row. (i) Gate: g_c = 1 when the reader's top-two margin under condition c is at least τ, else 0; τ on a fixed label-free grid of the 0th, 25th, 50th and 75th percentiles of the parent reader's seed-42 margin, both conditions pooled (the 0th percentile gives g ≡ 1, the parent itself). State the margin each base reader supplies (scaled Δ for R-a, probabilities for R-b; EIC W7). (ii) Fused score: (1 + λ_u)·z(B) + λ_a·g_c·z(T_c) on the existing 56 cells times the 4 thresholds (224 cells), with the cross-fit rule written out in place of "A′'s min-margin cross-fit": the cell maximising min(R@1 − R@1 of B, gain) on one parity half, applied to the other. (iii) Counterpart term: G_cf = (g_a·z(T_a) + g_b·z(T_b)) / 2, fused as (1 + λ_u)·z(B) + λ_a·G_cf on the same 224 cells with the max-R@1 rule. Use G_cf, not ḡ·z(T_cf): G_cf is the two-condition mean of the complete conditioned term, as for T_cf and R-b's expected term, while ḡ·z(T_cf) is a product of means that drops the gate-term covariance and removes more than the condition. Report the 0th-percentile cell as a sanity check: R-c and its counterpart should reproduce the parent and its counterpart up to z-scoring before rather than after the average.
- **Acceptance criteria**: The R-c row states the gate form, the τ grid, the margin per base reader, the λ grid, the cross-fit rule and the G_cf counterpart, the counterpart term is identical under both conditions so it passes `crossfit_condition_free`'s check, and the 0th-percentile sanity cell is listed as a report.

---

## Suggested Revisions (Should Fix or Consider)

| Transport ref | Revision Item | Sub-Claim(s) | Severity | Evidence Anchor | Confidence | Source Reviewer | Obligation class | Cost scope | Bounded consequence |
|---|---|---|---|---|---|---|---|---|---|
| S1 | Define pick for R-c and for R-b expected | SC-3 | major (EIC W1) | text: Appendix A §5.4 item 5 "raises pick accuracy by at least 10 points over its configuration's arg-max reader" and "or reaches the bar, no test is built" | 5 (EIC W1) | EIC W1 | should_fix | sentence: §5.3 | reporting_requirement_unmet; section §5.3 |
| S2 | One outcome-to-action table and a status line (needed for D2 pass) | SC-10, SC-11, SC-12, SC-15 | minor (EIC W4) | absence: Appendix A §5.4 — expected one outcome-to-action table naming the action for every development and test outcome, including a test NO-GO, a partial pass, R-b killed after clearing the bar, and what is computed on test seeds before the verdict; checked memo §0 to §4, Appendix A §5.1 to §5.6, §6 and §7 | 4 (EIC W4) | EIC W4, Q4 | should_fix | section: new table in `DECISION_RULE.md` | interpretive_ambiguity_remains; section §5.4 |
| S3 | Per-seed and per-pair results are descriptive | SC-14 | minor (EIC W4); minor (Meth W9) | absence: Appendix A §5.4 — expected one outcome-to-action table naming the action for every development and test outcome, including a test NO-GO, a partial pass, R-b killed after clearing the bar, and what is computed on test seeds before the verdict; checked memo §0 to §4, Appendix A §5.1 to §5.6, §6 and §7 | 4 (EIC W4); 4 (Meth W9) | EIC W4; Meth W9 | should_fix | sentence: §5.4 item 6 | interpretive_ambiguity_remains; section §5.4 item 6 |
| S4 | A total tie order, measured from the largest bar margin | SC-16 | minor (EIC W5); minor (Meth W9) | text: Appendix A §5.4 item 4 "simpler (R-a before R-b, no gate before gate, A0 before A1)" | 4 (EIC W5); 4 (Meth W9) | EIC W5; Meth W9 | should_fix | sentence: §5.4 item 4 | interpretive_ambiguity_remains; section §5.4 item 4 |
| S5 | A0's role: A1 priority, paired A1 minus A0, meaning of an A0 GO | SC-17, SC-18 | minor (EIC W6); minor (Meth W8) | text: Appendix A §5.1 "to show whether the CSD grouping helps once the reader works" | 3 (EIC W6); 4 (Meth W8) | EIC W6, Q3; Meth W8, Q3 | should_fix | sentence: §5.1, §5.3, §5.4 items 1 and 4 | claim_scope_unsupported; claim: what a GO says about the CSD grouping |
| S6 | Glossary, plain names and a sequence-date note | SC-22, SC-23, SC-24 | minor (EIC W8) | text: memo §3 "with T_N1u the centered factor term of A3" and Appendix A §4 table header "B = C2 with R@1 18.34" | 3 (EIC W8) | EIC W8 | should_fix | section: head of `DECISION_RULE.md` | reader_traceability_reduced; manuscript (Appendix A §5) |
| S7 | Timeline gates, user approval of the commit, a cutoff and a fallback order | SC-25, SC-26, SC-27 | minor (EIC W9); minor (Meth W11) | absence: Appendix A §5.5 timeline — expected scheduled slots for the user's go-ahead to commit the rule, the re-derivation of the test numbers and the whole-branch final review, plus a rule for a slipped day; checked §5.5, §6 process and git bullets, §7 and memo §0 | 4 (EIC W9); 3 (Meth W11) | EIC W9; Meth W11 | should_fix | section: §5.5; sentence: §5.4 | acceptance_criterion_unmet; section §5.5 |
| S8 | Memo §1 points to §5.4; RCA relabelled; the plan's prior in §0 | SC-28, SC-29 | minor (EIC W10) | text: memo §1 "bar margin ≥ +0.5 with a 95% lower bound above 0" and "RCA (13.38, the GO bar)" | 4 (EIC W10) | EIC W10 | should_fix | sentence: memo §0 and §1 | reader_traceability_reduced; section memo §0 and §1 |
| S9 | R-a's spread from the pooled within-episode standard error | SC-32 | major (Meth W3) | text: Appendix A §5.2 R-a row "Spread = root mean square of Δ_h over the seed-42 development episodes, both conditions pooled" | 3 (Meth W3) | Meth W3 | should_fix | sentence: §5.2 R-a row | interpretive_ambiguity_remains; claim: a miss by R-a shows that noise scaling fails |
| S10 | R-b specification paragraph, frozen before any seed-42 number | SC-33, SC-34, SC-35, SC-36, SC-37 | major (Meth W4) | absence: Appendix A §5.2 R-b row and R-b requirements — expected the cross-fitted heads' training-row count and recipe, R-b's regularisation, feature scaling, bank size and the A0 bank, fixed before any seed-42 number; checked §5.2, §5.4, §5.5, §6, memo §3 and Appendix D Step 1 | 4 (Meth W4) | Meth W4 | should_fix | section: §5.2 R-b requirements | method_reproducibility_unresolved; section §5.2 |
| S11 | Add B to the bar-margin comparator and to the GO list | SC-38 | minor (Meth W5) | text: Appendix A §5.4 item 2 "B′ = B plus the configuration's averaged-heads term" and memo §3 "so B′ is B rebuilt, not B plus a term" | 4 (Meth W5) | Meth W5 | should_fix | sentence: definitions block; §5.4 item 6 | acceptance_criterion_unmet; section §5.4 items 3 and 6 |
| S12 | Projected test sensitivity and a stated reading of a NO-GO | SC-39 | minor (Meth W6) | absence: Appendix A §5.4 item 6 and §5.5 — expected a projected pooled half-width or smallest detectable bar margin for seeds 49 to 51, and a stated reading of a NO-GO whose pooled point estimate is above 0; checked memo §1, §5.3, §5.4, §5.6, Appendix D §5 and Appendix F.2 | 3 (Meth W6) | Meth W6 | should_fix | section: §5.4 item 6; §5.5 Thursday row | evidence_gap_remains; claim: Friday NO-GO reading |
| S13 | A "Claim licensed" line and the label-policy disclosure | SC-40, SC-41 | minor (Meth W7) | text: Appendix E §2.2 "Different episode seeds reuse the same 6,451 selection paintings" | 4 (Meth W7) | Meth W7 | should_fix | sentence: §5.4 item 6 | claim_scope_unsupported; claim: GO result |
| S14 | R-c's base and item 3's clauses stated | SC-42, SC-43 | minor (Meth W9) | text: Appendix A §5.4 item 4 "R-a before R-b, no gate before gate, A0 before A1" | 4 (Meth W9) | Meth W9 | should_fix | sentence: §5.4 items 1 and 3 | interpretive_ambiguity_remains; section §5.4 items 1 and 3 |
| S15 | Per-seed cross-fitting declared part of the method; fixed picks disclosed | SC-44 | minor (Meth W10) | text: Appendix A §5.4 item 6 "rerun every cross-fit on each" and memo §3 "two cross-fit halves are the episodes' index parity" | 4 (Meth W10) | Meth W10 | should_fix (optional frozen-cells line: consider) | sentence: §5.4 item 6 | claim_scope_unsupported; claim: GO result |
| S16 | Full-precision point; reference pick accuracies moved to §5.3 | SC-45 | minor [SEVERITY-SOURCE: letter-fallback] (Meth Minor Issues) | none (Meth Minor Issues list) | 4 [CONFIDENCE-SOURCE: report-level] | Meth Minor Issues | should_fix | sentence: §5.4 item 3; §5.3 | interpretive_ambiguity_remains; section §5.4 item 3 |
| S17 | State the measurement basis of the two halvings behind the +0.5 bar | SC-46 | none (question-channel item) | none (Meth Q4) | none | Meth Q4 | should_fix | sentence: §5.4 item 3 rationale | evidence_gap_remains; claim: calibration of the +0.5 development bar |

### Suggested Item Details

**S1: Define pick for R-c and R-b expected.** Source: EIC W1, Suggestion (ii). Fix: in §5.3, "R-c inherits its base reader's pick accuracy; R-b expected is scored by the arg-max of P(h)", so the diagnostic exists for every candidate.

**S2: Outcome-to-action table and status line.** Source: EIC W4, Q4. Fix: add a table with (1) no candidate eligible: no test, Friday chooses between named options; (2) one eligible: carry it; (3) several within 0.05: tie order per S4 and S5; (4) R-b's check: per R4, a diagnostic only; (5) test, every interval check passes (R@1 against cosine, RCA, B′ and the counterpart, plus B if S11 is adopted; gain above 0; gain against RCA): GO; (6) any check fails, including a partial pass: NO-GO, with the options named and the reading per S12; (7) on test seeds, only the GO quantities until the verdict is written down (R1). Add: "Items 1 to 5 are development selection on seed 42; item 6 is the only confirmatory test." The Journal-Fit seat states this table moves D2 from warn to pass.

**S3: Per-seed and per-pair results are descriptive.** Source: EIC W4; Meth W9; Meth W7. Fix: in item 6, "per-seed results are reported and do not change the pooled verdict"; per-pair results are reported, not tested.

**S4: Total tie order.** Source: EIC W5; Meth W9. Fix: "Every candidate whose bar margin is within 0.05 of the largest is tied; ties go to the earliest in this order: R-a, R-b arg-max, R-b expected, R-c." With S5 the A1 and A0 question is a priority, not a tie-break. The methodology seat notes that the 0.05 window is far below the noise of paired differences, so the rule in effect carries the maximum, which is acceptable here because the fresh-seed test decides.

**S5: A0's role.** Source: EIC W6, Q3; Meth W8, Q3. Fix: add to §5.3 "paired A1 − A0 difference in R@1 and bar margin under each reader (seed 42, descriptive)". In item 4, replace "A0 before A1" by the methodology seat's recommended priority: "carry the best A1 candidate that clears the bar; carry an A0 candidate only if no A1 candidate clears it". State in item 1 that A0 runs are candidates under that priority, and add the Journal-Fit seat's sentence: "If an A0 configuration is carried, a GO supports the reader fix without the CSD grouping, and the CSD question stays open." Whether an A0 GO counts as a GO for plan (a) is the user's call; the rule should record the answer.

**S6: Glossary, names, dates.** Source: EIC W8. Fix: a one-line glossary of the codes the file keeps (B, B′, T_cf, T_N1u, T_6u, A0, A1, R-a to R-c); drop or gloss C2, E2 (and its AIC bank), N4, N6 and T6; write "the method-A checkpoint" instead of "A3" in B's definition; add "Folder and report dates are sequence numbers, not calendar dates."

**S7: Timeline gates and cutoff.** Source: EIC W9; Meth W11. Fix: in §5.5, Tuesday "user approves the rule commit" (EIC W9); Thursday evening "controller re-derives the test (GO) quantities"; Friday morning "whole-branch final review of the test before the GO or NO-GO is reported, then the user decides". In §5.4, a cutoff: Meth W11 proposes "Candidates without development numbers by Thu 8 Oct 12:00 are dropped from this round, and R-c is built on the best completed candidate"; EIC W9 gives Wed 23:00 as an example and leaves the choice to the user. State one time. Fallback order if time runs short (Meth W11): R-a and R-c on R-a first, R-b arg-max on A1 next, the remaining configurations last.

**S8: Memo fidelity and the prior.** Source: EIC W10. Fix: in memo §1, replace the bar line with "Development bar and GO: exactly as Appendix A §5.4 items 3 and 6"; rename RCA's label to "the strongest raw pair metric"; add to §0: "Our own estimate of the chance of a GO is moderate at best (Appendix B §10): fresh-seed tests have so far roughly halved development margins."

**S9: R-a's spread.** Source: Meth W3. Fix: σ_h = sqrt(mean over seed-42 episodes of (s²_S,h + s²_C,h) / 4), where s²_S,h and s²_C,h are the sample variances of the four support-pair and the four contrast-pair agreements on grouping h. It is the standard error of Δ_h implied by pair-to-pair scatter, identical under both conditions, label-free and frozen from seed 42. Do not use bank episodes (the heads' draw touches 86% of scorer-train paintings, so bank posteriors are largely in-sample and sharper). Replace the root mean square rather than adding a second R-a. The card's order-of-magnitude check: under the root mean square the image grouping's scaled Δ in emotion × genre falls from about 1.39 to about 0.94.

**S10: R-b specification.** Source: Meth W4. Fix: add a paragraph to §5.2, committed with the rule: cross-fitted heads with `fit_one_head`'s recipe and a 60,000-row draw within each half (each half holds about 92,000 rows), each refit head's held-out accuracy reported beside the memo §3 values before R-b is trained; a fixed bank size per half (for example 65,536 episodes, about 200 s each); features standardized on bank training episodes; C chosen by five-fold cross-validation on bank episodes only; the two half-readers' probabilities averaged; the same recipe for A0 on a three-grouping bank. State that no R-b setting changes after any seed-42 number is read, and that a change becomes a new candidate reported as such. Disclose the bank's label-space limit (classes are groupings with a uniform prior; genre is not a class). Optional, consider, only if declared now: a label-free EM correction of R-b's class prior on unlabelled seed-42 episodes (Saerens and colleagues, 2002).

**S11: B as a comparator.** Source: Meth W5. Fix: the bar-margin comparator is the largest R@1 of B, B′ and the counterpart, and B joins item 6's GO list; B is computed on every seed anyway.

**S12: Test sensitivity.** Source: Meth W6. Fix: on Thursday, before building the seeds, split the carried configuration's seed-42 per-episode margin variance into between-painting and within-painting parts (a one-way decomposition by anchor painting, minutes on CPU), project the pooled three-seed half-width and write it in the log. Add to item 6: "A NO-GO whose pooled point estimate is above 0 is reported as inconclusive at a detectable margin of x, not as evidence that the reader fails." The card's projections: a pooled half-width of about 0.12 to 0.14 if episode noise dominates and 0.18 to 0.20 if painting variation dominates; at a margin of +0.25 one comparison clears with probability about 0.96 or 0.73, at +0.15 about 0.62 or 0.34.

**S13: Claim licensed.** Source: Meth W7. Fix: add to item 6: "A GO shows that the carried configuration beats each comparator, pooled over the three aspect pairs, on new episodes drawn from the same 6,451 selection paintings. It does not show transfer to new paintings (the held split, 12,281 paintings, stays reserved for the paper test) or a margin on each pair (per-pair results are reported, not tested)." For the paper: the groupings may be described as built without evaluation labels, with the label-free diagnostics that also rank CSD above Gram (placeability P_ami 0.208 against 0.150, Leiden-seed stability 0.807 against 0.673, Appendix C), and the decision to continue with CSD after its told margins were read should be disclosed.

**S14: R-c's base and item 3's clauses.** Source: Meth W9. Fix: "R-c is built on the candidate with the largest bar margin, whether or not it clears the bar"; keep item 3's lower-bound and gain clauses, noted as kept for continuity (no step-1 arm failed the gain clause, and the lower bound is implied by +0.5 at seed-42 half-widths of 0.21 to 0.25).

**S15: Test-time cross-fits.** Source: Meth W10. Fix: state in item 6 that per-seed cross-fitting is part of the method's definition and that the intervals hold the picks fixed. Optional, consider: beside the GO, report the same comparisons with the seed-42 cells frozen, as a descriptive line and not a second verdict. Painting-level halves are not needed for a decision of this size.

**S16: Precision and placement.** Source: Meth Minor Issues. Fix: say that "≥ +0.5" applies to the full-precision point, as Appendix C's readings did. Under R1, move the reference pick accuracies (A1 43.2%, A0 54.7%, the seed-42 arg-max reader fused on B) to §5.3 as diagnostics.

**S17: Basis of the halvings.** Source: Meth Q4 ("The bar's calibration rests on them"). Fix: state in the bar's rationale whether the two halvings (0.52 to 0.26, 0.26 to 0.15) were measured on R@1 margins against a matched control or on another quantity.

---

## Revision Roadmap

### Source-traceability checklist

Immutable source order: seat order (EIC, then Meth), then findings before question answers, then ordinal, then sub-claim. An item both seats raised sits at its earliest raising reference. This order is not a work order and implies no priority; the Rewrite Map above is the presentation view for the rewrite. The author chooses `will_address`, `wont_address` or `not_on_point` later in the separate author-adjudication step. The machine `revision-roadmap/1.0` core is not emitted with this letter, because no block manifest was bound for the plan and its `proposed_targets` cannot be filled without inventing block ids.

- [ ] REV-1, R1 (obligation `must_fix`): kill decided by the development bar alone; told mapping out of every rule; test-seed computation order (EIC W1, W4; Meth W2)
- [ ] REV-2, S1 (obligation `should_fix`): define pick for R-c and R-b expected (EIC W1)
- [ ] REV-3, R2 (obligation `must_fix`): one definitions block (EIC W2; Meth W5, Minor Issues)
- [ ] REV-4, R3 (obligation `must_fix`): the rule file governs and is self-contained (EIC W2)
- [ ] REV-5, R4 (obligation `must_fix`): R-b's domain-shift check as a stated diagnostic, with a shift report (EIC W3; Meth W2; arbitrated)
- [ ] REV-6, S2 (obligation `should_fix`): outcome-to-action table and status line (EIC W4)
- [ ] REV-7, S3 (obligation `should_fix`): per-seed and per-pair results descriptive (EIC W4; Meth W9)
- [ ] REV-8, S4 (obligation `should_fix`): total tie order (EIC W5; Meth W9)
- [ ] REV-9, S5 (obligation `should_fix`): A0's role (EIC W6; Meth W8)
- [ ] REV-10, R5 (obligation `must_fix`): R-c fully specified with a condition-free counterpart (EIC W7; Meth W1)
- [ ] REV-11, S6 (obligation `should_fix`): glossary, names, sequence dates (EIC W8)
- [ ] REV-12, S7 (obligation `should_fix`): timeline gates, commit approval, cutoff, fallback order (EIC W9; Meth W11)
- [ ] REV-13, S8 (obligation `should_fix`): memo §1 and §0 (EIC W10)
- [ ] REV-14, S9 (obligation `should_fix`): R-a's spread (Meth W3)
- [ ] REV-15, S10 (obligation `should_fix`): R-b specification (Meth W4)
- [ ] REV-16, S11 (obligation `should_fix`): B as a comparator (Meth W5)
- [ ] REV-17, S12 (obligation `should_fix`): test sensitivity and the NO-GO reading (Meth W6)
- [ ] REV-18, S13 (obligation `should_fix`): claim licensed and label-policy disclosure (Meth W7)
- [ ] REV-19, S14 (obligation `should_fix`): R-c's base and item 3's clauses (Meth W9)
- [ ] REV-20, S15 (obligation `should_fix`): per-seed cross-fitting declared (Meth W10)
- [ ] REV-21, S16 (obligation `should_fix`): full-precision point; reference pick accuracies (Meth Minor Issues)
- [ ] REV-22, S17 (obligation `should_fix`): basis of the halvings (Meth Q4)

---

## Journal-Supplied Deadline (Optional Transport)

- **Exact deadline from source letter**: NOT PROVIDED. The plan's own dates (rule committed before any code; go/no-go on Friday 9 October 2026; CVPR abstract 10 November) are author context, not a review deadline.

---

## Response Letter Instructions

Please respond to every item using the format in `templates/revision_response_template.md`.

**Must include**:
1. A response and change description for each Required Revision (R1 to R5), including, for R4, whether the arbitrated option A was taken or the user chose the kill path.
2. A response for each Suggested Revision (S1 to S17): adopted, or the reason for not adopting it, plus the user's choices where a card leaves one open (the cutoff time in S7; whether an A0 GO counts as a GO for plan (a) in S5).
3. The revised §5 with changes marked, and the resulting `DECISION_RULE.md`.
4. A cross-reference table from each R and S item to the rule item, definition or table row that resolves it.

---

## Closing

We ask you to rewrite Appendix A §5 and its §5.4 rule as set out above and to return the rewritten rule for a scoped re-review before it is committed and before any code. Both seats found the plan's test architecture, its matched counterparts for R-a and R-b, and its order of work sound. The methodology seat judges that every change fits Tuesday 6 October and needs no new data, seeds or runs, and the Journal-Fit seat estimates a few hours for its items.

---

## Appendix: Full Reviewer Reports and Sub-Claim Inventory

The two complete Phase 2 cards are attached by reference in the same folder: `phase2_eic.md` (Journal-Fit Reviewer) and `phase2_methodology.md` (Peer Reviewer 1, methodology, with its seven arithmetic receipts). Their Phase 1 commitments are `phase1_eic.md` and `phase1_methodology.md`; the panel configuration is `phase0_reviewer_configuration.md`.

### Sub-claim inventory (Step 1b, compact form)

Positions: raised, corroborated, not-mentioned (silence), disputed. Severity and confidence are transported from the parent finding. "Disputed (severity)" marks a seat that raised the same sub-claim at a different severity; "disputed (remedy)" marks an incompatible fix.

| SC | Parent finding(s) | EIC position | Meth position | Severity (EIC / Meth) | Confidence (EIC / Meth) | Disposition | Item |
|---|---|---|---|---|---|---|---|
| SC-1 | Item 5 reads two ways; one outcome gets no action or contradicts §5.5 | raised (W1) | corroborated (W2a) | major / major | 5 / 4 | corroborated 2 of 2 | R1 |
| SC-2 | Pick accuracy, a diagnostic, decides a kill | raised (W1) | corroborated (W2c) | major / major | 5 / 4 | corroborated 2 of 2 | R1 |
| SC-3 | "Pick" undefined for R-c and R-b expected | raised (W1) | not-mentioned | major | 5 | single-seat | S1 |
| SC-4 | B′ has two incompatible definitions | raised (W2) | disputed (severity; raised in W5) | major / minor | 4 / 4 | SPLIT (severity), resolved | R2 |
| SC-5 | Fusion base (B or B′) not stated in §5 | raised (W2) | disputed (severity; raised in W5) | major / minor | 4 / 4 | SPLIT (severity), resolved | R2 |
| SC-6 | No precedence rule names a binding version | raised (W2) | not-mentioned | major | 4 | single-seat | R3 |
| SC-7 | Item 3's gain clause needs an exact statistic | raised (W2) | disputed (severity; Minor Issues) | major / minor [SEVERITY-SOURCE: letter-fallback] | 4 / 4 [CONFIDENCE-SOURCE: report-level] | SPLIT (severity), resolved | R2 |
| SC-8 | R-b's domain-shift kill has no numbers | raised (W3) | disputed (remedy; raised in W2b) | major / major | 4 / 4 | SPLIT (direction), arbitrated | R4 |
| SC-9 | R-b's kill outside §5.4, no precedence over the bar, effect on R-c unstated | raised (W3) | disputed (remedy; raised in W2b) | major / major | 4 / 4 | SPLIT (direction), arbitrated | R4 |
| SC-10 | No single outcome-to-action table | raised (W4) | not-mentioned | minor | 4 | single-seat | S2 |
| SC-11 | Test NO-GO has no named options | raised (W4) | not-mentioned | minor | 4 | single-seat | S2 |
| SC-12 | Partial GO pass only implied to be NO-GO | raised (W4) | not-mentioned | minor | 4 | single-seat | S2 |
| SC-13 | Test-seed computation before the verdict not restated | raised (W4) | disputed (severity; raised in W2c) | minor / major | 4 / 4 | SPLIT (severity), resolved | R1 |
| SC-14 | Per-seed and per-pair results not said to decide nothing | raised (W4) | corroborated (W9) | minor / minor | 4 / 4 | corroborated 2 of 2 | S3 |
| SC-15 | No line separating development selection from the confirmatory test | raised (W4) | not-mentioned | minor | 4 | single-seat | S2 |
| SC-16 | Tie-break is a partial order; window reference unstated | raised (W5) | corroborated (W9) | minor / minor | 4 / 4 | corroborated 2 of 2 | S4 |
| SC-17 | A1 minus A0 under the same reader not measured | raised (W6) | corroborated (W8) | minor / minor | 3 / 4 | corroborated 2 of 2 | S5 |
| SC-18 | A0 both reference and candidate; meaning of an A0 GO unstated | raised (W6) | corroborated (W8) | minor / minor | 3 / 4 | corroborated 2 of 2 | S5 |
| SC-19 | "A′" undefined; R-c's cross-fit rule not stated | raised (W7) | not-mentioned (its W1 fix states the rule) | minor | 3 | single-seat | R5 |
| SC-20 | R-c's gate form, margin, grids and counterpart formula unstated | raised (W7) | disputed (severity; raised in W1) | minor / critical | 3 / 4 | SPLIT (severity), resolved | R5 |
| SC-21 | R-c's score form differs from memo §3 | raised (W7) | not-mentioned | minor | 3 | single-seat | R5 |
| SC-22 | Undefined internal codes | raised (W8) | not-mentioned | minor | 3 | single-seat | S6 |
| SC-23 | Colliding names (A3; R1 to R3, R-a to R-c, R@1; a/b and A/B) | raised (W8) | not-mentioned | minor | 3 | single-seat | S6 |
| SC-24 | Sequence dates unexplained; three rule-file paths | raised (W8) | not-mentioned | minor | 3 | single-seat | S6 |
| SC-25 | No slot for test re-derivation and final review before Friday | raised (W9) | corroborated (W11) | minor / minor | 4 / 3 | corroborated 2 of 2 | S7 |
| SC-26 | No slot for the user's approval of the commit | raised (W9) | not-mentioned | minor | 4 | single-seat | S7 |
| SC-27 | No slip rule or cutoff | raised (W9) | corroborated (W11) | minor / minor | 4 / 3 | corroborated 2 of 2 | S7 |
| SC-28 | Memo §1 bar summary drifts from §5.4 | raised (W10) | not-mentioned | minor | 4 | single-seat | S8 |
| SC-29 | The plan's own prior absent from memo and Appendix A | raised (W10) | not-mentioned | minor | 4 | single-seat | S8 |
| SC-30 | R-c's counterpart g_c·z(T_cf) not condition-free | not-mentioned (outside its card) | raised (W1) | critical | 4 | single-seat | R5 |
| SC-31 | The same 10 points mean different things on A1 and A0 | not-mentioned | raised (W2c) | major | 4 | single-seat | R1 (fallback) |
| SC-32 | R-a's root mean square mixes signal into the noise scale | not-mentioned | raised (W3) | major | 3 | single-seat | S9 |
| SC-33 | Cross-fitted heads' size and recipe unstated (sharpness mismatch) | not-mentioned | raised (W4) | major | 4 | single-seat | S10 |
| SC-34 | R-b's regularisation, scaling, bank size and half-reader combination open | not-mentioned | raised (W4) | major | 4 | single-seat | S10 |
| SC-35 | A0 reader needs its own three-grouping bank | not-mentioned | raised (W4) | major | 4 | single-seat | S10 |
| SC-36 | Nothing forbids revising R-b after seed-42 numbers | not-mentioned | raised (W4) | major | 4 | single-seat | S10 |
| SC-37 | Bank label space limits what R-b can learn | not-mentioned | raised (W4) | major | 4 | single-seat | S10 |
| SC-38 | Rebuilt B′ can fall below B; B not a comparator | not-mentioned (transcribes memo §3) | raised (W5) | minor | 4 | single-seat | S11 |
| SC-39 | Test sensitivity unstated; NO-GO reading undefined | not-mentioned | raised (W6) | minor | 3 | single-seat | S12 |
| SC-40 | What a GO licenses is unstated | not-mentioned | raised (W7) | minor | 4 | single-seat | S13 |
| SC-41 | Label-policy disclosure for the CSD choice | not-mentioned | raised (W7) | minor | 4 | single-seat | S13 |
| SC-42 | Whether R-c's base must clear the bar | not-mentioned | raised (W9) | minor | 4 | single-seat | S14 |
| SC-43 | Item 3's lower-bound and gain clauses never bind | not-mentioned | raised (W9) | minor | 4 | single-seat | S14 |
| SC-44 | Test-time cross-fits read test labels; intervals hold picks fixed | not-mentioned | raised (W10) | minor | 4 | single-seat | S15 |
| SC-45 | Full-precision point; placement of reference pick accuracies | not-mentioned | raised (Minor Issues) | minor [SEVERITY-SOURCE: letter-fallback] | 4 [CONFIDENCE-SOURCE: report-level] | single-seat | S16 |
| SC-46 | Measurement basis of the two halvings | not-mentioned | raised (Q4) | none | none | single-seat | S17 |

## Attachment: Acronym Check (advisory, #849)

### Acronym check (advisory; no reply needed)
Coverage: body (partial)
Not in this input: English abstract, Chinese abstract.
Not checked:
- Body, line 918: AR (a definition form this check does not read)

| Scope | Line | Rule | Acronym | Uses |
|---|---|---|---|---|
| Body | 10 | Not defined | CoSiR | 3 |
| Body | 12 | Not defined | CSD | 41 |
| Body | 17 | Not defined | CVPR | 4 |
| Body | 28 | Not defined | GO | 10 |
| Body | 46 | Not defined | RCA | 11 |
| Body | 55 | Not defined | CLIP | 21 |
| Body | 91 | Not defined | ViT | 4 |
| Body | 137 | Not defined | ARS | 6 |
| Body | 156 | Not defined | PLAN | 5 |
| Body | 200 | Not defined | AMI | 23 |
| Body | 225 | Not defined | AIC | 1 |
| Body | 225 | Not defined | CPU | 8 |
| Body | 228 | Not defined | CACTU | 1 |
| Body | 250 | Not defined | SHA | 4 |
| Body | 285 | Not defined | CUDA | 1 |
| Body | 285 | Not defined | NUM | 2 |
| Body | 285 | Not defined | OMP | 1 |
| Body | 286 | Not defined | GPU | 3 |
| Body | 286 | Not defined | MKL | 1 |
| Body | 348 | Not defined | VGG | 3 |
| Body | 376 | Not defined | MLLM | 4 |
| Body | 473 | Not defined | kNN | 3 |
| Body | 498 | Not defined | ARI | 1 |
| Body | 577 | Not defined | HOW | 1 |
| Body | 577 | Not defined | WHAT | 1 |
| Body | 577 | Not defined | WHY | 1 |
| Body | 594 | Not defined | MFCVAE | 1 |
| Body | 594 | Not defined | SCE | 1 |
| Body | 595 | Not defined | ICLR | 1 |
| Body | 599 | Not defined | DEC | 1 |
| Body | 599 | Not defined | IIC | 1 |
| Body | 599 | Not defined | SCAN | 1 |
| Body | 599 | Not defined | SwAV | 1 |
| Body | 599 | Not defined | TEMI | 1 |
| Body | 602 | Not defined | LAION | 1 |
| Body | 655 | Not defined | PCA | 1 |
| Body | 877 | Not defined | AB | 2 |
| Body | 946 | Not defined | DINOv2 | 1 |
| Body | 963 | Not defined | MB | 4 |
| Body | 964 | Not defined | GB | 1 |
| Body | 1196 | Not defined | DAS6 | 1 |
| Body | 1196 | Not defined | RTX | 1 |
| Body | 1211 | Not defined | CRL | 1 |
| Body | 1211 | Not defined | VL | 1 |
| Body | 1243 | Not defined | ArtGAN | 1 |
| Body | 1342 | Not defined | CVS | 1 |
| Body | 1342 | Not defined | KISSME | 1 |
