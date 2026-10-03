# Editorial Decision

## Calibration Resolution

`calibration_status: NOT_CALIBRATED`

Current runtime boundary: this package is not upgraded from a candidate or prose-named profile. `PROFILE_MEASURED` stays unavailable until a closed profile artifact and replay validator bind the exact target fields to a completed panel's execution topology.

## Manuscript Information

- **Title**: Which repair to run first after the E3 NO-GO: a decision memo for method A′ (CoSiR v2), with the E3 report, the repair handoff and spec §3/§6 appended
- **Manuscript ID**: none assigned (internal decision memo; review folder `src/test/20261104_ars_repair_order_review/`)
- **Submission Date**: 2026-10-03 (memo date)
- **Decision Date**: 2026-10-03
- **Review Round**: 1
- **Mode and contract**: `reviewer_methodology_focus`, contract `reviewer/reviewer_methodology_focus/v2`, baseline v3.20.0, `panel_size` 2, generated 2026-10-03T12:32:12Z
- **Object of review**: the memo, §0 to §9. Appendices A to C are evidence. E3's own verdict was not re-reviewed (Phase 0 strategy).
- **Target binding**: `criteria_binding_unavailable`, disclosed in both Phase 1 cards. No venue criteria were applied and this letter makes no venue-fit claim. CVPR is author-stated context only.

## Review Panel Composition and Provenance (plain text)

This mode is not `reviewer_full`, so the typed `review-panel-provenance/1.0` artifact and its six-axis block do not apply and none was built. The panel is described here in plain text.

- **Seats.** Two scoring seats: the Journal-Fit Reviewer (`contract_role: eic`, owner of D2 `writing_and_structure`) and Peer Reviewer 1 (`contract_role: methodology`, owner of D1 `methodology_rigor`, mandatory). This mode has no domain seat, no perspective seat and no Devil's Advocate seat.
- **Execution.** Both seats ran on the same model family (Claude, provider Anthropic), each in its own fresh subagent context. Each Phase 1 was paper-content-blind (contract and metadata only). Neither seat saw the other's output before committing its card. No human reviewer sat on the panel. This synthesis was also written by a Claude model.
- **Conformance.** Per the orchestrator, both Phase 2 cards passed `check_phase_conformance.py` and `check_panel_synthesis.py --layer1-only` before synthesis. Usable cards: 2 of `panel_size` 2, so no `[PANEL-SHRUNK]`.
- **Correlated-error caveat.** Role separation is not independence. Two seats from one model family can share blind spots and make the same mistake, so agreement between them is weaker evidence than agreement between independent reviewers, and an error both seats share would not show up as a disagreement here. No binary or numeric independence claim is made. Context separation holds within this panel attempt only.
- **Evidence access note.** The methodology seat states that it read repository code that the manuscript only names: `train_fit_diagnostic.py` (confidence basis of its W1) and `crossfit_lambda` in `src/eval/aspect_scorers.py` (W10). The Phase 0 report had expected code to be out of reach. Those two findings therefore rest partly on material outside the manuscript.
- **Manuscript text aimed at the panel.** None found (Phase 0 report; Journal-Fit card, scope notes). The user's stated leaning toward H3 first is a disclosed preference and an anchoring cue; both seats judged the order on the evidence. Appendix B's imperatives address the project's implementing agent and were treated as evidence of the planned workflow.

---

## Decision

### Major Revision

The memo's decision rules need revising and re-checking before anything runs. Nothing fatal was found: the seed-45 episodes and the held rows are unspent. The methodology seat judges every required change repairable within about a day of the window, most of it rule text with no compute.

---

## Mechanical Synthesis (sprint contract, three steps)

**Step 1: role-scoped scoring matrix.** Only assessed scores from eligible seats count. Ineligible `not_assessed` values are excluded from numerator and denominator.

| Dimension | Priority | Eligible roles | eic seat | methodology seat | Assessed eligible seats | Verdict |
|---|---|---|---|---|---|---|
| D1 methodology_rigor | mandatory | methodology | not_assessed (ineligible, excluded) | block, `block_class: repairable` | 1 | block |
| D2 writing_and_structure | normal | eic | warn | not_assessed (ineligible, excluded) | 1 | warn |

Both dimensions have an assessed eligible seat, so there is no `[DIMENSION-UNASSESSED]`.

Criterion judgements on the sprint scale (`judgement_scale: sprint_contract`, values copied unchanged): D1 = block (methodology seat); D2 = warn (eic seat).

**Step 2: failure conditions.** Every expression parsed against the recognised vocabulary. The cross-reviewer quantifier is applied per dimension over its assessed eligible seats (n = 1 for each dimension here).

| Condition | Severity | Quantifier | Expression (pattern) | Evaluation | Fired | Action |
|---|---|---|---|---|---|---|
| F1 | 95 | any | `D1 has a fatal block` (6) | D1: 0 of 1 seats fatal (block is repairable) | no | reject |
| F2 | 90 | any | `D1 scores 'block'` (4) | D1: 1 of 1 seats score block | yes | major_revision |
| F3 | 70 | any | `D1 scores 'warn'` (4, exact match) | D1: 0 of 1 seats score exactly warn (the score is block) | no | major_revision |
| F4 | 40 | any | `D2 scores 'warn' or worse` (8) | D2: 1 of 1 seats at warn or worse | yes | minor_revision |
| F0 | 10 | all | `every dimension scores 'pass'` (9) | D1: 0 of 1 pass; D2: 0 of 1 pass | no | accept |

**Step 3: precedence.** Fired: F2 (severity 90) and F4 (severity 40). F2 has the highest severity, so its action applies. There is no DA seat, so there are no DA CRITICAL IDs to adjudicate, and since the decision is not accept, no DA-versus-accept marker applies. The fired action is not softened.

dimension_verdicts: [D1=block, D2=warn]
fired_conditions: [F2, F4]
da_critical_adjudications: []
editorial_decision=major_revision

`ARS_CROSS_MODEL` is not set, so no cross-model blind decision check was run.

---

## Blocking Issues (3, immutable source order)

The methodology seat's D1 block "rests on W1" with "W2, W4, W6 and W7" adding to it; the Journal-Fit seat's D2 warn is driven by its Major W1. These are grouped into three rows by the step they affect. Rows follow roadmap source order.

| Transport ref | Blocking issue | Source reviewer(s) | Evidence anchor | Resolving roadmap item |
|---|---|---|---|---|
| R1, R2, R3, R8, R9 | The H1 pilot has a "promising" reading but no complete outcome-to-action rule; "falls short" is undefined; seed 45 is not committed as a single look | EIC W1; Meth W3, W4 | EIC W1: text: §6 table, O2 row "H3 only if H1 falls short"; "if H1 looks promising on seed 42 we may skip H3". Meth W4: text: §4 "the H1 score on an H2-repaired model, if H1 alone falls short" | REV-1, REV-2, REV-3, REV-15, REV-17 |
| R4, R5, R6, R11 | H3's rules cannot separate its readings: the fit rule is anchored at zero with an unspecified scorer and does not cover every outcome, and the 3.0 transfer bar applies a halving that does not belong | Meth W1, W7 (EIC scope-note pointer on the coverage gap) | Meth W1: text: §5.1 "the gain on fresh label episodes over scorer-train rows (a new episode seed), with the". Meth W7: text: §5.1 "gains halved on the fresh draw in E3, fusion keeps only part of a term's gain" | REV-11, REV-12, REV-13, REV-20 |
| R7, R10 | Two design choices weaken the later GO test: a pick criterion that trades away the binding R@1 margin, and a "no fit" branch that would tune A′'s settings on evaluation-label episodes | Meth W2, W6 | Meth W2: text: §5.2 "parity halves, mean(R@1, gain) per half, as in E3". Meth W6: text: §5.1 "LAB, a fast testbed whose target is known to be learnable, before any pseudo-bank retraining" | REV-14, REV-19 |

Seat labels: "EIC" is the Journal-Fit Reviewer; "Meth" is Peer Reviewer 1 (methodology, seat R1 in the schema's seat enum). Transport refs R1 to R12 and S1 to S15 are roadmap positions, not seats and not a work rank.

---

## Reviewer Summary

| Seat | Role | Dimension scored | Recommendation | Confidence | Findings |
|---|---|---|---|---|---|
| Journal-Fit Reviewer (`eic`) | Research director who signs off on experiment decision memos (Phase 0 Card #1) | D2 warn | None issued (sprint seats score dimensions; per-seat decisions are retired) | Per finding, 3 to 5 | 7 strengths; 8 weaknesses (1 Major, 7 Minor) |
| Peer Reviewer 1 (`methodology`) | Statistician in ML evaluation and selective inference (Phase 0 Card #2) | D1 block, repairable | None issued; on the order question it recommends a modified O3 | Per finding, 4 to 5 | 6 strengths; 11 weaknesses (6 Major, 5 Minor); 13 arithmetic receipts (12 consistent, AR12 not computable because the rounding rule is ambiguous) |
| Peer Reviewer 2 (domain), Peer Reviewer 3 (perspective), Devil's Advocate | Not in this mode's panel | n/a | n/a | n/a | n/a |

---

## The Panel's Answer on the Order (§0 and §9 Q1)

**Where the seats agree.**
1. Commit every decision rule before anything runs, in one dated, hashed commit made before the LAB bank is built and before any H1 pilot number exists, with its hash in the §7 ledger. H1's rules go into the same commit if H1 runs first or in parallel (EIC Q3; Meth Q1 step 1 and Q3).
2. O1's one visible advantage, that only its first step has a complete rule chain, comes from how the memo was drafted, not from evidence for the order. Writing the H1 rule block removes it (EIC Q1 and W1; Meth W3 and W5 find the same missing H1 branches).
3. Whatever order is chosen, every step that runs first needs a full outcome-to-action rule set (EIC Q1; Meth Q2 to Q5).
4. Both seats read the memo's own §6 dependency note ("H3's transfer measurement is more informative when the nested scorer of H1 already exists") as bearing on the choice. Meth Q1 uses it against O1. EIC Q1 combines it with the memo's time column: O3 gives both readings in about the same day that O1 takes for H3 alone.

**Where one seat defers.** The Journal-Fit seat does not choose an order. It writes that "whether H3 first is worth delaying H1 by a day is the methodology seat's question" (EIC Q1). The order is therefore answered by the methodology seat, within D1.

**The methodology seat's answer: a modified O3** (Meth Q1).
- *Before any run:* the hashed commit with the revised H3 and H1 rules, the joint decision table and the single-look rule for seed 45.
- *Day 1:* write and test the 2-D cross-fit. Build LAB and a matched-k bank. Launch L3, L5, a fixed-τ run and the matched-k run in one GPU lock (about 10 minutes each). Run the H1 pilot with A3 as the pre-specified primary model and the other checkpoints as descriptive rows.
- *Day 2 morning:* score the LAB runs in-distribution and on seed 42 under both the term-only and the nested score, each paired against A3 and C0 on identical episodes. Apply the joint table. If both readings are negative, the repair stops on day 2.
- *If only one code stream can be reviewed per day:* H1 first and H3 on day 2, "never O1".
- *Reasons given:* H1 lies on every path after H3 and is the cheapest step (half a day, minutes of CPU); H3 answers a different question (is retraining worth the remaining days); the two seed-42 looks that the memo lists as O3's cost are not extra, since O1 makes the same looks a day apart; O2 with H3 skipped risks spending the single seed-45 look on a thin margin without knowing the ceiling; O4 is the least diagnostic.
- *On the user's leaning:* H3 is worth running, because it is the only step that can separate "the partitions are the limit" from "the architecture or loss is the limit". As designed it should not run first and alone: it delays the cheapest decisive step by a day, and Meth W1 leaves its most likely outcome without a clear next step.

**Disagreement.** None on the order. The only difference is timing presentation: the Journal-Fit seat reads O3 from the memo's cost column (about one day for both steps), while the methodology plan places the joint readout on day 2 morning because H3's transfer is scored once the nested scorer exists. No seat supports O1 as the first step: the methodology seat recommends against it, and the Journal-Fit seat finds its advantage to be a drafting artefact while leaving the call to the methodology seat.

**Scope of this answer.** No seat argued the case for stopping now and going to branch 3; this panel has no Devil's Advocate. The answer takes the user's decision to attempt the repair as given.

---

## Required Changes Before Anything Runs (summary)

Everything in groups A and C below belongs in the pre-run hashed commit (EIC Q3; Meth Q1 step 1).

**A. Needed for interpretability or to protect the GO test** (must_fix). This group follows the methodology card's own "Needed" list, plus Meth W7 (named in its D1 block rationale and Q3 rule changes but not repeated in that list) and the Journal-Fit seat's Major W1, which asks for the same H1 rule block.

| Ref | Change | Card(s) |
|---|---|---|
| R1 | H1 pilot rule block: promising, not promising and inconclusive, each with a reading and a next action, including when H3 is skipped and when branch 3 fires | EIC W1; Meth W3 |
| R2 | Define "falls short" as a development-pilot outcome only | EIC W1; Meth W4 |
| R3 | One set of outcome words across §5.1, §5.2 and §6; a reading that routes a decision is not "descriptive" | EIC W1 |
| R4 | H3: one deciding scorer (term-only agreement rule, λ = ∞) and a fixed episode count (4,096 per pair, stated seed) | Meth W1, Q2(b) |
| R5 | H3: paired A3 and C0 baselines on identical episodes; fit decided on LAB minus A3; loss bar descriptive only | Meth W1, Q2(a), Q3 |
| R6 | H3 readings that cover every outcome, with an inconclusive band | Meth W1, Q3; EIC scope-note pointer |
| R7 | Pick rule aligned with the binding R@1 comparison, reused unchanged for the A′ pick | Meth W2 |
| R8 | Pre-specified primary model (A3) and a "promising" bar at 2.80 paired standard errors, with negative and inconclusive branches | Meth W3 |
| R9 | Seed 45 scored once for one pre-registered A′; a failure ends the repair | Meth W4 |
| R10 | LAB as a yes or no gate only; H2 grid pre-registered before H3; LAB checkpoints excluded by hash | Meth W6 |
| R11 | Transfer bar restated in nested-score units without the halving (g* rule) | Meth W7, Q3 |
| R12 | L5 mandatory plus a fixed-τ run, if "no fit" is to stop the repair | Meth W8, Q2(c) |

**B. Desirable** (the methodology card's "Merely desirable" list).

| Ref | Change | Card |
|---|---|---|
| S8 | Predict seed-45 power from the pilot's paired standard errors; consider a larger seed-45 episode count if power is below 50% | Meth W3, Q5 |
| S13 | Matched-k control (k = 8, 23, 10), cheap and itself a label-free A′ candidate | Meth Q2(d) |
| S14 | A second model seed when a deciding statistic lands within one half-width of its threshold | Meth Q2(e) |
| S15 | More bootstrap resamples at boundary cases, pre-registered with the seed fixed | Meth needed-versus-desirable list |

**C. Raised but placed on neither side of that split by its card** (should_fix or consider). The methodology seat puts S9 and S10 inside the pre-run commit (Meth Q1 step 1; W10). S6 was raised by both seats; it is a one-line ledger edit that keeps seed 45 clean.

| Ref | Change | Card(s) |
|---|---|---|
| S1 | Resolve the H1 label collision (hypothesis versus the K8 run) | EIC W2 |
| S2 | Glossary with pointers; append spec §4 or quote C2 | EIC W3 |
| S3 | Qualify the header's provenance claim (AMI row from E2); support or soften "best"; relabel 33.44 | EIC W4 |
| S4 | One cost line per step, stating what it includes | EIC W5 |
| S5 | Say that the appended handoff proposed O3 and why the leaning differs | EIC W6 |
| S6 | Reassign any 8B probe to a seed other than 45 in the §7 ledger | EIC W7; Meth W9 |
| S7 | Explain the sequence dates in the appendix list | EIC W8 |
| S9 | Joint decision table with a dated stop | Meth W5 |
| S10 | State the bank seed, the fresh-episode count and seed, and the tie rule for collapsed control cells | Meth W10 |
| S11 | Disclose seed 45's painting overlap with earlier looks and H1's post-hoc origin; keep the look count | Meth W11 |
| S12 | Paired A3 minus C0 under the nested score, so a pass can be credited to aspect training | Meth Q5 |

---

## Consensus Analysis

**Counting in a two-seat panel.** The CONSENSUS-4 and CONSENSUS-3 labels are defined over four non-DA reviewers and cannot occur here. The labels below are the role file's lower-count labels over this panel's two seats: a sub-claim both seats raised is a *corroborated finding (2 of 2 seats)*, and one seat's sub-claim is a *single-seat finding*. Silence is not agreement and not dissent. The two seats score disjoint dimensions, and Phase 0 routed rule-changing ambiguities to the methodology seat and reader-facing ones to the Journal-Fit seat, so most findings being single-seat is the expected pattern, not a sign of disagreement. Confidence values are self-reported and were not used to weigh anything.

### Points of Agreement

**Corroborated findings (2 of 2 seats).**
1. **SC-1. The H1 pilot has no complete outcome-to-action rule.** EIC W1 (Major): §5.2 gives only a "promising" reading with no next action, and the actions that depend on it are scattered across §5.1 and §6. Meth W3 (Major): "The pilot reading also has no 'not promising' or 'inconclusive' outcome." Same severity; compatible remedies. Roadmap R1.
2. **SC-2. "Falls short" is undefined.** EIC W1 (Major): it is never defined, and "otherwise" leaves unstated which H1 result sends the project to branch 3. Meth W4 (Major): if it can mean a failed seed-45 test, H4 implies a second GO test. The remedies fit together: EIC asks for the term to be defined inside the H1 rule block, and Meth narrows it to the development pilot. Roadmap R2.
3. **SC-12. Seed 45 is also proposed for the 8B probe** in Appendix A §11. EIC W7 (Minor) and Meth W9 (Minor). Both ask for the probe to take another seed in the ledger. Roadmap S6.
4. **SC-17. H3's readings do not cover every outcome**: a run that is not "no fit" with a term-only gain of at least 3.0 and a lower bound at or below 0 falls under none. Raised by Meth W1 and Q3; the Journal-Fit seat records the same gap as a pointer left to D1 (scope notes) and does not score it. Roadmap R6.

**Corroborated strengths (2 of 2 seats).**
- The GO arithmetic and its identity are sound: EIC S2 (R@1 = (either rate + gain) / 2 defined once and used directly); Meth S4 (re-derived 6.19 and 12.07, matching 6.2 and 12.1).
- The seed and row ledger protects the GO test: EIC S6 (seed ledger and row scope as standing constraints); Meth S1 (held budget 0 of 2, RCA fixed, no rules added after results).
- Reproducibility affordances are concrete: EIC S6 (named functions and scripts); Meth S6 (hash-checked checkpoints, builders, unit tests).
- Weaknesses and risks are disclosed before results: EIC S4 (post-hoc and estimate labels, §8's seven weaknesses); Meth S5 (forking-paths risk and mitigations named).
- Committing H3's rules in advance is right: EIC S3 and Q3; Meth Q3.

**Single-seat findings, Journal-Fit seat** (all D2; none disputed by the methodology seat).
- SC-3 (EIC W1, Major): a reading labelled descriptive serves as a branch condition, in three wordings. R3.
- SC-4 (EIC W2, Minor): "H1" names both the nested-score hypothesis and the K8 run; no rule becomes ambiguous. S1.
- SC-5, SC-6 (EIC W3, Minor): project shorthand without definition or pointer; spec §4 C2 cited but not appended. S2.
- SC-7, SC-8, SC-9 (EIC W4, Minor): three AMI values appear in none of the appendices although the header says every §2 number comes from Appendix A; "best measured gains" cannot be checked; 33.44 is the uniform control's either rate, not the uniform term's (33.48). S3. The methodology seat's statement that "the memo's §2 table matches Appendix A wherever both report a value" concerns values, not labels, and does not conflict with SC-9.
- SC-10 (EIC W5, Minor): two cost figures for the H2 grid. S4.
- SC-11 (EIC W6, Minor): the memo does not say that its appended handoff proposed O3. S5.
- SC-13 (EIC W8, Minor): sequence dates unexplained. S7.
- Strengths: decision-first opening (S1); orders compared on common columns with the leaning's costs stated (S5); §9 questions scoped to the decision (S7).

**Single-seat findings, methodology seat** (all D1; none disputed by the Journal-Fit seat, which deferred D1 questions).
- SC-14, SC-15, SC-16 (Meth W1, Major): H3's deciding scorer is unspecified (A3 is "no fit" under one of the script's two gains and "fits" under the other); the zero-anchored bar has no paired baseline; for LAB, "fits, weak transfer" mostly means a weak fit. R4, R5.
- SC-18 (Meth W2, Major): mean(R@1, gain) = either/4 + 3·gain/4 prefers a cell whose R@1 falls 0.95 below its control. R7.
- SC-19 (Meth W3, Major): "promising" is a best-of-six-or-more look on the pick draw, with margins that predict about a coin flip on seed 45. R8. (The Journal-Fit seat left the leniency question to D1 explicitly.) SC-20, the power prediction, is S8.
- SC-21 (Meth W4, Major): seed 45 is not committed as a single look. R9.
- SC-22 (Meth W5, Minor): no joint decision table and no dated stop. S9.
- SC-23 (Meth W6, Major): the "no fit" branch would select A′'s settings with evaluation labels. R10.
- SC-24 (Meth W7, Major): the 3.0 bar halves an unselected measurement; without the halving the memo's own chain gives about 1.5. R11.
- SC-25 (Meth W8, Minor): L5 optional, no fixed-τ run. R12.
- SC-26 (Meth W10, Minor): bank seed, episode count and seed, and control tie rule unstated. S10.
- SC-27 (Meth W11, Minor): seed 45 shares paintings with every earlier look; H1 came from a profile that included the spent seed-43 draw. S11.
- SC-28 to SC-31 (Meth Q5, Q2(d), Q2(e), desirable list): S12 to S15.
- Strengths: the nested uniform control is the right counterfactual (S2); H3 is fenced off as an upper-bound diagnostic (S3).
- Arithmetic: the methodology seat's re-derivations (GO arithmetic, the mean(R@1, gain) identity, 3 × 0.99 = 2.97, 1.67 at 80% power, 56 cells collapsing to 30 λ sums) and its 13 receipts found no mismatch. Receipts attest auditability only, not correctness.

### Points of Disagreement

None. Both seats were checked for existence, severity and remedy conflicts on every shared sub-claim:
- SC-1 and SC-2: Major in both cards; remedies compatible (see above).
- SC-12: Minor in both cards; same remedy.
- SC-17: raised by one seat and recorded as an unscored pointer by the other.
- The 33.44 label (SC-9) against the methodology seat's §2 value check: no conflict, as noted above.

No arbitration was needed.

### Not Assessed by This Panel

The reduced panel has no domain seat and no Devil's Advocate. These points were not scored, and nothing in this letter should be read as a judgement on them:
- Domain plausibility: whether caption-side CLIP features can carry emotion, whether aspect-block factors are learnable by this architecture beyond what H3 itself would test, and whether the factor-basis mechanism is plausible.
- Novelty of the nested fusion score and positioning against the conditional-similarity and vision-language retrieval literature.
- An adversarial case for branch 3 (stopping the repair now) over any repair order.
- Venue fit (`criteria_binding_unavailable`).
- Data-level re-derivation from stored per-anchor arrays or checkpoints. The memo header's claim that E3's final review re-derived its numbers was treated as an unverified author statement (Phase 0).

---

## Decision Rationale

The decision follows mechanically from the contract. The methodology seat, the only seat eligible for D1, scored methodology_rigor block with block_class repairable. Its block rests on Meth W1: H3's fit rule is anchored at zero and its scorer is unspecified, so the outcome E3 makes most likely, a weak fit, can be read either as "no fit" or as "fits, weak transfer", and those readings lead to different next steps. Meth W2 (the pick criterion), W4 ("falls short" and the single look), W6 (evaluation labels reaching A′'s settings) and W7 (the 3.0 transfer bar) add to it. The Journal-Fit seat, the only seat eligible for D2, scored writing_and_structure warn, driven by EIC W1: the H1 pilot has a reading but no outcome-to-action rule. F2 (severity 90) and F4 (severity 40) fired, and F2 takes precedence.

Why not reject: F1 needs a fatal D1 block, and the methodology seat found nothing showing the seed-45 episodes or the held rows spent. Both seats credit the ledger (EIC S6, Meth S1); their only seed-45 concern is a suggestion in Appendix A for a future probe (EIC W7, Meth W9). Why not minor revision: the block names rule defects that would leave a step uninterpretable or weaken the GO test, which the contract's stage note treats as block-worthy, and a fired action may not be softened.

Major revision here means the rules change and are re-checked before any run, not a long delay. The methodology seat states that no change needs more than a day of the window and most cost no compute. The Journal-Fit items are presentation fixes. Both seats agree on the H1 outcome rules, "falls short" and the seed-45 conflict; the rest rests on one seat each, undisputed.

---

## Required Revisions (Must Fix)

Severity, evidence anchor and confidence are transported unchanged from the cards. Consequence codes come from the roadmap schema's closed set.

| Transport ref | Revision Item | Sub-Claim(s) | Severity | Evidence Anchor | Confidence | Source Reviewer | Obligation class | Cost scope | Bounded consequence |
|---|---|---|---|---|---|---|---|---|---|
| R1 | H1 pilot rule block covering every outcome, each with a next action | SC-1 | major (EIC W1); major (Meth W3) | text: §6 table, O2 row "H3 only if H1 falls short"; "if H1 looks promising on seed 42 we may skip H3" | 4 (EIC W1); 4 (Meth W3) | EIC W1; Meth W3 | must_fix | section: §5.2 Pilot reading | interpretive_ambiguity_remains; section §5.2 |
| R2 | Define "falls short" as a development-pilot outcome | SC-2 | major (EIC W1); major (Meth W4) | text: §4 "the H1 score on an H2-repaired model, if H1 alone falls short" | 4 (EIC W1); 4 (Meth W4) | EIC W1; Meth W4 | must_fix | sentence: §4 H4 and §6 O2 row | interpretive_ambiguity_remains; section §4 and §6 |
| R3 | Same outcome words in §5.1, §5.2 and §6 | SC-3 | major (EIC W1) | text: §6 table, O2 row "H3 only if H1 falls short"; "if H1 looks promising on seed 42 we may skip H3" | 4 (EIC W1) | EIC W1 | must_fix | sentence: §5.1, §5.2, §6 | reader_traceability_reduced; section §5.1, §5.2, §6 |
| R4 | One deciding scorer and a fixed episode count for H3 | SC-14 | major (Meth W1) | text: §5.1 "the gain on fresh label episodes over scorer-train rows (a new episode seed), with the" | 5 (Meth W1) | Meth W1 | must_fix | section: §5.1 Measurements | interpretive_ambiguity_remains; section §5.1 |
| R5 | Paired A3 and C0 baselines; fit decided on the paired difference | SC-15, SC-16 | major (Meth W1) | text: §5.1 "the gain on fresh label episodes over scorer-train rows (a new episode seed), with the" | 5 (Meth W1) | Meth W1 | must_fix | section: §5.1 Measurements and outcome rules | interpretive_ambiguity_remains; section §5.1 |
| R6 | H3 readings exhaustive, with an inconclusive band | SC-17 | major (Meth W1) | text: §5.1 "the gain on fresh label episodes over scorer-train rows (a new episode seed), with the" | 5 (Meth W1) | Meth W1; EIC scope-note pointer | must_fix | section: §5.1 Proposed outcome rules | interpretive_ambiguity_remains; section §5.1 |
| R7 | Pick rule aligned with the binding R@1 comparison | SC-18 | major (Meth W2) | text: §5.2 "parity halves, mean(R@1, gain) per half, as in E3" | 5 (Meth W2) | Meth W2 | must_fix | section: §5.2 Cross-fitting | acceptance_criterion_unmet; claim: A′ GO test on seed 45 |
| R8 | Primary model fixed; promising bar at 2.80 paired SEs; negative and inconclusive branches | SC-19 | major (Meth W3) | text: §5.2 "control on seed 42 on both R@1 and gain (lower bounds above 0)" | 4 (Meth W3) | Meth W3 | must_fix | section: §5.2 Models and Pilot reading | acceptance_criterion_unmet; claim: A′ GO test on seed 45 |
| R9 | Seed 45 is a single look; a failure ends the repair | SC-21 | major (Meth W4) | text: §4 "the H1 score on an H2-repaired model, if H1 alone falls short" | 4 (Meth W4) | Meth W4 | must_fix | sentence: §7 Episode seeds | claim_scope_unsupported; claim: A′ GO result |
| R10 | LAB as a yes or no gate only | SC-23 | major (Meth W6) | text: §5.1 "LAB, a fast testbed whose target is known to be learnable, before any pseudo-bank retraining" | 4 (Meth W6) | Meth W6 | must_fix | section: §5.1 No-fit rule and §5.3 | claim_scope_unsupported; claim: label-free method (spec §4 C2) |
| R11 | Transfer bar in nested-score units, no halving | SC-24 | major (Meth W7) | text: §5.1 "gains halved on the fresh draw in E3, fusion keeps only part of a term's gain" | 4 (Meth W7) | Meth W7 | must_fix | section: §5.1 outcome rules and Why 3.0 | interpretive_ambiguity_remains; section §5.1 |
| R12 | L5 mandatory and a fixed-τ run, if "no fit" is to stop the repair | SC-25 | minor (Meth W8) | text: §5.1 "Optionally L5 = A5's settings (β = 0), the one E3 run whose fresh-episode" | 4 (Meth W8) | Meth W8 | must_fix | section: §5.1 Runs | interpretive_ambiguity_remains; section §5.1 |

### Required Item Details

**R1: H1 pilot rule block covering every outcome**
- **Problem**: §5.2 gives the H1 pilot only a "promising" reading, labelled descriptive, with no next action and no negative or inconclusive branch. The actions that depend on it sit in §5.1 and §6 in different words, so only O1's first step has a complete rule chain.
- **Source**: EIC W1 ("as written the rules are complete only for the leaning's first step"); Meth W3 ("The pilot reading also has no 'not promising' or 'inconclusive' outcome").
- **Requirement**: Add an H1 rule block parallel to §5.1 in which promising, not promising and inconclusive each have a measured condition, a threshold, a reading and a next action, including when H3 is skipped and when branch 3 fires (EIC W1). Thresholds per R8; next actions per the joint table (S9).
- **Acceptance criteria**: §5.2 maps every possible pilot outcome to exactly one next action, and that block is in the pre-run hashed commit.

**R2: Define "falls short"**
- **Problem**: "falls short" (§4 H4; §6 O2 row) is never defined, so it is unclear which H1 result triggers H3, H4 or branch 3, and whether it can mean a failed seed-45 test.
- **Source**: EIC W1; Meth W4.
- **Requirement**: State that "falls short" refers to the development pilot only (Meth W4), using R1's outcome labels (EIC W1).
- **Acceptance criteria**: every use of "falls short" in §4 and §6 is defined as a named R1 pilot outcome on seed 42 and none refers to a seed-45 result.

**R3: One set of outcome words**
- **Problem**: a reading labelled descriptive in §5.2 serves in §5.1 and §6 as a branch condition, in three wordings.
- **Source**: EIC W1.
- **Requirement**: Use the same outcome words in §5.1, §5.2 and §6, and do not label a reading descriptive if it routes a decision.
- **Acceptance criteria**: §5.1, §5.2 and §6 use identical outcome labels and no reading labelled descriptive appears as a branch condition.

**R4: One deciding scorer and a fixed episode count for H3**
- **Problem**: the in-distribution gain uses "the procedure of `train_fit_diagnostic.py`", which reports two gains. On E3's fresh pseudo-aspect episodes, A3 is "no fit" under the cross-fitted agreement rule (0.36 [−0.12, 0.86]) and "fits" under the training score (0.95 [0.28, 1.62]).
- **Source**: Meth W1, Q2(b).
- **Requirement**: Make the term-only agreement rule (λ = ∞), whose gain replicated across draws (0.99 and 0.97), the one deciding scorer. Report the cross-fitted agreement rule and the training score as secondary. Use 4,096 fresh episodes per pair at a stated seed.
- **Acceptance criteria**: §5.1 names exactly one deciding scorer and states the episode count and seed.

**R5: Paired baselines for H3's fit reading**
- **Problem**: a fit bar anchored at zero cannot tell "labels help this architecture" from "labels reproduce E3's level"; for LAB, "fits, weak transfer" mostly means a weak fit, which is the "no fit" reading under another name.
- **Source**: Meth W1, Q2(a), Q3.
- **Requirement**: Score A3 and C0 on the same fresh label episodes and the same seed-42 episodes as each LAB run, and decide on the paired difference (LAB run minus A3, painting-clustered). Per Meth Q3: a run fits if the paired term-only gain over A3 has a lower bound above 0, shows no fit if its point estimate is at or below 0, and is otherwise inconclusive (a second model seed, about 10 minutes). The 5% loss bar becomes descriptive only.
- **Acceptance criteria**: H3's fit rule in the commit is a paired comparison against A3 on identical episodes and the loss level no longer decides.

**R6: H3 readings that cover every outcome**
- **Problem**: a run that is not "no fit" with a seed-42 term-only gain of at least 3.0 and a lower bound at or below 0 falls under none of the three readings.
- **Source**: Meth W1 and Q3; EIC scope notes (pointer, left to D1).
- **Requirement**: Make the readings exhaustive and mutually exclusive, with an inconclusive band (Meth Q3's proposed H3 rules give one form).
- **Acceptance criteria**: every combination of H3 measurements maps to exactly one reading and one next action.

**R7: Pick rule aligned with the binding comparison**
- **Problem**: the criterion mean(R@1, gain) equals either/4 + 3·gain/4. It strictly prefers a cell that adds 1.0 point of gain and loses 2.9 points of either rate, although that cell's R@1 falls 0.95 below the control's, and the R@1 comparison against the nested control is the one that binds a GO.
- **Source**: Meth W2.
- **Requirement**: The nested control maximises R@1; the nested score maximises min(R@1 minus the R@1 of the control tuned on the same half, gain), with the first cell in a stated order winning ties. The same rule is reused unchanged for the A′ pick (Meth Q5).
- **Acceptance criteria**: §5.2 states a pick rule whose trade-off matches the GO, with a tie order, and the A′ pre-registration reuses it unchanged.

**R8: Pre-specified primary model and a promising bar tied to seed-45 power**
- **Problem**: "the best run" is a maximum over six or more models on the draw where A3 was already picked, and E3's picked gain halved on the fresh draw (0.52 to 0.26). Margins that just clear 0 correspond to about a 50% chance of clearing on a fresh draw before any shrinkage.
- **Source**: Meth W3, Q5.
- **Requirement**: Pre-specify A3 as the primary model, with A1, A2, A4, A5, A6, C0 and SE as descriptive rows, or cross-fit the model choice together with λ. Promising: both point margins (R@1 against the nested control, and gain) are at least 2.80 times their paired standard errors. Not promising: either point margin is at or below 0. Inconclusive: anything else.
- **Acceptance criteria**: §5.2 names the primary model and states numeric promising, not-promising and inconclusive thresholds before any pilot number exists.

**R9: Seed 45 is a single look**
- **Problem**: §7 reserves seed 45 "for the A′ GO test" but does not say A′ is tested once or what follows a failure. A second GO test on another fresh draw roughly doubles the chance of a false GO, with no correction stated.
- **Source**: Meth W4.
- **Requirement**: Seed 45 is scored once, for one pre-registered A′, by Oct 12 at the latest (Meth joint table); a failure ends the repair; any further fresh seed needs an error split committed before the first test.
- **Acceptance criteria**: §7 states the single-look rule and the stop-on-failure rule.

**R10: LAB as a yes or no gate only**
- **Problem**: on the "no fit" path, H2's τ, β or schedule would be tuned on label episodes of the evaluation aspects and carried into A′, which §7 rules out for the method and which would weaken the label-free claim.
- **Source**: Meth W6.
- **Requirement**: Pre-register the H2 grid before H3 runs. LAB answers only whether any setting in that fixed family fits labels. The whole grid trains on pseudo banks, and the A′ cell is picked by the pseudo-bank fit gate and the seed-42 criterion. LAB checkpoints are excluded from the A′ candidate set by a hash list, and LAB's role is disclosed.
- **Acceptance criteria**: §5.1 and §5.3 leave no path by which a setting chosen on LAB enters A′, and the hash exclusion list is in the commit.

**R11: Transfer bar in nested-score units**
- **Problem**: the 3.0 bar applies E3's winner's-curse halving (0.52 to 0.26, a picked cross-fitted value) to an unselected fixed-λ measurement whose gain did not shrink between draws (0.99 against 0.97). Without the halving, the memo's own chain gives about 1.5.
- **Source**: Meth W7, Q3.
- **Requirement**: State the ceiling in GO units: the best fitting run's cross-fitted nested-score gain on seed 42 against its nested control is too low if it is below g* = max(5.6 × SE_R, 2.8 × SE_g), with SE_R and SE_g taken from A3's paired standard errors in the pilot (about 0.94 with E3's half-widths). Report the term-only seed-42 gain beside it. This also settles which end of §3's "roughly 0.7 to 1.0" the memo means; the methodology seat found that range runs from a coin flip (0.66) to about 80% power (0.94).
- **Acceptance criteria**: §5.1's transfer rule is stated in nested-score units with its formula fixed before the run and no halving applied to an unselected measurement.

**R12: L5 and a fixed-τ run, if "no fit" is to stop the repair**
- **Problem**: L5 is optional and no fixed-τ run is planned, so "no fit" may reflect settings tuned for pseudo banks, as the memo's own §8 item 3 warns.
- **Source**: Meth W8, Q2(c); listed as needed in the methodology card on that condition.
- **Requirement**: Make L5 mandatory and add one fixed-τ run in the same GPU lock; "no fit" then requires every LAB run to fail.
- **Acceptance criteria**: §5.1 lists L3, L5 and the fixed-τ run as required and the "no fit" rule quantifies over all of them.

---

## Suggested Revisions (Should Fix or Consider)

| Transport ref | Revision Item | Sub-Claim(s) | Severity | Evidence Anchor | Confidence | Source Reviewer | Obligation class | Cost scope | Bounded consequence |
|---|---|---|---|---|---|---|---|---|---|
| S1 | Rename the nested-score hypothesis or add a note in §1 that Appendix A's H1 is the K8 run | SC-4 | minor (EIC W2) | text: §2 table and §5.2 heading "K8 (H1 run, no genre partition)"; "H1: nested-score pilot (a development look)" | 5 (EIC W2) | EIC W2 | should_fix | sentence: §1 and §2 table | reader_traceability_reduced; section §5.2 |
| S2 | Glossary at the end of §1 with pointers into Appendix A; append spec §4 or quote C2 | SC-5, SC-6 | minor (EIC W3) | text: §2 table and §4 H3 "Term-only paired gain, A3 minus SE / minus C0"; "which spec §4 C2 forbids" | 4 (EIC W3) | EIC W3 | should_fix | section: §1 | reader_traceability_reduced; manuscript §0 to §9 |
| S3 | Qualify the header (AMI row from E2, cite E2); give λ = 4 and 16 or soften "best"; label 33.44 as the uniform control | SC-7, SC-8, SC-9 | minor (EIC W4) | text: header and §2 AMI row "comes from the E3 report (Appendix A), whose final review re-derived it from stored per-anchor arrays"; "(E2, scorer-train rows)" | 5 (EIC W4) | EIC W4 | should_fix | sentence: header, §2 table, §3 | reporting_requirement_unmet; table §2 |
| S4 | One cost line per step, stating what it includes | SC-10 | minor (EIC W5) | text: §6 table O4 row and cost paragraph "about 1.5 to 2 days including a pre-registration"; "including its pre-registration, under an hour of GPU" | 4 (EIC W5) | EIC W5 | should_fix | sentence: §6 | interpretive_ambiguity_remains; table §6 |
| S5 | One sentence naming the handoff's order (O3) and the reason for the leaning | SC-11 | minor (EIC W6) | text: Appendix B, Suggested first steps "Pilot H1 on seed-42 episodes with the existing A3 (and A1, A4, A5) checkpoints"; "In parallel, the H3 label-trained diagnostic on CPU or a short GPU run" | 4 (EIC W6) | EIC W6 | should_fix | sentence: §0 or §6 | reporting_requirement_unmet; section §0 |
| S6 | §7: any 8B probe takes a seed other than 45 | SC-12 | minor (EIC W7); minor (Meth W9) | text: Appendix A §11 "on a fresh episode seed (for example 45; seed 44 has been scored by both 2B runs)" | 3 (EIC W7); 5 (Meth W9) | EIC W7; Meth W9 | should_fix | sentence: §7 Episode seeds | acceptance_criterion_unmet; section §7 |
| S7 | One clause in the appendix list explaining the sequence dates | SC-13 | minor (EIC W8) | text: memo §0 and Appendix A first line "ended NO-GO on 2026-10-03 (Appendix A)"; "Date: 2026-11-01 (sequence date of plan step E3" | 4 (EIC W8) | EIC W8 | consider | sentence: Appendices list | reader_traceability_reduced; section Appendices |
| S8 | Predict seed-45 power from the pilot's paired SEs at half the observed margins; if below 50%, consider a larger pre-registered seed-45 episode count, checked first by subsampling seed 42 | SC-20 | major (Meth W3) | text: §5.2 "control on seed 42 on both R@1 and gain (lower bounds above 0)" | 4 (Meth W3) | Meth W3, Q5 | should_fix | section: §5.2 and the A′ pre-registration | evidence_gap_remains; claim: A′ GO test on seed 45 |
| S9 | Joint decision table mapping each H1 and H3 outcome to one next action, with a dated stop, committed with the rules | SC-22 | minor (Meth W5) | absence: memo §5 to §6 — expected a joint table that maps each H1-pilot outcome and each H3 outcome to one next action with a dated stop; checked §0 aims, §5.1 outcome rules, §5.2 pilot reading, §5.3, §6 table and its notes, §7, §8 item 7 | 4 (Meth W5) | Meth W5 | should_fix | section: new table after §6 | interpretive_ambiguity_remains; section §6 |
| S10 | State the bank seed, the fresh-episode count and seed, and the tie rule for the collapsed control cells, in the rules commit | SC-26 | minor (Meth W10) | absence: memo §5.1 Bank LAB and Measurements, and §5.2 Cross-fitting — expected the bank seed, the count and seed of the fresh label episodes, and a tie rule for the 56 cells that collapse to 30 control scores; checked §5.1, §5.2, the §6 cost note, §7, Appendix B Evaluation rules and Code entry points | 4 (Meth W10) | Meth W10 | should_fix | sentence: §5.1 and §5.2 | method_reproducibility_unresolved; section §5.1 |
| S11 | Disclose seed 45's painting overlap with earlier looks and H1's origin in a profile that included seed 43; keep the look count in the ledger | SC-27 | minor (Meth W11) | text: §8 "only the seed-45 test controls that, and only for the final claim" | 4 (Meth W11) | Meth W11 | should_fix | section: §8 and the later paper | claim_scope_unsupported; claim: A′ GO result |
| S12 | Paired A3 minus C0 comparison under the nested score | SC-28 | none (question-channel item) | none (Meth Q5 answer) | none | Meth Q5 | should_fix | section: §5.2 Outputs | claim_scope_unsupported; claim: credit to aspect training |
| S13 | Matched-k control run (k = 8, 23, 10) in the same lock | SC-29 | none (question-channel item) | none (Meth Q2(d) answer) | none | Meth Q2(d) | consider | section: §5.1 Runs | evidence_gap_remains; section §5.1 |
| S14 | Second model seed when a deciding statistic lands within one half-width of its threshold | SC-30 | none (question-channel item) | none (Meth Q2(e) answer) | none | Meth Q2(e) | consider | section: §5.1 | evidence_gap_remains; section §5.1 |
| S15 | More bootstrap resamples at boundary cases, pre-registered with the seed fixed | SC-31 | none (question-channel item) | none (Meth "Needed versus merely desirable" list) | none | Meth Review Body | consider | sentence: §7 Evaluation rules | evidence_gap_remains; section §7 |

---

## Revision Roadmap

### Source-traceability checklist

Immutable source order: seat order (EIC, then R1 methodology), then findings before question answers, then ordinal, then sub-claim. An item both seats raised sits at its earliest raising reference. This order is not a work order and implies no priority; the author chooses `will_address`, `wont_address` or `not_on_point` later in the separate author-adjudication step. The machine `revision-roadmap/1.0` core is not emitted with this letter, because no block manifest was bound for the memo and its `proposed_targets` cannot be filled without inventing block ids. A machine build that sorts by ordinal alone may interleave question-channel items; that would move only S12 to S15.

- [ ] REV-1, R1 (obligation `must_fix`): H1 pilot rule block covering every outcome (EIC W1; Meth W3)
- [ ] REV-2, R2 (obligation `must_fix`): define "falls short" as a development-pilot outcome (EIC W1; Meth W4)
- [ ] REV-3, R3 (obligation `must_fix`): one set of outcome words across §5.1, §5.2, §6 (EIC W1)
- [ ] REV-4, S1 (obligation `should_fix`): resolve the H1 label collision (EIC W2)
- [ ] REV-5, S2 (obligation `should_fix`): glossary with pointers; spec §4 C2 (EIC W3)
- [ ] REV-6, S3 (obligation `should_fix`): header provenance, "best", the 33.44 label (EIC W4)
- [ ] REV-7, S4 (obligation `should_fix`): one cost line per step (EIC W5)
- [ ] REV-8, S5 (obligation `should_fix`): name the handoff's order and the reason for the leaning (EIC W6)
- [ ] REV-9, S6 (obligation `should_fix`): keep any 8B probe off seed 45 (EIC W7; Meth W9)
- [ ] REV-10, S7 (obligation `consider`): explain the sequence dates (EIC W8)
- [ ] REV-11, R4 (obligation `must_fix`): one deciding scorer and a fixed episode count for H3 (Meth W1)
- [ ] REV-12, R5 (obligation `must_fix`): paired A3 and C0 baselines for the fit reading (Meth W1)
- [ ] REV-13, R6 (obligation `must_fix`): exhaustive H3 readings with an inconclusive band (Meth W1; EIC pointer)
- [ ] REV-14, R7 (obligation `must_fix`): pick rule aligned with the binding R@1 comparison (Meth W2)
- [ ] REV-15, R8 (obligation `must_fix`): primary model fixed; promising bar at 2.80 paired SEs; all branches (Meth W3)
- [ ] REV-16, S8 (obligation `should_fix`): predict seed-45 power before the pre-registration (Meth W3)
- [ ] REV-17, R9 (obligation `must_fix`): seed 45 is a single look; a failure ends the repair (Meth W4)
- [ ] REV-18, S9 (obligation `should_fix`): joint decision table with a dated stop (Meth W5)
- [ ] REV-19, R10 (obligation `must_fix`): LAB as a yes or no gate only (Meth W6)
- [ ] REV-20, R11 (obligation `must_fix`): transfer bar in nested-score units (Meth W7)
- [ ] REV-21, R12 (obligation `must_fix`): L5 and a fixed-τ run, if "no fit" is to stop the repair (Meth W8)
- [ ] REV-22, S10 (obligation `should_fix`): reproducibility parameters in the commit (Meth W10)
- [ ] REV-23, S11 (obligation `should_fix`): disclose painting overlap and H1's origin (Meth W11)
- [ ] REV-24, S12 (obligation `should_fix`): A3 minus C0 under the nested score (Meth Q5)
- [ ] REV-25, S13 (obligation `consider`): matched-k control (Meth Q2(d))
- [ ] REV-26, S14 (obligation `consider`): second model seed near a threshold (Meth Q2(e))
- [ ] REV-27, S15 (obligation `consider`): more bootstrap resamples, pre-registered (Meth Review Body)

---

## Journal-Supplied Deadline (Optional Transport)

- **Exact deadline from source letter**: NOT PROVIDED. The memo's own dates (branch decision Oct 9, repair past about Oct 12 costing paper time) are author context, not a review deadline.

---

## Response Letter Instructions

Please respond to every item using the format in `templates/revision_response_template.md`.

**Must include**:
1. A response and change description for each Required Revision (R1 to R12).
2. A response for each Suggested Revision (S1 to S15): adopted, or the reason for not adopting it.
3. The revised memo with changes marked, and the hash of the pre-run commit that fixes the rules.
4. A cross-reference table from each R and S item to the revised section or rule.

---

## Closing

We ask you to revise the memo's decision rules as set out above and to submit the revised rules for a scoped re-check before the LAB bank is built or any H1 pilot number exists. Both seats found the memo's framing, ledger and GO arithmetic sound. The methodology seat judges that the required changes fit within about a day of the window and that most of them need no compute.

---

## Appendix: Full Reviewer Reports and Sub-Claim Inventory

The two complete Phase 2 cards are attached by reference in the same folder: `phase2_eic.md` (Journal-Fit Reviewer) and `phase2_methodology.md` (Peer Reviewer 1, methodology, with its 13 arithmetic receipts). Their Phase 1 commitments are `phase1_eic.md` and `phase1_methodology.md`; the panel configuration is `phase0_reviewer_configuration.md`.

### Sub-claim inventory (Step 1b, compact form)

Positions: raised, corroborated, not-mentioned (silence), disputed. Severity and confidence are transported from the parent finding. No sub-claim is disputed.

| SC | Parent finding(s) | EIC position | Meth position | Severity | Confidence | Disposition | Item |
|---|---|---|---|---|---|---|---|
| SC-1 | H1 pilot lacks a complete outcome-to-action rule | raised (W1) | raised (W3) | major / major | 4 / 4 | corroborated 2 of 2 | R1 |
| SC-2 | "Falls short" undefined | raised (W1) | raised (W4) | major / major | 4 / 4 | corroborated 2 of 2 | R2 |
| SC-3 | Descriptive reading used as a branch condition; wording differs | raised (W1) | not-mentioned | major | 4 | single-seat | R3 |
| SC-4 | "H1" names two things | raised (W2) | not-mentioned | minor | 5 | single-seat | S1 |
| SC-5 | Shorthand without definition or pointer | raised (W3) | not-mentioned | minor | 4 | single-seat | S2 |
| SC-6 | Spec §4 C2 cited but not appended | raised (W3) | not-mentioned | minor | 4 | single-seat | S2 |
| SC-7 | Header provenance claim fails for the AMI row | raised (W4) | not-mentioned | minor | 5 | single-seat | S3 |
| SC-8 | "Best measured gains" cannot be checked | raised (W4) | not-mentioned | minor | 5 | single-seat | S3 |
| SC-9 | 33.44 labelled as the uniform term | raised (W4) | not-mentioned (its §2 value check covers values) | minor | 5 | single-seat | S3 |
| SC-10 | Two H2 cost figures disagree | raised (W5) | not-mentioned | minor | 4 | single-seat | S4 |
| SC-11 | Handoff's different first step not disclosed | raised (W6) | not-mentioned | minor | 4 | single-seat | S5 |
| SC-12 | Seed 45 also proposed for the 8B probe | raised (W7) | raised (W9) | minor / minor | 3 / 5 | corroborated 2 of 2 | S6 |
| SC-13 | Sequence dates unexplained | raised (W8) | not-mentioned | minor | 4 | single-seat | S7 |
| SC-14 | H3's deciding scorer unspecified | not-mentioned | raised (W1) | major | 5 | single-seat | R4 |
| SC-15 | Zero-anchored fit bar without a paired baseline | not-mentioned | raised (W1) | major | 5 | single-seat | R5 |
| SC-16 | For LAB, "fits, weak transfer" mostly means a weak fit | not-mentioned | raised (W1) | major | 5 | single-seat | R5 |
| SC-17 | H3 readings not exhaustive | corroborated (scope-note pointer, unscored) | raised (W1, Q3) | major | 5 | corroborated 2 of 2 | R6 |
| SC-18 | Pick criterion against the binding R@1 comparison | not-mentioned | raised (W2) | major | 5 | single-seat | R7 |
| SC-19 | Lenient promising bar; no pre-specified model | not-mentioned (deferred to D1) | raised (W3) | major | 4 | single-seat | R8 |
| SC-20 | Seed-45 power not predicted before pre-registration | not-mentioned | raised (W3, Q5) | major | 4 | single-seat | S8 |
| SC-21 | Seed 45 not committed as a single look | not-mentioned | raised (W4) | major | 4 | single-seat | R9 |
| SC-22 | No joint decision table, no dated stop | not-mentioned | raised (W5) | minor | 4 | single-seat | S9 |
| SC-23 | LAB-tuned settings would carry evaluation labels into A′ | not-mentioned | raised (W6) | major | 4 | single-seat | R10 |
| SC-24 | 3.0 bar applies a halving that does not belong | not-mentioned | raised (W7) | major | 4 | single-seat | R11 |
| SC-25 | L5 optional, no fixed-τ run | not-mentioned | raised (W8) | minor | 4 | single-seat | R12 |
| SC-26 | Reproducibility parameters unstated | not-mentioned | raised (W10) | minor | 4 | single-seat | S10 |
| SC-27 | Seed 45 shares paintings; H1's post-hoc origin | not-mentioned | raised (W11) | minor | 4 | single-seat | S11 |
| SC-28 | A3 minus C0 under the nested score missing | not-mentioned | raised (Q5) | none | none | single-seat | S12 |
| SC-29 | Matched-k control | not-mentioned | raised (Q2(d)) | none | none | single-seat | S13 |
| SC-30 | Second model seed near a threshold | not-mentioned | raised (Q2(e)) | none | none | single-seat | S14 |
| SC-31 | More bootstrap resamples at boundary cases | not-mentioned | raised (Review Body list) | none | none | single-seat | S15 |

## Attachment: Acronym Check (advisory, #849)

### Acronym check (advisory; no reply needed)
Coverage: body (partial)
Not in this input: English abstract, Chinese abstract.

| Scope | Line | Rule | Acronym | Uses |
|---|---|---|---|---|
| Body | 8 | Not defined | GO | 80 |
| Body | 8 | Not defined | NO | 8 |
| Body | 15 | Not defined | CVPR | 5 |
| Body | 40 | Not defined | CLIP | 16 |
| Body | 40 | Not defined | ViT | 3 |
| Body | 41 | Not defined | ReLU | 3 |
| Body | 49 | Not defined | RCA | 17 |
| Body | 112 | Not defined | CPU | 10 |
| Body | 132 | Not defined | AMI | 3 |
| Body | 146 | Not defined | LAB | 4 |
| Body | 153 | Not defined | GPU | 11 |
| Body | 186 | Not defined | SHA | 4 |
| Body | 220 | Not defined | MLLM | 24 |
| Body | 229 | Not defined | NaN | 8 |
| Body | 370 | Not defined | AI | 5 |
| Body | 370 | Not defined | AIC | 2 |
| Body | 371 | Not defined | IC | 3 |
| Body | 408 | Not defined | GiB | 3 |
| Body | 408 | Not defined | RTX | 2 |
| Body | 534 | Not defined | KISSME | 1 |
| Body | 846 | Not defined | VL | 5 |
| Body | 955 | Not defined | JSON | 2 |
| Body | 960 | Not defined | NMF | 1 |
| Body | 960 | Not defined | PCA | 1 |
| Body | 960 | Not defined | SAE | 1 |
| Body | 960 | Not defined | SpLiCE | 1 |
| Body | 993 | Not defined | DAS6 | 3 |
| Body | 993 | Not defined | GB | 1 |
| Body | 1271 | Not defined | OK | 1 |
| Body | 1386 | Not defined | DA | 1 |
| Body | 1392 | Not defined | CUB | 2 |
