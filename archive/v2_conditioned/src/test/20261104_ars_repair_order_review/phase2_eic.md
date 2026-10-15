contract_role: eic
## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: writing_and_structure
score: warn
trigger: "The commitments can be recovered but only with effort: terms left undefined at first use or renamed between sections, outcome-to-action mappings scattered across several places, memo claims without pointers to the appended evidence they rely on"

## Review Body

This seat scores D2 only. It judges the memo (§0 to §9) as a method-focused decision document, not against a venue template, and no criteria binding was supplied. The memo follows the conventions of that genre closely. §0 opens with the decision question, three numbered aims, the time window, the fallback and the user's leaning. §1 defines the task, metrics and GO test. §2 and §3 give the evidence and the arithmetic, §4 and §5 the hypotheses and designs, and §6 compares four orders on common columns. §7 fixes the constraints, §8 lists the memo's own weaknesses and §9 asks scoped questions. The appendices are clearly marked as verbatim evidence. A reader with an hour can find the question, the leaning, the H3 rules and the constraints.

Some commitments can be recovered only with effort, which is the warn condition of my Phase 1 plan. The H1 pilot is the first step of two of the four orders and the follow-on step of O1, yet it has a reading and no outcome-to-action rule; its next steps are spread over §5.1 and the §6 table in three different wordings (W1). The label H1 names two different things (W2). Several project terms are used in the memo body without definition or pointer (W3). The header's provenance claim does not hold for one §2 row (W4). The remaining items are small. No commitment is missing outright or stated in conflicting forms, so the block trigger does not apply. The absence of results played no part in the score.

### S1: Decision-first opening with aims, window, fallback and leaning
§0 states the question, three numbered aims, the six-day window with the date after which days come out of the paper schedule, the branch-3 fallback and the user's leaning, then points to §6, §8 and §9. That is the order a decision document should follow.
**Evidence Anchor**: text: §0 aim 1 "the first step yields the most decision-relevant information per day"

### S2: Core vocabulary defined once, with the identity the arithmetic uses
§1 defines R@1, the other-aspect rate, condition gain, the either rate, the uniform-weight control and cross-fitting at first use, and states R@1 = (either rate + gain) / 2. §3's GO arithmetic then uses that identity directly, so the reader can follow the numbers without the appendix.
**Evidence Anchor**: text: §1 Metrics "it is exactly 0 for any scorer that ignores the condition"

### S3: The H3 rules are written as a fixed-in-advance decision record
§5.1 gives each H3 outcome a measured condition, a threshold, a reading and a next action, says the rules are committed before the run, and justifies the 3.0 threshold with a pointer back to §3.
**Evidence Anchor**: text: §5.1, Why 3.0 bullet "a GO with a nested score needs a fused gain of roughly 1 point on fresh episodes (Section 3)"

### S4: Status, post-hoc and estimate labels are explicit
The header says nothing in §5 to §7 has run. §2 labels the two E3 limits as post-hoc and outside the decision map, §6 labels its costs as unmeasured estimates, and §8 lists seven weaknesses, several of which cut against the leaning.
**Evidence Anchor**: text: §6 cost paragraph "Cost estimates (ours, from E2 and E3 timings, not measured for these steps)"

### S5: The orders are compared on common columns, and the leaning's costs are stated
§6 gives every order the same five columns, records O1's main risk as delaying the cheapest test, and the paragraph below the table concedes a dependency that favours having the H1 scorer first. The leaning is disclosed as the user's and is not argued for beyond the evidence.
**Evidence Anchor**: text: §6 table, O1 row "delays the cheapest test (H1) by a day"

### S6: A clear boundary between memo and evidence, with reproducibility pointers
Each appendix is marked verbatim with its source path. §5 names the functions and scripts each design reuses (build_episode_bank, train_fit_diagnostic.py, zfuse). §7 states the seed ledger and the row scope as standing constraints.
**Evidence Anchor**: text: Appendix A heading "The E3 report (verbatim; headings demoted one level)"

### S7: The §9 questions are scoped to the decision
Each question maps to a section (§6, §5.1, §7, §5.2), and an answer to any of them would change the plan: the order, H3's discriminating power, rule commitment, validity of the later test, or the pilot design.
**Evidence Anchor**: text: §9 Q2 "Can H3 as designed discriminate its three readings?"

### W1: The H1 pilot has a reading but no outcome-to-action rule, and its next steps are scattered
§5.1 gives H3 a complete rule block. §5.2 gives the H1 pilot only a "promising" reading, labelled descriptive, with no next action. The actions that depend on the H1 outcome sit elsewhere, in different words. The §6 O2 row has "H3 only if H1 falls short" and "if H1 looks promising on seed 42 we may skip H3". §5.1's weak-transfer branch has "H1 on the existing checkpoints is the only short path; otherwise branch 3". "Falls short" is never defined, and "otherwise" leaves unstated which H1 result sends the project to branch 3. A reader can trace O1's first step from measurement to action in one place. The same reader cannot do that for the first step of O2 or O3, or for the H1 step that O1 itself can lead to. §0 asks "which step to run first, and under which decision rules", and as written the rules are complete only for the leaning's first step. The memo is also inconsistent with itself here: a reading labelled descriptive serves in §6 as a branch condition. Fix: add an H1 rule block parallel to §5.1, with promising and not promising each given a reading and a next action, including when H3 is skipped and when branch 3 fires. Then use the same outcome words in §5.1, §5.2 and §6. Whether the missing rule would change a decision, and whether the "promising" bar is lenient, belongs to D1.
**Severity**: Major
**Evidence Anchor**: text: §6 table, O2 row "H3 only if H1 falls short"; "if H1 looks promising on seed 42 we may skip H3"
**Confidence**: 4 (decision-document review; the gap is visible in the text, but the authors may have meant the pilot to stay rule-free until the A′ pre-registration)

### W2: The label H1 names two different things
The memo uses H1 for the nested-score hypothesis (§0, §4, §5.2, §6, §8). In its own §2 table it also uses H1 for the K8 training run on the AI bank. Appendix A uses H1 only for the K8 run: in the grid table, the pick text of its §4, the winner's-curse paragraph of its §5, and its §6.2, §6.4 and §7. Appendix B uses both meanings on one page. So a reader who checks §5.2 against Appendix A meets rows such as "H1 (not eligible)" whose gains have nothing to do with the nested score. No rule or ledger entry becomes ambiguous, since the §2 row says "H1 run", so this stays in D2. Fix: rename the hypothesis in the memo (for example N, for nested), or add a one-line note in §1 that Appendix A's H1 is the K8 run.
**Severity**: Minor
**Evidence Anchor**: text: §2 table and §5.2 heading "K8 (H1 run, no genre partition)"; "H1: nested-score pilot (a development look)"
**Confidence**: 5 (every occurrence read in the memo and both appendices)

### W3: Project shorthand is used in the memo body without definition or pointer
§1 defines the core vocabulary well (S2), but the memo body also relies on terms it never defines:
- SE and C0, the baselines behind the "about +1 point" claim that motivates H1.
- A1 to A6, the §5.2 model set.
- K8, E1, E2 and AMI.
- Claim C2, cited as "spec §4 C2". Spec §4 is not appended; Appendix C holds only §3 and §6.
- λ_aspect and λ_swap.
- The MLLM probe that spent seed 44.
- "Held rows".

Each is explained somewhere in Appendix A (mostly its §1, the "Terms" paragraph of its §2, and §7) or in Appendix B, but the memo gives no pointer. A reader of §0 to §9 alone cannot decode them. Fix: add a short glossary at the end of §1 with pointers into Appendix A, and either append spec §4 or quote C2's wording.
**Severity**: Minor
**Evidence Anchor**: text: §2 table and §4 H3 "Term-only paired gain, A3 minus SE / minus C0"; "which spec §4 C2 forbids"
**Confidence**: 4 (the project team knows these terms; the panel reader the memo addresses does not)

### W4: The header's provenance claim does not cover every §2 and §3 number
The header says every number in §2 and §3 comes from Appendix A. The §2 AMI row itself credits E2, and three of its six values (affect against emotion 0.20, caption against emotion 0.06, caption against style 0.06) appear nowhere in Appendices A to C. Image against genre (0.397) and caption against genre (0.161) are in Appendix A §7, and image against style (0.32) is only in Appendix C. These values carry H3's rationale in §4. Separately, §3 calls 0.97 and 1.14 "A3's best measured gains". Appendix A's fixed-λ table omits λ = 4 and 16 and refers to its Figure 4, which does not render here, so the reader cannot check "best". The same §2 table also has a small label slip: 33.44 is the either rate of the cross-fitted uniform control in Appendix A §6.1, and the memo calls it the "uniform term" (the term alone gives 33.48). Fix: qualify the header ("except the AMI row, from the E2 report") and cite E2. Add the λ = 4 and 16 values or soften "best", and label 33.44 as the uniform control.
**Severity**: Minor
**Evidence Anchor**: text: header and §2 AMI row "comes from the E3 report (Appendix A), whose final review re-derived it from stored per-anchor arrays"; "(E2, scorer-train rows)"
**Confidence**: 5 (full-text search of the manuscript for each value)

### W5: Two cost figures for the H2 grid disagree
The §6 table puts O4's first result at about 1.5 to 2 days including a pre-registration. The cost paragraph below it puts the H2 grid at about a day including its pre-registration. The gap may be scoring and reporting time, but the memo does not say so. Aim 1 is information per day, so the time column is part of the decision. Fix: give one cost line per step and say what it includes.
**Severity**: Minor
**Evidence Anchor**: text: §6 table O4 row and cost paragraph "about 1.5 to 2 days including a pre-registration"; "including its pre-registration, under an hour of GPU"
**Confidence**: 4 (both figures read in §6; their intended scope is not stated)

### W6: The memo does not say that its appended handoff suggested a different first step
Appendix B, written the same day, suggests piloting H1 first with H3 in parallel, which is the memo's O3. §0 presents the user's leaning, O1, in bold. §6 lists O3 as one alternative without saying it was the handoff's plan or what evidence moved the leaning away from it. The memo is otherwise even-handed (S5), but a reader answering Q1 should know that the plan of record changed and why. Fix: one sentence in §0 or §6 naming the handoff's order and the reason for the leaning.
**Severity**: Minor
**Evidence Anchor**: text: Appendix B, Suggested first steps "Pilot H1 on seed-42 episodes with the existing A3 (and A1, A4, A5) checkpoints"; "In parallel, the H3 label-trained diagnostic on CPU or a short GPU run"
**Confidence**: 4 (framing judgement; both texts read in full)

### W7: Seed 45 is reserved twice across the memo and its appendix
§7 reserves seed 45 for the A′ GO test. Appendix A §11 proposes seed 45 as its example for the 8B MLLM probe, an option Appendix B lists as not chosen for now, so it is still open. The memo's own ledger is clear. Even so, a reader who later revives the 8B option from Appendix A could spend the A′ test seed. Fix: §7 should say that any 8B probe takes another seed.
**Severity**: Minor
**Evidence Anchor**: text: memo §7 and Appendix A §11 "Seed 45 is reserved for the A′ GO test and is untouched"; "on a fresh episode seed (for example 45; seed 44 has been scored by both 2B runs)"
**Confidence**: 3 (the appendix names 45 as an example, not as a commitment)

### W8: The sequence dates are not explained in the memo
The memo is dated 2026-10-03 and says E3 ended that day. Appendix A is dated 2026-11-01, and its path in the memo's appendix list carries that date. Appendix A's first line explains that 2026-11-01 is a sequence date, and Appendix B explains the sequence, but a memo reader sees the later date first. Fix: add one clause to the appendix list.
**Severity**: Minor
**Evidence Anchor**: text: memo §0 and Appendix A first line "ended NO-GO on 2026-10-03 (Appendix A)"; "Date: 2026-11-01 (sequence date of plan step E3"
**Confidence**: 4 (both dates read; the explanation exists one click away)

### Answers to the §9 questions within this seat's remit
Q1 (framing). The memo frames the choice fairly (S5), with one omission (W6). As drafted, O1 is the only order whose first step already has a complete rule chain. That comes from how the memo was drafted, not from evidence for the order, and the W1 fix removes it. On the memo's own time column and its §6 dependency note, O3 gives both readings in about the same day that O1 takes for H3 alone. Whether H3 first is worth delaying H1 by a day is the methodology seat's question. Whatever order is chosen, the memo should commit rules for every step that order runs first.

Q3 (commitment). Yes, commit the rules before the run. Follow E3's practice: a dated file committed before the LAB bank is built and before any H1 pilot number exists, with its commit hash entered in the §7 ledger. If O2 or O3 is chosen, the H1 rule block from W1 goes into the same commit. Whether the thresholds are sound is a D1 question.

Q5 (presentation). The gap in this seat's remit is the action mapping (W1): §5.2 calls the pilot reading descriptive, yet §6 and §5.1 use it as a branch condition. Whether the reading is too lenient for the §3 margin is a D1 question.

Q2 and Q4 are methodology questions outside this seat.

### Scope and integrity notes
- Left to D1, noted only as pointers: the §5.1 readings do not cover a run that is not "no fit" and has a term-only gain of at least 3.0 whose lower bound is at or below 0; whether the undefined "falls short" changes a decision; and the λ set of the nested control.
- No text in the manuscript addresses the reviewers or asks for a verdict. Appendix B's imperatives ("Read this first", the environment rules, the suggested steps) are addressed to the project's implementing agent. I treated them as evidence of the planned workflow.
- Appendix A's figures and relative links do not resolve in this copy. No memo claim depends on a figure except the "best measured" wording in W4.
- The appendices are about four times as long as the memo. Appendix B repeats the §2 table and H1 to H4 nearly verbatim, and its environment, git and GPU rules have no bearing on the decision. That redundancy sits behind a clearly marked boundary and does not obscure the decision path, so I do not count it against D2.
