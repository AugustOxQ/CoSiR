# CoSiR v2 method repair: ARS methodology-focus review of the repair order

**Report date:** 2026-11-04 (sequence date in this folder; the review ran on 2026-10-03).
**What was reviewed:** the decision memo "Which repair to run first after the E3 NO-GO" (sections 0 to 9), with three
appendices that served as evidence: the [E3 report](2026-11-01_aspect_factor_gonogo.md), the repair handoff and spec
sections 3 and 6. E3's own verdict was not re-reviewed.
**Full record:** `src/test/20261104_ars_repair_order_review/`. It holds the memo, the reviewer configuration, the blind
Phase 1 plans, the Phase 2 cards, the editorial decision, the sprint contract and the review log.

## Verdict

**Major Revision. The one block is repairable, and neither seed 45 nor the held rows had been touched.**
- **D1 (methodology rigor) was at block** (`block_class: repairable`), decided by the methodology seat, the only seat
  eligible for D1.
- **D2 (writing and structure) was at warn**, decided by the Journal-Fit seat, the only seat eligible for D2.
- **Fired conditions: F2** (D1 scores block; major revision, severity 90) **and F4** (D2 at warn or worse; minor
  revision, severity 40). F2 has the higher severity, so its action applied. F1 (a fatal D1 block) did not fire,
  because nothing showed the seed-45 episodes or the held rows spent.
- The methodology seat judged every required change repairable within about one day of the window, most of it rule
  text with no compute.

## Why the review was run

E3, the pre-registered go/no-go of method A, ended NO-GO. The user chose to attempt a method repair, called **A′**,
before falling back to the analysis paper (branch 3). Three candidate first steps were on the table:
- **H1:** a nested test-time score (the "H1 pilot") evaluated on E3's existing checkpoints on the seed-42 selection
  episodes.
- **H2:** a repair of the training fit (a grid over temperature, weights and schedule).
- **H3:** a label-trained learnability diagnostic. It trains E3's architecture and loss on episodes built from the
  evaluation labels of scorer-train rows (bank LAB), to ask whether the architecture can learn the aspects at all.

The user leaned toward running H3 first. A full five-seat panel had already reviewed the plan on 2 and 3 October
([ARS plan review](2026-10-27_ars_plan_review.md)), so the user asked for a methodology-focus check of the repair order
instead of a second full panel.

**Terms used below.** *Condition gain* is R@1 minus the other-aspect rate (the rate at which the retrieved candidate
matches the contrast aspect); it is exactly 0 for any scorer that ignores the condition. The *either rate* is the
fraction of anchors whose top candidate matches either aspect, so either = R@1 + other-aspect rate. The *uniform-weight
control* is the same fusion with the condition-agreement weights replaced by uniform weights, so its gain is 0 by
construction. *Seed 42* is the development episode draw, used for picks and pilots. *Seed 45* is the reserved fresh
draw for the single A′ GO test. Seed 43 was E3's test draw and is spent.

## How the review was run

We used the ARS `academic-paper-reviewer` skill in methodology-focus mode (contract
`reviewer/reviewer_methodology_focus/v2`, baseline v3.20.0, `panel_size` 2).
- **Panel.** Two scoring seats, both confirmed unchanged by the user: the Journal-Fit Reviewer (a research director who
  signs off on experiment decision memos; owner of D2) and Peer Reviewer 1 (a statistician in ML evaluation and
  selective inference; owner of D1, mandatory).
- **Two phases per seat.** Phase 1 committed a scoring plan blind to the memo's content (contract and title metadata
  only). Phase 2, in a fresh context, scored the memo against that plan. The synthesis then applied the contract's
  three mechanical steps: the role-scoped matrix, the failure conditions, and precedence.
- **Checks.** `check_sprint_contract.py` passed on the contract. Both Phase 1 plans passed `check_phase_conformance.py`.
  Both Phase 2 cards passed it and `check_panel_synthesis.py --layer1-only`, and the synthesis passed
  `check_panel_synthesis.py`.
- **Provenance.** Both seats ran on one model family, each in its own fresh subagent context, neither seeing the other's
  output before committing its card. No human reviewer sat on the panel, and the synthesis was also written by a Claude
  model. Role separation is not independence: two seats from one family can share blind spots, so their agreement is
  weaker evidence than agreement between independent reviewers. The methodology seat also read repository code that the
  memo only names (`train_fit_diagnostic.py` for its W1, `crossfit_lambda` for its W10), so those findings rest partly
  on material outside the memo. No target venue criteria were bound (`criteria_binding_unavailable`), so the review made
  no venue-fit claim.
- **What the reduced panel did not assess.** It had no domain seat, so it did not judge whether caption-side CLIP
  features can carry emotion or whether the factor mechanism is plausible. It had no devil's advocate, so nobody argued
  the case for stopping now and taking branch 3; the answer takes the user's decision to attempt a repair as given. It
  did not assess novelty of the nested score, or venue fit, and it did not re-derive numbers from stored per-anchor
  arrays.

## The panel's answer on the order

**A modified O3: commit the revised rules first, then run the H1 pilot and H3 in parallel. Never H3 first and alone.**

The Journal-Fit seat did not choose an order and left that call to the methodology seat. It found that O1 (H3 first)
looked best only because its first step was the only one with a complete outcome-to-action rule chain, which came from
how the memo was drafted, not from evidence for the order. No seat supported O1.

The methodology seat's plan:
- **Before any run:** one dated, hashed commit with the revised H3 and H1 rules, the joint decision table and the
  single-look rule for seed 45, made before the LAB bank is built and before any H1 pilot number exists.
- **Day 1:** write and test the two-dimensional cross-fit; build LAB and a matched-k bank; launch L3, L5, a fixed-temperature run and the
  matched-k run in one GPU lock (about 10 minutes each); run the H1 pilot with A3 as the pre-specified primary model.
- **Day 2 morning:** score the LAB runs in distribution and on seed 42 under both the term-only and the nested score,
  each paired against A3 and C0 on identical episodes, and apply the joint table. If both readings are negative, the
  repair stops on day 2.
- **If only one code stream can be reviewed per day:** H1 first and H3 on day 2.

Reasons the methodology seat gave:
1. H1 lies on every path after H3 and is the cheapest step (half a day, minutes of CPU).
2. H3 answers a different question (is retraining worth the remaining days). It is worth running because it is the
   only step that separates "the partitions are the limit" from "the architecture or loss is the limit", but as
   designed its most likely outcome had no clear next step (W1 below).
3. The two seed-42 looks that the memo counted as O3's extra cost are not extra, because O1 makes the same looks a day
   apart.
4. O2 with H3 skipped could spend the single seed-45 look on a thin margin without knowing the ceiling, and O4 is the
   least diagnostic.

The only difference between the seats was timing: the Journal-Fit seat read O3 from the memo's cost column (about one
day for both steps), and the methodology plan puts the joint readout on day 2 morning, because H3's transfer is scored
once the nested score exists.

## Main findings

Card references: "Meth" is the methodology seat, "EIC" the Journal-Fit seat. Severity is as the cards gave it.

| Card | Sev. | Finding | Numbers |
|---|---|---|---|
| Meth W1 | Major | H3's fit rule is anchored at zero and its scorer is unspecified, so its most likely outcome has two readings; the readings do not cover every outcome | A3 on E3's fresh pseudo-aspect episodes: 0.36 [−0.12, 0.86] under the cross-fitted agreement rule ("no fit"), 0.95 [0.28, 1.62] under the training score ("fits"); A5 0.61 [0.07, 1.18] under the first; A4's lower bound under the second is −0.004 |
| Meth W2 | Major | The pick criterion mean(R@1, gain) works against the comparison that binds a GO | See the algebra below |
| Meth W3 | Major | The pilot's "promising" bar is lenient, a maximum over six or more models on the draw where A3 was picked, and has no negative branch | E3's picked gain halved on the fresh draw (0.52 to 0.26); margins just above 0 mean about a 50% chance of clearing on a fresh draw before shrinkage |
| Meth W4 | Major | "Falls short" is undefined, and seed 45 is not committed as a single look | A second GO test on another fresh draw would roughly double the chance of a false GO |
| Meth W6 | Major | The "no fit" branch would tune H2's settings (τ, β, schedule) on evaluation-label episodes and carry them into A′ | Spec §4 C2 forbids this for the method |
| Meth W7 | Major | The 3.0 transfer bar applies a winner's-curse halving to an unselected fixed-λ measurement | A3's term-only gain did not shrink between draws (0.99 and 0.97); without the halving the memo's own chain gives about 1.5, half of 3.0 |
| EIC W1 | Major | The H1 pilot has a reading but no outcome-to-action rule, and its next steps sit in three wordings | None |
| EIC W2 | Minor | "H1" names both the nested-score hypothesis and the K8 run | None |
| EIC W4 | Minor | The header's provenance claim fails for one row (see the next section) | Three AMI values appear in no appendix |

**Pick-criterion algebra (Meth W2).** The criterion equals either/4 + 3·gain/4. The nested control has gain 0, so it is
in effect tuned on R@1 alone. Along a line of equal criterion, the either rate changes by −3 times the change in gain,
so R@1 (which is either minus the other-aspect rate, at a fixed criterion) changes by minus the change in gain. A cell
that adds 1.0 point of gain and loses 2.9 points of either rate raises the criterion by 0.025, although its R@1 falls
0.95 below the control's. The comparison against the nested control on R@1 is the one that binds a GO. The methodology
seat named this as the mechanism by which E3's cross-fit kept λ at 0.25 to 0.5 and lost 2.96 R@1 to its control.

**H3 scorer ambiguity (Meth W1).** On E3's own fresh pseudo-aspect episodes, A3 reads "no fit" under the agreement rule
(gain 0.36, interval [−0.12, 0.86], which includes 0) and "fits" under the training score (0.95 [0.28, 1.62], lower
bound above 0). A LAB run that fits as weakly as E3 did is the outcome the "architecture or loss" reading predicts, and
for such a run the choice of scorer and bootstrap noise would decide the classification. The two classifications lead
to different next steps (H2 on LAB, or H1 only and then branch 3). For LAB, the in-distribution and transfer
measurements are the same labelled task on different rows, so "fits, weak transfer" mostly means a weak fit, which is
the "no fit" reading under another name.

**Controller re-derivation.** We re-derived both numbers ourselves, not only from the card: the W2 identity (at equal
mean(R@1, gain), R@1 changes by minus the gain change; +1.0 gain and −2.9 either gives criterion +0.025 and R@1 −0.95)
and the W1 example (the agreement rule's interval [−0.12, 0.86] reads "no fit", the training score's [0.28, 1.62] reads
"fits"). The methodology seat's own 13 arithmetic receipts found 12 consistent and one (AR12) not computable because
the rounding rule was ambiguous. Receipts attest auditability, not correctness.

**Where one seat deferred to the other.** Both seats raised the missing H1 outcome rule (EIC W1, Meth W3) and the
undefined "falls short" (EIC W1, Meth W4), at the same severity and with compatible remedies. H3's non-exhaustive
readings were raised by the methodology seat and recorded as a pointer by the Journal-Fit seat. The remaining
methodology findings are single-seat and undisputed, which is the expected pattern because the seats scored disjoint
dimensions.

## The error the panel caught in our own memo

The memo's header said that every number in sections 2 and 3 came from the E3 report (Appendix A), whose final review
had re-derived it from stored per-anchor arrays. EIC W4 found that this did not hold for the AMI row of the section 2
table: that row came from E2's build record (`build_record.json`), not from the E3 report. Three of its six values
(affect against emotion 0.20, caption against emotion 0.06, caption against style 0.06) appear in none of the
appendices. The controller confirmed the finding against the E2 record. Two smaller points came with it: "A3's best
measured gains" (0.97 and 1.14) cannot be checked because the appendix's fixed-λ table omits λ = 4 and 16, and 33.44 is
the either rate of the cross-fitted uniform control, which the memo called the "uniform term" (the term alone gives
33.48). The methodology seat's statement that the section 2 table matches Appendix A wherever both report a value
concerns values, not labels, and does not conflict with this.

**The "H1" name collision.** The memo used H1 for the nested-score hypothesis and, in its own section 2 table, for
the K8 training run on the AI bank. Appendix A uses H1 only for the K8 run, and Appendix B uses both meanings on one
page. No rule became ambiguous, so the Journal-Fit seat kept this at Minor.

## Required changes and where they were carried

The editorial decision listed twelve must-fix items (R1 to R12). Spec revision 3
([§15](../../../superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md), method A′) and the
stage pre-registration (`src/test/20261105_method_repair_diagnostics/PREREGISTRATION.md`, committed in eb58116,
abbreviated P below) carry them. P is the binding text for the diagnostics stage.

| Ref | Required change | Carried in |
|---|---|---|
| R1 | H1 pilot rule block: promising, not promising and inconclusive, each with a next action | P §6 (reading) and P §8 (joint table: next action per outcome); §15 "Diagnostics stage" |
| R2 | Define "falls short" as a development-pilot outcome only | P §6 and §8 replace the phrase with the named outcomes "inconclusive or not promising" of the seed-42 pilot; P §2 allows only seed 42 on selection rows in this stage |
| R3 | One set of outcome words across §5.1, §5.2 and §6 | P §6, §7 and §8 use one vocabulary (promising, not promising, inconclusive; fits, no fit; ceiling sufficient, ceiling too low) |
| R4 | One deciding scorer (term-only agreement rule) and a fixed episode count for H3 | P §5 (term-only agreement rule, rank-equivalent to λ = ∞) and P §7 (fresh label episodes, 4,096 per pair, seed 3042 + pair index) |
| R5 | Paired A3 and C0 baselines; fit decided on LAB minus A3; loss bar descriptive only | P §7 (fit of a LAB run is the painting-clustered paired gain against A3 on identical episodes; the loss over the last 10 steps is in the descriptive list; X minus C0 also descriptive) |
| R6 | H3 readings that cover every outcome, with an inconclusive band | P §7 (fits if the lower bound is above 0, no fit if the point is at or below 0, inconclusive otherwise, with one rerun at model seed 43) |
| R7 | Pick rule aligned with the binding R@1 comparison, reused for the A′ pick | §15 (cross-fit maximises min(R@1 minus the same-half control's R@1, condition gain), tie order stated; "ARS W2" cited) and P §5 |
| R8 | Pre-specified primary model (A3) and a 2.80 paired-SE promising bar | P §6 (primary model A3 by checkpoint SHA-256; promising if m_R and m_g are each at least 2.80 times their SE; not promising if either is at or below 0) |
| R9 | Seed 45 scored once for one pre-registered A′; a failure ends the repair | §15 "The A′ GO test" (one look, by Oct 12, failure takes branch 3; ledger 42, 43, 44, 45) and P §2, P §8 last row |
| R10 | LAB as a yes or no gate only; H2 grid pre-registered before H3; LAB checkpoints excluded by hash | P §4 (LAB checkpoints never A′ or H2 candidates, hashes in `results/label_checkpoints.json`), P §9 (grid pre-registered now), §15 H3 bullet |
| R11 | Transfer bar restated in nested-score units without the halving (g*) | P §6 (g* = max(5.6·SE_R, 2.8·SE_g) from the pilot) and P §7 (ceiling sufficient if the best fitting run's nested gain point is at least g*) |
| R12 | L5 mandatory plus a fixed-τ run, if "no fit" is to stop the repair | P §4 (L3, L5, LT, MK3 in one GPU lock) and P §7 (H3 no fit means none of L3, L5, LT fits) |

Several suggested items (S) were also carried: the seed ledger and 8B probe on seed 46 or later (S6, P §2 and §15), the
joint decision table with a dated stop (S9, P §8), the stated bank seeds and episode counts (S10, P §2 and §3), the
disclosures of seed 45's painting overlap and H1's post-hoc origin (S11, P §11 and §15), the A3 minus C0 row and the
predicted seed-45 power (S12 and S8, P §6 descriptive rows), the matched-k control (S13, run MK3) and the second model
seed for inconclusive runs (S14, P §7). We did not check the remaining S items (S1 to S5, S7, S15) against §15 or the
pre-registration for this report.

## What it changed

The user approved the revisions. Spec revision 3 (§15) and the diagnostics pre-registration were committed before any
script of the diagnostics stage ran, and the stage then ran under them. Its results (the H1 pilot on A3, the H3 fit
and ceiling, the matched-k reading and the joint decision) are not part of this report. They belong in the separate
report `2026-11-05_method_repair_diagnostics.md`, which is yet to be written.

## Caveats

- **One model family, two seats.** Agreement between the seats is weak evidence, and a mistake both share would not
  show as a disagreement.
- **No outside eyes and no adversary.** No human reviewer took part and nobody argued for branch 3.
- **Plan, not results.** The review judged rules written before any result, so it says nothing about whether A′ works.
- **Code access.** Two of the methodology findings rest partly on code the memo only names.

## Sources

- `src/test/20261104_ars_repair_order_review/memo.md`, `phase0_reviewer_configuration.md`, `phase1_eic.md`,
  `phase1_methodology.md`, `phase2_eic.md`, `phase2_methodology.md`, `editorial_decision.md`,
  `20261104_ars_repair_order_review_log.md`
- `docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md`, §15
- `src/test/20261105_method_repair_diagnostics/PREREGISTRATION.md` (commit eb58116)
- Precedent: [ARS plan review](2026-10-27_ars_plan_review.md); E3 results: [aspect factor go/no-go](2026-11-01_aspect_factor_gonogo.md)
