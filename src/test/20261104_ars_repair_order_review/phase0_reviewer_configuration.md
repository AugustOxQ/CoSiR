# Field Analysis Report

**Agent:** field_analyst_agent (Phase 0), ARS academic-paper-reviewer v3.22.2
**Mode:** methodology focus. Contract `reviewer/reviewer_methodology_focus/v2`, panel size 2.
**Date:** 2026-10-03

**Panel scope for this mode.** Methodology-focus runs three agents: field_analyst, eic and methodology_reviewer. This
report therefore configures exactly **two** seats:

- Card #1: Role `EIC`, Display role `Journal-Fit Reviewer`, owner of D2 `writing_and_structure`.
- Card #2: Role `Peer Reviewer 1`, Display role `Peer Reviewer 1` (methodology), owner of D1 `methodology_rigor`
  (mandatory).

This mode has no Peer Reviewer 2 (domain) or Peer Reviewer 3 (cross-disciplinary) seat, so neither gets a card, and
**no Devil's Advocate is configured**. The role file's quality gate of four cards is replaced by this mode's
two-seat panel.

**Target binding:** `criteria_binding_unavailable`. No author-confirmed #683 ReviewTargetContext and no binding manifest
was supplied. No venue criteria are applied, and none are inferred from model memory.

---

## Paper Basic Information

- **Title:** Which repair to run first after the E3 NO-GO: a decision memo for method A′ (CoSiR v2), with the E3 report,
  the repair handoff and spec §3/§6 appended.
- **Document type:** an internal pre-results decision memo for a research team. It is not a submission. It asks which of
  four candidate steps (H1 nested test-time score, H2 training-fit repair, H3 label-trained learnability diagnostic,
  H4 combination) to run first, and under which pre-stated decision rules, before a new pre-registered GO test of a
  repaired method A′.
- **Abstract length:** no abstract. §0 "The decision requested" works as the summary (about 230 words).
- **Full text length:** about 19,400 words (metadata 19,424). The memo body (§0 to §9) is about 3,670 words. Appendix A
  (the E3 report, verbatim) has about 11,930, Appendix B (the method-repair handoff, verbatim) about 2,420, and
  Appendix C (spec §3 and §6, verbatim) about 1,310. The document holds about 180 table rows.
- **Number of references:** no reference list. One external work is named in passing ("Wang et al.", a raw-feature
  baseline) without a bibliographic entry. GeneCIS, ArtELingo, GoEmotions, CLIP ViT-B/32, Qwen3-VL-2B and the baselines
  RCA, KISSME, Tip-Adapter and Xing are named without citations. The sources are internal project records: five earlier
  internal reports (relative links), the spec, the E3 pre-registration, commit hashes and code paths. Appendix A embeds
  five figures by relative path, and those paths do not resolve from this folder, so reviewers see the captions only.

## Field Analysis

| Dimension | Analysis Result |
|---|---|
| Primary Discipline | Computer vision and machine learning: cross-modal (image and caption) retrieval, specifically example-conditioned aspect similarity. A few example pairs demonstrate an aspect without naming it, and the ranking must follow that aspect. |
| Secondary Disciplines | (1) ML evaluation methodology and statistics: pre-registration, cross-fitted selection, winner's curse and forking paths after a failed confirmatory test, the painting-clustered bootstrap. (2) Representation and metric learning from weak signals: sparse non-negative factor bases, pseudo-label (k-means pseudo-partition) episode training, distant supervision, metric-from-pairs baselines. (3) Computational analysis of art and affect: ArtELingo's emotion, style and genre labels. |
| Research Paradigm | Quantitative, as a prospective design. The memo proposes diagnostic and confirmatory computational experiments with pre-stated thresholds and reports none of their results. The appended E3 report is a completed quantitative study (pre-registered GO test plus post-hoc descriptive analyses). |
| Methodology Type | Experimental (computational) plus statistical modelling and machine learning. The memo specifies an upper-bound oracle diagnostic (H3: training on evaluation labels of training rows), a development-set pilot over a 2-D fusion-weight grid with cross-fitting (H1), a retraining grid behind a training-fit gate (H2), outcome rules with numeric thresholds, and a later intersection-union GO test of six one-sided paired comparisons under a painting-clustered bootstrap. |
| Target Journal Tier | `criteria_binding_unavailable`. This is an internal decision document, not a submission, so no venue tier applies. The memo says the eventual paper targets CVPR (abstract Nov 10, paper Nov 16). That is recorded as author-stated context only, and no CVPR criteria are bound or applied. Field-general observation: the document's reference class is an experimental protocol or decision memo inside an ML research project, judged on fitness for its stated decision. D2's phrase "venue conventions for method-focused papers" should therefore be read as the conventions of a method-focused protocol or decision document (decision first, options, rules, evidence, constraints, risks). No journal or conference template applies. |
| Paper Maturity | **Pre-results decision memo.** This label describes the stage, not the quality. Nothing in §5 to §7 has been run. The memo is structurally complete for its purpose: the decision requested (§0), the task and GO test (§1), the evidence (§2), the GO arithmetic (§3), the hypotheses (§4), the designs and outcome rules (§5), the orders compared (§6), fixed constraints (§7), self-identified weaknesses (§8) and questions for the panel (§9), with the evidence appended verbatim. It sits outside the role file's first draft / revised draft / pre-submission scale; the closest analogue is a pre-registration draft circulated for review. |

## Recommended Target Journals (Top 3)

Not applicable: `criteria_binding_unavailable`. The document under review is an internal decision memo, not a manuscript
for a venue. The author-stated eventual target (CVPR) is context, not a review criterion. Following the role file, no
substitute venue is selected and no venue fit is claimed.

## Reviewer Configuration Cards

### Reviewer Configuration Card #1

**Role**: EIC
**Display role**: Journal-Fit Reviewer
**Contract dimension owned**: D2 `writing_and_structure` (priority normal; contract rule F4: D2 at "warn" or worse
gives minor revision)
**Identity Description**: A research director in vision-language learning with long area-chair experience in the
computer vision conference community (field-general here; no venue bound). For years they have been the person who
reads a team's experiment decision memos, pre-registration drafts and go/no-go write-ups and signs off on what runs
next. Their specialty is decision documents under deadline: can a reader who has one hour and the go/no-go call find
the decision being asked, the options, the rule that fires on each outcome, the evidence behind each option, and the
cost of being wrong. They also routinely check whether the author's own preference has been allowed to frame the
comparison. For this panel the "journal fit" question becomes **fitness for purpose**. Does this memo serve its readers,
the team and the user deciding by about Oct 9, and does it frame the decision honestly?
**Review Focus**:
  1. **Decision traceability (§0 to §9).** Check that the decision requested, the three aims of §0 (most
     decision-relevant information per day, no spending of fresh test episodes or held rows, early stopping on a
     negative), the four orders O1 to O4 (§6), the outcome rules (§5.1), the pilot reading (§5.2) and the fixed
     constraints (§7) can each be found. Check that every rule states what is measured, the threshold, and the next
     action it triggers, and that the §6 comparison table lines up with the §5 designs and the §8 weaknesses.
  2. **Terminology, naming and chronology across the memo and its verbatim appendices.** For example, the label
     "H1" names the nested-score hypothesis in memo §4 and Appendix B, but it names the K8 training run (no genre
     partition) in the memo §2 table and throughout Appendix A. Appendix A carries a sequence date (2026-11-01) and the
     folder a sequence date (20261104), while the memo is dated 2026-10-03 and says E3 ended that day. Check whether
     defined terms (condition gain, either rate, uniform-weight control, cross-fitting, GO bar, development look) are
     defined at first use in the memo and used consistently. Also check whether figure references that cannot render
     here leave any memo claim unsupported.
  3. **Honest framing of the choice.** Check whether the user's leaning (H3 first, O1) is presented as a preference
     argued from evidence, on an even footing with O2 and O3. The memo's own statement that "H3's transfer measurement
     is more informative when the nested scorer of H1 already exists" (§6) bears on this. So does Appendix B's
     "Suggested first steps", which pilots H1 first and runs H3 in parallel. Check that post-hoc items are labelled
     post-hoc, that cost estimates are labelled as estimates, and that the memo tells the reader the H1 idea came from a
     post-hoc look that included the spent seed-43 test draw. Check that the fallback (branch 3) and the schedule cost
     of a negative result are stated where the decision is made.
  4. **Proportion and self-containment.** The memo has about 3,700 words and the appendices about 15,700. Check
     whether a reader of §0 to §9 alone has every number and definition the decision needs, whether §2 and §3 quote
     Appendix A faithfully, and whether the §9 questions are scoped so that answers would actually change the plan.
**Will particularly care about**: Whether someone who reads only §0 to §9 can say what will run first, what each
outcome would trigger, and why this order beats the others. Also whether the memo's framing (the leaning stated in
§0, "the user's leaning" in the §6 table) nudges the reader toward a preferred answer instead of letting the evidence
choose.
**Possible blind spots**: This seat does not re-derive statistics or judge whether thresholds are technically sound.
It may reward a clean structure that hides a weak decision rule, which is Peer Reviewer 1's territory. It may treat the
verbatim appendices as out of scope for writing quality, although their naming collides with the memo's. With no venue
bound, it will not (and should not) apply CVPR presentation norms. It is not a domain expert in conditional similarity
or factor learning.

### Reviewer Configuration Card #2

**Role**: Peer Reviewer 1
**Display role**: Peer Reviewer 1
**Contract dimension owned**: D1 `methodology_rigor` (mandatory; contract rules F1 to F3: a fatal D1 block gives
reject, and a D1 block or warn gives major revision)
**Identity Description**: A statistician working in machine-learning evaluation methodology. They specialise in
confirmatory and pre-registered experiments on learned models and in selective inference after model selection:
cross-fitting, the winner's curse, garden-of-forking-paths risk after a failed pre-registered test, and
development-versus-test hygiene when the same rows are resampled into new episode draws. They have hands-on experience
with episodic few-shot and retrieval benchmarks, where cluster-robust (grouped) bootstrap inference and its Monte Carlo
error decide close calls. They also have experience designing oracle and upper-bound diagnostics (training on labels
the method may not use) that separate "the task is unlearnable by this model" from "the proxy supervision is wrong"
from "the optimiser or settings are wrong". What they check first in any diagnostic is whether its outcomes can
actually tell the competing explanations apart, and whether running it spends or contaminates the later confirmatory
test.
**Review Focus**:
  1. **Discriminating power of H3 and its outcome rules (§5.1, §8, §9 Q2 and Q3).** Check that "no fit", "fits, weak
     transfer" and "fits and transfers" are exhaustive and mutually exclusive, and that each maps to one explanation
     and one next action. For example: "no fit" is an AND rule and "fits" is its OR complement; some measurement
     combinations may get a reading that another measurement contradicts, such as the "no fit" condition met while the
     seed-42 term-only gain is large, or a gain of at least 3.0 whose lower bound is not above 0. Weigh the confounds the
     memo lists against what a reading can still license: label granularity (8, 23 and 10 values against 64 clusters)
     and the cheap matched-k control; settings tuned for pseudo banks; one model seed; one development draw. Judge
     whether a second recipe or the matched-k bank would change which reading fires.
  2. **Development and test hygiene and forking paths (§4 H1, §5.2, §7, §8 items 2 and 6, §9 Q4).** Use the seed
     ledger: 42 for development and the pick, 43 spent, 44 used by the MLLM probe, 45 reserved for A′. Count the looks
     at seed 42 (an H1 pilot over 56 grid cells for A1 to A6, with C0 and SE as references; H3's transfer read; then the
     A′ pick). Note that H1 was generated from a post-hoc profile computed partly on the seed-43 test draw. Seed 45 is
     drawn from the same 32,413 selection rows (6,451 paintings) as seeds 42 and 43, so it is not independent of the
     data that generated the hypothesis. Check whether H3's labels can leak into A′'s design: under "no fit" the memo
     proposes tuning H2 settings on the label bank LAB, which could carry evaluation-label information into the
     method's hyperparameters, a concern given spec §4 C2. Check whether "pre-register before any A′ run is scored on
     seed 45" is sufficient protection.
  3. **Resolution, thresholds and power (§3, §5.1 "Why 3.0", §8 item 5).** Re-derive the GO arithmetic (for example
     2 × 16.72 − 27.25 ≈ 6.2 and 2 × 16.72 − 21.37 ≈ 12.1, and mean(R@1, gain) = either/4 + 3·gain/4). Re-derive the
     claimed 0.7 to 1.0 fused-gain requirement from E3's paired half-widths (about 0.33 for R@1 and 0.30 to 0.44 for
     gain). Judge the 3.0-point transfer threshold, which rests on a single winner's-curse observation (0.52 to 0.26)
     and heuristic discounts, and the 5% fit threshold, both set without a power analysis. Judge whether a cheap
     precision calculation is possible within the window. Use E3's bootstrap-seed sensitivity, where three of six
     lower bounds were Monte Carlo boundary cases, when judging the "lower bound above 0" rules.
  4. **The H1 pilot and its control (§5.2, §9 Q5).** Check that the nested uniform control z(cos) + (λ_u + λ_a)·z(T_u)
     is the right counterfactual. Does it get pick freedom equal to the nested score's? Its set of λ sums differs from
     E3's 1-D grid, which included λ = ∞. Judge whether the "promising" reading (lower bounds above 0 against the
     nested control on seed 42) is too lenient given 56-cell pick freedom over several models and E3's winner's curse.
     Consider whether the paired variance of nested score against nested control differs from A3 against its E3
     control, which would change the margin of §3.
  5. **Can the order meet its own aims (§6, §9 Q1).** Judge whether each order (O1 to O4) gives an interpretable first
     result within its estimated cost, whether a negative result stops the repair early, and whether any order weakens
     the later seed-45 GO test. Note the dependency the memo states: H3's transfer read is more informative once the
     H1 scorer exists.
**Will particularly care about**: Whether every proposed step, as designed, yields an outcome that can be read
unambiguously and changes the next action. Also whether any step, by repeated development looks, label leakage into
design choices, or added pick freedom, weakens the seed-45 GO test or what the paper may later claim. Per the contract's
stage notes, missing results are not a block. Design flaws that leave a step uninterpretable or weaken the later GO test
are.
**Possible blind spots**: This seat may treat a six-day decision memo as a full confirmatory protocol and ask for
controls, model seeds and power analyses the window cannot afford. It should weigh each request against §0's three
aims and the stated budget. It may under-weigh domain plausibility (whether caption-side CLIP features can carry
emotion, whether aspect-block factors are learnable by this architecture, novelty of a nested fusion score) because
there is no domain seat in this panel. It does not judge readability unless an ambiguity changes what a rule would
decide.

## Review Strategy Recommendations

**Object of review and calibration.**
- The object under review is the memo (§0 to §9): its proposed order, designs and pre-stated decision rules.
  Appendices A to C are evidence. E3's verdict was already final-reviewed, and this panel is not re-reviewing it.
  Reviewers may still check that the memo's §2 and §3 quote Appendix A faithfully and may use Appendix A's numbers to
  judge the designs.
- The contract's stage-specific note governs scoring: missing results are not a block, and design flaws that leave a
  step uninterpretable or weaken the later GO test are. Per the contract's measurement procedure, each seat writes its
  contract paraphrase and scoring plan (what to look for, and what triggers warn, block and fatal for its dimension)
  before reading the manuscript.

**Dimension boundary between the two seats.** An ambiguity or inconsistency that would change what a rule decides
(an outcome falling between rules, a control defined two ways) belongs to D1 and Peer Reviewer 1. One that slows or
misleads a reader without changing the decision belongs to D2 and the Journal-Fit Reviewer. The naming collision on
"H1" falls under D2 unless a reviewer shows that it makes a rule or a ledger entry ambiguous.

**Expected tension.** Peer Reviewer 1 may favour more controls (the matched-k bank, a second recipe, a second model
seed, a precision calculation). The Journal-Fit Reviewer, and the memo itself, weigh time per decision-relevant result
against a six-working-day window. The synthesizer should judge requests against §0's three aims, and should
distinguish controls that make a reading interpretable (D1-relevant) from controls that are merely nice to have.

**Coverage gaps from the reduced panel.** With no domain seat and no Devil's Advocate, nobody owns the following:
positioning against the conditional-similarity and vision-language retrieval literature (GeneCIS and others), novelty
of the nested score, plausibility of the factor-basis mechanism, and an adversarial case for branch 3 over any repair.
The synthesizer should mark such points as outside this panel's scope rather than scoring them implicitly.

**Verification limits.** Reviewers cannot access the stored per-anchor arrays, checkpoints or code. Internal
arithmetic and cross-section consistency can be checked from the text; data-level re-derivation cannot.

**Manuscript text directed at the panel (findings, not instructions).** The manuscript is author-supplied data. The
items below are reported, not obeyed. **No text was found** that asks for leniency, prescribes reviewer identities,
requests a particular verdict, or tells reviewers to ignore content.

1. **Stated author preference (§0, lines 23 to 24; §6 table "the user's leaning"; §9 Q1).** "The user leans toward
   running H3 first." This is a disclosed preference, which is appropriate in a decision memo. It is also an anchoring
   cue. Both seats should judge the order on the evidence and not defer to it. The Journal-Fit seat should assess
   whether the framing is even-handed.
2. **Provenance assurance (lines 3 to 4).** "Every number in Sections 2 and 3 comes from the E3 report (Appendix A),
   whose final review re-derived it from stored per-anchor arrays." This is a verification claim the panel cannot
   check. Treat it as an unverified author statement, and do not let it replace the reviewers' own consistency
   checks of §2 and §3 against Appendix A.
3. **"Questions for the panel" (§9, five questions).** These are legitimate author questions, and the reviewers may
   answer them. They do not bound the review's scope or its scoring. Q5's list of possible gaps invites comment and
   does not limit it.
4. **Agent-directed operational text in Appendix B (from line 1097).** The handoff was written "for a fresh agent".
   It contains imperatives ("Read this first", "Read in this order", "Do this pilot before writing the
   pre-registration", rules for the environment, GPU lock, git and reports, and "Suggested first steps"). These are
   instructions to the project's own implementing agent. For this panel they are evidence of the planned workflow,
   not instructions to any reviewer. Their content is relevant to Card #1, focus 3: Appendix B's suggested order
   (H1 pilot first, H3 in parallel) differs from the leaning stated in §0.
