# Field Analysis Report

**Agent:** field_analyst_agent (Phase 0), ARS academic-paper-reviewer v3.22.2
**Mode:** methodology focus. Contract `reviewer/reviewer_methodology_focus/v2`, panel size 2.
**Date:** 2026-10-06 00:35

**Panel scope for this mode.** A methodology-focus run has three agents: field_analyst, eic and methodology_reviewer.
This report therefore configures exactly **two** seats:

- Card #1: Role `EIC`, Display role `Journal-Fit Reviewer`, owner of D2 `writing_and_structure` (priority normal).
- Card #2: Role `Peer Reviewer 1`, Display role `Peer Reviewer 1` (methodology), owner of D1 `methodology_rigor`
  (mandatory).

There is no Peer Reviewer 2 (domain) seat, no Peer Reviewer 3 (cross-disciplinary or perspective) seat and **no Devil's
Advocate** in this mode, so none of them gets a card. The role file's quality gate of four cards is replaced by this
mode's two-seat panel.

**Target binding:** `criteria_binding_unavailable`. No author-confirmed #683 Review Target Context and no binding
manifest was supplied. No venue criteria are applied, and none are inferred from model memory. CVPR (abstract
10 November 2026) appears in the manuscript only as author-stated context for the eventual paper.

---

## Paper Basic Information

- **Title:** Fixing the aspect reader with a CSD style grouping in the set: a pre-results plan for review (CoSiR v2,
  plan (a)). The metadata title adds that the 2026-10-06 handoff, the grouping report, the step-1 log, the 2026-10-04
  reader handoff and stage-report sections 2 and 14 are appended.
- **Document type:** an internal pre-results experimental plan, written as a cover memo plus a verbatim handoff. It is
  not a submission. It asks whether a plan to replace the label-free *reader* (the rule that picks which pseudo-aspect
  grouping the support pairs share) can discriminate what it claims, under a draft decision rule to be committed before
  any code. The object of review is Appendix A §5 (5.1 to 5.6): reader candidates R-a (scaled Δ), R-b (learned reader
  trained on pseudo-aspect bank episodes) and R-c (confidence gate), the measures, the draft decision rule §5.4, the
  timeline §5.5 and the review targets §5.6. The cover memo §0 to §4 frames six questions. Appendices B to F are
  evidence. (`memo.md` in this folder is identical to the manuscript's §0 to §4.)
- **Abstract length:** no abstract. §0 "The decision requested" serves as the summary (about 340 words).
- **Full text length:** about 19,100 words (metadata 19,092). Cover memo §0 to §4 about 1,620. Appendix A (the plan,
  handoff of 2026-10-06) about 2,260, of which the object of review, §5, is about 950. Appendix B (grouping draft
  report) about 6,190; Appendix C (step-1 log, results to controller review) about 2,980; Appendix D (reader handoff of
  2026-10-04) about 2,710; Appendix E (stage report §2 and §14) about 2,850; Appendix F (two recorded project lessons)
  about 340. About 200 table lines.
- **Number of references:** no reference list. Works are named in passing without bibliographic entries: CSD
  (Somepalli et al. 2024, preprint), Daunhawer et al. (ICLR 2023), CACTUs, MFCVAE, SCE-Net, DiscoverNet, SCAN, TEMI,
  SwAV, IIC, Long-CLIP, the GoEmotions checkpoint `SamLowe/roberta-base-go_emotions`, PercepT, ArtELingo, ArtEmis,
  WikiArt/ArtGAN, CLIP ViT-B/32, VGG-19 Gram statistics, Leiden, and the pair-metric baselines RCA, KISSME, Xing and
  CVS. The real sources are internal: earlier reports, plan and log files, commit hashes, code paths and two
  project-memory notes. Four figures are embedded by relative paths (three in Appendix B, one in Appendix E) that do not
  resolve from this folder, so reviewers see the captions only.

## Field Analysis

| Dimension | Analysis Result |
|---|---|
| Primary Discipline | Computer vision and machine learning: cross-modal (image and caption) retrieval under an example-conditioned aspect. Four support pairs and four contrast pairs show an unnamed aspect, and the ranking of 13 candidates must follow it (conditional similarity, few-shot task inference). The specific component under review is a task-inference module: a reader that infers which of several label-free pseudo-aspect groupings the examples share, fused into a retrieval score. |
| Secondary Disciplines | (1) **ML evaluation methodology and statistics:** pre-registered selection among several candidates on a reused development seed, carried by the maximum; cross-fitting of fusion weights; matched condition-only controls; a painting-clustered bootstrap; an intersection-union GO test on three fresh episode seeds. (2) **Meta-learning and learning from pseudo-tasks:** R-b is trained on episodes constructed from clustering/community groupings (unsupervised task construction in the CACTUs sense) and must transfer to real aspect episodes, with cross-fitted heads to avoid in-sample posteriors. (3) **Computational analysis of art and affect, with weak or distant supervision:** ArtELingo emotion (per viewer), style and genre (per painting); groupings from GoEmotions probabilities (Leiden), CLIP k-means and CSD style embeddings (Leiden). |
| Research Paradigm | Quantitative, as a prospective design. The plan reports no result of its own. The appendices are completed quantitative exploratory studies on the development seed (step 0, step 1, the Leiden sweep), with plans written before their numbers. |
| Methodology Type | Experimental (computational) plus statistical modelling and machine learning. Development-set model selection among seven configurations (R-a, R-b arg-max and R-b expected on A1 and on A0, plus R-c on the best), each fused through a 56-cell cross-fitted weight grid, each with a matched condition-free counterpart and a rebuilt condition-free bar B′. A development bar (bar margin at least +0.5 with a lower bound above 0), a kill rule, a tie-break, then one confirmatory test on fresh episode seeds 49 to 51 pooled with one cluster per painting, requiring lower bounds above 0 against cosine, RCA, B′ and the matched counterpart. |
| Target Journal Tier | `criteria_binding_unavailable`. This is an internal plan, not a submission, so no venue tier applies. CVPR is recorded as author-stated context only; no CVPR criteria are bound or applied. Field-general observation: the reference class is an experimental protocol or pre-registration draft inside an ML research project, judged on fitness for the decision it feeds. D2's phrase "venue conventions for method-focused papers" is read here as the conventions of a method-focused protocol or decision document (decision requested, fixed constraints, candidates, measures, rule, outcome-to-action mapping, timeline, risks). No journal or conference template applies. |
| Paper Maturity | **Pre-results plan (pre-registration draft).** The label names the stage, not the quality. Nothing in §5 has been run, and the rule file `DECISION_RULE.md` does not exist yet; the author states that the review may change any §5.4 item. For its purpose the document is structurally complete: decision requested and six questions (§0), task and bars (§1), fixed constraints (§2), implementation facts read from the code (§3), review targets (§4, quoting §5.6), and the plan with candidates, measures, draft rule, timeline and pitfalls, with the evidence appended verbatim. Draft-stage signs that reviewers should expect: some thresholds are stated informally (the R-b domain-shift kill), some terms are defined twice (B′), and internal codes from earlier work (A′, C2, E2, N6, T_N1u) appear in the object of review. The role file's first draft / revised draft / pre-submission scale does not fit; the nearest analogue is a registered-report protocol circulated for review. |

## Recommended Target Journals (Top 3)

Not applicable: `criteria_binding_unavailable`. The document under review is an internal plan, not a manuscript for a
venue. The author-stated eventual target (CVPR) is context, not a review criterion. Following the role file, no
substitute venue is selected and no venue fit is claimed.

## Reviewer Configuration Cards

### Reviewer Configuration Card #1

**Role**: EIC
**Display role**: Journal-Fit Reviewer
**Contract dimension owned**: D2 `writing_and_structure` (priority normal; contract rule F4: D2 at "warn" or worse
gives minor revision)
**Identity Description**: A research lead in vision-language representation learning and few-shot retrieval, with long
area-chair experience in the computer-vision conference community (field-general here; no venue bound). For years they
have chaired internal go/no-go reviews and read registered-report style protocols for ML experiments. Their speciality is
the executable decision document: can someone other than the author turn the plan into a rule file and apply it without
asking, are terms defined once and used one way, does every outcome map to a stated action, and does the framing let the
evidence choose. For this panel "journal fit" becomes **fitness for purpose**: does the cover memo plus Appendix A §5
serve its actual readers (the user deciding on Friday 9 October and the implementing agents), and is the rule ready to be
committed as `DECISION_RULE.md`?
**Review Focus**:
  1. **Outcome-to-action completeness of §5 as a document.** Enumerate the outcomes the plan can produce on the
     development seed and on the test, and check that the text names an action for each: no candidate clears the bar;
     one clears; several clear within the 0.05 tie window; R-b fails its domain-shift check; the carried configuration
     passes some GO comparators but not others; the test fails. Check that every rule names the quantity, the threshold
     and the action, and whether the plan would benefit from one outcome table. Check that each stated purpose has a
     listed measure: A0 runs "to show whether the CSD grouping helps once the reader works", but §5.3 lists paired
     differences only against each configuration's current arg-max reader, not A1 against A0 under the same reader.
  2. **Terms and names in the object of review and across the appendices.** B′ is "B rebuilt, not B plus a term" in §3
     and in Appendix A §4, but "B plus the configuration's averaged-heads term" in §5.4 item 2. Codes from earlier work
     appear in §5 or §3 without a definition in this document: "A′'s min-margin cross-fit" (R-c), "B = C2", E2, N6,
     T_N1u, T_6u, the AIC bank. Names collide: "A3" is a step-1 arm (Leiden image and caption) and also the method-A
     checkpoint whose centred factor term T_N1u sits inside B; "R1, R2, R3" (step-1 readings), "R-a, R-b, R-c"
     (readers), "R0" (an arm) and R@1 (the metric); "step 1" of the grouping work against "Step 1" of the 2026-10-04
     reader handoff; conditions "a/b" in the plan against "A/B" in Appendix E, where A and B also name the two aspects;
     "grouping" against "partition". Folder and report names carry sequence dates (20261117, "2026-11-12") while the
     calendar dates are 2026-10-05 and 2026-10-06. Check which of these would slow or mislead a reader of the committed
     rule file.
  3. **Fidelity and self-containment of the cover memo.** Check that §0 to §4 restate Appendix A and the evidence
     faithfully (the §4 quote of §5.6, the §1 bar definitions, the §3 implementation facts), and whether a reader of
     §0 to §4 plus Appendix A §5 has every definition needed to apply §5.4 without opening Appendices B to F. The
     object of review is about 950 of about 19,100 words. Check whether claims that rest on figures (Appendix B Figure 3,
     the pick tables) are still supported when the figures do not render.
  4. **Honest framing and demarcation.** Check that the line between what is fixed (§2, Appendix A §3 "not to be
     reopened") and what the review may change ("The review may change any item" of §5.4) is clear. Check that the
     motivations for R-a and R-b (failures 1 and 2, diagnosed post hoc from seed-42 pick tables under the told mapping)
     are labelled exploratory, that the chance assessment ("moderate at best", Appendix B §10) is visible where the
     decision is framed, and that the timeline (§5.5) shows the process steps the plan itself requires elsewhere (the
     controller re-derivation is scheduled; the whole-branch final review of Appendix A §6 is not; the user's "commit
     only when asked" rule meets "commit the rule before any code").
**Will particularly care about**: Whether someone holding only §0 to §4 and Appendix A §5 could write `DECISION_RULE.md`
unambiguously, and could say for each development and test outcome what happens on Thursday and on Friday.
**Possible blind spots**: This seat does not re-derive statistics or judge whether a threshold is technically sound, so
it may reward a clean structure that hides a weak rule (Peer Reviewer 1's territory). It may over-penalise internal
shorthand in a document whose actual readers share the project's vocabulary; it should weigh each naming issue by
whether the committed rule file or a fresh implementer would misread it. With no venue bound it does not, and should
not, apply CVPR presentation norms. It is not a domain expert in conditional similarity, style descriptors or
meta-learning.

### Reviewer Configuration Card #2

**Role**: Peer Reviewer 1
**Display role**: Peer Reviewer 1
**Contract dimension owned**: D1 `methodology_rigor` (mandatory; contract rules F1 to F3: a fatal D1 block gives reject,
and a D1 block or warn gives major revision)
**Identity Description**: A statistician in machine-learning evaluation methodology. They specialise in selective
inference after model selection on reused development data: the winner's curse, select-then-test protocols, and
cross-fitting or sample splitting at the level of the independent unit rather than the observation. They have hands-on
experience with episodic few-shot and meta-learning evaluation, including meta-learners trained on clustering-constructed
pseudo-tasks (CACTUs-style unsupervised task construction) and the pseudo-to-real task shift that limits them. They are
practised with cluster bootstrap inference when items recur across episodes and seeds, and with intersection-union GO
tests over several comparators. In score-fusion work they have designed matched negative controls that remove only the
condition, and they audit learned components for leakage through in-sample posteriors, label-informed diagnostics and
tuning on labelled evaluation episodes. What they check first is whether each candidate's outcome can be read
unambiguously against its control, and whether the selection step leaves the one confirmatory test meaningful.
**Review Focus**:
  1. **Matched counterparts and B′ for every candidate, the gate included (§5.2, §5.4 item 2, §3; memo Q1).** Check that
     each counterpart removes only the condition and keeps equal pick freedom: R-a and R-b arg-max (T_cf as the mean of
     the two conditions' picked-grouping terms), R-b expected (Σ P(h)·s_h averaged over the two conditions, which keeps
     an example-adaptive, condition-free weighting of the head terms). For R-c, the gate g comes from the reader's
     top-two margin, which differs between conditions (under condition b the margin is taken on −Δ, or on the swapped
     inputs for R-b). "The counterpart applies the same gate to T_cf" can then mean g_c·z(T_cf), which is not identical
     under both conditions and would trip `crossfit_condition_free`'s check, or a condition-averaged gated term. Judge
     which is the matched control and whether the counterpart gets the same λ-by-threshold grid. Check B′ (rebuilt
     against "plus a term"), the fusion base (§3 fuses on B with `crossfit_nested(B, B, T, parity)`, while Appendix D
     Step 4 fused on B′), and the bar-margin comparator chosen by the larger point R@1.
  2. **Leakage and pseudo-to-real shift in R-b (§5.2 R-b requirements; memo Q2).** Bank rows are scorer-train only, so no
     selection row enters training: confirm. Cross-fitted heads (painting halves) build the training features; the
     standard heads (a 60,000-row draw touching 31,287 paintings) are read at evaluation, so check calibration and
     sharpness mismatch (the cross-fitted heads' training size is not stated). Check the label space: the bank's classes
     are groupings, with a uniform prior and six grouping-pair blocks with no third grouping controlled, while real
     episodes have three aspect pairs, a third aspect controlled on candidates, and a caption grouping that is never the
     told answer. Judge whether the motivating capability ("CSD and image agreeing together means genre, CSD alone means
     style") is learnable from a bank whose classes are groupings, and how R-b for A0 is trained (a three-grouping bank
     differs structurally). Check whether any R-b choice (features, regularisation, bank size, arg-max or expected) could
     be tuned on seed-42 pick accuracy, which reads the told mapping. The domain-shift kill ("high", "does not move") has
     no number.
  3. **Selection, multiplicity and what the fresh-seed test can license (§5.4 items 1, 3, 4, 6; memo Q3).** Seven
     configurations, each with a 56-cell cross-fit (more for R-c), are chosen by the maximum bar margin on seed 42,
     which has been reused many times. R-c is then built on the selected best (select, then augment, on the same data).
     The +0.5 bar is a shrinkage heuristic resting on two observed halvings (0.52 to 0.26, 0.26 to 0.15). Fresh seeds are
     new episode draws on the same 6,451 selection paintings (seed 42 already anchors 4,602 of them), so painting-level
     idiosyncrasies exploited by selection carry into the test. Judge what a GO then licenses (new episodes, not new
     paintings; the held split stays reserved), whether the intersection-union structure and the single gain count are
     right, and whether the test has power: if the margin halves to about +0.25, what pooled half-width do three seeds
     on saturating painting clusters give, against seed-42 half-widths of about 0.20 to 0.25? Work within the user-fixed
     seed policy (§2, Appendix F.2) and propose only mitigations that fit it.
  4. **Cross-fitting, interval validity and R-a's spread (§3, §5.2 R-a, §5.4 item 6; memo Q4).** The cross-fit halves
     are episode-index parity, not painting halves, so both halves share anchor and example paintings. B's picks reuse
     the same halves (disclosed). Picks are fixed before resampling, so intervals omit selection variance. At test, every
     cross-fit is rerun on each seed's own halves with a rule that reads R@1 and gain, so fusion weights are tuned on
     labelled test episodes, cross-fitted; judge whether that is acceptable as part of the method definition and whether
     intervals should reflect it. For R-a: the root mean square over both conditions pooled mixes signal (episodes where
     grouping h is the right one) with noise, which penalises groupings with strong signal such as the image grouping
     under genre. Compare a per-episode standard error over the four pairs, bank-estimated spread (in-sample head
     posteriors on bank rows are sharper, Appendix D Step 1), and label-free re-estimation on each test seed against
     freezing it from seed 42.
  5. **Decision-rule coherence and thresholds (§5.4 items 3 to 5, §5.5; memo Q5 and Q6).** Item 5 kills "if no candidate
     raises pick accuracy by at least 10 points ... or reaches the bar": a candidate that gains 10 points but misses the
     bar is neither killed (item 5) nor carried (item 4), so the pick-accuracy clause either has no decision effect or
     conflicts with item 4. Pick accuracy reads the told mapping, so a kill on it sits uneasily with "the told mapping is
     a diagnostic only", and Appendix D's warning never to compute it on test seeds before the verdict is not restated.
     The same +10 applies to A1 (four groupings, chance 25%, injective told mapping, so no 83.3% arg-max cap) and A0
     (chance 33%, cap 83.3%). Item 3's gain clause is nearly non-binding (every step-1 arm met it). The tie-break order
     is partial (R-a on A1 against R-b on A0), A0 is both reference and candidate, and the plan does not say whether
     R-b's domain-shift kill overrides the bar. Label policy: the CSD grouping was kept over Gram on told margins read on
     labelled seed-42 episodes (Appendix B §10), while Appendix B §7 rejected the sweep's pick for reading labels; judge
     what the paper may claim as label-free, without reopening the fixed grouping. Judge whether Tue 6 to Fri 9 October
     can hold, and which step slips first.
**Will particularly care about**: Whether every candidate, as designed, yields a margin that can be read against a
control that removes only the condition, and whether the selection on seed 42 (seven configurations, carried by the
maximum, plus label-informed kill and diagnostic paths) leaves the one fresh-seed test able to tell a working reader from
a lucky one. Per the contract's stage note, missing results are not a block; flaws that leave a candidate
uninterpretable or weaken the test are.
**Possible blind spots**: This seat may treat a four-day plan as a full confirmatory protocol and ask for painting-level
splits, extra seeds or power analyses the window cannot afford, or for reopening items the user fixed (the seed policy,
the groupings). It should weigh each request against §0 and the contract's stage note, and prefer cheap mitigations that
fit the constraints. With no domain seat on this panel, it may under-weigh whether CSD and the CLIP heads can plausibly
carry style, whether R-b's hand-built features are sensible for the vision problem, and the novelty of the reader. It
judges readability only where an ambiguity changes what a rule decides. It can check arithmetic and cross-section
consistency from the text, not re-derive numbers from the stored arrays.

## Review Strategy Recommendations

**Object of review and calibration.**
- The object under review is Appendix A §5 (5.1 to 5.6): candidates, measures, draft rule, timeline and review targets.
  The cover memo frames the questions; Appendices B to F are evidence and are not re-reviewed. Reviewers may still use
  them to check §5's premises and the memo's quotations.
- The contract's stage note governs scoring: missing results are not a block; flaws that leave a candidate
  uninterpretable or weaken the test are. Per the contract's measurement procedure, each seat writes its contract
  paraphrase and scoring plan (what to look for, and what triggers warn, block and fatal for its dimension) before
  reading the manuscript.
- The author says the review may change any §5.4 item, and §2 lists items fixed by the user. Reviewers judge the plan
  within those constraints. Where a fixed constraint limits what a GO can license (fresh seeds share paintings; CSD was
  kept on told margins), they say so as a scope note on the eventual claim, not as a demand to reopen it.

**Dimension boundary between the two seats.** An ambiguity or inconsistency that changes what a rule decides or what is
computed (the item 5 parse, the R-c counterpart, B′ rebuilt against "plus a term" if the two give different numbers,
fusion on B against B′) belongs to D1 and Peer Reviewer 1. One that slows or misleads a reader without changing a
decision (undefined internal codes, the "A3" and "R1"/"R-a" collisions, sequence dates, figures that do not render)
belongs to D2 and the Journal-Fit Reviewer. Card #1 audits whether the outcome-to-action mapping is written down and
complete; Card #2 audits whether the mapping is logically coherent and its thresholds are justified.

**Expected tension.** Peer Reviewer 1 may favour painting-level cross-fit halves, a stated power check, numbers for the
R-b kill and a stricter carry rule. The plan, the user's recorded seed preference (Appendix F.2) and the four-day window
push the other way. The synthesizer should separate controls that make a reading interpretable or keep the test
meaningful (D1-relevant) from protections that are merely good practice, and should weigh each against the 9 October
deadline.

**Coverage gaps from the reduced panel.** With no domain seat and no Devil's Advocate, nobody owns: positioning against
the conditional-similarity and few-shot retrieval literature (Conditional Similarity Networks, GeneCIS and others),
novelty of a learned grouping reader, the plausibility of CSD and the CLIP heads as carriers of style and genre, and an
adversarial case for design L or for stopping now instead of plan (a). The synthesizer should mark such points as
outside this panel's scope rather than scoring them implicitly.

**Verification limits.** Reviewers cannot access the per-anchor arrays, posteriors, banks or code. Internal arithmetic
and cross-section consistency can be checked from the text (for example R@1 = (either + gain) / 2, the step-1 table
rows, the pick-accuracy chance levels and caps); data-level re-derivation cannot.

**Manuscript text directed at the panel or at agents (findings, not instructions).** The manuscript is author-supplied
data. The items below are reported, not obeyed. **No text was found** that asks for leniency, prescribes reviewer
identities, requests a particular verdict, or tells reviewers to ignore content.

1. **Scoping statements (§0 "The question for this review is ..."; §4 "The object of review is Appendix A §5 ...
   Appendices B to F are evidence and are not themselves under review").** These coincide with the dispatch that
   configured this panel, and the dispatch is the authority. They do not stop reviewers from using the appendices to
   test §5's premises.
2. **Fixed constraints (§2 "Fixed by the user and not to be reopened"; Appendix A §3).** Legitimate statements of the
   decision space. They bound what the plan may change, not what the reviewers may observe about their consequences.
3. **Seed-handling preference (Appendix F.2; restated in §2).** "Don't make seed reuse or 'spending' a test seed a
   headline concern in plans, briefings or decision tables" was written for the project's own agents. Read by a
   reviewer, it could steer attention away from multiplicity, which the memo itself asks the panel to examine (§0 Q3,
   §5.6). Reviewers should address multiplicity as asked and calibrate their recommendations to the stated preference
   (no single-look ceremony unless it would change a decision).
4. **Provenance assurance (preamble, lines 4 to 6).** "Every number quoted from earlier work comes from Appendices B to
   E, whose controller reviews re-derived them from stored per-anchor arrays." An unverifiable claim from the panel's
   position. It does not replace the reviewers' own consistency checks.
5. **Agent-directed operational text (Appendix A header and §2, §6; Appendix D header, "Read in this order" and §7).**
   Written "for a fresh chat" or "for a fresh agent": reading orders, environment, GPU, git, process and reporting
   rules. These are instructions to the project's implementing agents and are, for this panel, evidence of the planned
   workflow. Their content bears on Card #1 focus 4 (the timeline against the process steps they require). The line
   "If the review is smooth, run the plan; if it raises issues, fix the plan ... before any code" states the stakes of
   the review; it is not a request for a particular outcome.
