# Round 7, design L from the style source: design

> Date: 2026-10-12 02:10 · Status: approved 2026-10-12 02:34 (user: "Yeah thats good") · Kind: experiment
> Builds on: [round 6 held read](../../reports/auto/v2/2026-11-25_artelingo_held_test.md); the grill's record
> ([handoff §2](../handoffs/2026-10-12-r7-decide-2-handoff.md)); [plan page](2026-10-11-r7-design-l-plan.html);
> [facts brief](../../../src/test/20261126_design_l_style_source/step3_facts.md); literature notes
> [style sources](../../reports/literature/2026-11-27_style_sources.md) and
> [design L novelty](../../reports/literature/2026-11-28_design_l_novelty.md) · Constitution:
> [version 2](../constitution.md)
> Decision rule: [`src/test/20261126_design_l_style_source/DECISION_RULE.md`](../../../src/test/20261126_design_l_style_source/DECISION_RULE.md)
> @ c95b6ea · binding; it governs where it differs from this spec

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** AFF, our best method, steers only by affect and loses on style × genre (−1.24 R@1 against B′(A1)
on held paintings). Our CSD style grouping also follows genre. We build style groupings with the genre part removed,
with no labels, and let the reader steer by them ("design L").

**Plan.**
1. Ceiling check (exploration with a stop gate): a grouping from the 27 true style labels, scorer told the aspect.
2. Seed-42 screen (new idea): six label-free groupings, four from CSD and two from CLIP token statistics, content
   removed. Each gets a reader and two scorers: R1 and extended AFF (AFF whose gate also opens on the new grouping).
   A three-part gate; the best passing scorer of each family goes on.
3. Fresh seeds 55 to 57 (claim test), in a fixed order: beat the five condition-free scorers, then AFF, then B′(A1)
   on style × genre.
First, the reused code must reproduce AFF's and told CSD's seed-42 numbers.

**Compared against.** B, B′(A0), B′(A1) (the strongest condition-free scorer, 18.81 R@1 on seed 42), the scorer's own
B′ and its matched control; then AFF (19.14). References: raw CSD, and an off-the-shelf non-redundant clustering of CSD (NrKmeans). Each of the six new groupings
has a random control with the same group sizes.

**Outcomes.**
- Ceiling below +0.5 R@1 over its matched control, or its interval reaches 0: design L stops; Friday is a no-go (the
  ArtELingo paper with AFF). From +0.5 to +1.0: the run goes on with a flag and a phone note.
- No scorer passes the seed-42 gate: design L stops; Friday is a no-go.
- A pick passes stage 1 on fresh seeds: Friday is a go (the method paper with design L). It replaces AFF only if it
  also beats AFF.
- We expect: a ceiling near +0.9 (60% to clear +0.5); a gate pass about 40% if the ceiling passes (AFF itself is
  only +0.33 over B′(A1)); a Friday go about 12%.

**Cost.** About 4 CPU hours (14 readers, three at a time) and 15 min of the local GPU; no cluster. Run by Tuesday,
report by Thursday 15 October; not done by then means a Friday no-go.

**Check these.**
1. Stage types (agent default): ceiling an exploration with a stop gate, seed 42 a new idea, fresh seeds a claim test.
2. Token statistics (agent default): CLIP blocks 4, 6, 8; mean and spread of patch tokens; high-norm tokens dropped;
   content removed by regression on CLIP's image embedding (one variant) or Cui et al.'s projection (the other).
3. The ceiling stops below +0.5 (user); our recommendation was +1.0.
4. After our recommendation (user): CSD first with generic sources in parallel; a family that passes alone goes at
   the full one-sided 0.025, unlike B's order "generic only if CSD works" (the CSD-alone case: agent default).
5. The random controls and the NrKmeans reference decide nothing; a pick whose random control also passes the gate
   is flagged (agent default).

## 1. Context and goal

Asked: "Read /project/CoSiR/docs/superpowers/handoffs/2026-10-12-r7-decide-2-handoff.md first and follow it." The
task was set in the `r7 decide` chat on 2026-10-11: design L from the style source; first CSD without its genre part,
generic sources in parallel (decision B, handoff §2).

AFF is the current best: GO on fresh seeds 49 to 51 (round 3) and on held paintings (round 6: +0.60 R@1 over B′(A0)),
but with no margin over B′(A1) or R1, and a loss on style × genre. The facts brief shows that the style source is the
limit. Told CSD gives style × genre only +0.13 [−0.35, +0.60] R@1 over its matched control. CSD's grouping shares more
with genre (AMI 0.33) than the two label sets share with each other (AMI 0.22). Making a grouping non-redundant with a
known partition is old (Gondek and Hofmann 2004, Cui et al. 2007, FCMI 2023). Its use as the conditions an
example-conditioned scorer steers by is new to our knowledge (novelty note §4).

Goal: a label-free style grouping that carries style apart from genre, so that a reader-steered scorer beats every
condition-free scorer, and AFF, on new episodes.
Question this step answers: does removing the content part of a label-free style source let a reader-steered scorer
beat the strongest condition-free scorers on fresh seeds?

## 2. Scope

In: the ceiling check; six label-free groupings in two families, each with a size-matched random control; NrKmeans on CSD as a descriptive grouping-step comparator; one reader
per grouping (round 1's recipe); R1 and extended AFF on each; the seed-42 gate and pick; fresh seeds 55 to 57 for the
picks; descriptive AMIs; the glossary terms "ceiling check", "extended AFF" and "content grouping".

Out: held rows (C5; a held read of a design L pick gets its own spec after Friday); a wider screen of style encoders
(parked); X-Cluster, FCMI and ClustPy's Orth as comparators of the grouping step (paper comparators, a later round; Orth is
already V1's method); a
second dataset (COCO-4c, later); CSD features at scoring time; changes to the reader recipe; the second backbone.

## 3. Design

### 3.1 Hypothesis and prior

Claim: a style grouping whose content part is removed, label-free, lets the reader tell style episodes from genre
episodes. A scorer that steers by it beats B′(A1) on style × genre, keeps AFF's emotion gains, and beats every
condition-free scorer pooled.

Mechanism: condition gain on style × genre needs a style grouping that differs from the genre side's grouping (the
CLIP image grouping). Removing what the content grouping explains should lower that overlap, so the style term
separates the two conditions.

Prior, before any number (the rule's §1 records the same):
- Ceiling: told true styles on style × genre near +0.9 over its matched control, about 60% that it clears +0.5 with
  its lower bound above 0. True styles are placed at 60.9% from images and 25.4% from captions (27 groups); CSD's
  grouping at 85.1% and 40.4% (17 groups).
- Removal lowers the AMI with the content grouping and with genre. Cui et al.'s projection and the graph cuts should
  keep more style than the ridge residual, since CLIP carries style linearly (novelty note §1, 2b).
- Placement is the main risk: what CLIP does not carry, CLIP heads place poorly, captions above all.
- Seed-42 gate: about 25% overall that one of the twelve scorers passes, about 40% given that the ceiling passes. AFF
  itself is only +0.332 over B′(A1) pooled, so a pass needs about +0.6 R@1 over AFF on style × genre while it keeps
  AFF's emotion pairs.
- Given a pass, about 50% that stage 1 passes on fresh seeds (seed-42 margins shrink after selection). Friday go about
  12% (0.6 × 0.4 × 0.5).
- A random grouping that passes the gate, or a generic variant above every CSD variant, would surprise us.

### 3.2 Arms and controls

The content grouping is E2's CLIP image k-means grouping with 64 groups, the reader's own image grouping (label-free).

| Arm | What it is | Differs from raw CSD by | Role |
|-----|------------|-------------------------|------|
| T | the 27 true style labels as a grouping; CLIP heads place items; scorer told (emotion → affect, style → T, genre → image) | true styles in CSD's place | ceiling |
| V1 | CSD vectors projected off the content groups' mean directions (Cui et al. 2007, method 2), then Leiden | linear content removal | candidate, CSD family |
| V2 | CSD's neighbour graph with edges inside one content group removed (weight 0), then Leiden | graph cut | candidate, CSD |
| V3 | as V2, edge weight 0.5 | half cut | candidate, CSD |
| V4 | V1's vectors, then V2's cut | both | candidate, CSD |
| V5 | CLIP token statistics, CLIP image embedding regressed out (ridge), then Leiden | new source | candidate, generic |
| V6 | CLIP token statistics, V1's projection, then Leiden | new source | candidate, generic |
| R0 | raw CSD, round 1's A1 grouping (17 groups) | none | reference |
| N | ClustPy's NrKmeans on CSD, two subspaces of 17; the clustering least like the content grouping | off-the-shelf non-redundant clustering, no content grouping given | grouping-step comparator, reference |
| Z1 to Z6 | a random grouping with the group sizes of V1 to V6 | random assignment | control, decides nothing |

Each grouping X joins A0's three (affect, image, caption) as a fourth, in CSD's place in A1. Its reader is trained
with round 1's recipe, label-free. Two scorers per grouping: R1[X] (gate opens on the reader's confidence) and
extended AFF X-AFF[X] (gate opens on confidence and a pick of affect or X).

Comparators of every scorer: B, B′(A0), B′(A1), B′(X) (its own B′), its matched control (C2), and AFF (C3). All
fused scorers and their matched controls use the same 224-cell family and per-half cross-fit, so they get the same
tuning freedom.

### 3.3 Data roles

| Data or seeds | Role | Used in stage |
|---------------|------|---------------|
| Scorer-train rows (183,694 rows, 36,518 paintings) | fit transforms, groupings, heads, banks, readers | 1, 2 |
| Seed 42 on selection rows (12,288 episodes, 4,602 anchor paintings) | development: ceiling, gate, pick, thresholds | 0, 1, 2 |
| Style and genre labels of scorer-train rows | the ceiling's grouping (style) and descriptive AMIs only | 1, diagnostics |
| Style labels of seed-42 items | the second ceiling, descriptive | 1 |
| Seeds 55, 56, 57 on selection rows (4,096 per pair each) | test, one look | 3 |
| Held rows | not touched (C5) | none |

### 3.4 Measurement

Metrics, episodes, cross-fitting and the bootstrap follow C9 and round 3's rule (§2, D8, D9). Seed 42: paired per
anchor, 95% intervals. Fresh seeds: pooled over three seeds (36,864 episodes, one cluster per anchor painting across
seeds); Holm across stage 1's five checks; a fixed sequence across the three stages; one-sided 0.0125 per chain when
both families have a pick, 0.025 when one has; decided by round 6's integer count form. Three seeds as in round 3; the
detectable margin of each check is computed from seed 42 before the seeds are built.

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|-------|------|-------|--------------|-------------|
| 0 | regression check | reused code on seed 42 | AFF per anchor via the per-seed cross-fit path, with round 3's chosen cells and σ* (pooled 19.137); told CSD per anchor (style × genre 21.82); the A1 reader retrained equal to round 1's, with its intermediate outputs (posteriors, banks, chosen C) and R1 on A1 per anchor; B and B′(A0) equal to round 3's, AFF minus B′(A1) per episode equal to round 6's | stop; trace the cause; a note for the user |
| 1 | exploration with a stop gate | T on seed 42 | style × genre told margin over its matched control at least +0.5, lower bound above 0 | design L stops; Friday no-go |
| 2 | new idea | V1 to V6, two scorers each, seed 42 | (a) C10 bar over the strongest of the five condition-free scorers; (b) style × genre at or above its matched control; (c) emotion × style and emotion × genre each at least AFF − 0.3 | no scorer passes: design L stops; Friday no-go |
| 3 | claim test | each family's pick on seeds 55 to 57 | F1: five condition-free checks (Holm); then F2: beats AFF; then F3: beats B′(A1) on style × genre, at the one-sided α of its chain (rule §7.5) | read per C11; Friday no-go if F1 fails in every chain |

### 3.6 Diagnostics

Measured, decides nothing; seed-42 and scorer-train only until the fresh-seed verdict is written.
- Per grouping: group count and sizes; AMI with the content grouping (leftover content, label-free); AMI with style
  and genre labels (does removal take genre more than style?); Leiden stability across seeds 42, 43, 44; head
  accuracies; the share of tokens dropped as outliers (V5, V6).
- Per scorer: per-pair R@1, margin and condition gain; pick accuracy under the told map; gate-open shares; the random
  control's numbers beside it (a gain the random grouping matches is not the grouping's).
- The second ceiling (true labels on the seed-42 items): how much placement costs the ceiling.
- After the verdict: per seed and per pair numbers, the gain statistic, the frozen-cell line.

### 3.7 Frozen vs rerun

Frozen from seed 42: the variant definitions, transforms, groupings, heads, readers, thresholds, gate rules, the
224-cell family and the picks. Rerun on each fresh seed: B, B′(A0), B′(A1), B′(X), the reader's probabilities, the gates
against the frozen thresholds, the control's σ*, the fused min-margin and counterpart max-R@1 cross-fits; AFF likewise
from its frozen round-3 parts.

## 4. Decisions

All grill rows are from the `r7 decide` chat (handoff §2); "m" and "s" refer to its numbered minor answers.

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|----|----------|----------|----------------------------|------------|------|
| D1 | Type and size | New idea, full loop | a new mechanism (incremental rejected) | user | 2026-10-11 |
| D2 | Idea | CSD without its genre part first (residual and cut variants), generic sources in parallel; a wider encoder screen parked | the source is the limit (facts) | user, after seeing the recommendation | 2026-10-11 |
| D3 | Generality | Encoders fixed in advance, chosen without the target's aspect labels; the grouping step knows nothing of aspects | a method, not a fit to ArtELingo | user | 2026-10-11 |
| D4 | Stage types | Ceiling: exploration with a stop gate; seed 42: new idea; fresh seeds: claim test, claim per C14 | stricter than confirmation for the fresh seeds | agent default | 2026-10-12 |
| D5 | Ceiling (Q1) | True-style grouping, CLIP heads, genre on the image grouping, told; style × genre vs matched control. Stop below +0.5 or lower bound at or below 0; weak +0.5 to +1.0 (run on, flag, phone note); clear from +1.0 | ours: stop below +1.0 | user | 2026-10-11 |
| D6 | What picks (Q2) | Seed-42 episode R@1; labels only describe | labels picking breaks D3 | user | 2026-10-11 |
| D7 | By Friday (Q3) | Development and a pre-registered fresh-seed test, done by Thursday 2026-10-15 | development only | user | 2026-10-11 |
| D8 | Scorers (Q4) | R1 and extended AFF, each on its retrained reader | ours: extended AFF alone | user | 2026-10-11 |
| D9 | Seed-42 gate (Q5) | All three: C10 bar over the strongest of B, B′(A0), B′(A1), own B′, own matched control; style × genre at or above its matched control (point); emotion × style and emotion × genre each at least −0.3 R@1 against AFF, paired (point) | same as ours | user | 2026-10-11 |
| D10 | Fresh seeds (Q6) | Fixed sequence: F1 five condition-free checks with Holm; F2 a paired win over AFF; F3 style × genre above B′(A1), at the one-sided α of its chain (rule §7.5). Friday: F1 passes → method paper with design L; else ArtELingo paper with AFF | ours: F1 with AFF secondary | user | 2026-10-11 |
| D11 | How many picks (Q7) | One CSD-based and one generic, each chain at one-sided 0.0125 | one pick at 0.025 | user | 2026-10-11 |
| D12 | Generic source | Intermediate-layer CLIP ViT-B/32 token statistics (per-channel mean and standard deviation, high-norm tokens dropped), image direction removed; replaces a second CLIP grouping cut across the first | CLIP's final features cluster by shape (Clevr-4) | user, after seeing the recommendation | 2026-10-12 |
| D13 | Controls and leftover content | A size-matched random grouping per new grouping (V1 to V6; T, R0 and N have none); leftover content as the AMI with the content grouping | a linear probe misses nonlinear content | user, after seeing the recommendation | 2026-10-12 |
| D14 | The residual | Cui et al. 2007's orthogonal re-clustering, cited as such | an uncited ridge residual | user, after seeing the recommendation | 2026-10-12 |
| D15 | Minor answers m1 to m11 | Probe placement for the ceiling, label placement descriptive (m1); CSD variants and the content grouping (m2); CLIP heads only (m3); labels only for the ceiling and AMIs, seed 42 only, no held rows (m4); comparators (m5); folder and prefix `r7_` (m6); round 1's reader recipe (m7); seeds 55 to 57, 4,096 per pair, C9 (m8); the pick rule (m9, m11); F3 as superiority (m10) | handoff §2 | user, after seeing the recommendation | 2026-10-12 |
| D16 | s1, s2, s3 | The winner goes on even with a worse genre AMI than raw CSD (s1); no pass: stop, Friday no-go (s2); CSD fails and a generic passes: generic alone at 0.025 (s3) | s3 departs from D2's order | user, after seeing the recommendation | 2026-10-12 |
| D17 | Cui's method 1 or 2 | Method 2 (projection off the span of the 64 content-group means, on centred vectors) | method 1 removes one direction per item, too little | agent default | 2026-10-12 |
| D18 | CSD's fourth slot | V4 = V1's vectors with V2's cut | the plan page's fourth slot | agent default | 2026-10-12 |
| D19 | Token statistics | Blocks 4, 6, 8 of 12; 49 patch tokens; tokens above 3 × the image's median norm dropped; z-score, PCA to 256; V5 ridge on the CLIP image embedding, V6 V1's projection | brackets removal strength (one variant only) | agent default | 2026-10-12 |
| D20 | Only CSD passes | Its pick goes alone at 0.025, as s3 | 0.0125 for a chain that runs alone | agent default | 2026-10-12 |
| D21 | Thresholds of each new reader | Percentiles 0, 25, 50, 75 of its seed-42 top-two margins, as round 3's D6 | round 1's A0 values on another reader | agent default | 2026-10-12 |
| D22 | Regression checks | AFF, told CSD, B′(A1), plus the A1 reader retrained by the new path | AFF and told CSD only | agent default | 2026-10-12 |
| D23 | Random controls | Descriptive; a pick whose random control also passes the gate is flagged | a fourth gate item (the user fixed three) | agent default | 2026-10-12 |
| D24 | Pick ties | Larger gain statistic, then the order V1 to V6, R1 before extended AFF | none needed in practice | agent default | 2026-10-12 |
| D25 | A Friday go and C3 | A pick replaces AFF as current best only after F2 passes | reading C3 | agent default | 2026-10-12 |
| D26 | Late run | Not complete by Thu 2026-10-15 23:59 → FV NO-GO, design L resumes only with the user's approval | a partial verdict | agent default | 2026-10-12 |
| D27 | A grouping-step comparator in this round | N: ClustPy's NrKmeans on CSD, descriptive on seed 42; details as rule L7 | a reviewer will ask why not an off-the-shelf non-redundant clustering; Orth is already V1 | user, after seeing the recommendation ("That could be added"); details agent default | 2026-10-12 |

**Amendments:** none.

## 5. Assumptions and open questions

- Assumption: round 1's reader recipe is deterministic, so the A1 reader retrained by the new path equals round 1's,
  and X keyed `csd` reproduces A1's block layout. Checked: stage 0.
- Assumption: the token statistics come from the same CLIP weights as the cached image features. Checked: in the build
  (recomputed embeddings match the cache).
- Assumption: each painting has one image, so painting vectors are well defined. Checked: round 1's grouping code
  asserts it.
- Assumption: every variant's Leiden grouping keeps at least 2 groups after merging small ones. Not checked; a variant
  that does not is dropped and reported.
- Assumption: the content grouping is constant within a painting. Checked in the build.

No open questions: the grill settled the bar, the data and the scope.

## 6. Success criteria

**Decision mode.** Binding rule: `src/test/20261126_design_l_style_source/DECISION_RULE.md` @ c95b6ea,
committed before any code; it governs where it differs from this spec. Its criteria:

- G1 to G4: the regression checks (stage 0).
- K1: the ceiling's stop, weak and clear zones (stage 1).
- S1 to S3: the seed-42 gate; P1: the pick per family (stage 2).
- F1 to F3: the fresh-seed chain per pick; FV: the Friday verdict (stage 3).

**Promising and dig budget** (stage 2). Promising: no scorer passes, but one has style × genre at least +0.3 R@1
above AFF, at or above its matched control, and a pooled bar margin above 0. Dig budget: none before Friday; the
no-go stands, and the result goes to the direction log's parked ideas for decision A of the next loop.

**Licensed claim.** F1: "<pick> beats B, B′(A0), B′(A1), its own B′ and its matched control on R@1, pooled over the
three aspect pairs, on new episodes from the same selection paintings." F2 adds "and AFF"; F3 adds "and B′(A1) on
style × genre episodes". Not transfer to new paintings, not a margin on each pair; paper claims need a held read (C5,
C14).

**Selection rule.** Stage 2's pick (rule P1): per family, the largest pooled bar margin among the scorers that pass
the gate; ties as D24.

## 7. Constitution check

| Principle | Pass or exception | Note |
|-----------|-------------------|------|
| C1 | pass | B′(A1), the strongest condition-free scorer, is in every bar |
| C2 | pass | every scorer, the ceiling included, against its matched control |
| C3 | pass | F2; a pick replaces AFF only after F2 (D25) |
| C4 | pass | development on seed 42; fresh seeds 55 to 57, hash-checked and ledgered |
| C5 | pass | no held row |
| C6 | pass | joins through `row_to_image` and the extraction sample IDs |
| C7 | pass | the rule is reviewed on Opus and committed before code |
| C8 | pass | prior in §3.1 and the rule; disclosure in §8 |
| C9 | exception | Holm and a fixed sequence at one-sided 0.0125 per chain, integer form, in place of each 95% lower bound (rule §11) |
| C10 | pass | the bar's comparator set is wider (stricter) |
| C11 | pass | failed checks read with their detectable margins |
| C12 | pass | stage 0 |
| C13 | pass | verdict numbers re-derived by separate code before each rule step |
| C14 | pass | the licensed claim above |

Exceptions: C9 · several claims share one family-wise one-sided 0.025 (the user, D10 and D11) · each check at 95%
would leave up to 14 claims uncorrected.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|------------------------------------------------|---------------------|-----------------|
| Removal also deletes style; groupings near noise | AMI with style near the random control's; head accuracies near chance | reported; the gate decides |
| CLIP heads cannot place the new groups (captions) | caption head accuracy near the majority share | reported; it explains a fail |
| A gate pass comes from retraining the reader, not from the grouping | the random control passes too | flagged at the top of the report (D23) |
| Seed-42 selection inflates the pick | fresh-seed margins far below seed 42's | the fresh seeds decide; reported beside the disclosure |
| Reused code changed silently | stage 0 | stop and trace |
| Token extraction misaligned with rows | C6 hash checks in the extraction meta | stop the generic family |

**Selection disclosure.** On seed 42: about 50 label-free variants before AFF (round 3's brainstorm), about 10 in
rounds 4 and 5, and this round's 12 candidate scorers (plus 4 reference scorers and 12 random-control scorers, which pick nothing).
Seed-42 margins of a pick are therefore inflated. The fresh seeds protect the verdict; the report gives each pick's
seed-42 margin beside its fresh-seed margin.

## 9. Deliverables and review

- At the end: code and results in `src/test/20261126_design_l_style_source/` (prefix `r7_`); token statistics under
  `/data/SSD2/pre_extract/artelingo/`; the auto report in `docs/reports/auto/v2/` with its results brief, and a
  user-read report in `docs/user_read/`; ledger rows for seeds 55 to 57.
- Reviews: the rule by a fresh Opus reviewer before commit (C7); ticket reviews for verdict code (gate, pick, chain,
  bootstrap counts); C13 re-derivation before each rule step; an end-to-end smoke of the stage-3 code on smoke seeds
  9001 to 9003 before any build; the final whole-branch review (`final-review.md`).
- Stop points: none between approval and the reports (build and run run unattended); a stage 0 failure, a ceiling
  stop or a gate with no pass ends the run early with its report.
- Budget: local CPU about 4 hours at 8 threads (14 readers: V1 to V6, Z1 to Z6, N, and A1 for G3; about 10 min each,
  three at a time; groupings, heads and scoring under an hour); the local GPU about 15 min under its lock (token
  statistics for 61,402 images); about 1.2 GB of new features. No cluster. Time-box: report by Thursday 2026-10-15;
  Friday 2026-10-16 go or no-go. Not complete by Thu 2026-10-15 23:59: no partial verdict, FV is NO-GO (D26; rule §8).
