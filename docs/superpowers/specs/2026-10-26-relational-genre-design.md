# The relational genre term: a sweep on seed 0, a pick on fresh seeds and a held genre claim test: design

> Date: 2026-10-26 19:35 · Status: approved 2026-10-26 19:57 (from the brief, r10 decide 25) · Kind: experiment
> Builds on: [report 12-25](../../reports/auto/percept/2026-12-25_encoders_genre.md) (stage iii), its
> [spec](2026-10-25-encoders-and-genre-design.md) and [rule](../../../src/test/20261025_encoders_genre/DECISION_RULE.md);
> the [literature note 12-26](../../reports/literature/2026-12-26_relational_genre_term.md); the grill in the r10
> decide 24 chat (its record: the [decide 24 handoff](../handoffs/2026-10-26-r10-decide-24-handoff.md) §4 and this
> spec's §4) · Constitution: [constitution.md](../constitution.md)
> Decision rule: `src/test/20261027_relational_genre/DECISION_RULE.md` @ 2e9529a (first commit, after the review)

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** The relational genre term at weight 8 reached the instructed Qwen genre image's level on
development paintings (report 12-25). This loop sweeps the term further, picks the best variant on fresh seeds, and
tests it once on held paintings against the instructed genre image, the strongest genre reading we have.

**Plan.**
1. Stage A (sweep, development, seed 0): weights 16 and 32, then higher rungs while the ladder still rises; the
   relational term alone; a sharper teacher temperature (0.02, 0.01) at weight 8; teacher anchors at weight 8. The
   guarded arm with the highest genre is the seed-0 winner.
2. Stage B (confirmation, development, seeds 1 to 3): weights 2, 4 and 8 and a new seed-0 winner. The confirmed arm
   with the highest three-seed mean goes to held; weight 8 competes like the others.
3. Stage C (claim test, held): the picked variant trained at full scale, 5 seeds, tested on 9,949 labelled held
   paintings.

**Compared against.** Stages A and B: the current recipe (same data and seeds). Stage C: the instructed Qwen genre
image, 0.5062 on held (the recipe 0.4534; the strongest PercepT entry 0.4269).

**Outcomes.** AMI: agreement of k-means clusters with the labels. Seed range: the spread of the earlier model over 3
seeds (genre 0.0128).
- Stage A: a new winner when a new arm beats weight 8 (0.5018) within the guard (emotion and style lose at most one
  seed range); else weight 8 stays.
- Stage B: an arm is confirmed when it gains at least one seed range over the recipe on fresh seeds, interval above 0,
  guards met; the confirmed arm with the highest mean is picked.
- Stage C: GO when the student is level with the instructed genre image within one seed range (lower bound above
  −0.0128); partial when it closes at least half the held gap over the recipe (at least +0.0264, interval above 0);
  margin met and NO-GO as the other outcomes.
- We expect (guesses): a new seed-0 winner 0.6; the held variant not weight 8, 0.4; held: GO 0.15, margin met 0.15,
  partial 0.50, NO-GO 0.20.

**Cost.** Local 3090, light: about 1 GPU hour (around 25 short runs) and 3 CPU hours of scoring. Build tonight; the
run on 2026-10-27; reports by its evening.

**Check these.**
- Taken after our recommendation: the one-loop shape with a held test at the end (decide 23); the held bar with its
  partial level (option c); the anchors equivalence test as a build gate.
- Agent defaults: stage A typed as a sweep although the anchors arm is a new mechanism; "rises" as strictly above
  every other guarded rung so far, with the cap at 256; the four held outcomes; the equivalence test's form (the
  anchors loss with the batch's own student and teacher rows as anchors equals the in-batch loss).
- Most likely wrong: the held gap (0.0527) is almost twice the development gap (0.0295), so GO is unlikely even if the
  term works as on development; the full-scale lift is unknown.

## 1. Context and goal

Asked (decision A, r10 decide 23, 2026-10-26 16:40): "close the genre gap first" (option 2 then 1); decision B: all
three candidates in the sweep, "cheap; I don't really know what the sweep affects". The grill (this chat): "a higher
bar is necessary", then "K1 maybe (c) would actually be better, it's a bit soft but acceptable, I will switch to (c)";
"we always want better variants so possibly 'compete'"; "we can keep growing if 64 still shows growing trend, then
we try 128"; "Full candidate".

Report 12-25's stage iii added a relational term (the glossary: a loss that makes each painting's pattern of
similarities to the other paintings of the batch copy the teacher's) to the genre head of the current recipe. At
weight 8 it reached genre AMI 0.5018 on seed 0 against the recipe's 0.4674 and the instructed Qwen genre image's
0.4989 on 29,232 development paintings; on fresh seeds 1 to 3, 0.5022 against 0.4694 (+0.0328 [+0.0235, +0.0442]),
emotion and style unchanged. It was picked among seven variants on one seed, and only weight 8 ran on fresh seeds. On
held paintings the recipe trails the instructed genre image by 0.0527 (12-24), almost twice the development gap. The
literature check (12-26) found no published sweep with a steadily rising gain, no head-to-head of the relational term
alone against beside the direct term, and no label-free genre comparator for paintings: the instructed image stays
the bar, and the margin is ours to set before the read.

Goal: find the best form and weight of the relational genre term on development data, then test the selected variant
once on held paintings against the instructed genre image.
Question this step answers: does a model that reads the default image give genre clusters level with the instructed
genre image on held paintings, and if not, how much of the gap does it close?

## 2. Scope

In: the sweep's arms (§3.2), fresh seeds for weights 2, 4, 8 and a new winner, one held genre test of one variant at
full scale, the diagnostics of §3.6, the auto report with its user-read report.
Out: style (the same idea on the style head waits for this result; next decide chat); other encoders (the term is
tested on Qwen only); the write-up of the encoder claims (after this loop); aspect-named retrieval for the 11-10 gate
(decide 20); the centroid form of the anchors (parked, D12); a held test of a second variant.

## 3. Design

### 3.1 Hypothesis and prior

The default Qwen image holds the genre information (the ceiling probe reaches 0.5040 from it, 12-24), and the
relational term lets the genre head use it: the in-batch similarity pattern is a softer target than the painting's own
teacher vector, so the head copies the teacher's neighbourhoods without copying its coordinates. If that is the
mechanism, a higher weight, a sharper teacher row or a larger anchor set may copy the pattern more closely, until the
shared trunk pays on emotion or style. The prior, written before any number (rule §1): a new arm beats weight 8 on
seed 0 within the guard with probability 0.6; on held, the likeliest outcome is partial (0.50), since a held lift the
size of the fresh-seed lift (+0.033) leaves the student about 0.020 under the instructed image. A GO (0.15) or a NO-GO
(0.20) would both surprise us less than a held lift above 0.05.

### 3.2 Arms and controls

Every arm is the current recipe with one change to the genre head's loss; the emotion and style losses are unchanged,
and the shared trunk sees every change. The comparator of stages A and B is the recipe itself on the same paintings
and seeds (its exports frozen from 12-24 and 12-25).

| Arm | What | Differs from the recipe by | Role |
|-----|------|----------------------------|------|
| GR-8 | the in-batch relational term at weight 8 beside the direct term (12-25's run, frozen) | one added genre term | candidate; the fallback |
| GR-16, GR-32; GR-64, GR-128, GR-256 | the same at higher weights; the last three only while the ladder rises (rule §4.1) | the weight | candidates |
| GN-8 | weight 8 with the direct term removed from the genre head | one term removed, one added | candidate |
| GS-0.02, GS-0.01 | weight 8 with the teacher's row at temperature 0.02 or 0.01 (the student's at 0.05) | the teacher's temperature | candidates |
| GQ-8 | weight 8 with teacher anchors (the glossary: 4,096 fixed teacher rows) in place of the in-batch rows | the rows compared | candidate (full, behind the equivalence test) |
| GR-2, GR-4 | weights 2 and 4 on fresh seeds 1 to 3 | the weight | candidates (stage B) |
| R6-5 | the current recipe | — | comparator |
| T, probe | the instructed Qwen genre image; the ceiling probe's prediction | no training | references (development); T is the held bar |

Stage C's arms: the picked variant at full scale (five seeds) against T on held; the recipe's saved held outputs (K4,
12-24) for the partial level; PercepT core's entries, the frozen image and Leiden reported only.

### 3.3 Data roles

| Data or seeds | Role | Used in stage |
|---------------|------|---------------|
| 13,400 fit and 1,489 stopping paintings (the frozen sample) | training; label-free early stopping | A, B |
| 29,232 development paintings P (genre labels on 23,682) | development: stage A's effects and winner, stage B's means and pick | A, B |
| the old 5,000 evaluation paintings | development, reported | A, B |
| model seed 0; seeds 1 to 3 | development: the pick; the confirmation and the pick on fresh seeds | A; B |
| 42,969 train and 6,152 validation paintings | training and stopping at full scale | C |
| 12,281 held paintings (9,949 with a genre label) | held: images for inference, captions for labels only | C |
| ArtELingo-28 (746 paintings) | not touched: it has no genre labels | — |

P, seeds 0 to 3 and the old 5,000 chose 12-24's recipe and 12-25's winner; they stay development data. No held number
enters any choice (`held-data.md`): the held variant is written to the stage B verdict file before any full-scale run
of a candidate, and only that variant is tested.

### 3.4 Measurement

- **Metric.** AMI of k-means partitions (20 clusters, three random states, mean) against the genre labels; emotion
  and style the same way for the guards. Leiden reported beside.
- **Stages A and B.** The 12-25 rule's stage iii statistics (rule §3.1): effect = arm minus the recipe on P at seed 0
  with a paired bootstrap over paintings (2,000 draws); on fresh seeds, three-seed means and a paired bootstrap over
  paintings and seeds. Seed ranges 0.0046 / 0.0082 / 0.0128 as the units. One seed to pick, three fresh seeds to
  confirm and pick among the fresh-seed arms (`held-data.md`, "Seeds").
- **Stage C.** The claim tests' held protocol (rule §3.2): the student's five seeds, T's three partitions; d_T =
  student minus T, d_R = student minus the recipe's saved held outputs; paired bootstraps over paintings and seeds,
  2,000 draws; the margin X = 0.0128; the half gap G_half = 0.0264, recomputed from the frozen files. Five seeds, as
  every held claim test of this line.

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|-------|------|-------|--------------|-------------|
| VA1 | regression (C12) | the new trainer, switches off at seeds 0 and 1; with the weight-8 switch at seed 0 | the recipe's exports on P and 12-25's weight-8 exports, bit for bit; 12-25's seed-0 scores within 1e-6 | stop; trace; nothing runs in A, B or C |
| EQ | regression | the anchors loss with the batch's own student and teacher rows as anchors, on three real batches | equals the in-batch loss within 1e-6 | GQ-8 runs but cannot win; cause in the report |
| A | sweep, seed 0 | GR-16, GR-32, the climb, GN-8, GS-0.02, GS-0.01, GQ-8 | A-NEW or A-SAME | — |
| VB1 | regression (C12) | the fresh-seed scorer on 12-25's frozen fresh-seed exports | their hashes and 12-25's fresh-seed effects within 1e-6 | stop B and C; trace |
| B | confirmation, seeds 1 to 3 | GR-2, GR-4, a new winner; GR-8 and the recipe frozen | B-PICK (a held variant) or B-NONE | B-NONE: no held test; report |
| VC1 | regression | the full-scale trainer, switches off; its held inference; the held scorer on 12-24's K4 outputs | K4's seed-0 weights and held outputs bit for bit; K4's held margins within 1e-6 | stop; trace; no held output |
| C | claim test, held | the picked variant, five seeds at full scale | G-GO, G-MM, G-PART or G-NO | — |

Order: VA1 and EQ, then A (GR-16 and GR-32 first, the climb, then the other arms in any order), then B, then VC1 and
C. VA2, VB2 and VC2 (the re-derivations by separate code, C13) run before each stage's outcome is stated. The run chat
owns the queue; a failed gate stops only the stages behind it.

### 3.6 Diagnostics (decide nothing; after each stage's outcomes are written)

- **Loss scale.** The mean value of the direct term and of the relational term over the first epoch, from one run
  with the recipe's settings at seed 0 and the relational weight 1, both terms logged per step, no export on P (rule
  §5), so a reader can compare our weights with CompRess's and CLIP-KD's (the literature note's suggestion); GQ-8's
  loss value beside GR-8's.
- **Stage A and B.** Each arm's three aspects with intervals; the old 5,000; Leiden; best epochs; each arm's level
  against T_g; whether the fresh-seed arms sit within one seed range of each other.
- **Stage C.** The student's emotion and style on held with the claim tests' margins; its genre against PercepT
  core's entries and the frozen image; Leiden; seed spreads; best epochs.

### 3.7 Frozen vs rerun

Rule §5. In short: the recipe's exports on P (seed 0) and on fresh seeds, weight 8's exports and weights (seeds 0 to
3), the recipe's full-scale weights and held outputs, T's held store and the held list are frozen; every new arm,
the fresh-seed runs of weights 2 and 4 and a new winner, the picked variant at full scale and all scoring are new.

## 4. Decisions

Times are Amsterdam time on 2026-10-26, from the user's messages in the r10 decide 23 and 24 chats.

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|----|----------|----------|----------------------------|------------|------|
| D1 | Direction (A) | close the genre gap first: more development work on the term, then a held genre test, as one full loop with three stages | a held test of weight 8 alone (option 1); pause genre and write the encoder claims (option 3) | user (the direction); user, after seeing the recommendation (the one-loop shape; flag) | decide 23, 16:40 |
| D2 | Idea (B) | all three candidates in the sweep: the ladder up, relational alone and a sharper temperature at weight 8, teacher anchors | one candidate at a time ("cheap; I don't really know what the sweep affects") | user | decide 23 |
| D3 | The held bar (K1) | option c: GO when level with the instructed genre image within X; a partial level when the student closes at least half the held gap over the recipe, interval above 0 | option a alone (the user's first pick; a likely NO-GO would leave no sentence); option b (above PercepT by 0.01, which the recipe already meets) | user, after seeing the recommendation (flag) | 19:05 |
| D4 | The margin X | one genre seed range, 0.0128 | half the held gap, 0.026 ("a higher bar is necessary") | user | 19:05 |
| D5 | The four held outcomes | GO, margin met (inconclusive, with the partial condition reported), partial, NO-GO; "matches or exceeds" when the lower bound is above 0 | three outcomes without the partial level | agent default | 19:05 |
| D6 | Stage A's pick (K2a) | the guarded arm with the largest unrounded genre AMI on seed 0, weight 8 included; a new winner goes to fresh seeds; weight 8 is the fallback | a pick without a confirmation step | user | 18:58 |
| D7 | Weights 2 and 4 on fresh seeds (K2b) | they compete: the held variant is the confirmed arm with the largest three-seed mean among weights 2, 4, 8 and a new winner | readings only (the ladder's shape without a pick; "we always want better variants") | user | 19:10 |
| D8 | The ladder's stop (K2c) | climb while the ladder rises: the next rung runs while the newest rung is guarded and strictly above every other guarded rung so far; cap at 256 | stop at 32; one rung at 64 only ("keep growing if 64 still shows growing trend, then we try 128") | user (the climb); agent default (the cap and the reading of "rises") | 19:20 |
| D9 | The anchors arm (K3) | a full candidate under the same rule | an exploration arm whose result is a reading only | user | 18:58 |
| D10 | The anchors' gate | the equivalence test (the anchors loss with the batch's own student and teacher rows as its two anchor sets equals the in-batch loss) before it can win | no gate | user, after seeing the recommendation (flag) | 19:20 |
| D11 | Teacher temperature arms (M1) | two arms at weight 8: the teacher's row at 0.02 and 0.01, the student's at 0.05 | one value; both rows sharpened | user, after seeing the recommendation | 19:20 |
| D12 | The anchors' form (M2) | 4,096 teacher rows of the fit paintings, drawn once per run from the model seed; temperature 0.05; the direct term stays | 200 k-means centroids of the teacher (parked) | user, after seeing the recommendation | 19:20 |
| D13 | Relational alone (M3) | one arm at weight 8, the direct term's weight 0 on the genre head | a ladder of its own | user, after seeing the recommendation | 19:20 |
| D14 | The held protocol (M4, M5) | the claim tests' protocol; the recipe's saved held outputs, PercepT core, the frozen image and Leiden reported beside; ArtELingo-28 not read | new held readings of unselected arms | user, after seeing the recommendation | 19:20 |
| D15 | Regression gates (M6) | VA1 (sample scale, switches off and weight 8), EQ, VC1 (full scale), re-derivations VA2, VB2, VC2 | a gate at the sample scale only | user, after seeing the recommendation | 19:20 |
| D16 | Diagnostics (M7) | §3.6 | none | user, after seeing the recommendation | 19:20 |
| D17 | Names and places (M8, M9) | round folder `src/test/20261027_relational_genre/` (prefix `rg_`); glossary entries "relational term" and "teacher anchors"; "teacher term" stays the name of the direct term | a new name for the direct term | user, after seeing the recommendation | 19:20 |
| D18 | Timing (M10) | build tonight, unattended; the run on 10-27 (light GPU work) | a weekend run | user, after seeing the recommendation | 19:20 |
| D19 | The stage types | A sweep (the anchors arm, a new mechanism, is judged by the same selection rule); B confirmation with a selection rule; C claim test | A as a new idea with its own go bar | agent default | 19:35 |
| D20 | The plan as a page | no html-plan page; the closing summary in chat | a page | user | 19:20 |

**Amendments** (changes after approval; "none" until there is one)

| ID | When | Change | Reason | Decided by | Before or after results | Places updated |
|----|------|--------|--------|------------|-------------------------|----------------|
| none | — | — | — | — | — | — |

## 5. Assumptions and open questions

- Assumption: the new trainer wraps 12-25's genre loss unchanged, so VA1 passes bit for bit. Checked: VA1 (rule
  §4.1) before any arm runs.
- Assumption: the recipe's full-scale run at seed 0 reproduces 12-24's K4 weights bit for bit with the current
  environment (the X2 gate of 12-25 did so for K2 and M2). Checked: VC1.
- Assumption: the fresh-seed lift of weight 8 (+0.033) carries to held at full scale in a similar size. Not checked;
  the prior puts the partial outcome at 0.50.
- Assumption: a weight of 64 and above trains with a finite loss. Not checked; a non-finite loss ends the climb
  (rule §4.1).
- Open questions: none; the grill settled the key and minor questions (§4).

## 6. Success criteria

**Decision mode.** Binding rule: `src/test/20261027_relational_genre/DECISION_RULE.md` at the commit in the header,
committed before any code; it governs where it differs from this spec. Its criteria:
- Stage A: A-NEW, A-SAME; the guard; the climb; validity VA1 to VA4 and EQ.
- Stage B: confirmed (C-HOLDS), B-PICK, B-NONE; validity VB1 to VB4.
- Stage C: G-GO, G-MM, G-PART, G-NO; validity VC1 to VC4.

**Selection rules.** Stage A (rule §4.1): the guarded arm with the largest unrounded genre AMI on P at seed 0 among
weight 8 and the new arms; ties to weight 8, then the listed order. Stage B (rule §4.2): among the confirmed arms on
seeds 1 to 3, the largest unrounded three-seed mean; ties to weight 8, then weights 2, 4, then the new winner.

**Licensed claims.**
- Stages A and B: "On 29,232 development ArtELingo paintings (Qwen; picked on one seed and on three fresh seeds),
  <variant> gives genre AMI <three-seed mean> against the recipe's <mean> (instructed Qwen genre image: 0.4989), with
  emotion and style within one seed range of the recipe." At most "promising on development data".
- Stage C, G-GO: "On 9,949 held ArtELingo paintings with a genre label, a model whose image side reads the default
  Qwen image, trained on the 42,969 train paintings with Qwen3-VL-Embedding's instructed graphs, instructed teacher
  and default captions, gives genre clusters (k-means, 20 clusters) level with the instructed Qwen genre image within
  one seed range (0.0128): <d_T> [interval], 5 seeds, paired 95% interval over paintings and seeds." With the lower
  bound above 0: "matches or exceeds" in place of "level with ... within one seed range".
- Stage C, G-PART (and G-MM when the partial condition holds): "...closes at least half the gap between the current
  recipe and the instructed Qwen genre image: <d_R> [interval] of the recipe's <gap> under it." G-MM alone: no claim;
  reported as inconclusive. G-NO: no claim.
- Not covered: style, emotion (readings only), other encoders, other datasets, training without Qwen, retrieval.

## 7. Constitution check

| Principle | Pass or exception | Note |
|-----------|-------------------|------|
| C1 strongest real comparator | pass | the instructed Qwen genre image on held (0.5062), the strongest genre reading; the recipe for the partial level; PercepT reported |
| C6 sample IDs | pass | the exports keep P's and H's orders; VA3 and VC3 check the row counts and hashes |
| C7 rule before code | pass | the rule reviewed by a fresh Opus `xhigh` worker and committed before any code |
| C8 prior and disclosure | pass | rule §1; §8 below |
| C12 regression first | pass | VA1, EQ, VC1 |
| C13 independent re-derivation | pass | VA2, VB2, VC2 by separate code, given the rule and the exports, never our numbers |

Exceptions: none.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|---|---|---|
| A high weight costs emotion or style through the shared trunk | the guard; each arm's emotion and style effects | the guard blocks the arm; reported |
| A high weight gives a non-finite loss | the trainer stops the run | the rung ends the climb; reported |
| The anchors code is wrong | EQ | GQ-8 cannot win; reported |
| The pick on fresh seeds is by a difference under one seed range | the arms' three-seed means | the report says so; the pick follows the rule (D7) |
| The held gap is larger than the development gap | the prior; the held outcome | partial or NO-GO; the sentence follows the outcome |
| The full-scale run does not reproduce 12-24's weights | VC1 | no held output; the cause traced to the user |

Selection disclosure. P and seed 0 chose 12-24's recipe among six and 12-25's winner among seven variants; this loop
adds six to nine arms on the same paintings and seed, so the seed-0 winner's gain is inflated by the pick. The fresh
seeds protect, but they now also pick among three to four arms, by a rule fixed here before any number. The held
paintings were read by the claim tests and 12-24 (the recipe's genre, 0.4534, and T's, 0.5062, are known and are the
comparators here); no choice was made on them and none is made here: the held variant is written down before any
full-scale run of a candidate, and only it is tested. The arm list, the ladder's cap and the bar were set before any
new number.

## 9. Deliverables and review

- At the end: round folder `src/test/20261027_relational_genre/` with the rule, the verdict files per stage, the pick
  file and the re-derivation files (outputs on `/data/SSD2`, gitignored); one auto report (12-27) in
  `docs/reports/auto/percept/` with a results brief, and a user-read report; a row in
  `docs/superpowers/held_ledger.md` for the held read.
- Reviews: the rule and this spec by a fresh Opus `xhigh` worker before the commit (2026-10-26 19:55: pass after
  fixes; four must-fix, eleven should-fix and ten nit findings, all applied); the build per `matt-chain.md`, with ticket
  reviews and unit tests for verdict code (the three switches and the anchors loss with EQ, the climb, the seed-0
  winner, the stage B statistics and pick, the held scorer with d_T and d_R, the re-derivations); the final
  whole-branch review; Claude's claim review of the auto report.
- Stop points: none between approval and the reports, except a failed gate.
- Budget: GPU about 1 hour on the local 3090 under the lock (about 25 runs of about 2 minutes: six to nine stage A
  arms, nine fresh-seed runs, five full-scale seeds, three regressions, one loss-scale run); CPU about 3 hours at 8
  threads (k-means on P for every new export, the held partitions, the bootstraps, the re-derivations). Light work:
  the weekday and 11:00 rules are not touched. Time-box: reports by 2026-10-27 22:00.
- Length: the body is about 245 lines against about 210 for a claim test plus two stages, because of the twenty
  decisions carried from the grill.
