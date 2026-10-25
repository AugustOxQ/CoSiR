# Four more encoders against PercepT on held paintings, the student's lift per encoder on development paintings, and genre variants toward the instructed Qwen image: design

> Date: 2026-10-25 18:00, revised 18:30 after the rule review and the cold check · Status: approved 2026-10-25 18:35
> (from this brief, r10 decide 21; the closing summary was confirmed in r10 decide 20 at 17:29 by the chat log) · Kind:
> experiment
> Builds on: [report 12-24](../../reports/auto/percept/2026-12-24_recipe_backbones.md), its
> [spec](2026-10-25-recipe-and-backbones-design.md) and [rule](../../../src/test/20261025_recipe_backbones/DECISION_RULE.md);
> the grill record in the [decide 21 handoff](../handoffs/2026-10-25-r10-decide-21-handoff.md) §2 · Constitution:
> [constitution.md](../constitution.md)
> Decision rule: `src/test/20261025_encoders_genre/DECISION_RULE.md` @ f0f7c91 (first commit, after the review)

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** (i) Four more image encoders get the claim test that SigLIP 2 and Qwen passed in 12-24. (ii) For
all seven encoders, on development paintings: how far the student rises above that encoder's own PercepT arm, and
which encoder properties go with the rise. (iii) Five loss variants try to lift genre toward the instructed Qwen image.

**Plan.**
1. Stage i (claim test, held): CLIP B/16, CLIP L/14 at 224 px, SigLIP 1 So400m/14 at 384 px, DINOv2-large, under
   12-24's rule with the new encoders, DINOv2 on the new recipe, and new checks of the code and the new feature
   stores. Sample scale, then full scale unless NO-GO, then ArtELingo-28.
2. Stage ii (confirmation, development): the seven students trained on 13,400 paintings and their PercepT arms,
   scored on 29,232 development paintings, with encoder statistics. Then the held table, which chooses nothing. A
   method idea from it goes to development paintings first, in a later loop.
3. Stage iii (incremental, development, Qwen, one seed): five genre variants, then two more weights for the closest
   method; the best variant within the guard runs on three fresh seeds if it gains one genre seed range.

**Compared against.** Stage i: the strongest PercepT entry (the bar), the encoder's own PercepT arm included (PercepT
with that encoder's image in its Stage 2). Stage ii: that own arm. Stage iii: the current recipe; target 0.4989.

**Outcomes.** AMI: agreement of k-means clusters with the labels. Seed range: the spread of the earlier model over 3
seeds (genre 0.0128).
- Stage i, per encoder: GO when emotion AMI is at least 0.01 above the bar, interval above 0; else inconclusive or
  NO-GO (not above the bar). One verdict per encoder, no pooled sentence; how the paper reports them is open.
- Stage ii, per encoder: "holds" when the student is above its own arm on emotion, interval above 0.
- Stage iii: GO when genre reaches 0.4989, promising from 0.4861; no win if emotion or style loses over one range.
- We expect (guesses): SigLIP 1 the likeliest GO (0.45), DINOv2 the least (0.10); stage ii holds for most encoders;
  stage iii reaches neither level (0.60). Not in this loop: retrieval (next), the abstract, a held genre test.

**Cost.** Local 3090: about 3 to 3.5 GPU h and 5 to 6 CPU h (a guess; the new extractions are not measured). Build
tonight; the run after it if it can end by 11:00, else Monday 19:00. Reports by Tuesday 2026-10-27 10:00 at the latest.

**Check these.**
- Taken after our recommendation: the four encoders, the stage iii guard, and "keep weights" as both saving close
  variants and two more weights for the closest method.
- Agent default: stage ii as a confirmation with an outcome per encoder, not an exploration.
- Agent defaults in stage iii: the new terms sit beside the existing teacher term; regression weight 1; the two extra
  weights follow a fixed ladder; fresh seeds start on the point gain; a fresh-seed failure is reported "not repeated".
- Agent default: a content check per new store (image-to-caption retrieval at least 1%) can stop a broken encoder.
- Most likely wrong: the extraction time, so the run's end time.

## 1. Context and goal

Asked (decision A, r10 decide 20, 16:17): "I suggest doing (1) A deeper analysis of how different backbones work, and
how could that help us potentially improve the number better. (2) The teacher-faithful heads or losses as suggested.
(3) Extra backbone, as CLIP B/16, CLIP V/14, and SigLIP 1. Simply check a broader range of models with the current
setting." Then "as above" (16:23) to the three-stage medium loop we proposed.

Report 12-24 replaced D2 by R6-5, the recipe without own-caption terms at teacher share 1/5, and found that students on
SigLIP 2 (0.1024) and Qwen (0.1029) pass held emotion against PercepT core's 0.0874 at both scales, while CLIP B/32
(0.0885) is inconclusive. Every student stays below the instructed Qwen image on genre, although the ceiling probe
finds the genre information in the default image (0.5040 against 0.4989 on development paintings).

Goal: test the emotion lead on more encoders, explain how the lift differs between encoders, and narrow the genre gap.
Questions this step answers: (i) does a student on CLIP B/16, CLIP L/14, SigLIP 1 or DINOv2-large beat the strongest
PercepT entry on held emotion by 0.01? (ii) on development paintings, is each student above its encoder's own PercepT
arm, and does the lift follow the encoder's starting level or a property of its image space? (iii) does a
teacher-faithful genre loss bring the student's genre AMI to the instructed Qwen image's?

## 2. Scope

In: stage i (new stores, students, matched arms, sample and full scale, ArtELingo-28); stage ii (P inference and
scoring of seven encoders, encoder statistics, the held table); stage iii (five variants, two extras, fresh seeds).
Out: aspect-named retrieval and the 11-10 gate (the next loop, user); the abstract (not now, user); a held test of a
genre winner (a later loop's claim test, user); a method idea from stage ii (development first, a later loop, user);
the probe interpolation of idea 2 (it needs the Qwen image at test time); tuning per encoder; how the paper reports the
verdicts (open, D4); the parked ideas.

## 3. Design

Stage i inherits the [12-24 spec](2026-10-25-recipe-and-backbones-design.md) §3 for part 2 (students, matched arms,
bars, statistics, stage 5); changes: the encoder list, the recipe R6-5 for every student, DINOv2 without a trigger,
new regression gates. Stages ii and iii inherit part 1's protocol on the 29,232 (12-24 spec §3.4). IDs follow the rule
(§7 there).

### 3.1 Hypothesis and prior

- **Stage i (claim test).** 12-24 showed that the instructed Qwen graphs and teacher carry the gain to other image
  encoders: SigLIP 2 passed, CLIP B/32 did not. A larger or language-aligned encoder should behave like SigLIP 2;
  DINOv2 has no text alignment, so its student may lack the emotion cues the teacher points at. Guesses for d at the
  sample scale: CLIP B/16 −0.002 to +0.010, CLIP L/14 +0.002 to +0.014, SigLIP 1 +0.008 to +0.018, DINOv2 −0.012 to
  +0.006. A DINOv2 GO, or a SigLIP 1 NO-GO, would surprise us. Rule §1 gives the prior per outcome.
- **Stage ii (confirmation).** The claim checked (12-24, our reading): "the student lifts each encoder above its own
  PercepT arm by a similar amount" (held, sample scale: +0.0243, +0.0202, +0.0190 for SigLIP 2, CLIP B/32 and Qwen).
  Checked here: the sign and size on 29,232 development paintings and seven encoders. Whether the lift follows the
  starting level has no prior direction: CLIP B/32 starts above SigLIP 2 on held frozen emotion (0.0642 against
  0.0569) and lifts less.
- **Stage iii (incremental).** The default image holds the genre information (probe 0.5040), so a loss that copies the
  teacher's genre view more closely should raise genre. The risk is the shared trunk: more genre pull may cost emotion.
  Prior: GO 0.15, promising 0.25, neither 0.60 (rule §1). A genre gain above 0.03 with the guard intact would surprise
  us.

### 3.2 Arms and controls

Stage i: rule §2.1. Each student is R6-5 with the image network's first layer at the encoder's width (12-24 spec
§3.2); its matched arm is PercepT core P3 with the CT's Qwen Stage 1 topics and Stage 2 on the same image, so it drops
only the student. Stage ii: rule §2.2; the own PercepT arm is the matched control of each student.

Stage iii (rule §2.3): each variant changes only the genre head's loss of R6-5.

| Arm | What | Differs from R6-5 by | Role |
|-----|------|----------------------|------|
| R6-5 | 12-24's pick, seed-0 P exports frozen | — | base |
| GT-2, GT-4 | genre teacher weight ×2, ×4 | the genre teacher weight | candidates |
| GR-0.5, GR-2 | plus the relational term toward the genre teacher | one added genre term | candidates |
| GA-1 | plus standardised regression through a discarded two-layer adaptor | one added genre term | candidate |
| two extras | two more rungs of the closest method (rule §4.3) | as their method | candidates |
| T, ridge probe | the instructed Qwen genre image; the ceiling probe's prediction | no training | reported |

All arms share the data, settings, early stopping and seed; no arm is tuned beyond its listed weights.

### 3.3 Data roles

| Data | Role | Stage |
|------|------|-------|
| 13,400 fit and 1,489 stopping paintings (the frozen sample) | training; label-free early stopping | i (3), ii (every student and own arm), iii |
| 29,232 train-side paintings outside the sample (P) | development: stage ii's scores and statistics, stage iii's effects and pick; in stage 4, training (23,080) and stopping (6,152) | ii, iii, i (4) |
| first 1,000 paintings of P | the store check (V5) | i |
| 5,000 old evaluation paintings | development, reported | iii |
| 42,969 train paintings | training | i (4) |
| 12,281 held paintings | held: images for inference, captions for labels only | i (3, 4); ii's held table, read after the P table, choosing nothing |
| ArtELingo-28, 746 paintings, their own images | test | i (5) |

P chose 12-24's recipe; it stays development data. No held or ArtELingo-28 number enters a choice (`held-data.md`).

### 3.4 Measurement

- **Stage i.** The RB rule's part 2 protocol (12-24 spec §3.4): readings r20 and own; mean over k-means states, then
  over seeds 0 to 4; d = candidate minus the maximum over eligible entries; paired bootstrap over held paintings and
  seeds, maximum taken again per draw; Leiden beside.
- **Stage ii.** Rule §3.2: per encoder, the lift = the student's AMI minus the better of its own arm's two readings,
  on P, mean over five seeds; a bootstrap over paintings (shared by both arms) and seeds (drawn per arm). Emotion
  decides; style and genre reported.
- **Stage iii.** Rule §3.3: effect = variant minus R6-5 on P, seed 0, part 1's per-aspect bootstrap (2,000 draws);
  genre AMI against T_g = 0.498916; fresh seeds 1 to 3 with a paired bootstrap over paintings and seeds.
- **Seeds.** Stage i five, as the claim test. Stage ii five (the sample-scale weights stage i trains). Stage iii one
  seed to pick, three fresh seeds to test the pick (`held-data.md`, "Seeds").

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|-------|------|-------|--------------|-------------|
| X0 | regression (C12) | the extractor with three new models | the first 512 CLIP B/32 and SigLIP 2 rows equal the old extractor's bit for bit | stage i stops; ii (three encoders) and iii go on |
| S | engineering | four new stores, 61,402 and 746 paintings | V5 per store (counts, order, a content check) | traced and fixed, else that encoder has no outcome |
| X1 | regression | this round's stage i code, sample scale | K2 and M2 seed 0 outputs bit for bit; 12-24's stage 3 and stage 5 verdict values within 1e-6 | stage i stops; ii (three encoders) and iii go on |
| 3 | claim test, sample scale | K6, K7, K8, K5 and matched arms, seeds 0 to 4 | S1 outcomes | — |
| X2 | regression | the same at full scale | K2 and M2 seed 0 bit for bit; 12-24's stage 4 values within 1e-6 | stage 4 stops; stage 5 at the sample scale |
| 4 | claim test, full scale | candidates not S1-NO | S2 outcomes | — |
| 5 | confirmation (test data) | ArtELingo-28 | A outcomes | — |
| VL1 | regression | stage ii's P scorer and P inference | 12-24's P values within 1e-6; K4 seed 0 on P and M2 seed 0 on held bit for bit | stage ii stops |
| ii | confirmation | seven students and own arms on P | L outcomes | — |
| VG1 | regression | stage iii's trainer, switches off | R6-5 seed 0 P exports bit for bit | no variant runs |
| iii | incremental, seed 0 | five variants, then two extras | III outcomes | — |
| iii-c | confirmation, fresh seeds | winner and R6-5, seeds 1 to 3, when the winner gains one genre seed range | R6-5's weights equal 12-24's K4 of the same seed; C outcomes | no C outcome; trace |

Order (agent default): each gate before its stage; a candidate's stage 4 after its S1 outcome; candidates K6, K7, K8,
K5; deferral for time only, decided before a candidate's stage 3 starts, K5 first, then K8, K7, K6, never dropped;
stage ii's P inference once every candidate not deferred has stage 3 weights; its held table after its P table. Stage iii needs no new
store, so the run chat may start with it; the rest of the queue order is the run chat's.

### 3.6 Diagnostics (decide nothing; each after its stage's outcomes are written)

- **Stage i.** As 12-24 (style and genre margins against the three bars, T, Leiden, seed spreads, best epochs), and
  each new store's V5 values beside CLIP B/32's and SigLIP 2's, and the old stores' difference from the current
  extractor (X0).
- **Stage ii, encoder statistics** per encoder on its default image on P, with 12-24's sheet kernels: norms as stored
  (some stores hold unit rows), anisotropy (mean pairwise cosine, top centred component's variance share), 20-NN label
  purity per aspect with chance, cosine silhouette per aspect, hubness (skew of 20-NN in-degree, maximum, share with
  zero). Spearman correlations over the seven encoders between the emotion lift and each of: frozen AMI, the own
  arm's AMI, each statistic (seven points: descriptive only). Reading: a lift that tracks the starting level says the
  student adds a fixed amount; a lift that tracks purity or hubness names an image-space property to work on, for a
  later loop on development paintings.
- **Stage ii, the held table** (after the P table): per encoder, held frozen AMI, own arm, student and lift at both
  scales, from 12-24 and stage i, and the lift's rank on held against P. It chooses nothing.
- **Stage iii.** Each variant's three aspects with intervals; the old 5,000; Leiden; best epochs; for GA, the AMI of
  the adaptor's own output (does the adaptor absorb the gain, the literature note's risk?); the probe as reference.

### 3.7 Frozen vs rerun

Rule §5. In short: 12-24's students, matched arms, part 1 exports and stores are frozen; the four new encoders' arms,
every P inference and every stage iii run are new.

## 4. Decisions

Times are Amsterdam time on 2026-10-25, from the user's messages in the r10 decide 20 chat (16:17 to 17:29).

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|----|----------|----------|----------------------------|------------|------|
| D1 | Direction (A) | the three stages as one medium loop | the user's three proposals | user, after seeing the recommendation | 16:23 |
| D2 | Encoders | CLIP B/16, CLIP L/14 at 224 px, SigLIP 1 So400m/14 at 384 px, DINOv2-large | the user's "CLIP V/14" read as L/14; DINOv2 added | user, after seeing the recommendation (flag) | 16:23 |
| D3 | Stage i's claim | one verdict per encoder (3a) | a pooled sentence (3b); full scale only (3c) | user | 16:46 |
| D4 | Paper reporting of the verdicts | open: "we might exclude models that cannot pass, too soon to determine". Our note: a paper that leaves out failed encoders must say they were tested | — | user | 16:52 |
| D5 | Where stage ii looks | development first (1b): sample-scale students and own arms of all seven encoders on the 29,232; then the held table, which chooses nothing | held results only (1a) | user | 16:46 |
| D6 | Stage ii's what-if | a method idea goes to development first, to held only as a new claim test in a later loop | — | user (confirmed our check) | 16:52 |
| D7 | Stage iii's target | the instructed genre value on P, 0.4989 (2c) | part 1's guarded rule (2a); genre first (2b) | user | 16:46 |
| D8 | Stage iii's guard | no win if another aspect loses more than one seed range | genre alone | user, after seeing the recommendation (flag) | 16:52 |
| D9 | "Keep weights" | (i) save every variant with genre ≥ 0.4861, and (ii) two more weights for the closest method | (i) alone | user, after seeing the recommendation (flag) | 16:52 |
| D10 | Minor m1, stores | images and texts for CLIP B/16, CLIP L/14, SigLIP 1; images only for DINOv2 | images only | user, after seeing the recommendation | 16:52 |
| D11 | Minor m2 | DINOv2 under R6-5; each new encoder gets the ArtELingo-28 stage | the old R7-only trigger | user, after seeing the recommendation | 16:52 |
| D12 | Minor m3 | the five stage iii variants (§3.2); idea 2's probe interpolation left out | — | user, after seeing the recommendation | 16:52 |
| D13 | Minor m4 | stage ii's statistics (§3.6); the correlation is descriptive | — | user, after seeing the recommendation | 16:52 |
| D14 | Minor m5 | five glossary terms: recipe, teacher share, matched arm, frozen input, seed range (added in decide 21) | — | user, after seeing the recommendation | 16:52 |
| D15 | Minor m6, timing | the build tonight; the run after it if it can end by 11:00, else Monday 2026-10-26 19:00 | — | user, after seeing the recommendation | 16:52 |
| D16 | Plan page; approval; restart | none ("plain"); approval from this brief ("approve", 18:35); a container restart before the build chat ("yes need restart, otherwise GPU might crash", 18:35) | — | user | 16:52, 18:35 |
| D17 | Abstract; retrieval | not now; retrieval the next loop ("the retrieval is important") | retrieval in this loop | user | 16:23 |
| D18 | Loop size | medium; step 3 skipped (stage i reuses a reviewed rule) | full loop | agent default (not contested) | 16:21 |
| D19 | Stage types | i: claim test as 12-24; ii: confirmation of 12-24's lift reading, an outcome per encoder; iii: incremental, then a confirmation on fresh seeds | ii as an exploration (no outcome; weaker) | agent default | 18:00 |
| D20 | IDs | K5 DINOv2 (12-24's ID), K6 CLIP B/16, K7 CLIP L/14, K8 SigLIP 1; M and F of the same number; F4 the frozen Qwen image | — | agent default | 18:00 |
| D21 | Regression gates (C12) | X0 (changed extractor equal to the old one, bit for bit), X1 and X2 (12-24's K2 and M2 outputs bit for bit, its verdict values within 1e-6), VL1 (scorer; K4 and M2 inference), VG1, and R6-5's fresh seeds against K4's | the old CT-based gates (12-24 already passed them); the stored rows as X0's reference (older extractor versions wrote them) | agent default | 18:45 |
| D22 | Store check | V5: counts, order, finite, width; image-to-caption top-1 retrieval ≥ 1% on 1,000 development paintings for the three text encoders, the bar void if CLIP B/32 or SigLIP 2 fails it; DINOv2 cosine ≥ 0.99 against the archived DINOv2-large store; a failed store may be fixed before training | counts only (a preprocessing error would pass) | agent default | 18:45 |
| D23 | Text sides | each checkpoint's documented text use (CLIP 77 tokens; SigLIP 1 padded to 64 tokens); they serve V5 and the next loop's retrieval only | — | agent default | 18:00 |
| D24 | Stage ii's lift | student minus the better of its own arm's readings; per-encoder bootstrap; L-HOLDS, L-QUAL, L-FAILS on emotion | against the frozen image (not a trained comparator) | agent default | 18:00 |
| D25 | Stage iii's terms | relational and regression terms added beside the direct teacher term, genre head only; GA's adaptor two layers, 2048 wide, GELU, w = 1, the probe's loss | replacing the direct term (two changes at once) | agent default | 18:00 |
| D26 | The extras | fixed ladders (rule §2.3); the closest method by guarded genre; next two rungs up when its largest run is best and guarded, else one rung on each side | free choice after seeing results | agent default | 18:00 |
| D27 | Stage iii's pick and fresh seeds | winner = guarded, largest unrounded genre AMI, ties to the listed order; the targets are the unrounded 0.4989162 and 0.4861162; fresh seeds 1 to 3 when the point gain ≥ 0.0128 (the user's words); R6-5 retrained on the same seeds; C-HOLDS needs the 3-seed gain ≥ one range with interval above 0 and the guard; a run with a non-finite loss has no genre value and cannot win | the interval as the trigger (not in the user's words); rounding to 4 decimals (a tie could pick a variant below the target) | agent default | 18:45 |
| D28 | Weights | every run saves; the fresh-seed runs' weights are kept too; the other weights below 0.4861 go on the pending deletions list | saving only close variants (the condition is known after scoring) | agent default | 18:45 |
| D29 | Re-derivations (C13) | separate code for stage i's d (V2), stage ii's lifts (VL2), stage iii's effects, pick and outcomes (VG2) | — | agent default | 18:00 |
| D30 | Leiden | beside in stages i and iii; not in stage ii (descriptive analysis) | Leiden everywhere | agent default; for stage i, user ("a Leiden reading in all major tests", 2026-10-20) | 18:00 |
| D31 | Outputs location | the round's outputs on `/data/SSD2` (about 30 GB, a guess), linked from the round folder | the system disk (`storage.md`: bulk data on the data disks) | agent default | 18:00 |
| D32 | Names | round folder `src/test/20261025_encoders_genre/`, module prefix `eg_`; report sequence name 12-25 | — | agent default | 18:00 |
| D33 | Priors (C8) | rule §1 | — | agent default | 18:00 |
| D34 | Reading both stage iii outcomes | the seed-0 outcome stands as the user defined it; a GO or promising winner whose fresh seeds give C-FAILS is reported "not repeated on fresh seeds"; a winner below 0.4861 stays neither, whatever its fresh seeds | C-QUAL counted as promising (widens the user's level) | agent default | 18:45 |
| D35 | Dig budget (stage iii) | a GO or promising variant goes to the next decide chat, which decides whether a later development loop follows before a held claim test; none is planned here | a fixed extra loop (the user decided only that a held test comes in a later loop) | agent default | 18:45 |
| D36 | The GA adaptor's details | input the head's unit-norm output; targets standardized as the ceiling probe; built from its own random generator, so the model's random stream is unchanged; its output exported for the diagnostic, then discarded | the shared random stream (GA's effect would mix with a seed shift) | agent default | 18:45 |

**Amendments:** none.

## 5. Assumptions and open questions

- Assumption: the three new checkpoints download and load with the installed `transformers`. Checked: not checked;
  DINOv2-large is cached.
- Assumption: 12-24's saved weights (K2 to K4, M2, M3) and the CT's sample-scale P3 weights (B1) load and give their
  saved held outputs. Checked: files listed on 2026-10-25; K4 seed 0 equals 12-24's part 1 R6-5 seed 0 weights
  (`torch.equal`, the rule review); X1 and VL1 check K2, M2 and K4.
- Assumption: the local 3090 repeats training bit for bit. Checked: 12-24's stages 0, 2 and 3a; again in X1, X2, VG1.
- Assumption: extraction takes about 1.5 GPU hours for the four encoders. Checked: not checked (CLIP B/32 took 310 s,
  SigLIP 2 822 s on 61,402 images).
- Open questions: none.

## 6. Success criteria

**Decision mode.** Binding rule: `src/test/20261025_encoders_genre/DECISION_RULE.md` at the commit in the header,
committed before any code; it governs where it differs from this spec. Its criteria:
- Stage i, per candidate K5 to K8: S1-GO, S1-MM, S1-IN, S1-NO; S2-GO to S2-NO; A-HOLDS, A-QUAL, A-FAILS; gates X0, X1,
  X2; validity V1 to V5.
- Stage ii, per encoder: L-HOLDS, L-QUAL, L-FAILS; validity VL1 to VL4.
- Stage iii: III-GO, III-PROM, III-NONE; C-HOLDS, C-QUAL, C-FAILS; validity VG1 to VG4.

**Selection rule** (stage iii, rule §4.3): unrounded genre AMI on P at seed 0 among guarded variants, ties to the
order GT-2, GT-4, GR-0.5, GR-2, GA-1, then the extras.

**Promising and dig budget** (stage iii): promising is III-PROM (guarded, genre at least 0.4861, below 0.4989; user);
a fresh-seed C-FAILS is reported beside it as "not repeated" (D34). Dig budget: none fixed here; the next decide chat
decides whether a later development loop follows before a held claim test (D35).

**Licensed claims.**
- Stage i, S2-GO for encoder X: 12-24's sentence (its spec §6) with X; S1-GO with "trained on 13,400 paintings".
  Per encoder only: no sentence across encoders.
- Stage ii, L-HOLDS for X: "On 29,232 development ArtELingo paintings, the student on X (13,400 training paintings,
  5 seeds) gives emotion AMI <lift> [interval] above X's own PercepT arm." A development reading; no held claim.
- Stage iii, III-GO with C-HOLDS: "On 29,232 development ArtELingo paintings (Qwen; picked on one seed, tested on three
  fresh seeds), <variant> gives genre AMI <x> against the recipe's <y> on the fresh seeds (instructed Qwen image:
  0.4989), with emotion and style within one seed range of the recipe." <x> and <y> are the three-seed means, not the
  seed-0 values. At most "promising on development data".

## 7. Constitution check

| Principle | Pass or exception | Note |
|-----------|-------------------|------|
| C1 strongest comparator | pass | stage i: the maximum over B1 to B6 and the matched arm; ii: each encoder's own PercepT arm; iii: R6-5 and the instructed Qwen image |
| C6 sample IDs | pass | P's and the held list's hashes; joins by painting name and extraction IDs; new stores in the existing order (V5) |
| C7 rule before code | pass | reviewed by a fresh Opus `xhigh` worker; committed before any code |
| C8 prior and disclosure | pass | rule §1; §8 below |
| C12 regression first | pass | X0, X1, X2, VL1, VG1, R6-5's fresh seeds against K4's |
| C13 re-derivation | pass | D29 |
| `held-data.md` | pass | stages ii and iii choose on development data; the held table chooses nothing |

Exceptions: none.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|---|---|---|
| A new store with wrong preprocessing | V5's content checks; frozen AMI far from its family | traced and fixed before training, else that encoder has no outcome |
| Seven points make any correlation noisy | — | descriptive only; no claim |
| The genre change costs emotion through the shared trunk | the guard; each variant's emotion effect | the guard blocks the win; reported |
| The adaptor absorbs the gain (GA) | the adaptor output's AMI against the head's | reported; the head below it is what counts |
| One seed picks the winner among seven | the fresh seeds | C outcomes decide what is promising |
| The run ends after 11:00 | the end estimate at start | Monday 19:00 start (D15) |

Selection disclosure. P chose 12-24's recipe among six on seed 0, and stage iii adds seven variants on the same seed
and paintings, so the winner's seed-0 gain is inflated by the pick; the fresh seeds protect. Stage ii: K4 at seed 0
is the model that won 12-24's pick on P, so the Qwen lift on P, and the correlations, are inflated by that pick. The variant list and
weights were set before any stage iii number (D12, D26). Stage i: the held paintings were read by the claim test and
12-24; no choice was made on them and none is made here. The encoders were chosen without new held numbers.

## 9. Deliverables and review

- At the end: round folder `src/test/20261025_encoders_genre/` with the rule, verdict, pick and re-derivation files
  (outputs on `/data/SSD2`, gitignored); the new stores under `/data/SSD2/pre_extract/`; one auto report (12-25) in
  `docs/reports/auto/percept/` with a results brief, and a user-read report; a row per held read in
  `docs/superpowers/held_ledger.md`.
- Reviews: the rule by a fresh Opus `xhigh` worker before its commit, and a cold Sonnet `high` check of spec and rule
  against the grill record; the build per `matt-chain.md`, with ticket reviews and unit tests for verdict code (stage
  i's scorer reuse, stage ii's lift and bootstrap, stage iii's pick, extras and fresh-seed statistics, the
  re-derivations); the final whole-branch review; Claude's claim review of the auto report.
- Stop points: none, except a failed gate.
- Budget (a guess): GPU about 3 to 3.5 h on the local 3090 under the lock: stores about 1.5 h (with downloads),
  students and matched arms at both scales about 1 h, stage ii inference about 10 min, stage iii about 25 min,
  regressions about 15 min. CPU 5 to 6 h at 8 threads (12-24's report queues took about 1 h per stage for three
  candidates). Time-box: reports by 2026-10-27 10:00.
- Length: the body is about 270 lines against about 210 for a claim test plus two stages, because of the 36 carried
  and new decisions and the twelve gated stages.
