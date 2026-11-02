# c* on MMArt's unseen paintings, four encoders: design

> Date: 2026-11-02 18:30, revised after the rule review · Status: approved 2026-11-02 18:37 (from the brief, r15 decide; no
> container restart); amended 2026-11-02 21:00 (D19: the test cut at 0.95, user) · Kind: experiment
> Builds on: [report 2027-01-02](../../reports/auto/percept/2027-01-02_backbones_sweep.md) (c* holds on three more
> encoders; the photo model beats its teacher on EmoSet); decisions C and A of 2026-11-02 and the grill in the r15
> decide chat (2026-11-02, one round and a closing summary, about 18:15 to 18:20; its answers are §4) · Constitution:
> [constitution.md](../constitution.md) · Decision rule: inline in §6, with §3.4 and §3.5
> Inherits: the r14 spec ([2026-11-02-backbones-sweep-design.md](2026-11-02-backbones-sweep-design.md)), part a: its
> arms (§3.2), clustering, SC3 statistic, retrieval, outcomes and C13 (§3.4), stage 0 replays (§3.5) and D16, D18, D19;
> the MMArt preparation ([round log](../../../src/test/20261102_mmart_prep/20261102_mmart_prep_log.md)) · Facts: the
> [facts brief](../../../src/test/20261102_mmart_test/mt_facts_brief.md) (Codex, 2026-11-02 18:11)
> Reviewer: a fresh Opus `xhigh` worker, 2026-11-02 18:35 (pass after fixes: four must-fix, eight should-fix and
> nine nit items, all applied; [review](../../../src/test/20261102_mmart_test/mt_rule_review.md))

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** c* beats its plain image and PercepT on ArtELingo's held paintings, on four encoders, but every
painting result so far uses ArtELingo. This loop tests the frozen c* of each encoder on MMArt: WikiArt paintings that
are not in ArtELingo, with MMArt's own emotion labels. It is the first painting set that shaped nothing, so it can
also license "better than the teacher" on paintings.

**Plan** (build and run unattended; you read the report).
1. Stage 0 (checks): tonight's MMArt stores pass their checks; the new scoring code, fed ArtELingo held, reproduces
   r14's and earlier painting numbers exactly; PercepT's saved image mappers reproduce their stored outputs.
2. Stage 1 (sweep, development only): on the 5,769 validation paintings with MMArt labels, measure each c*'s seed ranges and
   pick each model's retrieval read-out (one per model, r11's rule). Committed before any MMArt test number.
3. Stage 2 (claim test): the MMArt test paintings, minus copies of ArtELingo train and validation paintings, scored
   for all four encoders; the test selects nothing.
4. Stage 3 (diagnostics, after the verdicts): the shared ArtELingo held paintings with MMArt labels, unseen artists,
   class-count clustering, and others.

**Compared against.** Per encoder: its plain image (WIN in six cells) and PercepT on that image (WIN in the three
clustering cells) carry the verdict. The Qwen image instructed per aspect (the teacher) is compared cell by cell.

**Outcomes.** Per encoder: holds with all nine WIN cells; qualified when only PercepT cells miss; fails otherwise,
and a failed verdict stands. Against the teacher, in each cell: in the cells where that encoder's c* won on
ArtELingo (Qwen: emotion and style clustering, genre retrieval; the other three: emotion and style clustering, style
and genre retrieval), a WIN licenses "better than the teacher"; in every cell, LEVEL or WIN licenses "level with or
better than"; BELOW licenses nothing. We expect holds on most encoders (0.65 each), and several "better than" cells, fewest in
emotion (MMArt's emotion labels agree with ours on only 43% of shared paintings).

**Cost.** CPU about 3½ to 4½ h (exports, checks, picks, scoring four encoders, diagnostics, re-derivation); no GPU
expected (the validation teacher views and all 20 PercepT mappers exist; checked). Build tonight (about 3 to 4 h), run when it ends;
report on Tuesday 11-03, at the latest 11-04 12:00.

**Check these.**
1. Your picks after seeing our recommendation: the WIN bar on the teacher cells won on ArtELingo (D3); the minor
   batch (D5), with PercepT frozen with its ArtELingo topics. The test cut moved from 0.85 to 0.95 after the
   near-copy counts (D19, user after seeing the recommendation): 16,384 test paintings.
2. The WIN-bar cells of the three other encoders follow the same rule from r14's results (agent default, D7).
3. The main stage is a claim test, not a confirmation, since it feeds paper claims (agent default, D6); the loop
   stays medium.
4. The validation set is ArtELingo's validation paintings (used for c*'s stopping and r10 to r14's picks) with MMArt
   labels; the picks are new, the images are not (D4).
5. No artist split: MMArt test paintings may share artists with ArtELingo train. A diagnostic reports the
   unseen-artist subset (D14).

## 1. Context and goal

Asked: "That's good result indeed, so we go with 2." (option 2 of report 2027-01-02: the PolyArt (MMArt) test), and
in the grill: "b. a more complete test" (all four encoders), "ok we switch to d" (the teacher bar), "I would go with
h" (picks on MMArt validation).

r14 showed that c*'s recipe holds on SigLIP 2, SigLIP 1 and CLIP L/14 on ArtELingo held, and that the photo model
beats its teacher on EmoSet, a photo set that shaped nothing. No painting set outside ArtELingo has been tested.
MMArt (PolyArt) indexes 74,234 WikiArt paintings; 16,782 are not in ArtELingo, and their seven views (four encoders'
default images, three Qwen teacher views) are extracted on 2026-11-02 from 18:01.

Goal: test whether frozen c* keeps its gains on unseen paintings with independent labels.
Question this step answers: on MMArt test, does each encoder's c* beat its plain image and PercepT, and where does it
beat or match the teacher?

## 2. Scope

In: frozen c* of four encoders (Qwen, SigLIP 2, SigLIP 1, CLIP L/14; seeds 0 to 4); MMArt test and validation; three
aspects (MMArt emotion, MMArt style, WikiArt genre); clustering and retrieval; the diagnostics of §3.6.
Out: any training of c* or the photo models; photo models on MMArt; MMArt captions and perspectives (perspective
retrieval and perspective instructions stay in "Parked ideas"); a topic refit of PercepT on MMArt; an artist-disjoint
split (diagnostic only); MMArt train paintings (they are ArtELingo train).

## 3. Design

### 3.1 Hypothesis and prior

Claim tested: c*'s gains over its plain image and over PercepT carry to WikiArt paintings outside ArtELingo, scored
with labels it never saw; where c* beat its teacher on ArtELingo, it beats it on MMArt too. Mechanism: c*'s heads
learn aspect maps from instructed Qwen graphs, not from labels, so they should carry to new paintings of the same
kind (WikiArt). Facts: on ArtELingo held, Qwen c* against the teacher: AMI emotion +0.0145 [+0.0113, +0.0177], style
+0.0041 [+0.0004, +0.0081], genre −0.0060 (LEVEL); mAP emotion +0.0007 (LEVEL), style +0.0017 (LEVEL), genre +0.0027
[+0.0013, +0.0042] (report 12-28; r11 stage 5). MMArt's emotion label equals our ArtELingo painting label on 43.4% of
11,398 shared held paintings (prep log); its style labels match ArtELingo's on 96.9% (merged to 20 classes); genre is
the same WikiArt set.
Prior, per encoder: SC1 and SC2 WIN in all six cells (0.85; ArtELingo gains over the plain image are many intervals
wide); SC3 WIN in all three (0.75); holds 0.65, qualified 0.20, fails 0.15. Teacher, Qwen: emotion AMI WIN 0.45 (new
labels), style AMI WIN 0.35 (narrow on ArtELingo), genre mAP WIN 0.45; LEVEL or WIN in the other cells 0.7. The
other encoders: as Qwen, and style mAP WIN 0.45. Surprise: a miss against the plain image in style or genre.

### 3.2 Arms and controls

Per encoder X (Qwen, SigLIP 2, SigLIP 1, CLIP L/14), on MMArt features only at test:

| Arm | What it is | Role |
|---|---|---|
| X-c* | frozen c* on X's default image, seeds 0 to 4: Qwen = r10's c* (k11=40), the others r14's X-c* | candidate (SC1 to SC4) |
| X frozen image | X's default image, no training | bar (SC1, SC2) |
| X matched arm | PercepT on X's image, applied frozen: the saved Stage 2 mappers with their ArtELingo Qwen topics (Qwen: the claim test's P3; SigLIP 2: M2; SigLIP 1: M8; CLIP L/14: M7), seeds 0 to 4; on MMArt the own reading is the argmax of its 40 topic probabilities and the 20-cluster reading k-means (states 0 to 2) on its L2-normalized probabilities, as r12's PercepT row; the larger reading as r14 D16, taken again in each draw | bar (SC3) |
| teacher | the Qwen image instructed per aspect (emotion_focus, style_word, genre_word), the ArtELingo instructions | own claims (SC4) |
| strongest PercepT entry | per aspect, the largest of the four matched arms' larger readings on MMArt test (C1) | descriptive |

### 3.3 Data roles

| Data | Role | Stages |
|---|---|---|
| MMArt validation: the 5,769 ArtELingo validation paintings with MMArt labels (emotion 5,720, style 5,769, genre 4,679 labelled) | development: seed ranges, read-out picks | 0, 1 |
| MMArt test: the 16,782 paintings not in ArtELingo, minus the cut below | held: scored only | 2, 3 |
| ArtELingo held, 12,281 paintings | regression replays (stage 0, ArtELingo labels); diagnostic with MMArt labels on its 11,528 MMArt paintings (stage 3) | 0, 3 |
| ArtELingo train and validation paintings (49,121) | near-copy reference for the test cut (the 5,769 validation paintings are also MMArt validation, above) | 0 |
| MMArt train (ArtELingo train paintings) | not touched | none |

Test cut (D5), in this order: drop the 34 exact byte copies of ArtELingo train or validation images; then every
test painting whose default Qwen image has cosine ≥ 0.95 (D19) with any of the 49,121 ArtELingo train and validation
paintings (the prep queue's `mmart_nearcopy.parquet`; float32, L2-normalized, joined by key); then, of the remaining
test paintings sharing an md5, keep the first in the order below (D10; 7 groups, 8 paintings dropped). The cut set is
the evaluation set of every arm: 16,384 paintings (computed 2026-11-02 21:00 from the index and the near-copy file;
the build recomputes it and gate (i) checks it). Labels: MMArt's 9 emotion classes as given ("something else" included), its 20
style classes, the 10 WikiArt genre classes.

Populations (r10, r11 and r14, unchanged; D16): clustering, for every arm and for the validation ranges, fits
k-means on every painting of the set (test: the cut set; validation: all 5,769); AMI and its bootstrap rows run over
all rows, and rows without the aspect's label are masked after indexing. Retrieval queries and gallery are the
aspect's labelled paintings (test, before the cut: emotion 190 and genre 3,043 without a label). Centering and
whitening maps are fitted on all 5,769 validation paintings; query expansion reads the labelled test gallery.

Keys and order (C6, D9): the MMArt `key` (its image id with its style folder), joined to every store through the
prep code's loader (validation stores by painting name). Evaluation order (bootstrap rows, query and gallery order,
cross-fit halves by parity of position) and D10's order: keys sorted ascending by
`hashlib.sha256(key.encode('utf-8')).hexdigest()`, on validation and test.

### 3.4 Measurement

- **Inherited, unchanged:** r14 part a's ArtELingo protocol (r14 §3.4), applied to MMArt: clustering per aspect
  head, k-means with 20 clusters, three states, mean AMI, frozen arms clustered the same way, paired bootstrap of
  2,000 joint draws over paintings and seeds; SC3's statistic and interval (`b_matched`, r14 D16); retrieval with
  the aspect head as query and gallery, relevance by the aspect label, mAP, 10,000 joint draws over queries and
  seeds; outcomes WIN, LEVEL, BELOW as r14 §3.4 with no multiplicity correction.
- **Seed ranges (stage 1).** r_X,a,clu and r_X,a,mAP: maximum minus minimum over X-c*'s seeds 0 to 4 on MMArt
  validation, per aspect a, of AMI (20 clusters) and of mAP at R0, with MMArt labels. Unrounded. They are the LEVEL
  unit of X's SC4 cells and the unit of X's picks.
- **Read-out picks (stage 1).** r11's read-out rule, menu, guard and ties on MMArt validation, with cross-fit
  halves: one pick per arm for its three aspects, scored as (1/3) Σ_a (mAP_a − mAP_a(R0)) / r_X,a,mAP; guard: no
  aspect more than r_X,a,mAP below its R0 value (D16). X-c* and X frozen image with X's ranges; the teacher (one arm,
  its three aspect views) once, with Qwen's ranges, used for all four encoders (D12). Each picked map is fitted on all validation features and
  applied to test (query expansion reads unlabeled test embeddings at test time with k fixed on validation, as r11).
- **C13.** Codex re-derives, from this rule and the saved exports and labels (no retraining), the test cut, the seed
  ranges, the read-out picks, SC1 to SC4 with intervals and outcomes, and the verdicts, within 1e-6, before any
  verdict is written.

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|---|---|---|---|---|
| 0 | regression check | MMArt stores; new export, matched-arm, cut and scoring paths | (i) to (vi) below | stop the affected path (no verdict for the encoders it serves, §6), trace, note it |
| 1 | sweep (development only) | MMArt validation | ranges and picks of every arm of every encoder written and committed before any MMArt test number | that encoder's verdict inconclusive; the others go on |
| 2 | claim test | MMArt test, all frozen | SC1 to SC4 | report |
| 3 | descriptive | all | §3.6, after the verdicts | report |

- (i) Stores: the prep queue's parity (each view, cosine ≥ 0.999 against the ArtELingo stores on 200 shared
  paintings), store check (one row per key, finite, unit norm within 1e-3, keys join) and the near-copy file (one row per test key, finite
  cosines, a reference of 49,121 paintings) pass.
- (ii) Export: c*'s export on each encoder, given ArtELingo held features, reproduces the stored r10 and r14 held
  exports within 1e-3, all five seeds.
- (iii) Scoring replay within 1e-9: the new scorer, fed ArtELingo validation and held features, exports and labels in
  r14's (r10's, r11's) order and with their recorded picks, gives r14's SC1 to SC3 and SC6 painting values and
  intervals for SigLIP 2, and report 12-28's c* and teacher held AMI and r11's stage 5 c* and teacher mAP, values
  and intervals, for Qwen.
- (iv) Matched arms (D11, D18): (a) each saved Stage 2 mapper (20 runs, trained on CUDA), applied on CPU to
  ArtELingo held features, gives its stored held probabilities within 1e-5 (max absolute); (b) the new reading code,
  fed the stored held probabilities, gives the stored partitions exactly and the `reported_stage4` readings within
  1e-9; (c) readings from the CPU probabilities are recorded beside them, no gate. A mapper that fails (a) is
  retrained from its saved Stage 1 topics with its recorded settings (local GPU, under the lock); its held readings
  (own and 20-cluster, each aspect) must lie within the maximum minus minimum over seeds 0 to 4 of the stored arm's
  held reading of that aspect, and the report flags it.
- (v) Validation teacher views: read from the ArtELingo `<view>_s2img` stores (r11's validation teacher views) by
  painting name (D8); gate: 5,769 keys found in each of the three views, finite, unit norm within 1e-3.
- (vi) Read-out rule: fed r11's stored validation outputs, it reproduces r11's picks and scores within 1e-9.

Which failed gate voids what (D17); an encoder whose verdict is inconclusive is never "qualified" from a missing SC3:

| Failed gate | Voids |
|---|---|
| (i) Qwen default store or the near-copy file | every encoder (the cut set is undefined) |
| (i) X's default store; (ii) X's export | X's verdict and SC4 cells |
| (i) a teacher store; (v) | that aspect's SC4 cells, all encoders; verdicts stand |
| (iii) | every encoder |
| (iv) X's mapper after the fallback | X's verdict; X's SC1, SC2 and SC4 rows reported, SC4 claims stand |
| (vi) | SC2 and the SC4 mAP cells, all encoders, so every verdict; SC4 AMI claims stand |
| stage 1: X's ranges or picks missing | X's verdict and SC4 cells |
| stage 1: the teacher's pick missing | every SC4 mAP cell |

### 3.6 Diagnostics

After the verdicts, deciding nothing: R0 rows for every arm; clustering with K = class count (9, 20, 10); per-class
AP; the 11,528 shared ArtELingo held paintings scored with MMArt labels (SC1, SC2 and SC4 cells, same protocol, picks
of stage 1), beside their ArtELingo-label numbers, which separates new labels from new images; the unseen-artist
subset (artist = the part of the WikiArt file name before the first underscore; test paintings whose artist has no
ArtELingo train painting) with its count and SC1, SC2 and SC4 cells (D14); the far subset (D19: the cut at 0.85, 10,403
paintings, SC1, SC2 and SC4 cells; it replaces the uncut test set); the near-copy count
with its 0.80 to 1.0 histogram and the removed paintings per class; test pairs with default-Qwen cosine ≥ 0.95 inside
the cut set; each encoder's share of the teacher's gain over the plain image; seed spreads on test
beside the validation ranges; the strongest PercepT entry (C1).

### 3.7 Frozen vs rerun

Frozen: every c* checkpoint and its recipe; the matched arms' Stage 1 topics and Stage 2 mappers; the ArtELingo
stores; stage 1's ranges and picks before test scoring. Rerun: MMArt exports of every arm (validation and test), any
mapper that fails its check (stage 0), all scoring.

## 4. Decisions

Grill rows: the r15 decide chat, 2026-11-02 (decision A about 18:00; round 1 about 18:15; closing about 18:20).

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|----|----------|----------|----------------------------|------------|------|
| D1 | Direction | the MMArt (PolyArt) test of c*; medium loop (type: D6, stricter than the user's confirmation, flagged) | test c* on a second art source (freeze and write; a fourth encoder) | user (direction, before seeing ours); user, after seeing the recommendation (loop size) | 2026-11-02 18:00 |
| D2 | Scope | all four encoders | "a more complete test" (Qwen c* only) | user (Q1 b) | 2026-11-02 18:15 |
| D3 | Teacher bar | per cell: WIN bar ("better than") in the cells c* won on ArtELingo, LEVEL or WIN ("level with or better than") in all cells | "good reason": no extra cost, licenses "better than" on paintings (LEVEL or WIN only, as r14; WIN in all six) | user, after seeing the recommendation (Q2 d) | 2026-11-02 18:20 |
| D4 | Where picks and ranges come from | MMArt validation with MMArt labels, as Affection (r12) and EmoSet (r14) | one rule for every new set (all frozen from ArtELingo; ranges only) | user (Q3 h) | 2026-11-02 18:15 |
| D5 | Minor batch | test cut (exact copies, near-copies at 0.85 to ArtELingo train and validation; the threshold changed to 0.95 by D19), unlabelled per aspect dropped; r14's verdict structure, a failed verdict stands; K = 20, seeds 0 to 4, no training, class-count K as a diagnostic; matched arms frozen with ArtELingo topics; the shared held diagnostic; local CPU, build tonight, run and report 11-03 | the grill's minor batch i to vi | user, after seeing the recommendation | 2026-11-02 18:15 |
| D6 | Stage types | stage 2 a claim test; stage 1 a sweep on development only; stage 3 descriptive | the result feeds paper claims (confirmation) | agent default | 2026-11-02 18:30 |
| D7 | WIN-bar cells per encoder | the cells WIN against the teacher on ArtELingo held: Qwen emotion AMI, style AMI, genre mAP (report 12-28, r11 stage 5); SigLIP 2, SigLIP 1, CLIP L/14 emotion AMI, style AMI, style mAP, genre mAP (r14 SC6) | D3's rule applied to each encoder's own record (the Qwen cells for all) | agent default | 2026-11-02 18:30 |
| D8 | Validation teacher views | read the stored ArtELingo `<view>_s2img` views (r11's) of the same images | byte-identical images (57 filed in another style folder, md5 checked); all 5,769 found in each view (rule review) (a fresh extraction of 17,307 images, about 25 min GPU) | agent default | 2026-11-02 18:30 |
| D9 | Evaluation order | keys sorted by sha256 of the key string | MMArt keys begin with the style folder (the style label) and then the artist, so sorted keys group labels (r14 D17's reason) | agent default | 2026-11-02 18:30 |
| D10 | Duplicates inside MMArt | keep the first key in sha256 order of each md5 group left in test after the cut | 7 groups (15 rows), 8 paintings dropped; a duplicate pair inflates retrieval for every arm (keep all) | agent default | 2026-11-02 18:30 |
| D11 | Matched-arm mappers | reuse the saved mappers (all 20 present) after the replay; retrain a failing one from its saved topics, flagged | the bar needs MMArt outputs, and only the Stage 2 mapper touches the image (refit topics on MMArt captions) | agent default | 2026-11-02 18:30 |
| D12 | Pick units | X's ranges for X-c* and X's frozen image; Qwen's ranges for the one teacher pick shared by all encoders | one teacher pick per aspect keeps one teacher across encoders (a pick per encoder) | agent default | 2026-11-02 18:30 |
| D13 | Folders | round folder `src/test/20261102_mmart_test/`, prefix `mt_`; outputs `/data/SSD2/cosir_r15/20261102_mmart_test/outputs/` | the project's layout | agent default | 2026-11-02 18:30 |
| D14 | Artist overlap | no artist split; an unseen-artist diagnostic | the user fixed ArtELingo's split (an artist-disjoint test set) | agent default | 2026-11-02 18:30 |
| D15 | Extraction time | the seven-view extraction from 18:01 on the local GPU, run by r15 decide; the "r14 mmart run" chat closed | "the extraction can start now" (after 19:00) | user | 2026-11-02 18:00 |
| D16 | Populations and read-out rule | clustering on all rows, unlabelled masked at AMI; retrieval on labelled rows; one read-out pick per arm by r11's rule | inherited unchanged from r10, r11 and r14, so the replays hold (drop rows before k-means; a pick per aspect) | agent default (rule review) | 2026-11-02 18:35 |
| D17 | Gate scope | the table under §3.5 | a failure voids only the cells that use the failed part (an encoder's whole verdict for any gate) | agent default (rule review) | 2026-11-02 18:35 |
| D18 | Mapper replay tolerance | CPU probabilities within 1e-5; readings within 1e-9 only from stored probabilities | the mappers ran on CUDA; a 1e-6 change can flip an argmax (readings within 1e-9 from CPU probabilities) | agent default (rule review) | 2026-11-02 18:35 |

| D19 | Test cut threshold | cosine ≥ 0.95 (16,384 test paintings); the 0.85 cut becomes the far-subset diagnostic in place of the uncut set | the near-copy check (19:55): at 0.85 the cut removed 6,375 of 16,782 (38%), mostly different paintings by the same artist (45% to 96% same artist per band); true copies sit at ≥ 0.99; 0.95 removes 2.4%, as EmoSet's 0.85 removed 2%, and keeps same-artist paintings as ArtELingo's own held split does | user, after seeing the recommendation | 2026-11-02 21:00 |

**Amendments**: D19 (2026-11-02 21:00, user) changes D5's cut from 0.85 to 0.95 after the near-copy counts; no test
score had been read.

## 5. Assumptions and open questions

- Assumption: the ArtELingo `<view>_s2img` stores hold the 5,769 validation paintings in each teacher view. Checked
  by key in the rule review, 2026-11-02; stage 0(v) checks it again.
- Assumption: the 20 Stage 2 mappers are saved (`.pt` beside each output). Checked in the rule review: all 20
  present.
- Assumption: the whole CPU work takes about 3½ to 4½ h (scoring about 15 min per encoder, r14's ArtELingo times
  scaled by 16,782 / 12,281; the rest from the rule review). Not checked.
- Assumption: MMArt's labels per painting are single-valued (one dominant emotion, one style, one genre). Checked in
  the index (one value per row; facts brief).
- Open questions: none.

## 6. Success criteria

**Decision mode.** Binding rule: this section with §3.4 and §3.5, committed with this spec before any code.

| ID | What is measured | Compared against | Bar | On which data |
|----|------------------|------------------|-----|---------------|
| SC1 | X-c*'s AMI, emotion, style, genre | X frozen image | WIN in each | MMArt test (cut) |
| SC2 | X-c*'s mAP, picked read-out, three aspects | X frozen image, its picked read-out | WIN in each | MMArt test (cut) |
| SC3 | X-c*'s AMI, three aspects | X matched arm, larger reading | WIN in each | MMArt test (cut) |
| SC4 | X-c*'s AMI and mAP (picked read-out), three aspects | the teacher, its picked read-out | each cell its own claim: WIN in D7's cells licenses "better than"; LEVEL or WIN in every cell licenses "level with or better than" | MMArt test (cut) |

- **Per encoder.** Holds when all nine cells of SC1 to SC3 WIN; qualified when SC1 and SC2 WIN in all six cells and
  one or more SC3 cells do not; fails otherwise. Holds GO, qualified GO for the qualified claim, fails NO-GO; the
  verdict stands after a failure and goes in the paper as it is.
- **SC4 cells.** A D7 cell at WIN gives "better than"; a D7 cell at LEVEL gives "level with or better than"; a
  non-D7 cell at LEVEL or WIN gives "level with or better than"; BELOW gives no claim and keeps its row.
- **Inconclusive (D17).** A failed gate voids what §3.5's table says; the rows that exist are reported; the rest go
  on. An inconclusive verdict is never "qualified".
- **Selection rules.** Stage 1 only: ranges, then read-out picks (§3.4, D12). Stage 2 selects nothing.
- **Licensed claim (holds):** "On <n> WikiArt paintings outside ArtELingo (MMArt; exact and near copies of ArtELingo
  train and validation paintings removed), c*'s recipe with the <X> image as its only input at test (frozen; never
  trained or picked on these paintings) clusters and retrieves paintings by emotion (MMArt's labels), style and genre
  better than the plain <X> image, and clusters them better than PercepT on the <X> image (AMI with 20 clusters; mAP
  with read-outs picked on validation paintings with MMArt labels; 5 seeds). <X> is one of four encoders; the three
  besides Qwen passed an earlier held emotion test on ArtELingo." Qualified: "… against PercepT on the <X> image,
  better in <aspects>." SC4: "<c* on X> is better than (or: level with or better than) the Qwen image instructed for
  <aspect> in <task> on MMArt."
- **Not licensed:** artists unseen in training (no artist split; §3.6 is a diagnostic); other art sources or photos;
  "better than the teacher" in a non-D7 cell; MMArt's perspectives.

## 7. Constitution check

| Principle | Pass or exception | Note |
|-----------|-------------------|------|
| C1 strongest comparator | pass | the frozen image and the matched arm as bars, the teacher per cell, the strongest PercepT entry beside |
| C6 sample IDs | pass | extraction keys through the prep loader; stage 0(i) |
| C7 rule before code | pass | §6 with §3.4 and §3.5, reviewed by a fresh Opus `xhigh` worker before the approving commit |
| C8 prior and disclosure | pass | prior §3.1; disclosure §8 |
| C12 regression check | pass | stage 0(ii) to (vi) |
| C13 re-derivation | pass | Codex, §3.4 |
| `held-data.md` | pass | ranges and picks on validation only; MMArt test scored, selecting nothing; the matched arm's larger reading is taken on test, which only makes the bar stricter; query expansion reads unlabelled test embeddings with k fixed on validation, as r11; the shared held paintings are read for replays and a diagnostic only |

Exceptions: none.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|---|---|---|
| Test paintings near-copy ArtELingo train or validation paintings | the near-copy count and histogram | the cut (0.95, D19); counts reported; the far subset (0.85) as a diagnostic |
| Shared artists make the test easier than "unseen paintings" suggests | the unseen-artist diagnostic | the claim says "paintings outside ArtELingo", not "unseen artists" |
| MMArt's emotion labels measure something else than ours (43% agreement) | per-class AP; the shared held diagnostic | emotion cells reported with this note; no label mapping |
| A mapper is missing or does not replay | stage 0(iv) | retrain from saved topics, flagged; else that encoder's verdict inconclusive (§3.5 table) |
| Ranges from 5 seeds on validation are very small, so LEVEL is strict | ranges beside test seed spreads (§3.6) | reported; ranges are not changed after results |
| Selection disclosure: the validation paintings carried c*'s stopping, r10's 59 runs and r11 and r14's picks; the three non-Qwen encoders passed a held emotion test on ArtELingo; stage 1 adds up to 25 read-outs per arm | validation numbers inflated | only test numbers are results; every pick frozen before test scoring; the claim names how the encoders were chosen |

## 9. Deliverables and review

- At the end: the round folder `src/test/20261102_mmart_test/` (prefix `mt_`) with code, tests, log and per-item
  outputs; outputs under `/data/SSD2/cosir_r15/20261102_mmart_test/outputs/`; the auto report
  `docs/reports/auto/percept/2027-01-03_mmart_test.md` with its results brief and index row; a user-read report.
- Reviews: this rule by a fresh Opus `xhigh` worker before the approving commit; tickets per `matt-chain.md`, with
  the test cut, the export and matched-arm paths, the picks, the scorer and the outcome code as verdict code; the C13
  re-derivation by Codex; the final whole-branch review.
- Stop points: none between approval and the reports. Time-box: the report by 2026-11-04 12:00.
- Budget: CPU about 3½ to 4½ h at 8 threads (exports of four encoders on 22,551 paintings, stage 0 replays, stage 1
  picks, scoring about 1 h, diagnostics, C13); local GPU none expected (a mapper retrain, if needed, takes minutes
  under the lock). Disk: about
  0.4 GB of MMArt stores (SSD2, shared cache) and about 2 GB of exports and outputs.

Body length: about 230 non-blank lines against a claim-test target of about 150 plus the rule: four encoders and a
cell-wise teacher claim need the per-encoder D7 cells and the mapper and teacher-view fallbacks.
