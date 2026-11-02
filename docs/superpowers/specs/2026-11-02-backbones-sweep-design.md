# c* on three more encoders, the photo model on EmoSet, and the large analysis sweep: design

> Date: 2026-11-02 03:40, revised after the rule review · Status: approved 2026-11-02 04:13 (from the brief, r14
> decide 3; no container restart) · Kind: experiment
> Builds on: [report 2027-01-01](../../reports/auto/percept/2027-01-01_photo_finetune.md) (the photo model above its
> teacher on Affection; licensed "level with or better than"); decision C and A of 2026-11-02 and the grill in the r14
> decide 2 chat (2026-11-02, two rounds, ending about 03:35; its answers are §4) · Constitution:
> [constitution.md](../constitution.md) · Decision rule: inline in §6, with §3.4 and §3.5
> Inherits: c*'s recipe and ArtELingo clustering protocol ([r10 spec](2026-10-27-recipe-sweep-design.md),
> [report 12-28](../../reports/auto/percept/2026-12-28_recipe_sweep.md)); the ArtELingo retrieval protocol and read-out
> rule ([r11 spec](2026-10-29-retrieval-design.md)); the Affection protocol, wording rule and read-out rule
> ([r12 spec](2026-10-30-affection-confirmation-design.md)); the photo fine-tune and its picked settings
> ([r13 spec](2026-10-31-photo-finetune-design.md)); the student on another encoder
> ([10-25 spec](2026-10-25-recipe-and-backbones-design.md), §3) · Facts:
> [facts brief](../../../src/test/20261102_backbones_sweep/bs_facts_brief.md)
> Reviewer: a fresh Opus `xhigh` worker, 2026-11-02 04:01 (pass after fixes: two must-fix, thirteen should-fix and
> thirteen nit items, all applied; [review](../../../src/test/20261102_backbones_sweep/bs_rule_review.md))

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** The photo model beat its teacher on Affection photos, but r13's bar licensed only "level with or
better than", and every result so far uses one encoder. You want wider tests before the 11-10 freeze. This loop
trains c* on three more image encoders, tests the frozen photo model on a second photo set with a WIN bar, and runs
the large analysis. c* and the photo model stay frozen, whatever part b finds.

**Plan** (build and run unattended; you read the report).
1. Stage 0 (checks): extract the new feature stores first (local GPU, in the day). The new code reproduces earlier
   numbers: c*'s and r13's training losses and exports, the r10 to r13 scores and picks, r13's near-copy set, the
   matched arms and r10's ablation runs.
2. Part a (claim test): c* and the photo fine-tune on SigLIP 2, SigLIP 1 and CLIP L/14 images, settings unchanged,
   five seeds each, on DAS6. Seed ranges and read-outs come from validation; then held ArtELingo and Affection test.
3. Part c (claim test): EmoSet validation picks the teacher's wording and each model's read-out by r12's rule; then
   its 17,716 test photos, minus near-copies of Affection train photos, are scored.
4. Part b (analysis, decides nothing): c*'s 13 knobs at three values (validation, 3 seeds); the ablation table with
   retrieval added (r10's existing runs); the photo fine-tune's settings (Affection validation, 3 seeds).

**Compared against.** Part a: each encoder's plain image and its own PercepT arm carry the bars; each comparison with
the Qwen teacher is its own claim. Part c: the plain Qwen image and the teacher, in the wording EmoSet validation
picks, carry the bars; frozen c*, CRL and the new encoders' photo models are plain rows.

**Outcomes.** Part a, per encoder, paintings: holds with a WIN over its plain image in all six aspect and task cells
and over its PercepT arm in clustering on all three aspects; qualified when only PercepT cells miss; fails otherwise.
Photos: holds with WINs over its plain image in both tasks. A teacher comparison at LEVEL or WIN licenses "level with
or better than the teacher" in that cell. Part c: holds with WINs over the plain image and the teacher in AMI and mAP
("better than the teacher"); qualified with WINs over the plain image in both tasks and, over the teacher, a WIN in
one task and LEVEL in the other; fails otherwise, and the verdict stands. An encoder whose runs fail twice, or a part
whose checks fail, gets no verdict (inconclusive); the others go on. We expect part a to hold on paintings for all
three, with teacher claims on emotion and style and none on genre; part c to WIN over the plain image, and over the
teacher in at least one task (likely), in both (possible).

**Cost.** Local GPU about 2½ h (extraction and two check trainings; daytime use allowed by you); DAS6 node402 about
4 to 5 h on 3 GPUs, up to about 12 h if the A6000 runs are slower than estimated; CPU about 4 h. Build Monday 11-02
(about 5 to 7 h), run when it ends; report about Tuesday 11-03, at the latest 2026-11-04 23:00.

**Check these.**
1. Parts a and c are claim tests, not confirmations (agent default): both feed paper claims. The loop stays medium.
2. Your picks after seeing our recommendation: the loop now, medium, in three parts (D1); part a's teacher
   comparisons as separate claims (D3).
3. The ablation table reuses r10's full-scale runs (eight removals, five seeds, saved exports): no new training,
   retrieval scored from them (agent default).
4. The three encoders are those that passed a held emotion test with the older recipe (10-25, 10-26); the report
   says so beside the claim.
5. Part b's grids (agent default): r10's knob values, with neighbours 10, 20 and 80 around c*'s 40; 14 photo rows
   around the photo model, among them fixed training lengths of 2, 5 and 20 epochs.
6. From the rule review (agent defaults): EmoSet photos are put in a hashed order, since each file path names its
   class; a failed encoder gets no verdict and the others go on; part b rows not done when node402 ends (11-04
   18:06) are listed as not run; the prior is corrected (the older recipe on these three encoders, at full scale,
   already beat the teacher on emotion and style).

## 1. Context and goal

Asked: "Thats a big improvement, so there's already a clear win against the teacher right? But only on a single arm
(emotion + artelingo + qwen). I would expect more robust and wider test." (decision C, 2026-11-02), then the three
parts of decision A and the grill answers (§4).

c* (glossary) beats its teacher in ArtELingo clustering on emotion (+0.0145 AMI) and style (+0.0041), is LEVEL on
genre (−0.0060) and LEVEL or WIN in retrieval (report 12-29). The photo model beats its teacher on 8,501 Affection test
photos: AMI 0.1838 against 0.1666, mAP 0.3172 against 0.2945, under a LEVEL-or-WIN bar (report 2027-01-01). All of it
uses the Qwen3-VL-Embedding-2B image as input and one photo set.

Goal: test whether c*'s gains hold on three more input encoders, whether the photo model beats its teacher on a photo
set that shaped nothing, and how c* and the photo fine-tune depend on their settings and parts.
Question this step answers: are c*'s and the photo model's results robust across encoders, datasets and settings?

## 2. Scope

In: part a (three encoders; ArtELingo three aspects and the Affection photo fine-tune; clustering and retrieval);
part c (EmoSet-118K validation and test, emotion); part b (§3.2); the feature stores these need; the auto report and
a user-read report.
Out: new recipes or settings for c* or the photo model (part b decides nothing, D6); CLIP B/32, CLIP B/16, DINOv2;
training without Qwen views; EmoSet's train split; a style or mood aspect on photos (parked, direction log); PolyArt
(the user's link); PercepT fitted on Affection or EmoSet.

## 3. Design

### 3.1 Hypothesis and prior

- **Part a.** Claim checked: report 12-28's held clustering claim and report 12-29's retrieval claim, with another
  input encoder; and r13's photo claim against the plain image. Mechanism: the gain comes from Qwen's instructed
  graphs and teacher at training; the input encoder only supplies the image. Under the older recipe (R6-5), at full
  scale, the students on these three encoders beat their frozen image on emotion by +0.034 to +0.046 AMI, beat the
  matched arm on style and genre, and beat the teacher on emotion (+0.0075 to +0.0116) and on style (+0.0108 to
  +0.0325), all WIN; on genre all three are BELOW the teacher (−0.036 to −0.041) (10-25 and 10-26
  `reported_stage4.md`). Prior: WIN over the frozen image in every cell (high confidence, 0.9 per encoder); WIN over
  the matched arm on all aspects (0.8); teacher claims in clustering on emotion (0.8 per encoder) and style (0.7), on
  genre none (0.9); in retrieval on emotion and style (0.5), on genre none (0.8). Photos: WIN over the frozen image
  (0.85); teacher BELOW (0.8). Surprise: a frozen-image miss on any aspect.
- **Part c.** Claim tested: the photo model is better than its teacher on a photo set that shaped nothing.
  Mechanism: its emotion map learned on Affection's photos carries to other photos. EmoSet was collected apart from
  Affection (its sources not checked); any photo it shares with Affection's train set leaves by the near-copy cut. Prior: a
  WIN over the plain image in both tasks (0.9); against the teacher a WIN in AMI (0.55) and in mAP (0.6); holds 0.45,
  qualified 0.25, fails 0.3. Surprise: BELOW the teacher in both tasks.
- **Part b.** No claim tested. Expectation: most knobs move c*'s mean gain by under one seed range (r10 at sample
  scale: 2 of 13 knobs moved it by more than 0.3 r); the photo model's gain shrinks with longer training (all five
  final runs stopped at best epoch 1).

### 3.2 Arms and controls

| Arm | What it is | Role |
|---|---|---|
| X-c* (X = SigLIP 2, SigLIP 1, CLIP L/14) | c*'s recipe with X's default image as the image trunk's input (first layer width = X's width); Qwen's instructed graphs, default captions and teachers as in c*; output width 2048 (10-25 spec §3); seeds 0 to 4 | candidate (SC1 to SC3, SC6) |
| X photo model | r13's photo fine-tune with r13's picked settings (80 neighbours, learning rate 0.001, correction on, teacher weight 0.25), from X-c*'s emotion checkpoint of seed s, training seed 100 + s; Qwen-built graphs as r13 | candidate (SC4 to SC6) |
| X frozen image | X's default image, no training (glossary: frozen input) | bar (SC1, SC2, SC4, SC5) |
| X matched arm | PercepT core P3 with its Stage 2 on X's image (glossary): the full-scale (stage 4) arms, seeds 0 to 4, with their stored held outputs and partitions: M2 (SigLIP 2, 10-25 round `20261025_recipe_backbones`), M8 (SigLIP 1) and M7 (CLIP L/14) (10-26 round `20261025_encoders_genre`) | bar (SC3) |
| teacher | the Qwen image instructed per aspect: r10's and r11's outputs (ArtELingo), r12's artwork wording (Affection), the wording EmoSet validation picks per task (EmoSet) | own claims (SC6); bar (SC9, SC10) |
| photo model | r13's five final runs, frozen | candidate (SC7 to SC10) |
| plain Qwen | the default Qwen image | bar (SC7, SC8) |
| frozen c*, CRL, X photo models, X frozen images, strongest PercepT entry (any encoder) | frozen c* (seeds 0 to 4), CRL (r12's Affection pick, CLIP L/14), on EmoSet; the strongest PercepT entry beside part a (C1) | descriptive |

Part b (analysis; each row changes one thing and is compared, paired, with its reference on the same machine). The
names below are plain words, so they do not clash with the PercepT entries B1 to B6 of earlier reports:
- **The knob sweep (c*'s knob sensitivity):** r10's 13 knobs at r10's three stage-1 values each (r10 spec §3.2),
  except graph neighbours at 10, 20 and 80 around c*'s 40; 39 rows; seeds 1 to 3; full scale; trained and scored on
  node402. Reference: c* rerun on node402, seeds 1 to 3.
- **The ablation table (c*'s ablation):** r10's eight removals (report 12-28, Ablation) at full scale, seeds 0 to 4,
  scored from r10's saved validation and held exports (no training); reference: r10's c* runs (same machine: all
  nine full-scale arms ran on the 3090, host `cbd437f8ce92`). An export that fails its sha256 is exported again from
  its weights on CPU (weights' hash checked first). Only if that fails, the removal and c* are retrained on node402,
  seeds 0 to 4, after the knob sweep, and that row uses the node402 c* as its reference.
- **The photo sweep (the photo fine-tune's sensitivity):** one change at a time from the photo model's settings:
  neighbours 20, 40 or 160; learning rate 0.0001, 0.0003 or 0.003; fixed training of 2, 5 or 20 epochs with no
  stopping (epoch 0 is the start point, so "2 epochs" is the checkpoint after two training epochs, r13 spec §3.4);
  teacher weight 0, 0.5 or 1; random start; content correction off; 14 rows; seeds 0 to 2 (start c* seed s, training
  seed 100 + s); trained and scored on node402. Reference: the photo model's settings rerun on node402, same seeds.
- **What the knob and photo sweep runs keep:** metrics, logs, per-painting (per-photo) partitions of the three
  states per head, and per-query AP; no weights or exports (D15). The paired intervals and the C13 spot check read
  these.

### 3.3 Data roles

| Data | Role | Stages |
|---|---|---|
| ArtELingo train, 42,969 paintings | training (part a, the knob sweep) | 1, 4 |
| ArtELingo validation, 6,152 paintings | development: stopping, X-c* seed ranges, read-out picks, the knob sweep's and the ablation table's scores | 1, 4 |
| ArtELingo held, 12,281 paintings (genre labels on 9,949) | held: part a scored; the ablation table scored (analysis, selects nothing) | 3, 4 |
| Affection train minus near-copies (69,927 photos, r13) | training, no labels (X photo models, the photo sweep) | 1, 4 |
| Affection validation, 4,251 photos | development: stopping, X photo model seed ranges and read-outs, the photo sweep's scores | 1, 4 |
| Affection test, 8,501 photos | held: scored only | 3 |
| EmoSet validation, 5,905 photos (minus near-copies) | development: wording picks, read-out picks, the photo model's seed ranges r_E | 2 |
| EmoSet test, 17,716 photos (minus near-copies) | held: scored only | 3 |
| EmoSet train | not touched (not unpacked) | none |

EmoSet keys (C6, D17): the image path string of each split-file entry (for example `image/awe/awe_08973.jpg`). The
path names the class, so no code reads the class from a key; labels come only from the entry's first field, read only
when scoring. Evaluation order (bootstrap rows, query and gallery order with its ties, cross-fit halves by parity of
position): the keys sorted by the sha256 of the key string, in place of r12's sorted keys; a unit test checks that
this order does not group classes. Near-copy cut (D9, D20): an EmoSet validation or test photo leaves its set when the
cosine between its default Qwen image and that of any of all 72,255 Affection train photos (r13's 2,328 removed
photos included) is ≥ 0.85 (r13's computation: float32, L2-normalized, joined by key); labels are never read; count
and 0.80 to 1.0 histogram reported. The cut sets are the evaluation sets of every EmoSet arm. Only EmoSet's validation
and test images and annotation files are unpacked.

### 3.4 Measurement

- **Training.** X-c*: c*'s recipe and stopping exactly (r10 spec, arm k11=40), only the image input changes. X photo
  model: r13's training with r13's picked settings, only the image input and the start checkpoint change. A run with a
  non-finite loss or a crash reruns once with the same seed; a second failure stops that encoder's part (reported).
- **ArtELingo clustering (part a, the knob sweep, the ablation table).** Report 12-28's protocol: per aspect head,
  k-means with 20 clusters, three states, mean AMI; X frozen images clustered the same way. Paired bootstrap, 2,000
  joint draws over paintings and seeds (frozen arms resample paintings only).
- **SC3's statistic (D16).** d = X-c*'s 20-cluster AMI, mean over seeds 0 to 4, minus the larger of the matched arm's
  two readings (own topics; 20 clusters, three states), each a mean over its seeds 0 to 4, from its stored held
  outputs and partitions. The interval is 10-26's matched margin (`b_matched`, 10-25 round `rb_score.py`): 2,000
  draws with `default_rng(0)`; each draw resamples the paintings once, then draws seeds independently for X-c* and
  for the matched arm (two separate trainings, so seeds are not paired); the larger reading is taken again in each
  draw.
- **ArtELingo retrieval (part a, the ablation table).** r11's protocol: the aspect head as query and gallery,
  relevance by the aspect's label, mAP; X-c*'s read-outs picked on validation by r11's rule with X-c*'s own
  validation mAP seed ranges. X frozen images (D19): r11's picks and per-query outputs for CLIP L/14 and SigLIP 2;
  SigLIP 1 picks by r11's stage 4 rule with r11's r_a, as the reused picks did. 10,000 joint draws over queries and
  seeds. The knob sweep: plain cosine (R0) only. The ablation table: R0, and the read-out form of c*'s r11 pick
  (centering plus k = 10 expansion), fitted on each run's own validation exports.
- **Affection (part a, the photo sweep).** r12's protocol: clustering with 20 clusters; retrieval with r12's
  read-out rule, X photo models using their own Affection validation seed ranges. X frozen images (D19): CLIP L/14
  keeps r12's pick (centering plus k = 10); SigLIP 2 and SigLIP 1 pick by r12's rule with r12's r_ret,mAP, as the
  reused pick did. The photo sweep: r13's sweep score path (20 clusters; cross-fit mAP with centering plus 10-neighbour
  expansion), no read-out selection.
- **EmoSet (part c).** r12's protocol on the new data, on the cut sets (§3.3): 20 clusters (and a K = 8 row, D10).
  The steps, in this order, all on EmoSet validation:
  1. r_E,clu: maximum minus minimum, over the photo model's five final runs, of emotion AMI (20 clusters, mean of
     three states). r_E,mAP: the same for emotion mAP at R0, the validation photos as queries and gallery.
  2. Each arm's read-out (the photo model, plain Qwen, each teacher wording) by r12's read-out rule, scored as
     (mAP − mAP(R0)) / r_E,mAP, seeded arms averaged over seeds.
  3. The teacher wording per task by r12's wording rule: clustering by validation AMI; retrieval by cross-fit mAP at
     each wording's own read-out pick; the photo wording counts if picked (D5), and is picked only when it beats the
     artwork wording by more than 0.001 r_E (r_E,clu; r_E,mAP).
  4. The descriptive rows (frozen c*, CRL, X photo models, X frozen images): R0, and the read-out picked by r12's
     rule with r_E,mAP.

  Each map is fitted on all EmoSet validation features (cut set) and applied to test. Bootstraps as r12.
- **Seed ranges for LEVEL.** Part a: each X-c*'s range over seeds 0 to 4 on ArtELingo validation, per aspect and task
  (AMI; mAP at R0); each X photo model's range on Affection validation. Part c: r_E,clu for SC9 and r_E,mAP for SC10.
  All unrounded. SC6's clustering LEVEL on ArtELingo uses X-c*'s own range, where report 12-28's genre LEVEL used the
  glossary's 0.0128; the report gives both units beside each cell, and the own range decides.
- **Outcomes.** WIN when the lower 95% bound > 0; LEVEL when not WIN and the lower bound > −r (the candidate's own
  range); BELOW otherwise. Per-cell intervals, no multiplicity correction; every verdict needs all its cells.
- **Part b effects.** Row minus reference, mean over seeds, with its paired interval (as above) and in seed ranges
  (glossary values: AMI 0.0046 / 0.0082 / 0.0128, mAP 0.0014 / 0.0022 / 0.0038; the photo sweep in r13's photo model
  ranges); the mean over aspects in seed ranges as r10's M score.
- **C13.** Codex re-derives, from this rule and the saved outputs (no retraining), the near-copy sets, the seed
  ranges, the wording and read-out picks, SC1 to SC10 with intervals and outcomes, and the verdicts, within 1e-6,
  before any verdict is written. Part b's sheets get a spot re-derivation of five rows.

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|---|---|---|---|---|
| 0 | regression check | new extractor paths, trainers, export, near-copy, pick and scoring paths | (i) to (xii) below | stop that part (no verdict, §6), trace, note it; the export and matched-arm branches below first |
| 1 | sweep (development only) | part a training on node402; validation | per encoder still in part a: its five X-c* runs and five X photo runs trained; its seed ranges and read-out picks written and committed before any held number of that encoder | that encoder stops (no verdict, §6); the others go on |
| 2 | sweep (development only) | EmoSet validation | r_E, read-out picks and wording picks committed before any EmoSet test number | part c stops (no verdict), note it |
| 3 | claim tests | ArtELingo held, Affection test, EmoSet test; all frozen | SC1 to SC10 | report |
| 4 | ablation (part b) | the knob and photo sweeps on validation; the ablation table on validation and held | the three sheets written | report |
| 5 | descriptive | all | §3.6 | report |

Order on node402: part a first, then the photo sweep, then the knob sweep. Part b never waits for stages 2 or 3, and
its numbers enter no verdict. The first X-c* runs are timed and the remaining node time is estimated from them. Rows
not finished by 2026-11-04 18:06 (node402's end) are listed as not run; the report does not wait for a new
reservation (D21). Stage 0:
- (i) Extraction: the extractor reproduces by key, cosine ≥ 0.999, the first 100 sorted Affection validation keys of
  the stored CLIP L/14 valtest store, of the default Qwen store and of both instructed emotion stores (artwork, photo),
  and the first 100 ArtELingo rows of the SigLIP 2 and SigLIP 1 stores.
- (ii) New stores: one row per key, finite, unit norm within 1e-3, keys join (C6); EmoSet unpack counts 5,905
  validation and 17,716 test photos in 8 classes.
- (iii) Near-copies: the EmoSet count and histogram written; no gate on the number.
- (iv) c* trainer: the generalized trainer with the Qwen encoder, seed 0, on the local GPU, gives c*'s seed-0 best
  epoch and best development loss per head within 1e-4 (c*'s sidecars). A test checks that another width changes
  only the image trunk's first layer.
- (v) Photo trainer: the generalized trainer with the Qwen encoder, r13's final settings, seed 0, on the local GPU,
  gives r13's final seed-0 best epoch and best development loss within 1e-4.
- (vi) Scoring replays, within 1e-9: report 12-28's c* and teacher held AMI and their intervals; r11's stage 5 c*
  and teacher mAP, intervals and c*'s pick; r13's SC1 to SC5 values and intervals.
- (vii) Stored arms: each matched arm's stored outputs and partitions give its entries in 10-25's and 10-26's
  `reported_stage4` (own and 20-cluster readings) within 1e-9. r10's ablation exports match their sidecars' sha256,
  and all eight removals' held changes reproduce report 12-28's held ablation table within 1e-9 (r10
  `outputs/ablation/held.json`; for example the relational term, −0.0440 [−0.0512, −0.0372]). Branches (D18): an export that
  fails is exported again from its weights on CPU (§3.2); a matched arm that fails its replay reruns on node402 with
  10-25's Stage 2 settings, seeds 0 to 4, after part a's training, and its new outputs pass the checks of (ii) before
  use. Any other failed stage 0 gate stops its part.
- (viii) Machines: c* seed 1 trained on node402; its validation AMI difference from the 3090 run is recorded; no
  gate.
- (ix) Export path: the generalized export code, given c*'s and r13's checkpoints and no training step, reproduces
  their stored ArtELingo validation and held exports and their Affection validation and test exports within 1e-3
  (r12's bar), all five seeds.
- (x) Near-copy code: given r13's inputs, it reproduces r13's removed set (2,328 keys) exactly.
- (xi) Wording rule: fed r12's validation outputs, it reproduces r12's wording picks and their scores within 1e-9.
- (xii) Bar paths, within 1e-9: the F2, F7 and F8 20-cluster held AMIs (0.0569, 0.0659, 0.0672 on emotion, and
  their style and genre entries); the `b_matched` d and intervals of K2, K7 and K8 (style, genre; `reported_stage4`);
  r11's stage 5 CLIP L/14 and SigLIP 2 frozen-image mAPs.

### 3.6 Diagnostics

After the verdicts, deciding nothing: R0 rows for every arm; EmoSet with 8 clusters; EmoSet per-class AP; the photo
model minus frozen c* on EmoSet, paired; the share of the teacher's gain over the plain image kept by the photo model
(Affection, EmoSet) and by each X model; seed spreads on held beside the validation ranges; r_E beside the photo
model's spread on EmoSet test; SC7 to SC10 on the uncut EmoSet test set, and the removed photos per class; the
node402 minus 3090 difference (0(viii)); the knob sweep's curves per knob (figure); the photo sweep's
training-length rows against the reference.

### 3.7 Frozen vs rerun

Frozen: c*'s and r13's weights and exports; r10's ablation runs; r11's, r12's and r13's picks and per-query outputs of
every reused arm; the matched arms' stored outputs; part a's seed ranges and picks before held scoring; part c's
r_E, read-out picks and wording picks before EmoSet test scoring. Rerun: the new stores, X-c* and X photo model
training and exports, EmoSet exports of every arm, the knob and photo sweeps' training, all scoring.

## 4. Decisions

Grill rows: the r14 decide 2 chat, 2026-11-02, two rounds (round 1 about 03:10, round 2 about 03:20, closing about
03:35). The r14 decide chat ran 2026-11-01 21:42 to 2026-11-02 03:00. D17 to D21 come from the rule review (04:01).

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|----|----------|----------|----------------------------|------------|------|
| D1 | The loop | one r14 loop now, three parts, medium, on DAS6; part b analysis only | wider tests before the freeze ("freeze and write") | user, after seeing the recommendation | 2026-11-02, r14 decide |
| D2 | Part a scope | SigLIP 2, SigLIP 1, CLIP L/14; ArtELingo three aspects and the Affection photo fine-tune; clustering and retrieval; frozen recipe; five seeds | three encoders from two families (one encoder; ArtELingo only) | user (Q1 c) | 2026-11-02 03:10 |
| D3 | Part a bars | WIN over the frozen image (all cells) and over the matched arm (clustering); each teacher comparison its own claim, LEVEL or WIN | "make it less strict" (every bar, the teacher's included, for a holds) | user, after seeing the recommendation (Q1′ b) | 2026-11-02 03:20 |
| D4 | Part c bar | holds: WIN over the plain image and the teacher in AMI and mAP; qualified: one WIN, one LEVEL; fails otherwise; after a failure the verdict stands and the next step is discussed | either task; clustering only | user (Q2 i; scenario check yes) | 2026-11-02 03:10, 03:20 |
| D5 | Part c protocol | r12's rule on EmoSet validation (wording per task, read-out per arm), 20 clusters; the photo wording counts if picked | "this makes life easier" (all carried from Affection; 8 clusters, plain cosine) | user (Q3 A; scenario check yes) | 2026-11-02 03:10, 03:20 |
| D6 | Part b's role | analysis only; c* and the photo model stay frozen whatever it finds, and the report says so | the rerun cost (about 7 h and a day of rework); scenario check | user | 2026-11-02, r14 decide; 03:20 |
| D7 | Part b content | the knob sweep, the ablation table and the photo sweep, at full scale | "this fits more" (one of them) | user (Q4) | 2026-11-02 03:20 |
| D8 | Timing | one spec, build and run; extraction first, on the local GPU, also on weekdays 10:00 to 19:00 | "one spec will be fine" (split part b off) | user (Q5 i) | 2026-11-02 03:20 |
| D9 | Minor batch | folder `src/test/20261102_backbones_sweep/`, prefix `bs_`; EmoSet validation and test only unpacked; near-copy cut 0.85; EmoSet rows (§3.2); r_E as LEVEL unit; X runs and picks (§3.2, §3.4); matched arms reused after a check; the knob and photo sweeps on 3 seeds and validation, the ablation table on 5 seeds, validation and held; training on node402 (the knob and photo sweeps also score there), extraction and other scoring local; glossary terms added | the grill's minor batch | agent default (accepted by the user); where the sweeps score, clarified by the rule review (D15 needs it) | 2026-11-02 03:20, 04:01 |
| D10 | EmoSet rows | K = 8 clustering as a plain row; CRL with r12's pick carried | readers expect the class count; CRL decides nothing here | agent default (accepted with D9) | 2026-11-02 03:20 |
| D11 | Stage types | parts a and c claim tests; stages 1 and 2 sweeps (development only); part b ablation | both feed paper claims (confirmation); stages 1 and 2 pick read-outs and wordings, as r12's stage 1; part b selects nothing (sweep) | agent default; stages 1 and 2 changed by the rule review | 2026-11-02 03:40, 04:01 |
| D12 | Knob sweep grid | r10's values; neighbours 10, 20, 80; seeds 1 to 3; node402 reference | spans both sides of 40 (r10's 10, 30, 40 around 20) | agent default | 2026-11-02 03:40 |
| D13 | Ablation table runs | r10's existing full-scale runs, scored again with retrieval | they are the same model and seeds (retrain 40 runs) | agent default | 2026-11-02 03:40 |
| D14 | Photo sweep grid | §3.2's 14 rows | tests the pick's neighbours, the step size, the training length and the teacher (r13's nine rows on one seed) | agent default | 2026-11-02 03:40 |
| D15 | Run outputs | part a keeps checkpoints and exports; the knob and photo sweeps keep metrics, logs, partitions and per-query AP only | a three-head run with weights and exports takes about 0.4 GB (r10's c*: 96 MB weights, 144 MB held and 144 MB validation exports per seed); the knob sweep alone would put about 46 GB on SSD2 | agent default | 2026-11-02 03:40 |
| D16 | Matched-arm bar and C1 | the bar uses the matched arm, with 10-26's `b_matched` statistic (§3.4); the strongest PercepT entry of any encoder is reported beside | the matched arm drops only the student; a Qwen-image PercepT entry differs in the encoder too; r10's shared seed draw pairs two separate trainings | agent default; statistic from the rule review | 2026-11-02 03:40, 04:01 |
| D17 | EmoSet order | keys sorted by the sha256 of the key string; no code reads the class from a key | each path names its class, so sorted keys would group classes in the cross-fit halves, bootstrap rows and ties (split-file order) | agent default (rule review) | 2026-11-02 04:01 |
| D18 | Failure outcomes | per encoder: no verdict (inconclusive) when its runs fail twice or its stage 0 gate fails; the others go on; an export retried on CPU; a matched arm rerun on node402 | one encoder's failure need not stop part a (stop all of part a) | agent default (rule review) | 2026-11-02 04:01 |
| D19 | Bar arms' read-outs | frozen images pick with the original round's rule and ranges (r11's r_a; r12's r_ret,mAP), so new picks match the reused ones | one rule for reused and new bar picks (each X model's own ranges) | agent default (rule review) | 2026-11-02 04:01 |
| D20 | Near-copy reference | all 72,255 Affection train photos | the larger set includes r13's 2,328 removed photos (the 69,927 the photo model used) | agent default (rule review) | 2026-11-02 04:01 |
| D21 | Node time-box | rows not done by 2026-11-04 18:06 listed as not run; the report does not wait | the report is due 11-04 23:00 (wait for a new reservation) | agent default (rule review) | 2026-11-02 04:01 |

**Amendments**: none.

## 5. Assumptions and open questions

- Assumption: the extraction takes about 2¼ h (throughputs of the ArtELingo stores and r13's Qwen extraction). Not
  checked. Large EmoSet images may slow Qwen's extraction (dynamic resolution); not checked.
- Assumption: a full-scale c* run takes about 4 to 5 min on an A6000, as on the 3090 (3.5 to 5.3 min). Not checked;
  r10's node figure (3 min) was at sample scale, and r10's held ablation took 252 GPU-min against 80 estimated. The
  first X-c* runs are timed (§3.5).
- Assumption: EmoSet was collected apart from Affection's sources. Not checked; the near-copy cut covers shared
  photos.
- Assumption: EmoSet's split files give one emotion label per image in its 8 classes (README). Not checked in the
  files; stage 0(ii) checks it.
- Assumption: the matched arms' per-painting outputs are stored for all three aspects (10-26's margins name them).
  Stage 0(vii) checks it.
- Assumption: node402 can read the new stores through the cluster CLI's data map (a new path is verified with a real
  job, project memory). Not checked.
- Open questions: none.

## 6. Success criteria

**Decision mode.** Binding rule: this section with §3.4 and §3.5, committed with this spec before any code.

| ID | What is measured | Compared against | Bar | On which data |
|----|------------------|------------------|-----|---------------|
| SC1 | X-c*'s AMI, emotion, style, genre | X frozen image | WIN in each | ArtELingo held |
| SC2 | X-c*'s mAP, picked read-out, three aspects | X frozen image, its picked read-out | WIN in each | ArtELingo held |
| SC3 | X-c*'s AMI, three aspects | X matched arm, larger reading | WIN in each | ArtELingo held |
| SC4 | X photo model's emotion AMI | X frozen image | WIN | Affection test |
| SC5 | X photo model's emotion mAP, picked read-out | X frozen image, its picked read-out | WIN | Affection test |
| SC6 | X-c*'s AMI and mAP (three aspects), X photo model's AMI and mAP | the teacher, its picked read-out | LEVEL or WIN, each cell its own claim | ArtELingo held, Affection test |
| SC7 | photo model's emotion AMI | plain Qwen | WIN | EmoSet test |
| SC8 | photo model's emotion mAP, picked read-out | plain Qwen, its picked read-out | WIN | EmoSet test |
| SC9 | photo model's emotion AMI | teacher, clustering wording | WIN | EmoSet test |
| SC10 | photo model's emotion mAP, picked read-out | teacher, retrieval wording, its picked read-out | WIN | EmoSet test |

- **Part a, per encoder.** Paintings: holds when all nine cells of SC1 to SC3 WIN; qualified when SC1 and SC2 WIN in
  all six cells and one or more SC3 cells do not; fails otherwise. Photos: holds when SC4 and SC5 WIN; fails
  otherwise. As claim-test outcomes: holds GO, qualified GO for the qualified claim, fails NO-GO. Each SC6 cell at
  LEVEL or WIN gives its claim; BELOW gives none and keeps its row.
- **Inconclusive (D18).** A paintings verdict needs X-c*'s five runs; a photos verdict needs X's five photo runs. If
  an encoder's runs fail twice (§3.4) or a stage 0 gate on its path fails, that encoder's claim and its SC6 cells
  are inconclusive: no claim, and the rows that exist are reported. A failed photo run leaves the paintings verdict
  as it is. The other encoders go on.
- **Part c.** Holds when SC7 to SC10 WIN; qualified when SC7 and SC8 WIN and one of SC9 and SC10 WINs while the other
  is LEVEL; fails otherwise (D4). Holds GO, qualified GO for the qualified claim, fails NO-GO; a failed stage 0 gate
  on its path or a failed stage 2 gate makes part c inconclusive.
- **Selection rules.** Stage 1: X-c*'s read-outs by r11's rule (ArtELingo) and X photo models' by r12's rule
  (Affection), each with its own ranges; the frozen-image bars as §3.4 (D19). Stage 2, in §3.4's order: r_E, each
  arm's read-out by r12's rule with r_E,mAP, then the teacher wording per task by r12's wording rule. Part b selects
  nothing.
- **Licensed claim (part a, paintings holds):** "On 12,281 held ArtELingo paintings, c*'s recipe with the <X> image
  as its only input at test (Qwen3-VL-Embedding-2B's instructed graphs, captions and teacher at training) clusters and
  retrieves paintings by emotion, style and genre better than the plain <X> image, and clusters them better than
  PercepT on the <X> image (AMI with 20 clusters; mAP with read-outs picked on validation; 5 seeds). <X> is one of
  three encoders that passed an earlier held emotion test." Qualified: "… better than the plain <X> image; against
  PercepT on the <X> image, better in <aspects>" (with no SC3 WIN: "…; against PercepT on the <X> image it is better
  in no aspect"). Photos: "On 8,501 held Affection photos, the same model on <X>,
  fine-tuned on Affection's unlabeled train photos with the settings picked for Qwen, clusters and retrieves photos by
  emotion better than the plain <X> image." SC6: "<model on X> is level with or better than the Qwen image instructed
  for <aspect> in <task> on <data>."
- **Licensed claim (part c, holds):** "On <n> EmoSet-118K test photos (photos with a near-copy in Affection's train
  set removed), the photo model (c*, trained on ArtELingo paintings, then fine-tuned on Affection's unlabeled train
  photos; frozen; never trained on EmoSet) clusters and retrieves photos by emotion better
  than the plain Qwen3-VL-Embedding-2B image and better than the Qwen image instructed for emotion (AMI with 20
  clusters; mAP; wording and read-outs picked on EmoSet validation by the rule fixed in r12; 5 seeds)." Qualified:
  "… better than the instructed image in <task> and level with it in <other task>."
- **Licensed (part b, as attribution only):** "On <data>, removing <part> from c* (or setting <knob> to <value>, or
  changing <setting> of the photo fine-tune) changes <aspect> AMI (or mAP) by <Δ> [<interval>] against c* (or the
  photo model) on the same machine (<seeds>)." It explains c*'s and the photo model's results; it never names a
  better setting.
- **Not licensed:** encoders not tested; training without Qwen views; other photo sets; EmoSet's other attributes;
  any setting from part b as better than c*'s or the photo model's.

## 7. Constitution check

| Principle | Pass or exception | Note |
|-----------|-------------------|------|
| C1 strongest comparator | pass | part a: the frozen image and matched arm as bars, the teacher and the strongest PercepT entry beside (D16); part c: the teacher in the wording EmoSet picks |
| C6 sample IDs | pass | extraction keys; EmoSet identifiers from its split files; stage 0(ii) |
| C7 rule before code | pass | §6 with §3.4 and §3.5, reviewed by a fresh Opus `xhigh` worker before the approving commit |
| C8 prior and disclosure | pass | prior §3.1; disclosure §8 |
| C12 regression check | pass | stage 0(i), (iv) to (vii), (ix) to (xii) |
| C13 re-derivation | pass | Codex, §3.4 |
| `held-data.md` | pass, with one disclosed earlier selection | stopping, ranges and picks on validation; held scored only; the knob and photo sweeps read validation only; the ablation table reads held and selects nothing; the matched arm's larger reading is taken on held, which only makes the bar stricter. Disclosed: the three encoders passed an earlier held emotion test (10-25, 10-26) on these held paintings, and the licensed sentence names it (D2, user). Query expansion reads unlabeled test embeddings at test time, with k fixed on validation, as in r11 and r12 |

Exceptions: none.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|---|---|---|
| EmoSet photos near-copy Affection train photos | the near-copy count and histogram | they leave EmoSet's sets (cut 0.85); the report gives the count |
| EmoSet's labels differ in kind from Affection's (one label per image, other annotators) | class counts, per-class AP | comparisons stay within EmoSet; reported |
| A new encoder's run fails or diverges | the rerun rule (§3.4) | that encoder's part stops; the others go on |
| Device differences (3090 against A6000) bias part b | 0(viii); the knob and photo sweeps use node402 references | reported; the ablation table stays on one machine |
| r_E is very small or very large, so LEVEL in SC9 and SC10 is too strict or too loose | r_E beside the photo model's spread on EmoSet test (§3.6) | reported; r_E is not changed after results |
| node402 runs take longer than estimated | the timed first X-c* runs | part a first; rows not done by 2026-11-04 18:06 are listed as not run (D21) |
| Part b shows a better setting than c*'s or the photo model's | the knob and photo sweep sheets | reported as analysis; nothing changes (D6) |
| EmoSet's data licence is unclear (repo MIT) | dataset note | evaluation only, no redistribution; the paper cites it |
| Selection disclosure: the three encoders passed a held emotion test with the older recipe (10-25, 10-26; CLIP B/32 and B/16 inconclusive, DINOv2 failed); ArtELingo validation carried r10's 59 runs and earlier rounds; Affection validation r12's 2 wordings, up to 25 read-outs per arm, 4 CRL versions and r13's 9 rows; EmoSet validation now 2 wordings and up to 25 read-outs per main arm | validation numbers inflated; the encoder set chosen on held data | only test numbers are results; every pick is frozen before scoring; the claim names how the encoders were chosen |

## 9. Deliverables and review

- At the end: the round folder `src/test/20261102_backbones_sweep/` (prefix `bs_`) with code, tests, log and
  per-item outputs; new stores under `/data/SSD2/pre_extract/affection_local/` and `/data/SSD2/pre_extract/emoset/`;
  EmoSet validation and test images under `/data/SSD/emoset/`; run outputs under
  `/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/`; the auto report `docs/reports/auto/percept/` (next
  sequence date) with its results brief and index row; a user-read report.
- Reviews: this rule by a fresh Opus `xhigh` worker before the approving commit; tickets per `matt-chain.md`, with
  the generalized trainers, the near-copy cut, the scoring paths, the picks and the outcome code as verdict code; the
  C13 re-derivation by Codex; the final whole-branch review.
- Stop points: none between approval and the reports. Time-box: the report by 2026-11-04 23:00 (node402 ends
  2026-11-04 18:06; rows not done by then are listed as not run, D21; the report does not wait for the user's
  re-reservation).
- Budget: local GPU about 2½ h, heavy, under the lock, in the day (D8): Affection SigLIP 2 and SigLIP 1 (85,007
  images each) and CLIP L/14 train (72,255), about 1 h; EmoSet validation and test (23,621) in three Qwen views and
  three other encoders, about 1¼ h; the 0(iv) and 0(v) check trainings, about 8 min. node402, 3 A6000: part a about
  1.7 GPU-h (30 runs), the knob sweep about 9 GPU-h (120 runs), the photo sweep about 1.7 GPU-h (45 runs), together
  about 4 to 5 h of wall time; 2 to 3 times that still ends before 11-04 18:06 if node work starts by the morning of
  11-03. CPU about 4 h at 8 threads (scoring, bootstraps, C13). Disk: about 2.5 GB of EmoSet images on SSD, about
  1 GB of new stores and about 10 GB of run outputs on SSD2 (15 X-c* runs at about 0.38 GB, 15 X photo models at
  about 0.18 GB, EmoSet exports about 1 GB).
- Contingencies (D18): an ablation export that fails is exported again on CPU; only if that fails, the removal and
  c* are retrained on node402, seeds 0 to 4, after the knob sweep (about 0.4 GPU-h per removal, plus 0.4 for c* once). A matched arm that
  fails its replay reruns on node402, seeds 0 to 4, after part a's training.

Body length: about 330 non-blank lines against a claim-test target of about 150 plus the rule: two claim tests and
an ablation share one spec (D8), and the rule review added the failure paths, the pick order and four replays.
