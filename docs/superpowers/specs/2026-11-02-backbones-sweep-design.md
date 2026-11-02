# c* on three more encoders, the photo model on EmoSet, and the large analysis sweep: design

> Date: 2026-11-02 03:40 · Status: draft · Kind: experiment
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

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** The photo model beat its teacher on Affection photos, but r13's bar licensed only "level with or
better than", and every result so far uses one encoder. You want wider tests before the 11-10 freeze. This loop
trains c* on three more image encoders, tests the frozen photo model on a second photo set with a WIN bar, and runs
the large analysis. c* and the photo model stay frozen, whatever part b finds.

**Plan** (build and run unattended; you read the report).
1. Stage 0 (checks): extract the new feature stores first (local GPU, in the day). The new code reproduces earlier
   numbers: c*'s and r13's training losses, the r11, r12 and r13 scores, the matched arms.
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
("better than the teacher"); qualified with a WIN in one task and LEVEL in the other; fails otherwise, and the
verdict stands. We expect part a to hold on paintings for all three, with teacher claims on emotion at most; part c
to WIN over the plain image, and over the teacher in at least one task (likely), in both (possible).

**Cost.** Local GPU about 2¼ h (extraction; daytime use allowed by you); DAS6 node402 about 4 to 5 h on 3 GPUs; CPU
about 4 h. Build Monday 11-02 (about 5 to 7 h), run when it ends; report about Tuesday 11-03, at the latest
2026-11-04 23:00.

**Check these.**
1. Parts a and c are claim tests, not confirmations (agent default): both feed paper claims. The loop stays medium.
2. Part a's teacher comparisons as separate claims: your pick after seeing our recommendation.
3. The ablation table reuses r10's full-scale runs (eight removals, five seeds, saved weights): no new training,
   retrieval scored from them (agent default).
4. The three encoders are those that passed a held emotion test with the older recipe (10-25, 10-26); the report
   says so beside the claim.
5. Part b's grids (agent default): r10's knob values, with neighbours 10, 20 and 80 around c*'s 40; 14 photo rows
   around the photo model, among them fixed training lengths of 2, 5 and 20 epochs.

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
  graphs and teacher at training; the input encoder only supplies the image. Under the older recipe (R6-5), every
  student on these encoders beat its frozen image on emotion (+0.021 to +0.036) and PercepT on style and genre; only
  SigLIP 1 and CLIP L/14 reached the teacher on style, none on genre (report 12-25). Prior: WIN over the frozen image
  in every cell (high confidence, 0.9 per encoder); WIN over the matched arm on all aspects (0.8); teacher claims on
  emotion for SigLIP 1 and SigLIP 2 (0.5), on style for SigLIP 1 and CLIP L/14 (0.4), on genre none (0.9). Photos:
  WIN over the frozen image (0.85); teacher BELOW (0.8). Surprise: a frozen-image miss on any aspect.
- **Part c.** Claim tested: the photo model is better than its teacher on photos from new sources. Mechanism: its
  emotion map learned on Affection's photos carries to other photos; EmoSet shares no source with Affection. Prior: a
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
| X matched arm | PercepT core P3 with its Stage 2 on X's image (glossary), stored outputs of 10-25 (SigLIP 2) and 10-26 (SigLIP 1, CLIP L/14) | bar (SC3) |
| teacher | the Qwen image instructed per aspect: r10's and r11's outputs (ArtELingo), r12's artwork wording (Affection), the wording EmoSet validation picks per task (EmoSet) | own claims (SC6); bar (SC9, SC10) |
| photo model | r13's five final runs, frozen | candidate (SC7 to SC10) |
| plain Qwen | the default Qwen image | bar (SC7, SC8) |
| frozen c*, CRL, X photo models, X frozen images, strongest PercepT entry (any encoder) | frozen c* (seeds 0 to 4), CRL (r12's Affection pick, CLIP L/14), on EmoSet; the strongest PercepT entry beside part a (C1) | descriptive |

Part b (analysis; each row changes one thing and is compared, paired, with its reference on the same machine):
- **B1, c*'s knob sensitivity:** r10's 13 knobs at r10's three stage-1 values each (r10 spec §3.2), except graph
  neighbours at 10, 20 and 80 around c*'s 40; 39 rows; seeds 1 to 3; full scale; on node402. Reference: c* rerun on
  node402, seeds 1 to 3.
- **B2, c*'s ablation table:** r10's eight removals (report 12-28, Ablation) at full scale, seeds 0 to 4, from r10's
  saved weights and exports (no training); reference: r10's c* runs (same machine, 3090). A run whose weights fail
  their hash or will not load retrains on node402, with its c* reference retrained there too.
- **B3, the photo fine-tune's sensitivity:** one change at a time from the photo model's settings: neighbours 20, 40
  or 160; learning rate 0.0001, 0.0003 or 0.003; fixed training of 2, 5 or 20 epochs with no stopping (checkpoint at
  the last epoch); teacher weight 0, 0.5 or 1; random start; content correction off; 14 rows; seeds 0 to 2 (start
  c* seed s, training seed 100 + s); on node402. Reference: the photo model's settings rerun on node402, same seeds.

### 3.3 Data roles

| Data | Role | Stages |
|---|---|---|
| ArtELingo train, 42,969 paintings | training (part a, B1) | 1, 4 |
| ArtELingo validation, 6,152 paintings | development: stopping, X-c* seed ranges, read-out picks, B1 and B2 scores | 1, 4 |
| ArtELingo held, 12,281 paintings (genre labels on 9,949) | held: part a scored; B2 scored (analysis, selects nothing) | 3, 4 |
| Affection train minus near-copies (69,927 photos, r13) | training, no labels (X photo models, B3) | 1, 4 |
| Affection validation, 4,251 photos | development: stopping, X photo model seed ranges and read-outs, B3 scores | 1, 4 |
| Affection test, 8,501 photos | held: scored only | 3 |
| EmoSet validation, 5,905 photos (minus near-copies) | development: wording picks, read-out picks, the photo model's seed ranges r_E | 2 |
| EmoSet test, 17,716 photos (minus near-copies) | held: scored only | 3 |
| EmoSet train | not touched (not unpacked) | none |

EmoSet keys: the image identifiers of EmoSet's split files, in extraction order (C6); labels: the one emotion per
image in those files, read only when scoring. Near-copy cut (D9): an EmoSet validation or test photo leaves its set
when the cosine between its default Qwen image and that of any Affection train photo is ≥ 0.85 (r13's computation:
float32, L2-normalized, joined by key); labels are never read; count and 0.80 to 1.0 histogram reported. Only EmoSet's
validation and test images and annotation files are unpacked.

### 3.4 Measurement

- **Training.** X-c*: c*'s recipe and stopping exactly (r10 spec, arm k11=40), only the image input changes. X photo
  model: r13's training with r13's picked settings, only the image input and the start checkpoint change. A run with a
  non-finite loss or a crash reruns once with the same seed; a second failure stops that encoder's part (reported).
- **ArtELingo clustering (part a, B1, B2).** Report 12-28's protocol: per aspect head, k-means with 20 clusters,
  three states, mean AMI; X frozen images clustered the same way. Paired bootstrap, 2,000 joint draws over paintings
  and seeds (frozen arms resample paintings only). Matched arm: its larger reading (own topics or 20 clusters), as
  10-26's matched margin, from its stored per-painting outputs and seeds.
- **ArtELingo retrieval (part a, B2).** r11's protocol: the aspect head as query and gallery, relevance by the
  aspect's label, mAP; read-outs picked on validation by r11's rule with X-c*'s own validation mAP seed ranges; X
  frozen images: r11's picks and per-query outputs for CLIP L/14 and SigLIP 2, SigLIP 1 by the same rule. 10,000
  joint draws over queries and seeds. B1: plain cosine (R0) only; B2: R0 and c*'s r11 pick applied unchanged.
- **Affection (part a, B3).** r12's protocol (clustering 20 clusters; retrieval with r12's read-out rule using each X
  photo model's own validation seed ranges). B3: r13's sweep score path (20 clusters; cross-fit mAP with centering
  plus 10-neighbour expansion), no read-out selection.
- **EmoSet (part c).** r12's protocol on the new data: 20 clusters (and a K = 8 row, D10); the teacher wording per
  task by r12's wording rule on EmoSet validation, then each main arm's read-out by r12's read-out rule on EmoSet
  validation, with the photo model's EmoSet validation seed ranges r_E (maximum minus minimum over its five runs, AMI
  and mAP at R0) in place of r12's c* ranges. Each map is fitted on all EmoSet validation features and applied to
  test. Bootstraps as r12.
- **Seed ranges for LEVEL.** Part a: each X-c*'s range over seeds 0 to 4 on ArtELingo validation, per aspect and task
  (AMI; mAP at R0); each X photo model's range on Affection validation. Part c: r_E. All unrounded.
- **Outcomes.** WIN when the lower 95% bound > 0; LEVEL when not WIN and the lower bound > −r (the candidate's own
  range); BELOW otherwise. Per-cell intervals, no multiplicity correction; every verdict needs all its cells.
- **Part b effects.** Row minus reference, mean over seeds, with its paired interval (as above) and in seed ranges
  (glossary values: AMI 0.0046 / 0.0082 / 0.0128, mAP 0.0014 / 0.0022 / 0.0038; B3 in r13's photo model ranges); the
  mean over aspects in seed ranges as r10's M score.
- **C13.** Codex re-derives, from this rule and the saved outputs (no retraining), the near-copy sets, the seed
  ranges, the wording and read-out picks, SC1 to SC10 with intervals and outcomes, and the verdicts, within 1e-6,
  before any verdict is written. Part b's sheets get a spot re-derivation of five rows.

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|---|---|---|---|---|
| 0 | regression check | new extractor paths, trainers and scoring paths | (i) to (viii) below | stop that part, trace, note it |
| 1 | measurement | part a training on node402; validation | 30 runs trained; seed ranges and read-out picks written and committed before any held number of part a | stop, note it |
| 2 | measurement | EmoSet validation | wording picks, read-out picks and r_E committed before any EmoSet test number | stop, note it |
| 3 | claim tests | ArtELingo held, Affection test, EmoSet test; all frozen | SC1 to SC10 | report |
| 4 | ablation (part b) | B1, B3 on validation; B2 on validation and held | the three sheets written | report |
| 5 | descriptive | all | §3.6 | report |

Order on node402: part a first, then B3, then B1. Part b never waits for stages 2 or 3, and its numbers enter no
verdict. Stage 0:
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
- (vii) Stored arms: each matched arm's outputs give 10-26's reported matched margins' entries within 1e-9; r10's
  ablation weights match their hashes and their held exports give report 12-28's held relational-term change
  (−0.0440 [−0.0512, −0.0372]) within 1e-9. A matched arm that fails reruns (PercepT Stage 2 on X, 10-25's settings).
- (viii) Machines: c* seed 1 trained on node402; its validation AMI difference from the 3090 run is recorded; no
  gate.

### 3.6 Diagnostics

After the verdicts, deciding nothing: R0 rows for every arm; EmoSet with 8 clusters; EmoSet per-class AP; the photo
model minus frozen c* on EmoSet, paired; the share of the teacher's gain over the plain image kept by the photo model
(Affection, EmoSet) and by each X model; seed spreads on held beside the
validation ranges; the node402 minus 3090 difference (0(viii)); B1 curves per knob (figure); B3's training-length rows
against the reference.

### 3.7 Frozen vs rerun

Frozen: c*'s and r13's weights and exports; r10's ablation runs; r11's, r12's and r13's picks and per-query outputs of
every reused arm; the matched arms' stored outputs; part a's seed ranges and picks before held scoring; part c's
wording picks, read-out picks and r_E before EmoSet test scoring. Rerun: the new stores, X-c* and X photo model
training and exports, EmoSet exports of every arm, B1 and B3 training, all scoring.

## 4. Decisions

Grill rows: the r14 decide 2 chat, 2026-11-02, two rounds (round 1 about 03:10, round 2 about 03:20, closing about
03:35).

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|----|----------|----------|----------------------------|------------|------|
| D1 | The loop | one r14 loop now, three parts, medium, on DAS6; part b analysis only | wider tests before the freeze ("freeze and write") | user, after seeing the recommendation | decide 1, 11-02 |
| D2 | Part a scope | SigLIP 2, SigLIP 1, CLIP L/14; ArtELingo three aspects and the Affection photo fine-tune; clustering and retrieval; frozen recipe; five seeds | three encoders from two families (one encoder; ArtELingo only) | user (Q1 c) | grill 1 |
| D3 | Part a bars | WIN over the frozen image (all cells) and over the matched arm (clustering); each teacher comparison its own claim, LEVEL or WIN | "make it less strict" (every bar, the teacher's included, for a holds) | user, after seeing the recommendation (Q1′ b) | grill 2 |
| D4 | Part c bar | holds: WIN over the plain image and the teacher in AMI and mAP; qualified: one WIN, one LEVEL; fails otherwise; after a failure the verdict stands and the next step is discussed | either task; clustering only | user (Q2 i; scenario check yes) | grill 1, 2 |
| D5 | Part c protocol | r12's rule on EmoSet validation (wording per task, read-out per arm), 20 clusters; the photo wording counts if picked | "this makes life easier" (all carried from Affection; 8 clusters, plain cosine) | user (Q3 A; scenario check yes) | grill 1, 2 |
| D6 | Part b's role | analysis only; c* and the photo model stay frozen whatever it finds, and the report says so | the rerun cost (about 7 h and a day of rework); scenario check | user | decide 1; grill 2 |
| D7 | Part b content | B1, B2 and B3 at full scale | "this fits more" (one of them) | user (Q4) | grill 2 |
| D8 | Timing | one spec, build and run; extraction first, on the local GPU, also on weekdays 10:00 to 19:00 | "one spec will be fine" (split part b off) | user (Q5 i) | grill 2 |
| D9 | Minor batch | folder `src/test/20261102_backbones_sweep/`, prefix `bs_`; EmoSet validation and test only unpacked; near-copy cut 0.85; EmoSet rows (§3.2); r_E as LEVEL unit; X runs and picks (§3.2, §3.4); matched arms reused after a check; B1 and B3 on 3 seeds and validation, B2 on 5 seeds, validation and held; training on node402, extraction and scoring local; glossary terms added | the grill's minor batch | agent default (accepted by the user) | grill 2 |
| D10 | EmoSet rows | K = 8 clustering as a plain row; CRL with r12's pick carried | readers expect the class count; CRL decides nothing here | agent default (accepted with D9) | grill 2 |
| D11 | Stage types | parts a and c claim tests; part b ablation | both feed paper claims (confirmation); part b selects nothing (sweep) | agent default | 03:40 |
| D12 | B1 grid | r10's values; neighbours 10, 20, 80; seeds 1 to 3; node402 reference | spans both sides of 40 (r10's 10, 30, 40 around 20) | agent default | 03:40 |
| D13 | B2 runs | r10's existing full-scale runs, scored again with retrieval | they are the same model and seeds (retrain 40 runs) | agent default | 03:40 |
| D14 | B3 grid | §3.2's 14 rows | tests the pick's neighbours, the step size, the training length and the teacher (r13's nine rows on one seed) | agent default | 03:40 |
| D15 | Run outputs | part a keeps checkpoints and exports; B1 and B3 keep metrics, per-item scores and logs only | a three-head run with weights and exports takes about 0.4 GB (r10's c*: 96 MB weights, 144 MB held and 144 MB validation exports per seed); B1 alone would put about 46 GB on SSD2 | agent default | 03:40 |
| D16 | Matched-arm bar and C1 | the bar uses the matched arm; the strongest PercepT entry of any encoder is reported beside | the matched arm drops only the student; a Qwen-image PercepT entry differs in the encoder too | agent default | 03:40 |

**Amendments**: none.

## 5. Assumptions and open questions

- Assumption: the extraction takes about 2¼ h (throughputs of the ArtELingo stores and r13's Qwen extraction). Not
  checked.
- Assumption: a full-scale c* run takes about 4 to 5 min on an A6000, as on the 3090 (3.5 to 5.3 min). Not checked.
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
  LEVEL or WIN gives its claim; BELOW gives none and keeps its row. A failed stage 0 gate leaves that part without a
  verdict.
- **Part c.** Holds when SC7 to SC10 WIN; qualified when SC7 and SC8 WIN and one of SC9 and SC10 WINs while the other
  is LEVEL; fails otherwise (D4).
- **Selection rules.** Stage 1: read-outs by r11's rule (ArtELingo) and r12's rule (Affection), each X model with its
  own ranges. Stage 2: the teacher wording per task by r12's wording rule, then read-outs by r12's rule, with r_E.
  Part b selects nothing.
- **Licensed claim (part a, paintings holds):** "On 12,281 held ArtELingo paintings, c*'s recipe with the <X> image
  as its only input at test (Qwen3-VL-Embedding-2B's instructed graphs, captions and teacher at training) clusters and
  retrieves paintings by emotion, style and genre better than the plain <X> image, and clusters them better than
  PercepT on the <X> image (AMI with 20 clusters; mAP with read-outs picked on validation; 5 seeds). <X> is one of
  three encoders that passed an earlier held emotion test." Qualified: "… better than the plain <X> image; against
  PercepT on the <X> image, better in <aspects>." Photos: "On 8,501 held Affection photos, the same model on <X>,
  fine-tuned on Affection's unlabeled train photos with the settings picked for Qwen, clusters and retrieves photos by
  emotion better than the plain <X> image." SC6: "<model on X> is level with or better than the Qwen image instructed
  for <aspect> in <task> on <data>."
- **Licensed claim (part c, holds):** "On <n> EmoSet-118K test photos (photos with a near-copy in Affection's train
  set removed), the photo model, trained only on Affection and frozen, clusters and retrieves photos by emotion better
  than the plain Qwen3-VL-Embedding-2B image and better than the Qwen image instructed for emotion (AMI with 20
  clusters; mAP; wording and read-outs picked on EmoSet validation by the rule fixed in r12; 5 seeds)." Qualified:
  "… better than the instructed image in <task> and level with it in <other task>."
- **Not licensed:** encoders not tested; training without Qwen views; other photo sets; EmoSet's other attributes;
  any setting from part b as better than c*'s or the photo model's.

## 7. Constitution check

| Principle | Pass or exception | Note |
|-----------|-------------------|------|
| C1 strongest comparator | pass | part a: the frozen image and matched arm as bars, the teacher and the strongest PercepT entry beside (D16); part c: the teacher in the wording EmoSet picks |
| C6 sample IDs | pass | extraction keys; EmoSet identifiers from its split files; stage 0(ii) |
| C7 rule before code | pass | §6 with §3.4 and §3.5, reviewed by a fresh Opus `xhigh` worker before the approving commit |
| C8 prior and disclosure | pass | prior §3.1; disclosure §8 |
| C12 regression check | pass | stage 0(i), (iv) to (vii) |
| C13 re-derivation | pass | Codex, §3.4 |
| `held-data.md` | pass | stopping, ranges and picks on validation; held scored only; B1 and B3 read validation only; B2 reads held and selects nothing |

Exceptions: none.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|---|---|---|
| EmoSet photos near-copy Affection train photos | the near-copy count and histogram | they leave EmoSet's sets (cut 0.85); the report gives the count |
| EmoSet's labels differ in kind from Affection's (one label per image, other annotators) | class counts, per-class AP | comparisons stay within EmoSet; reported |
| A new encoder's run fails or diverges | the rerun rule (§3.4) | that encoder's part stops; the others go on |
| Device differences (3090 against A6000) bias part b | 0(viii); B1 and B3 use node402 references | reported; B2 stays on one machine |
| Part b shows a better setting than c*'s or the photo model's | B1 and B3 sheets | reported as analysis; nothing changes (D6) |
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
  2026-11-04 18:06; later node work waits for the user's re-reservation).
- Budget: local GPU about 2¼ h, heavy, under the lock, in the day (D8): Affection SigLIP 2 and SigLIP 1 (85,007
  images each) and CLIP L/14 train (72,255), about 1 h; EmoSet validation and test (23,621) in three Qwen views and
  three other encoders, about 1¼ h. node402, 3 A6000: part a about 1.7 GPU-h (30 runs), B1 about 9 GPU-h (120 runs),
  B3 about 1.7 GPU-h (45 runs), together about 4 to 5 h. CPU about 4 h at 8 threads (scoring, bootstraps, C13).
  Disk: about 2.5 GB of EmoSet images on SSD, about 1 GB of new stores and about 8 GB of run outputs on SSD2.

Body length: about 230 lines against a claim-test target of about 150 plus the rule: two claim tests and an ablation
share one spec (D8).
