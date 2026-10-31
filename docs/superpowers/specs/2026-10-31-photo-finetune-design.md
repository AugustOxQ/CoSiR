# c* fine-tuned on Affection's unlabeled train photos, tested against its teacher on held photos: design

> Date: 2026-10-31 15:40 · Status: draft · Kind: experiment
> Builds on: [report 12-31](../../reports/auto/percept/2026-12-31_affection_confirmation.md) (frozen c* on Affection:
> above the plain image, below its teacher; fails); the grill in the r13 decide chat (2026-10-31, two rounds, ending
> about 15:20; its answers are §4) · Constitution: [constitution.md](../constitution.md)
> Decision rule: inline in §6, with §3.4 and §3.5 · Inherits: [r12 spec](2026-10-30-affection-confirmation-design.md)
> (Affection labels, keys, clustering and retrieval measurement, read-out menu and its selection rule, WIN / LEVEL /
> BELOW) and c*'s recipe ([report 12-28](../../reports/auto/percept/2026-12-28_recipe_sweep.md), arm k11=40)
> Reviewer: not yet (a fresh Opus `xhigh` worker reviews the rule before the commit that approves it)

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** Frozen c* beats the plain Qwen image on Affection photos but falls below its teacher, the Qwen image
instructed for emotion (AMI −0.0282, mAP −0.0190; report 12-31). You do not accept a student below its teacher. This
loop fine-tunes c* on Affection's 72,255 train photos, without labels, and tests it again on the 8,501 held photos.

**Plan** (build and run unattended; you read the report).
1. Stage 0 (checks): extract the instructed emotion views of the train photos and of their explanations. The new code
   reproduces earlier numbers: r12's verdict, c*'s Affection outputs and c*'s ArtELingo graphs. Train photos with a
   near-copy in validation or test leave the training set.
2. Stage 1 (sweep, validation, seed 0): 9 rows. c*'s settings; one change at a time (learning rate, graph neighbours,
   the content correction, the teacher share); one row from scratch; the best combination. The pick has the best
   weaker gap to the teacher (AMI or mAP), counted in c*'s r12 seed ranges.
3. Stage 2 (validation): the pick trains on seeds 0 to 4. Its seed ranges and read-out are measured and committed
   before any test number.
4. Stage 3 (claim test, Affection test): r12's rule against the plain image and the teacher, and the CRL bar.

**Compared against.** The plain Qwen image and the teacher (artwork wording, r12's pick), from r12's outputs; CRL
(CLIP L/14, painting wording, r12's pick). Beside them, without a bar: frozen c* from r12, which differs only by the
fine-tuning.

**Outcomes.** Holds: a WIN over the plain image in clustering and retrieval, and LEVEL or WIN against the teacher in
both. Qualified: the teacher LEVEL or WIN in one task only. Fails: anything else. LEVEL uses the new model's own seed
range. The CRL bar is its own claim. A failure is reported as it is; no second sweep row goes to the test. We expect a
WIN over the plain image and a much smaller gap to the teacher: holds or qualified are likely, fails is possible.

**Cost.** About 1.5 h of heavy GPU for the extraction, then about 1 h of light GPU for training, all local; CPU about
2 h. Build tonight; run on Sunday 2026-11-01 (no working-hours limit); the report about Monday 11-02, at the latest
2026-11-03 23:00.

**Check these.**
1. The from-scratch row: your pick after seeing our recommendation.
2. The 9-row grid: our proposal, approved by you.
3. Train photos with a near-copy in validation or test (cosine ≥ 0.98 on the default Qwen image) leave training
   (agent default). Visual Genome holds many COCO images, so copies across Affection's splits are possible.
4. Stage 3 is a claim test, not a confirmation (agent default): a confirmation cannot rescue r12's failed one, and
   the result is meant for the paper.
5. The decision to train on photos came after r12's test result; the report says so beside the claim.

## 1. Context and goal

Asked: "I think above plain is good but lower than teacher is not quite okey. So I will do 1." (option 1 of report
12-31: train c* on photos too), then "a. now" (before the 2026-11-10 freeze) (user, 2026-10-31).

Frozen c* (r10's arm k11=40, heads trained only on ArtELingo paintings) keeps 54.9% of the teacher's AMI gain over
the plain image on Affection test and 61.5% of its mAP gain (report 12-31). On ArtELingo, trained c* WINs over the
teacher in emotion clustering (+0.0145 AMI) and is LEVEL in retrieval (+0.0007 mAP). c* stays the model for
paintings; nothing on ArtELingo reruns.

Goal: fine-tune c*'s emotion pathway on Affection's unlabeled train photos and test it on held Affection photos.
Question this step answers: does a c* that has trained on photos reach its teacher on photos?

## 2. Scope

In: Affection train (training, no labels), validation (stopping, sweep, seed ranges, picks) and test (scored);
emotion only; the instructed emotion views of train photos and their explanations; the auto report and a user-read
report.
Out: one model for paintings and photos (option A1, not chosen); ArtELingo reruns; style and genre on photos; PolyArt
(the user's link); new CRL versions or teacher wordings; PercepT fitted on Affection; the large sweep (r14, after
the freeze).

## 3. Design

### 3.1 Hypothesis and prior

Claim tested: c* that trains on photos is LEVEL with or above its teacher on held photos, in clustering and in
retrieval. Mechanism: frozen c*'s emotion head learned its map from the default image toward the teacher's emotion
view on paintings only; with the same graphs and teacher term built on photos, the head can learn that map on photos.
Starting point on Affection validation (r12, seed 0 to 4): frozen c* AMI 0.1427 to 0.1461 against the teacher's
0.1719 (about −8 r_clu); mAP 0.2858 (centering plus 10-neighbour query expansion, cross-fit) against 0.3044 (about
−18 r_mAP). Prior, before any number: a WIN over the plain image in both tasks (high confidence); against the
teacher, LEVEL or BELOW in each task, with a much smaller gap than r12's; holds or qualified more likely than fails.
The from-scratch row about as good as fine-tuning (72,255 photos is more than c*'s 42,969 paintings). Surprise: the
best row below frozen c* on validation.

### 3.2 Arms and controls

| Arm | What it is | Role |
|---|---|---|
| photo model | the stage 1 pick: c*'s emotion pathway (image and caption trunks and the emotion head, from c*'s emotion checkpoint of seed s), trained further on Affection train; or, if row 8 is picked, the same pathway from random weights; seeds 0 to 4 | candidate |
| plain Qwen | r12's arm and outputs, its picked read-out | bar (SC1, SC2) |
| teacher | r12's arm, artwork wording (r12's pick for both tasks), its picked read-out | bar (SC3, SC4) |
| CRL | r12's Affection pick (CLIP L/14, painting wording), its picked read-out | bar (SC5) |
| frozen c* | r12's arm and outputs; differs from the photo model only by the fine-tuning | descriptive (matched control) |
| PercepT core, CLIP L/14 | r12's rows | descriptive |

The other arms reuse r12's exports, picks and per-query outputs unchanged; only the photo model is new.

### 3.3 Data roles

| Data | Role | Stages |
|---|---|---|
| Affection train, 72,255 photos, minus near-copies of validation or test photos | training, no labels read | 0a, 1, 2 |
| Affection validation, 4,251 photos | development: early stopping, sweep scores, seed ranges, read-out pick | 1, 2 |
| Affection test, 8,501 photos | held: scored only | 3 |
| ArtELingo | stage 0a's replays only | 0a |

Keys, label rule, evaluation order and ties are r12's (§3.3, §3.4). Train labels are never read. Near-copy rule: a
train photo whose default Qwen image has cosine ≥ 0.98 with any validation or test photo's default Qwen image leaves
the training set; the count, per source, and the 0.95 to 1.0 cosine histogram of the nearest pairs are reported.

### 3.4 Measurement

- **Training (c*'s recipe for the emotion head).** Fit nodes: the training set above. Graphs on the fit nodes and,
  separately, on the validation nodes (for the stopping loss), built as c*'s emotion graph: mutual top-k cosine
  neighbours of the instructed emotion explanation mean (E), minus the per-anchor content share λ times content (the
  mean of default explanation mean and default image similarity), all standardized as in c*'s builder; with the
  correction off, the mutual top-k of E alone. Loss: the four buddy terms (weight 1) plus the teacher term toward the
  instructed emotion image (weight 0.25, teacher share 1/5; row 7: weight 1, share 1/2). AdamW, weight decay 0.05,
  batch 512 graph pairs, at most 200 epochs, patience 15 on the validation loss; the checkpoint at the best epoch.
  Temperatures, dropout and everything else as c*'s emotion head. Instruction wording: the artwork wording (c*'s own,
  r12's teacher pick) for both images and explanations.
- **Sweep scores (stage 1).** On all 4,251 validation photos, seed 0: emotion AMI by r12's clustering protocol (20
  clusters, three states, mean) and cross-fit emotion mAP with the fixed read-out centering plus k = 10 query
  expansion (r12's procedure). Gaps: g_clu = (AMI − 0.1719…) / r_clu and g_map = (mAP − 0.3044…) / r_mAP, with the
  teacher's r12 validation values unrounded and r12's c* ranges r_clu = 0.003430790752866536 and
  r_mAP = 0.0010103233538674705. Score = min(g_clu, g_map).
- **Seed ranges (stage 2).** r12's stage 0b procedure for the photo model over seeds 0 to 4: r'_clu, r'_mAP,
  r'_nDCG. They serve the LEVEL bars of SC3 and SC4 and the read-out pick; r12's ranges are reported beside them.
- **Test scoring (stage 3).** r12's stage 2 and 3 protocol unchanged: the same labels, partitions, gallery, read-out
  menu, paired bootstraps (clustering 2,000 draws, images then seeds; retrieval 10,000 joint draws over queries and
  seeds), comparators' outputs from r12.
- **Outcomes.** WIN when the lower 95% bound > 0; LEVEL when not WIN and the lower bound > −r' (the photo model's
  range); BELOW otherwise.
- **C13.** Codex re-derives the near-copy count, the sweep scores and pick, the seed ranges, the read-out pick, the SC
  values, intervals and outcomes from this rule and the data, within 1e-6, before the verdict is written.

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|---|---|---|---|---|
| 0a | regression check | new code paths, new views | (i) to (v) below | stop, trace, note it |
| 1 | sweep | train, validation, seed 0 | the sheet of rows 1 to 9 and the pick written to one file and committed before any stage 2 run | report |
| 2 | measurement | validation | seeds 0 to 4 trained; r', the read-out pick committed before any test number | stop, note it |
| 3 | claim test | Affection test, everything frozen | SC1 to SC5 | report |
| 4 | descriptive | validation, test | §3.6 | report |

Stage 0a:
- (i) Views: the instructed extractor, run on the first 200 validation keys (sorted, FI included), reproduces r12's
  `emotion_focus` rows by key with cosine ≥ 0.999; the instructed text path reproduces the first 200 rows of
  ArtELingo's stored instructed emotion caption store with cosine ≥ 0.999. The train views have one row per train
  key (images) and per train explanation, all finite and unit norm within 1e-3; keys join to the index (C6).
- (ii) r12's replay: the new scoring path, fed r12's exports, gives r12's SC1 to SC4 and SC8 values and intervals
  within 1e-9.
- (iii) Inference: the photo model's code, given c*'s emotion checkpoints and no training step, gives r12's c*
  Affection validation and test exports within 1e-3 (r12's bar), all five seeds.
- (iv) Graphs: the generalized graph builder, given c*'s ArtELingo full-scale inputs, reproduces c*'s saved ArtELingo
  emotion graph edges exactly.
- (v) Near-copies: the count and histogram written; no gate on the number.

Sweep grid (seed 0; changed setting in bold; learning rate 0.001 unless stated):

| Row | Start | Learning rate | k | Content correction | Teacher weight |
|---|---|---|---|---|---|
| 1 | c* | 0.001 | 40 | on | 0.25 |
| 2, 3 | c* | **0.0003**, **0.0001** | 40 | on | 0.25 |
| 4, 5 | c* | 0.001 | **20**, **80** | on | 0.25 |
| 6 | c* | 0.001 | 40 | **off** | 0.25 |
| 7 | c* | 0.001 | 40 | on | **1** |
| 8 | **random** | 0.001 | 40 | on | 0.25 |
| 9 | c* | for each setting, the value of the best-scoring row among 1 to 7 that changes it (row 1's value if none beats row 1) | | | |

Row 9 is skipped when it equals an earlier row.

### 3.6 Diagnostics

After the verdict, deciding nothing: R0 (plain cosine) rows; graded emotion nDCG; the verdict with tied images
dropped; per-source and per-class AP; the photo model minus frozen c*, paired, both tasks; the share of the teacher's
gain over the plain image (as report 12-31); r' beside r12's ranges and the held per-seed spread; the full sweep sheet
with row 8 against row 1; per row, the share of validation graph edges whose two photos share the plurality emotion.

### 3.7 Frozen vs rerun

Frozen: c*'s weights (the start point); r12's exports, picks and per-query outputs of every other arm; the sweep's
pick, r' and the read-out pick before test scoring. Rerun: the train views (new), graphs, training, the photo
model's exports and scoring.

## 4. Decisions

Grill rows: the r13 decide chat, 2026-10-31, two rounds.

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|----|----------|----------|----------------------------|------------|------|
| D1 | Which model | fine-tune c* on Affection train (A3); c* stays the model for paintings | "in between A1 and A2" (A1 one joint model; A2 from scratch only) | user | grill 1 |
| D2 | From-scratch row | row 8 in the sweep, so validation decides between the two start points | training takes minutes (fine-tuning only) | user, after seeing the recommendation | grill 2 |
| D3 | The bar | r12's rule unchanged (B1) | the same as r12, so directly comparable (WIN in one task, B2) | user | grill 1 |
| D4 | The recipe | a sweep on Affection validation (C2) | "the sweep depends dataset-wise" (c*'s settings unchanged, C1) | user | grill 1 |
| D5 | Pick rule | the best weaker teacher gap, in c*'s r12 seed ranges (D2) | aims at the bar (AMI only, D1; mean gain over plain, D3) | user | grill 2 |
| D6 | The grid | §3.5's nine rows | one change at a time, then the best combination (a full grid, 72 rows) | user, after seeing the recommendation | grill 2 |
| D7 | After a failure | reported as it is; no second sweep row goes to the test; the next step is decided then | selection on held data | user (scenario check iii) | grill 2 |
| D8 | Minor batch | artwork wording; extraction on the local GPU this weekend; c*'s training settings, stopping on Affection validation; r12's protocol; the CRL WIN bar again; c* and report 12-31 stay the fallback | the grill's minor batch | agent default (accepted by the user) | grill 1 |
| D9 | Ties in the pick | scores within 0.5 (in r units) of the best go to the row with the fewest changed settings, then the lowest row number | prefer c*'s own settings | agent default (shown to the user in the D2 scenario) | grill 2 |
| D10 | Seeds | 0 to 4; seed s starts from c*'s seed s; the sweep's seed-0 run of the pick is seed 0 | as c* and r12 | agent default | 15:40 |
| D11 | What trains | the emotion pathway: both trunks and the emotion head of c*'s emotion checkpoint; style and genre heads unused | the photo model serves emotion only | agent default | 15:40 |
| D12 | Near-copies | train photos with cosine ≥ 0.98 to a validation or test photo leave training | Visual Genome and COCO share images; keeping them trains on held photos' copies and explanations | agent default | 15:40 |
| D13 | Sweep read-out | fixed: centering plus k = 10 query expansion | every main arm's r12 pick (a menu per row: slower, more selection) | agent default | grill 2 |
| D14 | Stage types | §3.5 | stage 3 feeds a paper claim and cannot be a confirmation of r12's failed one | agent default | 15:40 |

**Amendments**: none.

## 5. Assumptions and open questions

- Assumption: the instructed text extraction runs at about 310 texts/s (inferred from r12's FI step: 1,553 s for
  21,808 images and 134,418 texts). Not checked.
- Assumption: one sweep row trains in under about 5 minutes on the local GPU (c*: 212 to 317 s per seed for three
  heads on 42,969 paintings). Not checked.
- Assumption: Affection's train explanations describe the same photos as its images (keys join). Checked in r12 for
  validation and test; stage 0a(i) checks train.
- Open questions: none.

## 6. Success criteria

**Decision mode.** Binding rule: this section with §3.4 and §3.5, committed with this spec before any code.

| ID | What is measured | Compared against | Bar | On which data |
|----|------------------|------------------|-----|---------------|
| SC1 | photo model's emotion AMI | plain Qwen | WIN | Affection test |
| SC2 | photo model's emotion mAP, picked read-out | plain Qwen, its picked read-out | WIN | Affection test |
| SC3 | photo model's emotion AMI | the teacher (artwork) | LEVEL or WIN (r'_clu) | Affection test |
| SC4 | photo model's emotion mAP, picked read-out | the teacher, its picked read-out | LEVEL or WIN (r'_mAP) | Affection test |
| SC5 | photo model's emotion mAP, picked read-out | CRL (CLIP L/14, painting), its picked read-out | WIN | Affection test |

- **Verdict (stage 3):** holds when SC1 to SC4 pass; qualified when SC1 and SC2 pass and exactly one of SC3 and SC4
  does; fails otherwise. A failed stage 0a leaves no verdict. SC5 is its own claim; a failed one gives no claim and
  keeps its row.
- **Selection rule (stage 1):** the highest score of §3.4 among the rows run; ties as D9. Row 9's settings by §3.5.
- **Selection rule (stage 2):** the photo model's read-out by r12's Affection read-out rule (r11's menu, cross-fit,
  (mAP − mAP(R0)) / r'_mAP averaged over seeds).
- **Licensed claim (holds):** "On 8,501 held Affection photos, c* fine-tuned without labels on Affection's train
  photos [or: c*'s recipe trained from scratch on them], with its settings picked on Affection validation, clusters
  and retrieves photos by emotion better than the plain Qwen3-VL-Embedding-2B image, and level with or better than the
  Qwen image instructed for emotion (AMI with 20 clusters; mAP with read-outs picked on validation; 5 seeds)."
  Qualified: "… and level with or better than the instructed image in <task>; in <other task> it is BELOW." Not
  licensed: one model for paintings and photos; transfer without training on photos; style or genre on photos; other
  photo datasets.
- **Licensed claim (SC5):** "The photo model retrieves photos by emotion better than our re-implementation of CRL
  (CLIP L/14, picked on validation) on held Affection."

## 7. Constitution check

| Principle | Pass or exception | Note |
|-----------|-------------------|------|
| C1 strongest comparator | pass | the teacher in r12's picked wording; CRL in r12's picked version; frozen c* as the matched control |
| C6 sample IDs | pass | r12's keys; train views join by key; stage 0a(i) |
| C7 rule before code | pass once reviewed | a fresh Opus `xhigh` worker reviews §3.4, §3.5 and §6 before the approving commit |
| C8 prior and disclosure | pass | prior §3.1; disclosure §8 |
| C12 regression check | pass | stage 0a(ii) to (iv) |
| C13 re-derivation | pass | Codex, §3.4 |
| `held-data.md` | pass | stopping, sweep, ranges and picks on validation; test scored only; the direction came after r12's test result (disclosed in §8) |

Exceptions: none.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|---|---|---|
| Training on photos does not close the gap | SC3, SC4 | report "qualified" or "fails" (D7) |
| Copies of test photos in train inflate the test | the near-copy count | they leave training (D12); the report gives the count |
| A wider seed range r' makes LEVEL easier | r' beside r12's ranges | reported; r' not changed after results (the user accepted this, grill scenario ii) |
| One source (FI) carries the result | per-source AP | reported |
| 22.6% of test labels are ties | the ties-dropped diagnostic | reported |
| Selection disclosure: on Affection validation, r12 tried 2 wordings, up to 25 read-outs per arm and 4 CRL versions; this loop adds 9 sweep rows and up to 25 read-outs for the photo model. The decision to train on photos followed r12's test result | validation numbers inflated | only test numbers are results; every pick is frozen before test scoring; the report states the history beside the claim |

## 9. Deliverables and review

- At the end: the round folder `src/test/20261031_photo_finetune/` (module prefix `pf_`) with code, tests, log and
  per-query outputs; the train views under `/data/SSD2/pre_extract/affection_local/`; run outputs under
  `/data/SSD2/cosir_r13/20261031_photo_finetune/outputs/`; the auto report
  `docs/reports/auto/percept/2027-01-01_photo_finetune.md` with its results brief and index row; a user-read report.
- Reviews: this rule by a fresh Opus `xhigh` worker before the approving commit; tickets per `matt-chain.md`, with the
  near-copy rule, the graph builder, the sweep score and pick, and the outcome code as verdict code; the C13
  re-derivation by Codex; the final whole-branch review.
- Stop points: none between approval and the reports. Time-box: the report by 2026-11-03 23:00.
- Budget: heavy GPU about 1.5 h, local, under the lock, on Sunday 2026-11-01 (a weekend day): 72,255 instructed images
  at the measured 19.4 images/s (about 62 min) and about 448,000 instructed explanations (about 25 min, not checked).
  Light GPU about 1 h: the graphs, 9 sweep rows and 4 more seeds. CPU about 2 h at 8 threads (the protocol and its
  checks). No DAS6. Disk: about 0.3 GB of images and about 1.8 GB of explanation vectors (float16) on SSD2.
