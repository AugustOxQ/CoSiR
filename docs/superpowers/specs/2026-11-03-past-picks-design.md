# c* past our picks: three more encoders, SemArt's new artists and the cost table: design

> Date: 2026-11-03 19:55 · Status: draft · Kind: experiment
> Builds on: decision A of 2026-11-03 18:30 (the [experiment overview](../../user_read/2026-11-03_experiment_overview.md)'s
> plan, loop r19: items 16, 6 and 19) and the r19 grill (r19 decide chat, 2026-11-03: round 1 about 18:55, round 2
> about 19:10, closing summary confirmed about 19:20; its answers are §4) · Constitution:
> [constitution.md](../constitution.md) · Decision rule: inline in §6, with §3.4 and §3.5
> Inherits: r14 part a ([2026-11-02-backbones-sweep-design.md](2026-11-02-backbones-sweep-design.md): §3.2 to §3.5,
> §6, D16, D18, D19) for ArtELingo and Affection; r15 ([2026-11-02-mmart-test-design.md](2026-11-02-mmart-test-design.md):
> §3.2 to §3.5, §6, D9, D10, D12, D16 to D18) for MMArt and for PercepT's mappers applied frozen; through them r10 to
> r13 · Facts: the [facts brief](../../../src/test/20261103_past_picks/pp_facts_brief.md) (Codex, with Claude's checked
> additions); the [SemArt check](../../../src/test/20261103_semart_check/20261103_semart_check_log.md)
> Reviewer: (pending: a fresh Opus `xhigh` worker before the approving commit)

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** Reviewers will ask three things that our picks leave open. Does c* hold on the encoders that failed
or were unclear with an older model (DINOv2, CLIP B/32, CLIP B/16)? Does it hold on artists it never saw? Is its one
encoder pass really cheaper than the teacher's one pass per aspect? This loop tests the current c*, with no setting
changed, on each question.

**Plan** (build and run unattended; you read the report).
1. Stage 0 (checks): the new feature stores pass their checks; r14's trainer and the scoring code reproduce r14's and
   r15's CLIP L/14 numbers; PercepT's saved mappers reproduce their outputs; the SemArt set is recomputed.
2. Stage 1 (sweep, development only): c* and the photo model on the three new encoders, five seeds each, on DAS6;
   PercepT on DINOv2 at full scale; seed ranges and read-outs from validation. Committed before any test number.
3. Stage 2 (claim test): the three new encoders on ArtELingo held, Affection test photos and MMArt test; all seven
   encoders on SemArt's 11,162 paintings by new artists (genre only). Everything frozen.
4. Stage 3 (claim test): the cost probe, 2,000 MMArt paintings on the local GPU.
5. Stage 4 (descriptive, after the verdicts): SemArt without its 217 closest paintings, r17's new metrics, and others.

**Compared against.** Per encoder and test set: its plain image (WIN in every cell) and PercepT on that image (WIN in
the clustering cells), as in r14 and r15. The Qwen image instructed per aspect (the teacher), cell by cell. Cost: the
teacher's three instructed passes.

**Outcomes.** Each encoder gets one verdict per test set; one set never makes up for another. ArtELingo and MMArt:
holds when all nine cells WIN; qualified when only PercepT cells miss; fails otherwise. Photos: holds when both cells
WIN. SemArt: holds when its three cells WIN; qualified when only the PercepT cell misses. Cost: holds when c*'s
slowest of three timings beats the teacher's fastest. Every result goes in the paper's encoder table; a fail or a
qualified result gets a limit sentence that names the encoder and cells. A teacher cell at LEVEL or WIN licenses
"level with or better than". We expect WINs over the plain image almost everywhere, a PercepT miss most likely on
DINOv2, and the cost claim on all seven encoders.

**Cost.** DAS6 node402 about 3½ GPU h (about 1¼ h on three GPUs, beside r18). Local GPU: about 2 h of extraction,
heavy (evening, night or weekend only), and a probe of about 45 min. CPU about 3 to 4 h. Build tonight after r17
merges; report about 11-06, at the latest 11-07 23:00.

**Check these.**
1. All three items are claim tests (agent default), since each feeds a paper claim. The loop stays medium.
2. The new encoders' teacher cells license only "level with or better than", also on MMArt and SemArt (agent
   default): they have no earlier result against the teacher, and r15 needed one for "better than".
3. SemArt's seed ranges and read-outs come from ArtELingo validation (your m2), so they use ArtELingo's genre labels:
   ArtELingo paintings have no SemArt type label.
4. The cost claim uses model time, with images decoded beforehand; time with decoding is measured once, as context
   (agent default). The probe takes about 45 min, not the 30 min of the grill.
5. Your picks after seeing our recommendation: the plan (decision A), the minors m1 to m7 and the four scenario
   checks. Most likely wrong: the extraction time (not measured for these encoders) and SemArt's artist-name misses
   (at most a few hundred paintings, not checked).

## 1. Context and goal

Asked: "we update that with the current model" (the user on item 16, decision A, 2026-11-03 18:30); in the grill:
"1a is the current setting and (possibly) the least hard one", "they are not expensive, though we might expect some
encoders to fail already" (SemArt on all seven encoders), "the direct answer to the question, plus it's not expensive"
(the three test sets).

c* holds on Qwen, SigLIP 2, SigLIP 1 and CLIP L/14 on ArtELingo held (r14) and on MMArt (r15). DINOv2, CLIP B/32 and
CLIP B/16 were tested only with the older model R6-5, on another bar (emotion AMI at least 0.0100 above PercepT core's
strongest entry; 10-25 and 10-26 rounds, reports 12-24 and 12-25): B/32 +0.0011 and B/16 +0.0070, both inconclusive;
DINOv2 −0.0041 at the sample scale, NO-GO. SemArt sits in the paper's related work (`paper_v1.md` line 128); the
SemArt check found 11,162 of its paintings by 2,717 artists with no painting in ArtELingo's train set. The paper says
c* needs one encoder pass where the teacher needs one per aspect (§3.2, §3.6); no time was measured.

Goal: test whether the current c*, unchanged, holds on three more encoders and on new artists, and costs less than its
teacher at test.
Question this step answers: per encoder and test set, does c* beat its plain image and PercepT, where does it match
its teacher, and is it faster than the teacher?

## 2. Scope

In: item 16: X-c* and the X photo model for X in DINOv2, CLIP B/32 and CLIP B/16 (seeds 0 to 4), on ArtELingo held,
Affection test and MMArt test; DINOv2's matched arm at full scale. Item 6: frozen c* of all seven encoders on SemArt's
new-artist set, genre (SemArt's type) only, clustering and retrieval. Item 19: the cost probe and the cost table. The
feature stores these need; the auto report with its paper section; a user-read report.
Out: any change of c*'s or the photo model's settings (D2); EmoSet for the new encoders (D7); emotion or style on
SemArt (no labels); SemArt's school and period labels and its text side; PercepT refit on SemArt; an artist-disjoint
ArtELingo split; claims from r17's metrics (context only, D13); training cost.

## 3. Design

### 3.1 Hypothesis and prior

- **Item 16.** Claim checked: r14's and r15's claims, with three encoders our earlier picks left out. Mechanism (r14
  §3.1): the gain comes from Qwen's instructed graphs and teacher at training; the input encoder only supplies the
  image. On ArtELingo development, R6-5 beat both its frozen image and its own PercepT arm on emotion on all three
  (student / PercepT arm / frozen image: B/32 0.0905 / 0.0708 / 0.0624; B/16 0.0952 / 0.0728 / 0.0643; DINOv2
  0.0839 / 0.0722 / 0.0587), and c* is stronger than R6-5 on Qwen. Prior per encoder (B/32, B/16, DINOv2): SC1 and
  SC2 WIN in all six cells 0.8 each; SC3 WIN in all three 0.6, 0.65, 0.45; paintings holds 0.55, 0.6, 0.4, qualified
  0.3, 0.25, 0.35, fails 0.15, 0.15, 0.25. Photos holds 0.8 each. MMArt as ArtELingo, 0.05 lower. Teacher: LEVEL or
  WIN in emotion and style clustering 0.6, BELOW in genre clustering 0.6.
- **Item 6.** Claim tested: c*'s genre gains carry to artists absent from its training set. Mechanism: the genre head
  maps a painting's look and subject, which SemArt's types also describe. Prior: SC11 and SC12 WIN 0.85 per encoder
  for the four with r15 records, 0.75 for the three new; SC13 WIN 0.7 and 0.55; holds about 0.6 and 0.45. Teacher
  genre mAP LEVEL or WIN 0.5.
- **Item 19.** Claim tested: c*'s test-time cost (one encoder pass and three small heads) is below the teacher's
  (three Qwen passes). Prior: holds on all seven 0.95; one Qwen pass takes about 86 ms per image end to end (MMArt
  prep log), the slowest other encoder about 19 ms.
- Surprise: a miss against the plain image on any cell, on any set.

### 3.2 Arms and controls

Item 16 inherits r14 part a's and r15's arms (r14 §3.2, r15 §3.2) with X in {DINOv2 (first layer width 1024), CLIP
B/32 (512), CLIP B/16 (512)}. Changes:

| Arm | What it is | Role |
|---|---|---|
| X-c* | r14's X-c*, X's default image as the only input; seeds 0 to 4; trained on node402 | candidate (SC1 to SC3, SC6 to SC14) |
| X photo model | r14's X photo model (r13's picked settings) from X-c*'s emotion checkpoint of seed s, training seed 100 + s | candidate (SC4 to SC6) |
| X frozen image | X's default image, no training | bar (SC1, SC2, SC4, SC5, SC7, SC8, SC11, SC12) |
| X matched arm | B/32: M3 (10-25 round), B/16: M6 (10-26 round), full scale, seeds 0 to 4, their stored held outputs, partitions and mappers; DINOv2: M5 rerun at full scale with 10-25's Stage 2 settings and the claim test's saved Qwen Stage 1 topics, seeds 0 to 4, on node402 (D12). On MMArt and SemArt applied frozen with r15's reading | bar (SC3, SC9, SC13) |
| teacher | Qwen instructed per aspect: r10's and r11's outputs (ArtELingo), r12's artwork wording (Affection), r15's outputs and pick (MMArt), `genre_word` on SemArt (new extraction) | own claims (SC6, SC10, SC14) |

Item 6 (SemArt), all seven encoders (Qwen, SigLIP 2, SigLIP 1, CLIP L/14, DINOv2, CLIP B/32, CLIP B/16), every arm
frozen, genre output only (D18): X-c* (Qwen: r10's c*; SigLIP 2, SigLIP 1, CLIP L/14: r14's; the new three: this
round's), X frozen image, X matched arm (Qwen: the claim test's P3; SigLIP 2: M2; SigLIP 1: M8; CLIP L/14: M7; the
new three as above) read as r15 (own reading: argmax of the 40 topic probabilities; 20-cluster reading: k-means,
states 0 to 2, on the L2-normalized probabilities; the larger reading, r14 D16, taken again in each draw), teacher.

Item 19 (cost), per encoder X: c* on X is one X default pass plus three trunk-and-head networks (4,197,376 parameters
each, paper §3.6); the teacher is three Qwen instructed passes (`emotion_focus`, `style_word`, `genre_word`); the plain
X image (one pass) is context. Beside every set: the strongest PercepT entry per aspect (C1), descriptive.

### 3.3 Data roles

| Data | Role | Stages |
|---|---|---|
| ArtELingo train, 42,969 paintings | training (X-c*, DINOv2 M5) | 0, 1 |
| ArtELingo validation, 6,152 paintings | development: stopping, X-c* seed ranges, read-out picks; SemArt's LEVEL units and read-out maps (all seven) | 1 |
| ArtELingo held, 12,281 paintings (genre on 9,949) | held: SC1 to SC3, SC6 scored; stage 0 replays | 0, 2 |
| Affection train minus near-copies (69,927 photos, r13) | training, no labels (X photo models) | 1 |
| Affection validation, 4,251 photos | development: photo seed ranges and read-outs | 1 |
| Affection test, 8,501 photos | held: SC4 to SC6 scored | 2 |
| MMArt validation: the 5,769 ArtELingo validation paintings with MMArt labels | development: ranges and picks (r15 D4) | 1 |
| MMArt test, 16,384 paintings (r15's cut set) | held: SC7 to SC10 scored | 2 |
| SemArt new-artist set (below) | held: SC11 to SC14 scored; it selects nothing | 2 |
| SemArt's other 10,220 paintings | not touched (the SemArt check read their cosines and artists only) | none |
| 2,000 MMArt test paintings (the first 2,000 keys in sha256 order) | timing only; no label read | 3 |

**The SemArt set (D5, D19).** Recomputed from `sc_check.py`'s rule and `sc_nearcopy.parquet`: a painting is kept when
its default Qwen image has cosine < 0.95 to all 49,121 ArtELingo train and validation paintings, its author matches
none of the three name tiers against the artists of ArtELingo's 42,969 train paintings, and its author is not
"UNKNOWN MASTER". Gate: 11,162 paintings by 2,717 artists. Then, of the paintings sharing an image md5, keep the first
in evaluation order (r15 D10's form); the count is reported. Labels: SemArt's `TYPE`, all 10 classes (D10). Keys:
`image_file`, joined to every store through the store's own index (C6). Evaluation order (bootstrap rows, query and
gallery order with its ties, cross-fit halves): keys sorted by the sha256 of the key string (r15 D9). Pairs inside the
set at cosine ≥ 0.95 are counted, not removed. Populations: clustering and retrieval on every painting of the set
(each has a type). Read-out maps are fitted on the arm's outputs for all 6,152 ArtELingo validation paintings and
applied to SemArt; query expansion reads the SemArt gallery, with k fixed on validation.

### 3.4 Measurement

- **Inherited, unchanged.** r14 §3.4 for ArtELingo and Affection (training, clustering with 20 clusters and three
  states, SC3's `b_matched` statistic, retrieval, seed ranges, outcomes) and r15 §3.4 for MMArt (ranges and one pick
  per arm on MMArt validation; the teacher's pick reused from r15, D12 there). Outcomes: WIN when the lower 95% bound
  > 0; LEVEL when not WIN and the lower bound > −r (the candidate's own range); BELOW otherwise; per cell, no
  multiplicity correction.
- **Training.** r14's trainers imported unchanged. A non-finite loss or a crash reruns once with the same seed; a
  second failure makes that encoder's dependent verdicts inconclusive (r14 D18). DINOv2's M5: the matched-arm code
  that made M3, M6 and the sample M5 (10-25's Stage 2 settings), imported unchanged, at full scale; its held outputs, partitions, readings and mappers are saved as
  M3's and M6's are.
- **Bar arms' picks (D21).** The new frozen images pick by the original round's rule and ranges, as r14 D19 did for
  SigLIP 1: ArtELingo by r11's stage 4 rule with r11's r_a; Affection by r12's rule with r12's r_ret,mAP; MMArt with
  X's MMArt validation ranges (r15 D12).
- **SemArt (item 6).** Clustering as r15: k-means with 20 clusters, three states, mean AMI against the type labels;
  2,000 joint draws over paintings and seeds (frozen arms resample paintings only); SC13 with `b_matched` on the genre
  outputs. Retrieval: the genre output as query and gallery, relevance by type, mAP, 10,000 joint draws over queries
  and seeds. Read-out: each arm's existing ArtELingo pick (one per arm, r11's rule; D9, D20): Qwen c* and the teacher
  r11's; the Qwen, SigLIP 2 and CLIP L/14 frozen images r11's; the SigLIP 1 frozen image r14's; the SigLIP 2, SigLIP 1
  and CLIP L/14 X-c* r14's stage 1; the new encoders' X-c* and frozen images this round's stage 1. LEVEL unit (SC14):
  X-c*'s range over seeds 0 to 4 on ArtELingo validation, genre (AMI at 20 clusters; mAP at R0), from the five runs'
  validation exports with ArtELingo's genre labels.
- **Cost (item 19, D8, D22).** On the local 3090 under the GPU lock, with no other process on the GPU; batch 32 for
  every pass; the extractor's preprocessing, precision and settings; 3 warm-up batches not timed; CUDA-synchronized
  wall time over the 2,000 paintings. Two forms: model time (images decoded and resized beforehand and held in memory;
  timed: transfer, preprocessing to tensors, forward), three repeats; end to end (from the image file, decoding
  included, the extractor's loader with 8 workers; each image decoded once per arm), one repeat. The three
  trunk-and-head networks are timed on the stored default embeddings, three repeats. Per pass also: images per
  second, peak GPU memory, parameters. c*'s time per image = X pass + three networks; the teacher's = the sum of its
  three passes. Whole-set figure: the time to embed MMArt's 16,384 test paintings for three aspects, from the
  per-image rates, beside the MMArt prep log's measured Qwen times.
- **Read-out of claims (D13).** AMI at 20 clusters and mAP. r17's metrics (NMI, ARI, ACC, cluster counts), imported
  read-only from r17's module once it merges, are context (§3.6).
- **C13.** Codex re-derives, from this rule and the saved outputs (no retraining): the SemArt set, the seed ranges,
  the read-out picks, SC1 to SC14 with intervals and outcomes, SC15 from the probe's raw timings, and the verdicts,
  within 1e-6, before any verdict is written.

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|---|---|---|---|---|
| 0 | regression check | new stores, trainers, exports, matched arms, the SemArt set, scoring and pick paths, the probe | (i) to (viii) below | stop the affected path (table below), trace, note it |
| 1 | sweep (development only) | node402 training (15 X-c*, 15 X photo, 5 DINOv2 M5); ArtELingo, Affection and MMArt validation | per new encoder: its runs trained; its ranges and picks on each set written and committed before any test number of that encoder; SemArt's units and picks (all from ArtELingo validation) committed before any SemArt number | that encoder's dependent verdicts inconclusive; the others go on |
| 2 | claim test | ArtELingo held, Affection test, MMArt test, the SemArt set; all frozen | SC1 to SC14 | report |
| 3 | claim test | the cost probe | SC15 | report |
| 4 | descriptive | all | §3.6, after the verdicts | report |

Order on node402: stage 0 runs first, then the X-c* runs (one GPU per encoder), then the photo models (they need the
Affection stores), then DINOv2's M5. The first runs are timed and the rest estimated from them. Stage 0:
- (i) Stores. The extractor reproduces, by key, cosine ≥ 0.999: the first 100 ArtELingo rows of the stored DINOv2,
  CLIP B/32 and B/16 stores; the first 100 SemArt keys of the stored CLIP B/32 SemArt store; the first 100 Affection
  validation keys of the stored CLIP L/14 store; `genre_word` on the first 100 ArtELingo validation paintings against
  the stored `genre_word_s2img` view. MMArt: the prep queue's parity (200 shared paintings) for each new encoder.
  Every new store: one row per key, finite, unit norm within 1e-3, keys join (C6).
- (ii) Trainers. r14's trainers, imported unchanged, reproduce on node402 r14's CLIP L/14 c* seed 0 and its CLIP L/14
  photo model seed 0: best epoch and best development loss within 1e-4.
- (iii) Export. The export code, given r14's CLIP L/14 checkpoints, reproduces their stored ArtELingo held and
  Affection test exports within 1e-3, all five seeds.
- (iv) Scoring, within 1e-9: r14's CLIP L/14 SC1 to SC6 values and intervals; r15's CLIP L/14 SC1 to SC4 values and
  intervals; the SemArt path (genre only, maps fitted on ArtELingo validation), fed ArtELingo held genre inputs with
  r14's picks, gives r14's CLIP L/14 genre values of SC1 to SC3.
- (v) Matched arms. M3's and M6's stored outputs and partitions give their `reported_stage4` readings within 1e-9;
  their ten mappers on CPU give their stored held probabilities within 1e-5 (r15 (iv), with its fallback). DINOv2: the
  imported code reruns the 10-26 round's sample-scale M5 seed 0 on node402; its held readings (own and 20-cluster,
  each aspect) lie within the stored sample M5's range over seeds 0 to 4, the difference recorded (D23). The new full
  M5 outputs pass (i)'s store checks.
- (vi) SemArt set: 11,162 paintings by 2,717 artists, exactly.
- (vii) Read-out rule: fed r11's stored validation outputs, it reproduces r11's picks and scores within 1e-9.
- (viii) Probe: the Qwen default end-to-end time within 20% of the MMArt prep log's 86.3 ms per image.

Which failed gate voids what (D24); an inconclusive verdict is never "qualified":

| Failed gate | Voids |
|---|---|
| (i) X's new store on a set | X's verdict and teacher cells on that set |
| (i) the SemArt `genre_word` store or its parity | SC14, all encoders; SemArt verdicts stand |
| (ii) or (iii) | every verdict and teacher cell of the three new encoders |
| (iv) | every verdict and teacher cell except SC15 |
| (v) X's matched arm, after r15's mapper fallback | X's ArtELingo, MMArt and SemArt verdicts; X's other rows reported, its teacher claims stand |
| (vi) | every SemArt verdict and SC14 |
| (vii) | every mAP cell, so every verdict except SC15; AMI teacher claims stand |
| (viii) | SC15, until the cause is traced and fixed |

### 3.6 Diagnostics

After the verdicts, deciding nothing: R0 rows for every arm; clustering at the class count (SemArt 10; MMArt 9, 20,
10); r17's NMI, ARI, ACC and cluster-count values on the new cells (D13; skipped and noted if r17's module has not
merged); the SemArt cells without the paintings at nearest cosine ≥ 0.93 (217 before the md5 step; D14), with whether
any verdict would change; SemArt AP per type; SemArt pairs inside the set at ≥ 0.95, and md5 duplicates removed; R6-5's
held numbers on the three encoders (10-25 and 10-26 rounds) beside c*'s; each encoder's share of the teacher's gain over
its plain image; seed spreads on test beside the validation ranges; the strongest PercepT entry per set (C1); the cost
table's other rows (plain image pass, end-to-end rates, peak memory, parameters, passes per aspect, the whole-set time);
node402 time per run.

### 3.7 Frozen vs rerun

Frozen: c*'s recipe and r13's photo settings; r10's and r14's checkpoints and exports; the stored matched arms (topics,
mappers, outputs, partitions); r11's, r12's, r14's and r15's ranges and picks of every reused arm; r15's MMArt cut set
and teacher outputs; stage 1's ranges and picks before any test number. Rerun: the new stores, X-c* and X photo model
training and exports, DINOv2's M5 at full scale, SemArt and MMArt exports of every arm that needs one, all scoring, the
probe.

## 4. Decisions

Grill rows: the r19 decide chat, 2026-11-03 (decision A 18:30; round 1 about 18:55; round 2 about 19:10; closing
summary confirmed about 19:20). D16 to D25 are agent defaults of this spec (19:55).

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|----|----------|----------|----------------------------|------------|------|
| D1 | The loop | r19 = items 16, 6 and 19 of the overview's plan; medium loop, beside r17 and r18 | decision A: six loops, three decided tonight | user, after seeing the recommendation | 2026-11-03 18:30 |
| D2 | Which model | the current c* and photo model, no setting changed on any encoder | "we update that with the current model" (re-tune per encoder) | user | 2026-11-03 18:30 |
| D3 | Paper section | the report states shortly how much the results change the paper draft v1: related work, method, experiments | the user's own rule for every loop | user | 2026-11-03 18:33 |
| D4 | Bar (Q1, 1a) | r14's bar unchanged, per encoder; every result in the paper's encoder table whatever it is; a fail or qualified result gets a limit sentence naming the encoder and cells | "1a is the current setting and (possibly) the least hard one" (an extra margin for the new encoders only, unfair the other way) | user | 2026-11-03 18:55 |
| D5 | SemArt set (Q2, 2a) | the new-artist set at the 0.95 cut: 11,162 paintings by 2,717 artists, unknown masters out | the close-pair check found no big issue (a 0.90 cut drops real paintings of common subjects and skews the types; withdrawn) | user | 2026-11-03 19:10 |
| D6 | SemArt encoders (2f) | all seven | "they are not expensive, though we might expect some encoders to fail already" | user | 2026-11-03 18:55 |
| D7 | Test sets of the new encoders (3c) | ArtELingo held, MMArt test, Affection test (r14's photo cells) | "the direct answer to the question, plus it's not expensive" (ArtELingo only) | user | 2026-11-03 18:55 |
| D8 | Cost table (m1) | images per second per default pass and per Qwen instructed pass (fixed batch, warm-up, with and without decoding); peak memory; parameters; passes per aspect; c* = one pass plus three networks of 4,197,376 parameters; the whole-set figure for MMArt; one probe on 2,000 MMArt paintings, short and light | the reviewer's cost question needs measured numbers (the paper's sentences only) | user, after seeing the recommendation | 2026-11-03 19:10 |
| D9 | SemArt read-out (m2) | seed ranges and read-out picks from ArtELingo validation | SemArt shapes nothing (SemArt's own 1,069 validation paintings) | user, after seeing the recommendation | 2026-11-03 19:10 |
| D10 | SemArt labels (m3) | all 10 types ("other" 253 and "study" 24 kept); genre clustering and retrieval | no merge rule to defend (drop the small classes) | user, after seeing the recommendation | 2026-11-03 19:10 |
| D11 | Runs (m4) | seeds 0 to 4; c* and photo training on node402, one GPU per encoder, beside r18; extractions on the local 3090 | the r14 and r15 setup | user, after seeing the recommendation | 2026-11-03 19:10 |
| D12 | Matched arms (m5) | DINOv2's M5 rerun at full scale, seeds 0 to 4; B/32 and B/16 reuse M3 and M6 (full scale) | the bar needs full scale (reuse the sample-scale M5) | user, after seeing the recommendation | 2026-11-03 19:10 |
| D13 | Claim read-out (m6) | AMI at 20 clusters and mAP; r17's metrics as context after r17 merges | the bars are r14's (claims on the new metrics) | user, after seeing the recommendation | 2026-11-03 19:10 |
| D14 | SemArt context row (m7) | the SemArt cells again without the 217 paintings at cosine ≥ 0.93 to ArtELingo; no bar change; the report says whether any verdict changes | "yes"; answers a reviewer's copy question | user, after seeing the recommendation | 2026-11-03 19:10 |
| D15 | Scenario checks | (1) DINOv2 level with PercepT in style clustering: qualified, in the table, with a limit sentence; (2) a reviewer's copy (the Brueghel Proverbs pair): answered by D14; (3) B/32 holding on paintings and failing on photos: reported as such; (4) each test set keeps its own verdict | "yes" to all | user, after seeing the recommendation | 2026-11-03 19:20 |
| D16 | Stage types | items 16, 6 and 19 claim tests; stage 1 a sweep on development only; stage 4 descriptive | each feeds a paper claim (confirmation for item 16; descriptive for item 19) | agent default | 2026-11-03 19:55 |
| D17 | Teacher "better than" cells | WIN licenses "better than" only in r15's D7 cells: on MMArt as r15; on SemArt the genre mAP cell of Qwen, SigLIP 2, SigLIP 1 and CLIP L/14. Every other cell, and every cell of the new three, licenses "level with or better than" at LEVEL or WIN | r15's D3 rule (the user's) needs an earlier WIN on ArtELingo, which the new three lack (their ArtELingo cells come from this round) | agent default | 2026-11-03 19:55 |
| D18 | SemArt arms | all frozen; c*'s genre head and the matched arm's genre mapper; matched arms read as r15 | r15's frozen use (refit PercepT on SemArt) | agent default | 2026-11-03 19:55 |
| D19 | SemArt keys and duplicates | `image_file`; sha256 order; md5 duplicates keep the first; pairs at ≥ 0.95 counted | r15 D9 and D10's forms (sorted keys; keep all) | agent default | 2026-11-03 19:55 |
| D20 | SemArt units and maps | ArtELingo's genre labels for the ranges; maps fitted on all 6,152 ArtELingo validation paintings | D9 requires ArtELingo validation, whose paintings have no SemArt type (MMArt's 5,769 paintings) | agent default | 2026-11-03 19:55 |
| D21 | Bar arms' picks | the new frozen images pick by the original round's rule and ranges (r14 D19's form) | new picks match the reused ones (each X-c*'s own ranges) | agent default | 2026-11-03 19:55 |
| D22 | Probe form | SC15 on model time, three repeats, c*'s slowest against the teacher's fastest; end to end once; batch 32 (Qwen's extractor batch); the first 2,000 MMArt test keys in sha256 order; the GPU alone | model time tests "one pass against three" directly; three repeats cover noise (end to end as the bar; one repeat) | agent default | 2026-11-03 19:55 |
| D23 | Stage 0 gates | §3.5 (i) to (viii); DINOv2's M5 rerun within the stored sample arm's seed range | r15 (iv)'s fallback tolerance: GPU runs are not bit-exact across machines (exact) | agent default | 2026-11-03 19:55 |
| D24 | Failure scope | §3.5's table; r14 D18's rerun rule | a failure voids only what uses the failed part (an encoder's whole verdict) | agent default | 2026-11-03 19:55 |
| D25 | Folders and time-box | round folder `src/test/20261103_past_picks/`, prefix `pp_`; outputs `/data/SSD2/cosir_r19/20261103_past_picks/outputs/`; new stores in new folders under `/data/SSD2/pre_extract/<dataset>/<encoder>/`; report about 11-06, at the latest 11-07 23:00; node rows not done by 11-07 18:00 reported as not run | the project's layout; node402 ends about 11-08 14:00 (wait for the node's end) | agent default | 2026-11-03 19:55 |

**Amendments**: none.

## 5. Assumptions and open questions

- Assumption: node402 reads the ArtELingo DINOv2, B/32 and B/16 stores and the new Affection stores through the
  cluster CLI's data map. Not checked; a new path is verified with a real job (project memory).
- Assumption: the extractions take about 1¾ to 2 h (r10, r14 and MMArt prep rates). Not checked for these encoders.
- Assumption: r17's module merges into main tonight, before r19's build starts. Not checked; no verdict depends on it.
- Assumption: SemArt's artist-name misses and family or workshop copies are at most a few hundred paintings. Not
  checked; D14's row is the guard.
- Assumption: SemArt's `TYPE` has one value per painting. Checked: `sc_check.py` asserts it against the store index.
- Assumption: node402 has three free A6000s until about 11-08 14:00. Checked at 18:45 (115 h 14 min left); r18 shares
  it.
- Open questions: none.

## 6. Success criteria

**Decision mode.** Binding rule: this section with §3.4 and §3.5, committed with this spec before any code.

| ID | What is measured | Compared against | Bar | On which data |
|----|------------------|------------------|-----|---------------|
| SC1 | X-c*'s AMI, emotion, style, genre (new three) | X frozen image | WIN in each | ArtELingo held |
| SC2 | X-c*'s mAP, picked read-out, three aspects | X frozen image, its picked read-out | WIN in each | ArtELingo held |
| SC3 | X-c*'s AMI, three aspects | X matched arm, larger reading | WIN in each | ArtELingo held |
| SC4 | X photo model's emotion AMI | X frozen image | WIN | Affection test |
| SC5 | X photo model's emotion mAP, picked read-out | X frozen image, its picked read-out | WIN | Affection test |
| SC6 | X-c*'s AMI and mAP (three aspects); X photo model's AMI and mAP | the teacher, its picked read-out | each cell its own claim, LEVEL or WIN | ArtELingo held, Affection test |
| SC7 | X-c*'s AMI, three aspects | X frozen image | WIN in each | MMArt test |
| SC8 | X-c*'s mAP, picked read-out, three aspects | X frozen image, its picked read-out | WIN in each | MMArt test |
| SC9 | X-c*'s AMI, three aspects | X matched arm, larger reading | WIN in each | MMArt test |
| SC10 | X-c*'s AMI and mAP, three aspects | the teacher, r15's pick | each cell its own claim, LEVEL or WIN | MMArt test |
| SC11 | X-c*'s genre AMI (all seven) | X frozen image | WIN | SemArt set |
| SC12 | X-c*'s genre mAP, picked read-out | X frozen image, its picked read-out | WIN | SemArt set |
| SC13 | X-c*'s genre AMI | X matched arm, larger reading | WIN | SemArt set |
| SC14 | X-c*'s genre AMI and mAP | the teacher (`genre_word`), its picked read-out | each cell its own claim (D17) | SemArt set |
| SC15 | c* on X: time per image for three aspects, model time (all seven) | the teacher's three instructed passes | c*'s slowest of three repeats below the teacher's fastest | 2,000 MMArt paintings, 3090 |

X in SC1 to SC10: DINOv2, CLIP B/32, CLIP B/16. X in SC11 to SC15: all seven encoders.

- **Verdicts, per encoder and test set, each its own (D15 (4)).** ArtELingo (SC1 to SC3) and MMArt (SC7 to SC9):
  holds when all nine cells WIN; qualified when the six frozen-image cells WIN and one or more PercepT cells do not;
  fails otherwise. Photos (SC4, SC5): holds when both WIN; fails otherwise. SemArt (SC11 to SC13): holds when all three
  WIN; qualified when SC11 and SC12 WIN and SC13 does not; fails otherwise. Cost (SC15): holds or fails. Holds is GO,
  qualified GO for the qualified claim, fails NO-GO. A failed verdict stands and goes in the paper's encoder table
  with a limit sentence (D4).
- **Teacher cells (SC6, SC10, SC14).** Per D17: a WIN in a D7 cell licenses "better than"; LEVEL or WIN in any other
  cell licenses "level with or better than"; BELOW licenses nothing and keeps its row.
- **Inconclusive.** A failed gate voids what §3.5's table says; a verdict also needs all five runs of its arm (r14
  D18). The rows that exist are reported; the rest go on.
- **Selection rules.** Stage 1 only: ranges, then read-out picks (§3.4, D21). Stages 2 and 3 select nothing.
- **Licensed (item 16).** r14's paintings and photos sentences and r15's sentence, each with its verdict's form, with
  "<X> is one of three encoders that failed or were unclear in an earlier held emotion test on ArtELingo with an older
  model (R6-5)" in place of r14's and r15's encoder sentence.
- **Licensed (item 6, holds).** "On <n> paintings by <m> artists who have no painting in ArtELingo's train set
  (SemArt, from the Web Gallery of Art; artists matched by name; images at cosine ≥ 0.95 to an ArtELingo train or
  validation painting removed), frozen c* on <X> clusters and retrieves paintings by type (SemArt's 10 classes, close
  to genre) better than the plain <X> image, and clusters them better than PercepT on the <X> image (AMI with 20
  clusters; mAP with read-outs picked on ArtELingo validation; 5 seeds)." Qualified: "…; against PercepT on the <X>
  image it is not better." SC14: "<c* on X> is better than (or: level with or better than) the Qwen image instructed
  for genre in <task> on SemArt's new artists."
- **Licensed (item 19, holds).** "On one RTX 3090 at batch 32, c* on <X> embeds a painting for emotion, style and genre
  in <a> ms (one <X> pass and three small heads), against <b> ms for the Qwen image instructed per aspect (three
  passes); model time, images decoded beforehand (<c> and <d> ms with decoding)."
- **Not licensed:** emotion or style on new artists; "unseen artists" beyond name matching; "better than the teacher"
  outside D17's cells; SemArt's schools or periods; claims from r17's metrics; cost on other GPUs or batch sizes, or
  training cost; any setting as better than c*'s.

## 7. Constitution check

| Principle | Pass or exception | Note |
|-----------|-------------------|------|
| C1 strongest comparator | pass | the frozen image and the matched arm as bars, the teacher per cell, the strongest PercepT entry beside; cost against the teacher c* replaces |
| C6 sample IDs | pass | store indexes; SemArt by `image_file`; stage 0(i) |
| C7 rule before code | pass | §6 with §3.4 and §3.5, reviewed by a fresh Opus `xhigh` worker before the approving commit |
| C8 prior and disclosure | pass | prior §3.1; disclosure §8 |
| C12 regression check | pass | stage 0 (i) to (v), (vii), (viii) |
| C13 re-derivation | pass | Codex, §3.4 |
| `held-data.md` | pass | stopping, ranges and picks on validation only; test sets scored, selecting nothing; SemArt's units and picks from ArtELingo validation (D9); the matched arm's larger reading on test only makes the bar stricter; query expansion reads unlabeled test embeddings with k fixed on validation, as r11 |

Exceptions: none.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|---|---|---|
| SemArt copies of ArtELingo paintings below 0.95, or artist-name misses; both favor c* and PercepT (trained on ArtELingo) over the plain image and the teacher | D14's row; the close pairs of the SemArt check (mostly same-subject paintings by other artists, one family copy) | the report says whether any verdict changes without the 217; no bar change |
| SemArt's types differ from WikiArt genre (religious, mythological), while ranges and maps come from ArtELingo genre | AP per type; clustering at the class count | reported beside the claim; the claim says "type, close to genre" |
| A new encoder's run fails or diverges | the rerun rule | that encoder's dependent verdicts inconclusive; the others go on |
| node402 is busy with r18, or slower than estimated | the timed first runs | the runs queue for a free GPU; the time-box holds; rows not done by 11-07 18:00 are reported as not run |
| The heavy extraction cannot start before Wednesday 10:00 | the build's end time | it waits until 19:00 (weekday rule); the report moves to 11-07 at the latest |
| GPU timing noise or another process on the 3090 | the three repeats; gate (viii); the lock | the probe runs alone; a failed (viii) stops it for tracing |
| r17's module is not merged | the build's start | context rows skipped and noted; no verdict depends on them |
| Selection disclosure: the three encoders were left out of r14 because of their R6-5 results (B/32, B/16 inconclusive; DINOv2 NO-GO), so the choice leans toward hard cases; ArtELingo validation carried c*'s stopping, r10's 59 runs and r11 to r15's picks; Affection validation r12's and r13's picks; stage 1 adds up to 25 read-outs per arm; on SemArt, the SemArt check and the grill read type counts and cosines, no model score | validation numbers inflated | only test numbers are results; every pick is frozen before scoring; the claim names how the encoders were chosen |

## 9. Deliverables and review

- At the end: the round folder `src/test/20261103_past_picks/` (prefix `pp_`) with code, tests, log and per-item
  outputs; run outputs under `/data/SSD2/cosir_r19/20261103_past_picks/outputs/`; new stores under
  `/data/SSD2/pre_extract/{mmart,affection_local,semart}/`; the auto report `docs/reports/auto/percept/` (the next
  sequence date) with its results brief and index row; a user-read report. The report has a short section on how much
  the results change the paper (D3): related work (SemArt moves from related work, `paper_v1.md` line 128, to a test
  set), method (§3.7 names the encoders and says "We changed no setting for any encoder"), experiments (the encoder
  table with every verdict and its limit sentences; the cost table beside §3.2's and §3.6's one-pass sentences).
- Reviews: this rule by a fresh Opus `xhigh` worker before the approving commit; tickets per `matt-chain.md`, with the
  SemArt set, the export and matched-arm paths, the picks, the scorer, the probe's timing code and the outcome code as
  verdict code; the C13 re-derivation by Codex; the final whole-branch review.
- Stop points: none between approval and the reports. Time-box: the report about 2026-11-06, at the latest 11-07
  23:00.
- Budget: node402, 3 A6000: 15 X-c* runs about 2.2 GPU h (r14: about 516 s each), 15 photo models about 0.9 GPU h,
  DINOv2's M5 about 4 min, stage 0's three runs about 0.2 GPU h; together about 3½ GPU h, about 1¼ h of wall time on
  three free GPUs. Local 3090, under the lock: extractions about 1¾ to 2 h, heavy, after 19:00 on weekdays and ending
  by 11:00 (MMArt B/32, B/16, DINOv2, 16,782 each; Affection the same three, 85,007 each; SemArt SigLIP 1, CLIP L/14,
  B/16, DINOv2 and Qwen `genre_word`, 21,382 each); the probe about 45 min (Qwen's four passes dominate), short. CPU
  about 3 to 4 h at 8 threads (ArtELingo, Affection and MMArt scoring for three encoders; SemArt for seven; C13).
  Disk: about 6 GB of run outputs on SSD2 (15 X-c* runs at about 0.38 GB, 15 photo models at about 0.18 GB); new
  stores under 1 GB.

Body length: about 310 non-blank lines against a claim-test target of about 150 plus the rule: three claim tests on
four test sets share one spec, and most protocol is cited from r14 and r15.
