# Outside methods (TPIPS, CLAY, TURTLE, then TEMI) beside c*: design

> Date: 2026-11-03 19:55 (rule-review fixes applied 20:15) · Status: approved 2026-11-03 20:45 · Kind: experiment
> Builds on: decision A of 2026-11-03 18:30 (r16 report 3: the experiment overview's plan, loop 2 = r18), the TPIPS
> timing probe and the r18 grill (r18 decide chat, 19:00 to 19:20; [round log](../../../src/test/20261103_outside_methods/20261103_outside_methods_log.md))
> and the code-route answer of this chat (D12) · Constitution: [constitution.md](../constitution.md) · Decision rule:
> inline in §6, with §3.4 and §3.5
> Inherits: the r17 spec ([2026-11-03-new-scores-design.md](2026-11-03-new-scores-design.md)): arms X-c* and X plain
> image (§3.2), populations, class counts, keys and orders (§3.3), clustering, intervals, labels and retrieval
> (§3.4), its scorer (§3.7, imported read-only); through it r15 and r14 · Facts:
> [facts brief](../../../src/test/20261103_outside_methods/om_facts_brief.md) (Codex, 2026-11-03; pinned commits) ·
> Reviewer: Opus `xhigh` rule review 2026-11-03 20:15 (approve with fixes, applied; `om_rule_review.md` in the round folder)

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** Readers will ask how c* compares with outside methods that name the aspect in text, and whether a
stronger clustering method than k-means closes c*'s gap over the plain image. This loop adds TPIPS (an 8B embedder
tuned to a named factor) and CLAY (a training-free text projection of CLIP L/14) as context rows in both tasks, and
replaces k-means with TURTLE, then TEMI (two label-free clustering methods).

**Plan** (the build starts after r17's scoring code merges; build and run unattended; you read the report).
1. Stage 0 (checks): our CLAY and TURTLE code match the official code on a fixed input (made once by the official
   code, attended, in the spec chat; hashes in the round log); our loaders and r17's scorer replay earlier cells;
   TPIPS on DAS6 matches a local run. A failed check voids that method's cells.
2. Stage 1 (sweep, validation only): CLAY's text set (CRL's 50 values or 100 new values), the read-outs of CLAY and
   TPIPS, and Qwen c*'s missing seed range. Committed before any test number.
3. Stage 2 (claim test, labels only): TPIPS and CLAY against c*; TURTLE on the plain image and on c*'s heads.
4. Stage 3 (claim test, later): TEMI as TURTLE, on a fixed ordered list of whole runs; a timing run only sets how
   many fit in 10 GPU h. The c* heads against the plain image with TEMI may get no cell.
5. Stage 4 (descriptive): all of r17's cluster counts; c* with TURTLE beside c* with k-means.

**Compared against.** c* with its usual reading. TPIPS against c* on Qwen; CLAY against c* on CLIP L/14 and on Qwen.
TURTLE: c* with TURTLE against the plain image with TURTLE, and c* with k-means against the plain image with TURTLE.

**Outcomes.** Each cell gets WIN, LEVEL or BELOW for c* (your rule: no pass or fail, no verdict for the loop); the
label sets the paper's sentence for that cell. We expect c* to WIN against CLAY in most cells (0.8 to 0.85), even
odds against TPIPS in clustering (0.5) and less in retrieval (0.35), WIN with TURTLE on both sides (0.8), and WIN with k-means against the plain image with TURTLE (0.7;
genre 0.6).

**Cost.** TPIPS about 6.4 GPU h (an estimate from 219 ms per image per factor on 48 images), about 2.5 h of wall time
on DAS6's three A6000s (not measured there). TURTLE about 410 fits locally, about 2 to 14 h (20 s to 2 min per fit, not
measured). CLAY and the value lists: minutes. Scoring about 4 h of wall time. Build once r17's scorer merges (about
11-04); report by 11-06 18:00. TEMI: its own build later, up to 10 GPU h; results before 11-16 go to the paper.

**Check these.**
1. The code route: your pick after seeing our recommendation (your first pick was the clones under a permission
   rule). We re-implement CLAY and TURTLE and check them against the official code; TEMI's Apache-2.0 code is copied
   with its notice. No CLAY or TURTLE code enters the public repo.
2. TURTLE then TEMI: your own pick; ours was TURTLE only.
3. Comparators and margins (agent default): TPIPS against Qwen c*, CLAY against both c* arms. The LEVEL margin of a
   TURTLE or TEMI cell is c*'s k-means seed range, not the method's own (that would cost about 220 more fits).
4. TURTLE runs at its code defaults, without its paper's label-free choice among 50 settings (agent default; the
   grill said "paper defaults"). The paper sentences say so.
5. For the unattended run (agent default): TPIPS on DAS6 is not checked (env, images, 32 GB of weights); fallback,
   the local 3090 for one night, ending by 10:00. A failed check voids that method's cells and the run goes on, an
   exception to the constitution's C12, which says to stop (§7).

## 1. Context and goal

Asked: decision A (2026-11-03 18:30, r16 report 3 chat): items 13 then 14 (TPIPS), 15 (CLAY) and 3 ("a strong
clustering method on the plain image (TEMI or TURTLE), four encoders") of the experiment overview's plan. The
user's reading of the overview: "Missing: CRL and CLAY in clustering; PercepT, CRL, CLAY and TPIPS in retrieval."

c*'s claims (r10 to r15) stand against the plain image, PercepT and the instructed Qwen teacher on four encoders,
ArtELingo held and MMArt test, with k-means at 20 clusters and picked read-outs; r17 adds the standard metrics at the
class count and CRL on CLIP L/14. Two kinds of comparison are still missing: recent methods that name the aspect in
text (TPIPS, arXiv 2607.18237, concurrent; CLAY, CVPR 2026, arXiv 2604.11539), and a clustering method stronger than
k-means on the plain image (TURTLE, ICML 2024; TEMI, BMVC 2023). The TPIPS timing probe ran in the decide chat (219 ms
per image per factor on the 3090, round log).

Goal: put TPIPS, CLAY, TURTLE and TEMI beside c* on ArtELingo held and MMArt test, on our protocol.
Question this step answers: where does c* stand against two outside methods that name the aspect, and does its
clustering gain over the plain image hold when k-means gives way to a stronger label-free clustering method?

## 2. Scope

In: TPIPS 8B with three factors; CLAY on CLIP L/14 with text sets A and B; TURTLE on the plain image of four encoders
at r17's counts and on c*'s heads at the class count and 20; TEMI as TURTLE at stage 3's scope; ArtELingo held and
MMArt test; picks on their validation splits; §3.6.
Out: TPIPS 2B and TPIPS as a teacher (parked, user); CLAY on other encoders; TURTLE's label-free hyperparameter
selection and TURTLE on two spaces at once (a new spec, if a cell comes close); the photo sets; CRL and PercepT
(r17); new encoders (r19); intervals at counts other than the class count and 20 (as r17 D3).

## 3. Design

### 3.1 Hypothesis and prior

Claims tested: c* clusters and retrieves paintings by aspect at least level with TPIPS and better than CLAY; c*'s
clustering gain over the plain image holds when both sides use TURTLE (or TEMI), and c* with k-means still beats the
plain image clustered by TURTLE (or TEMI).
Mechanism: c*'s heads are trained per aspect on paintings (buddy graphs, the instructed teacher). CLAY projects
frozen CLIP onto a subspace of value texts with no training, as CRL scores CLIP against value texts; c* on CLIP L/14's
held emotion mAP (r14, 0.2671) is above CRL's (r11, 0.2554), with no paired interval. TPIPS fine-tunes the 8B Qwen embedder on human
odd-one-out votes under a named factor (its paper: focus attribute 31.35 against 30.85 zero-shot): a base four times
c*'s 2B, not trained on painting aspects (not checked). TURTLE and TEMI find better partitions of a fixed space than
k-means but add no aspect signal the space lacks; on the plain image the dominant structure is subject and genre, so
emotion and style gain least (our guess).
Prior (C8), per cell: c* WIN against CLAY 0.8 (CLIP L/14 c*) and 0.85 (Qwen c*), both tasks; against TPIPS 0.5 in
clustering, 0.35 in retrieval; c* + TURTLE against plain + TURTLE 0.8; c* with k-means against plain + TURTLE 0.7
(genre 0.6); TEMI as TURTLE. Surprise: the plain image with TURTLE above c* in emotion; TPIPS below CLAY.

### 3.2 Arms and controls

X = Qwen, SigLIP 2, SigLIP 1, CLIP L/14. Labels are for c*: d = c* minus the other arm.

| Arm | What it is | Role |
|---|---|---|
| X-c* (k-means) | r17's X-c*: the frozen aspect head, seeds 0 to 4; its usual reading (k-means; picked read-out) | reference of every label |
| TPIPS | TPIPS-Embed-Qwen3VL-8B (snapshot 86a1493c, late fusion), factor text = the aspect name; 4,096-wide unit vector per image and aspect; unseeded | context row (SC1, SC2) |
| CLAY | CLIP L/14's image conditioned per aspect by CLAY (§3.4), text set A or B picked on validation (D7, D8); 768-wide unit vector; unseeded | context row (SC3, SC4) |
| X plain + TURTLE | X's default image partitioned by TURTLE (§3.4), fitted on the scored set, starts 0 to 2 | comparator (SC5, SC6) |
| X-c* + TURTLE | each aspect head partitioned by TURTLE at the class count and 20, start 0 per c* seed | candidate (SC5) |
| X plain + TEMI, X-c* + TEMI | as the TURTLE rows, with TEMI (§3.4), at stage 3's scope | SC7, SC8 |

Tuning: c* had r10's validation runs and picks; TPIPS and CLAY get validation read-out picks (and CLAY's text set);
TURTLE and TEMI get none. Text set A: r11's CRL lists, 50 values per aspect, "a painting with the {aspect} of {value}", rank 49. Text set B:
100 values per aspect from the same LLM and prompt asking for 100, "a painting of {value}", rank 50 (D5).

### 3.3 Data roles

| Data | Role | Stages |
|---|---|---|
| ArtELingo validation, 6,152 (genre 4,960) | development: CLAY's text-set and read-out picks, TPIPS's read-out pick; the parity inputs | 0, 1 |
| MMArt validation, 5,769 (MMArt labels; its images are ArtELingo validation's) | development: the same picks for MMArt | 1 |
| ArtELingo held, 12,281 (genre 9,949) | held: replays, cells, descriptive; CLAY's mean image and the TURTLE and TEMI fits (TEMI's head choice and neighbour graph too) read its images, never its labels | 0, 2, 3, 4 |
| MMArt test, 16,384 (r15's cut set) | held, as above | 0, 2, 3, 4 |
| Every train split | not touched | none |

Class counts and cluster counts: r17 §3.3 and D2 (emotion 9: 5, 8, 9, 16, 20; genre 10: 5, 10, 16, 20; style 27 on
ArtELingo and 20 on MMArt: 10, 20, 27, 40). Populations, keys and orders: r17 §3.3; TURTLE and TEMI, like k-means,
partition every row and mask missing labels after. One plain-image partition per encoder, set, K and start serves all
three aspects (192 fits); c*'s heads add 220 (4 encoders × 11 aspect-K pairs × 5 seeds): 412 in all. TPIPS extracts 34,817 images (ArtELingo held and validation, MMArt
test); MMArt validation reads the ArtELingo validation store by painting name (C6: extraction keys only).

### 3.4 Measurement

- **Clustering, intervals, labels.** r17 §3.4 unchanged: at the class count AMI, NMI, ARI and ACC, at 20 AMI; 2,000
  joint draws, rows then seeds; unseeded arms resample rows only. TPIPS and CLAY: k-means as r10 on each aspect's
  vector. A TURTLE or TEMI partition takes k-means' place: the plain arm averages its 3 starts as k-means averages its
  states; X-c* + TURTLE draws seeds as c*. WIN when the lower 95% bound of d > 0; LEVEL when not WIN and the lower
  bound > −r; BELOW otherwise; no multiplicity correction. r is the reference c*'s validation seed range for that set,
  aspect, metric and K, from the table below; TURTLE and TEMI cells use the same ranges (D14).
- **Source of r** (stage 1 writes every range it computes, committed with the picks):

| Cell family | r from |
|---|---|
| Class count (four metrics), ArtELingo and MMArt | r17's stage 1 (Qwen from r10's validation exports); MMArt style's AMI (its class count is 20): r15's |
| AMI at 20 clusters: SigLIP 2, SigLIP 1, CLIP L/14 on ArtELingo | r14's records (`artclu`) |
| AMI at 20 clusters: the four encoders on MMArt | r15's records (`clu_ranges`) |
| AMI at 20 clusters: Qwen on ArtELingo | this stage 1: k-means as r10 on r10's validation exports of Qwen c* (seeds 0 to 4), mean over states 0 to 2 per seed, maximum minus minimum, unrounded (no recorded range exists) |
| Retrieval mAP | Qwen on ArtELingo: r11's `stage0a_ranges.json`; the other encoders on ArtELingo: r14's (`artret_ranges`); MMArt: r15's (`ret_ranges`) |

- **Retrieval.** r17 §3.4 (r11's protocol; paired bootstrap, 10,000 draws). New picks on the set's validation split
  by r11's rule, one per arm for its three aspects; units: Qwen c*'s validation mAP ranges for TPIPS, CLIP L/14 c*'s
  for CLAY (r15 D12's form). CLAY's menu clips the whitening sizes to its rank. CLAY's text set, per dataset: r11's
  rule picks a read-out per text set (guard against that set's own R0); the set whose pick has the higher mean over
  the three aspects of validation mAP divided by CLIP L/14 c*'s validation mAP range on that set is used; ties
  within 0.1 of that unit go to A (D7).
- **CLAY** (CVPR 2026, official code at a3f1890). Per aspect: unit text vectors T from CLIP L/14's text encoder;
  μ = the normalized mean of T; tangent vectors log_μ(T), centred; P from the top `rank` right singular vectors. Images:
  two Householder reflections take the scored set's normalized mean image to μ through their normalized midpoint
  (D6); then log_μ, P, normalize. The mean is over all rows of the split being scored, whatever their labels
  (6,152; 5,769; 12,281; 16,384). One CLAY vector per row and aspect serves clustering, retrieval and both cross-fit
  halves. Query and gallery are the same set.
- **TURTLE** (ICML 2024, official code at 4674efa). Code defaults: γ 10, 6,000 outer steps of 10 inner steps, Adam
  1e-3 for both, batch 10,000 (capped at the set), cold start; one space; K as the cell; the partition is the argmax on
  every row. Ours keeps the data on the GPU, draws its random numbers in the official order (gate (ii)) and reads no label (the
  official code logs validation accuracy every 20 steps).
- **TEMI** (BMVC 2023, official code at c6a81a6, Apache-2.0), one space, K as the cell. Paper Table 10: AdamW,
  weight decay 1e-4, batch 512, 50 heads, 200 epochs, β 0.6, teacher momentum 0.996 constant, teacher temperature 0.1
  throughout, learning rate warmed up over 20 epochs to 1e-4, then constant. Where Table 10 is silent, the copy's
  parser default: heads of 2 layers, hidden 512, bottleneck 256, no batch norm or dropout. 50 neighbours by cosine on
  the unit rows of the fitted set, self excluded; no input standardization; the head with the lowest mean training
  loss over the last epoch; its argmax on every row (D10).
- **C13.** Separate code re-derives, from this rule and the saved outputs, the picks and every SC cell's value, d,
  interval and label, within 1e-6, before any label is written. It recomputes CLAY's vectors from the CLIP rows and
  the text vectors (a second check of gate (i)) and k-means on TPIPS and CLAY (same threads); it reads the TURTLE and
  TEMI partitions (Codex if `codex-usage --room` says room, else a
  Claude worker).

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|---|---|---|---|---|
| 0 | regression check | parity references; r10, r11, r14, r15 records; the 48 probe images | (i) to (iv) | void per the table below, trace, note it |
| 1 | sweep (development only) | the validation splits; r10's validation exports | every range and pick written and committed before any stage 2 number | void per the table below |
| 2 | claim test, labels only | the held sets, all frozen | SC1 to SC6 | report |
| 3 | claim test (TEMI), later | as stage 2, after TEMI's timing run | SC7, SC8 within the scope rule | report what ran |
| 4 | descriptive | all | §3.6, after the labels | report |

- (i) CLAY parity: on the reference input (the first 2,000 ArtELingo validation paintings in sorted name order, CLIP
  L/14, emotion, text set A), our conditioned vectors equal the official code's within 1e-5 (largest absolute
  difference; float32, CPU; the same text vectors on both sides).
- (ii) TURTLE parity: on the reference input (the same 2,000 paintings, plain Qwen, K 9, 100 outer steps, batch 10,000
  capped at 2,000, CPU, seed 0), our final effective task-encoder weights (after weight-norm removal) equal the
  official code's within 1e-5 (largest absolute difference) and the partitions are identical; our GPU run of the same
  input, TF32 off as in every fit, gives a partition with AMI ≥ 0.99 against the CPU one. The reference batch is the
  full set, so batch sampling is checked apart, in numpy: our index draws equal the sequence of
  `np.random.choice(n, min(10000, n), replace=False)` after TURTLE's seed setting (seed 0), first 3 draws, n = 12,281
  and 16,384, exactly.
- (iii) Path replay within 1e-9, our loaders feeding r17's scorer. (a) k-means at 20: the AMI values and intervals of
  r15's CLIP L/14 and Qwen MMArt SC1 cells and r14's SigLIP 2 ArtELingo SC1 cell; r10's stage 4 Qwen held partitions at
  20, exactly. (b) The partitions of (a) fed back through the partition input used for TURTLE and TEMI (3 partitions
  for an unseeded arm) give the same values and intervals. (c) Retrieval: r15's CLIP L/14 and Qwen MMArt SC2 values
  and intervals, and one of r11's stage 5 cells of Qwen c* against the plain image (ArtELingo held), each at its
  recorded pick.
- (iv) TPIPS: on the 48 probe images, all three factors, every path used (DAS6, any batched path) gives a local
  batch-1 run's vectors within cosine 0.999 per row; every store holds the expected keys, once each, as unit rows.

The references for (i) and (ii) were made once by the official code, attended, in the spec chat (2026-11-03 19:58;
driver outside git, inputs from `om_parity_inputs.py`, hashes in the round log); the build never runs the clones. TEMI
before stage 3: a fresh reviewer agent checks our copy's diff against commit c6a81a6 (every change marked), and a toy
run (10 well-separated Gaussian blobs in 64 dimensions, 500 points each, K 10, seed 0) reaches ACC ≥ 0.95.

**TEMI scope rule (D10).** A unit is one encoder × one set × one arm × one K with all its starts (plain image: 3) or
seeds (c*'s heads: 5, for every aspect that uses that K). The list is fixed here, in three groups, encoders in the
order Qwen, SigLIP 2, SigLIP 1, CLIP L/14, then ArtELingo before MMArt, then K ascending: (1) the plain image at the
class count and 20 (28 units, 84 fits); (2) c*'s heads at the class count and 20 (28 units, 220 fits); (3) the plain
image's other counts (36 units, 108 fits). The timing run (Qwen plain, ArtELingo held, K 20, start 0; no metric
computed) gives t per fit; N = floor(10 h / (t × 16,384 / 12,281)). Whole units run in list order until the next unit
would take the total over N fits; none is skipped or cut. The list of units to run, with t and N, is committed in the
round log before the first TEMI fit. No TEMI cell is scored until every listed fit is done; a cell is scored only if
every fit it needs is in the list.

| Failed gate or missing part | Voids |
|---|---|
| (i) | SC3, SC4 |
| (ii) | SC5, SC6 |
| (iii) a | every clustering cell |
| (iii) b | SC5 to SC8 |
| (iii) c | SC2, SC4 |
| (iv) | TPIPS cells, until a local batch-1 extraction passes |
| TEMI diff review or toy run | SC7, SC8 |
| stage 1: a read-out or CLAY's retrieval text set missing | that arm's SC2 or SC4 cells |
| stage 1: CLAY's clustering text set missing | SC3 |
| a range missing (r17's, r14's, r15's, r11's or stage 1's) | the LEVEL word of its cells: a cell that is not WIN is inconclusive; WIN stands |
| r17's stage 0 voided a part of its scorer | the same cells here |

A failed gate may be rerun after a traced code fix, logged, before any cell of that method is computed. A gate that
still fails at the time-box voids as above. The report gives the user each failure and its cause.

### 3.6 Diagnostics

After the labels, deciding nothing:
- **Count sweep.** AMI and ARI at r17's counts for plain + TURTLE (and TEMI where run), TPIPS and CLAY (k-means),
  beside r17's k-means rows; point values (mean, minimum and maximum over starts or seeds).
- **What TURTLE adds.** X-c* + TURTLE against X-c* (k-means), and plain + TURTLE against plain (k-means), at the class
  count and 20: d with its paired interval, no label.
- **Fits.** The number of non-empty clusters per TURTLE and TEMI fit; TURTLE's final loss and entropy.
- **CLAY.** The text set picked per task and both sets' validation scores; class names found verbatim in each list
  (listed, not removed; D22).
- **TPIPS.** The mean cosine between an image's three factor vectors; the R0 rows of TPIPS and CLAY.

### 3.7 Frozen vs rerun

Frozen: every c* export and plain store; r10 to r17's picks, ranges and partitions; r11's CRL texts; the TPIPS
weights; text set B once generated; the parity references; stage 1's picks before stage 2. Rerun: the TPIPS
extraction; CLAY's vectors; the TURTLE and TEMI fits; k-means on TPIPS and CLAY; all scoring, through r17's scorer
imported read-only (a missing function is added in our round folder, never in r17's).

## 4. Decisions

Grill rows: the r18 decide chat, 2026-11-03 18:30 to 19:20. D12 and agent defaults: r18 spec chat, 19:45 to 19:55; rule-review fixes 20:15.

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|----|----------|----------|----------------------------|------------|------|
| D1 | Items and sets | TPIPS, CLAY, TURTLE then TEMI; ArtELingo held and MMArt test; both tasks for TPIPS and CLAY | the plan's loop 2 | user | 18:30 |
| D2 | TPIPS | the released 8B late fusion model; factors "emotion", "style", "genre"; not the 2B, not as a teacher | the 2B needs our own training | user (model); user, after seeing the recommendation (factors) | 18:30; 19:20 |
| D3 | TPIPS device | DAS6 node402, a third of the images per GPU; the local 3090 at night as fallback | about 2.5 h against 6.4 h | user, after seeing the recommendation | 19:20 |
| D4 | CLAY | CLIP L/14; text sets A and B, picked on validation | one set fixed blind | user (ours the same) | 19:00 to 19:20 |
| D5 | Text set B | Qwen3-VL-8B-Instruct, greedy, r11's prompt asking for 100 distinct values; duplicates (same text after lower-casing and trimming) dropped, the first 100 kept, rank min(50, n − 1); with fewer than 2 values or a failed run CLAY uses set A with no pick | CLAY's own lists are for photos | user (form); agent default (dedup, cap, fallback) | 19:20; 19:55; 20:15 |
| D6 | CLAY's alignment | the scored set's mean image, as CLAY's code (label-free) | the train split's mean: not CLAY's method | agent default | 19:55 |
| D7 | Read-out | r11 to r15's menu and rule on validation, one pick per method across aspects, per dataset; CLAY's whitening sizes clipped to its rank (15 per text set); for retrieval, r11's rule picks one read-out per text set (guard against that set's own R0), then the set whose pick has the higher mean over aspects of validation mAP divided by CLIP L/14 c*'s validation mAP range on that set; ties within 0.1 of that unit to A | sizes above the rank whiten noise; the rule score is a gain over each set's own R0, so it does not compare the sets (review B1) | user, after seeing the recommendation (menu); agent default (CLAY details) | 19:20; 19:55; 20:15 |
| D8 | Clustering of TPIPS and CLAY | k-means at r17's counts on each aspect's vector; CLAY's text set, per dataset, by the mean over aspects of validation AMI at 20 (k-means as r10, mean over states 0 to 2), ties within 0.001 to A | a pick per task gives each task its best | user, after seeing the recommendation; agent default (pick) | 19:20; 19:55; 20:15 |
| D9 | TURTLE | replaces k-means on the plain image (four encoders, r17's counts, 3 starts), fitted on the scored set; also on c*'s heads at the class count and 20, one start per seed; code defaults, no 50-setting selection | "paper defaults" (grill); the selection costs 50 fits per cell | user; agent default (code defaults) | 19:20; 19:55 |
| D10 | TEMI | after TURTLE; scope: the unit list fixed in §3.5, the timing run only sets how many whole units fit in 10 GPU h; results after 11-16 go to the supplementary; paper Table 10 settings, gaps filled by the copy's parser defaults (§3.4) | TURTLE only (ours); a run-time choice of cells after TURTLE's labels (review B4, S6) | user (own pick); agent default (budget, settings, list) | 19:00 to 19:20; 19:55; 20:15 |
| D11 | Labels | WIN, LEVEL or BELOW against c* per cell; no pass or fail, no loop verdict | a gate per family | user, after seeing the recommendation | 19:20 |
| D12 | Code route | CLAY's math and TURTLE's loop re-implemented in our repo, checked against references made once by the official code (attended); TEMI's Apache-2.0 code copied with its notice; no CLAY or TURTLE code in git | clones under a standing permission rule (first pick); authors asked for a licence (declined) | user, after seeing the recommendation | 19:45 |
| D13 | Reference c* per method | TPIPS: Qwen c*; CLAY: CLIP L/14 c* and Qwen c*; TURTLE and TEMI: the same encoder's c* | CLAY against Qwen c* only mixes encoders | agent default | 19:55 |
| D14 | Margin r of TURTLE and TEMI cells | c*'s k-means seed ranges, from the source table in §3.4 (Qwen on ArtELingo at 20 clusters computed in stage 1) | the method's own ranges: about 220 more validation fits | agent default | 19:55; 20:15 |
| D15 | Protocol | r17's metrics, draws, labels and no multiplicity correction | comparability with r10 to r17 | agent default | 19:55 |
| D16 | Stage types | 0 regression; 1 sweep; 2 and 3 claim tests (labels feed paper sentences); 4 descriptive | confirmation for TURTLE: its sentence is a paper claim | agent default | 19:55 |
| D17 | Order | the build starts after r17's scorer merges; TEMI built after TURTLE runs | | user | 19:20 |
| D18 | Folders | round `src/test/20261103_outside_methods/`, prefix `om_`; outputs `/data/SSD2/cosir_r18/20261103_outside_methods/outputs/`; TPIPS stores `/data/SSD2/pre_extract/{artelingo_full,mmart}/tpips_8b/`; clones and references `/data/SSD2/cosir_r18/ext/`, outside git | features belong in the shared cache (`storage.md`) | user (prefix); agent default (paths) | 19:20; 19:55 |
| D19 | Report | each report states how much it changes the paper draft: related work, method, experiments | | user | 18:33 |
| D20 | TPIPS batching | length-sorted batches if faster and within gate (iv); else batch 1 | batch 4 and 8 were slower (padding) | agent default | 19:55 |
| D21 | TURTLE and TEMI seeds | plain image starts 0, 1, 2; c*'s heads start 0 for each c* seed | as k-means' states 0 to 2 | agent default | 19:55 |
| D22 | Class names in LLM lists | listed, not removed | | user (scenario confirmed) | 19:20 |
| D23 | Gate numbers | 1e-5 (CLAY and TURTLE weights), 1e-9 (path replay), AMI ≥ 0.99 (GPU against CPU), cosine 0.999 (TPIPS), ACC ≥ 0.95 (toy run), 2,000 reference rows, 100 outer steps, 48 probe images | tight enough to catch an order or math error, loose enough for float32 | agent default | 2026-11-03 20:15 |
| D24 | Failed gates | a failed gate voids the cells in §3.5's table and the run goes on; a rerun after a traced fix is allowed before any cell of that method is computed; the report gives each cause; C12 exception (§7) | stopping the whole run idles every other arm; the run is unattended (D17) | agent default | 2026-11-03 20:15 |
| D25 | Paper wording of TURTLE and TEMI | the SC5 to SC8 sentences name "at its released code's default settings, without its hyperparameter selection" (TURTLE) and "with its paper's Table 10 settings" (TEMI); no selection run | a reader takes "clustered with TURTLE" as TURTLE as published; a one-off selection on validation (50 fits) is the option if a cell comes close | agent default | 2026-11-03 20:15 |

**Amendments**: none.

## 5. Assumptions and open questions

- Assumption: r17's scorer merges with its §3.7 interface and takes a partition as input. Not checked (r17 builds
  tonight).
- Assumption: node402 runs TPIPS: an env with transformers ≥ 5.0 and peft ≥ 0.19, the 32 GB of weights through the
  node's HF cache, the 34,817 images on the node. Not checked; the build's first ticket checks and syncs.
- Assumption: a TURTLE fit takes 20 s to 2 min with data on the GPU (412 fits: about 2 to 14 h). Not measured; the
  smoke times one fit and reads its power.
- Assumption: the official CLAY and TURTLE code runs on torch 2.11 for the references (TURTLE uses the deprecated
  weight-norm function). Not checked; the reference run shows it.
- Assumption: the 8B instruct model gives about 100 distinct values per aspect. Not checked.
- Open questions: none (the key point is D12).

## 6. Success criteria

**Decision mode, labels only** (D11). Binding rule: this section with §3.4 and §3.5, committed with this spec before
any code. Each cell gets WIN, LEVEL or BELOW by §3.4's numeric rule, which is the bar of every SC, with r from §3.4's source
table; no cell has a pass bar and the loop has no verdict.

| ID | What is measured | Compared against | On which data |
|----|------------------|------------------|---------------|
| SC1 | Qwen c*: AMI, NMI, ARI, ACC at the class count; AMI at 20; three aspects | TPIPS, k-means | ArtELingo held, MMArt test |
| SC2 | Qwen c*: mAP, picked read-out, three aspects | TPIPS, its pick | both |
| SC3 | CLIP L/14 c* and Qwen c*: as SC1 | CLAY, k-means | both |
| SC4 | CLIP L/14 c* and Qwen c*: as SC2 | CLAY, its pick | both |
| SC5 | X-c* + TURTLE: as SC1, four encoders | X plain + TURTLE | both |
| SC6 | X-c* (k-means): as SC1, four encoders | X plain + TURTLE | both |
| SC7, SC8 | as SC5, SC6 with TEMI, within stage 3's scope | X plain + TEMI | both |

Cells: SC1 29 (on MMArt style the class count is 20), SC2 6, SC3 58, SC4 12, SC5 and SC6 116 each, SC7 and SC8 up
to 116 each.
- **Licensed sentences.** WIN: "On <set>, c* on <X> clusters (retrieves) paintings by <aspect> better than <arm>
  (<metrics>, <K> clusters)." LEVEL: "... level with <arm>." BELOW: no claim for c*; if the upper bound is also below
  0, "<arm> clusters (retrieves) paintings by <aspect> better than c* on <X>." SC5: "With TURTLE (its released code's
  default settings, without its hyperparameter selection) on both sides, c* ... better than the plain <X> image."
  SC6: "c* on <X> clusters paintings by <aspect> better than the plain <X> image clustered with TURTLE at its released
  code's default settings, without its hyperparameter selection." SC7 and SC8: the same with TEMI at its paper's
  Table 10 settings. CLAY's text set may differ between tasks and datasets.
- **Not licensed:** counts other than the class count and 20; TPIPS or CLAY on other encoders; TURTLE with selected
  settings; sets not scored. **Selection:** stage 1 only (D7, D8); stage 2 selects nothing. **Inconclusive:** a
  failed gate voids what §3.5's table says.

## 7. Constitution check

| Principle | Pass or exception | Note |
|-----------|-------------------|------|
| C1 strongest comparator | pass | two outside methods; the plain image with stronger clustering |
| C6 sample IDs | pass | TPIPS stores keyed at extraction; joins by painting name or key (gate (iv)) |
| C7 rule before code | pass | §6 with §3.4 and §3.5, reviewed by a fresh Opus `xhigh` worker before the approving commit |
| C8 prior and disclosure | pass | prior §3.1; disclosure §8 |
| C12 regression check | exception | stage 0 replays reused paths before any candidate number; a failure voids cells and the run goes on (below) |
| C13 re-derivation | pass | §3.4 |
| `held-data.md` | pass | picks on validation; the held sets select nothing; CLAY's mean image and the TURTLE and TEMI fits (TEMI's head choice and neighbour graph too) read held images without labels, as k-means does; our TURTLE reads no label |

Exceptions: C12. It breaks the stop on a difference: a failed gate voids cells (§3.5), is rerun after a traced fix and
is reported, but the run goes on. Why: the run is unattended (D17, D24). Rejected: stopping the whole run, which idles
every arm that does not depend on the failed gate.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|---|---|---|
| Our CLAY or TURTLE differs from the official code | gates (i), (ii) | trace; no cell of that method until it passes |
| TURTLE at code defaults is under-tuned, so SC6 is easier than with the paper's selection | a cell within c*'s range; few non-empty clusters | the report says so; the selection is a new spec |
| TURTLE or TEMI leaves clusters empty at K 27 or 40 | §3.6 fit counts | reported; the partition is scored as is |
| CLAY's lists name the classes (for example a style name) | §3.6 verbatim matches | listed beside the cells (D22) |
| The margin from c*'s k-means ranges is too narrow or wide for TURTLE cells | the seed spread of c* + TURTLE beside | reported; not changed after results |
| node402 not ready or busy (r19 may use it) | the build's node check | local 3090 at night (about 7 h, ending by 10:00; no long heavy local run on weekdays 10:00 to 19:00) |
| Chance WIN or BELOW in near-tie cells: 337 labelled stage 2 cells (up to 232 more in stage 3), no multiplicity correction | counts per SC; d inside seed ranges | the report gives the counts |
| TURTLE and TEMI are fitted on the scored set, not on train and predicted on test as in their papers | none (by design) | one line in the report |
| TPIPS saw WikiArt paintings in training (not checked) | none | the report says so beside the TPIPS cells |
| GPU or CPU contention with r17 and r19 | lock and `uptime` | wait for the lock; fewer processes |
| Selection disclosure: on the held sets nothing of TPIPS, CLAY, TURTLE or TEMI was seen (the probe read 48 MMArt test images for timing and factor cosines only); c*'s AMI at 20, mAP and (via r17) its class-count values were seen. Validation carried r10's 59 runs and r11 to r17's picks; stage 1 adds 30 CLAY and 25 TPIPS read-outs per validation split (60 and 50 in all), and one text-set comparison per dataset and task | inflated validation numbers | only held numbers are results; labels by the fixed rule |

## 9. Deliverables and review

- At the end: the round folder (prefix `om_`) with code, tests, log and records; outputs and stores as D18; TEMI's
  copy with its licence and notice; the auto report in `docs/reports/auto/percept/` (next sequence date at report
  time) with its results brief, index row, a user-read report and the paper-change section (`docs/paper/
  cvpr2027_aspect_spaces/drafts/v1/`: related work, method, experiments); a short second report for TEMI.
- Reviews: this rule by a fresh Opus `xhigh` worker before the approving commit; tickets per `matt-chain.md`, with
  CLAY, TURTLE, TEMI, the picks and the label code as verdict code; the C13 re-derivation; the final whole-branch
  review.
- Stop points: none between approval and the reports. Time-box: the stage 0 to 2 and 4 report by 2026-11-06 18:00;
  TEMI's by 2026-11-15 18:00 for the paper, else by 2026-11-21 for the supplementary.
- Budget: TPIPS 6.4 GPU h on the 3090 or about 2.5 h of wall time on three A6000s, plus a 32 GB weight copy and the
  image sync; text set B minutes of local GPU; TURTLE about 410 fits, local GPU through the shared queue (`res-queue gpu`), about 2 to 14 h at 20 s to
  2 min per fit (not measured); TURTLE and TEMI fits over about 200 W run outside weekdays 10:00 to 19:00; scoring about 6 to 10 CPU h at 8 threads, about 4 h of wall time; C13 about 2 h. TEMI:
  up to 10 GPU h. Disk: TPIPS stores about 1.7 GB on SSD2; partitions and references under 1 GB.

Body length: about 280 non-blank lines (many are long table rows) against a claim-test target of about 150 plus the
rule: four methods over two tasks, four encoders and two sets need the per-method arms, gates, margin sources and
TEMI's fixed scope list.
