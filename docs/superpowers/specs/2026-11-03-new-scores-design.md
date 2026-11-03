# New scores from saved outputs (standard metrics, cluster counts, CRL, untold and PercepT retrieval): design

> Date: 2026-11-03 19:30, revised after the rule review 19:50 · Status: approved 2026-11-03 20:45 (from the brief,
> r17 decide 2; restart no); the shared-queue rule of 20:45 applied to §8 and §9 after approval · Kind: experiment
> Builds on: decision A of 2026-11-03 18:30 (r16 report 3: the experiment overview's plan, loop 1 = r17) and the r17
> grill (r17 decide chat, 2026-11-03 19:00 to 19:20; its answers are §4) · Constitution:
> [constitution.md](../constitution.md) · Decision rule: inline in §6, with §3.4 and §3.5
> Inherits: the r15 spec ([2026-11-02-mmart-test-design.md](2026-11-02-mmart-test-design.md)): arms, populations, keys
> and order (§3.2, §3.3), clustering and retrieval protocol and outcomes (§3.4), D7, D9, D12, D16; through it the r14
> spec ([2026-11-02-backbones-sweep-design.md](2026-11-02-backbones-sweep-design.md)) §3.4 and SC1 to SC10 · Facts:
> the [saved-outputs inventory](../../../src/test/20261103_new_scores/ns_saved_outputs.md) (Sonnet worker, 2026-11-03
> 19:05) · Reviewer: a fresh Opus `xhigh` worker, 2026-11-03 19:40 (pass after fixes: five must-fix, eleven
> should-fix and seven nit items, all applied; M2 applied to SC8 only, see D19;
> [review](../../../src/test/20261103_new_scores/ns_rule_review.md))

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** Every claim so far uses one number per task: AMI at 20 clusters and mAP. Readers expect the
standard clustering metrics (NMI, ARI, ACC) at the class count, a view of how results move with the cluster count,
and three comparisons we lack: CRL on c*'s encoder, PercepT in retrieval on every encoder, and retrieval where the
query names no aspect ("untold"). This loop scores saved outputs only: no training, no GPU.

**Plan** (build and run unattended tonight; you read the report).
1. Stage 0 (checks): the new code reproduces r11's to r15's numbers exactly; PercepT's mappers replay on the CPU.
2. Stage 1 (sweep, validation only): c*'s seed ranges for the new metrics; read-out picks for untold c*, PercepT
   and CRL on MMArt. Committed before any new test number.
3. Stage 2 (claim test): every claim cell again at the class count in AMI, NMI, ARI and ACC; c* on CLIP L/14
   against CRL on CLIP L/14; untold retrieval; PercepT retrieval.
4. Stage 3 (descriptive, after the verdicts): AMI and ARI over the cluster counts (a figure); emotion AP per class
   against the teacher; Qwen c* against CRL on MMArt; told beside untold retrieval.

**Compared against.** Each class-count cell keeps the comparator and bar of its 20-cluster cell: the plain image and
PercepT need a WIN; the teacher a WIN where r14 and r15 set one, else LEVEL or WIN. CRL on CLIP L/14: WIN in every
cell. Untold c*: the plain image (WIN). c*'s retrieval: PercepT's topic probabilities (WIN).

**Outcomes.** Each metric cell is its own claim: a cell that meets its bar licenses its sentence; a cell that misses
gives no claim and keeps its row. A failed check makes only the cells it serves inconclusive. There is no overall
verdict; the paper cites cells that pass, by family. Other cluster counts get no verdict word. We expect WIN against
the plain image in most cells (AMI and NMI 0.85, ARI 0.75, ACC 0.65), against CRL in most (0.85 to 0.9), and untold
c* to WIN in style and genre (0.8), with even odds in emotion (0.5).

**Cost.** CPU only: about 13 to 20 CPU hours, about 5 to 8 h of wall time in three processes (our estimate, not
measured). Build tonight (about 3 h), run after it, CPU free before 11:00; report on 11-04 by 18:00.

**Check these.**
1. The CRL bar, a WIN in every cell: your pick after seeing ours (your first pick was 3b). Our default extends it to
   the new metrics at the class count.
2. Your picks after seeing ours: the minor batch (metric definitions, untold and PercepT retrieval, CRL on MMArt, the
   per-class look, the protocol, the folder).
3. Our defaults: each class-count cell keeps its 20-cluster bar; stage 2 is a claim test (the loop stays medium);
   PercepT at the class count uses only its k-means reading; in retrieval PercepT's seeds are drawn apart from c*'s.
4. A correction to the seed-range answer: Qwen's new ranges come from c*'s own validation seeds; the glossary values
   cannot check them (they belong to an older model).
5. Some test values were seen before these bars, and about 410 cells have no multiplicity correction (§8).

## 1. Context and goal

Asked: decision A (2026-11-03 18:30, r16 report 3 chat): items 1, 2, 7, 8, 9 and 11 of the experiment overview's
plan, CRL clustering on paintings and PercepT retrieval on SigLIP 2, SigLIP 1, CLIP L/14 and MMArt (direction log,
"Decided (A)"). On the cluster counts the user said: "more reasonable than the random number I suggest".

r10 to r15 license c*'s claims with AMI at 20 clusters and mAP: against the plain image and PercepT on four encoders,
on ArtELingo held and MMArt test, and per cell against the Qwen teacher; the photo model on Affection and EmoSet.
Readers and co-authors ask for the standard clustering metrics at the class count, the effect of the cluster count,
CRL on the same encoder, PercepT in retrieval and retrieval with an image-only query. All inputs are saved (inventory),
so the loop is scoring only. Its scorer is shared: r18 and r19 import it after it merges.

Goal: extend the evidence to standard metrics and the missing comparators, from saved outputs.
Question this step answers: do c*'s claims hold at the class count in AMI, NMI, ARI and ACC, does c* beat CRL on the
same encoder, and does it beat PercepT in retrieval and the plain image when the query names no aspect?

## 2. Scope

In: c* (seeds 0 to 4) on Qwen, SigLIP 2, SigLIP 1 and CLIP L/14; ArtELingo held and MMArt test (r15's cut set);
the photo models on Affection test and EmoSet test (emotion); the plain image, matched PercepT, the teacher, CRL on
CLIP L/14; the items of §3.6.
Out: any training or GPU work; intervals at the counts other than the class count and 20 (a later option if time
allows, user: a new spec or an amendment); CRL on other encoders; c* exports on the photo sets for SigLIP 2, SigLIP 1
and CLIP L/14; PercepT on the photo sets; TPIPS, CLAY, TURTLE and TEMI (r18); new encoders (r19); text queries (r22);
the qualitative figure (r20).

## 3. Design

### 3.1 Hypothesis and prior

Claim tested: c*'s gains over the plain image, PercepT and (per cell) the teacher at 20 clusters hold at the class
count in all four standard metrics; c* beats CRL on its own encoder in clustering and retrieval; c* beats PercepT in
retrieval; when the query names no aspect, c*'s joined heads retrieve by each aspect better than the plain image.
Mechanism: c*'s heads separate aspect classes in the embedding, so any partition-quality metric and any count near the
class count should show it; CRL's 50 value scores and PercepT's 40 topic probabilities keep less of the image than a
2,048-wide head. In untold retrieval the joined vector also carries the two other aspects, which dilutes the emotion
signal most (emotion is the weakest head). Facts: ArtELingo held emotion mAP, c* on CLIP L/14 0.2671 against CRL
0.2554 (about 7 seed ranges); PercepT on Qwen 0.2483/0.1523/0.5049 against c* 0.2691/0.3235/0.6151 (r11).
Prior (C8), per cell: against the plain image WIN 0.85 (AMI, NMI), 0.75 (ARI), 0.65 (ACC); against matched PercepT
WIN 0.75; teacher cells keep their AMI-at-20 outcome 0.7; CRL retrieval WIN 0.9, CRL clustering WIN 0.85; PercepT
retrieval WIN 0.9; untold against the plain image WIN 0.8 (style, genre), 0.5 (emotion). Surprise: an ACC or ARI
BELOW against the plain image where AMI wins.

### 3.2 Arms and controls

Per encoder X (Qwen, SigLIP 2, SigLIP 1, CLIP L/14); r15 §3.2 holds for the first four rows.

| Arm | What it is | Role |
|---|---|---|
| X-c* (told) | the aspect head of frozen c*, seeds 0 to 4 (Qwen: r10's k11=40; others r14's) | candidate (SC1 to SC3, SC5, SC6, SC8) |
| X plain image | X's default image | bar (SC1, SC7) |
| X matched PercepT | r15's matched arm (saved mappers, ArtELingo topics), seeds 0 to 4; at the class count only its k-means reading on L2-normalized probabilities (D16); in retrieval its 40 probabilities, L2-normalized (r11's form) | bar (SC2, SC8) |
| teacher | the Qwen image instructed per aspect (photo sets: the wording picked for its AMI-at-20 cell) | per-cell claims (SC3, SC4) |
| X photo model | r13's (Qwen) and r14's photo models, seeds 0 to 4 | candidate (SC4) |
| CRL on CLIP L/14 | r11's form and function: float64 cosines of the unit CLIP L/14 image with the aspect's 50 value texts (r11's texts, painting wording), L2-normalized; unseeded | bar (SC5, SC6) |
| X-c* untold | per seed s, X-c*'s three heads of seed s, each unit-normalized, joined (6,144 wide) and divided by √3; one vector for all aspects (D8) | candidate (SC7) |

### 3.3 Data roles

| Data | Role | Stages |
|---|---|---|
| ArtELingo validation, 6,152 paintings (genre 4,960) | development: class-count seed ranges; picks of untold c* (four encoders) and PercepT (SigLIP 2, SigLIP 1, CLIP L/14) | 1 |
| MMArt validation, 5,769 paintings (all among ArtELingo's 6,152; checked) | development: class-count seed ranges; picks of untold c* and PercepT (four encoders) and CRL | 1 |
| Affection validation 4,251; EmoSet validation 5,770 (r14's cut) | development: the photo models' class-count seed ranges | 1 |
| ArtELingo held, 12,281 (genre 9,949) | held: replays, claims, descriptive | 0, 2, 3 |
| MMArt test, 16,384 (r15's cut set) | held | 0, 2, 3 |
| Affection test 8,501; EmoSet test 17,348 (r14's cut) | held | 0, 2, 3 |
| Every train split | not touched | none |

Class counts K (labelled classes present, label code, 2026-11-03; the same classes on each validation split, checked
by the rule review): ArtELingo emotion 9, style 27, genre 10; MMArt emotion 9 ("something else" included), style 20,
genre 10; Affection 9; EmoSet 8. MMArt style's K is 20, so its AMI cells in SC1 and SC3 repeat r15's SC1 and SC4
cells; there only NMI, ARI, ACC and SC2's one-reading cell are new. Populations (r15 D16): k-means on every row of the
set, rows without the aspect's label masked after clustering; retrieval queries and gallery are the aspect's labelled
rows. Keys and order: as each set's earlier round (ArtELingo sorted painting names; MMArt and EmoSet sha256 of the key,
r15 D9 and r14 D17; Affection r12's).

### 3.4 Measurement

- **Class-count metrics (D6).** At K = the class count: AMI, NMI (arithmetic normalization) and ARI as scikit-learn;
  ACC from the contingency table of the K clusters and the K classes on the labelled rows (rows indexed first, then
  masked, repeats counted), with the Hungarian matching that maximizes matched rows, recomputed in every draw; a
  cluster with no labelled row counts zero; the denominator is the draw's labelled rows. k-means as r10 (MiniBatch,
  n_init 3, batch 4,096, states 0 to 2, unit rows); only K changes. Threads per set as its earlier round (D20):
  ArtELingo and MMArt 8 threads, Affection and EmoSet 1 thread, at every K; the replays, the count sweep and C13 use
  the same. Per arm: the mean over states, then seeds. CRL is clustered on its aspect vector.
- **Intervals.** Clustering: per cell 2,000 joint draws, a fresh `default_rng(0)` per cell; per draw the rows, then
  c*'s five seeds, then the comparator's five seeds if it is seeded (r14's `draw_bootstrap` order; an unseeded
  comparator resamples rows only). The four metrics of a cell read the same draws. Retrieval: r11's paired bootstrap,
  10,000 draws over queries and seeds; with a seeded comparator (SC8) its seeds are drawn apart from c*'s, after c*'s
  (D19); with an unseeded comparator the draws are r11's.
- **Outcomes.** WIN when the lower 95% bound > 0; LEVEL when not WIN and the lower bound > −r; BELOW otherwise; no
  multiplicity correction (r14 §3.4). r at the class count: the maximum minus minimum over the candidate's 5 seeds of
  that metric on the set's validation split (c*: ArtELingo or MMArt validation; photo models: Affection or EmoSet
  validation), unrounded (D7). r at 20 clusters (SC5): CLIP L/14 c*'s recorded AMI ranges (r14 ArtELingo, r15 MMArt).
  r in retrieval (SC6 to SC8): X-c*'s recorded told validation mAP ranges on that set (D17).
- **Retrieval.** r15 §3.4 (r11's protocol): relevance by the aspect label, mAP, picked read-outs. New picks (stage 1)
  by r11's rule on the set's validation split (cross-fit halves, the menu at the arm's width, guard, ties): one pick
  per arm for its three aspects, in units of X-c*'s recorded validation mAP ranges on that set (D17). New picks: X-c*
  untold (8), X matched PercepT (7: all but Qwen on ArtELingo), CRL on MMArt (1). Qwen's ArtELingo PercepT cell keeps
  r11's pick and stored probabilities; CRL on ArtELingo keeps r11's pick (`none+k10`); every other arm keeps its
  recorded pick (r11, r14, r15). Every new PercepT pick reads the CPU validation probabilities of stage 0 (iv); the
  MMArt test probabilities are r15's CPU outputs. Each map is fitted on all validation rows and applied to test;
  query expansion reads the test gallery with k fixed on validation (r11).
- **Untold retrieval (D8).** The joined vector ranks the aspect's labelled rows for each query; the ranking is
  scored with that aspect's labels. The plain image is ranked the same way with its recorded pick.
- **C13.** Separate code re-derives, from this rule and the saved exports, labels and stage 1 records, the class
  counts, the seed ranges, the new picks and every SC cell's value, d, interval and outcome, within 1e-6, before any
  verdict is written (Codex if `codex-usage --room` says room, else a Claude worker).

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|---|---|---|---|---|
| 0 | regression check | new scorer, metric, mapper, CRL, join and pick paths | (i) to (vi) below | void per the table below, trace, note it |
| 1 | sweep (development only) | the validation splits | every range and pick written and committed before any new test number | the cells it serves are inconclusive |
| 2 | claim test | the test sets, all frozen | SC1 to SC8 | report |
| 3 | descriptive | all | §3.6, after the verdicts | report |

- (i) Clustering replay within 1e-9, the stored inputs fed in their rounds' orders: (a) paintings: r14's SigLIP 2
  ArtELingo SC1 and SC6 AMI values and intervals; r15's CLIP L/14 MMArt SC1 and SC4 AMI values and intervals; k-means
  at K = 20 gives r10's stage 4 c* held partitions (`stage4/partitions/full/k11=40`) exactly. (b) The matched-arm
  interval: r14's SigLIP 2 ArtELingo SC3 d and interval (two readings, `b_matched`), and a unit test that the
  one-reading path equals it with the own reading removed (same draws). (c) Photos: r14's SigLIP 2 Affection SC4 and
  SC6 AMI values and intervals; r14's EmoSet SC7 and SC9 values and intervals. (d) K ≠ 20: r14's EmoSet K = 8 values
  (`sheet_emoset.json`) and r15's class-count diagnostic (MMArt test, K = 9 and 10, every arm, four encoders).
- (ii) Retrieval replay within 1e-9: r15's CLIP L/14 SC2 values and intervals; r11's stage 5 SC10 to SC15 values and
  intervals (c* against the plain and the instructed image); r11's stage 5 CRL and PercepT held mAP values (PercepT
  per seed) and their per-query AP arrays at their picks, CRL from the rebuilt vectors (r11 recorded no interval
  against CRL or PercepT); r14's CLIP L/14 SC2 c* mAP.
- (iii) Metrics: AMI, NMI and ARI equal scikit-learn and ACC equals a brute-force best permutation on test cases,
  within 1e-12; the class counts equal §3.3's.
- (iv) Mappers (r15 (iv) a): each of the 20 saved mappers, on CPU, gives its stored held probabilities within 1e-5;
  Qwen's on ArtELingo validation gives r11's stored validation probabilities within 1e-5.
- (v) Untold join: MMArt, the `keys` arrays of the three heads are equal; ArtELingo held, the sidecars' `list_sha256`
  are equal; ArtELingo validation, the row counts equal the sorted validation list; rows unit norm within 1e-3.
- (vi) Read-out rule: fed r11's stored validation outputs, it gives r11's picks and scores within 1e-9 (r15 (vi)).

| Failed gate | Voids |
|---|---|
| (i) a, painting scorer | SC1 to SC3 and SC5 |
| (i) b, matched-arm interval | SC2 |
| (i) c, Affection or EmoSet replay | that set's SC4 cells |
| (i) d, K ≠ 20 replay | every class-count cell |
| (ii) r15 SC2 or r11 SC10 to SC15 (scorer, bootstrap) | SC6 to SC8 |
| (ii) CRL rebuild | SC5 and SC6 |
| (ii) PercepT per-query AP | SC8 |
| (iii) a metric | that metric's cells in every SC |
| (iii) a class count | that set's and aspect's class-count cells |
| (iv) X's mapper | X's SC8 cells with a new pick; X's MMArt SC2 cells |
| (v) X | X's SC7 cells |
| (vi) | every cell with a new pick (SC6 on MMArt, SC7, SC8 but Qwen ArtELingo) |
| stage 1: a range missing | the LEVEL word of its cells; its teacher cells (SC3, SC4) inconclusive unless WIN |
| stage 1: a pick missing | that arm's cells |

### 3.6 Diagnostics

After the verdicts, deciding nothing:
- **Count sweep (D3).** AMI and ARI at emotion 5, 8, 9, 16, 20 (every set), genre 5, 10, 16, 20, style 10, 20, 27,
  40 (both painting sets), for c*, the plain image, matched PercepT (k-means), the teacher and CRL (CLIP L/14), four
  encoders; on the photo sets the photo model, plain image and teacher. Point values: the mean over 5 seeds and the
  minimum and maximum over seeds; a figure per set. A lower point value at a count is written "lower point value",
  with no verdict word (D5). If 11-04 12:00 passes with stage 3 unfinished, the photo sets run last and may be cut
  (listed as not run).
- **Emotion per class (D11).** AP per class, c* minus the teacher, four encoders, ArtELingo held and MMArt test;
  per-class paired intervals (10,000 draws over that class's queries and c*'s seeds), class sizes beside.
- **Qwen c* against CRL on CLIP L/14 (D10)** on MMArt test: clustering at 20 and at the class count, retrieval;
  paired intervals, no outcome word (r12's form).
- **Told beside untold:** told c* beside untold c* per aspect, untold minus told with its paired interval (one shared
  seed draw: untold seed s is built from told seed s's runs).
- R0 rows of the new retrieval arms; each class-count cell beside its AMI-at-20 cell; test seed spreads beside the
  validation ranges.

### 3.7 Frozen vs rerun

Frozen: every model, export and mapper; r10 to r15's picks, ranges and partitions; r11's CRL texts; stage 1's ranges
and picks before stage 2. Rerun: k-means at the new counts; PercepT validation probabilities on CPU (ArtELingo
validation; MMArt validation is its subset of 5,769); CRL vectors by r11's function, so (ii) is exact; untold joins;
all scoring.

Shared scorer (D13): the round's scoring module takes embeddings or partitions and labels and returns a metric table
(arm, aspect, K, metric, per seed and mean), and a paired cell (candidate, comparator, labels, metrics, K, draws, r →
d, interval, outcome per metric); retrieval cells wrap r11's functions in the same form. Its interface is frozen when
it merges into main; r18 and r19 import it read-only; a later need adds a function and never changes a signature.

## 4. Decisions

Grill rows: the r17 decide chat, 2026-11-03 19:00 to 19:20. Agent defaults: r17 decide 2, 2026-11-03 19:30; D20 and
the changes to D19 from the rule review (19:40).

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|----|----------|----------|----------------------------|------------|------|
| D1 | Items | 1, 2, 7, 8, 9, 11 of the overview's plan; CRL clustering on paintings; PercepT retrieval on the three other encoders and MMArt; scoring of saved outputs only, CPU | the plan's loop 1 (later loops) | user | 2026-11-03 18:30 |
| D2 | Cluster counts | emotion 5, 8, 9, 16, 20; genre 5, 10, 16, 20; style 10, 20, 27, 40 | "more reasonable than the random number I suggest" (5, 8, 16, 20 for all) | user (Q1 1a; ours the same) | 19:00 to 19:20 |
| D3 | Intervals | class count: AMI, NMI, ARI, ACC with intervals and outcomes in every claim cell; other counts: AMI and ARI point values (mean, min to max over seeds); AMI-at-20 claims keep their intervals; intervals at every count later if time allows | readers' standard set at one count (intervals everywhere) | user (Q2 2a; ours the same) | 19:00 to 19:20 |
| D4 | CRL bar | WIN in the 12 cells of c* on CLIP L/14 against CRL on CLIP L/14 (AMI at 20 and mAP; ArtELingo held and MMArt test; three aspects); a failed cell gives no claim and keeps its row; the report says the ArtELingo retrieval values were seen before; the class-count metric cells take the same bar by D15 | first pick 3b | user, after seeing the recommendation (Q3 3a) | 19:00 to 19:20 |
| D5 | Scenario checks | a lower ARI point value at 5 emotion clusters: "lower point value", no verdict word, the class-count claim stands; a LEVEL CRL clustering cell fails its bar, row kept | | user | 19:00 to 19:20 |
| D6 | Metric definitions (m1) | ACC and NMI at the class count only; ACC by Hungarian matching; NMI arithmetic; AMI and ARI as scikit-learn | ACC needs one-to-one matching | user, after seeing the recommendation | 19:00 to 19:20 |
| D7 | LEVEL margin (m2) | each new metric's seed range over the candidate's 5 seeds on the set's validation split, at the class count, per encoder; Qwen's too, from r10's validation exports | the glossary's 0.0046/0.0082/0.0128 are an older model's ranges (D2 of r10, 5,000 development paintings), so the check put to the user cannot work (correction, agent default) | user, after seeing the recommendation; correction agent default | 19:00 to 19:20; 19:30 |
| D8 | Untold retrieval (m3) | heads unit-normalized and joined (6,144), cosine; pick on validation by r11's rule; against the plain image with its pick; WIN per aspect; ArtELingo held and MMArt test, four encoders; told beside, descriptive | the form decided earlier on 11-03 | user, after seeing the recommendation | 19:00 to 19:20 |
| D9 | PercepT retrieval (m4) | 40 probabilities L2-normalized; validation probabilities on CPU; pick on validation; WIN per aspect; 24 cells (Qwen ArtELingo included with r11's pick, agent default) | r11's PercepT row (Qwen ArtELingo left out: its values were seen) | user, after seeing the recommendation; Qwen cell agent default | 19:00 to 19:20; 19:30 |
| D10 | CRL on MMArt (m5) | r11's value texts; pick on MMArt validation; Qwen c* against CRL on CLIP L/14 beside, descriptive | r12's form | user, after seeing the recommendation | 19:00 to 19:20 |
| D11 | Emotion per class (m6) | AP per class, c* minus teacher, four encoders, two sets, paired intervals, no verdict, after the verdicts | explains the emotion retrieval cells below the teacher | user, after seeing the recommendation | 19:00 to 19:20 |
| D12 | Protocol (m7) | inherited: k-means settings, 3 states, 2,000 and 10,000 draws, no multiplicity correction; replays first | comparability with r10 to r15 | user, after seeing the recommendation | 19:00 to 19:20 |
| D13 | Folder and scorer (m8) | `src/test/20261103_new_scores/`, prefix `ns_`; outputs `/data/SSD2/cosir_r17/20261103_new_scores/outputs/`; interface frozen at merge (§3.7) | r18 and r19 import it | user, after seeing the recommendation (outputs path agent default) | 19:00 to 19:20 |
| D14 | Stage types | 0 regression; 1 sweep (development only); 2 claim test; 3 descriptive; loop stays medium | stage 2 feeds paper claims (confirmation) | agent default | 19:30 |
| D15 | Bars of the class-count cells | each takes its AMI-at-20 cell's bar: plain image and PercepT WIN; teacher WIN in r15 D7's MMArt cells (emotion and style, four encoders) and EmoSet (r14 SC9), else LEVEL or WIN; CRL WIN at 20 and at the class count; each metric cell its own claim | one bar per cell family keeps the grill's answers (a bar per metric) | agent default | 19:30 |
| D16 | Matched PercepT at the class count | its k-means reading at K only; seeds drawn apart from c*'s | the own reading has up to 40 topics, not K; the arms are separate trainings | agent default | 19:30 |
| D17 | Pick and LEVEL units in retrieval | X-c*'s recorded validation mAP ranges on that set for every new pick and word of X's arms; reused picks keep their round's rule (r14 D19) | one unit per encoder, as r15 D12 (each arm's own range) | agent default | 19:30 |
| D18 | Gate scope | §3.5's table | a failure voids only what uses the failed part | agent default; table from the rule review | 19:30; 19:40 |
| D19 | Retrieval bootstrap with two seeded arms (SC8) | r11's paired bootstrap with PercepT's seeds drawn apart, after c*'s (as SC2 and r14 D16); told against untold keeps one shared draw (same runs) | the arms are separate trainings; a shared draw pairs seed s with seed s arbitrarily (r11's draw; the first draft's "same expected width" holds only on average over pairings) | agent default (rule review M2) | 19:40 |
| D20 | k-means threads | per set as its earlier round: ArtELingo and MMArt 8, Affection and EmoSet 1 | 1 against 8 threads moved r15's AMI by 1.3e-5, over the 1e-9 and 1e-6 tolerances | agent default (rule review M3) | 19:40 |

**Amendments**: none.

## 5. Assumptions and open questions

- Assumption: every input in the inventory reads as listed (shapes from headers; loaders assert sidecars and
  hashes). Checked: headers only; stage 0 reads them all.
- Assumption: the 20 mappers replay on CPU (r15 gate (iv) a passed on 2026-11-02). Stage 0 (iv) checks it again.
- Assumption: about 13 to 20 CPU hours (r15's C13 took 5,178 s for 60 cells, this round has 412; a k-means fit
  takes about 13 s at one thread on 12,281 × 2,048; the count sweep adds about 1,400 fits; untold picks run about 10
  eigendecompositions at 6,144 per arm). Not measured.
- Assumption: ArtELingo validation style has 27 classes, the smallest with 5 paintings. Checked (inventory, rule
  review); its class-count ranges may be wide.
- Not checked: r13's records, r12's Affection ranges and wording record, the contents of `stage2_emoset.json`; stage
  0 (i) c reads them.
- Open questions: none.

## 6. Success criteria

**Decision mode.** Binding rule: this section with §3.4 and §3.5, committed with this spec before any code.

| ID | What is measured | Compared against | Bar | On which data |
|----|------------------|------------------|-----|---------------|
| SC1 | X-c*'s AMI, NMI, ARI, ACC at the class count, three aspects | X plain image | WIN in each metric cell | ArtELingo held, MMArt test |
| SC2 | the same | X matched PercepT, k-means at the class count | WIN | ArtELingo held, MMArt test |
| SC3 | the same | the teacher | D15: WIN in r15 D7's MMArt emotion and style cells; LEVEL or WIN in every cell | ArtELingo held, MMArt test |
| SC4 | X photo model's emotion AMI, NMI, ARI, ACC at the class count | X plain image (WIN); teacher (Affection LEVEL or WIN; EmoSet, Qwen only, WIN) | as named | Affection test (four encoders), EmoSet test (Qwen) |
| SC5 | c* on CLIP L/14: AMI at 20; AMI, NMI, ARI, ACC at the class count; three aspects | CRL on CLIP L/14 | WIN | ArtELingo held, MMArt test |
| SC6 | c* on CLIP L/14: mAP, picked read-out, three aspects | CRL on CLIP L/14, its picked read-out | WIN | ArtELingo held, MMArt test |
| SC7 | X-c* untold: mAP per aspect, picked read-out | X plain image, its picked read-out | WIN | ArtELingo held, MMArt test |
| SC8 | X-c* told: mAP, picked read-out, three aspects | X matched PercepT, its picked read-out | WIN | ArtELingo held, MMArt test |

- **Per cell.** Each metric cell is its own claim: WIN licenses its sentence; else no claim, and the row stays.
  Teacher cells (SC3, SC4) as r15 SC4: a WIN-bar cell at WIN gives "better than", at LEVEL "level with or better
  than"; another cell at LEVEL or WIN gives "level with or better than"; BELOW gives no claim.
- **Inconclusive.** A failed gate voids what §3.5's table says; the rows that exist are reported.
- **Selection rules.** Stage 1 only: ranges, then picks (§3.4, D17). Stage 2 selects nothing.
- **Licensed claims.** SC1 to SC3: "On <set>, c* on <X> clusters paintings by <aspect> better than <the plain X image
  | PercepT on X, its k-means reading with as many clusters as classes> (or: than / level with or better than the
  Qwen image instructed for <aspect>) with as many clusters as classes (<K>), in <the metrics that pass> (5 seeds)."
  SC4: "On <Affection | EmoSet> test, the photo model on <X> clusters photos by emotion better than <the plain X image
  | the Qwen image instructed for emotion> (or: level with or better than) with as many clusters as classes (<K>), in
  <the metrics that pass> (5 seeds)." SC5: "c* on CLIP L/14 clusters paintings by <aspect> better than CRL on CLIP
  L/14 on <set> (AMI with 20 clusters; <metrics> with as many clusters as classes, <K>)." SC6: "c* on CLIP L/14
  retrieves paintings by <aspect> better than CRL on CLIP L/14 on <set> (mAP)." SC7: "When the query names no aspect,
  c*'s joined heads on <X> retrieve paintings of the same <aspect> better than the plain <X> image (mAP; read-outs
  picked on validation)." SC8: "c*'s <aspect> head on <X> retrieves paintings by <aspect> better than PercepT's topic
  probabilities on <X> (mAP)."
- **Not licensed:** any count other than the class count and 20; CRL on other encoders; untold against told; per
  class; aspects or sets not scored.

## 7. Constitution check

| Principle | Pass or exception | Note |
|-----------|-------------------|------|
| C1 strongest comparator | pass | the plain image, matched PercepT, the teacher, CRL on the same encoder, all as bars |
| C6 sample IDs | pass | keys from extraction; the untold join checks keys or list hashes (0 (v)) |
| C7 rule before code | pass | §6 with §3.4 and §3.5, reviewed by a fresh Opus `xhigh` worker before the approving commit |
| C8 prior and disclosure | pass | prior §3.1; disclosure §8 |
| C12 regression check | pass | stage 0 (i), (ii), (iv), (vi): every reused path, K ≠ 20 and the matched-arm draw included |
| C13 re-derivation | pass | §3.4 |
| `held-data.md` | pass | ranges and picks on validation only; the test sets are scored and select nothing; PercepT's class-count reading is fixed before any test number |

Exceptions: none.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|---|---|---|
| 412 cells with no multiplicity correction: chance WINs in near-tie cells | the outcome counts per family; d in seed ranges | the report gives the counts; the paper cites families, not single near-tie cells |
| ACC and ARI weigh large classes (ArtELingo emotion classes 153 to 4,244 paintings) | class sizes beside the cells; per-class AP | reported with the note; no reweighting |
| Class-count ranges from 5 seeds are small, so LEVEL is strict | ranges beside test seed spreads (§3.6) | reported; ranges are not changed after results |
| PercepT's k-means reading is weaker than its own reading, so SC2 is easier than r14's SC3 | the AMI-at-20 SC3 (larger reading) beside | SC2's sentence names the k-means reading |
| CPU contention with r18 and r19 slows the run | the run log's times; `res-queue status` | every CPU step goes through `res-queue cpu` (one slot of 8 threads each, so three processes take three of the four slots) and waits its turn; the count sweep's photo sets last (§3.6) |
| Selection disclosure: on test, AMI at 20 and mAP of the main arms (r10 to r15), class-count AMI on MMArt (r15) and EmoSet (r14, K = 8), CRL's and c* on CLIP L/14's ArtELingo mAP (r11, r14) and PercepT's Qwen ArtELingo mAP (r11) were seen; NMI, ARI, ACC, CRL clustering, CRL on MMArt, other PercepT retrieval and untold retrieval were not. Validation carried r10's 59 runs and r11 to r15's picks; stage 1 adds up to 25 read-outs per new arm (16 arms) | inflated validation numbers; bars set near seen values | bars come from the grill and earlier rounds (§4); only test numbers are results; the report names the seen cells |

## 9. Deliverables and review

- At the end: the round folder `src/test/20261103_new_scores/` (prefix `ns_`) with code, tests, log and records;
  outputs under `/data/SSD2/cosir_r17/20261103_new_scores/outputs/`; the auto report in `docs/reports/auto/percept/`
  (next sequence date at report time) with its results brief and index row; a user-read report; in the report, a
  short section on how much it changes the paper draft (`docs/paper/cvpr2027_aspect_spaces/drafts/v1/`: related
  work, method, experiments).
- Reviews: this rule by a fresh Opus `xhigh` worker before the approving commit; tickets per `matt-chain.md`, with
  the metrics, the bootstraps, the picks, the untold join and the outcome code as verdict code; the C13
  re-derivation; the final whole-branch review.
- Stop points: none between approval and the reports. Time-box: the report by 2026-11-04 18:00.
- Budget: CPU about 13 to 20 h at 8 threads per process (C13 the largest part, then the count sweep, then the untold
  picks), about 5 to 8 h of wall time in three processes; CPU use ends before 11:00 if the run starts by 01:00; no
  GPU. Each CPU step is its own `res-queue cpu --label r17:<step> --est <duration>` job (one slot; checks under
  15 minutes `--short`), queued step by step, never a whole script; the k-means thread counts of D20 are set in code
  (the queue sets 8 threads per slot). At 20:45 the queue was idle and DAS6 had no waiting launches; r18 and r19 run
  mainly on GPUs (local and DAS6), which take no CPU slot. Disk: about 2 GB on SSD2 (partitions, per-query AP, PercepT validation probabilities).

Body length: about 235 non-blank lines against a claim-test target of about 150 plus the rule: eight families of
cells over four sets and four encoders need the per-family bars, the reused and new picks, the replays per path and
the gate scope.
