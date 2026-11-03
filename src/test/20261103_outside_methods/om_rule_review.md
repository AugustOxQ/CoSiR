# r18 rule review: outside methods beside c* (spec 2026-11-03-outside-methods-design.md)

Reviewer: Opus worker, constitution C7 rule review, read only · Date: 2026-11-03 20:15 · Spec state: untracked
draft of 19:55

**Verdict: approve with fixes.** The four blocking findings must be fixed before the approving commit. Each one has a
short replacement text below. Then there are 8 should-fix and 15 minor findings.

Read: the spec from §1 on and its brief; the r17 spec (§3.3, §3.4, §3.7, D2, D3, D7); the constitution; the facts
brief A to E; the round log; the hard rules of `spec_core.md`; the type table and parts table of `spec_experiment.md`;
`held-data.md`. Checked against files:
- the r14 spec §3.2 (part a trains SigLIP 2, SigLIP 1 and CLIP L/14 only) and r14's `records/` (no Qwen stage 1 file);
- r11's `stage0a_ranges.json` (Qwen's ArtELingo mAP ranges 0.00141/0.00223/0.00382);
- r17's inventory `ns_saved_outputs.md` line 163 (Qwen's ArtELingo AMI-at-20 range "not located");
- r15's `records/verdicts.json` (Qwen MMArt SC1 cells exist); the glossary's seed-range row;
- reports 2027-01-02 (CLIP L/14 c* emotion mAP 0.2671) and 2026-12-29 (CRL 0.2554).

No code was run. Not checked: the CLAY, TURTLE and TEMI code lines (the facts brief was taken as given), r17's scorer
interface (not built yet), node402's state.

## What holds (checked, no change needed)

- **Cell counts.** SC1 has 29 cells: 3 aspects × 2 sets × 5 metric cells, minus MMArt style's AMI at 20, which is its
  class-count cell. SC2 has 6, SC3 58, SC4 12, SC5 and SC6 116 each (4 encoders × 29). All correct.
- **TURTLE fits.** The count is 412 if one plain-image partition per encoder, set, K and start serves all three
  aspects. The plain image needs 4 encoders × 2 sets × 8 distinct K × 3 starts = 192 fits. The distinct K are
  {5, 8, 9, 10, 16, 20, 27, 40} on both sets. c*'s heads need 4 × (6 ArtELingo + 5 MMArt) × 5 seeds = 220 fits. "About
  410" is right, but the text does not say that the plain partitions are shared (minor m3).
- **Intervals per arm type.** TPIPS and CLAY are unseeded: k-means states 0 to 2 are averaged and only rows are
  resampled. Plain + TURTLE averages its 3 starts as states, rows only. c* + TURTLE has one start per c* seed, and the
  seeds are drawn as c*'s. All of this matches r17 §3.4.
- **Label rule.** WIN, LEVEL and BELOW are as in r17. They are well defined once r is defined for every cell (B2).
- **Held data.** The "like k-means" argument is sound. CLAY's alignment, the TURTLE fit and the TEMI fit read held
  images without labels and choose nothing by result. k-means in r10 to r17 does the same, and it also picks the best
  of 3 inits by inertia on the held rows. §3.3, D6 and §7 state this the same way. Two label-free parts are not named
  (m4).
- **TPIPS's read-out pick is complete.** It uses r11's menu at width 4,096 (25 read-outs) and r11's guard, score, tie
  band and order. The units are Qwen c*'s validation mAP ranges, with one pick per set.
- **D14 (agent default) is acceptable.** D14 uses c*'s k-means ranges as the margin of the TURTLE cells. c* + TURTLE
  has one start per seed, so its seed spread is likely wider than the k-means range. A narrower margin makes LEVEL
  harder for c*, so the error goes against our claim. The §8 row puts the spread beside each cell.
- **Parity design.** CLAY's math is deterministic and gets the same text vectors on both sides, so (i) is workable.
  TURTLE on CPU is workable if the reference and our code use the same torch build and weight-norm form. Gate (ii)
  has one gap (S2).

## Blocking

### B1. D7 and §3.4 "Retrieval": the rule compares CLAY's two text sets on scores that are not comparable

Problem: D7 picks one read-out "per text set, then the set with the higher rule score". r11's rule score is
(1/3) Σ_a (mAP_a − mAP_a(R0)) / r_a, the gain over the same arm's R0 (facts brief D, `rt_readout`). Each text set has
its own R0. Set B can have a lower R0 and a larger read-out gain, and then it beats set A even when set A's mAP is
higher at every read-out. So the rule can carry the weaker CLAY comparator (C1).

Fix (D7, the retrieval clause): "for retrieval, r11's rule picks one read-out per text set, with the guard against
that set's own R0. Then the set whose pick has the higher (1/3) Σ_a mAP_a(pick) / r_a on validation is used (r_a:
CLIP L/14 c*'s validation mAP ranges on that set); ties within 0.001 go to A."

### B2. §3.4 and §6: the LEVEL margin r is undefined for Qwen c* on ArtELingo at 20 clusters

Problem: §3.4 says r comes from "r14's and r15's at 20 and in retrieval". r14 trained only SigLIP 2, SigLIP 1 and
CLIP L/14, and its records hold `stage1_{siglip2,siglip1,clip_l14}.json` only. No AMI-at-20 range exists for Qwen c*
on the 6,152 ArtELingo validation paintings. r17's inventory did not find one (line 163). The glossary values
0.0046/0.0082/0.0128 belong to an older model (r17 D7). Qwen's ArtELingo retrieval range is r11's, not r14's.

Cells without a defined r: the ArtELingo AMI-at-20 cell of each aspect in SC1, SC3 (Qwen c*), SC5 and SC6 (Qwen).
That is 12 labelled cells, plus up to 6 in SC7 and SC8. As written, the run would have to choose r.

Fix:
- (a) Stage 1 adds: "Qwen c*'s AMI-at-20 seed range on ArtELingo validation, per aspect. k-means as r10 on r10's
  validation exports (k11=40, seeds 0 to 4); the mean over states 0 to 2 per seed; maximum minus minimum, unrounded;
  committed with the picks."
- (b) Replace the r sentence of §3.4 with: "r is the reference c*'s validation seed range for that set, aspect, metric
  and K. At the class count: r17's stage 1 ranges (MMArt style's AMI: r15's). AMI at 20: r14's for SigLIP 2, SigLIP 1
  and CLIP L/14 on ArtELingo; r15's for the four encoders on MMArt; stage 1's for Qwen on ArtELingo. mAP at R0: r11's
  for Qwen on ArtELingo; r14's for the other encoders on ArtELingo; r15's on MMArt."
- (c) Void row: "stage 1: a range missing | a cell of that range that is not WIN is inconclusive".
- (d) Optional: add a column "r from" to the §6 table.

### B3. §3.5 gate (iii) and C12: the replay leaves three reused paths unchecked

Problem: C12 requires each reused path to reproduce its recorded numbers before any candidate number exists. Gate
(iii) replays only clustering, with k-means, on CLIP L/14 (MMArt) and SigLIP 2 (ArtELingo). These paths have no
replay:
1. Retrieval: r11's protocol through r17's scorer, with our loaders and the recorded c* picks. It serves SC2 and SC4
   (18 cells).
2. Qwen c*'s exports: r10's stage 4 store, which is a different store from r14's. It serves SC1 to SC6.
3. A partition given from outside (TURTLE, TEMI), with 3 starts averaged for an unseeded arm. It serves SC5 to SC8.

The void table has no row for any of them.

Fix: replace (iii) with: "(iii) Path replay within 1e-9, with our loaders feeding r17's scorer.
(a) k-means at 20: the AMI values and intervals of r15's CLIP L/14 and Qwen MMArt SC1 cells and of r14's SigLIP 2
ArtELingo SC1 cell; and r10's stage 4 Qwen held partitions at 20, exactly.
(b) The partitions of (a), fed back through the partition input used for TURTLE and TEMI (3 partitions for the
unseeded arm), give the same values and intervals.
(c) Retrieval: r15's CLIP L/14 and Qwen MMArt SC2 values and intervals, and one of r11's stage 5 cells of Qwen c*
against the plain image (ArtELingo held), each at its recorded pick."

New void rows: "(iii) a | every clustering cell"; "(iii) b | SC5 to SC8"; "(iii) c | SC2, SC4".

### B4. §3.5 scope rule (D10): TEMI's scope is not fixed, so the run decides which cells exist

Problem: "cells then run in this order until 10 GPU h are used" orders three groups only. Three points are open:
- the order inside a group (encoders, sets, K, starts);
- a group cut midway leaves cells with fewer than 3 starts or 5 seeds;
- nothing stops the scoring of finished cells before the cut.

The cut likely falls inside the first group. That group holds 84 fits: 4 encoders × 7 distinct K (class count and 20,
both sets) × 3 starts. The second group holds 220 fits. The TEMI paper reports about 45 minutes on an A100 for 50 heads
on 50,000 CIFAR-100 images (facts brief B). Scaled to 12,281 to 16,384 rows, that is about 11 to 15 minutes per fit.
At that rate, 10 h gives about 40 to 55 fits (our estimate, not measured). SC7 may then get no cell, and the order the
rule leaves open decides which SC8 cells exist. TEMI is built after TURTLE's labels exist (D17), so a choice made then
can follow those results.

Fix (replace the scope rule): "Scope rule (D10). The timing run (Qwen plain, ArtELingo held, K 20, start 0; no
metric is computed from it) gives t per fit. N = floor(10 h / (t × 16,384 / 12,281)). Units run whole, in this order,
until the next unit would take the total over N fits:
(1) the plain image at the class count and 20: one unit per encoder (Qwen, SigLIP 2, SigLIP 1, CLIP L/14), each with
both sets, all its K and starts 0 to 2;
(2) c*'s heads at the class count and 20: one unit per encoder in the same order, each with both sets, three aspects
and seeds 0 to 4;
(3) the plain image's other counts: one unit per encoder.
The list of units goes into the round log and is committed before the first TEMI fit. No TEMI cell is scored before
every listed fit is done."

The brief should also say that SC7 may get no cell.

## Should fix

### S1. §3.4 CLAY, D6, D7, D8: CLAY's mean image and the pick granularity leave choices open

Problem: in retrieval, "the scored set's normalized mean image" can mean the mean over all rows of the split, or over
the aspect's labelled rows only. For genre, the labelled rows are 9,949 of 12,281 held and 4,960 of 6,152 validation.
The second reading uses which rows have a label. For the validation picks it is also open whether the mean comes from
the whole split or from each cross-fit half. D7 and D8 do not say that the text set is picked per dataset and per task
(up to four picks). D8 also does not say how its AMI at 20 is computed.

Fix:
- §3.4 CLAY, after "(D6)": "The mean is over all rows of the split being scored, whatever their labels (ArtELingo
  validation 6,152, MMArt validation 5,769, ArtELingo held 12,281, MMArt test 16,384). One CLAY vector per row and
  aspect serves clustering, retrieval and both cross-fit halves."
- D8: "... by the mean over aspects of validation AMI at 20 (k-means as r10, mean over states 0 to 2), per dataset;
  ties within 0.001 go to A."
- D7: add "per dataset". §6: add one line, "CLAY's text set may differ between tasks and datasets".

### S2. §3.5 gate (ii): the TURTLE parity input never checks the batch sampling that every real fit uses

Problem: the reference input has 2,000 rows, and the batch is 10,000 capped at the set. So every outer step uses the
full set. The real fits have 12,281 and 16,384 rows, so each step samples 10,000 rows. The random draw of batch
indices is the most likely place for an order mismatch, and it is never checked. Three more points are open: which
weights "within 1e-5" applies to (weight-norm g and v, or the effective matrix; the inner classifiers?), which norm
it uses, and TF32 on the 3090, which could move the GPU result.

Fix: "(ii) TURTLE parity, on the reference input: the first 2,000 ArtELingo validation paintings in sorted name order,
plain Qwen, K 9, batch 500, 100 outer steps, CPU, seed 0. Our final effective task-encoder weights equal the official
code's within 1e-5 (largest absolute difference), and the partitions are identical. Our GPU run of the same input,
with TF32 off as in every fit, gives a partition with AMI ≥ 0.99 against the CPU one."

### S3. §3.5 references, D12, brief Plan 1: when, where and how the references are made

Problem: §3.5 says the references are made "in the decide chat", and the brief says "in this chat". D12 was decided in
the spec chat at 19:45, after the decide chat closed at 19:20. Neither official code runs unchanged (facts brief A, B):
- CLAY has no CLIP ViT-L/14 loader and breaks on the output containers of transformers 5.6.2. It encodes text inside
  `make_subspace` and saves no conditioned matrix.
- TURTLE needs a dataset-to-K entry and a new representation name.

The rule does not say what the reference driver may change, or which dtype and device CLAY's reference uses.

Fix: "The official code makes the references for (i) and (ii) once, attended, in the r18 spec chat, before the
approving commit (else before the build's first ticket). The driver may only:
(a) feed our stored vectors (CLAY: the text-encoder call returns r11's set A text vectors, so the lines of
`make_subspace` itself run; images come from the CLIP L/14 store);
(b) add a dataset name, K and a representation name;
(c) save outputs.
No line of the math changes. The driver, its diff against the pinned commit, the input keys, the dtype and device
(float32, CPU) and the output hashes are kept beside the references in `/data/SSD2/cosir_r18/ext/` and are named in
the round log."

### S4. §3.5 void table, stage rows and §7 C12: missing rows, one conflict and C12's stop

Problems:
- Stage 1's row says "the arm's cells are inconclusive". The void table says "that arm's retrieval cells". CLAY's
  text-set pick for clustering (D8) has no void row.
- TEMI's checks (the diff review and the toy run) have no void row.
- r17's class-count ranges have no row. The parts of r17's scorer that r17's own stage 0 voids have no row either.
- §8 row 1 says "no cell of that method until it passes", which implies a retry. The void table reads as final. So the
  retry policy is open.
- C12 says that a difference "stops the work and goes to the user with its cause traced". The spec voids the cells and
  goes on, but §7 says "pass". The constitution does not allow a quiet reinterpretation.

Fix: add these void rows:
- "stage 1: CLAY's clustering text set missing | SC3";
- "stage 1: a read-out or CLAY's retrieval text set missing | that arm's SC2 or SC4 cells";
- "r17: a class-count range missing | a cell of that range that is not WIN is inconclusive";
- "r17's stage 0 voided a part of its scorer | the same cells here";
- "TEMI diff review or toy run | SC7, SC8".

Below the table, add: "A failed gate may be rerun after a traced code fix, logged, before any cell of that method is
computed. A gate that still fails at the time-box voids as above. The report gives the user each failure and its
cause." Then either list C12 in §7 as an exception (what it breaks: the run goes on; why: an unattended run, D17; the
simpler option rejected: stopping the whole run idles every other arm), or follow C12 as written.

### S5. §6 "The existing claim": a family verdict that no one decided, with overlapping words

Problem: "holds if every cell is WIN; qualified if some are LEVEL or BELOW; weakened if half or more are BELOW". A
family with half of its cells BELOW is both qualified and weakened. "Family" is not defined: SC5 alone, SC6 alone, or
SC5 to SC8 together. Void cells and TEMI's partial scope have no place in the count. No §4 row decides this paragraph,
and it is a verdict per family, while D11 (user) says labels only and no verdict.

Fix: delete the paragraph (recommended). The per-cell labels and the grill's "phrased per method" scenario already say
how the paper uses the labels. If it stays, it needs an agent-default D row and this text: "Per SC (SC5, SC6, SC7 and
SC8 separately), over its decided cells: holds if every decided cell is WIN; weakened if at least half are BELOW;
qualified otherwise. Void cells are listed and not counted."

### S6. §3.4 TEMI, D10: TEMI's settings are not fully fixed, and TEMI is built after TURTLE's results

Problem: these settings are open:
- which warmup "warmup 20" means (learning rate or teacher temperature);
- the head shape (the parser default is 2 layers, hidden 512, bottleneck 256);
- the neighbour metric, and which rows the neighbour graph is built on;
- input standardization (TEMI saves `mean.pt` and `std.pt`);
- the teacher temperature schedule (the parser has a 30-epoch warmup, from 0.1 to 0.1);
- when the training loss for the head choice is read.

D17 builds TEMI after TURTLE runs, so the builder can make these choices after seeing TURTLE's labels.

Fix: give a value for each open item now, in §3.4 or D10. Where the paper's Table 10 is silent, use the copy's parser
default and name it. Example form: "head shape: parser default (2 layers, hidden 512, bottleneck 256); neighbours: 50
by cosine on the unit rows of the fitted set, self excluded; no input standardization; the head with the lowest mean
training loss over the last epoch".

### S7. D9, §6 SC5 and SC6 sentences: TURTLE at code defaults is weaker than TURTLE as published (C1)

Problem: TURTLE's paper chooses among 50 settings with a label-free criterion; the spec runs one setting. The brief
(Check these 4) and §8 flag this. But the licensed sentence of SC6 says "the plain <X> image clustered with TURTLE",
and a reader takes that to mean TURTLE as published. D9's code-default part is an agent default, so it is open to
critique.

Fix (minimum): the SC5 and SC6 sentences say "... clustered with TURTLE at its released code's default settings,
without its hyperparameter selection"; SC7 and SC8 say "TEMI with its paper's Table 10 settings". A cheap middle
option, if wanted: run TURTLE's label-free selection once on ArtELingo validation (plain Qwen, K 20, 50 fits) and use
the chosen setting everywhere. That is a sweep on development data and needs a D row.

### S8. Brief "Cost", §5, §8, §9: TURTLE's time and the local GPU window do not agree

Problem:
- At the assumed 20 s to 2 min per fit (§5), 412 fits take 2.3 to 13.7 h, not "about 1 to 4 h" (brief, §9).
- The TPIPS fallback "ending by 11:00" (§8) breaks the machine rule: long local GPU runs over about 200 W do not run on
  weekdays from 10:00 to 19:00, and 11-04 to 11-06 are weekdays.
- The power draw of TURTLE and TEMI is not known, but both are planned on the local GPU at any hour.

Fix:
- Brief and §9: "TURTLE about 410 fits, about 2 to 14 h at 20 s to 2 min per fit (not measured; the smoke times one
  fit and reads its power)".
- §8: "local 3090 at night (about 7 h, ending by 10:00)".
- §9: "TURTLE and TEMI fits over about 200 W run outside 10:00 to 19:00 on weekdays".

## Minor

- **m1. §6 table.** It has no bar or r column. Hard rule 3 asks for a numeric bar per SC. D11 (labels only) is the
  user's choice, so the numeric label rule stands in for the bar. Add: "The bar of every SC is §3.4's numeric label
  rule, with r from the column 'r from'" (see B2 d).
- **m2. §4.** The gate thresholds and inputs have no D row: 1e-5, 1e-9, AMI ≥ 0.99, cosine 0.999, ACC ≥ 0.95 on 10
  blobs, 2,000 rows, 100 outer steps, 48 images. Add one agent-default row.
- **m3. §3.3 or §3.4.** Add "one TURTLE (or TEMI) partition of the plain image per encoder, set, K and start serves
  all three aspects". Then the count 192 + 220 = 412 follows from the text.
- **m4. §3.3 and §7.** Name three more points. TEMI's head choice by training loss, and its neighbour graph, read the
  held rows without labels. Say that our TURTLE reads no label; the official code logs validation accuracy every 20
  steps. Fitting on the scored set departs from the TURTLE and TEMI papers' protocol (fit on train, predict on
  test); give this one line in §8.
- **m5. §3.5 (iv).** Say that all three factors are checked on the 48 probe images. The probe ran "emotion" only.
- **m6. §3.1.** "c* on CLIP L/14 beat CRL in r11's ArtELingo emotion retrieval (0.2671 against 0.2554)" mixes two
  rounds. 0.2671 is r14's CLIP L/14 c* held mAP (report 2027-01-02, SC6 row). 0.2554 is r11's CRL (report 2026-12-29).
  They were never tested as a pair. Write: "c* on CLIP L/14's held emotion mAP (r14, 0.2671) is above CRL's (r11,
  0.2554); no paired interval exists." r17 §3.1 has the same wording.
- **m7. Brief.** Three fixes:
  - "TPIPS about 6.4 GPU h (measured on the 3090)" is an estimate from the rate measured on 48 images, without image
    loading.
  - §3.1's "genre 0.6" prior is not in the brief.
  - The Outcomes field does not say what a failed check does.
- **m8. §8 selection disclosure.** Stage 1 adds 30 CLAY and 25 TPIPS read-outs per validation split (60 and 50 in
  all), plus the text-set comparisons (one per split and task).
- **m9. §3.4 C13.** Say what C13 recomputes and what it reads. Recomputed: CLAY's vectors from the CLIP rows and the
  text vectors (a second check of gate (i)), and k-means on TPIPS and CLAY with the same threads. Read: the TURTLE and
  TEMI partitions.
- **m10. D5.** Three points are open: what counts as a duplicate (for example, the same text after lower-casing and
  trimming); what happens when the model gives more than 100 values (keep the first 100); and what happens when set B
  fails or has fewer than 2 values (CLAY uses set A, with no pick).
- **m11. §3.2.** The template's column "Differs from the reference by" and the tuning-budget line are missing. c* was
  tuned on validation (r10's 59 runs); TPIPS and CLAY get read-out picks; TURTLE and TEMI get no tuning. One line is
  enough.
- **m12. §8.** No row covers the 337 labelled stage 2 cells (and up to 232 in stage 3) that have no multiplicity
  correction; r17 §8 has one. Add: "chance WIN or BELOW in near-tie cells | counts per SC; d in seed ranges | the
  report gives the counts".
- **m13. §8.** TPIPS's training images may include WikiArt paintings, and our held sets come from WikiArt. This is not
  checked. Add a risk row that says so.
- **m14. Brief "Check these" (hard rule 4).** D11 (labels only, no loop verdict) is a key decision that the user took
  after seeing the recommendation. Name it there, for example by merging it into item 3.
- **m15. §3.5 TEMI checks.** Name who reviews the diff (a fresh reviewer agent). Fix the toy run's settings before
  TEMI's build, for example: 10 blobs in 64 dimensions, 500 points each, K 10, seed 0.

## Removal list

None.

## Applied (2026-11-03 20:15, spec `2026-11-03-outside-methods-design.md`)

- **B1** applied with change: the controller's rule (mean over aspects of validation mAP divided by CLIP L/14 c*'s
  range on that set; ties within 0.1 of that unit to A) replaces the review's 0.001 tie band. D7 and §3.4.
- **B2** applied with change: stage 1 computes Qwen c*'s AMI-at-20 range from r10's validation exports (as the
  review's fix a); a source-of-r table in §3.4 replaces the review's sentence (b) and covers retrieval too; void row
  voids only the LEVEL word (WIN stands), as the controller decided. Sources checked: r14 records `artclu` and
  `artret_ranges` (three encoders), r15 `clu_ranges` and `ret_ranges` (four encoders), r11 `stage0a_ranges.json`.
  The "r from" column (d) is the table, not a column in §6.
- **B3** applied: gate (iii) (a) to (c) as the review's text; void rows (iii) a, b, c.
- **B4** applied with change: units follow the controller's definition (encoder × set × arm × K, all starts or seeds),
  not the review's per-encoder units; the full list (88 units, 412 fits, three groups) and its order are in §3.5.
  Whole units, no skipping, list committed before the first fit, no cell scored before all listed fits are done.
  The brief says SC7 may get no cell.
- **S1** applied (mean over all rows of the split; one vector per row and aspect; per dataset in D7 and D8; §6 line).
- **S2** applied with change: the references were already made with batch 10,000 capped at 2,000 (the full set), so the
  review's batch 500 is not used; the batch-sampling order is checked by the numpy draw check (controller item 6);
  effective weights, largest absolute difference and TF32 off are written in.
- **S3** applied with change: the references were made in the spec chat at 19:58; the text says so, and points to the
  driver and hashes in the round log instead of restating the driver's rules.
- **S4** applied: five void rows, the rerun sentence, and C12 listed as an exception in §7 (new rows D24).
- **S5** applied (paragraph deleted; "CLAY's text set may differ between tasks and datasets" line added).
- **S6** applied: every open TEMI setting has a value in §3.4 (learning-rate warmup 20 to 1e-4 then constant; parser
  head; 50 cosine neighbours, self excluded; no standardization; lowest mean last-epoch loss).
- **S7** applied (minimum): SC5 to SC8 sentences name the settings; D25. The selection run was not added.
- **S8** applied: TURTLE 2 to 14 h in brief, §5, §9; local fallback ends by 10:00; the 200 W weekday line in §9.
- **m1** applied (§6). **m2** applied (D23). **m3** applied (§3.3). **m4** applied (§3.3, §7; §8 row). **m5** applied
  (gate (iv)). **m6** applied (§3.1; r17 §3.1 not touched). **m7** applied (estimate wording; genre 0.6; failed check in
  Plan 1). **m8** applied. **m9** applied. **m10** applied (D5). **m11** applied (tuning line in §3.2; no new column).
  **m12** applied. **m13** applied (§5 and §8). **m14** applied (merged into Check these 3). **m15** applied.
- Also done: a TURTLE partition per K shared by the three aspects (§3.3, controller item 9); the numpy draw check in
  gate (ii) (item 6); §3.5's reference paragraph (item 5).
- Not applied: none. Not checked: r17's stage 1 may not include AMI at the class count; the table assumes it does
  (a missing range falls under the void row).
