# r17 rule review: new scores from saved outputs (spec 2026-11-03-new-scores-design.md)

Reviewer: Opus worker, constitution C7 rule review, read only · Date: 2026-11-03 19:40 · Verdict: **pass after fixes**
(5 must-fix, 11 should-fix, 7 nits). The review was cut at the 200k context meter. Checked: the spec from §1 on, the
brief, the inventory, the decide-2 handoff, the r14 and r15 specs, the templates, the constitution, the records and code
named below. Not checked: r13's records, r12's Affection ranges and wording record, `stage2_emoset.json` contents. No
test code was run. Facts were checked with two read-only scripts (scratchpad).

## Must-fix

**M1. §3.5 (ii): the retrieval replay names intervals that do not exist.**
Problem: "(ii) … r11's stage 5 c*, teacher, CRL and PercepT held mAP values and intervals". r11 stage 5 recorded
intervals only for c* against the plain image and against the instructed image (SC10 to SC15). For CRL and PercepT it
recorded only mAP point values (mean, and per seed for PercepT) and per-query AP files. As written, the gate cannot pass.
Evidence: `/data/SSD2/cosir_r11/20261029_retrieval/outputs/stage5/verdict.json` key `sc` holds SC10 to SC15 only
(comparators `plain`, `instructed`); key `map` gives `crl` 0.25543…, `percept` 0.24833… with `per_seed`;
`stage5/perquery/crl__{aspect}__h{aspect}__s-.npz` and `percept__{aspect}__s{0..4}.npz` exist.
Fix: "(ii) Retrieval replay within 1e-9: r15's CLIP L/14 SC2 values and intervals; r11's stage 5 SC10 to SC15 values
and intervals (c* against the plain and the instructed image); r11's stage 5 CRL and PercepT held mAP values (PercepT
per seed) and their per-query AP arrays (`stage5/perquery`) at their picks, CRL from the rebuilt vectors (r11 recorded
no interval against CRL or PercepT); r14's CLIP L/14 SC2 c* mAP."

**M2. D19 and §3.4 "Intervals": the shared seed draw for SC8 rests on a wrong reason and contradicts D16.**
Problem: D19 says "the replays need it". No recorded retrieval interval has two seeded arms: r11's SC10 to SC15 and
r15's SC2 compare against unseeded arms. `rt_stats.paired` draws the seed vector S only when a side is seeded. So a
second, independent S for a seeded comparator changes no replay. The spec also uses two rules for the same pair of
separately trained arms: SC2 draws PercepT's seeds apart from c*'s ("the arms are separate trainings", D16, r14 D16),
but SC8 pairs c*'s seed s with PercepT's seed s. On "its expected width equals independent draws": this holds only on
average over random pairings. The seed covariance term Σ(c_i − c̄)(p_π(i) − p̄) averages to zero over permutations π.
For the one fixed pairing s↔s it does not hold. With 5 seeds, the realized covariance can make the interval narrower
or wider.
Evidence: `20261029_retrieval/rt_stats.py:45-85` (I, then S "shared by both sides", only `if seeded`);
`20261027_relational_genre/rg_held.py` `draws_d_r` (rows, then `s_g`, then `s_r`: independent);
`stage5/verdict.json` `sc` comparators.
Fix (D19 row and §3.4): "Retrieval with two seeded arms (SC8; the told-untold and per-class intervals of §3.6): r11's
paired bootstrap with the comparator's seeds drawn apart: per draw, I, then c*'s five seeds, then the comparator's five
seeds (as SC2 and r14 D16; the arms are separate trainings). With an unseeded comparator the draws are r11's, so the
replays hold; a unit test checks that." Why column: "the arms are separate trainings; a shared draw pairs seed s with
seed s arbitrarily (r11's draw, rejected)". If the shared draw is kept instead, rewrite the reason: "r11's form; its
seed-variance term equals that of independent draws only on average over pairings; this pairing may narrow or widen
the interval". Also list D19 under "Check these".

**M3. §3.4 "Class-count metrics": the k-means thread count is not fixed, and it changes partitions beyond both
tolerances.**
Problem: "k-means as r10 … only K changes" leaves the thread count open. The earlier rounds used two routes. ArtELingo
(r10, r14) and MMArt (r15) fit k-means at 8 threads: `ct_score.output_partitions`, `threadpool_limits(limits=8)`.
Affection (r12) and EmoSet (r14) fit at 1 thread: `rs_vh.r20_partition` and `bs_emoset_score.partitions_k`. r15's log
shows the effect. With one thread, 212 of 16,384 assignments changed (seed 2, state 1), and the emotion AMI moved by
1.3e-5. That is over C13's 1e-6 and over the replays' 1e-9.
Evidence: `20261020_claim_test/ct_score.py:101`; `20261027_recipe_sweep/rs_vh.py:246`;
`20261102_backbones_sweep/bs_emoset_score.py:215-225`; `20261102_mmart_test/mt_diag.py:159-169` (r15's class-count
diagnostic used `ct_score.partitions(x, k)` at 8 threads); `20261102_mmart_test/20261102_mmart_test_log.md`, 02:02
entry.
Fix (append to the k-means bullet): "Threads per set, as its earlier round: ArtELingo and MMArt 8 threads (r10's,
r14's and r15's route), Affection and EmoSet 1 thread (r12's and r14's route), at every K; the replays, the count
sweep and C13 use the same counts."

**M4. §3.5 (i): load-bearing paths have no replay (C12).**
Problem: three reused paths decide cells but no stage 0 gate replays them, although their records exist:
(a) the Affection photo path (loader, r12's key order, the artwork wording; 32 SC4 cells);
(b) the matched arm's bootstrap with independent seed draws (96 SC2 cells). (i) replays only the 20-cluster reading
(a point value), not r14's SC3 d and interval;
(c) k-means at K ≠ 20 (every class-count cell). (i) checks only K = 20.
Evidence: r14 `records/verdicts.json` `cells`: SC4 SigLIP 2 Affection d 0.078173 [0.072415, 0.084456]; SC6 SigLIP 2
Affection clustering d 0.025795 [0.020069, 0.031629]; SC3 SigLIP 2 emotion d 0.025262 [0.019781, 0.026655] with
`comparator_readings`. r14 `outputs/stage3/sheet_emoset.json` has `ami_k8` (photo_qwen 0.440270). r15
`outputs/diag/clustering_k_class.json` has K = 9 and 10 for every arm on MMArt test. The independent-seed draw order is
in `rb_score.draw_bootstrap` (rows, then seeds per arm in arm order).
Fix (add to (i)): "r14's SigLIP 2 Affection SC4 and SC6 AMI values and intervals; r14's SigLIP 2 ArtELingo SC3 d and
interval (`b_matched`, two readings), and a unit test that the one-reading path equals it with the own reading
removed (same draws); at K ≠ 20, r14's EmoSet K = 8 values (`sheet_emoset.json`) and r15's class-count diagnostic
values (MMArt test, K = 9 and 10, every arm, four encoders), within 1e-9."

**M5. §3.5 gate-scope table: one failure under-voids, others over-void.**
Problem:
- The rebuilt CRL vectors feed SC5 (clustering), but a CRL failure in (ii) voids only SC6 to SC8, so SC5 would stand.
- (i) as one row voids SC1 to SC5 for any part, so an EmoSet loader failure would void the painting cells.
- (iv) voids Qwen's ArtELingo SC8 cell, which uses r11's pick and stored probabilities, not the CPU mapper.
- (iv) voids X's ArtELingo SC2 cells, which use the stored GPU probabilities. Only MMArt's SC2 cells use CPU outputs
  (r15's exports).
- A failure in (iii) of the class counts has no row.

Fix: replace the table with:

| Failed gate | Voids |
|---|---|
| (i) painting scorer (r14 SigLIP 2, r15 CLIP L/14, r10 partitions) | SC1 to SC3 and SC5 |
| (i) matched-arm interval | SC2 |
| (i) Affection or EmoSet replay | that set's SC4 cells |
| (i) K ≠ 20 replay | every class-count cell |
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

## Should-fix

**S1. §3.4: ACC in the bootstrap is not pinned down.** Add: "ACC: the contingency table of the K clusters and the K
classes on the labelled rows (rows indexed first, then masked, repeats counted); Hungarian matching maximizing matched
rows, recomputed in every draw; a cluster with no labelled row counts zero; denominator = labelled rows of the draw."

**S2. §3.4 "Outcomes": r for SC7 and SC5 at 20.** "The candidate's own range" conflicts with D17 for untold c*, a new
arm with no recorded range. Add: "SC7: X-c*'s recorded told validation mAP ranges on that set (D17); SC5 at 20 clusters:
CLIP L/14 c*'s recorded AMI ranges (r14 `stage1_clip_l14.json` `artclu`; r15 `stage1_clip_l14.json` `clu_ranges`)."

**S3. §3.7 and §3.4: it is unclear which PercepT validation probabilities Qwen's MMArt pick uses.** Two sets exist:
r11's stored ArtELingo validation probabilities and the new CPU probabilities. They agree only within 1e-5 (iv), and
the pick's tie rule can turn on such a difference. Fix: "every new PercepT pick reads the CPU validation probabilities
(the MMArt test probabilities are r15's CPU outputs); Qwen's ArtELingo cell keeps r11's pick and stored
probabilities."

**S4. §3.5 (v): the key check cannot be run as written on ArtELingo.** The ArtELingo held exports hold only `img`
(sidecar `list_sha256`). The validation exports (`full/train/k11=40/R6-5/…`) have no keys and no list hash. Fix: "(v)
MMArt: the `keys` arrays of the three heads are equal; ArtELingo held: the sidecars' `list_sha256` are equal; ArtELingo
validation: row counts equal the sorted validation list. The joined vector is divided by √3; rows unit norm within
1e-3."

**S5. §6 licensed sentences vs §8.** §8 promises "SC2 named 'PercepT, k-means at the class count' in every sentence",
but the SC1 to SC3 sentence says "PercepT on X". Fix: "<the plain X image | PercepT on X, its k-means reading with as
many clusters as classes>". SC5: add "(AMI with 20 clusters; <metrics> with as many clusters as classes, <K>)".

**S6. Brief, "Check these" (hard rules 2 and 4, module type rule).** Three items are missing:
- the stage types (D14, agent default; the module requires them listed);
- the class-count bars (D15), the most consequential agent default;
- the minor batch m1 to m8 (D6, D8 to D13): the user's picks after seeing the recommendation, with untold and PercepT
  retrieval key to SC7 and SC8.

Fix: merge items 4 and 5 into one line. Add: "Agent defaults: the stage types (stage 2 a claim test, D14) and the bars
of the class-count cells (each keeps its AMI-at-20 bar, D15). Your picks after seeing our recommendation: the minor
batch (metric definitions, untold and PercepT retrieval, CRL on MMArt, per-class look, protocol, folder)."

**S7. Brief, "Outcomes".** The template asks for inconclusive and what follows each outcome. Add: "A failed check voids
only the cells it serves (no claim, inconclusive); a missed bar is reported as is; the paper cites only cells that
pass, by family."

**S8. §9 and brief "Cost": the budget is probably low.**
Evidence:
- r15's C13 derive took 5,178 s for 60 cells (r15 log, 03:47); this round has about 412 cells, 16 new picks and 6,144-wide untold vectors.
- A k-means fit takes about 13 s at one thread on 12,281 × 2,048 (r11 `summary.md`, `partition_seconds` about 40 s per seed of three states).
- The count sweep has about 1,400 fits at width 2,048 or more.

Estimate: about 13 to 20 process hours and 5 to 8 h of wall time. Biggest drivers: C13, then the count sweep, then the
untold picks (10 eigendecompositions at 6,144 per arm). The time-box still holds.
Fix: "CPU about 13 to 20 h … about 5 to 8 h of wall time; if 11-04 12:00 passes with stage 3 unfinished, the count
sweep's photo sets go last and may be cut (listed as not run)."

**S9. §3.5 (i): "r10's partitions" is not named.** Fix: "r10's stage 4 c* held partitions (`stage4/partitions/full/k11=40`)".

**S10. §3.7: the CRL rebuild form is not fixed.** Fix: "CRL vectors by r11's function (float64 cosines of the L2-normalized image with the 50 value texts), so (ii) is exact".

**S11. §3.4 "Intervals": the draw order is not given.** C13 re-derives from the rule. Add: "per draw: rows, then c*'s five seeds, then the matched arm's five seeds (r14's `draw_bootstrap` order); a fresh `default_rng(0)` per cell".

## Nits

- N1. Header and D14 to D19 give 19:45, which is later than this review's time (19:40). Use the time of writing.
- N2. MMArt style has 20 classes, so its AMI cells in SC1 and SC3 repeat r15's SC1 and SC4 AMI cells. Only NMI, ARI, ACC
  and SC2's one-reading cell are new. Say so beside §3.3.
- N3. The brief uses code names ("m2", "D7", "D16"). Use plain words: "the LEVEL-margin answer".
- N4. SC4's licensed sentence is "the same for photos". Write it out with "<Affection | EmoSet> test" and "photos".
- N5. SC8's sentence should say "c*'s <aspect> head", so readers see told c* against the aspect-free PercepT vector. In
  SC7, "the query names no aspect" is clearer than "the image as the only query" (the glossary form).
- N6. D4 says "all cells". The grill set WIN in 12 cells (AMI at 20 and mAP). The class-count metrics come from D15, an
  agent default. Say so in D4.
- N7. The brief is 40 lines against about 35.

## Checked and fine

- MMArt validation is a subset of ArtELingo validation. All 5,769 keys map to unique ArtELingo painting names, and all
  are among the 6,152 validation names (`mt_data.keys('val')`, `rt_data.names('val')`, run 2026-11-03).
- Class counts and class sets are equal on validation and test: ArtELingo 9/27/10 (smallest validation classes
  77/5/153), MMArt 9/20/10, Affection 9, EmoSet 8 (cut: 5,770 validation, 17,348 test).
- The read-out menu works at every width:
  - width 6,144: whitening at 64, 256 and 1,024 (`SIZES` capped at the width), 25 read-outs. Each eigendecomposition
    is 6,144 × 6,144 (about 0.3 GB, seconds to a minute). A cross-fit half has about 3,000 rows, more than 1,024 dims.
  - widths 40 and 50: whitening at the full width, 15 read-outs.
- r11's CRL used one pick (`none+k10`) for its three aspect vectors.
- The records exist:
  - r14: SigLIP 2 SC1 and SC6, SC3 readings (`M2-r20`), CLIP L/14 SC2 c* mAP 0.267091, EmoSet SC7 and SC9 with intervals.
  - r15: CLIP L/14 SC1, SC2 and SC4 with intervals, SC3 `M7-r20`.
  - r11: stage 5 c* and teacher intervals.
  - r10: stage 4 partitions.
- The cell count matches: 96 + 96 + 96 + 40 + 30 + 6 + 24 + 24 = 412 ("about 410").
- The bars match the cells they extend: r15 D7 (emotion and style AMI, four encoders, WIN); r14 SC6 and r10's teacher
  cells on ArtELingo (LEVEL or WIN); r14 SC9 on EmoSet (WIN); Affection (LEVEL or WIN).
- Hard rules 1, 2, 3, 5 and 7 hold. The parts table for a claim test and a sweep is complete. The body has about 228
  non-blank lines, and a reason is given.
- The machine has 32 cores, so three processes at 8 threads fit.

## What I would remove

Nothing.
