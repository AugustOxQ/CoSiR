# Outside methods: c* with k-means beats plain images with default TURTLE in every cell

> 2026-11-05 (sequence name 2027-01-06) · [Spec](../../../superpowers/specs/2026-11-03-outside-methods-design.md) · [Decision rule](../../../superpowers/specs/2026-11-03-outside-methods-design.md#6-success-criteria) · [Run log](../../../../src/test/20261103_outside_methods/20261103_outside_methods_log.md)

## Results brief

**What we tried.** Outside methods (r18) compared c*, our student with aspect heads, against outside embeddings and stronger clustering ([spec](../../../superpowers/specs/2026-11-03-outside-methods-design.md)).
We used TPIPS (factor-conditioned embeddings), CLAY (text projection of CLIP), and TURTLE (label-free clustering) on two painting sets.

**Outcome against the bar.** Of 337 cells, 330 are WIN, 6 LEVEL, 1 BELOW and none void.
Each cell has its own label; there is no overall verdict.
WIN means the difference's lower 95% bound exceeds zero; otherwise LEVEL means it exceeds −r; BELOW means it does not.
Here r is c*'s validation seed range, the maximum minus minimum over seeds.
Examples use ArtELingo Qwen emotion: AMI at nine clusters, or mAP.

|Comparison|Cells|WIN|LEVEL|BELOW|Example: candidate / comparator|Difference [95% interval]|
|---|---:|---:|---:|---:|---|---|
|Qwen c* / TPIPS, clustering|29|28|1|0|0.1024 / 0.0718|+0.0306 [+0.0274, +0.0340]|
|Qwen c* / TPIPS, retrieval|6|6|0|0|0.2691 / 0.2565|+0.0126 [+0.0114, +0.0138]|
|Qwen and CLIP L/14 c* / CLAY, clustering|58|54|3|1|0.1024 / 0.0676|+0.0348 [+0.0312, +0.0382]|
|Qwen and CLIP L/14 c* / CLAY, retrieval|12|12|0|0|0.2691 / 0.2486|+0.0205 [+0.0194, +0.0216]|
|c* + TURTLE / plain image + TURTLE|116|114|2|0|0.0998 / 0.0669|+0.0330 [+0.0286, +0.0373]|
|c* k-means / plain image + TURTLE|116|116|0|0|0.1024 / 0.0669|+0.0356 [+0.0317, +0.0394]|

The independent re-derivation of the scoring from saved outputs matched all 10,762 checked items; none failed or was skipped.
The largest gap on a cell was 3.89e-14; the largest overall was 2.06e-8, on a CLAY validation score.

**What it does not show.** TURTLE used its released code's defaults, without its paper's selection of settings.
At those defaults it scored below k-means in 203 of 232 diagnostic comparisons (170 with the whole interval below zero; c*'s intervals may be too narrow). So these cells do not show that c* holds against a clustering stronger than k-means.
TURTLE is our re-implementation, checked against the official code only in a short run (100 outer steps on 2,000 rows).
TURTLE cells use c*'s k-means seed range for LEVEL; no correction covers the 337 cells.
TPIPS ran with one library version; its vectors move with the transformers version. Whether it saw WikiArt paintings in training was not checked.
c*'s test values were known from earlier rounds; these comparator results were new.
TEMI, which trains clustering heads through neighbours and a teacher copy, did not run.

**Options for next.** r20's decide chat holds this choice. Head variants (r21) and text queries (r22) stay on the plan.

1. Put these rows into the paper: tables, related work, method text. Cost: writing only, no GPU.
   It shows readers where c* stands against TPIPS, CLAY and default TURTLE; it adds no evidence.
2. Build and run TEMI, this loop's third stage. Cost: one unattended build and up to 10 GPU h.
   It could show whether a second method, at its paper's settings, closes c*'s gap. Results before 2026-11-16 enter the paper; c*'s heads may get no cell.
3. Run the spec's one-off TURTLE selection on validation, then repeat plain-image TURTLE at the chosen setting.
   Cost: a small spec, a build, and about 134 fits (50 + 84): about 4 GPU h (3.7 to 4.4) by our arithmetic, not measured. A selected setting may change a fit's time.
   It could show whether selected TURTLE reaches c* with k-means, the test these cells leave open.

**Your reading.**

<details>
<summary>Our reading and recommendation (open after yours)</summary>

**Our reading.** c* stands above both outside embeddings in most cells, and above our priors.
TPIPS, on a base four times c*'s size, loses 34 of 35 cells with one LEVEL; it ran against Qwen c* only. CLAY is the close comparator: its four non-WIN cells of 70 are all style.
The TURTLE cells are weaker evidence than their 230 WINs suggest. Plain-image TURTLE mostly sits below plain k-means, so these cells add little beyond r17's k-means cells.
The closest ones are the ones to watch: d is below c*'s seed range in 5 of 116 c* k-means cells and 6 of 116 cells with TURTLE on both sides, all emotion ARI or ACC.
We are sure of the scoring: the re-derivation matches it from saved outputs and recomputes CLAY's vectors. It redoes no feature extraction or training. The gates check those, and for TURTLE training only by a short run against the official code and one full fit matched across machines.
We are less sure what default TURTLE says about stronger clustering.

**Our recommendation.** Our view: option 1 now, since the abstract is due 2026-11-10 and needs no new run.
Then option 2 with option 3 in the same build: one spec covers both, and a reader will first ask whether TURTLE was tuned.
Until then, keep the settings clause beside every TURTLE sentence, and do not call default TURTLE "stronger" than k-means in the paper.

</details>

## What ran

Stages zero, one, two and four finished; stage three did not run in this round.
All six gates passed; no cell was void.

|Stage|Type; work|Where; timing|
|---|---|---|
|0|Regression; method parity and replay|Local CPU/3090, node402 probes|
|1|Validation sweep; ranges and picks|Local CPU; 03:11 to 03:19 on 2026-11-05|
|2|Claim test; 28 units|Local CPU; 03:20 to 04:10|
|3|TEMI claim test|Not run; its own build follows|
|4|Descriptive diagnostics|Local CPU; 05:28 to 06:39|
|Re-derivation|Picks, vectors and cells|Local CPU, four slots; 04:46 to 05:27|

TPIPS used the local RTX 3090; six shards ended by 03:08 on 2026-11-05.
TURTLE used node402's three GPUs on 2026-11-04, 05:50 to 10:17; all sixteen jobs succeeded.
CLAY vectors and texts came from the local build.

|Data|Rows|Role|
|---|---:|---|
|ArtELingo validation|6,152; genre 4,960|Picks and ranges|
|MMArt validation|5,769|Picks and ranges|
|ArtELingo held|12,281; genre 9,949|Claims and diagnostics|
|MMArt test|16,384; emotion 16,218; genre 13,420|Claims and diagnostics|

Encoders were Qwen, SigLIP 2, SigLIP 1 and CLIP L/14; c* used seeds zero through four.
Plain-image TURTLE used starts zero through two, each c* seed start zero; k-means averaged states zero through two.
Clustering used 2,000 joint row/seed draws and retrieval 10,000 paired draws.
Metrics were AMI, NMI, ARI and ACC at the class count (nine emotion, ten genre, twenty-seven or twenty style) and AMI at twenty.
Labels masked rows only at scoring; TURTLE fitted the scored images without labels, and CLAY used the scored set's mean image. No train split was read.

Commits: labels 65f8ff9a; TURTLE jobs c29073c9; stage-one records 8ca05e1c, before stage two.
The scorer came from the preceding round, read-only.
TPIPS used the released late-fusion model; possible WikiArt training overlap was not checked.

## Results

The rule licenses each WIN and LEVEL sentence separately; the single BELOW cell licenses neither side's superiority.
WIN medians span metrics; they describe scale, not a pooled effect.
Values and labels come from [labels.json](../../../../src/test/20261103_outside_methods/records/labels.json).

|Spec ID|Comparison|Cells|WIN / LEVEL / BELOW|Median WIN d|
|---|---|---:|---|---:|
|SC1|TPIPS clustering, Qwen c*|29|28 / 1 / 0|+0.0306|
|SC2|TPIPS retrieval, Qwen c*|6|6 / 0 / 0|+0.0526|
|SC3|CLAY clustering, both c* encoders|58|54 / 3 / 1|+0.0246|
|SC4|CLAY retrieval, both c* encoders|12|12 / 0 / 0|+0.0209|
|SC5|c* + TURTLE / plain image + TURTLE|116|114 / 2 / 0|+0.0763|
|SC6|c* k-means / plain image + TURTLE|116|116 / 0 / 0|+0.1056|

All seven non-WIN cells follow; d is candidate minus comparator. Decisions use unrounded values.

|Spec ID; set; encoder; aspect|Metric; clusters|Candidate|Comparator|d [95% interval]|r|Label|
|---|---|---:|---:|---|---:|---|
|SC1; ArtELingo; Qwen; emotion|ARI; 9|0.0605|0.0550|+0.0056 [-0.0001, +0.0119]|0.0130|LEVEL|
|SC3; ArtELingo; Qwen; style|AMI; 27|0.4109|0.4061|+0.0049 [-0.0000, +0.0098]|0.0087|LEVEL|
|SC3; MMArt; Qwen; style|AMI; 20|0.4227|0.4240|-0.0013 [-0.0078, +0.0051]|0.0118|LEVEL|
|SC3; MMArt; Qwen; style|NMI; 20|0.4250|0.4263|-0.0013 [-0.0078, +0.0051]|0.0117|LEVEL|
|SC3; MMArt; CLIP L/14; style|ACC; 20|0.3518|0.3510|+0.0008 [-0.0104, +0.0113]|0.0081|BELOW|
|SC5; ArtELingo; SigLIP 2; emotion|ARI; 9|0.0532|0.0490|+0.0041 [-0.0008, +0.0089]|0.0293|LEVEL|
|SC5; MMArt; SigLIP 1; emotion|ACC; 9|0.1956|0.1872|+0.0084 [-0.0006, +0.0159]|0.0050|LEVEL|

Each LEVEL cell licenses “level with” for that cell, without establishing equality.
The positive-d BELOW cell has a lower bound below −r, so it licenses no claim for c*.
The “arm better than c*” sentence is also unlicensed because the upper bound exceeds zero.

Narrow WINs include ArtELingo Qwen style NMI against CLAY: +0.0048 [+0.0000, +0.0097], with an unrounded lower bound of 3.11e-5.
MMArt SigLIP 1 emotion ARI with TURTLE on both sides is also narrow: +0.0037 [+0.0006, +0.0067].

The [certificate](../../../../src/test/20261103_outside_methods/records/c13.json) covers labels, picks and ranges at full draw counts.
It recomputes CLAY's vectors, reads saved partitions and stores, and so does not reproduce TURTLE training or TPIPS extraction.

## Analysis

Our reading: the embeddings matter here, but TURTLE's default settings provide a limited test of stronger clustering.
The files under `OUT/diag/` supply descriptive results; they assign no label.

**Priors against outcome.** WIN shares exceed every tested prior.

|Comparison|Spec's prior per cell|Observed WIN cells|
|---|---|---|
|TPIPS clustering|0.50|28/29|
|TPIPS retrieval|0.35|6/6|
|CLAY, CLIP L/14 c*, both tasks|0.80|34/35|
|CLAY, Qwen c*, both tasks|0.85|32/35|
|TURTLE on both sides|0.80|114/116|
|c* k-means / plain image + TURTLE|0.70; genre 0.60|116/116; genre 40/40|
|TEMI|Same priors as TURTLE|Not run|

The first surprise did not occur: the plain image with TURTLE never exceeded either c* arm in emotion.
TPIPS fell below CLAY in sixteen of twenty-nine clustering cells and four of six retrieval cells, partly confirming the second surprise.
These point comparisons have no label; ArtELingo style mAP was 0.2365 for TPIPS against CLAY's 0.3137.

**What TURTLE adds.** At its defaults TURTLE usually scored below k-means on the same rows; this diagnostic has no label.
The table summarises `turtle_adds_*.json`: TURTLE minus k-means over 232 comparisons, the four class-count metrics and AMI at twenty.
On the plain image the direction depends on the metric: in ARI and ACC, d is above zero in 20 of 48 rows, 14 of them with a lower bound above zero.
In emotion, plain-image d is above zero in 11 of 40 rows, 8 of them with a lower bound above zero.

|Arm|Metrics|Comparisons|d below 0|Upper bound below 0|Lower bound above 0|
|---|---|---:|---:|---:|---:|
|c*|AMI, NMI|68|64|53|0|
|c*|ARI, ACC|48|45|35|0|
|Plain image|AMI, NMI|68|66|60|2|
|Plain image|ARI, ACC|48|28|22|14|

Median d is smallest in emotion: -0.0034 on c* and -0.0020 on the plain image, against -0.0336 and -0.0302 in genre.
For ArtELingo Qwen emotion AMI, TURTLE gives c* 0.0998 against k-means' 0.1024: -0.0026 [-0.0055, +0.0002].
For the plain image, it gives 0.0669 against 0.0696: -0.0027 [-0.0050, -0.0004].
The largest plain-image fall is ArtELingo SigLIP 1 genre AMI at twenty: -0.0804 [-0.0859, -0.0747]. The largest rise is ArtELingo SigLIP 2 emotion ACC: +0.0256 [+0.0216, +0.0295].
The c* comparison pairs rows only and draws the seed sets separately, so its interval may be wider; the widening is not measured.

Default TURTLE does not close the gap: c* with k-means wins every labelled comparison against it.
With TURTLE on both sides, two emotion cells are LEVEL; the rest WIN.
In five c* k-means cells d is below c*'s seed range, and in six cells with TURTLE on both sides: SigLIP 2 and CLIP L/14 emotion, ARI or ACC. Of those eleven, one is LEVEL and the rest are WIN under the rule.
Our reading: this supports the aspect spaces at these settings. It says little about stronger clustering, because the comparator mostly sits below plain k-means.

**Count sweep.** Outside arms usually stay below c* in `sweep_*.json`; point values give no claim.
Counts are 5, 8, 9, 16, 20 (emotion); 5, 10, 16, 20 (genre); 10, 20, 27, 40 (style).
The table gives outside minus c* k-means over both sets and every count, against Qwen c* for TPIPS, CLIP L/14 c* for CLAY, and the matching encoder for TURTLE.

|Outside arm|Metric|Point comparisons|Outside above c*|Difference range|
|---|---|---:|---:|---|
|Plain image + TURTLE|AMI; ARI|104 each|0; 0|-0.2040 to -0.0080; -0.1997 to -0.0026|
|TPIPS|AMI; ARI|26 each|0; 2|-0.1193 to -0.0091; -0.0958 to +0.0079|
|CLAY|AMI; ARI|26 each|0; 0|-0.0426 to -0.0036; -0.0650 to -0.0039|

**Fits.** `fits.json` records sixteen fits with empty clusters across ten groups, with a largest shortfall of two; no fit is missing and scoring kept every partition.
Thirteen use plain images at twenty-seven or forty clusters; three are c* emotion fits at twenty.
Entropy stayed within 0.0004 of log(K), despite empty hard clusters.
Our reading: balanced soft assignments can still leave hard clusters empty.
Our guess, not tested: the default entropy weight keeps clusters near equal size, which may explain the lower scores on classes of unequal size.

**CLAY and validation picks.** `clay.json` records text set A for both tasks and sets, by the spec's rule. A uses fifty prior values per aspect; B uses generated values.
TPIPS and CLAY both picked `none+k10`, no map with ten-neighbour query expansion, on both sets.

|Set|Clustering mean AMI at twenty: A / B|Retrieval selection score: A / B|
|---|---|---|
|ArtELingo|0.3280 / 0.3003|195.2334 / 176.8842|
|MMArt|0.3213 / 0.2930|231.8407 / 216.6322|

The selection score averages validation mAP divided by CLIP L/14 c*'s seed range.
Set A holds 4 of 9 emotion class names, 3 of 10 genre names (4 after normalising case and underscores) and one MMArt style name; none was removed. Our reading: this favours CLAY.

**TPIPS and plain cosine retrieval.** `tpips_cosine.json` records mean cosine between each image's three factor vectors.
It is 0.7510 on ArtELingo held and 0.7511 on MMArt test.
Our reading: the factors change the vectors but retain much shared direction.
Plain cosine mAP for TPIPS and both CLAY sets is in `r0_rows.json`; only CLAY set A supplied labelled cells.

## How the results change the paper

The paper gains outside-method comparisons and a clustering test; the architecture stays unchanged.
Lines refer to the latest joined draft, `drafts/v1/paper_v1.md`, dated 2026-11-02.

- **Related work:** lines 18 and 22 describe CLAY and TPIPS. Add these results and CLAY's style exceptions.
  Add TURTLE beside the clustering discussion at line 100; TEMI gains no result.
- **Method:** line 414 specifies twenty-cluster k-means and Leiden. Add class-count scoring and the default TURTLE evaluation.
  Lines 394 to 410 need the outside arms' retrieval picks and CLAY's text-set selection.
- **Experiments:** the section is unwritten. Add family tables, exceptions and descriptive diagnostics, with selection disclosure, library version and no multiplicity correction.

A licensed example is: “On ArtELingo held, c* on Qwen clusters paintings by emotion better than TPIPS (AMI, nine clusters).”
WIN retrieval cells use “retrieves”; LEVEL cells use “level with”. The BELOW cell supplies no superiority sentence.

TURTLE WINs license c* over the plain image, either with TURTLE on both sides or with c* using k-means.
Each sentence must name the set, aspect, encoder and metric/count.
Each requires “at its released code's default settings, without its hyperparameter selection”.

No claim covers other counts, unscored sets, TPIPS or CLAY on other encoders, selected TURTLE settings, or TEMI before its run.

## Deviations

**Run order and timing.** Stage two ran four units at a time through res-queue, an agent default, rather than serially as `om_queue.sh` does.
Each unit used its own process and part file, and caches lived in memory, so no number depended on order.
TPIPS extraction finished about 20 hours before the first handoff's Thursday 23:00 estimate; labels followed at 05:28 on 2026-11-05, well before the 11:00 end set for overnight runs.

**TPIPS parity.** Node vectors missed the fixed 0.999 cosine bar, with a lowest row cosine of 0.9987, even with identical packages and images; the hardware cause was not isolated.
The threshold stayed unchanged, TPIPS moved to the local run, and gate (iv) then passed at 0.99999994.
Reported vectors depend on the library version: median row cosine between transformers 5.6.2 and 5.16.1 was 0.9988. This run used 5.6.2.

**Gate reach and final build review.** The gates retain limits after a pass.

|Gate|What it cannot see|Evidence that carries the remaining claim|
|---|---|---|
|(i), CLAY|Log map arccos scale at rank forty-nine|Set B's scale matters by up to 0.011. Independent math restatement differed by 7.2e-7, or 0.001 degrees.|
|(ii), TURTLE|Batch draws in the short fit|A shared generator and oracle test check draws; sabotage caught changes.|
|(ii), GPU|3090 reference versus node402 fits|One full fit matched every partition row across machines; loss was 2.9114 on both.|
|(iv), TPIPS|Store rows against reference|`om_tpips_check.py` compares each shard's probe rows with the reference.|

Both final build reviewers approved with notes, and each stopped before full coverage. Their main checks:

|Check|Final review evidence|
|---|---|
|Cells and execution units|337 identifiers and 28 units re-derived from the spec|
|CLAY vectors; gate projector / mean / query|1.5e-8; 2.2e-7 / 6.0e-8 / 6.6e-7 against 1e-5|
|Earlier retrieval replay|36/36 expected values matched exactly|
|Rule sabotage|19 cases; two misses gained tests|

The controller's whole-suite run before the fits passed 314 tests.

Part files bind no input hash; changed inputs require forced recomputation.
CLAY and TPIPS gates bind no code hash; independent re-derivation checks final scoring.

**Text list and row form.** Set B has forty-four style values containing “gothic”; the log counts forty-three “gothic X” values, and the extra is “gothic” itself. The list stays unchanged.
Set B keeps the first hundred distinct generated values, with no fallback when at least two remain.

TURTLE reads c*'s unit rows; k-means reads float16 rows cast to float32, without division. The spec names no TURTLE row form.
On one seed and two encoders per set, row norms were 0.99997 to 1.00003, the largest element gap 3.7e-6 and the lowest cosine 0.99999976; all 412 fit inputs matched their hashes.
Our reading: this small difference leaves the labels specific to unit-row TURTLE.

**Independent re-derivation review.** Seven findings led to three fixes below the result boundary, one fix round and a re-review; derivation text stayed byte-identical.
None of the four unfixed items can change a label; the re-review found no defect that does.

|Finding|Disposition|
|---|---|
|TURTLE input hashes unchecked|Fixed; compare binds every input.|
|Unavailable ranges could cause false failure|Fixed; compare respects the recorded state.|
|Status hashes ignored|Fixed.|
|CLAY operation order|Unfixed; row difference up to 7.3e-12 risks a false failure only, and the full compare passed.|
|Recorded inputs not reread|Unfixed; a hand check found no TURTLE file newer than 04:00, before the derive.|
|Label writer does not require PASS itself|Unfixed; the controller required the full PASS first.|
|Edits above the boundary invalidate every cache|Retained; this affects recomputation cost only.|

The fix round passed seventy tests and caught six mutations. Four low notes on test fixtures, path checks and input trust stay in the round log.

## Files and storage

This task created only this report and deleted nothing.
The [round folder](../../../../src/test/20261103_outside_methods/) holds code, logs and records.
Outputs (`OUT`) are under `/data/SSD2/cosir_r18/20261103_outside_methods/outputs/`.
Keep `c13/compare.json`, `parts/turtle/`, `stage1/`, `stage2/`, `textset_b/` and all `diag/` files.
Shared TPIPS stores are under `/data/SSD2/pre_extract/{artelingo_full,mmart}/tpips_8b/`.

Would remove after review: `/data/SSD2/cosir_r18/20261103_outside_methods/c13_smoke/` and `c13_round_smoke/`.
The storage summary is left for the wrap-up, in the read chat.
