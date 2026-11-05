# Outside methods: c* with k-means beats plain images with default TURTLE in every cell

> 2026-11-05 (sequence name 2027-01-06) · [Spec](../../../superpowers/specs/2026-11-03-outside-methods-design.md) · [Decision rule](../../../superpowers/specs/2026-11-03-outside-methods-design.md#6-success-criteria) · [Run log](../../../../src/test/20261103_outside_methods/20261103_outside_methods_log.md)

## Results brief

**What we tried.** Outside methods (r18) compared c*, our student with aspect heads, against outside embeddings and stronger clustering ([spec](../../../superpowers/specs/2026-11-03-outside-methods-design.md)).
We used TPIPS (factor-conditioned embeddings), CLAY (text projection of CLIP), and TURTLE (label-free clustering) on two painting sets.

**Outcome against the bar.** Each cell has its own label; there is no overall verdict.
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

The independent re-derivation matched every cell; the largest cell difference was 3.89e-14 across 10,762 checked items.
No item failed or was skipped; the overall maximum was 2.06e-8, on CLAY validation.

**What it does not show.** TURTLE used its released code's defaults, without its paper's selection of settings.
TURTLE cells use c*'s k-means seed range for LEVEL; no correction covers the 337 cells.
TPIPS ran with one library version; its vectors move with the transformers version.
c*'s test values were known from earlier rounds; these comparator results were new.
TEMI, which trains clustering heads through neighbours and a teacher copy, did not run.

**Options for next.**

1. Add paper tables and the planned figure (r20). Writing, design work and CPU minutes could make the differences visible.
2. Build this loop's third stage for TEMI, within 10 GPU h. It could test another clustering method; results before 2026-11-16 enter the paper.
3. Continue head variants (r21) and text queries (r22), at about two GPU hours and under one GPU hour.
   These could test different heads or text inputs.

**Your reading.**

<details>
<summary>Our reading and recommendation (open after yours)</summary>

**Our reading.** The student stays ahead of the plain image with TURTLE at these settings.
The embedding results exceed our priors; CLAY makes style close. TURTLE usually reduces scores; selected settings remain untested.

**Our recommendation.** Our view: use the licensed sentences now and continue the planned TEMI build.
Keep the TURTLE settings clause beside every paper claim.

</details>

## What ran

Stages zero, one, two and four finished; stage three did not run in this round.
All six gates passed; no cell was void.

|Stage|Type; work|Where; timing|
|---|---|---|
|0|Regression; method parity and replay|Local CPU/3090 and node402 probes; replay took 45 min.|
|1|Validation sweep; ranges and picks|Local CPU queue; 03:11 to 03:19 on 2026-11-05.|
|2|Claim test; 28 units|Local CPU queue; 03:20 to 04:10.|
|3|TEMI claim test|Not run; its own build follows.|
|4|Descriptive diagnostics|Local CPU; duration unrecorded.|
|Independent re-derivation|Picks, vectors and cells|Four local CPU queue slots; 04:46 to 05:27.|

TPIPS used the local RTX 3090; six shards ended by 03:08 on 2026-11-05.
TURTLE used node402's three GPUs on 2026-11-04, from 05:50 to 10:17.
All sixteen jobs succeeded.
CLAY vectors and texts came from the local build; text generation took 5.8 minutes.

|Data|Rows|Role|
|---|---:|---|
|ArtELingo validation|6,152; genre 4,960|Picks and ranges|
|MMArt validation|5,769|Picks and ranges|
|ArtELingo held|12,281; genre 9,949|Claims and diagnostics|
|MMArt test|16,384; emotion 16,218; genre 13,420|Claims and diagnostics|

Encoders were Qwen, SigLIP 2, SigLIP 1 and CLIP L/14; c* used seeds zero through four.
Plain-image TURTLE used starts zero through two; each c* seed used start zero.
K-means averaged states zero through two.
Clustering used 2,000 joint row/seed draws; retrieval used 10,000 paired draws.
Class-count metrics were AMI, NMI, ARI and ACC; twenty clusters added AMI.
Class counts were nine for emotion, ten for genre, and twenty-seven/twenty for ArtELingo/MMArt style.
Clustering masked missing labels afterwards; TURTLE fitted scored images without labels.
CLAY used the scored set's mean image; no train split was read.

|Code|Commit|
|---|---|
|Labels|65f8ff9a|
|TURTLE jobs|c29073c9|
|Stage-one records, before stage two|8ca05e1c|

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
It reads saved partitions, so it does not reproduce TURTLE training.

## Analysis

Our reading: the embeddings matter here, but TURTLE's default settings provide a limited test of stronger clustering.
The files under `OUT/diag/` supply descriptive results; they assign no label.

**Priors against outcome.** WIN shares exceed every tested prior.

|Comparison|Spec's expected WIN odds|Observed WIN cells|
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

**What TURTLE adds.** It usually lowers both arms' scores.
The table summarises `turtle_adds_*.json`: class-count AMI differences, TURTLE minus k-means, across encoders.

|Set; aspect|c* difference range|Plain-image difference range|
|---|---|---|
|ArtELingo; emotion|-0.0026 to +0.0020|-0.0045 to -0.0012|
|ArtELingo; style|-0.0349 to -0.0216|-0.0471 to -0.0193|
|ArtELingo; genre|-0.0348 to -0.0167|-0.0458 to +0.0111|
|MMArt; emotion|-0.0034 to -0.0001|-0.0057 to -0.0003|
|MMArt; style|-0.0270 to -0.0173|-0.0426 to -0.0086|
|MMArt; genre|-0.0343 to -0.0265|-0.0602 to -0.0367|

For ArtELingo Qwen emotion, TURTLE gives c* AMI 0.0998 against k-means' 0.1024: -0.0026 [-0.0055, +0.0002].
For the plain image, it gives 0.0669 against 0.0696: -0.0027 [-0.0050, -0.0004].
The c* comparison pairs rows only; it draws the seed sets separately.
Its interval may be wider; the widening is not measured.
At twenty clusters, ArtELingo SigLIP 1 plain-image genre AMI also falls: -0.0804 [-0.0859, -0.0747].

Default TURTLE does not close the gap: c* with k-means wins every labelled comparison against it.
With TURTLE on both sides, two emotion cells are LEVEL; the rest WIN.
Our reading: this supports the aspect spaces at these settings.

**Count sweep.** Outside arms usually stay below c* in `sweep_*.json`; point values give no claim.
Emotion uses counts 5, 8, 9, 16 and 20; genre uses 5, 10, 16 and 20.
Style uses 10, 20, 27 and 40 on both sets.
The table summarises outside-minus-c* k-means differences over both sets and every count.
Comparators are Qwen c* for TPIPS, CLIP L/14 c* for CLAY, and each matching encoder for TURTLE.

|Outside arm|Metric|Point comparisons|Outside above c*|Difference range|
|---|---|---:|---:|---|
|Plain image + TURTLE|AMI|104|0|-0.2040 to -0.0080|
|Plain image + TURTLE|ARI|104|0|-0.1997 to -0.0026|
|TPIPS|AMI|26|0|-0.1193 to -0.0091|
|TPIPS|ARI|26|2|-0.0958 to +0.0079|
|CLAY|AMI|26|0|-0.0426 to -0.0036|
|CLAY|ARI|26|0|-0.0650 to -0.0039|

Five-cluster emotion examples give AMI / ARI, averaged over starts or seeds.

|Set|Qwen c* k-means|Qwen plain k-means|Qwen plain + TURTLE|TPIPS|CLIP L/14 c* k-means|CLAY|
|---|---|---|---|---|---|---|
|ArtELingo|0.0912 / 0.0744|0.0655 / 0.0510|0.0689 / 0.0554|0.0666 / 0.0761|0.0859 / 0.0699|0.0676 / 0.0524|
|MMArt|0.0441 / 0.0322|0.0345 / 0.0241|0.0353 / 0.0252|0.0349 / 0.0275|0.0422 / 0.0301|0.0301 / 0.0196|

**Fits.** `fits.json` records sixteen fits with empty clusters across ten groups; no fit is missing.
Thirteen use plain images at twenty-seven or forty clusters; three c* emotion fits use nineteen of twenty.
The largest shortfall was two; scoring kept every partition.
Loss ranged from 1.5291 to 3.6602; entropy ranged from 1.6093 to 3.6889.
Entropy stayed within 0.0004 of log(K), despite empty hard clusters.
Our reading: balanced soft assignments can still leave hard clusters empty.

**CLAY and validation picks.** `clay.json` records text set A for both tasks and sets, with reason `a_higher`.
Both TPIPS and CLAY picked `none+k10`, meaning no map and ten-neighbour query expansion, on both sets.
A uses fifty prior values per aspect; B uses generated values.

|Set|Clustering mean AMI at twenty: A / B|Retrieval selection score: A / B|
|---|---|---|
|ArtELingo|0.3280 / 0.3003|195.2334 / 176.8842|
|MMArt|0.3213 / 0.2930|231.8407 / 216.6322|

The retrieval selection score averages validation mAP divided by the corresponding CLIP L/14 c* seed range.
These class names occur verbatim in the lists; none was removed.

|List|Emotion, both sets|Genre, both sets|ArtELingo style|MMArt style|
|---|---|---|---|---|
|A|amusement, awe, disgust, fear|cityscape, landscape, portrait|None|baroque|
|B|awe, contentment, excitement, fear|landscape, portrait|None|baroque|

Matches after lower-casing and replacing underscores are recorded separately.

**TPIPS and plain cosine retrieval.** `tpips_cosine.json` records mean cosine between each image's three factor vectors.
It is 0.7510 on ArtELingo held and 0.7511 on MMArt test.
Our reading: the factors change the vectors but retain much shared direction.
`r0_rows.json` supplies plain cosine mAP below; only CLAY A supplied labelled cells.

|Set; arm|Emotion|Style|Genre|
|---|---:|---:|---:|
|ArtELingo; TPIPS|0.2555|0.2277|0.5386|
|ArtELingo; CLAY A|0.2449|0.2918|0.5279|
|ArtELingo; CLAY B|0.2472|0.2731|0.4570|
|MMArt; TPIPS|0.1945|0.2474|0.5441|
|MMArt; CLAY A|0.1910|0.3157|0.5354|
|MMArt; CLAY B|0.1930|0.2957|0.4685|

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

The final labels passed verification, but the report must retain these procedure and checker limits.

**Run order and timing.** Stage two ran four units at a time through res-queue, an agent default, rather than serially as `om_queue.sh` does.
Each unit used its own process and part file; caches lived in memory, so no number depended on order.
TPIPS extraction finished about a day before the first handoff's Thursday 23:00 estimate.
Labels followed at 05:28 on 2026-11-05, well before 11:00.
CPU slot-hours and diagnostic wall time remain uncomputed.

**TPIPS parity.** Node vectors missed the fixed 0.999 cosine bar, even with identical packages and byte-identical images.
The lowest row cosine was 0.9987; the remaining hardware cause was not isolated.
The threshold stayed unchanged, and TPIPS moved to the local run.
Gate (iv) then passed at 0.99999994.
Between transformers 5.6.2 and 5.16.1, median row cosine was 0.9988.
Reported vectors depend on the library version; this run used 5.6.2.

**Gate reach and final build review.** The gates retain limits after a pass.

|Gate|What it cannot see|Evidence that carries the remaining claim|
|---|---|---|
|(i), CLAY|Log map arccos scale at rank forty-nine|Set B's scale matters by up to 0.011. Independent math restatement differed by 7.2e-7, or 0.001 degrees.|
|(ii), TURTLE|Batch draws in the short fit|A shared generator and oracle test check draws; sabotage caught changes.|
|(ii), GPU|3090 reference versus node402 fits|One full fit matched every partition row across machines; loss was 2.9114 on both.|
|(iv), TPIPS|Store rows against reference|`om_tpips_check.py` compares each shard's probe rows with the reference.|

Both final build reviewers approved with notes but incomplete coverage. Their checks follow.

|Check|Final review evidence|
|---|---|
|Cells and execution units|337 identifiers and 28 units re-derived from the spec|
|Two MMArt Qwen emotion retrieval cells|Largest difference 6e-11|
|CLAY vectors|Largest difference 1.5e-8|
|CLAY gate projector / mean / query|2.2e-7 / 6.0e-8 / 6.6e-7 against 1e-5|
|Earlier retrieval replay|36/36 expected values matched exactly|
|Rule sabotage|19 cases; two misses gained tests|
|Fix re-review|168 tests passed; three sabotages caught|
|Pre-run whole suite|314 tests passed in two minutes thirteen seconds|

Part files bind no input hash; changed inputs require forced recomputation.
CLAY and TPIPS gates bind no code hash; independent re-derivation checks final scoring.

**Text list and row form.** Set B has forty-four style values containing “gothic”; the log reports forty-three “gothic X” values.
The extra value is “gothic” itself; the log count covers the extended phrases. The list stays unchanged.

|Decision|Retained behaviour|
|---|---|
|D5|Keep the first hundred distinct generated values; no fallback when at least two remain.|

TURTLE reads c*'s unit rows; k-means reads float16 rows cast to float32, without division.
The spec names no TURTLE row form.
The sample covered one seed and two encoders per set.

|Row-form check|Measured size|
|---|---|
|Row norms|0.99997 to 1.00003|
|Largest element gap|3.7e-6|
|Lowest cosine|0.99999976|
|Independent input binding|All 412 fit inputs matched hashes|

Our reading: this small difference leaves the labels specific to unit-row TURTLE. The diagnostic interval's widening remains unmeasured.

**Independent re-derivation review.** Seven findings led to three fixes below the result boundary, one fix round and a re-review.
Derivation text stayed byte-identical.

|Finding|Disposition and limit|
|---|---|
|TURTLE input hashes were unchecked|Fixed; compare now binds every input.|
|Unavailable ranges could cause false failure|Fixed; compare respects the recorded range state.|
|CLAY operation order|Unfixed; row difference up to 7.3e-12 risks false failure only. Full compare passed.|
|Recorded inputs were not reread|Unfixed; hand check found no TURTLE file newer than 04:00, before derive.|
|Status hashes were ignored|Fixed below the boundary.|
|Label writer did not require PASS itself|Unfixed; controller required certified full PASS before labels.|
|Edits above the boundary invalidate every cache|Retained; fifty-two sections depend on source identity. This affects recomputation cost.|

The fix round passed seventy tests and caught six mutations; the re-review found no defect that changes a label.
Residual notes concern colliding synthetic fixtures, unchecked path containment, and a low-draw test that cannot distinguish its guards.
The input checker trusts sidecar aspect and seed; the partition reader checks every pairing a cell uses.
These limit tests or malformed-input defence; actual pairings and the full comparison passed.

## Files and storage

This task created only this report and deleted nothing.
The [round folder](../../../../src/test/20261103_outside_methods/) holds code, logs and records.
Outputs (`OUT`) are under `/data/SSD2/cosir_r18/20261103_outside_methods/outputs/`.
Keep `c13/compare.json`, `parts/turtle/`, `stage1/`, `stage2/`, `textset_b/` and all `diag/` files.
Shared TPIPS stores are under `/data/SSD2/pre_extract/{artelingo_full,mmart}/tpips_8b/`.

Would remove after review: `/data/SSD2/cosir_r18/20261103_outside_methods/c13_smoke/` and `c13_round_smoke/`.
The storage summary is left for the wrap-up, in the read chat.
