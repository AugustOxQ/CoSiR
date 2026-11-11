# Paper baselines: c* wins all twelve cells against VLM2Vec-V2 and text-only BERTopic

> 2026-11-11 · Spec: [paper-baselines handoff](../../../superpowers/handoffs/2026-11-11-r25-paper-baselines-handoff.md#2-decided-by-the-user-not-to-be-reopened) · Decision rule: [r18's WIN / LEVEL / BELOW rule, r10](2027-01-06_outside_methods.md#results-brief) · [Run log](../../../../src/test/20261111_paper_baselines/20261111_paper_baselines_log.md)

## Results brief

**What we tried.** We compared c*, our student with aspect heads, with VLM2Vec-V2 images and BERTopic caption topics on ArtELingo test ([design record](../../../superpowers/handoffs/2026-11-11-r25-paper-baselines-handoff.md)).

**The level reached.** All 12 cells are WIN for c* at the class count and twenty clusters.
The closest difference is against VLM2Vec-V2 emotion at twenty: +0.0356 [+0.0320, +0.0391] AMI.
WIN permits “better than” for each cell.

**What it does not show.** This covers one set and Qwen c* only.
BERTopic reads captions, while c* and VLM2Vec-V2 read images. Its default topics are very unbalanced.
Prompt, image-size and batch-noise effects on AMI are not checked.

**Options for next.** Paper-row cutoff: 2026-11-14.

1. Put these rows into Table 2. Cost: writing only. The paper chat already has the numbers.
2. Run multimodal BERTopic with CLIP ViT-B/32 and MiniLM, as PercepT did. Cost: a new extraction and CPU fits, unmeasured.
   This could answer the modality caveat.
3. Run VLM2Vec-V2 on MMArt. Cost: another GPU extraction and CPU scoring, unmeasured. This could test the paper's second set.

**Your reading.**

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** These cells support c* at the reported settings, and we are sure of the scoring: the gate is exact and the recompute gap is 0.0.
Against VLM2Vec-V2, d is 2.9 to 10.7 times c*'s seed range r. Against BERTopic the margin is large except in emotion.
We are less sure what the BERTopic rows say about topic models in general: they read text only, with very unbalanced topics.

**Choices that could change a verdict.** MiniLM k-means retains WIN in all six alternative cells.
Its closest difference is emotion at twenty: +0.0151 [+0.0111, +0.0191].
Per-caption topics and VLM2Vec-V2 prompt or image-size alternatives remain unchecked.
The user added class-count scoring before the final run. Both requested counts yield WIN in every cell.

**Our recommendation.** Our view: use the rows now, with the text-only BERTopic caveat.

</details>

**Flags.** Agent defaults: joined captions per painting, and default BERTopic topics reduced to K with reassigned outliers and three UMAP seeds.
VLM2Vec-V2 defaults: the repository prompt and image size, plus bf16 with batch sixteen.
The B1 fix restored missing MiniLM files. The corrected rerun supplies all reported BERTopic rows.

## What ran

VLM2Vec-V2 used its released 2B adapter and each aspect instruction.
BERTopic used MiniLM, UMAP, HDBSCAN and c-TF-IDF, with a fresh fit per count and seed.

c* uses five model seeds. K-means and UMAP each use three states.
The scorer uses 2,000 joint row and c*-seed draws, seed zero, and averages comparator states per draw.
CPU scoring took eleven minutes. Records end at ac48d7e7.

VLM2Vec's Table 3 retrieval row belongs to another chat, “CoSiR paper sec4 tables 2”.

## Results

Each original comparison is WIN because its unrounded lower bound exceeds zero.
Otherwise, LEVEL requires a lower bound above minus r. BELOW covers the remaining outcomes.
r is c*'s own validation seed range for the same aspect and count.

There are 12,281 test paintings. Emotion and style score all rows. Genre scores the 9,949 labelled rows.
Methods fit all rows, then mask missing labels for scoring.
AMI is adjusted mutual information. Higher means closer agreement with labels.
d is c* minus the comparator. Intervals are paired 95% bootstrap intervals, without a correction across cells.

|Aspect|K|c*|Comparator|Comparator AMI|d [95% interval]|r|Label|
|---|---:|---:|---|---:|---|---:|---|
|emotion|9|0.1024|VLM2Vec-V2|0.0638|+0.0386 [+0.0349, +0.0423]|0.0036|WIN|
|emotion|9|0.1024|BERTopic|0.0375|+0.0649 [+0.0603, +0.0691]|0.0036|WIN|
|emotion|20|0.1066|VLM2Vec-V2|0.0710|+0.0356 [+0.0320, +0.0391]|0.0059|WIN|
|emotion|20|0.1066|BERTopic|0.0662|+0.0404 [+0.0360, +0.0443]|0.0059|WIN|
|style|27|0.4109|VLM2Vec-V2|0.3730|+0.0380 [+0.0335, +0.0421]|0.0087|WIN|
|style|27|0.4109|BERTopic|0.1050|+0.3059 [+0.2978, +0.3096]|0.0087|WIN|
|style|20|0.4066|VLM2Vec-V2|0.3700|+0.0366 [+0.0320, +0.0411]|0.0107|WIN|
|style|20|0.4066|BERTopic|0.0936|+0.3131 [+0.3053, +0.3176]|0.0107|WIN|
|genre|10|0.5101|VLM2Vec-V2|0.4593|+0.0508 [+0.0424, +0.0590]|0.0136|WIN|
|genre|10|0.5101|BERTopic|0.2605|+0.2496 [+0.2387, +0.2604]|0.0136|WIN|
|genre|20|0.5001|VLM2Vec-V2|0.4595|+0.0406 [+0.0346, +0.0469]|0.0139|WIN|
|genre|20|0.5001|BERTopic|0.3092|+0.1909 [+0.1816, +0.1994]|0.0139|WIN|

The gate equals r18 exactly at all six cells.
The [independent recompute](../../../../src/test/20261111_paper_baselines/records/check.json) has maximum gap 0.0 over twelve cells.
It refits VLM2Vec k-means and rereads BERTopic partitions. It checks the plumbing, not the shared sklearn library.
Source: [cells.json](../../../../src/test/20261111_paper_baselines/records/cells.json).

## Agent choices checked

The [decisions](../../../../src/test/20261111_paper_baselines/decisions.md) marked yes or maybe follow, ranked by our concern.

1. **BERTopic topic pipeline and read-out.** MiniLM k-means was recomputed through the same fit and scoring functions. All six cells retain WIN.
2. **Captions joined per painting.** Per-caption topics with majority vote were not recomputed because they need new fits and aggregation.
   BERTopic's smallest d is +0.0404, with lower bound +0.0360, at emotion and twenty clusters.
3. **VLM2Vec-V2 prompt, newline and image size.** Alternatives were not recomputed because they need new GPU extraction. The completed extraction took 153 minutes.
   Its smallest d is +0.0356, with lower bound +0.0320.
4. **Interval rule and validation range.** The alternative CLIP range was not recomputed.
   Every lower bound exceeds zero, so original WIN labels do not depend on r.
5. **BERTopic defaults, count correction and outlier handling.** Other settings were not recomputed.
   The correction accounts for the outlier topic. Every row gets a topic. No document is truncated.
6. **MiniLM fix and class-count scope.** See the stale comparison below. Both user-requested counts yield only WIN cells.

The [extra record](../../../../src/test/20261111_paper_baselines/records/alt_minilm_kmeans.json) gives this exploratory read-out:

|Aspect|K|MiniLM k-means AMI|d [95% interval]|Label|
|---|---:|---:|---|---|
|emotion|9|0.0841|+0.0183 [+0.0143, +0.0224]|WIN|
|emotion|20|0.0915|+0.0151 [+0.0111, +0.0191]|WIN|
|style|27|0.1074|+0.3035 [+0.2945, +0.3063]|WIN|
|style|20|0.1010|+0.3056 [+0.2971, +0.3091]|WIN|
|genre|10|0.2884|+0.2216 [+0.2114, +0.2310]|WIN|
|genre|20|0.2949|+0.2052 [+0.1965, +0.2129]|WIN|

Names and embedding hashes match. The full-draw check reuses the recorded candidate gate and states that in each cell.
MiniLM k-means tests the embedding read-out. BERTopic with an internal k-means cluster model remains unchecked.

## Analysis

Our reading: c* leads both comparators here. MiniLM k-means narrows the emotion gap, while the BERTopic setup caveats remain.
Emotion AMI is low for every method (c* 0.1024 at nine), so the emotion gaps are small in absolute terms.
Captions often name emotions. Their effect on BERTopic's emotion AMI is not checked.
The user chose text-only BERTopic against image-only comparators.
PercepT's multimodal BERTopic used CLIP ViT-B/32 and MiniLM with one hundred topics. That setup is not checked here.

BERTopic topics are very unbalanced. The `bertopic/topics_*.json` sidecars give these seed ranges:

|Count|Largest topic|Smallest topic|Four largest topics together|
|---|---|---|---|
|20|2,478 to 3,269|14 to 21|7,955 to 9,024|
|9|5,111 to 6,050|17 to 33|11,301 to 11,940|
|10|5,120 to 6,036|18 to 33|11,020 to 11,762|

Our reading: this imbalance can lower AMI and belongs to the default pipeline. Its isolated effect is not checked.

VLM2Vec-V2 bf16 vectors move with batch composition: cosine against batch one ranged from 0.9986 to 0.9997, around the 0.999 build bar.
The fp32 check gave 1.0. The run chat accepted bf16 at batch sixteen.
The AMI effect is not checked. Cosine noise and AMI differences use different units, so their sizes guarantee no label.

c*'s test values were known from earlier rounds.
Defaults were set at 2026-11-11 05:10 before comparator scores. The user supplied class counts.
The decision record says no round choice used test results. That process was not independently checked.

## Deviations

Review finding B1 identified missing sentence-transformers files in cached MiniLM.
The first run used mean pooling without Normalize and a fallback token limit.
The fix restored four files at the same revision and added a fallback guard, then reran BERTopic.
The corrected model uses Normalize and a 256-token limit. The longest document has 242 tokens.

Stale and final AMI from check.json, at twenty clusters:

|Aspect|Stale BERTopic AMI|Final BERTopic AMI|
|---|---:|---:|
|emotion|0.0720|0.0662|
|style|0.0927|0.0936|
|genre|0.3137|0.3092|

The handoff reports no stale-to-final label changes. check.json holds no stale bootstrap intervals.
The user added class counts through the paper chat. Twenty-cluster results remain supplementary.
Two side venvs over the CoSiR environment supplied dependencies. The build records say the CoSiR packages stayed unchanged.

## Claim review

The controller (Claude) recomputed five numbers of the brief and the results from the record files, not from this report.

|Number|In the report|Recomputed|File|
|---|---|---|---|
|VLM2Vec-V2 emotion, 20: d [95% interval]|+0.0356 [+0.0320, +0.0391]|+0.0356 [+0.0320, +0.0391]|`records/cells.json`|
|BERTopic style, 20: AMI and d|0.0936; +0.3131|0.0936; +0.3131|`records/cells.json`|
|MiniLM k-means emotion, 20: d [95% interval]|+0.0151 [+0.0111, +0.0191]|+0.0151 [+0.0111, +0.0191]|`records/alt_minilm_kmeans.json`|
|Stale and final BERTopic emotion AMI, 20|0.0720; 0.0662|0.0720; 0.0662|`records/check.json`|
|BERTopic topic sizes at 20: largest; four largest|2,478 to 3,269; 7,955 to 9,024|2,478 to 3,269; 7,955 to 9,024|`outputs/bertopic/topics_seed*.json`|

The genre labelled count (9,949) matches `n_labelled` in all four genre cells.
The one decisions.md line marked "yes" (the class-count scope) is in the brief's folded block. No gap was found.

## Files and storage

Keep final stores and partitions. The handoff records these sizes:

|Item|Location|Size|
|---|---|---:|
|VLM2Vec stores, kept|`/data/SSD2/pre_extract/artelingo/vlm2vec_v2/`|218 MB|
|Round outputs|`/data/SSD2/cosir_r25/20261111_paper_baselines/outputs/`|48 MB|
|VLM2Vec code|`/data/SSD2/cosir_r25/ext/`|26 MB|
|VLM2Vec venv|`/root/venvs/vlm2vec`|148 MB|
|BERTopic venv|`/root/venvs/bertopic`|177 MB|

No round item exceeded one GB at handoff. Its validation extraction belongs to the Table 3 chat.
The handoff reported no other running work or uncommitted round changes.
The constitution file was already modified before r25.

Nothing was deleted. Pending removals are the stale BERTopic run, the two merged build worktrees and branches, and scratch smoke files.
The pending-deletions list records exact locations.
