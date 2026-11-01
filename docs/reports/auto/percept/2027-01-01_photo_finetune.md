# Photo fine-tune: above the plain image and the teacher on Affection (holds); the CRL claim passes

> 2026-11-01 (sequence name 2027-01-01) · Spec: [design](../../../superpowers/specs/2026-10-31-photo-finetune-design.md) · Decision rule: [§6 with §3.4 and §3.5](../../../superpowers/specs/2026-10-31-photo-finetune-design.md#6-success-criteria) @ cf0444c (approval commit), no amendments · Run log: [round log](../../../../src/test/20261031_photo_finetune/20261031_photo_finetune_log.md) · Previous: [r12](2026-12-31_affection_confirmation.md)

## Results brief

**What we tried.** We fine-tuned c*, our network trained on paintings, on 69,927 Affection train photos and their free-text explanations, without emotion labels ([spec](../../../superpowers/specs/2026-10-31-photo-finetune-design.md)). Settings, seed ranges and read-outs were picked on validation and frozen. Then we tested it on the 8,501 held photos against the plain Qwen image and its teacher, the Qwen image instructed for emotion.

**Outcome against the bar.** **Holds.** The photo model is a WIN over the plain Qwen image and over the teacher in both tasks. The teacher bar was LEVEL or WIN, so the licensed text says "level with or better than" the teacher. It also passes the separate CRL retrieval bar.

|Task|Photo model|Teacher|Photo minus teacher [95%]|
|---|---|---|---|
|Clustering AMI|0.1838|0.1666|+0.0172 [+0.0121, +0.0225]|
|Retrieval mAP|0.3172|0.2945|+0.0227 [+0.0210, +0.0244]|

**What it does not show.** This covers emotion on one photo dataset and Qwen backbone, with five fresh runs. Photo training followed r12's test result on these same held photos. It does not license transfer without photo training, or one model for paintings and photos. The sweep's pick led the next row by less than half of the photo model's seed range.

**Options for next.**

1. Freeze this result and write: no new compute, but writing time. This fixes the licensed claim for the abstract (2026-11-10) and paper (2026-11-16).
2. Run r14's large sweep after the freeze: a new compute budget could test recipe and pick sensitivity.
3. Test a second photo dataset: data preparation and a new protocol budget could test reach beyond Affection.

**Your reading.**

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading:** photo training reverses the teacher gap in both tasks, across sources and without tied labels. Confidence here is high; device differences reduce confidence in the exact sweep pick.

**Our recommendation:** freeze this photo result and write before the abstract deadline. Keep c* for paintings and the large sweep after the freeze.

</details>

## What ran

The emotion pathway trained on photos; comparators reuse r12 outputs.

|Stage|Type|Role|
|---|---|---|
|0a|regression check|Views, parity, replays, graphs, loss and near-copies|
|1|sweep|Validation, seed 0; commit sheet before final runs|
|2|measurement|Five fresh runs; commit ranges and read-out before test|
|3|claim test|Affection held photos; all settings frozen|
|4|descriptive|Diagnostics after verdict; decide nothing|

|Data|Photos|Role|
|---|---|---|
|Affection train|72,255 before filter; 69,927 fit|Train without emotion labels|
|Affection validation|4,251|Stopping, sweep, ranges and read-out|
|Affection test|8,501|Held claim test|

c*'s emotion view (its image side and its explanation side) trains from c*'s weights, on fixed backbone features. Training uses corrected explanation graphs, buddy terms and the image teacher. The pick changes graph neighbours to 80.

Final runs start from matching c* seeds, 0 to 4, with training seed 100 + s. No sweep run enters test. Plurality emotion labels use alphabetical tie breaks. Train labels are never read. Default-image cosine ≥ 0.85 to validation or test removes train photos and their explanations.

Measurement is r12's, unchanged: AMI with 20 clusters (mean of three states), mAP with read-outs picked on validation, paired bootstraps (2,000 clustering, 10,000 retrieval draws), 95% intervals of photo model minus comparator. WIN: lower bound > 0; LEVEL: lower bound > −r', where r' is the photo model's validation seed range (maximum minus minimum over five seeds); otherwise BELOW.

Code: main `949ad04`, local RTX 3090 under the GPU lock. Full queue: 2026-11-01, 18:43–21:24; all steps exited 0. Timings and checks are in the run log.

All ten regression items passed (`stage0a/gates.json` and part files):

|Check|Evidence|
|---|---|
|Image parity, 200 rows|Minimum cosine 0.9997; bar ≥ 0.999|
|Text parity, 200 rows|Minimum cosine 0.9994; bar ≥ 0.999|
|New views|72,255 images; 474,051 explanations; keys join, finite, unit norm within 0.001|
|Replays|Within 1e-9; ranges, read-out scores and pick match|
|Frozen c* inference|Maximum difference 0, five seeds, both splits; bar 0.001|
|Painting graphs|All three saved graphs reproduced exactly|
|Training loss|6.161110545314756; absolute difference 0; bar 0.0001|
|Near-copies|2,328 removed; count and histogram saved|

`stage0a/nearcopy.json` records removals by source and nearest split, the cosine histogram, and within-train copies.

The sweep picked 80 neighbours. A row's clustering gap is its AMI minus the teacher's validation AMI, divided by r12's c* AMI range (0.00343). Its retrieval gap is the same for mAP, with r12's c* mAP range (0.00101). The score is the smaller gap. The small mAP range makes retrieval gaps look large. Teacher validation: AMI 0.1719, mAP 0.3044. Read-out: centering plus ten-neighbour expansion.

|Row|Changed setting from c* recipe|AMI|mAP|Clustering gap|Retrieval gap|Score|
|---|---|---|---|---|---|---|
|0|Frozen c*, reference|0.1427|0.2856|-8.51|-18.65|-18.65|
|1|None; c* start, lr 0.001, k 40, correction on, teacher weight 0.25|0.1769|0.3227|+1.45|+18.13|+1.45|
|2|Learning rate 0.0003|0.1873|0.3225|+4.49|+17.95|+4.49|
|3|Learning rate 0.0001|0.1841|0.3240|+3.55|+19.44|+3.55|
|4|k = 20|0.1857|0.3171|+4.04|+12.58|+4.04|
|5|k = 80; PICK|0.1905|0.3280|+5.42|+23.37|+5.42|
|6|Content correction off|0.1570|0.2932|-4.33|-11.06|-11.06|
|7|Teacher weight 1|0.1806|0.3147|+2.52|+10.15|+2.52|
|8|Random start|0.1801|0.3227|+2.40|+18.15|+2.40|
|9|lr 0.0003; k 80; teacher weight 1|0.1837|0.3167|+3.43|+12.17|+3.43|

All nine rows completed; the [sheet](../../../../src/test/20261031_photo_finetune/stage1_sheet.json) froze at `ee67eb2`. The pick leads row 2 by 0.93 score units, which is 0.0032 AMI: less than half of the photo model's own AMI seed range (0.00707). Training is not deterministic across devices (final review): in the build, one row trained on GPU and on CPU differed by about 0.6 score units, near the 0.5 tie margin. So the exact pick is close to noise.

All five final runs stopped at best epoch 1; the validation loss rose from epoch 2. The [record](../../../../src/test/20261031_photo_finetune/stage2_records.json) froze ranges and read-out at `dd51dde`, before test. The picked read-out subtracts the validation mean and expands each query with ten neighbours.

|Read-out, validation, cross-fit|mAP|
|---|---|
|Centering plus ten-neighbour expansion (pick)|0.3275|
|Plain cosine|0.3149|

Its score is 5.19, in units of the photo model's mAP range.

## Results

**Verdict under the rule: holds**, final. Both plain-image bars and both teacher bars pass; the separate CRL claim also passes. Frozen c* is shown beside them without a bar.

|ID|Metric|Comparator|Photo model|Comparator number|Difference [95%]|Bar|Outcome|
|---|---|---|---|---|---|---|---|
|SC1|AMI|Plain Qwen image|0.1838|0.1041|+0.0797 [+0.0741, +0.0857]|WIN|WIN|
|SC2|mAP|Plain Qwen image|0.3172|0.2452|+0.0720 [+0.0699, +0.0742]|WIN|WIN|
|SC3|AMI|Teacher, artwork wording|0.1838|0.1666|+0.0172 [+0.0121, +0.0225]|LEVEL or WIN|WIN|
|SC4|mAP|Teacher, artwork wording|0.3172|0.2945|+0.0227 [+0.0210, +0.0244]|LEVEL or WIN|WIN|
|SC5|mAP|CRL, CLIP L/14, painting|0.3172|0.2739|+0.0433 [+0.0411, +0.0455]|WIN|WIN|
|Descriptive|AMI|Frozen c*|0.1838|0.1384|+0.0454 [+0.0408, +0.0501]|None|No claim bar|
|Descriptive|mAP|Frozen c*|0.3172|0.2755|+0.0417 [+0.0399, +0.0436]|None|No claim bar|

Retrieval uses centering plus ten-neighbour expansion; teacher wording and CRL retain r12 picks. Frozen c* uses paired seeds.

|Independent check|Evidence|
|---|---|
|C13, `rederive/rederive.json`|`ok: true`; 378 items; 0 mismatches within 1e-6|
|Finality|`stage3/verdict.json` written 21:17; successful compare written 21:22, after it|

**Licensed claim, spec §6 (fine-tuned start):**

> On 8,501 held Affection photos, c* fine-tuned without emotion labels, from the photos' free-text explanations, on Affection's train photos minus near-copies of held photos, with its settings picked on Affection validation, clusters and retrieves photos by emotion better than the plain Qwen3-VL-Embedding-2B image, and level with or better than the Qwen image instructed for emotion (AMI with 20 clusters; mAP with read-outs picked on validation; 5 seeds).

**Selection disclosure:** the decision to train on photos came after r12's test result on these same held photos. Stage 3 is a **claim test**; r12's failed confirmation stays failed. On Affection validation, r12 tried 2 teacher wordings, up to 25 read-outs per arm and 4 CRL versions; this round adds 9 sweep rows and up to 25 read-outs for the photo model. Validation numbers are inflated by this selection; only test numbers are results. Every pick froze before test scoring.

The CRL sentence is licensed:

> The photo model retrieves photos by emotion better than our re-implementation of CRL (CLIP L/14, picked on validation) on held Affection.

Not licensed: one model for paintings and photos; transfer without photo training; style or genre on photos; other photo datasets. Our addition: no from-scratch claim (no from-scratch run went to test), and no claim from the descriptive frozen c* rows.

## Analysis

The photo model stays first in every diagnostic view: with plain cosine, in nDCG, with tied photos dropped and in every source. Per class, the teacher is slightly above it on contentment. All use `diag/diag.json` and decide nothing.

Teacher gain share = (model minus plain) / (teacher minus plain), point values only; frozen c* is unchanged from r12.

|Task|Photo model share|Frozen c* share (r12)|
|---|---|---|
|AMI|127.5%|54.9%|
|Picked mAP|145.9%|61.5%|

Plain cosine and graded nDCG keep the order. nDCG uses emotion overlap without query expansion, also after the picked transform.

|Arm|R0: plain-cosine mAP|R0: nDCG|Picked-transform nDCG|
|---|---|---|---|
|Photo model|0.3068|0.9396|0.9388|
|Frozen c*|0.2670|0.9315|0.9310|
|Plain Qwen image|0.2383|0.9250|0.9253|
|Teacher|0.2884|0.9372|0.9366|
|CRL|0.2639|0.9314|0.9319|

Untied: holds, all four WIN. Remove 1,917 ties; refit partitions, queries and gallery on 6,584 photos. Read-outs and r' stay fixed; CRL is not repeated.

|ID|Untied comparison|Difference [95%]|Outcome|
|---|---|---|---|
|SC1|AMI vs Plain Qwen image|+0.1098 [+0.1021, +0.1179]|WIN|
|SC2|mAP vs Plain Qwen image|+0.0949 [+0.0920, +0.0980]|WIN|
|SC3|AMI vs Teacher, artwork wording|+0.0312 [+0.0247, +0.0379]|WIN|
|SC4|mAP vs Teacher, artwork wording|+0.0267 [+0.0243, +0.0292]|WIN|

Every source has photo > teacher > frozen c* > plain. All share the test gallery.

|Source|Queries|Photo mAP|Teacher mAP|Frozen c* mAP|Plain mAP|
|---|---|---|---|---|---|
|coco_affective|2,308|0.3006|0.2795|0.2688|0.2423|
|emotional_machines_affective|1,013|0.3409|0.3134|0.2885|0.2516|
|flickr30k_affective|1,375|0.2981|0.2788|0.2675|0.2394|
|visual_genome_affective|1,671|0.3018|0.2806|0.2690|0.2428|
|you_aaai_affective|2,134|0.3483|0.3229|0.2869|0.2508|

Our reading: FI (`you_aaai_affective`) and Emotional Machines gain most over frozen c*. All sources exceed the teacher.

|Class|Queries|Photo AP|Teacher AP|Frozen c* AP|Photo minus teacher|
|---|---|---|---|---|---|
|amusement|1,651|0.2973|0.2732|0.2845|+0.0241|
|anger|81|0.0445|0.0377|0.0255|+0.0068|
|awe|1,460|0.3321|0.2785|0.2826|+0.0536|
|contentment|2,626|0.3718|0.3784|0.3454|-0.0065|
|disgust|351|0.3272|0.1975|0.1985|+0.1297|
|excitement|1,185|0.3167|0.2989|0.2656|+0.0178|
|fear|448|0.2005|0.1605|0.1511|+0.0400|
|sadness|501|0.2997|0.2872|0.1822|+0.0125|
|something else|198|0.0550|0.0526|0.0377|+0.0024|

Our reading: awe and disgust carry most of the query-weighted teacher lead; contentment stays slightly below. Against frozen c*, awe, contentment, excitement, sadness and disgust carry most gain.

The photo model's validation ranges are about twice r12's c* ranges in AMI and mAP. Its held spreads are wider than its own validation ranges and than r12's held spreads. Every criterion is WIN, so the wider LEVEL tolerance did not decide the verdict.

|Metric|New validation r'|r12 validation r|New held spread|r12 held spread|
|---|---|---|---|---|
|AMI|0.00707|0.00343|0.00801|0.00488|
|mAP, picked|0.00242|0.00101|0.00323|0.00085|
|mAP, plain cosine|0.00242|0.00101|0.00266|0.00096|
|nDCG, plain cosine|0.00032|0.00034|0.00035|—|

Ranges: five decimals; AMI and mAP points: four.

The random start (row 8) and the c* start (row 1) are level on validation, on one seed: AMI +0.0033, inside the photo model's AMI range (0.00707), and mAP +0.0000. Our reading: on photos the painting start may add little; one seed cannot show it.

|Sweep comparison, seed 0|Random start (row 8)|c* start (row 1)|Difference|
|---|---|---|---|
|AMI|0.1801|0.1769|+0.0033|
|mAP|0.3227|0.3227|+0.0000|

Every graph exceeds chance for edges whose photos share the plurality emotion. Labels enter only this diagnostic.

|Graph (rows)|Validation edges|Same-emotion share|
|---|---|---|
|k 40, correction on (1, 2, 3, 7, 8)|28,894|54.2%|
|k 20, correction on (4)|11,743|55.6%|
|k 80, correction on (5, 9)|68,562|52.5%|
|k 40, correction off (6)|48,439|55.6%|

Chance (random pairs): 19.3%. Row 0 has no photo graph.

Our reading: label share does not predict the best model. More neighbours won despite a lower share; removing content correction raises share but lowers score.

## Deviations

One departure in order, no amendment. §3.4 says C13 re-derives before the verdict is written; the queue wrote `stage3/verdict.json` first (21:17), as in r12, and C13 compared after it (21:22: 378 items, 0 mismatches). The verdict file was not changed. The final review accepted this order on the condition that the verdict counts only after the compare passes, which it did.

Two smoke-1 fixes preceded the full run: use full bootstrap draws for interval replay; use verdict read-out normalization for plain-cosine diagnostics (bare cosine differed by about 3e-9). Both reruns passed. Final-review guards preceded smoke 2, which passed, including the verdict/compare/diagnostic rerun.

Builds 1 to 5 set agent defaults (data subsets for smoke runs, parity keys, seeds of the sweep and final runs, run reuse, comparator sources); the run log lists them under "Agent defaults". The within-train near-copy count (10,114) counts photos; 1,057 of them are also near-copies of held photos.

The final review asked for a check of the text join on the full explanation view; it passed during the run. Matched cosine 0.5972, against 0.3842 to 0.3843 for shifted rows; the matched row won in 94.4% of rows.

Queue: 2 h 41 min versus the 5 to 6 h estimate. Extraction and training were faster (`queue_seconds.txt`). No second candidate went to test.

## Files and storage

Storage is **to be decided at the next wrap-up (disk ledger)**. Nothing was deleted.

- Outputs: `/data/SSD2/cosir_r13/20261031_photo_finetune/outputs/` (849 MB): gates, near-copy record, sweep rows, verdict, re-derivation, diagnostics and queue timings cited above; checkpoints, exports, partitions and per-query arrays.
- Views under `/data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b_aspect_prompts/`: `emotion_focus_train` (286 MB) and `emotion_focus_txt` (1.9 GB). Sizes: run log.
