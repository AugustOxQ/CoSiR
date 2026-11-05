# Text retrieval: c* passes the label bar, with weak known-pair hits

> 2026-11-05 (sequence name 2027-01-09) · Spec: none (small loop, design in the round log) · Decision rule: round log, “The rule” · [Run log](../../../../src/test/20261105_text_retrieval/20261105_text_retrieval_log.md)

## Results brief

**What we tried.** We tested c*'s caption side as a text encoder in text to image, image to text and text to text.
We scored label and known-pair hits on 6,152 validation paintings and 30,872 captions with seed 0.
We compared c* with plain Qwen, the teacher's text and three plain encoders.
RoBERTa supplied an emotion text to text comparator.
We also tested SemArt, MMArt perspectives and word queries.

**Outcome against the bar.** Outcome 1 passed: c* exceeded both main comparators in 6 of 9 label cells, against a bar of 5.
All 9 exceeded plain Qwen.
Style's 3 cells and genre's 3 exceeded the teacher; emotion's 3 fell below it.
Each cell below lists label mAP for text to image, image to text and text to text, in that order.

| Label mAP | c* | Plain Qwen | Teacher | Chance |
|---|---|---|---|---|
| emotion | 0.2650/0.2898/0.3455 | 0.2236/0.2407/0.2541 | 0.2714/0.3036/0.3790 | 0.1723/0.1723/0.1565 |
| style | 0.1720/0.1359/0.1072 | 0.1342/0.1079/0.0940 | 0.1578/0.1219/0.0953 | 0.0756/0.0756/0.0753 |
| genre | 0.4877/0.3989/0.3073 | 0.4179/0.3390/0.2669 | 0.4470/0.3932/0.2951 | 0.1404/0.1404/0.1401 |

Each fixed read-out gave 5 of 9 cells; cross-fit picks gave 6 of 9.
Style's three cells and genre text to image held in every check.
Genre image to text and genre text to text depended on the read-out.
Genre text to text exceeded the teacher only with expansion; raw cosine and centering fell below.

Pair hits fall below plain Qwen for every c* head in every direction.
Text to image and image to text fall far below, as the best-head R@10 table shows.
Text to text is close.

| Direction | c* (best head) | Plain Qwen | Teacher (best head) |
|---|---|---|---|
| text to image | 0.2118 (genre) | 0.3102 | 0.1604 (genre) |
| image to text | 0.3186 (genre) | 0.5068 | 0.2466 (emotion) |
| text to text | 0.1756 (genre) | 0.1823 | 0.1261 (genre) |

Two items are the user's to decide.
The first is whether the paper states this pair-hit limit.
The second is how to treat the free variant, which sends captions through the image side.
It exceeds both comparators in 3 of 9 cells, which would give Outcome 2; the rule reads c*.

**What it does not show.** This uses one seed and validation data only, so it supports no claim.
Read-out selection and scores use the same rows; cross-fit checks this.
`tx_checks.py` followed the first read.
A painting's emotion uses its captions' plurality label.
MMArt texts are generated and long, so only part is comparable.
Of the 5,769 emotional texts, 73.8% contain a word from their own emotion's word list, so that row leaks.
SemArt uses its validation split, not the published test split.
c*'s caption side trained on caption means, so single captions and short words are new inputs.
The code and export gate checked this training input.
Lenses and CANVAS were not re-checked.

**Options for next.**

1. The rule calls for a medium claim loop on held ArtELingo and the MMArt test rows.
   Our additions are five seeds and read-outs picked on validation.
   Held ArtELingo teacher extraction covers 61,744 captions by 3 instructions.
   At about 295 per second, this takes about 10.5 minutes, by our estimate; that rate comes from short captions.
   MMArt needs 11,528 rows by five texts; we have no cost estimate because these texts are longer.
   This loop tells which cells hold.
   The handoff estimates a claim test read by about 10 November.
2. Option u joins c* and plain scores with a weight picked on validation.
   It uses saved vectors and takes minutes.
   It tests whether joined scores recover pair hits and what that costs on label hits.
   The handoff allows it when pair hits are weak.
3. Option t retrains the caption side on single captions and freezes the image side.
   It takes hours by our estimate; the rule did not trigger it.
   It tests whether single-caption training changes the emotion gap to the teacher.

**Your reading.**

<details>
<summary>Our reading and recommendation (open after yours)</summary>

**Our reading.** c* works on label hits.
It exceeds plain Qwen with intervals and all three plain encoders by mean score in every Table 1 cell.
Style and genre exceed the teacher, which needs an encoder pass per aspect.
The count has a thin margin: four cells hold throughout, and two depend on the read-out.
Emotion falls below the teacher; its text to image gaps sit in contentment, fear and sadness.
Macro mAP is close for text to image and image to text, with gaps of 0.0011 and 0.0022.
It is not close for text to text, with a gap of 0.0298.
Low pair hits fit emotion and style graphs that subtract content.
c* works as an aspect retriever; as a caption matcher it is limited for text to image and image to text.

**Our recommendation.** We recommend option 1 after option 2, which takes minutes and checks the pair limit.
We would claim style and genre label hits and state emotion as below the teacher.
The user decides the pair-limit wording.

</details>

## What ran

Exploration, seed 0, no held rows. SemArt/MMArt validation: 1,069/5,769 paintings. RoBERTa: 768-d. mAP/macro mAP: query/class means; chance: relevant share. Pairs: own painting/own caption/another own caption; exclude self. Missing genre leaves both sides.

## Results

### Table 1: label mAP

Directions: text to image/image to text/text to text. Best mAP read-out: centering plus expansion (k = 10). Centering exceptions: c*/teacher genre image to text; plain emotion text to image; plain/teacher style image to text; plain encoders emotion text to text; SigLIP style image to text. RoBERTa: raw cosine. Dash: not checked.

Row|Emotion|Emotion|Emotion|Style|Style|Style|Genre|Genre|Genre
---|---|---|---|---|---|---|---|---|---
c*|0.2650|0.2898|0.3455|0.1720|0.1359|0.1072|0.4877|0.3989|0.3073
Free variant|0.2577|0.2768|0.3168|0.1467|0.1116|0.0921|0.4537|0.3968|0.3049
Plain Qwen|0.2236|0.2407|0.2541|0.1342|0.1079|0.0940|0.4179|0.3390|0.2669
Teacher|0.2714|0.3036|0.3790|0.1578|0.1219|0.0953|0.4470|0.3932|0.2951
CLIP L/14|0.2165|0.2232|0.2023|0.1516|0.1076|0.0909|0.4168|0.3569|0.2557
SigLIP 2|0.2116|0.2214|0.2029|0.1495|0.1056|0.0893|0.4293|0.3459|0.2447
SigLIP|0.2143|0.2229|0.2065|0.1529|0.1065|0.0892|0.4231|0.3511|0.2471
RoBERTa|—|—|0.3099|—|—|—|—|—|—
Chance|0.1723|0.1723|0.1565|0.0756|0.0756|0.0753|0.1404|0.1404|0.1401

Emotion macro mAP:

Row|text to image|image to text|text to text
---|---|---|---
c*|0.1911|0.2171|0.2962
Plain Qwen|0.1525|0.1615|0.1969
Teacher|0.1922|0.2193|0.3260
RoBERTa|—|—|0.2527

### The rule

Paired bootstrap: 95%, 1,000 painting resamples, seed 0; both intervals must exceed zero. c* minus comparator:

Aspect / direction|Minus plain Qwen|Minus teacher|Above both
---|---|---|---
emotion / text to image|+0.0414 [+0.0406, +0.0421]|-0.0064 [-0.0071, -0.0058]|no
emotion / image to text|+0.0491 [+0.0466, +0.0517]|-0.0139 [-0.0167, -0.0111]|no
emotion / text to text|+0.0914 [+0.0899, +0.0930]|-0.0335 [-0.0343, -0.0326]|no
style / text to image|+0.0378 [+0.0361, +0.0395]|+0.0142 [+0.0125, +0.0160]|yes
style / image to text|+0.0280 [+0.0271, +0.0289]|+0.0140 [+0.0134, +0.0147]|yes
style / text to text|+0.0132 [+0.0126, +0.0138]|+0.0119 [+0.0113, +0.0126]|yes
genre / text to image|+0.0698 [+0.0676, +0.0722]|+0.0407 [+0.0375, +0.0437]|yes
genre / image to text|+0.0599 [+0.0573, +0.0623]|+0.0057 [+0.0044, +0.0071]|yes
genre / text to text|+0.0404 [+0.0390, +0.0419]|+0.0122 [+0.0106, +0.0138]|yes

**Verdict under the rule:** Outcome 1.

### Read-outs

Cross-fit swaps painting halves; checked after first read.

Read-out|Above both|Genre image to text|Genre text to text
---|---|---|---
raw cosine|5 of 9|yes|no
centering|5 of 9|yes|no
centering plus expansion|5 of 9|no|yes
Cross-fit picks|6 of 9|yes|yes

### Pair hits

Best reciprocal-rank read-outs. Cells: R@1/R@5/R@10/median rank. c* text to text R@10 exceeds all three plain encoders (best 0.1516).

Row/head|text to image|image to text|text to text
---|---|---|---
c* emotion|0.0200/0.0615/0.0934/267.5|0.0348/0.1105/0.1666/107|0.0187/0.0485/0.0686/875
c* style|0.0036/0.0163/0.0278/727|0.0060/0.0221/0.0416/459|0.0289/0.0649/0.0911/600
c* genre|0.0571/0.1498/0.2118/88|0.0865/0.2285/0.3186/30|0.0597/0.1318/0.1756/198
Free variant best head|0.0655/0.1470/0.1984/137|0.1403/0.2848/0.3708/26|0.0468/0.1036/0.1394/320
Plain Qwen|0.1178/0.2442/0.3102/49|0.2081/0.4064/0.5068/10|0.0675/0.1397/0.1823/195
Teacher best head (genre / emotion / genre)|0.0447/0.1137/0.1604/156|0.0744/0.1775/0.2466/67|0.0376/0.0902/0.1261/312
Best plain encoder|0.1270/0.2514/0.3180/46 (SigLIP 2)|0.2326/0.4358/0.5382/8 (SigLIP 2)|0.0533/0.1136/0.1516/311 (SigLIP)

Free heads: emotion/emotion/genre. Centering except c* genre image to text/text to text, emotion text to text: raw cosine; style text to image: expansion.

## Analysis

### Leak checks

Without label-name captions, emotion: c* above plain Qwen, below the teacher. Style and genre: Table 1 order (c* above both; point values only). Full-row read-outs.

Label-name share|emotion|style|genre
---|---|---|---
ArtELingo|0.1146|0.0020|0.0282

Emotion mAP, no leak|text to image|image to text|text to text
---|---|---|---
c*|0.2616|0.2816|0.3095
Plain Qwen|0.2257|0.2400|0.2463
Teacher|0.2673|0.2936|0.3351

RoBERTa emotion text to text, all/without leak: 0.3099/0.2683.

Cross-painting order holds; emotion text to image c*/plain/teacher 0.2647/0.2232/0.2712.

Emotion AP, text to image|c*|Teacher
---|---|---
contentment|0.4684|0.5036
fear|0.2739|0.2954
sadness|0.1944|0.2047

### Caption means

Single captions fall below means on style and genre label mAP and all three pair R@10 for c*. Emotion label mAP differs less (0.2650/0.2905) than chance (0.1723/0.2019). Plain Qwen also gains (pair R@10 0.7243 versus 0.3102); the gap alone does not show training mismatch.

Training used one caption mean/painting; code and export forward-pass gate checked this. Means: label, centering; pair, best reciprocal rank.

Row/aspect|Label mAP single/mean|Pair R@10 single/mean
---|---|---
c* emotion|0.2650/0.2905|0.0934/0.3230
c* style|0.1720/0.2036|0.0278/0.0809
c* genre|0.4877/0.5378|0.2118/0.6174

### SemArt

SemArt: validation, not published test. Stored descriptions scrub author/title/date tokens. Raw: unscrubbed description only, no separate title/attribute fields; names appear only where the description contains them. Best reciprocal-rank read-outs:

Validation R@10|text to image|image to text
---|---|---
c* emotion|0.1787|0.1862
c* style|0.0926|0.1066
c* genre|0.3976|0.3452
Free variant genre|0.4687|0.3994
Plain Qwen|0.5968|0.5341
Plain Qwen, raw|0.6268|0.5529
CLIP B/32|0.4331|0.4200
SigLIP 2|0.5154|0.4808

Read-out: centering. CLIP L/14, SigLIP: not checked; text absent.

Published text ([source](../../literature/2027-01-08_text_cross_modal_retrieval.md)):

Published row|text to image R@1 / R@5 / R@10|image to text R@1|Median rank
---|---|---|---
2018 paper|0.144 / — / about 0.455|0.138|—
2018 README CML|0.164/0.386/0.505|—|10
2018 README CCA|0.117/0.283/0.377|—|25

Paper/README conflict, cause unstated. Metadata rows and CANVAS rows excluded.

### MMArt

Matching head best on pairs? No. c*: genre leads all five; emotion loses on emotional (0.1723/0.5526), style on formal (0.0530/0.3280). Free: emotion leads narrative/formal/unified; genre leads emotional/historical. Text to image R@10, centering:

Perspective|emotion|style|genre
---|---|---|---
narrative|0.3850|0.0979|0.6523
formal|0.1167|0.0530|0.3280
emotional|0.1723|0.0950|0.5526
historical|0.0279|0.0499|0.1182
unified|0.3456|0.1408|0.7339

Label mAP: emotion/style and genre labels; columns: c* heads; best read-outs.

Perspective|emotion|style|genre
---|---|---|---
narrative|0.2040/0.1358/0.4000|0.1947/0.2232/0.3487|0.1920/0.1738/0.4941
formal|0.1920/0.1377/0.3621|0.1916/0.2263/0.3088|0.1877/0.1894/0.4757
emotional|0.2648/0.1322/0.3079|0.1978/0.2252/0.3196|0.2029/0.1873/0.4798
historical|0.1854/0.1347/0.2602|0.1892/0.3383/0.3091|0.1827/0.2757/0.3603
unified|0.2180/0.1436/0.3821|0.1947/0.3085/0.3395|0.1941/0.2431/0.5080

Chance: emotion 0.1698, style 0.0852, genre 0.1402. Centering; emotion-label raw cosine: narrative, style head, unified genre head.

Matching heads: emotional text, emotion labels: emotion head leads (0.2648 versus 0.2029/0.1978). Formal text, style labels: style head leads (0.2263 versus 0.1894/0.1377). Style head leads all five texts on style labels. Historical text scores highest (0.3383 versus formal 0.2263).

README generators: Qwen3-VL-8B-Instruct, narrative/emotional/historical; GalleryGPT, formal; Qwen3-8B, unified. Emotional text also used ARTEMIS-v2 crowd reactions and metadata conditioning. **Our caution:** text may favour Qwen encoders.

### Word queries

Macro mAP, best read-out; one query/class. R: raw cosine; C: centering. No expansion.

Query|c*|Free variant|Plain Qwen|Teacher|Best plain encoder|Chance
---|---|---|---|---|---|---
emotion bare text to image|0.2671C|0.2261C|0.2027R|0.2718C|0.1827R SigLIP 2|0.1171
emotion bare text to text|0.4417C|0.3861C|0.3049C|0.5481C|0.1900C SigLIP 2|0.1110
emotion template text to image|0.3074R|0.2739C|0.2197C|0.2801R|0.1950R SigLIP|0.1171
emotion template text to text|0.5205R|0.4430C|0.2412C|0.5536C|0.1616C SigLIP|0.1110
style bare text to image|0.2680R|0.3477R|0.2319C|0.3883R|0.3664R SigLIP 2|0.0370
style bare text to text|0.0853C|0.0712R|0.0631C|0.0771C|0.0586C CLIP L/14|0.0370
style template text to image|0.3115R|0.3346C|0.2069C|0.3747C|0.3256R SigLIP 2|0.0370
style template text to text|0.0886C|0.0668R|0.0495C|0.0785C|0.0533C CLIP L/14|0.0370
genre bare text to image|0.5947R|0.6849C|0.5932R|0.7063C|0.6779R SigLIP 2|0.1000
genre bare text to text|0.2997C|0.3266C|0.2725C|0.3518C|0.2127C CLIP L/14|0.1000
genre template text to image|0.6103R|0.7138C|0.5444C|0.6875C|0.6237R SigLIP 2|0.1000
genre template text to text|0.3318C|0.3174C|0.2013C|0.3298C|0.1852C SigLIP 2|0.1000

Leaders: c*, emotion template text to image/style text to text/genre template text to text; free variant, genre template text to image; the teacher, elsewhere. SigLIP 2 leads plain encoders on style and genre text to image, above c*. None leads overall.

## Deviations

CLIP B/32 joined SemArt. Image to image: omitted from pair table; metric parity only. `tx_checks.py`: after first read. Word queries: no expansion. Teacher: not run on MMArt, by scope. Lenses and CANVAS: not re-checked.

## Files and storage

[Results](../../../../src/test/20261105_text_retrieval/results/): `tables.md`, `summary.json`, `checks.json`. Bulk `results/{core,mmart,words,semart,leak}.json` under `/data/SSD2/cosir_r22/20261105_text_retrieval/` (folder 419 MB, mostly text vectors). Summary/storage: wrap-up. User: Lenses PDF/MMArt numbers.
