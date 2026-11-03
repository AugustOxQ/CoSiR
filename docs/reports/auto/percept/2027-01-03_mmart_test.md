# MMArt test: holds, GO on all four encoders

> 2026-11-03 (sequence name 2027-01-03) · Spec and decision rule: [§6 with §3.4–§3.5](../../../superpowers/specs/2026-11-02-mmart-test-design.md#6-success-criteria) @ e96a7cd (approved 2026-11-02 18:37); amendment 471859e (D19: cosine 0.95, user) · Run log: [round log](../../../../src/test/20261102_mmart_test/20261102_mmart_test_log.md) · [r14](2027-01-02_backbones_sweep.md)

## Results brief

**What we tried.** We tested c*, our network trained on ArtELingo paintings, frozen, on 16,384 MMArt paintings
outside ArtELingo, scored with MMArt's own labels ([spec](../../../superpowers/specs/2026-11-02-mmart-test-design.md)).
Four image encoders, five seeds each; read-outs were picked on validation paintings only.

**Outcome against the bar.** **Holds, GO for each of the four encoders.** c* wins all nine cells per encoder against
the plain image (the encoder's own embedding) and against PercepT on the same encoder. A WIN means the 95% interval of
the difference is above 0. Example: Qwen emotion clustering, AMI 0.0609 for c* against 0.0407 for the plain image.
Against the teacher (the Qwen image given an instruction per aspect), all 15 cells set in advance for a "better than"
claim WIN. LEVEL means within the seed range r below 0; BELOW means further below.

|Teacher cells|WIN|LEVEL|BELOW|
|---|---:|---:|---:|
|MMArt, this round|18|4|2|
|ArtELingo, previous round|18|5|1|

**What it does not show.** Emotion retrieval is below the teacher for Qwen and CLIP L/14 (Qwen −0.0010
[−0.0015, −0.0004]): no teacher claim there. Test and training paintings share WikiArt and most artists; only 15 test
paintings have an artist without ArtELingo train paintings, too few to score. The three encoders besides Qwen
had passed an earlier held test on ArtELingo.

**Options for next.**

1. Freeze these results and write: no compute. Gives the paper a second painting set on four encoders; frees the
   week before the abstract (2026-11-10) and the paper (2026-11-16).
2. A second aspect on photos (style or mood, for example Flickr Style): about one loop, a new data path and a
   data-access check. Tells whether the photo result goes beyond emotion. Can wait until after the abstract.
3. A painting test with artists outside ArtELingo: a data search first (not done yet). Tells whether the gains
   hold for new artists, a claim this round cannot make.

**Your reading.**

<details>
<summary>Our reading and recommendation (open after yours)</summary>

**Our reading:** the gains over the plain image and over PercepT carry to new paintings and new labels on all four
encoders. Against the teacher the pattern of ArtELingo repeats: wins in clustering, level or slightly below in emotion
retrieval. We are sure of the verdict (the independent re-derivation matched every number); less sure about new
artists.

**Our recommendation:** option 1, freeze and write; the abstract is a week away.

</details>

## What ran

All gates and stage 1 records are ok; test selected nothing.

|Stage|Type|Result|
|---|---|---|
|0|Regression check|All ok|
|1|Development sweep|Picks/ranges committed before test: 8dd3857|
|2|Claim test|60 cells; C13 before final copies|
|3|Descriptive|Ten diagnostics; no decisions|

- **Data and seeds:** 5,769 ArtELingo validation paintings, MMArt labels; seeds 0 to 4. AMI: 20 clusters, three states, mask labels after clustering. Retrieval: labelled queries/gallery.
- **Terms:** PercepT's matched arm: PercepT on the same encoder, its saved topics and mappers applied frozen; per draw the larger of its own-topic and 20-cluster readings. Read-out: how embeddings are compared (centering or not, ten-neighbour query expansion).
- **Code:** bc27203 (build); C13 derive f093fb7, compare fixes c00fda2 and f13117d; diagnostics display fix a3b7e29. Local CPU, three parallel steps, eight threads; no training. [Build/run handoff](../../../superpowers/handoffs/2026-11-03-r15-run-handoff.md).

|Cut set|Cosine|Paintings|
|---|---|---|
|Test|0.95|16,384|
|Far diagnostic|0.85|10,403|

[Cutsets](../../../../src/test/20261102_mmart_test/records/cutsets.json); reference: ArtELingo train/validation.

|Gate (all ok in status.json)|Measured, round log|
|---|---|
|(i) Stores/cut|Stores/near-copy pass; test 16,384, far 10,403 match spec|
|(ii) Export|Maximum 6.1e-5, bar 1e-3|
|(iii) Scoring|Maximum difference 0, bar 1e-9; full draws|
|(iv) Mappers|20 pass; maximum 8.3e-7, bar 1e-5; stored readings replay|
|(v) Teacher views|All pass; 5,769 keys each (stage0/gate_v.json)|
|(vi) Read-outs|Maximum difference 0, bar 1e-9|

All 20 PercepT mappers passed, so the fallback (retrain a mapper) was not used. The stop for a failed teacher-view
gate was not triggered, and no encoder was voided.

Seed ranges r (validation, maximum minus minimum over five seeds), AMI / plain-cosine mAP, from the stage 1 records. Read-outs: center = centering, none = no map, k10 = ten-neighbour expansion. Teacher: center+k10, Qwen ranges.

|Encoder|Emotion r|Style r|Genre r|c* / plain read-out|
|---|---|---|---|---|
|Qwen|0.0048 / 0.0015|0.0118 / 0.0026|0.0097 / 0.0037|center+k10 / none+k10|
|SigLIP 2|0.0044 / 0.0011|0.0071 / 0.0029|0.0073 / 0.0044|center+k10 / center+k10|
|SigLIP 1|0.0038 / 0.0006|0.0190 / 0.0051|0.0100 / 0.0042|center+k10 / none+k10|
|CLIP L/14|0.0030 / 0.0006|0.0135 / 0.0038|0.0086 / 0.0019|center+k10 / center+k10|

## Results

**Verdict under the rule: holds, GO for all four encoders.** Qwen = Qwen3-VL-Embedding-2B.

Re-derivation (C13): **6,697 items compared, 0 failed, maximum absolute difference 0, scope full**, at 2,000 clustering and 10,000 retrieval draws (`c13/compare.json`, `derive_full.log`).

C13's record was committed at 03:51 (f13117d), before the verdicts were copied (5c323d8), so the verdicts are final.

Cells: `stage2/{cells_clu,cells_ret,outcomes}.json`; verdicts agree. Entries: **c*/comparator; d [95%]**, all **WIN**. Columns: plain-image AMI (SC1), plain-image mAP (SC2), matched-PercepT AMI (SC3).

|Encoder, aspect|SC1|SC2|SC3|
|---|---|---|---|
|Qwen, emotion|0.0609/0.0407; +0.0201 [+0.0178, +0.0227]|0.2011/0.1935; +0.0075 [+0.0070, +0.0082]|0.0609/0.0514; +0.0094 [+0.0055, +0.0101]|
|Qwen, style|0.4227/0.2314; +0.1913 [+0.1836, +0.1972]|0.3472/0.2046; +0.1426 [+0.1399, +0.1453]|0.4227/0.1959; +0.2268 [+0.2161, +0.2314]|
|Qwen, genre|0.5131/0.4498; +0.0633 [+0.0579, +0.0691]|0.6199/0.5151; +0.1048 [+0.1025, +0.1072]|0.5131/0.4257; +0.0873 [+0.0786, +0.0959]|
|SigLIP 2, emotion|0.0608/0.0332; +0.0276 [+0.0252, +0.0301]|0.2016/0.1897; +0.0120 [+0.0114, +0.0125]|0.0608/0.0451; +0.0158 [+0.0118, +0.0165]|
|SigLIP 2, style|0.4504/0.3243; +0.1261 [+0.1208, +0.1311]|0.3744/0.2779; +0.0965 [+0.0945, +0.0985]|0.4504/0.2298; +0.2206 [+0.2111, +0.2247]|
|SigLIP 2, genre|0.5181/0.3993; +0.1188 [+0.1114, +0.1271]|0.6291/0.5097; +0.1194 [+0.1170, +0.1218]|0.5181/0.4611; +0.0569 [+0.0486, +0.0655]|
|SigLIP 1, emotion|0.0620/0.0370; +0.0249 [+0.0226, +0.0274]|0.2021/0.1913; +0.0108 [+0.0103, +0.0113]|0.0620/0.0468; +0.0152 [+0.0114, +0.0156]|
|SigLIP 1, style|0.4546/0.3915; +0.0631 [+0.0580, +0.0677]|0.3831/0.3031; +0.0799 [+0.0777, +0.0823]|0.4546/0.2297; +0.2249 [+0.2149, +0.2290]|
|SigLIP 1, genre|0.5213/0.4325; +0.0888 [+0.0815, +0.0957]|0.6282/0.4932; +0.1350 [+0.1326, +0.1374]|0.5213/0.4594; +0.0620 [+0.0523, +0.0705]|
|CLIP L/14, emotion|0.0590/0.0386; +0.0204 [+0.0182, +0.0227]|0.2010/0.1912; +0.0098 [+0.0093, +0.0103]|0.0590/0.0454; +0.0136 [+0.0098, +0.0140]|
|CLIP L/14, style|0.4345/0.3558; +0.0787 [+0.0731, +0.0849]|0.3637/0.2967; +0.0670 [+0.0647, +0.0689]|0.4345/0.2329; +0.2016 [+0.1918, +0.2069]|
|CLIP L/14, genre|0.5115/0.4322; +0.0793 [+0.0715, +0.0864]|0.6245/0.5110; +0.1134 [+0.1113, +0.1156]|0.5115/0.4506; +0.0609 [+0.0517, +0.0694]|

All cells: 16,384 paintings; labelled emotion/style/genre 16,218/16,384/13,420. Joint row/seed bootstrap; no multiplicity correction.

Teacher cells (SC4): the D7 column marks the 15 cells where a WIN licenses "better than", set before the test from
the ArtELingo results; all 15 WIN. WIN: lo > 0; LEVEL: lo > −r (within the validation seed range); BELOW otherwise.
Outcomes use unrounded values.

Claim sentences: **"c* on [encoder] is [claim] the Qwen image instructed for [aspect] in [task] on MMArt."** AMI = clustering; mAP = retrieval. In the table, B = "better than", L = "level with or better than"; None = no sentence.

|Encoder, aspect|Task|d [95%]|r|Outcome|D7|Claim|
|---|---|---|---|---|---|---|
|Qwen, emotion|AMI|+0.0043 [+0.0023, +0.0063]|0.0048|WIN|yes|B|
|Qwen, emotion|mAP|-0.0010 [-0.0015, -0.0004]|0.0015|BELOW|no|None|
|Qwen, style|AMI|+0.0069 [+0.0011, +0.0127]|0.0118|WIN|yes|B|
|Qwen, style|mAP|-0.0004 [-0.0023, +0.0016]|0.0026|LEVEL|no|L|
|Qwen, genre|AMI|+0.0060 [+0.0022, +0.0102]|0.0097|WIN|no|L|
|Qwen, genre|mAP|+0.0027 [+0.0015, +0.0041]|0.0037|WIN|yes|B|
|SigLIP 2, emotion|AMI|+0.0043 [+0.0021, +0.0065]|0.0044|WIN|yes|B|
|SigLIP 2, emotion|mAP|-0.0005 [-0.0010, +0.0001]|0.0011|LEVEL|no|L|
|SigLIP 2, style|AMI|+0.0346 [+0.0297, +0.0392]|0.0071|WIN|yes|B|
|SigLIP 2, style|mAP|+0.0269 [+0.0253, +0.0284]|0.0029|WIN|yes|B|
|SigLIP 2, genre|AMI|+0.0110 [+0.0044, +0.0184]|0.0073|WIN|no|L|
|SigLIP 2, genre|mAP|+0.0119 [+0.0103, +0.0135]|0.0044|WIN|yes|B|
|SigLIP 1, emotion|AMI|+0.0054 [+0.0033, +0.0076]|0.0038|WIN|yes|B|
|SigLIP 1, emotion|mAP|-0.0000 [-0.0005, +0.0004]|0.0006|LEVEL|no|L|
|SigLIP 1, style|AMI|+0.0388 [+0.0338, +0.0433]|0.0190|WIN|yes|B|
|SigLIP 1, style|mAP|+0.0355 [+0.0336, +0.0374]|0.0051|WIN|yes|B|
|SigLIP 1, genre|AMI|+0.0143 [+0.0072, +0.0202]|0.0100|WIN|no|L|
|SigLIP 1, genre|mAP|+0.0110 [+0.0095, +0.0126]|0.0042|WIN|yes|B|
|CLIP L/14, emotion|AMI|+0.0024 [+0.0004, +0.0045]|0.0030|WIN|yes|B|
|CLIP L/14, emotion|mAP|-0.0011 [-0.0015, -0.0006]|0.0006|BELOW|no|None|
|CLIP L/14, style|AMI|+0.0187 [+0.0134, +0.0248]|0.0135|WIN|yes|B|
|CLIP L/14, style|mAP|+0.0161 [+0.0141, +0.0179]|0.0038|WIN|yes|B|
|CLIP L/14, genre|AMI|+0.0045 [-0.0034, +0.0108]|0.0086|LEVEL|no|L|
|CLIP L/14, genre|mAP|+0.0072 [+0.0058, +0.0087]|0.0019|WIN|yes|B|

Some teacher WINs are narrow: the lower bound is +0.0004 for CLIP L/14 emotion AMI and +0.0011 for Qwen style AMI.
Emotion AMI is low for every arm (c* about 0.06), so emotion differences are small in absolute terms.

<details>
<summary>Licensed sentences, records/verdicts.json</summary>

The four sentences differ only in the encoder, X = Qwen3-VL-Embedding-2B, SigLIP 2, SigLIP 1 or CLIP L/14:

> On 16,384 WikiArt paintings outside ArtELingo (MMArt; exact and near copies of ArtELingo train and validation paintings removed), c*'s recipe with the X image as its only input at test (frozen; never trained or picked on these paintings) clusters and retrieves paintings by emotion (MMArt's labels), style and genre better than the plain X image, and clusters them better than PercepT on the X image (AMI with 20 clusters; mAP with read-outs picked on validation paintings with MMArt labels; 5 seeds).

> X is one of four encoders; the three besides Qwen passed an earlier held emotion test on ArtELingo.

</details>

Not licensed (§6): unseen artists, other art sources/photos, perspectives, or "better than" outside pre-set bars. LEVEL permits only "level with or better than"; BELOW nothing.

## Analysis

All ten items are **ok**, deciding nothing: current `OUT/diag/<item>.{json,md}`.

Shared held: same 11,528 paintings, fixed picks/draws; MMArt beside ArtELingo labels. Qwen differences:

|Comparison|MMArt labels, d [95%]|ArtELingo labels, d [95%]|
|---|---|---|
|Emotion AMI, plain|+0.0192 [+0.0166, +0.0220]|+0.0364 [+0.0332, +0.0398]|
|Style AMI, plain|+0.1878 [+0.1810, +0.1927]|+0.1879 [+0.1809, +0.1923]|

Our reading: on the same images, the emotion gain with MMArt labels is about half the gain with ArtELingo labels;
the style gain does not change. The emotion gain on the test set (+0.0201) is close to the shared-held one with MMArt
labels, so the smaller emotion gain comes from the new labels, not the new images. Labelled populations differ;
genre labels match.

Far subset (the cut at 0.85): every plain-image cell and every teacher cell set for "better than" has a positive lower
bound. Qwen against the teacher:

|Far comparison|d [95%]|
|---|---|
|Emotion AMI|+0.0053 [+0.0026, +0.0079]|
|Emotion mAP|-0.0007 [-0.0013, -0.0001]|
|Style mAP|-0.0039 [-0.0058, -0.0020]|

Our reading: the gains over the plain image hold farther from the training images. On the far subset only, Qwen's
emotion and style retrieval are below the teacher; on the test set the style cell is LEVEL. The far subset decides
nothing, so no claim follows from either.

Teacher-gain share: (c* − plain)/(teacher − plain); AMI / mAP, percent.

|Encoder|Emotion|Style|Genre|
|---|---|---|---|
|Qwen|127.1% / 88.2%|103.7% / 99.8%|110.5% / 102.7%|
|SigLIP 2|118.2% / 96.4%|137.8% / 138.6%|110.2% / 111.1%|
|SigLIP 1|127.7% / 99.8%|259.2% / 179.9%|119.2% / 108.9%|
|CLIP L/14|113.6% / 90.3%|131.2% / 131.7%|106.0% / 106.8%|

Our reading: c* keeps more than the teacher's whole gain in every clustering cell; in emotion retrieval it keeps
88.2% to 99.8%.

Test seed spread against the validation range r (`seed_spreads`; ratio = test spread / r, over 12 cells each):

|Metric|Ratio, smallest to largest|Median|Cells above 1|
|---|---|---|---|
|AMI|0.33 (SigLIP 1 emotion) to 2.41 (SigLIP 2 genre)|1.07|6|
|Plain-cosine mAP|0.80 (CLIP L/14 emotion) to 1.09 (CLIP L/14 style)|0.91|2|

Our reading: r is a close guide to the test spread in retrieval and a loose one in clustering.

Strongest PercepT entry per aspect:

|Aspect|Entry|Reading|AMI|
|---|---|---|---|
|Emotion|Qwen P3|Own topics|0.0514|
|Style|CLIP L/14 M7|Own topics|0.2329|
|Genre|SigLIP 2 M2|20 clusters|0.4611|

|Remaining item|Result and limit|
|---|---|
|Plain cosine|Qwen emotion mAP c*/plain/teacher: 0.1998/0.1935/0.2026.|
|Class-count clustering|Nine-cluster emotion AMI, Qwen c*/plain/teacher: 0.0532/0.0381/0.0492.|
|Per-class AP|Qwen contentment 0.2413 versus teacher 0.2547; amusement 0.0828 versus 0.0710.|
|Unseen artists|15 paintings, 12 artists; no cells because fewer paintings than 20 clusters (finding 11; agent default).|
|Near copies|390 at cosine ≥0.95 to reference; within test: 48 pairs, 92 paintings. Histogram/class removals saved.|

## Deviations

No known rule departure; handoff/round-log notes:

|Item|Protocol or fix|
|---|---|
|Finding 8|Plain-image/teacher retrieval: float32 `unit(unit(raw))`; r11 normalises once. Once-normalised teacher `whiten1024+k5` stage 1 score moves up to 3.7e-4; picks unchanged. C13 matches round.|
|Threads, 02:02|Eight-thread round versus one thread: SigLIP 2, seed 2, state 1, emotion changes 212/16,384 assignments; c* AMI moves 1.3e-5. C13 uses eight (agent default; spec fixes none).|
|C13 source, 03:48|Compare-only source check fix; derive ran exactly f093fb7.|
|Display, 04:45|Two Markdown tables re-rendered from saved JSON after lookup fix. Cells/verdicts unaffected; current files used, never pre_fix_0450.|

The 1.3e-5 thread effect is below the smallest WIN lo: CLIP L/14 emotion AMI versus teacher, +0.0004.

Agent defaults of the spec and build: the claim-test type, the teacher bars and views, the sha256 and md5 orders,
picks scored in each encoder's own ranges, label masks after clustering, the gate scopes and the mapper tolerance.
In the run: C13 at eight k-means threads; the unseen-artist subset without cells.

|Work|Actual (Amsterdam; queue/log)|Handoff estimate|
|---|---|---|
|Stage 0|01:13–01:38|30 min|
|Stage 1|01:38–02:05; stages 0+1: 3,120 s|30 min|
|Stage 2|2,189 s|1–1.5 h|
|C13 derive|5,177.9 s; overlaps stage 2|1.5 h|
|Stage 3|3,053 s|1–1.5 h|

**Type of the test:** the direction log calls this round a confirmation; the spec runs stage 2 as a claim test, the
stricter type, since it feeds paper claims (agent default, flagged in the spec's brief).

## Files and storage

Nothing deleted; sizes from the round log, 04:46.

|Folder under `/data/SSD2/cosir_r15/20261102_mmart_test/`|Size|
|---|---|
|`outputs/` (OUT)|12 GB; exports 9.0 GB, stage 0 2.9 GB; rest <1 MB|
|`c13/`|19 MB, cache 16 MB|
|`c13_smoke/`|16 MB|
|`smoke_build3/`|12 GB|

Files: cells/outcomes, diagnostics, queue logs, C13 compare/derive log, [records](../../../../src/test/20261102_mmart_test/records/). Storage (decided 2026-11-03, r16 decide; disk ledger): `outputs/exports/` (9.0 GB) kept until the paper's numbers are frozen; the build smoke, stage 0 gate files and C13 smoke and caches deleted.
