# Recipe pick and backbone tests: SigLIP 2 and Qwen pass; CLIP B/32 stays inconclusive

> 2026-10-25 (sequence name 12-24) · Spec: [design](../../../superpowers/specs/2026-10-25-recipe-and-backbones-design.md) · Decision rule: [binding](../../../../src/test/20261025_recipe_backbones/DECISION_RULE.md) @ a32b63d · Run log: [round log](../../../../src/test/20261025_recipe_backbones/20261025_recipe_backbones_run_log.md)

## Results brief

**What we tried.** We chose loss terms and weights (a recipe), then tested students on different image encoders.

**Terms.**

- D2 is the previous recipe (report 12-22). Own-caption terms align each image with its caption. Teacher share 1/5 gives the teacher one fifth of loss weight.
- AMI measures cluster agreement with labels. A seed range is D2's AMI spread over three seeds in 12-22, the development bar's unit.
- PercepT core is our reimplementation. B1 is its Qwen default-image arm. Own reading selects the top topic; r20 uses k-means with 20 clusters.
- Held paintings are for evaluation. Sample and full scale mean training on 13,400 and 42,969 paintings.
- Matched arms retrain PercepT's second stage on the student's encoder images, keeping Qwen topics and settings. Frozen inputs are encoder embeddings clustered without a student.
- Gain = the student's AMI minus the strongest comparator's AMI. GO passes; inconclusive has a positive gain without passing.

**Outcome against the bar.** Removing own-caption terms at teacher share 1/5 won on 29,232 development paintings, seed 0. Image endpoints only at 1/5 was close: mean gain 1.55 against the winner's 1.60 seed ranges.

| Emotion change vs D2 [95% interval] | Style change vs D2 [95% interval] | Genre change vs D2 [95% interval] |
|---|---|---|
| +0.0061 [+0.0046, +0.0076] | +0.0240 [+0.0216, +0.0261] | +0.0070 [+0.0039, +0.0102]; below one seed range (0.0128) |

On the 12,281 held paintings, B1's own reading was strongest for all three students. GO required gain ≥ 0.0100 and a positive lower bound.

| Full-scale student | AMI | B1 own AMI | Gain [95% interval] | Verdict |
|---|---:|---:|---|---|
| SigLIP 2 | 0.1024 | 0.0874 | +0.0150 [+0.0096, +0.0163] | GO at both scales |
| CLIP B/32 | 0.0885 | 0.0874 | +0.0011 [-0.0043, +0.0025] | Inconclusive at both scales |
| Qwen, picked recipe | 0.1029 | 0.0874 | +0.0154 [+0.0100, +0.0169] | GO at both scales |

No model holds the lead on ArtELingo-28's 746 paintings with an interval above zero. Both Qwen recipes and SigLIP 2 are qualified: positive gains with intervals reaching below zero. CLIP B/32 is below core. SigLIP 2's matched arm sets its bar there.

**What it does not show.** Selection among five candidates on one seed can inflate the winner's gain. Every claim needs Qwen3-VL-Embedding graphs, teacher and captions at training. The claims are k-means-only: full-scale Leiden emotion intervals include zero for SigLIP 2 and Qwen. Style, genre, retrieval and untested backbones are excluded. ArtELingo-28 is small with wide intervals; CLIP's inconclusive held result is not NO-GO.
The user took these flagged decisions after our recommendation:

| D8 | D9 | D11 | D13 |
|---|---|---|---|
| We test SigLIP 2 then CLIP B/32; DINOv2 runs only if image endpoints only wins. | Every student needs Qwen captions at training. | The strongest entry, matched arm included, sets the bar. | Full scale runs unless the sample result is NO-GO. |

**Options for next.**

1. Test teacher-faithful heads or losses on development data ([ideas 2 and 3][lossideas]). Our reading: the probe puts genre information in the default image. Runs take minutes on the 3090 and test whether losses preserve it.
2. Write this for the CVPR 2026-11-10 abstract. “The emotion lead over PercepT core holds on two image encoders, Qwen and SigLIP 2, with Qwen3-VL-Embedding at training; CLIP B/32 is inconclusive.” Writing time is unestimated; this could set the abstract's scope.

**Your reading.** We record your reading at wrap-up.

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** Our view: dropping own-caption terms at teacher share 1/5 gains on emotion and style on one development seed. Genre is +0.0070, positive but below one seed range (0.0128). Raising the share alone loses on all aspects, averaging -1.36 seed ranges. Removal at share 1/3 gives no qualifying gain, averaging 0.10. This answers 12-22's confound only in that setting.

The [framing note][framing] requires a SigLIP 2 or CLIP student above PercepT core. SigLIP 2 meets that condition at both scales. Our view supports this narrow claim. “The emotion lead over PercepT core holds on two image encoders, Qwen and SigLIP 2, with Qwen3-VL-Embedding at training; CLIP B/32 is inconclusive.” ArtELingo-28 gives only qualified support.

**Our recommendation.** Our view: write the narrow claim, then test genre losses on development data.
</details>

## What ran

All stages finished without failure.
The round ran 2026-10-25, 06:59–13:07 Amsterdam time, in the main checkout with RTX 3090 and CPU scoring.

Part 1 selected a recipe; part 2 tested held claims at both scales, then ArtELingo-28.
Every part 2 student uses the picked recipe; only the image network's first layer changes with the encoder's width (rule §2).
Context: D2 passed both held emotion claim tests (full scale 0.1008 against PercepT core's 0.0874, report 12-19) but stayed below the instructed Qwen image on style and genre (direction log, 2026-10-21). In 12-22, dropping the own-caption terms also raised the teacher's loss share from 1/5 to 1/3; part 1's six recipes separate the two.
Stopping used 1,489 sample and 6,152 full-scale paintings.
AMI averages three k-means states, then five seeds in part 2. Settings were not tuned.
Intervals use 2,000 paired painting bootstraps; part 2 resamples seeds and recomputes maxima.

## Results

The pick and two emotion claims pass.

### Part 1 pick

Part 1 uses 29,232 paintings and seed 0; genre labels cover 23,682 paintings.
The emotion, style and genre seed ranges are 0.0046, 0.0082 and 0.0128.
Passing requires one gain of at least one range with a positive lower bound, and no loss exceeding one range per aspect.
The largest mean gain wins, rounded to two decimals.

| ID / recipe | Measured image AMI change [95% interval] | D2 comparator | Bar | Outcome |
|---|---|---|---|---|
| R6-5: no own-caption terms, teacher share 1/5 | Emotion: +0.0061 [+0.0046, +0.0076] | D2 emotion 0.1010 | Emotion gain and all guards pass; mean 1.60 | P-R6-5: picked |

**Verdict under the rule:** no own-caption terms at share 1/5 wins.
Both image-endpoints-only recipes pass: teacher share 1/5 gives mean gain 1.55, and share 1/3 gives mean gain 1.04.
D2 at share 1/3 fails the style and genre guards; removal at share 1/3 has no qualifying gain.
The recipe change triggers a Qwen student; DINOv2 runs only if image endpoints only wins. The licensed image AMI claim compares the pick with D2 on fresh development paintings, one seed. It covers development only.

### Part 2 held emotion: stages 3 and 4

Each comparison uses 12,281 held paintings and five seeds. GO requires gain ≥ 0.0100 and lower bound > 0.
Smaller positive gains are inconclusive.

| ID / input | Stage / AMI | Gain [95% interval] | Comparator / AMI | Bar | Outcome |
|---|---|---|---|---|---|
| K2: SigLIP 2 | 3 / 0.1020 | +0.0165 [+0.0110, +0.0177] | B1-own / 0.0855 | GO bar | S1-GO |
| K3: CLIP B/32 | 3 / 0.0901 | +0.0046 [-0.0006, +0.0057] | B1-own / 0.0855 | GO bar | S1-IN |
| K4: Qwen, picked | 3 / 0.1045 | +0.0190 [+0.0137, +0.0201] | B1-own / 0.0855 | GO bar | S1-GO |
| K2 | 4 / 0.1024 | +0.0150 [+0.0096, +0.0163] | B1-own / 0.0874 | GO bar | S2-GO |
| K3 | 4 / 0.0885 | +0.0011 [-0.0043, +0.0025] | B1-own / 0.0874 | GO bar | S2-IN |
| K4 | 4 / 0.1029 | +0.0154 [+0.0100, +0.0169] | B1-own / 0.0874 | GO bar | S2-GO |

Matched arms retrain Stage 2 on the candidate's images, keeping Qwen topics, settings and corresponding seeds.
Matched arms use their own reading below:

| ID / image | Stage 3 own AMI | Stage 4 own AMI |
|---|---:|---:|
| M2: SigLIP 2 | 0.0777 | 0.0793 |
| M3: CLIP B/32 | 0.0699 | 0.0716 |
| B1: Qwen, Qwen student's matched arm | 0.0855 | 0.0874 |

**Verdict under the rule:** SigLIP 2 and Qwen pass both scales; CLIP B/32 is inconclusive at both.
The claim is licensed separately for **X = SigLIP 2** and **X = Qwen3-VL-Embedding**:
“On 12,281 held ArtELingo paintings, a model whose image side reads the default X image, trained on the 42,969 train
paintings with Qwen3-VL-Embedding's instructed graphs, instructed teacher and default captions, gives emotion
clusters (k-means, 20 clusters) with AMI at least 0.01 above the strongest PercepT core reimplementation trained on
the same paintings, one whose Stage 2 reads the same X image included (5 seeds; paired 95% interval over paintings
and seeds above 0).” Sample GO licenses the same with “trained on 13,400 paintings”.

These claims do not cover training without Qwen, style, genre, other datasets, PercepT with its own encoders, retrieval or untested backbones.

### Stage 5 ArtELingo-28

Each comparison uses 746 paintings, their own images, English-translated captions and full-scale weights.
“Holds” requires gain > 0 and lower bound > 0.

| ID / model | Emotion AMI | Gain [95% interval] | Comparator / AMI | Bar | Outcome |
|---|---:|---|---|---|---|
| K1: previous Qwen | 0.1344 | +0.0067 [-0.0200, +0.0116] | B1-own / 0.1277 | holds bar | A-QUAL |
| K2 | 0.1536 | +0.0244 [-0.0044, +0.0271] | M2-own / 0.1292 | holds bar | A-QUAL |
| K3 | 0.1198 | -0.0079 [-0.0332, +0.0001] | B1-own / 0.1277 | holds bar | A-FAILS |
| K4 | 0.1443 | +0.0167 [-0.0103, +0.0212] | B1-own / 0.1277 | holds bar | A-QUAL |

**Verdict under the rule:** both Qwen recipes and SigLIP 2 are qualified; CLIP B/32 fails with a negative gain.
No model earns “holds”. A-FAILS applies to ArtELingo-28; CLIP B/32's held test was inconclusive.

All regressions passed: 9/9 exports bit for bit at stage 0; 27/27 checks at both scales.
Independent derivations agreed for part 1 and stages 3–5, without exclusions or skips.

## Analysis

**Our reading:** the probe supports spec §3.6: default images hold genre information; losses limit its use.
The probe predicts each instructed Qwen image from the default one (ridge and a 4-layer MLP, fit on the 13,400 sample paintings, stopped on the regression loss of the 1,489 stopping paintings, no labels); it decides nothing. Ridge reaches genre AMI 0.5040 against instructed 0.4989, default 0.4623 and D2 0.4604 on fresh development data.

**Our reading:** the [four-view sheet][sheet] shows better neighbours alongside more concentrated vectors.

**Our reading:** [part 1 diagnostics][part1report] retain the winner's lead over D2 on all aspects.
Image endpoints only at share 1/5 matches or exceeds it on several diagnostics.
This holds on the old 5,000 paintings, captions, Leiden and epoch-4 exports.
Its best epochs are 3 for emotion, 2 for style and 1 for genre.
The epoch-4 export is a fixed snapshot after epoch 4. Image-endpoints-only recipes lack a caption network.

**Our reading:** [stage 3][stage3report] exceeds PercepT on style and genre; all students miss instructed genre.
Cells give student AMI minus comparator AMI; style uses 12,281 paintings, genre 9,949 labelled paintings.
The Qwen student's matched arm is B1 in both reported tables.

| ID / aspect | Strongest PercepT entry (B1–B6) | Candidate matched arm | Candidate frozen image | T: aspect-instructed Qwen image |
|---|---:|---:|---:|---|
| K2 / style | +0.2016 | +0.2038 | +0.1014 | +0.0273 |
| K3 / style | +0.1434 | +0.1562 | +0.0738 | -0.0309 |
| K4 / style | +0.1702 | +0.2002 | +0.1752 | -0.0041 [-0.0089, +0.0007] |
| K2 / genre | +0.0391 | +0.0108 | +0.1038 | -0.0260 |
| K3 / genre | +0.0146 | +0.0149 | +0.0472 | -0.0505 |
| K4 / genre | +0.0179 | +0.0179 | +0.0070 | -0.0471 |

### Stage 4 reported readings

**Our reading:** [full scale][stage4report] retains style gains and the instructed genre gap.
Arms stay fixed; the strongest entry can change.

| ID / aspect | PercepT maximum | Matched arm | Frozen encoder | T |
|---|---:|---:|---:|---:|
| K2 / style | +0.2006 | +0.2024 | +0.1019 | +0.0278 |
| K2 / genre | +0.0418 | +0.0129 | +0.0923 | -0.0375 |
| K3 / style | +0.1395 | +0.1532 | +0.0714 | -0.0333 |
| K3 / genre | +0.0215 | +0.0198 | +0.0400 | -0.0577 |
| K4 / style | +0.1699 | +0.2046 | +0.1763 | -0.0029 |
| K4 / genre | +0.0265 | +0.0304 | +0.0014 | -0.0527 |

| Emotion gain | SigLIP 2 | CLIP B/32 | Qwen |
|---|---:|---:|---:|
| Against frozen input | +0.0455 | +0.0243 | +0.0321 |
| Against instructed Qwen | +0.0103 | -0.0035 | +0.0108 |

**Our reading:** CLIP B/32's Leiden emotion margin flips from positive to -0.0043 [-0.0088, -0.0016].

SigLIP 2's Leiden genre margin flips from uncertain negative to +0.0123 [+0.0041, +0.0206].
Leiden style and genre intervals are positive; SigLIP 2 and Qwen emotion intervals include zero.

| Diagnostic | SigLIP 2 | CLIP B/32 | Qwen |
|---|---|---|---|
| Emotion AMI seed min–max | 0.1016–0.1032 | 0.0877–0.0906 | 0.1017–0.1041 |
| Best emotion epochs | 5–12 | 5–12 | 5–8 |
| Best style epochs | 2–4 | 1–3 | 2–4 |
| Best genre epochs | 1–2 | 1 | 1 |

Previous Qwen with the picked student's entries gives +0.0133 [+0.0081, +0.0148].
CLIP B/32 mean cosine to the legacy store is 0.999993.

### ArtELingo-28 reported context

**Our reading:** [transfer remains uncertain][stage5report] in both image groups.
Of the 746 paintings, 248 also belong to ArtELingo; their different image files have mean Qwen embedding cosine 0.934
([spec §8](../../../superpowers/specs/2026-10-25-recipe-and-backbones-design.md#8-risks)). The cause was not checked.
The other 498 are outside our store. Cells give emotion AMI and gain against each subset's strongest comparator, with 95% intervals.

| Model | 248 shared: AMI; gain [95% interval] | 498 outside: AMI; gain [95% interval] |
|---|---|---|
| K1 | 0.1024; -0.0126 [-0.0493, +0.0036] | 0.1349; +0.0195 [-0.0164, +0.0247] |
| K2 | 0.1130; -0.0066 [-0.0459, +0.0068] | 0.1556; +0.0359 [-0.0013, +0.0388] |
| K3 | 0.1088; -0.0062 [-0.0480, +0.0135] | 0.1139; -0.0014 [-0.0344, +0.0044] |
| K4 | 0.1116; -0.0034 [-0.0396, +0.0121] | 0.1471; +0.0317 [-0.0046, +0.0365] |

SigLIP 2 uses its matched arm; the others use Qwen core's own reading. All intervals include zero.
Language-specific labels differ from pooled labels; this split does not test culture as a condition.

## Deviations

Timing and defaults qualify the run; there were no formal rule amendments.

| ID / note | Qualification |
|---|---|
| D32, split | No candidate deferred; descriptive queues pushed initial estimates past 11:00. |
| V4 | No forced rerun; runners refuse retraining once held output exists; code is deterministic. No seed failed V3. |
| Stage 0 | Random unit-norm [29232, 2048] data stood in for an export timing; actual part 1 times cover budget. |
| Stage 5 gate | Agent default requires passed stage 3a beyond the spec's stage 4; it passed. |
| Stage 5 scale | “Stage 4 ran” means reached an outcome. All did; K1–K4 used full scale. |
| D31 / rule §4.2 | Qwen held outcomes test the development pick; they do not choose recipes. |

Stage 3's report queue ran 08:52 to 09:52 after the chain, taking 3,613 seconds.
Stage 4's was planned for 19:00 and ran 12:06 to 13:07 on the user's word, taking 3,617 seconds without failure.
Neither queue decides an outcome.
Part 1 had no exclusions or forced rerun.

Old development data chose D2 among 13 rows and supplied the caption-free variants. Encoder choice used no held numbers.
No choice was made on held data, and none is made here. The comparator's maximum is taken on held data (spec §8).

## Files and storage

The [round folder][round] holds code, rule and log. We keep outputs until the wrap-up's storage summary.
The gitignored [outputs][outputs] exist only in the main checkout, including verdicts, diagnostics, queues and `part2/reported_stage4.{json,md}`.

| System disk storage, checked with du -sh | Size |
|---|---:|
| outputs/ | 23 GB (23G) |
| Part 1 train | 12 GB |
| Part 2 full | 5.8 GB |
| Part 2 sample | 5.3 GB |
| Part 2 regression | about 0.8 GB (760M) |

Build scratch trees are listed in [the cleanup list](../../../../.scratch/pending_deletions.md). Nothing was removed.

[round]: ../../../../src/test/20261025_recipe_backbones/
[outputs]: ../../../../src/test/20261025_recipe_backbones/outputs/
[sheet]: ../../../../src/test/20261025_recipe_backbones/outputs/part1/sheet.md
[part1report]: ../../../../src/test/20261025_recipe_backbones/outputs/part1/reported.md
[stage3report]: ../../../../src/test/20261025_recipe_backbones/outputs/part2/reported_stage3.md
[stage5report]: ../../../../src/test/20261025_recipe_backbones/outputs/part2/reported_stage5.json

[lossideas]: ../../literature/2026-12-20_ways_to_improve.md#candidate-ideas-one-ranking
[framing]: 2026-12-21_advantages_framing.md
[stage4report]: ../../../../src/test/20261025_recipe_backbones/outputs/part2/reported_stage4.md
