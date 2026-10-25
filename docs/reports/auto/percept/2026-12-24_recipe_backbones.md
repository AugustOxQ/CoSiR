# Recipe pick and backbone tests: SigLIP 2 and Qwen pass; CLIP B/32 stays inconclusive

> 2026-10-25 (sequence name 12-24) · Spec: [design](../../../superpowers/specs/2026-10-25-recipe-and-backbones-design.md) · Decision rule: [binding](../../../../src/test/20261025_recipe_backbones/DECISION_RULE.md) @ a32b63d · Run log: [round log](../../../../src/test/20261025_recipe_backbones/20261025_recipe_backbones_run_log.md)

## Results brief

**What we tried.** We chose a recipe, the loss terms and weights, then tested students on different image encoders.
Adjusted mutual information (AMI) measures cluster agreement with labels.

**Outcome against the bar.** The recipe without own-caption terms won on 29,232 development paintings with seed 0.
Own-caption terms align each painting's image with its own caption.
The winner kept teacher share 1/5, meaning the teacher supplies one fifth of the loss weight.
Its gains against the previous model, D2, were:

| Emotion gain [95% interval] | Style gain [95% interval] | Genre gain [95% interval] |
|---|---|---|
| +0.0061 [+0.0046, +0.0076] | +0.0240 [+0.0216, +0.0261] | +0.0070 [+0.0039, +0.0102] |

A seed range is D2's spread over three seeds in 12-22, the unit of the development bar.
The image-endpoints-only recipe at share 1/5 was close, with mean gain 1.55 against the winner's 1.60 seed ranges.

Held emotion required a gain of 0.0100 with a positive lower interval bound.
A matched arm retrains PercepT's second stage on the student's images, keeping Qwen topics and core settings.
Qwen core's own reading on its default image set every bar; both new matched arms were weaker.
The Qwen student's matched arm was the bar itself.

| Full-scale student | AMI | Strongest core entry (B1, own reading) | Gain [95% interval] | Verdict |
|---|---:|---:|---|---|
| SigLIP 2 | 0.1024 | 0.0874 | +0.0150 [+0.0096, +0.0163] | GO at both scales |
| CLIP B/32 | 0.0885 | 0.0874 | +0.0011 [-0.0043, +0.0025] | Inconclusive at both scales |
| Qwen, picked recipe | 0.1029 | 0.0874 | +0.0154 [+0.0100, +0.0169] | GO at both scales |

On ArtELingo-28's 746 paintings, no model holds the lead with an interval above zero.
The previous Qwen recipe, the picked Qwen recipe and SigLIP 2 are qualified.
Their gains are positive, but their intervals reach below zero.
CLIP B/32 is below PercepT core.

**What it does not show.** Selection among five candidates on one seed can inflate the winner's gain.
Every claim needs Qwen3-VL-Embedding at training time for graphs, teacher and captions.
The held claims exclude style, genre, retrieval and untested backbones.
ArtELingo-28 is small, with 746 paintings and wide intervals.
Inconclusive for CLIP B/32 is not a NO-GO.
The user took these flagged decisions after our recommendation:

| D8 | D9 | D11 | D13 |
|---|---|---|---|
| Encoder order; conditional DINOv2 | Qwen captions | Strongest comparator, matched arm included | Full scale unless NO-GO |

**Options for next.**

1. Test teacher-faithful heads or losses on development data ([ideas 2 and 3][lossideas]). The probe finds genre information in the default image. Runs take minutes on the 3090 and could test whether losses preserve it.
2. Write “any student backbone, Qwen teacher at training” into the CVPR framing for the 2026-11-10 abstract ([framing note][framing]). SigLIP 2 meets its condition at both scales; report CLIP B/32 as inconclusive. Writing and review time is not estimated; this could define the paper's scope.
3. Investigate CLIP B/32's weaker gain despite higher frozen emotion AMI: 0.0642 against SigLIP 2's 0.0569 at stage 3. Cost is not estimated; this could test how the loss transfers between encoders.

**Your reading.** We record your reading at wrap-up.

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** Our view: removal at teacher share 1/5 gains on all aspects and resolves 12-22's confound within these recipes.
Raising the share alone loses on all three, averaging -1.36 seed ranges.
Removal at share 1/3 gives no qualifying gain, averaging 0.10.
The gain comes from removing the terms, rather than the teacher's larger share.
SigLIP 2 meets the framing note's “any backbone” condition at both scales, with Qwen supervision at training.
CLIP B/32 leaves it unresolved; ArtELingo-28 gives only qualified support.

**Our recommendation.** Our view: write the claim with the Qwen caveat, then test genre losses on development data.

</details>

## What ran

All candidates finished; stage 4's descriptive readings remain pending.
The round ran on 2026-10-25 from 06:59 to 09:52 Amsterdam time, using the main checkout, local RTX 3090 and CPU scoring.

Part 1 selected a recipe; part 2 tested held claims at both scales, then confirmed on ArtELingo-28.
Both included exploration.

The sample used 13,400 training paintings and 1,489 stopping paintings; full scale used 42,969 and 6,152.
AMI averages three k-means states, then five model seeds in part 2; settings were not tuned per encoder.
Intervals use 2,000 paired painting bootstraps; part 2 also resamples seeds and recomputes comparator maxima.

## Results

The recipe pick and two emotion claims pass.

### Part 1 pick

The development comparison uses 29,232 paintings and seed 0; genre has labels on 23,682 paintings.
The emotion, style and genre seed ranges are 0.0046, 0.0082 and 0.0128.
A recipe must gain at least one range with a positive lower bound and lose at most one range on each aspect.
The largest mean gain wins, rounded to two decimals.

| ID / recipe | Measured image AMI change [95% interval] | D2 comparator | Bar | Outcome |
|---|---|---|---|---|
| R6-5: no own-caption terms, teacher share 1/5 | Emotion: +0.0061 [+0.0046, +0.0076]; style: +0.0240 [+0.0216, +0.0261]; genre: +0.0070 [+0.0039, +0.0102] | Emotion 0.1010; style 0.3748; genre 0.4604 | Gain and guard pass; mean 1.60 | P-R6-5: picked |

**Verdict under the rule:** no own-caption terms at share 1/5 wins.
Both image-endpoints-only recipes pass: teacher share 1/5 gives mean gain 1.55, and share 1/3 gives mean gain 1.04.
D2 at share 1/3 fails the genre guard; removal at share 1/3 has no qualifying gain.
Qwen runs; DINOv2 does not. The licensed claim covers these image AMI changes against D2 on fresh development paintings, with one seed.
It covers no held result.

### Part 2 held emotion: stages 3 and 4

Each comparison uses 12,281 held paintings and five seeds. GO requires gain ≥ 0.0100 and lower bound > 0.
Smaller positive gains are inconclusive.

| ID / input | Stage / AMI | Gain [95% interval] | Comparator / AMI | Bar | Outcome |
|---|---|---|---|---|---|
| K2: SigLIP 2 | 3 / 0.1020 | +0.0165 [+0.0110, +0.0177] | B1-own: core own / 0.0855 | GO bar | S1-GO |
| K3: CLIP B/32 | 3 / 0.0901 | +0.0046 [-0.0006, +0.0057] | B1-own / 0.0855 | GO bar | S1-IN |
| K4: Qwen, picked | 3 / 0.1045 | +0.0190 [+0.0137, +0.0201] | B1-own / 0.0855 | GO bar | S1-GO |
| K2 | 4 / 0.1024 | +0.0150 [+0.0096, +0.0163] | B1-own / 0.0874 | GO bar | S2-GO |
| K3 | 4 / 0.0885 | +0.0011 [-0.0043, +0.0025] | B1-own / 0.0874 | GO bar | S2-IN |
| K4 | 4 / 0.1029 | +0.0154 [+0.0100, +0.0169] | B1-own / 0.0874 | GO bar | S2-GO |

Matched arms retain saved Qwen Stage 1 topics, core settings and corresponding seeds, while retraining Stage 2 on the candidate's encoder images.
The table gives matched arms' emotion AMI, using their own reading:

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
No model licenses the ArtELingo-28 “holds” claim. Confirmation cannot rescue a failed backbone test.

All gates passed: stage 0 matched 9/9 exports bit for bit; sample and full regressions passed 27/27 checks each.
Independent derivations agreed for part 1 and stages 3–5, without exclusions or skips.

## Analysis

**Our reading:** the probe supports spec §3.6: default images hold genre information; losses limit its use.
With label-free stopping, ridge reaches genre AMI 0.5040 against instructed 0.4989, default 0.4623 and D2 0.4604 on fresh development data.

**Our reading:** the [four-view sheet][sheet] shows better neighbours but more concentrated instructed vectors.
Negative emotion and style silhouettes suggest a clustering limit.

**Our reading:** [part 1 diagnostics][part1report] retain the winner's lead on all aspects.
This holds on the old 5,000 paintings, fresh captions, Leiden clusters and epoch-4 exports.
The winner stops at emotion epoch 3, style epoch 2 and genre epoch 1.
Early exports run patience epochs; image-endpoints-only recipes lack a caption network.

**Our reading:** [stage 3][stage3report] exceeds PercepT on style and genre, but all students miss instructed genre.
Each cell is student AMI minus comparator AMI. Style uses 12,281 paintings; genre uses 9,949 labelled paintings.

| ID / aspect | Strongest PercepT entry (B1–B6) | Candidate matched arm | Candidate frozen image | T: aspect-instructed Qwen image |
|---|---:|---:|---:|---|
| K2 / style | +0.2016 | +0.2038 | +0.1014 | +0.0273 |
| K3 / style | +0.1434 | +0.1562 | +0.0738 | -0.0309 |
| K4 / style | +0.1702 | +0.2002 | +0.1752 | -0.0041 [-0.0089, +0.0007] |
| K2 / genre | +0.0391 | +0.0108 | +0.1038 | -0.0260 |
| K3 / genre | +0.0146 | +0.0149 | +0.0472 | -0.0505 |
| K4 / genre | +0.0179 | +0.0179 | +0.0070 | -0.0471 |

All students exceed their frozen image on emotion. SigLIP 2 and Qwen also exceed the instructed image; CLIP B/32 remains unresolved.
**Our reading:** Leiden weakens emotion leads; SigLIP 2's genre interval crosses zero, while CLIP B/32 and Qwen remain positive.
Claims remain specific to k-means.

[Diagnostics][stage3report] show narrow emotion seed spans and early best epochs.
Previous Qwen stops later on style and genre; with the picked student's comparators, its margin is +0.0146 [+0.0093, +0.0157].
CLIP B/32 store parity covers 61,402 paintings, with cosine mean 0.999993, minimum 0.997838 and none below 0.99.
**Our reading:** parity does not explain CLIP's result. SigLIP 2 extraction parity was not checked.

### Stage 4 reported readings

Pending: the stage 4 report queue runs 2026-10-25 from 19:00; this section is filled from part2/reported_stage4.{json,md}.

### ArtELingo-28 reported context

**Our reading:** [transfer remains uncertain][stage5report] in both image groups.
Of the 746 paintings, 248 also belong to ArtELingo; their different image files have mean Qwen embedding cosine 0.934
([spec §8](../../../superpowers/specs/2026-10-25-recipe-and-backbones-design.md#8-risks)). The cause was not checked.
The other 498 are outside our ArtELingo store. Cells give emotion AMI and gain against the subset's strongest comparator, with 95% intervals.

| Model | 248 shared: AMI; gain [95% interval] | 498 outside: AMI; gain [95% interval] |
|---|---|---|
| K1 | 0.1024; -0.0126 [-0.0493, +0.0036] | 0.1349; +0.0195 [-0.0164, +0.0247] |
| K2 | 0.1130; -0.0066 [-0.0459, +0.0068] | 0.1556; +0.0359 [-0.0013, +0.0388] |
| K3 | 0.1088; -0.0062 [-0.0480, +0.0135] | 0.1139; -0.0014 [-0.0344, +0.0044] |
| K4 | 0.1116; -0.0034 [-0.0396, +0.0121] | 0.1471; +0.0317 [-0.0046, +0.0365] |

SigLIP 2 uses its matched arm; other models use Qwen core's own reading. All subset intervals include zero.
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

Both descriptive queues moved to evening. After the chain, stage 3 fitted the morning: 08:52–09:52, taking 3,613 seconds.
Stage 4 stays at 19:00 because morning left no margin before 11:00. Neither decides an outcome. Part 1 had no exclusions or forced rerun.

Old development data chose D2 among 13 rows and supplied the caption-free variants. Encoder choice used no held numbers.

## Files and storage

We keep the outputs until the wrap-up's storage summary. The [round folder][round] holds code, rule and log.
The gitignored [outputs][outputs] exist only in the main checkout and hold the pick, verdicts, derivations, diagnostics and queue statuses.

| System disk storage, checked with du -sh | Size |
|---|---:|
| outputs/ | 23 GB (23G) |
| Part 1 train | 12 GB |
| Part 2 full | 5.8 GB |
| Part 2 sample | 5.3 GB |
| Part 2 regression | about 0.8 GB (760M) |

The build scratch trees are listed in [.scratch/pending_deletions.md](../../../../.scratch/pending_deletions.md). Nothing was removed.

[round]: ../../../../src/test/20261025_recipe_backbones/
[outputs]: ../../../../src/test/20261025_recipe_backbones/outputs/
[sheet]: ../../../../src/test/20261025_recipe_backbones/outputs/part1/sheet.md
[part1report]: ../../../../src/test/20261025_recipe_backbones/outputs/part1/reported.md
[stage3report]: ../../../../src/test/20261025_recipe_backbones/outputs/part2/reported_stage3.md
[stage5report]: ../../../../src/test/20261025_recipe_backbones/outputs/part2/reported_stage5.json

[lossideas]: ../../literature/2026-12-20_ways_to_improve.md#candidate-ideas-one-ranking
[framing]: 2026-12-21_advantages_framing.md
