# Stage report, 2026-10-27 to 2026-10-28: 40 graph neighbours put held emotion and style above the instructed Qwen image; genre stays level

> Covers 2026-10-27 to 2026-10-28, the r10 recipe sweep only. Previous stage report: [2026-10-26, the condition-free line](2026-10-26_condition_free_line.md).
> Every number comes from [report 12-28](../auto/percept/2026-12-28_recipe_sweep.md) or its [run log](../../../src/test/20261027_recipe_sweep/20261027_recipe_sweep_log.md). The previous recipe's starting held genre result comes from [12-27's results brief](../auto/percept/2026-12-27_relational_genre.md#results-brief).
> “Development” means the 29,232 development paintings, with genre labels on 23,682. “Held” means the 12,281 held paintings, with genre labels on 9,949.
> Report names keep the project's sequence dates, 12-27 and 12-28. Step dates and times are real Amsterdam dates and times.
> Table codes: G is the previous recipe, c* is the pick, T is the instructed Qwen image of an aspect. The glossary defines all codes.

## Summary

- The previous recipe passed its held emotion claim against PercepT core at both training scales.
- The search picked the previous recipe with graph neighbours raised from 20 to 40. All other settings stayed fixed.
- Held emotion AMI of the pick was 0.1066, against 0.0921 for the instructed Qwen image and 0.0874 for PercepT core.
- Held style AMI was 0.4066, against 0.4026 for the instructed Qwen image. The difference was +0.0041 [+0.0004, +0.0081], a narrow win.
- Held genre AMI was 0.5001, against 0.5062 for the instructed Qwen image. It was level under the agreed margin, not a win, and its whole difference interval lay below zero.
- The ablation removed one part at a time. It decides nothing. Our reading: emotion carries the clearest result, and the next choice awaits the user.

## Where we started

We asked whether a condition-free model could give emotion, style and genre spaces from one read of the default image. The model needs no aspect instruction at use. We compared its spaces with PercepT, the prior method that the project seeks to improve.

The previous recipe combined 12-24's recipe with a relational genre term and a sharper teacher row. Its held genre result entered this round from 12-27:

| Starting held genre comparison | AMI | Previous recipe minus comparator [95%] |
|---|---:|---|
| Previous recipe (G) | 0.5001 | reference |
| Instructed Qwen genre image (T) | 0.5062 | -0.0060 [-0.0105, -0.0014] |
| Strongest eligible PercepT entry | 0.4269 | +0.0732 [+0.0622, +0.0795] |

This result was level under the agreed margin of one genre seed range (0.0128). The whole interval against the instructed image lay below zero, so it was not a win. The [previous stage report](2026-10-26_condition_free_line.md) covered the earlier work.

We used two strong comparators for different questions. The instructed Qwen image gave Qwen3-VL-Embedding the instruction for each aspect, then clustered its embeddings directly. PercepT core was the project's PercepT re-implementation. We took the strongest eligible entry per aspect across its core and paper-settings entries.

We measured adjusted mutual information (AMI), the agreement between clusters and ArtELingo labels. ArtELingo is the painting dataset with emotion annotations, captions, and style and genre labels. We used image heads only, k-means with 20 clusters, and the mean over three cluster states. Paired 95% intervals used 2,000 painting and seed draws. The emotion test retook the strongest eligible PercepT entry in each draw.

A seed range is a fixed AMI spread across model seeds for one aspect, set before this round. It put gains on comparable scales. Mean gain was the difference over the previous recipe divided by each aspect's seed range, then averaged across aspects.

| Aspect | Seed range |
|---|---:|
| Emotion | 0.0046 |
| Style | 0.0082 |
| Genre | 0.0128 |

Sample and full scale describe training size, not the evaluation split. Both scales gave outputs on the same held paintings.

| Scale | Fit or train paintings | Stopping or validation paintings | Role this round |
|---|---:|---:|---|
| Sample | 13,400 | 1,489 | Previous-recipe claim, sweeps, confirmation, development ablation |
| Full | 42,969 | 6,152 | Previous-recipe claim, held test of the pick, held ablation |

The round asked whether the previous recipe still supported the held emotion claim against PercepT core. It then asked whether a sweep on development paintings could find a better recipe, for one held test of the pick.

Sources: [12-28, What ran](../auto/percept/2026-12-28_recipe_sweep.md#what-ran); [12-27, results brief](../auto/percept/2026-12-27_relational_genre.md#results-brief); [decision rule, statistics](../../../src/test/20261027_recipe_sweep/DECISION_RULE.md#3-the-statistics).

## The model tested in this round

The pick kept the previous recipe and raised graph neighbours from 20 to 40. Two paintings link when each lies among the other's nearest neighbours. At use, the model still reads the default image without an aspect instruction. Training uses Qwen's instructed views and captions.

The ablation removed each of these eight parts in turn, with all other settings fixed:

- The instructed graphs link nearby paintings in the instructed view for each aspect. Removal gives every aspect one default-view graph.
- The content correction subtracts content similarity before the emotion and style graphs are built. Genre uses no correction.
- The buddy terms pull graph-linked paintings closer through four image and caption terms.
- The caption endpoints let buddy terms use captions. Removal leaves only the image-to-image buddy term.
- The direct teacher makes each head copy the instructed Qwen image for its aspect.
- The relational term makes the genre head copy the teacher's pattern of similarities within a batch.
- The sharper teacher row gives nearby paintings more weight in that pattern. Removal sets the teacher row to the student's temperature.
- The shared trunk processes the input for all heads. Removal gives each aspect its own trunk.

![One default image gives three aspect spaces](../assets/2026-10-28_recipe_sweep/model_structure.png)

*Figure 1. Instructed views supply graphs and teacher targets; the model gives three outputs from one default image. Orange marks the neighbour change. Teal marks the relational term added since the previous stage diagram; the previous recipe already contained it. Content correction applies to emotion and style only.*

Sources: [spec, arms and ablation](../../superpowers/specs/2026-10-27-recipe-sweep-design.md#32-arms-and-controls); [previous stage, design](2026-10-26_condition_free_line.md#the-design-as-it-stands-on-2026-10-26).

## Timeline

The claim test and the search finished on the evening of 2026-10-27. The held ablation ended the next morning, at 03:53.

![Recipe sweep timeline](../assets/2026-10-28_recipe_sweep/timeline.png)

*Figure 2. Node and local lanes show overlapping work. The held ablation crossed the planned end and finished the next morning.*

## The steps, in date order

### 2026-10-27 · Stage 0: the previous recipe passed the held emotion claim (worked)

The previous recipe passed against PercepT core at both scales. We tested whether the recipe chosen for genre still supported the paper's emotion claim.

We reused its full-scale held outputs after checks, added the missing sample seed, and computed sample held outputs. Each comparison used five model seeds. The bar was fixed in advance. The gain had to reach at least 0.0100 AMI, and its 95% interval had to lie above zero. The claim required a full-scale pass and no sample-scale failure.

| Training scale | G AMI | PercepT core AMI | G minus PercepT core [95%] |
|---|---:|---:|---|
| Sample | 0.1043 | 0.0855 | +0.0188 [+0.0137, +0.0200] |
| Full | 0.1039 | 0.0874 | +0.0164 [+0.0107, +0.0182] |

Both comparisons used the held set. PercepT core's own reading was its strongest entry at both scales. The full-scale emotion score of the previous recipe was known before this test, so the point result was close to known.

Source: [run log, stage 0](../../../src/test/20261027_recipe_sweep/20261027_recipe_sweep_log.md#stage-0-gs-emotion-claim-and-analysis-2026-10-27).

### 2026-10-27 · Stage 1: graph neighbours led the single-setting sweep (worked)

Graph neighbours gave the largest guarded mean gain over the previous recipe. We varied thirteen settings alone to see where the recipe had room to improve.

We ran three alternative values of each setting on a reserved cluster node (node402), on development paintings, at sample scale and seed 0. All 39 candidate runs completed. The guard was fixed in advance: no aspect could lose more than half a seed range against the previous recipe's run on the same machine. The four settings with the largest positive guarded gains entered the joint grid.

| Setting and best value | Mean gain over the previous recipe on the node, seed ranges |
|---|---:|
| Graph neighbours, 40 | +0.924 |
| Teacher row temperature, 0.035 | +0.371 |
| Style relational weight, 8 | +0.288 |
| Shared trunk width, 512 | +0.164 |

![Curves for all thirteen settings](../assets/2026-10-28_recipe_sweep/knob_curves.png)

*Figure 3. Filled points pass the guard; open points fail. Blue bands mark the four settings sent to the grid.*

Emotion relational weights, weight decay and learning rate changes all lowered the mean gain. Patience changed it by at most 0.0012. Our reading: patience had no effect because the best checkpoints came early, at epochs 2 to 4 in the node's reference run.

Guard failures spanned the relational weights, teacher settings, learning rate, weight decay, trunk width, dropout, graph neighbours and buddy temperature. The attempts table lists each failed value. The [complete curves](../../../src/test/20261027_recipe_sweep/20261027_recipe_sweep_log.md#stage-1-single-knob-sweep-on-node402-2026-10-27) keep every attempt.

The node repeated its reference run bit for bit. Its genre score differed from the local reference by -0.0077, about 0.6 genre seed ranges. The rule allowed this machine difference, because every sweep comparison used the node's own reference. Scorer parity was exact.

Sources: [12-28, stage 1](../auto/percept/2026-12-28_recipe_sweep.md#stage-1-node402-knob-curves); [run log, stage 1](../../../src/test/20261027_recipe_sweep/20261027_recipe_sweep_log.md#stage-1-single-knob-sweep-on-node402-2026-10-27).

### 2026-10-27 · Stage 2: combinations did not beat the graph change alone (did not work)

The graph change alone stayed first after the joint grid. We tested whether the best single-setting gains added together. They did not.

The grid used each selected setting at its eligible positive values and at the previous recipe's value. It added 18 new runs, with the same seed, development set and half-seed-range guard. The pool held 21 guarded positive candidates. The fixed rule sent its top three to local confirmation.

| Rank and candidate | Emotion AMI | Style AMI | Genre AMI | Mean gain over the previous recipe on the node, seed ranges |
|---|---:|---:|---:|---:|
| 1. Graph neighbours 40 | 0.1107 | 0.4090 | 0.5060 | +0.924 |
| 2. Neighbours 40, style relational weight 2, trunk width 512 | 0.1103 | 0.4031 | 0.5144 | +0.874 |
| 3. Neighbours 40, teacher row temperature 0.035 | 0.1111 | 0.4018 | 0.5093 | +0.751 |

All three met the guard. A teacher row temperature of 0.035 alone gave +0.371. Neighbours 40 alone gave +0.924. Together they gave +0.751. The sources give no checked reason for this interaction.

The job monitor stopped at 19:27 after 30 minutes without output, while the verification job still ran. The run chat watched the same job with a longer limit and finished the ordered tail by hand. No training job reran and no result changed.

Source: [12-28, stage 2](../auto/percept/2026-12-28_recipe_sweep.md#stage-2-node402-top-runs).

### 2026-10-27 · Stage 3: confirmation picked 40 graph neighbours (worked)

The 40-neighbour model had the largest qualifying mean gain on local seeds 1 to 3. We checked the shortlist against the frozen local previous recipe on seeds not used in the search.

The bar was fixed in advance. Each aspect's difference interval had to start above minus one seed range. The 95% interval of the mean gain had to lie above zero. Among qualifiers, the largest mean gain set the pick, written before any full-scale run.

Each row used seeds 1 to 3 and the 29,232 development paintings. Gains are over the previous recipe.

| Candidate | Emotion AMI | Style AMI | Genre AMI | Mean gain [95%], seed ranges | Qualified? |
|---|---:|---:|---:|---|---|
| Previous recipe (G) | 0.1062 | 0.3931 | 0.5050 | reference | reference |
| Neighbours 40 (c*) | 0.1089 | 0.4072 | 0.5019 | +0.683 [+0.452, +0.959] | Yes |
| Neighbours 40, style relational weight 2, trunk width 512 | 0.1085 | 0.4020 | 0.4990 | +0.369 [+0.001, +0.724] | Yes |
| Neighbours 40, teacher row temperature 0.035 | 0.1082 | 0.3991 | 0.5006 | +0.272 [-0.187, +0.714] | No |

![Local confirmation of the three candidates](../assets/2026-10-28_recipe_sweep/confirmation.png)

*Figure 4. Confirmation compares the shortlist with the local previous recipe. Two candidates qualify; the softer teacher combination fails the mean-gain interval bar.*

Most of the pick's mean gain came from style. Its style difference over the previous recipe was +0.0141 [+0.0121, +0.0160]. Its emotion difference was +0.0026 [+0.0011, +0.0045]. Genre's interval crossed zero.

The mean-gain intervals of the two qualifiers overlapped. The rule reports this and takes the largest mean gain as the pick. The third candidate did not qualify, because its mean-gain interval crossed zero. It received no held test. No training run failed or repeated.

Our reading: broader graph links could supply useful positives and explain the development style gain. We did not check this mechanism.

Source: [12-28, stage 3](../auto/percept/2026-12-28_recipe_sweep.md#stage-3-local-confirmation).

### 2026-10-27 · Stage 4: the held pick exceeded the instructed image on emotion and style, and was level on genre (worked)

On held paintings the pick exceeded the instructed Qwen image on emotion, and narrowly on style. Genre was level under the agreed margin.

We trained the fixed pick at full scale on seeds 0 to 4, then tested it once on the held set. The bar was fixed in advance. A win needed the 95% difference interval above zero. Level needed its lower bound above minus one seed range. The emotion claim against PercepT core kept the stage 0 bar.

Held outcomes triggered no next experiment automatically. The comparisons below use the held set and five model seeds.

| Aspect | c* AMI | T AMI | c* minus T [95%] |
|---|---:|---:|---|
| Emotion | 0.1066 | 0.0921 | +0.0145 [+0.0113, +0.0177] |
| Style | 0.4066 | 0.4026 | +0.0041 [+0.0004, +0.0081] |
| Genre | 0.5001 | 0.5062 | -0.0060 [-0.0103, -0.0015] |

![Held differences against the instructed Qwen image](../assets/2026-10-28_recipe_sweep/held_results.png)

*Figure 5. Left: the pick, the previous recipe and the comparators, with broken AMI axes. Right: differences over the instructed image; genre lies below zero and above the level bar of minus one seed range.*

We also reported differences over the previous recipe and over the strongest eligible PercepT entry per aspect.

| Aspect | G AMI | c* minus G [95%] | Strongest PercepT entry / AMI | c* minus PercepT [95%] |
|---|---:|---|---|---|
| Emotion | 0.1039 | +0.0027 [+0.0001, +0.0054] | B1-own / 0.0874 | +0.0192 [+0.0137, +0.0207] |
| Style | 0.4057 | +0.0009 [-0.0044, +0.0065] | B6-own / 0.2297 | +0.1769 [+0.1663, +0.1809] |
| Genre | 0.5001 | 0.0000 [-0.0028, +0.0030] | B5-r20 / 0.4269 | +0.0732 [+0.0622, +0.0796] |

Emotion and style used PercepT core entries read by their own topic assignment. Genre used a paper-settings entry. The glossary defines the codes.

The development style gain did not become a clear held gain over the previous recipe. Both genre means rounded to 0.5001, although their held outputs differed in every head. Our reading: emotion was the strongest result, the style win over the instructed image was narrow, and genre was level, not a win.

Leiden, community detection on a neighbour graph, gave a descriptive genre check with no bar. The pick reached genre AMI 0.5082, against 0.4454 for PercepT's paper-settings entry: +0.0627 [+0.0529, +0.0736]. The binding rule limited the Leiden reading to genre. Emotion, style and a Leiden comparison with the instructed image were not computed.

Source: [12-28, stage 4](../auto/percept/2026-12-28_recipe_sweep.md#stage-4-held-full-scale).

### 2026-10-27 to 2026-10-28 · Ablation: removing each part (no bar)

Removing the relational term lowered held genre most. Removing the instructed graphs, buddy terms or content correction lowered held emotion and style most. We removed each part to measure its contribution within the fixed recipe.

Each arm removed one part and retuned nothing. Development used sample-scale training and seeds 1 to 3. Held used full-scale training and seeds 0 to 4. Each comparison used the pick's own outputs and paired intervals. The rule set no pass bar, so the ablation decides nothing. All removals and independent checks completed without a failed run or rerun.

![Changes when each model part was removed](../assets/2026-10-28_recipe_sweep/ablation.png)

*Figure 6. Held circles and development squares show removal minus pick, with paired intervals. The shared symmetric log scale separates large losses from small changes.*

Held AMI changes below are removal minus the complete pick, on the held set. A dagger marks an interval wholly above or below zero. The [full ablation table](../auto/percept/2026-12-28_recipe_sweep.md#ablation) gives all absolute scores and intervals, plus the development table.

| Part removed | Emotion change | Style change | Genre change |
|---|---:|---:|---:|
| Complete pick, AMI | 0.1066 | 0.4066 | 0.5001 |
| Relational term | +0.0009 | 0.0000 | -0.0440 † |
| Sharper teacher row | +0.0016 | +0.0008 | -0.0038 † |
| Direct teacher | +0.0035 † | +0.0041 † | -0.0031 |
| Buddy terms | -0.0277 † | -0.1512 † | -0.0015 |
| Caption endpoints | +0.0009 | +0.0072 † | -0.0034 |
| Content correction | -0.0090 † | -0.0752 † | +0.0072 † |
| Instructed graphs | -0.0327 † | -0.1903 † | +0.0054 |
| Shared trunk | -0.0001 | +0.0023 | -0.0001 |

Removing the relational term lowered held genre by 0.0440 [-0.0512, -0.0372]. Removing the sharper teacher row lowered it by 0.0038 [-0.0074, -0.0002]. The shared trunk showed no clear held change on any aspect. Its removal raised development genre, but that result did not repeat on held paintings.

Two removals raised the same aspect on development and held paintings. Without the direct teacher, emotion rose. Without the caption endpoints, style rose. Removing the direct teacher also raised held style.

| Removal and comparison | Development change [95%] | Held change [95%] |
|---|---|---|
| Direct teacher, emotion | +0.0022 [+0.0005, +0.0037] | +0.0035 [+0.0008, +0.0056] |
| Caption endpoints, style | +0.0042 [+0.0013, +0.0072] | +0.0072 [+0.0015, +0.0121] |

These results do not show that either part is unneeded, because the effects on other aspects were mixed or uncertain. Removing the content correction raised held genre by +0.0072 [+0.0030, +0.0113] and lowered emotion and style. Its development genre interval crossed zero.

Our reading: the large effects fit the stated roles of the instructed graphs, buddy terms, content correction and relational term. The direct teacher and the caption endpoints call for follow-up. This check changes no recipe.

The ablation cost more than planned.

| Item | Planned | Actual |
|---|---|---|
| Development training | about 17 GPU minutes | 63 GPU minutes |
| Held training | about 80 GPU minutes | 252 GPU minutes |
| End of the chain | 23:00 planned end, 03:00 report time | 03:53 on 2026-10-28 |

The buddy removal trained longest. In development, its genre head reached the 200-epoch cap in seeds 2 and 3 and stopped at epoch 165 in seed 1.

Sources: [12-28, ablation](../auto/percept/2026-12-28_recipe_sweep.md#ablation); [run log, ablation and deviations](../../../src/test/20261027_recipe_sweep/20261027_recipe_sweep_log.md#ablation).

### 2026-10-27 · Diagnostics: each label set favoured its named head (no bar)

Each aspect's labels agreed most with its named head in both models. After the relevant verdicts, we compared every head with every label set, to see what the separate spaces kept.

The diagnostics used full-scale held outputs across five seeds. They had no pass bar and chose nothing. Cells show AMI for the previous recipe, then the pick.

| Head | Emotion labels | Style labels | Genre labels |
|---|---|---|---|
| Emotion | 0.1039 to 0.1066 | 0.1647 to 0.1641 | 0.4048 to 0.3889 |
| Style | 0.0622 to 0.0610 | 0.4057 to 0.4066 | 0.2983 to 0.2903 |
| Genre | 0.0695 to 0.0701 | 0.2668 to 0.2720 | 0.5001 to 0.5001 |

The emotion head's agreement with genre labels fell. The heads also agreed slightly less with one another. Class F1 moved both ways. Overlap of neighbours with the teacher's neighbours rose for style and genre and fell for emotion. We did not check a visual reading of the saved galleries.

Source: [12-28, analysis](../auto/percept/2026-12-28_recipe_sweep.md#analysis); [run log, diagnostic detail](../../../src/test/20261027_recipe_sweep/20261027_recipe_sweep_log.md#diagnostic-detail-completed-after-the-relevant-verdicts).

## All attempts at a glance

The search gave a qualifying pick. Its softer-teacher combination failed confirmation. Single-setting rows give mean gain over the previous recipe's node run, in seed ranges. Their guard allowed at most half a seed range of loss per aspect. All training runs completed.

| Date | Step or attempt | Result against its bar | Outcome |
|---|---|---|---|
| 10-27 | Stage 0, held emotion claim | Full gain over PercepT core +0.0164; sample +0.0188; both above the required +0.0100 with positive intervals | worked |
| 10-27 | Stage 1, single-setting sweep | Neighbours 40 led by +0.924 under the half-range guard | worked |
| 10-27 | Emotion relational weight 2 | -0.559; guard failed | did not work |
| 10-27 | Emotion relational weight 8 | -0.700; guard failed | did not work |
| 10-27 | Emotion relational weight 16 | -1.197; guard failed | did not work |
| 10-27 | Style relational weight 16 | -0.119; guard failed | did not work |
| 10-27 | Genre relational weight 12 | -0.208; guard failed | did not work |
| 10-27 | Genre relational weight 16 | -0.368; guard failed | did not work |
| 10-27 | Teacher row temperature 0.015 | -0.121; guard failed | did not work |
| 10-27 | Teacher row temperature 0.025 | -0.303; guard failed | did not work |
| 10-27 | Direct teacher weight 0.125 | -0.225; guard failed | did not work |
| 10-27 | Direct teacher weight 0.5 | -0.454; guard failed | did not work |
| 10-27 | Learning rate 0.0003 | -0.391; guard failed | did not work |
| 10-27 | Learning rate 0.0005 | -0.101; guard met, below the previous recipe | did not work |
| 10-27 | Learning rate 0.002 | -0.339; guard met, below the previous recipe | did not work |
| 10-27 | Weight decay 0 | -0.239; guard failed | did not work |
| 10-27 | Weight decay 0.01 | -0.299; guard failed | did not work |
| 10-27 | Weight decay 0.2 | -0.146; guard failed | did not work |
| 10-27 | Batch pairs 256 | -0.169; guard met, below the previous recipe | did not work |
| 10-27 | Batch pairs 1024 | -0.146; guard met, below the previous recipe | did not work |
| 10-27 | Trunk width 1536 | -0.156; guard failed | did not work |
| 10-27 | Trunk width 2048 | -0.032; guard met, below the previous recipe | did not work |
| 10-27 | Dropout 0 | -0.414; guard failed | did not work |
| 10-27 | Dropout 0.2 | -0.136; guard met, below the previous recipe | did not work |
| 10-27 | Dropout 0.3 | -0.854; guard failed | did not work |
| 10-27 | Graph neighbours 10 | -0.191; guard failed | did not work |
| 10-27 | Graph neighbours 30 | -0.064; guard met, below the previous recipe | did not work |
| 10-27 | Buddy temperature 0.07 | -0.403; guard failed | did not work |
| 10-27 | Buddy temperature 0.1 | -0.766; guard failed | did not work |
| 10-27 | Positive values outside the top four settings: style relational weight 2, genre relational weight 4, direct teacher weight 0.0625, batch pairs 2048, buddy temperature 0.03 | +0.069, +0.067, +0.158, +0.133, +0.060; guard met | not sent to the grid, except style relational weight 2 |
| 10-27 | Patience 5, 10, 30 | +0.001, +0.000, +0.000 | no effect |
| 10-27 | Stage 2, joint grid | Graph change alone stayed first at +0.924; the teacher row combination gave +0.751, below +0.924 | did not work |
| 10-27 | Stage 3, confirmation and pick | +0.683 [+0.452, +0.959] over the local previous recipe; all aspect guards met | worked |
| 10-27 | Style relational weight 2 with trunk width 512 and neighbours 40 | +0.369 [+0.001, +0.724]; qualified, below the pick | not picked |
| 10-27 | Teacher row temperature 0.035 with neighbours 40 | +0.272 [-0.187, +0.714]; mean-gain interval crossed zero | did not qualify |
| 10-27 | Stage 4, held pick | Emotion and style intervals above the instructed image; genre lower bound above minus one seed range | worked |
| 10-27 to 10-28 | Eight-part ablation | Removing the relational term cost genre -0.0440 against the pick; no bar | no bar |
| 10-27 | Diagnostics, after each verdict | Each label set favoured its named head; no bar | no bar |

Source: [12-28](../auto/percept/2026-12-28_recipe_sweep.md) and the [complete single-setting attempts](../../../src/test/20261027_recipe_sweep/20261027_recipe_sweep_log.md#stage-1-single-knob-sweep-on-node402-2026-10-27).

## Where we stand

On held paintings the pick exceeded the instructed Qwen image on emotion and narrowly on style, and was level with it on genre. It supported the held emotion claim against PercepT core.

| Held aspect | Pick AMI | Previous recipe AMI | Instructed Qwen image AMI | Strongest eligible PercepT entry AMI |
|---|---:|---:|---:|---:|
| Emotion | 0.1066 | 0.1039 | 0.0921 | 0.0874, core own reading |
| Style | 0.4066 | 0.4057 | 0.4026 | 0.2297, core own reading |
| Genre | 0.5001 | 0.5001 | 0.5062 | 0.4269, paper settings at 20 clusters |

Our reading: emotion gave the clearest support. The style interval over the instructed image barely cleared zero, and its interval over the previous recipe crossed zero.

The evidence has limits:

- Selection among about 60 configurations inflated the search's seed-0 gains.
- Confirmation seeds 1 to 3 were new to these runs, but rounds 12-25 and 12-27 also used them for recipe choices.
- Earlier rounds read the held split and the instructed-image scores. The previous recipe became the recipe after its held genre read.
- The known previous-recipe emotion score made stage 0 close to known.
- The claims cover k-means, the Qwen image encoder and ArtELingo only. This round showed no transfer to another encoder.
- The ablation decides nothing. Retrieval still has no result.

**Verdict (reference).** Under the fixed rule, the pick replaces the previous recipe. Held outcomes trigger no next step automatically. This is not the user's reading.

Source: [12-28, results brief](../auto/percept/2026-12-28_recipe_sweep.md#results-brief).

## Next

The choice is the user's, taken in the next decide chat. The results brief offered three options:

1. Remove the caption endpoints from the pick. On development paintings, the mean gain over the complete pick was +0.315 [+0.121, +0.534] seed ranges, from style. Other aspects showed no clear change. This costs under an hour. Its held ablation result already exists, so another held test would be close to known.
2. Test the recipe with SigLIP 2, SigLIP 1 and CLIP L/14. This costs about half a day of GPU work. It shows whether the recipe transfers to other image encoders. Node402 is idle until about 2026-10-30 23:30.
3. Draft the CVPR abstract from the supported claims, for the 2026-11-10 deadline. This costs writing time and fixes what the paper can claim.

Our recommendation: take the caption-endpoint follow-up first, because it is small and could become the recipe for the encoder tests. Then run the encoder tests before the node reservation ends, and draft the abstract in parallel. The present emotion claim rests on the complete pick. A changed recipe needs its own claim test.

Source: [12-28, options and recommendation](../auto/percept/2026-12-28_recipe_sweep.md#results-brief).

## Glossary

The plain names below match the codes in the source reports.

- **Pick (c\*):** the previous recipe with graph neighbours raised from 20 to 40. Every other setting stayed fixed.
- **Previous recipe (G; GS-0.02):** 12-24's recipe, called R6-5, plus the relational genre term and the sharper teacher row.
- **Recipe:** the training losses and their settings.
- **Aspect:** emotion, style or genre.
- **Head:** the model output for one aspect.
- **Instructed Qwen image (T):** the Qwen3-VL-Embedding image embedding under that aspect's instruction, clustered directly as a comparator.
- **Qwen:** Qwen3-VL-Embedding, the image-text encoder that supplied instructed views and teacher targets.
- **PercepT core:** the project's PercepT re-implementation on its feature stores.
- **B1-own:** PercepT core's Qwen arm (P3) with its own topic reading. It is the strongest emotion entry.
- **B6-own:** PercepT core P2 with its own topic reading. It is the strongest style entry in this comparison.
- **B5-r20:** PercepT paper-settings entry P4, reclustered at 20. It is the strongest eligible genre entry in this comparison.
- **B4-r20:** PercepT paper-settings entry P1, reclustered at 20. It is the comparator for the reported Leiden genre result.
- **Own reading:** PercepT's top-topic assignment, not a separate k-means read of topic probabilities.
- **Seed:** the random choices for a model run or a cluster state. The report names which one.
- **Seed range:** a fixed AMI spread across model seeds for one aspect, set before this round. It is the unit for gains.
- **Mean gain:** each aspect's AMI gain over the reference divided by its seed range, then averaged across aspects. The ablation uses the complete pick as reference.
- **Guard:** the per-aspect loss limit that a candidate had to meet before its mean gain could count.
- **Sample and full scale:** the smaller fit and stopping sets, or the full train and validation sets. Neither names an evaluation split.
- **Development:** paintings used to compare recipes and choose the pick.
- **Held:** separate paintings for claim tests. Earlier rounds also read this split.
- **ArtELingo:** the painting dataset with emotion annotations and captions, plus style and genre labels.
- **AMI:** adjusted mutual information, the agreement between clusters and labels, adjusted for chance.
- **k-means:** the cluster method used for the main comparisons, with a fixed cluster count.
- **Leiden:** community detection on a neighbour graph, reported here for genre as a descriptive check.
- **Buddy graph:** links between nearby paintings that define positive pairs for the buddy terms.
- **Graph neighbours (k11; mutual top-k):** two paintings link when each is among the other's nearest neighbours under the graph score.
- **Instructed view:** embeddings made with an aspect instruction. The graph's view can differ from the direct image teacher.
- **Instructed graphs:** separate graphs from the aspect views. Their removal uses one graph from the default view.
- **Content correction:** subtraction of content similarity before emotion and style graph links are chosen.
- **Buddy terms:** losses that pull graph-linked paintings closer through image and caption endpoints.
- **Caption endpoints:** the caption side of buddy pairs. Removal leaves only image endpoints.
- **Direct teacher:** a loss that makes each head copy its instructed Qwen image target.
- **Relational term:** a loss that copies the teacher's pattern of similarities between paintings. The pick uses it on genre.
- **Sharper teacher row:** a lower-temperature similarity target that gives nearby paintings more weight.
- **Shared trunk:** the common first network layer before the aspect heads.
- **Ablation:** removal of one part at a time, without retuning, to measure its contribution in the complete recipe.

## Sources

The checked auto report and the run log supply this round's results. The spec and the rule supply the tests set in advance.

- [12-28: recipe sweep, main auto report](../auto/percept/2026-12-28_recipe_sweep.md).
- [Recipe sweep run log](../../../src/test/20261027_recipe_sweep/20261027_recipe_sweep_log.md).
- [12-27: relational genre, results brief only](../auto/percept/2026-12-27_relational_genre.md#results-brief), for the previous recipe's starting held genre result.
- [Recipe sweep spec](../../superpowers/specs/2026-10-27-recipe-sweep-design.md), especially arms, ablation and measurement.
- [Recipe sweep decision rule](../../../src/test/20261027_recipe_sweep/DECISION_RULE.md), for the bars and the order set in advance.
- [Previous stage report, 2026-10-26](2026-10-26_condition_free_line.md), for the earlier record and design.
