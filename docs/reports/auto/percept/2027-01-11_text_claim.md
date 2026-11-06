# Text claim: c* and late fusion hold on held ArtELingo and fail on MMArt

> 2026-11-06 (sequence name 2027-01-11) · Spec: [design](../../../superpowers/specs/2026-11-06-text-claim-design.md) · Decision rule: §3.4, §3.5, §6 @ 452e143e · Run log: [full detail](../../../../src/test/20261106_text_claim/20261106_text_claim_log.md)

## Results brief

**What we tried.** We froze c* and tested its caption side on data that shaped nothing: 12,281 held ArtELingo paintings, and 16,384 MMArt paintings outside ArtELingo with their generated narrative text. We also tested late fusion, a weighted sum of c* and plain Qwen cosines with weight 0.7 on plain Qwen ([spec](../../../superpowers/specs/2026-11-06-text-claim-design.md)). A cell is one aspect (emotion, style, genre) in one direction (text to image, image to text, text to text). Each set has 9 cells, scored by label mAP.

**Outcome against the bar.** c* and late fusion hold on ArtELingo and fail on MMArt. "Holds" needs a WIN over plain Qwen in 9 of 9 cells and over the teacher's text in 5 or more; 3 or 4 gives "qualified". A WIN means the interval's lower bound is above 0. Late fusion also needs pair R@10 no more than 0.02 below plain Qwen, at the lower bound. All checks pass, and no verdict is void.

| Row and set | WIN over plain Qwen | WIN over the teacher's text | Pair bar | Verdict |
|---|---|---|---|---|
| c*, ArtELingo | 9 of 9 | 6 of 9 | no bar | holds |
| c*, MMArt | 8 of 9 | 3 of 9 | no bar | fails |
| Late fusion, ArtELingo | 9 of 9 | 5 of 9 | passes | holds |
| Late fusion, MMArt | 9 of 9 | 0 of 9 | passes | fails |

The numbers that decided each verdict (label mAP difference [95% interval]):

- **c*, ArtELingo:** the weakest WIN over plain Qwen is style text to text, +0.0127 [+0.0121, +0.0133]. The weakest WIN over the teacher's text is genre image to text, +0.0062 [+0.0041, +0.0086].
- **c*, MMArt:** genre text to text is 0.4564 against plain Qwen's 0.4701, -0.0137 [-0.0190, -0.0078]. All five seeds are below plain Qwen, so seed noise does not explain the miss. Validation already showed it: 0.4628 against 0.4778.
- **Late fusion, ArtELingo:** the fifth WIN over the teacher's text is style image to text, +0.0009 [+0.0005, +0.0012].
- **Late fusion, MMArt:** no cell is a WIN over the teacher's text. The highest lower bound is style text to text, -0.0016 [-0.0023, -0.0009]. Validation also gave 0 of 9 above the teacher's text.
- **Pair bar:** the lowest of the four lower bounds is -0.0143 (MMArt, text to image), against the bar of -0.02.

**What it does not show.**

- **Late fusion's count over the teacher's text depends on the read-out.** At one fixed read-out, its cells above the teacher's text on ArtELingo number 4 (raw cosine), 2 (centering) and 5 (query expansion). These are mean differences with no intervals. c* keeps 6 at every read-out.
- **The cause of the MMArt miss.** We do not know it. A Qwen model generated the narrative text, and this may favour Qwen rows. MMArt's emotion labels agree with ArtELingo's on 43% of shared paintings.
- **Other encoders and human text.** The claims cover Qwen on these two sets, with five seeds and read-outs picked on validation. MMArt has no human text, and plain Qwen is its only plain encoder measured. Its teacher's text uses ArtELingo's instructions. Intervals have no multiplicity correction.
- **Why weight 0.7 helps.** The diagnostics describe late fusion at this weight and test no cause.
- **Pair numbers.** c*'s claims cover label retrieval. Its pair numbers stand in a table with no reading (spec, decision D1).

**Options for next.**

a. **Take the ArtELingo claims to the paper.** No GPU, one writing chat. The paper states both holds, MMArt's failed verdicts and its cell claims. It shows nothing new; it fixes the claims for the 11-10 target and the paper on 11-16. The wording of c*'s pair numbers still waits for your talk with the other authors.
b. **Read the missed cell per class.** Saved per-query scores, CPU, under 1 hour, a small loop. It could show whether the miss sits in a few frequent genres: macro mAP is 0.3921 against plain Qwen's 0.3901 (no interval).
c. **Test the shared 11,528 MMArt rows** (spec §2). These are held ArtELingo paintings with MMArt's text and labels. About 10 minutes of GPU for the narrative text, about 1 h for all five texts (our estimate from this run's rates; the forward pass is not measured), plus a short spec, a build chat and a run chat. It could show whether the failure comes from the new paintings or from the text and labels. It cannot change a verdict.

**Your reading.**

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** c* on ArtELingo is the firm result, and we are confident in it. Every cell keeps its validation sign against both comparators, and the counts do not move across read-outs or leak checks. Its claim over the teacher's text is a style and genre claim: all 3 emotion cells are BELOW, on both sets.

The MMArt miss on genre text to text is real: five seeds, all three read-outs and validation agree. In this cell c* is also below plain Qwen on three of the four other MMArt texts. Macro mAP has the opposite sign, which points weakly to frequent classes. We do not know the cause.

Late fusion's "holds" is correct under the rule, and it is thin. The firm part is 18 of 18 WIN over plain Qwen with the pair bar passed on both sets. The claim over the teacher's text is the weak part. Our reading of weight 0.7: the plain term's spread is about 2 times the c* term's, so late fusion's ranking follows plain Qwen more than c*. We are less sure of this, as nothing tested it.

**Our recommendation.** Option a, with option b beside it if you want a reason for the miss. c* on ArtELingo becomes the main claim. MMArt goes in per cell, with its failed verdict stated. Late fusion goes in as a second claim, and its sentence adds "read-outs picked on validation", as c*'s does. Option c can wait: it cannot change a verdict, and the target is close.

</details>

## What ran

- **Stages, data, seeds:** regression checks, validation sweep (seed ranges, picks), claim test on held rows, descriptive diagnostics; seeds 0 to 4.
- **Code:** local GPU. Stage 1 committed at 452e143e, before any held score; the re-derivation (C13) at 3c435bb9; verdicts at dcb43313.
- **Checks:** gates pass, 39 of 39 stores after one fix (Deviations). C13 passes with 0 failures; the largest per-query AP difference is 1.8e-7. The closest bound to a bar is 3.5e-4 away: late fusion, MMArt, genre text to text against the teacher's text, at the LEVEL line.

## Results

Verdict table: Results brief. Spec IDs: SC1 c*, ArtELingo; SC2 c*, MMArt; SC3 late fusion, ArtELingo; SC4 late fusion, MMArt.

Differences are label mAP; intervals pair paintings and seeds. Range is the validation seed range, the LEVEL unit. Directions: t2i text to image, i2t image to text, t2t text to text. Plain Qwen cells are WIN unless marked. Read-outs, query counts and [absolute label mAP](../../../../src/test/20261106_text_claim/20261106_text_claim_log.md#all-bar-cells-and-absolute-label-map) are in the round log.

Paintings per aspect: ArtELingo 12,281 for emotion and style, 9,949 for genre; MMArt 16,218 for emotion, 16,384 for style, 13,420 for genre.

### c*, ArtELingo (SC1)

| Cell | Against plain Qwen | Against the teacher's text | Range |
|---|---|---|---|
| emotion/t2i | +0.0417 [+0.0411, +0.0424] | -0.0036 [-0.0042, -0.0031] BELOW | 0.0023 |
| emotion/i2t | +0.0473 [+0.0454, +0.0491] | -0.0166 [-0.0187, -0.0145] BELOW | 0.0016 |
| emotion/t2t | +0.0923 [+0.0904, +0.0939] | -0.0308 [-0.0325, -0.0294] BELOW | 0.0044 |
| style/t2i | +0.0381 [+0.0368, +0.0394] | +0.0159 [+0.0146, +0.0172] WIN | 0.0018 |
| style/i2t | +0.0284 [+0.0278, +0.0291] | +0.0137 [+0.0132, +0.0142] WIN | 0.0011 |
| style/t2t | +0.0127 [+0.0121, +0.0133] | +0.0113 [+0.0107, +0.0119] WIN | 0.0014 |
| genre/t2i | +0.0657 [+0.0638, +0.0676] | +0.0403 [+0.0379, +0.0426] WIN | 0.0038 |
| genre/i2t | +0.0595 [+0.0570, +0.0622] | +0.0062 [+0.0041, +0.0086] WIN | 0.0063 |
| genre/t2t | +0.0427 [+0.0383, +0.0475] | +0.0162 [+0.0118, +0.0211] WIN | 0.0135 |

### c*, MMArt (SC2)

| Cell | Against plain Qwen | Against the teacher's text | Range |
|---|---|---|---|
| emotion/t2i | +0.0086 [+0.0074, +0.0097] | -0.0021 [-0.0033, -0.0009] BELOW | 0.0029 |
| emotion/i2t | +0.0057 [+0.0053, +0.0060] | -0.0028 [-0.0032, -0.0024] BELOW | 0.0006 |
| emotion/t2t | +0.0061 [+0.0056, +0.0065] | -0.0012 [-0.0017, -0.0008] BELOW | 0.0010 |
| style/t2i | +0.0684 [+0.0659, +0.0709] | +0.0149 [+0.0125, +0.0174] WIN | 0.0054 |
| style/i2t | +0.0496 [+0.0481, +0.0511] | +0.0183 [+0.0172, +0.0195] WIN | 0.0043 |
| style/t2t | +0.0273 [+0.0257, +0.0289] | +0.0171 [+0.0155, +0.0187] WIN | 0.0039 |
| genre/t2i | +0.0575 [+0.0549, +0.0601] | +0.0022 [-0.0007, +0.0050] LEVEL | 0.0046 |
| genre/i2t | +0.0138 [+0.0092, +0.0194] | -0.0324 [-0.0372, -0.0268] BELOW | 0.0189 |
| genre/t2t | -0.0137 [-0.0190, -0.0078] no WIN | -0.0264 [-0.0320, -0.0203] BELOW | 0.0205 |

### Late fusion (SC3, SC4)

Late fusion is a WIN over plain Qwen in all 18 cells, from +0.0036 [+0.0034, +0.0038] to +0.0717 [+0.0703, +0.0729]; each cell is in the round log. Against the teacher's text:

| Cell | ArtELingo | MMArt |
|---|---|---|
| emotion/t2i | -0.0254 [-0.0259, -0.0250] BELOW | -0.0056 [-0.0062, -0.0050] BELOW |
| emotion/i2t | -0.0272 [-0.0289, -0.0255] BELOW | -0.0049 [-0.0052, -0.0045] BELOW |
| emotion/t2t | -0.0514 [-0.0527, -0.0502] BELOW | -0.0035 [-0.0038, -0.0031] BELOW |
| style/t2i | +0.0076 [+0.0066, +0.0087] WIN | -0.0011 [-0.0026, +0.0004] BELOW |
| style/i2t | +0.0009 [+0.0005, +0.0012] WIN | -0.0124 [-0.0131, -0.0117] BELOW |
| style/t2t | +0.0052 [+0.0050, +0.0055] WIN | -0.0016 [-0.0023, -0.0009] BELOW |
| genre/t2i | +0.0227 [+0.0206, +0.0248] WIN | -0.0096 [-0.0120, -0.0072] BELOW |
| genre/i2t | -0.0148 [-0.0171, -0.0122] BELOW | -0.0310 [-0.0336, -0.0283] BELOW |
| genre/t2t | +0.0068 [+0.0048, +0.0090] WIN | -0.0044 [-0.0072, -0.0015] BELOW |

### Pair bar and pair numbers

Late fusion pair bar: lower bound above -0.02. Pair queries: ArtELingo 61,744 captions (t2i), 12,281 paintings (i2t); MMArt 16,384 in both.

| Set | Direction | Late fusion − plain Qwen, pair R@10 | Passes |
|---|---|---|---|
| ArtELingo | t2i | +0.0029 [+0.0010, +0.0049] | yes |
| ArtELingo | i2t | -0.0030 [-0.0079, +0.0019] | yes |
| MMArt | t2i | -0.0114 [-0.0143, -0.0085] | yes |
| MMArt | i2t | -0.0084 [-0.0108, -0.0061] | yes |

Pair metrics at frozen pair picks.

| Set | Row | Head | Direction | R@1 | R@10 | Median rank |
|---|---|---|---|---|---|---|
| ArtELingo | c* | genre | t2i | 0.0385 | 0.1517 | 174.8 |
| ArtELingo | late fusion | genre | t2i | 0.0910 | 0.2541 | 83.7 |
| ArtELingo | plain Qwen | all | t2i | 0.0905 | 0.2512 | 94.0 |
| ArtELingo | c* | genre | i2t | 0.0573 | 0.2309 | 61.4 |
| ArtELingo | late fusion | genre | i2t | 0.1575 | 0.4106 | 19.4 |
| ArtELingo | plain Qwen | all | i2t | 0.1692 | 0.4136 | 19.0 |
| MMArt | c* | style | t2i | 0.0075 | 0.0518 | 492.2 |
| MMArt | late fusion | style | t2i | 0.5902 | 0.8379 | 1.0 |
| MMArt | plain Qwen | all | t2i | 0.6149 | 0.8493 | 1.0 |
| MMArt | c* | emotion | i2t | 0.0856 | 0.2883 | 43.6 |
| MMArt | late fusion | emotion | i2t | 0.6558 | 0.8922 | 1.0 |
| MMArt | plain Qwen | all | i2t | 0.6718 | 0.9006 | 1.0 |

Pair R@10 on MMArt is near its ceiling, so late fusion's pair clause says less there than on ArtELingo; R@1 and median rank stand beside it.

**Verdict under the rule:** c* holds on ArtELingo and fails on MMArt; late fusion holds on ArtELingo and fails on MMArt. A failed verdict stands.

Licensed: spec §6's c* sentence for ArtELingo, with "6 of them"; its late fusion sentence, with "5 of them". The late fusion sentence names no read-out pick.

MMArt's teacher cells stand as their own claims despite the failed verdicts: c* on style in every direction, "better than the teacher's text"; c* on genre text to image, "level with or better than". No other MMArt teacher cell licenses a claim, and late fusion has none. On MMArt test the narrative text names its own emotion in 2.0% of rows, style in 0.04% and genre in 4.9% (spec, on validation: 2.3%, 0.0%, 4.7%).

## Analysis

**c*'s MMArt failure comes from one cell, and validation already showed it.** Genre text to text is below plain Qwen at every seed, on validation and on test. It also stays below at all three read-outs and in both leak checks. Stage 1 recorded the validation numbers at 452e143e, before any held score.

| MMArt, genre t2t | c* mean | c* seeds 0 to 4 | Plain Qwen | Difference |
|---|---|---|---|---|
| Validation | 0.4628 | 0.4535, 0.4634, 0.4740, 0.4597, 0.4632 | 0.4778 | -0.0150 |
| Test | 0.4564 | 0.4479, 0.4568, 0.4671, 0.4545, 0.4556 | 0.4701 | -0.0137 |

**The spec's brief flagged emotion's thin margin, and genre decided.** The spec had only text to image on MMArt, where emotion's margin was +0.0082. Emotion is a WIN in all three directions. Macro mAP has the opposite sign in the deciding cell: c* 0.3921, plain Qwen 0.3901, a mean difference with no interval.

| MMArt, genre t2t | Raw | Centering | Picked | No own painting | No label words |
|---|---|---|---|---|---|
| c* | 0.4151 | 0.4338 | 0.4564 | 0.4564 | 0.4477 |
| plain Qwen | 0.4237 | 0.4419 | 0.4701 | 0.4701 | 0.4620 |

**In genre text to text, c* is also below plain Qwen on three of the four other MMArt texts.** These are formal, historical and unified text; on emotional text it is above. They have no bar and cannot replace the narrative verdict. The round log has the values.

**Late fusion's MMArt failure is the teacher count.** It has 0 WINs against the 3 that "qualified" needs, and validation also gave 0 of 9 above the teacher's text.

**Late fusion's ArtELingo count depends on the read-out.** The table counts cells above each comparator, with both sides at one fixed read-out. These are mean differences with no intervals, so they replace no WIN count. Late fusion is above the teacher's text at all three read-outs in only 2 cells: style image to text and style text to text. c* keeps 6 at each read-out.

| Set | Row | Raw: plain/teacher | Centering | Query expansion |
|---|---|---|---|---|
| ArtELingo | c* | 9/6 | 9/6 | 9/6 |
| ArtELingo | late fusion | 9/4 | 9/2 | 9/5 |
| MMArt | c* | 8/3 | 8/3 | 8/4 |
| MMArt | late fusion | 9/0 | 9/0 | 9/0 |

**Our reading of weight 0.7: late fusion's ranking follows plain Qwen more than c*.** The plain term's spread is about 2 times the c* term's in most cells. Its median ratio is 1.98 on ArtELingo and 2.19 on MMArt; within one query it is 1.63 and 2.10. The weight alone gives 2.33, as the plain cosine's raw spread is a little smaller than c*'s. This fits two results. Late fusion's gain over plain Qwen is below c*'s in all 9 ArtELingo cells, and its MMArt genre text to text stays above plain Qwen.

| Held set | Spread ratio | Rank correlation of the two scores |
|---|---|---|
| ArtELingo | 1.01 to 2.63 | 0.16 to 0.56 |
| MMArt | 1.45 to 3.63 | 0.34 to 0.71 |

**Late fusion keeps most plain Qwen hits.** On ArtELingo text to image, late fusion hits 0.81 of the queries that only plain Qwen finds. The round log has the full split.

**Pair R@10 at weight 0.7 is 0.0017 to 0.0124 below the grid maximum.** The maximum lies at 0.8 to 1.0 (validation, seeds 1 to 4).

**Three of the four outcomes were the most likely verdict in our priors.** Late fusion's "holds" on ArtELingo had the lowest of its three priors. The written surprise, a c* ArtELingo cell with no WIN over plain Qwen, did not occur.

| Verdict | Holds | Qualified | Fails | Outcome |
|---|---|---|---|---|
| c*, ArtELingo | 0.60 | 0.32 | 0.08 | holds |
| c*, MMArt | 0.30 | 0.25 | 0.45 | fails |
| Late fusion, ArtELingo | 0.20 | 0.40 | 0.40 | holds |
| Late fusion, MMArt | 0.15 | 0.25 | 0.60 | fails |

**The spec's other risks leave the verdicts in force.** Means fitted on validation: raw and centering rows agree for c* and differ for late fusion (4 against 2 on ArtELingo). At the picks, removal of the own painting or of label-naming text changes no count of mean differences. On macro mAP the counts (plain Qwen/teacher's text) are c* 9/7 and late fusion 9/6 on ArtELingo, c* 9/4 and late fusion 9/3 on MMArt. Truncation affects 1 RoBERTa text and 0 others.

**Side rows have no bar.** The free variant, CLIP L/14, SigLIP 2, SigLIP and RoBERTa are in the round log. In text to image, c* is above all four non-RoBERTa rows on all three aspects.

## Deviations

- **Amendment (user, 2026-11-06 12:49, 60b22ac2):** the 19:00 wait was lifted. Only job timing changed; rules, rows, picks and bars stayed fixed.
- **Gate (i) stopped once, at 13:45:** one diagnostic-only store gave 0.99847019 against 0.999, on 1 of 200 rows, as length-sorted batches padded differently. Commit cbe5afce repeats the store's own batches: minimum 0.99999988, and the rerun passes 39 of 39. The bar, ticket 03's as in r15, is unchanged.
- **Chain restarts:** a context cut, the lifted wait, and one failed start with no job. No result depends on them.
- **C13:** Codex wrote it and fixed it in one round. It uses the run's numerical path and shares four input readers with the run, so a fault there would be shared; joins, caption labels and logic are separate.
- **Measured cost:** extraction 95.67 minutes against about 1.5 h; scoring 58.03 minutes; `decide` 55.1 seconds against about 1 h of CPU; all jobs under an hour. C13's cache (59 GB) far exceeded the output estimate.
- **Build notes and weak tests:** in the round log.
- **Exploration:** unrequested `label_word_share`, the within-query spread and `tcl_review_pair_estimate.py` (off the run path). Added in this report from saved outputs: validation per-seed values of the deciding cell and counts of mean differences. None changes a verdict.
- **User-read path:** `docs/user_read/2026-11-06_text_claim.md`, the real date in place of the planned one.

## Files and storage

- **Outputs:** `/data/SSD2/cosir_r23/20261106_text_claim` (79 GB, of which `c13` 59 GB); MMArt stores 4.3 GB. Records under `src/test/20261106_text_claim/records/`: `verdicts.json`, `c13.json`, `stage1_art.json`, `stage1_mm.json`, `status.json`.
- **Storage:** left on disk and listed on `.scratch/pending_deletions.md`; the decision is the user's at the wrap-up.
