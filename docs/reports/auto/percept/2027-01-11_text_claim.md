# Text claim: both ArtELingo rows hold; both MMArt rows fail

> 2026-11-06 (sequence name 2027-01-11) · Spec: [design](../../../superpowers/specs/2026-11-06-text-claim-design.md) · Decision rule: §3.4, §3.5, §6 @ 452e143e · Run log: [full detail](../../../../src/test/20261106_text_claim/20261106_text_claim_log.md)

## Results brief

**What we tried.** We tested frozen c* and late fusion on held paintings under the [spec](../../../superpowers/specs/2026-11-06-text-claim-design.md).

**Outcome against the bar.** P/T count WIN over plain Qwen/the teacher’s text. Q requires both pair lower bounds above -0.02.

Holds requires P=9 and T≥5; qualified requires P=9 and T=3 or 4. Late fusion also requires Q.

| Row/set | Verdict | P/T/Q |
|---|---|---|
| SC1 | holds | 9/6/None |
| SC2 | fails | 8/3/None |
| SC3 | holds | 9/5/True |
| SC4 | fails | 9/0/True |

MMArt c* misses WIN against plain Qwen on genre text to text; late fusion misses the teacher count. Deciding intervals follow.

**What it does not show.** Claims cover these sets and frozen validation picks. They exclude other encoders and human MMArt text.

**Options for next.** Test the shared 11,528 MMArt rows (spec §2): extraction and scoring cost; this could locate the transfer failure. Collect human MMArt text: annotation and scoring cost; this could test dependence on generated text.

**Your reading.**

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** TODO-CONTROLLER

**Our recommendation.** TODO-CONTROLLER

</details>

## What ran

Stages: regression checks, validation sweep, held claim test, descriptive diagnostics. Local GPU; seeds and validation picks committed before held scores at 452e143e. C13: 3c435bb9; verdicts: dcb43313.

## Results

| ID | Row / set | P / T / Q | Verdict |
|---|---|---|---|
| SC1 | c* / ArtELingo | 9 / 6 / none | holds |
| SC2 | c* / MMArt | 8 / 3 / none | fails |
| SC3 | late fusion / ArtELingo | 9 / 5 / true | holds |
| SC4 | late fusion / MMArt | 9 / 0 / true | fails |

Label differences are mAP; intervals pair paintings and seeds. Read-outs: R raw, C validation centering, K centering plus expansion. Directions: t2i text to image, i2t image to text, t2t text to text. [Absolute label mAP](../../../../src/test/20261106_text_claim/20261106_text_claim_log.md#all-bar-cells-and-absolute-label-map).

### SC1

| Cell | Against plain Qwen | Against teacher | Range | Read-outs C/P/T | Paintings/queries |
|---|---|---|---|---|---|
| emotion/t2i | +0.0417 [+0.0411, +0.0424] WIN | -0.0036 [-0.0042, -0.0031] BELOW | 0.0023 | K/C/K | 12281/61744 |
| emotion/i2t | +0.0473 [+0.0454, +0.0491] WIN | -0.0166 [-0.0187, -0.0145] BELOW | 0.0016 | K/K/K | 12281/12281 |
| emotion/t2t | +0.0923 [+0.0904, +0.0939] WIN | -0.0308 [-0.0325, -0.0294] BELOW | 0.0044 | K/K/K | 12281/61744 |
| style/t2i | +0.0381 [+0.0368, +0.0394] WIN | +0.0159 [+0.0146, +0.0172] WIN | 0.0018 | K/K/K | 12281/61744 |
| style/i2t | +0.0284 [+0.0278, +0.0291] WIN | +0.0137 [+0.0132, +0.0142] WIN | 0.0011 | K/C/C | 12281/12281 |
| style/t2t | +0.0127 [+0.0121, +0.0133] WIN | +0.0113 [+0.0107, +0.0119] WIN | 0.0014 | K/K/K | 12281/61744 |
| genre/t2i | +0.0657 [+0.0638, +0.0676] WIN | +0.0403 [+0.0379, +0.0426] WIN | 0.0038 | K/K/K | 9949/49986 |
| genre/i2t | +0.0595 [+0.0570, +0.0622] WIN | +0.0062 [+0.0041, +0.0086] WIN | 0.0063 | C/K/C | 9949/9949 |
| genre/t2t | +0.0427 [+0.0383, +0.0475] WIN | +0.0162 [+0.0118, +0.0211] WIN | 0.0135 | K/K/K | 9949/49986 |

### SC2

| Cell | Against plain Qwen | Against teacher | Range | Read-outs C/P/T | Paintings/queries |
|---|---|---|---|---|---|
| emotion/t2i | +0.0086 [+0.0074, +0.0097] WIN | -0.0021 [-0.0033, -0.0009] BELOW | 0.0029 | R/R/R | 16218/16218 |
| emotion/i2t | +0.0057 [+0.0053, +0.0060] WIN | -0.0028 [-0.0032, -0.0024] BELOW | 0.0006 | K/C/K | 16218/16218 |
| emotion/t2t | +0.0061 [+0.0056, +0.0065] WIN | -0.0012 [-0.0017, -0.0008] BELOW | 0.0010 | R/R/K | 16218/16218 |
| style/t2i | +0.0684 [+0.0659, +0.0709] WIN | +0.0149 [+0.0125, +0.0174] WIN | 0.0054 | K/K/K | 16384/16384 |
| style/i2t | +0.0496 [+0.0481, +0.0511] WIN | +0.0183 [+0.0172, +0.0195] WIN | 0.0043 | K/K/C | 16384/16384 |
| style/t2t | +0.0273 [+0.0257, +0.0289] WIN | +0.0171 [+0.0155, +0.0187] WIN | 0.0039 | K/K/K | 16384/16384 |
| genre/t2i | +0.0575 [+0.0549, +0.0601] WIN | +0.0022 [-0.0007, +0.0050] LEVEL | 0.0046 | K/K/K | 13420/13420 |
| genre/i2t | +0.0138 [+0.0092, +0.0194] WIN | -0.0324 [-0.0372, -0.0268] BELOW | 0.0189 | K/K/K | 13420/13420 |
| genre/t2t | -0.0137 [-0.0190, -0.0078] LEVEL | -0.0264 [-0.0320, -0.0203] BELOW | 0.0205 | K/K/K | 13420/13420 |

### SC3

| Cell | Against plain Qwen | Against teacher | Range | Read-outs C/P/T | Paintings/queries |
|---|---|---|---|---|---|
| emotion/t2i | +0.0199 [+0.0197, +0.0202] WIN | -0.0254 [-0.0259, -0.0250] BELOW | 0.0010 | C/C/K | 12281/61744 |
| emotion/i2t | +0.0366 [+0.0355, +0.0377] WIN | -0.0272 [-0.0289, -0.0255] BELOW | 0.0017 | K/K/K | 12281/12281 |
| emotion/t2t | +0.0717 [+0.0703, +0.0729] WIN | -0.0514 [-0.0527, -0.0502] BELOW | 0.0030 | K/K/K | 12281/61744 |
| style/t2i | +0.0299 [+0.0292, +0.0306] WIN | +0.0076 [+0.0066, +0.0087] WIN | 0.0008 | K/K/K | 12281/61744 |
| style/i2t | +0.0156 [+0.0153, +0.0159] WIN | +0.0009 [+0.0005, +0.0012] WIN | 0.0004 | C/C/C | 12281/12281 |
| style/t2t | +0.0066 [+0.0064, +0.0068] WIN | +0.0052 [+0.0050, +0.0055] WIN | 0.0003 | K/K/K | 12281/61744 |
| genre/t2i | +0.0482 [+0.0471, +0.0492] WIN | +0.0227 [+0.0206, +0.0248] WIN | 0.0015 | K/K/K | 9949/49986 |
| genre/i2t | +0.0385 [+0.0365, +0.0409] WIN | -0.0148 [-0.0171, -0.0122] BELOW | 0.0068 | K/K/C | 9949/9949 |
| genre/t2t | +0.0333 [+0.0314, +0.0354] WIN | +0.0068 [+0.0048, +0.0090] WIN | 0.0061 | K/K/K | 9949/49986 |

### SC4

| Cell | Against plain Qwen | Against teacher | Range | Read-outs C/P/T | Paintings/queries |
|---|---|---|---|---|---|
| emotion/t2i | +0.0051 [+0.0047, +0.0055] WIN | -0.0056 [-0.0062, -0.0050] BELOW | 0.0010 | R/R/R | 16218/16218 |
| emotion/i2t | +0.0036 [+0.0034, +0.0038] WIN | -0.0049 [-0.0052, -0.0045] BELOW | 0.0004 | K/C/K | 16218/16218 |
| emotion/t2t | +0.0038 [+0.0036, +0.0040] WIN | -0.0035 [-0.0038, -0.0031] BELOW | 0.0004 | R/R/K | 16218/16218 |
| style/t2i | +0.0523 [+0.0512, +0.0535] WIN | -0.0011 [-0.0026, +0.0004] BELOW | 0.0014 | K/K/K | 16384/16384 |
| style/i2t | +0.0189 [+0.0183, +0.0195] WIN | -0.0124 [-0.0131, -0.0117] BELOW | 0.0007 | C/K/C | 16384/16384 |
| style/t2t | +0.0086 [+0.0084, +0.0088] WIN | -0.0016 [-0.0023, -0.0009] BELOW | 0.0004 | K/K/K | 16384/16384 |
| genre/t2i | +0.0458 [+0.0447, +0.0469] WIN | -0.0096 [-0.0120, -0.0072] BELOW | 0.0019 | K/K/K | 13420/13420 |
| genre/i2t | +0.0153 [+0.0135, +0.0173] WIN | -0.0310 [-0.0336, -0.0283] BELOW | 0.0066 | K/K/K | 13420/13420 |
| genre/t2t | +0.0083 [+0.0063, +0.0104] WIN | -0.0044 [-0.0072, -0.0015] BELOW | 0.0069 | K/K/K | 13420/13420 |

Late fusion pair bar: lower bound above -0.02.

| ID | Direction | Late fusion − plain Qwen pair R@10 | Read-outs | Paintings/queries | Q cell |
|---|---|---|---|---|---|
| SC3 | t2i | +0.0029 [+0.0010, +0.0049] | C/C | 12281/61744 | true |
| SC3 | i2t | -0.0030 [-0.0079, +0.0019] | C/C | 12281/12281 | true |
| SC4 | t2i | -0.0114 [-0.0143, -0.0085] | C/C | 16384/16384 | true |
| SC4 | i2t | -0.0084 [-0.0108, -0.0061] | C/C | 16384/16384 | true |

Pair metrics at frozen pair picks.

| Set | Row | Head | Direction | Read-out | R@1 | R@5 | R@10 | Median rank |
|---|---|---|---|---|---|---|---|---|
| art_held | c* | genre | t2i | C | 0.0385 | 0.1053 | 0.1517 | 174.8000 |
| art_held | c* | genre | i2t | C | 0.0573 | 0.1615 | 0.2309 | 61.4000 |
| art_held | plain Qwen | all | t2i | C | 0.0905 | 0.1927 | 0.2512 | 94.0000 |
| art_held | plain Qwen | all | i2t | C | 0.1692 | 0.3289 | 0.4136 | 19.0000 |
| mm_test | c* | emotion | i2t | C | 0.0856 | 0.2107 | 0.2883 | 43.6000 |
| mm_test | c* | style | t2i | C | 0.0075 | 0.0310 | 0.0518 | 492.2000 |
| mm_test | plain Qwen | all | t2i | C | 0.6149 | 0.7936 | 0.8493 | 1.0000 |
| mm_test | plain Qwen | all | i2t | C | 0.6718 | 0.8530 | 0.9006 | 1.0000 |

**Verdict under the rule:** SC1 holds; SC2 fails; SC3 holds; SC4 fails. A failed verdict stands.

SC1 licenses: "On 12,281 held ArtELingo paintings, c* (five seeds, read-outs picked on validation) beats plain Qwen in label retrieval in all nine aspect and direction cells, and beats the teacher's text in 6 of them, with one encoder pass and a small head per aspect, where the teacher's text needs an encoder pass per aspect."

SC3 licenses: "A weighted sum of c* and plain Qwen cosines (a = 0.7) beats plain Qwen in label retrieval in all nine cells, and beats the teacher's text in 5 of them. Its pair R@10 is not more than 0.02 below plain Qwen's in text to image and image to text, at the interval's lower bound."

MMArt teacher claims stand despite failed set verdicts: c* style in every direction, “better than the teacher’s text”; genre text to image, “level with or better than”. Other MMArt teacher cells license nothing. Only WIN counts against plain Qwen.

## Analysis

MMArt c* fails on genre text to text: -0.0137 [-0.0190, -0.0078] against plain Qwen. Its lower bound misses zero by 0.0190. Genre decides, despite the spec’s emotion warning. 

MMArt late fusion fails: T=0 against at least three, despite P=9 and Q. Style text to text is closest to teacher WIN: lower bound -0.0023 against zero.

The deciding cell’s validation range is 0.0205, held spread 0.0192; seed values 0.4479, 0.4568, 0.4671, 0.4545, 0.4556.

The fifth ArtELingo late fusion teacher WIN is style image to text: +0.0009 [+0.0005, +0.0012]. C13's closest cell is SC4 / label / late / teacher / genre / t2t, lower -0.0072451194 against -0.0068928519, distance 0.0003522675. This is the LEVEL boundary.

Late fusion ArtELingo holds despite its lower prior. The other rows take their most likely verdict.

| Verdict | Holds | Qualified | Fails |
|---|---|---|---|
| c*, ArtELingo | 0.60 | 0.32 | 0.08 |
| c*, MMArt | 0.30 | 0.25 | 0.45 |
| Late fusion, ArtELingo | 0.20 | 0.40 | 0.40 |
| Late fusion, MMArt | 0.15 | 0.25 | 0.60 |

The surprise, an ArtELingo c* cell without WIN over plain Qwen, did not occur.

Our reading: late fusion retains hits from different query groups. The split describes the frozen weight.

| Set/direction | Both share/rate | Only c* share/rate | Only plain share/rate | Neither share/rate |
|---|---|---|---|---|
| art_held/t2i | 0.1139/0.9999 | 0.0379/0.4407 | 0.1373/0.8124 | 0.7109/0.0169 |
| mm_test/t2i | 0.0486/1.0000 | 0.0032/0.5009 | 0.8007/0.9681 | 0.1475/0.0850 |


Spread ratios and rank correlations on held rows:

| Set | Spread-ratio range | Spearman-mean range |
|---|---|---|
| art_held | 1.0066 to 2.6252 | 0.1553 to 0.5572 |
| mm_test | 1.4508 to 3.6345 | 0.3365 to 0.7098 |

The curve uses seeds 1 to 4; the verdict weight stays fixed.

| Set/direction | At frozen weight | Maximum on grid | Weight at maximum |
|---|---|---|---|
| art_val/i2t | 0.4990 | 0.5114 | 0.9 |
| art_val/t2i | 0.3189 | 0.3206 | 0.8 |
| mm_val/i2t | 0.9464 | 0.9509 | 1.0 |
| mm_val/t2i | 0.9083 | 0.9151 | 0.9 |

Fixed read-outs change positive mean-difference counts, shown as descriptive P/T. No saved intervals support replacement WIN counts.

| Set | Row | Raw P/T | Center P/T | Expansion P/T |
|---|---|---|---|---|
| art_held | c* | 9/6 | 9/6 | 9/6 |
| art_held | late fusion | 9/4 | 9/2 | 9/5 |
| mm_test | c* | 8/3 | 8/3 | 8/4 |
| mm_test | late fusion | 9/0 | 9/0 | 9/0 |

Risk checks leave verdicts in force. Genre text-to-text examples follow; full macro, leak and class tables are in the round log.

| Genre t2t | Row | Raw | Center | Picked | No own painting | No label words |
|---|---|---|---|---|---|---|
| art_held | c* | 0.2786 | 0.2917 | 0.3152 | 0.3149 | 0.3039 |
| mm_test | c* | 0.4151 | 0.4338 | 0.4564 | 0.4564 | 0.4477 |

Truncation affects 1 RoBERTa text and 0 other checked texts. MMArt uses generated text and different emotion labels. Token limits and seed spreads are in the log.

ArtELingo side rows use validation picks, without a bar.

| Row | Emotion t2i | Style t2i | Genre t2i |
|---|---|---|---|
| free variant | 0.2557 | 0.1462 | 0.4502 |
| CLIP L/14 | 0.2137 | 0.1498 | 0.4126 |
| SigLIP 2 | 0.2097 | 0.1489 | 0.4272 |
| SigLIP | 0.2110 | 0.1517 | 0.4188 |

RoBERTa emotion text to text: 0.3118.

Other MMArt texts have no bar and cannot replace narrative verdicts. Full rows are in the log; c* t2i mAP follows.

| Text | c* emotion/style/genre t2i |
|---|---|
| formal | 0.1924/0.2438/0.5405 |
| emotional | 0.2587/0.2422/0.5592 |
| historical | 0.1857/0.3623/0.4245 |
| unified | 0.2157/0.3321/0.5731 |

## Deviations

Amendment, user, 2026-11-06 12:49 (60b22ac2): lifted the 19:00 wait. Friday work between 10:00 and 19:00 followed the user's word. Only job timing changed; rules, rows, picks and bars stayed fixed.

Gate (i) stopped at 13:45: diagnostic-only `mm_val/formal/genre_word/teacher`, 0.99847019 versus 0.999. Shortfall 0.00052981; 1 of 200 rows. Length-sorted batches padded differently. Commit cbe5afce repeats store batches, minimum 0.99999988; rerun passes 39/39. Bar unchanged: ticket 03 as r15; §3.5 names none. Full trace is in the log.

Chain restarts: context cut, lifted wait, failed start without a job. Results do not depend on these stops.

C13 passed with 0 failures. Codex wrote it and fixed it in one round. Separate logic uses the run's numerical path: no numerical diversity, because matrix shapes affect near-tied ranks.

C13 shares four input readers: `rt_data.names`, `rt_data.labels`, `mt_data.keys`, `mt_data.labels`. Their faults would be shared; joins, caption labels and logic are separate.

C13 `pass` can retain differences when every affected SC is voided, recording `void_differences`. Both lists are empty here.

Measured extraction: 95.67 minutes, against about 1.5 h. Scoring: 58.03 minutes; `decide` takes 55.1 seconds against the CPU estimate of about 1 h. Wall times only. Longest step: `score_art_held`, 17.99 minutes; all logged jobs are under an hour.

| C13 measured chain interval | Minutes |
|---|---|
| c13 compare val | 2.0 |
| c13 derive held | 11.0 |
| c13 compare all | 10.0 |

`c13_void` also withholds cell claims. RoBERTa is normalised on read. A bad non-void score file stops `decide`. Plain Qwen LEVEL and BELOW words make no claim. Ticket 01 has no review trace; the independent numbers review stands in. C13's cache exceeded the output estimate. Weak tests remain disclosed. Full held-size C13 checks now completed, as its categories table shows.

Exploration: unrequested `label_word_share`, within-query spread and `tcl_review_pair_estimate.py` (off the run path). None changes a verdict.

User-read path: `docs/user_read/2026-11-06_text_claim.md`, the real date rather than the planned date.

## Files and storage

Outputs: `/data/SSD2/cosir_r23/20261106_text_claim` (79G); `c13` (59G); MMArt stores (4.3G). Records under `src/test/20261106_text_claim/records/`: `verdicts.json`, `c13.json`, `stage1_art.json`, `stage1_mm.json`, `status.json`.

Storage: left on disk and listed on .scratch/pending_deletions.md for the next storage summary; the decision is the user's at the wrap-up.
