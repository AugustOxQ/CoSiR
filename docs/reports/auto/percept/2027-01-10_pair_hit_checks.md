# Pair-hit checks: the own-caption term raises pair hits, and joined scores pass the rule at one weight

> 2026-11-05 (sequence name 2027-01-10) · Spec: none (short checks after the r22 read) · Decision rule: [round log, "Follow-up"](../../../../src/test/20261105_text_retrieval/20261105_text_retrieval_log.md) @ 86fd9711 · Run log: same file

## Results brief

**What we tried.** We ran two short checks on pair hits, a painting and its own caption.
Both used 6,152 validation paintings, 30,872 captions and seed 0, with no training and no held rows.
The first check scored two saved models that differ in the own-caption term: recipe R0-5 (term on) and recipe R6-5 (term off).
The second check joined c* and plain Qwen scores, with a weight a on plain Qwen from 0 to 1.

**Outcome against the bar.** Both checks passed their rules.

The own-caption term: supported.
The rise is above 0 and at least half of the gap to plain Qwen in text to image and image to text.

| Best-head pair R@10 | Term on | Term off | Difference [95% interval] | Share of the gap |
|---|---|---|---|---|
| text to image | 0.2538 | 0.1724 | +0.0814 [+0.0773, +0.0858] | 59% |
| image to text | 0.4161 | 0.2947 | +0.1214 [+0.1110, +0.1338] | 57% |
| text to text | 0.1748 | 0.1450 | +0.0298 [+0.0263, +0.0332] | 80% |

The term cost label hits in all 9 cells, in best-read-out label mAP:

- emotion: 0.0155 to 0.0329 lower
- style: 0.0006 to 0.0047 lower
- genre: 0.0288 to 0.0410 lower

Joined scores: one weight passed, a = 0.7.
The bar has two parts: pair R@10 within 0.02 of plain Qwen in text to image and image to text, and 5 or more of 9 label cells above both comparators.

| Weight a on plain Qwen | Label cells above both | Text to image | Image to text | Text to text |
|---|---|---|---|---|
| 0 (c* alone) | 6 | 0.2118 | 0.3186 | 0.1756 |
| 0.5 | 5 | 0.3061 | 0.4623 | 0.1977 |
| 0.7 | 5 | 0.3215 | 0.4954 | 0.1952 |
| 0.8 | 2 | 0.3221 | 0.5088 | 0.1935 |
| 1 (plain Qwen alone) | 0 | 0.3102 | 0.5068 | 0.1823 |

Under the rule, joined scores at a = 0.7 enter the claim loop as one extra row.

**What it does not show.**
This is one seed on validation, so it supports no claim.
The weight and the read-outs were picked on the same rows that scored them.
Only one weight on the grid passed: a = 0.6 misses the pair bar (0.4833 against 0.4868), and a = 0.8 falls to 2 of 9 cells.
Joined label cells use mean scores, without intervals.
R0-5 and R6-5 are older than c*: a sample of 19,889 paintings, earlier graphs, no genre relational term.
So the term's effect on c* itself is not known.
The check does not cover the term's cost in clustering.

**Options for next.**

1. Run the claim loop with c* as the main row and joined scores at a = 0.7 as the extra row.
   The rule calls for this. The joined row needs no new encoder pass, because plain Qwen is c*'s input.
2. Run the claim loop with c* alone and keep the joined curve as a note.
   This costs nothing and leaves pair hits below plain Qwen.
3. Retrain c* with the own-caption term.
   This takes hours by our estimate. It could give one model with better pair hits.
   It would change the model that every earlier result rests on.

**Your reading.**

<details>
<summary>Our reading and recommendation (open after yours)</summary>

**Our reading.** The user's explanation holds at sample scale: the own-caption term recovers more than half of the gap to plain Qwen.
The term is a trade-off, because it moves the model from aspect matching toward pair matching at a cost in label hits.
Joined scores reach most of both sides without a retrain.
At a = 0.7 they stand above plain Qwen in text to image and text to text, and 0.0114 below it in image to text.
They keep 5 of 9 label cells, against c*'s 6.
The margin is thin: one grid point passes, and style image to text clears the teacher by only 0.0018.
This gives the other authors a third choice on the pair-hit limit: state it, leave it out, or state it with joined scores as the answer.

**Our recommendation.** We recommend option 1. We would keep c* frozen, because a retrain puts every earlier result at risk 11 days before the paper deadline.

</details>

## What ran

- **Stages and types:** two short checks, type exploration, inside the r23 decide chat.
- **Data and seeds:** ArtELingo validation, seed 0. No held rows.
- **Code:** `tx_followup.py`, local GPU, 5 minutes, queue label `r23:short_checks`.

## Results

| ID | Measured | Number | Compared against | Bar | Outcome |
|---|---|---|---|---|---|
| Own term, text to image | best-head pair R@10, on minus off | +0.0814 [+0.0773, +0.0858] | gap to plain Qwen, 0.1379 | interval above 0, rise at least half of the gap | pass |
| Own term, image to text | best-head pair R@10, on minus off | +0.1214 [+0.1110, +0.1338] | gap to plain Qwen, 0.2121 | interval above 0, rise at least half of the gap | pass |
| Joined, a = 0.7 | pair R@10, text to image | 0.3215 | plain Qwen 0.3102 | within 0.02 | pass |
| Joined, a = 0.7 | pair R@10, image to text | 0.4954 | plain Qwen 0.5068 | within 0.02 | pass |
| Joined, a = 0.7 | label cells above both comparators | 5 of 9 | bar 5 | 5 or more | pass |

**Verdict under the rule:** the own-caption explanation is supported. Joined scores enter the claim loop as one extra row, weight a = 0.7.

Label mAP of joined scores, with Table 1's comparators of the r22 report:

| Cell | a = 0 | a = 0.7 | Plain Qwen | Teacher | a = 0.7 above both |
|---|---|---|---|---|---|
| emotion, text to image | 0.2650 | 0.2435 | 0.2236 | 0.2714 | no |
| emotion, image to text | 0.2898 | 0.2759 | 0.2407 | 0.3036 | no |
| emotion, text to text | 0.3455 | 0.3252 | 0.2541 | 0.3790 | no |
| style, text to image | 0.1720 | 0.1645 | 0.1342 | 0.1578 | yes |
| style, image to text | 0.1359 | 0.1237 | 0.1079 | 0.1219 | yes |
| style, text to text | 0.1072 | 0.1008 | 0.0940 | 0.0953 | yes |
| genre, text to image | 0.4877 | 0.4679 | 0.4179 | 0.4470 | yes |
| genre, image to text | 0.3989 | 0.3758 | 0.3390 | 0.3932 | no |
| genre, text to text | 0.3073 | 0.2992 | 0.2669 | 0.2951 | yes |

## Analysis

Our reading: at some weights, joined pair hits exceed both parts in all three directions.
Text to image peaks at a = 0.8 (0.3221 against plain Qwen's 0.3102).
Text to text peaks at a = 0.5 (0.1977 against 0.1823).
Image to text peaks at a = 0.9 (0.5125 against 0.5068).
The genre head leads pair hits at every joined weight.

A small weight also raises some label cells: style text to image goes from 0.1720 at a = 0 to 0.1775 at a = 0.3.
From a = 0.8 the count falls fast: 2 of 9 cells at a = 0.8, 1 at a = 0.9.

## Deviations

None from the rule. The rule's limits are agent defaults. The code had no separate review.
Its gates compare a = 0 and a = 1 with the saved r22 scores for every cell and read-out; the largest difference was 3.1e-6.
No painting of the recipes' training sample is a validation painting (0 of 6,152).

## Files and storage

- **Outputs:** [`results/followup.json`](../../../../src/test/20261105_text_retrieval/results/followup.json), code `tx_followup.py`, log `/data/SSD2/cosir_r22/20261105_text_retrieval/logs/followup.log`.
- **Storage:** nothing new over 1 MB.
