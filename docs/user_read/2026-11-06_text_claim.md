# c* and late fusion hold on ArtELingo and fail on MMArt

> 2026-11-06 · [Full report](../reports/auto/percept/2027-01-11_text_claim.md)

## Summary

- c*, our frozen student model, gets the verdict "holds" on held ArtELingo paintings, which shaped nothing.
- c* gets "fails" on MMArt, paintings outside ArtELingo with generated narrative text.
- Late fusion is a weighted sum of c* scores and plain Qwen scores. Plain Qwen is Qwen without aspect instructions.
- Late fusion also gets "holds" on ArtELingo and "fails" on MMArt.
- These verdicts use five seeds and read-outs picked on validation. A read-out turns model vectors into retrieval scores.

## How we got here

c* was picked and frozen before this loop.
An earlier exploration, on validation with one seed, had c*'s caption side above plain Qwen and the teacher's text in 6 of 9 cells.
The teacher is the aspect-instructed Qwen model that c* learns from.
The teacher's text is Qwen text embedded under each aspect's instruction.
A short check then put late fusion's pair hits near plain Qwen's. A pair hit is a painting's own caption in the first results.
This loop tests both on data that shaped nothing, for the paper.

## c* on held data

**Problem.** The earlier result used one seed on validation, so the paper could not claim it.

**Idea.** We froze c* and tested its caption side on data that shaped nothing.
The first set is 12,281 held ArtELingo paintings.
The second set is 16,384 MMArt paintings outside ArtELingo, with generated narrative text.

We compare c* with plain Qwen and with the teacher's text.
A cell is one aspect (emotion, style, genre) in one direction (text to image, image to text, text to text).
Each set has 9 cells.
Label mAP scores each cell: it measures how well retrieval ranks results with the same aspect label.
Read-outs were picked on validation before any held score.

**Did it work?** c* holds on ArtELingo and fails on MMArt.
The verdict words come from a rule written before the run.
A WIN means the 95% interval of the difference has its lower bound above 0.

- Holds: a WIN over plain Qwen in all 9 cells, and over the teacher's text in 5 or more.
- Qualified: a WIN over plain Qwen in all 9 cells, and over the teacher's text in 3 or 4.
- Fails: anything else.

On MMArt, c* misses the WIN over plain Qwen in one cell, genre text to text, and this cell decides the verdict.
A failed verdict stands.
c*'s claims cover label retrieval. Its pair numbers stand in a table in the full report, with no reading.

**Key evidence.** Counts are WINs out of 9 cells.

| Set | WINs over plain Qwen | WINs over the teacher's text | Verdict |
|---|---:|---:|---|
| ArtELingo | 9 of 9 | 6 of 9 | holds |
| MMArt | 8 of 9 | 3 of 9 | fails |

The deciding cell, in label mAP:

| Cell | c* | Plain Qwen | Difference [95% interval] |
|---|---:|---:|---|
| MMArt, genre text to text | 0.4564 | 0.4701 | -0.0137 [-0.0190, -0.0078] |

All five seeds put this cell below plain Qwen. Validation and all three read-outs agree.
We do not know the cause of the miss. Two facts may bear on it, and we tested neither:

- A Qwen model generated the narrative text, and this may favour Qwen rows.
- MMArt's emotion labels agree with ArtELingo's on 43% of shared paintings.

On both sets, all 3 emotion cells are below the teacher's text.
Each cell against the teacher's text is also its own claim, whatever the verdict.
On MMArt, c* is better than the teacher's text on style in all three directions.
On MMArt genre text to image, c* is level with or better than the teacher's text.

> [Full report: Results](../reports/auto/percept/2027-01-11_text_claim.md#results)

## Late fusion

**Problem.** Late fusion came from a short check, so the paper could not claim it yet.

**Idea.** Late fusion is a weighted sum of the c* and plain Qwen cosines, with weight 0.7 on plain Qwen.
We fixed this weight before the run.
Late fusion faces the same label rule as c* on each set.
It also has a pair bar.
Pair R@10 is the share of queries with the matching painting or caption in the first ten results.
Late fusion's pair R@10 may be at most 0.02 below plain Qwen, at the interval's lower bound.
This bar applies in text to image and image to text.

Read-outs were picked on validation before any held score.
The diagnostics describe late fusion at this weight and test no cause.

**Did it work?** Late fusion holds on ArtELingo and fails on MMArt.
It passes the pair bar on both sets.
On MMArt it has no WIN over the teacher's text, against the 3 that "qualified" needs.
It also has no claim of its own on any MMArt cell.
Pair R@10 on MMArt is near its ceiling, so the pair bar says less there than on ArtELingo.

**Key evidence.** Counts are WINs out of 9 cells.

| Set | WINs over plain Qwen | WINs over the teacher's text | Pair bar | Verdict |
|---|---:|---:|---|---|
| ArtELingo | 9 of 9 | 5 of 9 | passes | holds |
| MMArt | 9 of 9 | 0 of 9 | passes | fails |

The comparisons that decided this table:

| Comparison | Difference [95% interval] |
|---|---|
| ArtELingo, style image to text: the fifth WIN over the teacher's text | +0.0009 [+0.0005, +0.0012] |
| MMArt, style text to text: the highest lower bound against the teacher's text | -0.0016 [-0.0023, -0.0009] |
| MMArt, text to image: the lowest lower bound of pair R@10 against plain Qwen | -0.0114 [-0.0143, -0.0085] |

Late fusion's count over the teacher's text depends on the read-out.
We counted its ArtELingo cells above the teacher's text, with both sides at one fixed read-out:

- Raw cosine: 4 cells.
- Centering: 2 cells.
- Query expansion: 5 cells.

These counts are mean differences with no intervals, so they replace no WIN count.
c* keeps 6 at every read-out.

> [Full report: Analysis](../reports/auto/percept/2027-01-11_text_claim.md#analysis)

## Limits of both claims

- The claims cover Qwen on these two sets, with five seeds and read-outs picked on validation.
- They do not cover other encoders or human text.
- MMArt has no human text, and plain Qwen is its only plain encoder measured.
- MMArt's teacher's text uses ArtELingo's instructions.
- Intervals have no multiplicity correction.

<details>
<summary>Our advice (open after giving your reading)</summary>

**Our reading.** c* on ArtELingo is the firm result.
Every cell keeps its validation sign against both comparators, and the counts stay the same across read-outs and leak checks.
Its claim over the teacher's text is a style and genre claim.

The MMArt miss on genre text to text is real: five seeds, all three read-outs and validation agree.
Macro mAP has the opposite sign in this cell, which points weakly to frequent classes.
We do not know the cause.

Late fusion's "holds" is correct under the rule, and it is thin.
Its firm part is 18 of 18 WINs over plain Qwen, with the pair bar passed on both sets.
Its claim over the teacher's text is the weak part.
We are less sure of one more reading, as nothing tested it.
The plain term's score spread is about 2 times the c* term's, so late fusion's ranking follows plain Qwen more than c*.

**Our recommendation.** Option a, with option b beside it if you want a reason for the miss.
Make c* on ArtELingo the main claim.
Report MMArt per cell, with its failed verdict stated.
Make late fusion a second claim, and add "read-outs picked on validation" to its sentence, as c*'s has.
Option c can wait: it cannot change a verdict, and the target is close.

</details>

## Next

You give your reading first.
Then you choose among these options, for the target of 2026-11-10 and the paper on 2026-11-16.
The wording of c*'s pair numbers still waits for your talk with the other authors.

a. **Take the ArtELingo claims to the paper.** No GPU, one writing chat.
The paper states both holds, MMArt's failed verdicts and its cell claims. This shows nothing new.

b. **Read the missed cell per class.** Saved per-query scores, CPU, under 1 hour, a small loop.
This could show whether the miss sits in a few frequent genres.

c. **Test the 11,528 MMArt rows that are also held ArtELingo paintings.** They keep MMArt's text and labels.
Our cost estimate is about 10 GPU minutes for the narrative text, or about 1 hour for all five texts.
It also needs a short spec, a build chat and a run chat.
This could show whether the failure comes from the new paintings or from the text and labels. It cannot change a verdict.

## Glossary

- **c\*:** the frozen student model. It has one shared trunk with emotion, style and genre heads.
- **Plain Qwen:** Qwen without aspect instructions.
- **Teacher:** the aspect-instructed Qwen model that c* learns from.
- **Teacher's text:** Qwen text embedded under each aspect's instruction.
- **Late fusion:** the weighted sum of c* and plain Qwen cosines.
- **Cell:** one aspect in one direction.
- **Label mAP:** mean average precision, where a hit has the same aspect label.
- **Pair hit:** a painting's own caption in the first results.
- **Held data:** data that shaped nothing.
- **Read-out:** the method that turns model vectors into retrieval scores.
- **WIN:** the 95% interval of the difference has its lower bound above 0.
