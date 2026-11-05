# On older models the own-caption term raises pair hits; joined scores recover them for c*

> 2026-11-05 · [Full report](../reports/auto/percept/2027-01-10_pair_hit_checks.md)

## Summary

- We asked why c* gets fewer pair hits than plain Qwen. A pair hit is a painting's own caption in the first ten results.
- Two saved models support your explanation: a training term that pulls a painting to its own caption raises pair hits.
- That term also lowers label hits, which are results with the same emotion, style or genre.
- Joining c* scores with plain Qwen scores recovers pair hits without a retrain.
- Only one weight passed our bar, so the margin is thin.
- These validation results use one seed, so the paper cannot claim them yet.

## How we got here

The last exploration showed c* above plain Qwen and the teacher in 6 of 9 label cells, and below plain Qwen on pair hits.
The teacher is the aspect-instructed Qwen model that c* learns from.
You said this fits c*'s training, which does not pull a painting to its own caption.
We tested that, and we tested a cheap way to recover pair hits.

## The own-caption term

**Problem.** c* was trained without a term that pulls a painting to its own caption.
**Idea.** Two saved models from before c* differ in this term, so we compared them on pair hits.
They have the same teacher share and the same size.
**Did it work?** Yes. With the term, pair hits rise by more than half of the distance to plain Qwen.
It also lowers label hits, mostly on emotion and genre.
These models trained on a smaller sample with earlier graphs, so the size of the effect on c* is not known.

**Key evidence.** The table shows R@10, the share of queries with a pair hit in the first ten results.
Each model shows its head with the highest R@10.

| Direction | Term on | Term off | Plain Qwen |
|---|---:|---:|---:|
| Text to image | 0.2538 | 0.1724 | 0.3102 |
| Image to text | 0.4161 | 0.2947 | 0.5068 |
| Text to text | 0.1748 | 0.1450 | 0.1823 |

> [Full report: Results](../reports/auto/percept/2027-01-10_pair_hit_checks.md#results)

## Joined scores

**Problem.** A retrain would change the model that all earlier results rest on.
**Idea.** c* already takes plain Qwen vectors as input, so a joined score needs no new encoder pass.
The joined score is (1 - a) times the c* cosine plus a times the plain Qwen cosine.
**Did it work?** Yes, at one weight: a = 0.7.
There, pair hits are close to plain Qwen, and 5 of 9 label cells stay above plain Qwen and the teacher.
We tried a from 0 to 1 in steps of 0.1. The weights 0.6 and 0.8 fail, so the pass is narrow.

**Key evidence.** Each row is one weight. The last three columns are pair R@10.

| Weight a | Label cells above both | Text to image | Image to text | Text to text |
|---:|---:|---:|---:|---:|
| 0 | 6 | 0.2118 | 0.3186 | 0.1756 |
| 0.7 | 5 | 0.3215 | 0.4954 | 0.1952 |
| 1 | 0 | 0.3102 | 0.5068 | 0.1823 |

> [Full report: Results](../reports/auto/percept/2027-01-10_pair_hit_checks.md#results)

<details>
<summary>Our advice (open after giving your reading)</summary>

**Our reading.** Your explanation holds on the older models.
Joined scores give the other authors a third choice: state the limit with joined scores as the answer.

**Our recommendation.** We recommend the claim loop with c* as the main row and joined scores as an extra row.
We would keep c* frozen and not retrain it before the deadline.

</details>

## Next

Under the rule written before these checks, joined scores enter the claim loop as one extra row.
The claim loop tests the results on held data with five seeds.
You read these results first, and then we grill the design of that loop.
You still decide the pair-hit limit, after your talk with the other authors.

## Glossary

- **Pair hit:** a result that is the painting's own caption, the caption's own painting, or another caption of the same painting.
- **Own-caption term:** a training term that pulls a painting to its own caption.
- **Joined score:** a weighted sum of the c* score and the plain Qwen score.
- **Label cell:** one aspect (emotion, style, genre) in one direction (text to image, image to text, text to text), so 9 cells.
- **Head:** one output of c* for an aspect.
