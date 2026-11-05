# A missing training term explains weak pair hits, and joined scores recover them

> 2026-11-05 · [Full report](../reports/auto/percept/2027-01-10_pair_hit_checks.md)

## Summary

- We asked why c* finds a painting's own caption less often than plain Qwen.
- A saved model pair supports your explanation: the training term for own captions matters.
- That term raises pair hits and lowers label hits.
- Joining c* scores with plain Qwen scores recovers pair hits without a retrain.
- One weight passed our bar, so the margin is thin.
- These validation results use one seed, so they support no claim yet.

## How we got here

The last exploration showed c* above plain Qwen and the teacher on shared labels, and below plain Qwen on pair hits.
You said this fits a training that does not align a painting with its own caption.
We tested that, and we tested a cheap way to recover pair hits.

## The own-caption term

**Problem.** c* was trained without a term that pulls a painting to its own caption.
**Idea.** Two saved older models differ in this term, so we compared them on pair hits.
**Did it work?** Yes. The term recovers more than half of the gap to plain Qwen.
It also lowers label hits, mostly on emotion and genre.
These are older, smaller models, so the size of the effect on c* is not known.

**Key evidence.** The table shows pair recall among the first ten results, for each model's best head.

| Direction | Term on | Term off | Plain Qwen |
|---|---:|---:|---:|
| Text to image | 0.2538 | 0.1724 | 0.3102 |
| Image to text | 0.4161 | 0.2947 | 0.5068 |
| Text to text | 0.1748 | 0.1450 | 0.1823 |

> [Full report: Results](../reports/auto/percept/2027-01-10_pair_hit_checks.md#results)

## Joined scores

**Problem.** A retrain would change the model that all earlier results rest on.
**Idea.** c* already takes plain Qwen as input, so we can join both scores with a weight and run no new encoder.
**Did it work?** Yes, at one weight: 70% on plain Qwen.
There, pair hits are close to plain Qwen, and 5 of 9 label cells stay above both comparators.
The neighbouring weights fail, so the pass is narrow.

**Key evidence.** Each row is one weight on plain Qwen.

| Weight | Label cells above both | Text to image | Image to text | Text to text |
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

Under the rule, joined scores enter the claim loop as one extra row.
You read these results first, and then we grill the claim loop.
You still decide the pair-hit limit, after your talk with the other authors.

## Glossary

- **Pair hit:** a result that is the painting's own caption, or the caption's own painting.
- **Own-caption term:** a training term that pulls a painting to its own caption.
- **Joined score:** a weighted sum of the c* score and the plain Qwen score.
- **Label cell:** one aspect in one retrieval direction.
- **Head:** one output of c* for an aspect.
