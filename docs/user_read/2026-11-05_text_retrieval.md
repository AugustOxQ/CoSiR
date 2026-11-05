# c* finds shared labels; caption matching depends on direction

> 2026-11-05 · [Full report](../reports/auto/percept/2027-01-09_text_retrieval.md)

## Summary

- c* passed the exploration bar: six cells exceeded plain Qwen and the teacher, against a bar of five.
- Style and genre exceeded the teacher; emotion fell below it.
- Four cells held in every check; two depended on how we read the vectors.
- Known caption pairs were harder for c* than plain Qwen in text to image and image to text; text to text was close.
- These validation results use one seed, so they support no claim yet.

## How we got here

c* has separate outputs for emotion, style and genre.
We tested its caption side as a text encoder to find shared labels across captions and paintings.

## Captions and paintings

**Problem.** Label matches may still miss a painting's own caption.
**Idea.** We tested text to image, image to text and text to text, with label and known-pair hits.
We compared c* with plain Qwen, the aspect-instructed teacher, three plain encoders and an emotion text model.

**Did it work?** c* exceeded plain Qwen with intervals in all nine label cells.
Its mean score also exceeded all three plain encoders in all nine cells.
Emotion stayed below the teacher.
Painting emotion used caption plurality.
Read-out selection and scores used the same validation rows; a split-row check kept the count.
Some checks followed the first read.

**Key evidence.** The table shows best-head pair recall among the first ten results.

| Direction | c* | Plain Qwen | Teacher |
|---|---:|---:|---:|
| Text to image | 0.2118 | 0.3102 | 0.1604 |
| Image to text | 0.3186 | 0.5068 | 0.2466 |
| Text to text | 0.1756 | 0.1823 | 0.1261 |

c*'s caption side trained on caption means, so single captions and short words are new inputs.
The code and export gate checked this training input.
For style and genre label hits, and for pair hits, single captions scored below caption means.
For emotion label hits, the gap is within the chance difference.

> [Full report: Results](../reports/auto/percept/2027-01-09_text_retrieval.md#results)

## Other text inputs

**Problem.** Human captions may not predict other text results.
**Idea.** We tested SemArt, MMArt's generated perspectives and word queries.
**Did it work?** Word-query results were mixed.
**Key evidence.** For c*, MMArt's genre head led every perspective on pair hits; the matching heads did not.
Its emotional text often named its own label, which leaks the answer.
SemArt used validation; published rows used test and different text.
Lenses and CANVAS were not re-checked.

> [Full report: Analysis](../reports/auto/percept/2027-01-09_text_retrieval.md#analysis)

<details>
<summary>Our advice (open after giving your reading)</summary>

**Our reading.** c* works as an aspect retriever; as a caption matcher it is limited for text to image and image to text.

**Our recommendation.** We recommend the claim loop below, after joined c* and plain scores, which take minutes to test.
We would claim style and genre label hits and state emotion as below the teacher.

</details>

## Next

The rule calls for a medium claim loop on held ArtELingo and the MMArt test rows.
Our additions are five seeds and read-outs picked on validation.
Held ArtELingo teacher extraction covers 61,744 captions by 3 instructions.
At about 295 per second, our estimate is about 10.5 minutes; that rate comes from short captions.
MMArt needs 11,528 rows by five texts; we have no cost estimate because these texts are longer.
The loop tells which cells hold.
The handoff estimates a claim test read by about 10 November.
Caption-side retraining takes hours by our estimate; the rule did not trigger it.
You decide whether the paper states the pair-hit limit.
The free variant sends captions through the image side and exceeds both comparators in 3 of 9 cells.
That would give Outcome 2; the rule reads c*.
You decide how to treat that alternative.
The Lenses PDF and published MMArt numbers remain with you.

## Glossary

- A **head** is one output for an aspect.
- A **read-out** turns vectors into retrieval scores.
- **R@10** is the share of queries with a known pair among the first ten results.
