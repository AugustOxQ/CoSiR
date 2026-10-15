# AFF needs at least 3 example pairs; with 1 pair it is worse than the bar

> 2026-10-14 · Full report: [K-shot example pairs](../reports/auto/v2/2026-12-03_kshot_supports.md) (its file name uses
> CoSiR's sequence date).

## Summary

- We gave AFF fewer example pairs per episode: 1, 2 or 3 instead of 4. Nothing was re-trained or re-tuned.
- With 3 pairs, AFF keeps almost all of its lead over the bar. The change is within the noise.
- With 2 pairs, the lead falls from +0.70 to +0.19 R@1 points, and its interval includes zero.
- With 1 pair, AFF is 1.73 points below the bar. Its gate then opens in both conditions.
- AFF's reader and gate were fitted for 4 pairs. So each drop mixes less evidence with a reader that no longer fits.
- The lead comes from the two emotion pairs. On style × genre, AFF is below the bar at every number of pairs.
- These are development numbers on seed 42, where AFF's settings were picked.
- Split by the aspect the pairs show, the whole lead is in the emotion episodes. Fewer pairs mostly make AFF steer
  toward emotion when it should not (added 2026-10-14, after your question).

## How we got here

The project ranks captions for an image (or images for a caption) by one aspect, such as emotion.
The aspect is shown by examples. An **episode** is one such ranking: a query, example pairs and 13 candidates.
**Support pairs** show the aspect to use, **contrast pairs** show another aspect. Swapping the two flips the
**condition**, and the right answer changes with it.

AFF is our current best method. Its **reader** looks at the example pairs and estimates which aspect they show. If the
reader is sure the answer is emotion, AFF adds an emotion term to the score. Every episode so far had 4 support and 4
contrast pairs. On 2026-10-14 you asked for a small, cheap check in the spirit of k-shot learning: how does AFF do with
fewer pairs?

## Fewer example pairs

**Problem.** We did not know how many example pairs AFF needs. A user of the method may have only two or three.

**Idea.** We used the same development episodes and kept only the first 1, 2 or 3 pairs of each kind. All else
stayed the same. So each episode is compared with itself. With 1 pair, the reader's spread over the pairs has no
value. We set it so that it adds nothing to the reader's decision.

**Did it work.** With 3 pairs, AFF keeps almost all of its lead over the bar. With 2 pairs, the lead over the bar is no
longer clearly above zero. AFF still beats its matched control there, by about half as much as before. About two
thirds of AFF's drop is in the part that reads the condition (no interval for this split). With 1 pair, AFF is worse
than the bar and than its matched control.

Two caveats change what this means:

- The reader and the gate were tuned for 4 pairs. We cannot tell how much of the drop is less evidence and how much is
  a reader that no longer fits.
- The gate should open mostly when the supports show emotion. With 2 pairs it opens there less often (81% to 72%) and
  in the swapped condition more often (30% to 33%). With 1 pair it opens in 62% of the swapped condition, so it often
  steers the wrong way.

**Key evidence.** AFF's lead in R@1 points (the share of queries with the right answer ranked first), with 95%
intervals:

| Example pairs | AFF over the bar | AFF over its matched control |
|---|---|---|
| 4 (today) | +0.70 [+0.46, +0.94] | +0.74 [+0.52, +0.96] |
| 3 | +0.63 [+0.40, +0.87] | +0.71 [+0.48, +0.92] |
| 2 | +0.19 [−0.06, +0.44] | +0.36 [+0.12, +0.59] |
| 1 | −1.73 [−2.05, −1.41] | −1.25 [−1.55, −0.96] |

> Full report: "Results brief" and "Analysis".

## Split by the aspect the pairs show

AFF's whole lead is in the episodes whose pairs show emotion. AFF's gate opens only when its reader picks the affect
grouping, so its one move is to steer toward emotion. On style and genre, its best result is a tie with the bar.

| Example pairs | pairs show emotion | pairs show style | pairs show genre |
|---|---|---|---|
| 4 (today) | +2.88 [+2.40, +3.38] | −0.09 [−0.46, +0.28] | −0.69 [−1.07, −0.31] |
| 3 | +2.92 [+2.44, +3.42] | +0.04 [−0.34, +0.41] | −1.06 [−1.45, −0.67] |
| 2 | +2.58 [+2.09, +3.07] | −0.17 [−0.56, +0.20] | −1.84 [−2.27, −1.42] |
| 1 | +4.41 [+3.83, +5.02] | −1.15 [−1.66, −0.64] | −8.44 [−9.11, −7.73] |

AFF over the bar, R@1 points. Each aspect pools its two aspect pairs (8,192 episodes).

- The emotion gain does not need the pairs. It is largest with 1 pair.
- The pairs mostly tell AFF when not to steer. Genre pays most when AFF steers wrongly, because the bar already finds
  the genre match about three times as often as the emotion or style match.

**Your reading** (2026-10-14, after seeing ours; you note it may hold mistakes): "Emotion is so different from style and
genre that it needs fewer examples; style × genre stays hard at any k. This supports 'the gain is mostly from emotion,
from the GoEmotions encoders'."

> Full report: "Analysis", its last part, "Split by the aspect the pairs show".

## Advice

<details>
<summary>Our advice (open after giving your reading)</summary>

Stop here, as you chose, and add one line to AFF's model card (its fact sheet): it needs at least 3 example pairs, and
with 1 pair it is worse than the bar. The curve is flat from 4 to 3 pairs, so more than 4 pairs is not likely to help.
This is our view.

</details>

## Next

The question is answered for development data, for AFF used as it is. You chose to stop after 1 pair. Not run: 5 to 7
pairs on new episodes, and a re-fit of the reader and gate at 1 or 2 pairs to separate the two causes of the drop.
One question is open for your weekly meeting (2026-10-14): are aspect labels needed in training to separate style from
genre? No source without labels has separated them so far.

## Glossary

- **AFF (affect steering):** our current best method; it adds an emotion term when its reader is sure the examples
  show emotion.
- **Example pairs:** each pair is an image and a caption from two paintings that share one value of an aspect. Support
  pairs show the aspect to use, contrast pairs another one.
- **Condition:** which aspect the supports show; swapping supports and contrasts flips it.
- **Reader:** AFF's part that estimates which aspect the example pairs show.
- **Gate:** AFF's rule for when the emotion term enters the score.
- **The bar, B′(A0):** the strongest scorer that ignores the condition.
- **Matched control:** AFF with only the condition removed.
- **Development episodes, seed 42:** the episodes we tune on. The held test uses other paintings.
