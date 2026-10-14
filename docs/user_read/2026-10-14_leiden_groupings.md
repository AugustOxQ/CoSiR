# Leiden groupings tie k-means, and the old attention-H1 communities do not help

> 2026-10-14 · Full report: [Leiden groupings side check](../reports/auto/v2/2026-12-04_leiden_groupings.md).

## Summary

- We screened two other ways to group the training data. A screen is a quick, cheap test. It is not AFF.
- Leiden communities at 64 groups do as well as k-means at 64 groups. So the earlier Leiden loss came mostly from too
  few groups.
- The 19 communities of attention-H1 score worse than their matched control, which ignores the condition.
- Neither passed the screen's bar, so we did not rebuild AFF on either one.
- These are development numbers on seed 42.

## How we got here

The project ranks captions for an image (or images for a caption) by one aspect, such as emotion. Example pairs show
the aspect. AFF, our current best method, uses **groupings**: label-free partitions of the training data. AFF has
three: affect (from emotion probabilities), image and caption (both k-means on CLIP features, 64 groups each).

Before AFF, in September, the buddy line grouped paintings into Leiden communities and compared them with PercepT, a
published clustering method. Its best model, **attention-H1**, is a small network that mixes content and emotion. It
gives 19 communities.

On 2026-10-14 you asked two questions:
- Is k-means the right way to group images and captions, or is Leiden (graph communities) better?
- Do the H1 communities help in today's episodes?

## Two screens

**Problem.** Nobody ever compared k-means with Leiden for the image and caption groupings at the same group count. An
earlier Leiden run lost, but it had only 17 and 30 groups. H1 was never tried in today's episodes.

**Idea.** We used the cheap screen from round 3, the one that picked AFF's affect grouping. It reports two margins over
the matched control (the same score with only the condition removed):
- **Told margin:** the right grouping is given from the labels. It shows the most a grouping can give.
- **Reader margin:** a simple reader picks the grouping without labels.

A candidate passed if its reader margin was at least the reference's. We tuned Leiden to 64 groups to match k-means.
We used H1 as the only grouping, as in the buddy line.

**Did it work.** No, for both.
- Leiden at 64 groups ties k-means when told. The reader gets a little less from it, within the noise.
- H1 adds nothing that depends on the condition. The fitted score ignored H1's condition part.
- The screen is not AFF: it has a simpler reader and no gate (AFF's rule for when its emotion term enters the score).
- We ran one Leiden seed only.
- H1 saw the held test paintings when it was trained, without their labels. So its communities can feed only
  development checks.

**Key evidence.** R@1 points (the share of queries with the right answer ranked first) over the matched control, on
12,288 development episodes, with 95% intervals:

| Groupings | Told margin | Reader margin |
|---|---|---|
| k-means, 64 groups (reference) | +1.64 [+1.37, +1.92] | +0.35 [+0.15, +0.57] |
| Leiden, 64 groups | +1.70 [+1.41, +2.01] | +0.22 [+0.06, +0.38] |
| H1 alone, 19 groups | −0.46 [−0.63, −0.28] | −0.46 [−0.63, −0.28] |

> Full report: "Results brief", "Results" and "Analysis".

## Advice

<details>
<summary>Our advice (open after giving your reading)</summary>

Stop both checks. Keep k-means for the image and caption groupings. Add one line to AFF's model card (its fact sheet):
Leiden at 64 groups ties k-means. H1 likely fails because it gives one group per painting, while emotion often differs
between the captions of one painting. This is our view.

</details>

## Next

Both checks are answered for the screen. No AFF rebuild runs. Your options:
- Stop here.
- Rebuild AFF on Leiden at 64 groups anyway: about 1 h of runs plus 1 to 2 h of build.
- Retrain H1 on the training paintings only, so the held paintings stay unseen: hours.

## Glossary

- **AFF (affect steering):** our current best method; it adds an emotion term when its reader is sure the examples
  show emotion.
- **Grouping:** a label-free partition of the training rows; a group is one of its parts.
- **Leiden, k-means:** two ways to make a grouping: graph communities, and clusters around 64 centres.
- **Attention-H1:** the buddy line's best student network; its 19 Leiden communities mix content and emotion.
- **Matched control:** the same score with only the condition removed.
- **Told margin, reader margin:** the screen's margins over the matched control, with the grouping given from the
  labels, or picked by a simple reader.
- **Development episodes, seed 42:** the episodes we tune on; the held test uses other paintings.
