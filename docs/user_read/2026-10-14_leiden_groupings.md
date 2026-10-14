# Leiden ties k-means on the screen and bar margin; AFF on Leiden is 0.56 lower on its own R@1; H1 fails; kept as records

> 2026-10-14 · Full report: [Leiden groupings side check](../reports/auto/v2/2026-12-04_leiden_groupings.md).

## Summary

- Leiden ties k-means on the cheap screen, within the noise.
- AFF on Leiden is 0.20 lower on the bar margin, within the noise.
- AFF's own R@1 is 0.56 lower than with k-means, with an interval below zero.
- H1 retrained on training paintings alone still fails its screen. No H1 rebuild follows.
- You keep both checks as records. AFF keeps k-means; Leiden is kept as a record (a parked idea).
- These are development results only, on the seed used to pick every setting.

## How we got here

The follow-ups tested the groupings in AFF and H1 after retraining.

The project ranks captions for an image (or images for a caption) by one aspect, such as emotion. Example pairs show
the aspect. AFF, our current best method, uses **groupings**: label-free partitions of the training data. AFF has
three: affect (already Leiden, from emotion probabilities), image and caption (both k-means on CLIP features, 64 groups each).

Before AFF, in September, the buddy line grouped paintings into Leiden communities and compared them with PercepT, a
published clustering method. Its best model, **attention-H1**, is a small network that mixes content and emotion. It
gives 19 communities.

On 2026-10-14 you asked two questions:

- Is k-means the right way to group images and captions, or is Leiden (graph communities) better?
- Do the H1 communities help in today's episodes?

The first-screen reading came on 2026-10-14 01:55; you gave the rebuild reading on 2026-10-14, both before seeing ours.
The AFF rebuild ran from 2026-10-14 02:07 to 2026-10-14 02:54.

## One screen, three arms

**Problem.** Nobody ever compared k-means with Leiden for the image and caption groupings at the same group count. An
earlier Leiden run lost, but it had only 17 and 30 groups. H1 was never tried in today's episodes.

**Idea.** We used the told-oracle screen, the earlier quick test that selected AFF's affect grouping.
It has three arms: k-means reference, Leiden, and H1 alone. It reports two margins over the matched control.
The matched control uses the same score with the requested aspect removed.

- **Told margin:** the right grouping is given from the labels. It shows the most a grouping can give.
- **Reader margin:** a simple reader picks the grouping without labels.

Our default pass bar, set before the run, required a reader margin at least the reference's.
We tuned Leiden to 64 groups to match k-means.
We used H1 as the only grouping, as in the buddy line.

**Did it work.** Neither candidate passed the screen's bar.

- Leiden at 64 groups ties k-means when told. The reader gets a little less from it, within the noise.
- H1 adds nothing that depends on the requested aspect. The fitted score ignored its part that responds to that aspect.
- The screen uses a rule reader, without AFF's practice bank, gate or fitted thresholds. The later rebuild compares AFF itself.
- We ran one Leiden seed only, in the screen and the rebuild.
- H1 saw the held test paintings when it was trained, without their labels. So its communities can feed only
  development checks.

**Key evidence.** R@1 points (the share of queries with the right answer ranked first) over the matched control, on
12,288 development episodes, with 95% intervals:

| Groupings | Told margin | Reader margin |
|---|---|---|
| k-means, 64 groups (reference) | +1.64 [+1.37, +1.92] | +0.35 [+0.15, +0.57] |
| Leiden, 64 groups | +1.70 [+1.41, +2.01] | +0.22 [+0.06, +0.38] |
| H1 alone, 19 groups (one grouping, so told and reader are the same) | −0.46 [−0.63, −0.28] | −0.46 [−0.63, −0.28] |

> Full report: "Results brief", "Results" and "Analysis".

## H1 retrained on training paintings only

**Problem.** The old H1 saw held paintings as features, and its mapped communities might undersell the trained network.

**Idea.** You asked for H1 retrained on scorer-train paintings only, still as the only grouping.
The retrained H1 gives 20 communities, with AMI 0.49 against the old labels, a measure of agreement between partitions.
We screened it first under your "cheap is best" rule; that order was our default.

**Did it work.** H1 still fails. The cross-fit gives weight 0 to its part that responds to the requested aspect in both folds.
No rebuild, by the cheap rule. This tests the one-grouping use only.

**Key evidence.** R@1 points with 95% intervals:

| Retrained H1 alone | Result |
|---|---|
| Over its matched control | −0.27 [−0.48, −0.07] |
| Reader minus k-means reference | −0.63 [−0.94, −0.32] |

Training was set to end when recall stopped improving on training data, rather than on separate test data.
It reached all 200 epochs.
Affect inputs used stored GoEmotions probabilities averaged over training rows, without fresh extraction.
Text and image inputs per painting also came from training rows only.

> Full report: "What ran", "Results", "Analysis" and "Deviations".

## AFF rebuilt on Leiden

**Problem.** The told-oracle screen uses a simple reader, without AFF's practice bank, gate or fitted thresholds.
It could miss a benefit that AFF's learned reader can use.

**Idea.** You chose Leiden on a tie for a coherent design and asked us to rebuild AFF anyway.
We replaced image and caption k-means with Leiden at 64 everywhere, including B and B′(A0), the comparison scorers.
That scope was our default. The affect grouping stayed unchanged, already Leiden in AFF.
Only the image and caption groupings changed. We refit the picks and thresholds and reused CLIP's content graph.

**Did it work.** Leiden ties on the bar margin within the noise; AFF's own R@1 is lower, with its interval below zero.
The bar margin compares AFF with the stronger of B′(A0) and its matched control; here B′(A0) sets the bar.

On seed 42, B′(A0) itself is lower with Leiden: 18.08 against 18.44 R@1 points with k-means.
A lower bar makes the margin look better. Thus AFF's own R@1 falls more than its margin.
Leiden is 0.20 lower on the bar margin, within the noise; AFF's own R@1 is 0.56 lower.
The k-means comparison reproduces the value AFF had on this development seed before.

**Key evidence.** R@1 points on the same development episodes, with 95% intervals where available:

| Metric | Leiden at 64 | k-means | Paired Leiden minus k-means |
|---|---|---|---|
| Bar margin / over B′(A0) | +0.50 [+0.30, +0.71] | +0.70 [+0.46, +0.94] | −0.20 [−0.44, +0.05] |
| AFF's own R@1 | 18.58 | 19.14 | −0.56 [−0.80, −0.30] |
| Over its matched control | +0.53 [+0.34, +0.72] | +0.74 [+0.52, +0.96] | −0.21 (point only, no interval) |

The matched-control difference has no paired interval; it is the point difference between the two margins.
The k-means matched-control margin comes from the example-pair check, which reproduced the earlier scores exactly.
A summary step crashed after the scores were saved. We fixed it and recomputed the summary from saved per-episode arrays.
No score changed.

> Full report: "What ran", "Results: AFF rebuild", "Analysis" and "Deviations".

## Your reading

On the first screens, you chose Leiden on a tie for coherence and expected H1 to fail because its use had changed.
You then requested both follow-ups.

On the rebuild, you said: "I don't think the way we do grouping is that important, given big issue exposed just now".
You keep Leiden as a record that it "can also do the job", a minor open direction.
You also keep H1's one-grouping failure as a record. You gave both readings before you saw ours.
The "big issue" is the example-pair check's question: are aspect labels needed in training to separate style from genre?

> Readings: user, before seeing ours. Source: [direction log, Leiden side check](../superpowers/direction_log.md).

## Advice

<details>
<summary>Our advice (open after giving your reading)</summary>

Our reading: Leiden ties k-means on the bar margin within the noise.
On seed 42, B′(A0) is lower with Leiden: 18.08 against 18.44 R@1 points with k-means.
That lower bar explains why AFF's own R@1 falls more than its margin.
Keep k-means; record Leiden as a coherent alternative, 0.20 lower on the bar margin, within the noise.
AFF's own R@1 is 0.56 lower, with an interval below zero.

The two rebuild readings agree. Our first-screen advice was to stop both checks; you chose the follow-ups instead.
H1 still fails as the only grouping after retraining. This is our view.

</details>

## Next

**Decided.** AFF's recipe does not change: k-means stays in r9 and the paper, with no Leiden held read.
Leiden is kept as a record (a parked idea); H1's one-grouping failure is recorded too.
The user keeps Leiden as a record and a minor open direction; it was added to the parked ideas.
You asked to "keep this just as a record"; we read that as no recipe change and stated this to you.

These results cover seed-42 development episodes only, the seed every setting was picked on.
They use CLIP only, no separate held-test data, one Leiden seed and one Leiden group count (64).

> Full report: "Decision" and "Results brief: What it does not show".

## Glossary

- **AFF (affect steering):** our current best method; it adds an emotion term when its reader is sure the examples
  show emotion.
- **Grouping:** a label-free partition of the training rows; a group is one of its parts.
- **Leiden, k-means:** two ways to make a grouping: graph communities, and clusters around 64 centres.
- **Attention-H1:** the buddy line's best student network; its communities mix content and emotion. Old version: 19; retrained: 20.
- **Matched control:** the same score with only the condition removed.
- **Told margin, reader margin:** the screen's margins over the matched control, with the grouping given from the
  labels, or picked by a simple reader.
- **Development episodes, seed 42:** the episodes we tune on; the held test uses other paintings.

- **Bar margin:** AFF's R@1 above the stronger of B′(A0) and its matched control.
- **B:** the base comparison scorer, without the score part that responds to the requested aspect.
- **B′(A0):** the condition-free comparison scorer AFF must beat, "the bar" here; it also uses image and caption groupings.
- **Condition:** the requested aspect, such as emotion, shown by the example pairs.
- **Cross-fit:** fit score weights on one part of the data and evaluate them on another part.
- **Folds:** the two parts of the data used for the cross-fit.
- **GoEmotions:** the outside emotion classifier whose probabilities provide AFF's affect inputs.
- **r9:** the current frozen AFF recipe, used in the main experiment and the paper.
- **Held read:** evaluation on separate test paintings that were not used to pick settings.
