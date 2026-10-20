# Our model passes the full-scale emotion test; under Leiden its lead disappears

> 2026-10-20 · [Full report](../reports/auto/percept/2026-12-19_claim_test_stage2.md)

## Summary

- Our model passed the emotion bar against PercepT core, the strongest PercepT version that did not collapse.
- With more training data, PercepT core rose more than our model, so our lead shrank.
- The paper's own topic rule collapsed again; its 40-topic versions scored below PercepT core.
- Under Leiden clustering the emotion lead disappears; on style and genre our model is below the instructed image.
- The rule now requires the paper's sentence, then a facts brief and a spec for the next dataset, ArtELingo-28.

## How we got here

We want one model that finds emotion, style and genre clusters in paintings, without an instruction at test time.
We compare it with PercepT core, our reimplementation of PercepT.
A decision rule, fixed before the runs, sets the bar and what each outcome leads to.
Stage 1 trained on a smaller sample and passed the bar on held paintings, which no model sees in training.
Under Leiden, a graph-based clustering method, stage 1's lead was not seen.
Stage 2 retrained every model on all train paintings and tested the same held paintings with the same rule.

## Did the emotion pass hold with all the training data?

**Problem.** Stage 1's pass came from a smaller training sample; the claim for the paper needs full-scale training.

**Idea.** Our model has one shared layer and one head each for emotion, style and genre.
It trains on graphs and with a pull toward the instructed Qwen image: the image's embedding made with an aspect instruction.
At test time it reads only the default Qwen image, made without an instruction.
Every model retrained with its own training settings on the same precomputed features.

| Test setup | Value |
|---|---:|
| Train paintings, every model | 42,969 |
| Same held paintings as stage 1 | 12,281 |
| Training seeds per model | 5 |
| Our model's k-means clusters | 20 |

**Did it work?** Yes.
Our model cleared both parts of the bar.
An independent recomputation agreed exactly.

The run was planned on the cluster node and stopped there twice.
First, a safety check in our code had a bug; we fixed it.
Then the cluster GPU did not reproduce stage 1's numbers.
The local GPU reproduced them exactly, so the whole run moved there before any held painting was scored.

**Key evidence.** Adjusted mutual information (AMI) measures how well clusters agree with labels.
The bar has two parts: a margin of at least +0.0100, and a paired 95% interval whose lower end is above zero.
The interval comes from a bootstrap over paintings and seeds.
PercepT core, read by its own top topic, was the strongest comparator.

| Held emotion AMI | Stage 1 | Stage 2 |
|---|---:|---:|
| Our model | 0.1001 | 0.1008 |
| PercepT core | 0.0855 | 0.0874 |
| Our margin | +0.0146 | +0.0133 |
| Margin interval | [+0.0093, +0.0157] | [+0.0081, +0.0146] |

Both parts hold in both stages.
The interval's lower end must be above zero; it does not need to reach +0.0100.
From stage 1 to stage 2, PercepT core rose by 0.0019 and our model by 0.0007, so the margin shrank.
Both changes are about the size of the spread across seeds, or smaller.

The paper's sentence states the claim this result licenses.
Our model reads the default Qwen image, and k-means groups it into 20 clusters.
Its held emotion AMI is at least 0.01 above the strongest PercepT that did not collapse, with the interval above zero.

> Full report: [Results](../reports/auto/percept/2026-12-19_claim_test_stage2.md#results) and [Deviations](../reports/auto/percept/2026-12-19_claim_test_stage2.md#deviations).

## Did the paper's settings make PercepT stronger?

**Problem.** PercepT core does not use the paper's own training settings, so those settings could give a stronger PercepT.

**Idea.** Run the paper's settings on our features, with two encoder choices.
One version keeps topics by the paper's own rule; the other keeps a fixed 40 topics.

**Did it work?** No.
The paper's rule kept under 10 or over 90 topics in all ten runs, so every one counts as collapsed.
The 40-topic versions did not collapse, but they scored below core.

**Key evidence.** Each version is read by its own top topic.

| PercepT version | Stage 2 emotion AMI |
|---|---:|
| Paper's topic rule | All ten runs collapsed |
| Paper settings, 40 topics, CLIP + RoBERTa | 0.0711 |
| Paper settings, 40 topics, Qwen | 0.0707 |
| Core | 0.0874 |

> Full report: [Results](../reports/auto/percept/2026-12-19_claim_test_stage2.md#results) and the collapse table in [Analysis](../reports/auto/percept/2026-12-19_claim_test_stage2.md#analysis).

## How far does the lead reach beyond the verdict?

**Problem.** The verdict answers one question: emotion clusters under k-means.

**Idea.** Leiden finds communities in a nearest-neighbor graph of the held outputs.
It shows whether the lead depends on k-means.
We also compared style and genre with core, and all three aspects with the Qwen images.

**Did it work?** Partly.
Under Leiden the emotion lead disappears; k-means with Leiden's number of clusters keeps it.
Our model beats core on style and genre, but it is below the instructed image on both.
These readings decide nothing under the rule.

**Key evidence.** Each difference is our model minus the comparator.
For style and genre, core is read in whichever of its two ways scores higher.

| Against PercepT core | Difference [95% interval where computed] |
|---|---:|
| Emotion, Leiden | −0.0005 [−0.0048, +0.0023] |
| Emotion, k-means at Leiden's cluster count | +0.0190 |
| Style | +0.1388 [+0.1283, +0.1427] |
| Genre | +0.0208 [+0.0103, +0.0310] |

The Qwen images are clustered directly, with k-means seeds varied only, so these differences have no interval.

| Against a Qwen image | Instructed image | Default image |
|---|---:|---:|
| Emotion | +0.0087 | +0.0300 |
| Style | −0.0340 | +0.1452 |
| Genre | −0.0623 | −0.0082 |

On genre our model is also slightly below the default image.
Style uses all held paintings; genre uses the 9,949 that have genre labels.
The paper's settings ran emotion only.
The 10 November gate still needs its own test against the instructed image.

> Full report: [Analysis](../reports/auto/percept/2026-12-19_claim_test_stage2.md#analysis).

## What this test does not show

The claim excludes style, genre, other datasets, other backbones, the instructed Qwen bar and retrieval.
It also excludes PercepT with its own encoders.
The split keeps each painting on one side, but we did not measure whether train and held share artists.

## Advice

<details>
<summary>Our advice (open after giving your reading)</summary>

Follow the rule: write the paper's sentence, then the facts brief and spec for ArtELingo-28.
Prepare the work for the 10 November gate beside it.
The second track waits for your choice of a cluster-level loss.

</details>

## Next

1. Give your reading of these results before you open our advice.
2. Follow the rule: the paper's sentence, then ArtELingo-28's facts brief and spec.
3. Choose the cluster-level loss for the second track; that choice is yours.
4. Beside this, prepare the 10 November gate: the instructed Qwen bar and retrieval by aspect name.
   Checks of the graphs on development data could also explain the Leiden result.

> Full report: [Results brief](../reports/auto/percept/2026-12-19_claim_test_stage2.md#results-brief).

## Glossary

- **Our model:** D2, also K1; one shared layer, one head per aspect, graphs, and a pull toward the instructed Qwen image.
- **PercepT core:** B1; our reimplementation of PercepT.
- **Paper settings:** PercepT with its paper's training settings; at 40 topics B4 (CLIP + RoBERTa) and B5 (Qwen).
- **Held paintings:** the test paintings; no model sees them in training.
- **Top topic:** PercepT's own reading, which puts each painting in its most likely topic.
- **AMI:** adjusted mutual information, the agreement between clusters and labels.
- **Collapse:** R-COL; the paper's rule keeps under 10 or over 90 topics, or one topic takes over half the training pairs.
  The 40-topic versions collapse only by the second test.
- **Leiden:** a clustering method that finds communities in a graph; here it replaces k-means as a check.
- **Instructed and default Qwen image:** the Qwen embedding of a painting, made with or without an aspect instruction.
- **The bar:** the decision rule's test; passing it is the outcome S2-GO.
- **Your reading and loss choice:** decisions C and B of the research loop.
- **10 November gate:** 11-10; the instructed Qwen bar and retrieval by aspect name.
