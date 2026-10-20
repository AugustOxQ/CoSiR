# Our model passes the held emotion test; under Leiden clustering its lead is near zero

> 2026-10-20 · [Full report](../reports/auto/percept/2026-12-18_claim_test_stage1.md) (its file name keeps the
> project's sequence date, not the real date)

## Summary

- Our model passed the held emotion test against the strongest PercepT version that did not collapse.
- Its emotion lead was +0.0146 AMI; the bar asked for at least +0.0100.
- By the decision rule, the full-scale test (stage 2) runs next, and its outcome replaces this one.
- With Leiden clustering instead of k-means, the emotion lead was near zero.
- On style and genre, our model beat PercepT core but stayed below the instructed Qwen image.

## How we got here

We want one image-text model that finds emotion, style and genre clusters in paintings, with no instruction at test
time. PercepT (arXiv 2606.03345) is the method to beat. On development data, the last comparison of model versions
(the model sheet) picked our current model. There it beat the instructed-image target on emotion only.

On 2026-10-20 you chose a test on held paintings: paintings never used to build or choose a model. You also asked
for a Leiden reading beside the verdict, since you questioned k-means as the clustering step. The test ran overnight.

## Does our model pass the held emotion bar?

**Problem.** Development data chose our model, so its lead there could be too high.

**Idea.** Test it once on 12,281 held paintings, under a decision rule written before the test.

- Our model: one shared layer with emotion, style and genre heads. It reads the plain Qwen image embedding once.
- Its training: each head learns from pairs of similar paintings for its aspect (the buddy graphs). Each head is also
  pulled toward the instructed Qwen image, the embedding made with an instruction that names the aspect.
- The comparator: every PercepT version, each read two ways (k-means, or its own top topic). The strongest version
  that did not collapse sets the bar.
- Both sides trained on the same 13,400 paintings from our stored features, with five seeds each.

AMI (adjusted mutual information) measures how well clusters match the emotion labels; 0 is chance.

**Did it work?** Yes. Our model cleared the bar, and an independent recomputation gave the same numbers. The
interval's lower end, +0.0093, is below the 0.01 margin; the rule asks only that it is above 0.

**Key evidence.** Held emotion AMI, mean over five seeds. The interval is a paired 95% bootstrap over paintings and
seeds.

| Held emotion test | Value |
|---|---:|
| Our model, k-means at 20 clusters | 0.1001 |
| Strongest PercepT: core, its own top topic | 0.0855 |
| Difference | +0.0146 [+0.0093, +0.0157] |
| Bar | difference at least +0.0100, and the interval's lower end above 0 |

The resampled differences averaged +0.0125, so the point value sits near the top of its interval. The late-October
gate passes under the rule; that link from this outcome to the gate was an agent default, which you check in the
spec's brief. Stage 2 retrains every model on the 42,969 training paintings with the same recipes.

> Full report: "Results brief", "Results" and the bootstrap paragraph of "Analysis".

## Did a PercepT closer to the paper set a stronger bar?

**Problem.** PercepT core is our own version of PercepT. It differs from the paper in more than the backbone, for
example in its training settings and its fixed 40 topics.

**Idea.** Add "paper settings" versions: the paper's training settings on our features, so only the backbone differs.
They come with two ways to choose topics:

- strict: the paper's own rule for which of the 100 topics to keep;
- fixed count: the 40 most used topics, as in core.

A version collapses when it keeps under 10 or over 90 topics, or when one topic takes over half the training pairs. A
collapsed version leaves the bar.

**Did it work?** No. Every strict run collapsed, and the fixed-count versions scored below core.

**Key evidence.** Held emotion AMI, each version read by its own top topic, mean over five seeds.

| Comparator | Topics kept | AMI |
|---|---|---:|
| Paper settings, strict (both input versions) | 1 to 2, or 98 to 99, of 100 | out (collapsed) |
| Paper settings, 40 topics, CLIP + RoBERTa inputs | 40 | 0.0707 |
| Paper settings, 40 topics, Qwen inputs | 40 | 0.0687 |
| PercepT core | 40 | 0.0855 |

Collapse could change at full scale; stage 2 tries again.

> Full report: "Results" and the paper settings table in "Analysis".

## Does the lead hold under other readings?

**Problem.** The verdict uses one reading: emotion, with k-means at 20 clusters.

**Idea.** Look at other readings beside the verdict. By the rule, they decide nothing.

- Leiden: communities in a nearest-neighbour graph, the clustering you set against k-means.
- Style and genre against PercepT core's stronger reading (k-means or its own top topic, whichever scores higher).
  The paper settings versions ran emotion only.
- The instructed Qwen image and the plain Qwen image, each clustered directly. The instructed image needs one pass
  per aspect; our model needs one pass in total.

**Did it work?** Partly.

- Under Leiden the emotion lead is near zero, and a second Leiden reading reverses it. At Leiden's cluster count,
  k-means still gives a lead of +0.0223.
- Style leads core by a wide margin; genre leads only just (interval's lower end +0.0002).
- Our model beats the instructed image on emotion, but not on style or genre. On genre it is also just below the
  plain image.

**Key evidence.** AMI differences, our model minus the comparator. Intervals appear only where computed. The image
rows repeat k-means only, with no training seeds.

| Reading | Difference |
|---|---:|
| Emotion, Leiden on the held graph, against core | +0.0019 [−0.0030, +0.0053] |
| Emotion, Leiden communities from training, held paintings join by vote, against core | −0.0018 |
| Emotion, k-means at Leiden's cluster count, against core | +0.0223 |
| Style, against core | +0.1436 [+0.1339, +0.1491] |
| Genre, against core | +0.0079 [+0.0002, +0.0153] |
| Emotion / style / genre, against the instructed image | +0.0080 / −0.0307 / −0.0572 |
| Emotion / style / genre, against the plain image | +0.0293 / +0.1486 / −0.0031 |

These rows are diagnostics. The early-November checkpoint's instructed-image bar needs its own test.

> Full report: "Analysis" (Leiden, style and genre, target and encoder tables).

## What this test does not show

- Full-scale training: stage 2 tests it.
- Claims on style and genre, other datasets, other backbones, or retrieval.
- The instructed-image bar and the aspect-named retrieval of the early-November checkpoint.
- PercepT with its own encoders (CLIP L/14 and a ModernBERT tuned on GoEmotions): our PercepT versions read our
  stored features.
- An artist-level split: the split groups paintings and identical image vectors, not artists.

## Advice

<details>
<summary>Our advice (open after giving your reading)</summary>

Our reading: the held emotion lead fits our prior (+0.005 to +0.015), near its upper end. It depends on the
clustering method: under Leiden it nearly vanishes. Our model beats the instructed image on emotion, but stays below
it on style and genre.

Our recommendation: build stage 2 with the recipes unchanged. Choose the second track's loss idea separately. Develop
the early-November checkpoint's work on development data.

</details>

## Next

- Your reading of these results comes first (decision C), before you open our advice.
- By the rule, the stage 2 build has already started in its own chat, beside your reading. It builds and tests the
  code; the full run waits for the evening or the cluster node.
- You also check the brief of the claim test spec, which you approved before the night run without reading it.
- The second track waits for your choice of a cluster-level loss (decision B): a loss on whole clusters, not on
  single pairs. The literature note ranks three ideas for a one-day exploration:
  1. a teacher anchor per head: each head also learns the clusters that k-means finds on its instructed image;
  2. a cluster head that teaches itself from the buddy pairs, after the published method TEMI;
  3. a probe of published cluster methods on our features, with no new training of our model.
- The note found no evidence that DEC helps when it starts from a good partition.

> Source for the three ideas: [the cluster-level losses note](../reports/literature/2026-12-17_cluster_level_losses.md),
> "Candidate ideas for a one-day exploration loop (ranked)".

## Glossary

- **AMI:** adjusted mutual information; 0 is chance agreement with the labels, 1 is perfect.
- **Held paintings:** 12,281 test paintings never used to build or choose a model.
- **Our model:** D2 in the model sheet, K1 in the full report: one shared layer, three aspect heads, buddy graphs and
  a pull toward the instructed image ("direct distillation").
- **Instructed Qwen image:** the Qwen image embedding made with an instruction that names one aspect; the "target".
- **Plain Qwen image:** the Qwen image embedding with no instruction; the "encoder" in the full report.
- **PercepT core:** our own version of PercepT, B1 in the full report.
- **Paper settings versions:** PercepT with the paper's training settings; strict cut B2 and B3, 40 topics B4 and B5;
  inputs CLIP + RoBERTa (P1) or Qwen (P4).
- **Collapse:** the rule R-COL: under 10 or over 90 kept topics, or one topic with over half the training pairs.
- **Reading:** how clusters come from a model's outputs: k-means at 20 clusters, or PercepT's own top topic ("own").
- **Stage 1, stage 2:** the test after training on 13,400 paintings, then on all 42,969; this result is S1-GO.
- **Late-October gate:** the 10-29 checkpoint: held emotion AMI at least 0.01 above PercepT, interval above 0.
- **Early-November checkpoint:** the 11-10 gate: the instructed-image bar and aspect-named retrieval.
