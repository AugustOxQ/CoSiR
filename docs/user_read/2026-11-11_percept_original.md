# PercepT on its own encoders is a stronger comparator in style and genre, and CoSiR stays ahead

> 2026-11-11 · Full report: [PercepT on its own encoders](../reports/auto/percept/2027-01-15_percept_original.md)

## Summary

- We ran PercepT on the encoders its paper uses, so that the paper's clustering table has a PercepT row on them.
- The run worked: all 5 seeds of the 40-topic version gave usable topics.
- This PercepT clusters style and genre better than our earlier PercepT on Qwen3, and emotion about the same.
- CoSiR's mean is higher in all three aspects. On genre, the gap is not tested at the seed level.
- PercepT's own topic-count rule failed again on all 5 seeds, as on two other encoders before.

## How we got here

CoSiR, our model, gives one embedding per aspect (emotion, style, genre) for an image. PercepT is the published method
it improves on: it learns topics from image-caption pairs and then predicts them from the image alone. Our earlier
PercepT rows ran on Qwen3, the image-text embedding model that CoSiR is built on.

A reviewer could say that PercepT does better on the encoders it was built for. So on 2026-11-11 you chose a new run of
PercepT on its own encoders, on our data and test protocol.

## PercepT on CLIP L/14 and ModernBERT

**Problem.** The clustering table had no PercepT row on PercepT's own encoders.

**Idea.** PercepT's paper uses two encoders. CLIP ViT-L/14 reads the image and the caption. ModernBERT, a text model
tuned on an emotion dataset (GoEmotions), reads the caption's emotion. We used both.

The paper will state two differences from PercepT's own setup. We train on our 42,969 English paintings, not its 1.2M
translated image-caption pairs. We keep 40 topics, not the number its own rule picks. Three smaller choices are ours,
and we did not measure their effect.

We scored the test paintings as the clustering table does. K-means makes as many clusters as the aspect has classes.
AMI then measures how well the clusters match the labels. We also ran PercepT's own topic-count rule beside it.

**Did it work.** Yes for the 40-topic version. No seed collapsed, and an independent recompute gave the same numbers.

The topic-count rule did not work. On four seeds it kept 96 to 99 of 100 topics. On one seed it kept a single topic.

**Key evidence.** Test AMI at the class count, mean of 5 seeds:

| Table row | Emotion | Style | Genre |
|---|---:|---:|---:|
| PercepT, its own encoders (new) | 0.064 | 0.262 | 0.474 |
| PercepT, Qwen3 (earlier) | 0.068 | 0.192 | 0.424 |
| CoSiR | 0.102 | 0.411 | 0.510 |

Across its 5 seeds, the new row's genre value goes from 0.457 to 0.512. CoSiR's 0.510 lies inside that range.

> Full report: "Results brief" and "Analysis".

<details>
<summary>Our advice (open after giving your reading)</summary>

The run is descriptive: it had no pass or fail rule. If the paper says that CoSiR beats PercepT in all three aspects,
test the genre gap at the seed level first. Use the same rule as for the other paper comparators; it costs a few CPU
minutes. If the paper only shows the table, nothing more is needed.

</details>

## Next

The paper chat put the row in the clustering table. It also has the files for the retrieval table. This was a small
loop for the paper, so no next experiment is open.

Your decision, an optional follow-up: does the paper's wording need the seed-level genre test?

## Glossary

- **AMI:** adjusted mutual information between the clusters and the labels; 0 is chance level, 1 a perfect match.
- **Class count:** k-means with as many clusters as the aspect has classes (emotion 9, style 27, genre 10).
- **40-topic version:** PercepT keeps the 40 most-used of its 100 topics.
- **Topic-count rule:** PercepT's own way to choose how many topics to keep (Algorithm 1 in its paper).
- **Collapsed:** a run whose topics are unusable. Any run collapses when one topic holds over half the captions. A run
  of the topic-count rule also collapses when it keeps under 10 or over 90 topics.
