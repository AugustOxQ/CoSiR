# ArtELingo-native BERT ceiling on held-out val+test data

Generated automatically, 2026-09-22 16:18:14.

## Held-out setup

This run uses ArtELingo val+test combined: **46,813 rows** and **9,365 unique paintings** (loaded as 9,365 deduplicated nodes). The set has verified **zero painting-level overlap** with the train split. The same ArtELingo-native BERT checkpoint is evaluated on these held-out captions, then its mean-pooled 9-way softmax vectors form a single-modality mutual-kNN graph (K=20), repaired with alpha=0.5, with Leiden seed=42.

## Per-caption classifier sanity check

Anonymous checkpoint indices are re-mapped independently on held-out captions by empirical per-index majority vote; the train mapping is not reused for the accuracy calculation. The recovered held-out mapping matches the train-run recovered mapping label-for-label.

| checkpoint index | held-out recovered emotion | supporting caption rows | train recovered emotion |
|---|---|---:|---|
| LABEL_0 | amusement | 2,615 | amusement |
| LABEL_1 | awe | 4,048 | awe |
| LABEL_2 | contentment | 8,775 | contentment |
| LABEL_3 | excitement | 1,669 | excitement |
| LABEL_4 | anger | 320 | anger |
| LABEL_5 | disgust | 1,495 | disgust |
| LABEL_6 | fear | 3,436 | fear |
| LABEL_7 | sadness | 4,111 | sadness |
| LABEL_8 | something else | 3,671 | something else |

**Held-out recovered-mapping top-1 accuracy: 64.38% (30,140/46,813 English caption rows).**

## Comparison

Genre metrics use only paintings that overlap the genre-labelled diagnostic set. The held-out genre subset has n=159, so this diagnostic has low sample size and low statistical power here; unlike emotion, it does not use the full held-out graph.

| signal / split | per-caption accuracy | emotion AMI | emotion V-measure | genre AMI | genre V-measure |
|---|---:|---:|---:|---:|---:|
| GoEmotions-only (off-the-shelf, out-of-domain) | — | 0.1180 | — | 0.0396 | — |
| ArtELingo-native BERT, TRAIN split (likely leaky) | 93.67% | 0.2897 | — | 0.0582 | — |
| ArtELingo-native BERT, HELD-OUT val+test (this run; genre n=159) | 64.38% | 0.1693 | 0.1730 | 0.0138 | 0.2518 |

## Conclusion

Compared with the likely-leaky train split, held-out caption accuracy changed by +29.29% (93.67% to 64.38%) and held-out emotion AMI changed by +0.1204 (0.2897 to 0.1693). This supports mostly memorization (or a material generalization gap): at least one held-out drop exceeds the 5 percentage-point / 0.05-AMI practical thresholds, rather than remaining close to the train-split result.

Regardless of that generalization result, this is evidence only that a well-matched **supervised** emotion signal can survive routing through buddy-graph clustering. It is not evidence that buddy-graph/DEC can discover affect structure **unsupervised** from raw CLIP features. That scope limit is important because RedCaps, CoSiR's other main dataset, has no emotion labels with which to supervise a comparable classifier; even a perfectly strong ArtELingo held-out result therefore would not transfer to RedCaps.
