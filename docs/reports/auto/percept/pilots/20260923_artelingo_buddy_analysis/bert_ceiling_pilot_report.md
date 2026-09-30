# ArtELingo-native BERT emotion-classifier ceiling pilot

Generated automatically, 2026-09-22 16:04:20.

## Important interpretation caveat: likely train-split leakage

**This BERT checkpoint was almost certainly trained on ArtELingo's official train split, the same split used to build `artelingo_train.json` and these painting nodes. Its caption accuracy and graph ceiling may therefore partly reflect memorization rather than generalization. This is not a fair apples-to-apples generalization comparison with GoEmotions. It answers the narrower question: given the best available in-domain emotion signal, what ceiling does buddy-graph/Leiden clustering reach at all?**

## Per-caption classifier sanity check

The anonymous `LABEL_n` outputs were empirically mapped by assigning each predicted index to its most common ground-truth emotion among captions with that prediction. This check occurs before painting-level pooling.

| checkpoint index | recovered emotion | supporting caption rows |
|---|---|---:|
| LABEL_0 | amusement | 28,708 |
| LABEL_1 | awe | 45,568 |
| LABEL_2 | contentment | 82,982 |
| LABEL_3 | excitement | 22,427 |
| LABEL_4 | anger | 3,948 |
| LABEL_5 | disgust | 13,898 |
| LABEL_6 | fear | 27,565 |
| LABEL_7 | sadness | 31,773 |
| LABEL_8 | something else | 32,327 |

**Recovered-mapping top-1 accuracy: 93.67% (289,196/308,723 English caption rows).**

## Graph-based ceiling

Setup: 61,402 deduplicated painting nodes, mean-pooled 9-way BERT softmax probabilities, single-modality mutual-kNN graph (K=20), graph repair with alpha=0.5, and Leiden seed=42. Genre metrics use the 1,144-painting genre-labelled overlap.

| signal / ceiling | emotion AMI | emotion V-measure | genre AMI | genre V-measure |
|---|---:|---:|---:|---:|
| GoEmotions-only (off-the-shelf, out-of-domain) Leiden ceiling | 0.1180 | — | 0.0396 | — |
| ArtELingo-native BERT (in-domain, train-set, likely leaky) Leiden ceiling | 0.2897 | 0.2916 | 0.0582 | 0.1696 |

## Interpretation

If this ceiling is also modest (similar to 0.12–0.15) despite high per-caption accuracy and in-domain training, the clustering method—not signal quality—is the likely bottleneck. If it is substantially higher, the GoEmotions result was mainly limited by domain mismatch and signal quality. This run supports the signal-quality/domain-mismatch reading: the in-domain BERT graph ceiling is substantially above the GoEmotions-only ceiling, so the out-of-domain affect signal was likely a main limiter.
