# ArtELingo alternate affect-encoder pilot

Generated automatically, 2026-09-22 16:59:52.

**Setup:** `j-hartmann/emotion-english-distilroberta-base` supplies a 7-way softmax distribution over Ekman's six basic emotions (anger, disgust, fear, joy, sadness, surprise) plus neutral. Its mixed training domains are broader than Reddit-only GoEmotions and include Crowdflower, MELD dialogue transcripts, ISEAR personal narratives, SemEval-2018, and GoEmotions. **This is not a fully independent comparison: GoEmotions is one ingredient in the model's training mix, so any gain should be read as a broader-domain-trained encoder result, not proof from a completely unrelated encoder.** Mean-pooled per-painting affect vectors are L2-normalized, then used alone to build a mutual-kNN graph (K=20) with the same repairs as the reference pilot; Leiden uses seed=42. Genre metrics use the 1144-painting genre-labelled overlap.

| signal | emotion AMI | emotion V-measure | genre AMI | genre V-measure |
|---|---:|---:|---:|---:|
| GoEmotions-affect-only (Leiden, reference) | 0.1180 | 0.1189 | 0.0396 | 0.0937 |
| GoEmotions-affect-only (DEC v2, reference — different clustering method, not directly comparable to this pilot's Leiden-only rows) | 0.1492 | 0.1498 | 0.0554 | 0.0930 |
| ArtELingo-native BERT, held-out (reference — in-domain supervised, upper bound) | 0.1693 | 0.1730 | 0.0138 | 0.2518 |
| j-hartmann-affect-only (Leiden, this run) | 0.1080 | 0.1088 | 0.0369 | 0.0859 |

## Conclusion

The new encoder's Leiden emotion AMI is 0.1080; no improvement or worse. The predeclared bar is AMI > 0.177, a 50% relative improvement over the GoEmotions single-modality ceiling of 0.1180. Genre AMI dropped relative to the GoEmotions Leiden reference (0.0369 vs. 0.0396). GoEmotions is part of this encoder's training mix, so this remains a non-independent comparison and any improvement must be interpreted cautiously as evidence for a broader-domain-trained encoder, not a completely unrelated one.
