# ArtELingo single-modality buddy-graph pilot

Generated automatically, 2026-09-22 13:25:28.

**Setup:** Three independent mutual-kNN graphs (K=20) are built from CLIP image features, CLIP text features, and mean-pooled GoEmotions sigmoid probabilities respectively. Each graph receives the same minimum-degree and connectivity repairs as the normal buddy graph, then Leiden uses seed=42. Genre metrics use the 1144-painting genre-labelled overlap.

| graph | emotion_AMI | emotion_Vmeasure | genre_AMI | genre_Vmeasure |
|---|---:|---:|---:|---:|
| Existing CLIP img+txt UNION baseline (reference) | 0.0593 | — | 0.4384 | — |
| CLIP-image-only | 0.0540 | 0.0549 | 0.4290 | 0.4546 |
| CLIP-text-only | 0.0638 | 0.0657 | 0.2931 | 0.3488 |
| GoEmotions-affect-only | 0.1180 | 0.1189 | 0.0396 | 0.0937 |

## Interpretation

The GoEmotions-only graph's emotion AMI ceiling is 0.1180. Compare this directly with the affect-fusion sweep to determine whether the shared graph diluted a substantially stronger affect structure or was already approaching the single-signal ceiling. Its genre AMI is 0.0396, which is not near zero; this indicates whether affect and genre specialize in different structure rather than the affect signal being merely noise.
