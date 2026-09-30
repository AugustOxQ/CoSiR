# ArtELingo affect-aware buddy-graph pilot

Generated automatically, 2026-09-22 12:56:17.

**Setup:** GoEmotions (`SamLowe/roberta-base-go_emotions`) sigmoid logits are mean-pooled over each painting's English captions, L2-normalized, and concatenated with L2-normalized CLIP text features. Image features remain pure CLIP. Buddy graph: K=20, alpha=0.5, Leiden seed=42.

| affect_weight | community_vs_emotion_AMI | community_vs_emotion_Vmeasure | community_vs_genre_AMI | community_vs_genre_Vmeasure |
|---:|---:|---:|---:|---:|
| 0.0 | 0.0593 | 0.0600 | 0.4384 | 0.4572 |
| 0.5 | 0.0614 | 0.0621 | 0.4045 | 0.4199 |
| 1.0 | 0.0759 | 0.0763 | 0.4025 | 0.4163 |
| 2.0 | 0.1119 | 0.1125 | 0.2042 | 0.2322 |
| 4.0 | 0.1160 | 0.1166 | 0.0867 | 0.1237 |

## CLIP-only reproduction check

The affect_weight=0.0 control reproduced the original CLIP-only baseline within ±0.005: emotion AMI=0.0593 vs. 0.0593, genre AMI=0.4384 vs. 0.4384 (genre n=1144).

## Conclusion

No. No affect weight achieved AMI > 0.09 (the predeclared 50% relative improvement threshold over the 0.0593 full-graph baseline) while retaining at least 80% of the 0.4384 genre AMI baseline. The non-tied emotion reference is AMI=0.0758; inspect the table for trade-offs.
