# ArtELingo buddy Stage 2 patch-attention pilot

Generated automatically, 2026-09-27 00:54:06.

## Frozen Stage-1 topics and held-out transfer

The frozen `attention_h1_embedding_snapshot.npz` supplies post-training 32-D train and held-out embeddings and the train Leiden partition (communities 0–18). The held-out labels are assigned to the train vocabulary with the existing cosine k-NN majority-vote transfer at k=20; 19/19 train communities appear in held-out and there were 0 degenerate tie fallbacks. Both splits were checked for exactly matching unique painting-ID sets and reindexed by painting ID into the fresh `load_dedup_features()` order used by the patch caches.

## Frozen multi-label target statistics

| split | mean labels | median labels | max labels | fraction multi-labeled |
|---|---:|---:|---:|---:|
| train | 1.000 | 1.000 | 1 | 0.00% |
| held-out | 1.000 | 1.000 | 1 | 0.00% |

Buddy has exactly one label per painting by construction: Leiden communities form a hard partition. PercepT's own Stage 2 report also records 1.000/1.000/1 labels and 0.00% multi-labeled paintings in both splits, despite its multi-hot threshold rule. Thus both realized Stage 2 target sets are single-label, making buddy's hard partition a structurally faithful comparison rather than an apples-to-oranges one.

## Attention-pooling mapper training

The mapper consumes only cached `[painting, 50, 512]` image patch tokens. PercepT's unchanged single-query attention-pooling architecture has a 19-topic head. It trains for 100 full-batch epochs with Adam at learning rate 0.001, using **BCEWithLogitsLoss against one-hot targets**, as in PercepT Stage 2.

| epoch | full-batch BCE loss |
|---:|---:|
| 10 | 0.518788 |
| 20 | 0.384487 |
| 30 | 0.304364 |
| 40 | 0.259604 |
| 50 | 0.234871 |
| 60 | 0.220764 |
| 70 | 0.212254 |
| 80 | 0.206773 |
| 90 | 0.203010 |
| 100 | 0.200270 |

## Held-out per-topic AUC

| topic | mapper AUC | marginal-frequency baseline AUC |
|---:|---:|---:|
| 0 | 0.8123 | 0.5000 |
| 1 | 0.6063 | 0.5000 |
| 2 | 0.9043 | 0.5000 |
| 3 | 0.6706 | 0.5000 |
| 4 | 0.5264 | 0.5000 |
| 5 | 0.7683 | 0.5000 |
| 6 | 0.4780 | 0.5000 |
| 7 | 0.5605 | 0.5000 |
| 8 | 0.4526 | 0.5000 |
| 9 | 0.6116 | 0.5000 |
| 10 | 0.6748 | 0.5000 |
| 11 | 0.5102 | 0.5000 |
| 12 | 0.6922 | 0.5000 |
| 13 | 0.8080 | 0.5000 |
| 14 | 0.4375 | 0.5000 |
| 15 | 0.4983 | 0.5000 |
| 16 | 0.5478 | 0.5000 |
| 17 | 0.5437 | 0.5000 |
| 18 | 0.2541 | 0.5000 |

| scorer | macro AUC | min | median | max | skipped topics |
|---|---:|---:|---:|---:|---:|
| patch-attention mapper | 0.5978 | 0.2541 | 0.5605 | 0.9043 | 0 |
| train-marginal baseline | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0 |

## Direct comparison with PercepT Stage 2

The PercepT values below come from `../20260922_percept_topic_pipeline/percept_stage2_pilot_report.md`. Buddy has 19 topics; PercepT has 40.

| system and scorer | macro AUC | min | median | max | skipped topics |
|---|---:|---:|---:|---:|---:|
| buddy patch-attention mapper | 0.5978 | 0.2541 | 0.5605 | 0.9043 | 0 |
| buddy train-marginal baseline | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0 |
| PercepT patch-attention mapper | 0.5690 | 0.3618 | 0.5536 | 0.8744 | 0 |
| PercepT train-marginal baseline | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0 |

## Conclusion

**Buddy's image-only Stage 2 mapper meaningfully beats its train-marginal baseline.** Its macro AUC is 0.5978 versus 0.5000, a difference of +0.097758; the predeclared practical margin is 0.01. PercepT's macro AUC is 0.5690; buddy minus PercepT is +0.0288. Buddy's image-only topic space is better than PercepT's by this macro-AUC comparison.
