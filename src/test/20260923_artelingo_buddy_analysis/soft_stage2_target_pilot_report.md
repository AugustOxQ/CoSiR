# Soft, uncertainty-preserving Stage 2 target pilot

## Method

Uses the same frozen buddy train communities, patch-feature cache, AttentionPoolingMapper architecture, Adam lr/epoch budget, and downstream logistic-regression probe protocol as `run_buddy_percept_downstream_probe_pilot.py`. The hard one-hot and mean-pooled control rows are cited from that pilot's report, not rerun. Two new mapper targets are compared:

- **Soft vote-frequency**: each train painting's own k=20 nearest OTHER train paintings' (cosine, excluding itself) community-label frequency distribution over the 19 frozen train communities, used as a soft cross-entropy target (`-(target * log_softmax(logits)).sum(dim=1).mean()`).
- **Sparse multi-hot**: the same distribution thresholded at >= 0.1000 (2/20 votes, mirroring PercepT's own "at least 2 votes" multi-label convention at its own topic count), trained with `BCEWithLogitsLoss` as PercepT's original Stage 2 pilot does for its own multi-hot DEC targets.

Multi-hot label statistics: mean 2.067 labels per painting, 65.3% of paintings multi-labeled.

## Results

| target | emotion AMI | emotion accuracy | genre AMI | genre accuracy |
|---|---:|---:|---:|---:|
| hard one-hot (cited) | 0.0231 | 0.3350 | 0.2625 | 0.4403 |
| soft vote-frequency | 0.0239 | 0.3359 | 0.2610 | 0.4403 |
| sparse multi-hot | 0.0077 | 0.3280 | 0.2430 | 0.4340 |
| mean-pooled control (cited) | 0.0685 | 0.3882 | 0.3399 | 0.5660 |

Final train loss: soft=2.435665; multi-hot=0.318986.

## Verdict

Soft vote-frequency beats the hard one-hot target on 2/4 metrics; sparse multi-hot beats it on 0/4 metrics. Neither soft target is a clear, consistent improvement over the plain hard label by this single-seed/single-split screen. Softening the Stage 2 target is not a high-value lever on this evidence.
