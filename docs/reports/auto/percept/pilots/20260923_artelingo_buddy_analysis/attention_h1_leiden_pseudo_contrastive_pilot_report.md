# Attention-h1 Leiden pseudo-contrastive pilot

Generated 2026-09-24 14:27:18.

## Method

The original Attention-h1 student learned from CLIP-content and GoEmotions-affect teacher graph InfoNCE losses, then used Leiden only after training. This pilot tests whether recycled Leiden pseudo-label positives tighten and separate the learned embedding by adding a third, same-community symmetric InfoNCE loss. The original two losses and recall-based plateau rule remain in place.

Schedule: recluster every 20 epochs starting at epoch 1; LAMBDA_CLUSTER_MAX=1.0; CLUSTER_WARMUP_EPOCHS=50; DOMINANT_FRACTION_GUARD=0.8; batch=1024; temperature=0.1; seed=42; MAX_EPOCHS=200; CHECKPOINT_EVERY=5; learning_rate=0.001.

Stopped: reached MAX_EPOCHS=200. Wall-clock time: 432.7 seconds.

## Comparison with established Attention-h1 baseline

Baseline numbers are literal constants from the prior `run_attention_h1_embedding_snapshot_pilot.py` run. Silhouette uses a seed-42 6,000-point draw followed by `silhouette_score` with `sample_size=min(4000, len(idx))`, `random_state=42`.

| metric | baseline | this run | absolute difference |
|---|---:|---:|---:|
| train emotion AMI | 0.1351 | 0.1189 | 0.0162 |
| train genre AMI | 0.2397 | 0.2645 | 0.0248 |
| held-out emotion AMI | 0.1249 | 0.1165 | 0.0084 |
| held-out genre AMI | 0.2404 | 0.2351 | 0.0053 |
| train silhouette pre | -0.0024 | -0.0038 | 0.0014 |
| train silhouette post | 0.0253 | 0.0776 | 0.0523 |
| held-out silhouette pre | 0.0302 | 0.0247 | 0.0055 |
| held-out silhouette post | 0.0392 | 0.0877 | 0.0485 |

## Community counts and reclustering

Train Leiden communities: pre=20, post=17. Held-out Leiden communities: pre=13, post=16.

Degenerate recluster passes skipped: 0. Successful recluster trajectory:

| epoch | communities | dominant fraction |
|---:|---:|---:|
| 1 | 20 | 0.1357 |
| 21 | 14 | 0.1634 |
| 41 | 13 | 0.1384 |
| 61 | 15 | 0.1335 |
| 81 | 15 | 0.1480 |
| 101 | 17 | 0.1079 |
| 121 | 18 | 0.1067 |
| 141 | 18 | 0.1039 |
| 161 | 18 | 0.1070 |
| 181 | 18 | 0.1098 |

Train post-training silhouette improved versus the baseline (0.0776 vs 0.0253; signed change +0.0523).
Held-out post-training silhouette improved versus the baseline (0.0877 vs 0.0392; signed change +0.0485).
