# ArtELingo PercepT Stage 1 P-Topic Formation pilot

Generated automatically, 2026-09-22 23:11:20.

## Setup and documented deviations

The train-only input is a 2816-dimensional fused vector. Content is the existing independently normalized CLIP image/text concatenation, normalized again as a whole; affect is a 768-dimensional embedding from masked-token mean pooling of `SamLowe/roberta-base-go_emotions`, then caption-mean pooling by painting. This is an embedding-level affect signal, a methodological upgrade over all earlier investigation pilots' 28-dimensional label probabilities. RoBERTa substitutes for the paper's ModernBERT-family GoEmotions encoder because this cached, validated project encoder shares the fine-tuning objective but not the backbone family.

Because this repository's ViT-B/32 CLIP content vector and the 768-dimensional affect vector have unequal dimensions, literal Eq. 2 summation is impossible without an unvalidated projection. The documented substitute is `L2_normalize(concat([h_C', h_C', h_E]))`: repeated content preserves the paper's 2:1 content:affect norm-budget weighting.

DEC uses convergence-controlled full-batch training (lr=1e-4; stop at `fraction_changed < 0.001`, ceiling 500) rather than the paper's fixed 200 epochs. This is the deliberate, previously validated project deviation. Norm-threshold pruning is approximated by retaining the 67 highest-L2-norm centroids out of 100, matching the paper's reported retention rate because its exact threshold rule is unstated.

## Predeclared success criterion

**held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, both simultaneously.**

## Training trajectories

- Pretraining reconstruction: epoch 10: 0.000077; epoch 20: 0.000056; epoch 30: 0.000046; epoch 40: 0.000040; epoch 50: 0.000036; epoch 60: 0.000034; epoch 70: 0.000031; epoch 80: 0.000030; epoch 90: 0.000029; epoch 100: 0.000028.
- DEC stopped via the **stability criterion** at epoch **275** (threshold `fraction_changed < 0.001`, ceiling 500).
- Joint DEC: epoch 1: total=0.006925, KL=0.006898, recon=0.000027, fraction_changed=0.080519; epoch 25: total=0.019993, KL=0.019953, recon=0.000040, fraction_changed=0.022214; epoch 75: total=0.031034, KL=0.030977, recon=0.000057, fraction_changed=0.016449; epoch 125: total=0.041286, KL=0.041217, recon=0.000070, fraction_changed=0.012280; epoch 175: total=0.049895, KL=0.049815, recon=0.000079, fraction_changed=0.023029; epoch 225: total=0.057920, KL=0.057836, recon=0.000084, fraction_changed=0.001922; epoch 275: total=0.065077, KL=0.064992, recon=0.000085, fraction_changed=0.000847.

## Cluster-size collapse detection

- Train: sizes range from 0 to 30,498 (median 0.0); 65/67 clusters are below 1% of assigned nodes. This is collapsed under the predeclared rule (>50% of clusters below 1%).
- Held-out: sizes range from 0 to 5,057 (median 0.0); 65/67 clusters are below 1% of assigned nodes. This is collapsed under the predeclared rule (>50% of clusters below 1%).

## Results

| architecture | split | emotion AMI | emotion V-measure | genre AMI | genre V-measure | collapse verdict | silhouette |
|---|---|---:|---:|---:|---:|---|---:|
| PercepT Stage 1 fused AE+DEC (67 surviving topics) | train | 0.0478 | 0.0491 | 0.3281 | 0.3384 | Collapsed | 0.8514 |
| PercepT Stage 1 fused AE+DEC (67 surviving topics) | held-out | 0.0363 | 0.0436 | 0.3081 | 0.3367 | Collapsed | 0.8550 |

Genre metrics use the genre-labelled overlap for each split (train n=1,144; held-out n=159).

The paper's reported 0.97 silhouette was measured on its own held-out fused-embedding input directly. This pilot instead reports a 128-dimensional DEC latent alongside label-based external metrics, so a large silhouette gap is not itself a failure signal; AMI against real labels is the primary criterion.

## Decision

**Collapsed.** The cluster-size collapse rule fired.
