# ArtELingo PercepT Stage 1 stabilized P-Topic Formation pilot

Generated automatically, 2026-09-23 00:12:40.

## Setup and documented deviations

The train-only input is a 2816-dimensional fused vector. Content is the existing independently normalized CLIP image/text concatenation, normalized again as a whole; affect is a 768-dimensional embedding from masked-token mean pooling of `SamLowe/roberta-base-go_emotions`, then caption-mean pooling by painting. This is an embedding-level affect signal, a methodological upgrade over all earlier investigation pilots' 28-dimensional label probabilities. RoBERTa substitutes for the paper's ModernBERT-family GoEmotions encoder because this cached, validated project encoder shares the fine-tuning objective but not the backbone family.

Because this repository's ViT-B/32 CLIP content vector and the 768-dimensional affect vector have unequal dimensions, literal Eq. 2 summation is impossible without an unvalidated projection. The documented substitute is `L2_normalize(concat([h_C', h_C', h_E]))`: repeated content preserves the paper's 2:1 content:affect norm-budget weighting.

DEC uses convergence-controlled full-batch training (lr=1e-4; stop at `fraction_changed < 0.001`, ceiling 500) and retains the 67 highest-L2-norm centroids out of 100, exactly as in the collapsed base pilot. The sole stabilization intervention is a full-batch balanced-assignment penalty, `LAMBDA_BALANCE = 1.0`, on the KL divergence between the global mean soft assignment and uniform. `LAMBDA_RECONSTRUCTION` remains 1.0.

## Predeclared success criterion

**held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, both simultaneously.**

## Training trajectories

- Pretraining reconstruction: epoch 10: 0.000077; epoch 20: 0.000056; epoch 30: 0.000046; epoch 40: 0.000040; epoch 50: 0.000036; epoch 60: 0.000034; epoch 70: 0.000031; epoch 80: 0.000030; epoch 90: 0.000029; epoch 100: 0.000028.
- DEC stopped via the **stability criterion** at epoch **326** (threshold `fraction_changed < 0.001`, ceiling 500).
- Joint DEC: epoch 1: total=0.007581, KL=0.006926, recon=0.000027, balance=0.000628, fraction_changed=0.127765; epoch 25: total=0.019694, KL=0.019510, recon=0.000039, balance=0.000145, fraction_changed=0.023029; epoch 75: total=0.030950, KL=0.030638, recon=0.000056, balance=0.000256, fraction_changed=0.013615; epoch 125: total=0.042795, KL=0.042085, recon=0.000069, balance=0.000641, fraction_changed=0.007003; epoch 175: total=0.055152, KL=0.054042, recon=0.000077, balance=0.001034, fraction_changed=0.014609; epoch 225: total=0.068287, KL=0.066951, recon=0.000081, balance=0.001256, fraction_changed=0.004169; epoch 275: total=0.081765, KL=0.080284, recon=0.000083, balance=0.001399, fraction_changed=0.006270; epoch 325: total=0.094624, KL=0.092938, recon=0.000084, balance=0.001602, fraction_changed=0.003664; epoch 326: total=0.094881, KL=0.093190, recon=0.000084, balance=0.001607, fraction_changed=0.000847.

## Cluster-size distribution over training

Hard assignments are measured over all 100 pre-pruning DEC centers. Epoch 0 is the K-means initialization snapshot.

- Epoch 0: min 407; max 1,168; median 584.0; below 1%: 61/100.

- Epoch 25: min 31; max 10,932; median 173.0; below 1%: 79/100.

- Epoch 50: min 0; max 13,346; median 59.0; below 1%: 82/100.

- Epoch 75: min 0; max 15,428; median 21.0; below 1%: 85/100.

- Epoch 100: min 0; max 18,131; median 5.0; below 1%: 86/100.

- Epoch 125: min 0; max 20,413; median 1.0; below 1%: 85/100.

- Epoch 150: min 0; max 21,616; median 0.0; below 1%: 88/100.

- Epoch 175: min 0; max 22,377; median 0.0; below 1%: 91/100.

- Epoch 200: min 0; max 22,809; median 0.0; below 1%: 94/100.

- Epoch 225: min 0; max 23,037; median 0.0; below 1%: 95/100.

- Epoch 250: min 0; max 23,186; median 0.0; below 1%: 96/100.

- Epoch 275: min 0; max 23,333; median 0.0; below 1%: 97/100.

- Epoch 300: min 0; max 23,438; median 0.0; below 1%: 97/100.

- Epoch 325: min 0; max 23,555; median 0.0; below 1%: 97/100.

- Epoch 326: min 0; max 23,558; median 0.0; below 1%: 97/100.

## Cluster-size collapse detection

- Train: sizes range from 0 to 23,603 (median 0.0); 64/67 clusters are below 1% of assigned nodes. This is collapsed under the predeclared rule (>50% of clusters below 1%).
- Held-out: sizes range from 0 to 3,779 (median 0.0); 64/67 clusters are below 1% of assigned nodes. This is collapsed under the predeclared rule (>50% of clusters below 1%).

## Results

| architecture | split | emotion AMI | emotion V-measure | genre AMI | genre V-measure | collapse verdict | silhouette |
|---|---|---:|---:|---:|---:|---|---:|
| PercepT Stage 1 fused AE+DEC + balance (67 surviving topics) | train | 0.1210 | 0.1218 | 0.2596 | 0.2748 | Collapsed | 0.7238 |
| PercepT Stage 1 fused AE+DEC + balance (67 surviving topics) | held-out | 0.0925 | 0.0979 | 0.2631 | 0.3162 | Collapsed | 0.7128 |

## Before/after comparison with collapsed base pilot

| pilot | split | emotion AMI | genre AMI | collapse verdict |
|---|---|---:|---:|---|
| Collapsed base | train | 0.0478 | 0.3281 | Collapsed |
| Collapsed base | held-out | 0.0363 | 0.3081 | Collapsed |
| Stabilized balance | train | 0.1210 | 0.2596 | Collapsed |
| Stabilized balance | held-out | 0.0925 | 0.2631 | Collapsed |

Genre metrics use the genre-labelled overlap for each split (train n=1,144; held-out n=159).

The paper's reported 0.97 silhouette was measured on its own held-out fused-embedding input directly. This pilot instead reports a 128-dimensional DEC latent alongside label-based external metrics, so a large silhouette gap is not itself a failure signal; AMI against real labels is the primary criterion.

## Decision

**Collapsed.** The cluster-size collapse rule fired.
