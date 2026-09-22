# ArtELingo DEC-vs-Leiden pilot

Generated automatically, 2026-09-22 15:17:11.

**Setup:** The same 61,402 x 28 mean-pooled GoEmotions sigmoid-probability nodes used by the GoEmotions-only Leiden pilot were clustered with a small 28→64→32→16→32→64→28 autoencoder and DEC (K=28). Genre metrics use the 1144-painting genre-labelled overlap.

## Training trajectories

- Pretraining mean reconstruction loss: epoch 1: 0.027206, epoch 10: 0.001738, epoch 20: 0.001014, epoch 30: 0.000801, epoch 40: 0.000614, epoch 50: 0.000557.
- Joint DEC losses: epoch 1: total=0.080430, KL=0.079881, recon=0.000549; epoch 10: total=0.091904, KL=0.090930, recon=0.000974; epoch 20: total=0.100333, KL=0.099317, recon=0.001017; epoch 40: total=0.112656, KL=0.111594, recon=0.001062; epoch 60: total=0.124824, KL=0.123697, recon=0.001126; epoch 80: total=0.138097, KL=0.136722, recon=0.001376; epoch 100: total=0.151989, KL=0.150216, recon=0.001774.

## Cluster-size collapse detection

**Prominent collapse check:** cluster sizes range from 38 to 5,374 nodes (median 2078.5); 3 of 28 clusters contain <1% of nodes (<614). This is not a collapse under the predeclared rule (fewer than half of clusters must be below that threshold).

Final latent silhouette score (full 61,402 nodes): 0.1700.

## Comparison

| method | emotion AMI | emotion V-measure | genre AMI | genre V-measure |
|---|---:|---:|---:|---:|
| Leiden on GoEmotions-only (reference) | 0.1180 | — | 0.0396 | — |
| DEC on GoEmotions-only (this run) | 0.1258 | 0.1264 | 0.0365 | 0.0739 |

## Decision

DEC is not a real win under the predeclared criteria: emotion AMI=0.1258 did not clear AMI > 0.177. Both AMI improvement and non-collapse were required.
