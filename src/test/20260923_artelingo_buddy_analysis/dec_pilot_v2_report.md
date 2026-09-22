# ArtELingo DEC pilot v2 — convergence-controlled

Generated automatically, 2026-09-22 15:33:48.

**Setup:** The same 61,402 x 28 mean-pooled GoEmotions sigmoid-probability nodes used by the GoEmotions-only Leiden and DEC-v1 pilots were clustered with the same 28→64→32→16→32→64→28 autoencoder and K=28 DEC centers. Genre metrics use the 1144-painting genre-labelled overlap.

## Convergence and training trajectories

DEC stopped via the **stability criterion** at epoch **185** (threshold: fraction_changed < 0.001; ceiling: 500 epochs).

- Pretraining mean reconstruction loss: epoch 1: 0.025519, epoch 10: 0.001570, epoch 20: 0.000820, epoch 30: 0.000627, epoch 40: 0.000566, epoch 50: 0.000465.
- Hard-assignment fraction_changed: epoch 1: 0.015179; epoch 10: 0.007198; epoch 60: 0.001873; epoch 110: 0.001596; epoch 160: 0.001352; epoch 185: 0.000993.
- Joint DEC losses: epoch 1: total=0.065385, KL=0.064921, recon=0.000464; epoch 25: total=0.069178, KL=0.068708, recon=0.000471; epoch 75: total=0.075769, KL=0.075288, recon=0.000480; epoch 125: total=0.081020, KL=0.080537, recon=0.000483; epoch 175: total=0.085376, KL=0.084886, recon=0.000490; epoch 185: total=0.086184, KL=0.085692, recon=0.000492.

## Cluster-size collapse detection

**Prominent collapse check:** cluster sizes range from 693 to 6,580 nodes (median 1902.5); 0 of 28 clusters contain <1% of nodes (<614). This is not a collapse under the predeclared rule (fewer than half of clusters must be below that threshold).

Final latent silhouette score (full 61,402 nodes): 0.1284.

## Comparison

| method | emotion AMI | emotion V-measure | genre AMI | genre V-measure |
|---|---:|---:|---:|---:|
| Leiden on GoEmotions-only | 0.1180 | — | 0.0396 | — |
| DEC v1 (100 fixed epochs, lr=1e-3, non-converged) | 0.1258 | — | 0.0365 | — |
| DEC v2 (converged, lr=1e-4) | 0.1492 | 0.1498 | 0.0554 | 0.0930 |

## Decision

Convergence did occur: the stability criterion fired before the epoch ceiling. The result does not clear the predeclared emotion AMI > 0.177 bar (actual AMI=0.1492). v2 confirms that the v1 conclusion holds even with proper convergence control.
