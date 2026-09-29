# Reconstruction-anchored clustering head pilot (frozen buddy trunk)

Generated automatically, 2026-09-26 23:46:53.

## Method

The buddy trunk (Attention-h1) is frozen throughout: this pilot loads `attention_h1_embedding_snapshot.npz`'s already-trained 32-D `train_embedding_post`/`heldout_embedding_post` directly and never instantiates or trains a buddy model. A small, genuinely **undercomplete** autoencoder (32→24→16 encoder, 16→24→32 decoder) is pretrained by reconstruction alone (MSE, Adam lr=0.001, 100 epochs, batch 1024) before any clustering pressure. This differs from two prior attempts: the decoupled `ClusterHead` (un-detached and detached) trained DEC's KL loss with **no reconstruction term at all** and collapsed to near-zero cluster-latent silhouette; feeding buddy's embedding into PercepT's own unmodified autoencoder **had** reconstruction but an **overcomplete** 128-D latent (128>32) that let the decoder cheat without well-separated clusters, scoring negative silhouette back in buddy's native space. This pilot's 16-D latent is smaller than its 32-D input, and input/output are both 32-D so there is no PercepT-style dimensionality mismatch to distort the reconstruction/KL loss scale (LAMBDA_RECONSTRUCTION=1 fixed, matching PercepT's own convention at a matched scale this time).

K-means (100 clusters, `n_init=10`, `random_state=seed`) initializes on the pretrained encoder's clean train cluster-latent. The joint phase reuses PercepT's own `soft_assignments`/`target_distribution` (Student's-t kernel, appropriate for this unconstrained Euclidean latent) and highest-norm `prune_centers`, transcribed verbatim from `run_percept_stage1_pilot.py`. `LAMBDA_DEC` is screened over (0.1, 0.5, 1.0) with a linear 30-epoch warm-up, fixed Adam lr=0.001, 200 full-batch epochs (no early stop — no recall metric to plateau on here), checkpoint cadence=5. Silhouette is reported in both the 16-D cluster latent and the original 32-D buddy embedding, using the established seed-42 two-stage sampling convention (`np.random.default_rng(42)` draw of at most 6,000, `silhouette_score(sample_size=min(4000, len(idx)), random_state=42)`); the fused (32-D) silhouette is the headline comparable to every other pilot in this directory.

Held-out Pareto bar: emotion AMI > 0.1236 AND genre AMI > 0.1954.

## Single-seed screen (seed 42)

| LAMBDA_DEC | split | emotion AMI | genre AMI | cluster silhouette | fused silhouette | clusters | held-out Pareto bar |
|---:|---|---:|---:|---:|---:|---:|---|
| 0.1 | train | 0.1118 | 0.2006 | -0.0139 | -0.0826 | 66 | n/a |
| 0.1 | held-out | 0.1041 | 0.1264 | -0.0181 | -0.0872 | 65 | does not clear |
| 0.5 | train | 0.0984 | 0.2029 | -0.0103 | -0.0942 | 61 | n/a |
| 0.5 | held-out | 0.0924 | 0.1622 | -0.0104 | -0.0965 | 57 | does not clear |
| 1 | train | 0.0981 | 0.1997 | 0.0202 | -0.0967 | 55 | n/a |
| 1 | held-out | 0.0903 | 0.1635 | 0.0187 | -0.0901 | 51 | does not clear |

### Screen collapse diagnostics

| LAMBDA_DEC | split | min | max | median | below 1% / 67 | collapsed |
|---:|---|---:|---:|---:|---:|---|
| 0.1 | train | 0 | 5,126 | 414.0 | 44/67 | yes |
| 0.1 | held-out | 0 | 893 | 51.0 | 42/67 | yes |
| 0.5 | train | 0 | 10,392 | 291.0 | 47/67 | yes |
| 0.5 | held-out | 0 | 1,786 | 37.0 | 48/67 | yes |
| 1 | train | 0 | 8,716 | 246.0 | 48/67 | yes |
| 1 | held-out | 0 | 1,499 | 29.0 | 48/67 | yes |

## Winner selection

No screened value clears the held-out Pareto bar. Selected LAMBDA_DEC=0.1 for highest held-out fused silhouette among all three (-0.0872): best available, does not clear the Pareto bar.

## Four-seed stress of the selected LAMBDA_DEC

| seed | held-out emotion AMI | held-out genre AMI | held-out cluster silhouette | held-out fused silhouette | held-out clusters | Pareto bar |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 0.1041 | 0.1264 | -0.0181 | -0.0872 | 65 | does not clear |
| 7 | 0.1150 | 0.0920 | -0.0028 | -0.0720 | 66 | does not clear |
| 123 | 0.1307 | 0.0808 | 0.0250 | -0.0933 | 66 | does not clear |
| 2024 | 0.0991 | 0.1404 | 0.0090 | -0.0647 | 67 | does not clear |

### Held-out summary statistics

- Emotion AMI across four seeds: mean=0.1122; min=0.0991; max=0.1307.
- Genre AMI across four seeds: mean=0.1099; min=0.0808; max=0.1404.
- Cluster-latent silhouette across four seeds: mean=0.0033; min=-0.0181; max=0.0250.
- Fused-embedding silhouette across four seeds: mean=-0.0793; min=-0.0933; max=-0.0647.
- Both held-out Pareto bars clear in 0/4 seeds.

## Comparison against every prior DEC-hybrid attempt and established baselines

| result | held-out emotion AMI | held-out genre AMI | held-out silhouette |
|---|---:|---:|---:|
| Attention-h1 baseline (four-seed mean) | 0.1241 | 0.2406 | 0.0397 |
| Euclidean DEC hybrid (four-seed mean) | 0.1160 | 0.1321 | -0.0288 |
| vMF DEC hybrid (four-seed mean) | 0.1215 | 0.1504 | 0.0298 |
| Decoupled cluster head, un-detached (four-seed mean) | 0.0845 | 0.0423 | -0.1568 |
| Decoupled cluster head, detached (four-seed mean) | 0.0828 | 0.0481 | -0.1529 |
| PercepT-on-buddy-embedding (seed 42, native 32-D) | 0.1482 | 0.1765 | -0.0252 |
| PercepT replication faithful recipe (seed 42, its own 128-D latent) | 0.1092 | 0.3288 | 0.5120 |
| This pilot (four-seed mean, fused) | 0.1122 | 0.1099 | -0.0793 |

## Final verdict

This four-seed mean fused silhouette (-0.0793) does not beat the best prior DEC-hybrid attempt (0.0298). Both AMI Pareto bars clear in 0/4 seeds. Reconstruction anchoring plus an undercomplete latent is not a robust fix for the DEC-hybrid direction by this investigation's standing bar. This is now the fifth (Euclidean, vMF, decoupled un-detached, decoupled detached) or sixth (including PercepT-on-buddy-embedding) attempt at attaching a DEC-style clustering objective to Attention-h1's embedding, and every one has failed to produce a robust, non-collapsed, Pareto-clearing result. This investigation has now tested every mechanism the brainstorm memo identified for this line of attack (geometry correction, gradient isolation, reconstruction anchoring at matched and mismatched scale) — this is a well-evidenced, settled negative result for attaching DEC-style clustering losses to Attention-h1, not an open question.
