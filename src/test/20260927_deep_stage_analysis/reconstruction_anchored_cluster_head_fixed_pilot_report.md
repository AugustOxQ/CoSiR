# Gap 3 — reconstruction-anchored cluster head, BOTH known bugs fixed

Generated 2026-09-28 00:39:54. Companion/fixed re-run of [`reconstruction_anchored_cluster_head_pilot_report.md`](../../test/20260923_artelingo_buddy_analysis/reconstruction_anchored_cluster_head_pilot_report.md). Fixes: `prune_centers` now keeps the lowest-norm (paper-faithful) centers; the reconstruction loss now sums over the feature dimension before averaging over the batch, matching the paper's L_R = ||h - h_hat||^2 instead of PyTorch's default mean-over-all-elements reduction. Original (buggy) file and report are left unmodified for provenance.

## Single-seed screen (seed 42)

| LAMBDA_DEC | split | emotion AMI | genre AMI | cluster silhouette | fused silhouette | clusters | held-out Pareto bar |
|---:|---|---:|---:|---:|---:|---:|---|
| 0.1 | train | 0.1334 | 0.1802 | 0.0190 | -0.0156 | 67 | n/a |
| 0.1 | held-out | 0.1197 | 0.1500 | 0.0165 | -0.0198 | 67 | does not clear |
| 0.5 | train | 0.1268 | 0.1767 | -0.0069 | -0.0517 | 67 | n/a |
| 0.5 | held-out | 0.1114 | 0.1322 | -0.0097 | -0.0557 | 67 | does not clear |
| 1 | train | 0.1200 | 0.1749 | -0.0126 | -0.0713 | 67 | n/a |
| 1 | held-out | 0.1071 | 0.1474 | -0.0107 | -0.0668 | 67 | does not clear |

## Winner selection

No screened value clears the held-out Pareto bar. Selected LAMBDA_DEC=0.1 for highest held-out fused silhouette among all three (-0.0198): best available, does not clear the Pareto bar.

## Four-seed stress of the selected LAMBDA_DEC (fixed)

| seed | held-out emotion AMI | held-out genre AMI | held-out cluster silhouette | held-out fused silhouette | held-out clusters | Pareto bar |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 0.1197 | 0.1500 | 0.0165 | -0.0198 | 67 | does not clear |
| 7 | 0.1173 | 0.0648 | 0.0255 | -0.0082 | 67 | does not clear |
| 123 | 0.1178 | 0.0986 | 0.0216 | 0.0026 | 67 | does not clear |
| 2024 | 0.1047 | 0.1464 | 0.0174 | -0.0215 | 67 | does not clear |

- Both held-out Pareto bars clear in 0/4 seeds.

## Comparison against the original (buggy) run

| | held-out emotion AMI (4-seed mean) | held-out genre AMI (4-seed mean) | held-out fused silhouette (4-seed mean) |
|---|---:|---:|---:|
| Original (buggy prune_centers + loss scale) | 0.1122 | 0.1099 | -0.0793 |
| **Fixed (this run)** | **0.1149** | **0.1150** | **-0.0117** |

## Verdict

**The two bug fixes do not materially change this attempt's conclusion.** Fixed fused silhouette -0.0117 vs. original -0.0793 (+0.0676 if both finite), Pareto bar still clears in only 0/4 seeds. This is consistent with the §4 diagnosis that the failure mode here is the embedding geometry / DEC self-sharpening dynamics, not the specific center-selection or loss-scale bugs -- those bugs affect *which* centers survive pruning and how strongly reconstruction anchors the latent, but not the underlying separability the training dynamics produce. The §4/§6b negative conclusion for the DEC-hybrid direction is not overturned by this fix.
