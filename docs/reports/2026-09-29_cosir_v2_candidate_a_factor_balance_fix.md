# CoSiR v2 Candidate A: factor balance fix on ArtELingo

**Verdict: partly fixed.** PCA whitening with a `0.1` usage-balance weight reduced the top-two-factor activation-mass share from **87.7% to 8.66%** on the same 308,723 CLIP training pairs. The max single-factor share is **4.35%**. The previously working anti-split and dead-factor controls did not break: modality-private factors fell from 1/32 to 0/32 and dead factors remained 0/32. Community-spanning factors rose from 19/32 (59.4%) to 25/32 (78.1%). **Reconstruction is substantially worse:** mean relative L2 error on the whitened features is 0.9929/0.9931, versus 0.5542/0.5097 on raw features. The factors therefore have more balanced use, but the whitened 512-dimensional targets are barely reconstructed by this 32-factor dictionary. This is not yet a clean basis for the condition-interface stage.

## Run and comparison

The reproducible script is `src/test/20260929_factor_balance_fix/run_revalidation.py`. As in Task 3, it loads `FeatureManager` from `/data/SSD2/pre_extract/artelingo/features`, joins `artelingo_train.json` by unique in-range stored sample IDs, rebuilds the Block 1 content graph from **raw CLIP image and text features**, trains Stage 1 with its defaults, and detects communities from the resulting embeddings. The graph again has 3,130,544 edges and Stage 1 yields 21 communities. No other encoder or Block 1 ablation setting was used.

Only the factor-training inputs changed: image and text CLIP arrays were independently fit and transformed with full-rank, 512-component, float64 PCA whitening. The fitted PCA objects are returned by `pca_whiten` for reuse. Float64 is necessary here because float32 PCA estimated zero eigenvalues on these nearly rank-deficient CLIP arrays and produced component variances up to about `1e8`; the corrected run checks near-unit variance before training. Observed component-variance ranges were 0.999997–0.999997 for images and 0.999997–1.041514 for text. Factor training otherwise kept the Task 3 defaults: 32 factors, seed 42, 2,000 edge-sampled epochs, batch size 1,024, Adam learning rate `1e-3`, and the original five loss weights. The new `lambda_usage_balance=0.1` was chosen before the real run, at the same order as the `0.1` anti-split weight, because both are secondary balance controls. No loss weight was tuned after looking at the results.

| Measure | Task 3 raw CLIP | Balanced, whitened CLIP |
|---|---:|---:|
| Max single-factor activation-mass share | Not reported | **4.35%** (factor 7) |
| Top-two-factor activation-mass share | **87.7%** | **8.66%** (factors 7, 31) |
| Modality-private factors | **1/32** | **0/32** |
| Dead factors | **0/32** | **0/32** |
| Factors with at least half their mass in one community | **13/32 (40.6%)** | **7/32 (21.9%)** |
| Active factors spanning communities | **19/32 (59.4%)** | **25/32 (78.1%)** |
| Relative reconstruction L2, image / text | **0.5542 / 0.5097**, raw target | **0.9929 / 0.9931**, whitened target |
| Code cosine, matched / shuffled | **0.9263 / 0.6060** | **0.8900 / 0.5881** |
| Matched-minus-shuffled cosine | **0.3203** | **0.3019** |
| First five / last five sampled training-loss means | **1.027990 / 0.199325** | **1.699950 / 0.771236** |

The anti-split and dead-factor rules are identical to Task 3: a factor is dead when both modality means are below `1e-4`; an active factor is modality-private when its smaller modality mean is at most 5% of its larger one. Seven factors (11, 14, 15, 22, 24, 27, 30) still place at least half of their combined activation mass in one Stage 1 community; all other active factors span communities under that same 50% rule. A spanning factor is not thereby proven semantically meaningful. The shuffled-pair control uses the same seed-42 permutation as Task 3. Its matched advantage fell by about 0.0184 cosine points, so paired agreement remains present but somewhat weaker.

Whitening changes which directions contribute to the reconstruction target: every principal component is scaled to approximately unit variance, including formerly low-variance directions. Relative L2 remains the same definition (mean per-row reconstruction error divided by target norm), but the raw- and whitened-target values describe different tasks. The near-1.0 whitened error indicates that balancing activation did not preserve useful reconstruction under this full-rank target. The lower absolute matched cosine and slightly smaller matched advantage also warrant caution despite the improved usage and community counts.

## Numeric correction and limits

The first execution used the exact PCA interface on float32 arrays. On these data, float32 PCA was numerically unstable: estimated zero eigenvalues made component variances reach about `1e8`, with huge factor losses and uninformative agreement. That execution was discarded as an invalid whitening check; no regularizer weight changed. A regression test for a small but nonzero component was added, `pca_whiten` was changed to fit in float64, and the same `0.1` configuration was rerun. This report uses only the corrected run's diagnostics.

This is an in-sample, single-seed test on CLIP features. The factors have not been semantically named or evaluated on held-out data. The diagnostic thresholds are operational rules, not evidence that all active or spanning factors are useful. Wall-clock time for the corrected run was 134.6 seconds, including 9.3 seconds for the graph, 1.8 for Stage 1, 94.8 for community detection, 2.6 for whitening, and 24.0 for factor training and full-code generation.
