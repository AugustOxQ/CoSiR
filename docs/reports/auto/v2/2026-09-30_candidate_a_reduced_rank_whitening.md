# CoSiR v2 Candidate A: reduced-rank whitening on ArtELingo

**Verdict: not fixed.** At the primary 99% cumulative-variance threshold, the smallest shared width is **K=433/512** (independent image/text minima: 400/433). The top-two-factor activation-mass share is **7.76%**, lower than 87.7% on raw CLIP (Task 3) and 8.66% after full-rank whitening (Task 4). Anti-split and dead-factor checks remain at **0/32** each, and **31/32 (96.9%)** active factors span communities. Yet mean relative reconstruction L2 is **0.9912/0.9915** for image/text, essentially the full-rank result of 0.9929/0.9931 and far from the raw-feature result of 0.5542/0.5097. The reconstruction trade-off persists at all three prespecified thresholds.

## Method and width choice

Run `src/test/20260930_factor_reduced_rank_whitening/run_revalidation.py` in the CoSiR environment. It loads the same 308,723 ArtELingo CLIP image/text training pairs from `FeatureManager` at `/data/SSD2/pre_extract/artelingo/features`, uses the same unique in-range sample-ID positional join to `/data/PDD/artelingo/artelingo_train.json`, and rebuilds the Block 1 content graph from **raw CLIP** features. The rebuilt graph has 3,130,544 edges; default Stage 1 training and community detection yield 21 communities. The graph and labels are reused for each factor run. No cross-encoder features or ablation settings enter this experiment.

Image and text PCA are fitted independently at full rank in float64, as in Task 4. For each threshold, `select_pca_rank` finds each modality's smallest rank reaching that fraction of its own original variance. `train_factors` and `SharedFactorEncoder` require equal image/text widths, so the training width is the **smallest common K meeting both thresholds**: the maximum of the two modality minima. Each factor run receives only the top-K whitened columns, with no zero-padding. The same PCA fits are reused across thresholds; factor training is repeated with seed 42, 32 factors, 2,000 epochs, batch size 1,024, and `lambda_usage_balance=0.1`. All other loss weights and defaults match Task 4. The only changed training input is whitening rank.

| Variance target | Image minimum K | Text minimum K | Shared training K / 512 | Variance retained, image / text |
|---|---:|---:|---:|---:|
| 95% | 262 | 309 | **309** | 96.84% / 95.01% |
| **99% (primary)** | **400** | **433** | **433** | **99.46% / 99.00%** |
| 99.9% | 486 | 485 | **486** | 99.905% / 99.918% |

This shared-width rule is necessary for the current factor model: independently truncating to each minimum would yield mismatched tensor widths and cannot be passed directly to it. It also retains more image components than the image-only minimum at 95% and 99%. The whitening function itself returns the exact independent minimum when called on one modality.

## Outcome

| Measure | Task 3 raw CLIP | Task 4 full-rank whitening | 99% reduced-rank whitening |
|---|---:|---:|---:|
| Max single-factor activation-mass share | Not reported | 4.35% | **3.88%** |
| Top-two-factor activation-mass share | 87.7% | 8.66% | **7.76%** |
| Modality-private factors | 1/32 | 0/32 | **0/32** |
| Dead factors | 0/32 | 0/32 | **0/32** |
| Factors with at least half their mass in one community | 13/32 | 7/32 | **1/32** |
| Active factors spanning communities | 19/32 (59.4%) | 25/32 (78.1%) | **31/32 (96.9%)** |
| Relative reconstruction L2, image / text | 0.5542 / 0.5097 | 0.9929 / 0.9931 | **0.9912 / 0.9915** |
| Code cosine, matched / shuffled | 0.9263 / 0.6060 | 0.8900 / 0.5881 | **0.9026 / 0.5791** |
| Matched-minus-shuffled cosine | 0.3203 | 0.3019 | **0.3235** |

The operational checks are identical to Tasks 3–4: dead means both modality means below `1e-4`; modality-private means the smaller modality mean is at most 5% of the larger; a non-dead factor spans communities when no single community holds at least half its combined activation mass. At 99%, only factor 28 places at least half its mass in one community. The matched cosine remains above the fixed seed-42 shuffled control; spanning and agreement do not establish semantic usefulness.

### Prespecified rank sensitivity

| Variance target | Shared K | Max mass share | Top-two mass share | Relative L2, image / text | Cosine, matched / shuffled |
|---|---:|---:|---:|---:|---:|
| 95% | 309 | 3.58% | 7.15% | 0.9868 / 0.9867 | 0.9042 / 0.5802 |
| **99%** | **433** | **3.88%** | **7.76%** | **0.9912 / 0.9915** | **0.9026 / 0.5791** |
| 99.9% | 486 | 3.68% | 7.35% | 0.9925 / 0.9927 | 0.9018 / 0.5689 |

The smallest K tested still leaves roughly 309 near-unit-variance dimensions for a 32-factor dictionary. Reconstruction improves by only about 0.005–0.006 relative L2 versus the primary 99% run and remains close to 1.0. Lowering the retained variance to 95% does not recover the raw-feature reconstruction baseline, while activation mass stays balanced throughout. These errors are calculated against each run's whitened target; the raw-feature values describe a different reconstruction target and are a reference, not an exactly matched objective.

This is one in-sample seed, with no held-out or semantic factor evaluation. The real run took 167.2 seconds, including 78.7 seconds for community detection and about 23–24 seconds per factor model.
