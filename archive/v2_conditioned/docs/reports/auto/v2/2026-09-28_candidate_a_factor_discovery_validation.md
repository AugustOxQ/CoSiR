# CoSiR v2 Candidate A: factor discovery on ArtELingo

**Verdict: real problems found; this is not yet a clean foundation for the condition-interface stage.** In one default-configuration, seed-42 run on all 308,723 ArtELingo training pairs, **1/32 factors was effectively modality-private**, **0/32 were dead** at the stated threshold, and **13/32 (40.6%) concentrated at least half of their activation mass in one Stage 1 community**. The remaining **19/32 (59.4%) spanned multiple communities** by that criterion. Two broadly active factors alone carried **87.7% of all mean activation mass**. Reconstruction and matched-pair agreement were measurable, but these concentration and imbalance findings mean the dictionary is only a partial result, not a validated shared aspect basis. No hyperparameters were adjusted after the first real attempt; this report describes that first run.

## Run and decision rules

The validation script is `src/test/20260928_factor_discovery_validation/run_validation.py`. It used `FeatureManager` at `/data/SSD2/pre_extract/artelingo/features`, loaded `/data/PDD/artelingo/artelingo_train.json`, and verified that metadata, image rows, text rows, unique in-range sample IDs, and annotations all matched at 308,723 rows. It used the stored IDs for the positional annotation join. The same Block 1 defaults rebuilt the 3,130,544-edge content graph, trained Stage 1 for 200 epochs, and detected 21 communities from Stage 1 embeddings. Factor discovery took the **raw frozen CLIP image and text features** as input and used the graph only as a neighbor regularizer. Each of its 2,000 epochs sampled 1,024 graph edges, deduplicated their endpoints, and trained on that node set.

`FactorTrainingConfig()` used 32 factors, Adam at `1e-3`, seed 42, and weights 1.0 for reconstruction, paired agreement, and graph consistency; 0.01 for sparsity; and 0.1 for anti-split. **There is no validated reference for this exact loss-weight combination.** These are starting defaults, not established optimal values.

The diagnostics use the following declared rules:

- A factor is **dead** when both modality means are below `1e-4` (on the raw CLIP input scale).
- An active factor is **modality-private** when the smaller modality mean is at most 5% of the larger mean. Dead factors are excluded from this count.
- A factor is **concentrated in one community** when one community receives at least 50% of its combined image-plus-text activation mass. An active factor below that threshold is counted as spanning communities. The largest Stage 1 community contains only 10.8% of rows, so 50% mass is substantial concentration. This is a mass test, not a claim that every spanning factor is semantically useful.

## Observations

| Measure | Result |
|---|---:|
| Modality-private factors | **1/32**: factor 13 |
| Dead factors | **0/32** at both means below `1e-4` |
| Factors concentrated in one community | **13/32 = 40.6%** |
| Active factors spanning communities | **19/32 = 59.4%** |
| Share of all mean activation mass in factors 18 and 26 | **87.7%** |
| Mean relative reconstruction L2 error, image / text | **0.5542 / 0.5097** |
| Mean image–text code cosine, matched / shuffled | **0.9263 / 0.6060** |
| Equivalent `1 − cosine` agreement loss, matched / shuffled | **0.0737 / 0.3940** |
| Sampled training loss, first 5 / last 5 epoch mean | **1.027990 / 0.199325** |
| Loss range over 2,000 epochs | **0.188101–1.258440**, all finite |

Relative reconstruction error is the mean over rows of `||decoded code − input||₂ / ||input||₂`, evaluated separately for the two modalities on every training row. The shuffled control is one seed-42 random permutation of text rows and uses the same post-hoc cosine calculation as the matched pairs. The matched advantage is 0.3204 cosine points, although the relatively high shuffled cosine and the dominance of factors 18 and 26 show that much of the code is shared across unrelated rows too. All returned codes were finite and nonnegative; their overall means did not collapse to zero.

Per-factor means below are computed over all 308,723 rows, exactly the statistic used inside the anti-split penalty. Community share is the largest single community's fraction of that factor's combined activation mass. `C` marks the 13 factors above the 50% single-community threshold; `P` marks the modality-private factor.

| Factor | Image mean | Text mean | Largest community mass | Flag |
|---:|---:|---:|---:|:---|
| 0 | 0.002973 | 0.001909 | 0.156 | |
| 1 | 0.005607 | 0.006156 | 0.885 | C |
| 2 | 0.004869 | 0.005324 | 0.311 | |
| 3 | 0.004556 | 0.002151 | 0.657 | C |
| 4 | 0.013493 | 0.010453 | 0.796 | C |
| 5 | 0.031501 | 0.041698 | 0.451 | |
| 6 | 0.022715 | 0.029235 | 0.723 | C |
| 7 | 0.001910 | 0.002620 | 0.232 | |
| 8 | 0.034422 | 0.030728 | 0.448 | |
| 9 | 0.081096 | 0.077023 | 0.344 | |
| 10 | 0.000089 | 0.000587 | 0.269 | |
| 11 | 0.003143 | 0.003134 | 0.766 | C |
| 12 | 0.002487 | 0.000270 | 0.353 | |
| 13 | 0.005026 | 0.000007 | 0.776 | C, P |
| 14 | 0.004656 | 0.005418 | 0.726 | C |
| 15 | 0.027413 | 0.025800 | 0.308 | |
| 16 | 0.002615 | 0.002938 | 0.599 | C |
| 17 | 0.007505 | 0.010401 | 0.723 | C |
| 18 | 1.197402 | 1.028983 | 0.192 | |
| 19 | 0.002227 | 0.000257 | 0.720 | C |
| 20 | 0.011508 | 0.011040 | 0.856 | C |
| 21 | 0.005261 | 0.003443 | 0.805 | C |
| 22 | 0.002872 | 0.001625 | 0.227 | |
| 23 | 0.002241 | 0.002931 | 0.318 | |
| 24 | 0.003509 | 0.004691 | 0.735 | C |
| 25 | 0.003617 | 0.001621 | 0.455 | |
| 26 | 1.194536 | 1.177803 | 0.173 | |
| 27 | 0.002321 | 0.001474 | 0.258 | |
| 28 | 0.005518 | 0.003913 | 0.395 | |
| 29 | 0.003583 | 0.001658 | 0.167 | |
| 30 | 0.003068 | 0.002630 | 0.433 | |
| 31 | 0.028886 | 0.022553 | 0.358 | |

## Limits

This is an in-sample, single-seed diagnostic. Community labels are the rebuilt Stage 1 labels from this run, not independently annotated aspects. The 50% threshold is an operational falsification rule; changing it changes the reported spanning fraction. The zero dead-factor count applies to the stated `1e-4` threshold: factor 10, for example, has a small image mean (`8.9e-5`) but a text mean of `5.9e-4`, so it is weak rather than dead by this definition. The learned factors were not semantically named or tested on a held-out split.

Wall-clock times for this run were 9.225 s for graph construction, 1.795 s for Stage 1, 84.707 s for community detection, 23.160 s for factor training including the final full-dataset code pass, and 120.059 s overall. Graph and training used the local CUDA GPU; community detection used the existing CPU implementation. These are observed timings, not benchmark averages.
