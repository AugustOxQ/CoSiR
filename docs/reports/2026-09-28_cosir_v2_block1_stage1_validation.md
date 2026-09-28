# CoSiR v2 Block 1: content-only Stage 1 on ArtELingo

**Verdict:** Yes, the rebuilt content-only Stage 1 produces communities with a positive, modest association with real ArtELingo emotion labels **in-sample**: emotion AMI is **0.036942**. The graph, student training, and community detection ran end to end on all **308,723** real training samples. This is one default-configuration, seed-42 run; it does not establish held-out performance or seed robustness.

## Measured run

| Measure | Result |
|---|---:|
| Train samples and real emotion classes | 308,723; 9 |
| Content graph edges (undirected) | 3,130,544 |
| Communities | 21 |
| Community size, minimum / median / maximum | 35 / 9,395 / 33,338 |
| Empty communities | 0 |
| Emotion AMI, train labels against train communities | **0.036942** |
| Graph construction wall time | 9.317 s |
| Stage 1 training wall time, including final embedding pass | 1.765 s |
| Community detection wall time | 95.406 s |
| Total wall time, including data load and AMI | 107.818 s |

The script loaded `img_features` and `txt_features` from `/data/SSD2/pre_extract/artelingo/features` with `FeatureManager`, verified that its metadata, actual feature rows, sample IDs, and `/data/PDD/artelingo/artelingo_train.json` each had 308,723 rows, and used the stored sample IDs to index the annotation list. The IDs were unique and in range. It used `GraphConfig()`, `Stage1Config()` (200 epochs, one sampled 1,024-edge batch per epoch), and the default `detect_communities` settings. The reported first and final sampled-batch losses were 6.123320 and 3.051670; the returned 32-dimensional embeddings were finite. The graph and training stages used the local CUDA GPU; community detection used the existing CPU scikit-learn/Leiden implementation. Times are wall-clock readings from this run, not benchmark averages.

## Relation to the original Attention-h1 result

The [original ArtELingo investigation](/project/CoSiR-buddy_prototype_conditioning/docs/reports/2026-09-26_artelingo_buddy_vs_percept_stage1_report.md) reports approximately **0.1241 mean held-out emotion AMI across four seeds** for Attention-h1 (range 0.1213–0.1264). This run's 0.036942 is numerically lower by about 0.0872, but the figures are **not directly comparable**. Here, communities were detected on the *train split* and scored against that same split's emotion labels. The original used a separate held-out protocol. This rebuild also uses a content-only teacher, while Attention-h1 used content plus caption-derived affect; the implementations and seed coverage differ too. The observed gap is real as a difference between reported numbers, but this task cannot attribute it to a bug, to removing affect guidance, or to a generalization difference. There is no positionally aligned genre label for this full train store, so no genre AMI is reported.

An apples-to-apples comparison would first extract or obtain the same CLIP features for the held-out ArtELingo split, keep a train-derived community vocabulary, assign held-out samples under the same protocol as the original investigation, and score the same held-out emotion labels over matched seeds. A controlled content-only versus content-plus-affect ablation in the same implementation and evaluation protocol would then isolate the effect of the affect teacher. Held-out feature extraction and that comparison are follow-up work, outside this validation.
