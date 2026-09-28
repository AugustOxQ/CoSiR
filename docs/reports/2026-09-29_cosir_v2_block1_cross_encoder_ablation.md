# CoSiR v2 Block 1: cross-encoder ablation with e5 text

**Verdict: the trained-minus-raw emotion AMI gap widens.** With DINOv2 + e5, Stage 1 gains **0.008321** AMI over its own raw-feature baseline, versus **0.001161** with CLIP + CLIP. This single-seed, in-sample result is evidence that the buddy/InfoNCE training mechanism can add emotion alignment when given a different encoder pair; the near-zero CLIP gap is not universal. It does not isolate the encoder swap from the dimension-matching choice or establish held-out performance.

**The SigLIP + e5 vision swap confirms a wider-than-CLIP gap, but a smaller one than DINOv2 + e5.** Its trained-minus-raw gap is **0.005739** AMI: about 4.9 times CLIP's gap and 0.002582 below DINOv2 + e5's gap. Keeping e5 fixed across the two cross-encoder runs shows the widening is not unique to DINOv2 vision features. This comparison is still one seeded, in-sample run per pair; the DINOv2 run also used fixed zero-padding to match e5's width, while SigLIP already matches it, so these gaps do not measure a pure vision-encoder effect.

## Same-split comparison

| Encoder pair | Raw-feature baseline AMI | Trained Stage 1 AMI | Gap (trained − raw) |
|---|---:|---:|---:|
| CLIP + CLIP (from Block 1) | 0.035781 | 0.036942 | +0.001161 |
| DINOv2 + e5 (prior run) | 0.032652 | 0.040973 | +0.008321 |
| SigLIP + e5 | 0.034434 | 0.040173 | +0.005739 |
| vit_sup + e5 | 0.031267 | 0.052282 | +0.021015 |

The DINOv2 + e5 gain is about **0.007160 AMI greater** than CLIP's gain (about 7.2 times its size). As a secondary finding, the **raw** DINOv2 + e5 features score **0.003129 lower** than raw CLIP on emotion AMI, while their trained Stage 1 embeddings score **0.004031 higher** than trained CLIP. Thus the larger training gain is not simply a stronger raw-feature baseline. These are measured differences from one run per pair, not estimates of statistical significance or seed robustness.

| Encoder pair and variant | Communities | Occupancy, min / median / max | Empty |
|---|---:|---:|---:|
| CLIP + CLIP, raw | 23 | 111 / 11,959 / 36,537 | 0 |
| CLIP + CLIP, trained | 21 | 35 / 9,395 / 33,338 | 0 |
| DINOv2 + e5, raw | 23 | 93 / 8,140 / 38,664 | 0 |
| DINOv2 + e5, trained | 18 | 1,023 / 14,660.5 / 36,053 | 0 |
| SigLIP + e5, raw | 29 | 22 / 8,440 / 34,989 | 0 |
| SigLIP + e5, trained | 19 | 3,654 / 15,477 / 32,591 | 0 |
| vit_sup + e5, raw | 23 | 21 / 10,213 / 36,089 | 0 |
| vit_sup + e5, trained | 15 | 6,099 / 20,045 / 43,193 | 0 |

The CLIP figures and occupancy come from [Block 1's same-split validation](2026-09-28_cosir_v2_block1_stage1_validation.md). All AMI scores compare communities with the real `emotion` field for the same 308,723 ArtELingo training rows.

## Method and measured run

`src/test/20260929_cross_encoder_stage1/run_cross_encoder_validation.py` loaded Task 1's saved, L2-normalized float32 arrays (`dinov2_img.npy`, 308,723 × 384; `e5_txt.npy`, 308,723 × 768). It used the arrays' documented positional alignment to `/data/PDD/artelingo/artelingo_train.json`; there was no feature re-extraction. The teacher graph was constructed from the **original, unprojected** DINOv2 and e5 arrays with `build_content_graph(..., GraphConfig())` and had 3,396,806 undirected edges.

The unmodified `train_stage1` requires both modalities to have the same feature width. For that call only, the script **zero-padded each 384-dimensional DINOv2 vector to 768 dimensions**, then passed the padded images and original e5 texts to `train_stage1(..., Stage1Config())`. This fixed linear embedding preserves DINOv2 row norms, inner products, and pairwise distances exactly and adds no fitted projection parameters. It does change the student's input width relative to the 512-dimensional CLIP run; the trained-score difference cannot be attributed uniquely to the encoder pair. The graph and raw baseline use the unprojected arrays.

For the raw baseline, the script L2-normalized each original modality separately and concatenated them into 1,152-dimensional vectors, matching Block 1's `raw_clip_baseline.py` procedure. It clustered both the trained 32-dimensional embeddings and raw vectors with unchanged default `detect_communities`, then used `community_stats` and `adjusted_mutual_info_score` on the real emotion labels. Defaults include seed 42. The graph and student training used local CUDA; community detection used the pipeline's CPU nearest-neighbor search and Leiden implementation. No `cuml` or `cugraph` was used.

| Measured stage | Wall time |
|---|---:|
| Graph construction | 11.987 s |
| Stage 1 training and embedding pass | 1.930 s |
| Trained embedding community detection | 98.409 s |
| Raw-feature community detection | 570.244 s |
| Total, including load and AMI | 684.526 s |

Both partitions had no empty communities. The trained DINOv2 + e5 community sizes were 36,053, 33,275, 29,362, 26,272, 24,895, 24,512, 22,637, 21,791, 16,559, 12,762, 11,918, 10,316, 8,112, 7,683, 7,405, 7,333, 6,815, and 1,023. The raw sizes were 38,664, 36,787, 28,201, 27,876, 26,677, 26,598, 19,274, 18,833, 15,613, 14,197, 13,598, 8,140, 7,611, 7,301, 6,838, 5,665, 3,798, 1,267, 959, 373, 216, 144, and 93. PyTorch emitted a warning when it wrapped the read-only e5 memory map; the pipeline only read that tensor, and the run completed successfully.

## SigLIP + e5 extension

`extract_siglip.py` used `google/siglip-base-patch16-224` via `get_image_features`, unwrapped `pooler_output` where returned, and L2-normalized float32 output. A separate 256-row smoke test passed before the full run; the full command repeated that smoke gate. It encoded all 61,402 distinct images once and restored the original 308,723 annotation-row positions. Full extraction wall time was **454.390 s**, excluding model startup and the smoke gate. `siglip_v_img.npy` has shape `(308723, 768)`, finite values, and row-norm min / mean / max of 0.9999999 / 1.0000000 / 1.0000001. Task 1's existing `e5_txt.npy` was reused without re-extraction.

`run_siglip_e5_validation.py` built the teacher graph from the original SigLIP and e5 arrays using `GraphConfig()` and obtained **3,611,154 undirected edges**. Because both arrays have 768 features per row, `train_stage1(..., Stage1Config())` received them directly, with no padding or projection. The raw baseline separately L2-normalized each view and concatenated them into 1,536-dimensional vectors. Both trained and raw features used default `detect_communities`, and emotion AMI was computed against the same ArtELingo training labels. The unmodified pipeline used seed 42.

The precise SigLIP + e5 scores were **0.034434421 raw AMI** and **0.040173012 trained AMI**, a **+0.005738591** gap. Raw SigLIP + e5 exceeded raw DINOv2 + e5 by about 0.001782 AMI, while trained SigLIP + e5 fell below trained DINOv2 + e5 by about 0.000800. This makes the smaller training gap visible even though the SigLIP pair starts from a stronger raw baseline. The SigLIP gap is about 0.004578 above CLIP's near-zero gap, so DINOv2's widening is not pair-specific.

| SigLIP + e5 measured stage | Wall time |
|---|---:|
| Graph construction | 16.289 s |
| Stage 1 training and embedding pass | 1.916 s |
| Trained embedding community detection | 83.885 s |
| Raw-feature community detection | 593.962 s |
| Total, including load and AMI | 698.140 s |

Both SigLIP partitions had no empty communities. The trained partition had 19 communities, ranging from 3,654 to 32,591 rows (median 15,477); the raw partition had 29 communities, ranging from 22 to 34,989 rows (median 8,440). As in the DINOv2 run, PyTorch warned when it wrapped the read-only e5 memory map; training only read it, and validation finished successfully.

## vit_sup + e5 extension

`extract_vit_sup.py` used the `google/vit-base-patch16-224` backbone and its
`last_hidden_state[:, 0]` CLS token, then L2-normalized each float32 vector.
This encoder was supervised for **ImageNet classification**. That is a
different pretraining objective from DINOv2's self-supervision and SigLIP's
image–text contrastive learning; it is also distinct from the
GoEmotions-based affect teacher discussed in Block 1. A separate 256-row
smoke test passed before the full run, which repeated the smoke gate. Full
extraction encoded 61,402 distinct images once and restored all 308,723
annotation-row positions in **450.809 s**, excluding startup and smoke.
`vit_sup_img.npy` is `(308723, 768)` float32, finite, with row-norm min /
mean / max of 0.9999999 / 1.0000000 / 1.0000001. The existing `e5_txt.npy`
was reused without re-extraction.

`run_vit_sup_e5_validation.py` constructed the graph from the original
vit_sup and e5 arrays with `GraphConfig()` and obtained **3,170,297 undirected
edges**. Both inputs have 768 features per row, so the unchanged
`train_stage1(..., Stage1Config())` received them directly, without padding
or projection. The raw baseline separately L2-normalized each view and
concatenated them into 1,536-dimensional vectors. Both variants used the
default `detect_communities` and the same 308,723 training-row emotion labels
for AMI; the pipeline used seed 42.

The precise vit_sup + e5 scores were **0.031266983 raw AMI** and **0.052281892
trained AMI**, a **+0.021014909** gap. Its raw AMI was 0.004514 below raw
CLIP + CLIP, while its trained AMI was 0.015340 above trained CLIP + CLIP.
The raw partition had 23 communities, with sizes from 21 to 36,089 (median
10,213); the trained partition had 15 communities, from 6,099 to 43,193
(median 20,045). Neither partition had empty communities.

| vit_sup + e5 measured stage | Wall time |
|---|---:|
| Graph construction | 18.019 s |
| Stage 1 training and embedding pass | 2.002 s |
| Trained embedding community detection | 87.016 s |
| Raw-feature community detection | 600.762 s |
| Total, including load and AMI | 709.897 s |

As in the earlier cross-encoder runs, PyTorch warned when it wrapped the
read-only feature memory maps; training only read those arrays, and the run
completed successfully.

## Four-pair synthesis

| Encoder pair | Trained − raw emotion AMI gap |
|---|---:|
| CLIP + CLIP | +0.001161 |
| DINOv2 + e5 | +0.008321 |
| SigLIP + e5 | +0.005739 |
| vit_sup + e5 | +0.021015 |

**Verdict:** All three different-from-CLIP pairs widen the in-sample emotion
AMI gap relative to CLIP + CLIP. The ImageNet-classification-supervised
vit_sup pair gives the largest gap, about 18.1 times CLIP's. This converges
with the DINOv2 and SigLIP results: the near-zero CLIP gap is specific to that
tested pair rather than a general failure of the Stage 1 training mechanism.
It supports, but does not establish causally, the interpretation that the
CLIP pair limited the observed training gain. The encoder pairs also differ
in text tower and, for DINOv2, dimension matching; these are one-seed,
same-split results, not a held-out or seed-robustness estimate.

Across the three e5 pairs, the gap increases as prior **RedCaps** union-graph
agreement with CLIP + CLIP decreases: SigLIP + e5 had Jaccard 0.173 and a
0.005739 gap; DINOv2 + e5 had Jaccard 0.134–0.138 and a 0.008321 gap;
vit_sup + e5 had Jaccard 0.128–0.132 and a 0.021015 gap. This ordering is
suggestive, not conclusive: it uses only three non-CLIP pairs, and those
Jaccard values came from a different dataset and graph experiment.
