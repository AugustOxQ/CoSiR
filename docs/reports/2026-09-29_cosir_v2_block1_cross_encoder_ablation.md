# CoSiR v2 Block 1: DINOv2 + e5 cross-encoder ablation

**Verdict: the trained-minus-raw emotion AMI gap widens.** With DINOv2 + e5, Stage 1 gains **0.008321** AMI over its own raw-feature baseline, versus **0.001161** with CLIP + CLIP. This single-seed, in-sample result is evidence that the buddy/InfoNCE training mechanism can add emotion alignment when given a different encoder pair; the near-zero CLIP gap is not universal. It does not isolate the encoder swap from the dimension-matching choice or establish held-out performance.

## Same-split comparison

| Encoder pair | Raw-feature baseline AMI | Trained Stage 1 AMI | Gap (trained − raw) |
|---|---:|---:|---:|
| CLIP + CLIP (from Block 1) | 0.035781 | 0.036942 | +0.001161 |
| DINOv2 + e5 (this run) | 0.032652 | 0.040973 | +0.008321 |

The DINOv2 + e5 gain is about **0.007160 AMI greater** than CLIP's gain (about 7.2 times its size). As a secondary finding, the **raw** DINOv2 + e5 features score **0.003129 lower** than raw CLIP on emotion AMI, while their trained Stage 1 embeddings score **0.004031 higher** than trained CLIP. Thus the larger training gain is not simply a stronger raw-feature baseline. These are measured differences from one run per pair, not estimates of statistical significance or seed robustness.

| Encoder pair and variant | Communities | Occupancy, min / median / max | Empty |
|---|---:|---:|---:|
| CLIP + CLIP, raw | 23 | 111 / 11,959 / 36,537 | 0 |
| CLIP + CLIP, trained | 21 | 35 / 9,395 / 33,338 | 0 |
| DINOv2 + e5, raw | 23 | 93 / 8,140 / 38,664 | 0 |
| DINOv2 + e5, trained | 18 | 1,023 / 14,660.5 / 36,053 | 0 |

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
