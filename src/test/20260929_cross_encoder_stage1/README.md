# ArtELingo DINOv2 + e5 train features

`extract_features.py` extracts row-aligned, L2-normalized float32 features for all 308,723 rows of `/data/PDD/artelingo/artelingo_train.json`. Row `i` of each array corresponds to row `i` of that JSON list. Images resolve under `/data/PDD/wikiart_proj/wikiart/`; captions come from each row's `caption` field.

- **Image:** `facebook/dinov2-small` via `AutoProcessor` and `AutoModel`, using `last_hidden_state[:, 0]` (CLS), 384 dimensions. The 61,402 distinct image files are encoded once and copied to all corresponding annotation rows.
- **Text:** `intfloat/e5-base-v2` via `AutoTokenizer` and `AutoModel`, with `"query: "` prepended to each caption, attention-masked mean pooling over `last_hidden_state`, and 768 dimensions.

From the repository root, with the `CoSiR` environment and a local CUDA GPU:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate CoSiR
python -u src/test/20260929_cross_encoder_stage1/extract_features.py --smoke-only
python -u src/test/20260929_cross_encoder_stage1/extract_features.py
```

The full command runs its own 256-row smoke gate before extracting all rows. It writes `features/dinov2_img.npy` and `features/e5_txt.npy` beside this README; the repository's `*.npy` ignore rule excludes them from Git. Defaults are 128 images per batch, 256 captions per batch, and eight image-loading threads; see `--help` to adjust them.

## Measured run

The 256-row smoke gate passed: DINOv2 `(256, 384)` and e5 `(256, 768)`, both finite with row norms approximately 1. The full extraction took **515.553 seconds** of wall-clock time, measured from the start of full array creation through writing and validating the outputs, excluding model startup and the smoke gate. No rows were skipped. Pillow warned about two unusually large source images; both were processed successfully.

| File | Shape | dtype | Finite | Row norm min / mean / max | Value min / max |
| --- | --- | --- | --- | --- | --- |
| `dinov2_img.npy` | `(308723, 384)` | float32 | yes | 0.9999999 / 1.0000000 / 1.0000001 | -0.2817402 / 0.2824658 |
| `e5_txt.npy` | `(308723, 768)` | float32 | yes | 0.9999999 / 1.0000000 / 1.0000001 | -0.2043188 / 0.1518828 |

## SigLIP vision swap with the saved e5 text array

`extract_siglip.py` encodes the same 61,402 distinct ArtELingo images with
`google/siglip-base-patch16-224`. It uses `get_image_features` (including the
`pooler_output` return used by newer transformers), then L2-normalizes each
768-dimensional float32 vector and restores annotation row order. It does not
load or re-extract e5 captions. The output is `features/siglip_v_img.npy`.

Run from the repository root with the `CoSiR` environment and local CUDA GPU:

```bash
python -u src/test/20260929_cross_encoder_stage1/extract_siglip.py --smoke-only
python -u src/test/20260929_cross_encoder_stage1/extract_siglip.py
python -u src/test/20260929_cross_encoder_stage1/run_siglip_e5_validation.py
```

The full extraction command repeats the 256-row smoke gate before encoding all
images. `run_siglip_e5_validation.py` pairs the new image array with the
existing `features/e5_txt.npy`, checks their matching 768-dimensional widths,
and runs the same graph, Stage 1, raw baseline, and emotion AMI comparison as
the DINOv2 + e5 validation. No padding or projection is applied.

The separate 256-row SigLIP smoke run passed with finite `(256, 768)` float32
features and unit row norms. The full command passed its own smoke gate, then
encoded all 61,402 unique images and restored all 308,723 annotation rows in
**454.390 seconds** of extraction wall-clock time (excluding startup and smoke).
The saved `siglip_v_img.npy` is `(308723, 768)` float32, all finite, with row
norm min / mean / max **0.9999999 / 1.0000000 / 1.0000001** and value min /
max **-0.2986873 / 0.5251451**. Pillow warned about an unusually large source
image; the full run completed and validated the output.

## ImageNet-supervised ViT vision swap with the saved e5 text array

`extract_vit_sup.py` uses the `google/vit-base-patch16-224` backbone and its
`last_hidden_state[:, 0]` CLS token. It L2-normalizes the 768-dimensional
float32 vectors, encodes each distinct image once, and restores the ArtELingo
annotation row order in `features/vit_sup_img.npy`. The ImageNet classification
supervision of this vision model is separate from the emotion labels used only
to evaluate the resulting communities.

Run from the repository root with the `CoSiR` environment and local CUDA GPU:

```bash
python -u src/test/20260929_cross_encoder_stage1/extract_vit_sup.py --smoke-only
python -u src/test/20260929_cross_encoder_stage1/extract_vit_sup.py
python -u src/test/20260929_cross_encoder_stage1/run_vit_sup_e5_validation.py
```

The full extraction repeats the smoke gate. `run_vit_sup_e5_validation.py`
reuses the existing `features/e5_txt.npy` and verifies that both modalities
have 768 features per row before running the unchanged graph, Stage 1, raw
baseline, community detection, and emotion AMI pipeline. No padding or
projection is applied.

The separate 256-row smoke test and the full command's repeated smoke gate
passed. The full run encoded all 61,402 unique images and restored all 308,723
rows in **450.809 seconds** of extraction wall-clock time, excluding startup
and smoke. `vit_sup_img.npy` is `(308723, 768)` float32 and finite, with row
norm min / mean / max **0.9999999 / 1.0000000 / 1.0000001** and value min /
max **-0.3249589 / 0.3157847**. Pillow warned about two unusually large
source images, both of which were processed successfully.
