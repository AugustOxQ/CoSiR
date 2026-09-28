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
