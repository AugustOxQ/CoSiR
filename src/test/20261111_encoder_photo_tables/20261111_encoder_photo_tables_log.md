# 20261111 encoder_photo_tables: the paper's Tables 6 and 7 from recorded cells

Written 2026-11-11 (Amsterdam) by the "CoSiR paper sec4 tables 4" tab, at the user's request ("add the Affection
numbers and the backbone numbers to the LaTeX now"). No new scoring: every value is read from a gated record.

## What it does
`ept_make.py <REPO>/Tables` writes `encoders.tex` (Table 6) and `photos.tex` (Table 7) and `records/cells.json`
(102 values, each with its source id). Rounding to three decimals and the bold (best per column at printed precision,
ties all bold) are done in the script. Asserts: c* is the same model in r17's SC1, SC2 and SC3 cells per encoder; the
photo model is the same in SC4's plain and teacher cells; the EmoSet photo mAP agrees between B02 SC8 and SC10.

## Sources
- r17 `src/test/20261103_new_scores/records/verdicts.json`: AMI at the class count (SC1 plain, SC2 matched PercepT,
  SC3 teacher on ArtELingo; SC4 Affection k = 9 and EmoSet k = 8); ArtELingo mAP at the validation-picked read-out
  (SC7 plain, SC8 matched PercepT and c*).
- B02 `src/test/20261102_backbones_sweep/records/verdicts.json`: Affection mAP per encoder (SC5), EmoSet mAP (SC8, SC10).
- r13 `/data/SSD2/cosir_r13/20261031_photo_finetune/outputs/stage3/verdict.json`: Qwen3 Affection mAP
  (`rows.<arm>.map_picked.map`, read-out center+k10).
- r11 stage 5 `/data/SSD2/cosir_r11/20261029_retrieval/outputs/stage5/verdict.json`: the teacher's ArtELingo mAP
  (Table 3's teacher row).

## Checks
- The Qwen3 block of Table 6 equals Tables 2 and 3 at three decimals.
- r17's EmoSet class-count AMI (photo / teacher / plain 0.4403 / 0.3942 / 0.3492) equals B02's diagnostic row.
- Compiled, no Overfull after the teacher row moved into the Qwen3 block (first version 10.8 pt too wide).

## Not in the tables (open for the user)
- Frozen c* (no photo fine-tuning) on photos: only AMI at 20 clusters exists (Affection 0.1384) and mAP 0.2755;
  class-count AMI would need a k-means run on saved outputs.
- CRL on Affection: mAP 0.2739 exists (r13), no class-count AMI.
- The r19 encoders (DINOv2, CLIP B/32, CLIP B/16): the user's 2026-11-05 reading put them in the ablation or appendix.
