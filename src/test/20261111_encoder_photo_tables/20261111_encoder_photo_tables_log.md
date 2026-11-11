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

## Frozen c* on photos (class-count AMI), 2026-11-11 09:08
Request: the paper row "frozen c*" (ArtELingo-trained emotion head, no photo fine-tuning) on Affection (k = 9) and
EmoSet (k = 8), Qwen3, with r17's SC4 protocol. Script `ept_frozen.py` (`gate`, then `run`); records
`records/frozen_cstar_gate.json`, `records/frozen_cstar.json`; logs `logs/gate.log`, `logs/run.log`.
- Gate: r17's own path (`ns_clu_cells._cand_parts` / `_comp_parts`, `ns_clu.paired_cell` point values, 1 draw; and
  `ns_clu.metric_table`) reproduced the four recorded SC4 cells (photo vs plain / teacher, both datasets) with max abs
  difference 0.0 (tolerance 1e-9).
- Rows: r14 `exports/<set>/cstar_qwen/test__s<seed>.npz` (checkpoint = r10 frozen c* emotion head, sha d2310aeb...),
  joined by key as `ns_photo.photo_stored` does (EmoSet uncut 17,716 -> cut 17,348 by key). Same k-means call
  (`ns_metrics.partitions`), same `metric_table`.
- AMI at the class count, mean of seeds 0 to 4: Affection 0.122671 (k 9), EmoSet 0.384830 (k 8).
- Protocol checks: Affection k = 20 gives 0.13838015668340972, equal to r13 `rows.cstar.ami` (diff 0.0); EmoSet k = 20
  gives 0.4472745151510936, equal to r14 `photo_minus_cstar` cstar AMI (0.4473).
- mAP at the validation-picked read-out (center+k10), from records only: Affection 0.275528 (r13 verdict
  `rows.cstar.map_picked.map`); EmoSet 0.441309 (r14 `diag/photo_minus_cstar.json` `sections.emoset.map_at_pick.cstar`).
- Other encoders on Affection: no frozen c* test rows and no mAP record exist (only Qwen's `cstar_qwen` exports). The
  SigLIP 2, SigLIP 1 and CLIP L/14 c* emotion checkpoints exist (r14 `outputs/cstar/<enc>/weights/k11=40/emotion__s<seed>.pt`).
  Needed: inference of those checkpoints on the Affection val and test features (bs_photo.py has a `cstar_<enc>` export
  arm), then k-means (this script), and a read-out pick on val plus test mAP. Not done.
