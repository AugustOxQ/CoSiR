# r29 facts

## 1. Caption input in training
- `ct2_d2.load_training_data` uses `ct2_data.default_captions`; default prefers `/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts/default` (sealed store), falling back to `/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b/txt.npy`; both are per-caption Qwen features. [src/test/20261020_claim_test/ct2_d2.py:283; src/test/20261020_claim_test/ct2_data.py:26,144,161,223,252]
- `split2` maps CLIP index `sample_id` to annotation row and its `painting`, groups by painting, sorts painting then extraction ID; `pid` indexes painting and `first` starts each group. Loader joins each store by `sample_id`, checks its `painting`, converts to float32, averages raw rows per painting, then L2-normalizes the mean. [src/test/20261020_claim_test/ct2_data.py:58,76,144,223,252; src/test/20261017_aspect_premise_checks/apc_data.py:16,65]

## 2. SigLIP 1 text store alignment
- `/data/SSD2/pre_extract/artelingo_full/siglip_so400m/index.json` records dataset, backbone, dtype, dimension, counts, truncation and image/text arrays; text entries have `sample_id`, `painting`, `img` (image row). `progress.json` records completed counts, seconds, truncation, norm/finite gate and index hash. [scripts/extract_features.py:547,589,603,608; the two JSON files]
- Read-only comparison of full indices: each has 308,723 unique caption IDs; all ID-to-painting mappings match Qwen full, and ordered (ID, painting) pairs also match. Thus all Qwen-index val/held captions can join by ID and painting, without relying on order. [both stores' index.json; src/test/20261106_text_claim/tcl_data.py:102,203]
- `txt.npy`: (308723, 1152), float16. Progress gate reports zero bad norms, maximum norm deviation 0.00026513; independent first 4096 rows range 0.9997662–1.0002309. [store `txt.npy`, `progress.json`]

## 3. Caption widths along the path
- Training source: Qwen-specific `ct2_data._caption_store` asserts 2048 and `_caption_union` allocates 2048 (`ct2_data.py:144,156,223,227`); `ct2_d2.load_training_data` asserts every graph space is 2048 (`ct2_d2.py:283,295`). Keep Qwen graph spaces intact and substitute caption inputs separately.
- `ms_train.TrunkSide` first linear is input 2048, head output 2048 (`src/test/20261019_model_sheet/ms_train.py:32`); `rb_student.build_model` parameterizes image input only and makes caption with `ms_train.TrunkSide`; `train_cohort` asserts caption input 2048 (`src/test/20261025_recipe_backbones/rb_student.py:43,51,60,153,156`). Give caption trunk an independent input width, including inference/checkpoint reconstruction.
- `bs_cstar._validation_inputs` loads Qwen means; `_export` asserts cap input 2048; `_infer_sides` rebuilds using image width only. Held export is image-only but still reconstructs the full view/checkpoint (`src/test/20261102_backbones_sweep/bs_cstar.py:321,350,358,385`). `r28_trainer` delegates to these (`r28_trainer.py:120,144`).
- `r28_text.load_sides` uses `widths = dict(img=1152, cap=2048)`; per-caption source and replay means are Qwen (`r28_text.py:105,113,136,163`). Change those inputs. `tcl_data.PLAIN_WIDTH=2048` belongs to Qwen and must remain; use existing 1152 side reader (`src/test/20261106_text_claim/tcl_data.py:38,203,216,272`).
- 2048 output remains: heads, teacher/graph embeddings and loss target (`rb_student.py:43,57`; `r28_trainer_loss.py:25`; `bs_cstar.py:334`; `ct2_d2.py:40,69,138`; `r28_text.py:97,128,144`); don't globally replace widths.

## 4. Text-direction scoring
- Query text is an individual caption; CoSiR runs `tcl_data.plain_text` Qwen full per-caption rows, joined to frame sample IDs, checked against painting, and L2-normalized. Validation replay alone checks painting caption means. [src/test/20261112_siglip_main/r28_text.py:136,163; src/test/20261106_text_claim/tcl_data.py:102,203,216]
- Plain SigLIP 1 uses its own pre-extracted text rows with `tcl_data.side(..., 'siglip_so400m')`, the same ID/painting checked join, plus SigLIP image rows. [src/test/20261106_text_claim/tcl_data.py:203,272]
- For new checkpoints load caption input width 1152 and the SigLIP per-caption rows; change replay to SigLIP painting means; refit validation centering means and read-out picks and then score held. All validation 30,872 and held 61,744 caption IDs are covered by the full-index equality in §2. [src/test/20261020_claim_test/ct2_data.py:34; src/test/20261112_siglip_main/r28_text.py:113,136,171,211,242]

## 5. Read-out selection
- Yes: `r28_text.validation_picks` calls `pick_readout` separately for each aspect and direction, maximizing mean validation label mAP across five seeds; `held_cells` applies that pick. [src/test/20261112_siglip_main/r28_text.py:58,242,255,290]

## 6. Photo model
- Yes. Affection fine-tuning inputs include `D`, the unit-normalized default explanation mean; `bs_photo._tensors` passes D as cap to the switched loss, alongside Qwen-derived graph/teacher. Fit has 69,927 photos and validation 4,251. [src/test/20261031_photo_finetune/pf_data.py:229; src/test/20261102_backbones_sweep/bs_photo.py:7,138,198,297]
- `/data/SSD2/pre_extract/affection_local/siglip_so400m/` contains `img.npy`, `keys.json`, `meta.json`, no text file; Qwen sibling has `txt.npy`. A SigLIP 1 Affection caption store elsewhere under pre_extract: not found in directory-name search; absence across all possible names not established. [paths above; `find /data/SSD2/pre_extract -iname '*affect*' -o -iname '*siglip*'`]

## 7. MMArt
- Table 4 is image AMI and image retrieval mAP; `r28_mmart.py` reads old r17/r18 records, it does not rescore. [src/test/20261112_siglip_main/r28_mmart.py:1,26,62]
- For new checkpoints, adapt r15 `mt_export.infer`/`export` to reconstruct the new full model and checkpoint paths, exporting image-side embeddings for MMArt and validation; rerun validation pick and MMArt AMI/mAP scorers (`mt_ret`, `mt_clu`, r17 scoring as applicable), then update record reader. [src/test/20261102_mmart_test/mt_export.py:141,165; src/test/20261102_mmart_test/mt_ret.py:160,198; src/test/20261112_siglip_main/r28_mmart.py:89]

## 8. Ablation
- Reference c* plus removal of instructed graphs, content correction, buddy, relational; paper also includes plain embedding. Seeds 0–4 for trained arms, plain unseeded; held AMI and mAP. [src/test/20261112_siglip_main/r28_ablation.py:1,9,50,55,70]
- Recorded `instructed_graphs` runs s0–s4: 498.3, 500.2, 422.9, 432.2, 440.4 s. Other removal run times not found in current r28 train records. The scorer is export-based, output width 2048, so needs new roots/checkpoint exports, not a caption-width parameter itself; training needs the parameter. [corresponding `/data/SSD2/cosir_r28/20261112_siglip_main/cstar/siglip1/train/instructed_graphs/run__s*.json`; `r28_ablation.py:55,79,160`]

## 9. Settings check
- SigLIP 1 seed 0 validation: c* k11=40 against k11=10, 20, 80 and k6=0.0005, fallback mean of six AMI/mAP cells (r10 criterion unavailable). k6 is optimizer learning rate: baseline 0.001 vs 0.0005. [src/test/20261112_siglip_main/records/settings_check.json; src/test/20261027_recipe_sweep/rs_knobs.py:12,63; r14 `train/k11=40/run__s0.json`]
- c* 0.3848908; k11=10 0.3796932; k11=20 0.3837708; k11=80 0.3853601; k6 0.3855929 (+0.0007021). Log flags both positive margins inside c* seed spread 0.3849–0.3879, no winning rows scored on test. [src/test/20261112_siglip_main/records/settings_check.json; src/test/20261112_siglip_main/20261112_siglip_main_log.md:34]

## 10. Runtime
- r14 SigLIP 1 c* seed 0 `seconds=340.8912`; this times `train_cohort` including its normal validation output, not data loading, separately invoked held export, or later scoring. [ `/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/cstar/siglip1/train/k11=40/run__s0.json`:103; src/test/20261102_backbones_sweep/bs_cstar.py:271,300]
- Separate validation + held export timing: not found. A different combined five-seed SigLIP 1 export job (held plus Affection/EmoSet photo) took 46 s, not a per-seed val+held figure. [src/test/20261102_backbones_sweep/bs_run_log.md:44]
- Text scoring logs: validation 3.28 min and held 6.90 min queue run, each for all three aspects and five seeds; ablation reference held AMI+mAP scoring 234.1 s for 15 aspect-seed cells. [ `/data/SSD2/cosir_r28/20261112_siglip_main/text/val.log`, `text/run_held.log`, `ablation/logs/replay.log`]

## 11. Text encoder timing
- Yes. `r28_time.py` times Qwen `encode_texts` with tokenization inside timed loop and reports per-caption repeated batch statistics; replace/add encoder with `load_encoder('siglip_so400m')` and same strings, adjust/check 1152-wide outputs, warmups and benchmark conditions. The separate adapter timing must also use the new caption checkpoint and 1152-wide features. [src/test/20261112_siglip_main/r28_time.py:1,48,63,86,105; src/data/feature_extract.py:175]

## 12. Prefix, truncation and extraction time
- SigLIP 1 uses `google/siglip-so400m-patch14-384`, bf16 model, lowercases text, tokenizer padding `max_length`, truncation at 64, get_text_features then L2 normalization. [src/data/feature_extract.py:25,137,175; scripts/extract_features.py:589]
- Local cached tokenizer, offline, counted all 308,723 indexed annotation captions with special tokens and extractor lowercasing: 0 exceed 64 tokens with `Represent the caption: `, and 0 without prefix. [cached `/data/SSD2/HF_home/hub/models--google--siglip-so400m-patch14-384`; annotation and store index; counting rule `scripts/extract_features.py:470`]
- Recorded unprefixed extraction `n_truncated=0`; `txt_seconds=369.8577` s, accumulated only around encoding (excludes separate truncation count and other overhead). [store `progress.json`; scripts/extract_features.py:470,589]

## Implementer notes
1. Add an ID/painting-checked SigLIP caption loader with painting mean then L2 normalization; retain Qwen default and instructed spaces for graph construction, correction and teacher targets.
2. Parameterize caption input width separately from image width in model factory and training; preserve 2048 head outputs, teacher/loss widths and Qwen graph assertions; store input configuration in checkpoint metadata.
3. Update validation input/export, held image export model reconstruction and checkpoint compatibility for the new caption trunk; train fresh seeds and verify exports.
4. Update text scoring and replay to use individual SigLIP caption rows / SigLIP validation means, refit validation means and picks, then score held.
5. For MMArt and ablation reruns, export and score the new checkpoints with suitable roots and validation picks; extend photo training only if requested, first extracting Affection caption text features.
6. Adapt text encoder and adapter timing, and decide explicitly whether prefixing requires new extraction and truncation handling.

## Would remove
- Nothing identified.
