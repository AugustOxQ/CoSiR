# 20261021 attention maps extraction

Written 2026-10-21. Extraction script for Qwen3-VL-Embedding-2B attention maps (4 instructions x img / txt / joint, 5,000 paintings).

- att_core.py: token types, rollout, suffix rollout (shared-start shortcut), writer, load_painting (numpy only at import).
- att_extract.py: CLI (--out --start --limit --views --check-full --device).
- att_diag_gate_a.py, att_diag_fp32.py, att_diag_cross.py: diagnostics behind three deviations from the brief.
  1. Gate a: eager bf16 against stored sdpa batch embeddings gives mean cosine 0.9987 to 0.9993 (sdpa bs=1 alone gives 0.9994), so mean > 0.998 / min > 0.99 plus a nearest-row order check replace mean > 0.999.
  2. Gate b: bf16 shortcut against plain pass has cosine min 0.9971 (eager vs sdpa plain passes: 0.9965); with the model in float32 the shortcut matches the plain pass to 1e-6 (cosine 1 - 1e-7, image-map L1 8e-7, cross L1 1.4e-6). Cosine floor set to 0.995 (mean 0.999); image-map L1 max 0.05 unchanged.
  3. Cross maps are stored as float32: values are 1e-10 to 1e-6, float16 would give L1 about 1 against the true map.
Smoke: `att_extract.py --limit 16 --check-full 4 --out <scratchpad>/att_smoke`.

## 2026-10-21 16:47  Raw attention instead of rollout

- Finding: rollout fails on this causal model. 99.999% of the pooled token's rollout mass goes to token 0, and the image remainder is mostly position (the top image rows).
- User decision (after seeing the recommendation): the maps are raw attention, the head-mean attention row of the query token summed over a layer range. Default is layers 16 to 27; last layer and all layers are the other choices (att_core.BANDS, layer index 0..27 = order of out.attentions). `att_core.band_map(p, band, kind)` returns the renormalised map and the mean per-layer share.
- Kept: `layers` (raw per-layer rows of the pooled token) and `roll` (pooled rollout row, as a record).
- Cross maps changed: now the raw head-mean attention of every caption-token row over the image columns, summed per band, float16, shape (3, n_caption_tokens, n_image_tokens). The caption-row rollout is no longer computed (suffix rollout kept for the pooled row only). full_pass (gate b) also returns raw layers and raw cross maps.
- Gate b: cross_l1 is now the L1 between band-renormalised raw cross rows; new img_l1_raw = L1 between shortcut and full-pass band_map image maps. Both fail above 0.05 (threshold unchanged).
- Smoke (16 paintings, check-full 4): gate b fails the new thresholds in bf16 (img_l1_raw max 0.074, last layer; cross_l1 max 0.102); see the hand-back.

## 2026-10-21 16:50  Gate b limits for raw maps (controller)

- `att_diag_raw_fp32.py` (4 paintings, default and style_word): against the float32 shortcut as reference, the bf16
  shortcut and the bf16 plain pass have the same L1 (image map, layers 16-27: 0.0928 and 0.0928 mean, 0.170 and 0.165
  max; caption-word rows: 0.085 and 0.084 mean). So the shortcut adds no error; the gap is bf16 rounding.
- bf16 moves about 5% of a map's mass against float32 (L1 0.06 to 0.14 mean by band). Kept bf16: every embedding store
  of the project is bf16. The report gives this as the noise floor for differences between views.
- Gate b limits: raw image map L1 max 0.10, raw cross L1 max 0.20 (about 1.5x the measured shortcut-vs-plain maxima,
  0.074 and 0.102). A wrong cache or position gives L1 near 1 or more.

## 2026-10-21 16:53  Pilot, 100 paintings (`--limit 100 --check-full 8`, scratchpad)

- Gates passed. Gate a: min cosine image 0.9960, text 0.9987. Gate b: embedding cosine min 0.9971; raw image map L1
  max 0.074 (last layer), 0.060 (layers 16-27); raw cross L1 max 0.102.
- 1.03 s per painting with gate b on 8 paintings; 0.82 s without, so about 70 min for 5,000. Peak GPU memory 5.0 GB.
- 158 MB for 100 paintings (about 8 GB for 5,000).
- GPU power 300 W mean, 347 W max, utilisation 86%: a heavy run, so the full run starts at 19:00 (weekday rule).

## 2026-10-21 17:05  Viewer and analysis script

- `att_viewer.py` (+ html, js, css): local server, one painting at a time; endpoints checked on the pilot; not yet seen
  in a browser. `att_label_names.json` is a gitignored cache it rebuilds when missing.
- `att_analysis.py`: the summary measures 1 to 7 (shares by layer, map shape, change from the default view with a
  shuffled baseline, adding the caption, caption words, labels with Kruskal-Wallis and Bonferroni, word maps). Pilot
  (100 paintings, `--min-class 5`): default-band L1 to the default view 0.68 to 0.79 against 1.15 to 1.22 shuffled and
  a bf16 floor of about 0.09. Full run after the extraction: `att_analysis.py --root <attention root> --workers 8`.

## 2026-10-21 20:08  Full run, 5,000 paintings (`att_queue_full.sh`, local 3090)

- Started 19:00 under the GPU lock; done 20:08 (4,116 s, 0.82 s per painting, peak 5.6 GB). Exit 0; 5,000 files,
  8.1 GB in `/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_attention/` (run record `run_0_all.json`).
- Passes: 20,000 img, 100,640 txt, 100,640 joint (5,000 paintings and 25,160 captions, 3 to 12 per painting, x 4 views).
- Gate a (first 32 paintings): mean cosine 0.9989 to 0.9993 for images, 0.9996 to 0.9997 for captions; min 0.9960;
  each image's nearest stored row is its own. Gate b (8 paintings, 184 passes): cosine min 0.9971, mean 0.9994;
  raw image-map L1 max 0.060 (layers 16-27), 0.074 (last layer), 0.039 (all layers); cross L1 max 0.102. All passed.
- Log: `outputs/full_run.log`.

## 2026-10-21 20:09  Analysis, 5,000 paintings

- `att_analysis.py --workers 8`: 29 s; outputs in `outputs/qwen3vl_emb_2b_attention/` (analysis.md, analysis.json,
  8 figures, per_painting.csv.gz). Max |share sum - 1| 1.5e-3; no cross rows dropped.
- Chance level for "hottest token on the border ring": the ring's share of tokens, mean 0.209 (median 0.178) over the
  5,000 grids (computed from per_painting.csv.gz grid_rows and grid_cols; 48 to 512 image tokens).
- Viewer restarted on the full root (port 8765, detached; log `outputs/viewer_full.log`).
- Report: `docs/reports/auto/percept/2026-12-23_attention_maps.md`.

## 2026-10-22 03:13  Check: are the scattered image maps a bug? (user question after seeing the viewer)

- Problem: in the viewer the image maps look scattered, almost random. Question: pipeline bug or the model?
- Scripts: `att_diag_spatial.py` (Codex, commits c954012, 6712018; parts geometry, recompute, structure, sinks,
  objects, sheet), `att_diag_order_real.py` and `att_diag_pos_pattern.py` (controller). Outputs in
  `outputs/spatial_diag/` (GPU parts: `summary_gpu_recompute_sinks_objects.md`; CPU parts:
  `summary_cpu_structure_sheet_geometry.md`, whose recompute and sinks rows are a 1-painting float32 smoke only).
- No bug found:
  1. Recompute (GPU bf16, plain full passes, 4 paintings x 2 views x 3 bands): L1 to the stored maps 0.02-0.03
     (layers 16-27), max 0.068 (last layer); the query row is `<|endoftext|>`. The viewer's `/api/maps` equals the
     stored maps to 5e-7.
  2. Token order: 6 real paintings, a region painted black; the image tokens that change are found in the map's
     row-major order (AUC of the hidden-state change, changed against unchanged cells: 96 px square 0.976 at the LLM
     input, 0.98-1.00 at layers 1-16; left 3/8 and top 1/4 0.74-0.90), against 0.33-0.55 for the transposed and random
     orders. A red disc on grey: the hottest token (default, layers 16-27) lies on the disc in 3 of 4 positions.
     Codex's first order test (flat synthetic colours, k-means, then a differential test) could not decide: on flat
     images the vision encoder's last layer spreads a change over many tokens; real paintings keep it local.
- The maps really are speckled:
  1. Moran's I (rook neighbours), 5,000 paintings, layers 16-27: default 0.138 [0.064, 0.233], instructed
     0.065-0.088, shuffled positions about 0; all layers 0.20-0.23.
  2. A plain grey canvas gives the same speckle (objects figure), so it comes from the model, not the painting.
  3. Single heads are no cleaner: per-head Moran's I 0.05-0.19 (16 paintings).
  4. Not a fixed position pattern: paintings with the same grid size share little (pairwise Spearman 0.06-0.09,
     leave-one-out group mean 0.15-0.21); dividing by the group mean does not smooth the maps.
  5. Hidden-state norm and attention: Spearman 0.47-0.61; no outlier image tokens (max norm 1.4x the median).
- Coarse structure exists: rank-coloured maps follow frames, margins and sky/ground bands (`sheet_ranks.jpg`); a red
  disc draws 1.4-2.3x its token share (default, layers 16-27), drawn faces about 1x, the last layer 0.3-0.5x.
- Consequence: "more focused" instructed maps put more mass on a few scattered tokens, not on one region. Raw
  attention is a weak picture of where the model looks; a paper figure would need another attribution method.
