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

## 2026-10-22 13:53  Value-weighted attention: does weight parked on the sink drive the speckle or any result? (user, option g)

- Question (user, after the sink explanation): the first token takes 0.62-0.67 of the pooled token's attention. Is
  this the known attention sink, and do our maps or results depend on weight that idle heads park there? The user
  wants this before giving their reading of the results brief.
- Method: `att_diag_value_weighted.py`. Hooks on each layer's `v_proj`; per layer, the pooled row's terms
  w_j = ||sum_h a_hj W_O^h v_j^(h//2)|| (Kobayashi et al. 2020; 16 query heads, 8 value heads) and the variant
  w2_j = sum_h a_hj ||W_O^h v_j||, summed per band and renormalised like the raw maps. Image only on 300 random
  development paintings (seed 0) plus k = 0, 1250, 2500, 3750 for a sheet; caption only on all 25,160 captions.
  Checks: input ids equal the store's; recomputed raw maps match the stored ones (L1 < 0.1) for the first 4
  paintings and painting 0's captions. Also: token 0's value size (mean over heads of ||W_O^h v||) against the median
  image or caption token, and its hidden-state max-abs and norm against the median token (massive activations).
- Reading rules, fixed before the run: (1) token 0 is a no-op sink if its weighted share is far below 0.62-0.67;
  (2) weighting makes maps less speckled if the median Moran's I rises by >= 0.05 and in >= 75% of paintings;
  (3) a headline result holds if its direction holds under weighting (default image share above the instructed ones;
  L1 to the default above 0.09 and below shuffled; the emotion view's top caption words still emotion words).
- Smoke (6 paintings, 191 captions; scratchpad): passed the checks; full GPU part about 13 min, light.

### 14:08  Results (GPU 13:54-14:07, 2 s per painting for 4 views, captions 632 s; outputs/value_weighted/)

Image only on 304 paintings (300 random + the 4 sheet paintings), captions only on all 25,160 captions. All store
checks passed. Medians over paintings, layers 16-27 unless named. Full tables: `outputs/value_weighted/summary.md`.

- Rule 1, met: `<|im_start|>` is a classic sink. Its hidden state has max-abs 275-365x and norm 198-224x the median
  token at the outputs of layers 2, 4 and 8 (92-97x and 73-76x at layer 16), and its value vector is 0.065-0.067x a median image token's
  (layers 16-27). Weighted, its share falls from 0.62-0.67 to 0.12-0.13 (image input) and from 0.65-0.71 to
  0.15-0.17 (caption input).
- Rule 2, not met: weighting does not reduce the speckle. Spearman raw vs weighted map 0.96-0.97. Moran's I raw vs
  weighted: default 0.134 vs 0.111, emotion 0.064 vs 0.094 (higher in 83% of paintings, but +0.03 < 0.05), style
  0.088 vs 0.066, genre 0.082 vs 0.064; shuffled about 0. The variant w2 gives the same. The sheet looks the same.
- Rule 3, met for every headline result:
  - image share, default vs instructed: raw 0.216 vs 0.119-0.123; weighted 0.638 vs 0.359-0.371 (the image passes on
    most of the attention output once the sink is discounted);
  - L1 to the default: raw 0.68 / 0.79 / 0.80, weighted 0.65 / 0.87 / 0.89 (emotion / style / genre), shuffled
    1.18-1.31;
  - top-5% mass: default 0.20 raw, 0.21 weighted; instructed 0.37-0.40 raw, 0.35-0.43 weighted;
  - captions: top-word means raw vs weighted Spearman 0.97-0.98 (834 tokens with >= 50 occurrences); emotion top
    weighted: bored, awe, amused, excited, confused, proud; largest rises: style cartoon, abstract, distorted,
    realistic; genre abstract, portrait, adventure (same as raw).
- New: the border peaks are partly parked weight. Hottest token on the border, raw vs weighted: layers 16-27 default
  0.25 vs 0.15, emotion 0.60 vs 0.41, style 0.47 vs 0.31, genre 0.40 vs 0.23; all layers 0.73-0.99 vs 0.15-0.42; last
  layer 0.59-0.67 vs 0.51-0.58. Over all layers the weighting also changes the map more (Spearman 0.80-0.85) and
  halves Moran's I (0.20-0.23 vs 0.07-0.11): the early layers' smooth edge pattern is weight on small-value tokens.
- Reading: the sink is the known no-op sink (massive activation, small value). It does not explain the speckle and does
  not change any headline result. It explains most of the all-layer border peaks and part of the layers 16-27 ones;
  the emotion map still peaks on the border more often than the default (0.41 vs 0.15, chance about 0.21).
- Limits: value weighting still reads one attention sublayer; it is not importance for the final embedding (no
  gradient or masking check). Joint input and word maps not re-checked.

## 2026-10-24 16:42  User decisions; viewer switches; border check (option 1, small)

- Decisions (user): reading "thats very interesting results, provide some potential basis for deeper analysis";
  keep the 8.1 GB store (ledger row added); option 1 "a bit"; Codex report estimate 3% to 7% (routing review
  recorded); viewer switches yes; value-weighted maps for all 5,000 paintings no.
- The container restarted at 16:39; the viewer server (old pid 197563) was gone and was started again on port 8765.
- Viewer (commit 668eaae): switches Rank (colour by percentile rank within the map) and Blur (Gaussian, sigma 1
  token, radius 2, edge weights renormalised); both remembered per browser. Endpoint test passed; the two map
  functions checked with node on a small grid.
- Border check: `att_diag_border.py`. (1) Interior-only maps: drop the outer ring, renormalise, re-measure the L1 to
  the default and the part of the full L1 on the ring; stored raw maps on all 5,000 paintings, raw and weighted on the
  304 of the value-weighted check. (2) Grey-ring test on the same 304: each prepared painting shrunk by 64 px
  (bicubic) and pasted at (32, 32) on grey 128, so the grid is unchanged, the outer ring is grey and the painting's
  edge sits on ring 1. Raw and weighted maps from one pass per view.
- Reading rules, fixed before the run (layers 16-27, emotion view first): position if the grey ring holds the
  hottest token about as often as the original outer ring (within 0.10); content if that falls to the ring's token
  share or below and ring 1 gains it; mixed otherwise. The view differences do not need the border if the
  interior-only L1 keeps at least 80% of the full L1 for every instruction (medians).
- Smoke (5 paintings, scratchpad): ran; analysis and sheet work.

### 16:49  Border check results (grey-ring pass 185 s; outputs/border_check/)

Layers 16-27, image only; ring 0 holds 0.207 of the tokens on average in the 304 paintings. Full tables:
`outputs/border_check/summary.md`; sheet `sheet_original_vs_padded.jpg` (weighted maps).

- Grey-ring test, hottest token on the outer ring, original vs grey ring:
  - emotion: raw 0.60 vs 0.70, weighted 0.41 vs 0.54; ring 1 (now the painting's edge) does not gain it (raw 0.08
    vs 0.08, weighted 0.10 vs 0.07). Ring-0 enrichment rises: raw 1.11 vs 1.45, weighted 0.86 vs 1.14.
  - style: raw 0.47 vs 0.39, weighted 0.31 vs 0.24; genre: raw 0.40 vs 0.38, weighted 0.23 vs 0.26; default: raw
    0.25 vs 0.17, weighted 0.15 vs 0.10.
  - Reading against the rule: position. For emotion raw the change is +0.102 (at the band); weighted it is +0.122,
    just outside the band but upward: the grey ring takes the peak even more often. No view shows the content pattern
    (a fall to about 0.21 with ring 1 gaining). Style, genre and default stay within 0.10.
- Interior-only maps (outer ring dropped): the interior L1 to the default keeps 0.91 (emotion), 1.01 (style) and
  1.01 (genre) of the full L1 on the 5,000 stored raw maps (medians), 0.94 / 1.00 / 1.00 weighted (304); rule met.
  The ring holds 0.28 (emotion) and 0.21 (style, genre) of the full L1. Peakedness survives inside: top-1% mass of the
  interior 0.136 / 0.150 / 0.150 against 0.060 for the default (5,000 raw).
- Reading: the sharper instructed maps and the view differences are not a border effect. The emotion view's border
  peak is a pull to the edge of the token grid, whatever the edge shows; the frames and margins of the paintings are
  not what draws it. Why edge tokens draw it (for example, edge tokens holding image-wide information, as register
  tokens do in ViTs) is not checked.
