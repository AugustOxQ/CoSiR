# r17 saved-outputs inventory (Sonnet worker, 2026-11-03 19:05; read only; shapes from npz/npy headers, counts from the label code)

Abbreviations: Q = Qwen3-VL-Embedding-2B, S2 = SigLIP 2, S1 = SigLIP 1, L14 = CLIP L/14. E/S/G = emotion/style/genre.
`SSD2` = `/data/SSD2`. `MAINT` = `/project/CoSiR/src/test` (frozen, gitignored outputs live only in the main checkout).
Existence and shapes were read from file headers; array contents, hashes and sidecars were not re-verified except where noted.

Key facts that apply to every cell
- "c* three heads" = three separately trained checkpoints per seed (`weights/k11=40/{emotion,style,genre}__s{seed}.pt`), each exporting its own 2048-wide float16 vector per image. Every encoder (Q, S2, S1, L14) gives 2048 wide c* vectors, so three heads concatenated = 6144 wide.
- Plain embeddings are width 2048 (Q), 1152 (S2, S1), 768 (L14), float16, stored once for all images of the dataset and joined by painting name or key.
- PercepT matched outputs are topic probabilities `prob`/`probs` float32 [n, 40], per seed. They are not only cluster assignments.
- CRL L14 image-side vectors are NOT saved anywhere. r11 CRL = L14 image embedding times the 50 value text embeddings (cosine scores, width 50 per aspect), computed on the fly in `rt_data.raw('crl', ...)`. The saved parts are the L14 stores and the text arrays (below).
- Row order: ArtELingo held/val = sorted painting names (`rt_data.names(split)`); MMArt exports = keyed (`keys` array inside the npz) over all 16,782 "new" paintings, test = 16,384 subset of those via `records/cutsets.json`; Affection/EmoSet = keyed.

## Table 1. ArtELingo held (12,281; genre 9,949 labelled)

| method x encoder | path | shape | seeds | present | note |
|---|---|---|---|---|---|
| c* Q, 3 heads | `SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/held/k11=40/{emotion,style,genre}__s{s}.npz` (+`.json`) key `img` | [12281, 2048] f16 | 0-4 | yes (15 npz) | r10's c* (k11=40). Also removal arms (`k11=40-no_*`) in the same folder |
| c* S2 | `SSD2/cosir_r14/20261102_backbones_sweep/outputs/cstar/siglip2/held/k11=40/{aspect}__s{s}.npz` key `img` | [12281, 2048] f16 | 0-4 | yes (15 npz) | |
| c* S1 | same, `.../cstar/siglip1/held/k11=40/` | [12281, 2048] f16 | 0-4 | yes (15 npz) | |
| c* L14 | same, `.../cstar/clip_l14/held/k11=40/` | [12281, 2048] f16 | 0-4 | yes (15 npz) | |
| plain Q | `SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b/img.npy` (+`index.json`) | [61402, 2048] f16 | none | yes | gather held rows by painting name (`rt_data._store`, `rb_encoders.full_images`) |
| plain S2 | `.../artelingo_full/siglip2_so400m/img.npy` | [61402, 1152] f16 | none | yes | |
| plain S1 | `.../artelingo_full/siglip_so400m/img.npy` | [61402, 1152] f16 | none | yes | |
| plain L14 | `.../artelingo_full/clip_l14/img.npy` | [61402, 768] f16 | none | yes | |
| matched PercepT Q (P3) | `MAINT/20261020_claim_test/outputs/stage2/full/held/percept/core__k40/P3__s{s}.npz` key `prob` | [12281, 40] f32 | 0-4 | yes | topic posteriors; `.pt` mappers in `.../stage2/full/percept/core/stage2/k40/` (10 .pt: P2 and P3) |
| matched PercepT S2 (M2) | `MAINT/20261025_recipe_backbones/outputs/part2/full/held/M2/P3__s{s}.npz` key `prob` | [12281, 40] f32 | 0-4 | yes | mapper `.pt`: `.../part2/full/matched/M2/P3__s{s}.pt` (5 present) |
| matched PercepT S1 (M8) | `SSD2/cosir_r10/20261025_encoders_genre/outputs/part2/full/held/M8/P3__s{s}.npz` | [12281, 40] f32 | 0-4 | yes | `.pt` in `.../full/matched/M8/` (5 present) |
| matched PercepT L14 (M7) | `SSD2/cosir_r10/20261025_encoders_genre/outputs/part2/full/held/M7/P3__s{s}.npz` | [12281, 40] f32 | 0-4 | yes | `.pt` in `.../full/matched/M7/` (5 present) |
| matched PercepT partitions (old protocol) | `MAINT/20261025_recipe_backbones/outputs/part2/partitions/full/{M2,F2,K2,T,B1..}/` and `SSD2/cosir_r10/20261025_encoders_genre/outputs/part2/partitions/full/{M7,M8,F7,F8,K7,K8,T}/` | labels [3 or 1, 12281] int | per arm | yes | only needed to replay r14; k-means/own readings are recomputable from `prob` |
| teacher Q, E | `SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts/emotion_focus_held/img.npy` (+`ids.npz`, `meta.json`) | [12281, 2048] f16 | none | yes | instruction `emotion_focus` |
| teacher Q, S | `.../style_word_held/img.npy` | [12281, 2048] f16 | none | yes | `style_word` |
| teacher Q, G | `.../genre_word_held/img.npy` | [12281, 2048] f16 | none | yes | `genre_word` |
| CRL L14 | not saved as vectors. Inputs: L14 plain store (above) + `SSD2/cosir_r11/20261029_retrieval/outputs/crl/{emotion,style,genre}_text.npy` | text [50, 768] each | none | inputs yes, vectors no | `rt_data.raw('crl','held',aspect)` returns [12281, 50] f64 cosine scores; CPU, seconds |
| labels E/S/G | code only: `rt_data.labels('held')` (needs `rs_vh`, `ct2_data` on path; annotations `rs_vh.ANNOTATIONS`, genre CSVs `rs_vh.GENRE_CSVS`) | int64 [12281], genre -1 = unlabelled | none | yes (code) | |

## Table 2. MMArt test (16,384; validation 5,769)

All c* exports, matched probs and teacher views cover all 16,782 "new" paintings (keys in file). Test = subset by `records/cutsets.json` (`test`, n 16,384, cosine cut 0.95; also `far`, n 10,403, cut 0.85). Select with `mt_data.keys('test')`.

| method x encoder | path | shape | seeds | present | note |
|---|---|---|---|---|---|
| c* Q/S2/S1/L14, 3 heads, test | `SSD2/cosir_r15/20261102_mmart_test/outputs/exports/new/{qwen,siglip2,siglip1,clip_l14}/cstar/{aspect}__s{s}.npz` keys `keys`, `emb` | [16782, 2048] f16 | 0-4 | yes (15 npz each, 4 encoders) | frozen held-trained c*, no MMArt training |
| c* validation | `.../exports/val/{enc}/cstar/{aspect}__s{s}.npz` | [5769, 2048] f16 | 0-4 | yes (15 each) | used for the read-out picks and seed ranges |
| c* shared held | `.../exports/shared_held/{enc}/cstar/...` | [11528, 2048] f16 | 0-4 | yes (15 each) | diagnostic set |
| plain Q | `SSD2/pre_extract/mmart/qwen3vl_emb_2b/img.npy` + `keys.json` | [16782, 2048] f16 | none | yes | |
| plain S2 | `.../mmart/siglip2_so400m/img.npy` | [16782, 1152] f16 | none | yes | |
| plain S1 | `.../mmart/siglip_so400m/img.npy` | [16782, 1152] f16 | none | yes | |
| plain L14 | `.../mmart/clip_l14/img.npy` | [16782, 768] f16 | none | yes | also the input CRL would need |
| plain, validation | ArtELingo stores (Table 1) gathered by `mt_data.features(view, val_keys)` | 5,769 rows | none | yes | `mt_data.features` joins by painting name |
| matched PercepT Q/S2/S1/L14, test | `.../exports/new/{enc}/matched/probs__s{s}.npz` keys `keys`, `probs` | [16782, 40] f32 | 0-4 | yes (5 each) | CPU inference of the held mappers (Table 1 `.pt`) |
| matched PercepT, validation or shared held | none | - | - | no | `exports/val/*/matched` and `shared_held/*/matched` are empty (0 files) |
| teacher Q, E/S/G, test | `SSD2/pre_extract/mmart/qwen3vl_emb_2b_aspect_prompts/{emotion_focus,style_word,genre_word}/img.npy` | [16782, 2048] f16 each | none | yes | |
| teacher, validation | ArtELingo `..._s2img` stores (`artelingo_full/qwen3vl_emb_2b_aspect_prompts/{emotion_focus,style_word,genre_word}_s2img/img.npy` + `ids.npz`) | 29,232 rows (per code assert; not opened) | none | yes | `mt_data.features` reads them for val keys |
| CRL L14 | none scored in r15. Inputs present: `mmart/clip_l14/img.npy` + r11 `crl/{aspect}_text.npy` | text [50, 768] | none | vectors no, inputs yes | CPU, seconds |
| labels E/S/G | `mt_data.labels(keys, aspect)`; index `/data/SSD/mmart/mmart_index.parquet` | int64, -1 = none | none | yes | MMArt style 20 classes, genre = WikiArt 10 classes |

## Table 3. Affection test (8,501; validation 4,251) and EmoSet test (17,348 after cut; validation 5,770 after cut)

Affection and EmoSet are emotion only. "Photo model" = r13/r14 photo-fine-tuned model, single emotion head.

| method x encoder | path | shape | seeds | present | note |
|---|---|---|---|---|---|
| Affection c* Q, 3 heads | `SSD2/cosir_r12/20261030_affection_confirmation/outputs/exports/cstar/{val,test}/{emotion,style,genre}__s{s}.npz` key `img` | test [8501, 2048], val [4251, 2048] f16 | 0-4 | yes | Q only. No Affection c* exports for S2/S1/L14 |
| Affection photo model Q | `SSD2/cosir_r14/.../outputs/exports/affection/photo_qwen/{val,test}__s{s}.npz` keys `keys`, `emb`; r13 copy `SSD2/cosir_r13/20261031_photo_finetune/outputs/exports/final/s{s}/{val,test}/emotion.npz` | [8501 or 4251, 2048] f16 | 0-4 | yes | |
| Affection photo model S2/S1/L14 | `.../exports/affection/photo_{siglip2,siglip1,clip_l14}/{val,test}__s{s}.npz` | [8501 or 4251, 2048] f16 | 0-4 | yes | |
| Affection c* Q, one head | `.../exports/affection/cstar_qwen/{val,test}__s{s}.npz` | [8501 / 4251, 2048] f16 | 0-4 | yes | |
| Affection PercepT Q (P3) | `SSD2/cosir_r12/.../exports/percept/{val,test}/P3__s{s}.npz` key `prob` | [8501 / 4251, 40] f32 | 0-4 | yes | Q only; no S2/S1/L14 PercepT on photos |
| Affection plain Q/S2/S1 | `SSD2/pre_extract/affection_local/{qwen3vl_emb_2b,siglip2_so400m,siglip_so400m}/img.npy` | [85007, 2048 or 1152] f16 | none | yes | join by key |
| Affection plain L14 | `.../affection_local/clip_l14_valtest/img.npy` [12752, 768] and `clip_l14_train` [72255, 768] | f16 | none | yes | two folders |
| Affection teacher Q | `.../affection_local/qwen3vl_emb_2b_aspect_prompts/{emotion_focus,emotion_focus_photo}/img.npy` (+ `_train`, `_txt`) | 85,007-row store expected (not opened) | none | yes | artwork and photo wordings |
| Affection CRL L14 | scores built on the fly (`ac_data._crl`); text: `SSD2/cosir_r12/.../outputs/crl/{clip,qwen}_{painting,photo}_emotion.npy`; cached partitions in `.../outputs/partitions/affection__test__crl_*` | text [50, 768] | none | vectors no | |
| EmoSet photo model Q/S2/S1/L14 | `SSD2/cosir_r14/.../exports/emoset/photo_{qwen,siglip2,siglip1,clip_l14}/{val,test}__s{s}.npz` keys `keys`, `emb` | test [17716, 2048], val [5905, 2048] f16 (uncut) | 0-4 | yes | cut to 17,348 / 5,770 with `bs_emoset.keys(split, cut=True)` (record `MAINT/20261102_backbones_sweep/records/nearcopy_emoset.json`) |
| EmoSet c* Q | `.../exports/emoset/cstar_qwen/{val,test}__s{s}.npz` | same shapes | 0-4 | yes | single head (head name not checked) |
| EmoSet plain Q/S2/S1/L14 | `SSD2/pre_extract/emoset/{qwen3vl_emb_2b,siglip2_so400m,siglip_so400m,clip_l14}/img.npy` + `keys.json` | [23621, 2048/1152/1152/768] f16 | none | yes | all EmoSet splits in one store |
| EmoSet teacher Q | `.../emoset/qwen3vl_emb_2b_aspect_prompts/{emotion_focus,emotion_focus_photo}/img.npy` | [23621, 2048] f16 | none | yes | |
| EmoSet CRL L14 | on the fly in `bs_emoset_score` from `emoset/clip_l14` + r12 CRL text | - | none | vectors no | |
| EmoSet PercepT | none | - | - | no | |
| labels | Affection: `ac_data.labels(split)` (int, 9 classes); EmoSet: `bs_emoset.labels(keys)` (strings, 8 classes) | | | yes (code) | |

## Table 4. Validation splits (where read-outs and seed ranges were set)

| cell | path | shape | seeds | present | note |
|---|---|---|---|---|---|
| ArtELingo val (6,152; genre 4,960) c* Q | `SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/train/k11=40/R6-5/{emotion__Gres,style__Fres05,genre__R0o}__s{s}.npz` keys `img`, `cap` | [6152, 2048] f16 each | 0-4 | yes (15 npz) | file names carry the graph row (`rt_data.GRAPHS`); read via `rt_data.raw('cstar','val',aspect,seed)` |
| ArtELingo val c* S2/S1/L14 | `SSD2/cosir_r14/.../outputs/cstar/{siglip2,siglip1,clip_l14}/train/k11=40/R6-5/...` | [6152, 2048] f16 | 0-4 | yes (15 each) | |
| ArtELingo val plain | Table 1 stores, gathered by the 6,152 val names | | none | yes | |
| ArtELingo val teacher Q | `..._s2img` stores (29,232 rows, gather) | | none | yes | |
| ArtELingo val PercepT Q | `SSD2/cosir_r11/20261029_retrieval/outputs/percept_val_full/P3__s{s}.npz` key `prob` | [6152, 40] f32 | 0-4 | yes | |
| ArtELingo val PercepT S2/S1/L14 | none | - | - | no | held-only; not exported |
| MMArt val (5,769) | Table 2 (`exports/val/...`) | | 0-4 | yes | MMArt labels; val = ArtELingo val paintings shared with MMArt |
| Affection / EmoSet val | Table 3 | | 0-4 | yes | |
| Read-out picks and ranges | `MAINT/20261102_mmart_test/records/stage1_{qwen,siglip2,siglip1,clip_l14}.json`, `stage1_teacher.json`; `MAINT/20261102_backbones_sweep/records/stage1_{siglip2,siglip1,clip_l14}.json`, `stage2_emoset.json`; r11 `MAINT/20261029_retrieval/stage0a_ranges.json`; r12 `SSD2/cosir_r12/.../outputs/stage0b/ranges.json`, `stage1/picks.json` | JSON | | yes | pick for c* on MMArt is `center+k10` for all four encoders; frozen picks `none+k10` (Q, S1) or `center+k10` (S2, L14) |

## a. Class counts (distinct labelled classes actually present) and class sizes

Counted from the label code on 2026-11-03 (`rt_data.labels`, `mt_data.labels`, `ac_data.labels`, `bs_emoset.labels`).

| dataset, split | aspect | labelled rows | classes present | min class size | max class size |
|---|---|---|---|---|---|
| ArtELingo held (12,281) | E | 12,281 | 9 | 153 | 4,244 |
| | S | 12,281 | 27 | 15 | 1,949 |
| | G | 9,949 | 10 | 296 | 2,230 |
| ArtELingo val (6,152) | E / S / G | 6,152 / 6,152 / 4,960 | 9 / 27 / 10 | 77 / 5 / 153 | 2,148 / 930 / 1,058 |
| MMArt test (16,384) | E | 16,218 | 9 (incl. "something else") | 413 | 4,437 |
| | S | 16,384 | 20 | 171 | 2,665 |
| | G | 13,420 | 10 | 356 | 3,092 |
| MMArt val (5,769) | E / S / G | 5,720 / 5,769 / 4,679 | 9 / 20 / 10 | 168 / 72 / 143 | 1,454 / 915 / 1,018 |
| Affection test (8,501) | E | 8,501 | 9 | 81 (anger) | 2,626 (contentment) |
| Affection val (4,251) | E | 4,251 | 9 | 54 | 1,387 |
| EmoSet test (17,348) | E | 17,348 | 8 | 1,641 (disgust) | 2,954 (excitement) |
| EmoSet val (5,770) | E | 5,770 | 8 | 510 (anger) | 956 (excitement) |

Note: MMArt E has 9 classes including "something else"; ArtELingo E has the same 9 names. K for k-means stays 20 in all r10-r15 scoring (EmoSet also has a descriptive K=8 row).

## b. Existing scoring code (import paths)

Round folders go on `sys.path`; each module prefix is unique. Order that works (as `mt_ret.py` does): `20261102_backbones_sweep`, `20261030_affection_confirmation`, `20261029_retrieval`, `20261027_recipe_sweep` under `/project/CoSiR/src/test/`, then `import ac_paths` (it pulls the older rounds in) and the repo root `/project/CoSiR` for `src.*`. All of these run on CPU. Note that `rs_paths`/`rt_paths` point to frozen inputs in the MAIN checkout.

Clustering
- k-means partitions, k fixed at 20: `ac_cluster.partitions(x, k=20, states=(0,1,2))` -> int32 [3, n]; refuses any other k (asserts). Underlying `rs_vh.r20_partition(x)` also fixes K=20, MiniBatchKMeans, n_init 3, batch 4096, states 0-2, unit norm checked.
- k as a parameter: `bs_emoset_score.partitions_k(x, k)` -> int32 [3, n] (same settings; k=20 equals `ac_cluster.partitions`). This is the only function with a k argument.
- PercepT readings: `ac_cluster.percept_partitions(prob [n,40])` -> {'r20': int32 [3,n] (k-means 20 on L2-normalised probs), 'own': int32 [1,n] (argmax)}. `mt_clu.matched_from_probs({seed: prob})` does the same for five seeds.
- mean AMI: `ac_cluster.ami(parts, labels)`; parts int [3,n] (unseeded), [5,3,n] (seeded) or own [1,n]/[5,1,n]; labels < 0 masked; returns float (mean over states, then seeds).
- paired bootstrap: `ac_cluster.paired(cand_parts [5,3,n], comp_parts [3,n] or [5,3,n], labels, draws=2000, workers=8, shared_seeds=False)` -> {A_cand, A_comp, d, interval [lo, hi], bootstrap_d}. Uses r10's `rg_held.bootstrap_d_t` (unseeded comparator) or `bootstrap_d_r` (seeded). Rows resampled, then seeds.
- r14/r15 orchestration (MMArt aware): `mt_clu.score_encoder(enc, keyset='test', scs=('SC1','SC3','SC4'), n_draws=None, workers=8, write=True)` and the lower-level `mt_clu.score_aspect(enc, aspect, labels, c_fixed, frozen=, matched=, teacher=, scs, r=, n_draws, workers, data, matched_name)`; builders `mt_clu.cstar_fixed({seed: x})`, `single_fixed(x)`, `matched_fixed(parts)`; k fixed at 20 (`mt_clu.kmeans_partitions(x)` -> `ct_score.output_partitions(x,'k1')['r20']`). `score_encoder` is bound to the committed stage 1 records and `mt_data` and writes `OUT/stage2/cells_clu.json`. ArtELingo equivalent: `bs_artclu.score_encoder`; margins `bs_artclu.frozen_margin / matched_margin / teacher_margin(fixed_c, fixed_x, labels, n_resamples=2000, workers)` (matched takes max of own and r20 readings per draw; d = candidate minus the larger).
- outcome rule: `rt_stats.outcome(lower, r)` -> 'WIN' (lower > 0), 'LEVEL' (lower > -r), else 'BELOW'. Same rule in the r14/r15 `make_cell` code paths.

Retrieval
- `rt_metrics.retrieval(x float32 unit [n,d], labels int [n], qe_k=0, workers=8)` -> {'ap' float64 [nq], 'p10', 'dropped', 'rows'}; label mAP = mean of 'ap'; full gallery ranking; self excluded; labels < 0 dropped; `qe_k` = query expansion neighbours (k10 = `qe_k=10`). `rt_metrics.graded(x, dist)` gives nDCG.
- read-out: `rt_readout.ReadOut(w, size, k)` with w in none/center/whiten, k in (0,1,3,5,10); `rt_readout.menu(width)`, `fit(ro, x)`, `apply(params, x)`, `stage5_apply(ro, x_val, x_eval)` (fit the map on all validation rows, apply to the scored rows), `crossfit_all`, `select(rows, ranges)` (guard + score + tie rule).
- per-arm AP at a pick: `ac_stats.arm_retrieval(get_val(seed), get_eval(seed), readout, labels, seeds=None or (0..4), workers=8)` -> {ap [nq] or [5,nq], rows, dropped, point, per_seed}. `mt_ret.arm_ap` wraps it.
- paired bootstrap on AP: `rt_stats.paired(cand float64 [n] or [5,n], comp, draws=10000)` -> {d, interval [lo, hi], boot}. Both sides must be float64; query counts must match.
- MMArt wrapper: `mt_ret.score_encoder(enc, keyset='test', scs=('SC2','SC4'), n_draws=10000, workers=8, ranges=None, picks=None, teacher=None)`; ranges/picks are read from committed records (`mt_paths.read_committed_record` refuses uncommitted files outside smoke). The generic pieces (`ac_stats.arm_retrieval`, `rt_stats.paired`, `rt_readout`) have no such guard.
- Not found: a ready scorer for "untold" retrieval (concatenated heads), and none for retrieval on matched PercepT posteriors (r11/r12 had `rt_data._percept`/`ac_data.percept_prob` for Q).

## c. Seed ranges (LEVEL margin)

- Definition: r = max minus min over c* seeds 0-4 of the validation score (AMI at k=20 mean of three states, or mAP at R0), per aspect, unrounded (`bs_artclu.seed_range`, `rt_stats.seed_range`, `bs_artret.own_ranges`, `mt_clu.clustering_ranges`, `mt_ret.validation_ranges`). Each encoder uses its own range; validation labels and validation exports only.
- MMArt (validation 5,769; own c*): `MAINT/20261102_mmart_test/records/stage1_<enc>.json`, keys `clu_ranges`, `ret_ranges` (with `per_seed`). Values clu E/S/G: Q 0.0048/0.0118/0.0097; S2 0.0044/0.0071/0.0073; S1 0.0038/0.0190/0.0100; L14 0.0030/0.0135/0.0086. ret mAP E/S/G: Q 0.0015/0.0026/0.0037; S2 0.0011/0.0029/0.0044; S1 0.0006/0.0051/0.0042; L14 0.0006/0.0038/0.0019. Copies with sources under `SSD2/cosir_r15/.../outputs/stage1/{clu,ret}_ranges_<enc>.json`.
- ArtELingo val (6,152): S2/S1/L14 in `MAINT/20261102_backbones_sweep/records/stage1_<enc>.json` (`artclu`, `artret_ranges`; also `affection` clu/map/ndcg). Clustering E/S/G S2 0.0029/0.0160/0.0070; S1 0.0038/0.0085/0.0115; L14 0.0037/0.0086/0.0159. Q: retrieval map E/S/G 0.00141/0.00223/0.00382 in `MAINT/20261029_retrieval/stage0a_ranges.json` (key `map`, with per_seed); Q clustering uses the glossary values 0.0046/0.0082/0.0128 (`bs_artclu.GLOSSARY_R`; the original r10 file was not located).
- Affection: r12 `SSD2/cosir_r12/.../outputs/stage0b/ranges.json` (Q); r14 records `affection` block for S2/S1/L14. EmoSet: `MAINT/20261102_backbones_sweep/records/stage2_emoset.json` (not opened).
- Note the MMArt ranges differ from the ArtELingo ones (different validation labels), so they are not interchangeable.

## d. PercepT matched: are retrieval-usable outputs saved?

- Yes. Held ArtELingo: topic posteriors `prob` [12281, 40] float32 for Q (P3), S2 (M2), S1 (M8), L14 (M7), seeds 0-4 (Table 1). MMArt test: `probs` [16782, 40] float32 for all four encoders, seeds 0-4 (Table 2). The embedding per image is the 40-dim posterior vector (k-means 20 on its L2-normalised form, or argmax = own topic). Cluster assignments are not the only artefact.
- Not saved: matched PercepT on ArtELingo val for S2/S1/L14, on MMArt val/shared held, on Affection (only Q) and EmoSet (none), and the "40-topic multi-hot" soft assignments beyond `prob`.
- Rerun for the missing sets: no GPU needed. `mt_matched.infer_cpu(mapper_path, images float32)` runs the saved `.pt` mapper on CPU; the r15 log says matched exports cost about 14 s per encoder (MMArt new, 16,782 images, 5 seeds). Mappers exist for all four encoders (5 seeds each). Image inputs must be the encoder's normalised plain store. Training a new topic fit on MMArt is not needed and was not done (caption embeddings deferred).

## e. Untold retrieval (three c* heads concatenated, one image query, scored against each aspect's labels)

Confirmed. For every encoder and seed, the three heads are separate files (`emotion__s{s}`, `style__s{s}`, `genre__s{s}`), each [n, 2048] float16 (ArtELingo held: key `img`; MMArt / Affection exports: key `emb` or `img`). Concatenated width = 6144. Same seed index across the three heads is the natural pairing, but the heads of one seed are three independent training runs (not one joint model). Row order is identical across the three heads of one set (ArtELingo held = sorted names; MMArt = `keys` array, check equality before concatenation). The three vectors need unit normalisation per head before concatenation if the intended read-out is cosine on the concatenation (rows are near unit norm already: `rt_data._cstar` asserts |norm-1| < 1e-3). No concatenated vector is saved; it is built on the fly (MMArt: 16,782 x 6144 f16 = 206 MB per seed). Read-out picks were set per head with each aspect's own pick; none exist for a concatenated vector, so a new pick on validation (`rt_readout`, `ac_select.arm_table`) is needed.

## Missing outputs and cost to regenerate (from logged times where stated)

1. CRL L14 image-side vectors on ArtELingo held, MMArt, Affection, EmoSet: not saved; inputs all present (L14 stores plus the r11/r12 text arrays). Cost: seconds on CPU (matrix product). Nothing logged for it as a separate step.
2. CRL on MMArt: never scored in r15; no text-array check for MMArt-specific wording (r11 texts are aspect-wide, 50 values per aspect). Cost as item 1.
3. Matched PercepT S2/S1/L14 on ArtELingo validation, MMArt validation/shared held, and EmoSet, and S2/S1/L14 on Affection: not saved. Only needed if the new round selects something for PercepT on validation. Cost: about 14 s per encoder for MMArt-sized sets on CPU (r15 log), no GPU.
4. Concatenated c* vector (untold retrieval): not saved, built from the saved heads. Cost: negligible; the new read-out pick on validation costs on the order of 9-16 min per encoder (r15 logged picks 540-957 s).
5. c* heads on Affection/EmoSet for S2, S1, L14: not exported (only Q c*; photo models for all four). c* weights exist, so export is CPU; r14 logged exports at 32-46 s per encoder for held plus photo sets, c* on MMArt about 72 s per encoder. No training needed.
6. Teacher views for S/G aspects on Affection/EmoSet: not applicable (emotion only datasets).
7. Original r10 clustering seed-range file for Q on ArtELingo val: not located (only the glossary constants in `bs_artclu.GLOSSARY_R`). Recompute from the saved Q val c* exports (clustering of 5 seeds x 3 aspects; r14 logged 390-520 s per encoder for held scoring).
8. Not checked: head name of `exports/emoset/cstar_qwen`, size of the Affection teacher stores, sidecar/hash validity of the files (the loaders assert these on read), `stage2_emoset.json` content, whether the MMArt `unseen_artist` set exists as a list.

Reference timings from logs (for sizing r17 scoring): r15 stage 2 scorer, clustering 637-825 s per encoder (SC1, SC3, SC4, 2000 draws), retrieval 250-488 s per encoder (10,000 draws); r14 held scoring 390-520 s clustering and about 120 s retrieval per encoder.

## What I would remove (nothing deleted)

Nothing needs removal for r17. Candidates only if space is needed (not checked as unused): `SSD2/cosir_r14/20261102_backbones_sweep/c13_smoke` (5.4 MB, smoke), `SSD2/cosir_r10/20261025_encoders_genre/outputs/part2/regression/*.lock` files, `SSD2/cosir_r11/.../stage0b/check_failed_A1_sample_scale.json` (a failed-gate record: keep as record). These are small; the large folders (r10 `full` 17 GB, r14 `cstar` 5.5 GB, r15 `exports` 9.0 GB) hold the inputs r17 needs.
