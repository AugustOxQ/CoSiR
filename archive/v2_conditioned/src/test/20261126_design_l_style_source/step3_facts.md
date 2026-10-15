# Round 7 grill: facts brief (design L, from the style source)

Facts only, no recommendations. Written 2026-11-26 for the round 7 grill. All numbers are seed 42 (12,288 episodes, 4,602 anchor paintings) unless stated. No held row was read; the held numbers are copied from round 6's own result files. Numbers I computed myself from stored arrays or by subtraction are marked "(computed)".

## Five facts most likely to change the plan

1. "Told" never uses labels at test time. It is a one-hot pick of which grouping's heads to use, and the heads are logistic regressions on frozen CLIP features, trained on the label-free grouping. With CSD told, style × genre is only +0.134 [-0.350, +0.595] margin (gain +3.668) on A1c, because the heads place captions poorly (CSD caption head 40.41%). A grouping built from true style labels plugs in as one more grouping, but its heads are CLIP probes: on the 27 true styles they reach 60.9% (image) and 25.4% (caption) (Q4).
2. A label-probe oracle on all true labels already exists (D0 told, 2026-10-04 diagnostics): style × genre +12.65 [+11.83, +13.46] R@1 over B, against +0.35 for the label-free image grouping told (`diagnose_fixes.txt` lines 24 and 27). It is on emotion, style and genre posteriors together, not a style-only grouping.
3. Style and genre labels are only weakly tied: AMI 0.2170, NMI 0.2172 on the 148,956 scorer-train rows with both labels (below the 0.34/0.33 of CSD against each). The largest genre holds 23% to 58% of each of the 10 most frequent styles, except Abstract Expressionism (90%). 22 of 27 styles have at least 3 genres with 50 or more rows; 13 of 27 when counted in paintings (Q3).
4. The run costs are small: Leiden on the CSD painting graph about 10 s, one head 19 to 36 s, scoring one configuration about 20 s (the whole step 1 evaluation of five arms 129 s), round 1's reader about 10 min on local CPU. CSD plus Gram extraction took 754 s on the local GPU (Q5).
5. B′(A1) is the bar to beat on style × genre: seed 42 R@1 21.42 against AFF 20.74 (computed, -0.68); on held rows AFF minus B′(A1) is -1.24 [-1.51, -0.96] (Q2). Round 6's reader and AFF use A0's three groupings (no CSD); the `csd` head is refit in round 6 only as a comparator.

## Q1. How the told ceiling was computed

**Meaning of "told".** For each aspect pair, condition a is about aspect A and b about aspect B. A fixed map sends each aspect to one grouping; the told term for an episode is that grouping's head score, p_h(query) · p_h(candidate), summed in the usual two directions. No label of the episode's own items is used. The map was fixed with the labels (by AMI), so it is a diagnostic ceiling.

| Condition | emotion | style | genre |
|---|---|---|---|
| A0 | affect | image | image |
| A1, A1c | affect | style_csd | image |

Sources: `src/test/20261116_grouping_step1_style/PLAN.md` §4 (arms) and `run_step1.py` `_arm` (`"told": {"emotion": "affect", "style": style_to, "genre": "image"}`); `src/test/20261117_reader_fix_csd/common.py` `TOLD` (line 45).

**Two implementations, same idea.**

| Where | Function | What it does |
|---|---|---|
| Leiden report (+2.23 CSD A1, +1.64 A0) | `run_step1.py::evaluate_general(post, parts, told, ctx, B, pB)` (generalises `run_told_oracle.evaluate_arm`); inside: `diagnose_fixes.stacked(post, ep, parts)` gives probe dots per grouping, `diagnose_fixes.pick(stack, part_idx)` selects the told grouping per episode | the told term is fused on B (cross-fit `crossfit_nested`), compared with its matched counterpart (`crossfit_condition_free` of the two-condition mean); margin = fused minus counterpart (R@1 points) |
| A1c spike (+0.134), commit a3e8003 | `src/test/20261119_no_caption_csd_spike/sp_run.py::onehot_probs` + `common.told_index(parts, TOLD[config], pair_index)`; probabilities are `np.eye(len(parts))[told_index]`, fed through round 2's fusion (`run_r2_fusion`, 224 cells) | same fusion as R1 on the same cells, with the reader's probabilities replaced by a one-hot on the told grouping |

**Inputs.** Per condition: the posterior of every grouping's two heads (an image head and a caption head, each a `LogisticRegression(C=1, max_iter=300)` on CLIP ViT-B/32 features, fitted on the same 60,000 scorer-train rows, `rto.fit_one_head`); a group label for each of the 183,694 scorer-train rows (183,694 rows = 36,518 paintings) to fit the heads; the seed-42 episodes; B (= stored C2, R@1 18.34). The told term is evaluated on the selection rows (32,413).

**Commands.**
- Leiden report arms: stages `group`, `heads`, `eval`, `describe` of `run_step1.py --set style` (per `run_step1.py` docstring; results in `20261116_grouping_step1_style/results/step1_eval_style.txt`). The exact shell lines are not in the log; the docstring gives the stages. Not found: a verbatim command line for the eval stage.
- A1c spike: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python src/test/20261119_no_caption_csd_spike/sp_run.py` (docstring of `sp_run.py`).

**Could true-style labels plug in directly?** Needed changes (facts about the code, not advice):
- A grouping enters `run_step1.py` through the group stage's npz (`gz[name]`, local labels over scorer-train rows), the heads stage (`fit_one_head`, `fit_image_head`) and an arm dict in `_arm`/`ARMS` (the `parts` tuple and `told` map). A true-style grouping would need a new name in `SETS` and `ARMS`, labels from `artelingo_aspect_labels(data)["style"][scorer_train]` (27 classes, smallest class 224 rows; `merge_small` merges below 200 rows), and its own heads.
- `common.py` (round 1) hard-codes `CONFIGS`, `TOLD` and `STEP1_NAME` for A0, A1, AR only; `sp_run.py` imports it, so a new configuration needs entries there too.
- The old posterior route for true labels: `diagnose_fixes.py` line 219 `rc.fit_probes(ctx, artelingo_aspect_labels(ctx.data), scorer_train, rc.PROBE_ROWS)` (D0 probes, CLIP features, 60,000-row draw) with `told_scores`; that is the label-probe oracle of Fact 2.
- Round 1's reader, `rb_*` banks and round 3 AFF are built on A0/A1 groupings; none is retrained by changing the told map.

## Q2. Per-pair numbers, seed 42

R@1 points (percent). e×s = emotion × style, e×g = emotion × genre, s×g = style × genre.

**Condition-free scorers** (`20261119_no_caption_csd_spike/results/sp_tables.txt`, "R@1 (%) fused / counterpart / B / B′ per pair" lines in the log; `B' ` column per section):

| Scorer | e×s | e×g | s×g | pooled |
|---|---|---|---|---|
| B | 12.32 | 21.57 | 21.14 | 18.341 |
| B′(A0) | 12.55 | 21.69 | 21.07 | 18.437 |
| B′(A1) | 13.29 | 21.71 | 21.42 | 18.805 |
| B′(A1c) | 13.11 | 22.13 | 21.55 | 18.931 |

**Told (sp_tables.txt, sections `told_A0_224`, `told_A1_224`, `told_A1c_224`).** Margin = fused minus counterpart; gain = condition gain. A1 and A1c are identical (the told term never uses the caption grouping).

| Told | e×s margin / gain | e×g margin / gain | s×g margin / gain | s×g fused R@1 |
|---|---|---|---|---|
| A0 | +2.643 / +5.267 | +3.003 / +13.159 | -0.397 [-0.640, -0.159] / +0.000 | 21.14 |
| A1 | +2.533 / +6.482 | +2.954 / +13.434 | +0.134 [-0.350, +0.595] / +3.668 [+2.870, +4.415] | 21.82 |
| A1c | +2.533 / +6.482 | +2.954 / +13.434 | +0.134 [-0.350, +0.595] / +3.668 | 21.82 |

Told s×g fused R@1 minus B′ (computed): A0 +0.07, A1 +0.40, A1c +0.27. Told "either" rate on s×g: A1 -3.400.
The Leiden report's version (different fusion, `step1_eval_style.txt` §2), told margin: A0 e×s +2.50, e×g +3.03, s×g -0.61; A1 e×s +3.27, e×g +3.04, s×g +0.39; pooled A0 +1.64, A1 +2.23.

**Label-probe oracle (not a grouping), `20261109_fix_diagnostics/results/diagnose_fixes.txt` lines 24 to 27, Δ vs B (R@1 / gain):**

| Term | e×s | e×g | s×g |
|---|---|---|---|
| T6oracle (label-free heads, A0 told) | 1.72 / 3.02 | 2.42 / 10.66 | 0.35 / 0.00 |
| D0told (label probes) | 10.10 / 12.09 | 14.08 / 26.83 | 12.65 [11.83, 13.46] / 22.72 |

**R1 (round 1's reader, R-c) and AFF.** R1 values are in sp_tables.txt (sections `R1_A0_224`, `R1_A1_224`, `R1_A1c_224`). AFF (A0 reader, round 3) is computed by me from `20261121_round3_affect_gate/results/seed42_arrays.npz` (`aff_fused__r1` by `pair_index`; pooled 19.137, which reproduces the stated +0.700 over B′(A0) and +0.332 over B′(A1) of the round 4 report line 77). Differences by subtraction of means (computed, not paired).

| Scorer (fused R@1) | e×s | e×g | s×g | minus B′(A0): e×s / e×g / s×g | minus B′(A1): e×s / e×g / s×g |
|---|---|---|---|---|---|
| R1 on A0 | 13.35 | 23.02 | 20.39 | +0.80 / +1.33 / -0.68 | +0.06 / +1.31 / -1.03 |
| R1 on A1 | 13.54 | 22.03 | 21.33 | +0.99 / +0.34 / +0.26 | +0.25 / +0.32 / -0.09 |
| R1 on A1c | 13.93 | 22.19 | 21.57 | n/a | +0.64 / +0.48 / +0.15 |
| AFF (A0) | 13.53 | 23.14 | 20.74 | +0.98 / +1.45 / -0.33 | +0.24 / +1.43 / -0.68 |

R1 margins over its own counterpart (sp_tables.txt): A0 +0.708 / +1.221 / -0.598; A1 +0.244 / -0.012 / -0.262; A1c +0.427 / +0.269 / +0.073.
AFF on fresh seeds 49 to 51, pooled, vs B′(A0) (`20261121_round3_affect_gate/results/descriptive.txt`, "AFF per aspect pair"): e×s +0.810, e×g +1.544, s×g -0.580 [-0.793, -0.363].

**Round 6 held read (seeds 52 to 54 pooled; 12,288 episodes per pair), AFF minus comparator.** Source: `docs/reports/auto/v2/2026-11-25_artelingo_held_test.md` lines 93 to 97; the vs B′(A1) column and pooled +0.136 re-read in `20261125_artelingo_held_test/results/descriptive.json`, key `aff_minus_b1.per_pair`.

| Pair | vs B | vs B′(A0) | vs control | vs B′(A1) |
|---|---|---|---|---|
| e×s | +0.86 [+0.64, +1.10] | +0.59 [+0.35, +0.83] | +0.45 | +0.35 [+0.06, +0.62] |
| e×g | +1.60 [+1.38, +1.81] | +1.64 [+1.40, +1.90] | +1.73 | +1.30 [+1.00, +1.60] |
| s×g | -0.57 [-0.75, -0.38] | -0.43 [-0.65, -0.21] | -0.28 | -1.24 [-1.51, -0.96] |

Pooled vs B′(A1): +0.136 [-0.028, +0.303] (`aff_minus_b1.pooled`). Held means (`bar_margin.AFF.pooled`): B′(A0) 18.837, counterpart 18.805, B 18.806, AFF fused 19.436; a B′(A1) mean is not in that key (not found).

## Q3. Labels

**Where.**
- Style: `artelingo_train.json` (`/data/PDD/artelingo/artelingo_train.json`, 308,723 rows), field read by `src/data/artelingo.py` `join_art_styles` into `data.art_styles` (one string per row, one style per painting asserted). 27 classes.
- Genre: not ArtELingo's own; ArtGAN WikiArt Genre CSVs `/data/SSD/wikiart_genre/genre_train.csv` and `genre_val.csv` (rows "path,label"), joined by file stem in `src/data/wikiart_genre.py::load_wikiart_genre`; -1 where unknown. 10 classes (abstract_painting, cityscape, genre_painting, illustration, landscape, nude_painting, portrait, religious_painting, sketch_and_study, still_life).
- Codes for both: `src/data/artelingo_splits.py::artelingo_aspect_labels(data)` (sorted-name codes, -1 missing).
- The painting folder in the image path (e.g. `Impressionism/...`) also names a style (`style_csd_vitl/meta.json` image_paths); I did not check that it equals `art_styles`.

**Rows of the earlier AMIs** (`step1_describe_style.txt` line 13, `run_step1.py::stage_describe`): scorer-train rows, 183,694 (36,518 paintings); AMI with style on all 183,694 rows; AMI with genre on the 148,956 genre-labelled rows. Stored values: style_csd 0.3413 / 0.3302; image_leiden 0.2844 / 0.4358 (the "CLIP image 0.28 / 0.44"); E2 image k-means 0.3176 / 0.3970; style_gram 0.1435 / 0.1871.

**My computation** (`r7_label_stats.py` in this folder; scorer-train rows only, rows with both labels = 148,956):

| Statistic | Value |
|---|---|
| AMI(style, genre) rows | 0.2170 |
| NMI(style, genre) rows | 0.2172 |
| AMI / NMI, one row per painting (29,593 paintings with both) | 0.2171 / 0.2184 |
| Styles with at least 3 genres of 50 or more rows | 22 of 27 (rows); 13 of 27 (paintings) |

Largest genre in the 10 most frequent styles (rows with both labels):

| Style | n rows | Largest genre | Share |
|---|---|---|---|
| Impressionism | 25,546 | landscape | 0.387 |
| Realism | 21,549 | landscape | 0.303 |
| Post_Impressionism | 13,438 | landscape | 0.309 |
| Expressionism | 12,704 | portrait | 0.308 |
| Romanticism | 11,737 | portrait | 0.300 |
| Baroque | 7,590 | portrait | 0.313 |
| Symbolism | 6,795 | landscape | 0.386 |
| Art_Nouveau_Modern | 6,317 | genre_painting | 0.234 |
| Abstract_Expressionism | 6,074 | abstract_painting | 0.899 |
| Northern_Renaissance | 4,797 | religious_painting | 0.576 |

Style class sizes (rows, all 183,694): 28,529 down to 224 (27 classes; the script prints them). Genre rows: portrait 32,039; landscape 31,369; genre_painting 24,606; religious 14,586; abstract 11,744; cityscape 10,781; sketch_and_study 8,489; still_life 6,718; nude 4,363; illustration 4,261. 34,738 scorer-train rows (18.9%) have no genre.
The script touches the held split only through `artelingo_splits` (to get indices); it indexes `scorer_train` alone. It loads `load_artelingo()` (all rows) as every round script does.

## Q4. Placement

Held-out accuracy of the heads on 10,000 scorer-train check rows (not used for fitting); image head / caption head.

| Grouping | Groups | Image head | Caption head | Majority | Source |
|---|---|---|---|---|---|
| style_csd (CLIP heads) | 17 | 85.10% | 40.41% | 13.08% | `step1_heads_style.txt` |
| style_csd, image head on CSD features | 17 | 91.67% | (caption head unchanged) | 13.08% | same |
| style_gram | 11 | 71.53% | 31.92% | 13.12% | same |
| style_rand | 17 | 15.64% | 13.20% | 13.18% | same |
| E2 image k-means | 64 | 92.7% | 22.9% | n/a | Leiden report Table 2 |
| E2 caption k-means | 64 | 21.6% | 89.7% | n/a | same |
| affect k-means | 64 | 13.5% | 34.6% | n/a | same |

Placeability P_ami (`step1_describe_style.txt` lines 5 to 9): style_csd 0.2084 (CLIP heads), 0.1981 (CSD image head); style_gram 0.1497 / 0.1242; E2 image 0.2416; E2 caption 0.1959.

**"Style from captions at 25.4%".** `docs/reports/stage/2026-10-04_aspect_conditioned_similarity_methods.md` line 281: a logistic-regression probe on frozen CLIP ViT-B/32 caption features predicting the true 27-class ArtELingo style, scoring 25.4% on selection rows, against 60.8% from images (the backbone-check log gives 60.9). Probes: `LogisticRegression(C=1, max_iter=300)` fitted on scorer-train rows, tested on selection rows (`src/test/20261025_backbone_check/run_backbone_check.py::artelingo_measure`; table row in `20261025_backbone_check_log.md` line 17: emotion img 35.1, txt 56.9, style img 60.9, txt 25.4). Four backbones moved it only to 26.3. It is a probe accuracy on true labels, not a head on a grouping.

## Q5. Cost (wall time)

| Step | Time | Source |
|---|---|---|
| Leiden on CSD, 36,518 scorer-train paintings (the 61,402 images are not all in the training rows) | graph 5 s + Leiden 4.5 s; stage 26 s per grouping incl. seeds 43/44; whole group stage 61 s (CSD, Gram, random) | `20261116_grouping_step1_style/results/run_group_style.log` |
| Leiden on affect probabilities (183,694 rows) | 7 to 8.5 min per cell, three CPU processes | Leiden report §5 |
| Caption Leiden on 183,694 rows | 297 s | `20261116_grouping_step1_style_log.md` line 52 |
| One head, CLIP features, 60,000 rows | style_csd 19 s; source image head 36 s; affect head 23 s; 3 style groupings' heads stage 86 s | `run_heads_style.log`, `run_r3_seed42.log` |
| Round 1's reader (R-b A0/A1 chain, local CPU, 3 processes) | heads and banks 8 min, train 2 min, eval 3 min; none on DAS6 | `20261117_reader_fix_csd_log.md` line 23 |
| Scoring one configuration on seed 42 | R1 candidate 21 s per configuration; spike of 3 configurations plus told 177 s; step 1 eval of 5 arms 129 s; round 3 seed-42 run 114 s (bundle 33 s); round 6 bundle with RCA and PM 155 s | `sp_run.log`, `run_eval_style.log`, `run_r3_seed42.log`, `run_r6_picks.log` |
| Extracting CSD and Gram for 61,402 images (one joint pass, local RTX 3090) | total 753.7 s: Gram fit pass 103 s, PCA 23 s, joint CSD plus Gram pass 613 s (100 img/s); CSD alone not timed | `style_csd_vitl/meta.json` timing, `results/extract_full.log` |

## Q6. Code to reuse

**Cached features** (`/data/SSD2/pre_extract/artelingo/`):
- `style_csd_vitl/embeddings.npy` float32 (61402, 768), L2-normalised; `style_vgg19_gram/embeddings.npy` float32 (61402, 640). One row per distinct image, in order of first appearance in `artelingo_train.json`.
- Join (C6): `row_to_image.npy` int32 (308723,) maps `load_artelingo()` row r to the image row (= `row_to_image_annotation_order.npy[data.sample_ids]`); features in dataset row order are `embeddings[row_to_image]`. `meta.json` holds the image path list, hashes of both maps and the annotation file (`6a4e5b17...`). `run_step1.py::real_style_features` re-asserts the alignment (hash checks, one-to-one image to painting, `sample_ids` identity order False).
- Caveat: sample ids are not the identity order (`sample_ids_identity_order: False`), so never index by `range(n)`.

**Code locations.**
- CSD extraction: `src/test/20261116_grouping_step1_style/extract_style_features.py` (CSD plus Gram, one pass) and `csd_model.py`; weights `tomg-group-umd/CSD-ViT-L` (revision c94c4d51, sha256 40e92fad...).
- Groupings (kNN graph, Leiden, merge small, random slot), heads, arms, describe: `run_step1.py` (`painting_grouping`, `stage_group`, `stage_heads`, `fit_image_head`, `evaluate_general`, `stage_describe`); calls `20261111_community_told_oracle/run_told_oracle.py` (`fit_one_head`, `merge_small`, `evaluate_arm`). Graph and Leiden: `src/model/communities.py::detect_communities`.
- Regressing the CLIP image embedding out of CSD: no code found (grep of the grouping folders for residual/regress gave no hit beyond `csd_model.py`'s unrelated text).
- Reader and scoring, latest chain: round 1 `20261117_reader_fix_csd/` (`common.py` `CONFIGS`/`TOLD`/`load_bundle`, `rb_build.py`, `rb_eval.py`, `rc_core.py`), round 2 `20261118_reader_fix_round2/` (`r2_common`, `run_r2_fusion`), spike `20261119_no_caption_csd_spike/` (A1c reader `sp_reader.py`, run `sp_run.py`).
- AFF (frozen): `20261121_round3_affect_gate/` (`r3_bundle.build_bundle`, `r3_fusion.score_frozen`, `run_r3_seed42.py`).
- Latest full pipeline (heads incl. `csd`, bundle, frozen-pick scoring, descriptive): `20261125_artelingo_held_test/` — `r6_heads.py` (`fit_all`, 5 heads: affect 41, affect_km 64, image 64, caption 64, csd 17), `r6_bundle.py` (`build_bundle_r6`), `r6_score.py` (`score_seed`, `frozen_cells`), `r6_picks.py`, `r6_context.py` (`RowContext`, modes selection or held), `run_r6_descriptive.py`. Its reader is round 1's A0 half-readers (`load_readers`); it has no A1/A1c reader.
- Pitfalls: new round folders need a new module prefix (`r7_`); develop on seed 42; held rows and test seeds are ledgered.

## Not found
- A verbatim shell command for `run_step1.py` eval stage (only the stage names and the docstring).
- A CSD-only extraction time; code for regressing CLIP out of CSD; a check that the image-path folder equals `art_styles`.
