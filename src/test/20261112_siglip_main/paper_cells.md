# r28 numbers sheet: SigLIP 1 cells for paper Tables 2 to 8

Written 2026-11-12 by r28 job H. Check: `python r28_cells_check.py` (same folder) reads every source named below and the .tex files and compares at three decimals.

How to read: Table numbers follow the order of the Tables folder (2 main_clustering, 3 main_retrieval, 4 mmart, 5 ablation, 6 encoders, 7 photos, 8 cost). Each cell group is three columns: `now` (the current cell in the .tex, with `L<n>` = line n of that file), `SigLIP 1` (the new value, a mean over five seeds unless the source is a single scored value), `source` (record file and key; `{a}` = emotion / style / genre in that order, `{k}` = class count 9 / 27 / 10). Cells are Emotion / Style / Genre. `r28` = `src/test/20261112_siglip_main/records/`, `r17` = `src/test/20261103_new_scores/records/`, `r18` = `src/test/20261103_outside_methods/records/`, `r20` = `src/test/20261111_encoder_photo_tables/records/`. All values are ArtELingo test unless said. Rows with no SigLIP 1 value: the cell says why (the model keeps its own backbone, or it is outside this round).

## Encoder pick on validation (`r28/valpick.json`)

ArtELingo validation, 6152 paintings, c* (`k11=40`), five seeds, means. Rule (`rule` key): highest mean of the six cells (AMI at 9 / 27 / 10 and plain-cosine mAP). Pick: **siglip1** (`pick/pick`), margin 0.0007 over the second (`pick/margin`, a close call). Qwen3 is an information row, not a candidate.

| Encoder | AMI emotion | AMI style | AMI genre | mAP emotion | mAP style | mAP genre | mean of six | source |
|---|---|---|---|---|---|---|---|---|
| siglip1 | 0.104 | 0.457 | 0.538 | 0.268 | 0.346 | 0.604 | 0.386 | `r28/valpick.json :: encoders/siglip1/cells/ami or map, slash, aspect/mean`, `encoders/siglip1/score` |
| siglip2 | 0.102 | 0.457 | 0.543 | 0.267 | 0.341 | 0.603 | 0.386 | `r28/valpick.json :: encoders/siglip2/cells/ami or map, slash, aspect/mean`, `encoders/siglip2/score` |
| clip_l14 | 0.102 | 0.437 | 0.528 | 0.264 | 0.326 | 0.596 | 0.376 | `r28/valpick.json :: encoders/clip_l14/cells/ami or map, slash, aspect/mean`, `encoders/clip_l14/score` |
| qwen | 0.103 | 0.418 | 0.523 | 0.267 | 0.312 | 0.594 | 0.370 | `r28/valpick.json :: encoders/qwen/cells/ami or map, slash, aspect/mean`, `encoders/qwen/score` |

Gates in the file: AMI and mAP of siglip1 and siglip2 equal r17 / r14 records (`encoders/<e>/gate/*/pass`). Validation numbers are not paper cells, so the check script does not compare this section.

## Table 2 (main_clustering.tex): AMI at the class counts

<!-- table: T2 -->

| Row | now AMI | SigLIP 1 AMI | source AMI |
|---|---|---|---|
| Plain embedding, Qwen3 -> SigLIP 1 | L9: 0.070 / 0.235 / 0.425 | 0.061 / 0.382 / 0.402 | `r17/verdicts.json :: id=SC1/artelingo/siglip1/{a}/ami/{k}/plain => comp` |
| + TURTLE, Qwen3 -> SigLIP 1 | L10: 0.067 / 0.215 / 0.436 | 0.057 / 0.337 / 0.365 | `r18/labels.json :: id=SC5/artelingo/siglip1/{a}/ami/{k}/plain_turtle => comp` |
| Instructed Qwen3 | L11: 0.089 / 0.406 / 0.501 | none: stays Qwen3, 3 passes | none |
| TPIPS (8B) | L13: 0.072 / 0.332 / 0.496 | none: own 8B model; TPIPS not run on SigLIP (handoff section 5) | none |
| CLAY, CLIP L/14 -> SigLIP 1 | L14: 0.068 / 0.406 / 0.485 | 0.069 / 0.437 / 0.504 | `r28/clay_crl.json :: cells/artelingo/{a}/ami/clay/siglip1` |
| CRL, CLIP L/14 -> SigLIP 1 | L15: 0.077 / 0.390 / 0.428 | 0.075 / 0.375 / 0.393 | `r28/clay_crl.json :: cells/artelingo/{a}/ami/crl/siglip1` |
| BERTopic (MiniLM) | L17: 0.038 / 0.105 / 0.260 | none: keeps its own model | none |
| VLM2Vec (Qwen2-VL) | L18: 0.064 / 0.373 / 0.459 | none: keeps its own model | none |
| PercepT, CLIP L/14 | L19: 0.064 / 0.262 / 0.474 | none: keeps its own model | none |
| PercepT, Qwen3 -> SigLIP 1 (matched) | L20: 0.068 / 0.192 / 0.424 | 0.070 / 0.225 / 0.466 | `r17/verdicts.json :: id=SC2/artelingo/siglip1/{a}/ami/{k}/matched => comp` |
| CoSiR (ours), Qwen3 -> SigLIP 1 | L22: 0.102 / 0.411 / 0.510 | 0.099 / 0.441 / 0.528 | `r28/per_seed.json :: ours_siglip1/ami/{a}/mean` |

## Table 3 (main_retrieval.tex): label mAP, picked read-out

Image to image: plain, PercepT and CoSiR cells are the SigLIP 1 cells of Table 6 (read-out picked on validation). CLAY and CRL read-outs: `clay_crl.json cells/artelingo/<aspect>/map/<method>/readout` (CLAY none+k10, CRL center+k10, picked on SigLIP 1 validation). Text directions: read-out per cell in `text.json cells/<aspect>/<direction>/readout`; plain SigLIP 1 uses SigLIP 1's own text tower, CoSiR's caption tower stays on the Qwen3 plain caption embedding.

<!-- table: T3 -->

| Row | now mAP | SigLIP 1 mAP | source mAP |
|---|---|---|---|
| Image to image: Plain embedding -> SigLIP 1 | L10: 0.248 / 0.192 / 0.536 | 0.245 / 0.296 / 0.504 | `r17/verdicts.json :: id=SC7/artelingo/siglip1/{a}/map/-/plain => comp` |
| Instructed Qwen3 | L11: 0.268 / 0.322 / 0.612 | none: stays Qwen3 | none |
| TPIPS (8B) | L12: 0.256 / 0.236 / 0.563 | none: own 8B model | none |
| CLAY -> SigLIP 1 | L13: 0.249 / 0.314 / 0.569 | 0.249 / 0.351 / 0.585 | `r28/clay_crl.json :: cells/artelingo/{a}/map/clay/siglip1` |
| CRL -> SigLIP 1 | L14: 0.255 / 0.282 / 0.503 | 0.256 / 0.300 / 0.513 | `r28/clay_crl.json :: cells/artelingo/{a}/map/crl/siglip1` |
| VLM2Vec | L15: 0.248 / 0.280 / 0.559 | none: keeps its own model | none |
| PercepT, CLIP L/14 | L16: 0.242 / 0.177 / 0.540 | none: keeps its own model | none |
| PercepT, Qwen3 -> SigLIP 1 (matched) | L17: 0.248 / 0.152 / 0.505 | 0.244 / 0.166 / 0.548 | `r17/verdicts.json :: id=SC8/artelingo/siglip1/{a}/map/-/matched => comp` |
| CoSiR (ours) i2i | L18: 0.269 / 0.323 / 0.615 | 0.269 / 0.355 / 0.623 | `r28/per_seed.json :: ours_siglip1/i2i/{a}/mean` |
| Text to image: Plain embedding | L21: 0.222 / 0.134 / 0.418 | 0.211 / 0.152 / 0.419 | `r28/text.json :: plain_siglip1/cells/{a}/t2i/mean` |
| Text to image: Instructed Qwen3 | L22: 0.268 / 0.157 / 0.443 | none: stays Qwen3 | none |
| Text to image: CoSiR (ours) | L23: 0.264 / 0.172 / 0.484 | 0.264 / 0.179 / 0.489 | `r28/text.json :: ours_siglip1/cells/{a}/t2i/mean` |
| Image to text: Plain embedding | L26: 0.240 / 0.110 / 0.343 | 0.220 / 0.107 / 0.354 | `r28/text.json :: plain_siglip1/cells/{a}/i2t/mean` |
| Image to text: Instructed Qwen3 | L27: 0.304 / 0.124 / 0.397 | none: stays Qwen3 | none |
| Image to text: CoSiR (ours) | L28: 0.287 / 0.138 / 0.403 | 0.286 / 0.141 / 0.407 | `r28/text.json :: ours_siglip1/cells/{a}/i2t/mean` |
| Text to text: Plain embedding | L31: 0.253 / 0.096 / 0.273 | 0.206 / 0.092 / 0.253 | `r28/text.json :: plain_siglip1/cells/{a}/t2t/mean` |
| Text to text: Instructed Qwen3 | L32: 0.377 / 0.098 / 0.299 | none: stays Qwen3 | none |
| Text to text: CoSiR (ours) | L33: 0.346 / 0.109 / 0.315 | 0.346 / 0.109 / 0.316 | `r28/text.json :: ours_siglip1/cells/{a}/t2t/mean` |

## Table 4 (mmart.tex): MMArt test, AMI at 9 / 20 / 10 and image mAP

Read-outs picked on the ArtELingo validation paintings that MMArt labels (`r28/mmart.json :: picks/siglip1/*`: CoSiR center+k10, plain none+k10, PercepT center+k0; CLAY and CRL as in `clay_crl.json cells/mmart/...`).

<!-- table: T4 -->

| Row | now AMI | SigLIP 1 AMI | source AMI | now mAP | SigLIP 1 mAP | source mAP |
|---|---|---|---|---|---|---|
| Plain embedding | L11: 0.038 / 0.231 / 0.458 | 0.035 / 0.391 / 0.410 | `r28/mmart.json :: siglip1/plain_ami/values[{i}]` | L11: 0.194 / 0.205 / 0.515 | 0.191 / 0.303 / 0.493 | `r28/mmart.json :: siglip1/plain_map/values[{i}]` |
| + TURTLE | L12: 0.038 / 0.223 / 0.422 | 0.030 / 0.349 / 0.372 | `r28/mmart.json :: siglip1/turtle_ami/values[{i}]` | L12: -- / -- / -- | none: TURTLE gives clusters only (stays --) | none |
| Instructed Qwen3 | L13: 0.049 / 0.416 / 0.526 | none: stays Qwen3 | none | L13: 0.202 / 0.348 / 0.617 | none: stays Qwen3 | none |
| TPIPS (8B) | L15: 0.038 / 0.334 / 0.490 | none: own 8B model | none | L15: 0.194 / 0.257 / 0.567 | none: own 8B model | none |
| CLAY | L16: 0.040 / 0.424 / 0.504 | 0.043 / 0.447 / 0.509 | `r28/clay_crl.json :: cells/mmart/{a}/ami/clay/siglip1` | L16: 0.192 / 0.338 / 0.576 | 0.193 / 0.375 / 0.593 | `r28/clay_crl.json :: cells/mmart/{a}/map/clay/siglip1` |
| CRL | L17: 0.034 / 0.412 / 0.442 | 0.040 / 0.384 / 0.409 | `r28/clay_crl.json :: cells/mmart/{a}/ami/crl/siglip1` | L17: 0.192 / 0.310 / 0.508 | 0.198 / 0.325 / 0.519 | `r28/clay_crl.json :: cells/mmart/{a}/map/crl/siglip1` |
| PercepT, Qwen3 -> SigLIP 1 (matched) | L19: 0.039 / 0.192 / 0.439 | 0.038 / 0.220 / 0.480 | `r28/mmart.json :: siglip1/percept_ami/values[{i}]` | L19: 0.193 / 0.167 / 0.504 | 0.190 / 0.182 / 0.547 | `r28/mmart.json :: siglip1/percept_map/values[{i}]` |
| CoSiR (ours) | L21: 0.053 / 0.423 / 0.531 | 0.053 / 0.455 / 0.539 | `r28/mmart.json :: siglip1/cstar_ami/values[{i}]` | L21: 0.201 / 0.347 / 0.620 | 0.202 / 0.383 / 0.628 | `r28/mmart.json :: siglip1/cstar_map/values[{i}]` |

## Table 5 (ablation.tex)

AMI at the class counts and plain-cosine mAP, five seeds (c* is the replay of r14's saved SigLIP 1 models, gate exact against r17: `ablation_ref_replay.json gate_pass`). The plain row is `ablation_plain.json` (single deterministic run). The four removal rows are the change from c* and are not run yet; the paper chat fills them from the ablation runs.

<!-- table: T5 -->

| Row | now AMI | SigLIP 1 AMI | source AMI | now mAP | SigLIP 1 mAP | source mAP |
|---|---|---|---|---|---|---|
| CoSiR (ours) = c* | L11: 0.102 / 0.411 / 0.510 | 0.099 / 0.441 / 0.528 | `r28/ablation_ref_replay.json :: ami/{a}/mean` | L11: 0.264 / 0.311 / 0.590 | 0.263 / 0.340 / 0.599 | `r28/ablation_ref_replay.json :: map_plain_cosine/{a}/mean` |
| Instructed graphs (removal) | L13: -0.033 / -0.186 / +0.010 | pending (ablation runs) | none | L13: -0.021 / -0.122 / +0.001 | pending (ablation runs) | none |
| Content correction (removal) | L14: -0.016 / -0.068 / 0.000 | pending (ablation runs) | none | L14: -0.007 / -0.058 / -0.001 | pending (ablation runs) | none |
| Graph-positive loss (removal) | L15: -0.029 / -0.140 / +0.008 | pending (ablation runs) | none | L15: -0.018 / -0.098 / 0.000 | pending (ablation runs) | none |
| Relational term (removal) | L16: +0.001 / -0.002 / -0.071 | pending (ablation runs) | none | L16: 0.000 / -0.001 / -0.129 | pending (ablation runs) | none |
| Plain embedding | L18: 0.070 / 0.235 / 0.425 | 0.061 / 0.382 / 0.402 | `r28/ablation_plain.json :: encoders/siglip1/cells/{a}/ami` | L18: 0.242 / 0.180 / 0.476 | 0.244 / 0.268 / 0.455 | `r28/ablation_plain.json :: encoders/siglip1/cells/{a}/map` |

## Table 6 (encoders.tex): SigLIP 1 block, already in the .tex

Nothing new. The SigLIP 1 cells (lines 23 to 25) are listed with their sources; the `now` and `SigLIP 1` columns are equal. They are also the SigLIP 1 cells of Tables 2 and 3.

<!-- table: T6 -->

| Row | now AMI | SigLIP 1 AMI | source AMI | now mAP | SigLIP 1 mAP | source mAP |
|---|---|---|---|---|---|---|
| Plain embedding | L23: 0.061 / 0.382 / 0.402 | 0.061 / 0.382 / 0.402 | `r17/verdicts.json :: id=SC1/artelingo/siglip1/{a}/ami/{k}/plain => comp` | L23: 0.245 / 0.296 / 0.504 | 0.245 / 0.296 / 0.504 | `r17/verdicts.json :: id=SC7/artelingo/siglip1/{a}/map/-/plain => comp` |
| PercepT | L24: 0.070 / 0.225 / 0.466 | 0.070 / 0.225 / 0.466 | `r17/verdicts.json :: id=SC2/artelingo/siglip1/{a}/ami/{k}/matched => comp` | L24: 0.244 / 0.166 / 0.548 | 0.244 / 0.166 / 0.548 | `r17/verdicts.json :: id=SC8/artelingo/siglip1/{a}/map/-/matched => comp` |
| CoSiR (ours) | L25: 0.099 / 0.441 / 0.528 | 0.099 / 0.441 / 0.528 | `r28/per_seed.json :: ours_siglip1/ami/{a}/mean` | L25: 0.269 / 0.355 / 0.623 | 0.269 / 0.355 / 0.623 | `r28/per_seed.json :: ours_siglip1/i2i/{a}/mean` |

## Table 7 (photos.tex): Affection and EmoSet, columns Affection AMI / Affection mAP / EmoSet AMI / EmoSet mAP

`now` is the Qwen3 row for the same method (lines 12 to 14), except the SigLIP 1 rows, where `now` is the SigLIP 1 row already in the .tex (lines 19, 20; `--` = EmoSet empty). "Paintings only" has no SigLIP 1 row in the .tex yet (new row). Affection photo model 0.182 / 0.323: source `r20/cells.json :: photo/affection/siglip1/photo/*/value` (also plain 0.099 / 0.255). EmoSet photo model, painting-only and plain: `r28/photos.json` cells `a` (arm photo_siglip1), `b_aff` / `b_emo` (cstar_siglip1), `c` (plain_siglip1); read-outs picked on each set's validation split (`cells/<c>/readout`). `photos.json` cells `g_aff_cstar_qwen`, `g_emo_cstar_qwen` replay the Qwen3 painting-only row and reproduce it (0.123 / 0.276 / 0.385 / 0.441, gate pass).

<!-- table: T7 -->

| Row | now Aff AMI / Aff mAP / Emo AMI / Emo mAP | SigLIP 1 Aff AMI / Aff mAP / Emo AMI / Emo mAP | source Aff AMI / Aff mAP / Emo AMI / Emo mAP |
|---|---|---|---|
| Instructed Qwen3 | L11: 0.155 / 0.295 / 0.394 / 0.452 | none: stays Qwen3 | none |
| Plain embedding, Qwen3 -> SigLIP 1 (Affection cells already in the .tex) | L19: 0.099 / 0.255 / -- / -- | 0.099 / 0.255 / 0.267 / 0.364 | `r20/cells.json :: photo/affection/siglip1/plain/ami/value` ; `r20/cells.json :: photo/affection/siglip1/plain/map/value` ; `r28/photos.json :: cells/c/ami/mean` ; `r28/photos.json :: cells/c/map/mean` |
| CoSiR paintings only, Qwen3 -> SigLIP 1 (new row) | L13: 0.123 / 0.276 / 0.385 / 0.441 | 0.150 / 0.288 / 0.374 / 0.446 | `r28/photos.json :: cells/b_aff/ami/mean` ; `r28/photos.json :: cells/b_aff/map/mean` ; `r28/photos.json :: cells/b_emo/ami/mean` ; `r28/photos.json :: cells/b_emo/map/mean` |
| CoSiR (ours), Qwen3 -> SigLIP 1 (Affection cells already in the .tex) | L20: 0.182 / 0.323 / -- / -- | 0.182 / 0.323 / 0.432 / 0.483 | `r20/cells.json :: photo/affection/siglip1/photo/ami/value` ; `r20/cells.json :: photo/affection/siglip1/photo/map/value` ; `r28/photos.json :: cells/a/ami/mean` ; `r28/photos.json :: cells/a/map/mean` |

## Table 8 (cost.tex)

New: the time of the instructed Qwen3 text pass and of the caption tower, per caption, one RTX 3090, batch 32, model time only, 2000 ArtELingo captions, generic instruction "Represent the user's input.", mean of 3 repeats (`time_text.json`). The .tex prints one decimal (ms); values here are three decimals. Existing image rows are unchanged (19.5 ms SigLIP 1, 88.6 ms Qwen3, 266.8 ms instructed; r19 `/data/SSD2/cosir_r19/20261103_past_picks/outputs/stage3/cost_table.json`, not re-read here, not checked by the script).

<!-- table: T8 -->

| Row | now ms | SigLIP 1 ms | source ms |
|---|---|---|---|
| Qwen3 text pass, ms per caption (new; Qwen3-VL-Embedding-2B, 1 pass) | n/a | 3.616 | `r28/time_text.json :: qwen_text_pass/ms_per_caption_mean_of_repeats` |
| Caption tower, three aspects, ms per caption (new) | n/a | 0.017 | `r28/time_text.json :: caption_tower_three_aspects/ms_per_caption_mean_of_repeats` |

Repeat range for the text pass: 3.588 to 3.643 ms; for the caption tower 0.016 to 0.018 ms (`ms_per_caption_repeat_min/max`). The caption tower has 4,197,376 parameters per aspect head (`caption_tower_params_each`).

## Per-seed values

Not copied here. `r28/per_seed.json`: for every Table 2 and Table 3 image cell of CoSiR on SigLIP 1, `ours_siglip1/<ami|i2i>/<aspect>/{per_seed, mean, min, max, seeds_above_instructed_qwen3, instructed_qwen3}`, with the Qwen3 model's own five seeds under `ours_qwen3/...` and the unseeded instructed Qwen3 under `instructed_qwen3/...`. `ours_siglip1/{t2i,i2t,t2t}/<aspect>` hold the text directions. `r28/text.json`: `cells/<aspect>/<direction>/per_seed` for ours, plain and the Qwen3 rows. Seeds of ours on SigLIP 1 above the instructed Qwen3 (of 5; the instructed value is one run):

| Cell | emotion | style | genre |
|---|---|---|---|
| ami | 5 of 5 | 5 of 5 | 5 of 5 |
| i2i | 4 of 5 | 5 of 5 | 5 of 5 |
| t2i | 0 of 5 | 5 of 5 | 5 of 5 |
| i2t | 0 of 5 | 5 of 5 | 5 of 5 |
| t2t | 0 of 5 | 5 of 5 | 5 of 5 |

(`ours_siglip1/<cell>/<aspect>/seeds_above_instructed_qwen3`; this small table is read from the file by the generator, not compared by the check script.)

## Cells not filled

Cells of Tables 2 to 8 with a SigLIP 1 slot but no value:

- Table 5, rows Instructed graphs, Content correction, Graph-positive loss, Relational term, all six cells each (24 cells): "pending (ablation runs)". The four removals have not been trained on SigLIP 1.
- Table 4, + TURTLE, the three mAP cells: TURTLE gives clusters only, no retrieval embedding (stays `--`).
- Table 7, Plain embedding and CoSiR on SigLIP 2 and on CLIP L/14, the four EmoSet cells per row (stay `--`) and the SigLIP 2 / CLIP L/14 "paintings only" rows: outside this round (the plan scores EmoSet on Qwen3 and SigLIP 1 only).
- Table 3, CLAY and CRL: only the image-to-image block exists in the .tex, so no text-direction slot.
- No SigLIP 1 slot by plan (backbone kept): Instructed Qwen3 (Tables 2, 3, 4, 7), TPIPS, VLM2Vec, BERTopic, PercepT on CLIP L/14.
- Table 8: no new cell besides the two text-pass times; the image-time rows were not re-measured.
