# r29 numbers sheet: the SigLIP-only adapter beside the current SigLIP 1 model (K0)

Written 2026-11-14 01:48 by `r29_ladder.py sheet` from the cell records present in `records/`; a missing piece shows `not run`. K0 values are r28's (`per_seed.json`, `mmart.json`, `photos.json`, r20 `cells.json`; for SC4 r17 SC1 and r14 SC6): five-seed means on test, three decimals. New cells: value (Q1 cell's value; for image mAP the aspect head, the P1 cell is the untold candidate and shows its own value), the paired difference d from the comparator with its 95% interval, r (the LEVEL unit) and the outcome by spec §3.4: WIN, LEVEL, BELOW, or INCONCLUSIVE. Q1 claim: WIN "above", LEVEL "level with", BELOW none. A failed pass rule (D9) does not hide the cells of SC3: they stay on this sheet. The sheet decides nothing: the levels are the ladder records'.

"Level" lines come from `records/<claim>_<arm>.json`; "K* - K0" is diagnostic (d), measured after the levels.

Pass rule (SC2): status `fail`, level 0. stage 2 ran on the fail under A1.

## SC3 (ArtELingo test, 15 cells; Tables 2 and 3): K* against P1 (plain embedding) and Q1 (instructed Qwen3)

Arm `siglip1_prefix` (encoder siglip1); cell files: image read, text read.

| Cell | K0 (r28) | new value | vs P1 | vs Q1 | Q1 claim | K* - K0 (d) |
|---|---|---|---|---|---|---|
| emotion ami | 0.099 | 0.102 | WIN d +0.0406 [+0.0363, +0.0446] r 0.0110 | WIN d +0.0129 [+0.0093, +0.0165] r 0.0110 | above | +0.0033 [+0.0006, +0.0061] |
| emotion map | 0.269 | 0.270 | WIN d +0.0157 [+0.0151, +0.0164] r 0.0015 (cell value 0.261) | WIN d +0.0012 [+0.0001, +0.0022] r 0.0015 | above | +0.0008 [+0.0001, +0.0016] |
| emotion t2i | 0.264 | 0.262 | WIN d +0.0508 [+0.0497, +0.0518] r 0.0023 | BELOW d -0.0060 [-0.0070, -0.0051] r 0.0023 | none | -0.0023 [-0.0035, -0.0012] |
| emotion i2t | 0.286 | 0.281 | WIN d +0.0603 [+0.0583, +0.0624] r 0.0029 | BELOW d -0.0229 [-0.0253, -0.0207] r 0.0029 | none | -0.0058 [-0.0071, -0.0045] |
| emotion t2t | 0.346 | 0.327 | WIN d +0.1211 [+0.1193, +0.1230] r 0.0040 | BELOW d -0.0493 [-0.0509, -0.0477] r 0.0040 | none | -0.0191 [-0.0213, -0.0169] |
| style ami | 0.441 | 0.446 | WIN d +0.0644 [+0.0594, +0.0692] r 0.0080 | WIN d +0.0401 [+0.0356, +0.0441] r 0.0080 | above | +0.0057 [+0.0022, +0.0092] |
| style map | 0.355 | 0.355 | BELOW d -0.0090 [-0.0107, -0.0074] r 0.0038 (cell value 0.287) | WIN d +0.0333 [+0.0315, +0.0353] r 0.0038 | above | -0.0005 [-0.0023, +0.0015] |
| style t2i | 0.179 | 0.175 | WIN d +0.0237 [+0.0222, +0.0253] r 0.0037 | WIN d +0.0189 [+0.0172, +0.0205] r 0.0037 | above | -0.0040 [-0.0054, -0.0024] |
| style i2t | 0.141 | 0.137 | WIN d +0.0295 [+0.0284, +0.0305] r 0.0023 | WIN d +0.0127 [+0.0117, +0.0135] r 0.0023 | above | -0.0036 [-0.0046, -0.0026] |
| style t2t | 0.109 | 0.107 | WIN d +0.0147 [+0.0139, +0.0156] r 0.0017 | WIN d +0.0089 [+0.0081, +0.0097] r 0.0017 | above | -0.0029 [-0.0038, -0.0019] |
| genre ami | 0.528 | 0.517 | WIN d +0.1155 [+0.1046, +0.1261] r 0.0261 | WIN d +0.0157 [+0.0061, +0.0255] r 0.0261 | above | -0.0107 [-0.0209, -0.0013] |
| genre map | 0.623 | 0.623 | WIN d +0.0457 [+0.0431, +0.0482] r 0.0037 (cell value 0.549) | WIN d +0.0105 [+0.0086, +0.0122] r 0.0037 | above | +0.0002 [-0.0015, +0.0018] |
| genre t2i | 0.489 | 0.485 | INCONCLUSIVE | WIN d +0.0417 [+0.0393, +0.0442] r 0.0038 | above | -0.0039 [-0.0060, -0.0018] |
| genre i2t | 0.407 | 0.404 | WIN d +0.0502 [+0.0483, +0.0521] r 0.0032 | WIN d +0.0077 [+0.0062, +0.0092] r 0.0032 | above | -0.0022 [-0.0042, -0.0002] |
| genre t2t | 0.316 | 0.317 | WIN d +0.0645 [+0.0622, +0.0667] r 0.0064 | WIN d +0.0184 [+0.0162, +0.0206] r 0.0064 | above | +0.0011 [-0.0025, +0.0044] |

Ladder (SC3, `SC3_siglip1_prefix.json`): **level 1**; 13 of 15 WIN over P1, 12 WIN or LEVEL over Q1 (level 1 needs 12 of 15). Sentence: "With no Qwen3 pass at test time, training adds over the plain embedding in 13 of 15 cells."

## SC9 (validation encoder pick)

stage 2 ran on the fail under A1

| Encoder | Five-seed mean | Seed range | Per-seed groups |
|---|---|---|---|
| siglip1 (K*) | +0.3882 | +0.0055 | s0: +0.3900, s1: +0.3893, s2: +0.3862, s3: +0.3905, s4: +0.3851 |
| siglip2 (E2) | +0.3838 | +0.0061 | s0: +0.3859, s1: +0.3845, s2: +0.3845, s3: +0.3841, s4: +0.3798 |
| clip_l14 (E3) | +0.3762 | +0.0034 | s0: +0.3761, s1: +0.3770, s2: +0.3759, s3: +0.3776, s4: +0.3742 |
| Qwen (information, not a candidate) | +0.3697 | +0.0028 | s0: +0.3706, s1: +0.3677, s2: +0.3705, s3: +0.3703, s4: +0.3696 |

Pick: siglip1; margin +0.0045; ahead in 5 of 5 seeds: s0, s1, s2, s3, s4; close call: True.

## SC4 (Table 6, ArtELingo test, 6 cells per encoder; K0 on that encoder shown beside, descriptive)

Arm `clipl14_prefix` (encoder clip_l14); cell files: image read.

| Cell | K0 (r28) | new value | vs P1 | vs Q1 | Q1 claim | K* - K0 (d) |
|---|---|---|---|---|---|---|
| emotion ami | 0.098 | 0.096 | WIN d +0.0330 [+0.0288, +0.0374] r 0.0089 | WIN d +0.0067 [+0.0027, +0.0109] r 0.0089 | above | -0.0026 [-0.0056, +0.0010] |
| emotion map | 0.267 | 0.267 | WIN d +0.0117 [+0.0111, +0.0124] r 0.0041 (cell value 0.260) | LEVEL d -0.0016 [-0.0027, -0.0004] r 0.0041 | level with | -0.0002 [-0.0012, +0.0007] |
| style ami | 0.422 | 0.422 | WIN d +0.0593 [+0.0543, +0.0641] r 0.0112 | WIN d +0.0157 [+0.0109, +0.0203] r 0.0112 | above | -0.0002 [-0.0038, +0.0035] |
| style map | 0.335 | 0.336 | WIN d +0.0025 [+0.0013, +0.0037] r 0.0029 (cell value 0.276) | WIN d +0.0143 [+0.0127, +0.0160] r 0.0029 | above | +0.0011 [-0.0003, +0.0026] |
| genre ami | 0.516 | 0.516 | WIN d +0.1188 [+0.1099, +0.1277] r 0.0262 | WIN d +0.0141 [+0.0059, +0.0220] r 0.0262 | above | +0.0000 [-0.0075, +0.0077] |
| genre map | 0.618 | 0.618 | WIN d +0.0463 [+0.0440, +0.0485] r 0.0012 (cell value 0.544) | WIN d +0.0051 [+0.0033, +0.0069] r 0.0012 | above | -0.0002 [-0.0009, +0.0006] |

Ladder (SC4, `SC4_clip_l14_clipl14_prefix.json`): **level 3**; 6 of 6 WIN over P1, 6 WIN or LEVEL over Q1 (level 1 needs 5 of 6). Sentence: "With no Qwen3 pass at test time, training adds over the plain embedding in every cell and it is level with or above the instructed Qwen3 in 6 of 6 cells."

Arm `siglip2_prefix` (encoder siglip2); cell files: image read.

| Cell | K0 (r28) | new value | vs P1 | vs Q1 | Q1 claim | K* - K0 (d) |
|---|---|---|---|---|---|---|
| emotion ami | 0.098 | 0.101 | WIN d +0.0472 [+0.0429, +0.0512] r 0.0111 | WIN d +0.0121 [+0.0085, +0.0156] r 0.0111 | above | +0.0027 [-0.0000, +0.0052] |
| emotion map | 0.269 | 0.269 | WIN d +0.0169 [+0.0162, +0.0176] r 0.0026 (cell value 0.261) | LEVEL d +0.0011 [-0.0001, +0.0021] r 0.0026 | level with | +0.0003 [-0.0005, +0.0010] |
| style ami | 0.439 | 0.439 | WIN d +0.1153 [+0.1095, +0.1194] r 0.0187 | WIN d +0.0324 [+0.0277, +0.0361] r 0.0187 | above | -0.0006 [-0.0038, +0.0025] |
| style map | 0.348 | 0.348 | WIN d +0.0254 [+0.0239, +0.0269] r 0.0036 (cell value 0.283) | WIN d +0.0260 [+0.0244, +0.0276] r 0.0036 | above | -0.0007 [-0.0019, +0.0007] |
| genre ami | 0.527 | 0.527 | WIN d +0.1717 [+0.1620, +0.1816] r 0.0181 | WIN d +0.0251 [+0.0168, +0.0337] r 0.0181 | above | -0.0004 [-0.0102, +0.0093] |
| genre map | 0.623 | 0.622 | WIN d +0.0521 [+0.0496, +0.0547] r 0.0024 (cell value 0.549) | WIN d +0.0096 [+0.0079, +0.0113] r 0.0024 | above | -0.0005 [-0.0020, +0.0009] |

Ladder (SC4, `SC4_siglip2_siglip2_prefix.json`): **level 3**; 6 of 6 WIN over P1, 6 WIN or LEVEL over Q1 (level 1 needs 5 of 6). Sentence: "With no Qwen3 pass at test time, training adds over the plain embedding in every cell and it is level with or above the instructed Qwen3 in 6 of 6 cells."

## SC5 (Table 4, MMArt test, 6 cells)

Arm `siglip1_prefix` (encoder siglip1); cell files: mmart read.

| Cell | K0 (r28) | new value | vs P1 | vs Q1 | Q1 claim |
|---|---|---|---|---|---|
| emotion ami | 0.053 | 0.055 | WIN d +0.0201 [+0.0174, +0.0229] r 0.0061 | WIN d +0.0055 [+0.0032, +0.0079] r 0.0061 | above |
| emotion map | 0.202 | 0.202 | WIN d +0.0059 [+0.0056, +0.0062] r 0.0004 (cell value 0.197) | LEVEL d +0.0001 [-0.0004, +0.0005] r 0.0004 | level with |
| style ami | 0.455 | 0.458 | WIN d +0.0664 [+0.0613, +0.0712] r 0.0127 | WIN d +0.0420 [+0.0372, +0.0470] r 0.0127 | above |
| style map | 0.383 | 0.383 | WIN d +0.0036 [+0.0020, +0.0051] r 0.0039 (cell value 0.307) | WIN d +0.0351 [+0.0333, +0.0370] r 0.0039 | above |
| genre ami | 0.539 | 0.535 | WIN d +0.1250 [+0.1167, +0.1327] r 0.0212 | WIN d +0.0090 [+0.0015, +0.0154] r 0.0212 | above |
| genre map | 0.628 | 0.628 | WIN d +0.0446 [+0.0429, +0.0465] r 0.0038 (cell value 0.538) | WIN d +0.0113 [+0.0098, +0.0127] r 0.0038 | above |

Ladder (SC5, `SC5_siglip1_prefix.json`): **level 3**; 6 of 6 WIN over P1, 6 WIN or LEVEL over Q1 (level 1 needs 5 of 6). Sentence: "With no Qwen3 pass at test time, training adds over the plain embedding in every cell and it is level with or above the instructed Qwen3 in 6 of 6 cells."

## SC6 (Table 7, photo model F1, 4 cells: Affection and EmoSet test)

Arm `siglip1_prefix` (encoder siglip1); cell files: photo read.

| Cell | K0 (r28) | new value | vs P1 | vs Q1 | Q1 claim |
|---|---|---|---|---|---|
| affection ami | 0.182 | 0.183 | WIN d +0.0843 [+0.0786, +0.0901] r 0.0076 | WIN d +0.0277 [+0.0216, +0.0332] r 0.0076 | above |
| affection map | 0.323 | 0.325 | WIN d +0.0705 [+0.0673, +0.0733] r 0.0058 | WIN d +0.0305 [+0.0275, +0.0331] r 0.0058 | above |
| emoset ami | 0.432 | 0.432 | WIN d +0.1652 [+0.1530, +0.1796] r 0.0309 | WIN d +0.0382 [+0.0265, +0.0528] r 0.0309 | above |
| emoset map | 0.483 | 0.487 | WIN d +0.1234 [+0.1193, +0.1275] r 0.0069 | WIN d +0.0354 [+0.0313, +0.0393] r 0.0069 | above |

Ladder (SC6, `SC6_siglip1_prefix.json`): **level 3**; 4 of 4 WIN over P1, 4 WIN or LEVEL over Q1 (level 1 needs 3 of 4). Sentence: "With no Qwen3 pass at test time, training adds over the plain embedding in every cell and it is level with or above the instructed Qwen3 in 4 of 4 cells."

## Table 5 (ablation, ArtELingo test): K0 rows; removals of K*: from `ablation.json`

| Row | K0 AMI emotion / style / genre | K0 mAP emotion / style / genre | K* AMI emotion / style / genre | K* mAP emotion / style / genre |
|---|---|---|---|---|
| CoSiR (ours) = c* | 0.099 / 0.441 / 0.528 | 0.263 / 0.340 / 0.599 | 0.102 / 0.446 / 0.517 | 0.263 / 0.339 / 0.599 |
| Instructed graphs (removal), change from c* (K*: change from K*, paired 95% interval) | -0.0309 / -0.2101 / -0.0074 | -0.0197 / -0.1435 / -0.0011 | -0.0357 [-0.0406, -0.0308] / -0.2147 [-0.2181, -0.2069] / +0.0042 [-0.0080, +0.0155] | -0.0199 [-0.0214, -0.0184] / -0.1416 [-0.1445, -0.1385] / -0.0010 [-0.0026, +0.0003] |
| Content correction (removal), change from c* (K*: change from K*, paired 95% interval) | -0.0113 / -0.0795 / -0.0043 | -0.0061 / -0.0686 / +0.0000 | -0.0128 [-0.0160, -0.0094] / -0.0876 [-0.0913, -0.0822] / +0.0108 [-0.0003, +0.0240] | -0.0057 [-0.0066, -0.0043] / -0.0674 [-0.0693, -0.0655] / +0.0003 [+0.0001, +0.0006] |
| Graph-positive loss (removal), change from c* (K*: change from K*, paired 95% interval) | -0.0326 / -0.1638 / -0.0058 | -0.0178 / -0.1178 / -0.0004 | -0.0359 [-0.0398, -0.0319] / -0.1695 [-0.1736, -0.1620] / +0.0049 [-0.0022, +0.0125] | -0.0180 [-0.0189, -0.0171] / -0.1167 [-0.1193, -0.1141] / -0.0007 [-0.0018, +0.0004] |
| Relational term (removal), change from c* (K*: change from K*, paired 95% interval) | +0.0013 / +0.0062 / -0.0665 | +0.0000 / -0.0006 / -0.1143 | +0.0001 [-0.0027, +0.0031] / -0.0016 [-0.0059, +0.0030] / -0.0597 [-0.0753, -0.0469] | -0.0001 [-0.0004, +0.0003] / -0.0004 [-0.0017, +0.0013] / -0.1144 [-0.1190, -0.1104] |

## Table 8 (cost, ms per caption, one RTX 3090, batch 32)

| Row | K0 (r28) | new |
|---|---|---|
| Qwen3 text pass | 3.616 | n/a (no Qwen3 pass) |
| SigLIP 1 text pass (K* caption form: prefixed) | n/a | 1.138 |
| Caption tower, three aspects | 0.017 | 0.017 |
