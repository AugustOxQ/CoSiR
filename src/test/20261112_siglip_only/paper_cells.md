# r29 numbers sheet: the SigLIP-only adapter beside the current SigLIP 1 model (K0)

Written 2026-11-13 06:28 by `r29_ladder.py sheet` from the cell records present in `records/`; a missing piece shows `not run`. K0 values are r28's (`per_seed.json`, `mmart.json`, `photos.json`, r20 `cells.json`; for SC4 r17 SC1 and r14 SC6): five-seed means on test, three decimals. New cells: value (Q1 cell's value; for image mAP the aspect head, the P1 cell is the untold candidate and shows its own value), the paired difference d from the comparator with its 95% interval, r (the LEVEL unit) and the outcome by spec §3.4: WIN, LEVEL, BELOW, or INCONCLUSIVE. Q1 claim: WIN "above", LEVEL "level with", BELOW none. A failed pass rule (D9) does not hide the cells of SC3: they stay on this sheet. The sheet decides nothing: the levels are the ladder records'.

"Level" lines come from `records/<claim>_<arm>.json`; "K* - K0" is diagnostic (d), measured after the levels.

Pass rule (SC2): status `fail`, level 0. SC4 to SC7: not run (D3).

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

## SC4 (Table 6, ArtELingo test, 6 cells per encoder; K0 on that encoder shown beside, descriptive)

Not run (D3): the pass rule did not pass.

No encoder arm yet: not run.

| Cell | K0 (r28) | new value | vs P1 | vs Q1 | Q1 claim |
|---|---|---|---|---|---|
| emotion ami | 0.098 (SigLIP 2 K0) | not run | not run | not run | not run |
| emotion map | 0.269 (SigLIP 2 K0) | not run | not run | not run | not run |
| style ami | 0.439 (SigLIP 2 K0) | not run | not run | not run | not run |
| style map | 0.348 (SigLIP 2 K0) | not run | not run | not run | not run |
| genre ami | 0.527 (SigLIP 2 K0) | not run | not run | not run | not run |
| genre map | 0.623 (SigLIP 2 K0) | not run | not run | not run | not run |

Ladder (SC4): not run.

## SC5 (Table 4, MMArt test, 6 cells)

Not run (D3): the pass rule did not pass.

New cells: not run (no cell record yet).

| Cell | K0 (r28) | new value | vs P1 | vs Q1 | Q1 claim |
|---|---|---|---|---|---|
| emotion ami | 0.053 | not run | not run | not run | not run |
| emotion map | 0.202 | not run | not run | not run | not run |
| style ami | 0.455 | not run | not run | not run | not run |
| style map | 0.383 | not run | not run | not run | not run |
| genre ami | 0.539 | not run | not run | not run | not run |
| genre map | 0.628 | not run | not run | not run | not run |

Ladder (SC5): not run (no ladder record).

## SC6 (Table 7, photo model F1, 4 cells: Affection and EmoSet test)

Not run (D3): the pass rule did not pass.

New cells: not run (no cell record yet).

| Cell | K0 (r28) | new value | vs P1 | vs Q1 | Q1 claim |
|---|---|---|---|---|---|
| affection ami | 0.182 | not run | not run | not run | not run |
| affection map | 0.323 | not run | not run | not run | not run |
| emoset ami | 0.432 | not run | not run | not run | not run |
| emoset map | 0.483 | not run | not run | not run | not run |

Ladder (SC6): not run (no ladder record).

## Table 5 (ablation, ArtELingo test): K0 rows; removals of K*: not run (D3)

| Row | K0 AMI emotion / style / genre | K0 mAP emotion / style / genre | K* AMI emotion / style / genre | K* mAP emotion / style / genre |
|---|---|---|---|---|
| CoSiR (ours) = c* | 0.099 / 0.441 / 0.528 | 0.263 / 0.340 / 0.599 | not run (D3) | not run (D3) |
| Instructed graphs (removal), change from c* (K*: change from K*, paired 95% interval) | -0.0309 / -0.2101 / -0.0074 | -0.0197 / -0.1435 / -0.0011 | not run (D3) | not run (D3) |
| Content correction (removal), change from c* (K*: change from K*, paired 95% interval) | -0.0113 / -0.0795 / -0.0043 | -0.0061 / -0.0686 / +0.0000 | not run (D3) | not run (D3) |
| Graph-positive loss (removal), change from c* (K*: change from K*, paired 95% interval) | -0.0326 / -0.1638 / -0.0058 | -0.0178 / -0.1178 / -0.0004 | not run (D3) | not run (D3) |
| Relational term (removal), change from c* (K*: change from K*, paired 95% interval) | +0.0013 / +0.0062 / -0.0665 | +0.0000 / -0.0006 / -0.1143 | not run (D3) | not run (D3) |

## Table 8 (cost, ms per caption, one RTX 3090, batch 32)

| Row | K0 (r28) | new |
|---|---|---|
| Qwen3 text pass | 3.616 | n/a (no Qwen3 pass) |
| SigLIP 1 text pass (K* caption form: prefixed) | n/a | 1.138 |
| Caption tower, three aspects | 0.017 | 0.017 |
