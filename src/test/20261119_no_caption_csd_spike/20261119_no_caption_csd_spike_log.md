# Spike: R1's reader without the caption grouping (A1c), 2026-10-06 18:17

Exploratory. Decides nothing. One more look at seed 42 (development episodes, no held rows).

## Question
If R1's reader uses (affect, image, csd) (A1c), dropping the CLIP caption grouping and keeping the CSD style grouping,
does the style x genre margin improve, and what happens to the other pairs and the pooled margin?

## What was done (all CPU; code in this folder, round-1 and round-2 modules imported unchanged)
1. `sp_reader.py`: A1c half-readers from round 1's A1 banks. Kept the three blocks without caption ((affect, csd),
   (affect, image), (csd, image); 16,384 episodes each, 49,152 per half). Features recomputed with parts
   (affect, image, csd) on the whole A1 bank and compared with round 1's `rb_reader_A1.npz` `half{j}__X` minus
   caption's 6 columns: bit-identical on both halves. Labels via `rb_features.bank_labels` (condition a = block's first
   grouping, condition b = its second) in A1c order (affect 0, image 1, csd 2). Trained with `rb_build.fit_half_reader`
   (C grid, 5-fold CV over episodes). Result: both halves chose C = 10; out-of-fold bank accuracy 70.36% (half 0) and
   70.49% (half 1), chance 33.3 (A0: 79.7/79.1 on 3 classes with caption; A1: 62.9 on 4 classes).
2. `sp_run.py`: seed-42 probabilities with `rb_eval.seed42_features` and the standard heads, averaged over the two
   half-readers; B'(A1c) rebuilt as `load_bundle` does it (`uniform_probe_scores` over the configuration's groupings,
   then `crossfit_condition_free(ctx.cos, t_n1u, t6u, parity)`); round 2's `run_r2_fusion` functions for T, tau_0..3
   (0/25/50/75th percentile of the reader's own seed-42 top-two margins), gate, cells, integer cross-fits,
   counterpart and assembly; `C.evaluate_fused`. 224 cells = `n_kappa=1` (k_top = 13), 896 cells for completeness.
   Same code for A0, A1, A1c. Told ceiling: P = one-hot on the told grouping (emotion: affect; style: csd on A1 and
   A1c, image on A0; genre: image); an evaluation-label oracle (diagnostic ceiling), through the same fusion.
3. `sp_tables.py` prints `results/sp_tables.txt` from `results/sp_results.json`. Per-config JSON/npz in `results/sp_cand_*`.

## Checks (all passed)
- Regression: R1 on A0, 224 cells: bar margin 0.4435221354166667 and its interval equal round 1's R-c exactly.
- Bank features: recomputed A1c features equal the A1 bank features without caption (both halves, both conditions).
- B' reproduction: my B'(A0), B'(A1) equal `bundle.Bp` scores, per-anchor arrays and `step1_eval_style.npz` exactly.
- 896-cell code path: R1 A0 and A1 bar and margin equal round 2's stored values exactly (A0 +0.472, A1 -0.0102).
- B' means (R@1, %): A0 18.437, A1 18.805, A1c 18.931; B 18.341.

## Results, 224-cell family (margin = fused minus matched counterpart, pp, painting-clustered 95% interval)
In every row below the bar comparator is the counterpart, so the bar margin equals the margin.

| config | pooled | emotion x style | emotion x genre | style x genre |
|---|---|---|---|---|
| A0 (affect, image, caption) | +0.444 [+0.216, +0.674] | +0.708 [+0.326, +1.067] | +1.221 [+0.799, +1.647] | -0.598 [-0.966, -0.225] |
| A1 (+ csd) | -0.010 [-0.169, +0.156] | +0.244 [-0.024, +0.520] | -0.012 [-0.322, +0.293] | -0.262 [-0.557, +0.018] |
| **A1c (affect, image, csd)** | **+0.256 [+0.094, +0.425]** | +0.427 [+0.170, +0.689] | +0.269 [-0.031, +0.565] | **+0.073 [-0.208, +0.350]** |
| told A0 (oracle) | +1.750 [+1.482, +2.021] | +2.643 | +3.003 | -0.397 [-0.640, -0.159] |
| told A1 (oracle) | +1.874 [+1.569, +2.180] | +2.533 | +2.954 | +0.134 [-0.350, +0.595] |
| told A1c (oracle) | +1.874 [+1.569, +2.180] | +2.533 | +2.954 | +0.134 [-0.350, +0.595] |

Told A1 and A1c are identical (the told term never uses caption; the cross-fit picks the same cells; only B' differs and
is not the comparator). Told A0 maps style to image, the same grouping as genre, hence its negative style x genre.

R@1 (%) fused / counterpart / B / B' per pair:
- A0: pooled 18.919 / 18.475 / 18.341 / 18.437; e x s 13.35 / 12.64 / 12.32 / 12.55; e x g 23.02 / 21.80 / 21.57 / 21.69; s x g 20.39 / 20.99 / 21.14 / 21.07
- A1: pooled 18.970 / 18.980 / 18.341 / 18.805; e x s 13.54 / 13.30 / 12.32 / 13.29; e x g 22.03 / 22.05 / 21.57 / 21.71; s x g 21.33 / 21.59 / 21.14 / 21.42
- A1c: pooled 19.230 / 18.974 / 18.341 / 18.931; e x s 13.93 / 13.51 / 12.32 / 13.11; e x g 22.19 / 21.92 / 21.57 / 22.13; s x g 21.57 / 21.50 / 21.14 / 21.55

A1c, other statistics (bar margin vs counterpart; comparator means B' 18.931, counterpart 18.974, B 18.341):
- Bar margin pooled +0.256 [+0.094, +0.425]; does not clear the bar (0.5 point clause fails). Gain statistic +0.834 [+0.617, +1.051].
- Gain margin per pair: e x s +0.708 [+0.361, +1.051]; e x g +1.282 [+0.872, +1.690]; s x g +0.513 [+0.144, +0.895].
- Either-rate margin: pooled -0.321 [-0.558, -0.085]; e x s +0.146 [-0.217, +0.518]; e x g -0.745 [-1.181, -0.312]; s x g -0.366 [-0.793, +0.042].
- Pick accuracy (told mapping) 55.79% [55.14, 56.45] (A0 51.26, A1 48.69; chance 33.3 / 25). Per pair, condition a / b (%): e x s 77.5 / 53.6; e x g 81.2 / 43.4; s x g 33.3 / 45.8.
- Pick shares (%), condition a / b: e x s a: affect 78, image 6, csd 16; b: affect 19, image 27, csd 54. e x g a: affect 81, image 3, csd 16; b: 9, 43, 47. s x g a: affect 60, image 7, csd 33; b: 14, 46, 40. Overall affect 43.6, image 22.0, csd 34.4.
- Chosen cells: k_top 13, tau_0 on both halves (gate always open: 100% in every pair), lambda_u/lambda_a = 0/1 (half 0) and 16/16 (half 1).
  (A0 chose tau_2, gate open 50% overall: e x s 51.0, e x g 52.5, s x g 46.5. A1 chose tau_0, 100%.)
- 896-cell version for A1c: identical to 224 (same cells chosen, all numbers equal). A0 896: pooled +0.472, e x s +0.745, e x g +1.270, s x g -0.598. A1 896: same as 224.

## Paired per-anchor differences, A1c minus other (224 cells; pp, painting-clustered 95% intervals)

| contrast | metric | pooled | e x s | e x g | s x g |
|---|---|---|---|---|---|
| A1c - A0 | fused R@1 | +0.311 [+0.004, +0.615] | +0.586 [+0.116, +1.036] | -0.830 [-1.383, -0.269] | +1.178 [+0.656, +1.691] |
| A1c - A0 | bar margin | -0.187 [-0.445, +0.067] | -0.281 [-0.686, +0.129] | -0.952 [-1.432, -0.475] | +0.671 [+0.242, +1.096] |
| A1c - A1 | fused R@1 | +0.260 [+0.092, +0.429] | +0.391 [+0.136, +0.653] | +0.153 [-0.148, +0.452] | +0.238 [-0.067, +0.548] |
| A1c - A1 | bar margin | +0.267 [+0.082, +0.450] | +0.183 [-0.103, +0.457] | +0.281 [-0.044, +0.599] | +0.336 [+0.012, +0.654] |

(896-cell versions differ only for A0: A1c - A0 bar margin pooled -0.216 [-0.474, +0.043], e x g -1.001, s x g +0.671.)

## Answer
Yes for style x genre: dropping caption lifts it from -0.598 (A0) and -0.262 (A1) to +0.073 (interval spans zero), a gain of
+0.67 [+0.24, +1.10] over A0 and +0.34 [+0.01, +0.65] over A1. But the pooled margin (+0.256 [+0.094, +0.425]) sits between
A1 (-0.010) and A0 (+0.444) and is -0.19 [-0.45, +0.07] below A0, because emotion x genre falls from +1.221 to +0.269 and
emotion x style from +0.708 to +0.427; it does not approach the 0.5 bar. Fused R@1 pooled is highest for A1c (19.23 vs 18.92
and 18.97), mostly because the counterpart itself gains from csd (18.97 vs 18.48), not because the reader adds more.

## Caveats
- Exploratory spike, decides nothing; one more look at seed 42 development episodes; no held rows, no GPU.
- The configuration was chosen after seeing seed-42 per-pair results of A0 and A1, so selection inflation applies to every
  A1c number and to the A1c - A0 and A1c - A1 contrasts; intervals are painting-clustered, not corrected for that selection.
- A1c's reader is trained on the A1 bank's three non-caption blocks, not on a bank built natively for A1c (different
  episode draw than a native bank would give); the A1c bank has 3 blocks, so the reader sees 3 classes in a 3-block mix.
- The pooled A1c - A0 contrast is not significant (interval spans zero); the A0 vs A1c gap in e x g is.
- Told A0's style x genre is degenerate (style and genre both mapped to image); read the told column on A1 and A1c.
- The told ceiling is an evaluation-label oracle (diagnostic only).
- Same code and family as round 1's R-c for A0 (regression exact); A0 224-cell numbers differ slightly from round 2's 896 ones.
