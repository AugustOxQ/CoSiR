# 20261023 aspect-episode spike log

## Question
On aspect episodes (the condition names emotion or art style through 4 cross-item example pairs whose value differs from the anchor's), can SE / C0 / R3 factor models select the right aspect, and how do simple baselines do? Throwaway; nothing committed; selection rows only (asserted; all other feature and code rows NaN). Run: `run_aspect.py`, 35 s, one GPU process only for the CLIP text encoding.

## Construction
4,096 anchors, seed 42, 0 draw failures. Eligible: 8 emotions (not "something else") and 23 styles with at least 30 selection paintings. Each episode has 13 candidates [p_emo, p_style, 11 n_k] shared by both conditions, and 16 example rows (P_emo: 4 pairs sharing an emotion v_i != e, style != s and differing within the pair; P_style: 4 pairs sharing a style t_j != s, emotion differing within the pair, emotion != e, paintings clean of e). Condition emotion: supports P_emo, contrasts P_style, positive p_emo. Condition style: roles swapped, `_target_first(cands, 1)`. All 30 rows (anchor included) come from 30 distinct paintings. The other-aspect candidate is column 1 in both conditions. Chance R@1 = 7.7%.
Caution on the example design: the example values are never the anchor's, so the condition tells the model the aspect, and the target is the candidate that shares the anchor's value of that aspect. This is a candidate-vs-anchor match on an aspect, which CLIP cos already does partly (CLIP R@1 11.1 vs chance 7.7).

## Checks (all passed)
1. All 30 rows per episode are selection rows; every constraint asserted on the final arrays (distinctness, p_emo/p_style/n_k, P_emo, P_style, eligibility, clean-of-e).
2. SE naive beta 0.3 on the old selection label episodes reproduces the stored `naive__SE__0.3__*` ranks exactly (both labels, both directions).
3. CLIP swap success = 0 in both directions; 0 exact ties between p_emo and p_style.
4. 20 random selection captions re-encoded with raw HF CLIP (text_projection(text_model().pooler_output)): cosine to cached txt_features min 1.00000.

## Main table (mean of directions, %; cross-fitted by anchor parity)
| scorer | R@1 emotion | R@1 style | R@1 pooled | other-aspect rate | swap | strict swap |
|---|---|---|---|---|---|---|
| clip | 10.14 | 12.11 | 11.13 | 11.13 | 0.00 | 0.00 |
| SE_agree_cv | 10.34 | 12.30 | 11.32 | 10.99 | 4.96 | 0.31 |
| SE_agree_b0.3 | 10.39 | 11.51 | 10.95 | 10.05 | 16.25 | 0.98 |
| C0_agree_cv | 10.03 | 12.15 | 11.09 | 10.77 | 4.43 | 0.32 |
| C0_agree_b0.3 | 9.92 | 11.54 | 10.73 | 9.67 | 15.99 | 0.78 |
| R3_agree_cv | 10.27 | 11.84 | 11.05 | 10.71 | 6.65 | 0.48 |
| R3_agree_b0.3 | 10.62 | 10.78 | 10.70 | 9.69 | 18.03 | 0.77 |
| SE_value (b0.3) | 11.91 | 11.98 | 11.94 | 9.94 | 17.83 | 1.01 |
| raw_agree_cv | 10.29 | 12.02 | 11.16 | 10.71 | 6.96 | 0.45 |
| raw_agree_relu_cv | 10.14 | 12.11 | 11.13 | 11.13 | 0.00 | 0.00 |
| proto_cv | 10.16 | 12.01 | 11.08 | 10.91 | 3.49 | 0.21 |
| names_cv (privileged) | 13.31 | 11.96 | 12.63 | 9.58 | 20.12 | 1.44 |

## Paired differences (pp, 5,000 bootstrap, seed 42, over anchors)
Pooled R@1 / swap, vs clip: SE_agree_cv +0.20 [-0.12,+0.50] / +4.96 [+4.49,+5.44]; C0 -0.04 [-0.34,+0.26] / +4.43; R3 -0.07 [-0.47,+0.31] / +6.65; raw_agree +0.03 [-0.25,+0.32] / +6.96; raw_agree_relu +0.00 / +0.00; proto -0.04 [-0.26,+0.17] / +3.49; names +1.51 [+0.94,+2.09] / +20.12 [+19.24,+21.01].
vs SE_agree_cv: C0 -0.23 [-0.59,+0.12] / -0.52 [-1.11,+0.05]; R3 -0.27 [-0.64,+0.08] / +1.70 [+1.12,+2.27]; raw_agree -0.16 [-0.52,+0.19] / +2.00; raw_agree_relu -0.20 / -4.96; proto -0.24 [-0.57,+0.09] / -1.46; names +1.31 [+0.73,+1.92] / +15.16 [+14.17,+16.16]. Per-condition and strict diffs are in `results/aspect_results.json`.

## lam picks (tuned on parity 0, tuned on parity 1)
SE 0.25/0.25; C0 0.5/0; R3 0.25/0.5; raw_agree 0.25/0.25; raw_agree_relu 0/0; proto 0.25/0; names 1/2. Tuning is on R@1, so lam is small: swap rises monotonically with lam while R@1 falls.

## Zero-shot names (selection rows)
Emotion prompts on captions: 31.3% (uniform chance 12.5%, majority class 31.8%, so no better than always guessing the majority). Style prompts on images: 27.5% (chance 4.3%, majority 16.1%).

## Surprising / caveats
- Nothing selects the aspect: every non-privileged scorer is within about +-0.3 pp of CLIP on pooled R@1 (CIs include 0); the other-aspect rate stays near 10-11%. Only the privileged `names` gains (+1.5 pp), limited by weak zero-shot accuracy.
- Swap success is misleading here. Under lam=inf, `proto` and `raw_agree` are exactly antisymmetric between the two conditions (swapping roles negates T), so swap is about 50% for any ordering, and it reaches 50% (proto 51.6, raw_agree 51.8) while R@1 is 8.0 and 8.6, at or below chance 7.7. An independent random scorer gets 25%. Swap should be read next to R@1 and the other-aspect rate, never alone. The cv rows have small swap only because R@1-based tuning picks small lam.
- `SE_agree_b0.3` etc. have 16 to 18% swap because the factor T dominates the 0.3 cos at raw code scale, again with R@1 at chance.
- `raw_agree_relu_cv` equals clip exactly: lam=0 was picked on both folds (no lam > 0 improves R@1).
- The SE value rule at b0.3 (11.94 pooled) is slightly above clip on emotion (+1.8), not on style.
