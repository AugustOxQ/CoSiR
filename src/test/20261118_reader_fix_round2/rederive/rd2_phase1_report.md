# Reader fix round 2: independent re-derivation, phase 1

Written 2026-10-06 16:30 (Amsterdam) by the independent re-derivation agent, under `../DECISION_RULE.md` (SHA-256
368bec11363b222d348622a37a6e3aedcfe795d772781d7a255d79b60fab265c, asserted by every script). The implementation files of
this folder (`r2_fusion.py`, `run_r2_fusion.py`, `r2_apply_rule.py`, `r2_readers.py`, `run_r2_readers.py` and their
tests) were never opened or imported.

**No candidate number was computed.** No cell with k_top < 13 (cells 224 to 895) was scored, and no R@1, bar margin or
gain statistic of R2 or R3 exists. The only fusion-family numbers here are R1's on cells 0 to 223, which reproduce
round-1 R-c (rule §4.7).

## 1. What was computed, and with what

| Script (`rederive/`) | What | Runtime |
|---|---|---|
| `rd2_core.py` | shared own code: features, reader probabilities, picks and margins, thresholds, weighted term, z-score via `zscore_rows` (D6), own float32 `combine`, own integer rank counts, top-k positions and restriction, comparators, bar margin and gain statistic via `cluster_bootstrap` (5,000, seed 42, chunk 250); phase guard | |
| `rd2_prep.py` | seed-42 cache: round 1's `common.load_bundle` (standard heads, B, B′, step-1 check), then our own A0/A1 features and grouping scores s_h | 105 s |
| `rd2_family.py` | the fusion family (§4.5, §4.6): gates, G_cf, top-k sets from B, restriction, cells, control σ*, integer min-margin and max-ρ cross-fits, assembly; refuses any k_top that the phase does not allow | |
| `rd2_regress.py` | §4.7 regression check (k_top = 13 only) | 9 s |
| `rd2_r2.py A0`, `A1` | §4.3: μ42, σ42, code check, EM π̂, P′, picks, margins, τ, shift report | < 1 s each |
| `rd2_r3.py A0`, `A1` | §4.4 / §4.8: draws, impure banks, features, purity-4 checks, SHA-256s, SMD, D(k), k*, training at k*, R3 probabilities and τ | 27 s (A0), 98 s (A1) |
| `rd2_selftest.py` | synthetic checks only: cell numbering (116, 119, 58, 123, 0..895 round trip), restriction invariants with ties, integer counts equal to `per_anchor` (with ties and NaN rows), own `combine` equal to `aspect_nested._combine` on all 56 weight cells, EM fixed point | all passed |
| `rd2_phase2.py` | prepared for phase 2, not run (needs `out/PHASE2_AUTHORISED`) | |

Run conditions: CPU only, `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1`, one
heavy process at a time, scikit-learn 1.6.1, numpy 2.2.6. Every D14 input that a script reads is SHA-256 asserted
through `r2_common.assert_inputs`. Outputs: `rederive/out/` (JSON at full precision, npz arrays, logs; gitignored).

Imports from earlier code, as the brief allows: round 1's `common.load_bundle` (bundle, B, B′(A0), B′(A1)),
`rb_eval.seed42_features` and `common.grouping_stack` (used only to check that our own features and s_h equal them),
`rb_features.both_conditions` (only to check that our impure-bank features equal it), the stored round-1 half-reader
pickles (loaded as data), `cluster_bootstrap`, `zscore_rows`, `per_anchor` (used only to cross-check our integer counts
on the assembled scores of the chosen cells).

## 2. Seed-42 inputs (cache)

- Bundle checks passed (step 1 reproduced exactly for A0, A1, AR, B, B′). B is condition-free (float32).
- B 18.341064453125, B′(A0) 18.436686197916664, B′(A1) 18.804931640625: equal to the rule's values at full precision.
- Our 18 (A0) and 24 (A1) features equal `rb_eval.seed42_features` bit for bit (max difference 0.0); Δ^b = −Δ^a exactly.
- Our s_h (float32 einsum, query's own modality) equal `common.grouping_stack` bit for bit, both directions.

## 3. Regression check (§4.7): PASSED

R1 = round 1's A0 half-readers (C 1.0 on half 0, 100.0 on half 1), P^c = mean of the two `predict_proba(scaler.transform(x))`.

| Comparison | Result |
|---|---|
| T^c, all four `T__{a,b}__{i2t,t2i}` | equal exactly (max difference 0.0) |
| top-two margins `margin__{a,b}` | equal exactly |
| picks `pick__{a,b}` (int8 stored vs int64) | equal in value |
| τ_0..τ_3 | 3.8684538364530674e-05, 0.21702129490553143, 0.47973989883399526, 0.7502585816077211: equal to `rc_tau.json`, to the rule's text and to the stored `extra__taus` exactly |
| gates (4 × 2 × 12,288) | equal to the stored `extra__gate_{a,b}` |
| control σ* | 0 on both halves; ρ_ctrl 4,532 (half 0) and 4,483 (half 1) of 6,144 × 4 rankings; R@1 on the tune half equal to round 1's JSON exactly |
| fused cells (integer min-margin) | 116 (τ_2, (0, 2)) on tune half 0, 119 (τ_2, (0, 16)) on tune half 1; criterion 175 and 225; half 0 had 4 cells tied at the maximum, the lowest wins |
| fused cells by round 1's float mean criterion | also 116 and 119 |
| counterpart cells (integer max-ρ) | 58 (τ_1, (0, 0.5)) and 123 (τ_2, (0.5, 1)); ρ 4,647 (2 cells tied) and 4,502 (4 cells tied) |
| cell settings vs round 1's JSON (τ index, τ, λ_u, λ_a) | equal |
| per-anchor `fused__{r1,gain,other,swap,strict}`, `cf__{...}`, `bar_v` | equal exactly (also equal to `per_anchor` on the literally assembled scores) |
| fused and counterpart R@1 | 18.918863932291664 and 18.475341796875, equal to the rule |
| bar comparator | counterpart |
| bar margin | 0.4435221354166667 [0.21646171563312194, 0.6735669710776852], equal to the rule exactly |
| gain statistic | 2.667236328125 [2.325087836946873, 3.012361650695922], equal to the rule exactly |

One non-rule diagnostic differs: round 1's JSON `gate_open_share` per condition (for example τ_1 condition a
83.77278447151184 against our 83.77278645833334). Round 1 averaged float32 gates; we average booleans in float64. The
overall shares (75, 50, 25) agree, the gates themselves are identical, and the share enters no rule.

## 4. R2 (§4.3)

**Code check** (each half's own `scaler.mean_`, `scaler.scale_`, no EM) reproduces R1's P bit for bit (max difference 0.0,
picks identical) on A0 and on A1. No σ42 entry is 0.

**μ42 and σ42** (numpy `X.mean(axis=0)`, `X.std(axis=0)`, X = vstack(F_a, F_b), 24,576 rows). The A0 values are the
first 18 rows. μ of the three Δ features is exactly 0.0 (Δ^b = −Δ^a), so an agreement check on μ42 should be absolute
there, not relative. μ42 agrees with an exact `math.fsum` mean to 6.5e-15 relative.

| feature | μ42 | σ42 |
|---|---|---|
| affect__S | 0.03750044020747131 | 0.007849502536376857 |
| affect__C | 0.03750044020747131 | 0.007849502536376863 |
| affect__Delta | 0.0 | 0.011123818700800807 |
| affect__sd_support | 0.012186092079906357 | 0.00984874049618295 |
| affect__sd_contrast | 0.012186092079906256 | 0.009848740496182914 |
| affect__argmax_match | 0.09319051106770833 | 0.14525199401530955 |
| image__S | 0.03357454000131099 | 0.031862780677996125 |
| image__C | 0.03357454000131099 | 0.03186278067799615 |
| image__Delta | 0.0 | 0.048657327089135255 |
| image__sd_support | 0.03537311514088052 | 0.04698367349249152 |
| image__sd_contrast | 0.03537311514088042 | 0.04698367349249159 |
| image__argmax_match | 0.062255859375 | 0.12397833376970978 |
| caption__S | 0.029346030526260353 | 0.020214559591368704 |
| caption__C | 0.029346030526260353 | 0.020214559591368718 |
| caption__Delta | 0.0 | 0.030127717583177564 |
| caption__sd_support | 0.026659937123360792 | 0.027450448085455683 |
| caption__sd_contrast | 0.02665993712336069 | 0.027450448085455614 |
| caption__argmax_match | 0.04376220703125 | 0.10264521961959072 |
| csd__S (A1) | 0.1524330437274557 | 0.08530088332858536 |
| csd__C (A1) | 0.1524330437274557 | 0.08530088332858543 |
| csd__Delta (A1) | 0.0 | 0.13139085109192547 |
| csd__sd_support (A1) | 0.12861106949342735 | 0.09249128735846526 |
| csd__sd_contrast (A1) | 0.12861106949342618 | 0.0924912873584648 |
| csd__argmax_match (A1) | 0.21924845377604166 | 0.21288854824495887 |

Against round 1's bank scalers, seed 42 sits 0.2 to 0.49 bank standard deviations lower on S and C, and its spread is
0.48 to 1.13 times the bank's (full table per half in `out/rd2_r2_A0.json`, `shift_report.per_half`).

**EM** (π_train = 1/H, start 1/H, tolerance 1e-10 on the max absolute change, cap 10,000):

| | π̂ (configuration order) | stop | cap reached |
|---|---|---|---|
| A0 | [0.33954936858584167, 0.31214248949301643, 0.348308141921142] | first s with change < 1e-10 is s = 38, so π̂ = π^(39) after 39 updates (last change 8.46e-11) | no |
| A1 | [0.2819924347120968, 0.24152272833894686, 0.2683388597125684, 0.20814597723638714] | first s with change < 1e-10 is s = 118, so π̂ = π^(119) after 119 updates (last change 8.82e-11) | no |

(The count convention matters when comparing: "s at stop" is 38 / 118, "updates computed" is 39 / 119.)

**R2's thresholds** (margins of P′ over the 24,576 values):

| | τ_0 | τ_1 | τ_2 | τ_3 |
|---|---|---|---|---|
| R2 / A0 | 3.771746264230602e-05 | 0.2907081713292494 | 0.5994124414690893 | 0.861841000813007 |
| R2 / A1 | 4.306062818093537e-06 | 0.14316071236581546 | 0.3553452483633919 | 0.6440802110318591 |

**Shift report (diagnostic).** Picks differing from R1: A0 21.598307291666664% (condition a 20.30%, b 22.90%; before EM
21.18%); A1 27.079264322916668% (a 19.98%, b 34.18%). Pick shares A0, R2 vs R1: affect 37.1 vs 55.2, image 28.9 vs 25.2,
caption 33.9 vs 19.6. Top probability mean on seed 42: R2 0.7569 (before EM 0.7559) against R1's 0.6943 (round 1:
0.694) and the bank's out-of-fold 0.781; A1: R2 0.6271, R1 0.5643.

## 5. R3 (§4.4)

**Draws.** Seeds 21700 / 21800, one generator per half in the rule's call order. All validity checks hold (rows in
half j, no anchor painting, caption painting differs from image painting, `order` rows are permutations). One redraw
round sufficed everywhere (A0 half 0: 17 image and 50 caption entries redrawn; half 1: 23 and 50). The share of drawn
slots whose painting also occurs in the original episode is 0.308% (half 0) and 0.340% (half 1), matching the rule's
dry run (0.31%, 0.34%), which suggests our reading of the draw procedure is the rule writer's.

**Banks.** Impure banks of purity 3, 2, 1 replace 1, 2, 3 positions per side; they are nested (every pair replaced at
k + 1 carries the same replacement at k). Our features equal `rb_features.both_conditions` exactly at every purity and
half. Purity-4 checks: features equal round 1's `half{j}__X` exactly (both halves), labels equal `half{j}__y`; the 18
purity-4 SMDs equal `rb_diag_A0.json` `c_shift_report.smd` exactly.

**SHA-256** (int64 C order for `order`, `img`, `cap`, each (N, 2, 4); "X" = vstack(F_a, F_b) float64 C order, (2N, 18);
F_a, F_b and the four pair arrays per purity are also hashed in the JSON):

| half | array | SHA-256 |
|---|---|---|
| 0 | order | 8ef89646cb27fe53a07ba210eed5afcc581000fce74d8e980fe81a171597a9db |
| 0 | img | 60418d2fe5350e53d117de1137833673af516066d4ff817a3e79858cea5f4a6c |
| 0 | cap | afcb783e03d791a0a7b892a2a0d1253456f1ea01dc5acbc485f21172d081f695 |
| 0 | X purity 1 | 1da07a4ec2a4f130005042caca49561f5f91fba8cb19b5303de89614c328bd5a |
| 0 | X purity 2 | 48692d40d8bd37b59239c29f4626e4085c77346e7f4873ebc4f24cc36ce07f49 |
| 0 | X purity 3 | c296740a785e9d259946eedcdf0c7c1fa6f6b94a8a7f0bd2f74746242d48460a |
| 0 | X purity 4 | fc0fa4c9e022f7688f159c1da326afc5fad879ec2b14c953b5190e26e794d7a8 |
| 1 | order | 43a0f2f8d92fcef6b08d5741714f3fa700c590d4d86541b61a7c72c3014eeb6e |
| 1 | img | 7947a3fafedc170666eee83a0ba86cfedf13ff3fc2d2f0c3c181d1efb751be05 |
| 1 | cap | f9501f8372c17aee3825e856fe68ece4828eed6d76ca6e6c382f1c5dd33900c8 |
| 1 | X purity 1 | fae2f096f7e4d642546671bc1e140c3784283b819e9495e771efdd0077957c2c |
| 1 | X purity 2 | 9b3fa584df66109b3c7eda07f9883642dcd8eb9f018e8d535d20a08b998975bb |
| 1 | X purity 3 | c2752dab47159f49eeb4eeed262c708bf463ef274898bf52363746ee382bfac0 |
| 1 | X purity 4 | dc2c411dfa649b04cd9763c1077d49aeec8b52d89fe293d038157c2427702688 |

**D(k) table** (SMD = (mean seed 42 − mean bank) / sqrt((var seed 42 + var bank) / 2), ddof 1; 24,576 seed-42 rows
against 196,608 bank rows; D = mean |SMD| over 18):

| feature | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| affect__S | +0.199817 | +0.065255 | -0.064941 | -0.191804 |
| affect__C | +0.199817 | +0.065255 | -0.064941 | -0.191804 |
| affect__Delta | 0 | 0 | 0 | 0 |
| affect__sd_support | +0.071712 | +0.036251 | +0.011153 | -0.003393 |
| affect__sd_contrast | +0.071712 | +0.036251 | +0.011153 | -0.003393 |
| affect__argmax_match | +0.087013 | +0.023153 | -0.038430 | -0.095264 |
| image__S | +0.278504 | -0.020656 | -0.252293 | -0.436272 |
| image__C | +0.278504 | -0.020656 | -0.252293 | -0.436272 |
| image__Delta | 0 | 0 | 0 | 0 |
| image__sd_support | +0.097170 | -0.103757 | -0.228887 | -0.309320 |
| image__sd_contrast | +0.097170 | -0.103757 | -0.228887 | -0.309320 |
| image__argmax_match | +0.187867 | -0.011635 | -0.176685 | -0.314422 |
| caption__S | +0.120911 | -0.198042 | -0.433111 | -0.615544 |
| caption__C | +0.120911 | -0.198042 | -0.433111 | -0.615544 |
| caption__Delta | 0 | 0 | 0 | 0 |
| caption__sd_support | -0.072499 | -0.275840 | -0.400701 | -0.482259 |
| caption__sd_contrast | -0.072499 | -0.275840 | -0.400701 | -0.482259 |
| caption__argmax_match | +0.014091 | -0.179250 | -0.333944 | -0.460855 |
| **D(k)** | 0.10945540348228223 | **0.08964658887594207** | 0.18506833918606608 | 0.27487353947498383 |

**k\* = 2** (A0). Full-precision SMDs in `out/rd2_r3_A0.json` (`D_table`). Mean |Δ_h| (diagnostic): seed 42 affect
0.00812, image 0.03317, caption 0.02119; bank half 0 at k = 4 / 3 / 2 / 1: image 0.0541 / 0.0433 / 0.0327 / 0.0225,
caption 0.0457 / 0.0371 / 0.0287 / 0.0207, affect 0.0080 / 0.0077 / 0.0075 / 0.0073 (half 1 within 0.001).

**Training at k\* = 2** (StandardScaler on the half's whole impure bank, multinomial lbfgs `max_iter=2000`, 5-fold CV
over episodes, mean held-out log loss, ties to the smaller C; no convergence warnings anywhere):

| half | C 0.01 | C 0.1 | C 1 | C 10 | C 100 | chosen C | gap to 2nd | OOF acc. |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.8505882241892271 | 0.849358876042633 | 0.8493484703189432 | 0.8493467041805429 | 0.8493429269438575 | **100.0** | 3.78e-06 | 62.02% |
| 1 | 0.8590656198139847 | 0.8577236430388551 | 0.8576982503865652 | 0.8576923270490717 | 0.8576954618008727 | **10.0** | 3.13e-06 | 61.54% |

Refit iterations 50 and 51. R3's τ on seed 42: **3.399954748600775e-05, 0.12456908995572917, 0.2854599358396749,
0.4974364340057035**. Top probability mean: bank out-of-fold 0.6003, seed 42 0.5997. Picks differ from R1 on 17.24% of
(episode, condition) values; pick shares affect 41.3, image 35.2, caption 23.5.

## 6. A1 versions (§4.8; descriptive, lower priority)

- **R2/A1**: code check passed bit for bit; π̂, iterations and τ in §4. μ42 and σ42 = the 24 rows of the table in §4.
- **R3/A1**: draws valid (half 0: 42 image, 95 caption entries redrawn; half 1: 52, 84); purity-4 features equal
  `rb_reader_A1.npz` `half{j}__X` exactly, the 24 purity-4 SMDs equal `rb_diag_A1.json` exactly; own features equal
  `rb_features.both_conditions` at every purity. D(1) 0.1763440292007937, **D(2) 0.10549306802418701**, D(3)
  0.13217396029550066, D(4) 0.19036018247246586, so **k\*_A1 = 2**. Chosen C: **1.0 on both halves**; half 0's margin
  over C = 100 is only 3.7e-07 in mean log loss (half 1: 2.6e-06 over C = 10). OOF accuracy 46.42% and 46.19%
  (chance 25%). τ: 1.1906487280111122e-05, 0.05877819082834039, 0.14845353048236493, 0.3081392192786174.
- SHA-256 for A1 (order, img, cap, X per purity) are in `out/rd2_r3_A1.json`:
  half 0 order 32d1240f…, img 03020f40…, cap 253985fe…; half 1 order 4ccdf80f…, img 4ccaea32…, cap 512ef9fc….

## 7. Things to watch in phase 2

1. **Near ties in the choice of C.** R3/A0: half 0 C 100 beats C 10 by 3.8e-6 in mean log loss, half 1 C 10 beats C 100
   by 3.1e-6; R3/A1 half 0 C 1 beats C 100 by 3.7e-7. A different thread count or BLAS could flip a choice; both of our
   runs used 8 threads. Round 1's refits were bit-identical between implementation and re-derivation.
2. **Cell ties.** On round 1's cells the fused pick on tune half 0 is the lowest of 4 cells tied at criterion 175, and
   the counterpart picks are the lowest of 2 and 4 tied cells. The integer criteria make this exact, but any
   implementation that compares floating means could choose differently on the 896 cells.
3. **Zero μ.** μ42 of the Δ features is exactly 0.0; compare those entries absolutely.
4. **EM count convention.** Ours: stop at s = 38 (A0) / 118 (A1), that is 39 / 119 updates.
5. **Feature-matrix hash convention.** We hash X = vstack(F_a, F_b) (float64, C order); F_a and F_b separately are also
   in the JSON in case the implementation hashed another layout.

## 8. Phase 2 readiness

`rd2_phase2.py` computes, for R1, R2, R3 on A0 from our own phase-1 probabilities: T, margins, τ (asserted equal to phase
1), all 896 cells with the top-k restriction, σ*, the integer cross-fits, chosen cells, per-anchor arrays, comparator,
bar margin, gain statistic, D12 clauses, per-pair numbers, R1 − round-1 R-c, R2 − R1, R3 − R1, pick accuracy, gate
shares and the §5 item 4 carry; `--a1 <reader>` gives the A1 ablation. It refuses to run until `out/PHASE2_AUTHORISED`
exists, which will be created only after the controller's message that the main run's regression check has passed. A
comparison script against the main run's result files will be written once their paths are known.

Storage: `rederive/out/` holds 506 MB on the system disk (the two `rd2_r3_*_bankfeat.npz` caches are 415 MB of it); they
are gitignored and can be deleted after phase 2.
