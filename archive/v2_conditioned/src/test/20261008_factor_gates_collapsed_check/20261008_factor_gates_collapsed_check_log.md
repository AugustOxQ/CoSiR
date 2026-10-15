# 2026-10-08 factor-gates collapsed-code check log

**Problem.** The factor-discovery gates from the earlier plan (top-2 mass, dead, modality-private, community spanning) measure factor *usage*, so a 32-factor dictionary made of near-copies of one axis can pass all of them. The 2026-09-29 code review found exactly that collapse in the seed-42 codes. Task 1 adds `src/eval/factor_gates.py`, geometry gates that must FAIL on these known-collapsed codes. This run checks that they do.

**What was run.** `run_check.py` (from the repository root, CoSiR env, GPU, seed 42 throughout):

- Loaded the cached codes `src/test/20261007_naive_rule_mechanism_analysis/cache/factor42_{img,txt}.npy` (all 308,723 rows; the factor model was fit on the row-split train rows, and `factor42_meta.json` confirms `factor_fit_items = 246,978`, matching `split_items`).
- Loaded the real ArtELingo CLIP features with `load_real_features`, and the row split with `split_items(EXPECTED_SAMPLES)`: fit rows = 246,978 train rows, eval rows = 61,745 held rows.
- Community labels: rebuilt the train-row content graph (`GraphConfig()`, 2,463,679 edges), ran `train_stage1(..., Stage1Config())`, then `detect_communities` (21 communities), the same steps as `prepare_item_disjoint_codes` minus factor training.
- Ran `evaluate_factor_gates` with default `FactorGateThresholds()`. Full values are in the (gitignored) `gates_collapsed.json`; console output in the (gitignored) `run_check.log`.
- Runtime: 80.9 s end to end.

**Results (default thresholds, eval = held rows).**

| Gate | Value | Threshold | Result |
|---|---|---|---|
| participation_ratio | img 1.3242, txt 1.3262 (min) | >= 8.0 | FAIL |
| redundancy | max abs pair corr 0.99977; 374 of 496 pairs >= 0.9; no constant factors | <= 0.90 and no constant | FAIL |
| readout | img 0.4936 vs PCA-10 0.4943 (beats it); txt 0.4669 vs PCA-10 0.4452 (loses) | both modalities <= PCA-10 | FAIL |
| sparsity | active fraction img 0.7133, txt 0.7176 | <= 0.375 | FAIL |
| dead | 0 dead (indices []) | <= 0 | PASS |
| modality_private | 0 private (indices []) | <= 1 | PASS |
| usage_concentration | top-2 mass share 0.0782 | <= 0.20 | PASS |
| community_spanning | spanning fraction 1.0 (32/32 spanning, 0 topic-like; 21 train communities) | >= 0.75 | PASS |
| pair_retrieval | code R@10 0.1276 / CLIP R@10 0.3440 = ratio 0.3709 | >= 0.5 | FAIL |

`all_passed = False`.

**Reference numbers for Tasks 3 and 5.** Paired retrieval, pool 1000, k 10, 61 pools over the 61,745 held rows, mean of image-to-text and text-to-image: CLIP feature recall@10 = **0.3440**; cached collapsed-code recall@10 = **0.1276**; retrieval ratio = **0.3709**. Readout baselines (held rows, relative L2): CLIP PCA-10 = 0.4943 img / 0.4452 txt; affine readout from the collapsed codes = 0.4936 img / 0.4669 txt.

**Conclusion.** All four expected gates (`participation_ratio`, `redundancy`, `readout`, `sparsity`) FAIL on the collapsed codes, and `pair_retrieval` fails too. The numbers agree with the earlier probe (participation ratio 1.32, 374/496 pairs with |r| >= 0.9, about 0.71 active fraction). The four legacy usage gates (`dead`, `modality_private`, `usage_concentration`, `community_spanning`) PASS, which reproduces the reason the collapse went unnoticed. No thresholds or gate code were changed after seeing these values.

Note: the `readout` gate fails on the text side only. The image-side affine readout from the collapsed codes is marginally better than PCA-10 (0.4936 vs 0.4943), so the gate's failure rests on the text modality; the gate requires both modalities, as designed.
