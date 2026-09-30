# 20261018 affect factor learning: run script, prepare and timing smoke

## Problem
Cells E and SE train factor encoders (R3 config, pair agreement, condition episodes) with conditions drawn from
GoEmotions affect clusters of the captions. Task 2 builds `run_affect.py`, extracts the affect signal, clusters it,
measures diagnostics, and times the runs so the local-vs-DAS6 choice can be made.

## Steps
1. `run_affect.py`: imports `run_grid.py` as `grid`; `prepare()`, `affect_cell_config`, `affect_source`,
   `run_affect_cell` (overwrite guard on full runs), `smoke()`, `main()` (`--evaluate/--replicate/--tables` raise
   `NotImplementedError("Task 3")`, Controller Ruling 2).
2. `--prepare` (125 s total), then `--smoke`.

## Verified
- Row scope: only `cache["scorer_train"]` captions reach the model; asserted `len(captions) == len(st)` and that
  no selection/val/held row overlaps scorer-train. Probe split asserted inside scorer-train and painting-disjoint.
- Affect array (183,694, 28), finite, in [0, 1] (asserted). Extraction 84.7 s on the RTX 3090.
- C0 and S reference checkpoints: stored configs equal `grid.cell_config("C0"/"S", 42)` (asserted); SHA-256 in
  `cache/affect_prepare.json` (C0 7653caf0985b..., S 33d35943ec62...).
- Image and caption AMIs reproduce the spec's 0.035 / 0.318 and 0.056 / 0.058.

## Prepare diagnostics (measured, never used for a choice)
k-means k=64: 64 non-empty groups, all 64 have >= 200 rows; sizes min / median / max = 244 / 1690 / 21098.

| Partition | AMI with emotion | AMI with art style |
|---|---|---|
| affect k=64 | 0.196 | 0.016 |
| CLIP image | 0.035 | 0.318 |
| CLIP caption | 0.056 | 0.058 |

Probe (logistic, C=1, standardized; fit on 146,960 rows of 80% of scorer-train paintings, scored on 36,734 rows):

| Input -> emotion | Accuracy |
|---|---|
| affect-28 | 0.517 |
| CLIP caption features | 0.577 |
| majority class | 0.285 |

## Step 5 smoke table (local RTX 3090)

| Cell | s/step | fixed s | projected full run (min) | peak GiB |
|---|---|---|---|---|
| E | 0.2687 | 0.74 | 8.97 | 3.87 |
| SE | 0.2781 | 0.27 | 9.28 | 3.86 |
| C0 (2x2 smoke) | 0.2477 | 1.37 | 8.28 | 3.86 |

Projected: selection (E + SE) 18.2 min; replication 2 x (SE + C0) 35.1 min; total 53.4 min sequential; peak 3.87 GiB.
Thresholds: one run > 45 min, total > 3 h, peak > 20 GiB: none exceeded. Decision: `run_locally`.

# Task 3: runs E and SE, selection evaluation, selection report

## Steps
1. `--run E --seed 42` and `--run SE --seed 42` as two parallel processes on the local RTX 3090 (launched 00:15,
   done 00:25): E 634.8 s, SE 633.4 s (10.6 min each; smoke projected 9.0 / 9.3 min alone), peak 3.94 / 3.93 GiB.
   Codes finite (asserted in `run_affect_cell`); every logged loss, condition loss and tau finite. No rerun needed.
2. Rule: `BINDING_GATES`, `apply_affect_rule`, `_check_affect_rule` verbatim from the brief; the six hand-checked
   cases pass (run first thing in `evaluate()`).
3. `evaluate()`: 4,096 selection label episodes per label; prefix-2,048 SHA-256 = stage (d)'s (asserted), every
   episode row a selection row (asserted), null targets `default_rng(42).integers(1, 13, 4096)` in (emotion,
   art_style) order. Models R3 (stage (d) cache), C0 / S (the 2x2's checkpoints; SHA-256 equal to prepare's record and
   configs asserted), E / SE (this folder; config = `grid.cell_config("S", 42)` asserted). Codes on scorer-train +
   selection rows only, NaN elsewhere (asserted); evaluation codes and CLIP features NaN outside selection rows
   (asserted). Per model: 9 gates (8 binding + sparsity reported), naive ranks on the beta grid, oracle at 0.3 / 0 and
   its null at 0, tie breakdown at 0 / 0.3. Reproduction check: R3 / C0 / S naive ranks at every beta and CLIP-only
   on the first 2,048 episodes equal the 2x2's stored ranks (identical share 1.000, asserted). Extras (reported,
   never gating): guard power, balance-matched beta (E, SE vs C0, re-scored and interpolated), per-target
   E / SE - C0, code probes + within-painting caption-residual probe (run_posthoc helpers, this folder's model list),
   argmax-factor AMI incl. the affect clusters, condition-loss / tau summaries.
4. A dry run of `evaluate()` (E and SE pointed at S's checkpoint, output to the scratchpad, discarded) ran during the
   last ~6 minutes of training to catch code errors; it passed. Its R3 / C0 / S rows equal the real run's except the
   512-d CLIP residual probe (47.04 vs 46.98) under a different BLAS thread count (OMP_NUM_THREADS=6).
5. `--evaluate` (248 s) -> results/selection_results.json, results/selection_ranks.npz, run_evaluate.log.

## Result (seed 42, selection rows, naive beta 0.3, paired R@1 points vs C0)
| Cell | binding gates | sparsity (img / txt) | D_emo | D_style | pooled |
|---|---|---|---|---|---|
| C0 | 8/8 | 0.419 / 0.479 pass | | | |
| E | 8/8 | 0.448 / 0.472 pass | +3.99 [+3.14, +4.83] | -1.23 [-2.08, -0.40] | +1.38 [+0.78, +1.99] |
| SE | 8/8 | 0.422 / 0.529 FAIL (reported only) | +1.31 [+0.54, +2.06] | +1.12 [+0.29, +1.93] | +1.21 [+0.65, +1.78] |
| S (ref.) | 8/8 | 0.424 / 0.559 FAIL | -0.85 [-1.64, -0.06] | +3.60 [+2.69, +4.48] | +1.37 [+0.78, +1.98] |

Rule (recomputed by hand): C0 binding OK -> no stop; E fails the style guard (lower bound -2.08 <= -1.5); SE
qualifies (+0.54 > 0, +0.29 > -1.5); tie band [SE] -> **picked SE**. Guard power from the measured style SE 0.427:
P(pass | true 0) 94.0%, P(pass | true -0.5) 65.0% (spec's pre-run 0.42: 94.6% / 66.3%).

Context: E's oracle beats its own naive rule by +5.16 pooled at beta 0 (C0 +0.51); E's style loss is naive-only
(oracle +1.81 over C0 at beta 0, image->style probe -0.44 n.s.); caption->emotion probe E +4.24, SE +2.28 over C0.
E's condition loss stays near uniform (last-10 mean 1.24 vs log 4 = 1.386); SE 0.90; S 0.44.

## Report
docs/reports/auto/v2/2026-10-18_candidate_a_affect_factor_learning_selection.md; figures
docs/reports/assets/2026-10-18_affect_factor_learning/ (build_2026-10-18_affect_factor_learning_figures.py).
Next (not this task): Task 4 replication of SE and C0 (seeds 43, 44), then Task 5 held test.

## Task 4 replication (seeds 43 and 44 of SE and C0)
1. Trained in this folder as separate processes (`run_affect.py --run CELL --seed S`): SE 43, SE 44 and C0 43 in
   parallel (about 620 s each, three on one RTX 3090), then C0 44 alone (535 s). All finite, peak 3.93 to 3.94 GiB;
   SE condition loss at the end 1.037 (seed 43) and 0.883 (seed 44), tau 0.0452 and 0.0492. No start-up failure. The
   first launch attempt died at once because of a shell quoting mistake (no process reached training, nothing written),
   and a cleanup pkill also killed the launching shell; both were rerun from a script, nothing else affected.
2. `--replicate` (69 s) -> results/replication.json, run_replicate.log. It reuses `selection_episodes`, `model_codes`
   (new optional `seed` argument; config asserted equal to `grid.cell_config("S" / "C0", seed)`; the reference-SHA
   check applies to seed 42 only), `grid.gate_report` with the same fit/eval rows and readout reference,
   `label_episode_weights`, `probe.fixed_weight_ranks` at beta 0.3 and `probe.r1_diff`. `model_path` and
   `training_record` gained the same optional `seed`. `evaluate()` itself is unchanged: `--tables` output for
   selection_results.json is byte-identical before and after (313 lines compared), and the JSON is untouched.
   Seed 42 is recomputed through the new path and equals the stored D_emo / D_style points exactly (asserted).
3. Result (SE minus same-seed C0, naive beta 0.3, R@1 points): seed 43 D_emo +0.50 [-0.26, +1.29], D_style +2.14
   [+1.25, +3.00], pooled +1.32 [+0.73, +1.89]; seed 44 D_emo +1.28 [+0.49, +2.06], D_style +1.04 [+0.21, +1.90],
   pooled +1.16 [+0.56, +1.75]. Both models pass 8/8 binding gates at both seeds; SE fails the report-only sparsity
   cap (caption 0.515 and 0.518), C0 passes it. Reported only; the verdict stays on seed 42 (rule untouched).
4. Report: a Replication section in docs/reports/auto/v2/2026-10-18_candidate_a_affect_factor_learning_selection.md.
   reports_sum.md row is not touched here (another session has uncommitted edits to it); its row still says
   "replication and held test pending".
