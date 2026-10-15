# 20261016 factor-learning grid: script, prepare, timing smoke

## Problem
Task 3 of the v2 factor-learning plan: build `run_grid.py` for the 2x2 grid (C0, A, S, AS) on scorer-train rows,
run its prepare phase, and measure run times to decide local RTX 3090 versus a DAS6 node.

## Steps
1. Wrote `run_grid.py` (constants, `cell_config`, `prepare`, `run_cell`, `smoke`, `main`). `--evaluate` and
   `--tables` raise `NotImplementedError("Task 4")`.
2. `--prepare`: 183,694 scorer-train rows, 32,413 selection rows, 36,518 paintings, 1,873,347 graph edges
   (5.0 s), 64 CLIP image groups, R0 SHA-256 verified, R0 readout reference [0.4930, 0.4691]; 15.9 s total.
3. `--smoke`: each cell at 10 and 60 steps; per-step time is the slope between them.

## Smoke table (local RTX 3090, seed 42)
| cell | s/step | projected full run (min) | peak GiB |
|---|---|---|---|
| C0 | 0.248 | 8.3 | 3.86 |
| A  | 0.247 | 8.2 | 3.13 |
| S  | 0.277 | 9.2 | 3.86 |
| AS | 0.259 | 8.6 | 3.14 |

Projected grid 2,061 s (34.4 min); replication 2,100 s (35.0 min); total sequential 4,162 s (1.16 h);
peak 3.86 GiB. Decision: `run_locally` (thresholds: 45 min per run, 3 h total, 20 GiB).

## What was verified
- CUDA present; prepare numbers match expectations (183,694 rows, 36,518 paintings, 64 groups, finite reference).
- All four cells train 10 and 60 steps with finite codes; checkpoints and histories written.
- Smoke checkpoints (`*_smoke*.pt`) are discarded, never evaluated. Cache, checkpoints, results and logs are gitignored.

# Task 4: full runs, selection evaluation, rule (2026-09-30)

## Steps
1. Trained the four cells at seed 42, 2,000 steps, locally. 4 x 3.86 GiB <= 20, so all four were launched as
   parallel processes. AS died at start-up with CUDA OOM (the four processes reserved 5-7 GiB each, more than
   the 3.9 GiB allocated peak; see `run_AS_seed42_oom_attempt1.log`). C0, A and S finished in parallel
   (673.9 / 667.6 / 676.2 s); AS was rerun with the same command alone afterwards (528.2 s). All codes finite.
2. Added `apply_rule` + `_check_rule` (verbatim from the brief), `model_codes`, `selection_episodes`,
   `evaluate`, `tables` and diagnostics (`weight_summary`, `term_spread`, `history_record`, `reproduce_probe`).
   Probe helpers are imported via importlib as `probe`.
3. `--evaluate` (104 s; `run_evaluate.log`), run twice with identical results (the second run added the
   cell - C0 comparison on the beta grid as context).

## What was verified
- `_check_rule()` passed; selection episode SHA-256s equal stage (d)'s; all episode rows are selection rows.
- Codes finite on scorer-train + selection rows and NaN elsewhere; evaluation inputs NaN outside selection.
- Checkpoint configs equal `cell_config(cell, 42)`; histories are 2,000-step runs; smoke checkpoints unused.
- Original R3 reproduces the headroom probe's stored ranks exactly (naive at all betas, CLIP-only, oracle at
  0.3 and 0, oracle null): identical share 1.000.

## Result (selection rows, naive R@1 at beta 0.3, pooled mean of directions)
| model | gates | naive R@1 | D vs C0 | D_emotion vs C0 |
|---|---|---:|---:|---:|
| R3 (reference) | 9/9 | 19.36 | | |
| C0 | 9/9 | 19.71 | | |
| A | 8/9 (readout) | 16.69 | -3.03 [-3.85, -2.20] | -1.90 [-3.03, -0.81] |
| S | 8/9 (sparsity, caption 0.559) | 21.15 | +1.44 [+0.61, +2.26] | -0.73 [-1.83, +0.39] |
| AS | 8/9 (readout, caption +0.0008) | 18.95 | -0.77 [-1.62, +0.07] | -1.59 [-2.69, -0.51] |

Rule outcome: **STOP, no cell qualifies** (C0 passes all gates). Without the gates, S would still fail the
emotion guard (lower bound -1.83 <= -1.0). Replication and the held test are not run.

Report: `docs/reports/auto/v2/2026-10-16_candidate_a_factor_learning_selection.md`.

# Final fix wave (whole-branch review, 2026-09-30)

## Problem
The final review kept the stop verdict (no cell qualifies) but found attribution and wording errors in the
selection report: S's emotion guard failure read as "emotion got worse" (at beta 0.3 D_emotion is not
significant; the guard lacked power), the emotion loss placed in text->image (at beta 0 it is equal in both
directions), cell A presented as a clean test of the agreement hypothesis (the graph term kept a same-painting
pull), plus minor points (ties, mixed-beta oracle column, single-batch losses, an uncommitted check).

## Steps
1. `run_posthoc.py` (new): post-hoc diagnostics on selection rows, probes fit on scorer-train rows; nothing
   retrained; val and held never read (asserted). `--run` (97 s, GPU for encoding, 12 CPU workers for 26 probes)
   writes `results/posthoc_results.json`; `--tables` reprints. Contents: per-modality code probes, within-painting
   caption-residual emotion probe (CLIP reference 46.98%, reproduces 47.0%), AMI of the argmax pair-code factor,
   per-target S - C0, guard power, per-direction emotion table, balance-matched S vs C0 (interpolated and
   re-scored), same-beta oracle columns, ties at beta 0 (random tie-break), graph-term confound, history summary.
2. `weak_world_check.py` (committed copy of the controller's throwaway check, repo-relative paths); rerun on CPU
   (22 s wall): amp 0.25, 300 steps none 0.285 / cond 0.270; 2,000 steps none 0.3175 / cond 0.550. Matches.
3. `run_grid.py --run` refuses to overwrite an existing full checkpoint unless `--overwrite` (F8).
4. Figure 5 (`per_target.png`) added to the build script; the four existing PNGs rebuild byte-identical.
5. Report revised (verdict wording, Results 2, 4, 5, 6, new post-hoc section, next steps, caveats).

## Verified numbers (post-hoc)
- Graph: all 386,439 same-painting scorer-train pairs are graph edges (20.6% of 1,873,347). Replaying training
  steps 1-50: same-painting share of the graph term's positive pairs 18.4% edge-sampled -> 54.9% expanded.
- Guard: SE 0.565; pass needs point > +0.107; P(pass | 0) 42.5%, P(pass | -0.5) 14.1%; n 8,192 -> half-width 0.55.
- Balance-matched S - C0 (matched beta: C0 0.197, S 0.457): interpolated D +1.23 / +1.26 (12-15% of D);
  re-scored +1.05 [+0.18, +1.89] / +1.15 [+0.34, +1.94] (20-27% of D), about 55-60% of it via emotion masking.
  The reviewer's "about 15%, all by emotion masking" holds only for the interpolation.
- Probes (acc %, C0 / A / S): caption->emotion 45.77 / 38.28 / 45.63; image->style 45.53 / 39.82 / 47.36.
- Caption residual -> emotion: C0 35.62, A 30.95 (-4.68 [-5.08, -4.26]); A's within-painting variance share 0.525
  vs C0 0.447.
- AMI argmax factor vs CLIP clusters / style / emotion: C0 0.349 / 0.143 / 0.055; S 0.423 / 0.187 / 0.050.
- Ties: A beta-0 naive 16.38% tie-aware, 16.62% random tie-break; positive tied at the top in 17-120 episodes.
- Condition loss, mean of last 10 logged steps: S 0.443, AS 0.619; AS above S at every logged step after step 1.

## Result
The pre-registered verdict, rule, D / D_emotion and gate outcomes are unchanged. Report:
`docs/reports/auto/v2/2026-10-16_candidate_a_factor_learning_selection.md`.
