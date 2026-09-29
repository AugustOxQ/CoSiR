# 2026-10-12 condition-interface evaluation on the repaired factors (held-out)

## Problem

Plan Task 7 (docs/superpowers/plans/2026-10-08-cosir-v2-candidate-a-factor-repair.md). Candidate A scores an image I and a text T under a condition c as `s(I,T|c) = beta*cos(CLIP_I, CLIP_T) + sum_l w_l(c)*a_I,l*a_T,l` over 32 non-negative factors. The naive rule sets `w(c) = ReLU(mean support code - mean contrast code)`, L1-normalized.

Task 6 selected the repaired recipe R3 (InfoNCE agreement plus decorrelation) on val, under gates the user amended after the fact. This task measures, on held rows never used for any decision, whether R3 improves conditional cross-item matching over the collapsed recipe R0. The primary test uses human-label episodes (emotion and art style) that are identical for both models, so the comparison is paired. Outcome, as narrowed by the final review's post-hoc interaction analysis (report, "Post-hoc analysis from the final review"): a condition-specific improvement over R0 is shown for text→image only. In image→text, R3 is better than R0 overall, but most of that gain (83%) is condition-independent, and the interaction CI spans 0.

Inputs are Task 6's two checkpoints only. No factor training, no `src/` change.

## Investigation steps

1. **Script** `run_eval.py`. It imports, without modification:
   - from Task 6's `run_grid.py`: `build_val_episodes` (Ruling 13 construction plus its three integrity asserts), `encode_rows`, `load_art_styles` (Ruling 12 positional join plus the one-style-per-painting assert);
   - from `run_ranking_eval.py`: `choose_swap_pairs`, `episode_arrays`, `validate_roles`, and `score_pool` (parity only);
   - from `run_mechanism.py`: `bootstrap_difference`, `role_outrank`, and `swap_reversal` (cross-check only).
2. **Decision rules fixed in code before the real run:**
   - β per variant: the highest mean bidirectional R@1 on the val episodes over {0, .001, .003, .01, .03, .1, .3, 1, 3}, with ties going to the smaller β. The grid is never extended.
   - Top-k: β is chosen per k, then k ∈ {1, 3, 5} by the same utility at its own β, with ties going to the smaller k.
   - Criteria: applied at each variant's selected β, with β=0 reported as information only.
   - Criterion 2 compares swap-reversal *rates*, because the two models have different numbers of valid pairs.
   - "CI above 0" means a lower bound strictly greater than 0.
3. **Smoke run** (`--smoke`, 26.5 s, output under `cache/smoke/`):
   - val rows stood in for held rows, with 256 label and 256/512 mined episodes;
   - it exercised every code path;
   - it found that `choose_swap_pairs` raises when a model has no valid pair (R0 on 256 stand-in episodes). The script now records "0 pairs" and marks criterion 2 as not evaluable in that case, instead of crashing. This was a code-path fix; no number from the smoke run informed any choice.
   - Like the real run, the smoke run encoded all rows, including held, for the all-codes finiteness assertion. No held-row metric was computed. The smoke results JSON was deleted; only `smoke.log` remains (gitignored).
4. **Real run**, once: `run_eval.py`, seed 42, RTX 3090, torch 2.11.0+cu130. **97.5 s.** Log: `run_eval.log`. Results: `results/eval_results.json` (both gitignored).
   - Split: 216,107 / 30,872 / 61,744 rows, with zero leakage (asserted). 27 art styles, one per painting and per leakage group (asserted).
   - **Val label episodes** were rebuilt with Task 6's call: 2,048 emotion over 8 targets (with "something else" excluded) and 2,048 art style over 23 targets, from val, seed 42. Their metadata equals Task 6's stored `summary.json` episode metadata (asserted). In addition, `condition_lift` on these episodes with the reloaded checkpoints reproduces Task 6's stored val lifts **exactly**: 24/24 values per model, covering naive and uniform, R@1, R@3 and tie counts. So the episodes and codes are Task 6's.
   - **Held label episodes** were built once, from held, seed 42: 1,024 emotion over 8 targets and 1,024 art style over 24 targets (New_Realism is eligible on held). The same arrays were used for R0 and R3.
     - SHA-256 emotion: `e62ab41f...c85`.
     - SHA-256 art style: `3a58cf9d...67f`.
   - **Checkpoints:** `load_factor_checkpoint`; R0 is cosine with no decorrelation, and R3 is InfoNCE with decorrelation 1.0 (asserted). All 308,723 rows were encoded in 8,192-row batches under `no_grad`. All codes are finite (asserted).
   - **Factor-mined episodes**, per model: 4,096 on val and 1,024 on held, mined split-locally under single-threaded BLAS and remapped to global rows. `validate_roles` and `episode_arrays` passed.
     - `conditional_score` against `score_pool` on the first 16 episodes: maximum absolute difference 3.0e-8 (R0) and 2.4e-7 (R3), below the 1e-5 limit.
     - Swap counts from `conditional_score` equal Task 9's `score_pool`-based `swap_reversal` for every variant.

## Result

Readiness criteria (brief Step 3), applied literally:

| criterion | verdict | numbers (held, selected β, R@1 points [95% CI]) |
|---|---|---|
| 1 primary: label episodes, pooled | **met** | R3 naive − R0 naive: +4.5 [+2.7, +6.3] i2t, +4.2 [+2.2, +6.2] t2i. R3 naive − R3 uniform: +2.0 [+0.5, +3.6] i2t, +5.2 [+3.5, +6.8] t2i |
| 2 secondary: naive swap reversal above R0's | **not met** | R3 160/256 = 62.5% i2t, 167/256 = 65.2% t2i. R0 9/9 = 100% i2t, 4/9 = 44.4% t2i. Higher in t2i only; R0 has only 9 valid pairs |
| 3 floor: mined naive − uniform | **met** | R3 +21.4 [+18.5, +24.2] i2t, +22.7 [+19.8, +25.6] t2i |

The full tables are in the report.

Criterion 1 is met as written, but it has no interaction test. The final review's post-hoc difference-in-differences, (R3 naive − R3 uniform) − (R0 naive − R0 uniform), is +0.78 [−1.03, +2.64] i2t and +4.39 [+2.39, +6.40] t2i at the selected β. It is cited in the report, informed no decision, and was not recomputed.

## Root cause / interpretation notes

- On mined episodes, R3 and R0 differ mostly because the *episodes* differ.
  - With decorrelated factors, "anchor-only" distractors (similar to the anchor on every non-target factor) are real competitors. With uniform weights they outrank the positive in 87-89% of R3's held episodes, against 3-19% for R0's (depending on direction and β).
  - So R3-mined episodes are far harder for any scorer that is not conditioned. Absolute mined R@1 is therefore not comparable across models.
- Mined roles are distinct rows, not distinct paintings (the protocol is unchanged, for comparability).
  - 791/1,024 of R3's held episodes, and 331/1,024 of R0's, repeat a painting among their 22 roles.
  - 34 of R3's and 2 of R0's contain a candidate from the anchor's or the positive's painting.
  - In t2i, a candidate from the positive's painting has the identical image. That gives an exact tie, which counts against the positive. At the selected β, 20-21 of R3's held t2i episodes are tied in every variant, including CLIP-only.
- R0 finds only 9 valid swap pairs on the painting split, against 61 in Task 9 on the row split. Its swap rates therefore have wide exact intervals: [66.4, 100] and [13.7, 78.8].

## Solution implemented

This task is an evaluation, so no code fix was needed. Report: `docs/reports/2026-10-12_cosir_v2_candidate_a_condition_eval_repaired_factors.md`. The stage (d) decision is the user's.

## Files

- `run_eval.py`: the evaluation. `--smoke` runs the smoke test; `--tables` reprints the tables from the results JSON.
- `20261012_condition_eval_repaired_factors_log.md`: this log.
- `.gitignore`: `*.npy`, `*.json`, `*.pt`, `cache/`, `*.log`.
- Gitignored:
  - `run_eval.log`
  - `results/eval_results.json`
  - `cache/smoke/smoke.log`

**Disclosure (final review):** `run_eval.py` was edited after the real run: the `clopper_pearson` helper and the `--tables` option were added (file mtime 23:36:48, results written 23:35:53; `run_eval.log` has no Clopper-Pearson line). No computed result changed.
