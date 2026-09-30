# 2026-10-14 stage (d) final held-out test: G3 vs the naive rule on held rows

## Problem

Plan Task 7 (`docs/superpowers/plans/2026-09-30-cosir-v2-candidate-a-stage-d.md`); spec
`docs/superpowers/specs/2026-09-30-cosir-v2-candidate-a-stage-d-design.md` §5 (replication), §6 (final
test, pre-registered) and §10 (caveats). Task 6 selected **G3** (CLIP-cluster conditions, no swap term,
seed 42). This task retrains G3's recipe with seeds 43 and 44, then touches the held split for the first
time in stage (d) to test the two pre-registered criteria:

1. **Criterion 1 (primary):** condition-use gain Δ = [R@1(G3) − R@1(G3 | wrong c)] − [R@1(naive) −
   R@1(naive | wrong c)] on the held label episodes (1,024 emotion + 1,024 art style, pooled). Met iff
   the 95% CI lower bound is > 0 in **both** i2t and t2i. Judged on seed 42.
2. **Criterion 2 (human swap test):** 1,024 held emotion-vs-style swap episodes; met iff G3's success
   rate minus naive's has a pooled paired-bootstrap 95% CI lower bound > 0.

**Outcome:** criterion 1 **NOT MET** (i2t +0.83 [−1.12, +2.78], t2i +1.61 [−0.54, +3.81]); criterion 2
**MET** (+6.40 [+4.20, +8.54] points; success 25.4% / 26.6% vs naive's 18.8% / 20.3%). Seeds 43 and
44 give the same verdicts on both tests.

No existing `src/` file was modified, so there is no `.claude/` change-log entry.

## Investigation steps

1. **Script** `run_final.py`, with phases `--prepare`, `--run 43|44`, `--selection`, `--final [--smoke]`
   and `--tables`. It **imports** Task 6's `src/test/20261013_stage_d_selection/run_selection.py` by file
   path (the folder name is not a package name) rather than copying it. Reused: Task 6's cache
   (`cache/prepare.{npz,json}`), `build_source` / `clip_source_from_labels`, `masked` / `scoped_inputs`,
   `same_label_fraction`, `recall_summary`, `pool`, `weight_stats`, `concat_episodes`,
   `history_summary`, `run_config`, and the constants. Only committed interfaces from Tasks 1-5 are
   called otherwise.
2. **`--prepare`** (7.8 s, `run_prepare.log`, `cache/prepare_checks.json`, `cache/held_codes.npz`),
   every item asserted:
   - split 216,107 / 30,872 / 61,744; the leakage groups, `split.train`, scorer-train (183,694) and
     selection (32,413) rebuilt from scratch equal Task 6's cached arrays; held groups are disjoint from
     train and val;
   - R3 SHA-256 `1c299fc0…e453f` and its config; all 308,723 rows re-encoded (as Task 6 did); the
     train-part codes and `factor_scale` are **bit-identical** to Task 6's cache;
   - **CLIP-cluster source:** `ClipClusterSource` refit on scorer-train rows (k=64, seed 42, 4.3 s)
     gives labels equal to Task 6's cached labels for both views; the source rebuilt from the cache has
     the same 128 valid groups and members;
   - G3 checkpoint SHA-256 `8d9317c2…fd63`, config = `ScorerTrainingConfig()` defaults.
   - Only the held rows' codes were cached for `--final`; no held metric was computed here.
3. **`--run 43` / `--run 44`** (`run_seed43.log`, `run_seed44.log`): two parallel OS processes on the RTX
   3090, `OMP/MKL/OPENBLAS_NUM_THREADS=6`, config = G3's with only `seed` changed (asserted), trained on
   the CLIP-cluster source rebuilt from Task 6's cached labels with inputs NaN-masked outside
   scorer-train rows. 2.9 min each; histories finite. β fell 0.30 → 0.042 (seed 43) and 0.056 (seed
   44), like seed 42 (0.050).
4. **`--selection`** (9.9 s, `run_selection.log`, `results/selection_seeds.json`): Task 6's selection
   episodes rebuilt; SHA-256s equal Task 6's (emotion `84056321…439ea0`, art style `e1cfe1ba…cc4c46`,
   asserted). Naive built by Task 6's exact call (step-0 `train_scorer`, β 0.3) and pinned with
   `torch.equal`. **G3 seed 42 reproduces Task 6's recorded selection gain exactly** (every scope,
   asserted). Selection scores: seed 42 +2.54, seed 43 +3.00, seed 44 +2.32 (all CIs above 0).
5. **`--final --smoke`** (9.1 s, `run_smoke.log`, `results/smoke_final.json`): the whole final code path
   on **selection rows** in place of held rows (no held row read; SHA asserts skipped). Its numbers were
   discarded and informed nothing. The one path the smoke skips (comparison with the 2026-10-12 report's
   recorded numbers) was exercised separately on the smoke output so the held run could not crash
   before saving.
6. **`--final`**, run **once** (23.0 s, `run_final.log`, `results/final_results.json`,
   `results/final_ranks.npz`). The phase refuses to run again once its JSON exists.
   - Inputs NaN-masked outside the 61,744 held rows.
   - Held label episodes: `standard_label_episodes(data, groups, split.held, label, 1024, seed=42)`;
     SHA-256s **equal** the recorded ones (emotion `e62ab41f…8c85`, art style `3a58cf9d…422e167f`,
     asserted). 8 emotion targets, 24 art-style targets.
   - `wrong_condition(seed=42)` per label type; traced back, it hands an episode a same-label condition
     in **12.9%** of emotion and **4.6%** of art-style held episodes.
   - Naive: Task 6's build, τ equal to Task 6's, weights `torch.equal` `label_episode_weights` on the
     first 16 held episodes per label, ranks identical to `label_episode_recall(rule, β 0.3)` on 100% of
     episodes. Naive and CLIP-only held R@1/R@3 **equal** the 2026-10-12 report's recorded values in
     every cell (max |diff| 0).
   - Human swap episodes: `build_human_swap_episodes(data, groups, split.held, 1024, seed=42)`; checked
     in the script: every row held, no leakage group repeats inside an episode, one-aspect clean
     (supports, contrasts, `p_emo`, `p_style` and negatives as defined), 8 anchor emotions, 24 anchor
     styles, SHA-256 `f1195dcc…0062`. `human_swap_success` was cross-checked against a component
     recomputation (`p_emo > p_style` under c_emo, `p_style > p_emo` under c_style).

## Result

| Test (held) | G3 seed 42 (judged) | seed 43 | seed 44 | Verdict |
|---|---|---|---|---|
| Criterion 1, Δ i2t | +0.83 [−1.12, +2.78] | +1.17 [−0.73, +3.12] | +0.78 [−1.22, +2.78] | |
| Criterion 1, Δ t2i | +1.61 [−0.54, +3.81] | +1.37 [−0.73, +3.56] | +2.05 [+0.00, +4.25] | **NOT MET** |
| Criterion 2, success diff pooled | +6.40 [+4.20, +8.54] | +8.11 [+6.01, +10.16] | +5.91 [+3.81, +7.96] | **MET** |

Held R@1 (i2t / t2i): naive 17.63 / 20.90, G3 16.70 / 21.34, CLIP-only 11.52 / 14.89, uniform
(β 0.3) 15.43 / 16.26. G3 − naive plain R@1: −0.93 [−2.44, +0.59] / +0.44 [−1.17, +2.10]. Swap
success: G3 25.39% / 26.56%, naive 18.85% / 20.31%; CLIP-only and uniform 0% by construction.
Seed 44's t2i lower bound is exactly 0.00, so its t2i fails the strict "> 0" test.

Full tables: `--tables`, and the report
`docs/reports/auto/v2/2026-10-14_candidate_a_stage_d_final.md`.

## Solution / follow-up

Nothing needed fixing. The pre-registered verdict is: criterion 1 not met, criterion 2 met. Held rows
were used once, for these measurements only, and informed no choice. The stage (e) decision is the
user's.
