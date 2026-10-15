# 2026-10-13 stage (d) selection run: G1-G5 vs the naive rule on the selection set

## Problem

Plan Task 6 (`docs/superpowers/plans/2026-09-30-cosir-v2-candidate-a-stage-d.md`); spec
`docs/superpowers/specs/2026-09-30-cosir-v2-candidate-a-stage-d-design.md` §3-§5 and §10. Five
conditional scorers were trained on **frozen** R3 factor codes, using self-generated conditions
(G1/G2 factor combinations, G3/G4 CLIP clusters, G5 Stage-1 communities; G2/G4 add the swap term).
They were compared with the naive rule (the step-0 model, β = 0.3) on human-label episodes from a
15% selection carve-out of the train paintings. The selection score is the condition-use gain
Δ = [R@1(m) − R@1(m | wrong c)] − [R@1(naive) − R@1(naive | wrong c)], pooled over emotion and art
style and averaged over i2t and t2i, in R@1 points. The plan has a pre-registered stop point at
best Δ ≤ +0.5.

**Outcome:** no stop. The best score is G5 at +3.00 [+2.05, +3.99]. G3 (+2.54) and G4 (+2.36) tie
with it (within 1.0 point). The tie goes to a no-swap run, then to table order, so the rule selects
**G3** (CLIP clusters, no swap). Task 7 is not started here; the controller dispatches it.

No `src/` file was modified, so there is no `.claude/` change-log entry.

## Investigation steps

1. **Script** `run_selection.py`, with phases `--prepare`, `--run G{k}`, `--evaluate` and `--tables`.
   It uses only committed interfaces from Tasks 1-5. The only non-public access is
   `ClipClusterSource._setup`, used to rebuild the source from cached k-means labels. `--prepare`
   asserts the rebuilt source equals the fitted one (same valid keys, same member arrays).
2. **Row-scope guards**, in addition to the plan's asserts:
   - The cache holds codes for train-part rows only; val and held entries are NaN.
   - `--run` NaN-masks CLIP features and codes outside scorer-train rows.
   - `--evaluate` NaN-masks everything outside the train part (scorer-train + selection).
   - So any read of val, held, or (during training) selection rows would show up as a non-finite
     loss or score. Every training history is asserted finite.
   - The selection episodes are asserted to use selection rows only.
   - Held rows were never used: the brief's "encode all rows" finiteness check ran once in
     `--prepare`, then the held and val codes were discarded (not cached).
3. **`--prepare`** (60.8 s, `run_prepare.log`, `cache/prepare.{npz,json}`):
   - The split is 216,107 / 30,872 / 61,744 (asserted).
   - Scorer-train has 183,694 rows (36,518 leakage groups). Selection has 32,413 rows (6,451 groups),
     which is 15.0% of train rows. The two parts are group-disjoint and their union is the train
     part (asserted).
   - The R3 SHA-256 matched `1c299fc0...e453f`, and its config is R3 (asserted). All 308,723 rows
     encode to finite codes. `factor_scale` (std of scorer-train pair codes + 1e-6) lies in
     [0.168, 0.335].
   - **factor_combo:** all 1,000 random draws were valid. Inside groups have 18,370-18,371 rows (top
     10%) and outside groups 91,847-94,752. Swap overlaps over 100 pairs: 48-6,436, median 1,231.
   - **clip_cluster:** 64 + 64 groups, all valid (k-means took 4.2 s). Image clusters have 1,605-4,962
     rows; caption clusters 635-7,431. Swap overlaps: 25-1,461, median 112.5.
   - **community:** content graph with 1,873,347 edges; Stage 1 loss 6.12 → 3.18 over 200 epochs;
     Leiden found 19 communities, all valid, of 2,875-17,410 rows (42.2 s).
   - **Mined-batch scope check:** one real batch per source (64 episodes, plus 64 swap episodes
     where the source can swap), using the seed-42 generator. All rows are scorer-train rows and no
     painting repeats inside an episode.
4. **Timing pilot and smoke run.** Both ran from the scratchpad, and their numbers were discarded;
   no choice depended on them.
   - The pilot trained 30 steps per run config on the GPU with 6 threads. Projected 3,000-step
     times were: G1 11.5 min, G2 27.6, G3 3.6, G4 10.7, G5 3.3. All are under the 60-minute limit.
   - `--evaluate` was smoke-tested on those 30-step checkpoints. It exercised every code path. The
     only fix needed was tolerating missing loss-curve steps in `--tables`.
5. **Real runs:** five parallel OS processes, one per run, with `OMP/MKL/OPENBLAS_NUM_THREADS=6` and
   a 65-minute timeout each. Each used `ScorerTrainingConfig()` defaults exactly (3,000 steps, batch
   64, lr 1e-3, seed 42), with only `swap` set per the plan's table (asserted). Training ran on the
   RTX 3090 with torch 2.11.0+cu130.
   - Train times: G1 13.1 min, G2 31.5, G3 5.6, G4 14.9, G5 5.4. All exited 0 with finite histories.
   - Checkpoints are `checkpoints/G{k}.pt` and histories `results/history_G{k}.json` (both
     gitignored). The SHA-256s are in `results/selection_results.json`.
6. **`--evaluate`** (25 s, CPU, `run_evaluate.log`, `results/selection_results.json`,
   `results/selection_ranks.npz`):
   - Selection episodes are `standard_label_episodes` on selection rows, seed 42:
     - emotion: 2,048 episodes over 8 targets, SHA-256 `84056321...439ea0`;
     - art style: 2,048 episodes over 23 targets, SHA-256 `e1cfe1ba...cc4c46`.
   - `wrong_condition(seed=42)` was applied per label type, and the derangement was traced back. It
     hands an episode the condition of another **same-label** episode in 11.9% of emotion episodes
     and 4.1% of art-style episodes.
   - Naive is `train_scorer(..., replace(ScorerTrainingConfig(), steps=0), device="cpu")`, using the
     factor_combo source only to set τ, which does not affect ranking. β = 0.30000001.
     - **Pin:** its weights `torch.equal` `label_episode_weights` on the first 16 emotion and the first
       16 art-style selection episodes.
     - Its ranks equal `label_episode_recall(rule weights, β=0.3)` on 100% of episodes, in both
       directions and for both label types.
7. **Descriptive contrasts** (G2−G1, G4−G3, G5−G3). These were added after the first evaluation and
   are not part of the rule. `--evaluate` was then re-run: the models, baselines, selection, meta and
   history blocks are **bit-identical** to the first evaluation (compared as JSON; the first copy is
   kept as `results/selection_results_first_eval.json` and `run_evaluate_first.log`).

## Result

Selection score: Δ pooled over label types, R@1 points, with the 95% paired-bootstrap CI (5,000
resamples, seed 42).

| Run | Source | Swap | Score | i2t | t2i |
|---|---|---|---:|---:|---:|
| G1 | factor_combo | no | +0.46 [−0.54, +1.51] | +0.54 [−0.90, +1.95] | +0.39 [−1.00, +1.76] |
| G2 | factor_combo | yes | +0.79 [−0.23, +1.81] | +0.61 [−0.76, +2.00] | +0.98 [−0.37, +2.34] |
| G3 | clip_cluster | no | +2.54 [+1.44, +3.67] | +1.83 [+0.32, +3.32] | +3.25 [+1.76, +4.79] |
| G4 | clip_cluster | yes | +2.36 [+1.29, +3.49] | +2.08 [+0.59, +3.59] | +2.64 [+1.12, +4.15] |
| G5 | community | no | +3.00 [+2.05, +3.99] | +2.69 [+1.34, +4.03] | +3.32 [+2.00, +4.69] |

How the rule applied:
1. The best run is G5 (+3.00).
2. G3 (+2.54) and G4 (+2.36) are within 1.0 point of it, so G3, G4 and G5 tie.
3. The tie goes to a no-swap run, leaving G3 and G5, then to the earlier run in the table: **G3**.
4. The stop point does not apply, since +3.00 > +0.5.

Naive R@1 is 18.36 (i2t) / 20.36 (t2i). G3 reaches 18.87 / 21.66 and G5 19.68 / 21.66. The ceiling
(per-episode oracle weights, β = 0.3) is 84.72 / 86.65. Every run drove β from 0.30 to 0.040-0.050.
The full tables are in the report `docs/reports/auto/v2/2026-10-13_candidate_a_stage_d_selection.md`,
and `--tables` reprints them.

## Solution / follow-up

Nothing needed fixing. The selected run is **G3** (CLIP clusters, no swap, seed 42), checkpoint
`checkpoints/G3.pt`. Task 7 (seeds 43/44 and the final held-out test) is the controller's call.

## Reproduction recipe

From the repository root, CoSiR environment:

```bash
python src/test/20261013_stage_d_selection/run_selection.py --prepare
# the five runs as parallel OS processes, each with 6 BLAS threads:
for r in G1 G2 G3 G4 G5; do
  OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 \
    python src/test/20261013_stage_d_selection/run_selection.py --run $r > src/test/20261013_stage_d_selection/run_$r.log 2>&1 &
done; wait
python src/test/20261013_stage_d_selection/run_selection.py --evaluate
python src/test/20261013_stage_d_selection/run_posthoc.py --run          # post-hoc diagnostics (below)
```

- **BLAS threads:** the five training runs used `OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS = 6`
  (recorded in each `results/history_G{k}.json` as `omp_num_threads`). `--evaluate` and the post-hoc run used
  the default thread count (the post-hoc run recorded 16 torch threads).
- **Parallel vs sequential:** every run owns its seed-42 generators, so running the five in parallel gives
  the same result as running them one after another **up to floating-point nondeterminism** (GPU training
  is not bit-deterministic, about 1e-6). This was argued by construction and was not tested empirically.

## Post-hoc diagnostics (final fix wave)

**Every number in this section is post-hoc, on selection rows (scorer-train rows for the mechanism
check), and informed no pre-registered decision.** The pre-registered selection (G3) and Task 7's two
verdicts are unchanged. No model was trained and no held row was read.

### Problem

The final whole-branch review (`.superpowers/sdd/2026-09-30-cosir-v2-candidate-a-stage-d/final-fix-findings.md`)
found the code and verdicts sound but the interpretation wrong in four places:
- criterion 2's pass is reproduced by the naive rule at G3's learned β (C1);
- the per-episode ceiling is uninformative (C2);
- criterion 1's miss is underpowered, not negative (I1);
- about a third of Δ is the β drop (I2), and the β-collapse mechanism needed a check (I3).

### Investigation steps

1. **`src/` additions (TDD, change log in `.claude/20261013_log.md`):**
   - `ceiling_ranks(..., target_column=0)`: an int or one int per episode declares another candidate the
     target; a random distractor gives the ceiling's null.
   - `label_oracle_ranks(...)`: one softmax-parametrized weight vector per target label, fitted on the
     label's other folds (2 folds, 200 Adam steps, lr 0.1, seed 42; smooth margin as in the ceiling,
     both directions, labels weighted equally), ranks out of fold.
2. **Script** `run_posthoc.py`. It imports `run_final.py` (and through it `run_selection.py`) and reuses
   their cache, episode construction, masking, swap checks and swap halves.
   - `--run`: 134 s on CPU, 16 torch threads (`run_posthoc.log`).
   - Output: `results/posthoc_results.json` and `results/posthoc_ranks.npz` (gitignored).
   - `--tables` reprints everything.
3. **Row scope.** The selection analyses NaN-mask every row outside the selection set. The mechanism
   check NaN-masks everything outside scorer-train. Held rows are not read. The I1/M2 numbers are
   arithmetic on Task 7's stored `final_results.json`.
4. **Reproduction checks, all exact:**
   - naive (β 0.3) and G1-G5 ranks equal Task 6's stored ranks on 100% of episodes, right and wrong
     condition, every scope; their Δ equal Task 6's bit for bit;
   - the selection swap episodes (SHA-256 `41f8215f…2478`) and the naive and G3 success rates equal Task
     7's discarded smoke run;
   - the per-episode ceiling equals Task 6's stored ceiling ranks on 100% of episodes.

### Results (post-hoc, selection rows)

- **C1, swap test (1,024 selection episodes; success i2t / t2i).**
  - naive at β 0.3: 18.75 / 21.00. Success rises steadily as naive's β falls:
    - β 0.2: +2.00 [+1.32, +2.73] pooled over naive at 0.3;
    - β 0.1: +4.00 [+2.98, +5.08];
    - G3's β 0.0496: +5.22 [+3.91, +6.54];
    - β 0.02: +6.05 [+4.64, +7.47].
  - G3: +5.27 [+3.22, +7.37].
  - **G3 − naive at G3's β: +0.05 [−2.10, +2.20].**
  - G3's interface at β 0.3: +0.68 [−1.17, +2.64].
  - Of the five runs, only G5 adds swap success beyond its own β drop: +2.59 [+0.73, +4.44], and its
    interface at β 0.3 gives +2.69 [+0.98, +4.39]. G5 was not selected and not evaluated on held.
- **I2, Δ decomposition (pooled mean, R@1 points).** Δ(run, naive@0.3) = β drop + beyond, exactly.

  | Run | Total | β drop | Beyond |
  |---|---|---|---|
  | G3 | +2.54 | +0.88 [+0.31, +1.44] (35%) | +1.66 [+0.55, +2.80] (i2t +0.56 [−0.95, +2.05], t2i +2.76 [+1.27, +4.30]) |
  | G1 | +0.46 | +0.90 | −0.44 [−1.45, +0.63] |
  | G2 | +0.79 | +0.95 | −0.16 [−1.15, +0.85] |
  | G4 | +2.36 | +0.88 | +1.48 [+0.37, +2.61] |
  | G5 | +3.00 | +0.95 | +2.05 [+1.06, +3.08] |

  - Each run's interface at β 0.3: G3 +1.62 [+0.59, +2.71], G1 −0.29, G2 +0.09, G4 +1.53, G5 +2.12.
  - The review's reference values for G1/G2/G5 (−0.42, −0.09, +2.12) were computed against naive at
    G3's β. The values here use each run's own β, as the finding specifies. G3's values match.
- **C2, ceilings (R@1 i2t / t2i, β 0.3; chance 7.69%):**
  - naive 18.36 / 20.36;
  - per-episode ceiling 84.72 / 86.65, and its random-target null 77.12 / 79.27;
  - **label oracle (CV) 18.41 / 21.04**, and its random-target null 7.69 / 7.62.
  - Label oracle − naive: +0.05 [−1.12, +1.22] / +0.68 [−0.49, +1.88].
  - At G3's β, the label oracle scores 19.19 / 21.61 against naive's 18.80 / 20.24: +0.39 [−0.85, +1.64] /
    +1.37 [+0.15, +2.61].
- **I3, mechanism check (1,024 mined scorer-train episodes per source, seed 42; pairwise accuracy, ties
  count one half).**
  - CLIP alone ranks a positive above a hard negative only 28.7-35.3% of the time, and above a random
    negative 68.2-77.0% of the time.
  - As naive's β goes 0.3 → 0, positive-over-hard rises by 2.7-3.5 points in every source and direction
    (clip_cluster 49.6 → 52.3 / 46.1 → 48.8). Positive-over-random stays within 0.6 points (clip_cluster
    85.6 → 85.3 / 85.7 → 85.8).
- **I1 / M2, from the stored held numbers (normal approximation).**
  - Held SE is 1.00 (i2t) and 1.11 (t2i).
  - If the held effect equalled the selection points (+1.83, +3.25), the chance that each lower bound is
    above 0 would be 0.45 (i2t) and 0.83 (t2i). Both at once: 0.38 assuming independence, 0.39 with the
    selection i2t/t2i correlation (0.10).
  - The held CIs contain the selection points.
  - Dropping the 8.74% same-label wrong conditions scales the point by ×1.096 and the SE by ×1.047, so z
    rises only ×1.047. The lower bounds become −1.13 and −0.51.

### Solution / follow-up

Both reports now carry these post-hoc sections and corrected wording; the pre-registered verdict lines
are unchanged. The β-matched control on held rows was **not** run: it is listed as a user option in the
final report.
