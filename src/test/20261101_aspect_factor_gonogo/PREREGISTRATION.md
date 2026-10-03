# E3 go/no-go pre-registration (CoSiR v2 CVPR plan, Tasks 12 to 14)

Written and committed on 2026-10-03, before any training run of this experiment, smoke runs included.

- **Binding authority:** the spec `docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md`
  as of commit 703af5d (file SHA-256 `d76a5bab850bbcf5510bcb6bf6c0c08f630e6aed84567512fe34aeb11e6a3a33`), §4, §6
  and §10, and the plan `docs/superpowers/plans/2026-10-03-cosir-v2-cvpr-e0-e5.md`, Tasks 12 to 14.
- **Mechanical:** every rule below is evaluated by `run_gonogo.py` in this folder from stored arrays (per-anchor
  metrics, λ picks, checkpoint and episode SHA-256s are written next to every result).
- **Frozen:** nothing here changes after a training run has started. A change can only be a dated addendum at the end
  of this file, committed before the result it affects is computed, with its reason.

## 1. Spec §6, copied verbatim

### Held-out-aspect test

**Held-out-aspect test** (review DA C1, validated; claim K8). It is pre-registered and runs in the go/no-go week.
- **Leave-one-aspect-out on ArtELingo:** train with the emotion-like and style-like partitions only (no
  caption-content partition), then evaluate the episodes that involve genre (genre × emotion, genre × style).
- **Rule:** K8 holds if genre-pair condition gain has a clustered 95% CI lower bound above 0 against the
  uniform-weight control. Otherwise C2 narrows to "selection among aspects represented in training", and the paper
  says so.
- **Generic partitions on CUB** (table above) give a second, cross-dataset test of the same question.
- **Supervision ablation:** an emotion run without the affect partition separates the GoEmotions signal from the
  rest (review R2 W4).

### Go/no-go

**Go/no-go.** Pre-registered before the first run; ArtELingo, CLIP B/32, selection rows; decided Fri Oct 9.
- **Grid and picking:** at most about 10 runs (partition set × L × loss weights), about 10 minutes each locally.
  The best run is **picked on seed-42** selection aspect episodes.
- **Fresh test episodes:** the GO test uses **fresh seed-43 episodes** from the same selection rows (the same rows,
  so the test is not independent of development; disclosed). Picking the best of about ten runs therefore does not
  inflate the result.
- **GO** if, on those fresh episodes and with the painting-clustered bootstrap of §10, the picked run beats
  **each** of the following on **both** R@1 and condition gain (95% CI lower bound above 0):
  - backbone-only (R@1 11.1, condition gain 0);
  - the best raw-feature metric-from-pairs baseline, fusion weight cross-fitted;
  - the *uniform-weight control*: our factors with uniform weights, i.e. the condition removed.
- **Strong GO:** R@1 at least 4 points above backbone-only (about 15 against 11.1), with condition gain at least
  +4 points. The label-probe reference (about 23) is a diagnostic, not a bound.
- **Early MLLM probe** (decides branch 2 vs 3, §4):
  - **Setup:** Qwen3-VL-2B-Instruct as an in-context reranker, given the 4 support and 4 contrast pairs, the query
    and the 13 candidates.
  - **Size:** 300 selection episodes per aspect pair, seed 44.
  - **Cost:** t2i prompts carry about 30 images each, so the 8B model is used only if time allows.
  - **Rule:** it "works" if condition gain and R@1 over backbone-only both have a CI lower bound above 0.
- **NO-GO:** GO is missed. The user chooses branch 2 or 3 (§4) from the numbers.
- **Replication:** a winner is re-run at seeds 43 and 44 before it moves to other datasets and Qwen.

## 2. Data, rows and episodes

- **Backbone:** frozen CLIP ViT-B/32 features from `src.data.artelingo.load_artelingo()`.
- **Training rows:** the 183,694 scorer-train rows of `artelingo_splits()`, as local rows 0..n-1 in that order.
- **Evaluation rows:** the 32,413 selection rows (6,451 paintings). Val and held rows are never read. Every code array
  that enters an evaluation is NaN outside the selection rows, and the script asserts it.
- **No evaluation labels in training.** Emotion, style and genre labels are used only for episodes and the
  value-sharing diagnostic. The pseudo-partitions are E2's (`src/test/20261031_pseudo_partitions/`), k = 64 each:
  *affect* (GoEmotions affect k-means of captions; distant supervision, spec §4 C2), *image* (CLIP image k-means),
  *caption* (CLIP caption k-means).
- **Selection episodes** (the pick): Task 9's `results/episodes_seed42.npz`. **Test episodes** (GO, K8, ablation):
  Task 9's `results/episodes_seed43.npz`. Both hold the three aspect pairs in the pooled order emotion×style,
  emotion×genre, style×genre, with the third aspect controlled, 4,096 episodes per pair (12,288 pooled). Their SHA-256s
  are checked against Task 9's `baselines_seed{42,43}.json` before use.
- **Clusters** for every bootstrap: the anchor's painting, i.e. `groups[anchor]` with `groups` the leakage-group ids of
  `artelingo_splits()` (rows that share a painting or an exact image vector form one group).

## 3. The training grid (Task 12)

- **Base config of every run:** the C0 recipe of the factor-learning grid, `cell_config("C0", seed)` in
  `src/test/20261016_factor_learning_grid/run_grid.py`. It equals the plan's
  `dataclasses.replace(R3_CONFIG, painting_batches=True)` field by field (checked on 2026-10-03; no field differs).
- **On top of it:** `epochs = 2000` training steps, model seed 42, `aspect_episodes_per_step = 32`,
  `aspect_beta = 0.3` unless a row says otherwise.
- **Inputs:** scorer-train features only, `group_ids = partitions.npz["local_groups"]`, E2's content graph
  `graph.npz`, and the bank named in the table (E2's `bank_<name>.npz`, 65,536 pseudo-aspect episodes on local rows).

| Run | Bank (partitions) | Changes to the base config | Eligible to be picked |
|---|---|---|---|
| A1 | AIC (affect, image, caption) | `lambda_aspect=1, lambda_swap=1` | yes |
| A2 | AIC | A1 with `num_factors=64` | yes |
| A3 | AIC | A1 with `lambda_aspect=3` | yes |
| A4 | AIC | A1 with `lambda_swap=0` | yes |
| A5 | AIC | A1 with `aspect_beta=0.0` | yes |
| A6 | AIC | A1 with `lambda_sparsity=0.0, lambda_decorrelation=0.1` (denser codes, so values can share factors) | yes |
| H1 | AI (affect, image) | A1's settings: the **held-out-genre model** (K8) | no (test only) |
| S1 | IC (image, caption) | A1's settings: the **supervision ablation** (no affect) | no (descriptive) |

- No other run enters the pick.
- A run that ends with non-finite codes is recorded as failed and is not eligible. If every A run fails, there is
  no picked run and GO is missed.
- An identical rerun is allowed only after an infrastructure failure (a crash, or out of memory caused by another
  process), never with changed settings.
- Replication at model seeds 43 and 44 happens only after a GO (spec §6, plan Task 17).

## 4. Scoring and metrics (used in every section below)

- **Backbone-only score:** cosine of unit CLIP features (`src.eval.aspect_scorers.cosine_scores`).
- **A factor model's score:** the agreement rule on its codes (`agreement_term`),
  w = ReLU(mean over S of a_I ⊙ a_T − mean over C of a_I ⊙ a_T), L1-normalized, term = Σ_l w_l q_l c_l. It is fused
  with the cosine by per-episode z-fusion (`zfuse`: z(cos) + λ·z(term); λ = inf means the term alone, λ = 0 the cosine
  alone).
- **Uniform-weight control:** the same model and fusion with w = 1/L (`agreement_term(..., uniform=True)`). It removes
  the condition, so its condition gain is 0 by construction.
- **λ cross-fitting** (`crossfit_lambda`):
  - within each episode set (seed 42 for the pick, seed 43 for the tests), over the three pairs pooled, with parity
    `np.arange(n) % 2`;
  - each half picks λ from {0, 0.25, 0.5, 1, 2, 4, 8, 16, inf} by the mean of R@1 and condition gain on that half
    (ties go to the earlier grid value); if that half picks 16 its grid is extended once to {32, 64};
  - the half's pick is applied to the other half;
  - every method is cross-fitted on its own, its uniform control included.
- **Metrics** (`src.eval.aspect_metrics.per_anchor`): R@1 (mean over both directions and both conditions), condition
  gain (R@1 minus the other-aspect rate), other-aspect rate, swap success and strict swap. Ties and rows with any
  non-finite score count as misses.
- **Uncertainty** (`summarize`, `compare`): a clustered bootstrap over anchor paintings, 5,000 resamples, seed 42,
  95% percentile intervals. `compare(A, B)` bootstraps the per-anchor difference A − B. Points are in percentage
  points.

## 5. The pick (Task 12, seed-42 selection episodes)

- Among A1 to A6, the **picked run** maximizes 0.5 × (R@1 point + condition gain point) of its cross-fitted scores on
  the pooled seed-42 episodes.
- Ties go to the lower run id (A1 before A2, and so on).
- H1 and S1 appear in the table, marked not eligible.
- `--select` writes `results/select_seed42.json` and `results/picked.json`, which records the picked checkpoint's
  SHA-256.
- The pick is frozen before any seed-43 number of a trained run is computed. `--gonogo` refuses a picked checkpoint
  whose SHA-256 differs from `picked.json`, and refuses to overwrite an existing `results/gonogo.json`.

## 6. The GO rule (Task 13, seed-43 episodes)

- **GO baseline:** the first entry of `go_bar_ranking` in Task 9's `baselines_seed42.json`. That is the top scorer
  among diag, diag_relu, bilinear, kissme, rca, xing, wang, probe and tip by 0.5 × (R@1 point + condition gain point)
  on the seed-42 episodes (value_prototype is excluded because it is not a metric-from-pairs baseline). The script
  re-checks that entry against the per-scorer points in the same file. The baseline's seed-43 per-anchor arrays are
  read from Task 9's `per_anchor_seed43.npz` (cross-fitted on the seed-43 episodes there).
- **Comparators:** (1) backbone-only `cosine`; (2) the GO baseline; (3) the picked run's uniform-weight control.
- **beats(c)** iff `compare(picked, c, clusters, "r1")["ci95"][0] > 0` and
  `compare(picked, c, clusters, "gain")["ci95"][0] > 0`.
- **GO** iff beats(c) holds for all three comparators.
- **Strong GO** iff GO holds, the R@1 point of `compare(picked, cosine)` is at least 4.0, and the condition gain point
  of `compare(picked, cosine)` is at least 4.0. Cosine's condition gain is exactly 0 (asserted), so the second
  condition is the picked run's own gain point.
- **Multiplicity:** these are unadjusted 95% intervals, as §6 states for the go/no-go. The Holm–Bonferroni family of
  §10 applies to the final held reads, not to this development decision.

## 7. The K8 rule (held-out aspect, Task 13)

- H1 is A1's settings trained on the AI bank (affect and image partitions; no caption partition).
- H1 and H1's uniform control are cross-fitted on the pooled seed-43 episodes (all three pairs, as in §4).
- The test restricts both per-anchor arrays to the episodes whose pair includes genre (emotion×genre and style×genre,
  8,192 episodes), with their anchor-painting clusters.
- **K8 holds** iff `compare(H1, H1_uniform, clusters, "gain")["ci95"][0] > 0` on that subset. Otherwise C2 narrows to
  "selection among aspects represented in training", and the paper says so.
- K8 is its own pre-declared family (§10): one test, no adjustment.

## 8. Descriptive rows (they decide nothing)

- **Supervision ablation:** S1 (IC bank, no affect partition) against the picked run on the emotion-containing pairs
  (emotion×style and emotion×genre), `compare` on R@1 and condition gain with CIs. SE, C0 and R3 rows on the same
  subset and on all pairs come from Task 9's seed-43 per-anchor arrays.
- **Value-sharing diagnostic** (selection rows; evaluation labels are used only here):
  - aspects: emotion (without "something else"), style and genre; values with at least 30 selection paintings
    (the episodes' eligibility rule);
  - models: the picked run, H1, S1, SE, C0 and R3; modalities: image codes and caption codes;
  - m[v, f] is the mean code of factor f over the selection rows labelled v, and M_f = max over v of m[v, f];
  - factor f is *shared* iff M_f > 0 and m[v, f] ≥ 0.1·M_f for at least half of the values (count × 2 ≥ number of
    values);
  - reported: the share of shared factors among all L factors, and the number of dead factors (M_f = 0).

  The agreement rule needs the values of an aspect to share factors (Task 4's one-hot test), so this diagnostic
  explains a GO or a NO-GO.

## 9. The early MLLM probe rule (Task 14)

- Qwen3-VL-2B-Instruct as an in-context reranker, 300 selection episodes per aspect pair at episode seed 44
  (900 pooled).
- The MLLM **works** iff `compare(mllm, cosine, clusters, "r1")["ci95"][0] > 0` and
  `compare(mllm, cosine, clusters, "gain")["ci95"][0] > 0`, with the same clustered bootstrap (anchor painting,
  5,000 resamples, seed 42).

## 10. Decision map (spec §4, copied verbatim)

**Decision branches on Oct 9** (EIC review W6). Each branch has its claim table fixed now.

| Branch | Condition | Paper | Claims carried | Venue |
|---|---|---|---|---|
| **1, GO** | §6 GO rule met | method paper | C1, C2, C3; K1 to K8 | CVPR |
| **2, NO-GO, MLLM works** | GO missed, and the in-context MLLM probe (§6) has a condition gain CI lower bound > 0 and an R@1 gain over backbone-only with CI lower bound > 0 | benchmark paper: a solvable, released task that embedders and metric-from-pairs fail while an MLLM given the examples partly succeeds | C1, C3; K1; K3 (with the MLLM as the example scorer); embedder and baseline tables | CVPR |
| **3, NO-GO, nothing works** | neither | an analysis and negative-results paper (the task, the value-episode shortcut, modality asymmetry across annotation protocols) | C1, C3, K1, K3 | a workshop or a datasets-and-benchmarks track, chosen with the user |

The user decides the switch on Oct 9 from the pre-registered numbers.

Applied mechanically: GO gives branch 1; GO missed and the MLLM works gives branch 2; neither gives branch 3.

## 11. Notes

- The `probe` GO-bar candidate is the standardized L2 logistic pair probe (controller ruling during Task 8,
  documented in the E1 report), not the plan's literal pair probe, which duplicated the diag rule.
- Smoke runs (`--smoke`: 50 training steps on Task 10's smoke bank, Task 9's smoke episodes, the A1 smoke
  checkpoint standing in for every run) write only under `checkpoints/smoke/` and `results/smoke/`. A real `--select`
  or `--gonogo` never reads them.
- The test episodes come from the same selection rows as the selection episodes (fresh seed only), so the GO test is
  not independent of development (disclosed, spec §6).

## Addenda

None.
