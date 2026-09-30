# CoSiR v2 Candidate A: condition-aware factor learning, first experiment (design)

**Status:** design approved section by section in conversation (2026-09-30); awaiting review of this written
spec. **Parent spec:** `docs/superpowers/specs/2026-09-28-cosir-v2-ground-up-redesign.md` (Candidate A §1-4).
**Follows:** stage (d) (`docs/superpowers/specs/2026-09-30-cosir-v2-candidate-a-stage-d-design.md`, §9 gate) and
the headroom probe (`docs/reports/auto/v2/2026-10-15_candidate_a_factor_headroom_probe.md`). **Handoff:**
`docs/superpowers/handoffs/2026-09-30-candidate-a-factor-learning-handoff.md`.

## 1. Intent

Learn factors that carry more of the conditions people ask for, and show it with the zero-parameter naive rule
on human-label episodes.

**Why this experiment.** Stage (d) found that no reweighting of R3's frozen factors beats the naive rule on
emotion and art-style episodes: the cross-validated label oracle scored about the same as naive. The headroom
probe then showed the limit is R3's code, not frozen CLIP:
- A 36-dimensional code that linearly reads the labels from the same CLIP features reaches a label-oracle
  R@1 of 49.8%, against 20.5% for R3 and 19.4% for the baseline (naive on R3, β 0.3).
- Style is read from the image (linear probe 59.8%, caption 26.0%); emotion from the caption (57.9%, image
  35.2%).
- CLIP image clusters line up with art style (AMI 0.32); no self-generated source lines up with emotion
  (≤ 0.06).
- A post-probe check found that within-painting caption variation carries emotion (probe 47.0%, majority
  28.4%) but not along its top-32 directions (36.8%).

So this first experiment tests two things with honest expectations:
1. **The agreement hypothesis:** R3's per-pair agreement pulls every caption code toward its painting's single
   image code, which may keep one-sided information (emotion, which the image barely shows) out of the factors.
2. **The style signal:** training the factors on CLIP-image-cluster conditions, the one label-free signal known
   to line up with a human condition.

Style gains are expected; emotion gains are not, and a guard (§6) ensures emotion does not get worse.

## 2. Settled decisions (user-approved, 2026-09-30)

| Decision | Choice | Reason |
|---|---|---|
| First target | **2×2: agreement {per pair, per painting} × conditions {none, CLIP image k-means}** | Tests the agreement hypothesis and the style signal and separates them. |
| Signal form | **Naive-rule episode loss**, β frozen at 0.3, random negatives only | Trains the factors for what evaluation measures; no trainable interface, so no β or interface confound (stage (d)'s lessons). |
| Condition source | **CLIP image-view k-means (64)**, stage (d)'s cached labels | Strongest measured style alignment (AMI 0.32); external to the factor space, so not circular. |
| Testing | **Selection on selection rows, then one held test on fresh held episodes (seed 43)** | Selection rows are fully out-of-sample for the new factors; held rows were read twice before, so new episodes plus disclosure. |
| Primary criterion | **Naive R@1 at β 0.3, pooled over labels, mean of directions, vs the matched control; emotion guard** | Measures what the system delivers; averaging directions avoids stage (d)'s power trap. |

## 3. Data and rows

- ArtELingo via `load_artelingo`; the painting-grouped split (seed 42): 216,107 train / 30,872 val / 61,744
  held rows.
- Stage (d)'s sub-split of train (cache `src/test/20261013_stage_d_selection/cache/prepare.npz`):
  **scorer-train** 183,694 rows (36,518 paintings) and **selection** 32,413 rows (6,451 paintings). A
  "painting" is a leakage group (`groups` in the cache); images are identical within a painting (99.8% of
  same-painting row pairs), with about 5 captions each.
- **All four factor models train on scorer-train rows only.** The content graph (`build_content_graph`,
  `GraphConfig()`), the condition partition and every fitted statistic come from scorer-train rows.
- **Val is not used. Held is read only by the held test (§7).**

## 4. The four cells

All cells start from `R3_CONFIG` (32 factors, ReLU, reconstruction 1.0, graph 1.0, sparsity 0.01, anti-split
0.1, usage balance 0.1, decorrelation 1.0, Adam 1e-3, 2,000 steps, seed 42) and share one batch sampler.

| Cell | Agreement | Condition loss | Role |
|---|---|---|---|
| **C0** | per pair (R3's InfoNCE, leakage-group masked) | none | **matched control** |
| **A** | per painting | none | agreement hypothesis |
| **S** | per pair | CLIP image k-means episodes | style signal |
| **AS** | per painting | CLIP image k-means episodes | both |

**Shared batch sampler.** Each step samples 1,024 content-graph edges as R3 does, takes their unique rows,
then **adds every scorer-train row of each sampled painting** (about 10k rows per step). Every loss term sees
the expanded batch; the graph term uses the graph induced on it. Painting-level agreement needs complete
paintings, and a shared sampler keeps the cells comparable. Original R3 (trained on all train rows with the
original sampler) is reported as a reference row, never as the criterion's control.

## 5. Losses

**Per-pair agreement (C0, S).** R3's `cross_modal_infonce_loss` on rows, temperature 0.1, leakage-group ids
masking same-painting negatives. Unchanged.

**Per-painting agreement (A, AS).** Replaces the per-pair term. For each painting `p` in the batch: the image
code `u_p` (its rows share one image; take the mean of their image codes) and the mean caption code
`v_p = mean over p's rows of a_T(caption)`. Symmetric InfoNCE between the L2-normalized `{u_p}` and `{v_p}`
over the batch's paintings, temperature 0.1, weight 1.0. A caption's code may then differ from the image code
as long as the painting's mean matches.

**Condition episode loss (S, AS).** Weight `λ_condition = 1.0` (fixed, no tuning).
- Source: stage (d)'s cached CLIP image-view k-means labels (64 clusters, scorer-train rows) wrapped as a
  single-view partition source (`CommunitySource(cache["clip_image"], scorer_train)`); min group 200 rows.
- Each step mines 64 episodes with `mine_condition_episodes` (`episodes_per_condition=4`, its default): anchor,
  4 supports, 4 positives, 4 contrasts,
  **12 random negatives and no hard negatives** (`num_hard=0, num_random=12`). No painting repeats within an
  episode (leakage-group keys). Mining RNG seeded from the cell seed.
- The episode rows are encoded by the current encoders. Weights `w` = the naive rule on pair codes
  (`ReLU(mean_S − mean_C)`, L1-normalized; an all-zero row stays zero), differentiable.
- Score `s = 0.3·cos(CLIP_q, CLIP_c) + Σ_l w_l q_l c_l` (`conditional_score`, **β frozen at 0.3**), over the 16
  candidates (4 positives + 12 negatives), in both directions (image anchor vs captions; caption anchor vs
  images).
- Loss: multi-positive softmax `−log(Σ_pos e^{s/τ} / Σ_all e^{s/τ})`, averaged over episodes and directions.
  `τ` is learnable (log-parametrized, initialized to the standard deviation of the step-0 scores on the first
  batch); it does not change rankings.

**Known subtlety.** The factor codes' scale is learned, so the code term can outgrow `0.3·cos`, which acts
like lowering β. Training and evaluation use the same β and the same kind of negatives, so nothing rewards
ignoring CLIP the way stage (d)'s CLIP-hard negatives did. Every cell is still reported on a β grid (§6).

**Collapse guards.** R3's losses stay on in every cell. Every cell is checked with `evaluate_factor_gates`:
- `thresholds=AMENDED_2026_09_29_THRESHOLDS`;
- fit rows = scorer-train, eval rows = selection;
- `readout_reference` = the readout of the R0 checkpoint
  (`src/test/20261011_factor_repair_grid/checkpoints/R0_seed42.pt`) measured by the same function on the same
  rows;
- community spanning on stage (d)'s cached Block 1 communities (scorer-train rows).

## 6. Selection (selection rows) and stop points

**Episodes.** Stage (d)'s selection label episodes: `standard_label_episodes(data, groups, selection, label,
2048, seed=42)` for emotion and art style, SHA-256s asserted equal to `run_final.SELECTION_SHA256`.

**Primary score** of a cell `X`: `D(X) = R@1_naive(X) − R@1_naive(C0)` in R@1 points, where R@1 is the naive
rule at β 0.3 (`label_episode_weights` + `label_episode_recall`), per episode averaged over i2t and t2i, pooled
over both labels (4,096 episodes). Uncertainty: paired bootstrap over episodes, 5,000 resamples, seed 42.

**Guard:** `D_emotion(X)`, the same difference on the 2,048 emotion episodes only.

**Rule.**
1. **Eligible:** all 9 gates pass (§5).
2. **Qualifies:** eligible, `D(X)` 95% CI lower bound > 0, and `D_emotion(X)` lower bound > −1.0.
3. **Pick:** the qualifying cell with the highest `D`. Cells within 0.5 points of it tie; a tie goes to the
   cell with fewer changes (A or S before AS), then to the higher `D`.

**Stop points.**
- **C0 fails any gate:** the setup is broken. Stop and report.
- **No cell qualifies:** stop, report, and do not run the held test.

**Replication.** The picked cell and C0 are retrained with seeds 43 and 44 (sampler, mining and init). Their
`D` and gates are reported. The verdict rests on seed 42.

**Reported, not gating** (for every cell, C0 and original R3):
- the label oracle (`label_oracle_ranks`, 2 folds, 200 steps) at β 0.3 and β 0, with its random-target null;
- per label and per direction numbers;
- naive R@1 on the β grid {0, 0.03, 0.1, 0.3, 1};
- CLIP-only;
- all gate values;
- the condition loss and the learned τ over training.

## 7. Held test (pre-registered; run once)

**Episodes.** New held label episodes: `standard_label_episodes(data, groups, held, label, n, seed=43)` per
label. `n` is fixed before the run from the selection result:
- `SE_sel` = the half-width of the picked cell's `D` CI / 1.96;
- the assumed held effect `e = 0.75 × D_sel` (shrunk for selection optimism);
- for `n` in (2048, 4096, 8192) per label: `SE_n = SE_sel × sqrt(2048 / n)`, power `= Φ(e / SE_n − 1.96)`;
- the smallest `n` with power ≥ 0.8; if none, `n = 8192` and the power is stated.

**Criterion.** The picked cell vs C0, both seed 42, on the same held episodes:
- **Primary met iff** the 95% CI of `D_held` (pooled, mean of directions) has lower bound > 0;
- **Guard met iff** the lower bound of `D_emotion,held` > −1.0.
- The verdict is "confirmed" only if both hold.

**Reported alongside:** the label oracle and its null, per label and per direction, the replication seeds'
`D_held`, original R3 naive, and CLIP-only.

**Row scope.** Factors are trained on scorer-train rows and only encode held rows. The held phase refuses to
run twice. A smoke run of the full held code path uses selection rows first, with its numbers discarded.

**Disclosure.** Held rows were read in two earlier final tests (the repair plan's Task 7 and stage (d)'s Task
7), both with seed-42 episodes. These episodes are new, but the rows and paintings are not.

## 8. Constraints

- **Human labels are evaluation-only.** No external condition taxonomy or pretrained affect signal is used.
- **Rows:** training, graphs and partitions use scorer-train rows; selection rows for choosing; held rows only in
  §7.
- **No tuning:** every weight, step count and threshold above is fixed before the runs.
- **Environment:** seed 42 unless stated; no `cuml`/`cugraph`; conda env `CoSiR`; `/project/CoSiR`, `main`;
  commit locally, never push.
- **Code:** function/class-formal code in `src/` with unit tests (TDD); new config fields default off, so
  `FactorTrainingConfig()` still reproduces R0 and `R3_CONFIG` still reproduces R3 (golden tests stay green).
  Real runs in dated `src/test/yyyymmdd_<name>/` folders with a log and a local `.gitignore`. Each modified `src/`
  file gets a `.claude/yyyymmdd_log.md` entry.
- **Reports:** `docs/reports/auto/v2/2026-10-16_candidate_a_factor_learning_selection.md` and
  `2026-10-17_candidate_a_factor_learning_held.md`, verdict first, every
  number against a named baseline (C0 for criteria; naive on R3 as the current system), figures where they help,
  a row in `docs/reports/reports_sum.md`, and `scripts/check_reports_sum.py` OK.
- **Execution:** SDD with Claude Code subagents as implementers (model sized per task), per-task reviewers, an
  Opus whole-branch final review, then one fix wave. Overnight automation is allowed and stops only at the stop
  points (§6) or before destructive actions. Another session may be committing in this repo: stage only the
  task's own files.

## 9. Code structure

| Module | Change |
|---|---|
| `src/train/factors.py` | Add `painting_infonce_loss(img_codes, txt_codes, painting_ids, temperature)`. |
| `src/train/factor_condition_loss.py` (new) | `naive_episode_loss(...)`: naive weights from support/contrast pair codes, `conditional_score` at a fixed β, multi-positive softmax over both directions, learnable log-τ. Pure function of codes and CLIP features of the episode rows. |
| `src/train/train_factors.py` | `FactorTrainingConfig` gains `agreement_level` (`"pair"` default, `"painting"`), `painting_batches` (default False), `lambda_condition` (default 0.0), `condition_episodes_per_step` (64), `condition_beta` (0.3). `train_factors` gains an optional condition source (plus keys for painting distinctness); painting agreement requires `painting_batches=True` and integer `group_ids`. Checkpoints store and restore the new fields; old checkpoints (R0, R3) still load. |
| `src/test/20261016_factor_learning_grid/run_grid.py` | `--prepare` (reuse stage (d)'s cache; build the scorer-train content graph; cache it), `--run CELL [--seed]`, `--evaluate` (gates, selection metrics, rule), `--tables`. |
| `src/test/20261017_factor_learning_held/run_held.py` | `--power` (n from the selection result), `--smoke` (selection rows), `--run` (once), `--tables`. |

**Key tests.**
- `painting_infonce_loss`: equals the per-row InfoNCE when every painting has one row; invariant to caption
  order within a painting; unaffected by within-painting caption spread when the painting means are fixed.
- `naive_episode_loss`: hand-computed value on a tiny episode; gradients reach the codes through both the
  weights and the candidate scores; an all-zero weight row falls back to the CLIP term without NaN.
- The expanded sampler contains every scorer-train row of each sampled painting and nothing outside the
  training rows.
- `mine_condition_episodes(num_hard=0)` returns 12 random negatives outside the group with no repeated
  painting.
- Defaults unchanged: `FactorTrainingConfig()` and `R3_CONFIG` golden tests pass; R0/R3 checkpoints load.
- A synthetic task where a condition is carried by a low-variance direction: the condition loss raises naive
  R@1 over training while the no-condition run does not.

**Build order (the plan's tasks).**
1. The two losses and config fields (TDD).
2. `train_factors` integration: sampler, painting agreement, condition loss, checkpoints.
3. Grid prepare, the four cells, gates and selection evaluation, and the selection report (**stop points**).
4. Replication seeds 43/44 for the picked cell and C0.
5. Power, held smoke, the held test, and the held report.
6. Final review and one fix wave.

## 10. Out of scope

- A label-free emotion signal (the open problem this experiment does not solve), and any external affect
  signal (a parent-spec decision for the user).
- A trained condition interface, learned β, or CLIP-hard negatives.
- Capacity changes (more factors, TopK) and whitening.
- Hyperparameter sweeps, including `λ_condition` and temperatures.
- Stage (e), the human-judged evaluation set; the free-text condition path; Candidate B.

## 11. Risks and caveats to carry into the reports

- **The control is not R3.** C0 uses the expanded sampler and scorer-train rows; original R3 is shown beside it.
  A large C0-vs-R3 gap would itself be a finding.
- **Selection rows have been read before** (stage (d) selection and post-hoc, the headroom probe). The design
  choices of this spec used those readings; the held test exists to confirm on new episodes.
- **Style, not emotion.** The only condition signal is style-aligned; emotion gains are not expected. The guard
  only prevents a loss.
- **Content vs style.** CLIP image clusters also follow content (AMI with style 0.32 is far from 1), so factors
  may learn content clusters and gain little on style episodes.
- **Code scale acts like β** (§5); the β grid shows it.
- **Emotion labels are per annotation**; the image carries little of them (image→emotion 35.2%, majority 28.4%).
- **One split, one held set.** Bootstrap CIs resample episodes under fixed models; seeds 43/44 cover training
  randomness only partly. Large-n held episodes reuse rows, so bootstrap CIs are somewhat optimistic.

## Glossary

- **Naive rule:** condition weights = ReLU(mean support pair code − mean contrast pair code), L1-normalized.
- **Pair code:** `0.5·(image code + caption code)` of one row.
- **Label oracle:** one weight vector per human label, fitted on half of that label's episodes and ranking the
  other half (cross-validated); measures how much label information a code carries.
- **Painting (leakage group):** all rows sharing one image; the unit of splitting and of painting-level
  agreement.
- **C0:** the matched control, R3's recipe refit on scorer-train rows with the shared sampler.
