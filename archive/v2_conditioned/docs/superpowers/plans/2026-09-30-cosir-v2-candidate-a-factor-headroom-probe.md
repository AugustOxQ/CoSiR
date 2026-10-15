# CoSiR v2 Candidate A: factor headroom probe (spike before factor learning)

**Status:** user-approved as a spike (2026-09-30, "Probe first"). Output is an answer, not kept model code.
**Why:** stage (d) showed the cross-validated label oracle on frozen R3 factors ≈ the naive rule
(selection rows, β 0.3: 18.41 / 21.04 vs 18.36 / 20.36 R@1). That says R3's 32 factors hold no more
emotion/style information than naive already uses. It does **not** say whether the limit is R3's code or
frozen CLIP ViT-B/32 itself. R3's encoder is one linear layer + ReLU on CLIP, so every factor-learning
design in the handoff (`docs/superpowers/handoffs/2026-09-30-candidate-a-factor-learning-handoff.md` §4)
re-mixes the same CLIP features. This probe measures the headroom before any design is chosen.

**Question:** on the stage-(d) selection label episodes, how far above naive-on-R3 can the label oracle get
when the code is (a) raw CLIP, (b) CLIP PCA, (c) a code that linearly encodes the evaluation labels
themselves (a diagnostic ceiling)? And which self-generated condition sources are aligned with the labels?

**Hard rules**
- Selection rows only for evaluation; scorer-train rows only for anything fitted (PCA, probes, k-means,
  scale factors). **Held and val rows are never read** (NaN-masked, as in stage d).
- Human labels are used only in (c) and in the alignment diagnostic, both diagnostics like the existing
  label oracle. No method trains on them.
- No `src/` change. If one seems needed, stop and report to the controller.
- CPU by default (`--device cpu`; the GPU is currently down). Seed 42 everywhere.
- Commit locally on `main`, only this task's files (another session has uncommitted weekly-report files
  in the tree: never stage them). Never push.

## Task 1: `src/test/20261015_factor_headroom_probe/run_probe.py` + CPU smoke test

**Files:** `run_probe.py`, `20261015_factor_headroom_probe_log.md`, `.gitignore` (copy
`src/test/20261013_stage_d_selection/.gitignore`: `*.npy *.npz *.json *.pt *.log cache/ checkpoints/ results/`).

**Reuse, don't copy.** Import stage (d)'s modules the way `src/test/20261013_stage_d_selection/run_posthoc.py`
does (importlib on `src/test/20261014_stage_d_final/run_final.py`, then `sel = fin.sel`):
`sel.load_prepared()` (cache: `groups`, `split_train`, `scorer_train`, `selection`, R3 `img_codes`/`txt_codes`,
`clip_image`, `clip_caption`, `community`), `sel.masked`, `sel.pool`, `sel.LABELS/SCOPES/DIRECTIONS`,
`fin.SELECTION_SHA256`. From `src/`: `load_artelingo`, `standard_label_episodes`, `label_episodes_sha256`,
`label_oracle_ranks`, `paired_bootstrap`, `conditional_score`, `label_episode_weights`, `tie_aware_rank`.

**Modes:** `--run` (full), `--smoke` (same code path, 128 episodes per label, 20 oracle steps, probe/PCA/k-means
fits on a 20,000-row scorer-train subsample; writes `results/smoke_*`, numbers discarded, prints per-phase
timings), `--tables` (reprint all tables from `results/probe_results.json`), `--device {cpu,cuda}`.

### Step 1: data, rows, episodes
- Load `load_artelingo()` and the stage-(d) cache. Assert `scorer_train` ∩ `selection` = ∅ and both ⊂ `split_train`.
- Episodes: `standard_label_episodes(data, cache["groups"], cache["selection"], label, 2048, seed=42)` per label
  in `("emotion", "art_style")`; assert `label_episodes_sha256(eps) == fin.SELECTION_SHA256[label]` (full run only)
  and that every episode row is a selection row.
- Random-target null targets: exactly as stage d, `rng = np.random.default_rng(42)`, then for each label in
  `LABELS` order `rng.integers(1, 13, n_episodes)`.

### Step 2: codes (each an `(img_code, txt_code)` pair over all rows, NaN outside `split_train`)
Fit everything on scorer-train rows only; assert the fit index set equals (or, in smoke, is a subset of)
`cache["scorer_train"]`.

| Name | Construction |
|---|---|
| `R3` | cache `img_codes`, `txt_codes` (unchanged). |
| `clip512` | CLIP features L2-normalized per row, image → img code, text → txt code (signed). |
| `pca32`, `pca128` | Per-modality centering with scorer-train means; one shared PCA basis fitted on the stacked centered scorer-train image and text features (`sklearn.decomposition.PCA`, `svd_solver="randomized"`, `random_state=42`); project each modality (signed). |
| `labelprobe` | **Diagnostic ceiling.** Four multinomial logistic regressions on scorer-train rows (standardize inputs with scorer-train mean/std; `sklearn.linear_model.LogisticRegression(C=1.0, max_iter=500)`; run the four fits in parallel processes if slow): image→art style, text→art style, image→emotion, text→emotion (all 9 emotions, incl. "something else"). img code = `[P_style(img), P_emotion(img)]`, txt code = `[P_style(txt), P_emotion(txt)]` (non-negative, ≈ S+9 dims). Also record each probe's top-1 accuracy on selection rows and the majority-class accuracy. |

**Scale.** β mixes a CLIP cosine with the code term, and the code term's size depends on raw code scale
(see the `conditional_score` docstring). Rescale every non-R3 code by one global scalar so its mean per-dimension
RMS over scorer-train rows (image and text codes pooled) equals R3's. Record the scalars. Rankings at β = 0 are
unaffected.

### Step 3: scorers on the selection episodes (both labels, both directions)
Evaluate with `sel.masked`-style inputs: CLIP features and codes NaN outside selection rows.
- **CLIP-only:** zero weights, β 0.3 (once).
- Per code, for β ∈ {0, 0.03, 0.1, 0.3, 1.0}:
  - **naive:** `label_episode_weights(img_code, txt_code, eps)` weights, ranks via `conditional_score` +
    `tie_aware_rank` (equivalently `label_episode_recall`);
  - **label oracle:** `label_oracle_ranks(..., beta=β, folds=2, steps=200, lr=0.1, seed=42)`.
- Per code: **oracle null** (`target_column` = the Step 1 null targets) at β = 0 and at the code's best oracle β.
- "Best β" for a scorer = the grid β with the highest pooled R@1 averaged over directions. This is one scalar
  picked in-sample from 5; disclose it in the report.

**Sanity checks (assert, full run):**
1. `R3` naive at β 0.3 and `R3` oracle at β 0.3 reproduce stage (d): compare ranks with
   `src/test/20261013_stage_d_selection/results/posthoc_ranks.npz` if the keys exist, else pooled R@1 with
   `posthoc_results.json` (naive 18.36 / 20.36; oracle 18.41 / 21.04) to 2 decimals.
2. `clip512` with uniform weights at β 0 ranks identically to CLIP-only.
3. Every oracle null's pooled R@1 is within 2 points of chance (7.69).

### Step 4: comparisons (paired bootstrap, 5,000 resamples, seed 42, per episode, R@1 points)
For every code, per scope (`pooled`, `emotion`, `art_style`), per direction and mean of directions:
- oracle(code, best β) − oracle(R3, best β);
- oracle(code, best β) − naive(R3, β 0.3);
- naive(code, best β) − naive(R3, β 0.3).

### Step 5: condition-source alignment (scorer-train rows; labels only measured)
`sklearn.metrics.adjusted_mutual_info_score` between each label (emotion; art style) and each partition:
cache `clip_image`, cache `clip_caption`, cache `community`; a new **caption-residual** k-means (per row:
text feature − mean text feature of that painting's scorer-train rows (`cache["groups"]`), paintings with ≥ 2
scorer-train rows only, L2-normalized, `MiniBatchKMeans(64, random_state=42, n_init=3, batch_size=4096)` as
in `ClipClusterSource`); and a uniform random 64-way partition (null). Restrict each AMI to rows where the
partition label is ≥ 0. Report n rows per cell.

### Step 6: outputs, smoke test, commit
- `results/probe_results.json` (all R@1, CIs, best β, scale scalars, probe accuracies, AMIs, timings, sanity
  results) and `results/probe_ranks.npz`; `--tables` prints every table.
- Run `--smoke` on CPU, confirm it completes, and record per-phase timings and an estimate of the full `--run`
  time on CPU in the log. Do **not** run `--run` in this task.
- Log file per project convention (problem, steps, what was verified). Commit the three files with
  message `feat(v2): factor headroom probe script (label oracle on CLIP/PCA/label-probe codes; source AMI)` and
  the session's trailers.

## Task 2 (controller, after Task 1 review): full run
`/root/miniconda3/envs/CoSiR/bin/python src/test/20261015_factor_headroom_probe/run_probe.py --run --device cpu`
(or `cuda` after the container reboot). Numbers are selection-row, post-hoc diagnostics.

## Task 3 (controller): report and reading
`docs/reports/auto/v2/2026-10-15_candidate_a_factor_headroom_probe.md`, a figure in
`docs/reports/assets/2026-10-15_factor_headroom_probe/` (oracle and naive R@1 per code and label type, with
R3-naive, CLIP-only and chance lines), a row in `docs/reports/reports_sum.md`, then
`scripts/check_reports_sum.py` must print OK. Baseline for every number: naive on R3 at β 0.3.

**Reading rule (stated before the run), per label type, mean of directions:**
- `labelprobe` oracle − `R3` oracle ≥ **+5** R@1 points with CI lower bound > 0 → **headroom**: linear codes on
  frozen CLIP can carry much more of this label; factor learning is worth designing, and the question becomes
  which self-generated signal gets close (Step 5 informs the source).
- < **+2** points, or CI including 0 → **no headroom**: frozen CLIP-B/32 is the limit for this label; factor
  learning is the wrong lever and the next decision is about features/backbone (user decision).
- Between → **modest**; report and let the user decide.
- `clip512` / `pca128` vs `R3` oracle says whether more unsupervised dimensions alone help (handoff option iii).
