# 2026-10-15 — factor headroom probe (plan Task 1: script + CPU smoke test)

Plan: `docs/superpowers/plans/2026-09-30-cosir-v2-candidate-a-factor-headroom-probe.md`, Task 1.
Script: `src/test/20261015_factor_headroom_probe/run_probe.py`. The full `--run` is Task 2 (controller) and has
**not** been run here.

## Problem

Stage (d) showed that the cross-validated label oracle on the frozen R3 factors roughly equals the naive rule
(selection rows, β 0.3: 18.41 / 21.04 vs 18.36 / 20.36 R@1). That does not tell us whether the limit is R3's code
or frozen CLIP ViT-B/32. The probe reruns naive and the label oracle on the stage-(d) selection label episodes
with other codes (`clip512`, `pca32`, `pca128`, and `labelprobe`, a diagnostic ceiling), then measures how
well self-generated partitions align with the labels (AMI).

## What the script does

- **Reuse:** it imports `run_final.py` by file path and takes `sel = fin.sel`, as `run_posthoc.py` does. It
  reuses `sel.load_prepared`, `sel.masked`, `sel.pool`, `sel.LABELS/SCOPES/DIRECTIONS`, `sel.log` and
  `fin.SELECTION_SHA256`. From `src/`: `load_artelingo`, `standard_label_episodes`, `label_episodes_sha256`,
  `label_episode_weights`, `label_episode_recall` (naive ranks, the plan's "equivalently"), `label_oracle_ranks`
  and `paired_bootstrap`. It makes no change under `src/`.
- **Codes:** `R3` is the cache codes. `clip512` is L2-normalized CLIP. `pca32`/`pca128` come from separate
  `PCA(k, svd_solver="randomized", random_state=42)` fits on the stacked, per-modality-centered fit rows, with
  each modality projected on the shared basis. `labelprobe` is four sklearn `LogisticRegression(C=1.0,
  max_iter=500, random_state=42)` probes (lbfgs, multinomial) on standardized inputs, run as 4 parallel joblib
  (loky) processes: img code = `[P_style(img), P_emotion(img)]`, txt code likewise, 27 + 9 = 36 dims.
  Every non-R3 code is multiplied by one scalar so its mean per-dimension RMS over the fit rows (image and text
  pooled) equals R3's.
- **Scorers:** CLIP-only is zero weights at β 0.3. Naive and the label oracle (`folds=2, steps=200, lr=0.1,
  seed=42`) run at β ∈ {0, 0.03, 0.1, 0.3, 1}. The oracle null (stage-(d) null targets) runs at β 0 and at the
  code's best oracle β. The best β is the grid β with the highest pooled R@1 averaged over directions, chosen
  in-sample; on a tie the first β in grid order wins.
- **Comparisons:** paired bootstrap (5,000 resamples, seed 42) per scope and direction, plus the mean of
  directions.
- **AMI:** `clip_image`, `clip_caption` and `community` from the cache; a caption-residual k-means; a uniform
  random 64-way null. All are computed on fit rows that have a partition label.
- **Outputs:** `results/probe_results.json` and `results/probe_ranks.npz`. Smoke writes
  `results/smoke_probe_*`. `--tables` reprints every table.

## Row-scope guards

- Val and held CLIP features are replaced by NaN right after `load_artelingo()`, via `dataclasses.replace` with
  `sel.masked(..., split_train)`. Every code is therefore computed only from train-part rows. The script asserts
  that each code (and the cached R3 codes) is finite on `split_train` and NaN everywhere else.
- `scorer_train ∩ selection = ∅` and both sets lie inside `split_train` (asserted). The cached partitions must be
  labelled exactly on scorer-train rows (asserted).
- PCA, the probes (including their standardization), the caption-residual k-means and the scale scalars are
  fitted on the fit rows only. The fit rows must equal `scorer_train` in the full run and be a subset of it in
  smoke (asserted).
- Evaluation inputs (CLIP features and every code) are NaN outside the selection rows, and every episode row is
  asserted to be a selection row. AMI rows are asserted to be scorer-train rows.
- The episode builder is stage (d)'s `standard_label_episodes`, unchanged. As in stage (d), it reads the full
  label array to exclude target paintings from negatives. It reads labels only, never features.

## Smoke test (CPU, `--smoke --device cpu`, torch 16 threads)

- Completed in **55 s**. Per-phase timings (s): load 2.3, episodes 0.4, codes (clip512 0.5, pca32 1.9, pca128
  1.7, labelprobe 38.6), evaluation (all five codes, naive + oracle + nulls) ≈ 4.4, comparisons 1.8,
  caption-residual k-means 1.2, AMI 0.2.
- **Sanity 2** (clip512 with uniform weights at β 0 ranks like CLIP-only; asserted in both modes): identical-rank
  share 1.0000. Passed.
- **Sanity 3** (oracle nulls within 2 points of chance; reported only in smoke): 4 of 10 cells fell outside,
  with values from 5.3 to 10.2. Each cell has 512 pooled episodes and 20 steps, so SE ≈ 1.2 points and 2 points
  is ≈ 1.7 SE; misses are expected at this size. In the full run, 4,096 pooled episodes give SE ≈ 0.4. Stage
  (d)'s R3 null at β 0.3 was 7.69 / 7.62.
- **Sanity 1** (reported only in smoke): the smoke episodes are the first 256 of stage (d)'s 2,048 per label
  (`build_label_episodes` draws sequentially; checked). On that prefix, the R3 naive β 0.3 ranks and the
  CLIP-only ranks equal the stored `selection_ranks.npz` ranks exactly (share 1.0000). The smoke oracle cannot be
  compared, because it uses 20 steps and a different fold split.
- **Full-size pre-check** (scratch timing script, not `--run`): 2,048 episodes per label reproduce
  `fin.SELECTION_SHA256` for both labels. The R3 label oracle at β 0.3 on CPU gives **ranks identical to the
  stored `posthoc_ranks.npz`** (share 1.0 in every label and direction; pooled 18.41 / 21.04). Stage (d)'s
  post-hoc also ran the oracle on CPU (`device=None`).
- Smoke R@1 numbers are discarded.

## Estimated full `--run` on CPU: about 20–30 min

- **Label oracle: about 15 min, the largest phase.** Measured per β for both labels at full size: 24 s (R3,
  32-d), 22 s (128-d), 34 s (512-d). Each code runs 5 β plus 1–2 nulls, so R3, pca32 and labelprobe each take
  about 170 s, pca128 about 155 s and clip512 about 240 s.
- **Probe fits: about 4–10 min.** One full-size image→style fit alone took 166 s with 32 BLAS threads, stopping
  at 500 lbfgs iterations without converging. The four fits run in parallel. Expect `ConvergenceWarning`s at
  `max_iter=500`; `n_iter` and `converged` are recorded for each probe.
- **Everything else: about 2 min.** PCA ~0.5 min, bootstraps ~1–2 min, caption-residual k-means and AMI under
  1 min.

## Deviations from the plan

1. **Smoke uses 256 episodes per label, not 128.** With seed 42, the first 128 art-style episodes contain
   `Contemporary_Realism` only once, and `label_oracle_ranks(folds=2)` raises. The smallest prefix that works is
   156. With 256, every label has at least 5 episodes. The full run is unaffected.
2. **"Ranks identically" is asserted as an identical-rank share ≥ 0.999**, for sanity 2 and for the naive part
   of sanity 1. This allows for float rounding between `F.cosine_similarity` and a dot product of normalized
   codes. The observed share was 1.0000. The naive part of sanity 1 also requires pooled R@1 equal to 18.36 /
   20.36 at 2 decimals. The oracle part requires pooled R@1 within 0.25 points of 18.41 / 21.04, and the
   identical-rank share is reported alongside.
3. **R@1 CIs are computed only for the headline rows** (each code's best β, the nulls and CLIP-only), to keep
   the bootstrap count down. The β grid is reported as point values. Comparisons have CIs, as planned.
4. The R3 `oracle − R3 oracle` comparison is identically zero, so it is omitted. The script adds one extra row,
   `CLIP-only − naive(R3, 0.3)`.

## Open interpretation choices (flagged for review)

- **PCA input** is raw CLIP features, centered per modality and **not** L2-normalized. The plan says
  L2-normalized for clip512 and for the residual, but not for PCA, so the script follows the plan literally.
- **Caption residual** is the raw caption feature minus the raw mean caption feature of its leakage group's fit
  rows, then L2-normalized. Only groups with at least 2 fit rows are used.
- **AMI with emotion** uses all 9 emotions, including "something else".
- **Smoke fits** (probes, PCA, k-means, scale scalars and AMI rows) use the 20,000-row subsample throughout.

## Quality gate

`/ccg:verify-quality` returned 0 errors and 11 warnings: long `tables` and `sanity_checks` functions, many
parameters, and a file over 500 lines. The neighbouring stage-(d) scripts follow the same pattern, so nothing
was changed.
