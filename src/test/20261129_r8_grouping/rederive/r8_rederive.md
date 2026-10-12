# Ticket 11 stored-file interface

`r8_rederive.py` (RD) is separate code for decision-rule §7(a) to (d), including §6.1 to §6.5.
It imports only `r8_common` and the four frozen library pieces R7 §9 phase A permits (`EvalContext`,
`crossfit_condition_free`, `zscore_rows`, `cluster_bootstrap`). It loads the frozen A3 object on CPU to recover the
N1u inputs and fits no grouping, head, reader or builder. All real joins come from `r8_common`. Own-map image
posteriors recovered from φ drive its scoring, after comparison with the stored ones.

Run from the round folder with `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`, using
`/root/miniconda3/envs/CoSiR/bin/python`.

```bash
python rederive/r8_rederive.py --step trigger --pending results/trigger_pending.json
python rederive/r8_rederive.py --step gate --pending results/gate_pending.json \
    --index results/seed42/index.json results/seed42/index_nbc.json \
    --groupings results/groupings/r8_groupings.json results/groupings/r8_groupings_nbc.json results/groupings/r8_drops_main.json
```

`--index` and `--groupings` take one or more files (`nargs="+"`), as ticket 10 does; give RD the same lists as 10.
`--root` defaults to `results/` (`results/smoke/gate` with `--smoke`). Defaults: `ROOT/seed42/index.json` and
`ROOT/groupings/r8_groupings.json`. `--aff-reader` overrides AFF's reader; `--fixture` (smoke only) supplies tiny
synthetic joins for the CPU tests.

## What RD reads

- **Pending file:** ticket 10's layout (`<step>_pending.json` as `run_r8_gate.py` writes it): the stats at the top
  level (`main`, `screen`, `fallback`), the trigger under `trigger`, plus `groupings`, `dropped`, `stage_sha256`,
  `probe_trigger_sha256`, `input_sha256`, `step`, `smoke`, `rule_sha256`. Every file in `input_sha256` is hashed and
  its SHA-256 compared, used or not. RD asserts `step`, `smoke` and `rule_sha256`. It compares `main`, `screen`,
  `trigger`, `groupings`, `dropped` and the booleans of `fallback`. It skips `report_flags`, `written` and the two
  constant `fallback` strings (they must be strings). A key present on one side only is a disagreement, at any depth.
  The index, batch, grouping and stage files RD reads must be listed in `input_sha256`.
- **Groupings:** all `--groupings` JSONs are merged first. An NPZ (same stem) is read only for a file that has
  groupings. A drop overrides a grouping. Z<X> is skipped when X is dropped, unless Z<X> itself is recorded dropped.
  A drops-only file (`{groupings: {}, dropped: {...}}`) needs no NPZ and may come in any order.
- **Seed-42 index (09):** each entry gives `file`, `sha256`, `groupings`, `taus_file`, `taus_sha256`,
  `readers_sha256`. RD reads `taus_<batch>.json` (key `taus`) and `reader_<X>.npz` beside the index, with those SHA-256s.
  τ^X is recomputed from the reader margins (percentiles 0, 25, 50, 75) and compared.
- **Space and head files (07, 08):** `spaces/<X>.npz` for every grouping, TP included (its space is the one-hot of
  the final TP map, so M11 and M12 return the stored map), `heads/post_<X>.npz`, `heads/xheads_<X>.npz` (R7's
  format: no `rows` key; the row count must equal `len(scorer_train)`). Z<X> comes from the M13 seeds
  (7100 + i, 7200 + i, 7300 + i, i = index of X in the rule's list; ZNB100 uses 7104, 7204, 7304).
- **Centres (index "Centres"):** μ_g = float64 mean of the float32 φ_p, divided by its float64 norm, kept float64;
  the map is the argmax of the float64 dot, ties to the lowest group.
- **A0 (R7, read only):** `20261126_design_l_style_source/results/heads/post_{affect,image,caption}.npz`, `post_R0.npz`
  (the legacy R0 part of B1) and `results/seed42/reader_A0.npz` (AFF's reader). AFF keeps the rule's four thresholds.
- **Stage records (04, 06):** `ROOT/stage0_regression.json`, `stageE_<AL|DI|SG>.json` (`evidence.embeddings`,
  `evidence.image_index`, `esc1.{b_shape,c_finite,d_index_hash}.ok`) and `stage1_fcmi.json` (`loss_log`,
  `loss_log_sha256`, `labels_before_merge`, `labels_sha256`, `tests.{finite_loss,clusters}`, `clusters_before_merge`,
  `steps`). RD recomputes shape, finiteness and the expected index hash (from the common `nonheld_images()` as
  int64) for ESC1, and the finite-loss flag from the `total_loss` values (never the stored `finite` flag) and the
  cluster count for SC2. A stage with `pass: false` and no evidence file gives "not produced".
- **Integer bars (L16):** ⌈a·N/25⌉ with exact fractions (a = −0.3, 0.5, 0), as ticket 10's `need`: −49, 246 and 0
  at N = 12,288; smaller in smoke. Intervals use RD's own `cluster_bootstrap` call (5000 draws, seed 42) on each
  scope's clusters; only keys ending `ci95` have the 1e-9 tolerance.

## Output

CLI writes `ROOT/rederive/<step>_rederive.json` through the write-once writer: `status` ("agree" or "disagree"),
`step`, `smoke`, `disagreements`, `input_sha256` (absolute path to SHA-256 of every file RD read, the pending file
included), the re-derived verdict fields and the stage checks. Exit 0 on agreement, 1 on disagreement; malformed or
missing files assert.

**The record is write-once.** After a disagreement, a rerun needs the controller to archive the old
`<step>_rederive.json` and the pipeline's `<step>_pending.json` first (ticket 10 refuses to overwrite its pending file
too). Smoke records may be rewritten.

**Long step.** The audit hook hashes every file read under `/project/CoSiR`, `/data` and the worktree, including the CLIP
feature store; ticket 10's `publish_checked` rehashes all of them. A real run takes more than four minutes and is for the
controller. The known-answer check of the derivation (RD's family and comparators on R7's stored R0 inputs) is
`/project/CoSiR/.scratch/r8-grouping/notes/known_answer_11.py`.

## Tests

```bash
cd src/test/20261129_r8_grouping && python -m pytest -q -p no:cacheprovider --rootdir=. test_r8_rederive.py
```

The tests use pytest's `tmp_path`. They feed RD the pending layout produced by ticket 10's own `gate_stats` on
identical arrays (both steps must agree) and include a guard test for each comparison (missing or extra key,
wrong declared SHA-256, missing or extra array, S2 scope, SC3 strictness, M13 seeds, stage and probe hashes).
