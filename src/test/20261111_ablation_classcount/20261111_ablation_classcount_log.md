# 20261111 ablation clustering at the class count

Written 2026-11-11 (Amsterdam time). Script: `t5_ablation_ami.py`. No file outside this folder was edited; nothing committed.

## What it does
Scores clustering AMI at the class count (emotion 9, style 27, genre 10) on the held split of ArtELingo (12,281 rows;
genre labelled on 9,949) for arm `k11=40` (our model, the gate) and its four removals (`no_instructed_graphs`,
`no_content_correction`, `no_buddy`, `no_relational`), seeds 0 to 4. Writes `out/ablation_ami.json` and prints a Markdown
table (arm by aspect, means to 5 decimals).
Subcommands: `gate` (only `k11=40`), `full` (gate arm, gate check, then the four removals; exits before the removals if
the gate fails), `smoke` (one arm, aspect, seeds; writes `smoke.json`).

## Reuse map
| Need | Taken from |
|---|---|
| export path, sha256, 135-sidecar check (rows 12,281, `list_sha256`, arm/aspect/seed) | `20261109_r20_figures/r20_ret.py`: `export_path`, `check_sidecars`, `sha256` (imported, not copied) |
| row binding: names and `list_sha256` | `rb_part2_report.held_labels()` as in `r20_ret.run`; asserted equal to `ns_data.keys` and to `ns_data.labels` for all 3 aspects |
| labels (-1 = unlabelled genre) | `20261103_new_scores/ns_data.labels` (record's labels) |
| rows in clustering form (float16 cast to float32, not renormalised) | same form as `ns_data.cstar(..., unit=False)`; loaded from the `r20_ret` path. For `k11=40` the script asserts bit equality with `ns_data.cstar(unit=False)` |
| k-means (MiniBatchKMeans, n_init 3, batch 4,096, states 0-2, 8 threads, all rows clustered) | `ns_metrics.partitions(x, k, 'artelingo')` |
| AMI (3 states averaged per seed, mask y>=0 after clustering, mean over seeds) | `ns_clu.metric_table(..., metrics=('ami',))` -> `ns_metrics.score_partitions` |
| class counts | `ns_paths.CLASS_COUNT`; equals the record's `class_count` and `ns_metrics.class_count(labels)` (all asserted) |
| gate values | `/data/SSD2/cosir_r17/20261103_new_scores/outputs/stage3/sweep/artelingo__qwen.json`, rows `arm=cstar, metric=ami, k=class count`; per-seed (5) and mean, tolerance 1e-9 |
The record was written by `ns_diag.sweep` -> `ns_clu_cells._cand_parts` -> `ns_data.cstar(unit=False)` + `ns_metrics.partitions`;
`ns_data.cstar` hardcodes arm `k11=40` (through `rt_data._cstar`), so it cannot load removals. The removals therefore use the `r20_ret`
export path; the gate arm uses the same loader, so the gate checks the path the removals take.

## Commands
Prefix every call: `cd /project/CoSiR/src/test/20261111_ablation_classcount && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python`
- gate: `t5_ablation_ami.py gate`
- full: `t5_ablation_ami.py full` (through `res-queue cpu`)
- smoke: `t5_ablation_ami.py smoke [--arm A] [--aspect X] [--seeds 0]`

## Smoke (machine load average about 12 at the time, so timings are on the high side)
- `k11=40`, emotion, seed 0: AMI 0.10170414948655143 against record 0.10170414948655143; max abs difference 0.0 (pass); 3.4 s for the fit (6.7 s whole run).
- Timing per seed, one `no_buddy` fit set each: emotion 3.4 s, style 15.3 s, genre 5.2 s, about 24 s per arm and seed.
  Estimated full run: 25 arm-seed units x 24 s = about 10 min, plus start-up (about 10 s).
- `no_buddy` seed 0 (not a result, a smoke only): style 0.27813, genre 0.51798.
- The smoke files `out/smoke.json` and `out_smoke_removal/smoke.json` are throw-away.

## Open questions
- None blocking. The gate has not been run on all five seeds and the three aspects yet (the controller runs `gate` or `full`).
- Seeds are the r10 training seeds 0 to 4 (the file `s<seed>`), the k-means states 0 to 2 inside each seed, as in the record.

## To remove (not deleted)
`out/smoke.json`, `out_smoke_removal/` (smoke output only).

## Full run, by the controller (2026-11-11, `res-queue cpu --short`, label `paper:t5_ablation_ami`)

Pass. 7.95 min in the queue (473 s in the script). Gate: arm `k11=40` equals the r17 record's `cstar` rows at the class
count, max abs difference 0.0 on every per-seed value and the mean, all three aspects. Log `logs/full.log`, record
`out/ablation_ami.json` (kept in git as `records/ablation_ami.json`; `out/` and `logs/` are gitignored). AMI at the class count (9 / 27 / 10), mean over seeds 0 to 4:

| arm | emotion | style | genre |
|---|---:|---:|---:|
| k11=40 (ours) | 0.10241 | 0.41093 | 0.51007 |
| no_instructed_graphs | 0.06981 | 0.22516 | 0.51986 |
| no_content_correction | 0.08626 | 0.34277 | 0.51021 |
| no_buddy | 0.07374 | 0.27050 | 0.51762 |
| no_relational | 0.10293 | 0.40863 | 0.43935 |

Same direction as the 20-cluster ablation of report 2026-12-28: the instructed graphs, the buddy terms and the content
correction carry emotion and style, the relational term carries genre. Filled into the paper's Table 5 (REPO commit
after `2ca85c6`). Throw-away smoke outputs: `out/smoke.json`, `out_smoke_removal/` (not committed; can be removed).
