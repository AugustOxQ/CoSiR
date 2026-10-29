# Ticket 06: stage 0 checks (2026-10-29)

`rt_stage0.py {0a,0b} [--outputs DIR] [--workers 8] [--smoke N]`
uses the shared data, metrics and statistics modules without changing them or
the frozen rounds. Full 0a writes identical JSON to the output directory and
`rt_paths.RANGES_FILE`; smoke never writes the latter. The queue commits ranges.
Input hashes cover all 15 normalized c* matrices, the three label arrays, the
emotion distributions and the ordered validation names (r10's newline encoding).
The validation-only input wrapper rejects any request for held data.

0b loads matrices inside a fork pool with one BLAS thread per worker. It calls
r10's `r20_partition` / `own_partition`, `point_table`, `_point_val` and
`_mean_over`, retaining the original representations, label mask and averaging.
Every comparison records name, value, reference, tolerance and pass; the stage
records elapsed times and returns a failing exit status for failed comparisons.

## Validation

Environment: `/root/miniconda3/envs/CoSiR/bin/python`,
`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`, CPU only.
Commands were limited to 220 seconds. No unit tests or full stages were run.

Both CLI stages ran with `--workers 8 --smoke 300 --outputs`
`/tmp/claude-0/-project-CoSiR/377b90b0-e3cd-4025-88b4-c754376deadf/scratchpad/codex/t06_smoke`.

| Check | Result | Seconds |
|---|---|---:|
| 0a, all aspects and five seeds, 300 rows | PASS; 20 input hashes; range/schema checks passed | 10.68 |
| 0b, 300 validation rows per head | PASS, 54/54; seven AMI comparisons explicitly skipped | 10.84 |
| Full held c* emotion, all five seeds | AMI 0.10658729378204021; reference 0.10658729378204021; difference 0 | 28.19 |
| Full validation emotion seed 0, 6,152 rows, mAP and nDCG | mAP 0.2663909953213407; nDCG 0.9249494206137872 | 3.13 |
| Full validation re-inference, emotion seed 0 | max absolute difference 0.00006103515625, below 0.001 | 3.39 |

The full validation metric seed excludes 3.63 seconds of label/distribution
setup. The re-inference measurement includes plain input loading and saved
output loading. Smoke's maximum difference over all 15 heads was also
0.00006103515625. Both smoke processes exited 0. No ranges file was created in
the source folder.

Estimate full 0a at about one minute: 15 times the measured emotion seed plus
setup is about 51 seconds; only emotion actually needs nDCG. Estimate full 0b
at 3–5 minutes with eight workers: 18 r20 tasks in three waves, allowing 8/5
contention scaling from the five-task measurement (about 128 seconds), plus
roughly 51 seconds for 15 full head checks and input overhead. PercepT's five
own readings are argmax only. This is below the 15-minute budget; the full run
has not been measured.

Smoke choice: slice all retrieval and re-inference matrices to the first N
rows after frozen-loader validation. Items ii/iii still verify full-source
hashes and genre counts because their fixed references describe the complete
held input. AMI checks are recorded as `skipped (smoke)` with pass true.

`r11-retrieval` was merged before the ticket commit (already up to date).
Nothing was deleted. Cleanup candidates: the scratch smoke output directory
above, and ignored local measurement files `t06_ami_timing.json`,
`t06_validation_timing.json`, `t06_inference_timing.json` in this folder.
