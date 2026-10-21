# D2 ablation queues

Run from this worktree's round folder:

```bash
cd /project/CoSiR-wt/r10-d2-ablation-t06/src/test/20261021_d2_ablation
bash ab_queue_stage0.sh
bash ab_queue_stage1.sh
bash ab_queue_stage2.sh
```

Outputs default to `/project/CoSiR/src/test/20261021_d2_ablation/outputs`.
Set `AB_OUTPUTS` to use another destination. Logs are
`$AB_OUTPUTS/logs/stage<N>_<label>.log` (replaced on rerun); status history is
`$AB_OUTPUTS/logs/stage<N>_status.txt` (appended). With no override, these are
under the default outputs folder's `logs/`.

Stage 0 trains R0 and R2 for seeds 0, 1, 2 at full length, then compares all
18 exports and best epochs with the model sheet. It takes about **5 min GPU**.
Stage 1 requires a passing `stage0.json`, builds missing graphs in the main
round folder on CPU, then trains R1 and R3 through R12, seeds 0, 1, 2. A failed
row is logged and the queue continues. Budget about **25 min GPU**.
Each GPU step first checks the driver and notes existing compute apps, then
waits up to 7,200 seconds for `/tmp/gpu0.lock` using `flock -o`.
A driver check failure stops the queue.

Stage 2 always analyses the four Qwen views, even after stage 0 fails. When
any row metadata exists besides R0 and R2, it also scores and analyses rows.
It uses CPU only: 4 workers for analysis, 8 for scoring. Budget **1 to 1.5 h
CPU**: views 4 to 13 min, score 50 to 70 min, rows a few minutes.

For the controller's smoke run, use an explicit scratch destination:

```bash
export AB_SMOKE=1
export AB_OUTPUTS=/tmp/claude-0/-project-CoSiR/594d01b4-f9b6-4254-9058-fab6ca0ffbe4/scratchpad/t06/queue_smoke
bash ab_queue_stage0.sh
bash ab_queue_stage1.sh
bash ab_queue_stage2.sh
unset AB_SMOKE AB_OUTPUTS
```

Smoke mode refuses an unset `AB_OUTPUTS` or a resolved path outside
`/tmp/claude-0/`. Stage 0 still does full GPU training: the regression check
requires real outputs. Stage 1 trains only R3, R5, R7, R9, R10 with two epochs,
all three seeds. Its graph store is still built/read in
`/project/CoSiR/src/test/20261021_d2_ablation/graphs`, including in smoke mode.
Stage 2 adds `--smoke` to both analysis calls and uses 20 bootstrap resamples,
emotion only, for scoring. Smoke analysis writes `views_smoke`/`rows_smoke`
reports; scoring writes its usual outputs in the scratch destination.
Do not set `AB_SMOKE_ROOT` to a narrower path that excludes `AB_OUTPUTS`.

Status lines:

- `AB_STAGE<N>_START <timestamp>`: queue invocation began.
- `<label>=0`: step succeeded; a nonzero code means failure. Read its log.
- `GPU_COMPUTE_APPS_NOTE pid,used_memory:` and following entries: compute
  processes observed before a GPU step; informational, not a gate.
- `GPU_CHECK_FAIL`: driver query failed; no training starts for that step.
- `AB_STAGE<N>_DONE <label>=<code> ...`: summary, also printed to the terminal.
  The queue exits 1 if any step failed, otherwise 0. A GPU check failure adds
  `gpu_check=1`. A failed `require_stage0` or graph build stops stage 1.

The stage 0 check log prints `AB_STAGE0 PASS 18/18` or
`AB_STAGE0 FAIL <passing pairs>/18 <first reasons>`. The atomic `stage0.json`
holds all pairs, array equality, NPZ hashes, best epochs and failure reasons.
The gate prints `AB_STAGE0 REQUIRE PASS` or `AB_STAGE0 REQUIRE FAIL ...`.
It checks the saved pass flag; rerun stage 0 after changing training outputs.
Files with `.pending` in their name are never selected by the checker.
