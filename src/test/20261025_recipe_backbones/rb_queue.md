# Run notes: queues of the recipe-and-backbones round (ticket 16)

Everything runs from the repo root of the merged main checkout (`/project/CoSiR`), never from a worktree: the extraction
script `scripts/extract_features.py` and the round folder are found from the queue script's own place. Start a queue with
`nohup bash <script> > <scratch>/<name>.out 2>&1 &` and read the status file. Queues never wait for input.

## Environment (set by `rb_queue_common.sh`; override only when written here)

| Variable | Default | Use |
|---|---|---|
| `RB_OUTPUTS` | `src/test/20261025_recipe_backbones/outputs` of the main checkout | outputs folder; passed as `--outputs` to every runner |
| `RB_SMOKE=1` | off | adds `--smoke` where the runner has it; `RB_OUTPUTS` must be under `/tmp/claude-0/` (else exit 2) |
| `RB_PYTHON` | CoSiR env python | |
| `RB_LOCK_WAIT`, `RB_GPU_LOCK` | 14400 s, `/tmp/gpu0.lock` | GPU steps: `flock -o -w $RB_LOCK_WAIT $RB_GPU_LOCK env CUDA_VISIBLE_DEVICES=0 python -u ...` |
| `RB_DEFER` | empty | e.g. `K5`: candidates left out of stages 3, 4, 5 for time (spec D32; logged; never dropped; run the stage again without it) |
| `RB_DEVICE` | `cuda` | `cpu` only for a plumbing test without the GPU |
| `RB_DRY=1` | off | print each step's command, run nothing |
| `RB_CPU_LOCK` | `/tmp/rb_cpu.lock` | the CPU-only queues (1x, reports) run one at a time |

Always `PYTHONDONTWRITEBYTECODE=1`, 8 threads, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. CPU steps hide CUDA.

## Status lines

`RB_<QUEUE>_START <date> smoke=<0|1>`, then one line per step `label=<exit> seconds=<n>` (also in
`<outputs>/queue_<queue>_status.txt`; logs in `<outputs>/logs/<queue>_<label>.log`), `RB_GATE <name> ok: ...` for a gate,
`RB_<QUEUE>_NOTE ...`, and `RB_<QUEUE>_STOP ...` with the reason when a step or gate fails; the queue then ends with
`RB_<QUEUE>_DONE failed=1`. A non-zero exit of a queue means stop; nothing later in the order runs.

## Order and queues

| # | Queue (`rb_queue_*.sh`) | Resource | What | Gate before it |
|---|---|---|---|---|
| 1 | `stage0` | GPU | budget line; `rb_stage0` (V0) | none (skipped when `stage0.json` passed) |
| 2 | `stage1` | GPU, then CPU | per recipe `rb_run vp3` then `rb_part1` (R0-5, R0-3, R6-5, R6-3, R7-5, R7-3); `rb_pick`; `rb_vp1`; then the pick line | stage 0 passed |
| 3 | `stage1x` | CPU, probe on GPU | `rb_sheet`, `rb_part1_report`, `rb_probe`; a failed step is "reported as missing", not a stop | VP1 agrees |
| 4 | `stores` | GPU | CLIP B/32 full; DINOv2 full and 28 (only if K5 triggered); the three instructed Qwen stores of the 746 | part 2 gate |
| 5 | `stage2` | GPU, CPU | regressions: `rb_student regress`, `rb_matched --regress`, `rb_score --check` (`stage2`) | part 2 gate |
| 6 | `stage3` | GPU, CPU | per candidate K2, K3, K4, K5: student train and infer, matched arm; `rb_score --stage 3`; `rb_v2 --stage 3` | stage 2 gate |
| 7 | `cpu 3` | CPU only | `rb_part2_report --stage 3` (30 to 40 min per candidate, estimated) | V2 agrees |
| 8 | `stage3a` | GPU, CPU | the same three regressions at full scale | part 2 gate |
| 9 | `stage4` | GPU, CPU | per candidate with an S1 outcome other than S1-NO: full scale; `rb_score --stage 4`; `rb_v2 --stage 4` | stage 3a gate |
| 10 | `cpu 4` | CPU only | `rb_part2_report --stage 4` | V2 agrees |
| 11 | `stage5` | GPU, CPU | K1 ArtELingo-28 at full, B1 and B6 outputs; `rb_a28`; `rb_v2 --stage 5` | stage 3a gate |

`rb_queue_gpu.sh` runs queues 1, 2, 4, 5, 6, 8, 9, 11 in this order and stops at the first failed one; it starts 3, 7 and
10 in the background as soon as their inputs exist and waits for them at the end. Single queues: `bash
src/test/20261025_recipe_backbones/rb_queue_<name>.sh` (CPU queue: `rb_queue_cpu.sh 3` or `4`). Every queue can be
started again after a stop: finished work is skipped by the runners' own checks.

### Gates (`rb_run.py gate NAME`, exit 1 with an `RB_STOP` line)

- `stage1`: `stage0.json` has `pass: true` (V0). `vp1`: `part1/rederive_vp1.json` has `agree: true` and is newer than
  `pick.json`. No pick is stated before: stage 1 ends with `rb_pick`, then `rb_vp1`; the pick line (`RB_PICK ...`) is
  printed by the driver only after the `vp1` gate, and the pick step's log is not echoed. `part2` = `stage1` + `vp1`.
- `stage3` (also `stage2` result): `part2/regression/stage2.json`, `stage3a` for stages 4 and 5: every expected item
  present and passing, the file's `pass` true. Expected names, from the three runners' code: `student/emotion`, `student/style`,
  `student/genre`, `student/held_list`; `matched/qwen_s0/{eval_prob,held_prob,mapper_state,config}`,
  `matched/ct_mapper_infer/{P3_s0,P2_s0}`; `scorer/{A_K1,entries,d,interval_lo,interval_hi,bootstrap_d,outcome}`, at least
  one `scorer/A_e/*` and one `scorer/partitions/*`. (`rb_score.require_gate` checks again inside the runners.)
- `stage4`: stage 3a items as above and `rederive_v2_stage3.json` agrees. A candidate's own gate: `rb_run candidates 4` lists
  those with an `outcome_id` that is not None and not `S1-NO` (from `verdict_stage3.json`).
- `vp3 --recipe R [--forced]`: the plan is the six recipes (`part1/plan.json`); a recipe with P exports for all three aspects
  is not trained again without `--forced`; every call is appended to `part1/vp3_log.jsonl`.
- `v2_3`, `v2_4`: the re-derivation agrees before the reported readings run (rule §4.2).

### VP2 after a forced rerun

`rb_part1.check_p` asserts before a P export is written. If `rb_part1 --recipes R` fails and the log's last line is the
VP2 assertion (`nonfinite rows`, `norm error`, dtype/shape), the queue notes it, runs `vp3 --forced` and reruns once
(logged). Failing again, R0-5 stops part 1 (`RB_STAGE1_STOP`); another recipe is passed to `rb_pick --vp2-failed R=CAUSE`
and the pick goes on without it. A failure that is not a VP2 assertion stops the queue.

### Triggers, K5, stores

`pick.json` `triggers {K4, K5}` are read by `rb_run candidates|triggers` (never re-derived). K4 exists unless P-KEEP; K5, M5
and the two DINOv2 stores only when K5 is triggered. Candidates run in the order K2, K3, K4, K5. Stage 3 sets `soft=1` on the
scorer: its exit 1 (a candidate without an outcome, V failure) is reported; `rb_run check-verdict` then demands a block for
every candidate, so a crash still stops the queue.

### Stage 5 scale

K1 at full scale always. Any other model at full scale only when stage 4 gave it an outcome (`outcome_id` is not None),
else at sample scale. "Stage 4 ran" means "ran to an outcome". `rb_run a28` prints the models, the scales that need B1 and B6
outputs, and each model's scale (read from `verdict_stage4.json`); `rb_a28` and `rb_v2` read the same file.

### Stores

From the repo root: `scripts/extract_features.py --dataset artelingo_full --backbone clip_b32`; `... --dataset artelingo_full
--backbone dinov2_large --image-only`; `... --dataset artelingo28 --backbone dinov2_large --image-only` (K5 only);
`rb_extract_a28.py --keys emotion_focus style_word genre_word` for the missing keys. A store whose `index.json`
(`done.json` for the Qwen stores) exists is skipped. Smoke: `extract_features.py ... --limit 16 --out-root <outputs>/stores`
and `rb_extract_a28.py --limit 16 --dest <outputs>/stores/a28`; smoke stores never replace the real ones, so the part 2
smoke reads the real stores in `/data/SSD2/pre_extract` (extract them first with a real `stores` run).

## Smoke (the controller's; one command per queue, scratch under `/tmp/claude-0/`)

```bash
export RB_SMOKE=1 RB_OUTPUTS=/tmp/claude-0/<scratch>/rb   # the same scratch folder for all queues, in this order
cd /project/CoSiR/src/test/20261025_recipe_backbones
bash rb_queue_stage0.sh; bash rb_queue_stage1.sh; bash rb_queue_stage1x.sh; bash rb_queue_stores.sh
bash rb_queue_stage2.sh; bash rb_queue_stage3.sh; bash rb_queue_cpu.sh 3; bash rb_queue_stage3a.sh
bash rb_queue_stage4.sh; bash rb_queue_cpu.sh 4; bash rb_queue_stage5.sh      # or: bash rb_queue_gpu.sh
```

Smoke runners shrink their work (seed 0, at most 2 epochs, 20 draws); the regressions (`stage2`, `stage3a`) are never shrunk
and run for real into the scratch folder. A smoke stage 0 exits 1 by design (its record says `smoke`); the queue accepts that.
`RB_DRY=1` prints all commands without running (plumbing check only).

## End estimate against 11:00 (D32)

`rb_run.py budget` (first lines of the stage 0 queue): it adds the step times in chain order, skips steps whose output file
exists, and puts the two report queues beside the chain (stage 3's starts after its V2; stage 4's after its V2 and after stage
3's report, one CPU job at a time). End = the later of the chain's end and stage 4's report end; low and high from the
table in `rb_run.py` (`M` measured: part 1 bootstrap 7 min at 8 workers, VP1 7 min, one P export's partitions 7.5 s;
`S` spec §9: part 1 25 min, CLIP store 10, students and matched arms at both scales 45, K4 15, DINOv2 35, ArtELingo-28 5,
regressions 10; `E` this ticket's estimates: scorer bootstrap and V2 7 to 15 min, reports 30 to 40 min per candidate per stage).
The deadline is the next 11:00 Amsterdam time. If the high end is past it, the run chat splits the run by itself
(`RB_DEFER="K5"`, then `"K4 K5"`), says why in the report, and runs the deferred candidates later: never dropped. Reports
by 2026-10-26 10:00. The estimate does not model CPU contention between the reports and the chain's own CPU steps.
