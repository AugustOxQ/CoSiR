# Run notes: queues of the relational genre round (ticket 11)

Run from the merged main checkout, not from a worktree. Start the chain with
`nohup bash src/test/20261027_relational_genre/rg_queue_gpu.sh > <scratch>/gpu.out 2>&1 &` and read the status files
(`<outputs>/queue_<a|b|c|gpu>_status.txt`). Queues never wait for input. Style and variables are those of 12-25's
`eg_queue.md`; this file lists what differs.

## Environment (set by `rg_queue_common.sh`)

| Variable | Default | Use |
|---|---|---|
| `RG_OUTPUTS` | `/data/SSD2/cosir_r10/20261027_relational_genre/outputs` | outputs folder; passed as `--outputs` to every runner |
| `RG_SMOKE=1` | off | adds `--smoke` where the runner takes it; `RG_OUTPUTS` must be under `/tmp/claude-0/` (else exit 2); the pending list goes to `<outputs>/pending_deletions.md`; the regression gates VA1, VB1 and VC1 only note a failure (2 epochs cannot equal 12-25's weights) |
| `RG_DRY=1` | off | prints every command in order and runs none; writes no file (status goes to `/dev/null`). Gates and reads of verdict files are replaced by placeholders (`NEWEST`, `G`, one climb rung shown) |
| `RG_PYTHON`, `RG_WORKERS` | CoSiR env python, 8 | |
| `RG_LOCK_WAIT`, `RG_GPU_LOCK` | 14400 s, `/tmp/gpu0.lock` | every GPU step: `flock -o -w $RG_LOCK_WAIT $RG_GPU_LOCK env CUDA_VISIBLE_DEVICES=0 python -u ...`; the queue waits, never `-n`; CPU steps take no lock and hide CUDA |
| `RG_DEVICE` | `cuda` | `cpu` only for a plumbing test |
| `RG_PENDING` | `/project/CoSiR/.scratch/pending_deletions.md` | where weights that may go are listed (never deleted) |

Always `PYTHONDONTWRITEBYTECODE=1`, 8 threads, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## Status lines

`RG_<Q>_START`, one line per step `label=<exit> seconds=<n>` (logs: `<outputs>/logs/<queue>_<label>.log`),
`RG_<Q>_NOTE ...`, `RG_<Q>_STOP <reason>` when the queue ends on a failed step or gate, then `RG_<Q>_DONE failed=<0|1>`.
`Q` is `A`, `B`, `C` or `GPU`. Stage B starts only when the last `RG_A_DONE` line says `failed=0`; stage C only when
the last `RG_B_DONE` says `failed=0` and stage B's outcome is B-PICK. B-NONE: stage C prints a note and ends with
`failed=0` (no held test).

## Queues, order, commands (`rg_queue_*.sh`; GPU = takes the lock)

Runner flags: `--outputs`, `--smoke` and `--workers` are added as each runner takes them. `rg_train` and `rg_full` take
`--device`.

### `a` (stage A, seed 0)

| # | Step (label) | Command | Res. | Gate / stop |
|---|---|---|---|---|
| 1 | `va1_train` | `rg_train regress va1` | GPU | the VA1 runs' weights go on the pending list |
| 2 | `va1_scorer` | `rg_score check va1` | CPU | VA1 = `regression/va1.json` and `va1_scorer.json` both `pass` true; otherwise **stop**: nothing runs in A, B or C |
| 3 | `eq` | `rg_train eq --device cpu` | CPU | fails: noted, GQ-8 still trains and cannot win (rule §4.1); no `eq.json`: stop |
| 4 | `train_GR-16_GR-32_s0` | `rg_train train --arms GR-16 GR-32 --seeds 0` | GPU | a failed run (non-finite loss; VA3 after the trainer's one rerun) is noted, the scorer reads its `run.json`; a crash without `run.json` stops |
| 5 | climb loop, at most four decisions: `climb_i`, `va2_rung_<newest>`, `train_<next>` | `rg_score climb`; `rg_v2 --phase rung --arm <newest>`; `rg_train train --arms <next> --seeds 0` | CPU, CPU, GPU | `<newest>` and `<next>` come from the last decision of `stage_a/climb.json` (`next` null ends the climb; GR-64, GR-128, GR-256 only). **`rg_v2` exit 1 stops the queue before the next rung** (rule §4.1) |
| 6 | `train_GN-8_GS-0.02_GS-0.01_GQ-8_s0` | `rg_train train --arms GN-8 GS-0.02 GS-0.01 GQ-8 --seeds 0` | GPU | one call, the four arms in this order |
| 7 | `score_a` | `rg_score a` | CPU | after it: weights listed in `stage_a/verdict.json` `pending_deletions` go on the pending list |
| 8 | `va2_a` | `rg_v2 --phase a` | CPU | exit 1: **stop**, no stage A outcome |
| 9 | `loss_scale` | `rg_train lossscale` | GPU | after the outcome (spec §3.6); a failure is noted, B goes on; its weights go on the pending list |

### `b` (stage B, seeds 1 to 3)

| # | Step | Command | Res. | Gate / stop |
|---|---|---|---|---|
| 0 | start | stage A's last `DONE` has `failed=0`; `stage_a/verdict.json` exists | | else **stop** |
| 1 | `vb1` | `rg_score check vb1` | CPU | `stage_b/vb1.json` `pass` true; otherwise **stop** (B and C) |
| 2 | `train_GR-2_GR-4[_v*]_s123` | `rg_train train --arms GR-2 GR-4 [<v*_A>] --seeds 1 2 3` | GPU | v*_A is added only when `stage_a/verdict.json` `outcome` is A-NEW (`winner`); failed runs as in stage A (arm not confirmed) |
| 3 | `score_b` | `rg_score b` | CPU | |
| 4 | `vb2` | `rg_v2 --phase b` | CPU | exit 1: **stop**, no B outcome |
| 5 | `report_ab` | `rg_score report` | CPU | |

### `c` (stage C, only on B-PICK)

| # | Step | Command | Res. | Gate / stop |
|---|---|---|---|---|
| 0 | start | stage B's last `DONE` has `failed=0`; `stage_b/verdict.json` `outcome` = B-PICK; G = `v_held` | | B-NONE: note and end, exit 0 |
| 1 | `vc1_train` | `rg_full regress vc1` | GPU | the VC1 run's weights go on the pending list |
| 2 | `vc1_scorer` | `rg_held check vc1` | CPU | VC1 = `stage_c/regression/vc1_train.json` and `vc1_scorer.json` both `pass` true; otherwise **stop**, no held output of G |
| 3 | `train_g` | `rg_full train --arm G --seeds 0 1 2 3 4` | GPU | failure (VC4: non-finite loss, not rerun): **stop**, no stage C outcome |
| 4 | `held_g` | `rg_full held --arm G --seeds 0 1 2 3 4` | GPU | guarded; VC3's one rerun is the runner's; a second VC3 failure: **stop** |
| 5 | `score_c` | `rg_held score --arm G` | CPU | |
| 6 | `vc2` | `rg_vc2 --arm G` | CPU | exit 1: **stop**, no stage C outcome |
| 7 | `report_c` | `rg_held report --arm G` | CPU | |

`gpu` runs `a`, `b`, `c` in this order; a queue that stops is noted and the chain goes on to queues that refuse
themselves with the reason (a failed gate stops only the stages behind it, spec §3.5). Exit 1 when any queue stopped.
Every runner skips work that is finished, so any queue can be started again after a stop (same command). Flags guessed
for `rg_v2.py` (not on the integration branch when this was written): `--phase rung|a|b`, `--arm`, `--outputs`,
`--workers`, `--smoke`; the controller adjusts the spelling in `rg_queue_a.sh` and `rg_queue_b.sh` after ticket 09 lands.

## Pending deletions

Rows go to the pending list (header `| added | chat | path or branch | size | what it is | why it can go |`), once per
folder, from the `*.pt` files under it; nothing is deleted: VA1's three runs `regression/va1_*_s*/`; the arms in
`stage_a/verdict.json` `pending_deletions` (`train/<arm>/s0/`; GR-8's frozen weights are 12-25's and not listed); the
VC1 run `stage_c/regression/vc1_run/`; the loss-scale run `loss_scale/run/`. Kept (rule §4.3 Weights): the stage A arms
at or above T_g - r_genre, every stage B run, G's five full-scale seeds.

## Expected times (8 CPU threads; one RTX 3090)

Measured in 12-24 and 12-25 (status files of `eg_queue_stage_iii.sh` 10-25, `rb_queue_stage3a/4.sh`), applied to the
same kind of step: a sample-scale training call is about 2 min per run plus about 1 to 2 min to load the data (12-25: five
variants in one call 613 s); a full-scale seed about 100 s plus the data (K4: 178 s for the one regression seed, 515 s
for a full call); held inference about 15 s; a scorer call with the bootstraps 2 to 6 min per five arms (12-25
`score_first` 370 s); a re-derivation 6 to 15 min (12-25 VG2: 387 s first, 570 s all, 864 s fresh); the report 9 min
(12-25: 561 s). The unmeasured steps (the first climb, the held scorer, VC2, `check`) are estimates marked `~`.

| Step | Minutes |
|---|---|
| a: VA1 train 8, VA1 scorer ~6, EQ ~1 | 15 |
| a: GR-16 and GR-32 | 6 |
| a: climb, expected (GR-32 rises, GR-64 stops): decision 1 (`climb` ~4, `rg_v2 rung` ~5), GR-64 3, decision 2 ~9 | 21 |
| a: GN-8, GS-0.02, GS-0.01, GQ-8 (one call) | 10 |
| a: `rg_score a` ~12, `rg_v2 --phase a` ~15, loss scale ~3 | 30 |
| **a total, expected** (climb to GR-256: +24, so 106) | **82** |
| b: VB1 ~5; GR-2, GR-4 (and v*_A) x 3 seeds 20 to 25; `rg_score b` ~12; `rg_v2 b` ~18; report 10 | 65 to 70 |
| c: VC1 train 4, `rg_held check vc1` ~6; G x 5 seeds 10; held 2; `rg_held score` ~6; `rg_vc2` ~12; report ~12 | 52 |
| **Total, expected (A-SAME or A-NEW, B-PICK)** | **about 200 min (3.3 h)** |
| Worst case (climb to GR-256, A-NEW) | about 230 min (3.8 h) |

If started at 22:00 the chain ends about 01:20 (worst case about 01:50), plus any wait for the GPU lock (14,400 s
limit per GPU step). Without B-PICK it ends after stage B, about 00:30. GPU share about 1 h, CPU about 2.5 h; the spec's
budget (1 h GPU, 3 h CPU) covers it.

## Failed runs and `--failed`

The trainer reruns once itself after a VA3 failure and records `status failed` (kinds `nonfinite_loss`, `va3`) in the
run's `run.json`; the scorer reads that, so the queue passes no `--failed`. `--failed ARM=CAUSE` of `rg_score` is only for
a run whose `run.json` says ok but whose exports fail VA3 or VB3 (declared by hand; it needs the one rerun first). Run it
by hand, and the matching `rg_v2` call with the same flag if `rg_v2` takes one, then `rg_score report`:
`python rg_score.py a --outputs "$RG_OUTPUTS" --failed 'GS-0.01=VA3 after forced rerun'`.

## Recovery (after any stop)

1. Read the last lines of `<outputs>/queue_<queue>_status.txt` and the named log in `<outputs>/logs/`.
2. A runner crashed or the machine stopped: start the same queue again; finished runs, scores and re-derivations are
   skipped (runner side). A climb restarted after a stop recomputes the decision (it is not added twice) and goes on.
3. A gate failed (VA1, VB1, VC1): trace the cause; move the failed regression file aside to a dated folder under
   `<outputs>/aside/` (`mv`, nothing is deleted) after the fix, then start the queue again. VA1 failing stops everything
   behind it.
4. `rg_v2` disagrees (rung, a, b, vc2): both values go in the report; the fix is by the rule's text. After the fix move
   the stale scorer files of that phase and the re-derivation file aside (`stage_a/climb.json` and `verdict.json`,
   `stage_b/verdict.json`, `stage_c/verdict.json` and the matching `rederive_*` files; the training outputs stay) and start
   the queue again. A rung disagreement must be settled before the next rung runs.
5. A training run failed (non-finite loss, VA3 after its rerun): it stays failed and is reported (rule "Failed runs"; VA4
   and VB4: no other run). VC4: a G seed that fails is not rerun beyond the rule's one VC3 rerun; no stage C outcome.
6. GPU lock wait over 4 h: the step ends with a failure line; start the queue again when the GPU is free.

## Smoke (the controller's; GPU; scratch under `/tmp/claude-0/`)

```bash
cd /project/CoSiR/src/test/20261027_relational_genre   # the merged main checkout
export RG_SMOKE=1 RG_OUTPUTS=/tmp/claude-0/<scratch>/rg
bash rg_queue_gpu.sh                     # the whole chain at 2 epochs, 5,000 pairs, 20 draws
# or per queue: bash rg_queue_a.sh ; bash rg_queue_b.sh ; bash rg_queue_c.sh
```

In a smoke the regression gates do not hold and are only noted; stage C needs a B-PICK, which a smoke may not give: then
stage C ends with its note. To run stage C anyway, write a `stage_b/verdict.json` under the scratch outputs by hand
(`{"outcome": "B-PICK", "v_held": "GR-8"}` is not enough for the scorer's other readers: copy a real smoke one and edit
`outcome` and `v_held`), then `bash rg_queue_c.sh`. Dry run of the whole chain: `RG_DRY=1 bash rg_queue_gpu.sh`.
