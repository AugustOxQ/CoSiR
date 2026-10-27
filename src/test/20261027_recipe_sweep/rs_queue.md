# Run notes: the queues of the recipe sweep (ticket 15)

Run from the merged main checkout, not from a worktree, after the build's gates exist. Style, variables and status lines are
those of 12-27's `rg_queue.md`; this file lists what differs. Queues never wait for input; each runner skips finished work, so
any queue can be started again after a stop (same command). Start a queue with
`nohup bash src/test/20261027_recipe_sweep/rs_queue_<name>.sh > <scratch>/<name>.out 2>&1 &` and read
`<outputs>/queue_<name>_status.txt`. Each node step (sync, N1, N2, every launch and pull) adds an entry to
`src/test/20261027_recipe_sweep/rs_node_log.md`; the node plan prints the reminder, it does not write the log.

## Files

| File | What |
|---|---|
| `rs_queue_common.sh` | sourced by the queues: env, status lines, `step`, `softstep`, `gate`, the weekday rule, pending rows |
| `rs_queue_stage0.sh` | seed 4 train, sample held inference, V0c, V0d, the stage 0 verdict, `rs_vh stage0` (V0b); then G's analysis |
| `rs_queue_fallback.sh` | stages 1 and 2 locally (rule §4.2 fallback: 12-27's GS-0.02 seed-0 run is G0) |
| `rs_node_plan.sh stage1` / `stage2` | stages 1 and 2 on node402: prints (`RS_DRY=1`) or runs the `cluster` commands |
| `rs_queue_stage3.sh` | TOP at seeds 1 to 3, the verdict, `rs_vp stage3` (V3c) |
| `rs_queue_stage4.sh` | c* at full scale, held inference, the verdict, `rs_vh stage4` (V4b), reported margins; then c*'s analysis |
| `rs_queue_ablation_dev.sh` | the eight removals at seeds 1 to 3, `rs_ablation dev`, `rs_vp ablation` |
| `rs_queue_ablation_held.sh` | the eight removals at full scale, `rs_ablation held`, `rs_vh ablation`; weekday rule |
| `rs_queue_names.py` | F's name, the eight arm names, the 13 knobs' runs, the shards (plumbing, no decision) |

`scripts/run_rs_node.sh` gained one mode, `vp` (`rs_vp partitions --machine node402`, V1c part (a) on the node's CPU); the plan
needs it, ticket 14 had none for part (a).

## Environment (`rs_queue_common.sh`)

| Variable | Default | Use |
|---|---|---|
| `RS_OUTPUTS` | `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs` | outputs folder; passed as `--outputs` to every runner |
| `RS_SMOKE=1` | off | adds `--smoke` where the runner takes it; `RS_OUTPUTS` must be under `/tmp/claude-0/` (else exit 2); a missing real gate is noted and the chain goes on |
| `RS_DRY=1` | off | prints every command in order, runs none, writes no file; gates and verdict reads become notes with placeholders (`F`, `TOP_1`, `C_STAR`, `POINT_n`; the removal arms are shown for F = G) |
| `RS_PYTHON`, `RS_WORKERS` | CoSiR env python, 8 | |
| `RS_LOCK_WAIT`, `RS_GPU_LOCK` | 14400 s, `/tmp/gpu0.lock` | every GPU step: `flock -o -w ... env CUDA_VISIBLE_DEVICES=0 python -u ...`; never `-n`; CPU steps hide CUDA |
| `RS_DEVICE` | `cuda` | `cpu` only for a plumbing test |
| `RS_PENDING` | `/project/CoSiR/.scratch/pending_deletions.md` | pending deletions list (rows appended, nothing deleted) |
| `RS_SWEEP_MACHINE` | local if `sweep/local/stage1.json` exists, else `node402` | the machine whose stage 1 and 2 files stage 3 reads |
| `RS_FORCE_HOUR=1` | off | skips the weekday rule |
| `RS_DAY_OK=1` | off | fallback: the first local run drew under 200 W, the rest may run on weekday daytime |

Always `PYTHONDONTWRITEBYTECODE=1`, 8 threads, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Status lines: `RS_<Q>_START`, one line per
step `label=<exit> seconds=<n>` (log `<outputs>/logs/<queue>_<label>.log`), `RS_<Q>_NOTE`, `RS_<Q>_STOP <reason>`,
`RS_<Q>_DONE failed=<0|1>`; `Q` is `STAGE0`, `FALLBACK`, `STAGE3`, `STAGE4`, `ABLATION_DEV`, `ABLATION_HELD`.

## Order, gates and what stops what (spec §3.5, rule §4.0)

Order: the gates V0a, V1a, V3a, V4a (the build; their files are under `<outputs>/gates/`); then stage 1 on the node while
`stage0` and G's analysis run locally; stage 2 (node); stage 3; stage 4; c*'s analysis; the development ablation by day; the
held ablation from 19:00. A failed gate stops only the stages behind it: stage 0 blocks nothing; stage 1 blocks 2, 3, 4 and
the ablation; V4a blocks stage 4 and the held ablation; V3a blocks stage 3 and both ablations.

| Queue | Step (label) | Gate checked first, by whom | Failure |
|---|---|---|---|
| `stage0` | `train_seed4` | queue: `gates/v1a_train.json` and `v1a_scorer.json` pass (`rs_train train` does not check V1a), `gates/v0a.json` pass | stop; a failed seed 4 stops the queue and is NOT rerun (rule V0d: seed 4 trains once): the sample scale then has no outcome and the full scale is scored as usual (the stage 0 code is being fixed to do so); the remaining steps are then run by hand |
| | `held_sample` | runner: `rs_stage0 held` refuses without V0a, and for seed 4 without a passing `v1a_train.json` for the trainer on disk | stop |
| | `v0c`, `v0d`, `verdict` | V0c soft (a failing scale gets no outcome, the verdict scores the other); V0d and the verdict are hard (the runner refuses without V0a and V0d) | stop |
| | `v0b` | `rs_vh stage0` (exit 1 on disagreement) | stop: no outcome stated, G's analysis does not run |
| | `analysis_g` | runner: refuses without `stage0/verdict.json` | stop |
| `fallback` | first step: the node's stage files aside | if `sweep/node402/` exists it is moved (never deleted) to `aside/node402_<YYYYmmdd-HHMM>/sweep/node402/`; the queue refuses when that folder exists and prints the rs_node_log.md line (see "Fallback conditions") | stop on a refusal |
| | `score_g0` | queue: V1a both halves | stop |
| | per knob `train_k<n>`, `score_k<n>` | k11: skipped when `gates/k11_sample.json` has pass false (the runner `rs_sweep stage1` enters them as failed); the graphs refuse by themselves | a crash stops; a failed run is a recorded outcome |
| | `stage1`, `v1c_a_stage1`, `v1c_b_stage1` | `rs_vp` part (a) then (b) (part (b) refuses without (a)) | stop before stage 2 |
| | `train_grid<n>`, `stage2`, `v1c_a_stage2`, `v1c_b_stage2` | SWEEP-NONE ends the queue (exit 0; G stays) | stop before stage 3 |
| `rs_node_plan.sh` | see "The node plan" | `gates/n1.json`, `n2.json`, `n3.json`, `v1a_train.json`, `v1a_scorer.json` pass; deadlines `RS_START_BY` and `RS_END_BY` (date-times) | `RS_NODE_STOP`: fallback |
| `stage3` | `train_top` | queue: V1a, V3a (the runner `rs_stage3 verdict` also refuses without it), V1c of the sweep machine (`scores/<m>/rederive_partitions.json`, `sweep/<m>/rederive_stage2.json`); SWEEP-NONE or TOP-NONE: note, exit 0 | stop |
| | `verdict`, `v3c` | `rs_vp stage3` (exit 1 on disagreement) | stop: stage 4 does not run |
| `stage4` | `train_c`, `held_c` | queue: V1a, V4a (both halves; `rs_full` also refuses), V3c (`stage3/rederive_stage3.json`) | stop |
| | KEEP-G, or no stage 3 file (F = G) | no training, no held inference. Gates: V4a scorer half (`gates/v4a_scorer.json`); V3c under KEEP-G, V1c of the sweep machine (`sweep/<m>/rederive_stage1.json`, and `rederive_stage2.json` when stage 2 ran) with no stage 3 file. Two CPU steps: `verdict` = `rs_stage4 verdict --candidate G`, then `v4b` = `rs_vh stage4 --candidate G`; G's outcomes are this round's stage 4 verdict on G's saved held outputs, checked by V4b (rule §4.4, last paragraph). No report, no analysis | stop: no outcome stated |
| | `verdict`, `v4b` | `rs_vh stage4` | stop: no outcome stated, the analysis does not run |
| | `report` (soft), `analysis_c` | after the verdict file (rule §4.4) | report failure is noted |
| `ablation_dev` | `train_<removal>` | queue: V1a, **V3a first** (see run note g), then the re-derivation behind F, also for F = G: V3c when `stage3/verdict.json` exists (PICK or KEEP-G), else V1c of the sweep machine (`sweep/<m>/rederive_stage1.json`, and `rederive_stage2.json` when stage 2 ran); a graph removal whose `gates/ablation_graphs.json` item failed is skipped with a note (run note b) | a trainer failure of one arm is noted, the others go on; scoring starts when every arm has a `run.json` |
| | `score_dev`, `v_ablation` | `rs_ablation dev` and `rs_vp ablation` | stop |
| `ablation_held` | weekday rule, `train_<removal>`, `held_<removal>` | queue: weekday rule; V1a, V4a, V3a; the re-derivation behind F, also for F = G (V3c when `stage3/verdict.json` exists, else V1c of the sweep machine, as in `ablation_dev`); for F = c*: `stage4/verdict.json` names F, V4b (`stage4/rederive_stage4.json`) | an arm's failure is noted, the others go on |
| | `score_held`, `v_ablation` | `rs_ablation held` refuses an arm without held outputs and without a failure record | stop |

The weekday rule: `ablation_held` refuses to start on a weekday (Monday to Friday) before 19:00 Amsterdam time unless
`RS_FORCE_HOUR=1`. This is stricter than the spec's loud window (10:00 to 19:00): it follows the ticket. The fallback's day
rule (rule §4.2): its first knob's runs are timed with nvidia-smi power sampled every 10 s; on a weekday between 10:00 and
19:00 the queue stops after that knob unless `RS_DAY_OK=1` (under 200 W) or `RS_FORCE_HOUR=1`; otherwise start it again from
19:00. The overnight runs end by 11:00; the heavy local GPU runs (the fallback, the held ablation) stay out of weekdays 10:00
to 19:00.

## The node plan (`rs_node_plan.sh`)

`RS_DRY=1 bash rs_node_plan.sh stage1` prints it. Real mode is for the run chat, from the merged main checkout, after N1 and
N2 passed (the controller's) and `das6_sync_rs.py --inputs --run` was done. The commands, in order:

1. Stage 1: `gates/n1.json` and `n2.json` pass, and V1a (`gates/v1a_train.json`, `v1a_scorer.json`, local files, as the local queues check
   them); now is not after `RS_START_BY`; `cluster status --node node402`; `cluster sync`.
   Deadlines: `RS_START_BY` and `RS_END_BY` are full Amsterdam date-times `YYYY-MM-DD HH:MM` (the old `HH:MM` form is refused). Unset,
   they are the rule's 14:00 and 22:00 today, which is strict. Rule §4.2, exception of 2026-10-27 15:31 (user): the run chat sets
   `RS_START_BY` to the real stage 1 start on the node and `RS_END_BY` to that start plus 8 hours; the real mode notes it when
   they are unset. A deadline is a real check (`RS_NODE_STOP`, then the local fallback); DRY mode only notes what it would do.
2. Three shards of G, G-repeat (both first, in shard a) and the 39 stage 1 runs (14, 14 and 13 runs), one
   `cluster launch --node node402 --tag <rsMMDD>-stage1-<a|b|c> -- bash scripts/run_rs_node.sh shard --configs ... --seeds 0`
   each (one GPU each; the command has no shell metacharacters), then `cluster watch` and `cluster pull --tag` for each, then
   `das6_sync_rs.py --fetch scores/node402`.
3. N3 (`run_rs_node.sh n3`) and `--fetch gates/n3.json`; a failing N3 stops the plan (fallback).
4. V1c part (a) on the node (`run_rs_node.sh vp`), fetch `scores/node402` (it holds `rederive_partitions.json`); a disagreement stops.
   (Stage 2 does this step again; see the move above.)
5. Locally `rs_sweep stage1 --machine node402`, then `rs_vp sweep --machine node402` (part (b)).
6. `rs_node_plan.sh stage2`: needs `gates/n3.json`; the grid's new points from `sweep/node402/stage1.json` `grid.new` in three
   shards; the same launch, watch, pull and fetch; part (a) again on the node over all runs; locally `rs_sweep stage2 --machine
   node402` and `rs_vp sweep`. Stage 2 is checked against `RS_END_BY` at its start and again right before `rs_sweep stage2`.
   The node's `rs_vp partitions` rewrites `scores/node402/rederive_partitions.json` (now all runs; it is the only file that job
   writes). Before the second fetch the plan moves the home copy from stage 1 to `rederive_partitions.stage1.json` (a move; it
   stops if that name exists with other content), else the fetch would refuse a differing local file. The readers keep reading
   `rederive_partitions.json`. A full second run of stage 2 after a good fetch stops at that move: the run chat decides.

Fallback conditions (rule §4.2): N1, N2 or N3 fails, stage 1 cannot start on the node by `RS_START_BY`, stage 2 has not ended by `RS_END_BY`,
or the node is lost and one rerun (a new tag) does not fix it. Then run `rs_queue_fallback.sh`; the node results are reported only.
If the node wrote stage files before the fallback (for example stage 1 ended there and stage 2 missed `RS_END_BY`), the fallback's
first step moves `sweep/node402/` (never a delete) to `<outputs>/aside/node402_<YYYYmmdd-HHMM>/sweep/node402/` and prints one line
`RS_FALLBACK_NOTE rs_node_log.md entry: ...`: add it to `rs_node_log.md`. The readers that decide F (`rs_full.sweep_ended_without_candidates`,
`rs_ablation.resolve_final`, `rs_stage4.stage_candidate`, `rs_vp` development ablation) count every machine that has a
`sweep/<m>/stage1.json`; without the move a node GRID-n beside a local SWEEP-NONE would make "F cannot be resolved". The queue
refuses when the target folder exists (nothing is changed). `scores/node402/` is read by none of them and stays; the files in
`aside/` stay readable for the report. A dry run only says what it would move.
The node's outputs stay on the node: add a pending row with their size (`cluster du --node node402`). G0 and TOP's weights and
exports are copied home before 2026-10-30 (the reservation ends about 2026-10-31 00:40). No decision compares a node number
with a local number.

## Pending deletions

Rows go to the pending list, once per folder, from the `*.pt` files under it; nothing is deleted: the V1a regression runs
`regression/v1a_s0` and `v1a_s1` (`stage0` queue); the V4a replay `regression/v4a_run` (`stage4`); every fallback run of
stages 1 and 2 not in TOP (`fallback`). Kept (rule §4.5): G0, TOP's runs, every stage 3, stage 4 and ablation run, G's seed 4.

## Expected times (8 CPU threads, one RTX 3090)

From 12-27's status files (`rg_queue.md`: three sample seeds of one arm in one call 387 s; four arms one seed 187 s; G five full
seeds 631 s, held 15 s, scorer calls 1.5 to 9 min, report 18 min) and the spec's cost lines (D22). `~` marks an estimate.

| Queue | Minutes |
|---|---|
| `stage0`: seed 4 ~3, held ~2, V0c, V0d and verdict ~5, `rs_vh stage0` ~10, analysis ~15 | ~35 |
| `fallback`: 39 runs ~1.5 each plus scoring ~1.5 each, about 2 to 2.5 h; stage 2 grid up to 72 runs more (up to 3.5 h); V1c ~15 each | 3 to 6 h |
| node: 14 runs per GPU ~1.5 to 2 h with scoring on the node's CPU; stage 2 about the same | 2 h + 2 h |
| `stage3`: nine runs ~15, verdict ~10, `rs_vp stage3` ~10 | ~35 |
| `stage4`: five seeds ~11, held ~1, verdict ~3, `rs_vh stage4` ~15, report ~18, analysis ~15 | ~65 |
| `ablation_dev`: 24 runs (spec: about 17 GPU minutes), scorer ~12, `rs_vp ablation` ~10 | ~40 |
| `ablation_held`: 40 full-scale seeds (spec: about 80 GPU minutes, from 19:00), scorer ~15, `rs_vh ablation` ~15 | ~110, ends by about 23:00 |

## Held-ledger rows to add (plain records; `docs/superpowers/held_ledger.md` format; the run chat fills the values after V0b, V4b and the held V of the ablation agree; the queue writes nothing)

| Row | When | Fill in |
|---|---|---|
| H20 stage 0 | after `stage0` ends | date; ArtELingo held split, 12,281 paintings (H11's list); Qwen3-VL-Embedding-2B default image, G = GS-0.02 (sample seeds 0 to 4: 12-27's weights and this round's seed 4, held inference new; full seeds 0 to 4: 12-27's held outputs, read in H19); PercepT B1 to B6 saved; purpose: stage 0, G's emotion claim at both scales (rule §4.1, spec D7); held list `a0b1139c6056fd858ddbd2ee1dd4bafd1453e590818c6aa59883364c0288a876`; round folder `src/test/20261027_recipe_sweep/` at the commit; `stage0/verdict.json` SHA-256 and outcomes (G-ES-*, G-EF-*, E0), V0b agrees; report path |
| H21 stage 4 | after `stage4` ends | as H20 for c* at full scale, seeds 0 to 4 (c*'s name); purpose: d_T per aspect against the instructed image, PE-* emotion margin over PercepT (rule §4.4); `stage4/verdict.json` SHA-256 and outcomes per aspect, V4b agrees; recipe chosen before this read (the stage 3 verdict SHA-256) |
| H22 held ablation | after `ablation_held` ends | the eight removals of F at full scale, seeds 0 to 4; purpose: the held ablation (rule §3.5); `ablation/held.json` SHA-256 and `inputs.ablation_graphs_sha256`; F's name; the removals that did not run, with the cause |

Under KEEP-G or SWEEP-NONE or TOP-NONE there is no H21 and no new held read: G's held outcomes come from this round's stage 4 verdict on G's saved held outputs (`stage4` queue: `rs_stage4 verdict --candidate G`, then V4b `rs_vh stage4 --candidate G`; rule §4.4, last paragraph), which equal 12-27's by V4a (H19's read); G's emotion claim is stage 0's.

## Run notes (decided in the build)

a. `shared_trunk` removal: one cohort, one trunk per aspect (build 8's reading).
b. A failing graph check stops only its own removal (`ablation_graphs.json` item false: the queue skips that arm with a note, `rs_ablation`
   lists it as failed with the cause); the ablation outputs record the gate file's SHA-256 as `inputs.ablation_graphs_sha256`.
c. F = G without a stage 3 file: `rs_full.sweep_ended_without_candidates` (every machine with a stage 1 file ended in SWEEP-NONE or
   TOP-NONE) is the one rule for tickets 08, 09 and 10.
d. The genre outcome of 12-27's G at stage 4 reads T-LEVEL by the rule (lower bound -0.0105 > -r_genre -0.0128), but the whole
   interval is below zero: write it as "below the instructed image by less than one seed range", never "matches T".
e. `gates/v1a_scorer.json` was written by `rs_sweep.py` before ticket 06's one-line edit (`written` in the ignore list of
   `_keep_or_write`); its `code_sha256` (3f1803...) now differs from `rs_sweep.scorer_code_sha256()` (5f6fb8...), checked on
   2026-10-27. No guard reads that hash (the queues read only its `pass`), and the edit does not touch the scoring, so the run
   chat need not rerun it. A rerun is cheap (CPU, about a minute) if a matching hash is wanted: move the file aside first
   (`mv` to `<outputs>/aside/`, never delete), then `rs_sweep check v1a`.
f. `gates/v1a_train.json` must never be rewritten after stage 0's seed 4 trains (V0d compares mtimes; `rs_train regress v1a` writes it).
g. V3a passes before the development ablation, whatever stages 1 and 2 give (`rs_ablation` refuses without it); the `ablation_dev`
   queue checks `gates/v3a.json` first and says so.
h. The ablation outputs record `inputs.ablation_graphs_sha256`, the SHA-256 of the graph gate file they relied on (`ablation/dev.json`
   and `held.json`; the queue notes it after scoring).
i. On the node, a k11 refusal stops the rest of its shard (`run_rs_node.sh shard` has `set -e`). Left as it is: the k11 gate
   (`gates/k11_sample.json`) passes (18 of 18 arrays equal), so no k11 run is refused. If it ever fails, run the other shard
   members again in a new shard without the `k11=` names.

## Recovery (after any stop)

1. Read the last lines of `<outputs>/queue_<name>_status.txt` and the named log in `<outputs>/logs/`.
2. A runner crashed or the machine stopped: start the same queue again; finished runs and scores are skipped.
3. A gate failed: trace the cause; move the failed gate file aside to `<outputs>/aside/` (`mv`, never delete) after the fix.
   A failed re-derivation (`rs_vp`, `rs_vh`): both values go in the report, the fix is by the rule's text; move the stale verdict
   and its `rederive_*` file aside, then start the queue again.
4. A failed run (non-finite loss, V1b after its one rerun) stays failed and is reported (rule V1d, V3d, V4d).
5. GPU lock wait over 4 h: the step ends with a failure line; start the queue again when the GPU is free.

## Smoke (the controller's)

```bash
cd /project/CoSiR/src/test/20261027_recipe_sweep   # the merged main checkout
RS_DRY=1 RS_OUTPUTS=/tmp/claude-0/<scratch>/dry bash rs_queue_stage0.sh      # any queue, any time; writes nothing
export RS_SMOKE=1 RS_OUTPUTS=/tmp/claude-0/<scratch>/smoke
bash rs_queue_stage0.sh      # trains seed 4 at 2 epochs and ends: rs_stage0 and rs_vh have no smoke mode
# stage 3: first write a smoke sweep/<m>/stage2.json (stage 'stage2', smoke true, outcome TOP, p_sha256, machine, top
# [{rank, name}]) under the scratch outputs, then: RS_SWEEP_MACHINE=<m> bash rs_queue_stage3.sh
```

Add `RS_DEVICE=cpu` for a plumbing test without the GPU (one smoke run takes about 40 s on the CPU).
