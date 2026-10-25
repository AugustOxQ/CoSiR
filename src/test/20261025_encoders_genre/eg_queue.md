# Run notes: queues of the encoders-and-genre round (ticket 16)

Run from the merged main checkout (`/project/CoSiR`), never from a worktree: `scripts/extract_features.py` and the X0
git archive are found from the queue script's own place. Start a queue with
`nohup bash src/test/20261025_encoders_genre/eg_queue_<name>.sh > <scratch>/<name>.out 2>&1 &` and read its status
file. Queues never wait for input. The driver that decides between steps is `eg_run.py` (read its docstring).

## Environment (set by `eg_queue_common.sh`)

| Variable | Default | Use |
|---|---|---|
| `EG_OUTPUTS` | `/data/SSD2/cosir_r10/20261025_encoders_genre/outputs` | outputs folder; passed as `--outputs` to every runner |
| `EG_SMOKE=1` | off | adds `--smoke` where the runner has it; `EG_OUTPUTS` must be under `/tmp/claude-0/` (else exit 2); pending list goes to `<outputs>/pending_deletions.md` |
| `EG_PYTHON` | CoSiR env python | |
| `EG_LOCK_WAIT`, `EG_GPU_LOCK` | 14400 s, `/tmp/gpu0.lock` | GPU steps: `flock -o -w $EG_LOCK_WAIT $EG_GPU_LOCK env CUDA_VISIBLE_DEVICES=0 python -u ...` |
| `EG_CPU_LOCK` | `/tmp/eg_cpu.lock` | the report queues (`cpu_i`) run one at a time |
| `EG_DEVICE` | `cuda` | `cpu` only for a plumbing test |
| `EG_DRY=1` | off | print each step's command, run none (plumbing check only) |
| `EG_PENDING` | `/project/CoSiR/.scratch/pending_deletions.md` | where stage iii appends weights that may go (never deleted) |

Always `PYTHONDONTWRITEBYTECODE=1`, 8 threads, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. CPU steps hide CUDA.

## Status lines

`EG_<QUEUE>_START <date> smoke=<0|1>`, one line per step `label=<exit> seconds=<n>` (also in
`<outputs>/queue_<queue>_status.txt`; logs in `<outputs>/logs/<queue>_<label>.log`), `EG_GATE <name> ok: ...` for a gate
that holds and `EG_STOP ...` for one that does not, `EG_<QUEUE>_NOTE ...`, and `EG_<QUEUE>_STOP <reason>` when the queue
ends because of a failed step or gate; the queue then ends with `EG_<QUEUE>_DONE failed=1`. Queue names: `STORES`,
`STAGE_I`, `CPU_I3`, `CPU_I4`, `CPU_I5`, `STAGE_II`, `STAGE_III`, `GPU`. A failed gate stops what the rule says stops and
nothing else; the table says which.

## Queues and order

| Queue (`eg_queue_*.sh`) | Resource | What | Stops |
|---|---|---|---|
| `stores` | GPU, CPU | X0 (`eg_x0`); the eight extractions (K6 CLIP B/16, K7 CLIP L/14, K8 SigLIP 1, K5 DINOv2-large `--image-only`; each for `artelingo_full` and `artelingo28`); V5 (`eg_v5 --calibrate`, then each store) | X0 fails: no store is written, queue ends (stage i stops; ii with K2 to K4 and iii go on). A failed extraction or V5: noted with the cause; full split failure removes that candidate's outcome, A28-only failure removes only its A outcome |
| `stage_i` | GPU, CPU | budget line; gate X0; X1 (three halves); per candidate K6, K7, K8, K5: `gate stage3`, student train and infer, matched arm; `eg_score --stage 3`; `check-verdict 3`; `eg_v2 --stage 3`; X2; per candidate with an S1 outcome other than S1-NO: `gate v2 --stage 3`, `gate stage4`, student and matched at full scale; `eg_score --stage 4`; `eg_v2 --stage 4`; stage 5 (B1 and B6 sample inference when `eg_run a28` lists the sample scale; `eg_score --stage 5`; `eg_v2 --stage 5`) | X0 or X1 fails: stage i ends. V5 failed on the full split: that candidate gets a block without an outcome, with the V5 cause. A28-only failure: both arms use `--kinds held` at both scales; only A has no outcome, with the V5 cause. No stage 3 outcome: no A outcome. X2 fails: no stage 4, stage 5 at the sample scale. A V2 that disagrees: stage i ends |
| `cpu_i N` (3, 4 or 5) | CPU | `gate v2 --stage N`; `eg_report_i --stage N` (about 30 to 40 min per candidate) | For outcomes, V2 gate fails: no report. With no candidate outcome: report the causes and stores without re-deriving |
| `stage_ii` | GPU, CPU | P inference and VL1 (b), (c), VL3 (`eg_pinfer all`); VL1 (a) (`eg_lift check`); `gate vl1`; `eg_run lift-score`; `eg_vl2`; `gate vl2`; `eg_stats stats` (the P table); `eg_stats held` (needs the reports of `cpu_i`; `bash eg_queue_stage_ii.sh held` runs it alone) | VL1 fails: stage ii ends. Stage 3 not finished for a candidate that is not deferred (and X0, X1 hold): ends at once. X0 or X1 failed: runs with K2 to K4 and builds the held table without waiting for stage i reports. Missing gate files: stops before P inference |
| `stage_iii` | GPU, CPU | VG1 (`eg_train3 regress vg1`, `eg_score3 check vg1`); the five variants; `eg_run score3 first`; `eg_vg2 --phase first`; the two extras; `score3 all`; pending list; `eg_vg2 all`; when triggered: winner and R6-5 on seeds 1 to 3, `eg_train3 parity`, `gate parity`, `eg_score3 fresh`, `eg_vg2 fresh`; `eg_score3 report` | VG1 fails: no variant runs. A VG2 that disagrees: ends. A failed parity: no C outcome (the scorer records the cause), the queue goes on |
| `gpu` | all | `stage_iii`, `stores`, `stage_i`, `stage_ii` in this order (stage iii needs no new store, spec §3.5), the CPU report queues 3, 4, 5 in the background one after the other once stage i ends, then the held table | none: a queue that stops is noted and the chain goes on; exit 1 if any queue stopped |

Every queue can be started again after a stop: finished work is skipped (a store whose `index.json` exists, a gate file
that exists, a trained run, a score or re-derivation file). Move a failed regression file aside to a dated folder before
rerunning after a fix; delete nothing. X0, V5 and the gate files that exist in the build outputs (`stores/x0.json`,
`part2/regression/x{1,2}_*.json`, `stage_ii/vl1_scorer.json`, `stage_iii/{vg1,vg1_scorer}.json`) are skipped.
Stage iii first checks every existing `score_{first,all,fresh}.json` with `eg_score3.py check-code`: a stale or missing
top-level `code_sha256` stops with "move <file> aside, then rerun", before training or skipping scores. Manual scoring
also refuses a stale destination. The hash covers `eg_score3.py` and its transitive top-level imports in this round
and the RB round (static AST traversal, excluding lazy imports inside functions); it stays outside `identity`, whose
input hashes VG2 compares. The non-smoke parity step rechecks the weights,
but keeps `fresh_parity.json` byte for byte when its result is unchanged (commit, dirty state and timing are bookkeeping).

## Gates and stops (`eg_run.py gate NAME`, exit 1 with an `EG_STOP` line)

`x0`, `v5 --cand K`, `x1`, `x2`, `stage3 --cand K` (X0, V5 of K, X1), `stage4 --cand K` (X2, S1 outcome not S1-NO),
`v2 --stage N` (agrees, not older than the verdict, every block with an outcome re-derived), `vl1`, `vl2`, `vg1`,
`vg2 --phase first|all|fresh`, `parity`. Before each candidate's stage 4 the queue also calls `gate v2 --stage 3`
(agent default; the rule does not require it). If `gate stage3` fails because V5 has no entry for K (the stores queue
has not run), the candidate is skipped and gets no block; if V5 says fail, its block is written without an outcome.

## Rerun of a run that failed VG3 (VG4)

After every `eg_train3 train` the queue looks at each run's `run.json`: a run with `status failed`, `failure_kind vg3`
and no `rerun_of` gets one `train --variants V --seeds S --rerun-failed` call (same seed and code; the trainer moves the old
folder aside). A run already rerun is never called again (the trainer would refuse). The rerun comes before the scoring
(`eg_score3` refuses a VG3 failure that was not rerun). A run that stays failed after its rerun goes into the scoring as a failed
run. If an otherwise successful `run.json` has exports that still fail VG3 after the forced rerun, score it by hand
with `--failed`. Both CLIs accept repeatable flags, but their fresh-seed keys differ:

- Scorer: `--failed VARIANT=CAUSE` for `first`, `all` and `fresh`. On `fresh`, it checks every seed of that variant and
  marks only those whose exports fail VG3; it refuses the declaration if none fails.
- VG2: `--failed VARIANT=CAUSE` for seed 0 (`first`/`all`); `--failed variant/sN=CAUSE` for each failed fresh seed.
  The variant ID keeps its real case (for example `GT-4/s2`). Repeat the fresh declaration for every affected seed.

A hand-run scorer with `--failed` needs VG2 run by hand with the matching flag; the queue supplies no `--failed` flags.
For seed 0, for example, run `python eg_score3.py first --outputs "$EG_OUTPUTS" --failed 'GA-1=VG3 after forced rerun'`
and `python eg_vg2.py --phase first --outputs "$EG_OUTPUTS" --failed 'GA-1=VG3 after forced rerun'` (repeat for `all`
if that variant is still included). For fresh seeds, assuming GT-4 won and only its seed 2 still fails:

```bash
# fresh VG3 failure
python eg_score3.py fresh --outputs "$EG_OUTPUTS" --failed 'GT-4=VG3 after forced rerun'
python eg_vg2.py --phase fresh --outputs "$EG_OUTPUTS" --failed 'GT-4/s2=VG3 after forced rerun'
```

Add `--smoke` to both commands for a smoke tree. These declarations require the one forced rerun first; they do not
permit another training run. After VG2 agrees, run `python eg_score3.py report --outputs "$EG_OUTPUTS"` (matching
`--smoke` if used). `report` itself requires an agreeing VG2 file no older than its score for `first`, `all`, and
`fresh` when triggered.

## Stage iii recovery after a scorer or VG2 fix

After fixing a VG2 disagreement or a stale-file refusal, preserve the old evidence in a dated folder before rerunning.
From this round's code folder, set `EG_OUTPUTS` to the output tree being repaired and use the following conservative
recipe. It moves aside all three scores, their VG2 records, both reports, the partition/bootstrap/reading caches, and
the VG1 scorer result so that VG1 is checked on the final code. Nothing is deleted. Keep `train/`, `vg1.json` and
`fresh_parity.json`: saved training exports/weights stay fixed under VG4, and unchanged parity evidence keeps its hash.
If the parity result itself was wrong, trace the cause and move that file aside too before the non-smoke parity rerun.

```bash
# stage iii recovery
: "${EG_OUTPUTS:?set EG_OUTPUTS to the stage iii output tree}"
stage="$EG_OUTPUTS/stage_iii"
mkdir -p "$stage/aside"
aside=$(mktemp -d "$stage/aside/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
for name in score_first.json score_all.json score_fresh.json rederive_vg2_first.json rederive_vg2_all.json rederive_vg2_fresh.json reported.json reported.md partitions vg1_scorer.json; do
    if [[ -e "$stage/$name" ]]; then mv "$stage/$name" "$aside/"; fi
done
"${EG_PYTHON:-python}" eg_score3.py check vg1 --outputs "$EG_OUTPUTS"
bash eg_queue_stage_iii.sh
```

The queue reruns `first`, VG2 first, `all`, VG2 all, and, when triggered, parity, `fresh`, VG2 fresh, then `report`.
Finished training runs are reused. Keep the existing queue settings (`EG_SMOKE`, `EG_DEVICE`, `EG_PYTHON`, workers).
When manual `--failed` declarations are needed, run the scorer and VG2 phases by hand with the matching flags above,
gating each phase with `python eg_run.py gate vg2 --phase PHASE --outputs "$EG_OUTPUTS"` before continuing; finish
with `report`. A smoke recovery still needs the copied parity evidence. Stage iii weight candidates for later removal
remain on the pending deletions list; this recovery moves only derived files and never removes weights.

## Deferral for time (D15)

The `budget` line is the first output of `stage_i`. If its high end is after 11:00: `python eg_run.py defer
--estimate-end HH:MM --outputs ...` defers the next of K5, K8, K7, K6 (one per call; call `budget` again).
The estimate and decision are recorded in `part2/deferral.json`; this is the only deferral route.
Deferred candidates get no store, no stage 3 and no stage 4 in the run; stage ii starts when
every candidate not deferred has its stage 3 block. A deferred candidate is never dropped: `python eg_run.py defer --release K`,
then `eg_queue_stores.sh` and `eg_queue_stage_i.sh` again (they train and score only
candidates without a block; stage 5 includes the new model). Stage ii for the deferred encoders is later reported as missing
(rule §2.2); nothing is added after P outputs exist (VL4).

## Replace a kept no-outcome block after a fix

`eg_score` refuses to recompute a finished block. After a fix, move the stale partition caches of that candidate aside
(they are under `<outputs>/part2/partitions/<scale>/<arm>/`; `mv` them to a dated folder, delete nothing), then
`python eg_run.py replace --stage 3 --cand K6 --reason "<V-check cause fixed>" --outputs <dir>` (add `--print-only` to see the
`eg_score.py --replace` calls first; a replaced stage 4 block that moves the candidate's stage 5 scale replaces its stage 5
block too). Then rerun `eg_v2` for the stage and the `cpu_i` queue.

## Weights below target

After `score3 all` the queue appends a row per variant listed in `score_all.json` `pending_deletions` (A_genre below
T_g - 0.0128, or a failed run) to the pending deletions list, once (rows are not repeated). Seed-0 weights of kept variants and all
fresh-seed weights are not listed. Nothing is deleted.

## Smoke (the controller's; scratch under `/tmp/claude-0/`)

`EG_SMOKE=1` shrinks only what the runners shrink (seed 0, 2 epochs, 20 draws, 1,000 P rows, `--limit 16` extraction).
The regressions (X1, X2, `eg_train3 regress vg1`) and X0, V5 and `parity` have no smoke mode. Use the build's gate files by
copying them into the scratch folder first. With `S=/tmp/claude-0/<scratch>/eg` and `R=/data/SSD2/cosir_r10/20261025_encoders_genre/outputs`:

```bash
S=/tmp/claude-0/<scratch>/eg ; R=/data/SSD2/cosir_r10/20261025_encoders_genre/outputs
export EG_SMOKE=1 EG_OUTPUTS=$S ; cd /project/CoSiR/src/test/20261025_encoders_genre
mkdir -p $S/stores $S/part2 $S/stage_ii $S/stage_iii
cp $R/stores/x0.json $R/stores/v5.json $S/stores/          # v5.json holds only the calibration until the new stores exist
cp -r $R/part2/regression $S/part2/                         # X1 and X2 gate files (real, from the build)
cp $R/stage_ii/vl1_scorer.json $S/stage_ii/
cp $R/stage_iii/vg1.json $R/stage_iii/vg1_scorer.json $R/stage_iii/fresh_parity.json $S/stage_iii/
bash eg_queue_stores.sh        # X0 skipped (x0.json), smoke extraction into $S/stores/_smoke, V5 only for real stores
bash eg_queue_stage_iii.sh     # five variants, first, VG2 first, extras, all, VG2 all, fresh seeds, fresh, VG2 fresh, report
bash eg_queue_stage_i.sh       # needs V5 entries for the new stores in $S/stores/v5.json (a real 'stores' run, or the 12-24 stores for K2/K3 only)
bash eg_queue_cpu_i.sh 3 ; bash eg_queue_cpu_i.sh 4 ; bash eg_queue_cpu_i.sh 5
bash eg_queue_stage_ii.sh      # P cut to 1,024 rows; VL1 (b)/(c) in smoke are the runner's own
bash eg_queue_gpu.sh           # or all of the above in one
```

Stage iii alone is the cheapest end-to-end test (it needs no store). The smoke stage i needs the new stores in
`/data/SSD2/pre_extract` (V5 and the runners read them there): run the real `stores` queue once first, or smoke only
`stage_iii`, `cpu_i`/`stage_ii` on fakes. `EG_DRY=1` prints all commands without running (used for the build check).

Replace smoke (after a stage 3 block without an outcome exists in the scratch folder; `eg_run replace` has no smoke
flag, so it scores with the full 2,000 draws, a few minutes per candidate):
```bash
python eg_run.py replace --stage 3 --cand K6 --reason "smoke: cause fixed" --outputs $S --print-only
mv $S/part2/partitions/sample/K6 $S/part2/partitions/sample/K6.stale_$(date +%s)   # not delete
python eg_run.py replace --stage 3 --cand K6 --reason "smoke: cause fixed" --outputs $S --workers 8
```
