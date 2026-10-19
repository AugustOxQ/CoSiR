# Ticket fix — Final-review fix wave

Branch `r10-ms-fix`, based on `r10-model-sheet`. Ticket identifier is `fix` (the follow-up to numbered tickets
01–04). Implemented F1, F2, F3, F5, F6, F7, F8 and F9 only. Review 4 remains the controller's manual back-fill.
Single agent; CPU only; no CCG records, other models, unit tests, push, merge or deletion.

## Definitions settled

- F1: the emotion gate considers only rows with `config.inputs == 'default'`; its threshold is the sheet's full
  PercepT emotion AMI mean plus .01, displayed to six decimals. Why: instructed I1 cannot establish the default-input gate.
- F1: results record `emotion_gate_reached` and `emotion_gate_threshold`. Why: the boolean names the actual gate
  and the numeric threshold remains available without rounding.
- F2: `SCORER_VERSION` is SHA256 of concatenated `ms_score.py` then imported `pav_score.py` bytes. Every metric,
  direct-API reference and head-cache identity includes it. Why: editing either scorer source invalidates reuse.
- F2: head-cache filenames hash the full identity; stale direct references are recomputed. Why: existing caches
  cannot block or supply scoring under the new code, and older metric/head cache files remain on disk.
- F2: stage 0 records current trainer `code_version` and `scorer_version`, lists every stale array path, and
  fails a separate `stale` gate while still checking the original numerical gates. Why: a bitwise-identical older
  array does not prove that the current trainer ran.
- F2: `--require-pass` reads only the stage0 manifest and compares PASS and both current versions; stage 1 runs
  this first and finishes immediately on failure. Why: training cannot bypass the code-bound stage 0 prerequisite.
- F3: `head_loss` defaults to including mask L1; dev evaluation disables only that addition. Form-c history logs
  its current mask penalty separately, and best states/patience use unpenalized dev loss. Why: mask shrinkage
  must not drive validation selection. CPU form a still directly returns untouched `ags_head.pair_loss`.
- F5: aggregation checks one trainer code version over all aspects and seeds of each named row and reports the
  row and versions on failure. Why: a per-aspect assertion would still permit mixed head versions.
- F6: picks require all 33 canonical seed-0 heads from the 11 `ROWS` entries. Stage 2 regenerates its manifest
  through this check before validation/training and finishes on failure. Why: an old manifest cannot bypass an
  incomplete stage 1. Existing rankings and derived-row exclusions are unchanged.
- F7: existing files are verified against their recorded configs before comparing with current configs; any
  missing file or config mismatch trains and saves the whole cohort. Each file is replaced through the existing
  atomic JSON/NPZ writes, with one replacement line per NPZ. Why: every head must come from the replayed cohort;
  corrupt files still crash instead of being accepted. An entirely matching cohort still skips.
- F8: aggregates retain each aspect's seed-0 own-label image AMI separately; D17's D2–D5 selection averages
  only those values. Why: available extra seeds must not change the D-row comparison. T4/T5 triggers still use
  row means against T1's mean plus SD, including the existing trigger tie rule.
- F9: every table has Inputs immediately after Seeds; bars show `—`. Why: readers can distinguish default
  inputs from instructed inputs even when rows are ranked together.

## Commands run

Read the final review, binding index, round log/defaults, prior build notes and named code in order; inspected
`ags_head.verify_output` read-only. Routine inspection used `cat`, `sed`, `rg`, `ls` and git status/diff. All writes
were in the round folder or this ticket's smoke folder. Earlier artifacts came from absolute main-checkout paths;
the integration worktree was read-only.

Abbreviations below:

```text
R = /project/CoSiR-wt-msfix/src/test/20261019_model_sheet
S = /tmp/claude-0/-project-CoSiR/56b9b751-6946-4c76-b52d-3761e520be8f/scratchpad/ms_smoke/fix
B = S's parent (ms_smoke)
```

Every Python process used:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='' timeout 235 /root/miniconda3/envs/CoSiR/bin/python
```

Arguments and log names (all logs under S):

```text
-u R/ms_train.py --rows A1 --aspects genre --seeds 0 --device cpu --max-epochs 2 --outdir S/full
    genre_full.log
-u R/ms_train.py --rows T1 T4 T5 D2 --seeds 0 --device cpu --smoke --max-epochs 2
    rows_smoke.log
-u R/ms_train.py --rows T5 --seeds 0 --device cpu --smoke --max-epochs 4 --outdir S/four
    t5_four.log
-u R/ms_train.py --rows T1 --seeds 0 --device cpu --smoke --max-epochs 2
    partial_resume.log (after the mv below)
-u R/ms_train.py --rows T1 --seeds 0 --device cpu --smoke --max-epochs 2 --outdir S/stale
    stale_resume.log (copied B/02/outputs/T1; all three configs stale)
-u R/ms_stage0.py --require-pass --outputs S/stage0_old
    require_old_before.log (exit 1)
-u R/ms_stage0.py --check --outputs S/stage0_old
    stage0_old.log (exit 1, expected stale gate)
-u R/ms_stage0.py --require-pass --outputs S/stage0_old
    require_old_after.log (exit 1)
-u R/ms_sheet.py --outputs S/sheet --score --sheet
    sheet.log
-u R/ms_sheet.py --outputs S/sheet --pick-seeds
    pick_seeds.log (exit 1, expected 29 missing heads)
-u S/verify_training.py
    verify_training.log
-u S/verify_resume.py
    verify_resume.log
-u S/verify_sheet.py
    verify_sheet.log
-u S/verify_stage0.py
    verify_stage0.log
```

Copied with `cp -a`:

- `/project/CoSiR-wt-msint/src/test/20261019_model_sheet/outputs/{stage0,stage0.json}` into `S/stage0_old/`.
- `B/03/outputs/I1`, `S/outputs/T1` and `S/full/A1` into `S/sheet/`.
- Fresh `S/outputs/T1` into `S/resume_before/`, then used `mv` to put its genre NPZ and JSON into `S/moved/`.
- `B/02/outputs/T1` into `S/stale/`.
- The four queue scripts plus stage0/sheet scripts into `S/queue_probe/`; linked its outputs to `S/sheet`.

Executed each copied stage-1/stage-2 queue under `timeout 235 bash`, with `PYTHONPATH=R` and the same CPU/thread/
bytecode environment. Both intentionally exit 1 before GPU calls; status files and logs stay in the probe folder.
The verification scripts make small synthetic in-memory inputs and scratch JSONs only; they are smoke diagnostics,
not repository unit tests. `verify_sheet.py` reuses actual precomputed bar scores for its controlled I1 counterexample.
`verify_stage0.py` recomputes one deliberately stale direct reference, then enforces replay without scoring.

`bash -n` passed for each `R/ms_queue_*.sh`; `git diff --check` passed. Reviewed the final diff and committed code
and this note with `r10 ms fix: Final-review fixes`. No command reached its 235-second timeout.

## Smoke numbers

Full A1 genre: 62,906 fit / 8,045 dev pairs, 123 steps per epoch, best epoch 1, 31.75 s through output save:

| Epoch | Train loss | Dev loss |
|---|---:|---:|
| 1 | 5.772502180768421 | 7.159665240316824 |
| 2 | 4.082042087861413 | 7.192388301728452 |

Both arrays are bitwise equal on uint16 views to main-checkout stage 1's `genre__R0o__s0.npz`:
10,240,000 values per side. All four losses are exactly equal.

T1: all 6 arrays bitwise equal to ticket 02 and all 12 train/dev losses exactly equal. T4 and D2 additionally
each have 6 bitwise-equal arrays and 12 exactly-equal train/dev losses against ticket 03. Totals: **20 arrays /
204,800,000 float16 values bitwise equal; 40 unchanged full-precision train/dev losses**. T1 second-epoch dev:
emotion 10.319878227382572, style 10.36016113246104, genre 8.194670393157256.

T5's six two-epoch training losses are also exactly unchanged against ticket 03. Each dev value plus its logged
penalty recovers the old formula within **4.18e-7** (float32 addition before versus after batch averaging).
Example, emotion epoch 1:

```text
dev_loss     10.883575047606
mask_penalty  0.995258152485
sum          11.878833200091
old formula  11.878833378905
abs delta     0.000000178814
```

| T5 head | 2-epoch best | 2-epoch best unpenalized dev | 4-epoch best | 4-epoch best unpenalized dev |
|---|---:|---:|---:|---:|
| emotion | 2 | 10.407165693065 | 4 | 9.873534564221 |
| style | 2 | 10.455514107927 | 4 | 9.868858937927 |
| genre | 2 | 8.082840531327 | 4 | 7.465163618471 |

All 18 T5 history entries (6 at two epochs, 12 at four) log a positive penalty. Best epochs and best losses equal
the minimum unpenalized dev entry for every head/run.

Partial T1 resume: all **3 heads / 6 files** have newer mtimes and fresh histories, **2 replaced lines + 1 saved
line**, and the moved genre pair reappears. Fully verified replay skips before model construction with all 6
mtimes unchanged. Stale complete T1 replay replaces all **3 heads**, successfully upgrading pre-fix configs.

Old stage 0: `FAIL stale`, exactly **7 stale paths**, no missing files. Original gates still pass **a=2/2,
b=36/36, c=45/45**, direct API equality true, **7/7 best epochs equal**. Require-pass exits 1 both before and
after the check. Synthetic require-pass covers missing file, non-PASS status, trainer mismatch, scorer mismatch,
and matching PASS with data loading, training and scoring forbidden.

Scorer-cache smoke: current identities replay without k-means; a synthetic scorer-version change prevents both
metric and head-cache reuse. A stale `pc_ami_reference.json` is recomputed exactly once, then replays without
scoring. Its updated identity contains the actual concatenated-source SHA256.

Sheet: 7 real heads / 14 arrays (A1 genre, T1 all aspects, I1 all aspects), three **13-column** tables with Inputs.
PercepT emotion bar is 0.077174 to six decimals; threshold prints **0.087174**. Controlled I1=0.2 and all default
emotion rows=0.05 produces **emotion_gate_reached=false** using the real bar scores. Actual smoke T1 is above
the gate; these truncated-training values are smoke results, not research conclusions.

Mixed trainer versions are rejected both across heads and across seeds, naming T1 and both versions. Incomplete
CLI seed picking exits 1 and lists all **29 missing canonical heads**; a synthetic complete 33-head population
succeeds. D17 chooses **D3 (seed-0 mean .5)** over **D2 (.2)** even when aggregate means favour D2 (.8 vs .1);
T4 still triggers from its row mean (.4), despite seed-0 .1 against T1 mean .2 / SD .01.

Queue probes end with exactly `MS_STAGE1_DONE require_stage0=1` and `MS_STAGE2_DONE manifest=1`; neither attempts
training. Shell syntax **4/4** passes.

## Open points and would remove

No further defect found. The controller still needs a real stage 0 regeneration under current code before stage 1;
the copied old run intentionally fails. Review 4's back-fill remains a manual controller action. GPU execution is
outside this CPU-only ticket; no GPU was used.

Deleted nothing. No repository removal candidates. After validation, would remove only this ticket's `S` scratch
folder (logs, diagnostics, copied/generated arrays and caches, preserved pre-resume files and queue probes).
