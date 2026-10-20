# Claim-test queues (ticket 09)

Run from this worktree; the scripts resolve their own round directory. They are
foreground queues, fail at the first failed step, and can be submitted with
`nohup bash ... > session-scratch/queue.log 2>&1 &` by the controller.
All GPU steps use `flock -o -w 14400 /tmp/gpu0.lock`; `CT_LOCK_WAIT` overrides
that wait. `CT_PYTHON` overrides the CoSiR Python executable. CPU work uses at
most eight threads and hides CUDA.

```bash
bash src/test/20261020_claim_test/ct_queue_stage0.sh
bash src/test/20261020_claim_test/ct_queue_stage1.sh
bash src/test/20261020_claim_test/ct_queue_cpu.sh
```

Wait for each queue to return success before starting the next. Step summaries,
including nonzero lock/worker exit codes and durations, are printed once and
appended to `outputs/queue_{stage0,stage1,cpu}_status.txt`; detailed stdout/stderr
is under `outputs/logs/`. Stage 1 records the local GPU/CPU end estimate against
11:00 in its budget log. The run chat owns the D23 decision to move PercepT to
node408, where a separate Stage 0 is required; these scripts do not submit
remote work.

Stage 0 trains seed 0 D2 and core (both Stage 1 versions, Stage 2 P1–P4), then
writes `stage0.json` with 23 individual results. The reference NPZ/JSON files and
recorded AMIs come from the main checkout, read-only. Scorer replay uses every
D2 head and P1–P4 image probabilities, each on all three labels. Inference replay
uses the 5,000 development names in evaluation order and the checkpoint device.
For a Stage 1 hash difference, the AMI escape checks both dependent image mappers
(P1/P2 or P3/P4), since Stage 1 itself has no image-only mapper.

A hash difference fails unless all three image-side AMIs differ by at most 1e-4
and the controller supplies a traced cause. Supply a JSON object keyed by the
failed regression item ID, for example
`{"regression/stage1/clip_roberta": "<cause established from the failure>"}`.
Recheck without retraining:

```bash
flock -o -w 14400 /tmp/gpu0.lock /root/miniconda3/envs/CoSiR/bin/python src/test/20261020_claim_test/ct_stage0.py --causes session-scratch/causes.json
```

A pass binds code, the rule, source hashes, training host and device. Changed
sources/code or a partial report fail V1. Stage 1 fixes seeds to 0–4; after any
held NPZ exists it refuses incomplete/changed training artifacts. Existing held
arrays are verified before inference can reuse them; damaged artifacts stop the
queue. A hash-verified non-finite paper output is preserved, skipped by inference,
and passed to the scorer to count as collapsed under V4. A V-check-forced rerun needs controller handling and a report entry under
V4; there is no automatic seed replacement or deletion.

The adapter `ct_run.py` supplies explicit paths to the merged modules without
editing them. Training, held artifacts, partitions and instructed-image stores
are written below this worktree's `outputs/`; verdict and diagnostics are written
in this round. Instructed-image stores go to `outputs/targets/`, and diagnostics
read them there. The queues never write to the main checkout or shared feature
cache. Diagnostics run after the saved verdict; independent V2 reproduction
remains the run chat's prerequisite for reporting an outcome.

The isolated smoke uses two seeds and one epoch, real optimizers/serialization,
5000-row development replay, tiny synthetic held inputs, both paper cuts,
R-COL/scoring/bootstrap/Leiden, and a synthetic external image encoder. It does
not load Qwen or read real held features. It writes no production Stage 0 pass
or production verdict. Choose a **new** session-scratch destination:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
flock -o -w 14400 /tmp/gpu0.lock \
/root/miniconda3/envs/CoSiR/bin/python src/test/20261020_claim_test/ct_smoke.py SESSION_SCRATCH/ct-smoke --device cuda
```

`--device` defaults to `cuda`; a tiny CPU check may explicitly use `--device cpu`.
The smoke checks file keys, saved hashes, exact development inference, held-list
hashes, scorer inputs and verdict/diagnostic keys at the file boundaries. Its
outputs are `smoke.json`, `smoke_verdict.json`, `smoke_diagnostics.json` and the
scratch `outputs/` tree. Full-size counts in the merged R-COL/held writers and
full diagnostics runner remain strict: smoke uses their pure judgment/inference/
diagnostic functions with synthetic stores and never licenses a claim result.
