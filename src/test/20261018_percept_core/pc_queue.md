# PercepT queues

From the repository root, start the GPU queue first with `--foreground` inside a
background shell, then start the CPU queue beside it:

```bash
round=src/test/20261018_percept_core
(
    bash "$round/pc_queue_gpu.sh" --foreground --lock-wait 14400
) > "$round/pc_queue_gpu.log" 2>&1 &
(
    bash "$round/pc_queue_cpu.sh"
) > "$round/pc_queue_cpu.log" 2>&1 &
```

The CPU queue waits for the GPU queue before writing the final sheet. Logs and
the shared `pc_queue_status.txt` are in the round folder.

`--foreground` waits for the locked job and returns its exit code. The plain
`bash src/test/20261018_percept_core/pc_queue_gpu.sh` form uses nohup and returns
at once; its PID is the background launcher shell. The lock is
`flock -o -w SECONDS /tmp/gpu0.lock`; children do not inherit the lock descriptor.
`--lock-wait` (default 14400 seconds) limits the wait to acquire the GPU lock.
On timeout the worker never starts, flock returns nonzero, and the launcher
records `PC_gpu_DONE lock_or_worker=<code> failed=1`. It also records that line
if the worker exits without writing DONE. Every launch records `PC_gpu_SUBMIT`
before waiting for the lock, including the nohup form.
Stage 0 checks that the September scripts and every module they could import are
byte-identical between record commits `3b3ce61`, `4e4480b`, and `1b70d7f`, and that
every function, class, and constant copied by `pc_sept.py` is text-identical to its
source at `1b70d7f`. Stage 0 no longer re-runs the September scripts (amendment A1).
It re-scores the saved September snapshot
(`res/percept/percept_stage1_embedding_snapshot.npz`) to held-out emotion AMI
0.122469 and genre AMI 0.227419 (tolerance 1e-6). K-means on the snapshot's
pre-DEC latents (K 60, 10 restarts, seed 42, one CPU thread) must reproduce its
saved initial labels exactly. Stage 0 also checks row 2 (tolerance 1e-6) and runs
the four-fix unit tests. A failed
Stage 0 blocks all PercepT training, while extraction still runs. A finished
RoBERTa store with `done.json` count 99999 produces `roberta=skip`. A failed Stage 1
run blocks its dependent arms; independent runs continue. All failures reach the
queue exit code and final status line.

PercepT core's Stage 1 runs its k-means start on one CPU thread (A2). After the
Stage 2 loop, a successful `stage1_clip_roberta_s0` triggers `stage1_repeat`,
which repeats that run into `outputs/repeat` using `--outroot`. The next step,
`repeat_check`, compares the arrays' key sets, dtypes, shapes, and bytes, and all
sidecar keys except `seconds`. It writes `outputs/stage1_repeat.json`, including
per-array and per-key agreement and both NPZ SHA-256 hashes. Differences are
reported, not a gate: the comparison exits 0 and the queue continues. A crash or
missing file fails the step. If the first run failed or was blocked, both repeat
steps are recorded as `blocked`.
If `stage1_repeat` fails, `repeat_check` is recorded as `blocked` so older repeat
files cannot be compared as a fresh run.

The independent CPU queue runs in the foreground:

```bash
bash src/test/20261018_percept_core/pc_queue_cpu.sh
```

It packages graphs, invokes the gate training CLI (including its existing-output
verification), checks the gate, then launches three aspect workers. Each worker
runs rows 6, 10, 13 sequentially with eight CPU threads; genre omits row 10, which
already exists. Scoring runs even after a gate failure so missing rows remain
visible. The head jobs do not depend on Stage 0. Before the sheet step, the queue
polls every 30 seconds for a `PC_gpu_DONE` after the last `PC_gpu_SUBMIT`, for up
to `PC_GPU_WAIT` seconds (default 18000). It records one
`gpu_wait=ok|none|timeout seconds=<n>` line: `none` means no GPU submission was
recorded. A timeout marks the CPU queue failed and still writes the sheet so the
incomplete run is visible. A completed GPU queue may itself have failed; its
results and status remain visible on the sheet.

For smoke checks, use a fresh folder within the worktree, the same folder for
both commands if their timing records and sheet should be combined:

```bash
bash src/test/20261018_percept_core/pc_queue_gpu.sh --foreground --smoke "$PWD/src/test/20261018_percept_core/outputs/queue_smoke"
bash src/test/20261018_percept_core/pc_queue_cpu.sh --smoke "$PWD/src/test/20261018_percept_core/outputs/queue_smoke"
```

Both smoke queues share that folder's status file; concurrent smoke runs use
the same GPU wait protocol, polling every second.

GPU smoke runs 512-caption extraction, all three Stage 1 versions, and all five
Stage 2 arms at seed 0. Stage 1/2 still read the existing full feature stores.
The Stage 1 repeat uses `--smoke <smoke root>/repeat`, and its comparison report
is `<smoke root>/stage1_repeat.json`. Stage 0 smoke runs every check described
above into the smoke folder and writes `SMOKE` instead of `PASS`.
CPU smoke uses each existing two-epoch head mode at seed 0 and runs
both scoring commands. Its gate is explicitly a smoke check, not a regression
pass. The unchanged scoring CLI computes row 1 on the frozen sample.

Smoke outputs live entirely in the requested folder. The rows adapter relocates
`pc_rows.HERE` before invoking its unchanged CLI, so graphs are in
`rows/outputs/graphs/` and heads and row scores in `rows/outputs/smoke/`.
`smoke_times.json` merges separate `gpu` and `cpu` timing maps under a file lock.
Each step records its exit code (or `skip`/`blocked`) and wall seconds, including
input loading and startup. Logs are in `logs/`; `pc_queue_status.txt` appends each
step immediately and one final summary per queue. Repeat a smoke run in a new
folder because the extractor refuses to overwrite a finished store.
