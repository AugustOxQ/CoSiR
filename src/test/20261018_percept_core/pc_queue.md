# PercepT queues

From the repository root, submit one background GPU job (the PID is the `flock`
process; its log and status are in the round folder):

```bash
bash src/test/20261018_percept_core/pc_queue_gpu.sh --lock-wait 14400
```

`--foreground` waits for the locked job and returns its exit code. The lock is
`flock -o -w SECONDS /tmp/gpu0.lock`; children do not inherit the lock descriptor.
Stage 0 exports commit `1b70d7f` into `sept_stage0/`, runs both unchanged September
scripts in fresh interpreters (CM-1–3), checks their measured report columns at
four decimals, checks row 2, and records the four-fix pytest result. A failed
Stage 0 blocks all PercepT training, while extraction still runs. A finished
RoBERTa store with `done.json` count 99999 produces `roberta=skip`. A failed Stage 1
run blocks its dependent arms; independent runs continue. All failures reach the
queue exit code and final status line.

The independent CPU queue runs in the foreground:

```bash
bash src/test/20261018_percept_core/pc_queue_cpu.sh
```

It packages graphs, invokes the gate training CLI (including its existing-output
verification), checks the gate, then launches three aspect workers. Each worker
runs rows 6, 10, 13 sequentially with eight CPU threads; genre omits row 10, which
already exists. Scoring runs even after a gate failure so missing rows remain
visible. The queue does not depend on Stage 0. Run it after GPU training when the
final sheet should include the PercepT outputs.

For smoke checks, use a fresh folder within the worktree, the same folder for
both commands if their timing records and sheet should be combined:

```bash
bash src/test/20261018_percept_core/pc_queue_gpu.sh --foreground --smoke "$PWD/src/test/20261018_percept_core/outputs/queue_smoke"
bash src/test/20261018_percept_core/pc_queue_cpu.sh --smoke "$PWD/src/test/20261018_percept_core/outputs/queue_smoke"
```

GPU smoke runs 512-caption extraction, all three Stage 1 versions, and all five
Stage 2 arms at seed 0. Stage 1/2 still read the existing full feature stores.
September exposes no smoke-size training mode: Stage 0 smoke exports the tree,
imports both entries in separate processes, checks row 2 and runs the fix tests.
Its status is `SMOKE`, never `PASS`; it does not check September's numerical
regression. CPU smoke uses each existing two-epoch head mode at seed 0 and runs
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
