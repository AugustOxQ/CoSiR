# Stage 2 queues (ticket 09)

Stage 2a and stage 2 run from the main checkout `/project/CoSiR` after the merge into main, never from a
worktree: the scripts find their own round folder and write `outputs/stage2/` and the reports there. They are foreground queues like stage 1's (`ct_queue.md`), so run them with
`nohup bash ... > session-scratch/queue.log 2>&1 &`. They stop at the first failed step. GPU steps run under
`flock -o -w 14400 /tmp/gpu0.lock` (`CT_LOCK_WAIT` changes the wait, `CT_GPU_LOCK` the lock file). `CT_PYTHON`
changes the Python. CPU steps use eight threads with CUDA hidden. No `ct_*` file is used or changed except
through the ct2 modules' imports.

```bash
bash src/test/20261020_claim_test/ct2_queue_gpu.sh     # extraction ... held inference
bash src/test/20261020_claim_test/ct2_queue_cpu.sh     # after the GPU queue returned success
```

One status line per step goes to `outputs/stage2/queue_{gpu,cpu}_status.txt` (and to the terminal), logs to
`outputs/stage2/logs/<queue>_<step>.log`. The GPU queue prints a budget line first (see below).

## Steps (`ct2_run.py <step> --root R --reports D --device cuda`)

| Step | Does | Skips when |
|---|---|---|
| `budget` | prints the end estimate against the next 11:00 | never |
| `extract` | `ct2_extract.extract(store, device)` for each store that is missing | a store exists and passes `ct2_data.check_s2_store`; an existing store that fails the check stops the queue (never overwritten) |
| `graphs` | `ct2_graphs.py --mode full --out R/full/d2/graphs`, on CPU | the folder's manifest exists and its identity matches the full split (a mismatch stops the queue) |
| `stage2a` | `ct2_stage2a.main`: sample-size training and the check, report `D/stage2a.json`; exit 1 on FAIL | `require_pass` already holds |
| `d2` | `ct2_d2.py --mode full --train`, seeds 0 to 4 | per seed, the cohort is verified (ct2_d2's own resume) |
| `core` | `ct2_percept.py` full mode: stage 1 both versions, stage 2 P2 and P3 at k40 | per run, ct2_percept's own resume |
| `paper` | stage 1 both versions, stage 2 P1 and P4 at strict and k40 | same |
| `rcol` | `ct2_verdict.py --rcol-only` (all train pairs), CPU | the saved R-COL file is identical (ct_rcol refuses a changed one) |
| `held` | writes `R/full/held/held_paintings.json` if missing, then `ct2_d2.py --held` and `ct2_percept.py --stage held` for every run | a held array already verified is reused |
| `verdict` | `ct2_verdict.py`: `D/verdict_stage2.json` | `--replace REASON` is the V-check rerun path |
| `diagnostics` | `ct2_diag.py`: `D/diag_stage2.json`, `.md` | none |

`R` defaults to `outputs/stage2` and `D` to this round folder. Every step after `stage2a` first calls
`ct2_stage2a.require_pass(D/stage2a.json, device)` and the ct2 modules call it again themselves. V4: `d2`,
`core` and `paper` refuse to start once any `R/full/held/**/*.npz` exists. The seeds are fixed to 0 to 4.
The stage 2a pass binds host, device and the code identity of every `ct2_*.py`, so all ct2 code (including
`ct2_run.py` and `ct2_smoke.py`) must stay unchanged from stage 2a on.

The extraction may be running from another worktree when the queue starts. `extract` then waits for the GPU
lock, finds the stores, checks them and moves on. A `.pending` folder (an unfinished extraction) makes
`ct2_extract.extract` stop with a clear message; wait for that extraction to end.

## Budget line

`ct2_run.py budget` prints JSON: start time, low and high end estimate, and the route against the next 11:00.
The numbers are the ticket's per-step estimates, not measurements: graphs 3 to 23 CPU min, stage 2a 25 GPU min,
D2 10, core 45, paper stage 1 85 and stage 2 25, held 5 GPU min, verdict 5 and diagnostics 35 to 50 CPU min.
K-means (one thread, n_init 10, at 216,107 pairs) and the rest of the extraction are not in the sum.
If the high end is past 11:00 the route says so and the run chat decides (node408 needs its own stage 2a).

## Smoke

`ct2_smoke.py SCRATCH --device {cuda,cpu}` runs every step through `ct2_run.main` on synthetic features
(1,500 fit and 150 validation paintings with two captions each, 12,281 held paintings), one epoch, seeds
0 to 4 (R-COL reads all five), into the new folder SCRATCH. Real code: graph build, D2 and PercepT trainers,
held writers, R-COL, scorer, diagnostics. Replaced: the data providers and splits, epochs, the stage 2a
gate (a recorder that reads the report), `ct2_stage2a.train` and `check` (the stage 2a item stays proven by
`test_ct2_stage2a.py` and by the real run), the extractor and store check, and the bootstrap (8 resamples,
one worker). On `cpu` the PercepT steps run with `--smoke` (the trainers refuse a non-smoke run off the GPU),
so only `--device cuda` runs the non-smoke path and the modules' own gate calls. It also checks that a
failing stage 2a returns exit code 1, that an existing store failing its check stops `extract` untouched,
that V4 refuses training once held arrays exist, that a failing gate stops every later step, and that
production folders and reports are unchanged. `--steps a,b` reruns steps in an existing SCRATCH.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
flock -o -w 14400 /tmp/gpu0.lock \
/root/miniconda3/envs/CoSiR/bin/python src/test/20261020_claim_test/ct2_smoke.py SESSION_SCRATCH/ct2-smoke --device cuda
```
