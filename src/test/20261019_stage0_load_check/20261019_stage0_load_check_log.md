# Stage 0 load check (r10 run, 2026-10-19)

## Problem

Stage 0 of the PercepT core spec (§3.5) failed again in the run, with new numbers. The build (00:40 to 00:52) gave
0.1238 / 0.2617 (held-out emotion / genre AMI) for both unchanged September scripts. The run's queue (01:32 to 01:44,
beside the CPU queue's three 8-thread workers) gave 0.1244 / 0.2617 for script 1 (Stage 1 snapshot) and 0.1287 /
0.2804 for script 2 (extended-seed Stage 2). The build's reading, "this machine reproduces its own September record
bit for bit", did not hold.

## Investigation

1. **Where the runs split** (log diff, build thread test vs tonight's script 1): the ten printed pretraining losses are
   equal; the first difference is DEC epoch 10 (2,973 vs 3,016 changed nodes), right after the k-means start.
2. **k-means under load** (`kmeans_repeat.py loaded 3`, `kmeans_loaded.txt`, `center_diffs_loaded.txt`): the
   September start, `KMeans(60, n_init=10, random_state=42)` with 8 threads on the saved node404 pre-DEC latents,
   three times at load 26 to 28. Labels identical in all runs (and equal to the saved September labels); centers differ
   at the last float32 bit (max abs difference 3e-8, a third of the entries); a new center hash every run.
3. **k-means idle** (`kmeans_idle.txt`, load about 4): the same, a new center hash every run. So the noise does not
   need load.
4. **One thread** (`kmeans_single.txt`, `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`): the same center hash
   in three of three runs, 25 s per run instead of 10 s. (`kmeans_single.py` and `kmeans_single1.txt` are a first try
   with `threadpoolctl`; it had no effect, because sklearn loaded OpenMP after the limit was set.) Cause of the noise:
   sklearn's Lloyd step adds the per-thread partial center sums in the order the threads finish.
5. **Both scripts on the idle machine** (`stage0_idle.sh`, `stage0_idle.txt`, logs in `runs/idle/`; tree exported
   from `1b70d7f` into the session scratchpad, GPU lock, same env as `pc_stage0.py`): script 1 gave 0.1238 / 0.2617
   (DEC log equal to the build's), script 2 gave 0.1287 / 0.2804 (DEC log equal to tonight's queue run).
6. **Script 1 with one CPU thread, twice** (`stage0_single_thread.sh`, `stage0_single_thread.txt`, `runs/single_*`):
   one run at load 8.5 (beside the Codex re-derivation), one idle. Both gave 0.1287 / 0.2804, with identical
   pretraining and DEC log lines.

Eight local runs of the unchanged September recipe at 8 threads or more:

| Outcome (emotion / genre) | DEC epoch 10 changed nodes | Runs |
|---|---|---|
| 0.1238 / 0.2617 | 2,973 | 5: September local sweep (2026-09-23), build script 1 and 2, build thread test, tonight idle script 1 |
| 0.1244 / 0.2617 | 3,016 | 1: tonight queue script 1 |
| 0.1287 / 0.2804 | 3,140 | 2: tonight queue script 2, tonight idle script 2 |

## Root cause

The September k-means start is not bit-reproducible with more than one CPU thread: the thread order of its center
sums changes per run, idle or loaded. DEC turns these last-bit differences into one of a few outcomes, 0.005 apart in
emotion and 0.019 in genre (159 held-out paintings). With one CPU thread the whole fit repeats (two of two runs, one
under load). That one thread is enough was shown for the whole process; that k-means is the only threaded source is
an inference from steps 2 to 4. Whether node404 repeats itself is not checked (its five September fits were equal).

## Fix

None in the September code (stage 0 runs it unchanged). Consequences, for the user's stage 0 decision:

- A 4-decimal re-fit check cannot pass reliably on any machine we checked; the parts that repeat exactly are the saved
  node404 snapshot re-score (0.122469 / 0.227419), the byte-identical code, and the k-means labels on the saved
  latents.
- PercepT core's Stage 1 (`pc_sept.py`, under `threadpool_limits(8)`) uses the same k-means start, so its runs carry
  the same noise. A one-thread k-means there would make them repeat; that is a code change to reviewed code.
- The PercepT arms stay blocked (spec §3.5); the run chat changed nothing in the round's code.
