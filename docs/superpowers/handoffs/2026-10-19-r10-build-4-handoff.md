# Handoff: build the stage 0 amendment (A1) and one-thread k-means (A2) of the PercepT core spec, then hand to the run

## Brief
- Done: decision C on the r10 run; the user amended stage 0 to the checks that repeat exactly (A1) and added one-thread k-means to PercepT core (A2).
- State: rows 1, 6, 10, 13 are filled; the five PercepT core arms are still blocked by the old stage 0.
- Next: one verdict-code ticket (new `pc_stage0.py` checks, one-thread k-means start in `pc_stage1.py`), review, merge, then the run chat.
- Blocked: nothing.

> Written 2026-10-19 08:35 by "r10 decide 8". Loop step reached: step 7 (build) of the amended r10 spec; step 8
> (run) follows. Direction log entry:
> [2026-10-19, PercepT core run read](../direction_log.md).

## 1. Read in this order

1. The spec [2026-10-18-percept-core-design.md](../specs/2026-10-18-percept-core-design.md): §3.5 (the amended stage 0
   row), §4 "Amendments" (A1, A2), §6, §9.
2. The load check log `src/test/20261019_stage0_load_check/20261019_stage0_load_check_log.md` (why the re-fit does not
   repeat; the threadpoolctl import-order finding in step 4).
3. The stage 0 trace `src/test/20261019_stage0_trace/`: `compare_artifacts.py` and `artifact_comparison.txt` already do
   checks (i) and (iii) by hand; `kmeans_probe.py` does (iv). The new `pc_stage0.py` turns them into gate checks.
4. The current code: `src/test/20261018_percept_core/pc_stage0.py`, `pc_stage1.py` (the k-means start, near line 202),
   `pc_sept.py` (its docstring names the source line ranges), `pc_queue_gpu.sh`.

Skills: none (the build follows `~/.claude/references/matt-chain.md`).

## 2. Decided by the user, not to be reopened

- The spec's §4 (D1 to D15), approved 2026-10-18 23:34.
- A1, stage 0 gate: only the checks that repeat exactly; the September re-fit and the Stage 2 AUC (0.8296) leave the
  gate (user, option a, 2026-10-19 08:20).
- A2, one-thread k-means in PercepT core's Stage 1; `pc_sept.py` stays verbatim (user, after seeing the
  recommendation, 2026-10-19 08:20).
- No container restart before this chat (user, 2026-10-19 08:20). GPU check at 08:20: driver, module and library all
  595.99.02; container up since 2026-10-18 15:53.
- The user asked for the PercepT core runs now. They saw the end-time range: GPU use ends about 10:30 by the build's
  timing (about 25 GPU minutes), or about 13:00 by the spec's estimate (2.5 to 3 hours); they said to go on.

## 3. Where things stand

- The new stage 0 checks (spec §3.5, A1), with the values they must give:
  - (i) the September scripts and every module they import: the same git blob at `3b3ce61`, `4e4480b` and `1b70d7f`
    (the trace found them equal, and no diff in `src/utils` or `src/conditional_buddy`).
  - (ii) every function, class and constant that `pc_sept.py` copies: text-identical to its source at `1b70d7f`. No test
    checks this today. The docstring lists the Stage 1 sources; the build finds the Stage 2 ones
    (`AttentionPoolingMapper`, `evaluate_auc`, `auc_summary`).
  - (iii) `res/percept/percept_stage1_embedding_snapshot.npz` (gitignored, on disk) re-scores to held-out emotion AMI
    0.12246902855745419 and genre AMI 0.22741920423971046 (genre on the 159 paintings with a genre label), to 1e-6.
  - (iv) `KMeans(60, n_init=10, random_state=42)` on its `train_latent_pre` (61,402 rows) under the one-thread limit
    gives `train_initial_label` exactly (about 25 s).
  - Row 2 (1e-6) and the 21 fix tests: unchanged.
- A2's checks: three k-means starts on the snapshot latents under the limit give one center hash (a test of the code
  path `pc_stage1.py` uses); in the run, a second Stage 1 `clip_roberta` seed 0 compared bitwise with the first
  (reported, not a gate). The build decides whether this repeat is a non-blocking queue step or a command for the run
  chat.

## 4. Open points for the next step

1. **Where the one-thread limit goes (minor, the build's choice):** `threadpool_limits(limits=1)` around
   `pc_sept.initialize_cluster_centers` in `pc_stage1.py`, nested in the current 8-thread block, works only if sklearn
   has loaded OpenMP before the limit is set. `pc_stage1.py` imports `pc_sept` (sklearn) at the top, so it should
   work. The test must use that same import order, or call a helper in `pc_stage1.py`. A fallback is
   `OMP_NUM_THREADS=1` for the Stage 1 process (GPU work is not affected).
2. **The run after the merge (minor):** rerun only the GPU queue and the rescore (`pc_score.py`); the 24 head runs are
   done. The build checks how the CPU queue and the sheet behave when they are not rerun, and writes the run chat's
   commands in its handoff.

## 5. Code and pitfalls

- The GPU queue runs `pc_stage0.py` first and blocks every Stage 1 and Stage 2 run on its exit code; only the checks
  change, not the queue's gate logic.
- The September modules collide with main's `src` package: run them only in fresh processes.
- `pc_queue_status.txt` is appended across runs; the CPU queue's sheet waits for the `PC_gpu_DONE` after the last
  `PC_gpu_SUBMIT`. Start one GPU queue only.
- Known before the run (build timing, seed 0): Qwen Stage 1 keeps 1 topic at γ 2 and CLIP + RoBERTa keeps 99. The
  spec reports it as it is (§8); P3 and P4 may then not be "established".
- `/project/CoSiR-buddy_prototype_conditioning` is a worktree at `1b70d7f` that another line uses; read it only.
- `routing-check` printed one open watch item (the intermittent Codex launch refusal, user 2026-10-12: watch, no rule
  change); nothing new.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing (after this chat's commit).
- On disk over 1 GB: the deferred head outputs (`src/test/20261018_aspect_graph_sheet/outputs`, `outputs_fu`); the 10
  merged ticket worktrees were deleted at this wrap-up (user, 2.2 GB in all).
