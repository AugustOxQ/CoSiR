# Handoff: r21 build 3: review gate 0b, build ticket 04, GPU smoke, final review, merge, then open the r21 run chat

## Brief
- Done: tickets 01, 02, 05, 06 reviewed, fixed, merged; 03 merged (its gate script not yet reviewed); gates 0a, 0b pass, all diffs 0.
- State: 06's table reproduces r10's ablation cells exactly (39 cells, diff 0); DEC λ ∈ {1, 10} may be weak (flat q).
- Next: review of gate 0b's script, ticket 04 (held exports, queue scripts), GPU smoke, final whole-branch review.
- Blocked: nothing; the GPU smoke waits for the local GPU (held by r25 and r26 until about 08:50 at the earliest).

> Written 2026-11-11 07:47 by the r21 build 2 chat (cut at 200k tokens). Loop step reached: step 7, mid-build.
> Direction log entry: [2026-11-11 · r21 decide](../direction_log.md).

## 1. Read in this order

1. This handoff, then the first build handoff's sections 2 and 5 (decisions, code pitfalls; still true):
   [2026-11-11-r21-build-handoff.md](2026-11-11-r21-build-handoff.md). The spec governs:
   [2026-11-11-r21-dec-heads-design.md](../specs/2026-11-11-r21-dec-heads-design.md).
2. The tickets: `.scratch/r21-dec-heads/issues/00-index.md` and `04-runner-held-queues.md`.
3. The round log and the decision lines on branch `r21-dec-heads` (worktree `/project/CoSiR-r21`):
   `src/test/20261111_r21_dec_heads/20261111_r21_dec_heads_log.md`, `decisions.md`.
4. `~/.claude/references/matt-chain.md` (final review, fix wave, re-review, merge) and
   `~/.claude/references/loop-chats.md` ("Unattended chats", "How to cut").

Skills: none by name; the chain is in `matt-chain.md`.

## 2. Decided by the user, not to be reopened

As the first build handoff's section 2 (spec §4, D1 to D7). Run mode is night: no questions to the user.

## 3. Where things stand

| # | Ticket | State |
|---|--------|-------|
| 01 | DEC math (`r21_dec.py`) | done, merged |
| 02 | Trainer (`r21_train.py`) | Opus `high` review (no code defect; 7 of 16 mutations survived), fixed (ce46aa8e; all 7 killed; teacher rows checked in fit order), merged |
| 03 | Runner train + gate 0b (`r21_run.py`, `r21_gate0b.py`) | built, merged (0ded9542); gate 0b PASS on CPU, all diffs 0, two mutations give FAIL; **its gate script needs a review**; **GPU smoke not run** |
| 04 | Runner held + teacher command + queue scripts | not started |
| 05 | Scoring (`r21_score.py`, `r21_arms.py`) | Opus `high` review (F1 to F9), fixed (86095ff9), merged; full gate 0a PASS, all diffs 0 |
| 06 | Stage 2 table + diagnostics (`r21_table.py`) | built; regression on r10's `k11=40-no_buddy` exact (39 cells, diff 0); Opus `high` review (F1 to F7), fixed (2d3917d9; 53 tests; 11 mutations killed; emotion regression still exact), merged |

- Integration branch `r21-dec-heads` (worktree `/project/CoSiR-r21`), pushed. Ticket worktrees: `/project/CoSiR-r21-t01`,
  `-t02`, `-t03`, `-t05`, `-t06` (branches `r21/t0N`).
- **The flag for the run handoff (top item, for the user):** Student-t q is almost flat on unit head outputs (mean max q
  0.133 / 0.061 / 0.138 against 1/k 0.111 / 0.037 / 0.100), so λ ∈ {1, 10} may be weak; a cheap amendment before stage 1
  is λ ∈ {100, 1000}, which needs no code. The spec's grid stays (night mode).
- Gate 0a: `/data/SSD2/cosir_r21/20261111_dec_heads/scores/gate0a.json` (c* and teacher AMI equal r17's file, c* mAP
  equals r20's, all diffs 0).
- Gate 0b record so far: `/tmp/claude-0/-project-CoSiR/888cc12a-f7e7-4899-8318-8e3cfaaf1c99/scratchpad/r21_smoke/r21_gate0b_20261111_072007.json`
  (written from an uncommitted state; rerun on merged code for the permanent record).

## 4. Open points for the next step

None for the user. The build, in order:
1. **Review of `r21_gate0b.py`** (verdict code: an Opus `high` worker): does it fail whenever DEC-off parity breaks?
   Its part (i) splits "DEC on at λ = 0" in three because DEC's extra forward shifts the dropout RNG (a decision line:
   kept, not fixed). Then rerun gate 0b on the merged code through `res-queue cpu` (about 7 min) and record it in the
   round log; write its record under `/data/SSD2/cosir_r21/20261111_dec_heads/scores/` or the round log.
2. **Ticket 04** (Sonnet `medium`, plumbing): `r21_run.py held` and `teacher`, the four queue scripts. The held sidecar
   fields that ticket 06's loader requires are listed below (section 5). Check `codex-usage --room implementer` first.
3. **GPU smoke** (ticket 03's command, below) once the GPU is free, and 04's end-to-end smoke (train 2 epochs, then
   `held`).
4. **Final whole-branch review** (Opus `xhigh`), one fix wave, a scoped re-review (Sonnet `high`), merge
   `r21-dec-heads` into main, push.
5. **Run handoff** `docs/superpowers/handoffs/2026-11-11-r21-run-handoff.md` with the flat-q flag at its top, the ticket
   list (title, blocked by, spec IDs, verdict code; from `00-index.md`), the GPU check, then
   `loop-next <run handoff> --label "r21 run" --unattended`.

## 5. Code and pitfalls

- **Held sidecar fields for ticket 04** (ticket 06's review): write with `rb_student.write_outputs(path, {'img': x},
  meta)`, only the array `img`, float16 [12281, 2048], finite unit rows (1e-3). `meta`: `kind='held'`, `arm` = the
  folder name exactly, `scale='full'`, `aspect`, `seed` (int), `rows=12281`, `width=2048`, `dtype='float16'`,
  `list_sha256` = `rs_paths.HELD_SHA256` computed from `rb_student.held_names()`, `sha256` = `rb_student.sha_array(x)`,
  `image_sha256` = the hash of the input rows (must equal c*'s; r10's sidecars carry `5af25f2a…`), `weights_sha256`.
  Reusing `rs_stage0.infer_set`'s meta (about line 365) gives them all. Without `image_sha256`, `r21_table.py table`
  refuses the r21 tree.
- **GPU smoke** (from ticket 03; after the merge, from `/project/CoSiR-r21`):
  `PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 res-queue gpu --est 20m --label r21:smoke -- sh -c "cd /project/CoSiR-r21/src/test/20261111_r21_dec_heads && /root/miniconda3/envs/CoSiR/bin/python r21_run.py train --arm dec-off dec-plain_l1_w1_S1 dec-start_l1_w1_S2 dec-teacher_l1_w1_S1 --seeds 0 --device cuda --smoke --outputs <empty folder in the new chat's scratchpad> > <log> 2>&1"`.
  Checks: each `run__s0.json` has status ok and, for DEC arms, `dec_start_epoch = 2` and `stop_epoch = 2` on every
  head; histories finite; `config.rg.arm` equals the folder name; `history[-1].dec.stop_reason` equals the record's;
  exports 6,152 × 2,048 float16; a rerun prints "verified weights; skip"; r10's tree unchanged. The CPU runs of the
  plain and start arms took 2.5 min each, mostly k-means on 43k rows: check the cost assumption ("a DEC run under
  10 min") from `history[*].dec.seconds_end`.
- **Test files in worktrees: one per process.** Frozen `20261021_d2_ablation/ab_paths.py` (via r17's
  `ns_paths.add_rounds()`) pins the main checkout's round folders, while `r21_train.py` adds the worktree's own; in one
  pytest process `test_r21_score.py` and `test_r21_train.py` clash in a worktree. On main they are the same files.
  `r21_run.py` and `r21_gate0b.py` import only c*'s chain and work in a worktree.
- **Ticket 06's real table** takes about 1 to 2 h of one CPU slot (`r21_table.py table`, then `diag`); the stage 2
  queue script should run it through `res-queue cpu`.
- `/data/SSD2/cosir_r21/20261111_dec_heads/scores/` holds `gate0a.json`, `gate0a_partial.json` and `val/cstar__s0.json`.
- At the end of the build: worktrees `/project/CoSiR-r21-t01` to `-t06` and their branches go on
  `.scratch/pending_deletions.md`, not deleted (night mode). Scratch of this chat (no item over 1 GB):
  `/tmp/claude-0/-project-CoSiR/888cc12a-f7e7-4899-8318-8e3cfaaf1c99/scratchpad/` (`r21_smoke/`, `t06/`, `rev02/`,
  `rev05/`, `rev06/`, `fix02/`, `sc_t02/`); and the first build chat's scratchpad
  (`/tmp/claude-0/-project-CoSiR/c6e18fab-59ed-4012-8270-0b34e3297dd2/scratchpad/`).

## 6. State at handoff

- Running: nothing of this chat.
- Uncommitted: nothing of r21 (ticket branches hold their commits locally; `r21-dec-heads` is pushed).
- On disk over 1 GB: nothing.
