# Handoff: finish the r10 build 4 work: GPU smoke, final whole-branch review, merge to main, then open the run chat

## Brief
- Done: tickets 10 (stage 0 checks that repeat exactly, A1; one-thread k-means start, A2) and 11 (Stage 1 repeat check in the GPU queue) built by Codex, reviewed by Opus, fixed, merged on `r10-stage0-a1`.
- State: the full stage 0 passes every check on the real data (snapshot AMIs, k-means labels and row 2 exact); the round suite passes on the merged branch (195 passed in 158 s).
- Next: a GPU smoke of the GPU queue, the final whole-branch review (Opus `xhigh`), fix wave, merge to main, then the run chat.
- Blocked: nothing.

> Written 2026-10-19 09:00 by "r10 build 4" (cut at its context meter, after both tickets merged). Loop step reached:
> step 7 (build) of the amended r10 spec, before the final review. Direction log entry:
> [2026-10-19, PercepT core run read](../direction_log.md).

## 1. Read in this order

1. This handoff, then the previous one,
   [2026-10-19-r10-build-4-handoff.md](2026-10-19-r10-build-4-handoff.md) (§2 the user's decisions, §3 the values
   each check must give, §5 pitfalls). Both still hold.
2. The spec [2026-10-18-percept-core-design.md](../specs/2026-10-18-percept-core-design.md): §3.5 (stage 0 row), §4
   "Amendments" (A1, A2), §6, §9.
3. The tickets `.scratch/r10-percept-core/issues/10-stage0-a1-one-thread.md`, `11-stage1-repeat.md` and the index
   `00-index.md` ("Shared interfaces", build 4 line).
4. `~/.claude/references/matt-chain.md`, "The final whole-branch review".

Skills: none.

## 2. Decided by the user, not to be reopened

- A1 and A2 as in the spec's §4 (user, 2026-10-19 08:20); the spec's §4 D1 to D15 (2026-10-18 23:34).
- No container restart before the build (2026-10-19 08:20). The user asked for the PercepT core runs now and accepted
  an end between about 10:30 and 13:00 (build 4 handoff §2).

## 3. Where things stand

**Branches.** Integration branch `r10-stage0-a1` (pushed), worktree `/project/CoSiR-wt/r10-stage0-a1` (with `res` and
`src/test/20261017_prompted_aspect_views/sample.npz` linked to the main checkout; both gitignored). Ticket branches
`r10-s0-10`, `r10-s0-11` (worktrees `/project/CoSiR-wt/r10-s0-10`, `-11`), merged. Main is not yet merged.

**Tickets** (record for the run chat's handoff; agent choices, Codex `gpt-6.1-sol` at `high`):

| # | Ticket | Blocked by | Spec IDs | Verdict code | Review |
|---|---|---|---|---|---|
| 10 | Stage 0 checks that repeat exactly (A1), one-thread k-means start (A2) (`pc_stage0`, `pc_stage1`) | none | §3.5 stage 0, A1, A2 | yes | Opus `high`: changes needed (check (ii) missed some edits; (iii) tolerance untested); one fix round; 87 tests pass |
| 11 | Stage 1 repeat check in the GPU queue (`pc_repeat`, `pc_queue_gpu.sh`, `pc_queue.md`) | none | A2 run check | yes (small) | Opus `high`: approve with two fixes (no compare after a crashed repeat; stale output); one fix round; 41 tests pass |

Coverage: §3.5 stage 0 and A1, ticket 10; A2 code path, ticket 10; A2 run check, ticket 11.

**What the code now does.**
- `pc_stage0.py` runs no September script. Gate: (i) every `.py` under `src/` outside `src/test/`, and under the two
  pilot folders, has the same blob at `3b3ce61` and `4e4480b` as at `1b70d7f` (74 and 77 files; a superset of the
  scripts' import closure, since they load siblings by path); (ii) every top-level node of `pc_sept.py` is a mapped,
  text-identical copy of its source at `1b70d7f` (decorators included; fixed import list; `base` exact); (iii) the
  snapshot re-scores to 0.12246902855745419 / 0.22741920423971046 (1e-6; 9,365 and 159 rows); (iv) `KMeans(60,
  n_init=10, random_state=42)` under `pc_stage1.one_thread()` gives the saved labels (0 differ, 25 s); row 2 and the
  21 fix tests as before. `--smoke DIR` runs every check and writes `SMOKE`.
- `pc_stage1.py`: `one_thread()` (threadpool limit 1, asserting sklearn's OpenMP and every pool at 1), `kmeans_start`
  wraps the verbatim `pc_sept.initialize_cluster_centers`; config `kmeans_threads=1`; `--outroot DIR`. `pc_sept.py` is
  byte-identical (blob `d7c064b`).
- `pc_queue_gpu.sh`: after Stage 2, `stage1_repeat` (clip_roberta seed 0 into `outputs/repeat/`) and `repeat_check`
  (`pc_repeat.py`, writes `outputs/stage1_repeat.json` with `bitwise_equal`); reported, not a gate; blocked when the
  first run or the repeat failed.
- Implementer's full stage 0 runs (CPU, in the ticket 10 worktree): `outputs/stage0_smoke/stage0.json` and
  `outputs/stage0_smoke2/stage0.json` there, all checks PASS.

## 4. Open points for the next step

1. **GPU smoke (first):** in the integration worktree, under the lock, the GPU queue in smoke mode, e.g.
   `bash src/test/20261018_percept_core/pc_queue_gpu.sh --foreground --smoke "$PWD/src/test/20261018_percept_core/outputs/queue_smoke_b5"`
   in a background Bash. Check: stage 0 `SMOKE` with all checks PASS, 3 Stage 1 runs, 5 Stage 2 runs, the repeat and
   `stage1_repeat.json` (`bitwise_equal` should be true; if false, that is a finding for the final review, not a
   stop), and each step's seconds (for the end-time estimate).
2. **Final whole-branch review** (`worker`, Opus `xhigh`), then one fix wave and a scoped re-review (Sonnet `high`), per
   `matt-chain.md`. The reviewer re-derives the stage 0 values and the A2 claim; brief it with the spec, the tickets,
   that section, hard rule 6 (`~/.claude/rules/spec-templates.md`) and `docs/superpowers/constitution.md`.
3. **The ablation table's PercepT rows (minor, agent's choice):** `pc_ablation.py` prints rows 1 to 3 and 6 to 13
   only; rows 4a, 4b, 5a, 5b, 5c (the arms' image-side AMI in `outputs/pc_results.json`) are not in it. Recommended: a
   small ticket 12 in this build that adds them, with a unit test (it carries numbers into the user's table), so the
   run chat only runs commands.
4. **Then:** merge `r10-stage0-a1` into main locally, push main, update the round log
   (`src/test/20261018_percept_core/20261018_percept_core_log.md`, a "Build 4 (A1, A2)" section), list the worktrees
   and branches on `.scratch/pending_deletions.md`, write the run chat's handoff with the ticket list above and the
   commands below, run the GPU check (night mode: check only), and open `loop-next <handoff> --label "r10 run 2"
   --unattended`.

**The run chat's commands** (build 4's finding; check them in the smoke):
- Only the GPU queue reruns: `(bash src/test/20261018_percept_core/pc_queue_gpu.sh --foreground --lock-wait 14400) >
  src/test/20261018_percept_core/pc_queue_gpu.log 2>&1` in a background Bash, from the repo root. The 24 head runs
  and rows 6, 10, 13 are done; the CPU queue is not rerun. Nothing waits for the GPU queue's `PC_gpu_DONE` then (the
  wait is in the CPU queue only); the status file gets one more SUBMIT/START/DONE block.
- After it: `python src/test/20261018_percept_core/pc_score.py` (rewrites `outputs/pc_results.json`, `pc_tables.md`
  with the arms and row 1; reads `outputs/stage0.json`), then `python src/test/20261018_percept_core/pc_ablation.py`.
- Timing (estimate, not measured): stage 0 about 2.5 minutes, then about 25 GPU minutes for 9 Stage 1 and 15 Stage 2
  runs (build 3 timing) plus the one-thread k-means (about +30 s per Stage 1 run) and the repeat (one Stage 1 run).
  About 35 minutes in all; the smoke's seconds refine it.

## 5. Code and pitfalls

- The build 4 handoff's §5 still holds (September modules only in fresh processes; one GPU queue at a time; the
  status file is appended; Qwen Stage 1 keeps K = 1 at γ 2, CLIP + RoBERTa K = 99 at seed 0).
- Merges: build 4 checked out `r10-stage0-a1` in the main checkout once; do merges in the integration worktree.
- The ticket worktrees need `res` and the frozen `sample.npz` linked from the main checkout (gitignored; `/res` is in
  `.git/info/exclude`).
- Codex sessions for a fix round by resume: ticket 10 `01a152de-cd8d-7210-87d5-ebba31923f90`, ticket 11
  `01a152de-d565-7cd1-a6de-cfa64499dae6` (logs in the build 4 session scratchpad).

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing (after this chat's commit).
- On disk over 1 GB: unchanged (the deferred head outputs, `.scratch/pending_deletions.md`).
