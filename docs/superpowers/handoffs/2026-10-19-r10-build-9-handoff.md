# Handoff: run the fix wave of the final review (Codex, brief ready), re-review, merge to main, stage 0, then open the run chat

## Brief
- Done: all four tickets merged into `r10-model-sheet`; CUDA smoke of every row type passes; stage 0 pre-check PASS.
- State: final review (Opus `xhigh`): 3 must-fix items (§6 line counted I1; stage 0 not tied to code version; form c
  stopped on the mask penalty); scorer, bars, form b, distillation re-derived exactly.
- Next: launch the prepared Codex fix brief, diff read and smoke, scoped re-review, merge to main, stage 0 from main.
- Blocked: nothing.

> Written 2026-10-19 20:59 by "r10 build 8" (cut at the context meter, 200k). Loop step reached: step 7 (build),
> fix wave of the final review. Direction log entry:
> [2026-10-19 · PercepT core with K = 40 read](../direction_log.md).

## 1. Read in this order

1. [build 7 handoff](2026-10-19-r10-build-7-handoff.md) §5 (pitfalls) and [build 8 handoff](2026-10-19-r10-build-8-handoff.md)
   §4 steps 5 to 7 and §5 (worktree pitfalls, stage 0, the run handoff's contents). This file replaces the rest.
2. The fix brief `.scratch/r10-model-sheet/notes/codex_brief_fix.txt` (items F1 to F9, checks a to f) and the review
   it rests on, `.scratch/r10-model-sheet/notes/final_review.md` (read its summary and findings; skip the detail
   unless a fix result needs it).
3. The round log `src/test/20261019_model_sheet/20261019_model_sheet_log.md` on `r10-model-sheet`: agent defaults
   1 to 16 (12 to 16 are new: D17 tie, seed 0 for the D17 rerun and I1, the §6 line, form c stopping, D17's best D on
   seed 0) and the timeline.
4. `~/.claude/references/codex-jobs.md` and `matt-chain.md` ("The final whole-branch review": one fix wave, then a
   scoped re-review at Sonnet `high`).

Skills: none.

## 2. Decided by the user, not to be reopened

- The spec, approved as written (2026-10-19 19:58); no container restart before the build (user, 19:58).
- Run mode set to day (user, 2026-10-19 20:51): ping the phone when something important needs the user's decision
  (`loop-chats.md`, "Unattended chats": ask with a default, 30-minute timer).

## 3. Where things stand

- Branch `r10-model-sheet` (pushed, `45e5da0`), worktree `/project/CoSiR-wt-msint`: tickets 01 to 04 merged
  (`a2f92fb`, `c7a4087`, `4b4ee36`, `92fa621`), build notes `ms_01_build.md` to `ms_04_build.md`.
- Fix worktree `/project/CoSiR-wt-msfix`, branch `r10-ms-fix` at `45e5da0`, sample symlink in place, nothing
  launched yet. Smoke folder `/tmp/claude-0/-project-CoSiR/56b9b751-6946-4c76-b52d-3761e520be8f/scratchpad/ms_smoke/fix/`
  (created).
- GPU smokes (controller, CUDA): ticket 02 T1, A1; ticket 03 T4, T5, D1, D2, D5, I1: valid outputs. Full-size joint
  rows: 1 to 3 s per epoch per head; CUDA losses close to CPU (not bitwise, as spec §8 expects).
- Stage 0 pre-check (20:36 to 20:47, pre-fix code, in `/project/CoSiR-wt-msint/.../outputs/stage0/`): STAGE0 PASS,
  genre arrays bitwise equal, 36/36 AMI equal at 6 decimals, 7/7 best epochs equal, PercepT 45/45. Three parallel
  aspects took 11 min. The fix changes the trainer's code version, so stage 0 must run again from main.
- Final review (`worker`, Opus `xhigh`, about 20 min): must fix 1 to 3; can wait 4 to 9. F1 to F9 in the brief cover
  1 to 3 and 5 to 9; item 4 (the back-fill) runs by hand after stage 0 (build 8 handoff §4 step 6). The reviewer did
  not run `pick_seeds`, `d17` or `sheet` on synthetic data, nor the GPU path; the fix brief's check e covers the first.
- Codex: five-hour window 100% left at 20:57 (new window); weekly about 67%. Ledger rows for 03 and 04 are in
  (`claude-config` `8d895d0`).
- GPU check at 20:59: driver 595.99.02 in all three places, container up since 19:08, GPU free, no lock.

## 4. Open points for the next step

No user question is open. Steps, in order:
1. `codex-usage --room implementer`, then launch Codex on the fix brief at `gpt-6.1-sol` high with
   `-C /project/CoSiR-wt-msfix`, log to `.scratch/r10-model-sheet/notes/codex_fix.log` (background Bash).
2. Diff read; check its smoke numbers against the brief's checks a to f; merge `r10-ms-fix` into `r10-model-sheet` in
   `/project/CoSiR-wt-msint`; GPU smoke of T5 `--smoke --device cuda` (the changed form c path) under the lock.
3. Scoped re-review of the fix diff only (`worker`, Sonnet `high`): does each of F1 to F9 do what the review asked,
   and do the re-derived checks of the review still hold (the bitwise path above all). Ledger row for the fix job.
4. Steps 5 to 7 of the build 8 handoff §4: merge into main from the main checkout and push; stage 0 from main (three
   `--aspects` processes, then `--check`, about 11 min); `ms_score.py --backfill --stage0-dir <round>/outputs/stage0`;
   the run handoff (ticket list, agent defaults 1 to 16, queue commands, timing, the back-fill to fold into the K = 40
   report, the review's "can wait" item 4 status), pending deletions, then `loop-next --unattended --label "r10 run 4"`.
5. Before that cut (day mode), ask the user whether to restart the container first (`loop-chats.md`, "How to cut"
   step 3), with the GPU check and the default "no restart" after 30 minutes.

## 5. Code and pitfalls

- Worktrees lack the data: each new worktree needs the `sample.npz` symlink (build 8 handoff §5). Keep the main
  checkout on `main`; merges into the integration branch run in `/project/CoSiR-wt-msint`.
- Stage 0 must run on the merged code from `/project/CoSiR` (its `outputs/stage0/` does not exist yet, so it is fresh).
  After F2, `ms_stage0.py --check` fails `stale` on arrays from older code, and the stage 1 queue's first step
  (`--require-pass`) refuses to train without a current PASS.
- If stage 0 fails: trace the cause, put it at the top of the run handoff, send it to the phone, and do not open the
  run chat (spec §3.5).
- Delete nothing; list items in `.scratch/pending_deletions.md` (build 8 handoff §6 and §6 below).

## 6. State at handoff

- Running: nothing of this chat (no Codex job, no subagent, no GPU lock, no background Bash).
- Uncommitted: nothing in git; `.scratch/r10-model-sheet/` (tickets, briefs, review) is outside git by design.
- Worktrees: `/project/CoSiR-wt-ms01` to `-ms04` (merged ticket branches), `-msint` (integration; its gitignored
  `outputs/` holds the stage 0 pre-check, 274 MB), `-msfix` (fix wave). Delete none; list them in pending
  deletions at the end of the build.
- On disk over 1 GB: `src/test/20261018_aspect_graph_sheet/outputs_fu/` (1.2 GB; deferred until the back-fill).
