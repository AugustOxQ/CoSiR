# Handoff: re-review the final fix wave (two Sonnet `high` workers), rerun the X1 and X2 scorer checks and the full tests, merge into main; then open "r10 run 21"

## Brief
- Done: final review by stage (all FIX); fix wave by Codex, both fix branches merged into the integration branch.
- State: integration branch at f0b5566, pushed; the fresh-seed phase now runs end to end at smoke scale (forced winner GR-4: C-HOLDS, VG2 fresh agrees, 0 differences).
- Next: Sonnet high re-review of the two fix diffs, X1 and X2 scorer reruns (eg_score.py changed), full tests, merge into main.
- Blocked: nothing.

> Written 2026-10-25 22:22 by "r10 build 21f" (cut at 200k tokens of context, at a clean point: no subagent, Codex
> job, GPU job or background job running). Loop step reached: step 7 (build), final review and fix wave done, scoped
> re-review next. Direction log entry: [2026-10-25 · Recipe and backbones round read](../direction_log.md).

## 1. Read in this order

1. This handoff, then `.scratch/r10-build-21/notes/build_progress.md`, section "r10 build 21f progress".
2. The review notes in `.scratch/r10-build-21/notes/`: `final_review_stage_i_part1.md`, `final_review_stage_i_part2.md`,
   `final_review_stage_ii.md`, `final_review_stage_iii.md`; Codex's per-item notes for stages i and ii,
   `final_fix_i_ii_codex.md`. Give them to the re-reviewers; read only what a decision needs.
3. The 21f handoff §4 c and d (merge into main, the run chat's handoff, its three warnings) and the 21b handoff §4 d
   and e; the 21b, 21c, 21d and 21f handoffs' §5 pitfalls; they still hold.
4. `~/.claude/references/matt-chain.md` ("The final whole-branch review"), `~/.claude/rules/agent-routing.md`, and
   before the cut `~/.claude/references/loop-chats.md` ("How to cut").

Skills: `writing-for-agents` (briefs).

## 2. Decided by the user, not to be reopened

As the 21b handoff §2: the spec's §4 rows D1 to D17 marked "user" or "user, after seeing the recommendation", the
brief's approval at 18:35, the container restart at 18:44. Run mode is night (checked 21:29).

Agent defaults of this chat (record them in the run handoff):
- A candidate without a stage 3 outcome gets no A outcome: it is left out of the stage 5 plan and its stage 5 block
  says "no outcome" with "no stage 3 outcome: <cause>" (RB rule §4: "no outcome is stated for a candidate until V1 to
  V4 hold for it"; §4.3 "it never rescues a failed part 2 test").
- `EG_DEFER` is removed: deferral goes only through `eg_run.py defer --estimate-end` (recorded in
  `part2/deferral.json`, rule §2.1).
- The stage iii scorer's code hash covers `eg_score3.py` and the static closure of its module-level imports
  (`eg_paths`, `eg_variants`, `rb_ami`, `rb_data`, `rb_paths`, `rb_pick`, `rb_rows`), so a later stage i or ii fix
  does not invalidate stage iii score files.

## 3. Where things stand

- **Integration branch** `r10-encoders-genre` at f0b5566, pushed (worktree `/project/CoSiR-r10-int`). Merged in
  this chat: `r10-fix-iii` (3d6250f, 17331fa; merge a25d1df) and `r10-fix-i-ii` (ed5cc1b, 67728c9; merge f0b5566).
- **Final review** (three Opus `xhigh` workers by stage; the stage i and iii reviewers cut at 200k, stage i finished by
  a second worker): all FIX. Re-derived by the reviewers: X1 and X2 166 of 166 items; VL1 (a) 5 of 18 at diff 0; VG1,
  R6-5 parity (9 of 9), T_g and R6-5 genre A at diff 0. Mutations: stage ii 9 of 9 killed; stage iii 11 of 12 (M12
  now has a test); stage i 3 of 4 (S1-GO `>=` now pinned by a stage i test).
- **Defects fixed** (details in the notes):
  - stage i: a failed ArtELingo-28 store blocked the matched arm (rule: it removes only the A outcome); v5.json was
    written only at the end; `EG_DEFER` stopped stage i at `check-verdict 3`; no-outcome causes missing from blocks
    and report; the report's stores block never refreshed;
  - stage ii: the VL4 plan lock could be written before K5 to K8 weights exist (a missing gate file read as failed);
    "missing" was decided by weight files, not by the stage 3 outcome; held() waited for a file that never comes
    after a failed X0 or X1; the queue smoke stopped at VL1; forced-VL3 reruns did not check weights and code;
  - stage iii: VG2 fresh never built `identity.img_sha256`, so it disagreed every time a C outcome existed (shown by
    the reviewer's forced-winner trace); stale score files were kept silently; `fresh_parity.json` was rewritten on
    every run; the report did not check VG2 itself.
- **Checks after the fix wave**:
  - VG1 scorer on the real outputs at a25d1df: pass, R6-5 emotion, style, genre and T_g at diff 0, content equal to
    the old file (moved to `outputs/aside/20261025_final_fix/stage_iii/vg1_scorer.json`).
  - Forced-winner stage iii trace on a25d1df (`/tmp/claude-0/eg_checks/trace3/`, smoke scale, CPU; GR-4 made to win):
    VG1 gate, score first, VG2 first agrees, extras GR-1 GR-4, score all, VG2 all agrees, winner GR-4 with fresh seeds, parity gate, score fresh C-HOLDS, VG2 fresh agrees (0 differences, 338 s), report, DONE failed=0 (22:06 to 22:22). Only the merged stage iii code ran; the stage i and ii fixes merged after it.
  - Codex's tests: 176 (stage iii files) and 191 (stage i and ii files, 12 of them queue dry runs on stub trees).
  - Not yet run: the full round tests on the merged head; the X1 and X2 scorer checks (`eg_score.py` changed).
- **Codex:** five-hour window 69% left at 22:11 (reset 02:51), weekly 78%.

## 4. Open points for the build (agent work; nothing for the user)

a. **Scoped re-review of the fixes**, two `worker`s at Sonnet `high`, in parallel: one on
   `git -C /project/CoSiR-r10-int diff b4e11f0 r10-fix-iii` (stage iii; notes `final_review_stage_iii.md`, briefs in the
   21f session scratchpad `codex_fix_iii.txt`, `codex_fix_iii_b.txt`), one on `git diff b4e11f0 r10-fix-i-ii` (stages i
   and ii; notes `final_review_stage_i_part{1,2}.md`, `final_review_stage_ii.md`, `final_fix_i_ii_codex.md`, brief
   `codex_fix_i_ii.txt`). Scratchpad: `/tmp/claude-0/-project-CoSiR/56959795-a4cd-4060-876c-8a6c6accfb43/scratchpad/`.
   Each checks that every finding is fixed as the rule says, no new gap, tests bite; verdict PASS or FIX. A FIX goes to
   a fix round (Codex by default; Claude takes it back if it fails), then a short re-check.
b. **Checks on the merged head:** the full round tests (about 5 min; pytest under `/tmp/pytest-of-root`); the X1 and
   X2 scorer checks: move `outputs/part2/regression/x{1,2}_scorer.json` aside to `outputs/aside/20261025_final_fix/`
   first (a regression file that exists is never rerun), then `/tmp/claude-0/eg_checks/cpu_x12.sh` (about 25 min,
   background). A dry run of `eg_queue_gpu.sh` (`EG_DRY=1`) after the merge, since `eg_queue_common.sh` changed.
c. **Then** as the 21f handoff §4 c and d: merge into main, push, `ln -sfn` the `EG/outputs` link in the main checkout,
   worktrees and branches on `.scratch/pending_deletions.md` (now also `/project/CoSiR-r10-fix3`,
   `/project/CoSiR-r10-fix12`, branches `r10-fix-iii`, `r10-fix-i-ii`), and the run chat's handoff (ticket list, queue
   order, gates already passed, end estimate against 11:00 Monday 2026-10-26, else Monday 19:00 (D15), deferral order
   K5, K8, K7, K6, and the three warnings of 21f §4 d, of which (2) is now covered by the trace above).

## 5. Code and pitfalls

- The 21b, 21c, 21d and 21f pitfalls hold. `EG_DEFER` no longer exists; `eg_run.py defer` is the only way to defer.
- A trace tree needs `score_first.json` written before the extras' run folders exist: VG4 refuses run folders of
  variants outside the first five and the extras (`/tmp/claude-0/eg_checks/trace3/run_trace3.sh` shows how).
- After any change to `eg_score3.py` or its static imports, existing stage iii score files are refused ("stale
  code_sha256; move … aside"); `eg_queue.md` lists what to move.
- Codex's planning-with-files habit commits `task_plan.md`, `findings.md`, `progress.md` into the folder it works in:
  check its commits for them.

## 6. State at handoff

- Running: nothing (no subagent, Codex job, GPU lock or background job of this chat). GPU check at 22:22: driver 595.99.02 answers, kernel module and library 595.99.02, GPU idle.
- Uncommitted: nothing in git; notes in `.scratch/` (outside git).
- On disk over 1 GB: `/tmp/claude-0/final_review_stage_iii/` (5.3 GB: two clones and a trace tree, partly hard links);
  `/tmp/claude-0/eg_checks/trace3/` (3.4 GB); `/tmp/claude-0/codex-fix3` (5.2 GB, Codex pytest trees); all on `.scratch/pending_deletions.md`. The 21f rows unchanged.
