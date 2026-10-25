# Handoff: the final review's fix wave and scoped re-review, the merge into main, then the run handoff and open "r10 run 14"

## Brief
- Done: ticket 16 smoked on the GPU end to end (every queue and runner) and merged; final review (Opus `xhigh`, two parts): FIX, no blocker; part 1's findings fixed; real CLIP B/32 and ArtELingo-28 Qwen stores built.
- State: 16 of 16 tickets merged on `r10-recipe-backbones` (tip 366c4e2); round suite 326 passed, 14 skipped.
- Next: fix wave for review part 2 (three small code fixes, two notes), scoped re-review, merge into main, run handoff.
- Blocked: nothing. Night mode; GPU check passes; no container restart.

> Written 2026-10-25 06:36 by "r10 build 19", cut at the context meter (over 200k). Loop step reached: step 7 (build),
> final review done, fix wave next. Direction log entry: [2026-10-21 · D2 ablation read](../direction_log.md).

## 1. Read in this order

1. This handoff.
2. `/project/CoSiR/.scratch/r10-build-16/notes/final_review_2_findings.md`: review part 2 and, at its end, the
   controller's decision per finding (what the fix wave does).
3. `notes/final_review_findings.md` (part 1, with the controller's section at its end), for the scoped re-review.
4. The [build 19 handoff](2026-10-25-r10-build-19-handoff.md) §4 c and d (merge, the run handoff's contents) and
   `notes/build18_reports.md` (run times).
5. `~/.claude/references/matt-chain.md` ("The final whole-branch review", merge) and `loop-chats.md` ("How to cut").

Skills: none.

## 2. Decided by the user, not to be reopened

As the [build 16 handoff](2026-10-25-r10-build-16-handoff.md) §2: the spec's §4 (D1 to D19, D30). Night mode; no
restart.

## 3. Where things stand

- Integration branch `r10-recipe-backbones` (worktree `/project/CoSiR-wt/r10-recipe-backbones`, pushed, tip 366c4e2) =
  `r10rb-t16` fast-forwarded. Full round suite there: 326 passed, 14 skipped (5.9 min).
- Ticket 16, controller diff read against rule §4.1, §4.2 and the ticket: gates as the rule says. `vp2_cause` matches
  `rb_part1.check_p`'s real AssertionError lines (nonfinite rows, norm error; tested on real failures). Controller
  fixes (ca7bd70): a failed candidate in stages 3 and 4 leaves only itself without an outcome (spec §9: stops only at
  stages 0, 2, 3a); stage 1 is skipped once VP1 agrees, finished recipes and 1x outputs are skipped on a restart.
- GPU smoke (`rb_queue_gpu.sh`, smoke mode, on a hardlinked copy of build 18's stage 0 to 1x tree): every queue ran to
  its end, `RB_GPU_DONE failed=0`. Restart skips worked; K5 failed as expected (no DINOv2 store) and the others went
  on; stage 2 and 3a regressions passed (27 items each, run for real); stage 5 gave blocks for K1 to K4. Stage 3 gave
  no outcome in smoke (below, §5), so stage 4 was smoked on a second copy with stage 3 outcomes set by hand: the
  full-scale student, inference and matched arms of K2, K3, K4 ran.
- Final review part 1 (stopped at its context limit): re-derived exactly, by its own code, the CT's d at both scales
  (sample 0.014596, full 0.013341, every A_e and all 2,000 bootstrap draws), part 1's 15 effects and the pick on the
  smoke tree, the ArtELingo-28 join (746 paintings, 17 ties) and d_A28. Finding 1 (reruns after a deferral) fixed:
  `rb_run.candidates(..., todo=True)` (c793253) and `rb_part2_report` per-block hashes (c7fcd2c, 366c4e2). Finding 2
  (a failed 3a also stops stage 5) kept as a documented agent default in `rb_queue.md`.
- Final review part 2: 18 mutations of rule thresholds and verdict checks all caught; hard rule 6 no gap in V0, item
  names, C6 joins, `extract_features` (D26, D27), probe, sheet, part 1 report; every SC has a file and key.
- Real stores built this chat (GPU, ticket 08's commands): `/data/SSD2/pre_extract/artelingo_full/clip_b32/` (61,402
  images, 308,723 captions, 397 MB, 6 min) and `artelingo28/qwen3vl_emb_2b_aspect_prompts/{emotion_focus,style_word,
  genre_word}/` (746 each, 3 min). The stores queue skips them; DINOv2 stores are built only if K5 is triggered.
- Codex: five-hour window 70% left at 06:34, resets 07:23.

## 4. Open points for the next step (agent work; nothing for the user)

a. Fix wave (`final_review_2_findings.md`, controller decisions): items 1, 3, 4 are code with tests in `rb_run.py` and
   `test_rb_run.py` (plumbing): one fixer, Codex if `codex-usage --room` says room, else a Sonnet `medium` worker,
   on a new branch from `r10-recipe-backbones`. Items 2 and 5 are notes: 2 in `rb_queue.md` (controller), 5 for the
   run handoff.
b. Scoped re-review (Sonnet `high`): the fix-wave diff plus build 19's fixes (`git diff 982e4e2 r10-recipe-backbones`
   on `rb_run.py`, the queue scripts, `rb_part2_report.py`, `rb_queue.md`). Then merge into main and push; worktrees
   `r10rb-t04` to `t16` and earlier ones still listed, and their branches, go on `.scratch/pending_deletions.md`.
c. The run chat's handoff: as the build 19 handoff §4d, and also:
   - start time and end: run `rb_run.py budget` at start (from 08:30 it gave 14:23 to 16:40 before the stores were
     built; the part 2 report queues, CPU, set the end). Past 11:00, so the run chat splits by D32 and records why
     (Sunday: the weekday-hours rule does not apply);
   - a rerun after a deferral now trains and scores only the new candidates (`rb_queue.md`, "Reruns");
   - report notes: part 2 makes no forced rerun (review 2 item 2); stage 0's scoring time is a stand-in on a random
     array (item 5); stage 5 needs a passed 3a (agent default); "stage 4 ran" means "ran to an outcome".
   Then the GPU check (night mode: check only) and `loop-next <handoff> --label "r10 run 14" --unattended`.

## 5. Code and pitfalls

- As the build 18 handoff §5. Also:
  - Smoke mode trains seed 0 only (`rb_student`, `rb_matched`), while `rb_score --smoke` reads all five seeds, so a
    smoke stage 3 or 4 never gives an outcome. The real run trains five seeds; this matters only for smoke tests.
  - A `reported_stage{3,4}.json` written before c7fcd2c has `verdict_sha256` in its header and is refused; only the
    smoke trees hold one.
  - `rb_queue_gpu.sh` restarts cleanly now (stage 1, 1x finished work skipped); the stage 2 and 3a regressions run
    again on a restart (5 to 8 min each).
  - The shell is zsh-like: an unquoted `=====` fails; quote it.
  - Do not `pkill -f "subagent-watch --watch"` from a Bash call (matches the calling shell).

## 6. State at handoff

- Running: nothing (no subagents, no background jobs, no GPU lock).
- Uncommitted: nothing (`.scratch/` is outside git).
- On disk over 1 GB (all on `.scratch/pending_deletions.md`): this chat's scratchpad
  `/tmp/claude-0/-project-CoSiR/d14991a9-632f-476b-84dc-a5e9238e928d/scratchpad` (16 GB as counted; its `rb/part1/train`
  is hardlinked to build 18's `s0`, so both go together), plus build 17 and 18's items. System disk: 211 GB free.
