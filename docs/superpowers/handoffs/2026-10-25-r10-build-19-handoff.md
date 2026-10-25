# Handoff: smoke and merge ticket 16 (driver and queues), then the final whole-branch review, the merge into main, and open "r10 run 14"

## Brief
- Done: tickets 04, 05, 06, 07, 13, 15 merged (05 and 15 after Opus `high` reviews and fixes); VP1 agrees with the pick end to end; round tests 313 passed.
- State: 15 of 16 tickets merged; 16 (stage driver, queues) built, CPU smoke of stages 0, 1, 1x passed; GPU smoke not run.
- Next: diff read and GPU smoke of 16, merge; final review (Opus `xhigh`); one fix wave; merge into main; run chat handoff.
- Blocked: nothing. Night mode; GPU check passes; no container restart.

> Written 2026-10-25 05:47 by "r10 build 18", cut at the context meter (over 200k). Loop step reached: step 7 (build) in
> progress. Direction log entry: [2026-10-21 · D2 ablation read](../direction_log.md).

## 1. Read in this order

1. This handoff.
2. The ticket index `/project/CoSiR/.scratch/r10-build-16/issues/00-index.md` and ticket 16 (`16-driver-queues.md`,
   controller notes at its end).
3. `/project/CoSiR/.scratch/r10-build-16/notes/build18_reports.md`: ticket 16's report (smoke commands, env, choices,
   what it did not run) and every run-time estimate of this build.
4. Review findings with the controller's decisions: `notes/review_05_findings.md`, `notes/review_15_findings.md`
   (earlier: 03, 10, 11, 12, 14). Briefs: `notes/implementer_brief.md`, `notes/reviewer_brief.md`.
5. `~/.claude/references/matt-chain.md` (final review, merge) and `loop-chats.md` ("How to cut") before the run handoff.

Skills: none.

## 2. Decided by the user, not to be reopened

As the [build 16 handoff](2026-10-25-r10-build-16-handoff.md) §2: the spec's §4 (D1 to D19, D30). Night mode; no
restart.

## 3. Where things stand

- Integration branch `r10-recipe-backbones` (worktree `/project/CoSiR-wt/r10-recipe-backbones`, pushed, tip 982e4e2).
  Full round suite there: 313 passed, 15 skipped (5.6 min; no `-n`, xdist is not installed).
- Merged in build 18:
  - 07 `rb_sheet`: diff read (formulas checked); real-data run 3 min on CPU; numbers sane (stores unit norm; each
    instructed view raises 20-NN purity on its own aspect).
  - 13 `rb_part2_report`: diff read against rule §3.2; real readers checked on real data (held labels 12,281, genre
    9,949 labelled; T and Fk images; one Leiden reading 10 s). CPU 30 to 40 min per candidate per stage (estimate).
  - 05 `rb_vp1`: review FIX (computation equal to `rb_pick` exactly; real labels identical); four fixes by the
    controller (excluded candidates' exports checked; export record's P hash asserted; `n_draws` compared; dtype and
    width). The part 1 agreement check (fake P exports, 2,000 draws) agrees before and after the fixes: 0 disagreements.
  - 15 `rb_v2`: review FIX; Sonnet `high` fixer (outcome-less blocks skipped; stage 5 scale as `rb_a28`; unchecked
    partitions fail V2; `--ct-check` fails on every mismatch; smoke seeds as the writers'). `--ct-check 1` exact on the
    CT's real outputs; the review's synthetic run of the real writers (`/tmp/claude-0/rv15/synth.py`) agrees after
    the fixes.
  - 04 `rb_part1_report`: diff read; reader keys equal `rb_part1`'s writer; smoke reproduced the pick's R0-5 A to 1e-9.
  - 06 `rb_probe`: diff read; one controller fix (R0-5 P export read only when its record names P.json's P); smoke on
    real images.
- Built, not merged: 16 on branch `r10rb-t16` (worktree `/project/CoSiR-wt/r10rb-t16`, commit c2a79b1, merged with
  982e4e2). Plumbing: no review; controller diff read and smoke decide.
- Codex: the five-hour window had 70% left at 05:43 and resets at 07:23; an implementer or fixer needs 80%.

## 4. Open points for the next step (agent work; nothing for the user)

a. Ticket 16, controller: read the diff of `rb_run.py`'s gates against rule §4.1 and §4.2 and the ticket's list;
   check `vp2_cause` against `rb_part1.check_p`'s real assert messages (the forced-rerun path never ran); then the GPU
   smoke into the scratchpad, under the lock (`flock -o -w … /tmp/gpu0.lock`): `stores`, `stage2`, `stage3`,
   `rb_queue_cpu.sh 3`, `stage3a`, `stage4`, `rb_queue_cpu.sh 4`, `stage5` (or `rb_queue_gpu.sh`), commands in
   `build18_reports.md`. The part 2 smoke reads the real stores; the CLIP B/32 full store does not exist yet. Either
   build it for real first (about 10 min GPU, needed by the run anyway; ticket 08's command) or smoke stage 3 without
   K3, and say which. Small fixes: the controller; larger: a fresh Sonnet `medium` fixer. Then merge 16.
b. Final whole-branch review (Opus `xhigh`, `matt-chain.md`: re-derive load-bearing numbers, hard rule 6, the
   constitution); one fix wave; a scoped re-review (Sonnet `high`). Points for it: `build18_reports.md` "Other open
   choices"; ticket 13's use of `rb_score` private helpers and its Leiden bootstrap copy.
c. Merge into main and push; worktrees `r10rb-t04` to `t16` (and earlier ones still listed) and their branches go on
   `.scratch/pending_deletions.md`.
d. The run chat's handoff: ticket list (index table), queue order (spec §3.5), commands from `rb_queue.md`, the end
   estimate against 11:00 (D32; 16's budget gives 194 to 306 min for the whole chain with four candidates, so a split
   is likely; Sunday, so the weekday-hours rule does not apply), the store commands; "stage 4 ran" means "ran to an
   outcome" (stage 5 scale). Then the GPU check (night mode: check only) and
   `loop-next <handoff> --label "r10 run 14" --unattended`.

## 5. Code and pitfalls

- As the build 17 handoff §5. Also:
  - `rb_vp1` and `rb_probe` now read each P export only after its training record's `p_sha256` equals P.json's.
  - `rb_part2_report` and `rb_part1_report` refuse to rewrite an existing output with different content (a rerun after a
    code change needs `--replace` for 13).
  - Part 2's reported diagnostics are long CPU jobs; `rb_queue_cpu.sh` runs them beside the GPU chain behind its own
    CPU lock.
  - Do not `pkill -f "subagent-watch --watch"` from a Bash call: the pattern matches the calling shell too (exit 144).

## 6. State at handoff

- Running: nothing (no subagents, no background jobs, no GPU lock).
- Uncommitted: nothing (`.scratch/` is outside git).
- On disk over 1 GB (all on `.scratch/pending_deletions.md`): `/tmp/claude-0/-project-CoSiR/aef61a93-9b15-4433-a322-fab20ac96c0f/scratchpad/s0`
  (12 GB, ticket 16's smoke of stages 0, 1, 1x), `/tmp/claude-0/rb18_t04_smoke` (9.5 GB, ticket 04's smoke tree),
  `/tmp/claude-0/rb17_e2e_p1` (4.1 GB, the agreement check; its result is read). System disk: 216 GB free.
