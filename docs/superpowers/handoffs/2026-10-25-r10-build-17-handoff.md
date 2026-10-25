# Handoff: go on with the r10 build (tickets 03 to 16), the final review and the merge, then open "r10 run 14"

## Brief
- Done: 16 tickets written; 01, 02, 09, 12 merged (stage 0: 9 of 9 exports bit for bit; scorer check exact).
- State: 10 (student) built, its GPU regressions pass at both scales; three review fixes owed before its merge.
- Next: 10's fixer, then the frontier (03, 05, 07, 08, 11, 15; then 04, 06, 13, 14; 16), final review, merge.
- Blocked: nothing. Night mode; no container restart (user).

> Written 2026-10-25 04:18 by "r10 build 16", cut at the context meter (over 200k). Loop step reached: step 7 (build) in
> progress. Direction log entry: [2026-10-21 · D2 ablation read](../direction_log.md).

## 1. Read in this order

1. This handoff.
2. The ticket index `/project/CoSiR/.scratch/r10-build-16/issues/00-index.md`: the table with each ticket's state,
   the shared interfaces (binding) and the coverage line. Tickets are the `NN-*.md` files beside it.
3. The two briefs every subagent gets: `.scratch/r10-build-16/notes/implementer_brief.md` and `reviewer_brief.md`.
   The code map `.scratch/r10-build-16/notes/code_map.md` is for subagents; read only what you need.
4. The spec `docs/superpowers/specs/2026-10-25-recipe-and-backbones-design.md` (§3, §9) and the rule
   `src/test/20261025_recipe_backbones/DECISION_RULE.md` only as far as a ticket or review needs.
5. `~/.claude/references/matt-chain.md` (build, reviews, final review, merge) and `codex-jobs.md`.

Skills: none (`to-tickets` was done; `implement-spec` per `matt-chain.md`).

## 2. Decided by the user, not to be reopened

As the [build 16 handoff](2026-10-25-r10-build-16-handoff.md) §2: the spec's §4 (D1 to D19, D30). Night mode; no
restart.

## 3. Where things stand

- Integration branch `r10-recipe-backbones` (worktree `/project/CoSiR-wt/r10-recipe-backbones`, pushed), from main
  at cf8f13e. Main checkout stays on main. Tests: `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
  CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider
  src/test/20261025_recipe_backbones` in that worktree (94 passed, 8 skipped after 12; `RB_SLOW=1` adds real-data
  tests of `rb_data`).
- Skeleton by the controller: `rb_paths.py` (frozen artifacts read from the main checkout only), `rb_rows.py` (the
  six recipes; R0-5, R6-3, R7-3 equal 12-22's R0, R6, R7).
- Merged and reviewed:
  - 01 `rb_data` (Opus `high` review: pass; P sha256 `169f137e…1ef7`, 29,232 = 23,080 + 6,152; genre labelled on
    23,682; labels and feature hashes re-derived from raw sources).
  - 09 `rb_encoders` (Opus `high` review: one fix, applied by the controller: the instructed ArtELingo-28 reader checks
    revision, count and ID hash; ArtELingo-28 list hash `c1f1b27e…2b70`).
  - 02 `rb_part1`, `rb_stage0` (plumbing; controller diff read; real stage 0 on the GPU, 2026-10-25: 9 of 9 old-5,000
    exports and best epochs bit for bit equal to 12-22's; scoring one P export: partitions 7.5 s, one AMI call
    0.0145 s, the part 1 bootstrap about 26 min on one process).
  - 12 `rb_score` (Opus `high` review: two should-fix and two minor, applied by a fresh Sonnet `high` fixer: sidecar
    checks for new arms, the stage gate needs `student/`, `matched/` and `scorer/` items, one candidate's V3/V4 failure
    no longer stops the others, locked regression writes). `--check stage2` reproduces the claim test's stage 1 values
    exactly (every A_e, d, both bounds, all 2,000 bootstrap values; run twice); `--check stage3a` passed once in the
    implementer's run (not rerun after the fixes; the run chat runs both into the real outputs).
- Built, not merged: 10 `rb_student` on branch `r10rb-t10` (worktree `/project/CoSiR-wt/r10rb-t10`). Controller GPU
  regressions passed (2026-10-25): `regress stage2` and `regress stage3a`, Qwen + R0-5 seed 0, weights, best epochs
  and held outputs of all three heads bit for bit equal to the claim test's, and the K1 replay. Opus `high` review:
  FIX, findings in `.scratch/r10-build-16/notes/review_10_findings.md` (3 should-fix: V1 gate, V4 any-seed refusal,
  use `rb_score.write_regression_items`; 3 minor). Next: a fresh Sonnet `high` fixer on that branch, the controller's
  diff read, then merge (the review re-derived the input path; no new review needed for these fixes).
- Records of the controller's GPU checks (stage 0, student regressions): `.scratch/r10-build-16/notes/gpu_checks/`.
- Ready now (blockers merged): 03 (pick), 05 (VP1, independent), 07 (sheet), 08 (stores), 11 (matched arm),
  13 and 14 (need 09, 12), 15 (V2, independent). Then 04, 06 (need 03); 16 (needs all).
- Codex: the five-hour window had 70% left at 03:45 and resets at 07:23; an implementer needs 80%
  (`codex-usage --room implementer`). One Codex implementer at a time after the reset; the rest go to Claude.

## 4. Open points for the next step (agent work; nothing for the user)

a. Dispatch at most three implementers at once (`worker`, Sonnet `high` for verdict code, `medium` for plumbing),
   each in its own worktree from the integration branch (`git worktree add -b r10rb-tNN /project/CoSiR-wt/r10rb-tNN
   r10-recipe-backbones`); the prompt names the worktree, the brief and the ticket. Keep `subagent-watch --watch`
   running in a background Bash.
b. Verdict-code tickets (03, 05, 10, 11, 12, 14, 15) get an Opus `high` review (`reviewer_brief.md`) before the merge;
   others a controller diff read plus their smoke run (GPU smoke under `flock -o -w … /tmp/gpu0.lock`, outputs in the
   scratchpad). Small, fully specified review fixes: the controller applies them; larger ones go to a fresh fixer.
c. Controller GPU checks still owed: ticket 10's `--regress stage2` and `stage3a`, ticket 11's matched-arm
   regressions, 12's `--check stage2|stage3a` into the real outputs only in the run chat (the build's runs go to the
   scratchpad).
d. Re-derivations 05 and 15: the implementer must not open the main scorer (the ticket lists the files). Codex is a
   natural fit for one of them after 07:23 (independent of the Claude builders).
e. Final whole-branch review (Opus `xhigh`, `matt-chain.md`), one fix wave, a scoped re-review (Sonnet `high`); merge
   into main and push; worktrees and ticket branches go on `.scratch/pending_deletions.md`.
f. The run chat's handoff: ticket list (index table), the queue order (spec §3.5), commands from `rb_queue.md`, the end
   estimate against 11:00 (Sunday: if part 2 ends after 11:00, the run chat moves it and records why, D32); then
   `loop-next <handoff> --label "r10 run 14" --unattended`.

## 5. Code and pitfalls

- Change no code in earlier rounds; `rb_paths.py` and `rb_rows.py` only by the controller.
- Frozen artifacts (12-22's exports, the claim test's outputs, weights, graphs, R-COL) exist only in the main
  checkout; worktrees import code but read artifacts through `rb_paths`.
- Regression files `part2/regression/stage{2,3a}.json`: items are a dict merged by name (index); the stage 2/3a gate
  must require every expected item name (ticket 16).
- Ticket 08 must write stores in the exact form 09's readers check (listed at the end of ticket 08).
- `rb_part1` saves an epoch-4 state too, so P exports can be added to a P-off run without retraining (`add_p`).
- 12's scorer: `seedless=` arms (Fk) draw nothing; the verdict draw order is the rule's.

## 6. State at handoff

- Running: nothing of this chat.
- Uncommitted: nothing (`.scratch/` is outside git).
- On disk over 1 GB: nothing new in the project; the stage 0 check's 1.6 GB in this session's scratchpad is deleted.
