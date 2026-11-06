# Handoff: r23 build 4: build tickets 9 and 10, end-to-end smoke, final review, merge to main, cut to the run chat

## Brief
- Done: tickets 5 and 6 are fixed, rechecked and merged, so tickets 1 to 8 are done. All 411 tests pass on the merged code.
- State: no held score exists. Replay gate (iii) still passes all 8 clauses: Table 2's largest difference is 2.65e-4 against a bar of 3e-4.
- Next: build tickets 9 and 10 side by side, run the end-to-end smoke, final review, merge to main, cut to the run chat.
- Blocked: nothing. The cut came from the context meter (past 200k tokens), not from a problem.

> Written 2026-11-06 07:12 by tab "r23 build 3". Loop step reached: inside step 7. Direction log entry: [Pair-hit checks read](../direction_log.md), "Grill and spec".

## 1. Read in this order

1. `/project/CoSiR/.scratch/r23-text-claim/issues/00-index.md`: ticket list and shared interfaces (now with `gate_iii.json`). Tickets sit beside it, outside git: read `09-run.md` and `10-diag.md`.
2. The [spec](../specs/2026-11-06-text-claim-design.md), from §1. Its §3.4, §3.5 and §6 are the rule; §3.6 is ticket 10's list. Skip the brief.
3. The [third build handoff](2026-11-06-r23-build-3-handoff.md) §4 (open points for the final review and the run chat, all still open) and §5 (the ticket 09 notes and "Working here"). Its fix lists are done, so skip those.
4. `~/.claude/references/matt-chain.md` and `codex-jobs.md`: the chain and Codex's launch form.
5. `src/test/20261102_mmart_test/mt_run.py` and `src/test/20261105_text_retrieval/tx_queue.sh`: the patterns ticket 9 follows.

Skills: none for ticket 9; the guard's test is a plain mutation check.

## 2. Decided by the user, not to be reopened

All of the spec's §4, D1 to D14. A change to the rule is an amendment (hard rule 5).

## 3. Where things stand

| Ticket | State | Evidence |
|---|---|---|
| 1, 2, 3, 4, 7, 8 | merged | see the third build handoff, §3 |
| 5 scoring | merged | 92 tests. 57 of 58 faults caught on the controller's rerun, against 37 of 52 before the fixes. The survivor is the smoke check in `main()`; `tcl_paths.py:33` holds the guard that matters |
| 6 replay | merged | 51 tests. 67 of 70 faults caught. The survivors are 55, 56 and 58: the real gate catches two, and the third only touches smoke runs. Review fixes 1 to 7, 9, 10, 11 are in; optional fix 8 is skipped |
| 9 runner, 10 diagnostics | not started | blocked by nothing now |

Gate (iii) on scratch after the fixes, in 2.1 minutes on the GPU, largest difference against its bar: `core_label` 3.6e-7, `late_label` 7.7e-7, `late_pair` 0, `mmart_label` 6.0e-8, `mmart_pair` 3.0e-8 (all 1e-5); `interval_hand` 1.7e-17, `interval_tie` 5.6e-17 (1e-12); `table2` 2.65e-4 (3e-4).

The integration worktree is `/project/CoSiR/.claude/worktrees/r23-int`, branch `r23-text-claim`, f2cc789f, pushed. All round tests on it: 411 passed in 4 minutes.

## 4. Open points for the next step

None for the user. The four notes for the final review and the run chat are in the third build handoff, §4: C13's "within 1e-6", scoring time, block size, and exact ties in label picks.

## 5. Code and pitfalls

Who builds:
- Codex at 07:10: 35% left of five hours, reset 09:53; weekly 39%. `--room implementer` needs 80%, so before the reset ticket 9 goes to a Sonnet `medium` worker (plumbing).
- Ticket 10 goes to a Sonnet `high` worker: it is not verdict code, but the report cites its numbers.
- A Codex fix round asked for fixes and tests only cost 4% (ticket 5). The controller reran the faults.

What ticket 9 must know, beyond the third handoff's list:
- `tcl_stage1.py` and `tcl_data.gate_iv()` have no command line. `tcl_outcome.py`, `tcl_replay.py cuda`, `tcl_forward.py cuda` and `tcl_extract.py <job>` do.
- Nothing writes a gate (iv) file. Ticket 9 writes `OUT/stage0/gate_iv.json` with `art` and `mm`. `gate_iv()` asserts both sets at once, so a crash marks both false, and the run chat traces which set failed.
- `tcl_replay.py` exits 1 on a failed gate and on a crash. It removes the old `gate_iii.json` at its start, so a crash leaves no file. Use `mt_run`'s `ok_rcs` with the output check, and do not retry a gate that reports a fail.
- `tcl_forward.py`'s main prints the result and exits 0 even when `pass` is false. Read `gate_ii.json`'s `pass`.
- `tcl_stage1.run` overwrites its record but does not remove it first. Remove it before the step (ticket 7's reviewer).
- Put the held guard at the top of `tcl_score.run()`. `score_unit`'s tests call it on `art_held` directly on tiny stores. `test_tcl_score.py` also calls `S.run` on `art_held` and `mm_test` with the data modules patched, so that test must stub the guard or supply smoke records.
- `unit_done` now takes the frame: `unit_done(set_name, kind, row, head, seed, fr, out=None)`.
- `status.json` keeps `c13_void` when it is rewritten.

Between tickets 9 and 10: agree the command line in both briefs, `tcl_diag.py [--device D]`, exit 0 on success, and the runner calls it for stage 3. Ticket 9 owns `tcl_score.py`'s changes; ticket 10 adds only `tcl_diag.py` and its tests.

Ticket 10's worker needs score files for quick checks. Earlier smoke outputs: `/tmp/claude-0/-project-CoSiR/7da119dc-f293-4d1e-86a7-6a9ce75014d8/scratchpad/t05_smoke/out/` (read only, 15 MB). The real smoke is the controller's, after both merges.

Working here:
- Worktrees by hand, from the integration branch, with absolute paths: `git -C /project/CoSiR/.claude/worktrees/r23-int worktree add /project/CoSiR/.claude/worktrees/r23-t09 -b r23-t09 r23-text-claim`.
- Tests of all round files take 4 minutes on 8 threads, so use `res-queue cpu --short`.
- `.claude/20261106_log.md` is tracked although `.claude` is ignored. Two tickets that append to it conflict at merge. Keep both sides, then `git add -f`. Tell ticket 9 and ticket 10 to append; the controller resolves it.
- Fault rerun scripts, updated to the merged code: `/tmp/claude-0/-project-CoSiR/0ee403f2-8ccd-4a1f-8732-ea673e82aed0/scratchpad/t05_mut/mutate.py` (no argument; 3 minutes) and `t06_mut/mutate.py` (argument: its folder; 7 minutes). Run them through `res-queue cpu`.
- In zsh, `echo =====` fails. An unquoted heredoc eats backslashes in Python patterns: use `<<'EOF'`.
- After the final review (`opus` `xhigh`) and the scoped re-review (`sonnet` `high`): merge into main, push main, list worktrees and ticket branches on `.scratch/pending_deletions.md`, then cut to the run chat with the ticket list in its handoff.
- Run timing (D13): 2026-11-06 is a Friday. Heavy jobs that would fall between 10:00 and 19:00 wait for 19:00. Report by 2026-11-07 12:00, at the latest 2026-11-08.

## 6. State at handoff

- Running: nothing. No queue job, Codex job or subagent.
- Uncommitted: nothing of this chat, apart from one Codex ledger row in `/project/claude-config/codex/jobs.md`, which the next `sync.sh` commits.
- On disk over 1 GB: nothing new. This chat's scratchpad holds 1 MB, and the next chats use its scripts, so it stays. Ticket branches 05 and 06 and their worktrees stay until the build's end, when they go on `.scratch/pending_deletions.md` with the rest.
- GPU check before this unattended chat: passed. Driver 595.99.02 in the module and the library.
