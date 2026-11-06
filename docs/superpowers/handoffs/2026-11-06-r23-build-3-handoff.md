# Handoff: r23 build 3: fix tickets 5 and 6, merge, build tickets 9 and 10, final review, merge to main

## Brief
- Done: tickets 1, 2, 3, 4, 7 and 8 are reviewed and merged. Tickets 5 and 6 are built and reviewed; each passes after named fixes.
- State: no held score exists. Replay gate (iii) passes all 8 clauses on a scratch run: Table 2's largest difference is 2.65e-4 against a bar of 3e-4.
- Next: fix ticket 5, merge, fix ticket 6, merge, build tickets 9 and 10, final review, merge to main, cut to the run chat.
- Blocked: nothing. The cut came from the context meter (past 200k tokens), not from a problem.

> Written 2026-11-06 06:56 by tab "r23 build 2". Loop step reached: inside step 7. Direction log entry: [Pair-hit checks read](../direction_log.md), "Grill and spec".

## 1. Read in this order

1. `/project/CoSiR/.scratch/r23-text-claim/issues/00-index.md`: ticket list, shared interfaces (updated: `status.json` with `c13_void`, the `cells.json` fields), coverage line. Tickets sit beside it, outside git.
2. The [spec](../specs/2026-11-06-text-claim-design.md), from §1. Its §3.4, §3.5 and §6 are the rule. Skip the brief.
3. `/project/CoSiR/.scratch/r23-text-claim/notes/t05_review.md` and `t06_review.md`: the two fix lists.
4. `~/.claude/references/matt-chain.md` and `codex-jobs.md`: the chain, and Codex's launch form.
5. The [second build handoff](2026-11-06-r23-build-2-handoff.md) §5 and the [first build handoff](2026-11-06-r23-build-handoff.md) §5: code pitfalls, all still true.

Skills: tdd (verdict code).

## 2. Decided by the user, not to be reopened

All of the spec's §4, D1 to D14. A change to the rule is an amendment (hard rule 5).

## 3. Where things stand

| Ticket | Branch, commit | State | Next |
|---|---|---|---|
| 01 metrics | merged | Opus review passed | done |
| 02 data door | merged | 55 tests; all 20 review faults caught on the controller's rerun | done |
| 03 extraction | merged | controller read the diff; smoke passed on every job | done |
| 04 forward | merged | gate (ii) passes; 26 tests, CPU and GPU; 37 of 38 faults caught, one equivalent. `mm` now includes the caption check (controller's decision, since MMArt reads the same caption side) | done |
| 07 stage 1 | merged | 31 tests; 49 of 49 faults caught on the controller's rerun | done |
| 08 outcomes | merged | 92 tests; a plain-loop recomputation matched all 76 cells | done |
| 05 scoring | `r23-t05`, 05afcdb6, worktree `.claude/worktrees/r23-t05` | 44 tests; GPU smoke passes on every row type (864 files in 2 minutes); review: all rule points pass, recompute within 1.8e-7, 0 of 216,648 first ranks differ | fix list in `t05_review.md` (fixes 1 to 6), then merge |
| 06 replay | `r23-t06`, c7c1209d, worktree `.claude/worktrees/r23-t06` | 40 tests; full gate passed on the GPU in 2.2 minutes; review: no fix changes a number | fix list in `t06_review.md` (fixes 1 to 6), after ticket 5's merge, then merge |
| 09 runner, 10 diagnostics | none | not started | blocked by 05 and 06 only |

Gate (iii) on scratch, largest difference against its bar: `core_label` 3.6e-7, `late_label` 7.8e-7, `late_pair` 0, `mmart_label` 6.0e-8, `mmart_pair` 3.0e-8 (all 1e-5); `interval_hand` 1.7e-17, `interval_tie` 5.6e-17 (1e-12); `table2` 2.65e-4 (3e-4).

Integration worktree: `/project/CoSiR/.claude/worktrees/r23-int`, branch `r23-text-claim`, 6b7f3384, pushed. Tickets 5 and 6 sit on an older ticket 4 commit, so merge `r23-text-claim` into each first. Both touch new files only, so merges should be clean.

Ticket order: ticket 5 fixes and merge; then in `r23-t06`, merge the integration branch and do its fix 1, which replaces `means_for` by `tcl_score.query_gallery_means`. A rerun of the full gate follows. Then 9 and 10 can run side by side.

## 4. Open points for the next step

None for the user. Notes for the final review and the run chat:
- **C13's "within 1e-6" (from ticket 5's review, measured on full `art_val`, plain, t2t, raw cosine).** Tie order alone moves a single query's AP by up to 8.3e-5, while mAP moves at most 3.2e-9. Spec §3.4 lists per-query AP among C13's items within 1e-6, so a re-derivation with another tie order fails at query level for no real reason. The reviewer proposes holding 1e-6 at cell level. This touches the rule: the final review decides whether it is a reading of the rule or an amendment, and the run chat's C13 brief must follow that. Giving C13 the same ranking (torch float32, the same block order) is the option that needs no amendment; check it first.
- **Scoring time.** 4,002 rankings, about 1.5 h in total against the spec's "about 1 hour" (reviewer's estimate, not measured at full size). The longest job, `art_held`, takes about 43 minutes, so one job per set and kind stays under 1 hour and ticket 09 needs no split. The report states the true time.
- **Block size.** `BLOCK = 1024`: 4.9 GiB allocated, 6.8 GiB reserved, measured at 61,744 by 61,744, width 4,096, on synthetic rows.
- **Exact ties in label picks (ticket 7's recheck, in `t07_review.md`).** A float mean over seeds can break a tie that is exact in fractions. Real AP values tie exactly only when two rankings are identical, and then the floats are identical too. Left as it is; the final review may reopen it.

## 5. Code and pitfalls

For ticket 09 (`09-run.md`):
- `status.json`'s format is in the index. Set `gate_iii` from `gate_iii.json`'s `overall`, and only after `tcl_replay.py` exits 0. `gate_ii` and `gate_i` keys: as the index says. `c13_void` is written only when a C13 difference stays after its trace.
- The end-to-end smoke must score all five MMArt kinds, since stage 1 reads them (ticket 5's smoke scored one). A failed rerun of stage 1 must not leave the earlier `stage1_*.json` in place as if it were current (ticket 7's reviewer).
- The held guard goes in `tcl_score.py`'s path too. `tcl_score.py` has no guard yet.
- `tcl_replay.py` lifts `TCL_SMOKE` inside `full_rows()`, so gate (iii) always replays full validation rows, about 2 minutes on the GPU. Smoke needs no subset.
- `--smoke` on `tcl_score.py` must agree with `TCL_SMOKE`, or it crashes.
- Add `gate_iii.json` to the index (ticket 6's review). The log time at `.claude/20261106_log.md:3` is wrong: fix it.

Working here:
- Codex at 06:55: 39% left of five hours, reset 09:53; weekly 40%. `--room implementer` says no room below 80%, so before the reset implementer tickets go to Sonnet `high` workers (verdict code) or Sonnet `medium` (plumbing). `--room bug` needs 20%. A fix round is a `bug` job.
- Cost of a Codex fix round: 6% to 10%, but about 27% when the brief asked it to rerun 38 faults with GPU tests. Ask Codex for the fixes and tests only. The controller reruns the faults: the reviewers' scripts are in this chat's scratchpad, `/tmp/claude-0/-project-CoSiR/7da119dc-f293-4d1e-86a7-6a9ce75014d8/scratchpad/` (`t05_review/`, `t06_review/`), and `t02_rerun/rerun.py` rebuilds faults from mutant copies.
- Codex stops on "if a number disagrees, stop": ticket 4 lost a run on 16,782 against 16,384, which are r15's `new` export and its test cut. Name such pairs in the brief.
- A resumed Codex job without `--json` writes plain text: its final note is at the log's tail, not in an `agent_message`.
- `git worktree add` from inside a worktree nests the new one when the path is relative. Use absolute paths. The Bash working folder moves with `cd`.
- `*.json` is gitignored: results files need `git add -f`.
- After the final review (`opus` `xhigh`) and the scoped re-review (`sonnet` `high`): merge into main, push main, list worktrees and ticket branches on `.scratch/pending_deletions.md`, then cut to the run chat with the ticket list in its handoff.
- Run timing (D13): 2026-11-06 is a Friday. Heavy jobs that would fall between 10:00 and 19:00 wait for 19:00. Report by 2026-11-07 12:00, at the latest 2026-11-08.

## 6. State at handoff

- Running: nothing. No queue job, Codex job or subagent.
- Uncommitted: nothing of this chat, apart from the Codex ledger rows in `/project/claude-config/codex/jobs.md`, which the next `sync.sh` commits.
- On disk over 1 GB: nothing new. This chat's scratchpad holds 138 MB with no file over 100 MB; the next chats need its review scripts, so it stays.
- GPU check before this unattended chat: passed. Driver 595.99.02 in the module and the library.
