# Handoff: r13 build 4: fix and merge ticket 12, build ticket 13 and run the smoke, then the final review and merge to main

## Brief
- Done: build chat 3 merged tickets 04, 08, 09 (review fixes applied, gates still pass), 10 and 11; 12 of 13 tickets built.
- State: ticket 12 (the C13 re-derivation) is reviewed and waits on one Codex fix round; ticket 13 is not started.
- Next: send ticket 12's fix by Codex resume, build 13, run the smoke, then the final Opus review, fix wave, merge to main.
- Blocked: nothing.

> Written 2026-11-01 16:55 by "r13 build 3". Loop step reached: step 7 (build, 12 of 13 tickets built, 11 merged).
> Direction log entry: [2026-10-31](../direction_log.md). Run mode: night.

## 1. Read in this order

1. `/project/CoSiR/.scratch/r13-photo-finetune/notes/fix_wave.md`: every review finding still open, per ticket; the
   sections for t08, t10, t11 and t12 are new. It is the state of this build.
2. The ticket index `/project/CoSiR/.scratch/r13-photo-finetune/issues/00-index.md`, tickets 12 and 13 beside it,
   `notes/implementer_rules.md`.
3. The ticket 12 fix brief, ready to send: `/project/CoSiR/.scratch/r13-photo-finetune/codex/t12_fix_task.txt`.
4. The spec [2026-10-31-photo-finetune-design.md](../specs/2026-10-31-photo-finetune-design.md) (rule §6 with §3.4,
   §3.5), `~/.claude/references/matt-chain.md` (the final review), `~/.claude/references/codex-jobs.md`.
5. The build 3 handoff [2026-11-01-r13-build-3-handoff.md](2026-11-01-r13-build-3-handoff.md) §3 to §5 and the build 2
   handoff §4 (agent defaults for the run handoff), build 1 §3 to §5 (stores, timing for the run chat).

Skills: `implement-spec` (matt-chain.md overrides it).

## 2. Decided by the user, not to be reopened

The spec's §4 (D1 to D14), approved 2026-11-01 15:33. Nothing new was asked in build chats 1 to 3.

## 3. Where things stand

Integration branch `r13-photo-finetune` at `5d8e103`, pushed (worktree `/project/CoSiR/.claude/worktrees/r13-int`);
183 fast tests pass (`-m 'not slow'` on the round folder).

| # | State | Check |
|---|---|---|
| 01 to 03, 05 to 07 | merged (build 2) | as build 3 handoff §3 |
| 04 | merged; Codex fix round (F1 fallback test with mutation proof, F5 pf_paths out of the code hash) | (iv) passed before the fix; the fix changes no edge |
| 09 | merged; Sonnet fixer: all five SCs' (comparator, metric, r) asserted, three mutants now fail | (ii) SC part passed before; test-only fix |
| 08 | merged; Opus review approve with fixes; Codex fix round: record() checks its own r'_mAP and the 25-item menu; stored-rows test | (ii) stage 2 part rerun after the fix: passes, 135 s CPU |
| 10 | merged; controller diff read; held guard tests with mutation check (5 mutants, each fails a test) | smoke (ticket 13) |
| 11 | merged; controller read of `main()`; self-check on r12's c* test exports (R0 rows equal r12's within 1e-9, gain share as report 12-31) | smoke (ticket 13) |
| 12 | branch `r13-t12` (Codex), worktree `.claude/worktrees/r13-t12`; Opus review: approve with fixes; NOT merged | Codex: 21 fast, 8 slow tests pass (r12 replays within 1e-9); reviewer: near-copy set equal to ours on all 56 items (2,328 removed) |
| 13 | not started | |

- **Ticket 12 next:** send `codex/t12_fix_task.txt` by resume to session `01a197bb-07c0-7a00-9e9f-57081b5dff6d`
  (form in codex-jobs.md, "One fix round by resume"; kind `other`). Fixes before merge: the sheet format (row 0 and a
  skipped row 9; `compare` fails on a correct pipeline today) and tests that `compare` rejects; the brief also folds in
  the r' tests, run.json checks, smoke rows and smoke draws. Then read the fix diff, rerun the fast tests, merge. If
  the fix round fails review, Claude (Sonnet `high`) takes it back from the last commit.
- **Ticket 13** (Sonnet `medium` worker): the queue script and the smoke. Add to its brief: (a) `pf_run.py` stages 1,
  2 and 3 must refuse unless `OUTPUTS/stage0a/gates.json` exists with `ok` true (t10 note 1); (b) the smoke runs
  extraction smoke first (stage 0a needs `OUTPUTS/smoke_views`), then 0a (all parts), 1 with `--rows 1,8`, 2, 3 and
  diag, then `pf_rederive.py derive` and `compare` with the same smoke draws as stage 3 (t12 item 7), with
  `OUTPUTS/records/` written before stage 2; (c) every GPU step under `flock -o -w <seconds> /tmp/gpu0.lock`. The
  controller runs the smoke itself (light GPU, under the lock), in the background.
- **Final review:** after 13 merges and the smoke passes: Opus `xhigh` whole-branch review (matt-chain.md), one fix
  wave with fix_wave.md's notes, a Sonnet `high` re-review of the fixes, merge into main, push, then the run handoff.
- **Codex:** 5h 60% left at 16:55 (reset 20:36), weekly 50%. Ledger rows for 04's and 08's fix rounds and ticket 12
  are in `/project/claude-config/codex/jobs.md` (update 12's outcome after its fix round).
- GPU check at 16:55: driver 595.99.02 answers, module and library match, GPU idle, no lock.

## 4. Open points for the next step

None for the user. Agent defaults to add to the run handoff, beside builds 1 and 2: ticket 10's and ticket 11's
decisions as listed in fix_wave.md (t10 item 2, t11 paragraph); ticket 12 was classed as the `rederivation` Codex kind
(C13 is a re-derivation), so it ran before the 20:36 reset.

## 5. Code and pitfalls

- Build 3's §5 and build 2's §5 still hold.
- `pf_run.py` reuses a finished run (run.json ok or failed); to retrain one, move its folder aside.
- Smoke-only flags of `pf_run.py`: `--rows`, `--max-epochs`, `--draws-clu`, `--draws-ret` (refused in a full run).
- The shell is zsh: a bare `echo ====` fails (`=` expansion); use printf.
- At the end of the build, list on `.scratch/pending_deletions.md`: the worktrees `.claude/worktrees/{r13-t04,
  r13-t08, r13-t12, agent-*}` (nine `agent-*` folders now) and their ticket branches (r13-t01 to r13-t12, worktree-agent-a19d52358fe5356fc, worktree-agent-a0b6a80528eda077e), plus the scratchpads of build
  chats 2 and 3 (small: test outputs, review scripts, mutant copies).

## 6. State at handoff

- Running: nothing (no subagents, no Codex job, no GPU lock).
- Uncommitted: nothing of this chat's in the project (`.scratch/` is outside git by design); the Codex ledger in
  `/project/claude-config` is committed with this cut.
- On disk over 1 GB: nothing new.
