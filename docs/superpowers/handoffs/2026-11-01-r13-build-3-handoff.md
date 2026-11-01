# Handoff: r13 build 3: fix and merge 04 and 09, review and merge 08, then build 10 to 13 and run the final review

## Brief
- Done: build chat 2 merged tickets 01, 02, 03, 05, 06 and 07; their gates (i), (ii) sweep, (iii), (iv) and (vi) pass.
- State: 04 and 09 are reviewed, each waiting on one test fix; 08 is built by Codex with its gate passing, review not started.
- Next: fixers for 04 and 09, an Opus review of 08, merge all three; then tickets 10, 11, 12, 13 and the final review.
- Blocked: nothing.

> Written 2026-11-01 16:24 by "r13 build 2". Loop step reached: step 7 (build, 7 of 13 tickets merged).
> Direction log entry: [2026-10-31](../direction_log.md). Run mode: night.

## 1. Read in this order

1. `/project/CoSiR/.scratch/r13-photo-finetune/notes/fix_wave.md`: every review finding still open, per ticket,
   with what is fix-before-merge and what is a note for the final fix wave. It is the state of this build.
2. The ticket index `/project/CoSiR/.scratch/r13-photo-finetune/issues/00-index.md` (`rows_sha256` is now defined in
   "New views") and tickets 08, 10 to 13 beside it; implementer rules in `notes/implementer_rules.md`.
3. The spec [2026-10-31-photo-finetune-design.md](../specs/2026-10-31-photo-finetune-design.md) (rule §6 with §3.4,
   §3.5), `~/.claude/references/matt-chain.md`, `~/.claude/references/codex-jobs.md`.
4. The build 2 handoff [2026-11-01-r13-build-2-handoff.md](2026-11-01-r13-build-2-handoff.md) §3 to §5 (wave plan,
   agent defaults for the run handoff, pitfalls) and the build 1 handoff §3 to §5 (stores, timing for the run chat).

Skills: `implement-spec` (read its SKILL.md; matt-chain.md overrides it), `tdd` for verdict-code implementers.

## 2. Decided by the user, not to be reopened

The spec's §4 (D1 to D14), approved 2026-11-01 15:33. Nothing new was asked in build chats 1 and 2.

## 3. Where things stand

Integration branch `r13-photo-finetune` at `7dbab24`, pushed (worktree `/project/CoSiR/.claude/worktrees/r13-int`);
112 fast tests pass (`-m 'not slow'` on the round folder).

| # | State | Gate run by the controller (scratch outputs) |
|---|---|---|
| 01 | merged; Opus review, fixes applied | none (E and T wait for 03's real views) |
| 02 | merged; Opus review approve | review re-derived 2,328 removed keys exactly |
| 03 | merged; diff read | (i) GPU parity: min cos 0.99974 images, 0.99935 texts |
| 05 | merged; diff read | (vi) GPU: within 1e-4 (CPU run 1.2e-8), 35 s |
| 06 | merged; diff read | (iii) CPU: ten arrays within 1e-3, 26 s |
| 07 | merged; Opus review approve | (ii) sweep part: AMI and 5-seed mAP difference 0.0 |
| 04 | branch `r13-t04` (Codex), worktree `.claude/worktrees/r13-t04`; review: one test fix (F1) | (iv) CPU: all three r10 graphs exact, 260 s |
| 09 | branch `r13-t09`, worktree `.claude/worktrees/agent-a1828fa2903be6e27`; review: one test fix | (ii) SC part, full draws: passes, 94 s |
| 08 | branch `r13-t08` (Codex), worktree `.claude/worktrees/r13-t08`; Opus `high` review not yet started | (ii) stage 2 part: passes, 182 s |
| 10 to 13 | not started; 10 needs 04, 08, 09 merged | |

- Fixes before merge: 04 F1 (a test that sees the zero-degree fallback) and F5 (drop `pf_paths.py` from
  `pf_graphs`' code hash), 09 finding 1 (assert the comparator and r for all five SCs). A fresh Sonnet `high` fixer
  per ticket in the existing worktree (the 5-minute rule); the controller reads the fix diff, reruns the tests, merges.
- Ticket 10 must take two review notes: pass `pf_graphs.load(...)`'s dict to `train_run`, never the `emotion.npz` path
  (04 F4); assert in stage 3 that the read-out is `STAGE2_FILE`'s pick, r is the photo model's r', and `val_embed` has
  4,251 rows (09 note 4). Ticket 11 must carry the PercepT core and CLIP L/14 rows (09 note 6).
- Codex: 5h 76% left at 16:24 (reset 20:36), weekly 52%. No room for an implementer (needs 80%) until the reset;
  ticket 12 is planned for Codex after 20:36 (Claude Sonnet `high` fallback). Ledger rows for 04 and 08 are in
  `/project/claude-config/codex/jobs.md` (update 08's outcome after its review).
- Timing measured for the run chat: graph build about 2.2 min at 42,969 nodes (k 40), estimated 4.5 to 5.5 min at
  70,000 (up to about 2× under CPU load), peak about 7 to 8 GB; near-copy run about 40 s; gate (iv) 4.3 min CPU.
  Trainer: about 8 s per epoch estimated (5 to 16 s), not measured.
- GPU check at 16:24: driver 595.99.02 answers, module and library match, GPU idle, no lock.

## 4. Open points for the next step

None for the user. Agent defaults to add to the run handoff, beside build 2's list: train keys are derived from the
index's text rows (equivalent to r12's rule, re-derived; `ac_data.keys` still parses the CSV's train emotion column in
an assert, no value used); gate (i) image parity uses the first 100 sorted val keys plus the first 100 FI val keys;
the within-train copy count (10,114) counts photos, 1,057 of them also removed.

## 5. Code and pitfalls

- Build 2's §5 and build 1's §5 still hold.
- `pf_data` checks the new views strictly (meta `view` = folder name, `smoke` = `PF_SMOKE > 0`, `rows_sha256`, unit
  rows); `pf_extract` writes them to pass. `fit_keys()` refuses a near-copy file whose `n_train` differs.
- Slow tests write into pytest's tmp folders, not the real `OUTPUTS`; the real `NEARCOPY_FILE` and stage 0a evidence
  are written by the stage runner (ticket 10) in the run chat.
- `/usr/bin/time` is not installed; time a command with `date +%s`.
- At the end of the build, list on `.scratch/pending_deletions.md`: the worktrees `.claude/worktrees/{r13-t04,
  r13-t08, agent-ad9ecc90885013acf, agent-ac290cd9ba087ab4b, agent-ae22522710e8e13e2, agent-a1828fa2903be6e27,
  agent-a9ae6f2143e926c06, agent-adc47d6b7426a9c7c, agent-ae2f9b14efc2c26e7}` and their ticket branches, plus this
  session's scratchpad (small: test outputs and review scripts).

## 6. State at handoff

- Running: nothing (no subagents, no Codex job, no GPU lock).
- Uncommitted: nothing of this chat's in the project (`.scratch/` is outside git by design).
- On disk over 1 GB: nothing new.
