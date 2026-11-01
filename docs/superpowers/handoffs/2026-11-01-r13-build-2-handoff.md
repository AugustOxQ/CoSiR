# Handoff: r13 build 2 — dispatch the 13 tickets, review, final review, merge; then open the run chat

## Brief
- Done: build chat 1 wrote the code map (Codex), the 13 tickets and the shared paths module (`pf_paths`); cut at 200k context.
- State: integration branch `r13-photo-finetune` (one commit, `db2a006`); no ticket started yet.
- Next: dispatch wave 1 (tickets 01, 04, 05), then the rest by the frontier; reviews; final Opus `xhigh` review.
- Blocked: nothing.

> Written 2026-11-01 15:55 by "r13 build". Loop step reached: step 7 (tickets written, build not started).
> Direction log entry: [2026-10-31](../direction_log.md). Run mode: night.

## 1. Read in this order

1. The ticket index `/project/CoSiR/.scratch/r13-photo-finetune/issues/00-index.md` (table, shared interfaces,
   coverage line, checked) and the 13 ticket files beside it. Implementer rules:
   `/project/CoSiR/.scratch/r13-photo-finetune/notes/implementer_rules.md`.
2. The spec from §1: [2026-10-31-photo-finetune-design.md](../specs/2026-10-31-photo-finetune-design.md) (decision
   rule §6 with §3.4 and §3.5).
3. `~/.claude/references/matt-chain.md` and `~/.claude/references/codex-jobs.md` (the chain and Codex jobs).
4. The code map `/project/CoSiR/.scratch/r13-photo-finetune/notes/code_map.md` only where a ticket review needs it;
   implementers read it themselves.
5. The previous handoff, [2026-11-01-r13-build-handoff.md](2026-11-01-r13-build-handoff.md) §3 to §5 (stores,
   outputs, pitfalls, verdict-code list, timing for the run chat).

Skills: `implement-spec` (read `~/.claude/skills/implement-spec/SKILL.md`; matt-chain.md's changes override it), `tdd`
for verdict-code implementers.

## 2. Decided by the user, not to be reopened

The spec's §4 (D1 to D14), approved 2026-11-01 15:33. Nothing new was asked in build chat 1.

## 3. Where things stand

- Integration worktree `/project/CoSiR/.claude/worktrees/r13-int`, branch `r13-photo-finetune`, pushed. It holds
  `src/test/20261031_photo_finetune/pf_paths.py` (paths, constants, grid, records; asserts `AC_SMOKE` and
  `AC_OUTPUTS` unset, so r13 reads r12's real outputs and never writes them) and `conftest.py`.
- The round folder on main also holds the facts brief, the rule review and `review_nearcopy/`.
- Codex: 5h window 91% left at 15:53 (reset 20:36); weekly 54%. The implementer estimate is 40% (room needs 80%), so
  one Codex implementer now (ticket 04, planned), then Claude workers until the reset; ticket 12 (C13 script) is
  planned for Codex after 20:36, with Claude (Sonnet `high`) as the fallback.
- Wave plan (at most three implementers at once): 01, 04, 05 first; then 06, 07, 08, 09, 02, 03 by the frontier;
  then 10, 11, 12, 13. Verdict-code tickets (01, 02, 04, 07, 08, 09, 12) get an Opus `high` review before merge.
  The controller runs every `slow` test (gates (ii) to (vi), light, CPU or GPU under the lock) after the hand-backs,
  and the light GPU parity of gate (i).
- GPU check at 15:53: driver 595.99.02 answers, module and library match, GPU idle, no lock.

## 4. Open points for the next step

None for the user. Agent defaults set in the tickets, to record in the run handoff:
- Smoke mode subsets train only (`PF_SMOKE=N`, the first N sorted train keys); validation and test stay full.
- Gate (ii)'s "its r12 mAP" for frozen c* is read as r12's five-seed c* `center+k10` mAP (0.28584595252536343),
  reproduced by the sweep-score path over five seeds; the seed-0 value is reported (r12 kept no seed-0 value).
- Row 8's random start: c*'s factory built in c*'s original aspect order under the run's training seed, emotion view
  kept (code map A2e).
- Sweep rows use c*'s seed-0 start and training seed 0; test exports are made only in stage 3, after the guard.
- Comparators' clustering partitions and per-query AP are read from r12's outputs (hash- and row-checked).

## 5. Code and pitfalls

- The handoff of build chat 1 (§5) still holds: keys `<source>/<image_name>`; 1,640 repeated file names; ignored
  rows out of E, D and the caption inputs; train labels never read; only gate (iii) reads c*'s hashes.
- r12's `crossfit_all` accepts only `[None]` or five seeds (code map C3d); r12's runners hard-code `cstar`.
- The sidecar's loss string "buddy+own-caption" is stale; `own=0` (code map E7).
- Unattended: delete nothing; list items on `.scratch/pending_deletions.md`.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing of this chat's in the project (the `.scratch/` tickets are outside git by design).
- On disk over 1 GB: nothing new.
