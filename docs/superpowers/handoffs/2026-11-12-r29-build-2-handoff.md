# Handoff: r29 build 2: review and merge tickets 01, 03, 04, 06, then build tickets 02, 05, 07 to 15, final review, merge, open the run chat

## Brief
- Done: 15 tickets written; tickets 01, 03, 04 and 06 built (104 tests pass across them); ticket 01 reviewed: pass after fixes.
- State: nothing merged into the integration branch yet. Gate (iii) passes on the real stores; gate (v) emotion cells replay exactly.
- Next: fix ticket 01, review 03, 04, 06, merge; start ticket 02 on Codex; build the rest.
- Blocked: nothing. Risk: gate (i)(c)'s validation part may fail from the machine alone (§5).

> Written 2026-11-12 23:10 by the "r29 build" tab (cut at the context meter's 200k note). Loop step reached: step 7
> (build) in progress. Direction log entry: [2026-11-12 · r29 decide](../direction_log.md).

## 1. Read in this order

1. The first build handoff, sections 2 to 5 (decisions, where things stand, code and pitfalls, the end of the build):
   [2026-11-12-r29-build-handoff.md](2026-11-12-r29-build-handoff.md). It still holds; this file adds the build state.
2. The tickets and their index (shared interfaces, coverage line, build order):
   `/project/CoSiR/.scratch/r29-siglip-only/issues/00-index.md` and the 15 ticket files beside it.
3. The notes (`/project/CoSiR/.scratch/r29-siglip-only/notes/`): `t01_review.md` (the review's must-fix and
   should-fix list), `t03_handback.md`, `t04_handback.md`, `t06_handback.md` (CLIs, real-data checks, layout
   assumptions, commands left for the controller), `scoring_map.md` (Codex's map of the scoring side),
   `brief_common.md` (shared part of every implementer brief: point new briefs at it), `t02_codex_task.txt` (ticket 02's
   Codex brief, ready to launch).
4. The spec §1 onward: [2026-11-12-siglip-only-adapter-design.md](../specs/2026-11-12-siglip-only-adapter-design.md),
   and `~/.claude/references/matt-chain.md`, `~/.claude/references/codex-jobs.md`.

Skills: none by name; the chain is in `matt-chain.md`.

## 2. Decided by the user, not to be reopened

As in the first build handoff, section 2 (spec §4, D1 to D9 and D25). Run mode is night (`run-mode`).

## 3. Where things stand

Integration branch `r29-siglip-only`, worktree `/project/CoSiR-r29`, pushed at the handoff commit: `r29_paths.py`
(controller-owned constants), `decisions.md` (18 lines from tickets 01, 03, 04, 06), `records/gates/`.

| Ticket | Branch, worktree, HEAD | State |
|---|---|---|
| 01 caption join, caption-width model | r29/t01, /project/CoSiR-r29-t01, 403448ba | built; review: pass after fixes (`t01_review.md`): must-fix 1, should-fix 2, 3, minors 4, 7, 9 |
| 03 trainer and exports | r29/t03, /project/CoSiR-r29-t03, 613bc992 (on t01) | built, 41 tests pass; GPU smoke ran; not reviewed |
| 04 text scorer | r29/t04, /project/CoSiR-r29-t04, 2a7693b6 (on t01) | built, 19 tests pass; gate (ii) real run left; not reviewed |
| 06 image test cells | r29/t06, /project/CoSiR-r29-t06, 9c36c7ed | built, 24 tests pass; gate (v) image: emotion replays with diff 0.0; full replay left; not reviewed |
| 02 prefixed store | r29/t02, /project/CoSiR-r29-t02, 166dc6c3 (empty) | not started: held for Codex (Sol high); brief `t02_codex_task.txt` |
| 05, 07 to 15 | none | not started |

Next steps, in order:
1. A fixer on r29/t01 for the review's must-fix 1 and should-fix 2, 3 and minors 4, 7, 9 (Sonnet `high`, or Codex).
   Then adapt 03's and 04's `load_view` call sites (named in their notes) when merging.
2. Verdict-code reviews of 03, 04, 06 (Opus `high` workers, one per ticket, as 01's). 03's review must check the
   caption substitution (only the caption tower's input changes; graph space D stays Qwen3) and gate (i)(b)'s test.
   06's review must check the read-out choices 5 and 7 (`t06_handback.md`) against spec §3.4 and D18.
3. Merge 01, 03, 04, 06 into the integration branch (expected conflict: `records/gates/iv.json` between 02 and 04; merge
   both parts). Run the full suite under `res-queue cpu`.
4. Launch ticket 02 on Codex once `codex-usage --room implementer` says room (the five-hour window reset at 23:43);
   the command form is in `codex-jobs.md`; worktree `/project/CoSiR-r29-t02` exists. Add the ledger row after.
5. Tickets 05, 07, 08, 09 (stage 0 and 1 path), then 10 to 14 (stage 2), then 15. At most three implementers at once.
6. Gate commands left for the controller (real data; run after the merge, from the integration worktree so the records
   land there): gate (v) image full replay (`t06_handback.md`, CPU 40 to 60 min); gate (ii) K0 validation scoring
   (`t04_handback.md`, GPU about 4 min). These are stage 0 checks: run them in the build if the queue allows, or list them
   for the run chat.
7. Final whole-branch review (Opus `xhigh`), one fix wave, scoped re-review (Sonnet `high`); merge into main, push;
   run handoff; GPU check; `loop-next` (first build handoff, section 5).

Codex: one ledger row added (the scoring map, 22:50). At 23:08 the five-hour window had 77% left, too little for an
implementer job (80% needed) until the 23:43 reset.

## 4. Open points for the next step

None for the user (night mode).

## 5. Code and pitfalls

- **Gate (i)(c)'s validation part (flag).** K0 has no `val/` export. The only stored validation file is r14's
  training-time `train/k11=40/R6-5/<aspect>__Gres__s0.npz`, written on node402's GPU. On CPU, the new export differs
  from it (max 1.2e-4); the held part is bit-equal (stored on CPU), and the new export code equals r14's own export code
  bit for bit. A local GPU may still differ. Gate (i) failing parks all training (§3.5), so settle before the run how
  the gate code records a failure caused only by the machine (`vs_r14_export_code` equal, `val/*` unequal). Constitution
  row 4: never loosen a tolerance; a cause in the rule parks the piece and is flagged. Check first whether the local GPU
  reproduces the file (a `gate-i-c --no-record` run through `res-queue gpu --short`).
- Ticket 03 writes `records/pass_rule.json`'s guard key as `"status": "pass"`; ticket 05 must write that key.
- Ticket 03's encoder names are bs_paths' keys (`clip_l14`), while `r29_paths.CAPTION_SOURCES` uses `clipl14`.
  Ticket 10 must map between them.
- Tickets 04 and 06 assume r14's file layout for new arms (`train/<arm>/R6-5/...`, `held/<arm>/...`, `val/<arm>/...`);
  check against 03's actual output in the reviews.
- `*.json` is gitignored: records go in with `git add -f` (as r28's records).
- Every subagent brief carries the unattended line: delete nothing; list what you would remove.

## 6. State at handoff

- Running: nothing of this chat (no subagents, no background jobs). r28's run chat uses the CPU queue.
- Uncommitted: nothing of r29 after this handoff's commit. Ticket branches r29/t01, t03, t04, t06 are committed locally,
  not pushed.
- On disk over 1 GB: nothing. Smoke and scratch outputs under `/data/SSD2/cosir_r29/20261112_siglip_only/`
  (`smoke/`, `scratch_t03_ic/`) are small; list them on `.scratch/pending_deletions.md` at the end of the build.
