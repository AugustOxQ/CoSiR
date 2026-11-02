# Handoff: finish the r15 build (end-to-end smoke, final whole-branch review, fix wave, merge into main), then open the run chat

## Brief
- Done: all 8 tickets built, checked and merged into `r15-mmart-test` (eb7f205, pushed); 251 tests pass.
- State: every stage 0 gate checked on real data passes: (ii) 4 of 4, (iii) clustering and retrieval, (iv), (vi), max difference 0 (6.1e-5 for Qwen's export).
- Next: the end-to-end smoke (`mt_run_cpu.sh smoke`) and the final Opus `xhigh` review, in parallel; fix wave; merge; open the run chat.
- Blocked: nothing.

> Written 2026-11-02 22:16 by tab "r15 build 2" (cut at the context meter, 200k). Loop step reached: step 7 (build),
> 8 of 8 tickets merged, smoke and final review not done. Direction log entry: [2026-11-02 · Backbones sweep read ... next, the MMArt (PolyArt) test (r15)](../direction_log.md).

## 1. Read in this order

1. This handoff, then the earlier build handoffs for stores, timings and git pitfalls:
   [2026-11-02-r15-build-handoff.md](2026-11-02-r15-build-handoff.md) sections 3 and 5,
   [2026-11-02-r15-build2-handoff.md](2026-11-02-r15-build2-handoff.md) section 5. The spec:
   [2026-11-02-mmart-test-design.md](../specs/2026-11-02-mmart-test-design.md), from §1 on.
2. The ticket folder `/project/CoSiR/.scratch/r15-mmart-test/` (outside git): `issues/00-index.md` (shared
   interfaces, coverage line, "Controller decisions"), the 8 tickets (all `Status: done`), and
   `notes/controller_notes.md` (the timeline, every review finding and fix).
3. `~/.claude/references/matt-chain.md` ("The final whole-branch review", "The record"),
   `~/.claude/references/codex-jobs.md` before any Codex job.

Skills: none.

## 2. Decided by the user, not to be reopened

All in the spec's §4 (D1 to D19) and the direction log, 2026-11-02. Nothing new in builds 1 and 2.

## 3. Where things stand

Integration branch `r15-mmart-test` (eb7f205, pushed; from main 88cb70f). Round folder `src/test/20261102_mmart_test/`:
`mt_paths`, `mt_data` (01), `mt_export` (02), `mt_matched` (03), `mt_clu` (04), `mt_ret` (05), `mt_outcome` (06),
`mt_run.py` and `mt_run_cpu.sh` (07), `mt_diag.py` (08).

| # | Ticket | Check | Result |
|---|---|---|---|
| 01 | data, cut, gates (i), (v) | Opus review, fix e7e9e4e | cut 16,384 / 10,403; gates pass |
| 02 | c* exports, gate (ii) | Opus review, fix c73d9f5 | gate (ii) passes, 4 encoders, 21 items each (15 held by r14's path, 3 own path on shared held, 3 own path on validation); max 6.1e-5 (Qwen; stored on cuda), others 0 to 6.1e-5 |
| 03 | matched arms, gate (iv) | Opus review, fix 5c6c845 | 20 mappers on CPU within 8.3e-7 |
| 04 | clustering scorer | Opus review, fix 845a467 | replay (iii) clustering, 2,000 draws: max difference 0 |
| 05 | retrieval scorer, picks | Opus review, fix 870ad7d | replay (iii) retrieval, whole: 108 values, 10,000 draws, max difference 0, `passed true`, 391 s; replay (vi), whole: 7 arms, 620 values, max difference 0, 1,242 s |
| 06 | outcome code | Opus review, fix 7b179e2 | 120+ tests |
| 07 | runner, CPU queue | controller diff read | stub wiring run; real `gate_i`, `gate_v`, `cutsets` steps ran |
| 08 | diagnostics | controller diff read | `near_copy` and `unseen_artist` on real data (counts match the prep file: 6,375 / 2,249 / 390 / 49; unseen artists 15 paintings, 12 artists); other 8 items on stubs only |

Build-check outputs (not records): the replay JSONs in
`/tmp/claude-0/-project-CoSiR/877af933-b98a-4928-b297-dab9007ed6b9/scratchpad/build_checks/t05_replay/stage0/`; ticket
04's in the build 1 session scratchpad (path in the build 2 handoff).

## 4. Open points for the next step

None for the user. Agent defaults carried on: the unseen-artist subset has no cells (15 paintings, fewer than the
20 clusters; the report says why). The C13 guard: `mt_run.py verdicts` copies `OUT/stage2/outcomes.json` to
`records/verdicts.json` only when `records/c13.json` is committed with `"pass": true`.

## 5. Code and pitfalls

- **Next steps, in order:**
  1. **The end-to-end smoke** (ticket 07's), in the background, never with `--enc` (the `smoke` part passes extra
     arguments to `verdicts` and `stage3`, which reject them):
     `cd src/test/20261102_mmart_test && MT_SMOKE=1 MT_OUTPUTS=/data/SSD2/cosir_r15/20261102_mmart_test/smoke_build3 bash mt_run_cpu.sh smoke`
     (on the data SSD: the exports are about 9 GB). Not timed; estimate 1 to 1.5 h: stage 0 about 10 min (replays at
     reduced scope, exports about 8 min), stage 1 at full size (no draws; r14 took about 11 min per encoder), stage 2
     at 40 and 100 draws, stage 3 `mt_diag.py all --workers 3` is the heaviest. Pass: `QUEUE smoke OK`, every
     `MT_RUN_SUMMARY ... failed=0`, `records/status.json` all `ok`, `diag/summary.json` without `failed` or `not run`.
     Stage 1 under smoke reads the real validation data, so its ranges and picks are the real values; the run
     recomputes them for the record.
  2. **The final whole-branch review**, Opus `xhigh` (`matt-chain.md`), may start beside the smoke (it reads code
     first; point it at the smoke folder for real stage 1 records and cells). Its brief names the spec, the tickets,
     the index, hard rule 6 (`~/.claude/rules/spec-templates.md`) and `docs/superpowers/constitution.md`. Points
     for it: ticket 08's shared-held loop copies the scorers' bodies with a second label set (`mt_diag.py:332-378`;
     check it equals `mt_clu.score_encoder` and `mt_ret.score_encoder`); ticket 07's `void_plan`
     (`mt_run.py:391`) against the §3.5 table and ticket 06's voids; the stage 2 SC4 cells of a partly voided
     encoder; `mt_ret.view_rows` normalises rows a second time (float rounding only, accepted); gate (ii)'s
     validation join relies on the trainer's dev order (`ct2_data.split2('full')`, 6,152 rows, asserted). Then one
     fix wave and a Sonnet `high` re-review of the fixes.
  3. **Merge into main**: `git merge-base --is-ancestor main HEAD && git update-ref refs/heads/main HEAD` (or merge
     main in first if it moved), push main; then list the 8 worktrees under `.claude/worktrees/` and their branches
     (`r15-t01` to `r15-t08`, and the `worktree-agent-*` branches the Agent tool made) on
     `.scratch/pending_deletions.md`.
  4. **The run handoff** with the ticket list (`matt-chain.md`, "The record": title, blocked by, spec IDs, verdict
     code; from `00-index.md`), the timing, and the run commands: `bash mt_run_cpu.sh first` (stage 0 and 1; then
     commit the printed command), `stage2`, C13, `verdicts`, `stage3`. Open the run chat with
     `loop-next ... --label "r15 run" --unattended`.
- **C13 has no code yet.** The run chat writes it with Codex (r14 had `bs_c13.py`, derive and compare,
  `records/c13.json` with `pass` and a sha256 map; C13 full derive took 5,334 s in r14). Say so in the run handoff.
- **Timing for the run handoff:** estimates, not measured: stage 0 about 25 min, stage 1 about 25 min, stage 2 at
  full draws about 1 to 1.5 h, C13 about 1.5 h, stage 3 about 1.5 h (2 to 3 h under load). From about 01:00 it ends
  about 06:30 to 08:00, before 11:00. CPU only.
- **Codex** had no room until 2026-11-03 02:02; check `codex-usage --room` for the C13 job.
- **Git:** `*.json` is gitignored (`git add -f` for records); commit only your own paths; other chats commit here.
- **Unattended:** delete nothing; `.scratch/pending_deletions.md` has the two scratch rows of build 2 (3.3 GB in
  `.scratch/r15-mmart-test/smoke/t02/`, 6.4 GB in this session's scratchpad).

## 6. State at handoff

- Running: nothing (all subagents and background jobs ended).
- Uncommitted: nothing of this chat's.
- On disk over 1 GB: `.scratch/r15-mmart-test/smoke/t02/` (3.3 GB) and this session's scratchpad (6.4 GB), both on
  the system disk and on `.scratch/pending_deletions.md`.
