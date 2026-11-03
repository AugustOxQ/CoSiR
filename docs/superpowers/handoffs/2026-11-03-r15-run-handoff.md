# Handoff: run r15 (the MMArt test): stages 0 and 1, commit the records, stage 2, C13 by Codex, verdicts, stage 3, then the reports

## Brief
- Done: r15 build merged into main (main = bc27203): 8 tickets, the final review in 3 parts with its fix wave and re-review; 274 round tests pass; the end-to-end smoke ends `QUEUE smoke OK` (94.5 min).
- State: on real data in the smoke, every stage 0 gate passes for all four encoders and stage 1 writes every range and pick (13 steps, none failed).
- Next: `bash mt_run_cpu.sh first`, commit the printed records, then stage 2, C13 (Codex), verdicts, stage 3, reports.
- Blocked: nothing.

> Written 2026-11-03 01:15 by tab "r15 build 3". Loop step reached: step 7 done (build merged); step 8 next. Direction
> log entry: [2026-11-02 · Backbones sweep read ... next, the MMArt (PolyArt) test (r15)](../direction_log.md).

## 1. Read in this order

1. This handoff.
2. The spec [2026-11-02-mmart-test-design.md](../specs/2026-11-02-mmart-test-design.md), from §1 on (§3.4 to §3.6 and
   §6 bind; §9 lists the deliverables).
3. The module docstring of `src/test/20261102_mmart_test/mt_run.py` (the stages, the records, the `c13.json` format).
4. r14's C13 code as a model for the C13 brief: `src/test/20261102_backbones_sweep/bs_c13.py` (derive and compare),
   and [2026-11-02-r14-run-handoff.md](2026-11-02-r14-run-handoff.md) §5 for how r14 ran it.
5. `~/.claude/references/codex-jobs.md` before the C13 job. For step 9: `report-kinds.md`, `report-writing.md`,
   `user-read-reports.md` and `~/.claude/templates/results_brief.md`.

Skills: none.

## 2. Decided by the user, not to be reopened

All in the spec's §4 (D1 to D19) and the direction log, 2026-11-02. Nothing new in the build.

## 3. Where things stand

**The tickets** (the record, `matt-chain.md`; branch `r15-mmart-test`, merged into main):

| # | Ticket | Blocked by | Spec IDs | Verdict code | Check |
|---|---|---|---|---|---|
| 01 | MMArt keys, order, labels, cut sets, subsets, feature loading; gates 0(i), 0(v) | none | §3.3, 0(i), 0(v), §3.6 sets, D5, D8, D9, D10, D14, D19 | yes | Opus review, fix |
| 02 | c* exports on MMArt (four encoders; val, new, shared held); gate 0(ii) | 01 | §3.2 X-c*, 0(ii), §3.7 | yes | Opus review, fix |
| 03 | Matched arms on CPU: mapper replay 0(iv), MMArt probabilities, readings | 01 | §3.2 matched arm, 0(iv), D11, D18 | yes | Opus review, fix |
| 04 | Clustering scorer: SC1, SC3, SC4 (AMI), validation AMI ranges; replay 0(iii) clustering | none | SC1, SC3, SC4, §3.4, 0(iii), D16 | yes | Opus review, fix |
| 05 | Retrieval scorer and stage 1 picks: SC2, SC4 (mAP), mAP ranges, picks; replays 0(iii) retrieval, 0(vi) | none | SC2, SC4, §3.4, 0(iii), 0(vi), D12, D16 | yes | Opus review, fix |
| 06 | Outcome code: cell outcomes, verdicts, D7 claims, D17 voids, licensed sentences | none | §6, §3.4 outcomes, §3.5 table, D3, D7, D17 | yes | Opus review, fix |
| 07 | Stage runner, records and the CPU queue; end-to-end smoke | 01 to 06 | §3.5 stages 0 to 3, §3.7, §6 selection rules | no | diff read, smoke |
| 08 | Diagnostics `mt_diag.py` | 04, 05 (and 01 to 03) | §3.6, D14, D19 | no | diff read, smoke |

Coverage (hard rule 6), from the ticket index: every SC and D ID has its ticket; C13 is this chat's (Codex).

**Build checks on real data:** gate (ii) 4 of 4 (max 6.1e-5, Qwen); replay (iii) clustering at 2,000 draws and
retrieval at 10,000 draws, max difference 0; replay (vi), 7 arms, 620 values, max difference 0; gate (iv) 20 mappers
within 8.3e-7; cut sets 16,384 (test) and 10,403 (far).

**The full test suite** on the merged branch (`src/test`, 80 min): 5,459 passed, 7 failed, 15 skipped. The 7 are
outside the r15 folder, in files this branch does not change. Four (`test_bs_cstar.py` width patch, three;
`test_rb_matched.py`, one) pass when their folder runs alone: an order effect between round folders in one process.
Three (`test_eg_queue_final.py`) fail alone too: the 2026-10-25 queue script `eg_queue_stage_ii.sh` refuses `EG_OUTPUTS` outside `/tmp/claude-0/`,
and pytest's `tmp_path` is under `/tmp/pytest-of-root/`. Neither touches this run; the round folder alone: 274 passed.

**The final whole-branch review** (Opus `xhigh`, three parts, each cut at its context limit; the notes are in
`.scratch/r15-mmart-test/notes/final_review.md`, outside git). Verdict: APPROVE with notes, once the fix wave is merged
and this C13 brief carries finding 8. Re-derived with the reviewers' own code, all against the smoke's real stage 1:
the cut sets (sha256 of the kept keys equal), the label counts, every retrieval seed range (1.1e-16), Qwen's
clustering ranges (0), the frozen SigLIP 2 and teacher picks (same pick, `center+k10`, tables within 1.4e-14), one
SC2 cell at 100 draws (2.6e-10). Sabotage: 24 of 26 caught at review; S13 is an equivalent change; S11 (bootstrap
pairing) now has its test.

| # | Finding | Now |
|---|---|---|
| 1 | a failed teacher pick made stage 2 refuse for all encoders | fixed (1ce7712) |
| 2 | a failed matched export did not fail gate (iv) | fixed (1ce7712) |
| 3 | a failed teacher-aspect gate would crash the scorer | stage 2 stops with exit 2 and names it (agent default; these gates pass on the real data) |
| 4 | `c13.json` not tied to the outcomes it checked | fixed: `c13.json` holds the sha256 of three files (§5) |
| 5, 6 | smoke records path; stale stage 1 status | fixed (1ce7712, 72ef2d2) |
| 7 | no round log | yours: write `src/test/20261102_mmart_test/20261102_mmart_test_log.md` |
| 8 | retrieval rows of the frozen and teacher arms are normalised twice | no code change: the C13 brief must do the same (§5) |
| 9 | the runner always runs 3 steps at once | `MT_PARALLEL=N` (bc27203) |
| 10 | D11's mapper retrain fallback is commands in the `mt_matched.py` docstring, not code | only if gate (iv) fails (it passes on all four); then run them and flag the mapper |
| 11 | the unseen-artist subset has 15 paintings, so no cells | the report says why (agent default) |

**The smoke** (`MT_SMOKE=1`, `/data/SSD2/cosir_r15/20261102_mmart_test/smoke_build3/`, log `smoke_queue.log`, code
before the fix wave): `QUEUE smoke OK rc=0
  seconds=5671`. Stage 0: 22 steps, none failed, about 8 min (replays at reduced scope). Stage 1: 13 steps, none failed,
  27 min at full size; `records/status.json` all `ok`. Stage 2: 8 steps, none failed, about 20 min at 40 and 100
  draws, 60 cells. Verdicts copied (smoke: no C13). Stage 3: `mt_diag.py all`, 10 items, 2,392 s, none failed or not
  run. After the merge, the new runner rebuilt `status.json` from the smoke's step records byte for byte and copied
  the verdicts again.

## 4. Open points for the next step

None for the user. Agent defaults carried on: finding 3's stop; finding 8 in the C13 brief instead of a code change;
the unseen-artist subset without cells.

## 5. Code and pitfalls

- **The run, in order** (from `src/test/20261102_mmart_test/`; no `MT_SMOKE`, no `MT_OUTPUTS`, so `OUT` is
  `/data/SSD2/cosir_r15/20261102_mmart_test/outputs/`; CPU only):
  1. Check `uptime` and `free -g` (`shared-resources.md`). Under a load above about 24, set `MT_PARALLEL=2`.
  2. `bash mt_run_cpu.sh first` (stage 0, then stage 1). Read `QUEUE first OK`, both `MT_RUN_SUMMARY ... failed=0`,
     and `records/status.json`. It prints the `git add -f ... && git commit ...` command for `records/stage1_*.json`,
     `stage1_teacher.json`, `status.json`, `cutsets.json`: run it and push. Stage 2 refuses until they are committed
     unchanged.
  3. `bash mt_run_cpu.sh stage2` writes `OUT/stage2/cells_clu.json`, `cells_ret.json`, `outcomes.json`.
  4. C13 by Codex (below). It writes `records/c13.json`; commit it (`git add -f`) and push.
  5. `bash mt_run_cpu.sh verdicts` copies `outcomes.json` to `records/verdicts.json` only when `c13.json` is
     committed with `"pass": true` and the three sha256 values equal the current files. Run its printed commit command.
  6. `bash mt_run_cpu.sh stage3` (`mt_diag.py all`), then the auto report
     `docs/reports/auto/percept/2027-01-03_mmart_test.md` with its results brief and index row, and the user-read
     report (spec §9).
- **C13 has no code yet.** Write it with Codex (a `rederivation` job, `gpt-6.1-sol` high; check
  `codex-usage --room rederivation` first; at 23:00 the five-hour window had 80% left and resets 02:43; the weekly
  limit had 19% left and resets 11-04 08:27). Brief it from the spec's rule and r14's `bs_c13.py`, never from our
  numbers; its own code, no `mt_*` imports. It re-derives, each within 1e-6 (spec §3.4): the test cut from the index and
  near-copy parquet files (against `records/cutsets.json`); every seed range from the validation exports and views;
  every pick and its score table (cross-fit halves by position parity in sha256 order, the guard, the tie rule; the
  teacher with Qwen's mAP ranges); all SC1 to SC4 cells (values, d, lo, hi at 2,000 and 10,000 draws, in the inherited
  draw order) and their outcomes, from the test exports, the matched probabilities, the test views and the labels;
  and the verdicts and SC4 claims from those cells and `records/status.json`. Two protocol facts the brief must state:
  - Finding 8: the frozen and teacher retrieval rows are `unit(unit(raw))` in float32 (`mt_data.features` normalises,
    then `mt_ret.view_rows` normalises again). Normalised once, the teacher's `whiten1024+k5` score differs by up to
    3.7e-4, above the 1e-6 bar; the picks do not change.
  - The c* rows are the float16 exports cast to float32 without a second normalisation (as r14).
  `records/c13.json` format: `{"pass": true, "sha256": {"stage2/outcomes.json": ..., "stage2/cells_clu.json": ...,
  "stage2/cells_ret.json": ...}}`, paths relative to `OUT` (the `mt_run.py` docstring). r14's full derive took 5,334 s.
- **Timing** (estimates; the smoke measured stage 0 at reduced replay scope 8 min, stage 1 at full size 27 min under a
  load of about 40): stage 0 about 30 min (full replays: (iii) clustering 15 min, retrieval 7 min, (vi) 21 min, in
  parallel with the exports), stage 1 about 30 min, stage 2 at full draws about 1 to 1.5 h, C13 about 1.5 h, stage 3
  about 1 to 1.5 h (the smoke took 40 min). Started at about 01:30 it ends about 07:00 to 08:00 (reports after that), before 11:00. CPU only.
- **Git:** `*.json` is gitignored (`git add -f` for records); commit only your own paths (`git commit -- <paths>`); other
  chats (the paper chat) commit on this repo too. `git fetch .` fails here.
- **Unattended:** delete nothing; `.scratch/pending_deletions.md` has the build's rows (worktrees, ticket branches,
  smoke folders).

## 6. State at handoff

- Running: nothing. GPU check at 01:12: driver, kernel module and library all 595.99.02 (the run is CPU only).
- Uncommitted: nothing of this chat's.
- On disk over 1 GB: `/data/SSD2/cosir_r15/20261102_mmart_test/smoke_build3/` (12 GB, data disk), the 11 build worktrees (1.3 GB),
  `.scratch/r15-mmart-test/smoke/t02/` (3.3 GB) and build 2's scratchpad (6.4 GB); all on `.scratch/pending_deletions.md`.
