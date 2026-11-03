# Handoff: r17 run: run stages 0 to 3 of the new scores, write C13, then the reports

## Brief
- Done: the fix wave closed all 9 review items; the re-review passed; 524 tests pass; the code is on main (8cf743b2).
- State: nothing has run on the real output folder yet; it is empty. CPU idle at 00:20, GPU check passes.
- Next: open the round log, run stage 0, then stages 1 and 2 in three chains, with C13 code written beside them.
- Blocked: nothing. Stages 0 to 2 end about 06:30; stage 3 likely ends about 13:30, past 11:00.

> Written 2026-11-04 00:21 by tab "r17 build 4". Loop steps reached: step 7 (build) done, steps 8 and 9 next. Direction
> log entry: [2026-11-03 · MMArt test read](../direction_log.md), bullet "r17 spec approved".

## 1. Read in this order

1. The spec: [2026-11-03-new-scores-design.md](../specs/2026-11-03-new-scores-design.md): §3.4 (C13), §3.5 (stages,
   gates, void table), §3.6 (diagnostics), §6 (the rule), §9 (deliverables, time-box: report by 2026-11-04 18:00).
2. The plan: `src/test/20261103_new_scores/ns_run_cpu.sh`, and `ns_run.py list`.
3. `~/.claude/references/codex-jobs.md` before the C13 job.
4. `report-kinds.md`, `report-writing.md`, `user-read-reports.md` and `~/.claude/templates/results_brief.md` before the
   reports.
5. The build 3 handoff, [2026-11-03-r17-build3-handoff.md](2026-11-03-r17-build3-handoff.md), §5 (pitfalls), only when
   a step goes wrong.

Skills: none.

## 2. Decided by the user, not to be reopened

All in the spec's §4 (D1 to D20), approved 2026-11-03 20:45 from its brief; no container restart (user).

## 3. Where things stand

**Code:** `src/test/20261103_new_scores/` on main, 8cf743b2, pushed; the main checkout holds it. The run imports these
files, so **stay in `/project/CoSiR`** and do not switch its branch. Records go to `records/` in that folder, outputs to
`/data/SSD2/cosir_r17/20261103_new_scores/outputs` (empty now, about 2 GB expected).

**Tickets, all 11 done** (table with blockers and spec IDs: `/project/CoSiR/.scratch/r17-new-scores/issues/00-index.md`):

| # | Ticket | Module |
|---|---|---|
| 01 | Metrics and partitions; gate (iii) metrics | `ns_metrics` |
| 02 | The 412 claim cells, bars, outcomes, voids | `ns_outcome` |
| 03 | Painting data, untold join; gate (v), class counts | `ns_data` |
| 04 | Photo data, class counts | `ns_photo` |
| 05 | PercepT mappers on CPU; gate (iv) | `ns_mappers` |
| 06 | Clustering scorer; gate (i) a to d | `ns_clu`, `ns_clu_replay` |
| 07 | Clustering cells SC1 to SC5, class-count ranges | `ns_clu_cells` |
| 08 | Retrieval scorer and cells SC6 to SC8; gate (ii) | `ns_ret` |
| 09 | Stage 1 picks; gate (vi) | `ns_picks` |
| 10 | Runner, records, status, queue script, scorer facade | `ns_run`, `ns_run_cpu.sh`, `ns_score` |
| 11 | Stage 3 diagnostics | `ns_diag` |

**Checks so far.** Tickets 01 to 09 each had an Opus `high` review. The final whole-branch review (Opus `xhigh`) found no
blocker and re-derived the 412 cells, the bars, the void logic and the real row route. The build's full replays matched
the stored values with max diff 0.0 (table in build 3's §3). The re-review of the fix diff (Sonnet `high`) passed, and its
sabotage of each fix was caught.

**What the fix wave changed, as the run sees it:**

- A missing new pick skips only its SC; `cells --task ret` keeps the rest and prints the absent cells.
- `ns_diag r0` now computes held or test R0 through `arm_ap`, with the validation value beside it. It takes about 45 min
  (real smoke: about 4 min for 2 keys). Before, it read the record in a minute.
- Outside smoke, a step file with `smoke: true` counts as `failed`. Point no smoke at the real output folder.
- Every `ns_diag` command refuses without `records/verdicts.json`, `figure` and `tables` too, unless in smoke.
- `ns_run.py gate` exits 1 on any failed entry.
- `write_json` is atomic. Two chains can still overwrite each other's `status.json` with an older view, since the last
  writer wins: **run `ns_run.py status` after each stage, before any commit**. It rebuilds from the step files.
- `cells_ret` est 25 min per (set, encoder); `told_untold` stays 15 min.

**Round tests:** 524 pass on the merge with main, through `res-queue cpu`.

**Machine at 00:20:** run mode night; GPU check passes (driver 595.99.02, kernel module and library agree), though r17
needs no GPU. Queue idle, 4 CPU slots free, load 5, 52 GB RAM available, 1.8 TB free on SSD2. r19 build 4 is open and may
take slots. Codex: 5h 80% left, reset 04:35; an implementer job needed 80%, so check again at the C13 job.

## 4. Open points for the next step

None for the user. Agent steps, in order:

1. **Round log.** Open `src/test/20261103_new_scores/20261103_new_scores_log.md` at stage 0 (spec §9, no other owner).
   Keep the end-time estimate and every choice in it.
2. **Stage 0 in three chains.** One `ns_run_cpu.sh` line waits for its slot, so a stage alone runs in one slot. Use, from
   a background Bash:
   `bash ns_run_cpu.sh stage0 --dry | grep -v -E 'r17:iv ' | xargs -d '\n' -P 3 -I{} bash -c '{}'`,
   then the `gate iv` line alone, after `iv_export`. Check the form on one short step first. xargs goes on when a child
   exits non-zero and returns 123 at the end, so read the status, not the exit code.
3. **After every stage:** `ns_run.py status`; read the counts and any `failed` or `missing`. Stage 0's record must say
   what build 3's §3 says about `a_r15` (r17's own wiring slip, fixed in 86c3cc0; the real cells were never affected).
4. **Stage 1.** `stage1-ranges` and the picks in three chains, leaving out `stage1_records`; then `stage1-records`,
   `ns_run.py status`, and commit `records/stage1_ranges.json`, `stage1_picks.json`, `status.json` with `git add -f`
   (`*.json` is gitignored), and push. No stage 2 step before that commit.
5. **C13 code,** once stage 1 is committed, beside stage 2 so that it is ready when stage 2 ends. Separate code re-derives
   from the rule, the exports, labels and stage 1 records: class counts, seed ranges, new picks, and every SC cell's
   value, d, interval and outcome, within 1e-6 (§3.4). Codex `gpt-6.1-sol` high if `codex-usage --room` says room, else
   a Claude worker. It gets the rule and the data, never our cells or outcomes. Its CPU steps go through `res-queue`. It
   must write `records/c13.json` with `pass: true` and `outcomes_sha256` of `records/outcomes.json`; `ns_run.py
   c13-check` reads it.
6. **Stage 2.** `stage2-clu` and `stage2-ret` in three chains. Then `ns_run.py status`, commit `records/status.json`,
   and run `results`. `outcomes` refuses unless `status.json` equals git HEAD. It lets `missing` rows through and only
   prints a count, and nothing stops `verdicts` on them: look at that count yourself and at each row's cause.
7. **C13, then verdicts.** Run C13, `c13-check`, `verdicts`. Commit `outcomes.json`, `c13.json`, `verdicts.json`
   (`git add -f`) and push.
8. **Stage 3** in three chains; the figures need their sweeps, and `tables` needs the sweeps and `r0`. Photo sets run
   last, as the plan orders them. Then record what did not run.
9. **Reports** (step 9): the auto report in `docs/reports/auto/percept/` (next sequence date) with its results brief and
   index row, the user-read report, and the section on how much this changes the paper draft
   (`docs/paper/cvpr2027_aspect_spaces/drafts/v1/`). Auto report draft by Codex if room, then Claude's claim review.
   Then open the next loop's decide chat (`loop-next <handoff> --label "<next loop> decide" --new-loop --notify`).

**Report items that no code writes:**

- D4's "the ArtELingo retrieval values were seen before".
- D5's "lower point value", with no verdict word.
- The "not run" list in the text.
- The EmoSet sweep is Qwen only (SC4's scope); r14 has rows for the other three encoders.
- CRL clustering has no stored partitions, so its route is proved only by the run and C13.
- SC7 and SC8 sentences open with "On <set>," (not in §6's template; needed to tell cells apart).
- `meets_bar` against the strict `bar_met`; the guard decided CRL's MMArt pick; r15's diagnostic plain rows differ in form
  from SC1's; ticket 06's review notes 1, 2 and 4 (controller notes).

**Time.** Slot-hours at load 30 to 60: stage 0 2.1, ranges 1.2, picks 4, clustering cells 5, retrieval cells 3.3 to 5,
stage 3 about 12, C13 not known (r15: 5,178 s for 60 cells; at that rate 412 cells take about 10). From a 00:30 start
in three chains: stages 0 to 2 end about 06:30, C13 about 09:30, stage 3 about 13:30. So the run likely ends past 11:00.
Estimate again at the start with `res-queue status`, weigh it as `shared-resources.md` says, choose, and record why in
the log and the handoff. CPU use past 11:00 is a strong preference, not a hard limit. §3.6's rule: if 12:00 passes with
stage 3 unfinished, the photo sets run last and may be cut, listed as not run. A run already going is not stopped for this.

## 5. Code and pitfalls

Build 3's §5 holds: row forms, `paired_cell`'s own pool (one plain process per queue step), `ns_diag`'s private imports,
the gate status at smoke draws, reading stage 1 only through `read_committed_record`. Three more:

- A smoke writes new test numbers. Nobody reads them before stage 1 is committed; check structure and exit codes only.
- `subagent-watch --watch` exits at once when no subagent is live; start it after the dispatch. A worker idle on its own
  queue job counts as not running.
- `test_spec_fragments_in_spec_text` fails on copies without `docs/`; expected there only.

## 6. State at handoff

- Running: nothing of this chat (no subagent, no queue job, no Codex job).
- Uncommitted: nothing of this chat's. Notes, tickets and smoke results in `.scratch` are outside git by design.
- On disk over 1 GB: nothing new. The build's worktrees, branches and scratch are on `.scratch/pending_deletions.md`;
  `r17-int` stays until the run ends.
