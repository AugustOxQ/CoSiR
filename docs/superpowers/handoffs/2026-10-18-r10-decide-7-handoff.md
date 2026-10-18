# Handoff: r10 decide 7: write the stage 1 report (auto and user-read), take the user's reading (C), then decision A and the spec approval

## Brief
- Done: stages 0 and 1 of the aspect graph sweep. Stage 0 matched the pilot exactly; 63 graph rows and 39 head runs are scored.
- State: the results sit in two tables; no report yet, and nobody has read them.
- Next: write the short auto report and the half-page user-read report; the user gives the reading first (C), then A and the spec approval.
- Blocked: nothing. The user is back at about 18:45 (day mode).

> Written 2026-10-18 18:25 by `r10 sweep 1` (cut past 200k context). Loop step reached: step 8 done for stages 0 and
> 1, step 9 (report) not started; decision A still open; spec not approved. Direction log entry:
> [2026-10-18](../direction_log.md).

## 1. Read in this order

1. The run log, [20261018_aspect_graph_sheet_log.md](../../../src/test/20261018_aspect_graph_sheet/20261018_aspect_graph_sheet_log.md):
   timeline, the agent defaults set in the build, and what ran.
2. The two sheets: [graphs/sheet.md](../../../src/test/20261018_aspect_graph_sheet/graphs/sheet.md) (63 graph rows,
   lifts, diagnostics) and [ags_head_tables.md](../../../src/test/20261018_aspect_graph_sheet/ags_head_tables.md)
   (heads and the per-view references). The row definitions: [ags_rows.md](../../../src/test/20261018_aspect_graph_sheet/ags_rows.md).
3. The spec, [2026-10-18-aspect-graph-sweep-design](../specs/2026-10-18-aspect-graph-sweep-design.md): brief, §3.2,
   §3.4, §3.5, §8.
4. For the reports: `~/.claude/rules/report-kinds.md`, `user-read-reports.md`, `report-writing.md`,
   `~/.claude/templates/auto_report_template.md`, `results_brief.md`, `~/.claude/references/user-read-report.md`,
   `ste-writing.md`. The pilot's reports are the pattern:
   [auto](../../reports/auto/percept/2026-12-10_purified_buddy_pilot.md) and
   [user-read](../../user_read/2026-10-18_purified_buddy_pilot.md).
5. For the rule task: `~/.claude/references/maintaining-rules.md`, `~/.claude/rules/habit-scope.md`, and the project
   memory `feedback_exploration-number-sheet.md`.

Skills: humanizer:humanizer (the user-read report's last pass).

## 2. Decided by the user, not to be reopened

- Everything in the [r10 sweep 1 handoff](2026-10-18-r10-sweep-1-handoff.md), section 2.
- 2026-10-18 18:05 (user): the storage summary as recommended. Keep the aspect-instruction store (disk ledger
  updated). Delete the rest: the rows of `.scratch/pending_deletions.md` marked "approved by the user 2026-10-18
  18:05" (pilot outputs, smoke folders, Codex notes, this build's worktrees and branches). The unattended guard of
  `r10 sweep 1` refused the deletes.
- 2026-10-18 18:05 (user): yes, the number-sheet guidance becomes a global rule.
- 2026-10-18 18:05 (user): the spec approval comes after decision A. The user said "I got no clue now" about what A
  needs, so we said A comes after the stage 1 results (C first, as in the loop).

## 3. Where things stand

- Stage 0: `STAGE0 PASS 2.4204897561 / 2.3849313416 / 2.5528809893`; the pilot's D, I, C, E, P and EC graphs are
  pair-identical on fit and dev.
- Graph sheet, the aspect's own-label lift: the plain aspect graph (R0) is the best of the 21 rows for emotion (2.420)
  and style (7.065). For genre, the image-weighted fusion of both sides (Fside025, 5.352) beats R0 (4.412).
- Head sheet, target AMI (mean of the image and caption side, own label, 5,000 evaluation paintings):

  | Aspect | Best row (3 seeds) | Its target AMI | R0 (seeds) | R0's target AMI |
  |---|---|---:|---|---:|
  | emotion | Fres1 | 0.137 ± 0.002 | seed 0 | 0.120 |
  | style | Fres05 | 0.260 ± 0.003 | seed 0 | 0.218 |
  | genre | Fside025 | 0.420 ± 0.002 | 3 seeds | 0.402 ± 0.005 |

- The per-view clustering references (no training) on the side each aspect uses: emotion captions 0.217 against the
  best head's 0.194; style images 0.395 against 0.348; genre images 0.496 against 0.454. Image-side emotion: best
  head 0.091 (Gres), emotion-instruction images 0.093, default images 0.070, the pilot's purified student 0.094.
- Every head stops early: the lowest dev loss is at epoch 1 to 4, as in the pilot.
- The lift ratio favours rows that avoid same-style and same-genre pairs (the residual rows push those lifts below 1);
  read it beside the degree-0 counts.

## 4. Open points for the next step

1. **Key: decision C** on stages 0 and 1, from the results brief; the user's reading first, then ours.
2. **Key: decision A**, answer-first. The user had no clue what A needs: explain it in plain words, with the
   options and their cost, and no recommendation until they answer. The spec's agent default is the exploration
   sweep.
3. **Key: spec approval**, after A. Paste the brief. Amend the spec with this build's agent defaults (run log,
   "Agent defaults": rows R0o and R1c, genre's R0 from captions, sd scaling, the 20% eligibility floor, the head's
   loss and target AMI) and flag them in "Check these".
4. **Do: the approved deletes** (above), then the storage summary at the wrap-up with this run's outputs: `outputs/`
   1.1 GB (39 head output pairs; the scores are in the tables) and `graphs/` 67 MB (rebuildable in about 5 minutes).
   Both are now rows of `.scratch/pending_deletions.md`; outputs/ has a deferred disk-ledger row.
5. **Do: the global rule** for the number-sheet guidance (user's yes), after `maintaining-rules.md`.
6. **Minor:** the reports' checks per `user-read-report.md` ("Checks"); Codex has room (76% of the five-hour window at 18:00).

## 5. Code and pitfalls

- Report name: CoSiR keeps sequence-number dates; the last auto report is `2026-12-10_…`, so this one is
  `docs/reports/auto/percept/2026-12-11_aspect_graph_sheet.md`; add its row to `reports_sum.md` and run
  `python scripts/check_reports_sum.py`.
- `ags_head.py --score` caches per output; a rescore takes about a minute. JSON results (`ags_head_results.json`,
  `graphs/sheet.json`, `graphs/selected.json`, `head_seed_rows.json`) are gitignored, as are `graphs/*.npz`.
- Stage 2 builds on `ags_head.py` (the head is the trunk's starting point) and `graphs/<aspect>.npz`.
- This chat was marked unattended by `loop-next`; open the next one with `--notify` so it can delete.
- Held data: none read; everything ran on the frozen train-side sample.

## 6. State at handoff

- Running: nothing (no job, subagent, Codex job or GPU lock).
- Uncommitted: the untracked files from before this chat (archive result folders, `sample.npz`, earlier Codex notes
  marked for deletion).
- On disk over 1 GB: the aspect-instruction store (6.5 GB, keep, user 18:05); `src/test/20261018_aspect_graph_sheet/outputs/` (1.1 GB, deferred).
