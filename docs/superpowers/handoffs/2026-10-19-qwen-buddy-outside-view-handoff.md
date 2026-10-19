# Handoff: write a stage report and a quick outside-view report of the Qwen + buddy line (2026-10-15 to now), then upload it to the dev log after the user's look

## Brief
- Done: r10 decide 10 finished (comparator, ceiling check, stage 2 grill); r10 decide 11 now writes the stage 2 spec.
- State: our best network, default input, 0.081 / 0.348 / 0.458 image-side AMI (emotion / style / genre), against PercepT core 0.077 / 0.229 / 0.436 and the instructed Qwen image 0.093 / 0.395 / 0.496.
- Next: the stage report of the line, then the outside-view page, its dev-log post, and the upload the user asked for.
- Blocked: nothing.

> Written 2026-10-19 19:25 by "r10 decide 10" on the user's request. A side chat outside the loop's phases: no loop
> step, no decision of the loop. Direction log: [entries 2026-10-15 to 2026-10-19](../direction_log.md).

## 1. Read in this order

1. `~/.claude/references/outside-view-report.md`: the whole procedure (stage report first, who builds each part,
   pack, dev-log post, publishing). Then `~/.claude/templates/stage_report_template.md`,
   `~/.claude/rules/report-kinds.md`, `report-writing.md` and `~/.claude/references/ste-writing.md`.
2. The direction log entries from 2026-10-15 (the conditioned line stops) to 2026-10-19 (stage 2 grill): the chain
   of the line, with each decision and who made it. "Now" gives the Next part.
3. The experiments: rows 12-06 to 12-15 of `docs/reports/reports_sum.md` (auto reports in
   `docs/reports/auto/percept/`) and their user-read reports in `docs/user_read/` (2026-10-17 to 2026-10-19). The
   user-read reports are the quickest read; the auto reports hold the numbers.
4. Today's two pieces: the two-bar table in [2026-10-19_percept_core_k40.md](../../user_read/2026-10-19_percept_core_k40.md)
   and the ceiling log
   [20261019_emotion_ceiling_log.md](../../../src/test/20261019_emotion_ceiling/20261019_emotion_ceiling_log.md).
5. Literature for Background and the claim: [2026-12-05_percept_neighbourhood.md](../../reports/literature/2026-12-05_percept_neighbourhood.md)
   (PercepT's weaknesses: SI is DEC's own objective; no retrieval) and
   [2026-12-07_prompted_aspect_views.md](../../reports/literature/2026-12-07_prompted_aspect_views.md) (novelty
   verdict; the "just prompting" attack).
6. The previous page, from the archived conditioned line:
   `archive/v2_conditioned/docs/reports/stage/2026-10-14_aff_scope_limits.html` (look, Preliminaries, figure style).
   Its source may be in `.scratch/stage_pages/`.

Skills: none.

## 2. Decided by the user, not to be reopened

- Scope: the Qwen + buddy line, from its start after the conditioned line stopped (2026-10-15) to today. "Quick":
  keep the page short (2026-10-19).
- Upload: the user asked for it ("make a quick outside view report and upload", 2026-10-19). Show the local preview
  first; deploy with `devlog-publish` after the user has looked.
- The line's decisions (B2, comparator option b, stage 2 target) are in the direction log; the page reports them, it
  does not reopen them.

## 3. Where things stand

Image-side AMI, k = 20, 5,000 development paintings (no held data yet), emotion / style / genre:

| Row | AMI |
|---|---|
| Qwen default image, clustered directly | 0.070 / 0.222 / 0.449 |
| CLIP image, clustered directly | 0.065 / 0.300 / 0.412 |
| PercepT core with K = 40, best setting per aspect | 0.077 / 0.229 / 0.436 |
| Our network, default input (one pass) | 0.081 / 0.348 / 0.458 |
| Instructed Qwen image, clustered directly (three passes; stage 2's target) | 0.093 / 0.395 / 0.496 |
| Linear probe on default Qwen (labels; a ceiling, not a method) | 0.121 / 0.527 / 0.657 |

- Clear gains: style over every baseline; emotion level with PercepT core; genre level with the Qwen image.
- Stage 2 (one trunk with three heads) is being specified in r10 decide 11; it has no results.

## 4. Open points for the next step

- Agent defaults: the stem name and date (stage reports use real Amsterdam dates); which experiments are positive
  findings with a figure.
- Minor: whether the start page (dev-log primer) needs a line about the new line; check with `devlog-publish` and
  `devlog_primer_template.md`.

## 5. Code and pitfalls

- The GitHub repo is public: never name or link it on the page or in the post (double-blind); `devlog.json` in
  `/project/cosir-devlog` lists strings the site may never hold.
- Development data only: every number says so; no held-out claim exists yet.
- Routing: `agent-routing.md` and the "Who builds it" steps; the figure is a Codex job (`codex-jobs.md`).
- r10 decide 11 runs in parallel and edits specs and the direction log: commit only this chat's files.
- Index the stage report in `reports_sum.md` and run `docs-check`.

## 6. State at handoff

- Running: r10 decide 11 (stage 2 spec, with the user) and r10 main 3 (watching loop r10).
- Uncommitted: nothing (after this chat's commit).
- On disk over 1 GB: `src/test/20261018_aspect_graph_sheet/outputs_fu/` (1.2 GB, deferred; not this chat's).
