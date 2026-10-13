# Handoff: the outside-view reports topic is closed; the next one is made when the user says "make outside-view report"

## Brief
- Done: both stage pages rebuilt for outside readers and online with the new start page; the rules folded into `outside-view-report.md`.
- State: the site holds the start page, 2 to 4 October and 5 to 12 October; nothing is staged.
- Next: nothing in this topic. A new page only on the user's call.
- Blocked: nothing.

> Written 2026-10-13 22:04 by the outside-pages chat (2). Outside the research loop (a report and process topic).

## 1. Read in this order

1. `~/.claude/references/outside-view-report.md`: what an outside-view report is, when it is made, who builds it,
   how it is packed and published.
2. `~/.claude/templates/devlog_primer_template.md`: the start page, the project's only primer.

## 2. Decided by the user, not to be reopened

All on 2026-10-13, recorded in the reference above and in `report_templates_changelog.md`: the name
"outside-view report"; made only when the user says "make outside-view report" (one or two days apart is a good
interval, never a schedule); the length stays about 1,800 words of prose; the site's start page is the only primer
(no pinned Slack copy); Preliminaries is copied from the previous page; the 5 to 12 October page covers only the work
after the first page; publish all three pages.

## 3. Where things stand

- **Site** (https://cosir-devlog.pages.dev, PIN-gated): start page, `2026-10-04_aspect_conditioned_similarity_methods`
  and `2026-10-12_affect_steering_line` (deploy ef554d57; site repo pushed, 718fc2b). The 12 unused figures of the old
  12 October page stay in the site's asset folder (no page links them; the user was told).
- **Project repo** (`r9-scope`): the new 12 October page and its structure figure `aff_structure.png` (8c9a6a7).
- **claude-config** (e424ba6): the rule update; `stage-page.md` renamed `outside-view-report.md`.
- **Not on the page, by the no-Limits rule:** the held paintings were read once before in the earlier task, and 24 held
  rows have a near-copy in the training images.

## 4. Open points for the next step

- Minor: should box 2 of the head-reader figure (first page) show the heads? Not answered.
- The next outside-view report would cover round 8 (ended 2026-10-13 with no pick) and round 9, once a stage report
  covers them.

## 5. State at handoff

- Running: a local server on port 8765 (serves `docs/reports/`); it ends with this chat.
- Uncommitted: `.scratch/stage_pages/` (gitignored; the old 12 October source is kept as `*.src_v1_2026-10-12.html`).
- On disk over 1 GB: nothing.
