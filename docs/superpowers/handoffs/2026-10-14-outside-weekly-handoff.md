# Handoff: make the one or two outside-view reports the user names, then the weekly report, its slides and the pptx

## Brief
- Done: the Leiden and k-shot side checks closed; readings recorded; labels question logged for the meeting.
- State: last stage report and outside-view page cover 2 to 12 October; last weekly report 2026-10-07.
- Next: ask the user which one or two outside-view reports to make, then the weekly report, slides and pptx.
- Blocked: the user's choice of outside-view reports; the weekly meeting is today (2026-10-14).

> Written 2026-10-14 by the "Leiden groupings 2" side chat (outside the r9 loop), at the user's request to move to a
> new window. Not a loop phase chat. Direction log: "Now" paragraph and the entries from 2026-10-12 on.

## 1. Read in this order

1. This handoff.
2. `~/.claude/references/outside-view-report.md` (how an outside-view report is made: the stage report it rests on,
   content on Opus, cold read on Haiku, check on Sonnet, HTML on Sonnet; the dev-log post; publishing only on the
   user's word).
3. `~/.claude/references/weekly-report.md` before the weekly report (report, slides, process; drafts are not
   committed or indexed until the user approves).
4. The direction log `docs/superpowers/direction_log.md`: "Now", "Parked ideas", and the entries dated 2026-10-12 to
   2026-10-14.
5. Only for the material: the reports listed in section 3.

## 2. Decided by the user, not to be reopened

- First one or two outside-view reports, then the weekly report with slides and pptx (user, 2026-10-14). The user
  names which outside-view reports.
- Outside-view reports are made only on the user's call, and published to the dev-log site only on the user's word.
- The weekly meeting is today; the user takes the labels question there (below).

## 3. Where things stand

- **Already covered:** stage report [2 to 12 October](../../reports/stage/2026-10-12_affect_steering_line.md) (through
  round 7) with its `.html` page and dev-log post; the [AFF model card](../../reports/stage/2026-10-13_aff_model_card.md)
  (2026-10-13); the last weekly report [2026-10-07](../../reports/weekly/2026-10-07_aspect_task_to_affect_steering.md)
  with its slides.
- **New since then** (auto reports keep CoSiR's sequence dates; user-read reports in `docs/user_read/`):
  - Round 8, learned groupings: no pick, AFF stays best ([report](../../reports/auto/v2/2026-11-29_learned_grouping.md),
    direction log 2026-10-13).
  - r9 (AFF on more backbones and datasets): spec approved 2026-10-13; build closed 2026-10-14 (merged to main,
    71843c1); the run is in progress in another chat. Its state: `docs/superpowers/handoffs/2026-10-14-r9-run-handoff.md`
    and the round log under `src/test/20261202_r9_scope/`. Use only results the r9 chats have recorded.
  - Side checks, 2026-10-14: k-shot ([report](../../reports/auto/v2/2026-12-03_kshot_supports.md)): AFF needs at least 3
    example pairs, and its whole lead is in the emotion episodes; Leiden groupings
    ([report](../../reports/auto/v2/2026-12-04_leiden_groupings.md)): tie on the bar margin, kept as a record.
  - **Open, for today's meeting:** are aspect labels needed in training to separate style from genre? Evidence,
    options and our recommendation: the k-shot entry of the direction log ("Open"). The history behind it (R1 steered
    on all aspects; only emotion paid; rounds 7 and 8 found no label-free style route) is in the entries from
    2026-10-04 to 2026-10-13.
- The paper chat runs separately (`docs/paper/` is its own private repo).

## 4. Open points for the next step

1. **Key, ask first:** which one or two outside-view reports? A natural candidate is a stage report for 12 to 14
   October (round 8, r9 so far, the side checks, the labels question); the user may name another. Ask also for the
   meeting time, to plan the order of work.
2. Then the weekly report (2026-10-07 to 2026-10-14), its slides and the pptx (`docs/reports/pptx/`, gitignored; build
   scripts in `docs/reports/assets/`).

## 5. Code and pitfalls

- `/project/CoSiR` is on main and shared with the r9 run chat, which commits there: stage only your own files.
- Stage and weekly reports use real Amsterdam dates; auto reports keep CoSiR's sequence dates (`docs-layout.md`).
- The GitHub repo is public: no link to it in anything meant for the paper or outside reviewers.

## 6. State at handoff

- Running: nothing from this chat. Uncommitted: nothing of this chat's.
- The previous close handoff: `2026-10-14-side-checks-close-handoff.md`.
