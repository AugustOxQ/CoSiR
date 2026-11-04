# Handoff: weekly report of 2026-11-04: discuss v3 with the user, then build the pptx after their edits

## Brief
- Done: outside-view page published (update #6, 29 October to 3 November). Weekly report v3 and slides v3 are drafted and checked.
- State: 21 slides, about 11,000 words; no wrong number in the fidelity checks. Drafts are not committed or indexed.
- Next: discuss the weekly report with the user. After their edits, build the pptx.
- Blocked: waits for the user's comments on v3.

> Written 2026-11-04 10:30 by the report chat "stage report and weekly". Not a loop chat. Direction log: "Now" holds the r17 reporting
> rule; "Parked ideas" holds the from-scratch photo run.

## 1. Read in this order

1. `docs/reports/weekly/drafts/2026-11-04_datasets_and_retrieval_v3.md`: the report.
2. `docs/reports/weekly/drafts/2026-11-04_datasets_and_retrieval_slides_v3.md`: the deck.
3. `~/.claude/references/weekly-report.md`: the process, the slide rules, how to build the pptx.
4. `.scratch/stage_pages/2026-11-04_weekly_v3_brief.md`: the word rules below, in full.
5. `.scratch/stage_pages/2026-11-04_term_audit.md`: only when a term comes up.

Skills: none.

## 2. Decided by the user, not to be reopened

All from 2026-11-04.

- The weekly is standalone, and its contents only extend last week's. No re-introduction of last week's design or sweep.
- No "win", "claim", "beat" or "bar" in any form. Keep "level" and "below". "Clearly above" is the outcome; "higher" is a bare point estimate.
- Level means not worse than one seed range; it does not show that the two are equal.
- "Post-processing" for read-out, "score only" for descriptive, "in progress" in coverage cells.
- "Text side" and "text trunk" for the model part; "explanations" for the photo data.
- "Outside comparisons" is the plain name of the round called r18.
- **r17's results:** they enter this weekly only after the user's reading in the r20 chat. They never enter the stage report of 2026-11-04. The next stage report starts from r17.
- **A photo model from random weights** at the picked settings, five seeds on the Affection test photos, runs after r19 frees the GPU. It is in the direction log's "Parked ideas".
- The pptx is built only after the user adjusts the weekly markdown.

## 3. Where things stand

- **Published:** `https://cosir-devlog.pages.dev/p/2026-11-04_datasets_and_retrieval.html`, deployed 2026-11-04 10:24, behind the shared PIN. The site repo is pushed. Source: `docs/reports/stage/2026-11-04_datasets_and_retrieval.{md,html}`, indexed.
- **Post for Slack:** `docs/reports/stage/2026-11-04_datasets_and_retrieval_devlog.md`. The user pastes it; the saved file keeps a placeholder for the link.
- **Weekly checks:** report v1 fidelity 11 findings, no wrong number; cold read 169 items; slides v1 fidelity 9 findings, cold read 61 items; all triaged in `.scratch/stage_pages/2026-11-04_*_changes.md`. After v3: no forbidden word in running text, every number on the slides is in report v3 except PercepT's arXiv ID, all image paths resolve from `drafts/`.
- **Figures:** `docs/reports/assets/2026-11-04_weekly/` and `slides/`, plus `2026-11-04_datasets_and_retrieval/`, are viewed by the controller. `2026-11-04_weekly/` is not committed yet; commit it with the approved weekly.
- **Status lines in the report and deck** are from 2026-11-04 08:30. r17's auto report (`docs/reports/auto/percept/2027-01-04_new_scores.md`) appeared at 09:15; r18 and r19 go on.

## 4. Open points for the next step

- Key: the user's edits to report v3 and slides v3.
- Minor: 21 slides; about 18 was the aim. Slides 5 and 6, 8 and 9, 20 and 21 can merge.
- Minor: slides 17 and 19 are dense for a projector (fidelity finding 8). Simpler versions need a figure round.
- Minor: `slides/progress_timeline.png` keeps its dates, and says "Three rounds running".
- Minor: refresh the status lines if the meeting is later than today.

## 5. Code and pitfalls

- Draft image paths are `../../assets/...`. When the deck moves up to `docs/reports/weekly/`, they become `../assets/...`, and slide 1's Notes say so.
- `~/.claude/templates/build_weekly_slides.py` prints HTML comments as slide text; v3 has none. Keep it so.
- After approval: move or copy the final report and slides to `docs/reports/weekly/`, add the rows to `docs/reports/reports_sum.md` (weekly and pptx tables, and the "Latest weekly report" line), run `python scripts/check_reports_sum.py` and `docs-check`, then commit and push.
- `2026-11-04_weekly/build_coverage.py` reads its cells from `.scratch/stage_pages/2026-11-04_datasets_and_retrieval.draft.md`.
- This repo is public: never name it in paper or submission material, and write no secrets here.
- A worker's scratch files can collide in the shared scratchpad; give each worker its own prefix, and have it read a file before it writes it.
