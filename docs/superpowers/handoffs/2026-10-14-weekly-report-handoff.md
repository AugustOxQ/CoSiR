# Handoff: discuss weekly report draft v1 (8 to 14 October) with the user, then fix, check and finish it

## Brief
- Done: the outside-view card "Where Our Method's Gain Comes From" is published (dev-log update #2).
- State: weekly report draft v1 is written, uncommitted and unchecked; its checks, figures and slides were stopped.
- Next: discuss draft v1 with the user first; then figures, the two checks and one fix wave.
- Blocked: the user's comments on draft v1; slides wait for the user's word.

> Written 2026-10-14 09:22 by the "outside weekly" chat (outside the r9 loop; not a loop phase chat), at the user's
> request to move the weekly report to a new chat. Direction log: "Now" and the entries from 2026-10-07 on.

## 1. Read in this order

1. This handoff.
2. `~/.claude/references/weekly-report.md` (report, slides, process: drafts are not committed or indexed until the
   user approves).
3. The draft: `docs/reports/weekly/drafts/2026-10-14_weekly_v1.md` (636 lines; over the 350 to 450 target).
4. The published card, whose content the weekly report repeats in its section 4:
   `docs/reports/stage/2026-10-14_aff_scope_limits.html` (text: `.scratch/stage_pages/2026-10-14_aff_scope_limits.draft.md`).

## 2. Decided by the user, not to be reopened

- The detailed update goes in the weekly report; the outside-view card stays one card (user, 2026-10-14).
- The card is published (user's word, 2026-10-14 09:16).
- Outside-view reports use no agent-style words ("Our reading", "Our advice", "we think"), only neutral phrases such
  as "This shows that ..." (user, 2026-10-14; now in `~/.claude/references/outside-view-report.md`).
- The weekly slides are not made now (user, 2026-10-14 ~09:05: "no need to do the weekly slide now").
- The weekly meeting is today at 11:00 (user).

## 3. Where things stand

- **Draft v1** (Opus): Summary; Idea and contributions (copied from last week, updated); How we got here; §1 the held
  read; §2 design L (failed branch); §3 round 8 (failed branch); §4 where AFF's gain comes from (split by shown
  aspect, the big table, k-shot, CSD AMI, Leiden); §5 what the paper needs (r9 plan and status, paper review); §6
  go/no-go checklist; §7 status and open decisions (the labels question with three options); glossary; sources.
  14 tables; easy cuts named by the drafter: Table 6, Table 10, parts of the glossary.
- **Three `FIGURE NEEDED` markers** (lines 138, 349, 390): a week timeline; AFF − B′(A0) by the aspect the pairs show;
  the k-sweep by shown aspect. A Codex brief for all three exists:
  `.scratch/weekly/codex_weekly_figs.txt` (target folder `docs/reports/assets/2026-10-14_weekly/`, not yet
  created).
- **Checks not run:** the Codex fidelity check (brief beside the figure brief: `codex_weekly_fidelity.txt`) and the
  Claude cold read; both were stopped by the user's switch to the card.
- **New numbers the draft uses:**
  - The seed-42 big table, all eight scorers: `src/test/20261203_kshot_supports/results/ks_big_table.txt`, committed
    d3d70aa, all asserts passed. On development: AFF − B′(A1) +0.33 [+0.05, +0.62]; AFF − R1 +0.22 [+0.06, +0.37].
    On held (round 6): +0.14 [−0.03, +0.30] and +0.02 [−0.06, +0.11].
  - The r9 SigLIP 2 held read H7: GO under the rule, AFF − B′(A0) +0.51 [+0.36, +0.66], all seven checks
    (`src/test/20261202_r9_scope/results/held_verdict_H7.json`, r9 log 08:57). It is preliminary: no auto report yet
    and no user reading. The draft marks it preliminary. H6 (Qwen3-VL-Embedding-2B) has no held result yet; check the
    r9 log before the meeting.

## 4. Open points for the next step

1. **Key:** the user's comments on draft v1 (length, the questions it answers, what to cut). Ask before any fix wave.
2. **Key:** should the weekly report also drop agent-style words? The template has "Verdict (reference)" lines and
   "Our reading" marks; the user's 2026-10-14 rule covers outside-view reports only.
3. Minor: whether to add H6 if the r9 run records it before the meeting.
4. Then: the figures (Codex), the fidelity check (Codex) and the cold read (Haiku) in parallel, one fix wave, look at
   every figure; slides and the pptx only on the user's word.

## 5. Code and pitfalls

- `/project/CoSiR` is on main and shared with the r9 run chat, which commits there: stage only your own files.
  `src/test/20261202_r9_scope/held_started_H6.json` is r9's; do not touch it.
- `scripts/check_reports_sum.py` reports the unindexed draft; that is expected until the user approves.
- Report file names of auto reports carry CoSiR's sequence-number dates (2026-11-xx); they are not real dates.
- The GitHub repo is public: no link to it in the report or the slides.

## 6. State at handoff

- Running: a local preview server on port 8765 (`python3 -m http.server`, docs/reports) from the old chat; it stops
  with that chat. Nothing else.
- Uncommitted: `docs/reports/weekly/drafts/2026-10-14_weekly_v1.md` (on purpose, until the user approves).
- On disk over 1 GB: nothing from this chat.
