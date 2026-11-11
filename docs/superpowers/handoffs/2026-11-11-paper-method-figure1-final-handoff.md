# Handoff: after the user's meeting, finish Figure 1's style from their comments; then Figure 2, method 3.5 and Sec. 4.1 from their points

## Brief
- Done: Figure 1 loosened (3.32 in high) and now in the LaTeX repo with its caption; 3.4 written; Section 3 polished (no semicolons between clauses).
- State: the paper builds (6 pages); 3.5 test time is a heading only; Figure 2 is a placeholder with a TODO.
- Next: wait for the user's Figure 1 comments from the 2026-11-11 meeting, then fix the figure.
- Blocked: Figure 1's final style waits for the meeting; 3.5 and Sec. 4.1 wait for the user's points.

> Written 2026-11-11 04:28 by the "paper method" tab. Outside the research loop: no loop step, no direction log entry.
> PF below means `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`; REPO means `PF/CVPR2027-CoSiR-Paper/` (the
> Overleaf-synced LaTeX repo the co-authors see).

## 1. Read in this order

1. This handoff, all of it.
2. `PF/figures/fig1_overview.png` (look at it) and its script `PF/figures/src/fig1_overview.py` (inch coordinates,
   origin top left, y down; column constants at the top, rows inside `main()`; the test strip is placed relative to
   the hairline `HL`).
3. `PF/README.md`, the Decisions rows of 2026-11-11 (the last three are new in this chat).
4. `REPO/Sections/3_method.tex` (all of it, about 170 lines) and, only when a fact is in question,
   `PF/notes/trace/3_method.md`.
5. For 3.5: `PF/drafts/v1.1/03_method.md` lines 257 to 287 (inference, the four retrieval directions, one trunk per
   aspect at test time; the read-out paragraph after it is not decided).

Skills: none by default. Ducktor commands only when the user types or names one (read the command's SKILL.md under the
plugin root; Ducktor is not switched on in this project).

## 2. Decided by the user, not to be reopened

All of section 2 of `2026-11-11-paper-method-student-handoff.md` stands. New on 2026-11-11 (in `PF/README.md`):

- **Figure 1 is in REPO** (`Figures/fig1_overview.pdf`) although not final; the v1.1 figure moved to
  `REPO/Figures/old/fig1_overview_v1.1.pdf`, not deleted. Its caption: the figure report's four sentences, one
  sentence per component. The style is discussed after the user's meeting of 2026-11-11.
- **Figure 2** stays a placeholder until Figure 1's style is final; its caption carries the user's
  `\TODO{this borrows from earlier figures, need to redraw}`.
- **3.4 as written**, with the three sentences added beyond the points (trunk and head layers, only the towers
  trained, the stop rule).
- **Experiment settings** (batch size, temperatures, alpha 0.25, beta 8, patience 15, at most 200 epochs, AdamW with
  learning rate 1e-3 and weight decay 0.05, widths 2048 to 1024 to 2048, dropout 0.1) go into Sec. 4 as its first
  subsection. Sources: v1.1 TRACE lines 512 and 518.
- **Section 3 polish kept**, both flagged edits included ("whereas" in 3.2, "of the denominator" in 3.4).
- **No semicolons between clauses** in any paper prose: a global rule now ("Paper-draft style" in
  `~/.claude/rules/report-writing.md`). Use a comma, a connecting word or a full stop; semicolons inside math stay.
- Table 1's caption: "The instructions used with Qwen3-VL-Embedding-2B." (no "verbatim").

## 3. Where things stand

- REPO last commit `23b5c45` (Figure 1 with caption, Section 3 polish, Table 1, Figure 2 TODO), pushed. Before it,
  `dfb79e3` (3.4). `latexmk -pdf main.tex` exits 0, 6 pages (was 5; the taller figure), no undefined references, no
  overfull lines.
- `cosir-paper` repo (`/project/CoSiR/docs/paper`) last commit `254d27c`, pushed: Figure 1 script and renders
  (`6e35fc6`), README decisions, the captions note, the 3.4 and polish traces.
- Figure 1: canvas 6.875 by 3.32 in (was 2.6), placed at 100% width. Every element is at least 0.06 in from the
  edges, every arrow shaft at least 0.155 in, the bus split 0.20 in, no blue crossing. A bounds and arrow check
  script is in this chat's scratchpad only; rebuild it if needed (iterate `ax.texts`, `ax.patches`, `ax.lines`,
  `ax.images` after `main()`, since `main()` leaves the figure open).
- Ducktor polish report: `/project/CoSiR/.duck/reports/2026-11-11-polish.md` (in `.git/info/exclude`).

## 4. Open points for the next step

Key:
1. **Figure 1 style**, from the user's meeting comments. Look work stays in the main chat or one Opus `high` worker.
   After each change: render, check at 300 dpi, SendUserFile the PNG; when the user is satisfied, copy the PDF to
   `REPO/Figures/`, rebuild, look at page 2, commit and push both repos, add a line to
   `/project/CoSiR/.claude/20261111_log.md` (or the day's log). Then ask about Figure 2.
2. **3.5 Test time**: needs the user's points first (no direct writing). Likely material: one encoder pass under the
   generic instruction, the tower by input type, the four retrieval directions, each aspect's own trunk. The image
   retrieval read-out of v1.1 is not decided; ask before it enters.
3. **Sec. 4.1 settings**: needs the user's skeleton for Sec. 4 first.

Minor:
4. The licence of the ArtELingo captions shown in Figure 1 is not checked.
5. Is "no direct writing" (and "no notes in REPO") a rule for all the user's papers? Related work (c) is still a
   sentence and two TODOs.

## 5. Code and pitfalls

- Figure script rules (the user's): every arrow's last segment at least about 0.15 in; the split from the input bus at
  least 0.2 in before the boxes; the blue lines from the student must not cross (y turns up on the inner line at `XY`,
  z on the outer at `XZ`); the only crossing is the green teacher line with the two blue lines, drawn with a gap;
  nothing touches the canvas edge.
- Run it with `PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python fig1_overview.py` in
  `PF/figures/src/`; check `pdffonts ../fig1_overview.pdf` (three STIX TrueType fonts, no Type 3).
- Each finished text part: write into `REPO/Sections/*.tex`, `latexmk -pdf -silent main.tex`, render the page with
  `pdftoppm` and look at it, commit and push REPO (only our files), update `PF/notes/trace/<section>.md`, commit and
  push `cosir-paper`, then SendUserFile `REPO/main.pdf` (`display: render`).
- Wide equations overflow the column: split them with `aligned` (as Eq. 10).
- The shell is zsh: quote globs. Paper names, never internal ones. No "development data". No semicolons between
  clauses.
- `/project/claude-config` holds another session's uncommitted edits (`design/modules.yaml`, `design/map.html`, some
  templates): commit only your own hunks there.

## 6. State at handoff

- Running: nothing.
- Uncommitted: `REPO/Sections/2_related_work.tex` (the user's own edit; leave it);
  `/project/CoSiR/docs/superpowers/constitution.md` (modified before this chat, not ours);
  `PF/CBMI_IMP_CAMREADY (2).pdf` (the user's, untracked).
- On disk over 1 GB: nothing. The session scratchpad holds page and figure renders (a few MB); they go with the
  session.
