# Handoff: paper plan 12: discuss with the user where to start on the remaining paper work, for a semi-final version by Sunday

## Brief
- Done: Section 4 review complete (4.6, 4.7, all captions). A status report of what is left is written and checked.
- State: abstract, introduction and conclusion not written. Three figures need work. About 1.7 to 2 columns must be cut.
- Next: the user reads the status report, gives a view, then we decide where to start.
- Blocked: nothing.

> Written 2026-11-13 21:00 by the "paper plan 11" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens), at the user's request. Direction log entry: none (paper work; decisions are rows in the paper folder's
> README).

## 1. Read in this order

1. This handoff, all of it.
2. PF `user_read/2026-11-13_paper_status.md`: the status table the user asked for (a user-read report, checked by a
   cold read and a facts check). Show it to the user first; its advice is in a collapsed block, so ask for the user's
   view of where to start before opening ours.
3. PF `notes/2026-11-13_remaining_work.md`: the same list with file paths and line numbers.
4. Project memory `feedback_user-writes-paper.md` and global `~/.claude/rules/report-writing.md`, bullet "Results
   paragraphs in papers".
5. PF `README.md`, the last four rows (this chat's decisions).

Skills: none for the first step. For the whole-paper review round later: Ducktor's paper review or
`academic-research-skills:academic-paper-reviewer` (the user named either).

## 2. Decided by the user, not to be reopened

All on 2026-11-13, each a README row in PF or stated here:
- Section 4 review: 4.6 (recipe sentence, k = 80, "k = 40 for paintings" in Sec. 3), 4.7 and Figures 3 and 4 captions,
  Table 2 and supplementary Figure 5 captions. Table 8 marks kept open, memo point 3.
- The teaser becomes one column wide, at the top of the right column of page 1 (CVPR habit). Abstract, introduction
  and teaser together: about one page, a little more at most.
- The old full-width teaser (CoSiR on Qwen3) goes to the supplementary under another name, not "teaser".
- Method figures: Figure 1 redesigned (laid out for four blocks, now two, right side empty); Figure 2 redrawn and
  improved (the user finds it still an AI-made figure, although an agent restyled it on 2026-11-12).
- Method text is cut only if the Section 4 cuts are not enough. Section 4 cuts (figure layout, long captions, 4.1
  text) aim at about one page; the exact cut is decided later.
- The conclusion is written after the abstract and introduction (user).
- One whole-paper review round, with Ducktor's or ARS's reviewer, is on the list.
- A co-author reads a semi-final version on Sunday 2026-11-15, possibly before the final cut.

## 3. Where things stand

- REPO `bcb64b5`, built clean (only the old 0.7 pt overfull box); table check "groups checked 49 missing 0".
- Deadlines: paper 2026-11-16 23:59 AoE = **2026-11-17 12:59 Amsterdam (Tuesday)**; supplementary 2026-11-24 12:59.
- Length: the text ends on page 9 at y 342 of 720 pt (about 0.4 of a column over), and Figures 3 and 4 sit on page 10
  (about one column). With the conclusion and a full page 1: about 1.7 to 2 columns to cut. Not checked: empty space
  on float pages 2 to 8. PDF page 12 is references only.
- TODOs in the PDF: 5 (related work 2, Section 4 opening 1, Tables 6 and 8 one each). The introduction's bracketed
  notes are separate.
- Codex had no room at 20:45 (5h window resets 21:05).

## 4. Open points for the next step

Key, answer-first: where to start, and whether figure work (teaser on SigLIP 1, Figures 1 and 2) runs in parallel with
the user's writing. Our advice is in the report's collapsed block.

Minor: confirm the title (the user's edit of 2026-11-11, "Conditional"; report "Points to know"); check that the old teaser's model equals the Qwen3
row of Table 6 (not checked) before its supplementary caption; `figures/captions.md` holds old Qwen3 caption drafts
(ours).

## 5. Code and pitfalls

- PF = `docs/paper/cvpr2027_aspect_spaces/` (own private repo), REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.
- Pull REPO before each edit (Overleaf). After each push, tell the user Overleaf needs a pull. Stage only our lines.
- Apply accepted wording with a small Python script that asserts each old string occurs once.
- No agent notes, TRACE blocks or comments in REPO `.tex` files. A `\TODO` only when the user asks for it.
- No semicolons between clauses, no dashes, British spelling, "test set", no win/level/lose words.
- Build: `latexmk -pdf -interaction=nonstopmode -silent main.tex` in REPO. Supplementary: scratch copy with the
  `% \input{Sections/X_supplementary}` line uncommented.
- Crops: `pdftoppm -r 150`, positions from `pdftotext -bbox` (points times 150/72), PIL crop, SendUserFile.
- Paper status documents stay in PF (private), not in MAIN's `docs/user_read/` (MAIN is public).
- Teaser: `PF/figures/src/fig_teaser.py` (old, Qwen3, full width 495 x 165.6 pt). Figures 3 and 4 sources show how the
  SigLIP 1 data is read (`fig_qualitative.py`, `fig_clusters_classcount.py`, r28 qualitative candidates).

## 6. State at handoff

- Running: nothing of this chat. Other tabs' r31 runs may use the queue.
- Uncommitted: nothing of ours. PF has the user's untracked `CBMI_IMP_CAMREADY (2).pdf`, kept out of git.
- On disk over 1 GB: nothing. The scratchpad (crops, build copies, 13 MB) goes with the session.
- Repo heads: REPO `bcb64b5`, PF `b7624be`.
