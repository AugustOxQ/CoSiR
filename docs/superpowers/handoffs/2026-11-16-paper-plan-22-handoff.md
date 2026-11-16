# Handoff: paper plan 22: go on with the Overleaf comments one by one (the user pastes each), then the review findings, then the final checks

## Brief
- Done: wrong facts A1 to A3 fixed; Overleaf comments 1 to 7 solved one by one (log in PF notes); headings now bold text.
- State: REPO `dfff09a` pushed; 8 pages + references; page 8 ends at line 548, about 1 line free.
- Next: the user pastes the next Overleaf comment; propose, apply after the yes, build, check page 8 and the float pages.
- Blocked: nothing; paper due 2026-11-17 12:59 Amsterdam, co-author discussion 2026-11-17 morning.

> Written 2026-11-16 14:25 by the "paper plan 21" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens), at the user's yes. Direction log entry: none (paper work; decisions are rows in the paper folder's
> notes and the status table).

## 1. Read in this order

1. This handoff, all of it.
2. PF `notes/2026-11-16_overleaf_comments.md`: the seven comments solved today, with the text that went in and why.
   New comments go on as rows 8, 9, ...
3. PF `notes/review/2026-11-16_findings_sorted.md`: the review findings (A done; B to E open), for after the comments.
4. PF `user_read/2026-11-13_paper_status.md`: the status table the user reads; keep it current.
5. Project memory `feedback_user-writes-paper.md`: the user writes the paper; agents write only text the user sets.

Skills: none.

## 2. Decided by the user, not to be reopened

All on 2026-11-16, recorded in the comments log and the sorted findings file:
- **Order of work:** the Overleaf comments first, one by one (the user pastes each with its sentence), then the other
  review findings of the sorted file, then the final checks.
- **Each change:** propose the exact text (or apply the user's own wording), apply only after the user's yes, build,
  commit and push; the user pulls into Overleaf and resolves the comment there.
- **A1 to A3** fixed with the text the user approved (REPO `3a3674a`). A3 reads "no aspect instruction", not "default
  instruction" (the image goes through SigLIP with no instruction).
- **Comments 1 to 7:** intro paragraph 3 at a high level; comment 2 already solved as co-author point 1;
  contribution 3 standalone (option a, without "four directions"); Lenses (one line in related work, "The Lenses
  dataset was not available to us at the time of submission." in Sec. 4.1); scope at the conclusion's end (no change in
  related work); "aspect labels (such as awe or amusement for emotion)"; Qwen3-VL-Embedding named in Compared methods.
- **Headings:** `\noindent\textbf{X}` instead of `\paragraph{X}` (the PDF is the same; the preamble mapped them).
- Still in force from the previous handoff: the LLM Usage Statement is the user's alone (agents only remind);
  review findings stay in `PF/notes/review/`, not in the PDF.

## 3. Where things stand

- REPO `dfff09a` = origin; PF `ca14e3a` = origin. `main.pdf` 11 pages (8 + references), no undefined references,
  `supp.pdf` builds.
- Page 8: the conclusion ends at line 548, about 1 line free (about 10 lines were free at the start of this chat).
- Float pages: p1 Fig. 1; p3 Figs. 2, 3; p6 Tables 1 to 3; p7 Tables 4 to 6; p8 Fig. 4, Table 7.
- Overleaf comments: 7 solved. The user resolves them in Overleaf after pulling from GitHub.

## 4. Open points for the next step

Key, in order:
1. **The next Overleaf comments**, one by one.
2. **The review findings** of the sorted file after the comments (B1 caption retrieval protocol, B2 to B14, C, ...).
   The fact for B14 is already found: a node with no emotion neighbours gets the mean weight of the others
   (MAIN `src/test/20261020_claim_test/ct2_graphs.py:82`).
3. **Reminder: the LLM Usage Statement** before submission (the user's).
4. **Final checks after the last edit:** both PDFs, no undefined references, no visible notes, anonymous, PDF metadata
   empty, text ends on page 8 (`PF/notes/review/2026-11-16_checks_before_upload.md`).

With the co-authors (the user): the title, "label-free", Fig. 4(a), the limitations section (about 12 lines; about 1
is free now, so it needs a cut elsewhere), the overview caption (point 6), status row 23 (space around the tables).
Minor: table row names differ ("Instructed Qwen3" in four tables, "Instructed embedding" in the retrieval and photo
tables; finding C9).

## 5. Code and pitfalls

- PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.
- **Float sensitivity:** one added line in Sec. 3 or 4 can push Table 6 from page 7 to page 8 and the conclusion to
  page 9 (seen twice today). Test candidate wordings by building each and listing the float pages
  (`pdftotext -f N -l N main.pdf - | grep '^\(Table\|Figure\) [0-9]\.'`) and the line of "future work." on page 8.
- Overleaf comments are not in git; agents see them only when the user pastes them. The CVF open-access site returns
  403 to fetches (use the project page or the user's copy).
- The user may edit REPO in this container as well as in Overleaf: `git status` and `git fetch` before any edit.
- Build `latexmk -pdf -interaction=nonstopmode -silent main.tex`, then `supp.tex` (it reads `main.aux`).
- Paper prose: no semicolons between clauses, no dashes, British spelling, no win/level/lose words, no
  "significant", novelty only "to our knowledge". A resolved co-author note or TODO is commented out and ends with
  "[done]", never deleted. No notes of ours in REPO `.tex` files.
- The handoffs live in MAIN, a public repo: no co-author names here.
- Get times with `TZ=Europe/Amsterdam date` before writing them.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing of ours in REPO or PF. PF keeps the user's untracked `CBMI_IMP_CAMREADY (2).pdf` out of git.
- On disk over 1 GB: nothing. The scratchpad (page renders, a few MB) goes with the session.
