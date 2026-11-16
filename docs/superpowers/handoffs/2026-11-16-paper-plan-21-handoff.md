# Handoff: paper plan 21: after the co-author discussion (2026-11-17 morning), the fixes the user picks from the sorted findings, then the final build and upload checks

## Brief
- Done: checks before upload, numbers check, web checks, two reviews (ARS, Ducktor), one sorted findings file; co-author notes all solved except point 6; contribution 3 fixed.
- State: REPO `c124a38` pushed, 8 pages + references, page 8 ends with the conclusion, about 10 lines free.
- Next: the user brings the co-author discussion's outcome; apply the fixes the user picks, build, recheck page 8.
- Blocked: the co-author discussion (2026-11-17); paper due 2026-11-17 12:59 Amsterdam.

> Written 2026-11-16 12:50 by the "paper plan 20" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens), at the user's yes. Direction log entry: none (paper work; decisions are rows in the paper folder's
> README and the status table).

## 1. Read in this order

1. This handoff, all of it.
2. PF `notes/review/2026-11-16_findings_sorted.md`: one sorted list of every finding of the five check and review
   files of 2026-11-16 (groups A to E). The co-author discussion picks from it. The five source files sit beside it.
3. PF `user_read/2026-11-13_paper_status.md`: the status table the user reads; keep it current.
4. PF `notes/2026-11-15_coauthor_comments.md`, last section: what the co-author asked and what is solved.
5. Project memory `feedback_user-writes-paper.md`: the user writes the paper; agents write only text the user sets.

Skills: none.

## 2. Decided by the user, not to be reopened

All on 2026-11-16:
- **Edit freeze** on REPO until the co-author discussion and their edits and comments are in. Review findings stay as
  separate files in `PF/notes/review/`, not in the PDF. Exceptions so far, at the user's request: the "[done]" marks
  (REPO `1238983`, `c124a38`) and contribution 3 (`c124a38`).
- **LLM Usage Statement:** the user re-reads the CVPR 2027 LLM policy and writes it alone. Agents only remind the user
  before submission. It is mandatory (missing: desk rejection possible).
- **Co-author points:** all solved except point 6 (the overview issues "we discuss later" and the caption length).
  Resolved notes in the source end with "[done]".
- **Contribution 3** reads "retrieval in all four directions between images and texts, ... and emotion on the photos
  of Affection and EmoSet" (the user's wording choice; one line more).
- Title is now "CoSiR: Conditional Similarity Learning Across Aspects" and the abstract opens "We present \method, a
  lightweight method for ..." (user's commit `2a412b1`, from a co-author's edit).
- New status row 23: whether to reduce the space above and below the tables (not checked whether CVPR 2027 allows it).

## 3. Where things stand

- REPO `c124a38` = origin; PF `5219a94` = origin. `main.pdf` 11 pages (8 + references), no undefined references.
  Page 8: the conclusion ends in the right column with about 10 lines free; the references start on page 9.
- Not marked "[done]" (the user's own `% \TODO` lines): the lens dataset (`2_related_work.tex:22`; Lenses is cited
  as a model at line 30), the aspect-specific setting (line 51, a question for the co-authors), the opening sentence
  of Sec. 4 (none in the text), and the facts double-check (`4_experiments.tex:8`; done by the numbers check, three
  wording fixes open, group A of the sorted file).
- Overleaf's own comment-panel threads are not in git; agents see them only when the user pastes them.

## 4. Open points for the next step

Key, in order:
1. **The co-author discussion's outcome.** The user says which findings to fix. The user writes the text, or sets
   the exact text for an agent. After each change: build, check page 8 still ends the text.
2. **Reminder: the LLM Usage Statement** before submission (the user's).
3. **Final checks after the last edit:** build both PDFs, no undefined references, no visible notes, anonymous,
   PDF metadata empty, text ends on page 8 (the checks of `PF/notes/review/2026-11-16_checks_before_upload.md`).

With the co-authors (the user): the title, "label-free", Fig. 4(a) (labels differ from Fig. 1), the limitations
section (about 200 words, needs about 12 lines; 10 are free), the overview caption (point 6), status row 23.
The user: OpenReview title (now "... Across Aspects") and abstract.

After the deadline: the supplementary review (due 2026-11-24 12:59 Amsterdam).

## 5. Code and pitfalls

- PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.
- The user may edit REPO in this container as well as in Overleaf: `git status` and `git fetch` before any edit.
- Build `latexmk -pdf -interaction=nonstopmode -silent main.tex`, then `supp.tex` (it reads `main.aux`). Check the
  logs for "undefined". Page 8 end: `pdftotext -f 8 -l 8` and a 50 dpi render.
- Paper prose: no semicolons between clauses, no dashes, British spelling, no win/level/lose words, no
  "significant", novelty only "to our knowledge". A resolved co-author note is commented out and ends with
  "[done]", never deleted. No notes of ours in REPO `.tex` files.
- The handoffs live in MAIN, a public repo: no co-author names here.
- Get times with `TZ=Europe/Amsterdam date` before writing them (this chat wrote wrong times three times).

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing of ours in REPO, PF or MAIN's handoff folder. PF keeps the user's untracked
  `CBMI_IMP_CAMREADY (2).pdf` out of git.
- On disk over 1 GB: nothing. The scratchpad (page renders, Codex logs, a few MB) goes with the session.
