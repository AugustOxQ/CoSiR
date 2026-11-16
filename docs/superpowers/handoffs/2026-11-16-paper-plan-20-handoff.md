# Handoff: paper plan 20: the user's Overleaf layout change, then rows 16 and 17 (checks before upload, LLM policy re-read), then the rest of the status table

## Brief
- Done: abstract (156 words), Fig. 1 caption, conclusion (93 words), all from the user's skeletons; Sec. 4 TODOs out; Fig. 3 and Tables 2 and 5 moved; status table has one Status column.
- State: REPO `86aa696` pushed; paper 8 pages + references, about 12 lines free under the conclusion.
- Next: compile after the user's Overleaf change; discuss rows 16 and 17 (checks before upload, LLM policy re-read).
- Blocked: the user makes one layout change in Overleaf first.

> Written 2026-11-16 04:45 by the "paper plan 19" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens), at the user's yes. Direction log entry: none (paper work; decisions are rows in the paper folder's
> README).

## 1. Read in this order

1. This handoff, all of it.
2. PF `user_read/2026-11-13_paper_status.md`: the status table (one Status column: done, open, after the deadline).
   The user reads this table; keep it current after each finished item.
3. PF `README.md`, the rows dated 2026-11-16 from "Abstract skeleton" on (this chat's decisions).
4. Project memory `feedback_user-writes-paper.md`: the user writes the paper; agents write only a part the user has
   set (this chat wrote the abstract, the caption and the conclusion at the user's request, from the user's skeletons).

Skills: none.

## 2. Decided by the user, not to be reopened

All on 2026-11-16, each a README row in PF:
- Abstract: the direct skeleton ("We present CoSiR: ..., a lightweight way of ...", graphs, adapter, datasets and
  tasks, one results sentence), no gap sentence, "novel" dropped; results sentence on clustering only (14 times
  faster, improves on the instructed Qwen3 embeddings on every aspect, MMArt, photos). The user edits it later.
- "Trained label-free" in the abstract, as in the introduction; the term is still to discuss with the co-authors.
- Fig. 1 caption from the user's message; the conclusion from the user's skeleton (no limitation sentence, no photos).
- A limitations section of about 200 words is the user's idea, to discuss with the co-authors.
- Sec. 4 has no opening sentence; its two TODO lines are commented out.
- The supplementary review waits until after the paper deadline.
- Figures 1 and 2 keep their positions; Fig. 3 at the bottom of page 3's right column; the user's table layout
  (Tables 2 and 5 at the bottom of their columns). The user makes further layout changes in Overleaf.
- Note for the co-authors: the overview caption (36 words) may be too short; reviewers may read Fig. 2 rather than
  Sec. 3, so method details may go back into it.

## 3. Where things stand

- REPO `86aa696` = origin; PF `72e0d77` = origin. `main.pdf` 11 pages (8 + references), builds clean; `supp.pdf` 3
  pages. Floats: p1 Fig. 1; p3 Fig. 2 (top, full width) and Fig. 3 (bottom right); p6 Tables 1 and 2; p7 Tables 3 to
  5; p8 Fig. 4 (top, full width), Tables 6 and 7. Conclusion heading at y 421 in page 8's right column, text ends at
  y 559; about 12 lines free (about 100 words with a heading). A 200-word limitations section needs about 12 lines
  more; dropping Fig. 4(a) and setting 4(b) in one column saves about 23 lines (measured 2026-11-16).
- Paper deadline 2026-11-17 12:59 Amsterdam; supplementary 2026-11-24 12:59 Amsterdam.
- The status table: 6 rows done, 15 open, 1 after the deadline (the supplementary).

## 4. Open points for the next step

Key, in order:
1. **The user's Overleaf change.** The user pulls `86aa696` in Overleaf, changes the layout, pushes to GitHub, and
   asks for a compile. Pull, build both PDFs, send the changed pages; while the user is still adjusting, send only
   the pages with no summary (user, 2026-11-16).
2. **Rows 16 and 17**, to discuss with the user ("discuss in the new chat"). Row 17: re-read the CVPR 2027 policy on
   LLMs (web; it said it may change before the deadline), then list our uses of LLMs for the LLM Usage Statement,
   which the user writes (mandatory, after the conclusion, outside the 8 pages, anonymous). Row 16: the checks before
   upload (list in the status file): anonymity and PDF metadata, every citation, the numbers in the text against
   their sources (the commented-out Sec. 4 TODO asked for this after the main-model switch), the ArtELingo caption
   licence in Fig. 2, the 8-page limit, hidden notes, consistent terms ("plain embedding" has three meanings,
   "encoder" and "backbone"). Our recommendation, given twice: run them now, report only; the policy re-read on a
   web-fetch agent, the checks on Codex if it has room, else a Sonnet `high` worker.

With the co-authors (the user): the title, "label-free", Fig. 4(a) (its labels "Plain image" and "Ours, emotion"
differ from Fig. 1's "Plain SigLIP" and "CoSiR emotion"), the limitations section, the overview caption length.
The user: edits of the abstract, the Fig. 1 caption and the conclusion; the OpenReview title and abstract; confirm
that Fig. 3 stays as it is.

After the deadline: the supplementary review (numbering, a pointer to Fig. S1, the Affection panel, the old teaser,
how the Fig. 1 query was chosen: Sec. S5 covers only Fig. 4 and Fig. S2).

## 5. Code and pitfalls

- PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.
- The user may edit REPO files in this container as well as in Overleaf: run `git status` before a pull. When the
  user edited here, build what is on disk and commit it as the user's, after their OK.
- Build `latexmk -pdf -interaction=nonstopmode -silent main.tex`, then `supp.tex` (it reads `main.aux`). Check the
  logs for "undefined" and "contains only floats". Float positions: `pdftotext -bbox-layout`; short last lines:
  `PF/tools/find_short_lines.py <bbox html>` (it reads the bbox HTML, not the PDF; table rows and the page-5 line
  "not the trunk weights." are known hits).
- `cvpr.sty` sets `\bottomfraction` and `\topfraction` to 0.99, so `[b]` holds a tall float.
- Paper prose: no semicolons between clauses, no dashes, British spelling, no win/level/lose words, no
  "significant", novelty only "to our knowledge". A resolved co-author note is commented out, never deleted.
- The handoffs live in MAIN, a public repo: no co-author names here.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing of ours in REPO, PF or MAIN's handoff folder. PF keeps the user's untracked
  `CBMI_IMP_CAMREADY (2).pdf` out of git.
- On disk over 1 GB: nothing. The scratchpad (page renders, a few MB) goes with the session.
- Repo heads: REPO `86aa696`, PF `72e0d77`, MAIN (after this cut's commit).
