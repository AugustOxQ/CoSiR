# Handoff: paper plan 15: work through the co-author's comments in order, starting with the small text fixes

## Brief
- Done: introduction paragraphs 4 and 5; bold run-in headings; Sec. 3.1 title; "SigLIP" for SigLIP 1 (item 9).
- State: a co-author's first read gave nine points; their Overleaf edits are accepted for now; REPO push freeze.
- Next: points 2, 3 and 1 (CLAY and CRL pass, the four directions, PercepT and the tasks), then the results order.
- Blocked: pushing REPO waits until the user says the Overleaf tracked changes and comments are resolved.

> Written 2026-11-15 22:45 by the "paper plan 14" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens), at the user's yes. Direction log entry: none (paper work; decisions are rows in the paper folder's
> README).

## 1. Read in this order

1. This handoff, all of it.
2. PF `notes/2026-11-15_coauthor_comments.md`: the sync rule, the accepted edits, the nine points in the user's order,
   and their status. This is the work list.
3. PF `README.md`, the rows dated 2026-11-15 from "Introduction paragraph 4" on (this chat's).
4. REPO `Sections/1_introduction.tex` (now with the co-author's paragraph 3) and `Sections/4_experiments.tex`.
5. Project memory `feedback_user-writes-paper.md`: how a part is written (one part at a time into the tex, build,
   show the PDF) and the tone (the user's CBMI paper).
6. `docs/superpowers/handoffs/2026-11-15-paper-plan-14-handoff.md`, section 5 (code and pitfalls), which still holds.

Skills: none.

## 2. Decided by the user, not to be reopened

All on 2026-11-15, each a README row in PF:
- Paragraph 4: number sets N3, N1 and N2; no MMArt and photos clause. Sentence 1 starts "Like PercepT, we score
  clusters against the emotion and genre labels of ArtELingo, but we cluster the paintings of a test split." No
  "held-out" in the paper. `2_related_work.tex:17` aligned the same way.
- Paragraph 5: "Our contributions are as follows:", bold heads, no section references, no numbers.
- `\paragraph` prints as flush-left bold text with no space above (one line in REPO `preamble.tex`).
- Sec. 3.1 is titled "Instructed neighbour graphs".
- The co-author's comments are done in the order of the comments note (a to f).
- Push freeze on REPO until the user says Overleaf is clean. Local commits only meanwhile.
- The co-author's Overleaf edits are accepted for now, "label-free" included. "Label-free" is a discussion point for
  later (the user does not want it). Do not revert those edits on our own.
- "SigLIP" for SigLIP 1 everywhere; SigLIP 2 keeps its number.

## 3. Where things stand

- REPO: `e426bc3` (SigLIP naming) is committed locally, one commit ahead of origin, not pushed. Origin is at
  `a0b62a0` (the co-author's sync). Builds clean, 12 pages; only the old 0.7 pt overfull box.
- The `\paragraph` change moved the Conclusion up about 206 pt on page 9 (about 17 lines). The length cut (about 1.7
  to 2 columns before that change) is still to do.
- Introduction: 5 paragraphs, about 510 words before the co-author's edits; ends at the bottom of page 1 while the
  abstract is empty. The teaser sits at the top of the right column.
- The co-author's notes in the tex use `\nvn{...}` (teal) and `\todo{...}` (red), both defined in `preamble.tex`.
- The abstract is still empty. Its three questions (who sets the skeleton, about 120, 150 or 200 words, which results)
  wait: the user said "a bit later".
- Paper deadline: 2026-11-17 12:59 Amsterdam. Supplementary: 2026-11-24 12:59. OpenReview's registered abstract still
  needs the update before the paper deadline (the user does that).

## 4. Open points for the next step

Key, at points 2, 3 and 1: the wording of each fix. Propose it, write it into the tex after the user's OK, build, show
the PDF (as for the introduction). Point 1 changes the introduction's paragraph 4, which is now above 100 words.

Minor:
- The grammar errors in the co-author's sentence ("an image or text embeddings for every aspects"): propose the fix at
  point b. It does not revert their edit.
- When the freeze ends: pull Overleaf's sync, merge `e426bc3` and later local commits, build, push, ask the user to
  pull in Overleaf.
- Carried from plan 14, none answered: a limitations paragraph; the LLM Usage Statement (the user writes it); whether
  Fig. 3(a) stays; the title (with the co-authors); the teaser caption (the user's).

## 5. Code and pitfalls

- PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.
- During the freeze: never `git push` in REPO. PF and MAIN push as usual (they are not synced with Overleaf).
- Overleaf comments are not in git. Ask the user to paste them if the next step needs them.
- The handoffs live in MAIN, a public repo: no names of co-authors there; the PF notes are private.
- The teaser source is `PF/figures/src/fig_teaser_t1.py a10`; copy `PF/figures/fig_teaser_t1_a10.pdf` to REPO
  `Figures/fig_teaser.pdf`.
- Section 5 of the plan-14 handoff still holds (assert scripts, build command, crops, style rules, no notes in `.tex`).

## 6. State at handoff

- Running: nothing of this chat.
- Uncommitted: nothing of ours. REPO has one local commit, `e426bc3`, not pushed (the freeze). PF has the user's
  untracked `CBMI_IMP_CAMREADY (2).pdf`, kept out of git.
- On disk over 1 GB: nothing. The scratchpad (page renders, a few MB) goes with the session.
- Repo heads: REPO local `e426bc3` (origin `a0b62a0`), PF `46a4a61`, MAIN (after this cut's commit).
