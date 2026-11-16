# Handoff: paper plan 19: the abstract (about 150 words), the Fig. 1 caption and the conclusion (at most 100 words), then the supplementary review

## Brief
- Done: point 8 decided (Affection candidate 12, supplementary maybe); page budget measured; Fig. 5 to the supplementary, which now builds from its own `supp.tex`; 22 short last lines removed.
- State: REPO `2e31292` pushed; the text ends in page 8's right column, Conclusion heading at y 249 (was 434); page 1 has about 6 free lines for the abstract.
- Next: the abstract, the Fig. 1 caption and the conclusion; then the supplementary review.
- Blocked: nothing; the user pulls REPO into Overleaf.

> Written 2026-11-16 04:01 by the "paper plan 18" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens), at the user's yes. Direction log entry: none (paper work; decisions are rows in the paper folder's
> README).

## 1. Read in this order

1. This handoff, all of it.
2. PF `README.md`, the rows dated 2026-11-16 from "Page budget measured" on (this chat's decisions), and rows 147
   and 150 (the abstract's three open questions).
3. PF `submission/registration.md`: the registered abstract (about 250 words). The user: it was a placeholder for
   the registration; the final one is about 150 words.
4. Project memory `feedback_user-writes-paper.md`: the user writes the paper; agents guide, check facts, and write
   only a part the user has set, one part at a time.
5. PF `notes/2026-11-16_short_lines.md` only when a short-line question comes up.

Skills: none.

## 2. Decided by the user, not to be reopened

All on 2026-11-16, each a README row in PF:
- Point 8: Affection candidate 12 (query a chimp, fear; plain SigLIP finds other apes, CoSiR finds snarling big
  cats, 4 of 5), candidate 3 (dog, amusement) the backup. Place: the supplementary, maybe. A citation of each source
  dataset is enough for the photos.
- Fig. 5 (k-means clusters) and its Sec. 4.6 paragraph are in the supplementary (new Sec. S4 "More qualitative
  examples", Fig. S2); Sec. 4.6 keeps one pointer sentence.
- The abstract is about 150 words; the conclusion at most 100 words.
- All 22 short-line edits (list in PF `notes/2026-11-16_short_lines.md`).
- Order: the abstract, the Fig. 1 caption and the conclusion first; the supplementary review after them.

## 3. Where things stand

- REPO `2e31292` = origin. `main.pdf` 11 pages (8 + references), builds clean (only the old 0.7 pt overfull box);
  `supp.pdf` 3 pages. Fig. 4 sits at the top of page 8; the Conclusion heading is at y 249 in page 8's right column,
  with nothing under it.
- Page budget, measured in scratch copies before the 22 edits (a 150-word placeholder abstract, a 100-word conclusion,
  the `\clearpage` after the introduction removed): with Fig. 5 in the supplementary, 162 pt (13 lines) over 8 pages,
  6 of those lines the Fig. 5 paragraph, which has since moved. The 22 edits then saved about 15 lines on pages 2 to 8
  and 4 on page 1. Estimate, not measured: the paper fits with about 10 lines to spare. Measure again once the
  abstract and conclusion are in.
- Page 1 holds only the abstract and the introduction (`main.tex` has a `\clearpage` after the introduction). Page 1
  has about 6 free lines; a 150-word abstract needs about 16, so the introduction spills about 10 lines onto page 2.
  The `\clearpage` must go when the abstract is in, or Related Work jumps to page 3.
- Paper deadline 2026-11-17 12:59 Amsterdam.

## 4. Open points for the next step

Key, in the user's order:
1. **The abstract**, about 150 words. Its other two questions are open (README row 150): who sets the skeleton, and
   which results it names. The user writes it; agents give facts and numbers.
2. **The Fig. 1 (teaser) caption.**
3. **The conclusion**, at most 100 words.
4. Then remove the `\clearpage`, build, and measure the page count again; check short last lines with PF
   `tools/find_short_lines.py`.

Then the supplementary review (the user's "later"):
- Numbering S1, S2, ... and the separate `supp.tex` are agent defaults; the user may prefer letters or numbers that
  go on from the paper.
- Fig. S1 (AMI at several cluster counts) has no sentence that points to it (older gap).
- The Affection panel, if it goes in: candidate 12 from MAIN `src/test/20261116_affection_examples/`
  (`make_candidates.py` has the selection rule; seed 0). A panel in the style of Fig. 4(a) is still to be built.
  Sources of its 11 photos: 9 Emotional Machines, 1 You et al. (AAAI 2016), 1 Visual Genome; cite each source
  dataset and Affection. Which paper "Emotional Machines" is was not checked.

Minor, carried: the red TODO lines at the start of Sec. 4 (an opening sentence); a limitations paragraph; the LLM
Usage Statement (the user writes it, agent lists our uses); the CVPR 2027 LLM policy re-read; the title; whether
Fig. 4(a) stays (it is like the teaser); the old teaser to the supplementary; "label-free" with the co-authors; the
overview's remaining issues ("we discuss later"); the checks before upload; the status table (PF
`user_read/2026-11-13_paper_status.md`), last updated after point 5.

## 5. Code and pitfalls

- PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.
- Before each REPO edit: `git pull --ff-only`. Edit with an assert script, build with
  `latexmk -pdf -interaction=nonstopmode -silent main.tex`, then `supp.tex` (it reads `main.aux` through xr-hyper,
  so main comes first). Check both logs for "undefined" and "contains only floats", find positions with
  `pdftotext -bbox-layout`, render with `pdftoppm -r 150`, send with SendUserFile. Commit and push REPO after the
  user's OK, then add the README row in PF and push PF.
- Page-budget measure: copy REPO to the scratchpad (without `.git`), put placeholder text in
  `Sections/0_abstract.tex` and `5_conclusion.tex` (end it with a marker word), comment out the `\clearpage`, build,
  and read the marker's page and y. Fig. 4 must be defined on page 7 to land at the top of page 8.
- Short-line edits: a word cut in a paragraph saves a line only when it removes more than the short line's width;
  test each in a scratch build. `\looseness=-1` alone saved no line here.
- Paper prose: no semicolons between clauses, no dashes, British spelling, no win/level/lose words, no
  "significant". One name for one thing. A resolved co-author note is commented out on its own line, never deleted.
- The handoffs live in MAIN, a public repo: no names of co-authors there; the PF notes are private.
- Photos in a figure: the paintings figures say "public-domain"; Affection photos get a citation of each source
  dataset.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing of ours in REPO or PF. PF has the user's untracked `CBMI_IMP_CAMREADY (2).pdf`, kept out of
  git. MAIN `src/test/20261116_affection_examples/contact_sheet.png` and `preview.png` (third-party photos) stay
  untracked on purpose.
- On disk over 1 GB: nothing. The scratchpad (7 MB of page renders) goes with the session.
- Repo heads: REPO `2e31292` (= origin), PF `3907e76`, MAIN (after this cut's commit).
