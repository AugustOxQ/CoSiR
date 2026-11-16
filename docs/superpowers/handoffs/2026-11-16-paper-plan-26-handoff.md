# Handoff: point ii, the cold read and fact check of the final text; then the abstract and the reminder for iv

## Brief
- Done: points b ("intra-modal" and "cross-modal" retrieval), c (Fig. 1 query labels), d (Fig. 3 simplified) and the new point f ("seed" only in the setting).
- State: REPO `8267a7e` pushed; 8 pages + references, page 8 ends at line 553, 7 free lines; every float in place.
- Next: point ii, a cold read and fact check of the final text (numbers and claims must not contradict each other).
- Blocked: nothing; paper due 2026-11-17 12:59 Amsterdam.

> Written 2026-11-16 17:14 by the "paper plan 25" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens), at the user's yes. Direction log entry: none (paper work; decisions are rows in the paper folder's
> notes and the status table).

## 1. Read in this order

1. This handoff, all of it.
2. PF `notes/2026-11-16_final_list.md`: the user's list for the last day. Rows a to f and i, iii are done with their
   REPO commit; ii and iv open. It also holds the two abstract versions (A recommended).
3. PF `user_read/2026-11-13_paper_status.md`: the status table the user reads; keep it current.
4. PF `notes/review/` (the sorted findings of 16 November) and `notes/review/2026-11-16_refcheck_sorted.md`: what the
   earlier checks covered, so the cold read does not repeat them blindly.
5. Project memory `feedback_user-writes-paper.md`: the user writes the paper; agents propose exact text only.

Skills: none.

## 2. Decided by the user, not to be reopened

All on 2026-11-16, recorded in the final list and the notes files named there:
- **Order:** ii (cold read and fact check, now that the text is final), then the abstract (A or B), then the reminder
  for iv. iv is the user's.
- **Each change:** propose the exact text, apply only after the user's yes, build, check the float pages, commit and
  push; the user pulls into Overleaf.
- **Point b** (`notes/2026-11-16_point_b.md`): no "direction" in the paper. Retrieval is "intra-modal" or
  "cross-modal", defined at test time (Sec. 3.3); the loss "averages two intra-modal and two cross-modal terms".
- **Point c:** Fig. 1 query labels "Emotion:", "Style:", "Genre:" bold, capitalised, aligned at the colon; query panel
  1.00 in, labels 7 pt (option A).
- **Point d** (`notes/2026-11-16_point_d.md`): no pictograms and no toy with made-up real paintings. Fig. 3 is the old
  figure simplified: (a) the scores of painting i in words ("instructed score − λ_i × content similarity = corrected
  score", "here λ_i = 0.5" under the operator), (b) the mutual k-nearest-neighbour rule (the user: important to show);
  no note lines. Script `figures/src/fig2_graphs_v4.py`.
- **Point f:** "seed" appears in the main paper only in the experiment setting (`4_experiments.tex` :53, :60, :62).
  The supplementary may name seeds but no seed number ("one of the five trained models").
- Still in force: limitations go to the supplementary (due 2026-11-24); review findings stay in `PF/notes/review/`,
  not in the PDF; no notes of ours in REPO `.tex` files; the title stays for now; LLM Usage Statement: the user writes
  it, agents do not read or comment on it, they only remind the user if the last version has none.

## 3. Where things stand

- REPO `8267a7e` = origin; PF `90071a2` or later = origin. `main.pdf` 11 pages (8 + references), no undefined
  reference, no overfull box; `supp.pdf` 3 pages.
- Float pages: p1 Fig. 1; p3 Figs. 2, 3 (Fig. 3 now at the bottom of the right column, 2.30 in high); p6 Tables 1 to 3;
  p7 Tables 4 to 6; p8 Fig. 4, Table 7. Page 8 ends at line 553 with 7 free lines (557 at the start of this chat).
- Claims changed today that the fact check should cover: the retrieval paragraph of Sec. 4.2 (now "In Tab. 3, …",
  "keeps the higher mAP, most in text-to-text retrieval"); the clustering and image-to-image sentences without "with
  every seed" (now claims about the mean over five models); the intro sentence "on all aspects"; the Fig. 3 caption
  (toy values, k = 2).
- The abstract versions A and B in the final list do not mention seeds or directions; recheck them against the final
  text before proposing one.

## 4. Open points for the next step

Key, in order:
1. **Point ii:** a cold read and fact check of the final main text and the supplementary: numbers against the tables
   and their sources, claims against each other, terms used one way. Propose each fix as exact text.
2. **The abstract:** A or B, both re-checked against the final text.
3. **The reminder for iv:** only if the last version has no LLM Usage Statement (see section 6; do not read it).

Minor, with the co-authors (the user): the final title, Fig. 4(a), table row names ("Instructed Qwen3" against
"Instructed embedding", finding C9). Optional from the reference check: R7 soft notes.

## 5. Code and pitfalls

- PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.
- **REPO `main.tex` has the user's own uncommitted edit.** Leave it alone and stage files by name, never `git add -A`.
- **Test each change in a scratch copy first:** `rsync -a --exclude .git REPO/ <scratch>/x/`, apply a patch script
  that asserts each old string occurs once, build, then check the float pages, the ruler line of "future work." on
  page 8 and `grep Overfull main.log`. This chat's `measure.sh` does all three: copy it from
  `/tmp/claude-0/-project-CoSiR/4304b2f6-a704-442b-a403-395e83bacd60/scratchpad/b/measure.sh` (may be gone; it is a
  `latexmk` build plus `pdftotext` greps). The ruler number is the last field of the "future work." line.
- **A short bottom float can fall into the left column and leave a gap above it** (seen with Fig. 3): look at the page
  image, not only the float list; moving the `figure` block later fixed it.
- **Bottom floats slip to the next page** when text above their `\input` grows (Tables 3, 6 and 7, Fig. 3 are `[b]`).
- **refs.bib:** edit PF's copy (with `% verified:` lines), then `grep -v '^\s*%' PF/refs.bib | cat -s > REPO/refs.bib`.
- Number pairs in paper prose: "from 0.402 to 0.528" or "it reaches 0.441 and CLAY 0.437", never "X against Y".
- The main paper cannot `\cref` labels in `supp.tex`; it points with "see the supplementary material".
- Build `latexmk -pdf -interaction=nonstopmode -silent main.tex`, then `supp.tex` (it reads `main.aux`).
- Paper prose: no semicolons between clauses, no dashes, British spelling, no win/level/lose words, no
  "significant", novelty only "to our knowledge". A resolved note or TODO is commented out and ends with "[done]".
- Before a figure is redrawn, ask whether real data or a toy is wanted: the user rejected a pictogram toy today.
- The handoffs live in MAIN, a public repo: no co-author names here.
- Get times with `TZ=Europe/Amsterdam date` before writing them.

## 6. State at handoff

- Running: nothing.
- Uncommitted: REPO `main.tex` (the user's own edit, which holds their LLM Usage Statement section; not ours, not to be
  read). PF keeps the user's untracked `CBMI_IMP_CAMREADY (2).pdf` out of git. Nothing of ours.
- On disk over 1 GB: nothing. This chat's scratchpad holds 269 MB of scratch builds and sketches
  (`/tmp/claude-0/-project-CoSiR/4304b2f6-a704-442b-a403-395e83bacd60/scratchpad/`); everything needed is in git. The
  user decides on it.
