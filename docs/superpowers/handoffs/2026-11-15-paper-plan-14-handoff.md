# Handoff: paper plan 14: write introduction paragraphs 4 and 5 with the user, one at a time

## Brief
- Done: introduction paragraphs 1 to 3 written and approved (REPO `53e04d7`). Overview figure toy draft built; the user holds Figures 2 and 3 until after the co-authors' read.
- State: five-paragraph introduction; paragraphs 4 (evaluation and results) and 5 (contributions) are still the user's skeleton list.
- Next: write paragraph 4 from the layout note, build, show the user; then paragraph 5.
- Blocked: nothing.

> Written 2026-11-15 04:15 by the "paper plan 13" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens), at the user's yes. Direction log entry: none (paper work; decisions are rows in the paper folder's
> README).

## 1. Read in this order

1. This handoff, all of it.
2. PF `README.md`, the rows dated 2026-11-15 (the last eight are this chat's).
3. PF `notes/2026-11-14_intro_layout.md`: the "Updated" line at the top, then Section 5, paragraphs 4 and 5, and
   Section 6 (contributions without numbers, the number sets N1 to N3).
4. REPO `Sections/1_introduction.tex` and PF `notes/trace/1_introduction.md` (the source of each sentence so far).
5. Project memory `feedback_user-writes-paper.md`: how a paragraph is written (one part at a time into the tex, build,
   show the PDF) and the tone (the user's CBMI paper, `PF/CBMI_IMP_CAMREADY.pdf`, introduction only).
6. `docs/superpowers/handoffs/2026-11-15-paper-plan-13-handoff.md`, sections 4 and 5: the minor open points and the
   pitfalls, which still hold.

Skills: none.

## 2. Decided by the user, not to be reopened

All on 2026-11-15, each a README row in PF:
- Five paragraphs: (1) background and the teaser, (2) the gap, (3) CoSiR and its two parts, (4) the evaluation and its
  key results, (5) the contributions.
- All the introduction's numbers go in paragraph 4. The contribution items carry none. C3 stays as a short third item
  (what is new in the evaluation), without its results.
- Paragraph 3 leads with the instructed neighbour graphs as the training signal. The content correction is one
  supporting sentence after it. The user found the first draft ("Two problems arise. First, content…") out of focus.
- "Trained without aspect labels", never "label-free". "The embedding model" in the introduction, not "the MLLM".
- Paragraph 1 cites Tversky (checked) and CSN. Paragraph 2 says "which favours the content of an image" with CSD
  (checked in its PDF) and keeps the CLAY and CRL sentence.
- The teaser's message, for the caption that the user writes: plain SigLIP 1 finds 0 of 4 with the query's emotion,
  CoSiR 4 of 4 per aspect.
- Figures 2 and 3 (the overview and the neighbour-graph figure) are held until after the co-authors read the current
  version (Monday 2026-11-16). The user found the toy draft "pretty weird". The caption key for solid and dashed lines
  waits too.

## 3. Where things stand

- REPO `53e04d7` builds clean, with only the old 0.7 pt overfull box; 12 pages, Conclusion on page 9 (position not
  measured in this chat).
- Introduction so far: paragraph 1 about 65 words (lines 002 to 008), paragraph 2 about 100 (009 to 021), paragraph 3
  about 150 (022 to 038). The left column of page 1 is full; the list continues in the right column under the teaser.
  The plan gives paragraphs 4 and 5 about 80 words each, so about 475 in all, above the 420 target and near the 460
  ceiling. The abstract is still empty.
- The skeleton list at the end of the section ("(graph+buddy)…", "(model)", "evaluation?") is the user's; paragraphs 4
  and 5 replace it.
- Overview toy draft: PF `figures/drafts/fig1_overview_toy.{png,pdf}`, source `figures/src/fig1_overview_toy.py`
  (docstring lists the changes). 205 pt tall against 243 pt now; towers swapped (image above caption), no crossings,
  drawings for the alignment term and the graph loss, "(Fig. 2)" label removed. Not in REPO. A compact version was
  not built. The designer's notes: start from the toy source; the test-time row cannot go beside the training rows (it
  needs about 5.9 in).
- Main-model risk, unchanged: the teaser and Figures 4 and 5 are built on c* SigLIP 1 (r28). Project memory says the
  SigLIP-only model (r29) stays an ablation row; not checked whether r31 or r32 changed that.
- Paper deadline: **2026-11-17 12:59 Amsterdam**. Supplementary: 2026-11-24 12:59.

## 4. Open points for the next step

Key, answer-first, at paragraph 4: which number sets go in (N3 and N1 recommended, N2 optional), and whether the
optional clause on MMArt and photos goes in. At paragraph 5: the lead-in phrase and the three items.

Minor, carried from plan 13, none answered: a limitations paragraph at the end of the conclusion; the LLM Usage
Statement (by Monday; the user writes it); whether Fig. 3(a) stays (at the length cut); the title (with the
co-authors); the teaser's place at the top of the right column; the teaser caption (the user's). The length cut of
about 1.7 to 2 columns is still to do.

## 5. Code and pitfalls

- PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.
- Per paragraph: pull REPO (stash local changes first), replace the skeleton text with an assert script, build with
  `latexmk -pdf -interaction=nonstopmode -silent main.tex`, crop page 1 (`pdftoppm -r 150`, positions from
  `pdftotext -bbox`, PIL crop), send it with SendUserFile, quote the text in the reply. Commit and push REPO only after
  the user's OK; then tell the user Overleaf needs a pull. Add the paragraph's sources to PF `notes/trace/1_introduction.md`.
- No notes in `.tex` files. No semicolons between clauses, no dashes, British spelling, no win/level/lose words
  ("improves on", "gains", "higher than"). No "significant".
- Number cautions for paragraph 4 are in the layout note: "with every seed" holds only against the instructed Qwen3;
  text retrieval on emotion favours the instructed Qwen3; the cost claim is per image; CoSiR on photos is fine-tuned on
  Affection photos.
- Figure routing still holds: design judgment Opus `xhigh`, look work Opus `high`; the controller opens every PNG
  first. A design worker filled its context in one run here, so give the next one a single variant.

## 6. State at handoff

- Running: nothing of this chat.
- Uncommitted: nothing of ours. PF has the user's untracked `CBMI_IMP_CAMREADY (2).pdf`, kept out of git.
- On disk over 1 GB: nothing. The scratchpad (page renders, the CSD PDF, figure crops, a few MB) goes with the session.
- Repo heads: REPO `53e04d7`, PF (after this cut's commit), MAIN (after this cut's commit).
