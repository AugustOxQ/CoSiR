# Handoff: paper plan 16: co-author point 5 first (the datasets paragraph of Sec. 4.1), then the rest of the list

## Brief
- Done: co-author points 2, 3, 1 and 7 (CLAY and CRL, retrieval directions, PercepT and the tasks, results by dataset).
- State: push freeze ended; REPO pushed to `cd9c4aa`; 5 of the co-author's 9 points done; builds clean, 12 pages.
- Next: point 5, the datasets paragraph: propose the wording, write it after the user's OK, build, show the PDF.
- Blocked: the user pulls REPO into Overleaf (asked, not confirmed).

> Written 2026-11-16 01:45 by the "paper plan 15" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens), at the user's yes. Direction log entry: none (paper work; decisions are rows in the paper folder's
> README).

## 1. Read in this order

1. This handoff, all of it.
2. PF `notes/2026-11-15_coauthor_comments.md`: the nine points in the user's order and their status. The work list.
3. PF `user_read/2026-11-13_paper_status.md`: the user's status table (21 rows plus the nine points), updated
   2026-11-16. Keep it current when a row changes; the user reads it.
4. PF `README.md`, the rows dated 2026-11-16.
5. REPO `Sections/4_experiments.tex` (Sec. 4.1 "Datasets." and the co-author's red `\todo` at its top).
6. Project memory `feedback_user-writes-paper.md` (how a part is written, the tone).
7. `docs/superpowers/handoffs/2026-11-15-paper-plan-14-handoff.md`, section 5 (build, crops, style rules), which
   still holds, except the push freeze, which has ended.

Skills: none.

## 2. Decided by the user, not to be reopened

All on 2026-11-16, each a README row in PF:
- Point 2: intro paragraph 2 ends "CLAY and CRL encode an image once for all aspects, but they only modulate or project
  its frozen embedding with texts about each aspect." Grammar fix in the co-author's paragraph-3 sentence.
- Point 3: Sec. 4.2's Retrieval part opens "The query and the results can each be an image or a caption, which gives
  four directions (Sec. 3.3)." and says which table holds which direction.
- Point 1: intro paragraph 4 opens with what PercepT is; then "Like PercepT, we score clusters against these labels,
  but we cluster the paintings of a test split, once per aspect."; retrieval "counts as correct when it shares the
  label of the query".
- Point 7: Sec. 4 by dataset: 4.2 Results on ArtELingo (run-in Clustering. and Retrieval.), 4.3 MMArt, 4.4 Results
  on Affection and EmoSet, 4.5 Ablation and analysis, 4.6 Qualitative examples. Headings only.
- The push freeze ended ("No edits on overleaf now, free to push"). Commit and push REPO as usual again, after the
  user's OK of each change.
- Still from 2026-11-15: "label-free" (the co-author's word) stays for now, a later talk with the co-authors.

## 3. Where things stand

- REPO `cd9c4aa` = origin. Builds clean, 12 pages; only the old 0.7 pt overfull box.
- Table numbers after the reorder: photos 5, time per image 6, ablation 7, other encoders 8.
- Length: the text ends at about y 241 pt in the left column of page 9; Figures 4 and 5 sit on page 10, after the
  references start. A test build without the time-per-image table ended the text on page 8 (the user's own tex comment
  says that table goes to the supplementary). Not yet raised as a decision: it belongs to the length cut.
- Abstract and conclusion still empty; teaser caption empty.
- Paper deadline 2026-11-17 12:59 Amsterdam. The user updates the OpenReview title and abstract.
- Not checked: Overleaf's comment panel and the co-author's email (not in git). A second co-author may still send
  points; the user said "maybe some points coming later".

## 4. Open points for the next step

Key:
- Point 5 (the datasets paragraph): the co-author's note says it reads as if we only train on ArtELingo; list all
  datasets first, then how each is used. Datasets: ArtELingo (train, validation, test), MMArt (test only, 16,384
  paintings), Affection (photo fine-tuning on its unlabelled training photos and explanations, plus test), EmoSet
  (test only). Propose the wording; the user decides. Remove the `\todo` once done.

Then, in the user's order: point 6 (overview figure size, captions over 100 words, with the length cut), point 4 (the
related-work opening sentence, the user writes it), point 8 (Affection examples, only if the fine-tuned photo model's
test embeddings exist).

Minor, carried: the abstract's three questions; the teaser caption; a limitations paragraph; the LLM Usage Statement
(the user writes it, agent lists our uses); the CVPR 2027 LLM policy re-read (planned for 16 November); the title; whether
Fig. 4(a) stays; the old teaser to the supplementary; the checks before upload.

## 5. Code and pitfalls

- PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.
- Before each REPO edit: `git fetch` and pull (Overleaf may have pushed). Edit with an assert script, build with
  `latexmk -pdf -interaction=nonstopmode -silent main.tex`, check `main.log` for "undefined" and "contains only
  floats", render with `pdftoppm`, send with SendUserFile.
- Float trap: when the time-per-image table was read at the end of the ablation subsection, LaTeX left page 9's left
  column blank ("Text page 8 contains only floats" in the log). Its `\input` now sits right after the photos table.
  Watch that warning after any change in Sec. 4.
- `PF/tools/reorder_sec4_by_dataset.py` did the point-7 move. It does nothing on a file that is already reordered.
- The handoffs live in MAIN, a public repo: no names of co-authors there; the PF notes are private.

## 6. State at handoff

- Running: nothing of this chat.
- Uncommitted: nothing of ours. PF has the user's untracked `CBMI_IMP_CAMREADY (2).pdf`, kept out of git.
- On disk over 1 GB: nothing. The scratchpad (page renders, a scratch copy of the paper, a few MB) goes with the
  session.
- Repo heads: REPO `cd9c4aa` (= origin), PF `5a592d5`, MAIN (after this cut's commit).
