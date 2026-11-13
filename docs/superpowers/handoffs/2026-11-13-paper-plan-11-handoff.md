# Handoff: paper plan 11: go through 4.6, 4.7 and the last captions of the Section 4 review with the user

## Brief
- Done: Section 4 review applied for 4.1 to 4.5 with the user (E1 fixed, read-out defined, supplementary seed table, Tables 3 and 7 captions). Page 1 now holds only the abstract and the introduction.
- State: no wrong number left. The body runs about 0.45 of a column onto page 9, before the conclusion is written.
- Next: show the user the 4.6 items (photos, Table 8 TODO), then 4.7, then the last three caption points.
- Blocked: nothing.

> Written 2026-11-13 20:06 by the "paper plan 10" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens). Direction log entry: none (paper work; decisions are rows in the paper folder's README).

## 1. Read in this order

1. This handoff, all of it.
2. Project memory `feedback_user-writes-paper.md` and global `~/.claude/rules/report-writing.md`, bullet "Results
   paragraphs in papers" (the user's three rules).
3. PF `notes/review/2026-11-13_section4_findings_sorted.md`, sections "4.6 Photos", "4.7 Qualitative", "Tables and
   supplementary" (our check and recommendation per finding; IDs R = facts check, C = cold read).
4. PF `notes/review/2026-11-13_section4_polish.md`, sections 4.6, 4.7 and "Captions" (wording proposals). Do not
   trust the polish blindly: its Table 7 speed-up sentence had the ratio inverted (caught and fixed here).
5. PF `README.md`, the last four rows (this chat's decisions).
6. REPO `Sections/4_experiments.tex` from `\subsection{Results on photos}`, `Tables/photos.tex`, `Tables/main_clustering.tex` caption.

Skills: none.

## 2. Decided by the user, not to be reopened

All on 2026-11-13, each a README row in PF:
- E1: the ablation arm without instructed graphs is "one graph on the plain Qwen3 image embedding" (4.5 and Table 5
  caption); "lower than that of the plain SigLIP 1 embedding that CoSiR reads"; "not from a graph on the plain embedding".
- 4.1: encoder pick "by a small margin over SigLIP 2"; read-out defined as a retrieval setting; Table 4 caption; polish
  rows a to g; supplementary section "Spread over random seeds" (`tab:supp_seeds`) with a pointer in 4.1.
- 4.2 and 4.3: comparators named, content-correction sentence moved, Table 3 text-to-text backbone "Qwen3" and caption.
- 4.4 kept as is ("best result in every column", tie at 0.202). Encoder line "possibly because". Table 7 caption.
- Table 6 marks (with the user's TODO) and the Table 5 caption stay as they are; the points are in PF
  `notes/review/2026-11-13_section4_memo.md` ("a short memo", the user's words; a file, not a note in the paper).
- `main.tex`: page 1 for the abstract and introduction only (user), done with `\clearpage` (REPO `eed6796`).

## 3. Where things stand

- REPO `eed6796`, all built clean (only the old 0.7 pt overfull box); table check "groups checked 49 missing 0".
- Length: 11 pages; the body ends in the left column of page 9 at about y 342 of 720 pt (4.7 and the conclusion
  heading), so about 0.45 of a column over 8 pages. The conclusion is not written yet. The user's section 4 cut plan is
  PF `notes/section4_cuts.md` (README row "Section 4 cut in advance").
- Still not written: the Section 4 opening, introduction, abstract, conclusion (all the user's).

## 4. Open points for the next step

Key, in this order, each with our recommendation, one subsection at a time (apply, build, send a crop):
1. **4.6 Photos:** S1/C19 drop "while the instructed Qwen3 embedding reaches 0.155" (four number sets); C20 EmoSet "on
   both backbones, on SigLIP 1 from 0.394 to 0.432"; C18 the photo fine-tuning recipe is said nowhere in the paper:
   do a facts lookup first (MAIN `src/test/20261031_photo_finetune/`, r20 and r28 photo records), then propose one
   sentence; polish lines 131, 132 (comma), 137 ("\method learns"); Table 8 caption polish (comma, split). C21 (EmoSet
   validation picks the read-out) is checked and needs no change.
2. **Table 8 TODO and R7** (three-decimal ties in bold; CLIP L/14 Affection AMI is the larger unrounded): the user kept
   Table 6 as it is with a memo, so ask whether Table 8 gets the same treatment (add it to the memo).
3. **4.7:** R8/C23 "In these examples, each aspect embedding orders ..."; R9/C24 "illustrates" for "shows"; C22 and
   polish line 146 ("of that painting's neighbours in style"); C30 Figure 4 caption "(the percentage of its paintings
   with that class)" (one definition in code, `src/test/20261109_r20_figures/r20_cand.py:160`); C29 Figure 3(b) labels
   "A / B" are the left and right painting's (check the figure before wording); polish 145, 152, Figure 3 "check mark".
4. **Captions:** C27 Table 2 BERTopic sentence and the polish TURTLE phrase; S4 supplementary figure caption "CLAY is
   level with \method" (banned word) to "reaches the AMI of \method".

Minor (mention, do not edit): the polish's cross-sentence points (PF polish file, end): "plain embedding" has three
referents across Sections 3 and 4 (how E1 arose); "encoder" against "backbone"; "SigLIP~1" with a tie once. PF
`figures/captions.md` still holds old Qwen3 caption drafts (our file). The `% This goes to supplemental material.`
comment beside Table 7 is the user's.

## 5. Code and pitfalls

- Pull REPO before each edit (Overleaf). After each push, tell the user Overleaf needs a pull. Stage only our lines.
- Apply accepted wording with a small Python script that asserts each old string occurs once (fast and exact).
- No agent notes, TRACE blocks or comments in REPO `.tex` files. A `\TODO` only when the user asks for it.
- No semicolons between clauses, no dashes, British spelling, "test set", no win/level/lose words.
- Build: `latexmk -pdf -interaction=nonstopmode -silent main.tex` in REPO. Supplementary: scratch copy with the
  `% \input{Sections/X_supplementary}` line uncommented.
- Crops: `pdftoppm -r 150`, positions from `pdftotext -bbox` (points times 150/72), PIL crop, SendUserFile.
- Table check: `PF/tools/check_tables_vs_sheet.py` (CoSiR env).
- In two-column mode, two `\newpage` in a row end only one column; use `\clearpage` for a new page.
- PF = `docs/paper/cvpr2027_aspect_spaces/` (own private repo), REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.

## 6. State at handoff

- Running: nothing of this chat. Other tabs' r31 runs may use the queue.
- Uncommitted: nothing of ours. PF has the user's untracked `CBMI_IMP_CAMREADY (2).pdf`, kept out of git.
- On disk over 1 GB: nothing. The scratchpad (crops, build copies) goes with the session.
- Repo heads: REPO `eed6796`, PF `469502a`.
