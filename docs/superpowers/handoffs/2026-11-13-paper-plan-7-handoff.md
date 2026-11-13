# Handoff: paper plan 7: review the supplementary cluster-count figure on SigLIP 1 with the user, then the small items

## Brief
- Done: the 4.2 points after r29, "strongest of the three-pass comparators", the 4.1 genre clause, the 4.4 MMArt text, and one big Table 2 grouped by passes (REPO `de7731c`).
- State: Sections 4.1 to 4.4 have no open TODO left except the opening sentence; 4.5 and 4.6 have no text yet.
- Next: show the user the SigLIP 1 cluster-count figure (built 2026-11-12) and settle the CLAY style panel point.
- Blocked: nothing.

> Written 2026-11-13 17:37 by the "paper plan 6" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens). Direction log entry: none (paper work; decisions are rows in the paper folder's README).

## 1. Read in this order

1. This handoff, all of it.
2. Project memory `feedback_user-writes-paper.md`: the user writes the paper. Agents word only the user's points.
3. `docs/paper/cvpr2027_aspect_spaces/notes/paper_plan.md`: every paper item and its status.
4. The README rows of 2026-11-13 in `docs/paper/cvpr2027_aspect_spaces/README.md` (the last rows of its table).
5. For item 1 of section 4: `docs/paper/cvpr2027_aspect_spaces/notes/supplementary_items.md`, the "Range of cluster
   counts" row, and the figure folder `src/test/20261111_cluster_count_range/figures/siglip1/` (`values.md`).

Skills: none.

## 2. Decided by the user, not to be reopened

All on 2026-11-13, each a README row:
- 4.2 PercepT sentence: "On the same encoders, with Qwen3 in training and only SigLIP 1 at test time, ..." (user,
  after seeing the recommendation). r29's numbers stay out of the text.
- 4.2: "SigLIP 1 by itself is already good on style." One sentence, no comparison, no number (user's own pick).
- Table 3 (and now Table 2's retrieval half) reads each aspect's own embedding; the joined three-head vector is not
  mentioned (user, after seeing the recommendation).
- 4.2: the instructed Qwen3 is "the strongest of the comparators with three passes per image" (user, option c).
- 4.1 genre clause: "without an extra term, the genre head loses most of its gain in genre retrieval over the plain
  embedding" (user, option a).
- 4.4 MMArt paragraph from three points (transfer test; best in every column, AMI above the instructed Qwen3 on all
  three aspects; order of the other methods close to ArtELingo's). Nothing more runs for MMArt before 11-16.
- Table layout: Table 2 = one single-column table, Clustering (AMI) and Retrieval (mAP, image to image), rows grouped
  by passes per test image (one pass, three passes, no image pass, then CoSiR); Table 3 = the three text directions
  (user, from three previews).
- Earlier: default model Qwen3 text + SigLIP 1 image; Table 5 on SigLIP 1; Figures 3 and 4 picks (README rows).

## 3. Where things stand

- **4.2 to 4.4:** written on SigLIP 1; traces in `notes/trace/4_experiments.md` (last five sections).
- **Table 2 (merged):** CoSiR 0.099 / 0.441 / 0.528 AMI and 0.269 / 0.355 / 0.623 image mAP, bold in every column.
  Closest: CLAY on style (AMI 0.437, CoSiR above with 4 of 5 seeds; mAP 0.351, 5 of 5), the instructed Qwen3 on
  emotion mAP (0.268, "level"). Plain Qwen3 now has image mAP 0.248 / 0.192 / 0.536 (same as Table 6).
- **Page budget:** body text ends on page 7, but Figures 3 and 4 float to page 8 and Tables 6 to 8 to page 9. The
  content runs about one page over the 8-page limit, before the introduction and conclusion are complete.
- **Supplementary cluster-count figure:** the SigLIP 1 version exists (`figures/siglip1/supp_cluster_counts.pdf`,
  main `c54869a1`, 12 lines). Ours is the top line at 11 of 13 counts; level at style 10 (CLAY 0.4095, ours 0.4094)
  and genre 20 (ours 0.5065, instructed Qwen3 0.5062). Its rows were not yet checked against the filled Table 2.
- **Deadlines:** paper 2026-11-16 23:59 AoE (2026-11-17 12:59 in Amsterdam); supplementary 2026-11-23.

## 4. Open points for the next step

Key (ask the user which comes first; item 1 is the likely start):
1. Supplementary cluster-count figure: check its 12 lines against the new Table 2 rows, show it, and ask about CLAY
   under ours in the style panel (caption note or inset). Then into REPO `Sections/X_supplementary.tex` (switched
   off in `main.tex`), only after the user's yes.
2. Page budget: about one page over. The user decides what moves to the supplementary (candidates: Table 8, Table 7,
   one of Figures 3 and 4), best after the introduction and conclusion are written.
3. 4.5 (ablation) and 4.6 (qualitative) have headings but no text; the 4 opening sentence is a `\TODO`.

Minor (mention, do not edit; all the user's lines):
4. The introduction's `\enumerate{...}` never closes.
5. The comment beside `\input{Tables/cost}` ("This goes to supplemental material.") is out of date.
6. Table 8 still lists the Qwen3 row of CoSiR first.
7. Figure 1's left crossing: the user improves it later.
8. Table 6 caption keeps a `\TODO{either underline or just say here we are all best?}` (since REPO `7c34f00`, 2026-11-11).
9. PF `figures/captions.md` still holds the old Qwen3 caption drafts and traces of Figures 3 and 4. It is our file:
   update its trace part to the SigLIP 1 picks when convenient.

## 5. Code and pitfalls

- Pull REPO before each edit (Overleaf). After each push, tell the user Overleaf needs a pull. Stage only our lines.
- No agent notes, TRACE blocks or comments in REPO `.tex` files. Traces go in `notes/trace/<section>.md`. A `\TODO`
  or `\todo` goes in only when the user asks for it.
- Never name or link the public GitHub repo in paper material. No semicolons between clauses. Write "test set".
- Table check: `PF/tools/check_tables_vs_sheet.py` (CoSiR env): "groups checked 49 missing 0". It now looks up
  Table 3's sheet groups in both `main_retrieval.tex` and `main_clustering.tex` (the merged Table 2).
- Table 2 keeps the label `tab:clustering` and the file `Tables/main_clustering.tex`; it is `\footnotesize` (at
  `\small` it was 26 pt too wide).
- Build: `latexmk -pdf -interaction=nonstopmode -silent main.tex` in REPO. Two overfull boxes are old: Table 3
  (0.7 pt) and Table 8 (4.3 pt). Any new one is ours.
- PF = `docs/paper/cvpr2027_aspect_spaces/` (its own private repo), REPO = `PF/CVPR2027-CoSiR-Paper/`.

## 6. State at handoff

- Running: nothing of this chat.
- Uncommitted: nothing of ours. PF has the user's untracked `CBMI_IMP_CAMREADY (2).pdf`, kept out of git.
- On disk over 1 GB: nothing of this chat. The scratchpad (1.6 MB of page renders) goes with the session.
- Repo heads: REPO `de7731c`, PF `dd9a1f8`.
