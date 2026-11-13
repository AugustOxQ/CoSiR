# Handoff: paper plan 8: go on cutting Section 4 in advance, sub-subsection by sub-subsection, starting at 4.2

## Brief
- Done: supplementary cluster-count figure (CLAY on top, caption) and all of 4.1's advance cuts (Datasets, Compared methods reordered, Metrics, Implementation details to the supplementary) (REPO `5fc3f01`).
- State: 4.1 is final for now. 4.2 to 4.4 not yet gone through. The Section 4 opening is the user's to write.
- Next: ask the user's view on 4.2 first, then compare with the agent's table in `notes/section4_cuts.md`.
- Blocked: nothing.

> Written 2026-11-13 18:09 by the "paper plan 7" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens). Direction log entry: none (paper work; decisions are rows in the paper folder's README).

## 1. Read in this order

1. This handoff, all of it.
2. Project memory `feedback_user-writes-paper.md`: the user writes the paper. Agents word only the user's points.
3. `docs/paper/cvpr2027_aspect_spaces/notes/section4_cuts.md`: the agent's first cut table (IDs a to j), the 4.1
   decisions and status, and the text taken out for the supplementary.
4. The README rows of 2026-11-13 in `docs/paper/cvpr2027_aspect_spaces/README.md` (the last six rows are this chat's).
5. REPO `Sections/4_experiments.tex` from `\subsection{Clustering experiments}` on, and `Tables/main_clustering.tex`,
   `Tables/main_retrieval.tex` (their captions).

Skills: none.

## 2. Decided by the user, not to be reopened

All on 2026-11-13, each a README row:
- How we cut: Section 4 is cut in advance, one sub-subsection at a time. The user gives their view first. The agent
  compares it with its table and gives a final suggestion. Each change is applied, compiled and shown (a crop) before
  the next. No line counts: the exact cuts wait until every section is final (intro, abstract and conclusion are not
  written yet).
- Supplementary cluster-count figure: CLAY drawn over ours, caption sentence "In the style panel, CLAY is level with
  CoSiR at 10 clusters and just under it at 20 and 27", in `Sections/X_supplementary.tex`.
- 4.1 Datasets: "We also test on MMArt", the no-retraining sentence removed.
- 4.1 Compared methods: reordered by Table 2's pass groups (one pass; then a second paragraph with three passes and
  BERTopic). VLM2Vec and BERTopic introduced by what they are, not as PercepT's baselines (user's own pick).
- 4.1 Metrics: the bootstrap-interval sentence removed. Genre term: unchanged (user: fine).
- Implementation details: moved whole to the supplementary, with the PercepT build sentences. No pointer sentence in
  Section 4: the user writes the opening. Section 3 says "(see the supplementary material)" for settings selection.
- Earlier decisions (Table 2 layout, 4.2 to 4.4 wording, default model) stand: see `paper-plan-7` handoff section 2.

## 3. Where things stand

- **4.1:** final for now (REPO `2a71f09`, `b50c61c`, `b085c0c`, `5fc3f01`).
- **4.2 to 4.4:** unchanged since paper plan 7. The agent's only open candidate there is j: drop "with as many clusters
  as the aspect has classes" and "with the read-out that each method picked on the validation split" from the captions
  of Tables 2 and 3 (Metrics gives both, and "Every table follows this protocol unless its caption states otherwise").
  An earlier test showed that merging 4.2's first two sentences saves no line.
- **Supplementary** (switched off in `main.tex`): section "Implementation details" (numbered 6 while it follows the
  main text), then the cluster-count figure (`Figures/fig_supp_cluster_counts.pdf`).
- **Page budget (rough, not checked):** the content fills about 7.9 of 8 pages. The intro, abstract, 4.5, 4.6 and the
  conclusion will add about 0.8 page, and the 4.1 cuts took back about 0.3 page. Table or figure moves are decided
  only after every section is written.
- **Deadlines:** paper 2026-11-16 23:59 AoE (2026-11-17 12:59 in Amsterdam); supplementary 2026-11-23.

## 4. Open points for the next step

Key:
1. 4.2 (clustering), then 4.3 (retrieval), then 4.4 (MMArt): the user's view first, then the agent's suggestion.
2. Then 4.5 (ablation) and 4.6 (qualitative): headings only, no text yet.

Minor (mention, do not edit; all the user's lines):
3. The introduction's `\enumerate{...}` never closes.
4. The comment beside `\input{Tables/cost}` ("This goes to supplemental material.") is out of date.
5. Table 8 still lists the Qwen3 row of CoSiR first.
6. Figure 1's left crossing: the user improves it later.
7. Tables 6 and 7 captions keep `\TODO{either underline or just say here we are all best?}`.
8. PF `figures/captions.md` still holds the old Qwen3 caption drafts of Figures 3 and 4 (our file).

## 5. Code and pitfalls

- Pull REPO before each edit (Overleaf). After each push, tell the user Overleaf needs a pull. Stage only our lines.
- No agent notes, TRACE blocks or comments in REPO `.tex` files. A `\TODO` only when the user asks for it.
- Never name or link the public GitHub repo in paper material. No semicolons between clauses. Write "test set".
- Build: `latexmk -pdf -interaction=nonstopmode -silent main.tex` in REPO. Two overfull boxes are old (Table 3 0.7 pt,
  Table 8 4.3 pt). Any new one is ours.
- To see the supplementary, switch `% \input{Sections/X_supplementary}` on in `main.tex` locally, build, then
  `git checkout main.tex`. A `\cref` into the supplementary breaks once it becomes its own PDF: name sections by hand.
- Crops for the user: `pdftoppm -r 150` on the page, crop with PIL, send with SendUserFile.
- Table check: `PF/tools/check_tables_vs_sheet.py` (CoSiR env), "groups checked 49 missing 0".
- PF = `docs/paper/cvpr2027_aspect_spaces/` (its own private repo), REPO = `PF/CVPR2027-CoSiR-Paper/`.

## 6. State at handoff

- Running: nothing of this chat.
- Uncommitted: nothing of ours. PF has the user's untracked `CBMI_IMP_CAMREADY (2).pdf`, kept out of git.
- On disk over 1 GB: nothing of this chat. The scratchpad (a few MB of page renders and scratch builds) goes with the
  session.
- Repo heads: REPO `5fc3f01`, PF `f2122b9`, main `728d15a0` (cluster-count figure redraw).
