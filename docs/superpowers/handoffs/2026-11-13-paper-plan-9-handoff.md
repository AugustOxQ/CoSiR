# Handoff: paper plan 9: the user checks 4.5, then 4.7 (qualitative examples), the user's points first

## Brief
- Done: 4.2 to 4.4 rewritten by the user's three rules; photos moved to their own subsection 4.6; 4.5 text in (three paragraphs). REPO `a011509`.
- State: Section 4 has text in 4.1 to 4.6. The 4.5 number sets and analyses are the agent's picks and wait for the user's check.
- Next: take the user's corrections to 4.5, then 4.7 (qualitative examples), the user's points first.
- Blocked: nothing.

> Written 2026-11-13 18:53 by the "paper plan 8" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens). Direction log entry: none (paper work; decisions are rows in the paper folder's README).

## 1. Read in this order

1. This handoff, all of it.
2. Project memory `feedback_user-writes-paper.md`: the user writes the paper. Agents word only the user's points. The
   CBMI paper is a tone model for the intro and method only, never for the experiments.
3. Global `~/.claude/rules/report-writing.md`, bullet "Results paragraphs in papers": the user's three rules, which
   hold for all their papers.
4. `docs/paper/cvpr2027_aspect_spaces/notes/section4_cuts.md`: the three rules with the agent's reading, and the
   status per subsection.
5. The README rows of 2026-11-13 in `docs/paper/cvpr2027_aspect_spaces/README.md` (the last five rows are this chat's).
6. REPO `Sections/4_experiments.tex` from `\subsection{Clustering experiments}` on.

Skills: none.

## 2. Decided by the user, not to be reopened

All on 2026-11-13, each a README row:
- The three rules for results paragraphs: (1) no win/level/lose words (above, level, behind, ahead, below); (2) per
  paragraph no "(X against Y)" and at most three number sets, written "from X to Y"; (3) "We report X in Table N",
  then the number sets, then the analysis, key points only. They hold for all the user's papers (global rule).
- 4.2: style point folded into the genre sentence, "with every seed" kept (it says more than the Metrics line), an
  analysis sentence on the content correction (Tab. 5). Captions of Tables 2 and 3: class-count and read-out phrases
  cut, BERTopic "one topic per class".
- 4.3: three number sets (style image to image, genre text to image, emotion text to text "where it reaches 0.377 and
  CoSiR 0.346"); analysis unchanged.
- 4.4: MMArt only; three number sets (emotion mAP 0.202 for both, AMI emotion and style); analysis unchanged.
- 4.5: one paragraph per table (ablation, other encoders, cost), at most two number sets each. The three tables may be
  merged into one or two later, when the page budget is settled.
- 4.6 "Results on photos": its own subsection after 4.5 (the table has four backbones); text as committed.
- Earlier decisions stand: see the `paper-plan-8` and `paper-plan-7` handoffs, section 2.

## 3. Where things stand

- **Text:** 4.1 to 4.6 have text. 4.7 has its heading and Figures 3 and 4 with captions, no text. The Section 4 opening
  is the user's to write. Intro, abstract and conclusion are not written yet.
- **Table numbers** follow the float order now: 2 clustering, 3 text retrieval, 4 MMArt, 5 ablation, 6 other
  encoders, 7 cost, 8 photos. Older README rows use the old numbers (ablation was 5, photos 7, cost 8).
- **Layout:** 11 pages with references; floats pile up after 4.5 and 4.6. Paragraphs that follow a table `\input` get
  no indent, so they run together. Layout and table merges wait until every section is written.
- **r30 and r31 (asked by the user at the end):** no number now in the main text waits on them. The main model's
  numbers come from r28. r31 (r29's stage 2, the SigLIP-only model, running tonight) gives an ablation or
  supplementary row, not the main model (the user with the co-authors). r30 is an exploration from saved outputs
  (DEC map and emotion valence) with no table cell planned. The DEC ablation row comes from r21, whose numbers are on
  the Qwen3 image model (style AMI 0.4109, Table 6's Qwen3 row), not on SigLIP 1 as Table 5 is.
- **Deadlines:** paper 2026-11-16 23:59 AoE (2026-11-17 12:59 in Amsterdam); supplementary 2026-11-23.

## 4. Open points for the next step

Key:
1. The user's corrections to 4.5's number sets and analyses (README row "Sec. 4.5 text").
2. 4.7 qualitative examples: the user's points first. Do the captions carry Figures 3 and 4, with a short pointer
   sentence in 4.7, or does 4.7 get its own points?
3. Later: where the r31 SigLIP-only row and the DEC row go (ablation table or supplementary), once r31 reports.

Minor (mention, do not edit; all the user's lines):
4. The comment beside `\input{Tables/cost}` ("This goes to supplemental material.") now contradicts the cost paragraph.
5. The introduction's `\enumerate{...}` never closes.
6. Table 7 (cost, Table 8 before) still lists the Qwen3 row of CoSiR first. Tables 6 (encoders) and 8 (photos)
   keep `\TODO{either underline or just say here we are all best?}` in their captions.
7. Figure 1's left crossing: the user improves it later.
8. PF `figures/captions.md` still holds the old Qwen3 caption drafts of Figures 3 and 4 (our file).

## 5. Code and pitfalls

- Pull REPO before each edit (Overleaf). After each push, tell the user Overleaf needs a pull. Stage only our lines.
- No agent notes, TRACE blocks or comments in REPO `.tex` files. A `\TODO` only when the user asks for it.
- Never name or link the public GitHub repo in paper material. No semicolons between clauses. Write "test set".
- Build: `latexmk -pdf -interaction=nonstopmode -silent main.tex` in REPO. Two overfull boxes are old (0.7 pt and
  4.3 pt, in two table files). Any new one is ours.
- To commit only part of a file, save it, write the committed version, commit, then restore (no `git add -p` here).
- Crops for the user: `pdftoppm -r 150` on the page, crop with PIL, send with SendUserFile. `pdftotext -bbox` finds a
  word's place (points; times 150/72 for pixels).
- Table check: `PF/tools/check_tables_vs_sheet.py` (CoSiR env), "groups checked 49 missing 0".
- PF = `docs/paper/cvpr2027_aspect_spaces/` (its own private repo), REPO = `PF/CVPR2027-CoSiR-Paper/`.

## 6. State at handoff

- Running: nothing of this chat. r31 run 2 (another tab) uses the queue tonight.
- Uncommitted: nothing of ours. PF has the user's untracked `CBMI_IMP_CAMREADY (2).pdf`, kept out of git.
  claude-config has another session's `codex/jobs.md` change, not ours.
- On disk over 1 GB: nothing of this chat. The scratchpad (a few MB of page renders) goes with the session.
- Repo heads: REPO `a011509`, PF `e11aafc`, claude-config `de9f1f0`.
