# Handoff: the retrieval table of Sec. 4.3, then the other tables and figures of Sec. 4, then the writing

## Brief
- Done: Table 2 (clustering, class count) filled except BERTopic, VLM2Vec and PercepT on CLIP L/14; cluster-count range scored for 8 of 11 rows.
- State: ours best in all three Table 2 columns, 0.102 / 0.411 / 0.510 against the teacher's 0.089 / 0.406 / 0.501.
- Next: design and fill the retrieval table (4.3) with the user; skip the 4.2 text for now.
- Blocked: the user's 11:00 meeting; ask nothing about it until the user says it is over.

> Written 2026-11-11 06:00 by the "paper experiments" tab, which stays open: it receives r25 and r26 and fills Table
> 2's last rows. Outside the research loop: no loop step, no direction log entry. PF means
> `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`; REPO means `PF/CVPR2027-CoSiR-Paper/` (the Overleaf-synced
> LaTeX repo the co-authors see).

## 1. Read in this order

1. This handoff, all of it.
2. `PF/README.md`, the Decisions rows (all of 2026-11-11, and 2026-11-10's ablation row with its retrieval column).
3. `REPO/Sections/4_experiments.tex` (4.1 names every compared method and the retrieval metric; 4.3 to 4.5 are empty
   headings) and `REPO/Tables/main_clustering.tex` (Table 2, the format to match).
4. `PF/notes/trace/4_experiments.md` (the source of each fact in Sec. 4, and how a table's trace is written).
5. `PF/notes/supplementary_items.md` (what goes to the supplementary instead of Sec. 4).
6. The retrieval sources, through a fact-finding read (below), not in this chat's context.
7. Only when the user says the meeting is over: `PF/notes/after_meeting_2026-11-11.md` (project memory
   `paper-meeting-2026-11-11`).

Skills: none.

## 2. Decided by the user, not to be reopened

All of section 2 of `2026-11-11-paper-experiments-2-handoff.md` stands (it carries the earlier handoffs'). New:

- **Order of work (user, 06:00):** the retrieval table first (4.3), then the other tables and figures of Sec. 4, then
  the writing. The 4.2 text waits.
- **Table 2 format:** AMI with three decimals, no intervals, a Passes column, bold for the best per aspect, each
  aspect's class count (9 / 27 / 10). Use the same look in the other Sec. 4 tables unless the user decides otherwise.
- **Supplementary range of cluster counts:** a figure, one panel per aspect, AMI against the number of clusters, one
  line per Table 2 row (user: option b). The "paper experiments" tab builds it.
- **PercepT's own Table 2 (Stage 2 AUC):** not in the paper; a possible supplementary check of our PercepT build only.
- **Ablation table (2026-11-10):** a retrieval column (ArtELingo test, image to image, label mAP, plain cosine, seeds 0
  to 4; removal minus ours, with 95% intervals) beside clustering AMI.
- Retrieval is the secondary result (Sec. 4 structure, 2026-11-11). The 2026-11-05 flag stands: ours has fewer pair
  hits than plain Qwen3; the paper's wording on it waits for the user's talk with the other authors (project memory
  `pair-hit-limit`).

## 3. Where things stand

- REPO `7384d4f` (pushed): Table 2 at the class count. Filled rows: plain 0.070 / 0.235 / 0.425; + TURTLE 0.067 /
  0.215 / 0.436; teacher 0.089 / 0.406 / 0.501; TPIPS 0.072 / 0.332 / 0.496; CLAY 0.068 / 0.406 / 0.485; CRL 0.077 /
  0.390 / 0.428; PercepT (Qwen3) 0.068 / 0.192 / 0.424; ours 0.102 / 0.411 / 0.510. Empty: BERTopic, VLM2Vec, PercepT
  on CLIP L/14.
- cosir-paper (PF) `0cb6f56` (pushed): README rows, `notes/supplementary_items.md`, the trace.
- Retrieval results, in the project's auto reports (`docs/reports/auto/percept/`), not yet in the paper:
  `2026-12-29_retrieval.md` (image retrieval), `2027-01-09_text_retrieval.md` and `2027-01-11_text_claim.md` (text
  retrieval, three directions with a caption), `2027-01-13_ablation_retrieval.md` (ablation retrieval column),
  `2027-01-04_new_scores.md` and `2027-01-06_outside_methods.md` (r17 and r18: the comparators' retrieval cells),
  `2027-01-03_mmart_test.md` (MMArt). Which comparators have retrieval cells at the paper's protocol: not checked.

## 4. Open points for the next step

Key:
1. **The retrieval table's form**, with the user: rows (the same methods as Table 2, or only those with retrieval
   cells), columns (image to image per aspect; the text directions in the same table or a second one), the metric (label
   mAP, as 4.1 says), backbone (Qwen3 only, like Table 2). Start with a fact-finding read (a Sonnet `medium` worker or
   Codex `gpt-6.1-terra` medium, per `agent-routing.md`): every retrieval cell that exists for each Table 2 row on the
   ArtELingo test split, its exact value, file and protocol. Numbers come from result files, not report prose. Then
   propose the form, the user decides, then fill REPO.
2. The other Sec. 4 tables and figures after that, in the order the user picks: the ablation table (4.4, with the
   retrieval column), the other encoders and photos (4.4 or supplementary), the qualitative figure (4.5, r20's
   figures in `src/test/20261109_r20_figures/`).

Minor:
3. The writing of each subsection only after its tables and figures, from the user's points (no agent-written paper
   text: project memory `user-writes-paper`).
4. The meeting memo's points wait for the user's word that the meeting is over.

## 5. Code and pitfalls

- **Two chats share one working tree.** The "paper experiments" tab still edits REPO and PF in the same folders:
  `REPO/Tables/main_clustering.tex`, later a supplementary figure, the 4.1 sentence on PercepT's own encoders (after
  the user's yes), the Table 2 and cluster-count sections of the trace, and its own README rows. Do not edit
  `Tables/main_clustering.tex`. Before editing `Sections/4_experiments.tex` or the trace, check `git status` for the
  other tab's uncommitted hunks. Commit only your own paths (`git commit -- <paths>`, never `git add -A`), and push at once.
- Tables in `REPO/Tables/*.tex`, `\input{Tables/<name>}`. One-column tables at `\tabcolsep` 2.7pt overflow easily;
  check `main.log` for "Overfull".
- Each finished part: write into REPO, `latexmk -pdf -silent main.tex`, render the page with `pdftoppm` and look at it,
  commit and push REPO (only our files), update the trace, commit and push PF, then SendUserFile `REPO/main.pdf`.
- No notes of ours in REPO. Paper names, never internal ones (c* is `\method`). No semicolons between clauses. "Test
  set", never "held-out" or "held". TURTLE always "at its released code's default settings, without its
  hyperparameter selection".
- `REPO/Sections/2_related_work.tex` holds the user's own uncommitted edit: leave it. `PF/CBMI_IMP_CAMREADY (2).pdf`
  is the user's, untracked. `/project/claude-config` holds other sessions' uncommitted edits: commit only your hunks.
- The shell is zsh: quote globs.

## 6. State at handoff

- Running: the "paper experiments" tab (waits for r25's cells, due about 07:15, and r26's table, about 10:00, then
  fills Table 2's last rows and scores them on the cluster-count grid); the r25 and r26 tabs.
- Uncommitted: `REPO/Sections/2_related_work.tex` (the user's); in the main repo, files not ours
  (`docs/superpowers/constitution.md`, `src/test/20261111_paper_baselines/decisions.md` and others).
- On disk over 1 GB: nothing from this work.
