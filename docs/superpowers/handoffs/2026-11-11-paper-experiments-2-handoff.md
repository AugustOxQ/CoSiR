# Handoff: fill Table 2's empty rows at the class count when r25 and r26 report, then the supplementary range of cluster counts

## Brief
- Done: Table 2 (main clustering table) in the paper at each aspect's class count; r26 opened for PercepT on its own encoders.
- State: ours best in all three columns, 0.102 / 0.411 / 0.510 against the teacher's 0.089 / 0.406 / 0.501.
- Next: fill BERTopic and VLM2Vec (r25) and PercepT on CLIP L/14 (r26) at the class count when their reports land.
- Blocked: the user's 11:00 meeting; ask nothing about it until the user says it is over (memo).

> Written 2026-11-11 05:36 by the "paper experiments" tab (context passed 200k). Outside the research loop: no loop
> step, no direction log entry. PF means `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`; REPO means
> `PF/CVPR2027-CoSiR-Paper/` (the Overleaf-synced LaTeX repo the co-authors see).

## 1. Read in this order

1. This handoff, all of it.
2. `PF/README.md`, the Decisions rows of 2026-11-11 (the last four are from this chat).
3. `REPO/Tables/main_clustering.tex`, `REPO/Sections/4_experiments.tex` (the 4.1 metrics paragraph) and
   `PF/notes/trace/4_experiments.md` (sections "4.2 Table 2" and "Table 2 switched to the class count": every cell's
   exact value and file).
4. When r25 or r26 reports: its auto report's results table (the class-count cells), then the result file it names.
5. Only when the user says the meeting is over: `PF/notes/after_meeting_2026-11-11.md` (project memory
   `paper-meeting-2026-11-11`). Not before.

Skills: none.

## 2. Decided by the user, not to be reopened

All of section 2 of `2026-11-11-paper-experiments-table-handoff.md` stands, except that k = 20 is replaced as below.
New on 2026-11-11 (README rows):
- **Table 2 format:** AMI with three decimals (0.080), as PercepT reports it; no intervals; the Passes column (encoder
  passes per test image) kept after the user asked what it means.
- **Cluster count:** each aspect's class count, k = 9 emotion, 27 style, 10 genre, in Table 2 and the 4.1 metrics
  sentence; the full range of cluster counts goes to the supplementary (user, after the recommendation and after
  asking whether AMI compares across cluster counts: only methods at the same count are compared).
- **PercepT-original:** a new run, r26: PercepT at paper settings with 40 topics, on CLIP L/14 and ModernBERT tuned on
  GoEmotions, our protocol, all three aspects. The paper states two differences: our 42,969 English training
  paintings instead of PercepT's 1.2M translated pairs, and 40 topics instead of its topic-count rule.

## 3. Where things stand

- REPO: `51626f5` (Table 2 added), `7384d4f` (class count in Table 2 and 4.1), pushed. `latexmk` exits 0, 7 pages, no
  overfull lines, no undefined references. cosir-paper: `7f9b776`, `e4d3be9`, `71be7b4` (README rows, trace), pushed.
- Table 2 at the class count (exact values in the trace): plain 0.070 / 0.235 / 0.425; + TURTLE 0.067 / 0.215 /
  0.436; teacher 0.089 / 0.406 / 0.501; TPIPS 0.072 / 0.332 / 0.496; CLAY 0.068 / 0.406 / 0.485; CRL 0.077 / 0.390 /
  0.428; PercepT (Qwen3) 0.068 / 0.192 / 0.424; ours 0.102 / 0.411 / 0.510 (bold in all three). Empty: BERTopic,
  VLM2Vec, PercepT on CLIP L/14.
- c* minus the teacher at the class count (r17 verdicts, SC3): emotion +0.0135 [+0.0105, +0.0167], style +0.0046
  [+0.0015, +0.0080], genre +0.0086 [+0.0017, +0.0152]. At k = 20 genre was the one cell below the teacher.
- r17's sweep file (`/data/SSD2/cosir_r17/20261103_new_scores/outputs/stage3/sweep/artelingo__qwen.json`) holds c*,
  plain, teacher and PercepT (Qwen3) at k = 5, 8, 9, 16, 20 (emotion), 10, 20, 27, 40 (style), 5, 10, 16, 20 (genre).
  TURTLE ran at 5 to 40 (r18 `diag/fits.json`). TPIPS, CLAY and CRL cells at other counts: not checked.
- r25 tab (BERTopic, VLM2Vec): opened 05:03; at 05:35 its row in ListAgents said "shell", so its Claude may have ended
  (not checked). r26 tab (PercepT on its own encoders): opened 05:27, busy. At 05:35 both got a cross-session message
  to also score at the class count, pointing to the README row. r26 confirmed (it checked the README row): it scores
  the k-means reading at the class count (the paper row) and at k = 20, plus the top-topic reading, into
  `src/test/20261111_percept_original/records/po_table.md` and its report; extractions queued behind r25's GPU job
  (about 06:36); training expected to end before 10:00. r25 did not answer (not checked further).

## 4. Open points for the next step

Key:
1. **The three empty rows.** When each report lands, take its class-count cells (k = 9 / 27 / 10) from the result
   file, round to three decimals, fill REPO, rebuild, look at the page, commit and push, add the values to the trace.
   If a report has only k = 20 cells, score the saved embeddings or outputs at the class count (a short CPU job
   through `res-queue cpu`), or ask the tab to; BERTopic needs a topic reduction per count. Check first whether the
   r25 tab is still alive (`ListAgents`; `res-queue status`).
2. **The 4.1 sentence on PercepT's own encoders**, once r26's numbers exist: Table 2's CLIP L/14 PercepT row, with the
   two differences the user decided the paper states. Show the user the sentence before it enters REPO.
3. **The supplementary range of cluster counts:** a small figure or table of AMI against k for every row; propose the
   form to the user. Most cells exist (above); missing ones are cheap k-means runs on saved embeddings.

Minor:
4. The text of 4.2 needs the user's points; write nothing before them.
5. The meeting memo's points (Figure 1 style, pair hits, the read-out, the no-direct-writing scope, related work (c))
   wait for the user's word that the meeting is over.

## 5. Code and pitfalls

- Tables in `REPO/Tables/*.tex`, included with `\input{Tables/<name>}`. Table 2 is one column wide at `\tabcolsep`
  2.7pt; a longer label or backbone overflows (check `main.log` for "Overfull").
- Each finished part: write into REPO, `latexmk -pdf -silent main.tex`, render the page with `pdftoppm` and look at
  it, commit and push REPO (only our files), update the trace, commit and push `cosir-paper`, then SendUserFile
  `REPO/main.pdf` (`display: render`).
- No notes of ours in REPO. Paper names, never internal ones (c* is `\method`). No semicolons between clauses. "Test
  set", never "held-out" or "held". TURTLE always with "at its released code's default settings, without its
  hyperparameter selection"; never call default TURTLE stronger than k-means.
- Numbers come from result files, not report prose. The TPIPS, CLAY, CRL and TURTLE cells are printed in no report;
  their files are named in the trace.
- The shell is zsh: quote globs. `/project/claude-config` holds another session's uncommitted edits: commit only your
  own hunks there.

## 6. State at handoff

- Running: the r25 tab (state not checked) and the r26 tab (busy), each its own chat. Nothing of this chat.
- Uncommitted: `REPO/Sections/2_related_work.tex` (the user's own edit; leave it);
  `/project/CoSiR/docs/superpowers/constitution.md` and `src/test/20261111_paper_baselines/decisions.md` (not ours);
  `PF/CBMI_IMP_CAMREADY (2).pdf` (the user's, untracked).
- On disk over 1 GB: nothing. The session scratchpad holds a preview copy of REPO and page renders (a few MB); they go
  with the session.
