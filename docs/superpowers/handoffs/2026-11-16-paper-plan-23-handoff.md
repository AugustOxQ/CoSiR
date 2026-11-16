# Handoff: go on with the co-author's answers: the Sec. 4 opening first, then the Fig. 2 caption, the cost column, then the abstract

## Brief
- Done: all ten Overleaf comments solved; the "Retrieval" heading fixed; the new teaser in the old design (no plain SigLIP row).
- State: REPO `d87a90d` pushed; 8 pages + references; page 8 ends at line 545, about 9 lines freed on page 1.
- Next: propose the Sec. 4 opening (how we evaluate and why two tasks); apply after the yes, build, check floats.
- Blocked: nothing; paper due 2026-11-17 12:59 Amsterdam, co-author discussion 2026-11-17 morning.

> Written 2026-11-16 14:58 by the "paper plan 22" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens), at the user's yes. Direction log entry: none (paper work; decisions are rows in the paper folder's
> notes and the status table).

## 1. Read in this order

1. This handoff, all of it.
2. PF `notes/2026-11-16_coauthor_points_2.md`: the co-author's eight answers of today and their status (point 1 done;
   2 to 5 open; 6 to 8 need no work). Each solved point gets its resolution and REPO commit in its row.
3. PF `notes/2026-11-16_overleaf_comments.md`: comments 1 to 10, all solved (the style of each resolution).
4. PF `user_read/2026-11-13_paper_status.md`: the status table the user reads; keep it current.
5. PF `notes/review/2026-11-16_findings_sorted.md`: the review findings (A done; B to E open), after the points.
6. Project memory `feedback_user-writes-paper.md`: the user writes the paper; agents propose exact text only.

Skills: none.

## 2. Decided by the user, not to be reopened

All on 2026-11-16, recorded in the two notes files above:
- **Order:** the co-author's points 2, 3, 4, then the abstract polish (point 5), then the review findings, then the
  final checks.
- **Each change:** propose the exact text (or apply the user's own wording), apply only after the user's yes, build,
  check the float pages, commit and push; the user pulls into Overleaf.
- **Overleaf comments 8 to 10:** "Tasks and metrics" with "Clustering groups the test images without labels: ..."
  (`87d4a24`); the retrieval sentence in the user's wording (`8429bbc`); the Fig. 4 caption explains (a) then (b)
  (`e5c413e`). The "Retrieval." heading runs into its text (`3890593`).
- **Teaser (point 1):** the old design (query left; "Ours" with Emotion, Style, Genre rows; "4/4 <class>" under each
  name), 4 per row, not 5 (query a10 gives 4/5 on emotion; no candidate gets 5/5 on all three). New caption; the intro
  sentence on plain SigLIP neighbours removed (`d87a90d`). Script PF `figures/src/fig_teaser_t2.py`.
- **Limitations** go to the supplementary (due 2026-11-24). The **LLM Usage Statement** may sit on page 9; it is the
  user's alone (agents only remind). **Title** is "CoSiR: Conditional Similarity Learning Across Aspects" for now.
- Still in force: review findings stay in `PF/notes/review/`, not in the PDF; no notes of ours in REPO `.tex` files.

## 3. Where things stand

- REPO `d87a90d` = origin; PF `e84d6a9` = origin. `main.pdf` 11 pages (8 + references), no undefined references,
  `supp.pdf` builds.
- Float pages: p1 Fig. 1; p3 Figs. 2, 3; p6 Tables 1 to 3; p7 Tables 4 to 6; p8 Fig. 4, Table 7. Page 8 ends at
  line 545. The new teaser is 1.3 in high (was 2.6 in), so about 9 lines more are free than at the last count.

## 4. Open points for the next step

Key, in order:
1. **Point 2, the Sec. 4 opening** (the co-author: briefly how we evaluate, clustering and retrieval, and why two
   tasks, what each shows). Place: after `\label{sec:experiments}`, where the commented TODO "add an opening
   sentence" sits (`4_experiments.tex:7`; end it with [done]). A draft to start from (not shown to the user yet):
   "We test each aspect embedding in two tasks against the labels of its aspect. Clustering shows whether the classes
   of an aspect form groups over the whole test set without labels, and retrieval shows whether the nearest neighbours
   of an image or a caption share its class, also between images and texts." The new first sentence of "Tasks and
   metrics" ("We test each aspect embedding in two tasks, against the labels of its aspect.") then repeats it: offer
   to cut it.
2. **Point 3, the Fig. 2 caption** (`3_method.tex:10`): from 1.5 lines to about 3, on how to read the figure:
   training at the top with (1) the aspect neighbour graphs from instructed embeddings and (2) the aspect adapter,
   test time at the bottom; grey frozen, green training only, blue trained (the current second sentence). Look at
   PF `figures/fig1_overview.png` (may be older than the PDF in REPO `Figures/fig1_overview.pdf`; render the PDF).
3. **Point 4, a cost column**: the co-author asks for a small cost column in "Table 3, which already has passes",
   pointing to the cost table in the supplementary (`Tables/cost.tex`, `\label{tab:cost}`). Table 3 is
   `Tables/main_retrieval.tex` (captions; rows mix image and text passes). Timed facts (cost table): instructed Qwen3
   3 passes 266.8 ms per image; CoSiR on SigLIP 19.5 ms (heads 0.03 to 0.04 ms); CoSiR on Qwen3 88.6 ms; a text is
   one Qwen3 pass, 3.6 ms per caption plus 0.02 ms. Not timed: plain SigLIP text, three instructed text passes, TPIPS,
   VLM2Vec. Bring options (Table 3 as asked, with ms per image and a caption note on text; or Table 2, the main table,
   where TPIPS and VLM2Vec have no timing) with a recommendation.
4. **Point 5, the abstract** after 2 to 4: stronger wording within the claim rules; fold in finding B13 (no accuracy
   number in the abstract).
5. Then the review findings (B1 caption retrieval protocol first; B14's fact is at MAIN
   `src/test/20261020_claim_test/ct2_graphs.py:82`), then the final checks
   (`PF/notes/review/2026-11-16_checks_before_upload.md`), and the reminder for the LLM Usage Statement.

With the co-authors (the user): the final title, "label-free", Fig. 4(a), status row 23 (space around the tables).
Minor: table row names differ ("Instructed Qwen3" in four tables, "Instructed embedding" in the retrieval and photo
tables; finding C9).

## 5. Code and pitfalls

- PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.
- **Float sensitivity:** a line added in Sec. 3 or 4 can push Table 3 to page 7 or Table 6 to page 8 (seen four times
  today). Test each candidate in a scratch copy (`rsync -a --exclude .git REPO/ <scratch>/x/`), build, list the float
  pages (`pdftotext -f N -l N main.pdf - | grep '^\(Table\|Figure\) [0-9]\.'`) and the line of "future work." on
  page 8. The Metrics paragraph ends near a full line: about 10 extra characters there move Table 6. The freed teaser
  space may relax this; check, do not assume.
- The user may edit REPO in Overleaf: `git status` and `git fetch` before any edit.
- Build `latexmk -pdf -interaction=nonstopmode -silent main.tex`, then `supp.tex` (it reads `main.aux`).
- Paper prose: no semicolons between clauses, no dashes, British spelling, no win/level/lose words, no
  "significant", novelty only "to our knowledge". A resolved note or TODO is commented out and ends with "[done]".
- The handoffs live in MAIN, a public repo: no co-author names here.
- Get times with `TZ=Europe/Amsterdam date` before writing them.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing of ours in REPO or PF. PF keeps the user's untracked `CBMI_IMP_CAMREADY (2).pdf` out of git.
- On disk over 1 GB: nothing. The scratchpad (page renders and scratch builds, under 100 MB) goes with the session.
