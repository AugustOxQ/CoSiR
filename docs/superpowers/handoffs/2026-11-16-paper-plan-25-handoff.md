# Handoff: point b ("direction" out of the paper) as exact text, then c, d, ii, the abstract and the reminder for iv

## Brief
- Done: kit check (low risk), reference check (52 of 57 OK, 6 fixes applied), points a ("label-free" defined) and e (review findings B1, B2, B5, B7, B9, C12).
- State: REPO `9db589c` pushed; 8 pages + references, page 8 ends at line 557, 3 free lines.
- Next: propose exact text for point b, the 8 "direction" places, the loss sentence included (the user's yes).
- Blocked: nothing; paper due 2026-11-17 12:59 Amsterdam.

> Written 2026-11-16 15:59 by the "paper plan 24" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens), at the user's yes. Direction log entry: none (paper work; decisions are rows in the paper folder's
> notes and the status table).

## 1. Read in this order

1. This handoff, all of it.
2. PF `notes/2026-11-16_final_list.md`: the user's list for the last day (a to e, then i to iv). Rows a, e, i, iii are
   done with their REPO commit; b, c, d, ii, iv open. It also holds the two abstract versions (A recommended).
3. PF `notes/2026-11-16_points_a_e.md`: the exact old and new text of the changes just applied (the style of a proposal).
4. PF `notes/review/2026-11-16_refcheck_sorted.md`: the reference check, with the optional soft notes (R7) not applied.
5. PF `user_read/2026-11-13_paper_status.md`: the status table the user reads; keep it current.
6. Project memory `feedback_user-writes-paper.md`: the user writes the paper; agents propose exact text only.

Skills: none.

## 2. Decided by the user, not to be reopened

All on 2026-11-16, recorded in the notes files above:
- **Order:** b, c, d; then ii (cold read and fact check, after the text is final); the abstract at the very end; iv is
  the user's.
- **Each change:** propose the exact text, apply only after the user's yes, build, check the float pages, commit and
  push; the user pulls into Overleaf.
- **Point a:** "label-free" is the one phrase, defined in the intro as "no aspect label, such as awe for emotion,
  enters its loss" ("aspect label" is the user's pick over "class label").
- **Point b:** every "direction" goes, in retrieval passages and in the loss sentence (`3_method.tex:102`, "averages
  four directions"), so the word leaves the paper.
- **Point e and the reference check:** applied in `9db589c` (B1, B2, B5, B7, B9, C12; R1 to R6, R5 without ESMC).
- **Kit (i):** keep all formatting changes (`\floatgap`, table fonts and `\tabcolsep`, the second `caption` load).
- Still in force: limitations go to the supplementary (due 2026-11-24); review findings stay in `PF/notes/review/`,
  not in the PDF; no notes of ours in REPO `.tex` files; the title stays for now; LLM Usage Statement: the user writes
  it, agents do not read or comment on it, they only remind the user if the last version has none.

## 3. Where things stand

- REPO `9db589c` = origin; PF `a06c194` = origin. `main.pdf` 11 pages (8 + references), no undefined reference, no
  overfull box; `supp.pdf` 3 pages.
- Float pages: p1 Fig. 1; p3 Figs. 2, 3; p6 Tables 1 to 3; p7 Tables 4 to 6; p8 Fig. 4, Table 7. Page 8 ends at line
  557 with 3 free lines (550 and about 7 before a and e). Abstract A or B costs about 1 line of that.
- The 8 "direction" places: `1_introduction.tex:32` ("retrieval in all four directions"), `3_method.tex:102` (the loss),
  `3_method.tex:165` ("four retrieval directions"), `4_experiments.tex:100`, `:102`, `:104`, `:105`, `:106`. Lines 59 and
  the Table 3 caption already say "kind of retrieval" and "per aspect in each block".

## 4. Open points for the next step

Key, in order:
1. **Point b:** exact text for the 8 places, plain names such as "text-to-image, image-to-text and text-to-text
   retrieval"; measure the line cost (3 free lines).
2. **Point c:** in Fig. 1 the query labels "Emotion:", "Style:", "Genre:" capitalised, bold, aligned at the colon
   (PF `figures/src/fig_teaser_t2.py`); show the render first.
3. **Point d:** redesign Fig. 3 (PF `figures/src/fig2_graphs.py`; caption at `3_method.tex:36`): propose two or three
   simpler layouts with fewer symbols for the user to pick; an Opus `high` worker builds the pick.
4. Then ii (cold read and fact check of the final text), the abstract (A or B, both still say "trained label-free"),
   and the reminder for iv.

Minor, with the co-authors (the user): the final title, Fig. 4(a), table row names ("Instructed Qwen3" against
"Instructed embedding", finding C9). Optional from the reference check: R7 soft notes (RKD, CSN, TF-QARE,
MultiMAP, PercepT photos, X-Cluster mood, Tversky, CSD).

## 5. Code and pitfalls

- PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.
- **REPO `main.tex` has the user's own uncommitted edit.** Leave it alone and stage files by name, never `git add -A`.
- **Test each change in a scratch copy first:** `rsync -a --exclude .git REPO/ <scratch>/x/`, build, then list the float
  pages (`pdftotext -f N -l N main.pdf - | grep '^\(Table\|Figure\) [0-9]\.'`), the line of "future work." on page 8
  (`pdftotext -layout`), and the size of each overfull box (`grep 'Overfull' main.log`).
- **Measure each change's cost alone** (apply one change, find the "future work." line) before a combined build; a
  patch script that asserts each old string occurs once makes this quick (the last one:
  `/tmp/claude-0/-project-CoSiR/e9b69953-b03e-4e4c-a1c4-689f81fd0bf4/scratchpad/ae/patch.py`, may be gone).
- **Free lines on page 8:** append k one-line paragraphs ending with a marker after the conclusion in a scratch copy and
  find the largest k that stays on page 8.
- **Bottom floats slip to the next page** when text above their `\input` grows (Tables 3, 6 and 7 are `[b]`). The fix
  that worked twice: move the `\input` earlier, then confirm the caption's position is unchanged.
- **refs.bib:** edit PF's copy (with `% verified:` lines), then `grep -v '^\s*%' PF/refs.bib | cat -s > REPO/refs.bib`.
- Number pairs in paper prose: "from 0.402 to 0.528" or "it reaches 0.441 and CLAY 0.437", never "X against Y".
- Fig. 2 comes from `figures/src/fig1_overview_lr.py`, not `fig1_overview.py`.
- The main paper cannot `\cref` labels in `supp.tex`; it points with "see the supplementary material".
- Build `latexmk -pdf -interaction=nonstopmode -silent main.tex`, then `supp.tex` (it reads `main.aux`).
- Paper prose: no semicolons between clauses, no dashes, British spelling, no win/level/lose words, no
  "significant", novelty only "to our knowledge". A resolved note or TODO is commented out and ends with "[done]".
- The handoffs live in MAIN, a public repo: no co-author names here.
- Get times with `TZ=Europe/Amsterdam date` before writing them.

## 6. State at handoff

- Running: nothing.
- Uncommitted: REPO `main.tex` (the user's own edit, not ours). Nothing of ours in REPO or PF. PF keeps the user's
  untracked `CBMI_IMP_CAMREADY (2).pdf` out of git.
- On disk over 1 GB: nothing. This chat's scratchpad holds about 640 MB of scratch builds
  (`/tmp/claude-0/-project-CoSiR/e9b69953-b03e-4e4c-a1c4-689f81fd0bf4/scratchpad/`); the user decides on it.
