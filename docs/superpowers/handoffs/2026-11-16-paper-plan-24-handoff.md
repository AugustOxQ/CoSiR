# Handoff: start the reference check (iii) and the kit check (i) in the background, then the user's points a and e together, then b, c, d

## Brief
- Done: co-author points 2 to 4 (Sec. 4 opening, Fig. 2 caption, an ms column in Table 3), "any VLM" to "VLM", Table 3 width fix, float gaps 4 pt smaller.
- State: REPO `8804679` pushed; 8 pages + references, page 8 ends at line 550, about 7 free lines on page 8.
- Next: launch the kit check and the reference check in the background, then propose the "label-free" definition (point a) with the review findings (point e).
- Blocked: nothing; paper due 2026-11-17 12:59 Amsterdam.

> Written 2026-11-16 15:38 by the "paper plan 23" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens), at the user's yes. Direction log entry: none (paper work; decisions are rows in the paper folder's
> notes and the status table).

## 1. Read in this order

1. This handoff, all of it.
2. PF `notes/2026-11-16_final_list.md`: the user's list for the last day (a to e, then i to iv), the decisions, and
   the two abstract versions with their checked claims. Each solved point gets its resolution and REPO commit there.
3. PF `notes/review/2026-11-16_findings_sorted.md`: the review findings (A done; B and C open). Point e works from it.
4. PF `notes/2026-11-16_coauthor_points_2.md`: the co-author's answers, points 1 to 4 done (the style of each row).
5. PF `user_read/2026-11-13_paper_status.md`: the status table the user reads; keep it current.
6. Project memory `feedback_user-writes-paper.md`: the user writes the paper; agents propose exact text only.

Skills: none.

## 2. Decided by the user, not to be reopened

All on 2026-11-16, recorded in the notes files above:
- **Order:** start i and iii in the background at once; then a to e (a and e together, since finding B3 is point a);
  then ii (cold read and fact check, after the text is final); the abstract at the very end; iv is the user's.
- **Each change:** propose the exact text (or apply the user's own wording), apply only after the user's yes, build,
  check the float pages, commit and push; the user pulls into Overleaf.
- **Done today:** Sec. 4 opening, version A (`aff278b`); Fig. 2 caption, version A, 4 lines (`505dc78`); "any VLM" to
  "VLM" in Fig. 2 (`ff12762`); Table 3 ms column, option A: 19.4 / 266.8 / 19.5 ms per image, "--" / "--" / 3.6 per
  caption in text to text (`6cf16fa`); Table 3 at `\footnotesize`, `\tabcolsep` 2.3 pt so it fits (`b3579df`);
  `\floatgap` = `\vspace{-4pt}`, kept "for now" (`8804679`). The `\input` of Tables 3 and 6 moved up to keep them on
  their pages.
- **LLM Usage Statement (iv):** the user writes it. Agents do not read or comment on it; they only remind the user if
  the last version has none.
- **Abstract:** last. Version A recommended, not yet chosen; point a may change the wording of both.
- Still in force: limitations go to the supplementary (due 2026-11-24); review findings stay in `PF/notes/review/`,
  not in the PDF; no notes of ours in REPO `.tex` files; the title stays for now.

## 3. Where things stand

- REPO `8804679` = origin; PF `fdbb6b9` = origin. `main.pdf` 11 pages (8 + references), no undefined references, no
  overfull box, `supp.pdf` builds.
- Float pages: p1 Fig. 1; p3 Figs. 2, 3; p6 Tables 1 to 3; p7 Tables 4 to 6; p8 Fig. 4, Table 7. Page 8 ends at line
  550 with about 7 free lines (2 before the 4 pt cut). Abstract A or B costs about 1 line of that.
- Timing facts (r19 `/data/SSD2/cosir_r19/20261103_past_picks/outputs/stage3/cost_table.json`): plain SigLIP 19.44 ms
  per image, heads 0.034 ms, Qwen3 one pass 88.6 ms, three instructed passes 266.8 ms; caption pass 3.616 ms (r28
  `time_text.json`). Plain SigLIP text and three instructed caption passes were never timed.

## 4. Open points for the next step

Key, in order:
1. **Launch i and iii in the background first** (the user's yes). Brief each agent with "delete nothing; list what you
   would remove" and the two lines on short output and no long waits (`agent-routing.md`).
   - **i, the kit check:** a Sonnet `medium` general-purpose agent (it needs the web) gets the official CVPR 2027
     author kit and diffs it against REPO `cvpr.sty`, `main.tex`, `ieeenat_fullname.bst`, `preamble.tex`. Known
     changes to judge: `preamble.tex:20` loads `caption`; `preamble.tex:48` redefines paragraph headings; the
     `\floatgap` macro; `\small`/`\footnotesize` inside tables; the 2026-11-16 upload check, section 4, lists them. It
     reports, it does not edit. Also ask it for the CVPR 2027 guide's wording on altered formatting.
   - **iii, the reference check:** 57 cited keys (main and supplementary). About four Sonnet `high` general-purpose
     agents with web, about 15 keys each. Per key: the paper exists (title, authors, venue, year match), and the
     sentence that cites it says what the paper says (read the paper, not only the abstract). Output one file per
     batch in PF `notes/review/`, then one sorted file. The full `refs.bib` with `% verified:` lines is PF's copy; REPO
     gets it without comments.
2. **Point a with e:** propose one concrete definition of "label-free" (maybe in the intro) and one phrase used across
   the paper. The facts for finding B3: aspect labels never enter the loss; captions supervise the emotion graph;
   labels pick settings on the validation split and score the tests. Then the other critical findings, each as exact
   text: B1 (caption retrieval protocol), B5 (emotion graph sees captions), B2 ("highest" against CLAY's 0.437 and
   0.441 on style, the MMArt emotion retrieval tie), B7 (the plain baseline's text encoder differs), B9 (cite
   CIM-Embed), C12 ("loses" in Sec. 4). Watch the about 7 free lines.
3. **Point b:** every "direction" in the retrieval text (Sec. 3, Sec. 4, the Table 3 caption) to plain names such as
   "text-to-image, image-to-text and text-to-text retrieval".
4. **Point c:** in Fig. 1 the query labels "Emotion:", "Style:", "Genre:" capitalised, bold, aligned at the colon
   (PF `figures/src/fig_teaser_t2.py`); show the render first.
5. **Point d:** redesign Fig. 3 (PF `figures/src/fig2_graphs.py`; caption at `3_method.tex:31`): propose two or three
   simpler layouts with fewer symbols for the user to pick; an Opus `high` worker builds the pick.
6. Then ii (cold read and fact check of the final text), the abstract, and the reminder for iv.

Minor, with the co-authors (the user): the final title, Fig. 4(a), table row names ("Instructed Qwen3" against
"Instructed embedding", finding C9).

## 5. Code and pitfalls

- PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.
- **Test each change in a scratch copy first:** `rsync -a --exclude .git REPO/ <scratch>/x/`, build, then list the
  float pages (`pdftotext -f N -l N main.pdf - | grep '^\(Table\|Figure\) [0-9]\.'`), the line of "future work." on
  page 8 (`pdftotext -layout`), and the **size** of each overfull box (`grep 'Overfull' main.log`), not only their
  count: the ms column made Table 3 26 pt too wide while the count stayed at one.
- **Bottom floats slip to the next page** when text above their `\input` grows (Tables 3, 6 and 7 are `[b]`). The fix
  that worked twice: move the `\input` earlier in the source, then confirm the caption's position is unchanged
  (`pdftotext -bbox -f N -l N main.pdf - | grep -A1 '>Table<'`).
- **Free lines on page 8:** append k one-line paragraphs ending with a marker after the conclusion in a scratch copy and
  find the largest k that stays on page 8.
- Fig. 2 comes from `figures/src/fig1_overview_lr.py` (writes `figures/drafts/fig1_overview_lr.pdf`, copied to REPO
  `Figures/fig1_overview.pdf`), not from `fig1_overview.py`.
- The main paper cannot `\cref` labels in `supp.tex`; it points with "see the supplementary material".
- The user may edit REPO in Overleaf: `git status` and `git fetch` before any edit.
- Build `latexmk -pdf -interaction=nonstopmode -silent main.tex`, then `supp.tex` (it reads `main.aux`).
- Paper prose: no semicolons between clauses, no dashes, British spelling, no win/level/lose words, no
  "significant", novelty only "to our knowledge". A resolved note or TODO is commented out and ends with "[done]".
- The handoffs live in MAIN, a public repo: no co-author names here.
- Get times with `TZ=Europe/Amsterdam date` before writing them.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing of ours in REPO or PF. PF keeps the user's untracked `CBMI_IMP_CAMREADY (2).pdf` out of git.
- On disk over 1 GB: nothing. The old scratchpad holds about 300 MB of scratch builds; the user decides on it (storage summary of 2026-11-16 15:38).
