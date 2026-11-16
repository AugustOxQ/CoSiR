# Handoff: go on with Part A of the merged review list from A6, then put all approved fixes into the paper

## Brief
- Done: the author's own review is merged into the panel's list (PF `20696d8`). A1, A3 and A5 are approved; built together, they cost 0 lines.
- State: the approved fixes are rows in PF `tools/apply_fixes.py`, not yet in REPO (`a56c4c1`). Page 8 keeps about 1.4 free lines.
- Next: A6, then A15, A19, A13 and A10 with the user, one exact text at a time; then all rows into REPO, build, commit, push.
- Blocked: the user's abstract, the Overleaf pull, OpenReview. Paper due 2026-11-17 12:59 Amsterdam.

> Written 2026-11-16 20:35 by the "paper plan 31" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens), with the user's yes. Direction log entry: none (paper work; decisions are rows in the paper folder's
> notes and the status table).

## 1. Read in this order

1. This handoff, all of it.
2. PF `notes/review/2026-11-16_fix_round.md`: every approved fix so far, with old and new text, its source, who
   decided, and its built line cost.
3. PF `notes/review/2026-11-16_panel_merged.md`: the header, the section "The author's review (U): what it adds", the
   Part A table, then each group when you reach it (A6 next). Open a seat's report (`2026-11-16_panel_<seat>.md`) only
   for the exact text of a fix.
4. PF `notes/review/2026-11-16_user_review.md`: the user's own review, verbatim. Open it only for the exact words of a
   U point.
5. The previous handoff, `docs/superpowers/handoffs/2026-11-16-paper-plan-31-handoff.md`, sections 2 and 5.
6. Project memory: `feedback_user-writes-paper.md`, `project_paper-name-angle.md`, `feedback_paper-chats-claude-only.md`.

Skills: none. The chat starts with `claude-paper`, so ARS is loaded if a further ARS step is needed.

## 2. Decided by the user, not to be reopened

All on 2026-11-16, evening; each is a row in `fix_round.md` and in `tools/apply_fixes.py`:
- **A1** (all four after seeing the recommendation): "without annotation" to "without aspect labels"
  (`1_introduction.tex:22`); "impressionism" to "Impressionism" and "and instead only require the aspects themselves."
  to "and instead require one instruction per aspect and captions." (`:26`, the co-author's sentence); "read from their
  captions" to "read from annotators' captions" (`:27`). The user's own wording for line 22 was rejected for its line.
- **A3:** the conclusion's last sentence becomes "Whether other aspects can join once an instruction names them and
  their neighbour graphs are built is left to future work." (EIC-W1). The recipe detail and the fact that labels chose
  the recipes go to the limitations in the supplementary (status row 18), not the conclusion.
- **A5:** Sec. 4.5, `4_experiments.tex:146`, becomes "The instructed graphs and their loss carry emotion and style, and
  the relational term carries genre." (DA-M5). **The conclusion's "from instructed neighbour graphs" stays** (the user):
  the name ANGLE says neighbour graphs, and genre's caption head is trained only by its graph (`3_method.tex:136`,
  "The caption tower has no alignment term"). Both variants that keep the graphs were built at +1 line on page 8.
- **The way of working** (user, this chat): per group, the exact old and new text with options and a recommendation;
  the user picks; the row goes into `tools/apply_fixes.py` and `fix_round.md`; a scratch build of all rows so far
  checks the cost; commit and push in PF. REPO gets all rows at once after Part A.
- Still in force from the plan 31 handoff, section 2: the abstract and the Figure 1 caption are the user's own text;
  the LLM Usage Statement is never read or commented on; in this paper's chats, Claude subagents only, never Codex.

## 3. Where things stand

- REPO `a56c4c1` = origin, no fix applied yet. PF `d3f785f` = origin. MAIN: this handoff.
- Built with A1, A3 and A5 (scratch copy): 11 pages; floats p1 Fig. 1, p3 Figs. 2 and 3, p6 Tables 1 to 3, p7 Tables 4
  to 6, p8 Fig. 4 and Table 7 (all as before); "future work." at ruler 564; about 1.4 free lines on page 8; no overfull
  box, no undefined reference.
- The merged list now has 26 Part A groups. The merge proposes this order: A6 (planned); then A15, A19, A13 and A10,
  which the user's review presses and which each have zero-line wording; then A26, A25 and A24's small items. A2 stays
  input to the user's abstract, with the review's points added. Not yet discussed: A4, A6 to A26 (except A5).
- New groups: A25 (cite ModernBERT and GoEmotions at `4_experiments.tex:41`, page 5), A26 (the Figure 2 caption: each
  aspect uses its own saved trunk; the caption is the user's text, so the user's call), B16 (the code answers for the
  supplementary or the rebuttal), D10 (the review's claims that do not hold: 88.9 ms and 4.6 times, the Figure 2
  points, "nearly 30 times").

## 4. Open points for the next step

Key, in order:
1. **A6** (the photo paragraph omits that ANGLE trained on paintings only is lower than the instructed embedding):
   DA-m1 (built 0) or R3-W5.
2. **A15, A19, A13, A10**, as the merged list proposes. Then the rest of Part A in its order; A26 and A25 are the user's
   call (caption text; page 5).
3. **Into REPO:** `git -C REPO fetch` first (a co-author may have pushed from Overleaf; `apply_fixes.py` stops when an
   old text no longer matches). Then `python3 tools/apply_fixes.py REPO`, build, `layout_measure.sh`, `free_lines.py`,
   float pages, commit, push. Update status rows 14 and 25.
4. **The user's abstract**, the last build check (status row 16), then the user's Overleaf pull, OpenReview and upload
   (plan 31 handoff, section 4, points 3 to 5).

## 5. Code and pitfalls

- **`tools/apply_fixes.py <dir>`** applies every row; each old text must occur exactly once in its file, or it stops.
  Add a row only after the user's yes. Scratch test: `git -C REPO archive HEAD | tar -x -C <scratch>/fx`, then
  `python3 tools/apply_fixes.py <scratch>/fx`, `bash tools/layout_measure.sh <scratch>/fx`,
  `/root/miniconda3/envs/CoSiR/bin/python tools/free_lines.py <scratch>/fx`.
- **Page 8** has about 1.4 free lines. A clause of about 6 words in the conclusion costs a full line (built for A5).
  A7, A18 and A19 also edit page 8. **Page 5** takes no added line (A12 option 2, A15 and A25 sit there).
- **A25** adds two references. The bibliography is sorted by author, so most reference numbers shift.
- `.tex` sources hold one sentence per line, so the merged list's line numbers at `a56c4c1` still hold after the
  approved rows.
- **Subagents:** the first merge worker (Opus `xhigh`) ran out of context by reading every seat report. Brief a
  subagent narrowly and name the few files it may read.
- The rest as in the plan 31 handoff, section 5. The handoffs live in MAIN, a public repo: no co-author names here.

## 6. State at handoff

- Running: nothing. No subagent, background job or queued job of this chat.
- Uncommitted: nothing of ours (PF keeps the user's untracked `CBMI_IMP_CAMREADY (2).pdf` out of git).
- On disk over 1 GB: nothing. In this chat's scratchpad (41 MB, not needed after the fixes are in REPO): five scratch
  builds (`base`, `a1`, `fx`, `o2`, `o3`) and the merge worker's backup `panel_merged_before_partB.md`. The plan 31
  items in the old chat's scratchpad are unchanged (its handoff, section 6).
