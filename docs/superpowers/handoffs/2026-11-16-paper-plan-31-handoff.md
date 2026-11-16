# Handoff: merge the user's own review into the panel's list, then go through the fixes before the deadline

## Brief
- Done: the co-author's label-free sentence merged, and Figure 3's source block moved (REPO `a56c4c1`). A second review panel ran: eight Fable `xhigh` reviewers, each blind to the others, plus an Opus `xhigh` merge (PF `a0c8f93`).
- State: 8 pages, every float in place, about 1.4 free lines on page 8. The panel says borderline. Every number holds. The devil's advocate's critical point is validated.
- Next: the user sends their own review. Merge it into the panel's list, then go through Part A (24 groups) with the user, one exact text at a time.
- Blocked: the user's review and abstract, the Overleaf pull, OpenReview. Paper due 2026-11-17 12:59 Amsterdam.

> Written 2026-11-16 20:00 by the "paper plan 30" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens). Direction log entry: none (paper work; decisions are rows in the paper folder's notes and the status
> table).

## 1. Read in this order

1. This handoff, all of it.
2. PF `user_read/2026-11-13_paper_status.md`: the status table. Open rows: 2 (abstract), 16 (last build check), 19
   (co-author point 6), 20 (OpenReview), 21 (Overleaf pull), 25 (the second review panel).
3. PF `notes/review/2026-11-16_panel_merged.md`: the merged list. Its head, the critical item, Part A and the line
   budget first. Open a seat's report (`2026-11-16_panel_<seat>.md`) only for the exact text of a fix.
4. PF `notes/review/2026-11-16_panel_brief.md`: the brief every reviewer got (the known decisions, the line rules).
5. The previous handoff, `docs/superpowers/handoffs/2026-11-16-paper-plan-30-handoff.md`, sections 2 and 5.
6. Project memory: `feedback_user-writes-paper.md`, `project_paper-name-angle.md`, `feedback_paper-chats-claude-only.md`.

Skills: none. The chat starts with `claude-paper`, so ARS is loaded if a further ARS step is needed.

## 2. Decided by the user, not to be reopened

All on 2026-11-16, evening:
- **The co-author's label-free sentence stays for now**, and Figure 3's source block is moved up to keep the figure on
  page 3 (REPO `a56c4c1`). The panel's A1 now asks to reword it: the user decides that in the fix round, as for any
  other text.
- **The Figure 1 caption stays as it is** ("Nearest neighbours of a query ...", REPO `7c5097d`). The panel's points
  on Figure 1 go to the supplementary (how the query and neighbours were chosen).
- **The second panel:** the ARS full review with real, separate seats this time (the review of 16 November, 05:05,
  ran all five seats in one Sonnet context). The user added two seats: LLM research (R4) and general multimodal
  learning (R5). All reviewers on Fable `xhigh` (the user's explicit choice for this review).
- **The user's own review** (done outside, from the PDF of REPO `a56c4c1`) is merged into the same list in the next
  chat.
- Still in force: the abstract and the caption are the user's own text (propose abstract text only when asked; the
  panel's abstract points are inputs, and Ducktor's 168-word rewrite is reference only); each change gets exact text,
  the user's yes, a scratch-copy test, a build, a float-page check, a commit and a push; the LLM Usage Statement is
  the user's, never read or commented on; in this paper's chats, Claude subagents only, never Codex.

## 3. Where things stand

- REPO `a56c4c1` = origin. PF `a0c8f93` + this cut's commit (status table, panel brief) = origin. MAIN: this handoff.
- `main.pdf` 11 pages. Floats: p1 Fig. 1; p3 Figs. 2, 3; p6 Tables 1 to 3; p7 Tables 4 to 6; p8 Fig. 4, Table 7.
  About 1.4 free lines on page 8, "future work." at ruler line 564.
- **Room:** the abstract may grow to 19 lines (about 20 words more); at 20 lines the conclusion's last line goes to
  page 9 (measured on REPO `a56c4c1` with filler words). The devil's advocate found in its test build that one more
  line on page 5 also pushes the conclusion to page 9. The B6c clause (2 lines) no longer fits.
- **The panel (merged list):** borderline on CVPR's scale (ARS: Major Revision). Votes: seven borderline (R5 leaning
  weak reject), the devil's advocate weak reject for today's text, borderline after the fixes.
  - The critical item (A1, all eight reviewers): introduction line 22 "No method ... learns several aspects without
    annotation", while line 20 counts "human votes" as annotation and the emotion graph reads annotators' captions;
    also the co-author's "only require the aspects themselves" (line 26). Checked by the paper chat against the
    source. The devil's advocate's wording fixes it at 0 lines (built).
  - Then A2 (abstract inputs, +1 line built), A3 ("Other aspects can join", conclusion line 7), A4 (no chance level
    for mAP), A5 (graphs credited for genre, Table 6 credits the relational term), A6 (paintings-only photo rows not
    mentioned), A7 to A17 major, A18 to A24 minor. Each about 0 lines; together zero net if nothing is added on page 5.
  - Four reviewer claims rejected by the merge (VLM2Vec-V2 on emotion, some line numbers, "labelled split" for k = 80,
    "ten methods" on MMArt).
  - Seven reviewers flag "CoSiR (ours)" in the legend of supplementary Fig. S1: already status row 15 (due 24
    November, not uploaded on 17 November).

## 4. Open points for the next step

Key, in order:
1. **Merge the user's review.** When the user sends it, a Claude subagent (Opus `xhigh`, as the first merge) adds it
   to `2026-11-16_panel_merged.md`: same groups, the user's points as a ninth source, new groups where needed. The
   user's own points are decisions to weigh first, not reviewer claims.
2. **Part A with the user**, in the merged order (A1, A3, A5, A6 first; A2 is input to the user's abstract). Per fix:
   the exact old and new text (from a seat's report or the user), the user's yes, then all approved fixes together in
   a scratch copy, build, `layout_measure.sh`, `free_lines.py`, float pages, commit, push.
3. **The user's abstract**: scratch copy, fact check (emotion AMI 0.089 to 0.099 on ArtELingo; 266.8 ms against 19.5
   ms per image, about 14 times; no retrieval gain claim for emotion), at most 19 lines.
4. **The last build check** (status row 16): pages, float pages, overfull boxes, undefined references, anonymity (no
   names in the PDF metadata), the LLM statement present (`grep -c 'LLM Usage' main.tex`, never read it).
5. **The user:** the Overleaf pull (then delete the merged branch `overleaf-2026-11-16-1631` on GitHub), the OpenReview
   title and abstract, the upload.

After the deadline: Part B of the merged list goes to the supplementary (status row 15) and the limitations (row 18),
due 2026-11-24 12:59.

## 5. Code and pitfalls

- PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.
- Layout tools in PF `tools/`: `layout_measure.sh <dir>`, `free_lines.py <dir>` (CoSiR env python), `move_fig3.py`
  (already applied in REPO; do not run it again). Test copies: `git -C REPO archive HEAD | tar -x -C <scratch>/x`.
- The review copy of REPO `a56c4c1` (page texts without the LLM statement, page images, the brief) is in the old
  chat's scratchpad, `/tmp/claude-0/-project-CoSiR/8e6846bc-5c19-4993-ba28-51fd3d82c48e/scratchpad/review_a56c4c1/`
  (14 MB). The merged list's paths point there. It can go after the deadline.
- The merge worker gives `.tex` line numbers at REPO `a56c4c1`; recheck them after any edit above them.
- Reviewer seats R2, R4 and R5 ran web searches with topic words only. Papers they name as missing are "not checked"
  unless the merged list says otherwise; check a citation before adding it to `refs.bib`.
- The Overleaf sync: when Overleaf has unpushed edits, the user pushes first; on a clash Overleaf pushes a branch
  `overleaf-<date>-<time>`. A new co-author edit can cost lines: measure after every pull.
- Paper prose rules and the build commands: the plan 29 handoff, section 5.
- The handoffs live in MAIN, a public repo: no co-author names here.

## 6. State at handoff

- Running: nothing. No subagent, background job or queued job of this chat.
- Uncommitted: nothing of ours (PF keeps the user's untracked `CBMI_IMP_CAMREADY (2).pdf` out of git).
- On disk over 1 GB: nothing. Deleted with the user's yes: the devil's advocate's test builds (259 MB). Left in the old
  chat's scratchpad, not needed after the deadline: the review copy (14 MB), seven scratch builds (8 MB each), the two
  PDF copies sent to the user (4.6 MB), Ducktor's and R4's working files (small).
