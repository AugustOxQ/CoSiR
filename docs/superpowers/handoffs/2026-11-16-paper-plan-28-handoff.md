# Handoff: wait for the user's own abstract, keep Figure 3 on page 3, then the last build check before the upload

## Brief
- Done: final fact check and both cold reads reviewed; 10 fixes and options a to d applied (REPO `f37339e`,
  `cc6d4e5`); status table updated (PF `78e31d3`).
- State: 8 pages, every float in place, no overfull box, about 3.4 free lines on page 8.
- Next: the user writes the abstract; if it is about 2 lines longer, move Figure 3's source up, then a last build check.
- Blocked: the user's abstract, the Overleaf pull, the co-author talk. Paper due 2026-11-17 12:59 Amsterdam.

> Written 2026-11-16 18:05 by the "paper plan 27" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens), at the user's yes. Direction log entry: none (paper work; decisions are rows in the paper folder's
> notes and the status table).

## 1. Read in this order

1. This handoff, all of it.
2. PF `user_read/2026-11-13_paper_status.md`: the status table (updated 18:00); rows 2, 11 to 16, 21 and 24 are
   today's. Its "What's left" order is the plan. Keep it current.
3. PF `notes/review/2026-11-16_fix_list.md`: what was applied, what was dropped, and the supplementary items.
4. PF `notes/2026-11-16_final_list.md`: rows a to h and i to iv (ii done, iv done: the user's statement is in
   `main.tex`).
5. Project memory `feedback_user-writes-paper.md` and `project_paper-name-angle.md`.

Skills: none.

## 2. Decided by the user, not to be reopened

All on 2026-11-16:
- **The abstract is the user's own text.** Neither agent version (A or B) is used; they stay as reference in
  `notes/2026-11-16_final_list.md`. Do not propose abstract text unless the user asks.
- **Applied, with the user's yes:** fixes 1 to 10 and options a, b, d (REPO `f37339e`), option c as the dashes only
  (REPO `cc6d4e5`). Option e (the weight for a node with no emotion neighbours, +3 lines) goes to the supplementary.
- Still in force: the title "ANGLE: Learning Multiple Embedding Spaces From Instructed Embeddings"; "label-free"
  defined once in the intro; no "direction" for retrieval; "seed" only in the setting; limitations in the
  supplementary (due 2026-11-24); no notes of ours in REPO `.tex` files; each change: exact text, the user's yes, test
  in a scratch copy, build, check float pages, commit, push. The LLM Usage Statement is the user's: agents never read
  or comment on it.

## 3. Where things stand

- REPO `cc6d4e5` = origin, clean. PF `72d86ac` = origin (the user's untracked `CBMI_IMP_CAMREADY (2).pdf` stays out
  of git).
- `main.pdf` 11 pages (8 + references + LLM statement). Floats: p1 Fig. 1; p3 Figs. 2, 3; p6 Tables 1 to 3; p7 Tables
  4 to 6; p8 Fig. 4, Table 7. About 3.4 free lines on page 8 (from word positions), "future work." at ruler line 562.
- Measured today: abstract A (+39 words) or B (+20 words) each cost about 2 lines and push Figure 3 to page 4. With
  Figure 3's source block moved up (`tools/move_fig3.py`), Figure 3 stays at the bottom of the right column of page 3
  (page image checked), every float in place, about 1.4 free lines.

## 4. Open points for the next step

Key, in order:
1. **The user's abstract.** When it comes: apply it in a scratch copy, build, check the float pages. If Figure 3 is on
   page 4, run `move_fig3.py` on the scratch copy, check again, show the user, then apply to REPO with their yes.
   A fact check of the new abstract against the tables is useful (numbers: emotion AMI 0.089 to 0.099 on ArtELingo;
   266.8 ms against 19.5 ms per image; no retrieval gain claim for emotion, it is lower than the instructed Qwen3's).
2. **The Overleaf pull** (the user): GitHub into Overleaf, then the branch `overleaf-2026-11-16-1631` can be deleted
   on GitHub (it is merged). If Overleaf has edits of its own, follow the sync steps in section 5.
3. **The co-author talk** (17 November): Figure 4(a) and its "Plain image" label; C3 (ArtELingo captions in Figure 2);
   open review rows B6, B10, B11, C5, C6, C7, C13 (row 13 of the status table). Any edit: the usual test and yes.
4. **The last build check** after the abstract and the co-author edits: pages, float pages, overfull boxes, undefined
   references, anonymity (no names in the PDF metadata), the LLM statement section present (`grep -c 'LLM Usage'
   main.tex`, never read it).
5. OpenReview title and abstract, then the upload (the user).

After the deadline: the supplementary (status row 15, `notes/supplementary_items.md`) and the limitations (row 18).

## 5. Code and pitfalls

- PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.
- **Scripts** (PF `tools/`): `layout_measure.sh <dir>` (build, float pages, ruler line, overfull and undefined
  counts); `free_lines.py <dir>` (free lines on page 8; run with the CoSiR env python); `move_fig3.py <dir>` (the
  Figure 3 move). Test copies: `git -C REPO archive HEAD | tar -x -C <scratch>/x`.
- Bottom floats slip to the next page when text above their source grows (Tables 3, 6, 7, Fig. 3 are `[b]`).
  Figure 3 is the first to slip when page 1 grows.
- **Overleaf sync:** when Overleaf has unpushed edits, the user pushes from Overleaf first; on a clash Overleaf pushes
  a branch `overleaf-<date>-<time>`. Merge it in a scratch worktree (`git worktree add --detach <scratch>/merge main`),
  resolve with the user, commit, then `git merge --ff-only` in REPO and push. git here is 2.34.
- Paper prose: no semicolons between clauses, no dashes, British spelling, no win/level/lose words, no "significant",
  novelty only "to our knowledge"; number pairs "from X to Y". Build `latexmk -pdf -interaction=nonstopmode -silent
  main.tex`, then `latexmk -f … supp.tex` (two harmless `xr-hyper` errors).
- The handoffs live in MAIN, a public repo: no co-author names here.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing of ours.
- On disk over 1 GB: nothing. This chat's scratchpad holds 272 MB of scratch builds
  (`/tmp/claude-0/-project-CoSiR/7de77c4a-9ee8-47b2-b27d-9f42cce9e56d/scratchpad/`); everything needed is in git.
