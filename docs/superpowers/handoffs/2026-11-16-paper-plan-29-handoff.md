# Handoff: wait for the user's own abstract, test it (Figure 3 move if needed), then the last build check before the upload

## Brief
- Done: rows 11, 12, 13 and C3 settled with the user (REPO `bea5a4c`, `c22519a`); status table updated (PF `eb41e37`).
- State: 8 pages, every float in place, no overfull box, about 3.4 free lines on page 8.
- Next: the user's abstract: test in a scratch copy, move Figure 3 if it slips, fact check its numbers, apply with a yes.
- Blocked: the user's abstract, the Overleaf pull, OpenReview. Paper due 2026-11-17 12:59 Amsterdam.

> Written 2026-11-16 18:30 by the "paper plan 28" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens), at the user's yes. Direction log entry: none (paper work; decisions are rows in the paper folder's
> notes and the status table).

## 1. Read in this order

1. This handoff, all of it.
2. PF `user_read/2026-11-13_paper_status.md`: the status table (updated 18:30). Open rows: 2 (abstract), 16 (last
   build check), 19 (co-authors' new points), 20 (OpenReview), 21 (Overleaf pull). Keep it current.
3. PF `notes/supplementary_items.md`: what goes to the supplementary (due 2026-11-24), with today's new rows.
4. Project memory `feedback_user-writes-paper.md` and `project_paper-name-angle.md`.

Skills: none.

## 2. Decided by the user, not to be reopened

All on 2026-11-16 (status table rows 11 to 13, 16):
- **Figure 4:** "Plain SigLIP" label; every neighbour a square crop (left or top window for three paintings), the
  query whole; caption and Sec. 4.6 say "plain SigLIP embedding" (REPO `bea5a4c`).
- **Conclusion:** stays as it is, no limitation sentence (a one-line limitation reads like a template).
- **Row 13:** C5 no change; C6 kept (61st and 66th percentiles stay); C7 "Encoder" for "Backbone" and C13 "cf.~"
  (REPO `c22519a`), all equations stay numbered; B6, B10, B11 details go to the supplementary, main text unchanged.
- **C3:** both ArtELingo captions stay in Figure 2.
- **Open, for after the supplementary experiments:** whether the supplementary says that the final recipe was adopted
  after its test result (direction log 2026-10-27). Do not raise it before then.
- Still in force: the abstract is the user's own text (agent versions A and B are reference only; propose no abstract
  text unless asked); title "ANGLE: Learning Multiple Embedding Spaces From Instructed Embeddings"; limitations in the
  supplementary; no notes of ours in REPO `.tex` files; each change: exact text, the user's yes, test in a scratch
  copy, build, check float pages, commit, push. The LLM Usage Statement is the user's: agents never read or comment on
  it.

## 3. Where things stand

- REPO `c22519a` = origin, clean. PF `eb41e37` = origin (the user's untracked `CBMI_IMP_CAMREADY (2).pdf` stays out
  of git). MAIN clean apart from old untracked files.
- `main.pdf` 11 pages (8 + references + LLM statement). Floats: p1 Fig. 1; p3 Figs. 2, 3; p6 Tables 1 to 3; p7 Tables
  4 to 6; p8 Fig. 4, Table 7. About 3.4 free lines on page 8, "future work." at ruler line 562.
- Measured earlier: an abstract about 2 lines longer than today's pushes Figure 3 to page 4; `move_fig3.py` keeps it
  on page 3 and leaves about 1.4 free lines.
- Measured today: the B6c clause ("an extra term raised the genre AMI in our development runs on training paintings,
  and without it the genre head keeps little of its gain in genre retrieval on the test split") costs 2 lines. Offer
  it only if the abstract leaves at least 2 free lines on page 8 (the user's yes to "look again after the abstract").

## 4. Open points for the next step

Key, in order:
1. **The user's abstract.** Apply it in a scratch copy, build, check the float pages. If Figure 3 is on page 4, run
   `move_fig3.py` on the scratch copy and check again. Fact check its numbers against the tables (emotion AMI 0.089 to
   0.099 on ArtELingo; 266.8 ms against 19.5 ms per image, about 14 times; no retrieval gain claim for emotion, it is
   lower than the instructed Qwen3's). Report problems only, no rewording. Show the user, apply with their yes.
2. **The B6c clause**, only if room is left (section 3).
3. **The last build check** after the abstract and any co-author edits: pages, float pages, overfull boxes, undefined
   references, anonymity (no names in the PDF metadata), the LLM statement section present (`grep -c 'LLM Usage'
   main.tex`, never read it).
4. **The user:** the Overleaf pull (then delete the merged branch `overleaf-2026-11-16-1631` on GitHub), the
   OpenReview title and abstract, the upload.

After the deadline: the supplementary (status row 15, `notes/supplementary_items.md`) and the limitations (row 18).

## 5. Code and pitfalls

- PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.
- **Scripts** (PF `tools/`): `layout_measure.sh <dir>` (build, float pages, ruler line, overfull and undefined
  counts); `free_lines.py <dir>` (free lines on page 8; CoSiR env python); `move_fig3.py <dir>`. Test copies:
  `git -C REPO archive HEAD | tar -x -C <scratch>/x`.
- Figure 4's source is PF `figures/src/fig_qualitative.py` (writes PF `figures/fig_qualitative.pdf`; copy it to REPO
  `Figures/`). The crop windows are in its `ANCHOR` dict.
- Bottom floats slip to the next page when text above their source grows (Tables 3, 6, 7, Fig. 3 are `[b]`).
  Figure 3 is the first to slip when page 1 grows. A one-clause addition in Secs. 3 or 4 costs about 2 lines.
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
- On disk over 1 GB: nothing. This chat's scratchpad holds 86 MB of scratch builds
  (`/tmp/claude-0/-project-CoSiR/bb022a48-390a-4d9c-8d0e-783cbd593f9c/scratchpad/`); everything needed is in git.
