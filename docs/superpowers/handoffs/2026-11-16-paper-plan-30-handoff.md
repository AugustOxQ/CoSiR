# Handoff: wait for the user's Figure 1 caption and abstract, test them, then the last build check before the upload

## Brief
- Done: new Figure 1 (draft F, REPO `cbc0e23`): neighbours picked from the 20 nearest to vary in content; old figure and all drafts kept.
- State: 8 pages, every float in place, no overfull box, about 3.4 free lines on page 8.
- Next: the user's Figure 1 caption and abstract: test each in a scratch copy, fact check, apply with a yes.
- Blocked: the user's caption and abstract, the Overleaf pull, OpenReview. Paper due 2026-11-17 12:59 Amsterdam.

> Written 2026-11-16 19:12 by the "paper plan 29" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens). Direction log entry: none (paper work; decisions are rows in the paper folder's notes and the status
> table).

## 1. Read in this order

1. This handoff, all of it.
2. PF `user_read/2026-11-13_paper_status.md`: the status table. Open rows: 2 (abstract), 4 (Figure 1 caption), 16
   (last build check), 19 (co-author point 6, any new points), 20 (OpenReview), 21 (Overleaf pull).
3. PF `figures/teaser_candidates/README.md`: how Figure 1 was found, the drafts, the command.
4. The previous handoff, `docs/superpowers/handoffs/2026-11-16-paper-plan-29-handoff.md`, sections 2 and 5 (decisions
   and pitfalls still in force).
5. Project memory: `feedback_user-writes-paper.md`, `project_paper-name-angle.md`, `feedback_paper-chats-claude-only.md`.

Skills: none.

## 2. Decided by the user, not to be reopened

All on 2026-11-16, evening:
- **Figure 1 is draft F** (query Bogdanov-Belsky, a boy under a tree): four neighbours per aspect picked from the 20
  nearest, all of the query's class; header "Nearest neighbours"; no "4/4" counts. The rule behind the emotion row (the
  user): if the query shows people, one or two neighbours show none; if outdoors, one is indoors (and the reverse).
- **The Figure 1 caption is the user's own text.** The current caption still says "Four nearest neighbours", which is
  no longer true. Propose no caption text unless asked; check theirs for facts only (below).
- **Keep the old Figure 1 and all drafts A to G** for possible use in the supplementary (PF `notes/supplementary_items.md`).
- **Co-author point 8:** the Affection and EmoSet photo examples go to the supplementary. Row 19: all points solved
  except point 6 (the overview's remaining issues, for the co-author talk).
- **In this paper's chats, delegate to Claude subagents only, never Codex** (this paper only, not a general rule).
- Still in force from the previous handoff: the abstract is the user's own text; each change gets exact text, the
  user's yes, a scratch-copy test, a build, a float-page check, a commit and a push; the LLM Usage Statement is the
  user's, never read or commented on.

## 3. Where things stand

- REPO `cbc0e23` = origin. PF `94ccd96` = origin (the user's untracked `CBMI_IMP_CAMREADY (2).pdf` stays out of git).
  MAIN `3ff67b44` (teaser search scripts) = origin.
- `main.pdf` 11 pages (8 + references + LLM statement). Floats: p1 Fig. 1; p3 Figs. 2, 3; p6 Tables 1 to 3; p7 Tables
  4 to 6; p8 Fig. 4, Table 7. About 3.4 free lines on page 8, "future work." at ruler line 562. The new Figure 1 is
  0.04 in shorter than the old one; the text flow did not change.
- **Room for the abstract** (measured 2026-11-16 with filler words): today 18 lines. At 19 or 20 lines Figure 3 goes
  to page 4; `tools/move_fig3.py` keeps it on page 3 (1.4 free lines left). At 21 lines the conclusion's last line
  goes to page 9. So the limit is 20 lines, about 20 words more than today. A longer caption of Figure 1 also costs
  lines on page 1: measure caption and abstract together.
- The B6c clause (2 lines) fits only if the abstract and caption leave at least 2 free lines on page 8.

## 4. Open points for the next step

Key, in order:
1. **The user's Figure 1 caption.** Apply in a scratch copy, build, check the float pages. Fact check: picked from the
   20 nearest; all four share the query's class in each aspect; no painting in two rows; public-domain paintings of
   the ArtELingo test set; picks vary in genre (emotion and style rows) or style (genre row); class shares of the
   query's 20 nearest: emotion 14, style 16, genre 9. Report problems only.
2. **The user's abstract**, as in the previous handoff (section 4.1): scratch copy, `move_fig3.py` if Figure 3 slips,
   fact check (emotion AMI 0.089 to 0.099 on ArtELingo; 266.8 ms against 19.5 ms per image, about 14 times; no
   retrieval gain claim for emotion).
3. **The B6c clause**, only if room is left.
4. **The last build check** after the caption, the abstract and any co-author edits: pages, float pages, overfull
   boxes, undefined references, anonymity (no names in the PDF metadata), the LLM statement section present
   (`grep -c 'LLM Usage' main.tex`, never read it).
5. **The user:** the Overleaf pull (then delete the merged branch `overleaf-2026-11-16-1631` on GitHub), the
   OpenReview title and abstract, the upload.

After the deadline: the supplementary (status row 15) and the limitations (row 18), due 2026-11-24 12:59.

## 5. Code and pitfalls

- PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.
- Layout tools in PF `tools/`: `layout_measure.sh <dir>`, `free_lines.py <dir>` (CoSiR env python), `move_fig3.py
  <dir>`. Test copies: `git -C REPO archive HEAD | tar -x -C <scratch>/x`.
- Figure 1's script: PF `figures/src/fig_teaser_t2.py` (new options `--data`, `--header`, `--stem`, `--no-count`,
  `--qw`; defaults draw the old figure pixel-identical). The exact command is in `figures/teaser_candidates/README.md`;
  copy `figures/fig_teaser_f.pdf` to REPO `Figures/fig_teaser.pdf`.
- The zero-shot people tag (round 4) misses small or seated figures; check any new pick by eye.
- Paper prose rules, the Overleaf sync and the build commands: previous handoff, section 5.
- The handoffs live in MAIN, a public repo: no co-author names here.

## 6. State at handoff

- Running: nothing. No subagent, background job or queued job of this chat.
- Uncommitted: nothing of ours.
- On disk over 1 GB: nothing. New this chat: the teaser search outputs, 53 MB in
  `/data/SSD2/cosir_r28/20261112_siglip_main/qual/teaser/` (`teaser_t3*` to `teaser_t6*`, kept: the drafts' data);
  7.9 MB of drafts in PF `figures/teaser_candidates/` (in git); 70 MB of scratch builds in this chat's scratchpad
  (`/tmp/claude-0/-project-CoSiR/3d2feffb-acc6-46a3-a989-c72d49a62c1d/scratchpad/`), not needed.
