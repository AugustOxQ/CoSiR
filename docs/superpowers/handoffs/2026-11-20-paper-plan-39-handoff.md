# Handoff: the supplementary, from the six open points of the row 15 list (due 24 November, 12:59 Amsterdam)

## Brief
- Done: the list for status row 15, with exact texts: 11 fix rows, tested together (the supplementary builds again).
- State: nothing applied to REPO yet; REPO is still `adfdb3c`, the uploaded paper. The paper submission is closed.
- Next: show the user the six open points of PF `notes/review/2026-11-17_supp_list.md` and apply their picks.
- Blocked: nothing; the user picks each change. Supplementary due 24 November 12:59.

> Written 2026-11-20 16:36 by the "paper plan 38" tab (a paper chat, outside the loop). Cut at the user's request
> ("submission is done already, let's move to sup stuff"). Direction log entry: none (paper work; decisions are in
> PF `notes/review/2026-11-16_fix_round.md`, "Paper plan 38").

## 1. Read in this order

1. This handoff, all of it.
2. PF `notes/review/2026-11-17_supp_list.md`: the whole list (Part 1 rows SP1 to SP11, the Figure S1 legend, the
   numbering, Part 2: C2, the old teaser, the Affection example). The rows are in `2026-11-17_supp_rows.json`.
3. PF `notes/review/2026-11-17_supp_facts.md`: the facts behind C2 and the old teaser (spot-checked).
4. PF `notes/review/2026-11-16_fix_round.md`, sections "Paper plan 37" and "Paper plan 38".
5. PF `user_read/2026-11-13_paper_status.md`, rows 15, 18 and 19, only when needed for the later steps.
6. Project memory: `feedback_user-writes-paper.md`, `feedback_paper-chats-claude-only.md`.

PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.

Skills: none for the first step (show the six points, apply the picks).

## 2. Decided by the user, not to be reopened

- **The submission is closed** (2026-11-20). The uploaded paper is REPO `adfdb3c`. Its Figure 1 caption ("we show its
  four nearest neighbours") stays, although the four shown are at ranks 4 to 20 of the 20 nearest. The supplementary's
  C2 text says how they were picked; write it so it reads well beside that caption.
- From paper plan 37, still in force: Figure S1's legend says "ANGLE (ours)", never "CoSiR"; the MMArt reference
  stays; the heading "6. LLM Usage Statement" stays.
- Still in force: the user writes the paper (agents guide, check, and give exact old and new text with options and a
  recommendation; the user picks; a scratch build; commit and push); the claim rule (each result as strongly as the
  tables allow); the co-authors' text wins except for errors and plain slips; the LLM Usage Statement is never read;
  Claude subagents only, never Codex.

## 3. Where things stand

- REPO `adfdb3c` = origin, untouched by this chat. PF `56a4735` = origin. MAIN: this handoff.
- The list's Part 1 (SP1 to SP11), tested together on a scratch copy of `adfdb3c`: `supp.tex` builds, 3 pages, no
  undefined reference, no overfull box. Before, it failed on the `xr-hyper` import (fixed by SP2).
- Table S2 floats alone to page 3 of the supplementary; fix the float layout after the new material is in.
- Figure S1's values (MAIN `src/test/20261111_cluster_count_range/figures/siglip1/values.md`): ANGLE highest at 11 of
  13 cluster counts, level at style 10 (CLAY 0.4095, ANGLE 0.4094) and genre 20 (ANGLE 0.5065, instructed Qwen3
  0.5062).

## 4. Open points for the next step

Key, in order. The six points of the list, each with a recommendation in the list file:
1. Rows SP1 to SP11 and the numbering (recommended: all rows; keep S1, Table S1, Figure S1).
2. Figure S1's legend: a, Table 2's names, or b, the encoder in the name (recommended: a). A rebuild with MAIN
   `src/test/20261111_cluster_count_range/kr_figure.py` (label constants `S1_*`, lines 70 to 80) into REPO
   `Figures/fig_supp_cluster_counts.pdf`.
3. C2, how Figure 1 was chosen: option a (full paragraph) or b (one sentence) (recommended: a).
4. The old teaser a10 as "the four nearest neighbours without picking" (recommended: yes).
5. The Affection example: a, candidate 12; b, plus EmoSet; c, none (recommended: a; photo licences unknown).

Then, in this order: the details of B6, B10 and B11 (row 13); the 17 November fix list's "For the supplementary"
(PF `notes/review/2026-11-17_fix_list.md`); the limitations (row 18, agent proposes); the panel's Part B
(PF `notes/review/2026-11-17_panel_merged.md`); the recipe-disclosure sentence (the user decides after more
supplementary experiments); the float layout; an anonymity check of the supplementary PDF and its figures before
upload (as PF `notes/review/2026-11-17_anonymity_check.md` did for the paper).

## 5. Code and pitfalls

- **Build the supplementary after `main.tex`:** SP2 imports `main.aux`, so build main first in the same folder.
- **Never print page 9 of `main.pdf` or `main.tex` after line 55:** the LLM Usage Statement sits there.
- **One combined `\Cref` across files** prints "Figures S2 and 4" (cleveref sorts). Use two `\cref` (row SP9).
- **Fix rows:** PF `tools/apply_fix_rows.py <rows.json> <dir> <ids...>`; each old text must be unique. In zsh, pass the
  IDs as separate words.
- **Scratch builds:** `git -C REPO ls-files -z | (cd REPO && xargs -0 tar -cf -) | tar -x -C <dir>`, then
  `latexmk -pdf main.tex` and `latexmk -pdf supp.tex` in that folder.
- **The Affection panel:** reuse the Figure 4(a) rule as MAIN `src/test/20261116_affection_examples/make_candidates.py`
  did; join rows by the IDs, never by position (constitution C6).
- **Times:** `TZ=Europe/Amsterdam date`. The shell is zsh. The handoffs live in MAIN, a public repo: no co-author names.

## 6. State at handoff

- Running: nothing. No subagent, background job or queued job of this chat.
- Uncommitted: nothing of ours (PF keeps the user's untracked `CBMI_IMP_CAMREADY (2).pdf` and
  `submission/2026-11-17_main_adfdb3c.pdf` out of git).
- On disk over 1 GB: nothing. This chat's scratch builds (25 MB) are in its session scratchpad, which is cleared
  with the session.
