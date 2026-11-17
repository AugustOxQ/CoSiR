# Handoff: run the new five-seat ARS review panel on the fixed paper text, then sort its findings for the user

## Brief
- Done: "instruction-tuned" in 8 places; three Claude checks of the merged text; the user's picks in (REPO `4c98972`).
- State: 8 pages, about 1.4 free lines on page 8, every float on its page; a reserve of 11 short-line cuts (about 6.7 lines).
- Next: the five-seat ARS panel on Fable `xhigh` (editor, R1 to R3 with new fields, Devil's Advocate), then one merged list.
- Blocked: the abstract (the user's morning talk with the co-authors), the Overleaf pull, OpenReview. Paper due 12:59.

> Written 2026-11-17 03:30 by the "paper plan 35" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens), with the user's yes. Direction log entry: none (paper work; decisions are in the paper folder's notes
> and the status table).

## 1. Read in this order

1. This handoff, all of it.
2. PF `notes/review/2026-11-16_fix_round.md`, section "Paper plan 35": this chat's decisions.
3. PF `notes/review/2026-11-16_panel_brief.md`: the brief of the 16 November panel. Reuse it, with the changes in
   section 4 below.
4. PF `notes/review/2026-11-17_fix_list.md`: what the three checks found and what is settled (seats may re-find these).
5. PF `user_read/2026-11-13_paper_status.md`: rows 2, 14, 16 and 25.
6. Project memory: `feedback_user-writes-paper.md`, `feedback_paper-chats-claude-only.md`.

PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.

Skills: `academic-research-skills:academic-paper-reviewer` (the panel; the chat starts with `claude-paper`).

## 2. Decided by the user, not to be reopened

All on 2026-11-17, 03:00 to 03:27; exact texts in `fix_round.md`.
- **The title is final:** "ANGLE: Learning Multiple Embedding Spaces From Instruction-Tuned Models". The body says
  "instruction-tuned" too (REPO `a841f87`, user's yes).
- **The panel:** five ARS seats, all on Fable `xhigh` (the user's yes covers the five calls): the editor (area chair
  for representation learning and retrieval at CVPR), R1 multimodal representation learning and embedding distillation
  (MLLM embedders such as VLM2Vec and Qwen3-VL-Embedding; losses, ablations, the comparison with the teacher, the cost
  claim), R2 conditional similarity and multiple clustering (novelty against the closest work, baselines), R3
  affective computing and computational art (labels, emotion evaluation, what "label-free" means), the Devil's
  Advocate. User, after seeing the recommendation.
- **No Ducktor seat and no pasted outside review this time** (user's own call, after the counts from 16 November).
- **Fixes applied** (REPO `4c98972`): R1, R2, R3a, R3b2, R3c, R4 to R13, O1 of `2026-11-17_fix_list.md`. The user keeps
  "achieves improved performance" in the conclusion (own call). O2 and O3 not taken.
- **The abstract** (status row 2) stays as it is until the user's morning talk with the co-authors. The points for
  that talk are in `fix_list.md`, "For the morning talk".
- Still in force: the claim rule (each result as strongly as the tables allow; change a sentence only where it is
  false or wider than its evidence); the co-authors' text wins except for such errors and plain slips; the way of
  working (exact old and new text, options, a recommendation, the user picks, a scratch build, commit and push); the
  LLM Usage Statement is never read; Claude subagents only, never Codex.

## 3. Where things stand

- REPO `4c98972` = origin. PF `2df7552` = origin. MAIN: this handoff.
- Built at `4c98972`: 11 pages in all, main text on 8, about 1.4 free lines on page 8 (was 0.4 at `34ec426`); floats p1
  Fig. 1, p3 Figs. 2 and 3, p6 Tables 1 to 3, p7 Tables 4 to 6, p8 Fig. 4 and Table 7; no overfull box, no undefined
  reference.
- The three checks of this chat (Sonnet `high` workers) and a fact lookup (Sonnet `medium`) are in PF `notes/review/
  2026-11-17_*.md`. Their facts for the supplementary are listed at the end of `fix_list.md` (row 15 of the status table
  does not list them yet).

## 4. Open points for the next step

Key, in order:
1. **Build the panel materials** in the new chat's scratchpad from REPO `4c98972`, as on 16 November (`panel_brief.md`,
   "Materials"): the page text of pages 1 to 8 cut before "LLM Usage", the references, page images p-01 to p-08, the
   sources, `main.tex` lines 1 to 55 only, and the supplementary draft for counter-evidence.
2. **Update the brief** before the launch: five seats, not six; the new title; page 8 has about 1.4 free lines plus the
   short-line reserve, the abstract may grow by one line at most; known points 1 to 7 as before, plus: the abstract
   points of the morning talk, the conclusion's "best" with the MMArt emotion tie (the user's pick), the intro's
   "aspect categories" sentence (A1, the co-authors' text), and the new supplementary facts of `fix_list.md`. The
   deadline is 12:59, so ask each seat to put "before the deadline" findings first.
3. **Launch the five seats in parallel**, each blind, with the field above. Then merge their reports into one list
   (Part A before the deadline, with exact old and new text and line costs), for the user's picks.
4. Leave time: the abstract fix after the morning talk, the last build check (row 16), the Overleaf pull and the
   OpenReview entry (rows 20, 21), the upload by 12:59.

## 5. Code and pitfalls

- **Fix rows:** PF `tools/apply_fix_rows.py <rows.json> <dir> <ids...>` applies `{id, file, old, new}` rows; each old
  text must be unique. In zsh, pass the IDs as separate words: an unquoted `$IDS` is not split, and the script then
  skips every row silently (this chat lost one build to it).
- **Scratch builds:** `git -C REPO ls-files -z | (cd REPO && xargs -0 tar -cf -) | tar -x -C <dir>`, then
  `bash PF/tools/layout_measure.sh <dir>`, `PF/tools/free_lines.py <dir>` and, for short last lines,
  `pdftotext -bbox-layout -f 1 -l 8 <dir>/main.pdf bb.html && PF/tools/find_short_lines.py bb.html`.
- **Never print page 9 or `main.tex` after line 55:** the LLM Usage Statement sits there. This chat once printed the
  top of page 9 by mistake; use `pdftotext -l 8` and cut before "LLM Usage".
- **Times:** `TZ=Europe/Amsterdam date`. The shell is zsh. The handoffs live in MAIN, a public repo: no co-author names.

## 6. State at handoff

- Running: nothing. No subagent, background job or queued job of this chat.
- Uncommitted: nothing of ours (PF keeps the user's untracked `CBMI_IMP_CAMREADY (2).pdf` out of git).
- On disk over 1 GB: nothing. This chat's scratchpad holds 781 MB of scratch builds (none needed: everything is in
  REPO and PF).
