# Handoff: the abstract and the conclusion after the user's morning talk, the last build check, then the upload by 12:59

## Brief
- Done: the five-seat panel on REPO `4c98972` (all borderline, no CRITICAL); the user's picks applied (REPO `ef21c90`).
- State: 8 pages, about 0.4 free lines on page 8, every float on its page; P10c would free one line.
- Next: wait for the user's morning talk; then the abstract (row 2) and P10c, a build check, Overleaf and OpenReview.
- Blocked: the abstract and P10c wait for the user's talk with the co-authors. Paper due 12:59.

> Written 2026-11-17 04:10 by the "paper plan 36" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens), with the user's yes. Direction log entry: none (paper work; decisions are in the paper folder's notes
> and the status table).

## 1. Read in this order

1. This handoff, all of it.
2. PF `notes/review/2026-11-16_fix_round.md`, section "Paper plan 36": this chat's decisions.
3. PF `notes/review/2026-11-17_fix_list.md`, section "For the morning talk": every point for the abstract and P10c.
4. PF `notes/review/2026-11-17_panel_merged.md`, the top part (seats, Part A tables, abstract inputs). The exact rows
   are in `2026-11-17_panel_rows.json`.
5. PF `user_read/2026-11-13_paper_status.md`: rows 2, 14, 16, 20, 21.
6. Project memory: `feedback_user-writes-paper.md`, `feedback_paper-chats-claude-only.md`.

PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.

Skills: none for the first step (text fixes and builds).

## 2. Decided by the user, not to be reopened

All on 2026-11-17; exact texts in `fix_round.md` and the rows file.
- **The panel's picks** (04:05, user's yes to the recommendation): P1b, P3a, P4a, P4b, P4c, P5b, P8a, P9a, P9b, P10a,
  P11a, P12a, P12c, P13a, P1d, P15a, in REPO `ef21c90`. Not taken: P2a, P2b, P7a+P7b, P8b, P17a, P6a, P1c, P14a, P2c,
  P3b, P5c.
- **Recurring findings** (03:52, own call): a Major or CRITICAL finding that an earlier review also raised comes back
  with full options, even on the co-authors' text and even when decided before.
- **Held for the morning talk:** P10c, the conclusion's "achieves improved performance" -> "achieves this" (frees one
  line). The user first took it for an abstract point; it is the conclusion's last sentence, and the user brings it to
  the talk with the abstract.
- Still in force from plan 35: the title "ANGLE: Learning Multiple Embedding Spaces From Instruction-Tuned Models"; the
  claim rule (each result as strongly as the tables allow; change a sentence only where it is false or wider than its
  evidence); the co-authors' text wins except for such errors and plain slips; the way of working (exact old and new
  text, options, a recommendation, the user picks, a scratch build, commit and push); the LLM Usage Statement is never
  read; Claude subagents only, never Codex.

## 3. Where things stand

- REPO `ef21c90` = origin. PF `2828f05` = origin. MAIN: this handoff.
- Built at `ef21c90`: 11 pages in all, main text on 8, about 0.4 free lines on page 8 (was 1.4 at `4c98972`; P10c held
  back costs the line); floats p1 Fig. 1, p3 Figs. 2 and 3, p6 Tables 1 to 3, p7 Tables 4 to 6, p8 Fig. 4 and Table 7;
  no overfull box, no undefined reference.
- The panel (all Fable `xhigh`, blind): EIC, R1, R2, R3 borderline at confidence 4, DA borderline at 3, no CRITICAL.
  Part B (supplementary) is in the merged file: that the three spaces differ, the emotion captions against the
  instruction, chance levels, one recipe per aspect with labels, a distillation baseline, seed spread.

## 4. Open points for the next step

Key, in order:
1. **The abstract** (status row 2), after the user's talk: the claim on emotion caption retrieval, "each aspect
   adapter ... their", the emotion graph through captions, "opens the door", and the panel inputs PA1 to PA4. The
   abstract may grow by one line at most. Exact old and new text, options, a recommendation; the user picks.
2. **P10c** (the conclusion's "achieves improved performance"), the user's call after the talk.
3. **Line budget:** about 0.4 free lines now. If the abstract adds a line and P10c is not taken, use the short-line
   reserve L1 to L11 of `2026-11-17_grammar_lines.md`; check each old text against `ef21c90` first (they were written
   at `34ec426`).
4. Then the last build check (row 16), the Overleaf pull and the OpenReview entry (rows 20, 21; the user's), the
   upload by 12:59.

## 5. Code and pitfalls

- **Fix rows:** PF `tools/apply_fix_rows.py <rows.json> <dir> <ids...>` applies `{id, file, old, new}` rows; each old
  text must be unique. In zsh, pass the IDs as separate words: an unquoted `$IDS` is not split, and the script then
  skips every row silently.
- **Scratch builds:** `git -C REPO ls-files -z | (cd REPO && xargs -0 tar -cf -) | tar -x -C <dir>`, then
  `bash PF/tools/layout_measure.sh <dir> | grep -v "^p9"`, `python3 PF/tools/free_lines.py <dir>` and, for short last
  lines, `pdftotext -bbox-layout -f 1 -l 8 <dir>/main.pdf bb.html && python3 PF/tools/find_short_lines.py bb.html`.
- **Never print page 9 or `main.tex` after line 55:** the LLM Usage Statement sits there. Use `pdftotext -l 8`.
- **REPO commit `ef21c90` has no body or attribution lines** (pushed; not rewritten). Later REPO commits keep the
  usual body.
- **Times:** `TZ=Europe/Amsterdam date`. The shell is zsh. The handoffs live in MAIN, a public repo: no co-author names.

## 6. State at handoff

- Running: nothing. No subagent, background job or queued job of this chat.
- Uncommitted: nothing of ours (PF keeps the user's untracked `CBMI_IMP_CAMREADY (2).pdf` out of git).
- On disk over 1 GB: nothing. This chat's scratchpad holds 148 MB of scratch builds (none needed: everything is in
  REPO and PF).
