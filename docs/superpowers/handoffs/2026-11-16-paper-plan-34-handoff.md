# Handoff: go on with the rest of Part A from A8, under the new claim rule; merge into REPO after the co-authors finish

## Brief
- Done: A24, A6 and A10 (reopened, bolder) and nine underclaim fixes approved (PF `1363cd9`); a co-author asked for bolder claims, so A4 and A7 are dropped.
- State: all rows built together: every float on its page, no overfull box, but the main text runs about 4 lines onto page 9 (lines not weighed now, the user).
- Next: A8, A9, A11, A12, A14, A16, A17, A18, A20 to A23 with the user; REPO only after the co-authors finish in Overleaf.
- Blocked: the co-authors' Overleaf edits, the user's abstract, OpenReview. Paper due 2026-11-17 12:59 Amsterdam.

> Written 2026-11-16 21:25 by the "paper plan 33" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens), with the user's yes. Direction log entry: none (paper work; decisions are rows in the paper folder's
> notes and the status table).

## 1. Read in this order

1. This handoff, all of it.
2. PF `user_read/2026-11-16_review_changes.md`: the tracking table of Part A the user reads (new this chat). Keep it
   current after each decision.
3. PF `notes/review/2026-11-16_fix_round.md`: every approved fix; the sections from "The co-author's underclaim point"
   on are this chat's.
4. PF `notes/review/2026-11-16_panel_merged.md`: each open group when you reach it (A8 at line 343). Open a seat's
   report only for the exact text of a fix.
5. PF `notes/review/2026-11-16_underclaim_scan.md`: the scan, with the verdict table at its top.
6. The plan 33 handoff (`2026-11-16-paper-plan-33-handoff.md`), sections 2 and 5.
7. Project memory: `feedback_user-writes-paper.md`, `project_paper-name-angle.md`, `feedback_paper-chats-claude-only.md`.

Skills: none. The chat starts with `claude-paper`.

## 2. Decided by the user, not to be reopened

All on 2026-11-16 evening, "user, after seeing the recommendation" unless marked; rows in `fix_round.md` and
`tools/apply_fixes.py`:
- **The claim rule** (the co-author: "In many places you seem to underclaim"): state each result as strongly as the
  tables allow; change a sentence only where it is false or wider than its evidence. So A4 and A7 are dropped, and A18
  becomes "not seen during training" (u14 in the introduction already says "not seen in training").
- **Line cost** (user's own note): do not weigh the page limit when proposing fixes; the many short last lines get
  tightened later, and the co-authors' edits change the page anyway. Still avoid a change from 1 line to 3.
- **The conclusion** (`5_conclusion.tex:4`): the user picks "On ArtELingo and on the unseen paintings of MMArt,
  \method has the best clustering AMI and image-to-image retrieval mAP on every aspect." and carries it into Overleaf
  (the co-authors edit the conclusion there). Not a row in `apply_fixes.py`.
- **A6 reopened** (option a): the photo paragraph leads with the fine-tuned gain on both photo sets and every encoder.
- **A10 reopened** (option iv): `:154` adds that on every encoder the AMI is higher than the instructed Qwen3
  embedding's on all three aspects; `:156` "There, with one pass instead of three, \method also improves on the
  instructed embedding of the same model, most on emotion AMI, from 0.089 to 0.102."
- **A24**: all ten items (the Table 6 rows keep their labels; "(Qwen3 below)" at `:43`).
- **Underclaim fixes** u1, u2, u7 to u12, u14 (the scan's verdict table). Skipped: rows 3, 4, 5, 13 (gains of 0.001).
- **Merge into REPO waits** until the co-authors finish in Overleaf; conflicts are sure and get discussed with the
  user (user's own call).
- Still in force: A1, A3, A5, A13, A15, A19, A25, A26 and the earlier rows; the way of working (exact old and new text,
  options, a recommendation, the user picks, a row, a scratch build, commit and push in PF); the abstract and the
  Figure 1 caption are the user's; the LLM Usage Statement is never read; Claude subagents only, never Codex.

## 3. Where things stand

- REPO `a56c4c1` = origin (fetched 2026-11-16 21:05; no new commit; the branch `overleaf-2026-11-16-1631` still on
  GitHub). PF `1363cd9` = origin. MAIN: this handoff.
- All rows built (scratch, 21:20): 12 pages in all, every float on its page (p1 Fig. 1, p3 Figs. 2 and 3, p6 Tables 1
  to 3, p7 Tables 4 to 6, p8 Fig. 4 and Table 7), no overfull box, no undefined reference; the conclusion's last
  about 4 lines are on page 9. Before these rows: about 3.4 free lines on page 8.
- Part A: 13 groups approved, A4 and A7 dropped, A2 the user's abstract, 12 open: A8, A9, A11, A12, A14, A16, A17,
  A18, A20, A21, A22, A23.

## 4. Open points for the next step

Key, in order:
1. **The rest of Part A** in the list's order, under the claim rule: A8 (distinct spaces shown by picked examples),
   A9 (the intro's emotion headline; scan rows 3 to 5 were skipped for 0.001 gains, so keep the intro on AMI), A11,
   A12, A14, A16, A17 (+2 lines: weigh with the user), A18 (the rule gives "not seen during training"), A20 to A23.
2. **The merge, when the user says the co-authors are done:** the user pushes Overleaf to GitHub; `git -C REPO
   fetch`; apply on a scratch copy first (`apply_fixes.py` stops at the first old text that no longer matches: list
   all such rows for the user, do not guess new old texts); discuss each conflict with the user; then apply to REPO,
   build, `layout_measure.sh`, `free_lines.py`, float pages, commit, push; the user pulls into Overleaf. Update status
   rows 14 and 25 and `review_changes.md`. A3 (conclusion :7) is the likeliest conflict.
3. **Back to 8 pages:** after the merge, tighten short last lines with the user until the conclusion ends on page 8.
4. **The user's abstract**, the last build check (status row 16), the Overleaf pull, OpenReview, the upload.

## 5. Code and pitfalls

- **The variant builder** is PF `notes/review/plan33/variants.py <spec.json>` (REPO HEAD + `apply_fixes.py` + the
  spec's edits, built in parallel, then `free_lines.py`). Its `S` path points to the plan 33 scratchpad: set it to the
  new chat's scratchpad first. Spec scripts `spec_a24.py`, `spec_r.py`, `spec_u.py` there hold every text built.
- **`apply_fixes.py` rows apply in order:** the "A6 reopened", "A10 reopened" and A24 rows match texts that earlier
  rows produce. Keep new rows at the end.
- **Seed spread:** ANGLE's range over five seeds (`Tables/supp_seeds.tex`) is about 0.001 to 0.012 per cell. Gains of
  that size (emotion image-retrieval mAP 0.269 against 0.268, the Qwen3-encoder margins) do not carry "improves on".
- **Never print `main.tex` or the last text page:** the LLM Usage Statement sits in `main.tex` and after the
  conclusion. `pdftotext` of page 9 and a grep of `main.tex` both showed it by accident in plan 33. Grep `Sections/`
  and `Tables/` only, and cut page text before "LLM Usage".
- **The shell is zsh**; times with `TZ=Europe/Amsterdam date`.
- The rest as in the plan 33 handoff, section 5. The handoffs live in MAIN, a public repo: no co-author names here.

## 6. State at handoff

- Running: nothing. No subagent, background job or queued job of this chat.
- Uncommitted: nothing of ours (PF keeps the user's untracked `CBMI_IMP_CAMREADY (2).pdf` out of git).
- On disk over 1 GB: nothing. This chat's scratchpad holds 313 MB of scratch builds (none needed: every row is in
  `apply_fixes.py`, every spec in PF `notes/review/plan33/`).
