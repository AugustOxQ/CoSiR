# Handoff: hear the user's other issues on the merged paper, then the new five-seat review; upload by 12:59

## Brief
- Done: the co-authors' night edits merged; the fixes that still apply, the conclusion sentence, 9 slips, the
  co-author notes, A23, A11 and the reference list are in (REPO `34ec426`).
- State: the text fits 8 pages (about 0.4 free lines on page 8), every float on its page, no printed note.
- Next: the user's "other issues" (not yet named); then the new five-seat ARS review of the merged text.
- Blocked: the user's Overleaf pull, the abstract claim (status row 2), OpenReview. Paper due 2026-11-17 12:59.

> Written 2026-11-17 03:00 by the "paper plan 34" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens), with the user's yes. Direction log entry: none (paper work; decisions are in the paper folder's notes
> and the status table).

## 1. Read in this order

1. This handoff, all of it.
2. PF `user_read/2026-11-13_paper_status.md`: rows 2, 14, 16, 21 and 25 hold the state after the merge.
3. PF `notes/review/2026-11-16_fix_round.md`: the sections "Merge with the co-authors' night edits" and "After the
   merge" (this chat's decisions, exact texts).
4. PF `user_read/2026-11-16_review_changes.md`: the Part A table, now closed (✓, ◐, ↺ or skipped per group).
5. Project memory: `feedback_user-writes-paper.md`, `project_paper-name-angle.md`, `feedback_paper-chats-claude-only.md`.

PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.

Skills: none for the first step. The review uses `academic-research-skills:academic-paper-reviewer` (the chat starts
with `claude-paper`).

## 2. Decided by the user, not to be reopened

All on 2026-11-17, between 02:20 and 02:55; exact texts in `fix_round.md`.
- **The co-authors' edits come first** (user's own rule): where they edited a sentence, their text wins over the
  review fixes. 14 approved rows are superseded (kept in PF `tools/apply_fixes.py` under `SUPERSEDED`); A9 went with
  their cut of the introduction's numbers.
- **Figure 1 caption:** theirs (user's own pick: they removed "public-domain" on purpose).
- **Conclusion:** their text, with the user's pick at the start of its second sentence ("On ArtELingo and on the
  unseen paintings of MMArt, \method has the best clustering AMI and image-to-image retrieval mAP on every aspect, ...").
- **Abstract:** their rewrite kept (user's own call). Its claim "improving on every aspect across clustering and
  retrieval tasks" is a to-do in status row 2 (caption retrieval: the instructed Qwen3 embedding is higher on emotion).
- **Their other text:** kept; plain slips fixed (9 rows).
- **The co-author notes:** the emotion-label sentence keeps its first half (user's own call); the MMArt "We did not
  run ..." sentence is removed (after seeing the recommendation).
- **Part A after the merge:** A23, A11 and A20 in; A8, A12, A14, A16 skipped (after seeing the recommendation); A21
  skipped (user's own call: no reason in the text); A17 not taken up; A18 and A22 settled. Part A is closed.
- **A new five-seat ARS review** of the merged text comes later (user's own call); when and on which model is open.
- Still in force: the claim rule (state each result as strongly as the tables allow; change a sentence only where it
  is false or wider than its evidence); the way of working (exact old and new text, options, a recommendation, the
  user picks, a scratch build, commit and push); the LLM Usage Statement is never read; Claude subagents only, never
  Codex.

## 3. Where things stand

- REPO `34ec426` = origin: merge `8b0ce82` (branch `overleaf-2026-11-17-0115`), fixes `7eb1614`, notes `e35e20c`,
  MMArt sentence, A23, A11 `1ecfdce`, references `34ec426`. PF `e4e41ba` = origin. MAIN: this handoff.
- Built at `34ec426`: 11 pages in all; the main text ends on page 8 at ruler 564, about 0.4 free lines; floats p1
  Fig. 1, p3 Figs. 2 and 3, p6 Tables 1 to 3, p7 Tables 4 to 6, p8 Fig. 4 and Table 7; no overfull box, no undefined
  reference, no bibtex warning, no printed note.
- The title is now "ANGLE: Learning Multiple Embedding Spaces From Instruction-Tuned Models" (the co-authors).
- The co-authors are done in Overleaf (user). The user still has to pull GitHub into Overleaf and delete both merged
  branches on GitHub.

## 4. Open points for the next step

Key, in order:
1. **The user's other issues.** Asked about tightening the page (02:45), the user said "There are some other issues,
   so not now". Ask what they are first.
2. **The new five-seat review:** when, and on which model. The last panel ran on Fable `xhigh` (each call needs the
   user's yes); the routing default for review roles is Sonnet `high`. Leave time for fixes before 12:59, and note
   that any added line pushes the conclusion to page 9.
3. **The abstract claim** (status row 2), the user's to solve.
4. **The last build check** (status row 16), the Overleaf pull, OpenReview (the new title), the upload.

## 5. Code and pitfalls

- **REPO now holds every fix.** `tools/apply_fixes.py` is a record only: run on REPO it stops at its first row. PF
  `notes/review/plan33/variants.py` starts from REPO HEAD plus `apply_fixes.py`, so it no longer works as it is; build
  scratch copies of the working tree instead: `git -C REPO ls-files -z | xargs -0 tar -cf -` piped into `tar -x -C
  <dir>`, then `bash PF/tools/layout_measure.sh <dir>` and `PF/tools/free_lines.py <dir>`.
- **Co-author notes print with the co-author's name** (`\nvn` in `preamble.tex`, bold teal). Before the upload, grep
  `Sections/` for uncommented `\nvn{` and `\todo{`.
- **Never print `main.tex` or the text after the conclusion:** the LLM Usage Statement sits there. Cut page text
  before "LLM Usage".
- **Times:** `TZ=Europe/Amsterdam date`; this chat once wrote a guessed time and had to fix it.
- The shell is zsh. The handoffs live in MAIN, a public repo: no co-author names here.

## 6. State at handoff

- Running: nothing. No subagent, background job or queued job of this chat.
- Uncommitted: nothing of ours (PF keeps the user's untracked `CBMI_IMP_CAMREADY (2).pdf` out of git).
- On disk over 1 GB: nothing. This chat's scratchpad holds 164 MB of scratch builds (none needed: everything is in
  REPO).
