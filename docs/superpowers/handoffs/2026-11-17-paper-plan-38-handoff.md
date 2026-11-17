# Handoff: the supplementary material (due 24 November, 12:59 Amsterdam), from status row 15

## Brief
- Done: the paper is uploaded (REPO `adfdb3c`, the co-authors' final abstract); all checks passed.
- State: 8 pages, 2.4 free lines on page 8; the anonymity and desk-reject check found no fail.
- Next: the supplementary, from status row 15: the open items, Figure S1's legend, limitations.
- Blocked: nothing; the user picks each change. Supplementary due 24 November 12:59.

> Written 2026-11-17 09:32 by the "paper plan 37" tab (a paper chat, outside the loop). Cut at the end of the paper
> upload, with the user's yes. Direction log entry: none (paper work; decisions are in the paper folder's notes and
> the status table).

## 1. Read in this order

1. This handoff, all of it.
2. PF `user_read/2026-11-13_paper_status.md`: rows 15 (supplementary), 18 (limitations), 19 (co-author point 8).
3. PF `notes/supplementary_items.md`: the planned items.
4. PF `notes/review/2026-11-17_fix_list.md`, section "For the supplementary", with
   `notes/review/2026-11-17_facts_for_findings.md` (the facts behind each item).
5. PF `notes/review/2026-11-17_panel_merged.md`, Part B (what the panel asked the supplementary to show).
6. PF `notes/review/2026-11-16_fix_round.md`, section "Paper plan 37": this chat's decisions.
7. Project memory: `feedback_user-writes-paper.md`, `feedback_paper-chats-claude-only.md`.

PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.

Skills: none for the first step (reading and a list of options).

## 2. Decided by the user, not to be reopened

All on 2026-11-17; recorded in `fix_round.md`, "Paper plan 37".
- **The uploaded paper** is REPO `adfdb3c`. The abstract is the co-authors' final text ("whilst showing improvements
  across aspects in clustering and retrieval tasks", softened to not overclaim). "After fine-tuning" for the photo
  sets was left out on purpose. The genre relational term stays out of the abstract.
- **Kept as written:** "an order of magnitude", the MMArt tie wording, the conclusion's "achieves improved
  performance" (P10c).
- **The MMArt reference** stays (it names one author, cited in the third person only).
- **Figure S1's legend:** "CoSiR (ours)" becomes "ANGLE (ours)" when the supplementary is built
  (`Figures/fig_supp_cluster_counts.pdf`). The public repo is named CoSiR, so the string must not appear.
- **The heading "6. LLM Usage Statement"** stays as it is.
- Still in force: the user writes the paper (agents guide, check, and give exact old and new text with options and a
  recommendation; the user picks; a scratch build; commit and push); the claim rule (each result as strongly as the
  tables allow); the co-authors' text wins except for errors and plain slips; the LLM Usage Statement is never read;
  Claude subagents only, never Codex.

## 3. Where things stand

- REPO `adfdb3c` = origin (uploaded version). PF `f0f9e1f` = origin. MAIN: this handoff.
- Uploaded PDF: PF `submission/2026-11-17_main_adfdb3c.pdf` (outside git). 11 pages, main text on 8, every float on
  its page, no overfull box, no undefined reference, no "CoSiR" string, no external link, empty PDF author field.
- The supplementary source: REPO `supp.tex` inputs `Sections/X_supplementary.tex`; `supp.pdf` in REPO is an old local
  build (outside git; rebuild before reading it).
- The CVPR 2027 rules (fetched 17 November): the supplementary must not identify the authors (no names, links,
  identifying acknowledgements); it is a separate upload; single column is allowed. Report:
  PF `notes/review/2026-11-17_anonymity_check.md`.

## 4. Open points for the next step

Key, in order:
1. **The list from status row 15:** the numbering, the sentence that cites Figure S1, Figure S1's legend (also
   "SigLIP 1", "VLM2Vec"), the Affection example (co-author point 8), the old teaser under a new name, how the
   Figure 1 query was chosen, the five text fixes from the final checks. Exact texts, options, a recommendation.
2. **Limitations** (row 18): agent proposes, the user decides.
3. **The panel's Part B items** and the fix list's "For the supplementary": the user picks which to add.
4. Before upload: an anonymity check of the supplementary PDF and its figures (the same checks as for the paper).

## 5. Code and pitfalls

- **Never print page 9 of `main.pdf` or `main.tex` after line 55:** the LLM Usage Statement sits there.
- **REPO `main.pdf` is a local build outside git.** It was stale once (16 November) while newer builds lived
  elsewhere. After each change, copy the current build over it, or tell the user which file is current.
- **Fix rows:** PF `tools/apply_fix_rows.py <rows.json> <dir> <ids...>`; each old text must be unique. In zsh, pass the
  IDs as separate words.
- **Scratch builds:** `git -C REPO ls-files -z | (cd REPO && xargs -0 tar -cf -) | tar -x -C <dir>`, then
  `latexmk -pdf supp.tex` in that folder for the supplementary.
- **PDF margin numbers:** the user names lines by the PDF's margin numbers. `find_short_lines.py` gives those for most
  lines, but not on page 1 (it gave 046 for margin line 018).
- **Times:** `TZ=Europe/Amsterdam date`. The shell is zsh. The handoffs live in MAIN, a public repo: no co-author names.

## 6. State at handoff

- Running: nothing. No subagent, background job or queued job of this chat.
- Uncommitted: nothing of ours (PF keeps the user's untracked `CBMI_IMP_CAMREADY (2).pdf` and
  `submission/2026-11-17_main_adfdb3c.pdf` out of git).
- On disk over 1 GB: nothing. This chat's scratch builds are deleted (user's yes).
