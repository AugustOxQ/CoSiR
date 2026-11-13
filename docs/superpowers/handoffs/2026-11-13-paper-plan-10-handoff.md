# Handoff: paper plan 10: run the Section 4 polish subagent, then go through the three review reports with the user

## Brief
- Done: 4.5 checked by the user, 4.7 written, the Figure 3 and 4 captions, the selection rule in the supplementary, the open intro list closed. Cold read and facts check of Section 4 done (REPO `a2dd3ed`).
- State: no wrong number. One error: the ablation's "without instructed graphs" arm uses a graph on plain Qwen3, not SigLIP 1 (4.5 text and Table 5 caption).
- Next: launch the polish subagent with both reports as input, then show the user all findings.
- Blocked: nothing.

> Written 2026-11-13 19:28 by the "paper plan 9" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens). Direction log entry: none (paper work; decisions are rows in the paper folder's README).

## 1. Read in this order

1. This handoff, all of it.
2. Project memory `feedback_user-writes-paper.md` (the user writes the paper; agents word only the user's points) and
   global `~/.claude/rules/report-writing.md`, bullet "Results paragraphs in papers" (the user's three rules).
3. PF `notes/review/2026-11-13_section4_facts_check.md` (Sonnet `high` worker: 1 error, 10 risks, 4 style; every table
   cell checked by hand, checker "49 missing 0", every "with every seed" claim holds).
4. PF `notes/review/2026-11-13_section4_cold_read.md` (Haiku worker: 30 items; its line numbers may be off by one).
5. PF `README.md`, the rows of 2026-11-13 (the last seven are this chat's).
6. REPO `Sections/4_experiments.tex`, `Tables/ablation.tex`.

Skills: none (the polish subagent reads its skill files by path, see section 4).

## 2. Decided by the user, not to be reopened

All on 2026-11-13, each a README row:
- 4.5: the ablation opener "The instructed graphs and the graph-positive loss that uses them matter most for emotion
  and style"; the encoder paragraph without the instructed Qwen3 clause; its closing sentence "The heads thus depend
  little on the encoder they read, since the aspect graphs that train them come from Qwen3 on every encoder" (user:
  keep); the cost sentence on the heads' 0.03 to 0.04 ms (user: keep). Table 7: Method and Backbone merged, CoSiR rows
  under an italic group line.
- 4.7: two paragraphs, one per figure, with the recommended points.
- Captions of Figures 3 and 4: "public-domain paintings" in place of the 1955 and fixed-rule sentences; Figure 3
  explains the rank and percentile beside each pair. The selection rule is in supplementary section 7 "Choice of the
  qualitative examples" (`sec:supp_qual`), with no pointer from the main captions.
- Introduction: the contributions list closed (`\begin{enumerate}`); this restored paragraph indents and full column
  width after the introduction.
- The review round: cold read and facts check in parallel, then one subagent for the Ducktor polish, the humanizer and
  the tone match, as Claude subagents (user's words: "subagent"). The polish waits for both reports. The Section 4
  opening is the user's, later.

## 3. Where things stand

- **Text:** Section 4 has text in 4.1 to 4.7. The Section 4 opening, the introduction, the abstract and the conclusion
  are not written yet.
- **Error E1 (confirmed by this chat):** MAIN `src/test/20261112_siglip_main/decisions.md` row 17: the
  "instructed_graphs" removal keeps r10's single graph on the Qwen3 plain embedding (k=40). So 4.5's "one graph on the
  plain embedding serves all three aspects", "lower than that of the plain embedding itself" (0.231 against plain
  SigLIP 1's 0.382; plain Qwen3 has 0.235) and "not from the plain embedding that CoSiR reads", and Table 5's caption
  sentence "Without instructed graphs, one graph on the plain embedding serves all three aspects", name the wrong
  embedding. The numbers stand. The wording is the user's to decide.
- **Facts check, risks worth showing first** (all in its report): the new encoder closing sentence's "since" is
  untested (no run varies the graph source); 4.4 and Table 4 "best result in every column" while emotion mAP is a tie
  (0.20206 and 0.202082); 4.1 "did best on the validation split" by 0.0007 over SigLIP 2; near-ties bolded without
  comment in Tables 2, 6 and 8, and Table 6 underlines CLIP L/14 emotion mAP 0.267 although the instructed row has
  0.268; 4.7's "thus orders ... by its own aspect" and "shows the AMI gain" rest on selected examples ("in these
  examples", "illustrates"); the photo paragraph may hold four number sets (rule: at most three); supplementary
  Figure caption "level with".
- **Cold read, top items:** "it" changes reference between two 4.2 sentences; MMArt "not in ArtELingo" against Table
  4's "ArtELingo validation paintings that MMArt also labels"; "read-out" is not defined; "with every seed" with no
  per-seed values shown; the photo fine-tuning objective is not said; the content-correction and relational-term rows
  of Table 5 are not discussed; Figure 4 "share" has two denominators in text and caption. It also flags the EmoSet
  validation used for the read-out choice: check how that split is defined before anyone calls it a held-data issue
  (global `held-data.md`).

## 4. Open points for the next step

Key:
1. **Launch the polish subagent first** (Opus `medium` worker; paper-section wording). Brief: Section 4 text and
   captions only, proposals only (no REPO edits). Three passes as in the earlier runs (`.duck/reports/2026-11-12-polish.md`):
   Ducktor's `paper-polish` (`/root/.claude/plugins/cache/ducktor/duck/0.2.3/library/paper-polish/SKILL.md` and its
   `references/`), the humanizer (`/root/.claude/plugins/cache/humanizer/humanizer/3.1.0/SKILL.md`), and the tone of the
   user's CBMI paper (PF `CBMI_IMP_CAMREADY.pdf`, introduction and method only, never its experiments) and of this
   paper's Sections 2 and 3. Give it both review reports: it skips or marks sentences that E1 or a confirmed finding
   will change. It keeps every number, claim, the user's three rules, the decided wordings of section 2 and the
   README rows, British spelling, "test set", no semicolons, no dashes. Output: PF `notes/review/2026-11-13_section4_polish.md`
   (original, proposal, reason per sentence) and a Ducktor report in MAIN `.duck/reports/`.
2. **Then show the user all findings**, E1 first, then the risks and the cold-read items sorted, each with a
   recommendation. The user decides each; apply one subsection at a time, build, send a crop.

Minor (mention, do not edit; the user's lines):
3. Tables 6 and 8 keep `\TODO{either underline or just say here we are all best?}` (the near-tie findings bear on it).
4. The comment beside `\input{Tables/cost}` ("This goes to supplemental material.").
5. PF `figures/captions.md` still holds the old Qwen3 caption drafts of Figures 3 and 4 (our file).

## 5. Code and pitfalls

- Pull REPO before each edit (Overleaf). After each push, tell the user Overleaf needs a pull. Stage only our lines.
- No agent notes, TRACE blocks or comments in REPO `.tex` files. A `\TODO` only when the user asks for it.
- Never name or link the public GitHub repo in paper material. No semicolons between clauses. Write "test set".
- Build: `latexmk -pdf -interaction=nonstopmode -silent main.tex` in REPO. One overfull box is old (0.7 pt, a table
  file); any new one is ours. The supplementary is switched off in `main.tex`: test it in a scratch copy with the
  `\input{Sections/X_supplementary}` line uncommented.
- Crops for the user: `pdftoppm -r 150` on the page, crop with PIL, send with SendUserFile. `pdftotext -bbox` finds a
  word's place (points; times 150/72 for pixels).
- Table check: `PF/tools/check_tables_vs_sheet.py` (CoSiR env), "groups checked 49 missing 0".
- PF = `docs/paper/cvpr2027_aspect_spaces/` (its own private repo), REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.

## 6. State at handoff

- Running: nothing of this chat (both review subagents finished). r31 run 2 (another tab) uses the queue tonight.
- Uncommitted: nothing of ours. PF has the user's untracked `CBMI_IMP_CAMREADY (2).pdf`, kept out of git.
  claude-config has another session's `codex/jobs.md` change, not ours.
- On disk over 1 GB: nothing of this chat. The scratchpad (page renders, a PDF snapshot, the review copies) goes with
  the session.
- Repo heads: REPO `a2dd3ed`, PF `20fc133`, claude-config `de9f1f0`.
