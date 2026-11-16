# Handoff: point ii, step 2: review the fact check and both cold reads, then one list of exact text fixes for the user

## Brief
- Done: title and method name ANGLE (row g, REPO `e8a9295`); Overleaf branch merged with the user's picks (row h, REPO
  `4dcc88c`); the user's own `main.tex` edit committed at their request (REPO `b0db8a7`). Three checks of the final
  text written: the Codex fact check and two cold reads (files below).
- State: REPO `b0db8a7` = origin, working tree clean; 8 pages + references, every float in place, about 4 free lines
  on page 8.
- Next: review the three check files as controller, drop false alarms and known items, and give the user ONE list of
  exact text fixes, with the τ_t fix. Then the abstract (A or B), then the reminder for iv.
- Blocked: the user pulls GitHub into Overleaf (row h); then the branch `overleaf-2026-11-16-1631` can be deleted on
  GitHub (it is merged). Paper due 2026-11-17 12:59 Amsterdam.

> Written 2026-11-16 17:39 by the "paper plan 26" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens), at the user's yes. Direction log entry: none (paper work; decisions are rows in the paper folder's
> notes and the status table).

## 1. Read in this order

1. This handoff, all of it.
2. PF `notes/2026-11-16_final_list.md`: rows a to h; h is today's Overleaf merge, g the rename. ii and iv open.
3. The three check files, all on REPO `8267a7e` (before the rename and the merge; intro line numbers have shifted):
   - PF `notes/review/2026-11-16_final_fact_check.md` (Codex `gpt-6.1-sol` high; WRONG 5, CONTRADICTS 1, OVERSTATED 3,
     TERM 7; all 297 main-table cells, the 60 spread cells and the 32 cost cells match; both abstract candidates pass).
     NOT yet reviewed by the controller.
   - PF `notes/review/2026-11-16_final_cold_read_claude.md` (Sonnet high worker; 31 findings). Reviewed, see section 4.
   - PF `notes/review/2026-11-16_final_cold_read.md` (Codex `gpt-6-luna`; 7 findings, 5 are artefacts of a broken
     scratch build: the "??" references and AdamW "[?]" are fine in the real PDF). Real: supplementary title (fixed
     by the rename), τ_t undefined.
4. PF `notes/review/2026-11-16_findings_sorted.md`: the earlier findings (A to E). Many cold-read points repeat open B
   and C rows; say so instead of presenting them as new.
5. PF `user_read/2026-11-13_paper_status.md`: the status table; keep it current.
6. Project memory `feedback_user-writes-paper.md` and `project_paper-name-angle.md`.

Skills: none.

## 2. Decided by the user, not to be reopened

All on 2026-11-16:
- **Title:** "ANGLE: Learning Multiple Embedding Spaces From Instructed Embeddings" (not Overleaf's "… One Embedding
  Space per Aspect from a Frozen Instruction-Following Model"). Method: ANGLE (Aspect Neighbour-Graph Learned
  Embeddings), written out once in the intro's "We propose" sentence. `\method` = `\mbox{ANGLE}\xspace`.
- **"Label-free"** stays defined once in the intro ("trained label-free: no aspect label, such as awe for emotion,
  enters its loss") and is used everywhere; contribution 1 says "label-free supervision" (the user's pick of option a
  over the co-author's "without aspect labels").
- **The co-authors' other intro word fixes stay** as merged (row h lists them).
- **Text fixes are discussed all together:** the τ_t fix waits for the joint list (the user, 17:30).
- **Each change:** propose the exact text, apply only after the user's yes, build, check the float pages, commit and
  push; the user pulls into Overleaf.
- Still in force from earlier handoffs: no "direction" (intra-modal and cross-modal retrieval); "seed" only in the
  experiment setting; limitations in the supplementary (due 2026-11-24); no notes of ours in REPO `.tex` files; the
  LLM Usage Statement is the user's (agents do not read or comment on it). It is now committed in `main.tex`
  (`b0db8a7`), so the reminder for iv is likely moot: check only that the committed `main.tex` has a section of that
  name (`grep -c 'LLM Usage' main.tex`), never read it.

## 3. Where things stand

- REPO `b0db8a7` = origin, clean. PF `9143c58` or later = origin. MAIN: this handoff.
- `main.pdf` 11 pages, `supp.pdf` 3 pages, no overfull box, no undefined reference. Float pages: p1 Fig. 1; p3 Figs.
  2, 3; p6 Tables 1 to 3; p7 Tables 4 to 6; p8 Fig. 4, Table 7. Page 8: about 4 free lines, measured from word
  positions (left column bottom against the last text line). The ruler numbers do not give the free room: the
  "future work." line is now at 561, with the same position as at 557 before the merge.
- `supp.log` has 2 harmless "Undefined control sequence" errors: `xr-hyper` reads the Fig. 3 caption (with `\cref`)
  from `main.aux`. The PDF is right. A fix (a short optional caption `\caption[...]{...}` for Fig. 3) can go with the
  supplementary work. A first `latexmk -silent` of `supp.tex` stops on it; use `latexmk -f`.

## 4. Open points for the next step

Key, in order:
1. **Review the fact check** (its 16 rows) against the sources it names, as controller: confirm or drop each.
2. **One list for the user**: the confirmed fact-check rows, plus the cold-read points below, each with exact old and
   new text, its page cost (measured in a scratch copy) and our recommendation; known B/C rows marked as known.
   My review of the Sonnet cold read (line numbers on `8267a7e`):
   - Worth proposing (cheap, real): intro "improves on the instructed embeddings … on all aspects" holds only for
     clustering: insert "the clustering of" (#3); `3_method.tex:91` "the plain emotion graph" → "the uncorrected
     emotion graph" (#7, "plain" means p0 everywhere else); `4_experiments.tex:114` "We did not run BERTopic, VLM2Vec
     and PercepT on CLIP L/14 on MMArt" → VLM2Vec-V2, and the plain Qwen3 row is also missing from Tab. 4 (#27);
     image-retrieval read-out sentence `4_experiments.tex:56` reads as a contradiction with Sec. S1 (#1, not a real
     contradiction: images one read-out per method, text per method, aspect and kind; a clarity fix, such as "For
     images, we pick …"); τ_t undefined `4_experiments.tex:71` ("…, where $\tau_t$ is their temperature").
   - Check in code before proposing: #2, whether the caption tower reads one caption's p0 embedding at test time
     (training reads d_i, the mean over a painting's captions).
   - Optional: #13 "learns emotion on photos in general" (= B8), #14 Tab. 5 caption "paintings only" and the dash,
     #22 "trunk shared by all aspects in training".
   - Known rows, not new: #4, #5, #6 (B6), #9 (B10), #12 (D5), #16 (C6), #17 (C2).
   - False alarm: #23 (bold g is the embedding, italic g the encoder); #25.
   - Supplementary (due 2026-11-24): #15 Fig. S1 cited nowhere; #29 and the legend of Fig. S1 ("CoSiR (ours)",
     "SigLIP 1", "VLM2Vec"; script MAIN `src/test/20261111_cluster_count_range/kr_figure.py`); #30 speed-ups from
     unrounded times; #31 "English training paintings"; the `xr-hyper` error above.
3. **The abstract:** A or B (both pass the fact check; neither has stale terms); recheck against the merged intro.
4. **iv:** see section 2 (likely moot).

Minor, with the co-authors (the user): Fig. 4(a), table row names ("Instructed Qwen3" against "Instructed embedding",
C9). New in the merged intro: "As \Cref{fig:teaser} shows" prints "Figure 1" mid-sentence, while the rest of the
paper prints "Fig. 1"; mention it with the list, a co-author wrote it.

## 5. Code and pitfalls

- PF = `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`, REPO = `PF/CVPR2027-CoSiR-Paper/`, MAIN = `/project/CoSiR`.
- **Test each change in a scratch copy first:** `git -C REPO archive HEAD | tar -x -C <scratch>/x`, apply a patch
  script that asserts each old string occurs once, build, check float pages and `grep Overfull main.log`. Scripts of
  this chat (may be gone): `/tmp/claude-0/-project-CoSiR/ce2e730d-aa83-4ad1-9375-1d43ad73c68c/scratchpad/measure.sh`
  (build, float pages, ruler line) and `free.py` there (free lines on page 8 from word positions; run with the
  CoSiR env python and the build folder as argument).
- **Overleaf sync:** when Overleaf has unpushed edits, the user pushes from Overleaf first; on a clash Overleaf pushes
  a branch `overleaf-<date>-<time>`. Merge it in a scratch worktree (`git worktree add --detach <scratch>/merge main`),
  resolve with the user, commit, then `git merge --ff-only` in REPO and push. git here is 2.34 (no
  `merge-tree --write-tree`).
- Bottom floats slip to the next page when text above their `\input` grows (Tables 3, 6, 7, Fig. 3 are `[b]`).
- Paper prose: no semicolons between clauses, no dashes, British spelling, no win/level/lose words, no "significant",
  novelty only "to our knowledge"; number pairs "from X to Y". Build `latexmk -pdf -interaction=nonstopmode -silent
  main.tex`, then `latexmk -f … supp.tex`.
- The handoffs live in MAIN, a public repo: no co-author names here.

## 6. State at handoff

- Running: nothing (both Codex jobs and both subagents ended).
- Uncommitted: nothing of ours. PF keeps the user's untracked `CBMI_IMP_CAMREADY (2).pdf` out of git.
- On disk over 1 GB: nothing. This chat's scratchpad holds 53 MB of scratch builds
  (`/tmp/claude-0/-project-CoSiR/ce2e730d-aa83-4ad1-9375-1d43ad73c68c/scratchpad/`); everything needed is in git.
