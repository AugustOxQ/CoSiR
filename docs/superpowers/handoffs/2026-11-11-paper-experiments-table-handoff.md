# Handoff: build the main clustering table (ArtELingo test, Qwen3) with placeholders, and check what PercepT-original runs exist

## Brief
- Done: 3.5 test time and 4.1 settings written; Sec. 4 has the user's six headings; CRL, BERTopic, VLM2Vec in 4.1.
- State: paper builds (7 pages, text to page 5); r25 tab scores BERTopic and VLM2Vec (cutoff 2026-11-14).
- Next: gather each row's ArtELingo test AMI, build the table in the LaTeX repo; check PercepT-original runs.
- Blocked: the user's meeting at 11:00; ask nothing about it until the user says it is over (memo).

> Written 2026-11-11 05:09 by the "paper method" tab (context passed 200k). Outside the research loop: no loop step,
> no direction log entry. PF means `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`; REPO means
> `PF/CVPR2027-CoSiR-Paper/` (the Overleaf-synced LaTeX repo the co-authors see).

## 1. Read in this order

1. This handoff, all of it.
2. `PF/README.md`, the Decisions rows of 2026-11-11 (the last nine are from this chat).
3. `REPO/Sections/4_experiments.tex` (all, about 50 lines) and its trace `PF/notes/trace/4_experiments.md`.
4. For the numbers, the reports named in section 4 below (results briefs first; tables only where a cell is needed).
5. Only when the user says the meeting is over: `PF/notes/after_meeting_2026-11-11.md` (project memory
   `paper-meeting-2026-11-11`). Not before.

Skills: none by default. Ducktor commands only when the user types or names one.

## 2. Decided by the user, not to be reopened

All of section 2 of `2026-11-11-paper-method-figure1-final-handoff.md` stands. New on 2026-11-11 (README rows):

- **Meeting:** the user presents the whole initial draft at 11:00. No questions about the meeting or its comments
  until the user says it is over; then raise the memo's points.
- **3.5** as written (descriptive, no new symbols; its two added sentences kept).
- **Sec. 4 structure:** (a) opening, now `\TODO{add an opening sentence}`, the user writes it; (b) 4.1 experimental
  settings; (c) 4.2 clustering experiments (main); (d) 4.3 retrieval experiments (secondary); (e) 4.4 ablation study
  (other encoders and photos go here or to the supplementary); (f) 4.5 qualitative examples.
- **All compared methods are named once in 4.1.** CRL added. BERTopic and VLM2Vec (two of PercepT's clustering
  baselines) added with results later; CEMTM skipped.
- **Label wording:** labels select settings on training paintings outside the selection runs' training data and on
  the validation split, and evaluate on the test set. Paper text says "test set", never "held-out".
- **Main clustering table:** Qwen3 backbone only; ArtELingo test split only (MMArt later if it fits). Rows: plain
  Qwen3; plain Qwen3 + TURTLE; instructed Qwen3 (teacher); TPIPS; CLAY; CRL; BERTopic; VLM2Vec; PercepT-original;
  PercepT (Qwen3); ours. **Placeholders** (row listed, numbers empty) for BERTopic, VLM2Vec and PercepT-original.
- **PercepT-original:** check what runs exist on PercepT's own backbone. The user knows our build differs from the
  paper's description (followed fully, its clusters collapse) and asked "we only tested on Qwen?".

## 3. Where things stand

- REPO last commit `3a28612` (Sec. 4 TODO), pushed; before it `6e62871` (CRL), `10b007d` (4.1 and headings, Sec. 3
  opening on labels, BERTopic reference), `53c7224` (3.5). `latexmk -pdf main.tex` exits 0, 7 pages (text ends on
  page 5; references follow), no undefined references, no overfull lines.
- `cosir-paper` repo (`/project/CoSiR/docs/paper`) last commit `1e45a94`, pushed: README rows, traces of 3.5 and
  4.1, the meeting memo, the verified BERTopic entry in `PF/refs.bib`.
- r25 tab "r25 paper baselines" (unattended, opened 05:02, handoff `2026-11-11-r25-paper-baselines-handoff.md`):
  VLM2Vec-V2 under each aspect instruction and BERTopic on the test captions with 20 topics; its report gives the
  rows' numbers.
- What is known about PercepT runs (not fully checked): our PercepT core (four audit fixes) ran in five arms on 5,000
  development paintings (report `docs/reports/auto/percept/2026-12-14_percept_core_arms.md`; P2 = CLIP + RoBERTa
  topics, Stage 2 on CLIP, the only established arm there). The test-split comparator is the "matched arm" (GLOSSARY):
  PercepT core with Qwen Stage 1 topics and Stage 2 on each encoder's own image, run on four encoders in r14 and r15
  (Qwen, SigLIP 2, SigLIP 1, CLIP L/14). The paper-settings versions collapsed (direction log, about line 828). The
  faithful version (CLIP L/14 + ModernBERT tuned on GoEmotions, all eight audit changes) was never built (GLOSSARY:
  "needed before submission"). September's PercepT work is on branch `experiment/percept_topic_pipeline` (worktree
  `/project/CoSiR-buddy_prototype_conditioning`) and may hold an original-backbone run on the old protocol.

## 4. Open points for the next step

Key:
1. **The table's numbers.** A fact-finding read (Codex `gpt-6.1-terra` medium when `codex-usage --room other` says
   room, else a Sonnet `medium` worker): for each non-placeholder row, the ArtELingo test AMI at k = 20 per aspect
   (emotion, style, genre), mean over seeds, with the file and line it comes from, and the difference to ours with
   its 95% interval where a report gives it. Starting points: 12-28 (`2026-12-28_recipe_sweep.md`) for ours, plain
   Qwen and the teacher; `2027-01-06_outside_methods.md` for TPIPS, CLAY and TURTLE (check that k = 20 cells exist;
   its examples are at nine clusters); `2026-12-31_affection_confirmation.md` for CRL; the direction log near line 824
   and the 12-28 report for PercepT (Qwen3). Numbers come from the files, never from summaries.
2. **The table layout** (proposed, not yet confirmed by the user): columns Backbone, AMI emotion, style, genre;
   rows grouped (the encoder and its teacher; methods that take the aspect as text; topic and clustering methods;
   ours); a footnote that the teacher needs three encoder passes and BERTopic reads captions at test time.
   Intervals, if shown, as subscript and superscript via one macro (`report-writing.md`); no "verdict" words; bold
   explained in the caption. Show the user the layout with the numbers before it enters REPO.
3. **PercepT-original:** check the September branch and the reports for any run on PercepT's own encoders; tell the
   user what exists and on which protocol, and propose a small run if none fits.

Minor:
4. The text of 4.2 needs the user's points; write nothing before them.
5. The meeting memo's points (Figure 1 style, pair hits, the read-out, the no-direct-writing scope, related work (c))
   wait for the user's word that the meeting is over.

## 5. Code and pitfalls

- Tables go in `REPO/Tables/*.tex`, included from the section with `\input{Tables/<name>}` (as `Tables/instructions`).
- Each finished part: write into REPO, `latexmk -pdf -silent main.tex`, render the page with `pdftoppm` and look at
  it, commit and push REPO (only our files), update `PF/notes/trace/4_experiments.md`, commit and push `cosir-paper`,
  then SendUserFile `REPO/main.pdf` (`display: render`).
- No notes of ours in REPO (`\TODO` only where the user asked for it). Paper names, never internal ones (c* is
  `\method`). No semicolons between clauses. "Test set", never "held-out" or "held". No "development data".
- TURTLE always with "at its released code's default settings, without its hyperparameter selection"; never call
  default TURTLE stronger than k-means (project memory `r18-paper-rows`).
- The shell is zsh: quote globs; `====` at the start of an `echo` argument fails.
- `/project/claude-config` holds another session's uncommitted edits: commit only your own hunks there.

## 6. State at handoff

- Running: the r25 tab (unattended; its own chat). Nothing else of this chat.
- Uncommitted: `REPO/Sections/2_related_work.tex` (the user's own edit; leave it);
  `/project/CoSiR/docs/superpowers/constitution.md` (modified before this chat, not ours);
  `PF/CBMI_IMP_CAMREADY (2).pdf` (the user's, untracked).
- On disk over 1 GB: nothing. The session scratchpad holds page renders (a few MB); they go with the session.
