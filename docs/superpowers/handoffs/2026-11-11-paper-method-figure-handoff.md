# Handoff: redraw Figure 1 with an agent from the /duck-figure design, settle the user's loss question, then write method 3.3 from the user's points

## Brief
- Done: related work (b) trimmed; method opening, 3.1 and 3.2 written and revised after the user's read; /duck-figure audit of Figure 1.
- State: the paper builds (4 pages); 3.3 to 3.5 are headings only; Figure 1 and 2 are the old v1.1 drawings.
- Next: an agent redraws Figure 1 (user chose this); answer item 2 below; then 3.3.
- Blocked: OpenReview registration by 11-11 12:59 (the user's alone).

> Written 2026-11-11 03:29 by the "paper latex" tab. Outside the research loop: no loop step, no direction log entry.
> PF below means `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`; REPO means `PF/CVPR2027-CoSiR-Paper/` (the
> Overleaf-synced LaTeX repo the co-authors see).

## 1. Read in this order

1. This handoff, all of it.
2. `REPO/Sections/3_method.tex` (the method so far) and `PF/notes/trace/3_method.md` (the source of each fact, and
   the user's skeleton per part).
3. `/project/CoSiR/.duck/reports/2026-11-11-figure.md`: the /duck-figure report (audit, layout sketch in inches,
   palette, a four-sentence caption, the text lines to change with the figure).
4. `PF/README.md`, the Decisions table rows of 2026-11-10 and 2026-11-11.
5. As needed: `PF/drafts/v1.1/03_method.md` (the checked source of every method fact; read by section),
   `PF/drafts/v1.2/04_experiments.md` (ablation Table 9 for the genre evidence).

Skills: none by default. Ducktor commands only when the user types or names one (this chat ran `/duck-figure` on the
user's request; Ducktor is not switched on in this project).

## 2. Decided by the user, not to be reopened

- **No notes in REPO** (2026-11-11): no TRACE or other comments of ours in `.tex` files, no note lines in its
  `refs.bib`. Traces go to `PF/notes/trace/<section>.md`; `PF/refs.bib` is the master with `% verified:` lines, and
  REPO gets it without comment lines (`grep -v '^%' refs.bib | cat -s`). Recorded in `PF/README.md` and project memory.
- **Method structure** (2026-11-11): one opening paragraph with Figure 1 and the components labelled (1) to (4), no
  subsection references in it; 3.1 plain and instructed embeddings; 3.2 aspect neighbour graphs and the
  graph-positive contrastive loss; 3.3 relational distillation for genre; 3.4 the student (heads, loss, the rest);
  3.5 test time. Other encoders and photos leave the method (ablation part of Sec. 4 or the supplementary).
- **Trunk wording in 3.4** (option a): "a trunk shared in training", plus one sentence that each aspect keeps the
  trunk of its own best epoch. No new runs.
- **Numbers later**: the first method text keeps design reasons qualitative (no AMI values); a sample size may stand
  ("a subset of 5,000 training paintings"). Avoid "development data" in the paper; name the data.
- **Figure 1**: an agent redraws it (option b, a first mock-up for the user to judge), from the /duck-figure design.
  The agent picks the training painting and its caption at random ("you can pick one randomly").
- **Order**: 3.3 starts only after the opening, 3.1 and 3.2 are fixed to the user's satisfaction.
- **Name macro**: `\method` (now "CoSiR") in `REPO/preamble.tex`; the final name is open until 11-16.
- All decisions of the earlier handoffs stand (`2026-11-11-paper-latex-handoff.md` section 2): no direct writing, the
  tone of the user's CBMI paper, the supervision wording, D1, TURTLE, the related work as written.

## 3. Where things stand

- REPO last commit `3f8c9b6`; `PF` repo (`cosir-paper`) last commit is the trace update after it. Both pushed.
  `latexmk -pdf main.tex` exits 0; 4 pages; no undefined references.
- Related work (b) is 244 words (was 337); it ends with the PercepT sentence. The no-aspect-label and one-pass points
  moved to the method opening.
- Method: opening (about 100 words), 3.1 (about 200 words, Table 1 `Tables/instructions.tex` with the four
  instruction texts, checked verbatim against the code), 3.2 (three run-in paragraphs, Eqs. 3 to 6, Figure 2).
- Figures in `REPO/Figures/`: `fig1_overview.pdf`, `fig2_graphs.pdf` (v1.1 drawings, matplotlib scripts in
  `PF/figures/src/`). Captions in the tex are one line each, for the user to write.

## 4. Open points for the next step

Key:
1. **Figure 1 redraw.** Launch one Opus `high` worker (look work, `agent-routing.md`) to rewrite
   `PF/figures/src/fig1_overview.py` to the /duck-figure layout (report section 3): two training lanes under the four
   numbered components, the one-pass test strip under the student, real painting and caption, Qwen in both encoder
   boxes, no weights, no k, no legend, 8 pt minimum, project palette. It picks a public-domain training painting at
   random with one of its own ArtELingo captions; the test strip uses the teaser's query painting
   (`PF/figures/fig_teaser.*`, owned by the r20 tab; read, do not edit). Not checked: whether ArtELingo captions may
   be reproduced; say so to the user. Then a visual review by Claude, send the PNG to the user. Figure 2 likely needs
   the same restyle; ask after Figure 1.
2. **The user's question on 3.2, not yet answered:** "Graph-positive contrastive loss: is this loss new, or can we
   borrow the notion from the literature?" Answer: not new as such. Neighbours as contrastive positives are
   established: NNCLR uses an embedding's nearest neighbour as its positive (`dwibedi2021nnclr`); language-guided
   sampling takes positives from neighbours under another (language) model (`elbanani2023lgssl`, the closest
   precedent: one graph from an uninstructed text model); X-Sample encodes a sample-similarity graph
   (`sobal2025xsample`). v1.1 related work lines 178 to 196 hold the checked descriptions; the cut form says these
   citations move into Sec. 3 when related work (d) is dropped (it was). Offer one sentence in 3.2, written only on the
   user's yes, for example: "Using neighbours as contrastive positives follows NNCLR~\cite{dwibedi2021nnclr} and
   language-guided sampling~\cite{elbanani2023lgssl}; our positives come from one instructed graph per aspect."
   Possibly also rename the loss to the field's term (the name table also lists "neighbour contrastive loss").
3. **Points for 3.3**, proposed and not yet confirmed: what the term matches (the genre head's in-batch similarity
   distribution against the genre teacher's; v1.1 lines 217 to 230); why genre needs it (the genre graph alone does
   not carry genre; v1.2 Table 9, evidence later); genre only, its weight in 3.4.
4. **3.2 extra sentences**: the user has not said whether the two sentences of ours stay ("the cross-modal terms tie
   the caption tower to the image tower ..." and "An image is never paired with its own captions."). Its reason was
   removed: the source (v1.1 Table 9 line 456) does not name the split; check before a reason returns.

Minor:
5. The /duck-figure report also lists text changes that come with the new figure (the caption, the components in
   the opening; the opening already has (1) to (4)).
6. Earlier open items: is "no direct writing" (and "no notes in REPO") a rule for all the user's papers? Related work
   (c) is still a sentence and two TODOs. Old REPO commits still hold the removed TRACE comments; only a force-push
   would remove them (not done; the user was told).

## 5. Code and pitfalls

- Each finished part: write into `REPO/Sections/*.tex`, `latexmk -pdf -silent main.tex`, render the page with
  `pdftoppm` and look at it, commit and push REPO, update `PF/notes/trace/<section>.md` and commit and push the
  `cosir-paper` repo (`/project/CoSiR/docs/paper`), then SendUserFile `REPO/main.pdf` (`display: render`).
- The shell is zsh: `echo ====` fails (a leading `=` expands); `--include=*.py` needs quotes.
- Paper names, never internal ones: graph-positive contrastive loss (not "buddy"), aspect neighbour graph, content
  correction, plain and instructed embeddings. No "development data" in the paper.
- The figure scripts run with `PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python`.
- `.duck/` is kept out of the public repo by `.git/info/exclude`.

## 6. State at handoff

- Running: nothing.
- Uncommitted: `docs/superpowers/constitution.md` (modified before this chat, not ours); `PF/CBMI_IMP_CAMREADY (2).pdf`
  (the user's, untracked).
- On disk over 1 GB: nothing. The scratchpad holds page renders and the /duck-figure CBMI renders (a few MB); they go
  with the session.
