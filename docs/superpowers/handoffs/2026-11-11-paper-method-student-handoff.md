# Handoff: give Figure 1 a looser layout (the user's comment), then write method 3.4, the student, from the user's points

## Brief
- Done: 3.3 written; the 3.2 precedent sentence added; Figure 1 redrawn in the plainer style the user likes, with four rounds of the user's fixes.
- State: the paper builds (5 pages); 3.4 and 3.5 are headings only; the new Figure 1 is only in the private paper repo.
- Next: loosen Figure 1's layout (more space, as the method's main figure), then 3.4 from the user's confirmed points.
- Blocked: nothing.

> Written 2026-11-11 04:03 by the "paper method" tab. Outside the research loop: no loop step, no direction log entry.
> PF below means `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`; REPO means `PF/CVPR2027-CoSiR-Paper/` (the
> Overleaf-synced LaTeX repo the co-authors see).

## 1. Read in this order

1. This handoff, all of it.
2. `PF/figures/fig1_overview.png` (look at it) and its script `PF/figures/src/fig1_overview.py` (inch coordinates,
   origin top left, y down; column constants at the top, rows inside `main()`).
3. `PF/README.md`, the Decisions rows of 2026-11-11 (five rows: structure, trunk, Figure 1, loss name and 3.2/3.3,
   3.4 points).
4. `REPO/Sections/3_method.tex` (opening to 3.3) and `PF/notes/trace/3_method.md` (the source of each fact).
5. For 3.4: `PF/drafts/v1.1/03_method.md` lines 165 to 255 (3.4 the student; 3.5 batches, teacher distillation term
   lines 209 to 215, per-head loss and weights 231 to 238, sum over heads 239 to 246, optimisation 247 to 250,
   per-head early stopping 251 to 256) and lines 274 to 287 (one trunk per aspect at test time), plus its TRACE lines
   512 and 518 (architecture and settings, checked against the code on 2026-11-09).
6. Background only, if needed: `/project/CoSiR/.duck/reports/2026-11-11-figure.md` (the /duck-figure design and the
   four-sentence caption, section 4).

Skills: none by default. Ducktor commands only when the user types or names one (Ducktor is not switched on in this
project; the chat is opened with the plugin loaded).

## 2. Decided by the user, not to be reopened

All of section 2 of `2026-11-11-paper-method-figure-handoff.md` stands (no notes in REPO, method structure, trunk
wording option a, numbers later, `\method` macro, no direct writing, the earlier handoffs' decisions). New on
2026-11-11 (recorded in `PF/README.md`):

- **Loss name** stays "graph-positive contrastive loss". 3.2 got the precedent sentence (NNCLR, language-guided
  sampling) and keeps its two closing sentences. The opening, 3.1 and 3.2 passed this round's check.
- **3.3** is written from the three confirmed points; its sentence "Where the graph gives each image a few positives,
  ..." stays. **No further notions**: the user finds the terms and symbols already heavy; add a symbol only when an
  equation needs it.
- **Figure 1 style**: the redraw's plain use of colour is kept. Boxes say "Qwen3" (the full name goes in the
  caption). The caption tower is drawn above the image tower, so the student's two output lines do not cross. Both
  paintings are shown whole. The caption label is $c \in C_i$.
- **Figure 1 enters REPO only when the user is satisfied with it**, together with its caption. Figure 2 is restyled
  only after Figure 1 is final.
- **3.4 points** (confirmed): two towers, image and caption, read only the plain embeddings; each tower is a trunk
  shared in training plus one head per aspect, and each aspect keeps the trunk of its own best epoch; the teacher
  distillation term is on the image tower only; each head's loss adds its terms, the loss sums over heads, and each
  head stops early on its own validation loss; the settings (batch size, temperatures, weights) go in a separate
  implementation paragraph.

## 3. Where things stand

- REPO last commit `cd74293` (3.3 and the 3.2 precedent sentence), pushed. `latexmk -pdf main.tex` exits 0, 5 pages,
  no undefined references. REPO still includes the old v1.1 `Figures/fig1_overview.pdf`, with the one-line caption.
- `cosir-paper` repo (`/project/CoSiR/docs/paper`) last commits: `2eaa1c9` (Figure 1, towers swapped) and `44aca06`
  (README decisions), pushed. Trace for 3.2 and 3.3 is in `PF/notes/trace/3_method.md`.
- Figure 1 now: canvas 6.875 by 2.6 in (CVPR text width, placed at 100%), STIX 8 pt labels and 9 pt bold titles,
  three embedded TrueType fonts. Training painting `abraham-manievich_untitled-1925` (train split, seeded random
  pick; the rule is in the script docstring), test painting the teaser's query `childe-hassam_old-lyme-bridge`.

## 4. Open points for the next step

Key:
1. **Figure 1, looser layout.** The user (2026-11-11): "The overall space is pretty squeezed, since this is the
   method main figure it's okay to use more space and a bit looser layout." Keep the width (6.875 in); raise the
   height (for example to about 3.2 to 3.4 in) and open the gaps: titles to content, the two training lanes, between
   the towers, the loss rows in column E, the hairline to the test strip. Keep every fix of this chat (next item).
   Look work: do it in the main chat or with one Opus `high` worker (`agent-routing.md`); review the render at
   300 dpi and at print size, then SendUserFile the PNG. After the user is satisfied: copy the PDF to
   `REPO/Figures/`, replace the caption in `3_method.tex` with the four-sentence caption of the design report
   (section 4; name Qwen3-VL-Embedding-2B there, since the boxes say Qwen3), build, look at the page, commit and push
   both repos, update the trace. Then ask about Figure 2.
2. **3.4 the student**, from the confirmed points. Introduce only the symbols the equations need (the teacher
   distillation term needs one equation, v1.1 Eq. 11; the per-head loss v1.1 Eq. 14). Trunk wording: "a trunk shared
   in training", plus one sentence that each aspect keeps the trunk of its own best epoch. The settings paragraph can
   follow 3.4 or wait for the user's word on where it goes (end of 3.4, or Sec. 4). Then the usual cycle (section 5).

Minor:
3. The licence of the ArtELingo captions shown in Figure 1 is not checked.
4. Earlier open items: is "no direct writing" (and "no notes in REPO") a rule for all the user's papers? Related work
   (c) is still a sentence and two TODOs.

## 5. Code and pitfalls

- **Figure script rules the user asked for:** every arrow's last segment (the one with the head) at least about
  0.15 in; the split from the input bus at least 0.2 in before the boxes; the blue lines from the student must not
  cross each other (y turns up on the inner line at `XY`, z on the outer at `XZ`); the only crossing is the green
  teacher line over the two blue lines, drawn with a gap; nothing touches the canvas edge (`A0` = 0.03).
- Helpers: `arrow()` draws a right-angle polyline with the head on the last segment; `place()` shows a painting
  whole and returns its width; text widths are measured with `get_window_extent` after `fig.canvas.draw()`.
- Run it with `PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python fig1_overview.py` (in
  `PF/figures/src/`); check `pdffonts ../fig1_overview.pdf` (three STIX TrueType fonts, no Type 3).
- Edits to the script get an entry in `/project/CoSiR/.claude/YYYYMMDD_log.md` (gitignored, one file per day).
- Each finished text part: write into `REPO/Sections/*.tex`, `latexmk -pdf -silent main.tex`, render the page with
  `pdftoppm` and look at it, commit and push REPO (only our files), update `PF/notes/trace/<section>.md` and commit
  and push the `cosir-paper` repo, then SendUserFile `REPO/main.pdf` (`display: render`).
- The shell is zsh: `echo ====` fails; quote globs. Paper names, never internal ones (graph-positive contrastive
  loss, aspect neighbour graph, content correction, plain and instructed embeddings). No "development data".

## 6. State at handoff

- Running: nothing.
- Uncommitted: `REPO/Sections/2_related_work.tex` (the user's own edit, a meeting comment and a `\newline`; leave it);
  `/project/CoSiR/docs/superpowers/constitution.md` (modified before this chat, not ours);
  `PF/CBMI_IMP_CAMREADY (2).pdf` (the user's, untracked).
- On disk over 1 GB: nothing. The session scratchpad holds page and figure renders and the painting-selection script
  (a few MB); they go with the session.
