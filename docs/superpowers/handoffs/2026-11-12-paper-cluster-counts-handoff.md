# Handoff: redraw the supplementary cluster-count figure with SigLIP 1 lines, after r28's numbers are in

## Brief
- Done: the Qwen3 cluster-count figure has all 11 Table 2 rows (main `902e0539`). Table 2's PercepT (CLIP L/14) row is filled.
- State: the main result moved to SigLIP 1 (2026-11-12). The figure still shows Qwen3 lines only.
- Next: wait for r28's numbers sheet, find the saved SigLIP 1 test inputs per row, then score the grid and redraw.
- Blocked: r28's numbers (the unattended "CoSiR r28 run" tab, tonight).

> Written 2026-11-12 19:50 by the "paper experiments 2" tab (a paper chat). Outside the research loop: no loop step,
> no direction-log entry. PF means `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`. REPO means
> `PF/CVPR2027-CoSiR-Paper/` (the Overleaf-synced LaTeX repo the co-authors see). KR means
> `/project/CoSiR/src/test/20261111_cluster_count_range/`.

## 1. Read in this order

1. This handoff, all of it.
2. `PF/README.md`, the Decisions rows of 2026-11-11 (the figure's form, option b) and 2026-11-12 (the general method
   and its new names, SigLIP 1 as the main result, and this figure's redraw).
3. `PF/notes/supplementary_items.md`, the row "Range of cluster counts".
4. `KR/20261111_cluster_count_range_log.md`: how each row was scored, and each row's gate against its Table 2 cells.
5. `docs/superpowers/handoffs/2026-11-12-r28-run-handoff.md`, sections 2 to 4: what r28 computes on SigLIP 1 and
   where. Then r28's numbers sheet `src/test/20261112_siglip_main/paper_cells.md` when it exists.
6. `REPO/Tables/main_clustering.tex` once the "paper plan 2" tab has filled it: the figure has one line per row of
   that table.

Skills: none.

## 2. Decided by the user, not to be reopened

- **The figure's form** (2026-11-11, option b): one panel per aspect, AMI against the number of clusters, one line
  per Table 2 row. Grid: emotion 5, 8, 9, 16, 20; style 10, 20, 27, 40; genre 5, 10, 16, 20. A grey band marks each
  aspect's class count (Table 2 uses the class count: 9 / 27 / 10).
- **The redraw** (2026-11-12, user, after seeing the recommendation): SigLIP 1 lines after r28's numbers are in, in
  this chat, which first checks which saved inputs exist. The Qwen3 figure stays the reference until then. The
  figure goes into REPO only after the redraw.
- **Do not edit REPO Tables 2 to 8.** The "paper plan 2" tab fills them from r28's numbers sheet.
- The look (the worker's default, shown to the user on 2026-11-11, no change asked): 6.875 × 2.35 in, FreeSerif, a
  shared legend below, ours vermilion `#D55E00` solid and thick, the instructed embedding black dashed, Okabe-Ito
  colours and one marker per method for the rest. The legend follows the paper's new names (README 2026-11-12: the
  teacher is now "Instructed Qwen3").

## 3. Where things stand

- Qwen3 figure: `KR/figures/supp_cluster_counts.{pdf,png}`, values in `KR/figures/values.md` (11 rows). Ours is the
  top line at every count except genre at 20 clusters (instructed Qwen3 0.506, ours 0.500).
- Scorers, each gated against existing cells with max |diff| 0.0: `kr_grid.py` (TURTLE, TPIPS, CLAY, CRL from r18's
  inputs), `kr_grid_r25.py` and `kr_bertopic_fit.py` (VLM2Vec, BERTopic), `kr_grid_r26.py` (PercepT on CLIP L/14).
  Plain, instructed, PercepT (Qwen3) and ours come from the R17 sweep file
  `/data/SSD2/cosir_r17/20261103_new_scores/outputs/stage3/sweep/artelingo__qwen.json`. `kr_figure.py` draws from
  that file and `KR/records/cells.json`.
- Rows that do not depend on the student's encoder, already scored on the grid: Instructed Qwen3, TPIPS (8B),
  BERTopic, VLM2Vec, PercepT on CLIP L/14. Which rows the new Table 2 keeps: not checked.
- New SigLIP 1 rows: ours, plain, + TURTLE, PercepT (matched), CLAY, CRL. r28 does not score them on the grid.
  Saved SigLIP 1 test inputs for them: not checked. SigLIP 1 is `siglip_so400m`, width 1152 (r14 `bs_paths.py`).
- `REPO/Sections/X_supplementary.tex` exists but `main.tex` has it switched off. Nothing of this figure is in REPO.

## 4. Open points for the next step

Key:
1. **Wait for r28's numbers sheet** (a background `until` loop on the file, not polling). Then a fact-finding read
   (Codex `gpt-6.1-terra` medium when `codex-usage --room` says room, else a Sonnet `medium` worker): per SigLIP 1 row,
   the saved test-set input for a new cluster count (path, shape, how rows join to the test ids), the code of its
   class-count cell, and a cost estimate. For TURTLE: whether new fits are needed, on the GPU, and how long.
2. **Score and redraw:** gate each new row against its new Table 2 cell (and a k = 20 cell where one exists) before
   using it. Redraw, check `values.md` against the figure, look at the PNG, send it to the user.

Minor:
3. The CLAY line hid under the instructed embedding and ours in the Qwen3 style panel. If the overlap stays in the
   new figure, ask the user: a caption note (our recommendation) or a zoomed inset.
4. The figure into REPO (`Sections/X_supplementary.tex`) after the user has seen the new figure.

## 5. Code and pitfalls

- Joins are positional on `FeatureManager` sample ids, never `range(n)` (project CLAUDE.md, C6).
- New scripts in KR keep the `kr_` prefix. `*.json` is gitignored: add records with `git add -f`.
- GPU and CPU-heavy work through `res-queue` (r28 uses the GPU tonight). Subagents pass `--no-wait`.
- r28's folder `src/test/20261112_siglip_main/` belongs to the r28 tab: read only.
- Commit only your own paths (`git commit -- <paths>`), push at once, never force-push. Other chats share PF, REPO
  and the main repo. The main repo holds other sessions' uncommitted files (`docs/superpowers/constitution.md`,
  `docs/reports/reports_sum.md` and others): leave them.
- No notes of ours in REPO. Paper names only (c* is `\method`). Numbers come from result files, not report prose.
- The shell is zsh: quote globs. Python: `/root/miniconda3/envs/CoSiR/bin/python` with `PYTHONDONTWRITEBYTECODE=1`.

## 6. State at handoff

- Running: nothing of this chat. Other tabs: "CoSiR r28 run" (unattended), "paper plan 2".
- Uncommitted: nothing of ours.
- On disk over 1 GB: nothing. Small and kept: `/data/SSD2/cosir_paper/20261111_cluster_count_range/outputs/` (4.4 MB).
