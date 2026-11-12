# Handoff: paper plan 5: fill Table 5 from r28's SigLIP 1 ablation, then show the user the review page for Figures 3 and 4

## Brief
- Done: Sec. 4.1 to 4.3 on SigLIP 1; c*'s settings kept after r28's settings check, two sentences in 4.1 (REPO `fde9788`).
- State: r28's ablation runs on the GPU until about 23:12. Its rows go to `records/ablation_siglip1.json`, then to `paper_cells.md` T5.
- Next: watch r28, fill Table 5, recheck the 4.1 genre sentence, then show the user the review page for Figures 3 and 4.
- Blocked: Figures 3 and 4 wait for the user's picks.

> Written 2026-11-12 23:00 by the "paper plan 4" tab (a paper chat, outside the loop). Cut for context size (past
> 200k tokens). Direction log entry: none (paper work; decisions are rows in the paper folder's README).

## 1. Read in this order

1. This handoff, all of it.
2. Project memory `feedback_user-writes-paper.md`: the user writes the paper. Agents word only the user's points.
3. `docs/paper/cvpr2027_aspect_spaces/notes/paper_plan.md`: every paper item and its status.
4. `src/test/20261112_siglip_main/paper_cells.md`: the SigLIP 1 numbers sheet (Table 5 rows pending).
5. `docs/paper/cvpr2027_aspect_spaces/CVPR2027-CoSiR-Paper/Sections/4_experiments.tex` and `Tables/ablation.tex`.
6. `src/test/20261112_siglip_main/20261112_siglip_main_log.md` (r28's log, newest lines last).
7. Before you show an HTML page to the user: `~/.claude/references/container-ops.md`.

Skills: none.

## 2. Decided by the user, not to be reopened

- c*'s settings stay for SigLIP 1 (user, 2026-11-12 22:50, after the recommendation; README row of that date).
- 4.1 to 4.3 on SigLIP 1 with "two claims, two baselines" (README row 2026-11-12, REPO `d80702e`).
- "the strongest comparator" in 4.2 is marked with `\todo` and a `\TODO` as problematic (user, REPO `9eb0723`).
- SigLIP 1's own style strength and the PercepT sentence wait for r29, the SigLIP-only model (user).
- The user picks the figure examples. Agents do no picking or drawing.

## 3. Where things stand

- **Table 5** needs SigLIP 1 rows. The c* row and the plain row are in hand (`records/ablation_plain.json`):
  - c*: AMI 0.099 / 0.441 / 0.528, plain-cosine mAP 0.263 / 0.340 / 0.599.
  - plain SigLIP 1: AMI 0.061 / 0.382 / 0.402, mAP 0.244 / 0.268 / 0.455.
  - Removal rows, as differences from c*: instructed graphs, content correction, graph-positive loss, relational
    term. They come from r28's ablation (res-queue "r28:ablation", 22:37 to about 23:12) and land in
    `records/ablation_siglip1.json`. r28 then fills the T5 rows of `paper_cells.md`.
- **The 4.1 genre sentence** ("without an extra term, the genre head retrieves paintings of the same genre less well
  than the plain embedding does") rests on the Qwen3 ablation (genre mAP 0.461 against plain 0.476). Recheck it
  against the SigLIP 1 row without the relational term (plain SigLIP 1 genre mAP 0.455).
- **Review page** for Figures 3 and 4 (r28 item G, 20:00): candidates 3a 11, 3b 18; Figure 4 top classes amusement,
  Cubism, still life. Page `/data/SSD2/cosir_r28/20261112_siglip_main/qual/viewer/index.html`; record
  `records/qual_candidates.json`; tools `src/test/20261111_qual_figure/` (`qf_cand.py`, `qf_view.py`); rules in the
  paper README row "Qualitative figure candidate rules" (2026-11-11). Not yet checked or shown to the user.
- **r29** (SigLIP-only aspect adapter): spec approved 22:42, build chat "r29 build" open. Its numbers come later as
  its own numbers sheet. It does not edit REPO or the paper folder.
- **Deadlines:** paper 2026-11-16 23:59 AoE (2026-11-17 12:59 in Amsterdam); supplementary 2026-11-23.

## 4. Open points for the next step

Key:
1. Table 5 on SigLIP 1: fill from `paper_cells.md` once r28 writes T5, run the table check, compile, show the page.
2. The 4.1 genre sentence: keep, change or flag, from the SigLIP 1 row. The user decides the wording.
3. Figures 3 and 4: show the review page, the user picks.

Minor (mention, do not edit; all the user's lines):
4. The introduction's `\enumerate{...}` never closes.
5. The comment beside `\input{Tables/cost}` ("This goes to supplemental material.") is out of date.
6. Table 8 still lists the Qwen3 row of CoSiR first.
7. 4.4 MMArt text is a TODO. "Split tables by passes" and "one big table" are open (`paper_plan.md`).
8. Figure 1's left crossing: the user improves it later.

## 5. Code and pitfalls

- Pull REPO before each edit (Overleaf). After each push, tell the user Overleaf needs a pull. Stage only our lines.
- No agent notes, TRACE blocks or comments in REPO `.tex` files. Traces go in `notes/trace/<section>.md`.
- Never name or link the public GitHub repo in paper material. No semicolons between clauses. Write "test set".
- Table check: `PF/tools/check_tables_vs_sheet.py` (CoSiR env). Last run: "groups checked 37 missing 0".
- Build: `latexmk -pdf -interaction=nonstopmode -silent main.tex` in REPO. Three overfull boxes are old (about 5 pt,
  Tables 2, 3 and 8). Any new one is ours.
- r28's run chat (workspace "CoSiR r28") owns the GPU job and the numbers sheet. Do not stop its jobs. It was told that
  r29 is taken, so it opens no next-loop chat and sends its results to the paper chat.
- PF = `docs/paper/cvpr2027_aspect_spaces/` (its own private repo), REPO = `PF/CVPR2027-CoSiR-Paper/`.

## 6. State at handoff

- Running: r28 ablation (GPU, workspace "CoSiR r28"); r29 build chat; r21 run 3. None belongs to this chat.
- Uncommitted: nothing of ours. PF has the user's untracked `CBMI_IMP_CAMREADY (2).pdf`, kept out of git.
- On disk over 1 GB: nothing of this chat. The scratchpad PDFs (about 13 MB) go with the session.
- Repo heads: REPO `fde9788`, PF `c98d8ef`.
