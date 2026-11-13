# Handoff: paper plan 6: take the open paper items with the user, starting with the 4.2 points that waited for r29

## Brief
- Done: Table 5 on SigLIP 1 (REPO `a616b3b`); 4.1 genre clause marked TODO; Figures 3 and 4 on SigLIP 1 in REPO (`424d61a`).
- State: default model fixed: Qwen3 text + SigLIP 1 image (user and co-authors, 2026-11-13).
- Next: ask the user which open item comes first; the 4.2 points that waited for r29 are the likely start.
- Blocked: nothing.

> Written 2026-11-13 17:06 by the "paper plan 5" tab (a paper chat, outside the loop). Cut for context size (past 200k
> tokens). Direction log entry: none (paper work; decisions are rows in the paper folder's README).

## 1. Read in this order

1. This handoff, all of it.
2. Project memory `feedback_user-writes-paper.md`: the user writes the paper. Agents word only the user's points.
3. `docs/paper/cvpr2027_aspect_spaces/notes/paper_plan.md`: every paper item and its status.
4. The README rows of 2026-11-13 in `docs/paper/cvpr2027_aspect_spaces/README.md` (the last rows of its table).
5. Only when the user picks the 4.2 points: r29's user-read report `docs/user_read/2026-11-13_siglip_only.md`, then
   its auto report `docs/reports/auto/percept/2027-01-18_siglip_only.md` (the SigLIP-only model).

Skills: none.

## 2. Decided by the user, not to be reopened

- The default CoSiR model is Qwen3 text + SigLIP 1 image: the caption tower reads the Qwen3 plain caption embedding,
  the aspect adapter reads SigLIP 1 images, c*'s settings (user with the co-authors, 2026-11-13; README row).
- Table 5 on SigLIP 1; its caption names SigLIP 1 ("with \method and the plain embedding on SigLIP 1"; user).
- The 4.1 genre clause "less well than the plain embedding does" is marked `\todo` + `\TODO`; wording later (user).
- Figures 3 and 4: picks a1; b1, b5, b11; amusement, Cubism, still life. Five crops shown whole, panel (a) labels at
  6.5 pt, rank 1 reads "nearest", the two repeated paintings in Figure 4 kept (user; README rows of 2026-11-13).
- Earlier decisions (c*'s settings kept, "two claims, two baselines", "the strongest comparator" marked as problematic):
  README rows of 2026-11-12.

## 3. Where things stand

- **Table 5** (ablation, ArtELingo test, SigLIP 1, five seeds): c* AMI 0.099 / 0.441 / 0.528, plain-cosine mAP
  0.263 / 0.340 / 0.599; plain SigLIP 1 0.061 / 0.382 / 0.402 and 0.244 / 0.268 / 0.455. Style loses most without
  the instructed graphs (-0.210 AMI). The relational term acts on genre only (-0.067 AMI, -0.114 mAP without it).
  Source `src/test/20261112_siglip_main/records/ablation_siglip1.json`; trace `notes/trace/4_experiments.md`.
- **4.1 genre clause:** on SigLIP 1, genre mAP without the relational term is 0.484, above plain SigLIP 1 at 0.455
  (all five seeds). On Qwen3 it was 0.461, below plain Qwen3 at 0.476.
- **Figures 3 and 4:** in REPO (`424d61a`), page 8, captions unchanged and still correct. Sources
  `figures/src/fig_qualitative.py` and `fig_clusters_classcount.py` (PF `7ec60e1`, `77839c7`).
- **r29** (the SigLIP-only model) is done and reported (see section 1, item 5). The two 4.2 points waited for it.
- **Deadlines:** paper 2026-11-16 23:59 AoE (2026-11-17 12:59 in Amsterdam); supplementary 2026-11-23.

## 4. Open points for the next step

Key (ask the user which comes first):
1. 4.2, the PercepT sentence ("a bit problematic", user) and whether to state SigLIP 1's own style strength. Both
   waited for r29. With the default model now fixed, the user decides how r29's result enters (if at all).
2. 4.2, "the strongest comparator" (`\todo` + `\TODO`): CLAY on SigLIP 1 is above the instructed Qwen3 on style and
   genre AMI and style mAP.
3. 4.1 genre clause wording (TODO).
4. 4.4 MMArt text (a TODO); "split tables by passes" and "one big table" are open (`paper_plan.md`).
5. Supplementary cluster-count figure on SigLIP 1 lines (README row 2026-11-12): first check which saved inputs exist.

Minor (mention, do not edit; all the user's lines):
6. The introduction's `\enumerate{...}` never closes.
7. The comment beside `\input{Tables/cost}` ("This goes to supplemental material.") is out of date.
8. Table 8 still lists the Qwen3 row of CoSiR first.
9. Figure 1's left crossing: the user improves it later.
10. PF `figures/captions.md` still holds the old Qwen3 caption drafts and traces of Figures 3 and 4 (Metsu, Ukiyo-e).
    It is our file: update its trace part to the SigLIP 1 picks when convenient.

## 5. Code and pitfalls

- Pull REPO before each edit (Overleaf). After each push, tell the user Overleaf needs a pull. Stage only our lines.
- No agent notes, TRACE blocks or comments in REPO `.tex` files. Traces go in `notes/trace/<section>.md`. A `\TODO`
  or `\todo` goes in only when the user asks for it.
- Never name or link the public GitHub repo in paper material. No semicolons between clauses. Write "test set".
- Table check: `PF/tools/check_tables_vs_sheet.py` (CoSiR env) now covers Table 5 too: "groups checked 49 missing 0".
  r28's own `r28_cells_check.py` fails on its `now` columns by design (they are the pre-swap .tex cells): ignore.
- Build: `latexmk -pdf -interaction=nonstopmode -silent main.tex` in REPO. Three overfull boxes are old (about 5 pt,
  Tables 2, 3 and 8). Any new one is ours.
- Figures: rerun the two scripts in `PF/figures/src/` (CoSiR env), then copy the PDFs to `REPO/Figures/` only after
  the user's yes.
- The r20x candidates file's header text still says plain = Qwen3; its inputs and numbers are SigLIP 1 (checked
  against the qual record). Only the header text is stale.
- PF = `docs/paper/cvpr2027_aspect_spaces/` (its own private repo), REPO = `PF/CVPR2027-CoSiR-Paper/`.

## 6. State at handoff

- Running: nothing of this chat (the review page server was stopped).
- Uncommitted: nothing of ours. PF has the user's untracked `CBMI_IMP_CAMREADY (2).pdf`, kept out of git.
- On disk over 1 GB: nothing of this chat. The scratchpad (under 2 MB) goes with the session.
- Repo heads: REPO `424d61a`, PF `77839c7`.
