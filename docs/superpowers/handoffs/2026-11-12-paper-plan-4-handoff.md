# Handoff: rewrite Sec. 4.1 to 4.3 for SigLIP 1 from the user's points, then Table 5 from r28's ablation

## Brief
- Done: Figure 2 redrawn in Figure 1's style, in REPO with its new caption. Tables 2, 3, 4, 6, 7, 8 show SigLIP 1.
- State: Table 2 AMI of CoSiR on SigLIP 1 is 0.099 / 0.441 / 0.528, against 0.089 / 0.406 / 0.501 for the instructed Qwen3. The text of 4.1 to 4.3 still gives the Qwen3 model's numbers.
- Next: ask the user for points on 4.1 to 4.3, then write them. Fill Table 5 when r28's ablation lands.
- Blocked: Table 5 waits for r28 tier 2 (ablation runs tonight).

> Written 2026-11-12 20:43 by the "paper plan 3" tab, cut at about 200k tokens (user: yes). Outside the research loop
> (r28 runs in its own workspace, "CoSiR r28"). PF means `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`;
> REPO means `PF/CVPR2027-CoSiR-Paper/`.

## 1. Read in this order

1. This handoff, all of it.
2. `PF/notes/paper_plan.md`: every item with its status.
3. `docs/superpowers/handoffs/2026-11-12-paper-plan-2-handoff.md` sections 2, 3 and 5, and
   `2026-11-12-paper-plan-3-handoff.md` sections 2 and 5: the decisions and pitfalls that still stand.
4. `src/test/20261112_siglip_main/paper_cells.md`: every SigLIP 1 cell with its source, the validation encoder pick,
   and the per-seed counts against the instructed Qwen3 (its last table).
5. `REPO/Sections/4_experiments.tex`, lines 22 to 100 (compared methods, implementation details, 4.2, 4.3).

Skills: none.

## 2. Decided by the user, not to be reopened

All of section 2 of the paper plan 2 and paper plan 3 handoffs stands. New on 2026-11-12 (PF README rows):

- **Figure 2** keeps its three panels and toy values, restyled in Figure 1's style (user, after seeing the
  recommendation). Its caption replaces the TODO caption (user: yes). Sec. 3.1 keeps "the plain emotion graph" for
  M_k(e) (user; "uncorrected" was proposed).
- **Tables** (user, after seeing the recommendation): plain Qwen3 only in Table 2 (Tables 3 and 4 show plain SigLIP 1).
  Table 7 has the SigLIP 1 block first: instructed Qwen3 on top, then SigLIP 1, Qwen3, SigLIP 2, CLIP L/14. Table 8
  gives the Qwen3 text pass as one caption sentence (3.6 ms per caption plus 0.02 ms for the caption tower).
  Table 6 has the SigLIP 1 block first, with the other blocks in their old order (agent default).

## 3. Where things stand

- REPO at `dc94094` (pull first: Overleaf). PF at `eb1c4ed`.
- Tables 2, 3, 4, 6, 7, 8 are on SigLIP 1 (REPO `309540f`). Every SigLIP 1 group of the sheet is in the .tex
  (`PF/tools/check_tables_vs_sheet.py`, 37 groups). Table 5 is unchanged (the Qwen3 ablation).
- **The text that disagrees with the tables now** (the user's lines; change them only from the user's points):
  - 4.1 compared methods: "each compared method uses the same encoder as CoSiR" (images: SigLIP 1 now). PercepT "on
    the same encoder ... Qwen3 gives the image and caption embeddings" (now SigLIP 1 images, Qwen3 topics). CLAY
    "modulates frozen CLIP image embeddings" and CRL "here with CLIP, the encoder that did best on validation" (both on
    SigLIP 1 now, with SigLIP 1's text encoder).
  - 4.1 implementation details: "the trunk maps the 2,048-dimensional plain embedding": the SigLIP 1 image tower's
    input width is not checked (see r14's config).
  - 4.2: "0.102, against 0.089 ... at most 0.077 for the other methods" is now 0.099 against 0.089, at most 0.075
    (CRL). TURTLE on SigLIP 1 is below k-means on all three aspects now (0.057 / 0.337 / 0.365 against 0.061 / 0.382 /
    0.402). The PercepT sentence: SigLIP 1, 0.099 against 0.070 on emotion.
  - 4.3: image to image is now 0.269 / 0.355 / 0.623 against 0.268 / 0.322 / 0.612. Seeds above the instructed
    Qwen3: emotion 4 of 5, style and genre 5 of 5. Text to image genre 0.489 against 0.443. Text to text emotion 0.346
    against 0.377. The text directions are behind on emotion in every seed.
- **The validation pick of SigLIP 1 is confirmed**, so "the encoder that did best on the validation split" may go into
  4.1. It is a close call: mean of six cells 0.386 for SigLIP 1 and SigLIP 2, a margin of 0.0007.
- r28's settings check (tier 2) was on the GPU at 20:35. Its result decides the 4.1 sentence on c*'s settings for
  SigLIP 1. The ablation follows tonight.

## 4. Open points for the next step

Key:
1. **4.1 to 4.3 on SigLIP 1.** Ask the user for points first (the user writes the paper; agents word the user's
   points). Also raise the PercepT sentence and SigLIP 1's own style strength (plain SigLIP 1 0.382 against plain
   Qwen3 0.235), which the user said "we discuss this later".
2. **Table 5** from r28's ablation, when its sheet rows land. Then recheck the 4.1 genre sentence on SigLIP 1's
   relational removal (the Qwen3 ablation gave genre mAP 0.461 against the plain 0.476).
3. **Figures 3 and 4:** the user picks from r28's review page (`records/qual_candidates.json` exists; the review page
   is not checked).

Minor:
4. Figure 1's left crossing: improve later (user).
5. The introduction's `\enumerate{...}` never closes, and the comment beside `\input{Tables/cost}` ("goes to
   supplemental material") is wrong now. Both are the user's lines: mention them, do not edit.
6. Overfull boxes of about 5 pt in Tables 2, 3 and 8 were there before this chat's edits.
7. Table 8 still lists the Qwen3 row of CoSiR first. Moving SigLIP 1 first was not raised.

## 5. Code and pitfalls

- All pitfalls of the earlier paper handoffs stand: pull before editing (Overleaf), tell the user after each push that
  Overleaf needs a pull; stage only our lines; no notes of ours in REPO; paper names only; no semicolons between
  clauses; "test set", never "held-out". Update `PF/notes/trace/<section>.md` and the plan with each change.
- The Codex `gpt-6-luna` job for the table fill failed partway (apply_patch errors, one row dropped, bold not set).
  Small LaTeX table edits are faster in the main session or with a Claude worker.
- After a table edit: run `PF/tools/check_tables_vs_sheet.py` (CoSiR env) and `latexmk -pdf -interaction=nonstopmode
  -silent main.tex` in REPO; build files are gitignored.
- Deadlines: paper 2026-11-16 23:59 AoE (2026-11-17 12:59 in Amsterdam); supplementary 2026-11-23.

## 6. State at handoff

- Running: in this chat nothing. Elsewhere: r28 (settings check, then the ablation) in workspace "CoSiR r28".
- Uncommitted: nothing of ours (PF has the user's `CBMI_IMP_CAMREADY (2).pdf`, left out of git).
- On disk over 1 GB: nothing. `/tmp/edit_tables.py` (10 KB, left by the Codex job) and this chat's 1.1 MB scratchpad,
  which goes with the session.
