# Handoff: rewrite the paper for the general method and the SigLIP 1 main result, starting with the new names across Sections 3 and 4

## Brief
- Done: the user made the method general and moved the main result to the SigLIP 1 student; r28 (opened tonight) makes the SigLIP 1 numbers.
- State: the LaTeX still says teacher and student and shows Qwen3 rows. SigLIP 1 student: AMI 0.099 / 0.441 / 0.528 against the instructed Qwen3's 0.089 / 0.406 / 0.501.
- Next: apply the new names across Sections 3 and 4, then the Section 3 restructure and Figure 1.
- Blocked: the table numbers wait for r28's numbers sheet.

> Written 2026-11-12 19:40 by the "paper plan 1" tab, cut at about 200k tokens at the user's yes. Outside the research
> loop (r28 runs in its own workspace, "CoSiR r28"). PF means `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`;
> REPO means `PF/CVPR2027-CoSiR-Paper/`.

## 1. Read in this order

1. This handoff, all of it.
2. `PF/README.md`, the decisions table, last two rows (2026-11-12): the general method, the names, SigLIP 1 as main.
3. `PF/notes/paper_plan.md`: the plan items. Update its statuses from section 2 below at the start (its rows predate
   this chat's decisions).
4. `REPO/Sections/3_method.tex` and `4_experiments.tex`: the user's `%` comments at the top of each.
5. `docs/superpowers/handoffs/2026-11-12-r28-run-handoff.md`, section 4: what r28 produces and where
   (`src/test/20261112_siglip_main/paper_cells.md`, the numbers sheet).

Skills: none. For a Ducktor step, follow `~/.claude/plugins/marketplaces/ducktor/skills/duck-<name>/SKILL.md` by hand.

## 2. Decided by the user, not to be reopened

All sections 2 of the earlier paper handoffs stand. New on 2026-11-12 (PF README, paper repo `0af4751`):

- **The method is general** (co-authors). Section 3 holds the general recipe for any aspects named by
  instructions. The genre relational term moves to Section 4 as the one aspect-specific part, and the main model
  keeps it (user, after seeing the recommendation: without it, Qwen3 genre AMI falls from 0.510 to 0.439, below the
  instructed Qwen3's 0.501).
- **New names** (user, after seeing the recommendation): teacher becomes "instructed embedding" (table row "Instructed
  Qwen3"; the model that gives it is "the MLLM"); student becomes "aspect adapter" (the agent's reading of the user's
  "that's good"; the other option was "aspect heads"; told to the user, no objection); "teacher distillation term"
  becomes "alignment term"; "relational distillation" becomes "relational term". "Distillation" leaves the paper's
  own terms. Add the four names to `GLOSSARY.md`.
- **Figure 1** (user): the instruction box becomes "MLLM"; the Qwen3 boxes of the student's input become a general
  encoder box ("any VLM").
- **SigLIP 1 is the main result** (user, co-authors; SigLIP 1 over SigLIP 2: user, after seeing the recommendation).
  Tables 2 to 4: CoSiR on SigLIP 1. Table 5 (ablation) and Table 7 (photos): recomputed on SigLIP 1. Table 6:
  reordered, SigLIP 1 block first. Tables that may split by number of passes: a TODO for now, decided later.
- **Encoder-matched comparators move to SigLIP 1:** plain, + TURTLE, PercepT (matched arm), CLAY, CRL. TPIPS,
  VLM2Vec, BERTopic and PercepT on CLIP L/14 keep their own models (user).
- **The caption tower stays on the Qwen3 plain caption embedding** (option i, as r14 trained it; the agent's reading
  of the user's "yes", not objected). So texts use the MLLM at test time, images only SigLIP 1.
- **Two claims, two baselines** (user, after seeing the recommendation): what training adds, against plain SigLIP 1;
  a light model above the strongest comparator at about 1/14 of the test time, against the instructed Qwen3. Plain
  Qwen3 stays in Table 2 beside the instructed Qwen3, with no claim on it.
- **The speed claim is test time only** (user): 19.5 against 266.8 ms per image, 428M against 2.13B parameters on the
  image path. Table 8 returns to the main paper, with a Qwen3 text-pass time from r28.
- **r28's order** (user): tier 0 (the encoder pick on validation, scoring from saved models, the text-pass time),
  tier 1 (CLAY, CRL, TURTLE on SigLIP 1), tier 2 (a settings check on SigLIP 1 validation, then the ablation), then
  Figures 3 and 4 candidates on SigLIP 1. The user picks the figure candidates in this chat.

## 3. Where things stand

- REPO at `cb473ff` (pull first: Overleaf). PF at `0af4751`. The main repo at `082ab857`.
- **Encoder against training** (Table 6, ArtELingo test; plain Qwen3, plain SigLIP 1, ours on SigLIP 1, instructed
  Qwen3). AMI emotion 0.070, 0.061, 0.099, 0.089; style 0.235, 0.382, 0.441, 0.406; genre 0.425, 0.402, 0.528, 0.501.
  mAP emotion 0.248, 0.245, 0.269, 0.268; style 0.192, 0.296, 0.355, 0.322; genre 0.536, 0.504, 0.623, 0.612. Training
  adds to every cell over plain SigLIP 1. On style most of the lead is SigLIP 1 itself (plain 0.382). So the paper does
  not say that a SigLIP student learns better than a Qwen3 student (the SigLIP lead over the Qwen3 student is on
  style).
- Method consequence: two encoders. The MLLM gives the instructed embeddings, the graphs, the content correction
  (its plain image and caption embeddings, the c_ij equation of 3.2) and the caption tower's input. SigLIP 1 gives
  only the image tower's input, with no instruction. Section 3 needs separate symbols for the two plain image
  embeddings.
- The SigLIP 1 pick so far rests on test numbers (Table 6). "The encoder that did best on the validation split" may
  go into 4.1 only if r28's tier 0 confirms it.
- c*'s settings were picked on Qwen3 (r10). One sentence in 4.1 after r28's tier 2: kept unchanged for SigLIP 1, or
  what the check found.
- The "paper experiments 2" tab got a note (19:37): main rows move to SigLIP 1, its cluster-count figure needs SigLIP
  1 lines later, do not edit Tables 2 to 8.

## 4. Open points for the next step

Key:
1. **Order of the writing items.** Recommended: (a) the names in Sections 3 and 4, captions and table rows; (b) the
   Section 3 restructure (the graph part split from the model; the relational term out; Table 1 general, the current
   one to Section 4); (c) Figure 1 (render, check at 300 dpi, send the PNG; Figure 2 follows its style); (d) the
   tables from r28's numbers sheet, as it lands; (e) Figures 3 and 4 picks from r28's review page.
2. **State SigLIP 1's own style strength in 4.2?** The user: "no, we discuss this later". Raise it again when 4.2 is
   rewritten.
3. **The PercepT sentence in 4.2** (from the earlier handoff, still open): with PercepT matched now on SigLIP 1,
   reword it when 4.2 is rewritten.

Minor:
4. The introduction's `\enumerate{...}` never closes; mention it when the user works on the introduction.
5. The comment beside `\input{Tables/cost}` in 4.5 ("goes to supplemental material") is now wrong: Table 8 returns.
6. Pre-meeting point still open: the scope of the no-direct-writing rule (`PF/notes/after_meeting_2026-11-11.md`).

## 5. Code and pitfalls

- All pitfalls of the earlier paper handoffs' section 5 stand: pull before editing (Overleaf), tell the user after
  each push that Overleaf needs a pull; stage only our lines when a file holds the user's uncommitted lines; no notes
  of ours in REPO; paper names only; no semicolons between clauses; "test set", never "held-out".
- r28's chat does not edit REPO or PF. This chat fills the LaTeX from `src/test/20261112_siglip_main/paper_cells.md`.
- Deadlines: paper 2026-11-16 23:59 AoE (2026-11-17 12:59 in Amsterdam); supplementary 2026-11-23.

## 6. State at handoff

- Running: r28 run and r28 main (workspace "CoSiR r28"); r21 run; maybe "paper experiments 2".
- Uncommitted: in REPO and PF nothing of ours (PF has the user's `CBMI_IMP_CAMREADY (2).pdf`, left out of git).
- On disk over 1 GB: nothing. The old scratchpad holds a few MB (figure render, Codex task and log).
