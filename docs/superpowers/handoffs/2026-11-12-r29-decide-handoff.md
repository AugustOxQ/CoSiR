# Handoff: r29 decide, grill and spec a SigLIP-only aspect adapter (graphs stay on Qwen3, both towers on SigLIP 1)

## Brief
- Done: the user chose the next experiment in the paper chat: a SigLIP-only aspect adapter, as a medium loop.
- State: CoSiR on SigLIP 1 still reads Qwen3 caption embeddings in its caption tower. So texts need Qwen3 at test time.
- Next: a short grill on the open points below, then a short spec and decision rule. The build and run start after r28 frees the GPU.
- Blocked: nothing. The runs wait for r28 (ablation and settings check on the GPU tonight).

> Written 2026-11-12 21:12 by the "paper plan 4" tab (a paper chat, outside the loop). Loop step reached: after
> decisions A and B (the user's, in the paper chat), before step 5 (the grill). Step 1 and the wrap-up do not apply:
> r28's results go to the paper chat, and r28's own run chat writes its reports. Direction log entry: none yet; this
> chat adds r29's entry at the spec. Medium loop: step 3 skipped (the options are clear), a short grill, a short spec.

## 1. Read in this order

1. This handoff, all of it.
2. `docs/paper/cvpr2027_aspect_spaces/notes/paper_plan.md`: the paper's state and its deadlines (the SigLIP-only row,
   the open questions it settles).
3. `src/test/20261112_siglip_main/paper_cells.md`: every SigLIP 1 cell of the current model. The text-direction rows of
   Table 3 are the cells this loop changes first.
4. `docs/paper/cvpr2027_aspect_spaces/CVPR2027-CoSiR-Paper/Sections/3_method.tex`, Sec. 3.1 and 3.2 (`sec:graphs`,
   `sec:adapter`): what the MLLM gives and what each tower reads.
5. r14's trainer for any encoder, `src/test/20261102_backbones_sweep/bs_cstar.py` (docstring lines 1 to 30, and lines
   351 to 390: the caption input is the Qwen3 default-caption mean, asserted at width 2048).
6. `~/.claude/references/grill.md` before the grill; `spec-templates.md` and `spec_experiment.md` before the spec.

Skills: grilling, domain-modeling (the grill runs as grill-with-docs).

## 2. Decided by the user, not to be reopened

- **Direction (A):** test a SigLIP-only model, as a medium loop with its own decide chat, opened now (user,
  2026-11-12 21:10, paper chat). Why: the paper's PercepT sentence is "a bit problematic", since CoSiR "on SigLIP 1"
  still uses Qwen3 for texts. The SigLIP-only model settles it, and the question of SigLIP 1's own style strength waits
  for it (user: "discuss later after new backbone is settled").
- **The idea (B), in the user's words:** "the instruction part and buddy part still from qwen, just the later student
  part becomes all siglip". So the instructions, the instructed embeddings, the mutual kNN graphs, the content
  correction and the alignment targets stay on Qwen3. The aspect adapter reads SigLIP 1 only: the image tower the
  SigLIP 1 image embedding (as now), the caption tower the SigLIP 1 text embedding of the captions (new). At test time
  no input passes through Qwen3. The user also said earlier: "The instruction then becomes something like 'Represent
  the caption: xxx'" (see open point 2).
- Paper decisions stand (paper folder README, rows of 2026-11-12): SigLIP 1 is the main encoder; two claims, two
  baselines (against plain SigLIP 1 and against the instructed Qwen3); the speed claim is test time only.

## 3. Where things stand

- **Current SigLIP 1 model** (r14 models, five seeds, ArtELingo test), text directions, mAP emotion / style / genre:
  text to image 0.264 / 0.179 / 0.489, image to text 0.286 / 0.141 / 0.407, text to text 0.346 / 0.109 / 0.316.
  Instructed Qwen3: 0.268 / 0.157 / 0.443, 0.304 / 0.124 / 0.397, 0.377 / 0.098 / 0.299. Plain SigLIP 1 (its own text
  encoder): 0.211 / 0.152 / 0.419, 0.220 / 0.107 / 0.354, 0.206 / 0.092 / 0.253. CoSiR is behind the instructed Qwen3
  on emotion in every seed and every text direction (source: `paper_cells.md`, Table 3).
- **Image side** of the current model: AMI 0.099 / 0.441 / 0.528, image mAP 0.269 / 0.355 / 0.623.
- **The two towers are coupled.** The graph-positive loss has image-to-caption and caption-to-image terms (Sec. 3.1,
  `eq:graphloss`), and early stopping keeps both trunks by one validation loss per head. So a new caption input means
  retraining both towers, and the image numbers can move too (not checked).
- **Inputs:** SigLIP 1 text embeddings of all ArtELingo captions exist,
  `/data/SSD2/pre_extract/artelingo_full/siglip_so400m/txt.npy` (308,723 captions, 1,152 dims, none cut at the
  64-token limit). The Affection, EmoSet and MMArt SigLIP 1 stores hold images only. Whether the photo model's
  training reads captions: not checked.
- **Cost:** one SigLIP 1 training run took 341 s (r14 seed 0), so five seeds take about 30 min of GPU. Test time for a
  text: Qwen3 3.6 ms per caption now; the SigLIP 1 text tower's time is not measured.
- **r28** (workspace "CoSiR r28", one unattended run chat) holds the local GPU tonight: the ablation, then one settings
  row rerun (queued 21:07 to 21:22, behind the ablation). Its settings check tells whether c*'s settings hold on SigLIP 1.
- **Deadlines:** paper 2026-11-16 23:59 AoE (2026-11-17 12:59 in Amsterdam); supplementary 2026-11-23.

## 4. Open points for the next step

Key (asked answer-first, `grill.md`):
1. **Its place in the paper:** the new main model (all CoSiR SigLIP 1 rows of Tables 2 to 8 replaced, the ablation
   too), or one more row (Table 3's text rows and Table 8)? This sets the run list against the deadline.
2. **The caption input:** the plain SigLIP 1 text embedding, or the caption with a text prefix such as "Represent the
   caption: ..."? SigLIP's text encoder was not trained to follow instructions.
3. **The settings:** c*'s settings unchanged, or a check on validation first?
4. **The decision rule:** against what the SigLIP-only model must hold (the current SigLIP 1 model on the image cells,
   the instructed Qwen3 on the text cells, plain SigLIP 1 everywhere), and what the paper says if text emotion falls further.

Minor:
5. Device and time: local GPU after r28 (`res-queue`), or DAS6 if a node is reserved (`cluster-work.md`).
6. Table 8: measure the SigLIP 1 text tower's time per caption.
7. The photo model (Table 7) and MMArt (Table 4): rerun on the SigLIP-only model, or keep the current rows.

## 5. Code and pitfalls

- Reuse r14 (`src/test/20261102_backbones_sweep/`) and r28's trainer changes (`src/test/20261112_siglip_main/`,
  `r28_trainer.py`, `r28_trainer_loss.py`). The caption width is asserted at 2048 in `bs_cstar.py` (line 387) and the
  caption trunk is the 2048-wide `TrunkSide` (`src/test/20261025_recipe_backbones/rb_student.py:60`). Both need the
  1,152 width. This is verdict code: tests, then a Claude review (`matt-chain.md`).
- New round folder in `src/test/` with a new module prefix (`r29_`), per the project's CLAUDE.md.
- **The main chat:** r29 has no main chat yet. The rule opens one at brief approval (`loop-chats.md`, "The main
  chat"). r28 is still running with its own main chat. Check first whether a new main chat makes r28's `loop-watch`
  print `ENDED`; if so, open r29's main chat only after r28's run chat ends.
- r28's run chat ends with its own handoff. It was told (paper plan 4, 21:15) that "r29" is this loop, so it should
  not open a second "r29 decide".
- The paper chat edits REPO and PF. This loop does not; it writes a numbers sheet like r28's `paper_cells.md`.

## 6. State at handoff

- Running: in the paper chat, nothing. Elsewhere: r28 (ablation, then a settings rerun) in workspace "CoSiR r28".
- Uncommitted: nothing of ours.
- On disk over 1 GB: nothing.
