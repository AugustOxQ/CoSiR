# Handoff: r28, put CoSiR on SigLIP 1 in every main table: score from saved models, run CLAY, CRL and TURTLE on SigLIP 1, then a settings check and the ablation

## Brief
- Done: the user moved the paper's main result to the SigLIP 1 student and set three run tiers (paper chat, 2026-11-12).
- State: most SigLIP 1 cells exist (Table 6: AMI 0.099 / 0.441 / 0.528 against the instructed Qwen3's 0.089 / 0.406 / 0.501); the ablation and some scores are missing.
- Next: tier 0, first the encoder pick on the validation split, then scoring from saved models.
- Blocked: nothing.

> Written 2026-11-12 19:30 by the "paper plan 1" tab (a paper chat). Loop step reached: r28 decide happened in that
> paper chat (the user set every choice in section 2); this one unattended chat does steps 7 to 9: the small builds,
> the runs and short reports. Direction log entry: none yet; this chat adds a short one at its end. The paper chat
> keeps writing and fills the LaTeX tables from this chat's numbers sheet.

## 1. Read in this order

1. This handoff, all of it.
2. `docs/superpowers/constitution.md`, section "Decisions agents never make alone": how an unattended chat handles
   each kind of decision.
3. `src/test/20261112_siglip_main/siglip_inventory.md` (a Codex inventory, 2026-11-12, two facts checked at source by
   Claude, the rest not checked): what exists for SigLIP 1, which files to rescore from, what needs training.
4. The paper tables, for the exact protocol of each cell (read only, never edit):
   `docs/paper/cvpr2027_aspect_spaces/CVPR2027-CoSiR-Paper/Tables/{main_clustering,main_retrieval,mmart,ablation,encoders,photos,cost}.tex`
   (Tables 2 to 8 in that order).
5. The code to reuse, per tier (section 5).

Skills: none. Builds follow `~/.claude/references/matt-chain.md` at the size each job needs.

## 2. Decided by the user, not to be reopened

All on 2026-11-12 in the paper chat; recorded in the paper folder's README decisions table (paper repo `0af4751`).

- **SigLIP 1 is the main result.** CoSiR's image tower reads plain SigLIP 1; the graphs, the content correction, the
  teacher (now "the instructed Qwen3") and PercepT's topics stay on Qwen3, as in r14. Tables 2, 3, 4, 5 and 7 get
  SigLIP 1 cells; Table 6 is reordered (paper chat). User (co-author meeting), SigLIP 1 over SigLIP 2: user, after
  seeing the recommendation.
- **The caption tower stays on the Qwen3 plain caption embedding** (option i, as r14 trained it). No retraining of
  the text side. User (the agent's reading of the user's "yes", not objected).
- **Encoder-matched comparators move to SigLIP 1:** plain, + TURTLE, PercepT (matched arm), CLAY, CRL. TPIPS,
  VLM2Vec, BERTopic and PercepT on CLIP L/14 keep their own models. User, after seeing the recommendation.
- **Two claims, two baselines:** the training effect against plain SigLIP 1; the practical claim (one light pass)
  against the instructed Qwen3, 3 passes. Plain Qwen3 stays in Table 2 with no claim on it. User, after seeing the
  recommendation.
- **The speed claim is test time only.** Table 8 returns to the main paper and needs the time of a Qwen3 text pass.
  User, after seeing the recommendation.
- **The order (user's tiers, with three changes the user accepted):**
  - Tier 0: the encoder pick on the validation split first; then scoring from saved models (MMArt, photos, text
    directions); then the Qwen3 text-pass time.
  - Tier 1: CLAY, CRL and + TURTLE (MMArt) on SigLIP 1. Independent of tier 0; may run beside it.
  - Tier 2: a settings check on SigLIP 1 validation, then the ablation (Table 5) on SigLIP 1.
  - Then Figures 3 and 4 candidates on the SigLIP 1 model (the user picks later, in the paper chat).
- **The pick of SigLIP 1 rested on test numbers (Table 6).** The validation pick in tier 0 fixes that. If an encoder
  other than SigLIP 1 wins on validation, the user decides then.
- **The selection on Qwen3 may not carry over.** c*'s settings were picked on Qwen3 (r10). The tier 2 check tests the
  most important one on SigLIP 1 validation. If another setting wins, the user decides whether to retrain.

## 3. Where things stand

- SigLIP 1 cells that exist (inventory; Table 6 and r17 records): ArtELingo test AMI at the class counts and picked
  image mAP for ours (0.099 / 0.441 / 0.528; 0.269 / 0.355 / 0.623), plain (0.061 / 0.382 / 0.402; 0.245 / 0.296 /
  0.504) and PercepT matched (0.070 / 0.225 / 0.466; 0.244 / 0.166 / 0.548); + TURTLE AMI 0.057 / 0.337 / 0.365;
  MMArt class-count AMI and mAP for ours, plain, PercepT; Affection photo model 0.182 AMI / 0.323 mAP.
- Comparators against the instructed Qwen3 (3 passes): AMI 0.089 / 0.406 / 0.501, mAP 0.268 / 0.322 / 0.612.
- Missing: the encoder pick on validation; text directions for the SigLIP 1 model; plain SigLIP 1 text rows; MMArt +
  TURTLE; EmoSet and "paintings only" photo rows; CLAY and CRL on SigLIP 1; the Qwen3 text-pass time; the settings
  check; the SigLIP 1 ablation; Figures 3 and 4 candidates.
- GPU check passed at 19:20 (driver 595.99.02, libraries match). Queue idle. Run mode: night.

## 4. The jobs, with their rules

Round folder: `src/test/20261112_siglip_main/` (module prefix `r28_`; the folder goes on `sys.path`). Outputs under
`/data/SSD2/cosir_r28/20261112_siglip_main/`. One decision line per result-shaping choice in the folder's
`decisions.md` (`loop-chats.md`). Validation only for every pick; the test split is scored, never used to choose.

**Tier 0**

1. **Encoder pick on the validation split (first).** Our model on SigLIP 1, SigLIP 2 and CLIP L/14, five seeds each,
   from r14's saved validation exports (6,152 paintings). Six cells per encoder: AMI at the class counts 9 / 27 / 10
   and plain-cosine label mAP, means over the seeds. Pick: the highest mean of the six cells (agent default; report the
   per-cell table too). If SigLIP 1 is not the pick: flag it at the top of the handoff and in a push, and go on with
   SigLIP 1 (night); in day mode ask, default "go on with SigLIP 1".
2. **Table 3 text directions** for the SigLIP 1 model (text to image, image to text, text to text), ArtELingo test,
   label mAP at the read-out picked on validation, the protocol of the Qwen3 rows (`20261105_text_retrieval`,
   `20261111_table3_rows`). Also a plain SigLIP 1 row in each text direction, with SigLIP 1's own text tower (cheap; the
   user agreed to "run everything suggested").
3. **Table 4 MMArt** for ours, plain and PercepT on SigLIP 1: AMI at 9 / 20 / 10 and image mAP at the read-out picked
   on the ArtELingo validation paintings that MMArt labels. Reuse r17 / r15 values only after checking they follow this
   protocol; else rescore from the r15 exports.
4. **Table 7 photos** on SigLIP 1: the photo model on EmoSet (8 clusters, read-out picked on EmoSet validation);
   "paintings only" (the ArtELingo model before photo fine-tuning) on Affection (9) and EmoSet (8); plain SigLIP 1 on
   EmoSet. The Qwen3 rows' scripts: `20261111_encoder_photo_tables/ept_frozen.py`, `20261102_backbones_sweep/bs_emoset_score.py`.
5. **Per-seed values** behind the paper's prose, for ours on SigLIP 1 against the instructed Qwen3: every Table 2 and
   Table 3 image cell, per seed.
6. **Table 8: Qwen3 text-pass time** per caption, the setup of Table 8 (one RTX 3090, batch 32, model time only),
   generic instruction; plus the caption tower's time. Source of the image times: r19 `cost_table.json`.

**Tier 1**

7. **CLAY and CRL on SigLIP 1**, with SigLIP 1's text tower in place of CLIP L/14's. Same texts (CRL's saved value
   lists from Qwen3-VL-8B-Instruct; CLAY's text sets), same picks made the same way on SigLIP 1 validation. Cells: Table
   2 (AMI at class counts), Table 3 (image mAP at the picked read-out), Table 4 (MMArt AMI and mAP).
8. **+ TURTLE on MMArt, SigLIP 1**, as the Qwen3 MMArt TURTLE row was made (first check whether it already exists).

**Tier 2**

9. **Settings check.** c* (`ref`, graph neighbours 40) and the knob rows r14 names (graph neighbours 10, 20, 80;
   `k6=0.0005`), SigLIP 1, seed 0, validation only. Criterion: r10's stage 3 pick criterion, unchanged
   (`src/test/20261027_recipe_sweep/DECISION_RULE.md`). If it needs inputs SigLIP lacks, use the mean of the six
   validation cells, with a decision line. If c* wins or ties: go on. If another row wins: score nothing of it on test,
   flag it (handoff top, push), and run the ablation on c* anyway; the user decides from the validation numbers whether
   to retrain (constitution row 6). In day mode ask first, with that default.
10. **Ablation (Table 5) on SigLIP 1:** the four removals (`instructed_graphs`, `content_correction`, `buddy` = the
    graph-positive loss, `relational`), five seeds each, plus c* as the reference; AMI at the class counts and
    plain-cosine mAP, as `20261111_ablation_classcount/t5_ablation_ami.py` scored the Qwen3 version. The "instructed
    graphs" removal keeps one graph on the Qwen3 plain embedding (agent default from the paper chat, not objected).
    About 20 runs of about 6 min; local GPU tonight, ending by 11:00 (`run-planning.md`).

**After tier 2**

11. **Figures 3 and 4 candidates** on the SigLIP 1 model, by the rules in the paper folder's README (row of
    2026-11-11 "Qualitative figure candidate rules"), with `20261111_qual_figure/` (`qf_cand.py`, `qf_view.py`): the
    candidate lists and a review page. No picks and no drawing: the user picks in the paper chat.
12. **Numbers sheet** `src/test/20261112_siglip_main/paper_cells.md`: per paper table, each new cell beside the current
    Qwen3 cell, with its source file. Then a short auto report (results brief first) and a half-page user-read report,
    their `reports_sum.md` rows, a short direction-log entry, and the handoff.

## 5. Code and pitfalls

- Reuse, do not rewrite: r14 `src/test/20261102_backbones_sweep/` (`bs_cstar.py` trainer and exports for any encoder,
  `bs_cstar_loss.py`, `bs_paths.py` (SigLIP 1 = `siglip_so400m`, width 1152), `bs_photo.py`, `bs_emoset_score.py`);
  r10 `src/test/20261027_recipe_sweep/` (`rs_sheet.py` removal keys and knobs); r18
  `src/test/20261103_outside_methods/` (`om_paths.py`, `om_data.py`, `om_clay.py`, TURTLE); CRL
  `src/test/20261029_retrieval/rt_crl.py`; r15 `src/test/20261102_mmart_test/`; r17 `src/test/20261103_new_scores/`.
- **Two trainer gaps** (inventory Q4): r14's `--encoder` path takes only `ref` and knob rows for Qwen ("only Qwen has
  knob rows", `bs_cstar.py` line 30), and its non-Qwen loss asserts the original switches (`bs_cstar_loss.py` line 40).
  Tier 2 needs both opened for SigLIP 1. This is verdict code: tests, then a Claude review (`matt-chain.md`).
- **CLAY and CRL are fixed to CLIP L/14** (`om_paths.py` line 100, `rt_crl.py` line 1). They need an encoder switch.
- The paper repo and the paper folder belong to the paper chat and the user (Overleaf). Do not edit them.
- Sample ID consistency (constitution C6): joins by the extraction's `sample_ids`, never `range(n)`.
- GPU and CPU work through `res-queue` with `--label r28:<step>`. Each training run is under 15 min, but the ablation
  is about 2 h of heavy load: keep it out of weekday 10:00 to 19:00.
- Unattended: delete nothing; list removals in `.scratch/pending_deletions.md`.
- Not in this round: the supplementary figure of AMI against the number of clusters on SigLIP 1 (deadline
  2026-11-23); any retraining of the caption tower; TPIPS on SigLIP.

## 6. State at handoff

- Running: nothing of this chat. Others: the "paper plan 1" tab (writing), maybe "paper experiments 2" (the
  supplementary cluster-count figure on Qwen3 rows).
- Uncommitted: in the main repo, files not ours (see `git status`).
- On disk over 1 GB: nothing from the paper chat.
