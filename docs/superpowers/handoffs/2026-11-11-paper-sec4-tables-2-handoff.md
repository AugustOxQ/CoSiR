# Handoff: fill Table 3's two empty rows, build the MMArt table, then score the ablation at the class count

## Brief
- Done: Table 3 (retrieval, picked read-outs) and Table 4 (ablation, four removals) are in the paper. 4.1 has the read-out sentence.
- State: ours is best in all three image-retrieval columns of Table 3 (0.269 / 0.323 / 0.615 against the teacher's 0.268 / 0.322 / 0.612). The teacher leads on emotion in the three text directions.
- Next: build the MMArt table, then score VLM2Vec and PercepT (CLIP L/14) for Table 3 when their validation files exist.
- Blocked: the user's 11:00 meeting; ask nothing about it until the user says it is over.

> Written 2026-11-11 06:35 by the "paper sec4 tables" tab, cut at about 200k tokens. Outside the research loop: no
> loop step, no direction log entry. PF means `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`; REPO means
> `PF/CVPR2027-CoSiR-Paper/` (the Overleaf-synced LaTeX repo the co-authors see).

## 1. Read in this order

1. This handoff, all of it.
2. `PF/README.md`, the Decisions rows of 2026-11-11 from "Retrieval table (Table 3" down (four rows).
3. `PF/notes/trace/4_experiments.md`, sections "4.3 Table 3" and "4.4 Table 4": every printed cell with its exact
   value, file and key.
4. `REPO/Tables/main_retrieval.tex`, `REPO/Tables/ablation.tex`, `REPO/Tables/main_clustering.tex` (the format).
5. The earlier handoff `2026-11-11-paper-sec4-tables-handoff.md`, sections 2 and 5 (its decisions and pitfalls stand).
6. Only when the user says the meeting is over: `PF/notes/after_meeting_2026-11-11.md` (project memory
   `paper-meeting-2026-11-11`).

Skills: none.

## 2. Decided by the user, not to be reopened

All of section 2 of `2026-11-11-paper-sec4-tables-handoff.md` stands. New, all 2026-11-11, recorded in the PF README:

- **Table 3 (retrieval):** each method at the read-out it picked on the validation split (user, against the
  plain-cosine recommendation). One table, four direction blocks (image to image; text to image; image to text; text
  to text). Image rows: Table 2's rows without + TURTLE and BERTopic. Text rows: plain Qwen3, the teacher's text,
  ours. Three decimals, no intervals, bold per aspect and direction.
- **4.1 read-out sentence:** written from the agent's point, accepted by the user.
- **Ablation (Table 4):** first the four main removals (instructed graphs, content correction, graph-positive loss,
  relational distillation), with ours and the plain embedding as reference rows. Absolute values now, without
  intervals. Whether intervals are needed is decided later. Retrieval stays at plain cosine, and the caption says so.
  The other four removals: later.
- **Ablation clustering at the class count:** yes, but later (user). Placeholders until then.
- **AMI is compared only at the same cluster count** (the user's own understanding, confirmed). The class count in
  Table 4 is for consistency with Table 2's ours row.
- **MMArt:** its own one-column table (option a): Table 2's image rows that have MMArt cells, with clustering AMI at
  MMArt's class counts (9 / 20 / 10) and image retrieval mAP at the picked read-out, side by side. MMArt's text
  results go to the supplementary, with the failure stated in the limits (`PF/notes/supplementary_items.md`).

## 3. Where things stand

- REPO `df7d11d` (pushed): Table 3 `Tables/main_retrieval.tex` (4.3), the 4.1 read-out sentence (`3c1e3e0`), Table 4
  `Tables/ablation.tex` (4.4). Table 3's VLM2Vec and PercepT (CLIP L/14) rows are empty. Table 4's four removal rows
  have retrieval only (clustering empty).
- PF `7a37338` (pushed): README rows, the trace, the supplementary list.
- MMArt cells (Codex fact map, the session scratchpad's `mmart_facts.md`, not kept; values checked against the
  numbers the reports print). Clustering AMI at MMArt's class count (E / S / G): plain 0.03810 / 0.23142 / 0.45832;
  + TURTLE 0.03778 / 0.22277 / 0.42165; teacher 0.04917 / 0.41582 / 0.52557; TPIPS 0.03817 / 0.33403 / 0.49044;
  CLAY 0.04029 / 0.42401 / 0.50397; CRL 0.03381 / 0.41206 / 0.44232; PercepT Qwen3 0.03908 / 0.19243 / 0.43873;
  ours 0.05321 / 0.42269 / 0.53057. Files: `/data/SSD2/cosir_r17/20261103_new_scores/outputs/stage3/sweep/mmart__qwen.json`
  (`rows[arm, aspect, k, metric=ami].mean`), `src/test/20261103_outside_methods/records/labels.json`
  (`SC1/SC3/SC5 .../mmart/qwen/<a>/ami/<k>/<comp>`), CRL from the r17 cross-encoder record. Image retrieval mAP at
  the picked read-out: plain 0.19352 / 0.20459 / 0.51512; teacher 0.20208 / 0.34755 / 0.61721; TPIPS 0.19399 /
  0.25736 / 0.56725; CLAY 0.19168 / 0.33830 / 0.57554; CRL 0.19193 / 0.31010 / 0.50840; PercepT Qwen3 0.19261 /
  0.16734 / 0.50407; ours 0.20107 / 0.34720 / 0.61991. Files: `/data/SSD2/cosir_r15/20261102_mmart_test/outputs/`
  (`diag/r0_rows.json`, `stage2/cells_ret.json`), r18 `labels.json` (SC2, SC4), r17 `verdicts.json`. **Re-read each
  printed cell from its file before it enters REPO** (a Sonnet `medium` worker or Codex terra, then the controller's
  check). Unchecked: the CRL record's exact path, and which read-out each MMArt pick used (the fact map says picks
  used MMArt labels on 5,769 ArtELingo validation paintings).
- MMArt has no cells for BERTopic, VLM2Vec or PercepT on CLIP L/14. Leave them out and say so in the caption, unless
  the user asks for new runs.

## 4. Open points for the next step

Key:
1. **The MMArt table:** build it from the cells above, in Table 2's look (Method column; clustering and retrieval
   groups; three decimals; bold per column). Its place in Sec. 4 (end of 4.2, its own short part, or beside the
   other encoders) is for the user. Show the PDF.
2. **Table 3's empty rows:** score VLM2Vec and PercepT (CLIP L/14) image-to-image label mAP on the ArtELingo test
   split at a read-out picked on validation. Use the same menu and pick rule as r11's stage 4 and 5
   (`src/test/20261029_retrieval/`; for a cheap route use r18's outside-methods retrieval code, which picked TPIPS
   and CLAY the same way). Gate: the code reproduces one existing Table 3 cell exactly (for example TPIPS or CLAY)
   before the new rows are scored. Inputs:
   - VLM2Vec: test store `/data/SSD2/pre_extract/artelingo/vlm2vec_v2/<aspect>/` (from r25's test extraction) and
     the validation store `/data/SSD2/pre_extract/artelingo/vlm2vec_v2_val/<aspect>/`, being written by the job
     `r25:extract_val` (queued by the old tab; check `res-queue status` and
     `src/test/20261111_paper_baselines/logs/extract_val.log`). One embedding per aspect instruction: the aspect's
     own embedding scores that aspect.
   - PercepT on CLIP L/14 (r26): validation probabilities
     `/data/SSD2/cosir_r26/20261111_percept_original/outputs/val/k40/L1__s{0..4}.npz` (keys prob, names; names in
     `ns_data.keys('artelingo','val')` order), test probabilities in r26's outputs, collapse per seed in
     `outputs/score/po_scores.json`. Use only the seeds that did not collapse, as in Table 2's row. The r26 run chat
     sends the confirmed path to this tab's successor (it was told "paper sec4 tables"; check messages), about 09:00.
3. **Ablation clustering at the class count** (user: later): k-means at 9 / 27 / 10 on the four removals' saved test
   embeddings, `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/held/k11=40-no_*` (seeds 0 to 4), with the
   project's scorer (`src/test/20261103_new_scores/ns_metrics.py`, mini-batch, n_init 3, batch 4,096, states 0 to
   2). Gate: the arm `k11=40` (ours) gives Table 2's 0.10241 / 0.41093 / 0.51007 exactly. CPU through `res-queue cpu`.

Minor:
4. Table 4 is tight at `\tabcolsep` 1.7pt with "Emo." headers. A polish may be needed once the clustering cells exist.
5. Bold in Table 4 waits for the full numbers. Note: without instructed graphs, genre retrieval is 0.591 against
   ours' 0.590.
6. The other four removals and the intervals in Table 4: the user decides later.

## 5. Code and pitfalls

- All pitfalls of the earlier handoff's section 5 stand: two chats share REPO and PF (the "paper experiments" tab
  edits `Tables/main_clustering.tex`, the cluster-count figure and its own README rows); check `git status` before
  editing a shared file; commit only your own paths with `git commit -- <paths>`; push at once. No notes of ours in
  REPO; paper names, never internal ones (c* is `\method`); no semicolons between clauses; "test set", never
  "held-out".
- REPO has two uncommitted files that are not ours: `Sections/2_related_work.tex` (the user's) and `main.tex` (another
  tab's or the user's, seen at 06:25). Leave both.
- Read-outs: Table 3 uses picked read-outs; Table 4 and the 2027-01-13 report use plain cosine. Do not mix them in
  one table.
- MMArt's style class count is 20 (merged classes), so its style AMI at the class count equals the 20-cluster value.
- The text cells of Table 3 come from `/data/SSD2/cosir_r23/20261106_text_claim/stage3/part4_macro_cells_leak.json`
  (`entry plain`); "leak" in the name refers to the no-leak text entries, not to data leakage.

## 6. State at handoff

- Running: `r25:extract_val` (VLM2Vec validation extraction, GPU, queued from the old tab's background shell, expected
  start about 06:46, 40 to 60 minutes; log `src/test/20261111_paper_baselines/logs/extract_val.log`). The old tab
  stays open until this job ends. Also running: the "paper experiments 2", r25, r26 and r21 tabs.
- Uncommitted: in REPO, `Sections/2_related_work.tex` and `main.tex` (not ours); in the main repo, files not ours
  (`docs/superpowers/constitution.md` and older untracked files).
- On disk over 1 GB: nothing from this work. The VLM2Vec validation store will be a few hundred MB under
  `/data/SSD2/pre_extract/artelingo/vlm2vec_v2_val/` (r25's store; not checked).
