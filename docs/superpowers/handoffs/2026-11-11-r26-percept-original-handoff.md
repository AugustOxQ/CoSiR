# Handoff: r26, run PercepT on its own encoders (CLIP L/14, ModernBERT-GoEmotions) on the ArtELingo test split, for the paper's main table

## Brief
- Done: the paper chat built the main clustering table; the PercepT row on CLIP L/14 is empty.
- State: no PercepT run on its own encoders exists; earlier non-Qwen runs used CLIP B/32 with RoBERTa-GoEmotions.
- Next: run PercepT at paper settings, 40 topics, on CLIP L/14 and ModernBERT-GoEmotions; score test AMI at k = 20.
- Blocked: nothing; the GPU is held by r25's extraction until about 06:36, so the jobs wait in the queue.

> Written 2026-11-11 05:25 by the "paper experiments" tab (a paper chat). Loop step reached: r26 is a small loop
> opened from a paper chat, with the design set by the user (below); this chat does the small build, the run (step
> 8) and the short reports. Direction log entry: none yet; this chat adds a short one at its end.

## 1. Read in this order

1. This handoff, all of it.
2. `docs/reports/auto/percept/2026-12-19_claim_test_stage2.md`: the paper-settings arms at full scale (42,969 train
   paintings, five seeds, both topic cuts, 8,014 s for two arms), their scoring and the collapse rule. Its code, to
   reuse rather than rewrite: `src/test/20261020_claim_test/` (`ct2_percept.py`, `ct2_run.py`, `ct2_data.py`,
   `ct2_queue_gpu.sh`, `ct_score.py` lines 92 to 105 for the two readings) and the core in
   `src/test/20261018_percept_core/` (`pc_stage1.py`, `pc_stage2.py`, `pc_fixes.py`, `pc_data.py`, `pc_roberta.py`).
3. `GLOSSARY.md` rows 7 to 13 (PercepT versions: core, faithful, paper settings strict and K = 40, matched arm).
4. `docs/reports/auto/percept/2026-12-04_percept_code_audit.md` (section C2 and on: PercepT's encoders, Eq. 2 fusion,
   Stage 2 on patch tokens) and `docs/reports/literature/2026-12-05_percept_neighbourhood.md` line 31.
5. The paper's Table 2 and 4.1 metrics paragraph, for the protocol the row must match:
   `docs/paper/cvpr2027_aspect_spaces/CVPR2027-CoSiR-Paper/Tables/main_clustering.tex` and `Sections/4_experiments.tex`.

Skills: none.

## 2. Decided by the user, not to be reopened

On 2026-11-11 in the paper chat (`docs/paper/cvpr2027_aspect_spaces/README.md`, rows of that date):
- The main clustering table has a PercepT row on PercepT's own backbone. It is a **new run**: PercepT at **paper
  settings with 40 topics** (code tag `percept_paper_k40`, GLOSSARY row 10), on **CLIP L/14** (image and text) and
  **ModernBERT-base tuned on GoEmotions** (the audit names `cirimus/modernbert-base-go-emotions`), on our protocol, with
  **all three aspects** scored (user, after seeing the recommendation).
- The paper will state two differences from PercepT's own setup: our 42,969 English training paintings instead of its
  1.2M translated image-caption pairs, and 40 topics instead of its topic-count rule (Algorithm 1).
- Table 2 shows AMI with three decimals and no intervals (user). The paper chat fills the row; this chat does not
  edit the LaTeX repo.

## 3. Where things stand

- The protocol of Table 2: train on the 42,969 train paintings, five seeds 0 to 4; on the 12,281 test paintings (genre
  on 9,949), k-means with 20 clusters (mini-batch, n_init 3, batch 4,096, states 0 to 2) on the L2-normalised Stage 2
  topic probabilities (the "r20" reading, `ct_score.py` lines 92 to 103), AMI per aspect, mean over seeds and states.
  The PercepT (Qwen3) row beside it, read this way: 0.080 / 0.188 / 0.423; ours (CoSiR): 0.107 / 0.407 / 0.500.
- The closest earlier run, paper settings K = 40 on CLIP B/32 + RoBERTa-GoEmotions (arm B4), test emotion AMI 0.0699
  (r20) at full scale (stage 2 report line 86); style and genre were never scored for paper arms (line 158). The strict
  version (Algorithm 1, gamma 2) collapsed in all 10 runs on both backbones tried.
- Inputs on disk: CLIP L/14 image and caption features for all 61,402 ArtELingo paintings
  (`/data/SSD2/pre_extract/artelingo_full/clip_l14/`, global vectors only); CLIP B/32 patch tokens
  (`/data/SSD2/pre_extract/artelingo_percept_patch_features/`, 6.8 GB). Missing: CLIP L/14 patch tokens and ModernBERT
  caption features. The ModernBERT model is not in the HF cache; huggingface.co answered at 05:20.
- `/data/SSD2` has 1.7 TB free. GPU check passed at 05:21 (driver 595.99.02 in kernel, library and `nvidia-smi`).
  `res-queue`: r25's extraction holds the GPU until about 06:36.

## 4. Open points for the next step

Agent defaults (take them, write a decision line in the round folder's `decisions.md`, flag them at the top of the
next handoff and the report):
- key: **Stage 2 on CLIP L/14 patch tokens** with attention pooling, PercepT's main version, as arm B4 did on B/32
  patches. Extract the L/14 tokens in the same form as the B/32 store (check which layer and width it saved).
  Alternative: Stage 2 on the L/14 global vector, which exists now.
- key: **Both topic cuts**, K = 40 and the strict Algorithm 1 (gamma 2), as the claim test ran them. Only K = 40 fills
  the row; the strict result is reported with its collapse check (it tells the paper whether the rule also collapses
  on PercepT's own encoders).
- key: **ModernBERT caption embedding:** the same pooling as `pc_roberta.py` uses for RoBERTa (record it). If the
  download or the model fails, fall back to RoBERTa-GoEmotions (on disk) with CLIP L/14, and flag it: the row's
  backbone then differs.
- minor: the rest exactly as arm B4 at full scale (image-caption pairs, Eq. 2 fusion, multi-hot Stage 2 targets,
  paper training settings, label-free early stopping). No choice in this chat reads test results.
- minor: report the top-topic ("own") reading too, as earlier reports did; the paper uses the r20 reading.

## 5. Code and pitfalls

- New round folder `src/test/20261111_percept_original/`, module prefix `po_` (round folders go on `sys.path`; r25
  uses `pb_` in `src/test/20261111_paper_baselines/`). Reuse the `ct2_` and `pc_` code by import or a thin store
  switch; do not edit the old round folders.
- Features go to `/data/SSD2/pre_extract/artelingo_full/clip_l14_patch/` and
  `/data/SSD2/pre_extract/artelingo_full/modernbert_go_emotions/` (project-owned cache). Join by the extraction's
  sample IDs, never `range(n)` (constitution C6). The L/14 patch store is about 30 GB in fp16; that is fine on SSD2.
- ModernBERT needs `transformers` 4.48 or newer. Do not change the CoSiR env's packages; if its version is older,
  use a separate env (r25 may have made `/root/miniconda3/envs/CoSiR-vlm2vec`; check before making another) and say
  so in the report.
- GPU work through `res-queue gpu --est <duration> --label r26:<step>`. The claim test's paper block took 8,014 s for
  two arms; one arm with L/14 patches may take 1 to 2 hours. Before the training job, read
  `~/.claude/references/run-planning.md`: today is a weekday, so a long heavy local run must not run from 10:00 to
  19:00. If the estimate ends after 10:00, run the extractions now and queue the training for 19:00 (or DAS6 if a
  node is reserved), and say so in the next handoff.
- Reports: a short auto report under `docs/reports/auto/percept/` (CoSiR's sequence-number date, next after the
  latest; r25 adds one too, so check at write time) with its `reports_sum.md` row, and a half-page user-read report.
  The results table gives what the paper needs: mean AMI per aspect (r20 reading), three decimals.
- Do not call this run "faithful" (GLOSSARY row 8 needs all eight audit changes and PercepT's own data).
- Night or day mode: run `run-mode`; in night mode take the defaults above without asking.
- Delete nothing; list removals in `.scratch/pending_deletions.md`.

## 6. State at handoff

- Running: the r25 tab (its own chat; GPU job `r25:extract`, CPU job `r25:bertopic`). Nothing of this chat.
- Uncommitted: nothing of this chat (`docs/superpowers/constitution.md` was modified before; not ours).
- On disk over 1 GB: nothing.
