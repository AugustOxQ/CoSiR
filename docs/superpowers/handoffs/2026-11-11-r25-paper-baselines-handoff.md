# Handoff: r25, score BERTopic and VLM2Vec-V2 as clustering comparators on the ArtELingo test split, for the paper's main table

## Brief
- Done: the paper chat wrote 4.1 (compared methods name BERTopic and VLM2Vec, two of PercepT's clustering baselines).
- State: neither method has been run; the paper's main clustering table lists them with empty cells.
- Next: run both on the ArtELingo test split (k-means k = 20 AMI per aspect), short auto report and user-read report.
- Blocked: nothing; the cutoff is 2026-11-14, after which the rows leave the paper.

> Written 2026-11-11 05:02 by the "paper method" tab. Loop step reached: r25 is a small loop opened from a paper chat,
> with the design already set by the user (below); this chat does step 8 and the short reports. Direction log entry:
> none yet; this chat adds a short one at its end.

## 1. Read in this order

1. This handoff, all of it.
2. `docs/reports/auto/percept/2027-01-06_outside_methods.md` (r18): how outside methods were scored beside c* on the
   same cells, the interval form and the WIN/LEVEL/BELOW rule. Its code, to reuse rather than rewrite:
   `src/test/20261103_outside_methods/` (`om_data.py` data and splits, `om_cells.py` scoring, `om_paths.py`,
   `om_facts_brief.md`).
3. `docs/reports/literature/2026-12-05_percept_neighbourhood.md` lines 45 to 70: PercepT's baseline setups
   (BERTopic multimodal with ViT-B/32 and MiniLM, UMAP plus HDBSCAN, merged to 100 topics; VLM2Vec v1 and v2).
4. The paper's 4.1 metrics paragraph:
   `docs/paper/cvpr2027_aspect_spaces/CVPR2027-CoSiR-Paper/Sections/4_experiments.tex` (k-means with 20 clusters,
   mini-batch, three k-means seeds, AMI against each aspect's labels, test split). The instructions:
   `.../CVPR2027-CoSiR-Paper/Tables/instructions.tex`.

Skills: none.

## 2. Decided by the user, not to be reopened

All on 2026-11-11 in the paper chat (`docs/paper/cvpr2027_aspect_spaces/README.md`, rows of that date):
- BERTopic and VLM2Vec are comparators in the paper's main clustering table; CEMTM is skipped (it needs retraining).
- **VLM2Vec-V2** (the 2B model on Qwen2-VL), image embeddings under **each aspect instruction**, adapted to its prompt
  format; one embedding per aspect, like our teacher.
- **BERTopic on the test captions, with 20 topics**, to match k = 20.
- **ArtELingo test split only**, in the Qwen3 setting of the paper; MMArt maybe later, not in this chat.
- Cutoff: numbers by 2026-11-14, or the two rows come out of the paper.

## 3. Where things stand

- c* (Qwen) test AMI at k = 20, five seeds, is in the r18 and earlier reports (e.g. 12-28); the paper's main table is
  being prepared in the paper chat from those reports.
- No VLM2Vec features exist on disk; `sentence-transformers/all-MiniLM-L6-v2` is in the HF cache.
- GPU check passed at 05:00 (driver 595.99.02 in kernel, library and `nvidia-smi`); GPU idle.

## 4. Open points for the next step

Agent defaults (take them, write a decision line in the round folder's `decisions.md`, flag them at the top of the
next handoff and the report):
- key: **BERTopic per painting.** One document per test painting (its captions joined), so each painting gets one
  topic. Alternative: per-caption topics, painting topic by majority.
- key: **BERTopic topic count.** Its default pipeline (MiniLM, UMAP, HDBSCAN), reduced to 20 topics, outliers
  reassigned with `reduce_outliers` so every painting has a topic. Alternative: k-means with 20 clusters as its
  cluster model. Three UMAP seeds (0, 1, 2), averaged, like the three k-means seeds.
- minor: VLM2Vec prompt wording: the aspect instructions of Table 1 in VLM2Vec's own format; record the exact strings.
- minor: whether VLM2Vec needs its own conda env (see section 5).

## 5. Code and pitfalls

- New round folder `src/test/20261111_paper_baselines/`, module prefix `pb_` (round folders go on `sys.path`).
- Features go to `/data/SSD2/pre_extract/artelingo/vlm2vec_v2/` (project-owned cache); join by the extraction's
  sample IDs, never `range(n)` (constitution C6).
- VLM2Vec-V2 may need a different `transformers` version than the CoSiR env. Do not change the CoSiR env's packages;
  if needed, create a separate env (`/root/miniconda3/envs/CoSiR-vlm2vec`) and say so in the report.
- GPU work through `res-queue gpu --short` or `--est`, label `r25:extract`; the extraction is short and light, so it
  may run in weekday working hours. CPU-heavy steps (BERTopic UMAP) through `res-queue cpu --label r25:bertopic`.
- Scoring identical to the c* cells it sits beside (same labels, same painting set, same k-means settings), so the
  paper can put the rows in one table. The test split shapes nothing: no choice in this chat reads test results.
- Reports: a short auto report under `docs/reports/auto/percept/` (CoSiR's sequence-number date, next after the latest)
  with its `reports_sum.md` row, and a half-page user-read report. The numbers the paper needs (mean AMI per aspect,
  and the difference to c* with its 95% interval) also go in the report's results table.
- Night or day mode: run `run-mode`; in night mode take the defaults above without asking.
- Delete nothing; list removals in `.scratch/pending_deletions.md`.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing of this chat (`docs/superpowers/constitution.md` was modified before; not ours).
- On disk over 1 GB: nothing.
