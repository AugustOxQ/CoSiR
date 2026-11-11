# PercepT on CLIP L/14 and ModernBERT: round r26 run log

> 2026-11-11 · [Build handoff](../../../docs/superpowers/handoffs/2026-11-11-r26-percept-original-handoff.md) · [Run handoff](../../../docs/superpowers/handoffs/2026-11-11-r26-run-handoff.md) · Decisions: [decisions.md](decisions.md) · [Auto report](../../../docs/reports/auto/percept/2027-01-15_percept_original.md)

## Problem

The paper's Table 2 needs a PercepT row on PercepT's own encoders: CLIP ViT-L/14 for image and text, and
ModernBERT-base tuned on GoEmotions for the emotion vector. The run trains PercepT at paper settings with 40 topics
(code tag `percept_paper_k40`) on the 42,969 ArtELingo train paintings, seeds 0 to 4, and scores the 12,281 test
paintings with r17's scorer. The strict Algorithm 1 cut runs beside it and is reported with its collapse check.
Table 3 also needs the test topic probabilities with painting names.

## Timeline

All times are Amsterdam local time.

| Time | Evidence | Step |
|---|---|---|
| 2026-11-11 09:28 | build handoff, `done.json` | Both feature stores extracted and checked (build chat). |
| 2026-11-11 09:29 | run chat | Run handoff read. DAS6: node402 reserved, but each of its 3 GPUs holds a 9.6 GB process that is not a cluster job, so `cluster launch` picks none. |
| 2026-11-11 09:29 | `po_queue_run.sh --smoke` | Smoke queued behind r21's stage 0c. |
| 2026-11-11 09:36 | smoke logs | Smoke ends: `R26_RUN smoke rc=0 done: stage1 stage2 held score`. |
| 2026-11-11 09:40 | `logs/stage1.log` | Full run started as a power probe. |
| 2026-11-11 09:43 | monitor | Stage 1 one-minute means 255 W and 264 W (peak 320 W): heavy. Probe stopped; no output written. |
| 2026-11-11 09:45 | 855ab41b | Full run queued for 19:00 (`po_queue_run.sh --at 19:00`); decision line added. Paper tab told the new time. |

## Smoke

Seed 0, smoke sizes (200 fit paintings), outputs in the chat's scratchpad. Every step ran on the GPU in under 20 s.

| Step | Result | GPU power (3 s samples) |
|---|---|---|
| Stage 1 | rc=0, 13.9 s for seed 0 | max 182 W, mean 73 W |
| Stage 2 | rc=0, both cuts | max 193 W, mean 140 W |
| Held | rc=0, k40 6.5 s, strict 4.5 s | max 140 W, mean 113 W |
| Score | rc=0; k40 and strict both collapsed (expected on 200 paintings) | CPU |

The smoke power numbers are no guide to the full run: each step was mostly loading.

## Start time

The weekday rule keeps a long heavy local run (over about 200 W) out of 10:00 to 19:00. The smoke could not tell, so
the full run started at 09:40 as a short probe. Full-scale Stage 1 drew 255 W and 264 W over its first two minutes, so
the run is heavy. Stage 1 alone takes about 55 minutes, so it would have run past 10:00. The probe was stopped at
2.4 minutes, during autoencoder pretraining (epoch 74); it had written only its log. The full run waits until 19:00:
GPU about 19:00 to 20:45, then scoring and the test export, end about 21:15, well before 11:00.

## Full run

`po_queue_run.sh --at 19:00`, local RTX 3090, seeds 0 to 4, both cuts. End line:
`R26_RUN rc=0 done: stage1 stage2 held score export 2026-11-11 20:09`.

| Step | Start | Run time | Note |
|---|---|---|---|
| Stage 1 (GPU) | 19:00 | 55.9 min | 5 seeds, both cuts |
| Stage 2 (GPU) | 19:55 | 11.1 min | faster than the 25 to 40 min estimate |
| Held (GPU) | 20:07 | 1.5 min | |
| Score (CPU) | 20:08 | 0.2 min | `score --record`: `records/po_scores.json`, `records/po_table.md` |
| Export (CPU) | 20:08 | 0.3 min | `OUT/test/<cut>/`, `OUT/val/<cut>/` |

### Results

Test AMI, 12,281 paintings (genre on 9,949), mean over seeds and 3 k-means states. Full table: `records/po_table.md`.

| Cut | Seeds in the mean | Reading | Emotion | Style | Genre |
|---|---|---|---:|---:|---:|
| K = 40 | 5 of 5 | class count (9 / 27 / 10), Table 2 | 0.064 | 0.262 | 0.474 |
| K = 40 | 5 of 5 | k = 20, supplementary | 0.068 | 0.256 | 0.460 |
| K = 40 | 5 of 5 | own (argmax) | 0.076 | 0.251 | 0.419 |
| strict | 0 of 5 | all | n/a | n/a | n/a |

K = 40 per seed, class count: emotion 0.060 to 0.067, style 0.250 to 0.276, genre 0.457 to 0.512 (seed 0 highest).
Mean dev AUC of Stage 2: 0.965. Top-topic share of the fit pairs 0.051 to 0.072, far under the 0.5 collapse bar.

Strict Algorithm 1 (gamma 2) collapsed on all 5 seeds. Seeds 0, 1, 3 and 4 kept 96, 99, 99 and 98 of the 100
topics (the rule's bar is over 90), with top-topic shares of 0.018 to 0.031. Seed 2 kept a single topic. The claim test
saw the same on CLIP B/32 and Qwen (10 of 10 runs collapsed).

### Checks

- **Independent recompute** (`po_recheck.py`, log `records/po_recheck.log`): reads the exported test files, joins the
  claim test's labels (`ct_held.painting_labels`) by painting name, and runs scikit-learn k-means and AMI itself. All
  six class-count and k = 20 means equal `po_scores.json` at full precision.
- The test exports equal the held outputs `OUT/held/L1__k40__s<seed>.npz` bit for bit; names are unique and in the
  same order for every seed; probabilities are finite and in [0, 1].

### Comparators in Table 2 (class count, ArtELingo test; from the paper's `Tables/main_clustering.tex`)

| Row | Emotion | Style | Genre |
|---|---:|---:|---:|
| PercepT, CLIP L/14 + ModernBERT (this run) | 0.064 | 0.262 | 0.474 |
| PercepT, Qwen3 (r17's matched core version) | 0.068 | 0.192 | 0.424 |
| CoSiR (c*, Qwen3) | 0.102 | 0.411 | 0.510 |

On genre, c*'s mean (0.510) lies inside this run's seed range (0.457 to 0.512). A seed-level test against c* was not
run (c*'s per-seed test values are not in r17's records; not checked).

### Hand-over

| Time | To | What |
|---|---|---|
| 2026-11-11 09:46 | CoSiR paper sec4 tables 5 | New time: files at about 21:15. |
| during the wait | sec4 tables 5, 6, 7 | Each tab asked to move the hand-over to its successor; sent to the highest at the end. |
| 2026-11-11 20:15 | CoSiR paper sec4 tables 7 | Table 3: `OUT/test/k40/L1__s{seed}.npz`; k40 seeds 0 to 4, 5 of 5 not collapsed; Table 2 row for information. |
| 2026-11-11 20:15 | CoSiR paper experiments 2 | Table 2 row (class count) and k = 20 for the supplementary. The tab filled the Table 2 row (paper repo 609a3cf) and scored the supplementary cluster-count grid from `OUT/held/L1__k40__s*.npz`; its six gate cells equal `po_scores.json` exactly (main 902e0539). sec4 tables 7 also confirmed the row. |
