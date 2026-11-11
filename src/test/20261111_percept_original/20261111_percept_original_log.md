# PercepT on CLIP L/14 and ModernBERT: round r26 run log

> 2026-11-11 · [Build handoff](../../../docs/superpowers/handoffs/2026-11-11-r26-percept-original-handoff.md) · [Run handoff](../../../docs/superpowers/handoffs/2026-11-11-r26-run-handoff.md) · Decisions: [decisions.md](decisions.md) · Auto report: (added at report time)

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

(Filled in when the run ends.)
