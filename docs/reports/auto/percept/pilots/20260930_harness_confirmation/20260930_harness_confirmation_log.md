# 2026-09-30 harness confirmation checks (QC1, QC2)

## Problem

Master report §6i found that the buddy-percept sweep's winner `m8x7ifx4`
cleared the Stage 1 gate on 4/4 seeds. The final review then showed that the
gate used k-NN *transfer* AMI (held-out paintings labelled by a vote into the
merged train topics). The investigation's Pareto bar (emotion > 0.1236, genre
> 0.1954) was set with *independent* held-out re-clustering, and transfer reads
about 0.01 higher on emotion. The winner's margin over the bar (0.003–0.008)
was smaller than that offset.

## Investigation

- **QC1:** `m8x7ifx4` was run through the sweep harness at its original
  seeds (42/7/123/2024, `transfer_k=40`) and at new seeds
  (11/23/57/101, `transfer_k` 40 and 20). Each run also scored the pilots'
  independent re-clustering AMI (`scripts/buddy_percept_sweep/pilot_metrics.py`).
  DAS6 tags: `qc1-orig`, `qc1-k40-a/b`, `qc1-k20-a/b`.
- **QC2:** the pilots' unchanged attention-h1 snapshot pilot was re-fitted at
  seeds 42/7/123/2024 and scored on both yardsticks
  (`run_baseline_seed_snapshot.py`). DAS6 tags: `qc2-s<seed>`.
- Tooling: commit `a906a89`. All jobs ran on DAS6 node403/404/405.
  Logs are in `logs/`.

## Root cause / finding

| system | seeds | independent emotion AMI mean | clears bar (independent) | gate emotion AMI mean (k=40) | clears gate |
|---|---|---:|---:|---:|---:|
| baseline (pilot) | 42/7/123/2024 | 0.1230 | 2/4 | 0.1341 | 3/4 |
| `m8x7ifx4` | original 4 | 0.1164 | 0/4 | 0.1295 | 4/4 |
| `m8x7ifx4` | new 4 | 0.1213 | 1/4 | 0.1314 | 4/4 |

The winner's 4/4 is an artefact of the transfer yardstick. On emotion it
does not beat the untuned baseline on either yardstick. Its gains are genre
AMI and Stage 2 AUC (0.9376 on the new seeds).

Side finding: the pilot is GPU-dependent. The DAS6 seed-42 re-fit gives
independent AMI 0.1187 / 0.2639, against the local GPU's 0.1249 / 0.2404.

## Solution / follow-up

Written up as master report §6j. The matched head-to-head
(`docs/superpowers/specs/2026-09-30-matched-percept-buddy-h2h-design.md`)
scores both systems on both yardsticks and runs every arm on the same GPU type.
