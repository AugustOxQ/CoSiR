# Gap 1 — symmetric mapper LR/epoch tuning for PercepT's (bug-fixed) Stage 2

Generated 2026-09-28 00:50:30. Mirrors [`candidate2_mapper_sweep_pilot_report.md`](candidate2_mapper_sweep_pilot_report.md)'s LR_GRID=(0.0003, 0.001, 0.003, 0.01), EPOCH_GRID=(100, 200, 400), and screen-then-4-seed-stress discipline, applied to PercepT's own fixed Stage 2 mapper instead of buddy's. Baseline (PercepT's original untuned lr=1e-3, epochs=100), previously published: **0.5925**; this run's own re-fit: **0.5843**.

**Note on reproducibility:** this run's own fresh Stage-1 re-fit scored 0.5843 at the untuned lr=1e-3/epochs=100 point, not the previously-published 0.5925 (-0.0082). Neither this script nor the original fixed pilot sets `torch.use_deterministic_algorithms`/`cudnn.deterministic`, so a fresh 500-epoch DEC self-sharpening re-fit is not bit-reproducible run-to-run at the same seed -- this is itself a real finding about this pipeline's stability, not a bug in this script. Every comparison below uses this run's own re-fit as the internal baseline, so the LR/epoch effect is measured consistently; the previously-published number is reported for context only.


## Part A: learning rate sweep (seed 42, 100 epochs)

| LR | macro AUC |
|---:|---:|
| 0.0003 | 0.5094 |
| 0.001 | 0.5843 |
| 0.003 | 0.7464 |
| 0.01 | 0.8692 |

Best LR: **0.01**.

## Part B: epoch sweep (seed 42, best LR)

| epochs | macro AUC |
|---:|---:|
| 100 | 0.8692 |
| 200 | 0.9000 |
| 400 | 0.9226 |

Best config: lr=0.01, epochs=400.

## Verdict

(Margins below are measured against this run's own re-fit baseline, **0.5843**, not the previously-published 0.5925 -- see the reproducibility note above.)

**lr=0.01, epochs=400 beat the screening margin** (seed 42: 0.9226 vs. 0.5843 baseline, +0.3383). 4-seed stress:

| seed | macro AUC |
|---:|---:|
| 42 | 0.9226 |
| 7 | 0.9227 |
| 123 | 0.9226 |
| 2024 | 0.9226 |

Stress mean: 0.9226 (min 0.9226, max 0.9227, std 0.0001).

**Symmetrically tuned PercepT Stage 2 mapper: 0.9226** (vs. this run's own untuned re-fit 0.5843, +0.3384; vs. the previously-published untuned number 0.5925, +0.3301).

**Updated comparison: buddy 0.8534 vs. symmetrically-tuned PercepT 0.9226 — a -0.0692 margin** (was +0.2609 against the untuned, previously-published comparator). This changes the headline conclusion and needs immediate attention.
