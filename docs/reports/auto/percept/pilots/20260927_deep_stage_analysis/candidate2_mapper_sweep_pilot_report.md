# Candidate 2 — buddy Stage 2 mapper LR/epoch sweep

Generated 2026-09-27 20:42:03. Baseline: candidate 1's adopted Variant C (merge K=16 + class-balanced loss), 4-seed mean **0.6334**. All sweep points below use that same merged vocabulary and loss weighting; only LR/epochs change.

## Part A: learning rate sweep (seed 42, 100 epochs)

| LR | macro AUC | min | median | max | top-1 acc |
|---:|---:|---:|---:|---:|---:|
| 0.0003 | 0.5441 | 0.4053 | 0.5071 | 0.8558 | 0.0673 |
| 0.001 | 0.6372 | 0.4309 | 0.6106 | 0.8977 | 0.2351 |
| 0.003 | 0.7751 | 0.5508 | 0.8036 | 0.9736 | 0.3203 |
| 0.01 | 0.8147 | 0.5992 | 0.8359 | 0.9845 | 0.3526 |

Best LR: **0.01**.

## Part B: epoch sweep (seed 42, best LR)

| epochs | macro AUC | min | median | max | top-1 acc |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.8147 | 0.5992 | 0.8359 | 0.9845 | 0.3526 |
| 200 | 0.8343 | 0.6426 | 0.8425 | 0.9865 | 0.3699 |
| 400 | 0.8460 | 0.6613 | 0.8503 | 0.9867 | 0.3851 |

Best config: lr=0.01, epochs=400.

## Part C: capacity

Skipped -- see verdict below for whether Parts A/B showed enough headroom to justify it.

## Verdict

**lr=0.01, epochs=400 beat the screening margin** (seed 42: 0.8460 vs. 0.6334 baseline mean, +0.2126). 4-seed stress:

| seed | macro AUC |
|---:|---:|
| 42 | 0.8460 |
| 7 | 0.8463 |
| 123 | 0.8462 |
| 2024 | 0.8461 |

Stress mean: 0.8461 (min 0.8460, max 0.8463, std 0.0001). This beats the candidate-1 baseline robustly across seeds -- adopt this LR/epoch setting.
