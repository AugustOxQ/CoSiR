# ArtELingo PercepT Stage 2 architecture ablation pilot

Generated automatically, 2026-09-23 03:19:28.

## Shared Stage-1 K=60/40 seed-42 reproduction

Stage 1 was re-fit exactly once before the frozen `q > 1.2/40` targets and all four plain-linear runs. Every run shares that frozen encoder, 40 surviving centers, and global pooled CLIP image embeddings.

| metric | established | re-fit | absolute difference | status |
|---|---:|---:|---:|---|
| held-out emotion AMI | 0.1238 | 0.1238 | 0.0000 | reproduced |
| held-out genre AMI | 0.2617 | 0.2617 | 0.0000 | reproduced |

## Plain linear classifier on L2-normalized pooled CLIP embeddings

Each row uses the shared frozen `q > 1.2/40` targets, a single `nn.Linear(512, 40)` on the existing L2-normalized global pooled CLIP image embeddings, `lr=3e-3`, and 100 full-batch epochs. Per-topic AUC scoring and the train-marginal baseline are imported from the Stage-2 sweep pilot.

| classifier-init seed | held-out macro AUC | train-marginal macro AUC | min per-topic AUC | median per-topic AUC | max per-topic AUC | skipped topics |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.7170 | 0.5000 | 0.3047 | 0.7311 | 0.9466 | 0 |
| 7 | 0.7178 | 0.5000 | 0.3050 | 0.7279 | 0.9473 | 0 |
| 123 | 0.7171 | 0.5000 | 0.2965 | 0.7343 | 0.9470 | 0 |
| 2024 | 0.7158 | 0.5000 | 0.2913 | 0.7339 | 0.9489 | 0 |

| seed-summary statistic | held-out macro AUC |
|---|---:|
| mean | 0.7169 |
| min | 0.7158 |
| max | 0.7178 |

## Direct comparison with attention pooling

The attention-pooling result is cited, not retrained here, from `percept_stage2_best_config_stress_pilot_report.md`. Both architectures use the same frozen `q > 1.2/40` targets, `lr=3e-3`, 100 epochs, and four initialization seeds.

| architecture | held-out macro AUC mean | seed range |
|---|---:|---|
| attention pooling over 50 patch tokens | 0.8256 | 0.8248-0.8272 |
| plain linear on pooled embedding | 0.7169 | 0.7158-0.7178 |

**Verdict:** Patch attention pooling meaningfully outperforms the plain pooled-embedding linear baseline under the investigation's non-overlapping-range robustness standard. Its mean macro-AUC advantage is 0.1087, and the ranges are cleanly separated: attention 0.8248-0.8272 versus plain linear 0.7158-0.7178.
