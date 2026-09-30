# Candidate 1 — Variant C 4-seed stress

Generated 2026-09-27 20:37:29. Merge (K=16) + class-balanced loss, combined. Companion to [`candidate1_stress_pilot_report.md`](candidate1_stress_pilot_report.md).

| seed | macro AUC | top-1 acc |
|---:|---:|---:|
| 42 | 0.6372 | 0.2351 |
| 7 | 0.6299 | 0.2498 |
| 123 | 0.6338 | 0.2522 |
| 2024 | 0.6326 | 0.2624 |

Summary: macro AUC mean=0.6334, min=0.6299, max=0.6372, std=0.0026; top-1 acc mean=0.2499.

All four seeds clear the 0.005 practical margin over baseline (0.5978) by a wide margin. Variant C's 4-seed mean (0.6334) is the best-performing configuration tested in candidate 1 -- recommended as the adopted configuration for buddy's Stage 2 headline number.
