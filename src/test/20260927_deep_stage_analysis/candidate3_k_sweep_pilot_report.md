# Candidate 3 — buddy topic-count (K) sweep

Generated 2026-09-27 20:48:34. Current adopted K=19, Stage 2 macro AUC 0.8461 (candidate 2's result). K here is swept via Leiden resolution on the already-trained, frozen embedding (no InfoNCE retraining) -- see module docstring for why this makes the candidate cheap enough for local GPU, no DAS6 needed.

Each row: raw Leiden K, K after merging any topic below 1% train share, held-out emotion/genre AMI (Pareto bar: emotion > 0.1236 AND genre > 0.1954), and Stage 2 macro AUC using the candidate 1+2 adopted recipe (class-balanced loss, lr=0.01, epochs=400).

| resolution | raw K | merged K | emotion AMI | genre AMI | clears bar | Stage 2 macro AUC |
|---:|---:|---:|---:|---:|---|---:|
| 2 | 138 | 33 | 0.1282 | 0.1855 | no | 0.8568 |
| 1 | 120 | 17 | 0.1296 | 0.1997 | yes | 0.8406 |
| 0.5 | 111 | 9 | 0.1142 | 0.2555 | no | 0.8826 |
| 0.25 | 106 | 4 | 0.0834 | 0.1863 | no | 0.9078 |

## Verdict

**No tested K meaningfully beats the current K=19 configuration** on Stage 2 macro AUC while also clearing the Pareto bar (best: K=4 at 0.9078). Candidate 3 is closed as tested with a negative/neutral result; K=19 (merged to 16 for Stage 2) remains the adopted configuration. Move to candidate 4 (richer multi-label targets) or consider this investigation's Stage 2 improvements complete for now.
