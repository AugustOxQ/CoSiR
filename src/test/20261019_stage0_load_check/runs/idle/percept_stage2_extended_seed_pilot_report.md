# ArtELingo PercepT Stage 2 extended seed pilot

Generated automatically, 2026-10-19 03:03:47.

## Shared Stage-1 K=60/40 seed-42 reproduction

Stage 1 was re-fit exactly once, under forced-deterministic cuDNN/cuBLAS execution, before the frozen `q > 1.2/40` targets and all 14 mapper runs. Every mapper run shares that frozen encoder, 40 surviving centers, and cached patch features. The reference values below are this deterministic pipeline's own observed fixed point (confirmed identical across two prior back-to-back runs), not the original non-deterministic citation of 0.1238/0.2617 -- see the module docstring for why exact reproduction of that citation is not a meaningful target under forced determinism. All 14 mapper seeds (including the four previously cited: 42, 7, 123, 2024) are trained fresh against this fit, since none of them are target-consistent with the old non-deterministic clustering.

| metric | deterministic reference | re-fit | absolute difference | status |
|---|---:|---:|---:|---|
| held-out emotion AMI | 0.1225 | 0.1287 | 0.0062 | FAILED |
| held-out genre AMI | 0.2274 | 0.2804 | 0.0530 | FAILED |

**Reproducibility failure.** The shared Stage-1 re-fit was not within the absolute AMI tolerance of 0.002 of this pipeline's own deterministic reference; no Stage-2 mapper was trained, so this pilot cannot silently use a different clustering.
