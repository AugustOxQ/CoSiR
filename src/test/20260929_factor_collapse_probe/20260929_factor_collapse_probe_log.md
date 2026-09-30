# 2026-09-29 factor-collapse probe log

**Problem.** A code review asked whether each CoSiR-v2 component serves Candidate A's goal. The factor-discovery stage had passed its Task 3–6 checks, yet Task 9 found that factors were strongly redundant and that the naive rule's lift is small once weights are scale-fair.

**Investigation.** `probe.py` reads Task 9's cached seed-42 factor codes and the real ArtELingo features. It measures code sparsity, the principal-component variance spectrum and participation ratio, factor correlations, and matched vs. shuffled cross-modal code cosine. It compares reconstruction against mean-only and PCA baselines. It also hashes image feature rows to check the train/held split for image reuse. No training is involved. The output is in `probe_output.txt`.

**Findings.**
- The codes are effectively one-dimensional: principal component 1 holds 86.5% of centered variance, the participation ratio is 1.32, and 374/496 factor pairs have |r| ≥ 0.9.
- The codes are dense: about 23 of 32 factors are active per row.
- Reconstruction is at about PCA rank 1–4 level.
- The row-level split leaves 99.54% of held rows with their exact image vector in train.

**Root cause (probable).** The paired-agreement loss is non-contrastive, nothing decorrelates the factors, and the usage-balance loss can be satisfied by copies of one axis. Sparsity pressure is weak and the inputs are uncentered.

**Resolution.** Diagnostic only; no source changes. The full write-up and fix recommendations are in `docs/reports/auto/v2/2026-09-29_code_review.md`.
