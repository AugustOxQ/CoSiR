# 2026-09-29 factor-collapse verification log

**Problem.** The code review (`docs/reports/2026-09-29_cosir_v2_code_review.md`) makes two critical
claims: (1) the 32-factor space has collapsed to about one dimension; (2) the held-out split reuses
99.5% of training images. Asked to check both independently before planning stage (d).

**Investigation.**
- Reran `src/test/20260929_factor_collapse_probe/probe.py` unchanged: output byte-identical to the
  saved `probe_output.txt`.
- `verify.py` (this folder) tests two steps the probe leaves open. First, the probe's variance share
  and participation ratio (PR) are variance-weighted, and its factor-model reconstruction figure
  comes from the Task 6 report (a different, full-data fit). So `verify.py` fits a ridge linear
  readout from the cached codes' top-k principal components to CLIP features (100k train rows),
  scores held rows, and compares against CLIP's own PCA and spectrum. Second, it measures leakage
  by two independent keys: exact image-vector hash, and the annotation `painting` field. It also
  checks text-vector reuse. Output: `verify_output.txt`.

**Findings.**

| Held rows | Image | Text |
|---|---:|---:|
| CLIP centered: PC1 share / PR | 9.6% / 39.1 | 12.1% / 41.4 |
| Factor codes: PC1 share / PR | 86.5% / 1.32 | 86.5% / 1.33 |
| Readout from code PC1 only (rel. L2) | 0.616 | 0.538 |
| Readout from code PCs 1–10 | 0.547 | 0.506 |
| Readout from all 32 codes | 0.544 | 0.504 |
| CLIP mean-only / PCA-2 / PCA-5 / PCA-32 | 0.638 / 0.578 / 0.535 / 0.397 | 0.551 / 0.501 / 0.472 / 0.374 |

- **Claim 1 confirmed, with a refinement.** In variance terms the codes are one-dimensional.
  CLIP spreads variance over about 40 effective directions; the codes put 86.5% on one. In
  information terms they are "a few dimensions", not literally one: code PCs 2–10 still improve
  the readout (0.616 → 0.547 for image), and code PCs 11–32 add nothing. All 32 codes linearly
  carry about as much of CLIP as PCA rank 2 (text) to rank 4–5 (image). The conclusion stands:
  the space does not offer 32, or even about 10, separable axes for a condition to select among,
  and the dominant axis dominates any dot-product factor score.
- **Claim 2 confirmed by two independent keys.**
  - 99.54% of held rows have their exact image vector in train.
  - 99.71% of held rows show a painting present in train.
  - 99.89% of distinct held paintings appear in train.
  - Text vectors are almost never reused (0.19%).
  - So Tasks 7–9's held-out evaluations are text-disjoint but image-seen. How much this
    inflated those numbers is untested; it needs a painting-grouped rerun.

**Resolution.** Verification only; no source changes. Agrees with the review's recommendation to fix
factor discovery (with effective-rank / max-|r| gates) and move to a painting-grouped split before
stage (d). The review's "probable causes" (non-contrastive agreement loss, no decorrelation,
copy-satisfiable balance loss) are plausible but were not tested here.

## Correction (2026-09-29, later the same day): the readout numbers above were over-regularized

`verify.py`'s `ridge_readout` used an absolute penalty `lambda = 1e-3 * n` (i.e. 1e-3 in covariance
units). The codes' covariance eigenvalues have median ~2e-5 (min 1e-6, max ~0.21), so that penalty
suppressed nearly every non-dominant code direction. Re-measured on the same cached codes, split,
and 100k seeded fit rows (held-row relative L2):

| Readout from all 32 codes | Image | Text |
|---|---:|---:|
| lambda = 1e-3 * n (this log's table above) | 0.544 | 0.504 |
| lambda = 1e-3 * trace/d (scale-invariant, `src/eval/factor_gates.py`) | 0.494 | 0.467 |
| lambda = 0 (least squares) | 0.484 | 0.455 |

So the codes linearly carry about as much of CLIP as **PCA rank ~11-12 (image) / ~8 (text)**, not
rank 4-5 / 2 as stated above. The variance-based collapse findings stand unchanged (PC1 86.5%,
participation ratio 1.32, 374/496 pairs |r| >= .9). The corrected reading: the collapse is mainly a
**scale/geometry** collapse, not an information collapse. The information survives in directions
whose variance is ~10^4 times smaller than the dominant axis, which is why any dot-product score
over the codes (including s(I,T|c)) effectively sees only that axis. The code-PC readout curve in
the table above (k=1..32) is affected by the same over-regularization and should not be relied on.
