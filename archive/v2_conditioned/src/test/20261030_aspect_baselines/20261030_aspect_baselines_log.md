# E1 aspect baselines log (2026-10-30)

Problem: set the Tier-1 GO bar for Task 13 by scoring raw metric-from-pairs baselines, the agreement rule on SE/C0/R3
codes and an SE uniform-weight control on ArtELingo selection aspect episodes.

Steps:
1. Wrote `run_baselines.py` (CPU only; codes encoded once on CPU, cached to `results/codes_*.npz`). Smoke at n=64 and
   n=512 passed; cosine gain asserted 0, all per-anchor values finite.
2. Review passed (seed-42 cosine R@1 12.9618 and rca recompute bit exact). Hygiene commit: refuse to overwrite real
   outputs without `--overwrite`, codes provenance re-hash and `results/codes_provenance.json`, dead code removed.
3. Full runs, seeds 42 and 43 (about 400 s each, 4,096 episodes per pair).
4. Report: `docs/reports/auto/v2/2026-10-30_aspect_baselines.md`; figures built by
   `docs/reports/assets/2026-10-30_aspect_baselines/build_figures.py`.

Results: GO bar rca on seed 42 (mean 6.74; R@1 13.38, gain 0.10); on seed 43 rca 6.73 is below cosine 6.76. SE_uniform
R@1 16.30 / 16.58 at gain 0 against cosine 12.96 / 13.53.

Issues: four GO candidates (diag, diag_relu, bilinear, xing) plus C0 tie the cosine exactly on seed 43 (lambda 0 on both
halves). No source files edited, so no change-log entry.
