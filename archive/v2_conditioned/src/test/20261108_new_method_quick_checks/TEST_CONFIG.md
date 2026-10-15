# The configuration under test on fresh seeds (fixed before any test episode is built)

**Date:** 2026-10-04. Applies `DECISION_RULE.md` row 1 and §6 (commit 7e50f18). Nothing here changes a rule.

## Why this configuration

The seed-42 development run (`run_checks.py`, results `results/checks_seed42.json` and `results/decision.json`, run at
bc36e72) applied the decision table mechanically. Exactly one configuration passed: **N1-nested-A3**, with
R@1 minus its own control +0.24 [0.09, 0.39] and condition gain +0.26 [0.05, 0.47] (painting-clustered bootstrap, 5,000
resamples, seed 42). The decision was row 1. D0 read "close" (Inferred-hard gain 13.33 against the threshold
0.5 × 21.06 = 10.53).

## The fixed configuration

- **Model:** A3, `src/test/20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt`, SHA-256
  dadfef1bed95bbd647c39fa0ea483edea78627d440a23b5a032083e76dd469e2 (E3's pick).
- **Score:** the nested score z(cos) + λ_u·z(T_u) + λ_a·z(T_N1), with T_u A3's uniform factor term
  (`agreement_term(..., uniform=True)`) and T_N1 the centered agreement term (`centered_term`), from
  `src/eval/aspect_quick_checks.py` (SHA-256 ba97d39e…, commit f6c49f2), `src/eval/aspect_nested.py` (fadd1fd5…) and
  `src/eval/aspect_scorers.py` (0c5d0bf6…).
- **Weights:** chosen on each test seed by A′'s min-margin parity cross-fit (`crossfit_nested`, λ_u ∈ {0, 0.5, 1, 2, 4,
  8, 16}, λ_a ∈ {0, 0.25, 0.5, 1, 2, 4, 8, 16}), rerun on that seed's own anchor-parity halves, as E3 did on seed 43.
  On seed 42 both halves picked (8, 2); that pick is recorded, not carried over.
- **Own condition-free control:** the nested uniform control z(cos) + σ·z(T_u), cross-fitted the same way.

## Test (DECISION_RULE.md §6, unchanged)

Seeds 45, 47 and 48; selection rows; 4,096 episodes per aspect pair each, built with
`src/test/20261030_aspect_baselines/run_baselines.py --episodes-seed <s>`, which also gives cosine and RCA on them.
GO = on the three seeds pooled (clusters = anchor paintings, one cluster per painting across seeds), the paired R@1 and
condition-gain differences against cosine, RCA and the own control all have 95% lower bounds above 0 (5,000 resamples,
seed 42). Each seed is reported on its own, descriptively.
