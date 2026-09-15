# Condition-Space Steerability Audit (Exp. 17.1) — Log

Spec: docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md (Experiment 17.1)
Plan: docs/superpowers/plans/2026-09-15-condition-space-audit.md

## Checkpoint discovery
Real run (`python src/test/20260915_condition_space_audit/discover_checkpoints.py`, 2026-09-15):

- redcaps_150k: 12 candidates (all from `res/CoSiR_condition_freeze_ablation/redcaps_150k/*_CoSiR_Experiment`; the `res/CoSiR_init_ablation/redcaps_150k/*_CoSiR_Experiment` glob in `REDCAPS_150K_GLOBS` matched 0 — real experiments there live one level deeper, under `init_buddies/`/`init_imgtxt/` subdirectories)
- impressions: 0 candidates — **STOP condition per brief step 6.** `IMPRESSIONS_GLOBS` (`res/CoSiR_init_ablation/impressions/*_CoSiR_Experiment`) matched 0 because the real directory structure nests experiments one level deeper: `res/CoSiR_init_ablation/impressions/init_buddies/*_CoSiR_Experiment`. Verified manually that adding that level (`init_buddies/*_CoSiR_Experiment`) yields 6 matching candidates that pass the filter (combine_side='img', conditioning_mode default 'asymmetric', initialization_strategy='buddies'). `checkpoints.json` was written as-is per the brief's exact code (`impressions: []`); Task 5 will need a corrected/fallback glob for impressions before it can proceed.

## Results
(to be filled after the real run)
