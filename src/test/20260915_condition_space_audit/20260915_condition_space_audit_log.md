# Condition-Space Steerability Audit (Exp. 17.1) — Log

Spec: docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md (Experiment 17.1)
Plan: docs/superpowers/plans/2026-09-15-condition-space-audit.md

## Checkpoint discovery
Initial run (`python src/test/20260915_condition_space_audit/discover_checkpoints.py`, 2026-09-15, brief's original globs):

- redcaps_150k: 12 candidates (all from `res/CoSiR_condition_freeze_ablation/redcaps_150k/*_CoSiR_Experiment`; the `res/CoSiR_init_ablation/redcaps_150k/*_CoSiR_Experiment` glob in `REDCAPS_150K_GLOBS` matched 0 — real experiments there live one level deeper, under `init_buddies/`/`init_imgtxt/` subdirectories)
- impressions: 0 candidates — **STOP condition per brief step 6.** `IMPRESSIONS_GLOBS` (`res/CoSiR_init_ablation/impressions/*_CoSiR_Experiment`) matched 0 because the real directory structure nests experiments one level deeper: `res/CoSiR_init_ablation/impressions/init_buddies/*_CoSiR_Experiment`. Flagged to coordinator; ruling confirmed the bug and directed a fix (see below).

### Fix round (coordinator-directed, commit after `2a6e45e`)
Corrected `REDCAPS_150K_GLOBS` and `IMPRESSIONS_GLOBS` in `discover_checkpoints.py` to include the `init_buddies/` path segment (deliberately excluding the sibling `init_imgtxt/`, which the downstream `initialization_strategy == "buddies"` filter would reject anyway):

```python
REDCAPS_150K_GLOBS = [
    "res/CoSiR_init_ablation/redcaps_150k/init_buddies/*_CoSiR_Experiment",
    "res/CoSiR_condition_freeze_ablation/redcaps_150k/*_CoSiR_Experiment",
]
IMPRESSIONS_GLOBS = [
    "res/CoSiR_init_ablation/impressions/init_buddies/*_CoSiR_Experiment",
]
```

Re-run after fix:
- redcaps_150k: **15 candidates** (12 from `condition_freeze_ablation` + 3 newly found under `init_ablation/redcaps_150k/init_buddies/`)
- impressions: **6 candidates** (all newly found under `init_ablation/impressions/init_buddies/`)

Both counts now nonzero — Task 5 unblocked for both datasets. `checkpoints.json` was regenerated with the corrected candidate lists.

## Results

Gate verdict: **positive**. Phase A (raw CLIP, the primary gate) found one of eight
(dataset, axis, modality) cells clearing the decision rule: `redcaps_150k / warmth /
img`, z=2.53, folded AUC=0.965 vs. control 0.69±0.11 — image-modality-specific, all
other cells (including `warmth` on text) null. Phase B (existing checkpoints)
qualitatively corroborates `warmth` as decodable post-training, caveated by
duplicate checkpoint directories inflating the raw 15/15 count. Per the spec's
decision rule, Experiment 17.2 should be scoped to the `warmth` axis (RedCaps,
image modality). Full write-up: `docs/reports/2026-09-15_condition_space_audit.md`.
