# 2026-11-09 fix diagnostics (EXPLORATORY: decides nothing; seed 42 development episodes only)

**Problem.** After the quick checks (`docs/reports/auto/v2/2026-11-08_new_method_quick_checks.md`, Sections 8 and 9),
every conditioned score lost "either rate" (aspect finding) when fused with the best condition-free score C2 (R@1 18.34
on seed 42). N6c beat C2 by only +0.15 [−0.06, 0.38] R@1. This run ranks candidate fixes for the next handoff. It is
not a check or a gate, and no other episode seed was read.

**What was run.** `diagnose_fixes.py` (CPU, 86 s) builds B = C2 with `crossfit_condition_free(cos, T_N1u, T_6u)`. It
asserts that `per_anchor(B)` equals the stored `matched__*` arrays of `per_anchor_n6c_gate.npz` exactly (it did). It
then scores eight terms on their own: N6's hard and soft readers; a contrastive hard reader (s at argmax Δ minus s at
argmin Δ); a contrastive soft reader (Σ Δ_h s_h / Σ |Δ_h|); the told partition (emotion to affect, style and genre to
image) and its told-minus-other version; and the refit D0 label probes, told and told-minus-other. The refit
reproduced the stored D0 told arrays exactly. Each term is then fused on B with A′'s nested min-margin cross-fit,
`crossfit_nested(B, B, T, parity)`. For every term the control ranked exactly as B. T6, T6pm and T6pm_soft are also
run as cascades on B with k = 2 and 3. The run ends with the hard reader's correctness and per-direction metrics for
the two best fusions. Per-direction metrics come from `per_anchor` on a score dict that holds one direction in both
direction slots. Outputs: `results/diagnose_fixes.{json,txt}` (gitignored).

**Reading (exploratory, seed 42 only).** The reader's choice of partition is the lever; the shape of the term is not.
- **Wrong picks cause the either loss.** N6's hard pick matched the told partition in 52.4% [51.8, 53.0] of
  rankings, and under both conditions in only 28.1% of episodes. On those episodes, T6 fused on B gained +1.16
  [0.80, 1.54] R@1 and lost no either (+0.12 [−0.43, 0.66]). On the rest it lost −0.61 [−0.94, −0.29] either for
  −0.19 [−0.41, 0.03] R@1.
- **The told partition would help.** Fused on B, it reached +1.50 [1.22, 1.79] R@1 (gain 4.56, either −1.57), against
  +0.19 [0.01, 0.38] for N6's reader. So a better partition reader on the same heads is worth about 1.3 R@1. Most of
  that comes from emotion × genre and emotion × style. On style × genre the told partition is the same under both
  conditions (image), so its gain there is exactly 0.
- **The contrastive readers do not help.** T6pm and T6pm_soft raise term-only gain (5.10 and 6.07 against 4.41) but
  cost more either. Fused on B they give +0.07 and +0.12 R@1, below T6.
- **Subtracting the other aspect costs either, even for told terms.** T6oracle_pm gives +1.04 against +1.50 for
  T6oracle; D0pm gives +9.81 against +12.28 for D0told (either −4.27 against +4.00).
- **Cascades lose R@1 at k = 2 and 3** (−0.55 to −2.59), because the either cost outweighs the gain.

Caveats:
- For any condition-antisymmetric term (Δ_b = −Δ_a exactly, so T6pm and T6pm_soft flip sign between conditions), a
  k = 2 cascade's either rate is fixed by B's top two candidates. That is why the two k = 2 cascades share either
  30.37 (checked in the scratchpad, not a bug).
- B's cross-fit picks were tuned on the same parity halves that the fusion's cross-fit reuses, so the fused
  differences carry a small second-order leak.
- All of this is one seed and a development look.
