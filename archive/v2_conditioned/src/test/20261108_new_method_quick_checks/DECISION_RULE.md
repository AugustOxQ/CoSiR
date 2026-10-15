# Decision rule for the three quick checks D0, N1, N2 (committed before any check is run)

**Date:** 2026-10-04. **Spec:** `docs/superpowers/specs/2026-10-04-new-method-quick-checks-design.md` (approved by the
user on 2026-10-04), §4 (the checks) and §5 (the decision). This file restates §5 with the numbers the spec left open:
the D0 threshold (decided with the user: Inferred keeps at least half of Told's gain over cosine) and the stop rules of
§4.2 and §4.3. It was committed before any scorer of this folder existed and before any D0, N1 or N2 number was read.
The decision-table outcome goes to the user before any further step is taken.

## 1. Data, inputs and metrics

- **Development episodes:** E1's seed-42 selection episodes, `src/test/20261030_aspect_baselines/results/episodes_seed42.npz`
  (SHA-256 12af9794…; 4,096 per aspect pair, 12,288 pooled), loaded through E3's `EvalContext` (per-pair SHA-256 checked
  against `baselines_seed42.json`, NaN outside selection rows).
- **Eligible checkpoints** (may pass §4): **A3** (`src/test/20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt`,
  SHA-256 dadfef1bed95bbd6…, asserted equal to E3's `picked.json`), **C0** and **SE** (cached codes
  `src/test/20261030_aspect_baselines/results/codes_{C0,SE}.npz`, SHA-256 71559d05… and 845d6cd3…).
  **L3 and LT** (label-trained) are scored as diagnostics only and can never pass. D0 reads evaluation labels on
  training rows and is a diagnostic only; it can never pass §4 either.
- **Metrics** (spec §1): R@1, other-aspect rate, condition gain = R@1 − other, either rate = R@1 + other, per anchor
  averaged over both conditions and both directions, pooled over the three aspect pairs. All in percentage points.
- **Uncertainty:** 95% percentile intervals of the bootstrap that resamples anchor paintings (5,000 resamples, seed 42;
  `src.eval.aspect_metrics.compare` / `summarize`). A comparison "beats" when the paired difference's lower bound is
  above 0.

## 2. D0 (label-probe aspect reading; diagnostic)

- **Told:** p_A(query) · p_A(candidate) on the conditioned aspect A, each item scored with its own modality's probe.
- **Inferred:** Δ_h = S_h − C_h per aspect h (emotion, style, genre), from within-pair posterior agreement over the 4
  support pairs (S_h) and the 4 contrast pairs (C_h). *Hard:* the aspect with the largest Δ_h, scored as Told. *Soft:*
  weights max(Δ_h, 0) normalised to sum 1 over the three aspects, summed Told scores; when every Δ_h ≤ 0 the soft weights
  fall back to uniform (1/3 each), and the share of such episodes is reported.
- **The decision variant** is whichever of Hard and Soft has the larger pooled condition gain (point estimate); it is
  the variant N6 would use.
- **Threshold (decided with the user).** Cosine's condition gain is exactly 0, so "Told's gain over cosine" is Told's
  pooled condition gain G_told. D0 reads **close to Told** if the decision variant's pooled condition gain is at least
  **0.5 × G_told** (point estimates), and **far below Told** otherwise. The paired difference
  gain(Inferred) − 0.5 × gain(Told) with its interval, the R@1 analogue and the per-pair values are reported as
  description and do not change the reading.
- **Guard.** If Told's gain has a lower bound ≤ 0, D0 cannot be read: no table row is applied and the result goes to the
  user as is.

## 3. N1 and N2 configurations and their own condition-free controls

| Config | Score | Its own condition-free control |
|---|---|---|
| **N1-nested-{A3, C0, SE}** | Cross-fitted nested score z(cos) + λ_u·z(T_u) + λ_a·z(T_N1), with T_u the existing uniform factor term of that checkpoint and T_N1 the centered rule of spec §4.2; λ grid and min-margin parity cross-fit exactly as A′ (`src.eval.aspect_nested.crossfit_nested`) | The nested uniform control z(cos) + σ·z(T_u) of the same checkpoint, cross-fitted as in A′ |
| **N2-{2, 3, 5}-agree** | A3's cross-fitted nested uniform control ranks; its top k (k = 2, 3, 5) are reordered by A3's current agreement term T_a (ties inside the top k broken by the control score) | The unreordered A3 nested uniform control (the k = 1 cascade) |
| **N2-{2, 3, 5}-N1** | As above with T_N1 of A3 as the reorder term; computed only if N1 survives its stop rule on A3 (§5) | As above |

Term-only scores (T_N1 alone, the current T_a alone, diagonal KISSME on the same codes, N1's centered uniform term)
are reported for every checkpoint as readings and baselines; they are not configurations. Diagonal KISSME on codes:
each factor's code standardised by its std over scorer-train rows (image and caption codes pooled), per-factor second
moment of the image-minus-caption difference over the 4 pairs plus a ridge of 1 (the E1 KISSME analogue: unit-variance
coordinates plus I), m_l = 1/(v_S,l + 1) − 1/(v_C,l + 1), score −Σ_l m_l (q_l − c_l)².

## 4. The decision table (spec §5), applied in order, first matching row wins

A configuration of §3 **passes** if, on the pooled seed-42 episodes, both paired differences against its own
condition-free control have 95% lower bounds above 0: R@1 (config − control) and condition gain (config − control; the
control's gain is 0, so this is the config's gain).

| Result on the development episodes | Next step (taken only after reporting to the user) |
|---|---|
| **Row 1.** At least one configuration of §3 passes | Fix the passing configuration (if several pass: the one with the largest min(R@1 margin, gain margin) in points; ties to the earlier row of §3's table) and test it on 3 fresh episode seeds by the GO rule of §6 |
| **Row 2.** No configuration passes; D0 reads close to Told | Build N6 (cross-modal classifier heads on the label-free k-means partitions, the aspect picked by D0's decision variant), run the same checks, then the same test |
| **Row 3.** No configuration passes; D0 reads far below Told | Stop method work; move to branch 3, with D0, N1 and N2 reported as analysis results |

## 5. Stop rules of spec §4.2 and §4.3, with numbers (readings)

- **N1 stops on a checkpoint** if (a) its term-only condition gain minus the current agreement term's term-only gain on
  the same checkpoint (paired) has a point estimate ≤ 0, or (b) its term-only either rate is below cosine's either rate
  (point estimates; cosine 25.92 on seed 42). A stop on A3 means N2-{k}-N1 is not computed. A stop does not override
  §4: an N1-nested configuration that passes §4 still passes.
- **N2 readings:** at each k, the share of rankings in which both p_A and p_B are in the top k of the control (this caps
  any gain). "Rarely reaches the top k" means a share below 10%. "The reordering adds no gain" means the gain's lower
  bound is ≤ 0. Both are reported per k; §4 decides.

## 6. GO rule for the fresh test seeds (written now, before any test episode is built)

- **Seeds 45, 47 and 48**, selection rows, 4,096 episodes per aspect pair each, built with E1's runner
  (`src/test/20261030_aspect_baselines/run_baselines.py --episodes-seed <s>`, which also scores cosine and RCA).
- **Fixed configuration:** the checkpoint (by SHA-256), the rule, k for N2, and its cross-fit procedure (λ grid and
  pick rule as in §3, rerun on each test seed's own anchor-parity halves, as E3 did on seed 43).
- **Comparators:** cosine, RCA (the GO bar) and the configuration's own condition-free control, each scored on the same
  test episodes.
- **GO** = on the three seeds pooled (36,864 episodes; clusters = anchor paintings, one cluster per painting across
  seeds), every paired difference config − comparator, for R@1 and for condition gain against each of the three
  comparators, has a 95% lower bound above 0 (5,000 resamples, seed 42). Each seed is also reported on its own,
  descriptively.
