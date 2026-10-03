# Method-repair diagnostics pre-registration (CoSiR v2, method A′, spec §15)

Written and committed on 2026-10-03, before any script of this stage exists or runs, smoke runs included.

- **Binding authority:** spec `docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md` §15
  (revision 3; file SHA-256 `049a0ec6bc119a3512b5fe5488771f3e6f3f36454b18c0d7c734524927c2e61e` at this commit), with
  §6 and §10. The rules below carry the required changes R1 to R12 of the ARS methodology-focus review of the repair
  order (`src/test/20261104_ars_repair_order_review/editorial_decision.md`, Major Revision). Implementation plan:
  `docs/superpowers/plans/2026-10-03-method-repair-diagnostics.md`.
- **Mechanical:** every rule below is evaluated by the scripts of this folder from stored arrays
  (`score_pilot.py`, `score_h3.py`, `decide.py`; the readings are the tested functions of `src/eval/aspect_nested.py`).
- **Frozen:** nothing here changes after a script of this stage has run on real data. A change is a dated addendum at
  the end of this file, committed before the result it affects is computed, with its reason. Decision rules never
  change after a result they govern exists.

## 1. Purpose and scope

The stage decides, before any A′ pre-registration, whether the repair proceeds with A′ = the nested score on A3,
proceeds to the pre-registered H2 grid, or stops (branch 3). It runs two diagnostics in parallel:

- **H1 pilot:** the nested score of spec §15 on E3's existing checkpoints, seed-42 selection episodes only.
- **H3 learnability diagnostic:** the E3 architecture and loss trained on episodes built from the **evaluation labels
  of scorer-train rows** (bank LAB), plus a label-free bank at matched granularity (bank MK).

Out of scope: any A′ pre-registration, any seed-45 episode, any H2 grid run (the grid is pre-registered in §9 so that
no LAB result can choose its settings).

## 2. Data, rows and seeds

- **Backbone:** frozen CLIP ViT-B/32 features from `src.data.artelingo.load_artelingo()`.
- **Training rows:** the 183,694 scorer-train rows of `artelingo_splits()`, as local rows 0..n−1 in that order; painting
  groups = `np.unique(groups[scorer_train], return_inverse=True)[1]`, asserted equal to E2's `partitions.npz`.
- **Development rows:** the 32,413 selection rows (6,451 paintings), read only through E3's
  `run_gonogo.EvalContext(42)`: E1's `episodes_seed42.npz` (SHA-256 checked against `baselines_seed42.json`), 4,096
  episodes per aspect pair, pairs pooled in the order emotion×style, emotion×genre, style×genre; every feature and code
  array NaN outside the selection rows (asserted). Val and held rows are never read.
- **Episode seeds:** only seed 42 is scored on selection rows. Seeds 43 (spent), 44 (MLLM probe) and 45 (reserved for
  the single A′ GO test) are not built, loaded or scored by this stage. Any 8B MLLM probe uses seed 46 or later. The
  ledger is `docs/superpowers/episode_seed_ledger.md`.
- **New randomness:** LAB bank seed 1042; MK bank seed 2042; fresh label episodes seed 3042 + pair index (0, 1, 2);
  MK k-means `random_state` / `seed` 42; model seed 42 (43 only under the rule of §7).
- **Clusters** for every bootstrap: the anchor's painting group (local groups on scorer-train rows, `groups[anchor]`
  on selection rows).

## 3. Banks

- **LAB (H3 only).** Partitions = the evaluation labels of scorer-train rows from `artelingo_aspect_labels(data)`:
  `emotion` (catch-all "something else" = −1), `genre` (−1 where unlabelled), `style`. Bank =
  `build_episode_bank(LAB, local_groups, arange(n), n_per_pair=21846, seed=1042, min_paintings=30)`, the same builder,
  rules and third-aspect control as E2's banks, truncated to its first 65,536 episodes as in E2, validated with
  `validate_aspect_episodes` on 1,000 sampled episodes. LAB is a diagnostic: it reads evaluation labels on training
  rows, which spec §4 C2 forbids for the method.
- **MK (label-free).** Partitions at the evaluation aspects' granularity on E2's three feature spaces: `affect8` =
  `MiniBatchKMeans(n_clusters=8, random_state=42, n_init=3, batch_size=4096)` on the cached GoEmotions
  `affect_probs` (`src/test/20261018_affect_factor_learning/cache/affect_prepare.npz`, scorer-train order, as E2's
  affect partition); `image23` = `kmeans_partition(CLIP image features of scorer-train rows, k=23, seed=42)`;
  `caption10` = `kmeans_partition(CLIP caption features, k=10, seed=42)`. Same builder and rules, seed 2042, 65,536
  episodes. Its AMI with each evaluation aspect is computed afterwards as a descriptive diagnostic only.
- Banks, partition files and a build record with every SHA-256 are written once; the builder refuses to overwrite.

## 4. Runs

Base config of every run: E3's C0 recipe, `dataclasses.replace(R3_CONFIG, painting_batches=True, seed, epochs=2000,
agreement_level="pair", lambda_condition=0.0)`, plus `aspect_episodes_per_step=32`, `aspect_beta=0.3`,
`num_factors=32` unless a row says otherwise.

| Run | Bank | Changes | Role |
|---|---|---|---|
| L3 | LAB | `lambda_aspect=3, lambda_swap=1` (A3's settings) | H3 fit and ceiling |
| L5 | LAB | `lambda_aspect=1, lambda_swap=1, aspect_beta=0` (A5's settings) | H3 fit and ceiling |
| LT | LAB | L3 + `aspect_tau_fixed=True` (the aspect temperature stays at its step-0 value) | H3 fit and ceiling |
| MK3 | MK | `lambda_aspect=3, lambda_swap=1` | matched-k reading (label-free) |

- All four train in one GPU lock, model seed 42. A run that ends with non-finite codes is recorded as failed; a failed
  LAB run counts as *no fit*; a failed MK3 makes the matched-k reading negative.
- An identical rerun is allowed only after an infrastructure failure, never with changed settings.
- **LAB checkpoints (L3, L5, LT and any seed-43 rerun) are never A′ or H2 candidates.** Their SHA-256s are listed in
  `results/label_checkpoints.json`. No setting, λ or pick is chosen from a LAB result.

## 5. Scorers and statistics

- **Term-only agreement rule:** the per-anchor metrics of `agreement_term` itself (rank-equivalent to λ = ∞).
- **Nested score and nested uniform control** (spec §15, `src/eval/aspect_nested.py`): s = z(cos) + λ_u·z(T_u) +
  λ_a·z(T_a) on λ_u ∈ {0, 0.5, 1, 2, 4, 8, 16} × λ_a ∈ {0, 0.25, 0.5, 1, 2, 4, 8, 16}; control z(cos) + σ·z(T_u) over the
  30 distinct σ = λ_u + λ_a. Cross-fitting by parity `np.arange(n) % 2`: on each tuning half the control picks σ by
  R@1, then the nested score picks the cell maximising min(R@1 − that control's R@1 on the same half, condition gain);
  ties go to the first cell in row-major order (λ_u outer, λ_a inner, ascending) and to the smallest σ; each half's
  picks score the other half. A term with weight 0 is left out; a non-finite row in a used input is a miss.
- **Metrics:** `src.eval.aspect_metrics.per_anchor` (R@1, condition gain, other-aspect rate, swap; ties and non-finite
  rows are misses). Either rate = R@1 + other-aspect rate.
- **Uncertainty:** `compare` / `summarize`, painting-clustered bootstrap, 5,000 resamples, seed 42, 95% percentile
  intervals, points in percentage points. **SE** of a paired difference := (upper − lower) / (2 × 1.959964) of its 95%
  interval.

## 6. H1 pilot rule (seed-42 selection episodes)

- **Primary model: A3**, checkpoint SHA-256 equal to E3's `results/picked.json`
  (`dadfef1bed95bbd647c39fa0ea483edea78627d440a23b5a032083e76dd469e2`).
- m_R = R@1(nested) − R@1(control) and m_g = gain(nested) − gain(control), both paired on the pooled seed-42
  episodes; SE_R and SE_g as in §5.
- **Reading:** *not promising* if m_R ≤ 0 or m_g ≤ 0; *promising* if m_R ≥ 2.80·SE_R and m_g ≥ 2.80·SE_g;
  *inconclusive* otherwise.
- **g\* = max(5.6·SE_R, 2.8·SE_g)** from this pilot, fixed for §7 before any H3 transfer number is read.
- **Descriptive rows (decide nothing):** the same nested score and control for A1, A2, A4, A5, A6, C0 and SE (E1's
  cached codes); A3 minus C0 under the nested score; nested and control against backbone-only cosine and RCA (E1's
  seed-42 per-anchor arrays); A3's uncross-fitted 56-cell profile (R@1, gain, either) and the control's R@1 per σ;
  predicted seed-45 power per metric, Φ(m / (2·SE) − 1.96), and their product.

## 7. H3 rules

- **Fresh label episodes (in-distribution):** `build_aspect_episodes` on the LAB labels over scorer-train rows,
  pairs (emotion, style; third genre), (emotion, genre; third style), (style, genre; third emotion), seed 3042 + pair
  index, 4,096 per pair, `min_paintings=30`, each validated with `validate_aspect_episodes`.
- **Fit of a LAB run X:** r = compare(term-only X, term-only A3, painting clusters, gain) on those episodes (codes of
  scorer-train rows). *Fits* if r's lower bound > 0; *no fit* if r's point ≤ 0; *inconclusive* otherwise. An
  inconclusive X is retrained once with identical settings at model seed 43 and decided on the per-anchor mean of its
  seed-42 and seed-43 arrays against A3: *fits* if the lower bound > 0, else *no fit*.
- **H3 no fit:** none of L3, L5, LT fits.
- **Ceiling** (when at least one LAB run fits): on the seed-42 selection episodes, each fitting run's cross-fitted
  nested score (§5) against its own nested control; the *best fitting run* is the one with the largest nested gain
  point. **Ceiling sufficient** if that gain point ≥ g\* (§6); else **ceiling too low**.
- **Matched-k reading:** compare(term-only MK3, term-only A3, painting clusters, gain) on the seed-42 selection
  episodes; *granularity lever* if its lower bound > 0. It decides only which bank the H2 grid uses (§9).
- **Descriptive (decide nothing):** each LAB run minus C0 on the fresh label episodes; the aspect loss over the last
  10 logged steps against its constant-score value (3.258 with the swap term, 2.565 without); τ at the first and last
  log; seed-42 term-only gain and either rate of every run.

## 8. Joint decision table

| H1 pilot on A3 | H3 | Decision |
|---|---|---|
| promising | any | **Pre-register A′ = the nested score on A3** (§5 cross-fitting on seed 42; the GO test once on seed 45). H4 (an H2 model in place of A3) only if H3's ceiling is sufficient, an H2 model passes its gate and beats A3's pilot margins, and both happen by Oct 9, before the A′ pre-registration commit. |
| inconclusive or not promising | ceiling sufficient | **Run the H2 grid (§9)** behind its gate; re-pilot each gate-passer with the §6 rule; pre-register A′ by Oct 9 on the promising model with the largest min(m_R/SE_R, m_g/SE_g); if none is promising, branch 3. |
| inconclusive or not promising | no fit, or ceiling too low | **Stop the repair: branch 3.** |
| any | any | Seed 45 is scored once, for one pre-registered A′, by Oct 12 at the latest; a failure ends the repair. |

## 9. H2 grid (pre-registered now; run only if §8 says so)

- **Bank B:** MK if the matched-k reading (§7) is a granularity lever, otherwise E2's AIC.
- **Cells** (base: A3's settings on B, model seed 42, 2,000 steps unless stated): G1 fixed τ; G2 β 0; G3 fixed τ + β 0;
  G4 128 episodes per step; G5 6,000 steps; G6 fixed τ + 128 episodes per step; G7 learning rate 3e-4 + 6,000 steps;
  G8 λ_aspect 1 + fixed τ + β 0. When B = MK, MK3 counts as cell G0.
- **Gate:** compare(term-only cell, term-only A3, painting clusters, gain) on fresh pseudo-aspect episodes of B's
  partitions over scorer-train rows (seed 4042 + pair index, 4,096 per pair, third partition controlled); a cell
  passes if the lower bound > 0. Only gate-passers are re-piloted.

## 10. Provenance and outputs

- Every result JSON records the SHA-256 of the episode sets, banks, partition files and checkpoints it read, and the
  script's own SHA-256. `score_h3.py` asserts that A3's SHA-256 equals the pilot's.
- Results stay local (gitignored): `results/build_record.json`, `bank_{LAB,MK}.npz`, `partitions_{LAB,MK}.npz`,
  `history_*.json`, `label_checkpoints.json`, `pilot_seed42.json`, `per_anchor_pilot_seed42.npz`, `h3.json`,
  `per_anchor_h3.npz`, `decision.json`. `decide.py` refuses to overwrite `decision.json`.

## 11. Disclosures

- H1 (the nested score) came from E3's post-hoc fixed-λ profile, which included the spent seed-43 test draw.
- Seed 42 has been scored by E1, E3's pick and E3's post-hoc profile, and is scored again here (H1 pilot, H3
  transfer). Only the seed-45 test controls the resulting selection, and only for the final claim.
- Seed 45 draws its episodes from the same 6,451 selection paintings as seeds 42 and 43: an A′ GO would be a
  fresh-episode result, not a fresh-painting one.
- H3 reads evaluation labels on training rows. It bounds selection among trained aspects and is never the method.
- One model seed per run (seed 43 only under §7), one development draw.

## Addenda

(none)
