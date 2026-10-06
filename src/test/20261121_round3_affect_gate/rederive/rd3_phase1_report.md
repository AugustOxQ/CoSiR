# Reader fix round 3: independent re-derivation, phase 1 (seed 42)

Written 2026-10-06 20:03 (Amsterdam) by the independent re-derivation agent, under `../DECISION_RULE.md` (SHA-256
2d311dbeac1a561fc2a719b730075c30819f4f7f3a592041cb64b65821dc5925, asserted by every script). The round-3
implementation (`r3_common.py`, `r3_bundle.py`, `r3_fusion.py`, `r3_stats.py`, `run_r3_*.py`, `r3_apply_rule.py`,
`test_r3_*.py`) was never opened or imported. Every target value below is copied from the rule's text.

**Overall: agreement on every item.** Every value of §5 items 1 to 4 and of D7 agrees under rule §8's tolerances, and in
fact equals the rule's value bit for bit (absolute difference 0.0 everywhere, every per-anchor array identical, every
discrete quantity identical). No disagreement needed tracing.

## 1. What was computed, and with what

| Script (`rederive/`) | What | Runtime |
|---|---|---|
| `rd3_core.py` | own code: SHA-256 checks, the 18 features, half-reader probabilities and their mean, picks, margins, τ percentiles, the weighted term T, grouping scores s_h, float32 combine, integer rank counts and per-anchor metrics (R@1 strict, ties miss), bar comparator, evaluation (bar margin, margin, gain statistic, either, per pair, D13 clauses), D7 redundancy, §6.1 one-way decomposition and projection | |
| `rd3_bundle.py --seed 42` | the bundle in the rule's §4 item 1 call sequence, from the allowed loaders only | 34 s |
| `rd3_family.py` | R1 and AFF gates (D6), z before the gate, gated terms, G_cf (D9), the 224 cells (D8), integer ρ and γ, σ*, the min-margin and max-R@1 cross-fits, the assembly, and a literal re-scoring check of the assembled scores | |
| `rd3_phase1.py` | §5 items 1 to 4, D7, §6.1 | 10 s |
| `rd3_selftest.py` | synthetic checks: the rule's named cells (116, 119, 58, 123, 39, 149, 10) decode to the rule's (τ index, λ_u, λ_a) and round-trip; own integer counts equal `per_anchor` on rows with ties and NaN; the one-way decomposition equals the textbook balanced-design formulas; R1 and AFF gate definitions | passed |

Imports from earlier code, all on rule §8's list: `run_gonogo.EvalContext`, `run_checks.model_inputs` (imported by
path from `20261108_new_method_quick_checks/`, module file asserted) with `centered_term`, `run_n6.load_posteriors`,
`run_told_oracle.fit_one_head` and `global_labels`, `rb_build.load_readers("A0", False)`, `zscore_rows`,
`crossfit_condition_free`, `uniform_probe_scores`, `cluster_bootstrap`; also `artelingo_splits` (the rule's call sequence
uses it). Importing `rb_build` and `run_told_oracle` loads round 1's `common` and the 20261109 diagnostics modules as a
side effect of their own imports; no function of them is called. Round 1's `load_bundle`, `run_sweep.setup`,
`rb_eval`, `rb_features`, `rc_core` and round 2's `r2_fusion` were not imported. `src.eval.aspect_metrics.per_anchor` is
used only in `rd3_selftest.py` on synthetic data.

One substitution, recorded: the rule writes B's T_6u as `run_n6.n6_terms(post_E2, ctx.pooled)[2]`. `n6_terms` is not on
§8's import list; its third output is `uniform_probe_scores(post, ep, run_n6.PARTS)` (read from its source), so we call
`uniform_probe_scores(post_E2, ep, ("affect", "image", "caption"))` and assert `run_n6.PARTS` equals that tuple. B then
equals the stored references exactly (below).

Run conditions: CPU only, `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1`,
scikit-learn 1.6.1, numpy 2.2.6, one process at a time. Seeds 49 to 51 and 9001 to 9003 are refused by `guard_seed`
unless `out/PHASE2_AUTHORISED` exists (it does not). Every D15 input read is SHA-256 asserted.

Outputs: `out/phase1.json` (every quantity at full precision; `summary` holds the compact view),
`out/phase1_arrays.npz` (per-anchor arrays and per-episode gates and picks), `out/rd3_bundle_seed42.{npz,json}` (the
bundle cache, 61 MB), logs.

## 2. §5 item 1: bundle (PASSED)

References (round 1's `load_bundle` was not called): the episodes file, step 1's `step1_eval_style.npz` (`B__*`,
`A0__Bprime__*`, anchor groups, pair index), `per_anchor_seed42.npz`, `n6_posteriors.npz`, `told_oracle.json`, and round
2's independent re-derivation cache `20261118_reader_fix_round2/rederive/out/rd2_seed42_cache.npz`, whose B scores,
features and s_h were taken from or checked bit-equal to round 1's `load_bundle`, `rb_eval.seed42_features` and
`common.grouping_stack` (round 2's phase-1 report §2).

| Piece | Reference | Result |
|---|---|---|
| anchors, candidates (12,288 episodes, pair order emotion×style, emotion×genre, style×genre) | `episodes_seed42.npz` | identical |
| parity | index mod 2; rd2 cache | identical |
| anchor paintings (4,602), pair index | step 1; rd2 cache; `per_anchor_seed42.npz` | identical |
| cosine per anchor (own metrics on `ctx.cos`), all 5 metrics | `per_anchor_seed42.npz` `cosine__*` | identical (§4 item 2 assertions pass) |
| B scores, 4 × (12,288 × 13), float32, condition-free | rd2 cache `B__*` | identical |
| B per anchor, 5 metrics | step 1 `B__*`; rd2 cache | identical; R@1 18.341064453125 = rule D10 |
| B cross-fit picks (λ_u, λ_a) | `told_oracle.json` `B_picks` | identical: (16, 8) on half 0, (8, 2) on half 1 |
| B′(A0) per anchor, 5 metrics | step 1 `A0__Bprime__*`; rd2 cache | identical; R@1 18.436686197916664 = rule D11 |
| B′(A0) cross-fit picks | `told_oracle.json` arm L `B_prime.picks` | identical: (16, 16), (16, 8) (no stored B′ scores exist; picks and per-anchor arrays are the check) |
| image and caption posteriors (both modalities) | `n6_posteriors.npz` | identical; selection rows identical |
| affect heads (41 classes) | `told_oracle.json` arm L `head` | identical record (draw SHA-256 7be956c0…, held-out accuracy 9.81 / 35.72) |
| grouping scores s_h, A0 order, both directions (float32) | rd2 cache `stack__*` (first three groupings) | identical |
| 18 features, both conditions | rd2 cache `F_A0__*` | identical (max abs diff 0.0); Δ^b = −Δ^a exactly |

### D7 redundancy (PASSED)

Pearson over the 13 candidates between z(s_h) and z(B) (zscore_rows float32, then float64), mean over the 12,288 rows
of each direction; no row had a zero denominator.

| Grouping | i2t, rule | i2t, ours | t2i, rule | t2i, ours |
|---|---|---|---|---|
| affect | 0.35348060377541385 | 0.35348060377541385 | 0.3828024789253903 | 0.3828024789253903 |
| image | 0.7145397990123284 | 0.7145397990123284 | 0.7090878258485419 | 0.7090878258485419 |
| caption | 0.6182295729609555 | 0.6182295729609555 | 0.665152773464146 | 0.665152773464146 |

All six equal exactly (difference 0.0). Order affect < caption < image in both directions: affect is the smallest.

## 3. §5 item 2: R1 = round-1 R-c (PASSED)

| Quantity | Rule / stored | Ours | Agree |
|---|---|---|---|
| half-reader C | 1.0, 100.0 | 1.0, 100.0 | yes |
| T^c, `T__{a,b}__{i2t,t2i}` | stored npz | identical | yes |
| margins `margin__{a,b}`, picks `pick__{a,b}` (int8 vs int64) | stored npz | identical in value | yes |
| τ_0..τ_3 recomputed (`numpy.percentile` of 24,576 margins, a first) | 3.8684538364530674e-05, 0.21702129490553143, 0.47973989883399526, 0.7502585816077211 | identical to `rc_tau.json` and to the rule's text; τ_0 = min margin | yes |
| gates at τ_0..τ_3 | stored `extra__gate_{a,b}` | identical | yes |
| fused cells (integer min-margin) | 116 (τ_2, 0, 2) on half 0, 119 (τ_2, 0, 16) on half 1 | 116, 119; criterion 175 (4 cells tied, lowest wins) and 225 | yes |
| counterpart cells (integer max-ρ) | 58 (τ_1, 0, 0.5), 123 (τ_2, 0.5, 1) | 58, 123; ρ 4,647 (2 tied) and 4,502 (4 tied) | yes |
| σ* | 0, 0 | 0, 0 (ρ_ctrl 4,532 and 4,483) | yes |
| `fused__*`, `cf__*` (5 metrics each), `bar_v` | stored npz | identical; also identical to the literal re-scoring of the assembled scores | yes |
| bar comparator | counterpart | counterpart | yes |
| bar margin | 0.4435221354166667 [0.21646171563312194, 0.6735669710776852] | same, difference 0.0 | yes |
| gain statistic | 2.667236328125 [2.325087836946873, 3.012361650695922] | same, difference 0.0 | yes |
| pick accuracy (D14, diagnostic) | 51.261393229166664 | 51.261393229166664 [50.661759963034505, 51.84087009613146] | yes |

Descriptive, R1 on seed 42: fused R@1 18.918863932291664, counterpart 18.475341796875; either change against the
counterpart −1.7801920572916667; per-pair bar margins 0.7080078125, 1.220703125, −0.59814453125.

## 4. §5 item 3: AFF = the brainstorm's numbers (PASSED)

| Quantity | Rule | Ours | Agree |
|---|---|---|---|
| fused R@1 | 19.136555989583336 | 19.136555989583336 | exact |
| counterpart R@1 | 18.39599609375 | 18.39599609375 | exact |
| bar comparator | B′(A0) | B′(A0) (means: B′ 18.436686197916664, counterpart 18.39599609375, B 18.341064453125) | yes |
| bar margin | 0.6998697916666667 [0.4598852740816973, 0.9371680126852968] | same | exact |
| margin vs counterpart | 0.7405598958333333 [0.5196896694963071, 0.9598857494832738] | same | exact |
| gain statistic | 3.110758463541667 [2.780005709854805, 3.4559584315470384] | same | exact |
| either change vs counterpart | −1.629638671875 | −1.629638671875 [−1.9464780404744315, −1.3095998368766755] | exact |
| per-pair bar margin, emotion × style | 0.9765625 | 0.9765625 [0.5785321458573172, 1.3775886939702795] | exact |
| per-pair bar margin, emotion × genre | 1.45263671875 | 1.45263671875 [1.0342696066046755, 1.8759388816913407] | exact |
| per-pair bar margin, style × genre | −0.32958984375 | −0.32958984375 [−0.7089190499272904, 0.05458614674142936] | exact |
| fused cells | 39 (τ_0, 4, 16) on half 0, 119 (τ_2, 0, 16) on half 1 | 39, 119; criterion 188 and 314, unique maxima | yes |
| counterpart cells | 149 (τ_2, 4, 4), 10 (τ_0, 0.5, 0.5) | 149 (unique), 10 (2 tied, lowest wins) | yes |
| σ* | 0, 0 | 0, 0 | yes |
| AFF − R1, fused R@1 | 0.21769205729166666 [0.06425880757348419, 0.3709597330984391] | same | exact |
| AFF − R1, bar margin | 0.25634765625 [0.04280778303598444, 0.46195041633015954] | same | exact |
| AFF τ_0 gate open, condition a | 9,941 of 12,288 (80.90006510416667%) | 9,941 (80.90006510416667%) | exact (counts) |
| AFF τ_0 gate open, condition b | 3,627 (29.5166015625%) | 3,627 (29.5166015625%) | exact (counts) |

The float32 mean of the condition-a gate is 80.90006709098816%, the brainstorm's printed value, as the rule says.
AFF's gate-open counts at τ_1..τ_3 (a / b): 9,005 / 2,573; 7,326 / 1,368; 4,308 / 336.

## 5. §5 item 4: development bar D13 for AFF (recorded)

| Clause | Value | Holds |
|---|---|---|
| (1) bar margin point ≥ +0.5 | 0.6998697916666667 | yes |
| (2) bar margin lower bound > 0 | 0.4598852740816973 | yes |
| (3) gain statistic lower bound > 0 | 2.780005709854805 | yes |

AFF clears the development bar on seed 42 (selection disclosure of rule §6.11 applies; this selects nothing).

## 6. §6.1 sensitivity projection

AFF's seed-42 per-episode difference (percentage points), one-way decomposition by anchor painting: n = 12,288, P =
4,602 paintings, Σ m_p² = 43,612, n₀ = 2.6699523682578064. SE² = (σ_a²(9Σm_p² − 6n) + σ_ε²·3n)/(3n)²; half-width =
1.96·SE; x = 2.80·SE; the seed-42 bootstrap half-width is half the width of the seed-42 95% painting-bootstrap interval of
the same difference.

| Check | seed-42 point | σ_a² | σ_ε² | SE | 1.96·SE | x = 2.80·SE | seed-42 boot half-width |
|---|---|---|---|---|---|---|---|
| R@1, AFF − cosine | 6.174723307291666 | 8.659210753381196 | 420.1183543738188 | 0.11587793060011237 | 0.22712074397622023 | 0.3244582056803146 | 0.37415231630884227 |
| R@1, AFF − RCA | 5.755615234375 | 8.25918251129639 | 409.1397078992219 | 0.11417548937089449 | 0.22378395916695318 | 0.3196913702385045 | 0.3719848422912784 |
| R@1, AFF − B | 0.7954915364583334 | 1.4195629723547245 | 139.6118270264445 | 0.0641888663865091 | 0.12581017811755782 | 0.17972882588222547 | 0.21238499562483287 |
| R@1, AFF − B′(A0) | 0.6998697916666667 | 0 | 175.6986792481523 | 0.06903717626782965 | 0.13531286548494612 | 0.193304093549923 | 0.23864136930179974 |
| R@1, AFF − counterpart | 0.7405598958333333 | 0 | 154.3796488968718 | 0.06471333708403104 | 0.12683814068470084 | 0.1811973438352869 | 0.22009803999348332 |
| gain statistic | 3.110758463541667 | 3.0884416159446615 | 340.70777639034856 | 0.09983370431525766 | 0.195674060457905 | 0.2795343720827214 | 0.3379763608461166 |
| gain, AFF − RCA | 3.006998697916667 | 1.197340532435266 | 445.6281213636307 | 0.11121737935241137 | 0.21798606353072628 | 0.31140866218675184 | 0.38314019052271875 |
| secondary: R@1, AFF − R1 | 0.21769205729166666 | 0.5386102496918705 | 75.66339567095812 | 0.04667811858582208 | 0.09148911242821128 | 0.13069873204030183 | 0.15335046276247746 |

σ_a² is truncated at 0 for the B′(A0) and counterpart differences (between-painting mean square below the within). The
projected three-seed half-widths are about 0.58 times the seed-42 half-widths, close to 1/√3 as expected. For the GO
checks against the condition-free comparators, x runs from 0.180 (B) to 0.193 (B′(A0)) R@1 points, below the prior's expected
pooled bar margin of about +0.3; for the secondary check x is 0.13 against a seed-42 point of 0.22.

## 7. Disagreements

None. Nothing needed tracing.

## 8. For phase 2

`rd3_bundle.py --seed <s>` builds any seed's bundle in the rule's call sequence; `rd3_family.py` and
`rd3_core.evaluate` are seed-agnostic. Phase 2 needs the controller to create `out/PHASE2_AUTHORISED` (the seed guard),
the pooled-over-seeds evaluation (concatenation 49, 50, 51; clusters = anchor paintings; bar comparator chosen once over
the 36,864 pooled episodes), the §6.2 hash check, and the seven GO checks plus the secondary check.
