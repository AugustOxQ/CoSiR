# Round 5 (idea 3) independent re-derivation: stage A report

Written 2026-10-07 16:25 (Amsterdam). This is stage A of the rule's §8 phase 1
(`../DECISION_RULE.md`, SHA-256 19e59fc7…735e, asserted at every run, with round 4's and round 3's rules).
Stage A computed no GoEmotions number. It passed no selection caption to the model, fitted no GE head on real data,
and computed nothing for G-T, G-TF or B′_G with a GoEmotions placement. The implementation files (`r5_*.py`,
`run_r5_*.py`, `test_r5_*.py`, `.superpowers/`) and the implementation's results were not opened.

## 1. Result

Every stage-A target matched exactly: 76 of 76 checks. The record is `results/rd5_stageA_fix1.json`
(SHA-256 `01ba2ecb7ba5888afc87b1f017ded9373436660765e340e2fcbd91e8714dc9df`). The run took 142 s on CPU
(16:20 to 16:22) with numpy 2.2.6, scikit-learn 1.6.1, torch 2.11.0+cu130 and Python 3.11.15.

The first run, `results/rd5_stageA.json` (SHA-256 `638a2038…7682`, 16:17 to 16:20), is kept as written. It produced
the same numbers, but one check failed, `r1.bar_v_eq_stored`. The fault was in my check, not in the computation. I
had assumed the stored `bar_v` of `cand_Rc_Rb_expected_A0.npz` was the bar comparator's R@1 per anchor. It is the
per-anchor bar margin, the fused R@1 minus the bar comparator's R@1. For R1 the bar comparator is its counterpart, so
`bar_v = fused__r1 − cf__r1`; I verified that on the stored file. I corrected the check and reran it as `--fix 1`,
following the `_fix<n>` convention. No computed array changed between the two runs.

## 2. What I wrote (all in `rederive/`, none of it committed)

| File | Content |
|---|---|
| `rd5_paths.py` | Paths; the SHA-256s that the rules state (the three rules and 25 inputs or modules, asserted); hashing; the allowed imports, loaded only through `allowed()` and `load_bundle_round1()` |
| `rd5_core.py` | Own code for the grouping stack s_h, the 18 reader features, P, T, m and π, τ from margins, R1's and AFF's gates, z-scoring through `zscore_rows`, the 224-cell family (fused, G_cf counterpart and nested control), integer ρ and γ, σ*, both cross-fits with their tie rules, assembly, per-anchor metrics and `as_int4` |
| `rd5_stats.py` | `point_ci` (via `cluster_bootstrap`, ×100), the bar comparator with ties to the earliest, the development record (bar margin, margin against the counterpart, gain statistic, either change, per-pair bar margins, Δ_k as an integer with its point and interval, B′(A1) beside), the D10 clauses with 1e-12 boundary flags, the carry (band 24 inclusive, ties to G-T, gap-24 and Δ=0 flags) and round 3's §6.1 sensitivity giving SE, 1.96·SE and x = 2.80·SE |
| `rd5_placement.py` | The single placement function of D4 (draw rng 0 over 60,000 rows, check rng 1 over 10,000 rows, `LogisticRegression(C=1, max_iter)` called directly, NaN float32 scatter, `classes_` = 0..K−1 asserted, accuracy); `unit`; `global_labels`; the GE input `X[scorer_train[i]] = affect_probs[i]`, `X[rows] = probs`; the GE head with its fallback (300, then 3,000, then a stop); the swap that builds a new post dict; identity-and-value fingerprints |
| `rd5_bundle.py` | One seed's bundle in round 3's §4 item 1 call sequence, using only the allowed loaders, with B, B′(A0), cosine and RCA checks and B′_Q |
| `rd5_candidates.py` | D5's extension for any placement Q (post_Q is a new dict, the bundle's post is fingerprinted before and after, the image and caption parts are asserted unchanged); G-T and G-TF of D6 and D7; the D8 comparator order; a `Guard` that refuses a `ge` placement until it is released |
| `rd5_goemotions.py` | Stage B only. Checks on the GoEmotions file: its SHA-256 must appear in its record and on a run-log line, its probs-bytes SHA-256 in the record; rows and sample ids are checked against the join and the splits. Then the CPU spot check of §8: positions `sort(rng(6).choice(32413, 1024))`, the full selection join as D2, HF `refs/main` and the seven file hashes, batch 256, max_length 64, max-abs ≤ 1e-4 |
| `rd5_stageA.py` | The stage-A runner (this report's numbers) |
| `rd5_stageB.py` | The stage-B runner, written but not run: AFF re-verified in-process, spot check, GE head, positive check, G-T and G-TF, records, carry, x for a carried candidate; writes `results/rd5_stageB.json` and `rd5_stageB_arrays.npz` and prints their SHA-256s |
| `rd5_test_synthetic.py` | 52 checks, all passing, on synthetic data only. They cover `first_place` and per-anchor metrics; cell numbering; the tie rules (lowest cell, σ* = 0) and a hand-computed ρ_ctrl criterion; exact π ties; the AFF gate and τ′; D10 at its thresholds; the carry cases; the bar-comparator order; x on a hand-computed example. For the placement function they check equality with a direct `LogisticRegression` fit, the NaN scatter, a refused `classes_` and refused non-finite rows. For the GE input and head: the mapping, the refusals, the fallback used and the fallback stop. Then the guard refusal, the CLIP extension equal to the bundle, a GE-like Q changing only the affect slice and columns 0 to 5, the positive check, fingerprint detection of an in-place mutation, and the GoEmotions-file checks on synthetic files |

### Float paths found by matching the stored arrays (the rule names the functions, not the arithmetic)

| Quantity | Path that reproduces the stored values exactly | Evidence |
|---|---|---|
| Agreements | float32 einsum `nsc,nsc->ns` | — |
| S, C | float32 means cast to float64 | round 4's stored `v` equal |
| Δ | float32 S − C, cast | P equal |
| Support and contrast spreads | float64 sample std (ddof 1) of the float32 agreements | P equal (3 other combinations differ by up to 4.5e-7) |
| P | (p₀ + p₁)/2 of `predict_proba(scaler.transform(F))`, F float64 | P equal to `seed42_arrays.npz` |
| T | Σ_h P(h)·s_h accumulated in float64 in A0 order, then cast to float32 | T equal to `cand_Rc_Rb_expected_A0.npz` |

## 3. Imports (rule §8 list only) and their file SHA-256s

| Import | File | SHA-256 | Stated in a rule |
|---|---|---|---|
| `run_gonogo.EvalContext` | `20261101_aspect_factor_gonogo/run_gonogo.py` | 8353dbc118cf619494a8e8e5d83bee4f2034ebbaa52946be0bfce92e67ea63dc | yes, asserted |
| `run_checks.model_inputs` | `20261108_new_method_quick_checks/run_checks.py` | 5dee3bdf44e526dd6b22580bb5302b41e133e76d4d458224011ca3107dbc4610 | yes, asserted |
| `run_n6.load_posteriors` | `20261108_new_method_quick_checks/run_n6.py` | 57046023af8f4352dc90b563def5e5e27ba35e12489bf7f858a6d0580902f4d0 | yes, asserted |
| `run_told_oracle.fit_one_head` (CLIP heads) | `20261111_community_told_oracle/run_told_oracle.py` | b7ae64175f259bf17edf503a57c5befd47edb938bd4d241aa732abf14ffe464d | yes, asserted |
| `rb_build.load_readers("A0", False)` | `20261117_reader_fix_csd/rb_build.py` | 63c7310c890675cc63e49838feb364cdd1c5c68aafdfb0f0f7972128476cba7b | yes, asserted |
| `common.load_bundle` (seed 42 only) | `20261117_reader_fix_csd/common.py` | 99496dcf859ff0b0f746266125c975c5e2f9632e0c2c91d74df17102c65c10a7 | yes, asserted |
| `centered_term`, `crossfit_condition_free`, `uniform_probe_scores` | `src/eval/aspect_quick_checks.py` | fcd116f529082f81e163092921a91909bd358eb3347695a959c68f814dc7d985 | recorded |
| `cluster_bootstrap` | `src/eval/aspect_metrics.py` | ad95485ef887089d74bb685e2cdb1aab7057b87ad62d926f5c252219f4c6c2f1 | recorded |
| `zscore_rows` | `src/model/aspect_rule.py` | f1d83bd9a7eb535a054c3221d565b8b38514f1e85590bbde0fd2be8cd502fe34 | recorded |
| `artelingo_splits` | `src/data/artelingo_splits.py` | f130950355487b8537ba0c0b1e35230a08e2e07d3d0fddd4c3e5aad240886c13 | yes, asserted |
| `load_goemotions`, `goemotions_probabilities` (stage B) | `src/data/affect.py` | d37e306c9b8673068f6d74e74d7eff61f649ebdb53088f0d8ad7f592e17f3d33 | yes, asserted |
| `join_captions`, `ANNOTATIONS_PATH` (stage B) | `src/data/artelingo.py` | 623b7b02eb03b7a81ecf246b120bb7f49e1b929194830faeebcd0d64fa3c89e1 | yes, asserted |

`crossfit_condition_free` also reaches `src/eval/aspect_nested.py` (fadd1fd5…7fa3) transitively. The stored inputs
asserted at each run are the three rules, `seed42_arrays.npz`, `cand_Rc_Rb_expected_A0.npz`, `rc_tau.json`,
`rb_reader_A0.pkl` and `.json`, `per_anchor_seed42.npz`, `episodes_seed42.npz`, `baselines_seed42.json`,
`per_anchor_told_oracle.npz`, `told_oracle.json`, `partitions.npz`, `n6_posteriors.npz`, `A3_seed42.pt`,
`affect_prepare.npz` and `.json`, and `artelingo_train.json`, each against the SHA-256 the rules give.

**Independence note.** I read only the allowed modules' relevant functions and `src/eval` and `src/data`, which are
not on the forbidden list. When I printed round 1's `load_bundle`, the line window also showed a few lines of
neighbouring `common.py` functions: the tail of `_subset_ctx`, `sigma_from_agreements`, `scaled_delta_picks` and
`condition_sets`, R-a pieces that this round does not use. Nothing from them went into my code. I never opened
`rb_eval`, `rb_features` or `rc_core`, any round-2-to-5 module, any `rd3_`/`rd4_` code, or the brainstorm modules.

## 4. Reproduced targets (seed 42, full precision, all exact)

| Target (rule) | My value | Equal |
|---|---|---|
| Bundle: B, B′(A0), cosine and RCA per anchor = `seed42_arrays.npz`; B 18.341064453125; B′(A0) 18.436686197916664; `v` | same | yes |
| Bundle = round 1's `load_bundle()` (B, B′(A0), T_N1u and A0 posteriors as scores; episodes; its B′(A1) per anchor = stored `Bp1__*`) | same | yes |
| Affect head = told_oracle.json arm L (n_classes 41, draw 7be956c0…, 9.81 and 35.72, majority 6.41, uniform 2.4390243902439024) | same | yes |
| Reader: P (both conditions) = `seed42_arrays.npz`; T (4 rankings), margins and picks = `cand_Rc_Rb_expected_A0.npz` (and round 4's stored) | same | yes |
| τ = rc_tau.json (3.8684538364530674e-05, 0.21702129490553143, 0.47973989883399526, 0.7502585816077211) | same, elementwise `==` | yes |
| R1: gates = stored; cells 116, 119 / 58, 123; σ* 0 / 0; fused, cf and bar_v per anchor; bar comparator = counterpart; bar margin 0.4435221354166667 [0.21646171563312194, 0.6735669710776852]; gain statistic 2.667236328125 [2.325087836946873, 3.012361650695922] | same | yes |
| AFF: gates = stored; τ₀ open 9,941 / 3,627; cells 39, 119 / 149, 10; σ* 0 / 0; per-anchor arrays = `aff_*` | same | yes |
| AFF fused R@1 19.136555989583336; counterpart 18.39599609375; bar comparator B′(A0) | same | yes |
| AFF bar margin 0.6998697916666667 [0.4598852740816973, 0.9371680126852968] | same | yes |
| AFF margin against the counterpart 0.7405598958333333 [0.5196896694963071, 0.9598857494832738] | same | yes |
| AFF gain statistic 3.110758463541667 [2.780005709854805, 3.4559584315470384]; either change −1.629638671875 | same | yes |
| AFF per-pair bar margins 0.9765625, 1.45263671875, −0.32958984375 | same | yes |
| AFF minus R1: fused 0.21769205729166666 [0.06425880757348419, 0.3709597330984391]; bar margin 0.25634765625 [0.04280778303598444, 0.46195041633015954] | same | yes |
| AFF minus B′(A1) 0.33162434895833337 [0.048231414333532084, 0.6246158772581268]; B′(A1) 18.804931640625 | same | yes |
| Item 3: own placement function on CLIP caption features = `fit_one_head`'s `txt` posterior (shape, float32, values, NaN pattern); accuracy 35.72; `classes_` 0..40; draw SHA 7be956c0…; `scorer_train[pos] == draw`; n_iter_ 176 | same | yes |
| Item 3, image features: = `fit_one_head`'s `img`; 9.81; n_iter_ 154 | same | yes |
| Item 4: with Q_CLIP, stack_Q = stack, F_Q = F, B′_Q = B′(A0) (scores and per anchor) | equal | yes |
| Item 4: G-T and G-TF reader outputs (P, T, m, π) = AFF's; τ′ = τ elementwise; gates = AFF's at every τ and condition | equal | yes |
| Item 4: both families give cells 39, 119 / 149, 10, σ* 0 / 0 and AFF's per-anchor arrays; records give AFF's numbers with bar comparator B′_Q (it comes first in the tie order; mean 18.436686197916664 = B′(A0)); all D10 clauses true; Δ_k = 0 | equal | yes |
| The bundle's post is unchanged after the extension (identity and value fingerprints) | unchanged | yes |

## 5. What the rule leaves open for a re-derivation (none of it changed a number)

1. **`bar_v`.** Round 3's §5 item 2 lists it among the arrays to equal but does not define it. It is the per-anchor
   fused R@1 minus the bar comparator's (see §1).
2. **Float paths.** The rule defines the features, P and T by naming round 1's functions (`rb_eval.seed42_features`,
   `common.expected_term`), which a re-derivation may not read. The exact arithmetic had to be found by matching the
   stored P, margins, v and T (table in §2). Three of the four feature variants I tried differed in P by about 1e-7.
   Exact P equality is therefore a sharp check.
3. **Item 3's accuracies.** told_oracle.json stores them rounded to 2 decimals. I compared rounded values against the
   stated 35.72 and 9.81, and raw values against `fit_one_head`'s raw values, which were exactly equal.
4. **"AFF minus R1 (bar margin)".** I read it as the per-anchor difference of the two bar margins, each against its
   own seed-42 bar comparator (B′(A0) for AFF, the counterpart for R1). It matched.
5. **The GoEmotions file's SHA-256 "from the record and the run-log line".** The key names are not specified, so the
   check asserts that the hex string appears in the record's text and on a run-log line. The SHA-256 of the probs
   bytes must also appear in the record.
6. **x units.** Round 3's §6.1 formula is applied to per-episode differences in fraction units. SE, 1.96·SE and x are
   reported in points (×100) beside the seed-42 bootstrap half-width.

## 6. What stage B needs and will do

- **Inputs:** `cache/r5_goemotions_selection.npz` and its record `.json` (the npz's SHA-256 and the probs-bytes SHA-256
  in the record), and the npz's SHA-256 on a line of `20261123_idea3_goemotions_log.md`. The runner reads nothing else
  of the implementation.
- **Run:** `rd5_stageB.py`, about 3 to 5 min of CPU plus the GoEmotions pass over 1,024 captions. In order:
  1. re-verify AFF in-process;
  2. check the GoEmotions file (rows, sample ids, splits) and run the CPU spot check (max-abs ≤ 1e-4); only then is the
     guard released;
  3. fit my own GE head on `affect_prepare.npz` and the file, with its fallback, recording the accuracy, n_iter_ and
     majority share;
  4. build Q_GE, stack_G, F_G and B′_G, and run D5's positive check;
  5. for G-T and G-TF: τ′, gates, open counts, families, cells, σ*, every development number of item 5, D10 and Δ_k;
  6. the carry, then x for a carried candidate;
  7. write `results/rd5_stageB.json` and `rd5_stageB_arrays.npz` and print their SHA-256s.
- **After my files are written and hashed:** comparison with the implementation's `dev_seed42.json`, `carry.json`,
  `seed42_arrays.npz`, `placement.json` and `cache/r5_ge_posterior.npz`, under §8's agreement tolerances. I will write
  that comparison when I am resumed, once the file formats can be seen.

Storage: `results/` holds two JSON files of 15 KB each. There is no `__pycache__` under `rederive/`. The
`__pycache__` folders that appeared at 16:08 to 16:11 in `20261123_idea3_goemotions/`, `20261121_round3_affect_gate/`,
`20261118_reader_fix_round2/` and `20261117_reader_fix_csd/` hold `r5_*`, `test_r5_*`, `r3_*`, `r2_fusion`, `rb_eval`
and similar modules, which my code never imports. They are not mine, and I left them untouched.
