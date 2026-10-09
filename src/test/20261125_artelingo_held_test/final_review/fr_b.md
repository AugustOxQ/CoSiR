# Final review B (scoring path), round 6, snapshot b97b411

> Reviewer B (Opus), 2026-10-09 17:53 to 18:30. Worktree `/project/CoSiR-r6-fr` (detached at b97b411). Area: tickets
> 01, 02, 03, 05, 06, 07; rule §2, §5.1 to §5.4, §6.1 to §6.5, §10.1, §11; spec SC1, SC2, D3, D4, D5, D8 (split),
> D9, D14, D17, D19, D20, D23. Scripts: `fr_b_inputs.py`, `fr_b_regression_capture.py`, `fr_b_r3_independent.py`,
> `fr_b_check.py`, `fr_b_guards.py`, `fr_b_mutants.sh` (this folder). Outputs: the temp folder
> `/tmp/claude-0/-project-CoSiR/714507bd-709f-400b-92b9-40bd3c709f3b/scratchpad/fr/b/` (`check.json`, `r3.json`,
> `inputs.json`, `guards.json`, logs).

## 1. Verdict

**CONFIRMED WITH FIXES.** Every number my area fixes was reproduced exactly by my own code: the input SHA-256s, the held
split, the value sets, the twelve episode hashes, the eight stored posteriors (bit for bit), B, B0 and B1, all 131
regression items and the σ split. The guards I fired through the real entry points all fired, and every mutant was
caught. One should-fix: rule §5.2's `results/value_sets.json` is written by no runner. The other findings are nits.

## 2. What I re-derived

| Check | How (my code) | Mine vs the code's | Result |
|---|---|---|---|
| §6.1 input SHA-256s | parsed the SHA tables of the r6 rule and the R3 rule (D15 table, told_oracle.json in D15's text, §6.1's three files, §5.1's prepare.npz, §6.4's seed42_arrays.npz); hashed the files | 36 rule-named inputs plus the rule itself (7444a5e3…) and the R3 rule (2d311dbe…): `INPUT_SHA256` holds each with the rule's value, the files match, and `INPUT_PATHS` point at them. Another 13 assertions go beyond the rule (round 3's D1/D2/D10 inputs, held_codes, AB smoke jsons, r3 modules); all match the disk | equal |
| §5.1 held split | `artelingo_splits` and `grouped_split` recomputed; disjointness of rows, leakage groups and painting names | held 61,744 (19.9998%), train 216,107, val 30,872, scorer-train 183,694, selection 32,413; held disjoint from train, val, scorer-train and selection in rows, leakage groups and paintings; equals held_codes `held_rows` and prepare.npz's 4 arrays; 44,385 held rows with all three labels | equal |
| §5.2 development values | own count of distinct paintings ≥ 30 per value on the selection pool | 8 / 23 / 10, equal to `development_value_sets`; the four styles outside are Action_painting, Analytical_Cubism, New_Realism, Synthetic_Cubism | equal |
| §5.3 / §6.4 episodes | ORIGINAL `build_aspect_episodes` on selection rows, after checking per pair that V equals the pool's eligible lists (so the restriction is a no-op) | 12 / 12 per-pair hashes (seeds 42, 9001, 9002, 9003) equal AB's records and the runner's items; the seed-42 episode arrays equal the r6 bundle's | equal |
| §6.2 refits | the posteriors of the heads the regression process fitted, predicted on selection rows and compared bit for bit (uint32 view, dtype, shape) | 8 / 8 stored arrays equal; D1 affect equals round 3's own `fit_one_head` refit; E2 image and caption equal round 3's bundle; the coefficient SHA-256s of the regression process equal `refit_check.json`'s | equal |
| §6.3 B, B0, B1 | round 3's `r3_bundle.build_bundle(42)` (EvalContext, stored n6 posteriors); B1 = `crossfit_condition_free` over A1 with step 1's STORED csd posteriors; no r6 module imported | 18.341064453125, 18.436686197916664, 18.804931640625 exactly; picks equal `picks_seed42.json`; r6's frozen B, B0 and B1 per-anchor arrays bit-equal to round 3's cross-fit (5 metrics each) | equal |
| bundle on selection | the same round 3 bundle | cos, T_N1u, stack, F, cl, pair index and anchor bit-equal to the r6 bundle (21 / 21) | equal |
| §6.4 arrays | the per-anchor arrays `score_seed` returned in the regression run (captured), compared with seed42_arrays.npz (SHA 5ea4b09a… asserted first) and per_anchor_seed42.npz | 110 / 110 equal (aff/r1 fused and cf × 5 metrics, 4 gates, cl, pair index, parity, reader pick, margin and P; cosine, rca and 9 PM × 5 metrics, anchor_group, pair index); also equal to round 3's `score_frozen`; the runner's flags agree with mine on every item | equal |
| §6.4 scalars | means and `cluster_bootstrap` intervals (fractions, ×100 after the percentile) from the captured arrays; the bar comparator by my own max-with-ties rule | all 11 targets equal exactly (max abs diff 0): AFF 19.136555989583336, CF 18.39599609375, bar margin, gain statistic, AFF − R1, R1 margin and gain, AFF − B1, comparators "B_prime" and "counterpart", 9,406 quarter-hits; equal to the runner's `got` | equal |
| §6.5 σ split | own one-way decomposition on my own per-episode differences | σ_a², σ_ε² of P1 to P7, S1, S2 equal `sensitivity_seed42.json` (relative diff 0.0); my differences equal `seed42_per_episode.npz` bit for bit; N 12,288, 4,602 paintings | equal |
| seed-42 counts file | `cluster_bootstrap` of the per-episode differences | points and 95% intervals of all 9 checks equal `seed42_pass_counts.json` | equal |

Runs on the snapshot (they write to the snapshot's gitignored `F/results/`, 1.2 MB): `run_r6_refit.py` (exit 0, 8/8
and affect identity pass), `run_r6_picks.py` (exit 0, 3 means, 3 frozen-equals-crossfit, 10 lambda conventions pass),
`run_r6_held.py --mode regression` through its own `main()` (exit 0, 131 / 131 items), `run_r6_sensitivity.py`
(exit 0). Suites: test_r6_common (34, worktree tests deselected because their fixture runs `git worktree remove`),
test_r6_episodes (33), test_r6_heads (31), test_r6_context (28), test_r6_score (36), test_r6_regression (39): all
pass.

## 3. Findings

1. **should-fix** `r6_common.py:401` (`write_value_sets`) and `run_r6_held.py:294`: no runner writes
   `results/value_sets.json`. Rule §5.2 says the development value sets "are written, as label codes with their names,
   to `results/value_sets.json` before the read". The function exists and is tested, but nothing calls it
   (ticket 01 built the function; no ticket wired it into a runner). They are not recorded in any other record
   either. Failure in the run: the read starts without a rule-required pre-read record, so the report cannot cite
   the value sets that restricted the held episodes. Fix: in regression mode, after `load_rows()`, call
   `R.write_value_sets(results / "value_sets.json", env.value_sets, env.data)` and put its SHA-256 in
   `regression_seed42.json`. In held mode, before `held_started.json`, assert that the file exists and equals the
   recomputed sets (codes) and record its SHA-256 in the attempt. The regression then needs a rerun (rule §6.7).
2. **nit** `run_r6_sensitivity.py`: it asserts no input SHA-256 (rule §6.1 "Scripts assert …"). It reads only the
   regression's per-episode file, whose SHA-256 the regression record binds, and it copies the regression's
   `input_sha256`. Leave, or call `R.assert_input(R.RULE_REL)`.
3. **nit** `r6_common.py:265`: the code the scoring path imports outside the r6 folder and round 3 (`src/eval/*`,
   `src/data/*`, QC `run_n6.py` / `run_checks.py`, TO `run_told_oracle.py`, GG `run_gonogo.py`) is not hashed, and the
   smoke-record SHA guard covers only r6 files. An edit there between the regression and the read would go unseen.
   The rule does not require it. Suggest recording `git rev-parse HEAD` and `git status --porcelain -- src` in
   `held_started.json`.
4. **nit** Rule §5.1 defines held as the 20% part "with all three labels known" (44,385 rows). The code's
   `split.held` is all 61,744 rows: features, A3 codes and posteriors are finite on all of them, and the pool filter
   drops the 17,359 rows without all three labels. No number changes, since only pool rows enter episodes and every
   statistic is per episode. This is the same pattern as development's EvalContext.
5. **nit** Rule §6.1's "before reuse" items: no round 4 or round 5 code is copied (verified: no `r4_`/`r5_` import or
   copy; B′(A1) is written directly). The log's row of 05:10 says so, so the rule's condition is vacuous. The log
   could also name the lessons the r6 code applies: R3 N11 `cl == groups[anchor]` (`r6_bundle.check_context`),
   R4 T3a-1 inputs on every attempt (`run_read` step 3), R4 T3a-2 mandatory order guard (`order_guard`).

## 4. Guards (my area)

- §6.1 input SHA: caught. Fired through `run_r6_refit.main()` with one expected SHA changed in memory (SystemExit
  before any data, no output written); suite mutation `input_sha`.
- §5.1 split asserts (prepared, held_rows, held_disjoint, leakage, held_selection, held_scorer, subsplit): caught by
  the suite's split-scenario mutations. My independent split check agrees on real data.
- §5.2 value counts 8/23/10 and held eligibility ≥ 30: caught by the suite (29 fails, 30 passes).
- §5.3 restriction V ⊆ ok and the restriction itself: caught. My mutant (restriction lines removed, on a copy) fails
  `test_value_outside_the_set…`.
- §5.3 hash distinctness: caught by the suite (`distinct_new`, `distinct_recorded`, recorded file SHA).
- §5.5 members in rows, NaN outside and finite on the rows (features, codes, posteriors): caught by the suite
  (context tests on synthetic held data of the real shapes).
- §6.2 bit-for-bit refit: caught. My mutant (`np.isclose` in `compare_bits`) fails the one-ulp test; real run passes.
- §6.3 exact means: caught. `load_picks` refuses a copy with B's mean moved by one ulp.
- §6.4 order guard and exactness: caught. `run_regression(results=<empty>)` gives exit 4 before any input; a stale
  `r6_heads.py` SHA in `refit_check.json` is refused; the suite catches loose comparisons.
- §6.5 sensitivity order guard: caught (`run_r6_sensitivity.run(results=<empty>)` gives exit 4).
- §5.4 parity mapping (h scores 1 − h): caught. My mutant (`parity == h`) fails the parity-mapping tests.
- CF condition-free per cell and gain 0: caught by the suite (`cf_cell`, `cf_gain`).

## 5. Hard rule 6

| ID / section | Status | Where |
|---|---|---|
| SC1 | covered | regression 131 items; my comparison above |
| SC2 | covered | `r6_heads.check_selection`, `run_r6_refit.py` |
| D3 | covered | `r6_score.CELLS` asserted against `R3.AFF_CELLS`/`RC_CELLS`; readers, τ and A3 from round 3 (arrays exact) |
| D4 | covered | no pick in `r6_score`; picks from `picks_seed42.json`, λ from `baselines_seed42.json` |
| D5 | covered | `r6_score._apply_masks`; `frozen_equals_crossfit`; mutant caught |
| D8 (split) | covered | `r6_common.load_split`; seeds 52 to 54 and 4,096 per pair in `r6_context.ADMITTED` |
| D9 | covered | B1 = `uniform_probe_scores` over A1 plus the nested assembly (`r6_bundle`, `r6_score`); S1 in stats |
| D14 | covered | R1 fused scored on held (`CORE_SCORERS`); its counterpart only with `include_pm` |
| D17 | covered | `build_aspect_episodes_r6` restriction; New_Realism outside the development set (verified) |
| D19 | covered | csd heads refit (`r6_heads`); no round 4 or 5 code copied |
| D20 | covered | bit-for-bit `compare_bits`; exit 3 stop |
| D23 | covered (my part) | PM fits rerun and checked on seed 42; the held PM rows are reviewer C's |
| §2 | covered | scorer definitions in `r6_bundle`, `r6_score` |
| §5.1 | covered | `load_split` (nit 4) |
| §5.2 | **partial** | sets computed and asserted; `value_sets.json` never written (finding 1) |
| §5.3, §5.4 | covered | `r6_episodes`, `r6_score`, `r6_picks` |
| §6.1 | covered | `assert_inputs` in refit, picks (bundle), regression and held runs (nit 2) |
| §6.2 to §6.5 | covered | re-derived above |
| §10.1 | covered | folder, prefixes, `results/` gitignored |
| §11 | covered | frozen items; reruns: heads, PCA/scaler (`fit_rows_sha256` asserted), B/B0/B1 picks |
| unrequested | defensible | extra input SHAs (held_codes, AB smoke jsons, r3 modules); extra regression items beside the targets (R1 R@1s, quarter-hits) |

Contract amendments of my sections (§1 import order, §4 `on_episodes` and bundle fields, §5 `include_pm`, §7
`sensitivity_seed42.json` keys) are applied in every r6 module, runner and test that writes or reads them.

## 6. Deferred nits

- T02, test loops over the module's `R3_BUILD_SEEDS`: leave. `recorded_hashes` has its own completeness guard;
  pinning (49, 50, 51) is a one-line test change if wanted.
- T02, `members()` has no bound check: leave. Ids come from the builder on pool rows, `in_rows[...]` raises on
  ids ≥ N, and `load_episodes` re-hashes.
- T03, the threads guard reads only the environment: leave. The coefficient SHA-256 equality between stage 1a and the
  read (`head_guard`) catches any pool difference that changes a fit, before `held_started.json`.
- T03, extra keys in `refit_check.json`: leave (additive).
- T03 / CLAUDE.md, MKL=8 missing from the test command: fix at the merge (doc only).
- T05, `cl_groups` compares two views: leave. `load_split` anchors the groups to prepare.npz, and cl equals round 3's
  arrays.
- T06, no order guard on `refit_check.json` in the picks runner: leave. The regression's order guard and its
  `coef_sha256_equal_picks_seed42` item cover it.
- T06, a failed rerun leaves the old passed picks: leave. Changed code makes the old record stale (refused); with
  unchanged code the exit 3 is visible to the run chat.
- T06, `held_verdict.json` existence only: reviewer C's area.
- T06, array_equal without dtype: leave. The regression's `array_item` compares dtype.
- T07: none.
