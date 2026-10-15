# Reader fix, round 3 (one-sided affect steering on R1, fresh-seed test): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** test AFF (R1 with its gate opened only on affect picks) on the fresh episode seeds 49, 50 and 51, with R1
beside it, under the committed rule: seed-42 regression checks first, then the sensitivity projection, the seeds, the
GO pass, an independent re-derivation, the verdict, the descriptive pass.

**Architecture:** new modules in `src/test/20261121_round3_affect_gate/`, all named `r3_*.py`, `run_r3_*.py` or
`test_r3_*.py`, importing round 1's code (`src/test/20261117_reader_fix_csd/`) and round 2's `r2_fusion.py` by path and
changing nothing there. One seed-parameterised bundle builder serves seed 42 and the test seeds; on seed 42 it must
reproduce round 1's `common.load_bundle()` exactly. A fusion module turns a bundle into R1's and AFF's per-anchor
arrays on the 224 cells. The main session launches every real run on CPU.

**Tech stack:** numpy 2.2.6, scikit-learn 1.6.1, torch (CPU), the CoSiR env (`/root/miniconda3/envs/CoSiR/bin/python`).

**Spec:** `docs/superpowers/specs/2026-10-06-round3-affect-gate-design.md`; **binding rule:**
`src/test/20261121_round3_affect_gate/DECISION_RULE.md` (committed before any code; the rule governs; section numbers
below are the rule's).

## Global constraints

- CPU only: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1`; at most 3 processes.
- Every script asserts the rule's SHA-256 and the SHA-256 of every D15 input it reads; a mismatch stops it.
- Non-smoke results are never overwritten; smoke outputs go to `results/smoke/`. Smoke runs never print or log a metric
  value of any scorer on any seed (only shapes, counts, file names, assertion pass or fail); their files are opened only
  by assertions and deleted when the smoke test has passed (rule §10).
- No smoke or test run touches seeds 49 to 51 except the main session's real runs of Task 5. Wiring smoke tests use the
  smoke seeds 9001, 9002 and 9003 (`run_baselines.py --smoke --episodes-seed <s>`, 64 episodes per pair).
- Round 1's and round 2's folders are read only; their writing functions are not called (rule §10).
- No module of this folder may share a name with a round-1 or round-2 module (`common`, `rc_core`, `rb_*`, `r2_*`,
  `run_*` of those rounds): their folders are on `sys.path`.
- Cross-fit ties compare the integer sums ρ and γ (round 2's `r2_fusion`), never float means.
- Held rows are not read.

## Review focus

1. The bundle builder must not depend on seed 42: no step-1 array, no seed-42 path, no `run_sweep.setup` assertion
   inside it. A test runs it on a smoke seed and checks that nothing seed-42-specific is read; on seed 42 a separate
   layer compares it with `common.load_bundle()` exactly.
2. The AFF gate: closed for every non-affect pick, open on an arg-max tie that includes affect (affect is index 0), and
   AFF's counterpart built from AFF's own gates (not R1's). Tests on synthetic probabilities; a mutation that passes R1's
   gates to AFF's counterpart must fail a test.
3. Alignment of external baselines: `per_anchor_seed{s}.npz` rows must match the bundle's episodes (anchor painting,
   pair index, cosine per anchor equal); a test with a shuffled row order must fail.
4. Order of computation on the test seeds (§6.4): the GO phase writes no per-seed summary, no bar margin and nothing
   of R1's counterpart; the descriptive phase refuses to run without `results/test_verdict.json` carrying this rule's
   SHA-256.
5. Pooled statistics: concatenation in seed order 49, 50, 51; clusters = anchor paintings shared across seeds; the
   strict inequality "lower bound above 0"; the sensitivity formula on a synthetic case with known variance
   components.

---

### Task 0 (controller): shared constants

**Files:** create `src/test/20261121_round3_affect_gate/r3_common.py`: paths; `RULE_SHA` and `assert_rule()`; the D15
SHA-256 table with `assert_inputs(names)`; `res_dir(smoke)`, `refuse_existing(paths, smoke)`,
`write_json_once(path, rec, smoke)`, `now_ams()`, `git_head()`, `provenance(smoke)`; the `sys.path` setup that makes
round 1's modules importable (`import common as C`, `rc_core as K`, `rb_build as rb`, `rb_eval as rbe`,
`rb_features as rf`, with a check that `common` resolved to round 1's file) and round 2's `r2_fusion as F`; constants
`A0`, `TAUS` (read from `rc_tau.json`), `TEST_SEEDS = (49, 50, 51)`, `SMOKE_SEEDS = (9001, 9002, 9003)`, `RC_CELLS`
(fused 116, 119; counterpart 58, 123), `AFF_CELLS` (fused 39, 119; counterpart 149, 10), `AFF_BRAINSTORM` (the numbers
of §5 item 3 at full precision), `RC_NUMBERS` (§5 item 2).

- [ ] Write it; `python -c "import r3_common"` from the folder passes `assert_rule()` and the input checks.
- [ ] Commit.

### Task 1 (pipeline stream, Opus): seed-parameterised bundle

**Files:** create `r3_bundle.py`, `test_r3_bundle.py`.

**Interfaces:**
- Consumes: `r3_common`; exactly the call sequence of rule §4 item 1 (`run_gonogo.EvalContext`,
  `20261108_new_method_quick_checks/run_checks.model_inputs(..., False)` with `centered_term(..., uniform=True)`,
  `run_n6.load_posteriors`, `run_n6.n6_terms`, `run_told_oracle.fit_one_head` and `global_labels` with the arm-L identity
  asserted, `uniform_probe_scores`, `crossfit_condition_free`, round 1's `common.grouping_stack` and
  `rb_eval.seed42_features`, `rb_build.load_readers("A0", False)`); round 1's `common.verify_inputs()` and the
  `told_oracle.json` SHA-256 on every seed.
- Produces:
  - `build_bundle(seed: int, smoke: bool) -> SimpleNamespace` with `seed, smoke, n, ctx, cl` (anchor paintings, (n,)),
    `parity`, `pair_index`, `cos`, `B`, `pB`, `Bp` (B′(A0) scores), `pBp`, `post` (A0: `{h: {"img", "txt"}}`), `stack`
    (`{d: (n, 3, 13)}` grouping scores in A0 order), `F` (`{c: (n, 18) float64}`), `affect_head` (provenance). Scores
    are `{c: {d: (n, 13) float32}}`, per-anchor dicts are `{metric: (n,)}`. In smoke mode only the episode source
    changes; the checkpoint, heads and posteriors stay the real ones (`model_inputs(..., False)`).
  - `load_external(bundle) -> {"cosine": per_anchor, "rca": per_anchor}` from `per_anchor_seed{s}.npz` (smoke folder
    in smoke mode), asserting `anchor_group`, `pair_index` and `per_anchor(bundle.cos)` equal.
  - `redundancy(bundle) -> {h: {d: float}}` (D7, the brainstorm's `row_corr` definition).
  - `compare_with_round1(bundle) -> dict` (seed 42 only): every comparison of §5 item 1 against
    `C.load_bundle()`, exact; raises on any mismatch.
  - `save_bundle(bundle, path)` / `load_bundle_cache(path)` for the per-seed cache of §6.4.

- [ ] Tests first (`test_r3_bundle.py`): on smoke seed 9001 (built first with `run_baselines.py --smoke
  --episodes-seed 9001`) the builder runs, shapes and dtypes are right, B and B′ are condition-free, Δ^b = −Δ^a, and no
  seed-42 file is opened (patch `open`/`np.load` to record paths); `load_external` fails on a shuffled
  `per_anchor` file; `redundancy` equals a direct numpy Pearson on a synthetic row set.
- [ ] Implement.
- [ ] Dry check on seed 42 (writes to `results/smoke/`, prints only pass or fail): `compare_with_round1` passes every
  comparison and the redundancy order names affect first in both directions.
- [ ] Commit (explicit paths).

### Task 2 (fusion stream, Sonnet): readers, gates, families, statistics

**Files:** create `r3_fusion.py`, `r3_stats.py`, `test_r3_fusion.py`.

**Interfaces:**
- Consumes: `r3_common`; round 1's `rb_build.load_readers`, `rb_eval.half_reader_probs`, `rb_features.average_probs`,
  `common.expected_term`, `common.top_two_margin`, `rc_core.gated_terms`, `rc_core.g_cf`; round 2's
  `r2_fusion.rank_info`, `cell_statistics(..., n_kappa=1)`, `control_choice`, `select_fused`, `select_cf`, `assemble`,
  `decode_cell`; `src.eval.aspect_metrics.per_anchor`, `cluster_bootstrap`.
- Produces (`r3_fusion`):
  - `reader(bundle) -> {"P": {c: (n, 3) float64}, "T": {c: {d: (n, 13) float32}}, "m": {c: (n,)}, "pick": {c: (n,)}}`;
  - `gates_r1(m, taus)`, `gates_aff(m, pick, taus)`, `gates_random(g_r1, share, rng_seed, n)` → list of 4
    `{c: (n,) float32}` (§7 item 7: one generator, condition a drawn first, `random(n)` per condition);
  - `run_family(bundle, T, gates, fused_only=False) -> {"fpick", "cpick" (None if fused_only), "sigma", "fused",
    "cf" (per-anchor dicts), "details"}` on the 224 cells;
  - `score_frozen(bundle, T, gates, fused_cells, cf_cells)` for the frozen-cell line (cell from seed-42 tune half h on
    parity 1 − h);
  - `open_shares(gates, pair_index)`.
- Produces (`r3_stats`):
  - `pooled_check(arrays_by_seed, cl_by_seed) -> {"point", "ci95", "pass"}` (concatenate in seed order, painting
    clusters, `cluster_bootstrap`, percentage points, pass = lower bound > 0);
  - `go_checks(per_seed) -> dict` with the seven checks of §6.5 and the secondary check of §6.6;
  - `sensitivity(diff, cl) -> {"SE", "half_width", "x", "seed42_half_width"}` (§6.1 formula);
  - `bar_info_pooled(...)` for §7 item 2.

- [ ] Tests first (synthetic): AFF's gate is R1's gate times 1[pick = affect]; an arg-max tie including affect counts as
  affect; the counterpart built from AFF's gates is condition-free and differs from one built from R1's gates (mutation
  check); `run_family` on synthetic rows reproduces a hand-computed integer cross-fit with ties to the lowest cell;
  `gates_random` is reproducible and draws condition a first; `pooled_check` treats a painting shared across seeds as
  one cluster; `sensitivity` recovers known σ_a² and σ_ε² on a large synthetic balanced design within 5%.
- [ ] Implement.
- [ ] Commit (explicit paths).

### Task 3 (runners, Sonnet; after Tasks 1 and 2): scripts, rule application, wiring smoke test

**Files:** create `run_r3_seed42.py`, `run_r3_test.py`, `r3_apply_rule.py`, `test_r3_wiring.py`.

**Interfaces:**
- `run_r3_seed42.py`: §5 items 1 to 4 in order (no AFF number before items 1 and 2 pass), writing
  `results/regression_check.json`, `results/seed42_arrays.npz` (R1 and AFF per-anchor arrays, gates, picks), then the
  sensitivity projection of §6.1 to `results/sensitivity.json`. `--dry` writes to `results/smoke/` and prints only pass
  or fail.
- `run_r3_test.py --phase go --seeds S... [--smoke]`: per seed builds and caches the bundle and reader arrays
  (`results/cache_seed{s}.npz`), runs AFF's family and R1's fused-only family, writes `results/go_seed{s}.npz`
  (per-anchor arrays of §6.4 only), then `results/go_pooled.json` (the seven checks and the secondary check). No
  per-seed summary is printed.
- `r3_apply_rule.py [--smoke]`: reads `go_pooled.json`, `sensitivity.json`, the re-derivation's agreement record when
  not smoke, writes `results/test_verdict.json` and `.txt` (GO or NO-GO with the §6.7 reading, the secondary check,
  rule SHA, Amsterdam time).
- `run_r3_test.py --phase descriptive --seeds S... [--smoke]`: refuses without a matching `test_verdict.json`; writes
  `results/descriptive.json` with §7 items 1 to 7.
- `test_r3_wiring.py`: builds smoke seeds 9001 to 9003 if missing, runs the three phases on them end to end in
  `results/smoke/` (the smoke rule application reads the real `results/sensitivity.json`, so this test runs only after
  Task 5's first step), asserts every output exists and every assertion passed, prints no metric value, deletes its
  smoke outputs on success, and runs one wiring mutation (AFF's gated term passed as G_cf) showing that an assertion
  fires.

- [ ] Tests first: the wiring test and the mutation; the GO phase refuses non-smoke seeds outside (49, 50, 51) and
  smoke seeds outside (9001, 9002, 9003); the descriptive phase refuses without a verdict.
- [ ] Implement; `run_r3_seed42.py --dry` passes (prints pass or fail only); `test_r3_wiring.py` passes on a stand-in
  `sensitivity.json` written to `results/smoke/` (the real wiring run follows in Task 5).
- [ ] Commit (explicit paths).

### Task 4 (Opus, independent; starts after the rule commit): re-derivation

- [ ] A fresh agent writes its own code in `rederive/`, does not read Tasks 1 to 3's code, and re-derives in two phases
  (rule §8): phase 1 on seed 42 (§5 items 1 to 4, redundancy, D13), phase 2 after the GO pass (hash check, per-seed B,
  B′, gates, σ*, chosen cells, the seven checks and the secondary check). Agreement as rule §8. Writes
  `rederive/rd3_phase{1,2}_report.md` and `rederive/agreement.json`.

### Task 5 (main session): real runs

- [ ] `uptime`, `free -g`; `run_r3_seed42.py` (background); log the regression result and the sensitivity table.
- [ ] `test_r3_wiring.py` against the real `sensitivity.json` (rule §8 step 4); it must pass, mutation included, before
  any seed is built.
- [ ] Record `codes_provenance.json`'s SHA-256; build seeds 49, 50, 51 with `run_baselines.py --episodes-seed <s>`
  (background, at most 3 processes); hash check against seeds 42, 43, 45, 47, 48 and each other; `codes_provenance.json`
  unchanged; build logs not opened; ledger rows (49 to 51 test, 9001 to 9003 smoke).
- [ ] `run_r3_test.py --phase go --seeds 49 50 51` (background).
- [ ] After Task 4's phase 2 agrees: `r3_apply_rule.py`; log; commit; verdict to the user.
- [ ] `run_r3_test.py --phase descriptive --seeds 49 50 51`; log; commit.

### Task 6: final review and report

- [ ] Whole-branch final review on Opus (re-derives every load-bearing number; one fix wave; scoped re-review).
- [ ] The report `docs/reports/auto/v2/2026-11-21_round3_affect_gate.md` (paper-draft style, baselines beside every
  headline number, figures), its `reports_sum.md` row, `python scripts/check_reports_sum.py`; storage report (files
  over 1 GB); memory update.
