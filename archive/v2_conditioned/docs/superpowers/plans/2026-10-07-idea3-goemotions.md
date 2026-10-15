# Reader fix, round 5 (idea 3: GoEmotions placement of captions on AFF, seed 42, fresh-seed test if carried): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** compute GoEmotions on the 32,413 selection captions once, fit the GE head (the CLIP caption head's recipe on
the 28 raw GoEmotions probabilities), and develop G-T and G-TF on seed 42 under the committed rule: regression checks,
development step, carry or kill, measured diagnostics; if a candidate is carried, the sensitivity projection, the
test-seed code with its tests, the wiring smoke test, seeds 52 to 54, the GO pass, the verdict and the descriptive pass.
An independent re-derivation checks every decision number; the whole-branch final review comes before the report.

**Architecture:** new modules in `src/test/20261123_idea3_goemotions/`, all named `r5_*.py`, `run_r5_*.py` or
`test_r5_*.py`. They import round 3's `r3_common`, `r3_bundle`, `r3_fusion`, `r3_stats`, round 4's `r4_common`,
`r4_bundle` and `r4_stats.bar_comparator`, the told-oracle line's `run_told_oracle`, `run_checks`, `run_n6` and round
1's `rc_core` by path and change nothing there. A placement object (`clip` or `ge`) carries the caption-side affect
posterior through every function; a process guard refuses `ge` objects until §5 items 1 to 4 have passed. The bundle is
round 4's `build_bundle(s, smoke)` plus a GE extension (stack_G, F_G, B′_G) that never touches the bundle's own fields.
The fusion layer computes G-T's and G-TF's readers, τ′ and gates and runs round 3's `run_family`. The main session
launches every real run; only the GoEmotions step may use the GPU (about 5.5 minutes on CPU if the GPU is busy).

**Tech stack:** numpy 2.2.6, scikit-learn 1.6.1, torch 2.11.0+cu130, transformers 5.6.2, the CoSiR env
(`/root/miniconda3/envs/CoSiR/bin/python`).

**Spec:** `docs/superpowers/specs/2026-10-07-idea3-goemotions-design.md`; **binding rule:**
`src/test/20261123_idea3_goemotions/DECISION_RULE.md` (committed before any code; the rule governs; section, D and item
numbers below are the rule's; round 4's and round 3's rules are part of it by reference).

## Global constraints

- Every Python call, `--help` and one-liners included: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  PYTHONDONTWRITEBYTECODE=1`; at most 3 processes. Only the main session's GoEmotions run (Task 8) may use the GPU.
- Every script asserts this rule's SHA-256, round 4's and round 3's rules' SHA-256, every D12 input it reads and the
  file SHA-256 of every module it imports from earlier rounds (rule §8, T1 row).
- **No implementer or test computes a GE-placement result (D11) on real data.** Tests run on synthetic bundles,
  synthetic placements and smoke seeds; `run_r5_seed42.py --dry` stops at the guard. No implementer runs GoEmotions on
  a real caption (tests use a stub model) or fits the GE head on real data (`run_r5_placement.py --item3-only` checks
  the CLIP reproduction without the GE head). The only seed-42 numbers anyone may see before Task 8 are the regression
  targets of §5 items 1 to 4.
- No implementer touches seeds 52 to 54; only the main session's real runs do. Smoke seeds 9001 to 9003 only.
- Non-smoke results are never overwritten; smoke outputs go to `results/smoke/` and print or log no metric value.
- The folders of rounds 1 to 4, step 1, the brainstorm, the told-oracle line and the affect line are read only; their
  file-writing functions are not called.
- No module of this folder shares a name with a module of rounds 1 to 4 (`common`, `rc_core`, `rb_*`, `r2_*`, `r3_*`,
  `r4_*`, `run_*`, `test_*` of those rounds).
- Cross-fit ties and the carry compare integer sums (ρ, γ, Δ_k), never float means.
- Implementers never read `rederive/`; the re-derivation never reads `r5_*` code. Held rows are not read.
- No test the rule lists (§10 lists A and B) is skipped, deferred or replaced without asking the user.

## Review focus

1. **The shared affect-head cache** (D5). `r3_bundle._HEADS` is one read-only object reused by every bundle of a
   process. The GE extension builds a new `post_Q` dict and never assigns into `bundle.post` or `_HEADS`; a test that
   assigns into either must fail, and fingerprints of round 3's and round 4's fields are equal before and after.
2. **The guard** (D11). Every function that takes a placement refuses a `ge` object before release, called directly;
   `release()` accepts only a `results/regression_check.json` that carries this rule's SHA-256 and items 1 to 4 passed;
   the diagnostics refuse until `results/carry.json` exists.
3. **G-T against G-TF wiring** (D6). G-T reads P, m and π from F (CLIP) and T from stack_G; G-TF reads both from F_G;
   τ′ is computed only on seed 42 and read from `dev_seed42.json` (SHA-256 asserted) on every other seed; each
   counterpart is built from its candidate's own term and gates.
4. **Row alignment** (D2 to D4). Selection rows ascending and equal to `ctx.selection`; `sample_ids` recorded; the
   regression sample's positions index `affect_probs` in scorer-train order; the GE input X has `X[scorer_train[i]] =
   affect_probs[i]`; a shifted mapping or a shifted sample must fail a test.
5. **The carry and D10 at their boundaries** (§5 items 6 to 8): a gap of 24 ties and 25 does not; Δ_k = 0 is not in
   E; a bar margin point of exactly 0.5 passes; a lower bound of exactly 0 fails; ties go to G-T; the bar comparator's
   tie order is B′_G, B′(A0), counterpart, B.

---

### Task 0 (Sonnet): shared constants and the guard

**Files:** create `.gitignore` (a copy of round 4's, D12), `r5_common.py`, `r5_guard.py`, `test_r5_common.py`.

**Interfaces:**
- `r5_common`: paths (`ROOT`, `CACHE`, `RESULTS`, `SMOKE`); `RULE_SHA`, `assert_rule()` (this rule, plus round 4's and
  round 3's rule asserts); the D12 table and `assert_inputs(names)`; `MODULE_SHA` (every earlier-round module file this
  round imports) and `assert_modules()`; the `sys.path` setup importing by path, with resolved-file checks:
  `r3_common as R3`, `r3_bundle as RB3`, `r3_fusion as RF3`, `r3_stats as RS3`, `r4_common as R4C`, `r4_bundle as RB4`,
  `r4_stats as RS4`, `run_told_oracle as RTO`, `run_checks as RCHK`, `run_n6 as N6`, `rc_core`; on import
  `R3.TEST_SEEDS = (52, 53, 54)` and `assert R4C.TEST_SEEDS == (52, 53, 54)`. Constants: `TEST_SEEDS`, `SMOKE_SEEDS`,
  `EARLIER_SEEDS = (42, 43, 45, 47, 48, 49, 50, 51)`, `CANDIDATES = ("G-T", "G-TF")`, `TIE_BAND_UNITS = 24`,
  `GOEMO_BATCH = 256`, `GOEMO_MAXLEN = 64`, `REG_SAMPLE = (5, 2048)`, `SPOT_SAMPLE = (6, 1024)`, `GOEMO_TOL = 1e-4`,
  `GE_MAX_ITER = 300`, `GE_FALLBACK_MAX_ITER = 3000`, the CLIP-head record targets, the AUC target
  0.7870951145887375, the group lift 2.7111312041209863, AFF's item-1 targets at full precision, AFF minus B′(A1),
  AFF's either per gain; `GOEMO_FILE_SHA = None` and `GE_POST_SHA = None` (each set by a one-line commit in Task 8);
  helpers `res_dir`, `refuse_existing`, `write_json_once`, `now_ams`, `sha256_file`.
- `r5_guard`: `Placement(kind, Q, sha256)`; `clip_from_bundle(bundle)` (Q is the bundle's `post["affect"]["txt"]`,
  asserted equal by value); `ge_from_file(path)` (file SHA-256 must equal `r5_common.GE_POST_SHA`; scatters
  `post_sel` into a NaN float32 (308,723, 41)); `require(placement, what)` raises `GuardError` for `ge` before release;
  `release(regression_check_path)`; `require_carry(path)`.

- [ ] Tests first (list A item 1 and the guard): a patched input or module hash stops the run before any build; the
  seed guard admits 42, 52, 53, 54, refuses 49 and 55, admits 9001 to 9003 only with smoke; `R4C.TEST_SEEDS` equals
  this round's; `require` refuses `ge` before release and accepts `clip`; `release` refuses a regression record with a
  failed item, a missing item or another rule SHA-256; `ge_from_file` refuses a wrong SHA-256.
- [ ] Implement; `python -c "import r5_common, r5_guard"` from the folder passes every assert.
- [ ] Commit (explicit paths).

### Task 1 (Sonnet; after Task 0): the GoEmotions step

**Files:** create `r5_goemo.py`, `run_r5_goemotions.py`, `test_r5_goemo.py`.

**Interfaces:**
- `selection_captions(ctx) -> (rows, sample_ids, captions)` (D2 item 1 with its assertions);
  `regression_sample(ctx) -> (pos, captions)` (D3); `compare(rerun, stored, tol) -> dict` (max, mean, count above 1e-5,
  passed); `run(device, loaded, out_dir)`: regression sample first, then (only if it passed) the selection captions;
  writes `cache/r5_goemotions_selection.{npz,json}` once (D2 item 5, with the selection captions' largest token count and the count over 64).
- `run_r5_goemotions.py --device {cuda,cpu}`: asserts the device used equals the device given; prints the two file
  SHA-256s and the pass or fail of item 2, never a probability.

- [ ] Tests first (list A item 2, stub model returning deterministic probabilities): the row and join assertions fire on
  an unsorted, short, scorer-train-overlapping or held-overlapping row set; the `refs/main` snapshot and
  `COSIR_ARTELINGO_ANNOTATIONS` asserts (D2 item 2); D3's positions and order; tolerance 0.9e-4 passes and 1.1e-4 fails; a sample shifted by one row fails; a failed sample passes no selection caption to the model; the device tag
  and the device refusal; batch 256 and max_length 64 passed through; no overwrite.
- [ ] Implement.
- [ ] Commit (explicit paths).

### Task 2 (Sonnet; after Task 0): the placement step

**Files:** create `r5_placement.py`, `run_r5_placement.py`, `test_r5_placement.py`.

**Interfaces:**
- `place(F, transform, lab, scorer_train, rows, max_iter) -> {"post", "classes", "n_iter", "acc", "draw", "check"}`
  (D4: `fit_one_head`'s draw, check rows, fit, `predict_proba` scattered into NaN float32, held-out accuracy);
  `ge_input(ctx, goemo_npz) -> X` (D4 input mapping); `fit_ge_head(X, lab, scorer_train, rows)` with the fallback;
  `write_ge_posterior(...)` (`cache/r5_ge_posterior.npz` once).
- `run_r5_placement.py [--item3-only]`: §5 item 3 (the placement function equals `RTO.fit_one_head`'s `txt` and `img`
  posteriors exactly, its record equals told-oracle arm L's head, `scorer_train[pos] == draw`); then, without
  `--item3-only`, the GE head, Q_GE and `results/placement.json`. Prints pass or fail and file SHA-256s only; the GE
  head's accuracy goes to `results/placement.json`, which the main session reads.

- [ ] Tests first (list A item 3, synthetic data): `place` equals a direct `LogisticRegression` fit with the same draw,
  check rows and scatter; NaN outside the rows, float32; `classes_` other than 0..K−1 refused; the fallback (a fit
  that reaches max_iter refits once with the larger cap and records both counts; a second cap stops); a misaligned
  scorer-train mapping refused.
- [ ] Implement; `run_r5_placement.py --item3-only` passes on the real data (CLIP heads only; prints pass or fail).
- [ ] Commit (explicit paths).

### Task 3 (Opus; after Task 0): the GE extension and the candidates

**Files:** create `r5_bundle.py`, `r5_fusion.py`, `test_r5_bundle.py`, `test_r5_fusion.py`.

**Interfaces:**
- `r5_bundle.extend(bundle, placement) -> SimpleNamespace(stack, F, Bp, pBp, kind)` (D5; `r5_guard.require` first;
  new `post_Q`; fingerprints of round 3's and round 4's fields and of `_HEADS` before and after; the image and caption
  columns and slices equal; B′_Q condition-free); `save_ext` / `load_ext` (bound by SHA-256 to round 4's two cache
  files).
- `r5_fusion.candidate(name, bundle, ext, tau_prime=None) -> {"P", "T", "m", "pick", "gates", "taus"}` (D6; G-T:
  `RF3.reader` on F with `ext.stack`, AFF's τ and `RF3.gates_aff`; G-TF: `RF3.reader` on `ext.F` and `ext.stack`, τ′);
  `tau_prime(m) -> tuple` (`rc_core.thresholds`, condition a first); `run_candidate(name, bundle, ext, cand) ->
  family` (`RF3.run_family` with the D7 gate check and the gates stored).

- [ ] Tests first (list A items 4 and 5): on synthetic bundles, `extend` with a `clip` placement returns the bundle's
  own stack, F and B′(A0); with a synthetic Q ≠ Q_CLIP only the affect slice, F columns 0 to 5 and B′_Q change, and an
  extension that ignores its Q fails D5's positive check (`positive_check(bundle, ext, placement)`, boolean only); a
  mutation assigning into `bundle.post["affect"]["txt"]` or `_HEADS` fails, and the whole of `bundle.post` keeps its
  fingerprint; G-T's gates
  equal AFF's; G-T reading T from F's stack, or P from F_G, fails; τ′ never recomputed on a non-42 seed; the
  counterpart mutation (AFF's gates passed) fails; constructed exact π′ ties (M28); the ρ_ctrl criterion example
  (M06); every function refuses a `ge` object before release. A smoke-seed (9001) run with a `clip` placement.
- [ ] Implement.
- [ ] Commit (explicit paths).

### Task 4 (Sonnet; after Task 0): records, carry and diagnostics

**Files:** create `r5_stats.py`, `r5_diag.py`, `test_r5_stats.py`, `test_r5_diag.py`.

**Interfaces:**
- `r5_stats.comparators(pB, pBp0, pBpG, cf)` (D8 order B′_G, B′(A0), counterpart, B), `bar_comparator` via
  `RS4.bar_comparator`; `dev_record(name, fam, aff_fam, pB, pBp0, pBpG, pBp1, cl, pair_index) -> dict` (§5 item 5);
  `d10(record) -> {clauses, boundary flags}`; `delta_k(fam, aff_fam) -> int` (`r2_fusion.as_int4`); `carry(records) ->
  {"E", "M", "tied", "carried"}` (items 7, 8).
- `r5_diag`: `auc_emotion(P_affect, pair_index)` (bs_07's sets and order), `auc_delta(F)`, `pair_lift(Pi, Pt, labS,
  gS)` (`RTO.pair_stats_heads`), `sharper_term(bundle, fam, ...)` (diagnostic (d), re-assembled scores equal the
  family's arrays); every function calls `r5_guard.require` and `require_carry`.

- [ ] Tests first (list A items 6 and 8): the comparator order with a case where per-seed, pooled and per-pair choices
  differ (T2-M4); D10 pinned at its thresholds (0.5 passes, 0.49 fails, a lower bound of exactly 0 fails, clause 3
  reads the gain statistic) with boundary flags; Δ_k from integers (a non-multiple of 0.25 refused); the carry (24
  ties, 25 does not, Δ_k = 0 not in E, all three clauses required, ties to G-T, empty E is a kill); no `or` inside a
  test assertion; the AUC sets and order on a hand-made example; Δ_affect's column; the per-condition and
  per-direction definitions of (d).
- [ ] Implement.
- [ ] Commit (explicit paths).

### Task 5 (Opus; after Tasks 1 to 4): the seed-42 runner

**Files:** create `run_r5_seed42.py`, `test_r5_runner.py`.

**Interfaces:** `run_r5_seed42.py [--dry] [--continue-boundary <sha256>]`: the full input check on every entry and
resume path; item 1 (`RB3.compare_with_round1`, `RB4.compare_a1_with_round1`, R1 and AFF through `RF3`, equality with
round 4's `seed42_arrays.npz`, AFF minus B′(A1)); items 2 and 3 re-asserted from their records; item 4 with
`clip_from_bundle`; `results/regression_check.json`; `r5_guard.release`; D5's positive check; items 5 and 6 (`results/dev_seed42.json`,
`results/seed42_arrays.npz`); the boundary stop (`results/boundary_seed42.json`) or items 7 and 8 (`results/carry.json`,
console line "CARRY <name> (pending the phase-1 agreement, rule §8)" or "KILL (pending ...)"); then the diagnostics
(`results/diagnostics_seed42.json`). `--sensitivity` (§6.1): refused unless `results/carry.json` names a carried
candidate and the log holds the phase-1 agreement record; nine checks in §6.5's order via `RS3.sensitivity`;
`results/sensitivity.json`. `--dry` writes to `results/smoke/`, stops when the guard would be released and
prints pass or fail only.

- [ ] Tests first (list A item 7): the items run in the rule's order; a failing item leaves no GE-placement result in
  any output or on stdout (M20); the guard refuses each GE-taking function called directly before release (T3a-2); the
  development step's gate check fires under a mutated τ′ (T3a-3); every entry and resume path runs the full input
  check (T3a-1); the boundary continuation with a carried candidate and with a kill (T3a-4); the CARRY and KILL lines
  (T3a-5); the `--sensitivity` refusals and SE and x on a hand-computed example (rule check S1); no non-smoke
  overwrite; the dry run stops at the guard and its console and log hold no decimal number (the
  leak regex catches `0.5`, `.5`, `5e-03` and one-decimal forms).
- [ ] Implement; `run_r5_seed42.py --dry` passes (pass or fail only).
- [ ] Commit (explicit paths).

### Task 6 (Opus; only if a candidate is carried, after the phase-1 agreement): sensitivity and test-seed code

**Files:** create `run_r5_build.py`, `run_r5_test.py`, `r5_apply_rule.py`,
`test_r5_testseeds.py`, `test_r5_wiring.py`.

- [ ] Before this task starts, the main session has run `run_r5_seed42.py --sensitivity` (Task 5's code, tested in list
  A) and the re-derivation's x has been compared.
- [ ] Tests first (list B): the build runner (one invocation for 52 to 54; hash checks against `EARLIER_SEEDS` and among
  the new seeds; `codes_provenance.json` before and after; build records; crash handling; ledger rows); the GO-pass
  assertions (a) to (f) of §6.4, each firing under a mutation (the AFF swap by copy); the nine checks in the rule's
  order, B′(A1) in none, the non-carried candidate never computed (G-TF's reader and F_G only if G-TF is carried;
  stack_G and B′_G always), open counts never written; the rule application
  (§6.7 readings including the AFF-only failure; a lower bound within 1e-12 of 0 goes to the user); the descriptive
  pass reproduces the GO cache (M27); `--boundary-reported` and the phase-2 agreement bound to the current
  `go_pooled.json` SHA-256 (T3-1, T3-10); the wiring smoke test on 9001 to 9003 with its three mutations and the leak
  check.
- [ ] Implement; the wiring smoke test passes.
- [ ] Commit (explicit paths).

### Task 7 (Opus, independent; starts after the rule commit): re-derivation

- [ ] A fresh agent writes its own code in `rederive/` (prefix `rd5_`), never reads `r5_*` code or round 3's and round
  4's re-derivations, and imports only rule §8's list. Phase 1: its own §5 items 1, 3 and 4 (τ and B′ parts); once the
  GoEmotions file exists (SHA-256 read from its record and the log line), the CPU spot check (1,024 rows,
  `default_rng(6)`, 1e-4); then its own GE head, Q_GE, the
  accuracy, `n_iter_`, the fallback decision, stack_G, F_G, B′_G, every development number of item 5, D10, Δ_k, the
  carry and, for a carried candidate, x. Phase 2 after the GO pass (it may also read each test seed's
  `per_anchor_seed{s}.npz` and `baselines_seed{s}.json`, SHA-256 asserted against `results/build_seed{s}.json`). It writes and hashes its results before comparing;
  the controller records only "phase 1 finished" and the SHA-256 until the implementation's run has written
  `carry.json`. Reports `rederive/rd5_phase{1,2}_report.md`, `rederive/agreement_phase{1,2}.json`. The controller checks
  its import lines against the list and logs the check.

### Task 8 (main session): real runs

- [ ] `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`; if empty, `flock -n -o -E 75
  /tmp/gpu0.lock env CUDA_VISIBLE_DEVICES=0 ... run_r5_goemotions.py --device cuda` (background), otherwise CPU; log;
  commit `GOEMO_FILE_SHA` (one line).
- [ ] `run_r5_placement.py` (CPU); log the GE head's accuracy beside 35.72 and 9.81; commit `GE_POST_SHA`.
- [ ] `uptime`, `free -g`; `run_r5_seed42.py` (background); log the regression result, the development table, the carry
  line and the diagnostics.
- [ ] Phase 1 agreement (Task 7); on agreement record the carry or the kill. Kill: report to the user; go to Task 9.
- [ ] If carried: `run_r5_seed42.py --sensitivity`; the re-derivation's x compared; Task 6, then the wiring smoke test, `run_r5_build.py --seeds 52 53 54`, the GO pass, phase 2,
  `r5_apply_rule.py`, the verdict to the user, the descriptive pass; ledger rows; commits.

### Task 9: final review and report

- [ ] Whole-branch final review on Opus (re-derives every load-bearing number with its own code, including the
  measured diagnostics; mutation checks; one fix wave; scoped re-review).
- [ ] The report `docs/reports/auto/v2/2026-11-23_idea3_goemotions.md` (paper-draft style, AFF and the comparators
  beside every headline number, figures), committed after the final review and its fix wave, with its
  `reports_sum.md` row and `python scripts/check_reports_sum.py`; storage report; memory update.
