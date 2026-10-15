# Reader fix, round 2: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** compute the seed-42 development numbers of the three round-2 candidates (R1 current, R2 adapted, R3
realistic-practice learned reader, each with the 896-cell gated and top-k restricted fusion family), apply the
committed rule, and run the fresh-seed test only if a candidate clears the bar.

**Architecture:** new modules in `src/test/20261118_reader_fix_round2/`, all named `r2_*.py`, importing round 1's
verified code from `src/test/20261117_reader_fix_csd/` by path and changing nothing there. A reader stream writes each
reader's seed-42 probabilities to a file; a fusion stream turns any probability file into a candidate. The main session
launches every real run on CPU.

**Tech stack:** numpy 2.2.6, scikit-learn 1.6.1, torch (CPU), the CoSiR env
(`/root/miniconda3/envs/CoSiR/bin/python`).

**Spec:** `docs/superpowers/specs/2026-10-06-reader-fix-round2-design.md`; **binding rule:**
`src/test/20261118_reader_fix_round2/DECISION_RULE.md` (committed before any code; the rule governs).

## Global constraints

- CPU only: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1`; at most 3 processes.
- Every script asserts the rule's SHA-256 and the SHA-256 of every D14 input it reads; a mismatch stops it.
- Non-smoke results are never overwritten; smoke outputs go to `results/smoke/` (first 200 episodes per aspect pair,
  round 1's `common.load_bundle(smoke=True)`).
- Round 1's folder is read only; its writing functions (`common.save_candidate`, `common.write_json_once`,
  `common.res_dir`, the `rb_build.py` stages, `run_rc.py`) are not called.
- No round-2 module may be named like a round-1 module (`common`, `rc_core`, `rb_*`, `run_*`): round 1's folder is on
  `sys.path` and a same-named module shadows it.
- Held rows are not read; no number on seeds 49 to 51 before the rule says so.

## Review focus

1. The regression check (rule §4.7) passes only when every array matches exactly; a test must fail when the top-k
   restriction leaks into the k_top = 13 cells.
2. The counterpart stays condition-free under the restriction (asserted per cell); a test must fail if K were taken
   from the fused score instead of B.
3. Ties: B ties inside the top-k set, R@1 ties miss, cell ties to the lowest number; tests on synthetic rows.
4. R3's banks are bit-reproducible from the seeds and nested across k; purity 4 equals round 1's features exactly.
5. Each guard test is shown to fail under its mutation (round 1's final review found two unguarded tests).

---

### Task 0 (controller): shared constants

**Files:** create `src/test/20261118_reader_fix_round2/r2_common.py`: paths, `RULE_SHA`, `assert_rule()`, the D14
SHA-256 table with `assert_inputs(names)`, `res_dir(smoke)`, `write_json_once(path, rec, smoke)`, `now_ams()`, and the
`sys.path` setup that makes round 1's modules importable (`import common as C`, `rc_core`, `rb_build`, `rb_eval`,
`rb_features`).

### Task 1 (fusion stream, Sonnet): top-k restriction, 896-cell cross-fit, counterpart, regression check, rule application

**Files:** create `r2_fusion.py` (pure functions), `run_r2_fusion.py` (CLI), `r2_apply_rule.py`, `test_r2_fusion.py`.

**Interfaces:**
- Consumes: `r2_common`; round 1's `common` (bundle, `expected_term`, `grouping_stack`, `evaluate_fused`), `rc_core`
  (`thresholds`, `gates`, `gated_terms`, `g_cf`, `control_choice`), `rb_eval`/`rb_features` (R1's probabilities and
  `picks_and_margins`); for R2 and R3 the reader stream's `results/probs_<R>_<config>.npz` (`P__a`, `P__b`, float64,
  (n_episodes, H)) with its `.json` record.
- Produces: `run_r2_fusion.py --reader {R1,R2,R3} --config {A0,A1} [--smoke] [--regression]` writing
  `results/tau_<R>_<config>.json` before any score, then `results/cand_<R>_<config>.{json,npz,txt}`;
  `--regression` (R1, A0, cells 0 to 223 only) writes `results/regression_check.json` and stops on any mismatch.
  `r2_apply_rule.py` reads the three A0 candidates and writes `results/rule_application.{json,txt}` (rule §5).

- [ ] Tests first (synthetic data): restriction keeps S inside K and B's order outside; K from B with ties to the lower
  index; k_top = 13 leaves scores unchanged; cell numbering ((κ·4 + t)·7 + u)·8 + a; min-margin and max-R@1 picks with
  ties to the lowest cell; counterpart condition-free per cell. Each guard test checked against its mutation.
- [ ] Implement; the regression check on real data (R1, A0) must reproduce round-1 R-c exactly (rule §4.7).
- [ ] Smoke runs of R1 (A0 and A1); `r2_apply_rule.py` on smoke outputs.
- [ ] Commit (explicit paths).

### Task 2 (reader stream, Opus): R2 and R3 probabilities (A0 and A1)

**Files:** create `r2_readers.py` (pure functions), `run_r2_readers.py` (CLI), `test_r2_readers.py`.

**Interfaces:**
- Consumes: `r2_common`; round 1's `rb_build` (`load_readers`, `fit_half_reader`, bank and halves loaders), `rb_eval`
  (`seed42_features`, `half_reader_probs`), `rb_features` (`both_conditions`, `bank_labels`, `smd`, `average_probs`).
- Produces: `run_r2_readers.py {r2,r3} --config {A0,A1} [--smoke]` writing `results/probs_<R>_<config>.{npz,json}`;
  R2's json holds μ42, σ42, π̂, the iteration count, the code check and the shift report; R3's json holds the D(k)
  table, k*, the purity-4 checks, the chosen C, bank accuracies and top probabilities, and is preceded by
  `results/r3_k_<config>.json` (written before any half-reader is trained). If k* = 4, no R3 probability file is
  written and the json says R3 = R1 (rule §4.4 h).

- [ ] Tests first: R2's code check (own scaler statistics, no EM, equals R1 to 1e-12); EM on a synthetic shift
  recovers a known prior; R3's draws are reproducible, nested and respect the painting constraints; purity 4 leaves the
  bank unchanged. Each guard test checked against its mutation.
- [ ] Implement; smoke runs of r2 and r3 on A0.
- [ ] Commit (explicit paths).

### Task 3 (main session): real runs on seed 42

- [ ] `uptime`, `free -g`; then, in the background: the regression check; R1/A0; `r2 --config A0` then R2/A0; `r3
  --config A0` then R3/A0 (at most 3 processes).
- [ ] Log each result with its time; commit the log.

### Task 4 (Opus, independent): re-derivation

- [ ] A fresh agent writes its own code in `rederive/` (it may reuse round 1's `rederive/`), does not read Tasks 1 and
  2's code, and re-derives every decision number of rule §7: bar margins, intervals and comparators, gain statistics,
  the regression check, τ, μ42, σ42, π̂, the D(k) table and k*.

### Task 5 (main session): apply the rule

- [ ] `r2_apply_rule.py`; log; commit. No candidate clears: report to the user (no test). One clears: Task 6.

### Task 6 (only if a candidate clears): fresh-seed test

- [ ] Sensitivity projection (rule §6.1) to the log; seeds 49 to 51 built with
  `src/test/20261030_aspect_baselines/run_baselines.py --episodes-seed <s>`; hash check; `run_r2_test.py` (Sonnet)
  computes only the GO quantities and writes `results/test_verdict.json`; an independent re-derivation of the GO
  quantities; the seed ledger row; then the deferred diagnostics and the frozen-cell line.

### Task 7: A1 ablation, final review, report

- [ ] A1 ablation on the carried (or best) candidate (rule §4.8). Whole-branch final review on Opus (re-derives every
  load-bearing number; one fix wave; scoped re-review) when a test ran; the full report in `docs/reports/auto/v2/`
  with its `reports_sum.md` row and `python scripts/check_reports_sum.py`.
