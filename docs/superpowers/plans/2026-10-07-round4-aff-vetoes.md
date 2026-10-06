# Reader fix, round 4 (vetoes on AFF's gate, developed on seed 42, fresh-seed test): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** develop three vetoes on AFF's gate (V4: image abstention; V2: the A1 reader must also pick affect; V24: both)
on seed 42 under the committed rule, carry at most one by its paired gain over AFF, and test it on the fresh seeds 52,
53 and 54 with AFF beside it: regression checks, development step and carry, sensitivity, wiring smoke test, seeds,
GO pass, independent re-derivation, verdict, descriptive pass.

**Architecture:** new modules in `src/test/20261122_round4_aff_vetoes/`, all named `r4_*.py`, `run_r4_*.py` or
`test_r4_*.py`. They import round 3's `r3_common`, `r3_bundle`, `r3_fusion`, `r3_stats` (and the pure helpers of
`run_r3_build.py`) by path and change nothing there; through them, round 1's and round 2's code as round 3 uses it. One
seed-parameterised bundle = round 3's A0 bundle plus an A1 extension (CSD posteriors, 24 A1 features, A1 readers,
B′(A1), v). The fusion layer builds the candidates' gates from AFF's gates and runs round 3's `run_family` on them. The
main session launches every real run on CPU.

**Tech stack:** numpy 2.2.6, scikit-learn 1.6.1, torch (CPU), the CoSiR env (`/root/miniconda3/envs/CoSiR/bin/python`).

**Spec:** `docs/superpowers/specs/2026-10-07-round4-aff-vetoes-design.md`; **binding rule:**
`src/test/20261122_round4_aff_vetoes/DECISION_RULE.md` (committed before any code; the rule governs; section numbers
below are the rule's; round 3's rule is part of it by reference).

## Global constraints

- CPU only: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1`; at most 3 processes.
- Every script asserts this rule's SHA-256, round 3's rule's SHA-256 and the SHA-256 of every D11 input it reads.
- Non-smoke results are never overwritten; smoke outputs go to `results/smoke/`. Smoke and dry runs never print or log a
  metric value of any scorer on any seed (only shapes, counts, file names, assertion pass or fail); their value files
  are opened only by assertions and deleted when the smoke test has passed (rule §10).
- No implementer touches seeds 52 to 54; only the main session's real runs of Task 5 do. Smoke seeds 9001 to 9003 only.
- No number of V4, V2 or V24 on seed 42 is computed by any implementer or test (dry runs print pass or fail only); the
  only seed-42 numbers anyone may see before Task 5 are the regression targets of rule §5 items 1 to 4.
- Round 1, 2 and 3 folders, step 1 and the brainstorm are read only; their file-writing functions are not called.
- No module of this folder shares a name with a module of rounds 1 to 3 (`common`, `rc_core`, `rb_*`, `r2_*`, `r3_*`,
  `run_r3_*`, `test_r3_*`, ...): their folders are on `sys.path`.
- Cross-fit ties and the carry compare integer sums (ρ, γ, Δ_k), never float means.
- Held rows are not read.

## Review focus

1. **Round 3's seed guard** (rule §4 item 1). `r3_bundle.build_bundle` refuses seeds outside 42, 49 to 51 and 9001 to
   9003; `_check_seed` reads `r3_common.TEST_SEEDS` at call time. Round 4 does not edit round 3's file: `r4_common`
   sets `R3.TEST_SEEDS = (52, 53, 54)` in its own process on import, and no round-3 function that reads `TEST_SEEDS` or
   `EARLIER_SEEDS` (round 3's build seed checks, `r3_apply_rule.py`) is called. Tests: `RB3._check_seed` admits 42, 52,
   53, 54 and refuses 49 and 55 (guard function only, no build).
2. **The A1 extension leaves round 3's bundle untouched** (every A0 field identical before and after, asserted) and is
   seed-independent in its inputs (no seed-42 file read on a smoke seed; patch `open` / `np.load` to record paths).
3. **Gate algebra:** each candidate's gate ⊆ AFF's at every τ and condition; with both factors 1 the gate is AFF's
   exactly; V24 = V4 · V2; a_v is identical in both conditions; the counterpart of a candidate is built from the
   candidate's own gates (a mutation that passes AFF's gates to the candidate's counterpart must fail a test).
4. **The carry** on synthetic Δ_k: Δ = 0 is not "beats AFF"; gaps of 24 tie and 25 do not; ties go to V4, then V2, then
   V24; an empty E kills; a candidate failing D10 is excluded even with the largest Δ. Bar comparator: for V2 and V24
   the order B′(A1), B′(A0), counterpart, B decides ties; for V4 B′(A0), counterpart, B.
5. **Order on the test seeds** (§6.4): the GO phase computes nothing of non-carried candidates, no AFF counterpart, no
   per-seed summary; B′(A1) enters a check only for a CSD candidate; the assertions (a) to (d) of rule §6.4 (clusters
   = `groups[anchor]`; gates passed in = gates recomputed from the method's definition; build-record hash checks
   re-run; AFF's counterpart not cross-fitted) each fire under mutation, and their open counts are never written; the
   descriptive phase refuses without a verdict carrying this rule's SHA-256 and checks that its cache reproduces the GO
   arrays.

---

### Task 0 (controller): shared constants

**Files:** create `src/test/20261122_round4_aff_vetoes/r4_common.py`: paths; `RULE_SHA` and `assert_rule()` (this
rule) plus round 3's `R3.assert_rule()`; the D11 SHA-256 table and `assert_inputs(names)`; the `sys.path` setup that
imports round 3's modules by path (`r3_common as R3`, `r3_bundle as RB3`, `r3_fusion as RF3`, `r3_stats as RS3`) and
checks each resolved to round 3's file; constants `A1 = ("affect", "image", "caption", "csd")`, `TEST_SEEDS = (52, 53,
54)`, `SMOKE_SEEDS = (9001, 9002, 9003)`, `EARLIER_SEEDS = (42, 43, 45, 47, 48, 49, 50, 51)`, `V75 =
0.021043562795966864`, `V75_KEEP_42 = 9216`, `BPA1_MEAN_42 = 18.804931640625`, `IMGABST_TARGETS` (rule §5 item 3 at
full precision, cells 117, 119 / 58, 67), `CANDIDATES = ("V4", "V2", "V24")`, `READS_CSD = {"V4": False, "V2": True,
"V24": True}`, `TIE_BAND_UNITS = 24`; helpers `res_dir`, `refuse_existing`, `write_json_once`, `now_ams` with this
folder's paths; and, on import, `R3.TEST_SEEDS = (52, 53, 54)` (rule §4 item 1, Review focus 1).

- [ ] Write it; `python -c "import r4_common"` from the folder passes both rule asserts and every input check.
- [ ] Commit (explicit path).

### Task 1 (bundle stream, Opus): A1 extension and seed guard

**Files:** create `r4_bundle.py`, `test_r4_bundle.py`.

**Interfaces:**
- Consumes: `r4_common`; `RB3.build_bundle`, `RB3.save_bundle`, `RB3.load_bundle_cache`, `RB3.compare_with_round1`,
  `RB3.load_external`; round 1's `rb_eval.seed42_features`, `rb_build.load_readers("A1", False)`, `common.INPUT_FILES`;
  `run_step1.full_post`; `uniform_probe_scores`, `crossfit_condition_free`, `per_anchor`.
- Produces:
  - `build_bundle(seed: int, smoke: bool) -> SimpleNamespace`: round 3's bundle (built under round 4's seed guard)
    plus `post["csd"]`, `F1` (`{c: (n, 24) float64}`), `readers_a1`, `Bp1` (B′(A1) scores, `{c: {d: (n, 13)
    float32}}`), `pBp1`, `v` (`(n,) float64`), with asserts: `z["selection"] == ctx.selection`, csd posteriors finite on
    selection rows, `F1[c][:, :18] == F[c]` exactly, `min(F1["b"][:, 6], F1["b"][:, 7]) == v` exactly, B′(A1)
    condition-free, every round-3 field unchanged by the extension.
  - `compare_a1_with_round1(bundle) -> dict` (seed 42 only): post["csd"], `Bp["A1"]` and `pBp["A1"]`, the 24 A1
    features against `rb_eval.seed42_features(C.load_bundle(), A1)`, B′(A1) mean = `BPA1_MEAN_42`; exact; raises on any
    mismatch; round 1's `load_bundle` wrapped so it prints nothing (as `RB3.compare_with_round1` does).
  - `save_bundle(bundle, path)` / `load_bundle_cache(path, seed, smoke, sha256)`: round 3's cache plus an A1 npz beside
    it, both SHA-256-recorded.

- [ ] Tests first (`test_r4_bundle.py`): on smoke seed 9001 (build it first with `run_baselines.py --smoke
  --episodes-seed 9001` if missing) the builder runs; shapes and dtypes; the asserts above; no seed-42 file opened;
  the seed guard tests of Review focus 1; each bundle guard (head identity, finiteness on selection rows, parity,
  selection equality of the CSD heads) fires under a constructed violation (T1-1); cache round trip equal.
- [ ] Implement.
- [ ] Dry check on seed 42 (writes to `results/smoke/`, prints only pass or fail): `RB3.compare_with_round1` and
  `compare_a1_with_round1` pass every comparison; v₇₅ recomputed equals `V75` and the keep count `V75_KEEP_42`.
- [ ] Commit (explicit paths).

### Task 2 (fusion stream, Sonnet): A1 reader, candidate gates, development and carry, checks

**Files:** create `r4_fusion.py`, `r4_stats.py`, `test_r4_fusion.py`.

**Interfaces:**
- Consumes: `r4_common`; `RF3.reader`, `RF3.gates_r1`, `RF3.gates_aff`, `RF3.gates_random`, `RF3.run_family`,
  `RF3.score_frozen`, `RF3.open_count`, `RF3.open_shares`; round 1's `rb_eval.half_reader_probs`,
  `rb_features.average_probs`, `rb_features.picks_and_margins`; `RS3.pooled_check`, `RS3.sensitivity`;
  `src.eval.aspect_metrics.cluster_bootstrap`, `per_anchor`.
- Produces (`r4_fusion`):
  - `reader_a1(bundle) -> {"P": {c: (n, 4) float64}, "pick": {c: (n,) int64}}` (D3; ties to affect);
  - `abstain(v, v75=V75) -> (n,) float32` = 1[v < v₇₅];
  - `gates_candidate(name, g_aff, pick_a1, keep) -> list of 4 {c: (n,) float32}` for V4, V2, V24 (D6), asserting the
    gate algebra of Review focus 3; `gates_candidate("AFF", ...)` with both factors 1 returns `g_aff` exactly;
  - `gates_imgabst_r1(g_r1, keep)` for §5 item 3.
- Produces (`r4_stats`):
  - `comparators(name, bundle_like) -> ordered list of (label, per_anchor)` (D8 order), `bar_comparator(...)`;
  - `dev_record(name, fam, aff_fam, comps, cl) -> dict` (§5 item 6: fused, cf, cells, σ*, bar comparator, bar margin,
    margin, gain statistic, either change, Δ_k integer and its point and interval, the D10 clauses, per-pair bar
    margins);
  - `carry(records) -> {"E", "M", "tied", "carried" (name or None)}` (§5 items 8, 9);
  - `go_checks(name, per_seed) -> dict` (§6.5: 8 or 9 checks, the AFF check last), `reading(check, x)` (§6.7);
  - `sensitivity_all(name, seed42_arrays) -> dict` (§6.1 for every GO check, via `RS3.sensitivity`).

- [ ] Tests first (synthetic): Review focus 3 and 4; every gate is float32 0/1; `reader_a1` on a stub reader
  reproduces a hand-computed mean; constructed exact arg-max ties go to affect for both π and π_A1 (M28); `go_checks`
  lists exactly 8 checks for V4 and 9 for V2/V24, the AFF check last; a bar-comparator test with D8's four-way order in
  which the per-seed, pooled and per-pair choices differ (T2-M4); a ρ_ctrl criterion test (σ* and ρ_ctrl of round 2's
  `control_choice` against a hand-computed 2-half example; M06); Δ_k summed with `r2_fusion.as_int4` (a non-multiple
  of 0.25 raises); the counterpart mutation fails a test.
- [ ] Implement.
- [ ] Commit (explicit paths).

### Task 3 (runners, Opus; after Tasks 1 and 2): scripts, rule application, wiring smoke test

**Files:** create `run_r4_seed42.py`, `run_r4_build.py`, `run_r4_test.py`, `r4_apply_rule.py`, `test_r4_runners.py`,
`test_r4_wiring.py`.

**Interfaces:**
- `run_r4_seed42.py`: §5 items 1 to 5 in order, writing `results/regression_check.json`; only then items 6 to 9
  (`results/dev_seed42.json`, `results/carry.json`, `results/seed42_arrays.npz` with every candidate's and AFF's
  per-anchor arrays, gates and cells); then, if a candidate is carried, §6.1 to `results/sensitivity.json`. A guard
  object refuses to write or print any candidate number while a regression item is pending. `--dry` writes to
  `results/smoke/` and prints only pass or fail.
- `run_r4_build.py --seeds 52 53 54 [--smoke]`: one invocation; per seed `run_baselines.py --episodes-seed <s>` (logs
  to `results/build_seed{s}.log`, never opened); hash check against each other and `EARLIER_SEEDS` with round 3's
  `episode_pair_hashes` / `hash_check` helpers imported from `run_r3_build.py`; `codes_provenance.json` before and after
  each build; `results/build_seed{s}.json`; refuses seeds outside `TEST_SEEDS` (`SMOKE_SEEDS` with `--smoke`).
- `run_r4_test.py --phase go --seeds S... [--smoke]`: reads `carry.json`; refuses if nothing is carried; re-runs the
  build records' file and episode-hash checks (M31); per seed builds and caches the bundle and reader arrays (only
  AFF's and the carried candidate's gates; the A1 reader only for a CSD candidate), asserts rule §6.4 (a), (b) and (d)
  with open counts held in memory only, runs the carried candidate's family and AFF's fused-only family, writes
  `results/go_seed{s}.npz` (§6.4 arrays only), then `results/go_pooled.json`. No per-seed summary printed.
- `r4_apply_rule.py [--smoke] [--boundary-reported <sha256 of go_pooled.json>]`: reads `go_pooled.json`,
  `sensitivity.json` and, when not smoke, the re-derivation's phase-2 agreement record; writes
  `results/test_verdict.{json,txt}` (GO or NO-GO with every §6.7 reading, rule SHA, Amsterdam time). A boundary case
  stops unless the flag carries the current file's SHA-256.
- `run_r4_test.py --phase descriptive --seeds S... [--smoke]`: refuses without a matching verdict; writes
  `results/descriptive.{json,txt}` with §7 items 1 to 4, after checking that its cached arrays reproduce the GO pass's
  `go_seed{s}.npz` exactly (M27).
- `test_r4_runners.py`: the §5 order test (a failing regression item leaves no candidate result in any output or on
  stdout; M20); seed refusals; the descriptive refusal; each §6.4 assertion fires under its mutation (cluster swap,
  gates of another method, a tampered build record, AFF counterpart requested); `--boundary-reported` and the phase-2
  agreement record with a stale `go_pooled.json` SHA-256 refused (T3-1, T3-10).
- `test_r4_wiring.py`: builds smoke seeds 9001 to 9003 if missing; runs build checks, GO, rule application and
  descriptive end to end in `results/smoke/` with a stand-in carried candidate (each of V4 and V2 in turn) and the real
  `results/sensitivity.json` (the stand-in in `results/smoke/` before Task 5); asserts every output and assertion; a
  leak check over all stdout and logs that fails on any decimal number (one-decimal numbers included); three mutations
  that must each fire an assertion (candidate's gated term passed as G_cf; AFF's fused arrays swapped for the
  candidate's in the AFF check; the candidate's gate passed without its veto factor); deletes its value files on
  success.

- [ ] Tests first; implement; `run_r4_seed42.py --dry` passes (pass or fail only); `test_r4_wiring.py` passes against a
  stand-in sensitivity file.
- [ ] Commit (explicit paths).

### Task 4 (Opus, independent; starts after the rule commit): re-derivation

- [ ] A fresh agent writes its own code in `rederive/` (prefix `rd4_`), does not read Tasks 1 to 3's code, and
  re-derives in two phases (rule §8): phase 1 on seed 42 (§5 items 1 to 5, every candidate's development numbers, D10,
  Δ_k, the carry, x for the carried candidate), phase 2 after the GO pass (hash check, per-seed B, B′(A0), B′(A1) where
  a GO comparator, v, gates, σ*, the chosen cells, every GO check). Agreement as rule §8. Writes
  `rederive/rd4_phase{1,2}_report.md` and `rederive/agreement_phase{1,2}.json`. Phase 1 may run in parallel with
  Tasks 1 to 3; its own code computes no candidate number before its own §5 items 1 to 5 pass, and it compares with the
  implementation's files only after its own results are written and hashed.

### Task 5 (main session): real runs

- [ ] `uptime`, `free -g`; `run_r4_seed42.py` (background); log the regression result, the development table and the
  carry.
- [ ] Phase 1 of Task 4; on agreement, act on the carry. Kill: report to the user and stop here.
- [ ] `test_r4_wiring.py` against the real `sensitivity.json`; it must pass, mutations included, before any seed is
  built.
- [ ] `run_r4_build.py --seeds 52 53 54` (background); hash check passed; build logs not opened; ledger rows.
- [ ] `run_r4_test.py --phase go --seeds 52 53 54` (background).
- [ ] Phase 2 of Task 4; on agreement `r4_apply_rule.py`; log; commit; verdict to the user.
- [ ] `run_r4_test.py --phase descriptive --seeds 52 53 54`; log; commit.

### Task 6: final review and report

- [ ] Whole-branch final review on Opus (re-derives every load-bearing number; mutation checks; one fix wave; scoped
  re-review).
- [ ] The report `docs/reports/auto/v2/2026-11-22_round4_aff_vetoes.md` (paper-draft style, AFF and the comparators
  beside every headline number, figures), committed after the final review and its fix wave, with its
  `reports_sum.md` row and `python scripts/check_reports_sum.py`; storage report (files over 1 GB); memory update.
