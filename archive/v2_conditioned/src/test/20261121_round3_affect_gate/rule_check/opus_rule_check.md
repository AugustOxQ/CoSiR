# Rule check: round 3 decision rule (draft, before commit)

Written 2026-10-06 19:45 (Amsterdam) by a fresh Opus reviewer. Checked: `../DECISION_RULE.md` (uncommitted draft,
471 lines) against the spec (728f5d7), round 2's rule and final review, the handoff, the brainstorm's code and results,
and the code the rule cites. Scratch scripts in this folder (`rc3_*.py`) read only seed-42 arrays (brainstorm cache,
round-1 stored arrays, `per_anchor_seed42.npz`) on CPU. No episode seed was built, `run_baselines.py` was not run, and
nothing outside this folder was written (no `__pycache__` either: `PYTHONDONTWRITEBYTECODE=1`).

## Verdict: FIX BEFORE COMMIT (2 blocking, 7 should-fix, 16 nits)

All 36 SHA-256s match, every brainstorm number in the rule is exact, and AFF's seed-42 record reproduces bit for bit
through the rule's float32 path. The two blocking items are a check that would fail on correct code (B1) and a
sequencing contradiction around the smoke test (B2). Both have one-paragraph fixes.

## Blocking

**B1. §5 item 3, line 306: the condition-a open share is a float32 artefact, so the check fails on correct code.**
AFF's τ_0 gate is open on 9,941 of 12,288 condition-a values and 3,627 condition-b values
(`rc3_float32_check.json`, `AFF.open_share_tau0_counts`). 9,941 / 12,288 = 80.90006510416666%. The rule's
80.90006709098816% comes from `bs_10_subsets.py:26`, `float(aff[0][c].mean())` on a float32 array. §7 item 7 defines
the share in float64, and §5 item 3 requires full precision. A float64 implementation therefore fails item 3 and, by §5
and §9, stops the work. Condition b (29.5166015625% = 3,627/12,288) is exact in both precisions.
*Replace line 306 with:* "- AFF's τ_0 gate open on 9,941 of the 12,288 condition-a values and 3,627 of the
condition-b values (80.90006510416667% and 29.5166015625%), compared as integer counts. `bs_10_subsets.json`
(`"H=affect".open_share_tau0`) prints 80.90006709098816% for condition a, the float32 mean of the same 9,941 values."

**B2. The end-to-end smoke test cannot be run where §8 puts it without breaking §10, and its rule step needs a file
that does not exist yet.** §8 (line 406) runs the wiring smoke test in step (1), before the seed-42 checks (2) and the
sensitivity projection (3). §10 (lines 458-459) says that before §5 passes, smoke runs "print and write no AFF or R1
metric". But the same paragraph (lines 460-463) has that test run "the GO pass, the rule application and the
descriptive pass", and those steps compute and write check points, bounds and the verdict reading. The smoke rule
application also needs x from `results/sensitivity.json` (§6.7), and that file only exists after step (3). There is a
second problem. The smoke seeds are 576 fresh episodes drawn from the same selection paintings, so any AFF metric they
print previews the test. Round 2's final review S2 recorded the same slip.
*Fix, §8 order:* "(1) implementation with unit tests; (2) the seed-42 regression checks (§5); (3) the sensitivity
projection (§6.1); (4) the end-to-end wiring smoke test on seeds 9001 to 9003 (§10); (5) the three seeds built (§6.2);
(6) the GO pass; (7) the re-derivation of the GO quantities; (8) the rule applied; (9) the descriptive pass; (10) the
final review; (11) the report." *§10, replace "Before §5 passes they print and write no AFF or R1 metric (only shapes,
counts and the pass or fail of assertions)" with:* "Smoke runs never print or log a metric value of any scorer on any
seed. The console and the log show only shapes, counts, file names and the pass or fail of each assertion. Their files
in `results/smoke/` are opened only by the assertions and are deleted when the smoke test has passed. In smoke mode the
rule application reads `results/sensitivity.json`."

## Should-fix

**S1. D12 (lines 191-193): the bar comparator's scope has two readings, and they give different numbers.** "Whichever
… has the largest mean R@1 over the episodes considered" can mean the comparator is chosen once on all episodes and
reused for every pair. That is `common.bar_info`, which produced §5 item 3's per-pair values. It can also mean a
comparator chosen per pair (or per seed in §7 item 2). Round 2's D10 said "the same comparator is used for every aspect
pair". Round 3 dropped that sentence. Evidence (`rc3_perpair_comparator.py`, AFF on seed 42):

| Pair | Largest mean in the pair | Bar margin vs pooled comparator B′ (rule value) | vs per-pair comparator |
|---|---|---|---|
| emotion × style | counterpart (12.769 > B′ 12.549) | 0.976562 | 0.756836 |
| emotion × genre | B′ | 1.452637 | 1.452637 |
| style × genre | B (21.137 > B′ 21.069) | −0.329590 | −0.396729 |

*Add to D12:* "The bar comparator is chosen once per scope: over the seed's episodes for that seed's numbers, over the
36,864 pooled episodes for pooled numbers. A per-pair bar margin uses the comparator of the scope it breaks down
(`common.bar_info`)."

**S2. §4 item 1 and D10 (lines 177-178, 249-254): passing the smoke flag through would silently swap frozen
components, and the call sequence is not written down.**
- `run_checks.model_inputs(ctx, "A3", scorer_train, True)` loads `checkpoints/smoke/A1_seed42.pt`
  (`run_checks.py:138-139`, `run_gonogo.py:108-110`). The file exists.
- `rb_build.load_readers("A0", True)` loads round 1's `results/smoke/rb_reader_A0.pkl`, which also exists.
  `load_json_npz` checks only that the smoke records are consistent with each other (`rb_build.py:139-150, 669-680`),
  so neither swap raises an error.
- `common.load_bundle(smoke=True)` builds a 600-episode subset of seed 42, not a smoke seed.
- `run_sweep.setup()` is tied to seed 42: line 195 builds `rg.EvalContext(df.SEED, False)` with SEED 42, lines 206-209
  assert that B equals seed 42's C2 arrays, and lines 265-266 assert alignment with `per_anchor_told_oracle.npz`.

D10's "`rc.model_inputs(ctx, "A3", scorer_train, ...)`" leaves the flag open.
*Replace the bundle part of §4 item 1 with:* "ctx = `run_gonogo.EvalContext(s, smoke)`; scorer_train =
`artelingo_splits(ctx.data).scorer_train`; T_N1u = `centered_term(run_checks.model_inputs(ctx, "A3", scorer_train,
False)[0], ctx.pooled, uniform=True)`; post_E2 = `run_n6.load_posteriors(n6_posteriors.npz, ctx)`; B =
`crossfit_condition_free(ctx.cos, T_N1u, run_n6.n6_terms(post_E2, ctx.pooled)[2], ctx.parity)[0]`; the affect heads =
`run_told_oracle.fit_one_head(ctx, global_labels(partition_L, scorer_train, len(ctx.groups)), scorer_train,
run_n6.HEAD_ROWS)` with the arm-L identity asserted; post = {affect: those heads, image: post_E2["image"], caption:
post_E2["caption"]}; B′(A0) = `crossfit_condition_free(ctx.cos, T_N1u, uniform_probe_scores(post, ctx.pooled, A0),
ctx.parity)[0]`; the readers = `rb_build.load_readers("A0", False)`. The A3 checkpoint, the head draw (60,000 rows) and
the readers are the same in every mode. `run_sweep.setup` and `common.load_bundle` are called only by the seed-42
comparison of §5 item 1."

**S3. D15 and §10 (lines 204-206, 449): on the test seeds nothing asserts the D1, D2 and D10 inputs.** D15 relies on
round 1's `common.verify_inputs` "which the seed-42 bundle check calls through `common.load_bundle`". The test-seed
pipeline never calls `load_bundle`, so as written the partitions, `n6_posteriors.npz`, `per_anchor_told_oracle.npz`
and the A3 checkpoint go unasserted on seeds 49 to 51. (`model_inputs` asserts A3 itself; the other three are not
asserted.) `told_oracle.json`, used for D2's head identity, is not hashed anywhere. The new seed files get no recorded
SHA-256 after the build.
*Add to D15:* "The pipeline calls round 1's `common.verify_inputs()` on every seed (D1, D2, D10 inputs and §2's seed-42
episodes) and asserts `src/test/20261111_community_told_oracle/results/told_oracle.json`
(76d9ec896b941a518e7db6684805fe0bc329f6afe4a48e1993992fbb221d70d2). After each build (§6.2) the SHA-256s of
`episodes_seed{s}.npz`, `per_anchor_seed{s}.npz` and `baselines_seed{s}.json` are written to
`results/build_seed{s}.json` and asserted by every later step."

**S4. §6.7 (lines 360-364): the NO-GO reading leaves the mixed case and the two gain checks undefined.** If one failed
check has a point above 0 and another at or below 0, the text gives one sentence for "the result" and does not say how
the first check is read. "<comparator>" also has no name for the gain statistic or for the gain-versus-RCA check.
*Replace with:* "Each failed check is read on its own. If its pooled point is above 0, it is 'inconclusive at a
detectable margin of x' (that check's x from item 1, with the realised pooled half-width beside it). If the point is at
or below 0, it is 'AFF did not beat <name> on fresh episodes', where <name> is cosine, RCA, B, B′(A0) or the matched
counterpart for the R@1 checks, 'the condition-free comparators on condition gain' for the gain statistic, and 'RCA on
condition gain' for the last check. The NO-GO is reported with every failed check's reading. The secondary check is
read the same way with <name> = R1."

**S5. §7 item 7 (lines 395-396): the random-share control is hard-coded to 12,288 episodes, but §10 runs the
descriptive pass on smoke seeds of 192.** With `generator.random(12288)` and "the seed's 12,288 episodes" the smoke run
breaks or masks wrongly.
*Replace with:* "share_c = (the number of episodes whose AFF τ_0 gate is open in condition c) / E, float64, with E the
seed's episode count; keep^a = 1[u < share_a] with u = generator.random(E), then keep^b from a second call". The
seed-42 orientation line could add that the brainstorm used generators 0 and 1 and the float32 share, which this
definition does not reproduce. It is orientation only.

**S6. §8 (lines 410-415): the re-derivation's scope and output folder are not defined, and x is not re-derived.** "Its
own code for everything except the bootstrap" cannot work on test seeds as written. B needs the A3 factor-model
encoding and the affect-head refit. Round 2's final review used its own code "for everything except the bundle loader,
the D6 z-score and the bootstrap". A phase-1 bundle check that calls `load_bundle` would be circular. x (§6.1) enters
the NO-GO reading and is in neither phase.
*Replace with:* "The re-derivation may import the data loaders and frozen components (`EvalContext`, `model_inputs`
with `centered_term`, `load_posteriors`, `fit_one_head`, `load_readers`, `zscore_rows`, `crossfit_condition_free`,
`cluster_bootstrap`). It writes its own code for the features, P, T, margins, picks, gates, G_cf, the 224 cells'
integer statistics, σ*, both cross-fits, the assembly, the per-anchor metrics, the comparisons, the checks and x
(phase 1). It writes only to `rederive/` of this folder."

**S7. §6.2 and §9 (line 444): a crashed build hits two rules at once.** `run_baselines.py` writes
`episodes_seed{s}.npz` early (line 147) and `baselines_seed{s}.json` last (line 207). A crash in between leaves files
that the no-overwrite guard (lines 93-96) then refuses to replace. §9 says a crashed run "is repeated after its partial
outputs are deleted" and also "Seeds are never rebuilt".
*Add to §6.2:* "A build that crashed before `baselines_seed{s}.json` was written is repeated once for the same seed after
its partial outputs are deleted. The episodes are a deterministic function of the seed, and nothing has been computed
from them. The crash and both runs' episode SHA-256s, which must be equal, are logged. 'Never rebuilt' means that a
completed seed is never rebuilt or replaced by another seed."

## Nits

- **N1.** Lines 65, 246 and 406 cite §9 for the smoke test. It is in §10.
- **N2.** §4 item 1 (lines 251-254): "the grouping scores s_h (D4) in both directions; the 18 reader features of both
  conditions (D5)" come after the smoke parenthetical and read as part of "stay the real ones". Move them before "In
  smoke mode".
- **N3.** §10's write scope (lines 449-453) leaves out `src/test/20261030_aspect_baselines/results/smoke/`, where the
  smoke seeds are built (lines 461-462).
- **N4.** Line 306: name the key, `bs_10_subsets.json` → `"H=affect".open_share_tau0` (see B1).
- **N5.** Line 5, "before any script of this folder exists": this check's `rule_check/rc3_*.py` now exist. Say "before
  any implementation script".
- **N6.** §6.11 and the prior: the brainstorm also read 15 declared oracles (with labels) and 2 controls on seed 42
  (brainstorm §6, report lines 339-340). Add them to the disclosure.
- **N7.** §6.1, "the seed-42 bootstrap half-width": define it as half the width of the seed-42 95% interval of the same
  paired difference.
- **N8.** D7 (lines 143-145): require the six recomputed values to equal the stated ones exactly, not only their order.
  They do (`rc3_float32_check.json`).
- **N9.** §2, line 78: `per_anchor` returns r1, gain, other, swap and strict. "Either" is `common.either` (r1 + other).
- **N10.** §10: add `PYTHONDONTWRITEBYTECODE=1` to the run prefix, so that importing the read-only folders writes no
  `__pycache__` into them.
- **N11.** The seed ledger's own convention ("every scoring of an aspect-episode seed on selection rows is one row")
  calls for a row "9001 to 9003: smoke (wiring only, 64 per pair), never results", so these seeds are never mistaken for
  fresh ones. Say when the 49 to 51 row is written (after the hash check of §6.2).
- **N12.** §9 has no row for "the wiring mutation of §10 does not fire an assertion". Suggest: stop, tell the user, build
  no test seed.
- **N13.** §5 item 3 (lines 309-310) says "then governs if the user agrees", while §8 (line 420) says "settles it, and
  the user is told". Use one wording.
- **N14.** Spec §3 calls the seed-42 regression "the end-to-end wiring test (N8)". The rule adds a separate smoke-seed
  test. Say in §10 that this is a deliberate addition: the seed-42 run does not exercise the build, the pooling, the
  rule application or the descriptive pass.
- **N15.** §6.4 (lines 350-351): `run_baselines.py` prints every scorer's R@1 table (lines 211-219). Redirect its stdout
  to a log that is not opened before the verdict, and check the build by exit status and the files written.
- **N16.** Line 26, "seed 42 as two regression checks", and §5's four items: say that items 2 and 3 are those two checks
  and that items 1 and 4 are the bundle check and the recorded bar.

## Verified

- **SHA-256 (36 of 36 match, `sha256sum -c`).** These are the spec (also equal to `git show 728f5d7:…`), the round-1 and
  round-2 rules, §2's `episodes_seed42.npz`, D1 (both files), D2 (`n6_posteriors.npz`), D10 (`A3_seed42.pt`) and all 29
  rows of the D15 table.
- **Cells.** The formula (t·7+u)·8+a with NESTED_U and NESTED_A maps AFF fused (τ_0, 4, 16) to 39 and (τ_2, 0, 16) to
  119, AFF counterpart (τ_2, 4, 4) to 149 and (τ_0, 0.5, 0.5) to 10, and R1's (τ_2, 0, 2), (τ_2, 0, 16), (τ_1, 0, 0.5),
  (τ_2, 0.5, 1) to 116, 119, 58, 123. This equals `r2_fusion.cell_number(0, t, u, a)`.
- **Tune-half convention.** `select_fused` and `select_cf` pick on parity == h, and `assemble` applies the pick to
  parity ≠ h. `bs_lib.Family.crossfit`/`assemble` use the same dicts, so the brainstorm's `fused_cells[h]` is the pick
  on tune half h, as the rule states.
- **Float32 reproduction on seed 42 (`rc3_float32_check.py`).** This used `r2_fusion.cell_statistics(..., n_kappa=1)`,
  `rc_core.gates`/`gated_terms`/`g_cf`, `_combine`, `control_choice`, `select_fused`/`select_cf` and `assemble`, on the
  brainstorm cache. Every §5 item 3 number is equal at full precision:
  - fused 19.136555989583336, counterpart 18.39599609375, comparator B′;
  - bar margin, margin and gain statistic with their intervals; either −1.629638671875;
  - the three per-pair bar margins; cells 39/119 and 149/10; σ* 0/0 (ρ_ctrl 4,532 and 4,483);
  - AFF − R1: fused 0.21769205729166666 [0.06425880757348419, 0.3709597330984391] and bar 0.25634765625
    [0.04280778303598444, 0.46195041633015954].
- **Float32 against float64, cell by cell.** The brainstorm's float64 family ((1+λ_u)z(B) + λ_a·M + λ_a·D) and the
  rule's float32 `_combine` give identical integer statistics in all 224 × 12,288 entries of fused R@1, fused gain and
  counterpart R@1, for AFF and for R1. The near-tie concern of §5 item 3 does not arise on seed 42.
- **Tie sets at the cross-fit maxima (float32).**
  - AFF: fused h0 {39} (188 vs 185), h1 {119} (314 vs 309); counterpart h0 {149} (4,589 vs 4,588), h1 {10, 27}, where
    the lowest cell wins.
  - R1: fused h0 {116, 125, 133, 142}, h1 {119}; counterpart h0 {58, 75}, h1 {123, 140, 179, 196}.
- **R1 = round-1 R-c.**
  - T^c, margins and picks equal the stored arrays.
  - τ recomputed with `rc_core.thresholds` equals `rc_tau.json` exactly, and the minimum margin equals τ_0.
  - The 10 per-anchor arrays and `bar_v` are bit-identical.
  - Bar margin 0.4435221354166667 [0.21646171563312194, 0.6735669710776852] and gain statistic 2.667236328125
    [2.325087836946873, 3.012361650695922] are equal.
- **D7.** The six redundancy values equal `bs_05_aff.json` exactly, no row is left out, and affect is the smallest in
  both directions.
- **Other numbers.**
  - Random-share orientation +0.665 [+0.441, +0.894] and +0.564 [+0.332, +0.798] equal `bs_10_subsets.json`.
  - D10 18.341064453125 and D11 18.436686197916664 are correct; D14 51.261393229166664 equals `bs_04`'s reader pick
    accuracy.
  - 4,602 anchor paintings; `partition_L` has 41 groups and the E2 partitions 64 each; the `n6_posteriors` keys are as
    stated; D5's chosen C is 1.0 and 100.0.
  - scikit-learn 1.6.1 and numpy 2.2.6 match in the env and in the reader record.
  - The brainstorm report states "about 50 variants" and "+0.63, range +0.42 to +0.72".
- **External baselines.**
  - `per_anchor_seed42.npz` holds `cosine__*` and `rca__*` (5 metrics each).
  - Its `anchor_group` and `pair_index` equal the bundle's, and `per_anchor(cos)` equals `cosine__*` exactly.
  - `run_baselines.py` concatenates pairs in EvalContext's order, and `run_n6.e1_arrays` already implements §4 item 2's
    assertions.
- **`codes_provenance.json`.** `run_baselines.py` rewrites it on every non-smoke build (lines 115-116). The current file
  round-trips byte-identically through `json.dumps(indent=1)`, and the SE and C0 checkpoints still hash to the recorded
  values, so its SHA-256 should stay 8e6a5….
- **Smoke mode.** `--smoke` defaults to 64 per pair and writes to `results/smoke/`. `EvalContext(s, True)` reads that
  folder and skips the 4,096 check. Seeds 9001 to 9003 are unused, 49 to 51 are free in the ledger, and no
  `r3_*`/`run_r3_*`/`test_r3_*` name collides with a round-1, round-2 or brainstorm module.
- **Statistics.**
  - §6.1 is round 2's formula verbatim. With seed-42 m_p taken as Poisson draws, E Σ M_p² = 9Σm_p² − 6n checks.
  - Applied to R1 against its counterpart (`rc3_sensitivity_r1.py`), the formula gives SE 0.0672 and x 0.188, which is
    the prior's "about 0.2" and the handoff's 0.067 / 0.19.
  - The GO list, the bootstrap pooled by painting across seeds, the bar-comparator tie order, the strict "> 0" (a bound
    exactly at 0 fails; a point exactly at 0 reads "did not beat") and every cross-fit tie rule are defined and match
    round 2.
- **Spec against rule.** Candidate, freezes, GO list, secondary check, comparators, R1 descriptive, sensitivity logged
  only, random-share control descriptive, disclosure and prior all match. The rule's additions are refinements, plus
  the smoke-seed test (N14).

Scripts and outputs in this folder: `rc3_float32_check.py` (with `rc3_float32_check.json`, gitignored),
`rc3_perpair_comparator.py` and `rc3_sensitivity_r1.py` (print only). Nothing over 1 GB was written.
