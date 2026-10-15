# Final review B2: stage 0a, stage 0b, the stage 2 driver and pins, the recipe, C13 phase 1

Reviewer B2, 2026-10-14 01:55 (stopped at the context limit; what is left is listed at the end). Snapshot `/project/CoSiR-wt/r9-fr-b2` at 4458634 (the integration branch
r9-scope). F = `src/test/20261202_r9_scope/`. Follows the first area-B reviewer's note
(`.scratch/r9-scope/notes/fr_b_progress.md`); its six "Checked" items are taken as done.

## 1. Verdict

**CONFIRMED WITH FIXES.** Every load-bearing number I re-derived holds (stage 0b item 1 to 3 in full, the pins
re-hash, the CLIP dry run's 96 array hashes, gate share, means and P-points). No HIGH finding. One MEDIUM blocks the
reads: the C13 phase-1 compare cannot read a real `pins_<B>.json` (M-1). No fix touches the 17 closure modules,
except the LOW L-1, which can wait.

## 2. What I re-derived

All with my own code (no `r9_*` import), in `F/final_review/fr_b2_*.py`; outputs beside them as `fr_b2_*.json`.

| Check | How | Mine vs the record | Result |
|---|---|---|---|
| Re-hash of `pins_clip.json` | `fr_b2_rehash.py`: SHA-256 of every listed file (input set, parts, gate references) | 51/51 inputs, 29/29 parts, 3/3 gate references equal; the 16 parts also in the input set carry the same SHA-256 in both lists; `pins_clip.json` SHA-256 b78dd5ef… equals stage 0b's `reference_pins.pins_clip_sha256` | holds |
| Stage 0b item 1, parts | `fr_b2_replay.py`: the replay folder against CLIP's pinned files; npz keys both ways, dtype, shape, every element (NaN in place) | 12/12 npz parts equal in every key (partitions 4, graph 5, bank_AIC 10, n6_posteriors 7, step1_heads_style 9, five rb_heads 5 each, two banks 10 each); 10 of 12 are byte-identical files; the reader pickle is byte-identical | holds |
| Stage 0b item 1, JSON parts | whole-file diff, exact types, every path | `rc_tau`: taus, percentiles, n_values, config equal; only provenance fields and the method sentence differ. `rb_reader_A0.json`: every CV value and pick equal; only `fit_s` timings, provenance, file-hash fields, the `model_spec` wording and the `exploratory` note differ. `baselines_seed42.json`: fit_rows, fit_rows_sha256, episodes_sha256 and all ten lambda picks equal; the replay does not write the R@1 tables (see finding L-2). `picks_seed42.json`: B, B0, B1, mean_r1, coef/episode/fit-row hashes, lambda picks equal; only backbone, rule SHA, time, module/input SHAs and two new check flags differ | holds |
| Stage 0b item 2, A3 | `torch.load` on CPU of both checkpoints, every field | 10 state-dict tensors `torch.equal` with the same dtype and shape; `feature_dim` equal; config equal except `aspect_tau_fixed: False` (bool) added; the whole training history (2,000 steps, loss, aspect loss, tau) equals the pinned history; step-1 loss 20.68084716796875 exactly | holds (case 1) |
| Stage 0b item 3, picks | cells file and picks file diffs above | replay `cells_seed42.json` byte-identical to `F/cells_seed42_clip.json`: AFF 39/119, CF 149/10, R1 116/119 and 58/123, sigma* 0/0; B, B0, B1 picks equal `picks_seed42.json` (df1fff78…) | holds |
| CLIP dry run, 96 array SHA-256s | `fr_b2_arrays.py`: SHA-256 of C-contiguous bytes of round 3's `seed42_arrays.npz` and AB's `per_anchor_seed42.npz` | 96/96 equal (sha, dtype, shape); the 96 keys are exactly round 6's 39 + 57 array items | holds |
| `regression_seed42.json` split | count item names by prefix | SHA-256 ca168eb6… as rule 3.1; 131 items, 0 failed; 39 `seed42_arrays/`, 57 `per_anchor_seed42/`, 35 other | holds |
| Gate share | own count: tau index = cell // 56, gates a and b of tune half h on parity 1 - h | 11,071 / 24,576 = 0.4504801432291667 = the dry run's and `pins_clip.json`'s value; with the same parity it would be 0.45536 (so the parity choice is visible) | holds |
| Seed-42 means | AFF, CF: mean of round 3's `aff_fused__r1`, `aff_cf__r1` x 100; B, B0, B1: round 6's picks | 19.136555989583336, 18.39599609375, 18.341064453125, 18.436686197916664, 18.804931640625: all equal the dry run's | holds |
| P1 to P7 points and n_j | the dry run's `pass_counts` vs round 6's `seed42_pass_counts.json` (5618c59a…), all fields but the five run-specific ones | 0 differences in checks, coef_sha256, episodes_sha256, holm_order, mode, n_clusters, n_episodes, secondary, seeds | holds |

## 3. Findings

**M-1 MEDIUM (blocks H6, H7, H9, H10; outside the closure).** `rederive/rd9_compare.py --phase1` cannot read the
pins schema `build_pins` writes. `rederive/rd9_inputs.py:93` resolves `input_sha256` keys (relative to `src/test/`)
against the round folder, so you get a FileNotFoundError. With that patched, the logical keys carry the backbone's
SHA-256s, so you get "pin mismatch". `rd9_compare.py:281-283` needs `seed42.array_hashes` with keys such as `B__r1`,
but the pins hold `seed42.arrays` with keys such as `seed42_arrays/aff_fused__r1`. The 96 keys hold no B or B0 arrays,
but line 284 requires them. Shown on the real dry-run pins (`fr_b2_phase1_probe.py`, `.json`). It fails closed, so no
wrong verdict. Fix (rederive only): read only `parts` `{path, sha256}`; map `seed42.arrays` keys to the archive
names; drop B and B0 from the required arrays (their means are compared); add a test on
`t13_clip_dryrun/pins_clip_b32.json`. No rerun is needed.

**M-2 MEDIUM (outside the closure).** The phase-1 compare has no test for most of the fields rule §7.5 lists. If I
delete the comparison of sigma*, the AFF cells, the CF cells, the B0 picks, RCA's lambda, tau or the P-points, all
96 rederive tests still pass. The tau tolerance and the exact-picks branch are also untested (`rd9_compare.py:22-27`,
`262-279`). Only switched-off, the means, n_j and the array hashes are tested
(`rd9_test_arithmetic.py:129`). Fix: in that test, change each field once and assert its key fails. Test only.

**M-3 MEDIUM (test only; not in the closure).** These guards in `run_r9_stage2.py` survive deletion under all 74
stage 2 tests (section 5):
- provenance: upstream changed (:370), wrong producer (:364), another backbone (:360), file SHA vs record (:362),
  provenance key completeness (:200), reader feature layout (:226);
- `pins_evidence`: stage 0b changed (:558), closure changed (:561), receipt changed (:566), record changed (:573),
  feature store changed (:575), frozen file changed (:578), the frozen re-assert (:579), and seed-42 or sensitivity
  changed (:583).

Failure scenario: an edit drops line 370, every test stays green, and a reader fitted on a stale head file is
pinned (rule §3.4 "Heads" order). Fix: one test per guard with the existing `world` fixture. Do not edit
`run_r9_stage2.py` itself before both chains have written their pins: `pins_evidence` compares the record's
`runner_sha256` with the file.

**M-4 MEDIUM (outside the closure).** The suite fails at 4458634:
`test_r9_pins.py::test_no_r9_module_imports_an_r6_module`. `run_r9_descriptive.py:228-229` (`h5_inputs`, added by
t23 fix 1b15a8d) imports `r6_common` and `r6_descriptive`. Ticket 23's reviews ran only `test_r9_descriptive.py`.
Fix: allow exactly these two modules for `run_r9_descriptive.py` in the test, with the reason (§8, after the
verdicts, reads H5's R6 layout). Or read H5's constants without importing them.

**L-1 LOW (closure module `run_r9_stage0a.py`: can wait).** `item()` (:266), `compare_items` and `compare_counts`
(:173) compare with `==`, so 1 == 1.0 == True. A change of scalar type passes the gate, though rule 3.1 says
"scalars exactly". Shown in my run: `item(1, 1.0)`, `item(True, 1)` and `compare_counts({'n_clusters': 4602.0}, …)`
all pass. A value change is still caught. Fix later, or outside the closure: a type-strict JSON check of the 0a
record.

**L-2 LOW.** Stage 0b item 1 names "the RCA and PM fits". The replay compares only fit_rows, fit_rows_sha256,
episodes_sha256 and the ten lambda picks: the recipe writes no fit arrays. The fits are covered indirectly, because
the per-anchor arrays of RCA and the nine PM scorers (in the 57 per_anchor items) are bit-equal in the regression.
No fix is needed. Note it in the report.

**L-3 LOW (lead 2).** For `rb_reader_A0_json`, stage 2 provenance tests only `feature_names` (copied). The C picks and
CV values are listed as metadata and not compared. The fit is tested through the pickle's coefficient arrays. The
`feature_names` check itself is untested (see M-3).

**L-4 LOW.** A backbone that is switched off gets `stage2_<B>.json` (status, gate share, the cells file's SHA-256)
but no `pins_<B>.json` (`run_r9_stage2.py:516`, `:553`). Rule §3.4 says the picks and the gate share go to the pins
file. This matches ticket 13 ("the driver stops"). Fix: write a short pins file for that case, or change the rule's
wording. It matters only if a backbone is switched off.

**L-5 LOW (test gaps; the gate still holds).** If I delete the dtype or shape term of `array_item`
(`run_r9_stage0a.py:271`), the change survives, but `compare_items` still catches it through the "got" description
field. If I delete the type check of `deep_diff` (`run_r9_stage0b.py:188`), the change survives too. Fix: add a
direct assertion for each.

Leads: lead 1 = L-1 (confirmed). Lead 2 = L-3. Lead 3 holds: `pins_evidence` puts `sensitivity_seed42` into the pins
as `{path, sha256}`; in the dry run the SHA-256 equals the file's.

User decisions applied: the graph's GPU wording is in rule §3.2. `graph_gpu` stops a CPU build, and a test
catches its deletion. CLIP's replay graph is byte-identical to the pinned one. Qwen's live `graph.npz.record.json`
shows cuda on the RTX 3090 with `cpu_allowed: false`. The `picked.get("smoke", False)` fix is at
`r9_bundle.py:174` and `r9_recipe_baselines.py:267`, and nothing else reads `picked["smoke"]`. Qwen's `picked.json`
has no "smoke" key, and its chain got past that point.

## 4. Hard rule 6 gaps (SC1, SC2 in my scope)

| Gap | Kind | Where |
|---|---|---|
| C13 phase 1 cannot consume the real pins (M-1); 7 of its rule fields untested (M-2) | partial (blocking for H6, H7) | rederive/ |
| Switched-off backbone: no pins file, while the rule says the picks and the gate share go there (L-4) | partial | run_r9_stage2.py |
| "RCA and PM fits" not compared as arrays in 0b; covered through the per-anchor arrays (L-2) | partial | run_r9_stage0b.py, r9_recipe_baselines.py |
| `sensitivity_seed42` pinned by SHA-256 in pins (not asked by §3.4; harmless) | unrequested | run_r9_stage2.py |
| none found | contradicts | |

No completion claim was taken as evidence. Every "holds" in section 2 comes from my own scripts' JSON outputs.

## 5. Guards whose deletion survives

`fr_b2_mutate.py` makes each edit in place in the snapshot, runs the named tests and restores the bytes (SHA-256
checked). Results are in `fr_b2_mutate.json`: 99 mutants. All results are final except `0b_routed`, a bad mutant
that I replaced with `0b_routed2` (caught). One batch was killed by a timeout and left `run_r9_stage2.py` mutated. I
restored it with `git checkout`, checked its SHA-256 against stage 0b's record, and reran the affected mutants.

Caught: all of stage 0a except `array_item`'s dtype and shape terms; all of stage 0b except the `deep_diff` type check;
the stage 2 guards not listed below; `gate_share` (parity, tau index, condition b) and `switched_off` (any, index);
eight r9_data join and split asserts; `graph_gpu`. The copy-based mutation tests of r9_heads and r9_picks pass (21
tests) and enumerate every marker.

Survive (real gaps): the 14 guards of M-3, and in the phase-1 compare `c_sigma`, `c_cf`, `c_aff`, `c_b0`, `c_rca`,
`c_taus`, `c_points`, `c_tau_tol` and `c_picks_exact` (M-2). Survive but are covered elsewhere: `0a_array_item_dtype`
and `0a_array_item_shape` (field compare), `0b_deep_type` (L-5), `0b_routed_missing` (KeyError on the next line),
`2_prov_overlap`, `2_frozen_pm_rows` (file SHA bound first), `2_script_0b_passed` (`decide` requires item 4),
`2_feature_store` (`features == stores`), `d_dup_painting` (the image-join assert).

## Left (not done before the context limit)

- `pins_qwen3vl_emb_2b.json` was not written yet at 01:43 (Qwen's chain was in the baselines step). Its read-only check
  is left: provenance, switched-off, gate share, the pinned SHA-256s.
- The copy-based guard tests of `r9_recipe_reader` and `r9_recipe_baselines` were not rerun. Their markers are named
  in the tests.
- The phase-1 compare on real pins after the M-1 fix (controller: rerun `fr_b2_phase1_probe.py`).
