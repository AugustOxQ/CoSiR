# Re-review of the fix wave, part 2: fx-held and fx-tests

Reviewer rr_fixwave_2, 2026-10-14 (Amsterdam). Snapshot `/project/CoSiR-wt/r9-rr2`, detached at 1e6e55f (r9-scope with both
tickets merged). F = `src/test/20261202_r9_scope`. Code read-only; mutations ran in scratch copies of the checkout under the
session scratchpad (`wk`, `wk2`, `wk3`), never in the snapshot.

## 1. Verdict

**PASS WITH NOTES.** Every item of both tickets is met and its new test fails when the fix is undone. No HIGH finding.
One MEDIUM: the L3 fix of fx-held breaks 14 tests of `test_r9_a28_stats.py` on the merged branch (test-side fallout, one-line
fix). Five LOW notes follow in section 5.

| Check | Result |
|---|---|
| N5: closure modules and `run_r9_stage2.py` | empty diff, `e79d4a1..HEAD`, and for each branch against its own base |
| fx-held: its 7 test files + `fr_c_gate_test.py` | 782 passed (snapshot) |
| fx-tests: stage 2, 0a, 0b, A28 files, new prepare file | 471 passed, 14 failed (all in `test_r9_a28_stats.py`, see M1) |
| Rest of F (`rederive/`, review_fixes, common, pins) | 192 passed, 1 failed: `test_r9_pins.py::test_no_r9_module_imports_an_r6_module` (B2 M-4, still open, not in these tickets) |
| Whole-folder collection | 1922 tests, no error (B1 LOW-5 held up) |
| Hand-back mutants of fx-held | I reran all 44 in a scratch copy: 43 caught, 1 survives (the same one) |
| Mutants of fx-tests | see section 4 |

## 2. fx-held (base e79d4a1)

| Item | Verdict | Where, and how I checked |
|---|---|---|
| C MEDIUM-1, launch-gate sub-conditions | met | `test_launch_stop[...]`, `test_gate_*`. Mutants 0 to 3 (modules half, read, backbone, all-checks-passed) caught. |
| C MEDIUM-2, stage 2 / regression / CLIP / phase 1 stops | met | Mutants 9 to 21 all caught (stage 2 x3, regression fields x3, CLIP halves x3, phase 1 x6). |
| C LOW-1, script run of the held runner | met | `run_r9_held.py:968`; mutant "no setdefault" caught by the `__main__` test. |
| C LOW-2, held `--fix 1` names | met | `run_r9_preread.py:35,423,463`, `run_r9_held.py:356,499-500`, `run_r9_apply_rule.py:114,551`. 7 mutants caught. Choice (a) below. |
| C LOW-3, smoke stand-in only in the smoke folder | met | `run_r9_preread.py:268-280`, `:360`. 2 mutants caught. Choice (b) below. |
| C LOW-4, launch bound to this checkout's files | met | `run_r9_preread.py:441-445`. 4 form mutants caught. Choice (c) below. |
| A1 L1, re-derivation count printed | met | `run_r9_apply_rule.py:699`; mutant caught. |
| A1 L2, `quantity` checked | met | `:249`; mutant caught. |
| A1 L3, pass file names read and backbone | met in the code; **breaks other tests (M1)** | `:107,266-267`; the held runner already wrote both fields (`pr.update`). |
| A1 `assert_rule` untested | met | 6 mutants (rule SHA, R6 SHA, two phrase lists, call in both paths) caught. |

### Choices of the hand-back

**(a) heldfix1 names: sound.** I built every subset of the five smoke-record files (smoke, crash1, fix1, heldfix1, reserve) in
a temp folder and called both `latest_smoke_record` functions for a first launch / after-crash (`held`), a held `--fix 1`, and
a reserve. Result: 0 mismatches. The runner and the apply step always agree; a held `--fix 1` always names
`smoke_record_heldfix1.json`; a reserve always names its own file; for a first launch `heldfix1` outranks `fix1`, `crash1`
and the base record only when it exists, and it cannot exist before a read has completed. `preread_guard` for a first launch
looks at `_fix1` then `_crash1` then the base name and never at `_heldfix1`; for a held fix it looks only at
`_heldfix1`. So a `--fix 1` pre-read (first-launch repair) cannot authorise a held fix, and the other way round. One residual
design limit is LOW-1 below.

**(b) smoke stand-in: sound, with a wrong comment.** In smoke mode the apply step reads `out / rd9_<read>_agreement.json`
(`run_r9_apply_rule.py:639`, `out if smoke`). Real mode still reads `AR.AGREEMENT_DIR` and refuses a smoke agreement
(unchanged). The descriptive pass in smoke mode reads no agreement at all (`run_r9_descriptive.py:bind_read`, `smoke` branch
sets no `agr`); in real mode it tries `out/` then `AGREEMENT_DIR` and needs the verdict's SHA (unchanged). So the comment at
`run_r9_preread.py:277` ("the apply step and the descriptive pass read it from there") is right only for the apply step (LOW-2).
No code reads the removed `source` key.

**(c) LOW-4 paths: sound.** The keys of `input_file_sha256` are built as `str(Path(p).resolve())` in `checked_json` and
`inputs_gate`: the pins file by `checked_json(R.pins_path(backbone), ...)`, the stage 0a record by
`checked_json(R.RESULTS / S0A.OUT_NAME, ...)`, the A28 pins by `checked_json(RH.AD.PINS_PATH, ...)` (in `inputs_gate` for A28
reads and in `smoke_gate` for every read). The launch check builds the same three expressions and resolves them the same way.
`evidence` is the very dict stored as `rec['input_file_sha256']`. So a production pre-read in the main checkout lists
exactly those paths. The main checkout has no `stage0a_regression.json` yet, so I could not run one; this is a static reading,
plus `test_launch_*` and `test_launch_a28_pins_not_needed_for_a_held_read`.

**(d) second `READ_BACKBONE_IDS`: sound.** The apply step cannot import the held runner (the runner imports it). The five
entries equal `RH.READ_BACKBONES`; `test_r9_held.py:406` compares the two dicts.

**(e) surviving mutant: sound, equivalent.** `own` (pins, stage 0a record) is never empty, so an empty
`input_file_sha256` already fails the new `all(... in rec['input_file_sha256'])` line. Only the message differs.

## 3. fx-tests (base 8db7ceb)

| Item | Verdict | Evidence |
|---|---|---|
| B2 M-3, 14 stage 2 guards | met | 14 tests (7 provenance cases, feature-names, 9 `pins_evidence` cases). Mutants: see section 4. |
| 13 fix 2 L1, replay skip narrowness | met | `test_replay_skip_is_only_the_roles_own_file_hash`; mutant `2_replay_skip_widened` caught. |
| B2 L-5, `array_item` dtype and shape, `deep_diff` type | met | mutants `0a_array_item_dtype`, `0a_array_item_shape`, `0b_deep_type` caught. |
| B2 open item, guard tests of reader and baselines | met | reader unchanged (marker test already there). Baselines: 12 copy-based tests plus the new marker test pass in the snapshot. Choice (c) below. |
| B1 MEDIUM-1, 7 survivors | met | the seven mutants caught (section 4). |
| B1 LOW-3, three join/label guards | met | 3 mutants caught. |
| B1 LOW-5, worktree collection | met | I reverted the import order in a scratch copy: the new test fails. The folder collects 1922 tests in the snapshot. |

### Choices of the hand-back

**(a) `r9_backbone` off the forbidden list: sound.** `r9_backbone.py` imports only `os`, `pathlib`, `subprocess`, `sys`; it
holds names and paths, no feature or score. The rest of the test stays: no `r9_common`, `r9_data`, `r9_score`, `r9_heads`,
`feature`, `torch`, `pre_extract`, and no `np.load` or `.npy/.npz` in the source. The new test also pins the order
(`import r9_backbone` before the first `from src.`). `pins_a28.json` does not list `r9_a28.py`; the closure is unchanged.

**(b) `len` shadow: each test still exercises the real check.** `monkeypatch.setattr(P, "len", ...)` changes `len` only for
the two objects (`csv`, `trans`) by `id`; every other call falls through to the builtin. I mutated the runner in a scratch
copy:

| Mutant of `run_r9_a28_prepare.py` | Result |
|---|---|
| train side = train only | caught (2 tests) |
| train side = train+val, plus held | caught |
| counts assert removed | caught (`..._count_differs_from_the_rules`) |
| V values assert removed | caught (`..._V_has_the_right_sizes_but_other_values`) |
| V sizes and values asserts both removed | caught |
| V sizes assert alone removed | survives; equivalent, the values assert implies it |
| partition assert (held + train side = all) removed | survives; not in the ticket, `train side = train only` is caught by its test instead |
| row-count assert (`len(csv) == 173_745 ...`) removed | survives, by construction: the shadow disables it (LOW-3) |

**(c) baselines marker test, "by construction": acceptable, weak.** The test pins the seven marker names, requires each to
appear as a quoted string in the tests above it, and loads a copy of the module with each marker's statement replaced by
`pass` (`mutant` asserts the marker sits on a statement). I checked by simulation that removing the one use of any marker
name from the test file makes it fail (every name occurs 1 to 3 times as a quoted token). It counts quotes, not assertions:
a name in an unrelated string would satisfy it (LOW-4). The reader file's version matches `mutant(...)` calls by regex and is
stronger, but the baselines file uses parametrised names, so that form does not fit.

**(d) four stage 2 guards still surviving deletion.** Three sit behind a later check; one does not.

| Guard | Later check | Judgment |
|---|---|---|
| `frozen_pm_rows` (`run_r9_stage2.py:108`) | `assert_frozen` :97 binds the file's SHA-256 to `pins_clip.json`, and `pins_clip.json` equals CLIP's file (B2 re-hash); equal bytes give equal rows. Also :107 `fit_rows_sha256`. | covered |
| `script_0b_passed` (:116) | :114 `stage0b.passed`; `decide` (`run_r9_stage0b.py:379`) cannot give `passed` without `item4`; :120-121 script SHA | covered |
| `feature_store` (:340) | :372 `features == stores`: a wrong hash would be observed and then fail the equality | covered (its mutant `2_features_complete` is caught) |
| `prov_overlap` (:75) | none that catches the same fault. `made` and `copied` come from the producer's record (`keys_made`, `keys_copied`, :193), not from a constant. A key in both lists is treated as copied (:80), so "made differs from CLIP" is never asked of it. | **not covered, LOW-5** |

## 4. Mutation reruns (scratch copies)

fx-held: 44 mutants of the hand-back's own script, all rerun by me. 43 caught, 1 survived ("launch non-empty inputs",
equivalent). Each source file was compared with the snapshot after the run: restored.

fx-tests: 29 mutants of `fx_tests_mutate.py`, paths pointed at the scratch copies.

All 29 caught: the 14 stage 2 guards (`2_record_backbone`, `2_record_file_sha`, `2_wrong_producer`, `2_upstream`,
`2_prov_keys_complete`, `2_reader_feature_names`, `2_ev_stage0b_changed`, `2_ev_closure`, `2_ev_receipt`,
`2_ev_record_changed`, `2_ev_store_changed`, `2_ev_frozen_changed`, `2_ev_frozen_assert`, `2_ev_seed42_changed`), the replay
skip (`2_replay_skip_widened`), `0a_array_item_dtype`, `0a_array_item_shape`, `0b_deep_type`, and the 11 B1 mutants
(`b1_min_paintings`, `b1_groups_rows`, `b1_expected_rows`, `b1_expected_v_style`, `b1_expected_v_sizes`, `b1_train_side_val`,
`b1_prepare_counts`, `b1_prepare_v`, `b1_translation_painting`, `b1_one_label`, `b1_one_image`). Source files compared with the
snapshot after the runs: all restored. My own B1 prepare-runner mutants are in section 3b.

## 5. New findings

**M1 (MEDIUM). The L3 fix breaks 14 tests of `test_r9_a28_stats.py` on the merged branch.** `load_pass` now refuses a pass file
without `read` and `backbone` (`run_r9_apply_rule.py:266`). `test_r9_a28_stats.py::record()` (:96-111) builds its pass records
with `pass_record_a28` and no `read` or `backbone`, so `load_record` (:112-114) stops with "its read and backbone are not H8 and
clip_b32". With the check off the file passes (51 of 51; scratch). Failing: `test_a28_record_at_each_levels_boundaries`,
`test_real_bootstrap_no_secondary_and_tie_orders[False|True]`, `test_round_trip_to_the_real_decision[...]` (x9: three
decisions times H8, H9, H10), `test_seven_check_record_still_round_trips[H6|H7]` (x2): 14 in all. The hand-back's test list left this file out.
The code is right; the production pass files carry both fields. Fix (test only): add `rec["read"] = read` and
`rec["backbone"] = AR.READ_BACKBONE_IDS[read]` in `load_record`. Do this before the final test run of the build.

**LOW-1. A stale `smoke_record_heldfix1.json` shadows newer records and cannot be retired.** Precedence is `heldfix1`, `fix1`,
`crash1`, base, and `publish_smoke` refuses to replace a record whose module bytes differ ("canonical smoke stale"). After a
held `--fix 1` on one read, a later code change leaves every other read's first-launch pre-read unable to publish a current
record (`smoke_guard` picks the stale `heldfix1`). Only `--reserve` has its own name. The same limit already held for `fix1`
and `crash1` (R6 design); this adds one name. Reasoned from the code, not reproduced. Worth one line in the run order: no
code change after the first held `--fix 1`.

**LOW-2. Two wording slips.** `run_r9_preread.py:277` says the descriptive pass reads the smoke stand-in; it does not
(section 2b). The top docstring of `run_r9_apply_rule.py:9-10` still gives the smoke-record order without `heldfix1`.

**LOW-3. The prepare test's `len` shadow leaves the real-file row-count assert (`run_r9_a28_prepare.py:48`) unexercised.** That
assert (173,745 and 167,230) survives deletion. It was untested before as well. A small test with an unshadowed short CSV
would cover it.

**LOW-4. The baselines marker test counts quoted names, not assertions** (section 3c). It still detects a marker that loses
all its tests.

**LOW-5. `prov_overlap` (`run_r9_stage2.py:75`) has no test and no later check** (section 3d). Test only: call
`assert_array_provenance` with one key in both lists and expect "made/copied keys overlap". Do not edit `run_r9_stage2.py`.

## 6. B2's open read-only item: `pins_qwen3vl_emb_2b.json` against Qwen's stage 2 record

**Holds. Nothing differs.** All reads were in the main checkout, read-only (`/project/CoSiR/src/test/20261202_r9_scope/`).

| Check | Result |
|---|---|
| SHA-256 of `pins_qwen3vl_emb_2b.json` | `9a0fae5ed1714d4194094316ce7af2db5305a8a6f3c4f643d282ebbe5e77acd8` (same file in the main checkout and the snapshot) |
| `r9_pins.build_pins('qwen3vl_emb_2b') == ` the file | equal (run in the main checkout, 1 call, CPU) |
| Every `{path, sha256}` entry in the pins (118) | all files exist, all SHA-256 equal |
| Parts | 29 parts present, same role set as `BB.parts(...)`, each SHA equals the pinned one |
| Inputs | `R.input_pins('qwen3vl_emb_2b')`: 52 inputs, 0 changed, 0 missing |
| Stage 2 record `stage2_qwen3vl_emb_2b.json` | `passed` true, `dry_run` false, `status` "ready"; its SHA equals `pins['stage2'].sha256`; `rule_sha256` equals the live rule SHA; `runner_sha256` equals `run_r9_stage2.py`; the stage 0b record it names has the SHA it lists |
| Closure | the stage 2 record's 17 module SHAs equal the pins' 17, and equal the live files (no closure module changed since) |
| Seed-42 | the pins' `seed42` block equals `stage2_seed42_...json` field for field (8 fields); its SHA is pinned |
| Picks | `pins.picks` B, B0, B1, mean R@1 and the lambda picks equal `recipe/qwen3vl_emb_2b/picks_seed42.json` (B 18.2495, B0 18.4123, B1 19.2485) |
| Gate references | the three SHAs equal `pins_clip.json`'s |
| Switched off (rule §3, "Switched off") | `false`: AFF cells 47 and 103 give `F2.cell_values` = (13, 0, 8.0, 16.0) and (13, 1, 8.0, 16.0); the fourth value (lambda_a) is 16.0, not 0, in both halves |
| Gate share (rule §3, "recorded in the pins file") | 0.6134033203125 = 15,075 / 24,576; the same number in the cells file, the stage 2 record, `pins.stage2` and `pins.picks.cells` |

Not done: I did not recompute the 15,075 open gates from the Qwen features; that needs the seed-42 scoring run (GPU).
The arithmetic and the four copies agree, and `build_pins` (which the pre-read also runs) reproduces the file.

## 7. For the controller

- Fix M1 (test only) before the build's final run.
- Long run (not done here, about 7 minutes): the whole baselines file,
  `cd /project/CoSiR-wt/r9-rr2 && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider src/test/20261202_r9_scope/test_r9_recipe_baselines.py`
  (the fx-tests author ran it once: 30 passed).
- Scratch copies to remove later (nothing deleted): `wk`, `wk2`, `wk3` and `m/` under
  `/tmp/claude-0/-project-CoSiR/06992efa-c388-41ec-bd52-4b5c99c57e84/scratchpad/`.
