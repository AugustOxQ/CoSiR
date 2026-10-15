# Hand-back: fix ticket fx-tests (r9 build 10 fix wave)

2026-10-14 03:3x (Amsterdam). Worktree `/project/CoSiR-wt/r9-fx-tests`, branch `r9-fx-tests`. F = `src/test/20261202_r9_scope`.
The 17 closure modules and `run_r9_stage2.py` are unchanged (`git diff` shows none). Only non-test edit: `F/r9_a28.py`
(item 7). Mutation runner: `F/final_review/fx_tests_mutate.py` (adapted from `fr_b2_mutate.py` and
`fr_b1_mutations.py`), results beside it in `fx_tests_mutate.json`. Every mutant below was made in place, the test file
run (`pytest -x`), the bytes restored and checked; all new tests fail on their mutant.

## Per item

**1. B2 M-3: done.** `test_r9_stage2.py` (appended block "fix wave fx-tests"):
- `test_provenance_record_guards_refuse[...]` (7 cases, uses the `world` fixture and `D.provenance`):
  `other_backbone` (:360), `file_sha` (:362), `wrong_producer` (:364), `upstream_hash` and `upstream_unknown_role`
  (:370), `key_missing_from_record` and `key_missing_from_file` (:200). Each matches its own message, so a deleted
  guard cannot hide behind a later one.
- `test_provenance_accepts_the_correct_upstream_hash` (positive control for :370).
- `test_reader_json_feature_names_must_equal_clips` (:226, `assert_part` on two files).
- `test_pins_evidence_stage_snapshot_guards_refuse[...]` (9 cases, `D.run_pins`): `stage0b_changed` (:558),
  `closure_changed` (:561), `receipt_changed` (:566), `record_changed` (:573), `store_changed` (:575), `frozen_changed`
  (:578), `frozen_reassert` (:579; the stage snapshot is refreshed so only the re-assert can notice), `seed42_changed`
  and `sensitivity_changed` (:583). Each also asserts no pins file was written.
- Mutants (ids in `fx_tests_mutate.json`): `2_record_backbone`, `2_record_file_sha`, `2_wrong_producer`, `2_upstream`,
  `2_prov_keys_complete`, `2_reader_feature_names`, `2_ev_stage0b_changed`, `2_ev_closure`, `2_ev_receipt`,
  `2_ev_record_changed`, `2_ev_store_changed`, `2_ev_frozen_changed`, `2_ev_frozen_assert`, `2_ev_seed42_changed`: all 14
  caught (each by the test named for it). Both `seed42_record` and `sensitivity_seed42` are tested; the mutant deletes
  the one line that guards both.
- Not in the ticket's 14 and still uncaught by design (covered by a later check, per B2): `2_prov_overlap`,
  `2_frozen_pm_rows`, `2_script_0b_passed`, `2_feature_store`. I did not add tests for them.

**2. 13 fix 2 L1: done.** `test_replay_skip_is_only_the_roles_own_file_hash` (`test_r9_stage2.py`): in the replay
world, `rc_tau` reaches the feature stores only through the `partitions` file hash; `provenance(replay=True)` must
keep that edge (stores found, nothing skipped for `rc_tau`), and `partitions` still skips its own hash. Mutant
`2_replay_skip_widened` (`value in by_hash`): caught.

**3. B2 L-5: done.**
- `test_r9_stage0a.py::test_array_item_equal_flag_needs_dtype_shape_and_values`: the `equal` flag of `array_item` for
  other dtype, other shape with the same values (reshape and ravel), and one changed value. Mutants `0a_array_item_dtype`
  and `0a_array_item_shape`: caught.
- `test_r9_stage0b.py::test_deep_diff_type_change_is_a_difference[...]` (6 cases: 1 vs 1.0, 1.0 vs 1, True vs 1, [1.0] vs
  [1], {"k":1} vs {"k":True}, 0 vs False), `test_deep_diff_array_against_list_is_a_type_difference`,
  `test_deep_diff_same_type_same_value_is_empty`. Mutant `0b_deep_type`: caught.

**4. B2 open item: done (my reading).** B2's note says the copy-based guard tests of `r9_recipe_reader` and
`r9_recipe_baselines` were not rerun and "their markers are named in the tests". I read that as: rerun them, and make
sure every marker is covered. Reader: it already had `test_every_marked_guard_has_a_mutation_test`; full file run:
93 passed, 1 skipped. Baselines had no such check for its 7 markers; I added
`test_every_marked_guard_has_a_mutation_test` in `test_r9_recipe_baselines.py` (the marker set is pinned in the test,
every marker name must appear in the tests above it, and a copy of the module without each marker must load). The 7
copy-based guard tests (`a3_inputs` x5 cases, `head_fit` x2, the pinned-picks pair) pass (12 selected tests, 34 s).
The new test is proved only by construction, not by a mutation: adding a marker means editing `r9_recipe_baselines.py`,
a closure module. If B2 meant something else, the controller should say so.

**5. B1 MEDIUM-1: done.** The 7 mutants are all caught now.
- `test_r9_a28.py`: `test_value_sets_default_cut_is_30_paintings_not_29_and_not_rows` (default cut: 30 paintings in,
  29 paintings over 30 rows out; also `A.MIN_PAINTINGS == 30`) kills the cut-29 and groups = rows mutants
  (`b1_min_paintings`, `b1_groups_rows`). `test_expected_counts_and_value_sets_are_the_rules` pins `EXPECTED`,
  `EXPECTED_V_SIZES`, `EXPECTED_V` to the rule's numbers and checks that the rule text carries them (`b1_expected_rows`,
  `b1_expected_v_style`, `b1_expected_v_sizes`).
- New file `test_r9_a28_prepare.py`: runs `run_r9_a28_prepare.main()` with stand-ins for the data, split, CSV,
  translations and labels (r9_a28's and the runner's own code is real). `..._hands_the_train_and_val_paintings_...`
  (:41, `b1_train_side_val`), `..._stops_when_a_count_differs_from_the_rules` (:63, `b1_prepare_counts`),
  `..._V_has_the_right_sizes_but_other_values` (:73, `b1_prepare_v`). To get past the runner's hard-coded row
  counts the fixture shadows `len` inside the runner module only (`monkeypatch`, `raising=False`).

**6. B1 LOW-3: done.** `test_r9_a28_data.py::test_two_images_for_one_painting_fail` (r9_a28_data.py:155,
`b1_one_image`); `test_r9_a28.py::test_translation_painting_must_equal_the_csv_painting` (:79,
`b1_translation_painting`); `test_a_painting_with_two_styles_or_genres_crashes[style|genre]` (:90, `b1_one_label`; the
message `more than one {aspect}` is matched so the genre case is not caught by the later genre check).

**7. B1 LOW-5: done.** `r9_a28.py` now puts `HERE` on `sys.path` and imports `r9_backbone` before the first `src`
import (lines 11-20). `r9_a28.py` is not in the closure and not in `pins_a28.json`. Side effect: the old test
`test_module_reads_no_feature_and_no_score` forbade `r9_backbone` in `r9_a28`'s imports; I removed `r9_backbone` from
its forbidden list (it is the import guard, not a feature reader). New test
`test_importing_r9_a28_first_loads_src_from_main_and_r9_common_still_imports` (order check plus a subprocess import in the
checkout root). Proved: with the old `r9_a28.py` the new test fails, and
`pytest test_r9_a28.py test_r9_a28_data.py` fails at collection with the ImportError. Note: at HEAD the
whole-folder `pytest --collect-only` already collected without errors in this worktree (1848 tests, the same number as
with the fix), so I could not reproduce B1's "25 errors" for the whole folder; the two-file run is the
reproducer. After the fix the whole folder collects.

## Tests and counts

All from `/project/CoSiR-wt/r9-fx-tests`, `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider <files>`:

| File | Result | Tests added |
|---|---|---|
| `test_r9_stage2.py` | 93 passed (was 74) | 19: 7 + 1 + 1 + 9 + 1 |
| `test_r9_stage0a.py` | 52 passed (was 51) | 1 |
| `test_r9_stage0b.py` | 113 passed (was 105) | 8 (6 + 2) |
| `test_r9_a28.py` | 25 passed | 6: 1 + 2 + 1 + 1 + 1 |
| `test_r9_a28_data.py` | 25 passed | 1 |
| `test_r9_a28_prepare.py` (new) | 3 passed | 3 |
| `test_r9_recipe_reader.py` | 93 passed, 1 skipped (rerun, unchanged) | 0 |
| `test_r9_recipe_baselines.py` | 12 selected passed (guard tests + the new enumeration test); full file: 30 passed in 6 min 56 s (one run, after my edit) | 1 |

Whole round folder: `pytest -q --collect-only src/test/20261202_r9_scope` collects 1848 tests, no error.

## Left for the controller / not decided here

- LONG RUN (done once by me, in the background): the whole `test_r9_recipe_baselines.py` takes about 7 minutes (its
  `held`/`readers` fixtures): 30 passed. Command to repeat it: `cd /project/CoSiR-wt/r9-fx-tests && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider src/test/20261202_r9_scope/test_r9_recipe_baselines.py`.
- Choices the ticket did not fix: item 4's reading (above); the `len` shadow in the prepare test; the pinned marker set
  in the baselines enumeration test; testing both `seed42_record` and `sensitivity_seed42` through one parametrised
  test.
- `test_r9_pins.py::test_no_r9_module_imports_an_r6_module` (B2 M-4) is not in this ticket and I did not run or touch it.
- Nothing deleted. Files I created that could go after the round: none besides the tests (the runner and its json are
  committed as evidence).

Commit: d2362ed (tests and fixes); a second commit corrects this line only.
