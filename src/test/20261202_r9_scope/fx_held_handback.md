# fx-held hand-back

Worktree `/project/CoSiR-wt/r9-fx-held`, branch `r9-fx-held`. F = `src/test/20261202_r9_scope`. Commit: the head of branch `r9-fx-held` (hash in the final message).
No closure module, no `run_r9_stage2.py` and no file under `rederive/` changed. Nothing deleted.

## Items

| Item | Done | Change | Tests (all in F) |
|---|---|---|---|
| C MEDIUM-1 | yes | tests only. All 13 tests of `final_review/fr_c_gate_test.py` are in `test_r9_preread.py` as `test_gate_*` (they call the real `run_held`). The five missing sub-conditions are also cases of `test_launch_stop`. The reviewer's file is untouched and still passes. | `test_launch_stop[launch_modules\|launch_read\|launch_backbone\|launch_check_not_run\|launch_no_inputs]`, `test_gate_*` |
| C MEDIUM-2 | yes | tests only | `test_input_stop[stage2_switched_off\|stage2_dry_run\|stage2_backbone]`; `test_regression_stop_one_field_wrong[<6 fields>]`; `test_regression_all_fields_equal_passes`; `test_clip_regression_stop_one_half_wrong[items_unequal\|items_short\|counts]`; `test_phase1_stop[phase1_phase\|phase1_backbone\|phase1_failed\|phase1_item\|phase1_no_items\|phase1_compared]` |
| C LOW-1 | yes | `run_r9_held.py:964`: under `__main__`, `sys.modules.setdefault('run_r9_held', sys.modules[__name__])` before `main()` | `test_script_run_of_held_runner_exits_4_on_a_refused_gate` (runs the real `__main__` block in its own namespace) |
| C LOW-2 | yes, with a design choice (below) | a held `--fix 1` now has its own pre-read name `preread_<Hn>_heldfix1.json` and its own smoke record `smoke_record_heldfix1.json`. `run_r9_preread.py:35,423,463` (kind `heldfix1`, flag `--held-fix 1`); `run_r9_held.py:355,498-499`; `run_r9_apply_rule.py:114,551` | `test_held_fix_launches_on_its_own_preread_and_smoke`, `test_held_fix_preread_is_free_when_fix1_authorised_the_first_launch`, `test_held_fix_smoke_record_takes_precedence_in_both_latest_functions`, `test_cli_held_fix_flag_runs_the_heldfix1_kind`, `test_gate_fix_needs_held_fix_preread` |
| C LOW-3 | yes | `run_r9_preread.py:268-280`: the smoke stand-in is written only to the smoke output folder (the apply step and the descriptive pass read it there). `:360`: the F copy of the smoke record only when `results` is `R.RESULTS`. | `test_smoke_agreement_writes_only_into_the_smoke_folder`, `test_scratch_smoke_record_has_no_committed_copy` |
| C LOW-4 | yes | `run_r9_preread.py:441-445`: at launch the record's `input_file_sha256` must hold this checkout's pins file, `R.RESULTS/stage0a_regression.json` and (A28 reads) `pins_a28.json`, resolved | `test_launch_stop[launch_other_worktree_inputs\|launch_no_pins\|launch_no_stage0a\|launch_no_a28_pins]`, `test_launch_a28_pins_not_needed_for_a_held_read` |
| A1 L1 | yes | `run_r9_apply_rule.py:699` prints `re-derivation's count N` beside the runner's | `test_main_prints_the_rederivation_count_beside_the_runners` |
| A1 L2 | yes | `run_r9_apply_rule.py:249`: `quantity == ST.QUANTITIES[name][2]` | `test_pass_quantity_label_is_checked` |
| A1 L3 | yes | `run_r9_held.py` already wrote `read` and `backbone` (`pr.update`, `run_read`), so no change there. `run_r9_apply_rule.py:107,266`: `load_pass` checks both (`READ_BACKBONE_IDS`). | `test_pass_file_must_name_its_read_and_backbone`, `test_pass_file_of_another_read_with_the_same_backbone_is_refused`, `test_run_read_order_and_every_score_and_pass_argument` (new asserts: fields written; `READ_BACKBONE_IDS == READ_BACKBONES`) |
| A1 `assert_rule` | yes | tests only | `test_assert_rule_stops_a_read[rule_sha\|r6_sha\|phrase\|r6_phrase]`, `test_assert_rule_stops_a_not_run_verdict[...]` (the read path :636 and the not-run path :633 each) |

`rederive/` and L3: `rd9_compare.py` reads named keys of the reported pass file (`mode`, `stats`, `checks`, `set_aside`, ...) and does not check the key set. The new `read` and `backbone` keys cannot break it, so I did not touch it.

## Choice the ticket did not fix (C LOW-2)
A new pre-read name alone does not free the held `--fix 1`: `smoke_record_fix1.json` has the same clash (`publish_smoke` refuses "canonical smoke stale", and the apply step and `refuse_or_go` read that name). So I gave the held fix its own smoke record name too, and put `heldfix1` first in both `latest_smoke_record` functions (apply step and held runner). Effects: (1) held `--fix 1` needs `smoke_record_heldfix1.json` (was `smoke_record_fix1.json`); (2) the pre-read for it is run with `--held-fix 1`; `--fix 1` stays the pre-read re-run that can authorise a first launch. Tests that put `smoke_record_fix1.json` for a held fix now put the new name (`test_r9_held.py`, `test_r9_held_a28.py`). The rule text names neither file. If the user prefers the smaller change (pre-read name only), revert the `SMOKE_RECORDS`/`latest_smoke_record` lines.

## Other changes to existing tests
`test_r9_apply_rule.py`: `QUANT` now uses the real labels of `ST.QUANTITIES`; `make_pass` writes `read` and `backbone` (needed by L2 and L3; `test_r9_descriptive.py` builds on it). `launch_record` in `test_r9_preread.py` lists the pins file, stage 0a record and A28 pins (LOW-4); three tests that build a record by hand got the same.

## Tests
Command (each under 40 s; from F): `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python -m pytest -q test_r9_preread.py test_r9_held.py test_r9_held_a28.py test_r9_held_shared.py test_r9_apply_rule.py test_r9_descriptive.py final_review/fr_c_gate_test.py`
Result: 781 passed (preread 111, apply rule 152, held 180). `test_r9_common.py`, `test_r9_review_fixes.py`, `test_r9_stage0b.py`, `test_r9_stage2.py`: pass. `test_r9_pins.py::test_no_r9_module_imports_an_r6_module` fails with and without my change (`run_r9_descriptive.py` imports `r6_common` and `r6_descriptive`); it is not mine and I left it.

## Mutations
`F/fx_held_mutants.py`, results `F/fx_held_mutation_results.json`: 44 hand-made mutants (the 5 launch parts, 4 LOW-4 forms, stage 2 x3, regression fields x3, CLIP halves x3, phase 1 x6, LOW-1, LOW-2 x7, LOW-3 x2, L1 to L3, held pass fields, `assert_rule` x6). 43 caught; each source file restored and SHA-checked after every mutant; `git diff` shows only the intended edits.
Survivor: "launch non-empty inputs" (`bool(rec.get('input_file_sha256'))`). The new LOW-4 check refuses an empty `input_file_sha256` too, so the old clause is now redundant (equivalent mutant). I kept it as defence in depth.

## For the controller
- The r9 module bytes changed (`run_r9_held.py`, `run_r9_preread.py`, `run_r9_apply_rule.py`). Stage 0a, every smoke record and every pre-read record must be made on the final bytes (as planned for the merge).
- The t22 controller command with a scratch `--results` no longer writes `F/smoke_record.json` or anything under `rederive/rd9_out/smoke/`.
- Files I would remove: none.
