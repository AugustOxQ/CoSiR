# fx-rr hand-back (branch r9-fx-rr)

F = `src/test/20261202_r9_scope`. N5 kept: no closure module and not `run_r9_stage2.py` changed.

| # | Item | Status | What changed | Tests |
|---|---|---|---|---|
| 1 | RR2 M1 | done | `test_r9_a28_stats.py` `load_record`: adds `read` and `backbone` (`AR.READ_BACKBONE_IDS`) to the record | the file: 51 pass; with the old `load_record` 14 fail (checked) |
| 2 | RR1 M1 | done | tests only: synthetic `held_arrays.npz` (3 smoke seeds, 64 per pair) and a pass file from `ST.pass_record`, in `test_r9_descriptive.py` (end of file) | `test_h5_arrays_are_accepted_as_built`, `test_h5_load_core_refuses_a_missing_key`, `..._an_extra_key`, `..._a_bad_array[8 cases: wrong dtype, NaN, short array, seed order, pair order]`, `..._other_seeds_or_pair_size`, `test_h5_check_pass_refuses_a_changed_count[2]`, `..._each_changed_item[12: n, point, ci95, quantity on P1, P6, S1]`, `..._changed_arrays`, `test_h5_inputs_binds_smoke_pass_arrays_and_episodes` (28 in all). 15 mutants of the H5 copy, all caught |
| 3 | RR1 N1 | done | `rederive/rd9_compare.py`: `compare_stats(..., compare_margins=True)`; in main, `--mode selection` (derived `mode == 'selection'`) passes False and the agreement gets `margins_not_compared` (text names the reason). Held and smoke modes unchanged | `rd9_test_compare.py`: `test_selection_replay_does_not_compare_the_margins_and_says_so[False/True]`, `test_held_modes_still_compare_the_margins[held/smoke]`; `test_margins_without_a_sensitivity_file_disagree` now also asserts no note. 4 mutants caught |
| 4 | RR1 N2 | done | `rederive/rd9_commands.md`: `--reported-sensitivity` in the per-read block; paragraph reworded; note on the selection replay | doc |
| 5 | RR1 N3 | done | `run_r9_apply_rule.py` (after `load_sensitivity`, guard `agreement-sensitivity`): `agr["sensitivity_sha256"]` must equal the SHA of the file the step reads (`spath`; for a `_fix1` pass without its own file, the original). Also needed: the smoke stand-in agreement in `run_r9_preread.py` `smoke_agreement` now writes `sensitivity_sha256`. Helper `write_agreement` in `test_r9_apply_rule.py` writes it | `test_agreement_is_bound_to_the_sensitivity_file_bytes`, `test_agreement_without_a_sensitivity_sha_is_refused`, `test_fix_pass_agreement_is_bound_to_the_sensitivity_file_it_uses`; `test_smoke_agreement_writes_only_into_the_smoke_folder` extended. 3 mutants caught |
| 6 | RR2 LOW-2 | done | comment in `run_r9_preread.py` `smoke_agreement`; docstring of `run_r9_apply_rule.py` lines 9-10 now gives `heldfix1` first (and the AGREEMENT field list names `sensitivity_sha256`) | none |
| 7 | RR2 LOW-5 | done | `test_r9_stage2.py::test_a_key_in_both_made_and_copied_fails` (equal arrays, so only the overlap guard can stop it). Driver untouched | mutant (guard line removed) caught |
| 8 | controller | done | `test_r9_a28.py`: subprocess env gets `HOME` | passes in the worktree and from the main checkout's files (below) |
| 9 | controller | done | `test_r9_recipe_groups.py::test_heads_step_fits_on_the_new_partitions_file_of_a_non_default_out`: dropped `assert not default.exists()`; asserts the pinned path is `out/partitions.npz` and not the default path (its SHA assert was already there) | passes in the worktree and from the main checkout's files (below) |

Mutation script: `fx_rr_mutants.py` (about 2 minutes), results `fx_rr_mutation_results.json`: 23 of 23 caught, every file restored.

## Tests

Command (from F, python of the CoSiR env, `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=`):
`pytest -q -p no:cacheprovider test_r9_a28_stats.py test_r9_descriptive.py test_r9_apply_rule.py test_r9_stage2.py test_r9_preread.py test_r9_a28.py rederive/rd9_test_*.py test_r9_recipe_groups.py::<the item-9 test>`
-> 767 passed, 1 skipped (the real-file test that needs the integration checkout).
Also `test_r9_recipe_groups.py test_r9_pins.py test_r9_held.py test_r9_held_shared.py`: 264 passed.
Items 8 and 9 from the main checkout, read-only: I built a scratch folder of symlinks to the main round folder (so `r9_common.RESULTS`
is the main checkout's real `results/`, where Qwen's `partitions.npz` exists) with only the two changed test files copied in:
`test_r9_a28.py` and the item-9 test, 26 passed. Nothing in `/project/CoSiR` was written.

## Choices the ticket did not fix

- **N3 is strict:** an agreement without `sensitivity_sha256` is refused (the per-read command always passes the sensitivity
  file now). Consequence: any agreement file made before fx-c13 (or a smoke stand-in made by the old preread bytes) is refused
  by the apply step; redo them. The smoke records must be redone on final bytes anyway (`run_r9_preread.py` changed).
- **N1:** the margins are skipped by the derived record's `mode == 'selection'`, not by a flag.
- **Not done (not in the ticket):** RR1 N3's second part (require margins for every non-selection read) and RR1 N4.
- **Would a rerun of the H8 selection replay now agree?** Expected yes, with 136 items (the committed file has 129; the seven
  quantity labels are new and R6's `seed42_pass_counts.json` carries them; the margins are skipped). I could not run it:
  there is no `rd9_out` derived record in the main checkout. `rd9_H8_selection_agreement.json` is not rewritten.
- RR2 LOW-1 (stale `smoke_record_heldfix1.json`), LOW-3, LOW-4: not in the ticket, untouched.

## Things I would remove (nothing deleted)

Scratch only, in the session scratchpad: `mainview2` (symlinks to the main round folder plus two copied test files), `a28stats.bak`, `py.sh`.
