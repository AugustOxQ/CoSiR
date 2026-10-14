# fx-desc hand-back

Worktree `/project/CoSiR-wt/r9-fx-desc`, branch `r9-fx-desc`. F = `src/test/20261202_r9_scope`. Commit: head of branch
`r9-fx-desc` (hash in the final message). No closure module, no `run_r9_stage2.py`, no `rederive/` file changed. Nothing deleted.

## Items

| Item | Done | Change | Tests (all in `test_r9_descriptive.py`) |
|---|---|---|---|
| 1. A2 F1 MEDIUM | yes | `run_r9_descriptive.py` `bind_read`: the verdict's backbone is compared with `RH.LEDGER_NAMES[RH.READ_BACKBONES[read]]` (the pass keeps the code name). The tests build the verdict with `AR.apply` on a staged pass (`staged()`, from `test_r9_apply_rule.stage`). H9 and H10 bind H8 and each other in one results folder. | `test_bind_read_real_agreement_directory_and_missing[H6..H10]` (asserts the verdict holds `AR.READS[read]['backbone']`), `test_bind_read_h8_binding_of_the_two_other_a28_reads`, `test_bind_read_verdict_identity_field_by_field[6]` |
| 2. B2 M-4 | yes | `run_r9_descriptive.py`: `h5_load_core`, `h5_check_pass`, `H5_CORE`, `H5_EXTRA` (copies of `r6_descriptive.load_core` / `check_pass`, `r6_score.CORE_SCORERS`, `ARRAY_EXTRA`); `R.R6_RULE_SHA256` (already in r9_common) replaces `R6.RULE_SHA256`; no `sys.path` edit. `test_r9_pins.py::test_no_r9_module_imports_an_r6_module` passes; importing the module loads no `r6_*` module. | the pins test; equivalence by script (below) |
| 3. A2 F2 LOW | yes | tests only; the three probe cases moved (the `staged` fixture now builds the verdict through `AR.apply`) | `test_bind_read_refuses_another_reads_verdict`, `test_bind_read_refuses_a_not_run_verdict`, `test_bind_read_refuses_a_pass_changed_after_the_verdict[H6\|H8]`; `no_arrays` makes any array or score load fail |
| 4. A2b F3 LOW | yes | 159: test only. 191: `r9_descriptive.check_rebuilt(got, arrays, sl, n)` (same test as `RH.array_item`: shape, dtype, `array_equal`). 172: `r9_descriptive.check_directions(sc, rows)`. 139: `run_r9_descriptive.near_counts(rec)`, and it now also requires the file's SHA to equal `NEAR_DUPLICATES_SHA256` (`ba92530d...`, the value fx-c13 pins; own constant here because fx-c13 is not in this branch). 145: test only (A2b's). | 159: `test_bind_read_refuses_a_pass_changed_after_the_verdict`. 191: `test_check_rebuilt_accepts_equal_arrays_and_refuses_each_difference` (1e-12, NaN, dtype with equal values, shape, `seed_index`, gate array). 172: `test_check_directions_on_a_real_score_and_with_one_row_changed` (real `S.score_seed` on `make_bundle` with `on_scores=collect`; first and last cell changed). 139: `test_near_counts_reads_cos_092_from_the_pinned_file`, `..._refuses_a_file_that_is_not_the_one_the_pass_names`, `..._a_pass_that_names_another_sha`, `..._not_the_pinned_one`, `..._bad_counts[5]`, `test_the_near_duplicates_pin_is_the_stage_1_files_sha`. 145: `test_stage0b_identity[2]` |
| 5. duplicate `require` | yes | `score_inputs`: the second `A28 pins/pass binding` require is removed (the first stays at the top of the A28 branch); the `read_set_sha256` require stays | covered by the existing flow (no unit test: plumbing) |
| 6. 23 fix LOW-A, LOW-B | yes | tests only | `test_bind_read_agreement_of_wrong_bytes` (expects `Refused`); `test_smoke_tag_refusals[6]` (no `--smoke`; `/`, `.`, `..`, empty, space), `test_smoke_tag_never_replaces_an_earlier_file` |

## How item 2's copy was checked
`fx_desc_h5_equiv.py` (result `fx_desc_h5_equiv.json`; this script, not the runner, imports round 6 from the main checkout; about 1 minute).
On H5's real held files (`R6D/results/held_arrays.npz`, `held_pass.json`): `H5_CORE`, `H5_EXTRA`, the rule SHA, the check names and
all nine `QUANTITIES` texts equal round 6's; `load_core` returns equal arrays, equal dtypes and equal keys for 3 seeds x 8 scorers x 5
metrics; both `check_pass` functions accept the real pass; five edited passes (P3 count +1, S1 point +1e-9, P7 interval +1e-9,
P1 quantity text, `n_clusters` +1) are refused by both. `h5_inputs(False)` and `h5_inputs(True)` also run on the real files.

## Tests
From F: `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider test_r9_descriptive.py test_r9_pins.py test_r9_apply_rule.py`
-> 216 passed (about 12 s); `test_r9_descriptive.py` alone: 43 passed (was 12). Also `test_r9_preread.py test_r9_review_fixes.py test_r9_common.py`: 283 passed (before the last test edits; only tests changed since).
Mutants: `fx_desc_mutants.py` -> `fx_desc_mutation_results.json`: 21 of 21 caught (the first run caught 17; the 4 survivors were a verdict `read` change and a `rule_sha256` change that another check
also refused, the dtype comparison, and the pass-side near-duplicates SHA; each now has its own case). Files restored byte for byte.

## For the controller
- The module bytes changed (`run_r9_descriptive.py`, `r9_descriptive.py`), so `r9_module_shas()` changed: every smoke record and the CLIP pre-read must be redone on the final bytes. I ran no real or smoke descriptive run (no smoke folder exists here; the run is long).
- `final_review/fr_a2b_probe_tests.py` is untouched and now fails its F2/F3a cases (its fixture writes the code name as the verdict's backbone, the bug F1 fixed). It was never in the suite. Listed for deletion: it is superseded by the tests above.
- Merge note: fx-c13 (merged in main r9-scope, not in this branch) also holds `NEAR_DUPLICATES_SHA256` in `run_r9_held.py` and `rederive/rd9_inputs.py`. After both merge, `run_r9_descriptive.NEAR_DUPLICATES_SHA256` could be `RH.NEAR_DUPLICATES_SHA256`; I kept a copy so this branch stands alone. `test_the_near_duplicates_pin_is_the_stage_1_files_sha` checks it against the real file.
- Choice not fixed by the ticket: `near_counts` pins the SHA in smoke as well as real mode (both read the same real file).
- Untracked leftovers of other jobs in the main checkout were not touched.
