# fx-c13 hand-back (progress note; branch r9-fx-c13)

Cut at the context meter. Items 1 to 8 are coded and tested; the final mutation table, the Qwen phase-1 score run and the
probe rerun on both backbones are left (see "Left").

## Items

1. **B2 M-1 (done in code, probe left).** At HEAD, `Inputs('clip_b32')` and `Inputs('qwen3vl_emb_2b')` on the real pins
   fail (FileNotFoundError on `input_sha256` keys; checked on a HEAD copy). `rd9_inputs.py`: `parts_entries` (new), used
   when the pins have `parts` (`Inputs.__init__`); fitted roles are named by their `parts` key (CLIP's cells file is
   `cells_seed42_clip.json`); `role_paths` resolves the recipe folder (symlink-safe). `rd9_compare.py` main: the Qwen
   branch reads the pins' `sensitivity_seed42` {path, sha256} with `read_pinned`. Real `pins_clip.json` has no `seed42`
   record: its phase-1 comparison runs in the R6 (CLIP) branch. Tests: `test_inputs_read_the_real_pins_files[clip_b32|qwen3vl_emb_2b]`,
   `test_parts_entries_need_path_and_sha256`, `test_main_*` (rd9_test_phase1.py).
2. **A1 M1 (done).** Per read: `--reported-sensitivity` (new); `compare_stats(..., sensitivity)` compares SE, x, x95 of
   P1 to P7 when the derived record has them (absent file = differences); agreement records `sensitivity_sha256`. Phase 1:
   `compare_sigma` (N, clusters, sigma_a2, sigma_e2 of P1 to P7) in `compare_phase1` and in the CLIP branch (R6's
   `sensitivity_seed42.json`, pinned). Tests: `test_margins_*`, `test_a_changed_margin_disagrees[SE|x|x95]` (rd9_test_compare.py);
   `test_one_changed_pinned_quantity_disagrees[sigma_a2|sigma_e2|sigma_N|sigma_clusters]`, `test_clip_branch_compares_the_sigma_file`.
3. **B2 M-2 (done).** `test_one_changed_pinned_quantity_disagrees` (24 cases: sigma*, AFF, CF, switched_off, gate share,
   B, B0, RCA lambda, tau, five means, P-points, n_j, array hashes), `test_picks_are_compared_exactly_and_taus_to_a_relative_tolerance`.
4. **C MEDIUM-3 (done).** `test_a_derivation_without_a_floor_array_disagrees` (one case per array of the stage-2 floor),
   `test_clip_branch_derivation_without_a_floor_array_disagrees`, `test_clip_branch_compares_the_B_means[B|B0|B1]`.
   Also: the CLIP branch floor is now explicit (it was built from the derived file's own names, so a derivation without
   `P__a` shrank it).
5. **B1 LOW-2 (done).** `rd9_inputs.NEAR_DUPLICATES_SHA256` = `ba92530d433dded1c8d2b2a2037e23d015aed5361fb1249d1caa631e40da2206`
   (full SHA of the real file checked); `rd9_compare` reads it with `read_pinned`; `run_r9_held.py` (`setup_a28`, 5 lines, no
   guard marker) reads it pinned. Tests: `test_near_duplicates_file_must_equal_its_pin`, `test_near_duplicates_pin_constant`,
   `test_real_near_duplicates_file_matches_the_pin`, `test_near_duplicates_pin_equals_the_comparers` (held), and the extended
   `test_smoke_builds_selection_space...` (edited file refused).
6. **A1 L2, comparer half (done).** `rd9_stats.QUANTITIES`; `compare_stats` compares `checks.<j>.quantity`. Tests:
   `test_the_quantity_label_is_compared`, `test_a_missing_quantity_label_disagrees`.
7. **B1 nit (done).** `reported_set_aside` keeps a repeated entry (the assert is gone; the derived side is a set, so the
   lengths differ and the key `set_aside` disagrees). Test: `test_a_repeated_set_aside_entry_disagrees`.
8. **A1 L5 (done).** `rd9_inputs.unpickle` asserts no `r9_*`/`run_r9*` module was imported by the unpickling; used by
   `Inputs.readers`, `rd9_phase1.py` (score) and `rd9_phase2.py`. Tests: `test_unpickle_refuses_a_pickle_that_imports_an_r9_module`,
   `test_the_reader_pickle_is_loaded_through_unpickle`, `test_no_rederive_script_unpickles_without_the_check`.

## Tests

`cd src/test/20261202_r9_scope/rederive && python -m pytest -q rd9_test_phase1.py rd9_test_compare.py rd9_test_arithmetic.py rd9_test_a28.py -p no:cacheprovider`
(last full count before the two unpickle tests: 254 passed; rd9_test_phase1.py alone: 102 passed). `test_r9_held_a28.py`: 290 passed.
Final state: rd9_test_phase1 + rd9_test_compare + rd9_test_arithmetic + rd9_test_a28 + test_r9_held_a28: 546 passed;
test_r9_held.py: all passed; test_r9_preread.py: 62 passed (one test edited to pass the sigma argument).
`test_r9_pins.py`: 2 failures not mine: `test_no_rederive...`: `test_no_r9_module_imports_an_r6_module` is B2's M-4 (run_r9_descriptive.py);
`test_build_pins_names_a_missing_part` needs `results/stage2_qwen3vl_emb_2b.json`, which only the main checkout has.

## Mutations

`fx_c13_mutations.py` (breaks each new guard once, restores, checks SHA). All 43 mutants are caught
(`fx_c13_mutations.json`: 42 caught in the full run, the 43rd, `phase2_fits_unpickle_plain`, caught on its own rerun
with the corrected mutant text; each file restored and SHA-checked). Untracked leftovers of mine, not committed and not
deleted (a hook refused `rm`): `fx_c13_mutations_0_22.json`, `fx_c13_mutations_22_50.json`, `fx_c13_mutations_0_43.json`
(partial or duplicate runs from a shell-quoting slip); they can go.

## Left

- Qwen phase 1 score: prepare is done (`rd9_out/rd9_qwen3vl_emb_2b_{bundle.npz,fits.pkl,prepare.json}`, this worktree, ignored).
  The score stage was started in the background at ~03:20 (log `/tmp/claude-0/-project-CoSiR/06992efa-c388-41ec-bd52-4b5c99c57e84/scratchpad/logs/score_qwen.log`;
  it ends with `phase 1 complete: qwen3vl_emb_2b` and writes `rd9_out/rd9_qwen3vl_emb_2b_phase1.json` and `_seed42_arrays.npz`).
  If it died, rerun `rd9_phase1.py --backbone qwen3vl_emb_2b --stage score` (CPU).
  CLIP phase 1 is done (`rd9_out/rd9_clip_b32_phase1.json`, `_seed42_arrays.npz`).
- Then `python src/test/20261202_r9_scope/fx_c13_phase1_probe.py clip_b32 qwen3vl_emb_2b` (written, not yet run) and report per
  backbone; fix what still fails. `results/recipe` in this worktree is a symlink to the main checkout's (ignored; I made it).
- Re-run the committed selection replay `rd9_phase2.py --read H8 --mode selection` needs `--reported-sensitivity` now (controller).
- The apply step does not yet require the margin items in the agreement (A1's optional part); not done.
