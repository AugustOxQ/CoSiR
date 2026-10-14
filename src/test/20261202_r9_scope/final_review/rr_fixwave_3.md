# RR3: scoped re-review of fix ticket fx-rr (r9 build 11)

Snapshot: `/project/CoSiR-wt/r9-rr3`, detached at beca6ce. Diff reviewed: `git diff ab1367e...562c0c8`. Code was not changed.

## Verdict: PASS WITH NOTES

Every item is met. Every new guard has a test that fails when the fix is undone. There is no HIGH or MEDIUM finding. Two
LOW notes concern a test gap and a hardening idea (below).

## Checks

- **N5:** `git diff --stat e79d4a1 HEAD -- <the 18 closure files and run_r9_stage2.py>` prints nothing. Met.
- **Tests on the snapshot** (quiet, no GPU): `test_r9_a28_stats.py test_r9_descriptive.py test_r9_stage2.py
  test_r9_preread.py rederive/rd9_test_compare.py` gave 436 passed. `test_r9_apply_rule.py` gave 155 passed. I did not run
  `test_r9_a28.py` or the recipe-groups tests (the controller runs the full suite).

## Method for the mutants

A copy of the round folder outside a git checkout cannot import `r9_common` (it needs `git rev-parse` and a `src/` tree).
Without that fix, the ticket's `fx_rr_mutants.py` shows "caught" for every mutant from a collection error (I saw this in
my first run). So I copied the whole snapshot tree (no `.git`, 85 MB) to `scratchpad/rr3/repo3`, ran `git init` there, and
mutated only that copy. In that copy 5 tests fail on the baseline (3 in `test_r9_stage2.py`, 2 in `test_r9_descriptive.py`,
all from missing untracked `results/` of other rounds). I judged each mutant by the failing tests that are new compared
with the baseline (script `scratchpad/rr3/mut3.py`).

## Per item

1. **RR2 M1 (a28_stats fixtures): met.** `test_r9_a28_stats.py:` `load_record` adds `read` and `backbone`. The file passes (51
   in the ticket's count; it is part of the 436).
2. **RR1 M1 (H5 copy tests): met.** `test_r9_descriptive.py` (end of file). I rebuilt the 15 H5 mutants of the ticket and
   judged them against the baseline: all 15 are caught by the intended tests. Specifically the removal of the keys check
   fails `..._refuses_a_missing_key` and `..._an_extra_key`. Removing the n, point, interval or quantity comparison fails
   `..._refuses_each_changed_item` for P1, P6 and S1 (the `quantity` mutant fails all three cases; my one-line summary
   showed only one name because the test id contains " - "). The episode and cluster count checks, the dtype, shape,
   seed-order, pair-order checks and the S1/S2 loop are each caught. Also `h5_inputs` is checked by
   `test_h5_inputs_binds_smoke_pass_arrays_and_episodes`.
3. **RR1 N1 (margins skipped in selection): met.** `rederive/rd9_compare.py:256-262`. `own.get('mode')=='selection'` sets
   `compare_margins=False` and adds `margins_not_compared`; held and smoke derived records carry `mode == 'read'`
   (`rd9_phase2.py:94`, choices read or selection), so they still compare. Ticket mutants (4) caught. My own mutants, all
   caught: the flag always false; note written but margins still compared when a sensitivity file is given; skipping in
   every mode including `read`. `test_held_modes_still_compare_the_margins[held/smoke]` and
   `test_selection_replay_does_not_compare_the_margins_and_says_so[False/True]` do the work. Note: the `mode` argument of
   the test helper `stage()` sets the reported pass file's mode; the derived mode is always `read`, so the "held/smoke"
   parametrisation tests two reported modes, which is the right thing.
4. **RR1 N2 (commands doc): met.** `rederive/rd9_commands.md:94` carries `--reported-sensitivity` in the per-read block.
5. **RR1 N3 (agreement bound to the sensitivity file): met.** `run_r9_apply_rule.py:665-667`, after `load_sensitivity`, so it
   compares with the SHA of `spath` (the `_fix1` file, else the original). The preread stand-in now writes the field
   (`run_r9_preread.py:274`; the sensitivity file exists at that point, the call is after `RH.run_smoke`, line 333 versus
   341). Ticket mutants (guard removed; guard made optional; stand-in field removed) caught. My own mutants caught: compare
   with the pass SHA instead; compare with the original file for a `_fix1` pass when the `_fix1` pass has no own file; accept
   an empty string. **Survived:** a mutant that always hashes the original `sensitivity_<Hn>.json` even when the `_fix1`
   pass has its own `sensitivity_<Hn>_fix1.json` (LOW-1 below).
6. **RR2 LOW-2 (comment, docstring): met.** `run_r9_preread.py:276-278` says in smoke mode only the apply step reads the
   stand-in; `run_r9_apply_rule.py:9-10` names `heldfix1`.
7. **RR2 LOW-5 (prov_overlap test): met.** `test_r9_stage2.py::test_a_key_in_both_made_and_copied_fails` uses equal
   arrays so only the overlap guard stops it; the mutant that removes the guard line fails exactly that test. Driver
   untouched.
8. **Controller item 8 (`test_r9_a28.py` HOME): met.** One added env key; the tests of the file are unchanged otherwise.
9. **Controller item 9 (`test_r9_recipe_groups.py`): met, with a small weakening.** The test no longer asserts that the
   default location is empty (it may hold a real file); it asserts the pinned path is `out/partitions.npz`, not the
   default, and that the SHA is of the out file. That is enough to catch a step that uses the default path.

## The hand-back's own choices

- **(a) N3 strict: sound.** An agreement without the field is refused; the per-read command now always passes the
  sensitivity file; no agreement for a real held read exists yet, and the smoke records must be redone on final bytes anyway.
  The consequence (old agreements and old stand-ins are refused) is stated in the hand-back.
- **(b) N1 keys on the derived `mode`: acceptable, with one residual hole.** `rd9_phase2.py` sets `mode` from its
  `--mode` flag. A held read run by mistake with `--mode selection` has seeds [42] and other `n_episodes`, so the `seeds`
  and `n_episodes` items disagree and the agreement is false. So an honest mistake is caught indirectly. But a derived
  record whose `mode` is wrongly `selection` while its stats equal the held pass (a hand edit) would agree, would skip the
  margins (and, for H8 to H10, the set-aside, flags and duplicate checks, and the `seed_index` array), and nothing in the
  apply step reads `margins_not_compared`. See LOW-2.
- **(c) RR1 N3 second part not done: acceptable.** `compare_stats` still skips the margins silently when the derived
  record has no `SE`/`x`/`x95` (`any(k in o ...)`). `rd9_phase2.py` writes them in every mode, so this needs a hand edit; a
  non-selection read should require them (cheap guard; not a finding against this ticket).
- **(d) Rerun of the H8 selection replay gives 136 items: plausible, by reading.** The committed file has 129 items:
  84 `checks.*` (7 checks times 12 fields), 32 arrays, 7 `holm_order`, 3 `episodes_sha256`, `n_episodes`, `n_clusters`,
  `seeds`; it has no quantity items and no margin items. Now each of the 7 checks adds `quantity`, giving 136. The R6
  `seed42_pass_counts.json` carries all seven quantity labels, equal to `rd9_stats.QUANTITIES` (checked). The margins are
  skipped. The reported side has `mode: regression`, `seeds: [42]`. I could not rerun (no `rd9_out` record), so "agrees" is
  unproven; the count and the label match are checked.

## New findings

- **LOW-1** `test_r9_apply_rule.py` (the three N3 tests): no test has a `_fix1` pass with its own
  `sensitivity_<Hn>_fix1.json` that differs from the original. The code (`run_r9_apply_rule.py:665`) is correct, but a
  regression that hashes the original file in that case survives. Add one test.
- **LOW-2** `rederive/rd9_compare.py:256-262` and `run_r9_apply_rule.py:665`: the apply step does not look at
  `margins_not_compared`, and the comparer does not cross-check the derived `mode` against the reported file (R6's seed-42
  file says `mode: regression`; R9 pass files say `held` or `smoke`). Hardening: refuse in the apply step any agreement that
  carries `margins_not_compared`, or require the reported mode to be `regression` for the skip.
- **LOW-3** `fx_rr_mutants.py:55-62`: `caught` is `returncode != 0`, so a collection error counts as caught, and with `-x`
  a baseline failure (as in my copy) hides the real result. In the ticket's own tree its recorded results show real FAILED
  lines, so the 23 of 23 stands; it should record the failing test names or compare with a baseline.
- **LOW-4** (cosmetic) a selection run given `--reported-sensitivity` records `sensitivity_sha256` although the margins are
  not compared.

## Left in place (nothing deleted)

Scratch under `/tmp/claude-0/-project-CoSiR/0f172dc2-24f9-4491-a3eb-2c13b77a047f/scratchpad/rr3/`: `repo`, `repo2`,
`repo3` (copies of the tree), `wk`, `mut3.py`. A delete hook refused my attempt to remove the symlinks in `repo`; I left
them and wrote no pending-deletion line.
