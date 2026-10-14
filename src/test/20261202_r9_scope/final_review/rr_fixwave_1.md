# Scoped re-review of the fix wave, part 1: fx-c13 and fx-desc (snapshot ab1367e)

Reviewer: rr1. F = `src/test/20261202_r9_scope`. Method: diffs `8db7ceb...r9-fx-c13` and `039fc93...r9-fx-desc`, the findings,
the tickets' mutation scripts rerun on a scratch copy (`<scratchpad>/wk/F`, nothing in the snapshot changed), the probe
rerun, tests on the snapshot (and on the main checkout, which has the real files).

## Verdict: PASS WITH NOTES

No HIGH. One MEDIUM (the copied H5 loader and pass check have no test in the suite). Four LOW.

## N5 and nearby tests

- N5: `git diff --stat e79d4a1 HEAD` over the 16 closure modules, `run_r9_stage0a.py` and `run_r9_stage2.py`: empty. Met.
- Snapshot: `rederive/rd9_test_{phase1,compare,arithmetic,a28}.py` 255 passed, 1 skipped (the real-file test that needs the
  integration checkout); `test_r9_preread.py` 111; `test_r9_held_a28.py` 290; `test_r9_held.py` + `test_r9_held_shared.py` 206;
  `test_r9_pins.py` 21; `test_r9_descriptive.py` + `test_r9_apply_rule.py` 195. All pass.
- Main checkout (all real files): `rd9_test_phase1.py` + `rd9_test_compare.py` 207 passed, 0 skipped; `test_r9_pins.py` 21 passed
  (so `test_build_pins_names_a_missing_part` passes both where Qwen's recipe output exists and where it does not).

## fx-c13

Independence: no `r9_*` import in `rederive/` (the existing test passes; `rd9_compare.py` imports only `rd9_*`).

| Item | Verdict | Where / evidence |
|---|---|---|
| B2 M-1: `--phase1` reads a real `pins_<B>.json` | met | `rd9_inputs.py` `parts_entries`, `Inputs.__init__` (named roles via `parts`), `rd9_compare.py` main (`sensitivity_seed42` read pinned). Probe rerun by me (below). |
| A1 M1: SE, x, x95 per read | met | `rd9_compare.py` `compare_stats(...,sensitivity)`; `--reported-sensitivity`; tests `test_a_changed_margin_disagrees[SE\|x\|x95]`, `test_margins_without_a_sensitivity_file_disagree`. Mutants 28-30 caught. |
| A1 M1: seed-42 sigma inputs in phase 1 | met | `compare_sigma` (N, clusters, sigma_a2, sigma_e2 of P1..P7), called in `compare_phase1` and in the CLIP branch (file read pinned, SHA `bc4b0a92`, equals `run_r9_held.SIGMA_CLIP_SHA`). 6 mutants caught. |
| B2 M-2: tests for sigma*, AFF/CF, B0, RCA lambda, tau, P-points | met | `test_one_changed_pinned_quantity_disagrees` (24 cases) plus the tau-tolerance and exact-picks tests. Deleting each comparison (mutants 1-19) fails a test. |
| C MEDIUM-3: floor arrays and CLIP B/B0/B1 means | met | `rd9_compare.py` CLIP branch `floor` (explicit); tests per array; mutants `floor_*`, `clip_*` caught (7). |
| B1 LOW-2: pin `near_duplicates_a28.json` | met | `rd9_inputs.NEAR_DUPLICATES_SHA256` = `ba92530d...2206`, equals `sha256sum` of the real file (I computed it). `read_pinned` in the comparer; `run_r9_held.setup_a28` checks the same value. A test compares the two constants. 3 mutants caught. |
| A1 L2 (comparer half) | met | `rd9_stats.QUANTITIES` compared as `checks.<j>.quantity`. The nine labels equal R6's `seed42_pass_counts.json` and `r9_stats.QUANTITIES` (I compared P1..P7 against both). |
| B1 nit (`rd9_compare.py:128`) | met | assert removed, `sorted(entries)` keeps the repeat, the derived side is a set, lengths differ. Mutant caught. |
| A1 L5: unpickle check | met | `rd9_inputs.unpickle`, used by `Inputs.readers`, `rd9_phase1.py`, `rd9_phase2.py`; a test forbids a plain `pickle.load` in any rederive script. 2 mutants caught. |

Mutations: I reran `fx_c13_mutations.py` on the scratch copy: all 43 caught; the unmutated tests pass on the same copy (497 passed),
so a catch is a real failure.

Probe (B2's, `fx_c13_phase1_probe.py`): I reran it on the snapshot's comparer with the phase-1 outputs of the `r9-fx-c13` worktree
(same pins SHAs: `pins_clip.json` b78dd5ef, `pins_qwen3vl_emb_2b.json` 9a0fae5e). CLIP: agreement true, 169 compared, 0 failed.
Qwen: agreement true, 160 compared, 0 failed. Same as `fx_c13_phase1_probe.json`.

Choices:
- (a) `--reported-sensitivity`, fail closed without it: sound for the per-read agreement. No code launches the comparer (the pre-read
  and the apply step only read the agreement file; the descriptive step reads it too), so "callers pass it" means the documented
  command. Problems: see N1 and N2.
- (b) sigma for P1..P7 only, not S1/S2: sound. The derived `pass_record` computes no S1/S2 sensitivity, and the bars use P1..P7.
  The S1/S2 sigma values in the frozen files stay uncompared (note, not a finding).
- (c) own copy of the quantity labels: sound. It keeps C13 independent, and it equals both the R6 and R9 text.
- (d) `sys.modules` before and after: sound for its purpose, with the limit in N4.
- (e) CLIP floor explicit: sound. The old floor came from the derived file's own names, so a derivation without `P__a` shrank it. Now fails.

## fx-desc

| Item | Verdict | Where / evidence |
|---|---|---|
| A2 F1: verdict backbone is the ledger name | met | `run_r9_descriptive.py:79` `RH.LEDGER_NAMES[RH.READ_BACKBONES[read]]`. The tests build the verdict with `AR.apply`; mutant `verdict_backbone_code_name` caught. |
| A2 F2: bind_read refusal tests | met | three tests; mutants for status, rule SHA, read, pass SHA (159) caught. (The status mutant is caught by a message check: a later guard also refuses, so the status guard is defence in depth.) |
| A2b F3 (159, 191, 172, 139, 145) | met | `check_rebuilt`, `check_directions`, `near_counts` and tests; mutants for each caught. |
| B2 M-4: no r6 import | met | `test_no_r9_module_imports_an_r6_module` passes; no `r6_*` name in `run_r9_descriptive.py` code. |
| Duplicate `require` | met | second "A28 pins/pass binding" removed; the first stays at the top of the A28 branch (it was a repeat of the same bind). |
| LOW-A, LOW-B (23 fix) | met | `test_bind_read_agreement_of_wrong_bytes`; `test_smoke_tag_refusals`, `test_smoke_tag_never_replaces_an_earlier_file`; mutants caught. |

Mutations: I reran `fx_desc_mutants.py` on the scratch copy: 21 of 21 caught.

Choices:
- (a) H5 copy. Line by line against `r6_descriptive.load_core` / `check_pass`: same key set (`set(z.files) == want`), same shape
  test, same int64 test of the three index arrays, same seed-order and pair-order tests, same float64 and finite test, same
  count, point, interval and quantity comparison, same `check_diffs` call (`secondary=True` is r9's default and equals r6's
  only form). `H5_CORE` = `CORE_SCORERS`, `H5_EXTRA` = `ARRAY_EXTRA`, `METRICS` from the same import, `R.R6_RULE_SHA256` equals
  `r6_common.RULE_SHA256` (both `7444a5e3...`). Only the message texts differ. The ticket's equivalence run
  (`fx_desc_h5_equiv.json`) agrees. Sound in meaning. But see M1: nothing in the suite tests it.
- (b) near-duplicates pin in smoke too: sound. `setup_a28` (fx-c13) pins the same file in smoke, so a smoke pass already
  carries the pinned SHA; a smoke run reading another file would be refused either way.
- (c) `check_rebuilt`, `check_directions`, `near_counts`: same logic as the inline code. `check_rebuilt` = `RH.array_item(...)['equal']`
  (shape, dtype, `array_equal`), same `seed_index` special case and slice rule. `check_directions` is the same expression.
  `near_counts` is the same checks plus the pin. Mutants for each caught.

Controller commit ab1367e: `NEAR_DUPLICATES_SHA256 = RH.NEAR_DUPLICATES_SHA256` is right (one value; rederive keeps its own
copy and a test compares it); the pins test now passes with and without Qwen's output. Fine.

## New findings

**M1 (MEDIUM, test gap): the H5 copy has no test in the suite.** `h5_load_core`, `h5_check_pass` and `h5_inputs`
(`run_r9_descriptive.py:222-275`) are covered by no test; `grep` finds none. I removed the keys check
(`require(set(z.files) == set(want)...)`) and the whole count/point/interval/quantity comparison in a scratch copy:
`test_r9_descriptive.py` stays at 43 passed. The equivalence script is a one-time check (and needs the main checkout), so an
edit to the copy later would pass silently, and H5 is read on real files. Fix: a test with a small synthetic `held_arrays.npz`
(or the real H5 files, skipped when absent) that accepts it and refuses a missing key, a wrong dtype, a changed count and a changed
quantity. The ticket's mutants did not touch the copy, so its "21 of 21" does not cover it.

**N1 (LOW): the H8 selection replay can no longer agree.** `rd9_phase2.py` adds SE/x/x95 to the derived record in every mode,
including `--mode selection`, so `compare_stats` now compares them. The reported side is R6's `seed42_pass_counts.json`, and the
only sensitivity file for seed 42 (`sensitivity_seed42.json`) holds sigma_a2 and sigma_eps2, not SE, x or x95. Without the option
the keys differ; with it they still differ (`margins.get` finds no SE). The hand-back tells the controller to rerun the committed
selection replay "with `--reported-sensitivity`": that cannot give agreement true. Nothing consumes
`rd9_H8_selection_agreement.json`, so no verdict is at risk. Fix: skip the margin items when `own['mode'] == 'selection'` (the
sigma inputs are already compared in phase 1), or say in `rd9_commands.md` that the committed file stays as it is.

**N2 (LOW): the per-read command block in `rd9_commands.md` (line 94) does not carry `--reported-sensitivity`.** Only a paragraph
below says to add it. A user who copies the block gets agreement false (fail closed, so safe, but a wasted run). Add it to the block.

**N3 (LOW): the apply step does not bind the agreement to the sensitivity file it uses.** The agreement records
`sensitivity_sha256`; `run_r9_apply_rule` reads the agreement but does not compare that digest with its own `sensitivity_sha256`.
An agreement made with another sensitivity file would still be accepted. The hand-back lists this as not done (A1's optional part).
Also `compare_stats` compares the margins only `if any(k in o for k in ('SE','x','x95'))`: a derived record without any of them skips
the check. rd9_phase2 always writes them, so no gap today; requiring them for every non-selection read would close it.

**N4 (LOW): the unpickle check is vacuous if an `r9_*` module is already in `sys.modules`.** It compares before and after, so a
pickle that references an `r9_*` class passes when something in the process imported that module earlier (for example inside one
pytest session that also imports the chain). The rederive scripts run in their own process and import none, so in use it holds.
It also looks only at `r9_`/`run_r9` names. Note only.

## Items for the controller

- `final_review/fr_a2b_probe_tests.py` (named in the fx-desc hand-back as superseded) and the three untracked
  `fx_c13_mutations_*.json` leftovers in the `r9-fx-c13` worktree: could be removed; I deleted nothing.
- After fixing M1 (tests only) nothing else needs a rerun; the module bytes of `run_r9_descriptive.py` and `r9_descriptive.py`
  changed in fx-desc (the hand-back says smoke records and the CLIP pre-read must be redone on final bytes).
- No command here took more than a few seconds; no long run is left.
