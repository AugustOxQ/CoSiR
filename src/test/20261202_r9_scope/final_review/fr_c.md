# r9 final review C: ticket 22 (pre-read checks, smoke, launch gate, comparer edits)

2026-10-14 01:52. Snapshot `/project/CoSiR-wt/r9-fr-c` at fdd5acc (ticket 22 on top of 4458634). F = `src/test/20261202_r9_scope/`.

## 1. Verdict

**CONFIRMED WITH FIXES.** The claim holds for the code as written. No held read from H6 to H10 can launch unless a
pre-read passed all of rule §6's checks on the current bytes. The fixes are missing tests plus small LOW items. I found
no HIGH finding: no launch path skips the checks, and there is no C5 leak.

- The real launch gate refused every bad record I tried (13 tmp_path scenarios, real `run_held`, `preread_guard` and
  `smoke_guard`). The positive control launched.
- Several sub-conditions of the gate are correct in the code, but no test pins them down. With all tests green, a
  later edit could remove the module-bytes half of the code check, the read or backbone check, or the "every check
  passed" check (MEDIUM-1).

## 2. What I checked

| Check | How | Result |
|---|---|---|
| Ticket test file | `pytest -q test_r9_preread.py` (env as brief) | 62 passed in 5.6 s |
| §6 order and stops | Read `run()` and each gate function in `run_r9_preread.py` | Order: inputs, heads, regression, sensitivity, phase 1, smoke. A failure writes a failed record with every later check `not_run` and exits 4. |
| Item 1, stage 0a | `inputs_gate` l.100-107 | Reads only `stage0a_regression.json`. Needs `passed is True`, `module_sha256 == r9_module_shas()` and the runner SHA of `run_r9_stage0a.py`. |
| Item 1, stage 0b and stage 2 | l.120-132 | For non-CLIP backbones (H6, H7, H9, H10) it calls the owner's `S2.assert_modules`, which checks `passed` and the 17 closure SHAs. It then needs stage 2 `passed`, the right backbone, `switched_off False`, `dry_run False`, and its seed-42 and sigma files at their recorded SHAs. |
| Item 1, pins | l.108-118, 133-136 | Every part is re-hashed before the owner's `check_pins` (which calls `build_pins` and reads the cells). `R.assert_inputs` re-hashes the rules and inputs. A28 reads also call `AD.check_pins`. At launch, every file in `input_file_sha256` is re-hashed. |
| Item 2, head refits | `heads_gate` | Uses the owner's `RH.setup` (`check_selection`) and `head_guard` with R6's arm-L reference (CLIP only). |
| Item 3, seed-42 regression | `regression_gate` | Picks and cells load only after the input gate. CLIP: stage 0a's `regression_items` (all N_ITEMS) plus `compare_counts`. New backbones: 6 fields of `S2.seed42_record` against the pinned record (including all 96 array hashes). `E.identity_check` covers seeds 42 and 9001 to 9003. |
| Item 4, sensitivity | `sensitivity_gate`, `RH.sigma_guard` | Recomputed with `r9_stats.check_diffs` and `sigma_split`, and compared for exact equality with the owner's file (CLIP: R6's file, SHA pinned). The source is never written: the scratch copy is opened with `xb` and its SHA is checked. |
| Phase-1 agreement | `phase1_gate` | Takes fix1, then crash1, then the base name. Needs `agreement`, phase 1, backbone, `failed == 0`, all items equal, the current pins SHA, the independent record and arrays below `AGREEMENT_DIR` at their recorded SHAs, and the derived record bound to the current pins. |
| Item 5, smoke | `smoke_gate`, `wiring_mutation`, `publish_smoke` | Last. Held and A28 paths on selection rows, seeds 9001 to 9003, 64 episodes per pair, apply and descriptive steps. The inputs-only branch runs on the real read rows first. The wiring mutation runs an isolated copy of `r9_score.py` and must raise the CF condition-free error. The record holds `passed: true` and `r9_module_shas()` and is checked with `AR.smoke_guard`. |
| Launch gate, bad records | `final_review/fr_c_gate_test.py`: 13 cases through the real `run_held` | Refused: missing record; failed record; a check `not_run` with `passed: true`; stale module SHAs; H6's record under H9's name (same backbone); wrong backbone; smoke record from older bytes; missing smoke record; empty inputs; reserve or fix launch with only a base pre-read; malformed JSON (exception, no launch). Valid record: launches. 13 passed. |
| Gate before any held number | `run_held` l.898-901 | After `refuse_or_go` (ledger and names only), before `smoke_guard`, `a28_reference` and `run_read`. Tested by `test_held_launch_calls_preread_before_read`. |
| Test stubs of the gate | `configure` in `test_r9_held.py` | They replace the module attribute. `run_held` imports `preread_guard` at call time, so production always calls the real gate. There is no other held launch path: `run_read` with held or A28 mode is built only in `run_held`. |
| C5, projection | `input_read_set` l.72-91, `setup_a28`, `projected_a28_report` | The parquet is read with `columns=` (row_id, painting, language, held, dropped). The schema check reads only the footer. Labels are set to the sentinel -1 and deleted before `A28Inputs` on the inputs-only branch, which returns before any episode. The smoke builds its table from selection rows. The descriptive smoke's thin-value counts (sentinel based) are dropped. No read-row score is computed. Hashing the whole parquet file is a byte check, not decoding. The language column is a row attribute that the loader's checks need. It is not one of the three labels. |
| C13 independence | Imports in `rd9_compare.py` and `rd9_inputs.py` | Only `rd9_*` and `src.data`. No `r9_*` or `run_r9_*`. `phase1_certificate` uses the re-derivation's own `verify_reads`. |
| C13 lists | `compare_phase1` and the CLIP branch | Cells (aff, cf, σ*, switched_off, gate share), B and B0 picks, RCA λ, τ, seed-42 means (B, B0, B1, AFF, CF), P-points and n_j for P1 to P7, array hashes: all compared. Untested parts: MEDIUM-3. |
| Guard deletions | `final_review/fr_c_mutants.py`, 11 deletions (9 from t22's table, including the `run_held` call site, plus the 2 comparer markers) | All 11 caught. Agrees with `t22_mutation_results.json` (41 of 41). |
| Sub-condition mutants | Same script, 23 more | 18 survived (below). Source SHA-256 restored and checked after each mutant. |

## 3. Findings

**MEDIUM-1: the launch gate's sub-conditions are untested.** `run_r9_preread.py:429-438`.
- Scenario: someone edits `preread_guard` and drops `rec.get('module_sha256') == R.r9_module_shas(here)`. Every test
  stays green, and a pre-read made on older module bytes then authorises a launch if the held runner's bytes are
  unchanged. The smoke guard still compares module SHAs, but only those of the smoke record.
- The same holds for `rec.get('read') == read`, the backbone check, the "all six checks passed" check and the
  non-empty `input_file_sha256` check. `test_launch_stop` changes only `passed`, `runner_sha256`, one input file and
  the proof SHA.
- These 5 sub-condition mutants survive `test_r9_preread.py` and are all caught by `fr_c_gate_test.py`.
- Fix: move the stale-modules, other-read, other-backbone, `not_run` and empty-inputs cases of `fr_c_gate_test.py`
  into `test_launch_stop`. Tests only; no closure module.

**MEDIUM-2: the single-field stops of §6 items 1 and 3 and of phase 1 are untested.** `run_r9_preread.py:125-127,
176-178, 183-185, 214-216`.
- `regression_state` makes all 6 pinned seed-42 fields wrong at once. A mutant that compares only `seed` passes.
- The CLIP test makes the regression items and the counts both wrong, so removing either half survives.
- These stage 2 conditions survive: `switched_off`, `dry_run`, backbone. `read_guard` in `regression_gate` still
  stops a switched-off backbone through the cells record.
- These phase-1 conditions survive: `phase == 1`, backbone, `failed == 0`, all items equal. The pins binding and
  the comparer's own flag partly cover them.
- Scenario: a later edit narrows the seed-42 comparison to one field, and a backbone whose pinned means or array
  hashes no longer reproduce passes the pre-read.
- Fix: parametrize `test_regression_stop` over each field (one field wrong at a time) and over CLIP items versus
  counts. Add stage 2 cases for `switched_off: true`, `dry_run: true` and another backbone. Tests only.

**MEDIUM-3: the comparer's new C13 floors are untested.** `rederive/rd9_compare.py:306-311, 222-225`.
- In the stage 2 schema, a pinned array is compared only if the derivation emitted it (`if name in arrays`). The
  `required` set is the only floor.
- Removing `{'cl','pair_index','parity','aff_fused_cells','aff_cf_cells'}` or the reader-array floor
  (`aff_gate`/`P`/`margin`/`pick` × a/b) from `required` survives `test_r9_preread.py` and `rd9_test_compare.py`.
- The same holds for removing the new CLIP B/B0/B1 seed-42 mean comparison.
- Scenario: a phase-1 derivation that omits `P__a` or `cl` gets agreement true on the stage 2 schema.
- Fix: add tests in `rd9_test_compare.py`: (a) derived arrays missing `cl`, then `P__a`, give agreement false;
  (b) the CLIP branch with a changed `B__r1` gives agreement false. `rederive/` is not a closure module.

**LOW-1: a refused gate crashes when the held runner runs as a script.** `run_r9_held.py:70, 899-900, 912`.
- With `python run_r9_held.py`, `run_r9_preread` imports a second copy, `run_r9_held`. `preread_guard` raises
  `run_r9_held.Refused`, which `__main__`'s `except Refused` does not catch.
- Demonstrated with `final_review/fr_c_main_refused.py`, which loads the runner under another module name: the
  output was "uncaught run_r9_held.Refused - successful pre-read required before launch".
- The gate fails closed: no launch. But the CLI exits 1 with a traceback instead of 4 with "refused: ...".
  `input_read_set`'s schema stop behaves the same in smoke.
- Fix: put `sys.modules.setdefault('run_r9_held', sys.modules[__name__])` under `if __name__ == '__main__':` before
  `main()`, or catch `Exception` subclasses named `Refused` from both modules. Not a closure module.

**LOW-2: a corrected held pass can deadlock on the pre-read name.** `run_r9_preread.py:420-427`.
- If a pre-read `_fix1` authorised the first launch (the base pre-read had failed), a later held `--fix 1` needs
  `preread_<Hn>_fix1.json` on the fixed bytes. That name is taken, and `preread_once` refuses to write it again.
- Scenario: a traced runner bug after the first read; the held fix can never launch without manual intervention.
- Fix: give held-fix pre-reads their own name (for example `preread_<Hn>_heldfix1.json`), or allow `_fix2` for
  pre-read reruns. Not a closure module.

**LOW-3: smoke outputs are written outside `--results`.** `run_r9_preread.py:278-279, 359-367`.
- `smoke_agreement` always writes `rederive/rd9_out/smoke/<tag>/rd9_<read>_agreement.json` into the independent
  re-derivation's folder. `publish_smoke` always writes `F/smoke_record.json`, even for scratch runs.
- The hand-back's controller command runs with `--results .../results/t22_controller`. It therefore creates the
  folder copy F of `smoke_record.json` from a scratch run. A later production pre-read either keeps that scratch
  copy (same bytes) or fails with "canonical smoke stale" (other bytes).
- The apply step refuses smoke agreements in real mode (`load_agreement` l.504), so there is no contamination.
- Fix: write the stand-in only into the smoke output folder (the apply step reads it from there). Write the F copy
  only when `results == R.RESULTS`.

**LOW-4: a launch record is not bound to this checkout's own files.** `run_r9_preread.py:436-438`.
- `input_file_sha256` is keyed by absolute path. A passing record copied from another worktree with identical code
  is accepted while that worktree's files are unchanged, even if this checkout's pins or stage 0a record differ.
- This needs a manual copy, so the risk is low.
- Fix: at launch, require `R.pins_path(backbone)`, `R.RESULTS / S0A.OUT_NAME` and (A28 reads) `AD.PINS_PATH`,
  resolved, to be keys of `input_file_sha256`. Not a closure module.

**Open item (liveness, not safety).** Stage 2 wrote the new-backbone sigma file through R6's `run_r6_sensitivity`.
The pre-read recomputes it through `r9_stats.check_diffs` and `sigma_split` and needs exact float equality. If the two
paths differ in summation order or dtype, the pre-read stops on real data. The controller's full pre-read will show
this.

None of the fixes touches the 17 stage 2 closure modules (`r9_backbone`, `r9_bundle`, `r9_common`, `r9_context`,
`r9_data`, `r9_episodes`, `r9_heads`, `r9_picks`, `r9_pins`, `r9_recipe_{a3,baselines,cells,groups,reader}`, `r9_score`,
`r9_stats`, `run_r9_stage0a`).

## 4. Guards whose deletion survives

None of the 41 `# guard:preread_*` markers. 11 deletions repeated here (stage0a_modules, stage0b, episode_hashes,
sigma, mutation, bytes, the `run_held` call site preread_launch, real_inputs, phase1_pins, compare_pins,
compare_sources) were all caught.

These sub-condition mutants survived (no marker deleted; the condition was removed from a marked statement):

| Mutant | File:line | Note |
|---|---|---|
| launch: module_sha256 half | run_r9_preread.py:433 | MEDIUM-1 |
| launch: read field | :429 | MEDIUM-1 |
| launch: backbone field | :430 | MEDIUM-1 |
| launch: all checks passed | :431 | MEDIUM-1 |
| launch: non-empty inputs | :436 | MEDIUM-1 |
| stage 2: switched_off | :126 | MEDIUM-2; `read_guard` still stops it |
| stage 2: dry_run | :126 | MEDIUM-2 |
| stage 2: backbone | :125 | MEDIUM-2 |
| regression: compare only `seed` | :183 | MEDIUM-2 |
| CLIP regression: items only | :176 | MEDIUM-2 |
| CLIP regression: counts only | :177 | MEDIUM-2 |
| phase 1: phase == 1, backbone, failed == 0, items equal (4 mutants) | :214-216 | MEDIUM-2 |
| publish: `AR.smoke_guard` re-check removed | :368 | LOW, defence in depth |
| comparer: cl/pair/parity/cells floor | rd9_compare.py:309 | MEDIUM-3 |
| comparer: reader-array floor | rd9_compare.py:310 | MEDIUM-3 |
| comparer: CLIP B/B0/B1 means | rd9_compare.py:222-225 | MEDIUM-3 |

Files of this review: `final_review/fr_c_gate_test.py` (13 launch scenarios), `final_review/fr_c_mutants.py`
(34 mutants, restored and SHA-checked), `final_review/fr_c_main_refused.py` (LOW-1 demonstration). Temporary outputs
are in `/tmp/claude-0/-project-CoSiR/18057a5a-4e60-48cd-83b0-ff14c8a319ae/scratchpad/fr_c/` (`mut_a.json`,
`mut_b.json`, `mut_c.json`, `mut_gate.json`). I did not run the full CLIP pre-read; the controller runs it.
