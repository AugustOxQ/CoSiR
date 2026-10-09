# Round 6 scoped re-review of the fix wave and ticket 15's fix round

> Re-reviewer (Opus), 2026-10-09 23:40. Worktree `/project/CoSiR-r6`, branch `r6-held-test` at 4fc1a69. Scope: the fix
> wave `25660d6..ec27ee2` (five items of `final_findings.md`) and T15's fix-round commits 59dd4d9 and a186c2c. Scripts
> `final_review/rr_*.py`; outputs in the session scratchpad `rr/`. CPU only, one process at a time, no held row
> indexed, F/results unchanged (listing compared before and after), no `.pyc` written, no tracked file edited.

## 1. Verdict

**CONFIRMED.** Every fix does what its item asks, through the real entry points: the value-sets file, the time-box,
the git provenance, the settings-copy check, and T15's waivers, retries, stop binding, first-built time and pinned
clock. I re-derived each load-bearing decision with my own code and found no difference. The 28 mutations of the new
guards are all caught except three. Those three are test gaps on code that is correct, so they are nits, and so is
the scope of the settings-copy guard (finding 1). The full r6 suite passes: 24 files, 828 passed, 0 failed.

## 2. Re-derived

| Check | How (my code) | Mine | Code's | Result |
|---|---|---|---|---|
| Rule §5.2 development value sets | `rr_value_sets.py`: labels coded from the annotations (sorted names, catch-all excluded, genre from the WikiArt CSVs), groups and selection from `prepare.npz` (SHA asserted), pool = selection rows with all three labels, eligible at ≥ 30 distinct paintings; no feature loaded | 8 / 23 / 10 (pool 23,580 of 32,413 selection rows; no New_Realism) | `RH.load_rows()` then `R.write_value_sets` into a temp folder, as `run_regression` calls it: 8 / 23 / 10, SHA-256 4ea6d338… | equal (codes and names) |
| DTS budget, fresh build at 2026-10-10 12:29:00 | `rr_dts_budget.py`: crafted seed-42 records; later of sanity and chosen ≤ start + 24 h elapsed | built, no stop; first-built written | `record_first_built` "written"; `main --stage stop` exit 0, source `dts_first_built.json`; held guard accepts | equal |
| … chosen at 12:29:01 | same | not built, stop | first-built not written; exit 3, "not built within the 24-hour budget"; held guard: fired | equal |
| … sanity at 12:29:01, chosen at 12:00 | later of the two | not built, stop | exit 3 | equal |
| … in time, hits 9,407 / 9,406 | hits > 9,406 | stop / no stop | exit 3 / exit 0 | equal |
| Rerun on 2026-10-12, setting, settings SHA and fingerprints unchanged | first build's time (2026-10-10 09:00) | built, no stop | exit 0, `budget_time_source` = first-built, built time 09:00:00; held guard accepts | equal |
| Rerun with changed setting / settings SHA / sanity outputs / chosen outputs | the rerun's own time | not built, stop (4 cases) | exit 3, source "this run's sanity and chosen records" (4 cases) | equal |
| Seed-42 stop with another clock start | refused, nothing written | refused | exit 2, no `dts_stop.json` | equal |
| Time-box, real `amsterdam_today()` with a faked UTC clock, process TZ UTC and America/New_York | Amsterdam date = UTC + 2 h (CEST); boxed iff no flag and date > 2026-10-15 | UTC 21:59 → 10-15 allowed; UTC 22:00 → 10-16 boxed; UTC 10-14 23:30 → 10-15 allowed; UTC 10-15 23:30 → 10-16 boxed | `rr_time_box.py` through `run_r6_held.main(["--mode", "held", …])`: 40 of 40 agree; every flag passes the box; exit 4 and nothing written in every case | equal |
| `r6_stats.QUANTITIES` against rule §3/§4 | my reading of the table: P1 R@1 −COS, P2 R@1 −RCA, P3 R@1 −B, P4 R@1 −B0, P5 R@1 −CF, P6 gain −CF, P7 gain −RCA, S1 R@1 −B1, S2 R@1 −R1 (fused); every one AFF minus it | 9 pairs | `("r1","cosine")`, `("r1","rca")`, `("r1","B")`, `("r1","B0")`, `("r1","aff_cf")`, `("gain","aff_cf")`, `("gain","rca")`, `("r1","B1")`, `("r1","r1_fused")`; `check_diffs` computes AFF (`aff_fused`) minus the comparator | equal |
| `DTS_CLOCK_START` | `git log` of the round folder: first DTS commit 6beb360 at 12:29:20 Amsterdam; log line 32 | 2026-10-09 12:29 | `R.DTS_CLOCK_START` = `RD.DTS_CLOCK_START` = "2026-10-09 12:29" | equal (20 s stricter than the commit second) |

## 3. Findings

1. **nit**, `run_r6_dts.py:241-261` (`settings_copy_guard`), fix wave item 5. The implementer says `list-input --for
   sanity` is the first seed-42 GPU input in the order the run follows. That holds for the rule's stage order (§7.4)
   and for `run_r6_dts.py`'s docstring, but the code does not enforce it. A seed-42 verbaliser job folder
   (`results/prep_gpu/jobs/s42_verbalise_prestage`) already exists, `r6_gpu_inputs.py` and `scripts/run_r6_verbalise.sh`
   do not check the copy, and `notes/gpu_path.md` lists the verbaliser (J1) before the listing job (J2). Failure in the
   run: a seed-42 tuning verbaliser launched before `list-input --for sanity` makes the first seed-42 model call with
   no committed `results/dts_settings.json`, and nothing refuses it. No number changes, since every record hashes
   F's settings file. The run handoff's step 7 already copies the file first ("First copy F's `dts_settings.json` …"),
   so the procedure covers it. Fix (optional): one handoff line, "no seed-42 GPU job, the verbaliser included, before
   that commit", or the same check in `r6_gpu_inputs.py` when the episodes' seed is 42.
2. **nit**, `test_r6_held.py:874-876`, fix wave item 3. No test fixes the time-box clock's timezone deterministically.
   A mutant that reads the date in UTC survives the whole `test_r6_held.py` (run at 23:32 Amsterdam). It is caught only
   between 00:00 and 02:00 Amsterdam, when the two dates differ. The code is right (`rr_time_box.py`: UTC 2026-10-15
   22:00 gives 2026-10-16 and is refused). Fix: a test that patches `RH.datetime.now` to UTC 2026-10-15 22:00 and
   expects 2026-10-16.
3. **nit**, `run_r6_dts.py:443` and `:518`, T15 fix round. Nothing tests the "later of the sanity and chosen
   records' times" rule. Mutants that take the chosen record's time alone, in `record_first_built` or in the stop,
   both survive the whole `test_r6_dts.py`, because every test has sanity before chosen. The code is right (my case:
   sanity 2026-10-10 12:29:01, chosen 12:00:00, stop). Fix: one stop test with sanity later than chosen.
4. **nit**, `run_r6_held.py:895-933` (`dts_stop_problem`). When the stop took its built time from
   `dts_first_built.json`, the held guard binds sanity, tune, chosen and the per-anchor file, not that file. The
   stop record holds `first_built_sha256`, and `write_new` never overwrites the file, so only a hand edit could
   change it. Fix (optional): compare `first_built_sha256` with the file when `budget_time_source` names it.
5. **nit, leave**, `run_r6_held.py:719-724`. The time-box is checked once, at step 1, and the first write comes after
   `setup()` (minutes later). A launch at 23:58 on 2026-10-15 can write `held_started.json` after midnight. This is
   defensible under "has not started by", and the read is planned days earlier.

No blocking or should-fix finding. Other checks, all clean:
- Only F files changed (14 in the fix wave, 7 in T15's fix round). `DECISION_RULE.md` and `dts_settings.json` are
  untouched. No number, setting or decision logic changed beyond the five items and the T15 round.
- The new fields are additive: `value_sets_sha256`, the attempt's `value_sets` and `git`, the pass record's
  `value_sets`, `seed42_records_sha256` gaining `value_sets.json`, and the stop's `budget_time_source` and
  `first_built_sha256`. No consumer checks a record's exact key set (`read_attempts`, the apply step, the descriptive
  pass). The smoke chain end to end (`test_r6_smoke_e2e.py`) and the synthetic held read pass with them.
- Item 4's sensitivity part is covered by the existing `guard:per_episode_sha` (`run_r6_sensitivity.py:61`), which
  binds `seed42_per_episode.npz` to the regression record. The value-sets file must also match the SHA-256 the
  regression record names (stricter than the rule; agent default, accepted). Smoke seeds are exempt from the copy
  check, and the smoke's started file gets git provenance: both harmless.

## 4. Guards

Each guard was mutated on a copy in the scratchpad and loaded by a pytest plugin under the module's name. "Caught"
means a test fails that also passes on an unmutated copy (control runs: 0 failures). `rr_mutate.py`,
`rr_mutate.json`.

- value_sets_file (step 3 refusal removed): caught, 6 tests
- value_sets_equal, SHA check only (content compare dropped): caught, 5
- value_sets_equal, content only (SHA compare dropped): caught (`test_value_sets_changed_after_step_3_refuses`)
- `value_sets_problem` without the SHA compare: caught, 6
- regression does not write `value_sets.json`: caught, 9
- regression record without `value_sets_sha256`: caught, 2
- smoke step 1's value-sets check removed: caught
- attempt without `value_sets`: caught
- time-box `<` for `<=` (refuses on 10-15): caught (`test_a_first_read_starts_until_2026_10_15_and_not_after`)
- time-box one day late: caught
- time-box applied to the flags: caught, 3
- time-box clock in UTC: **not caught** (finding 2)
- git provenance not recorded: caught, 2
- `settings_copy_guard` call removed: caught, 2
- settings copy existence only (bytes not compared): caught
- first-built match without the fingerprints, the settings SHA, or the setting: caught (each)
- first-built written outside the budget: caught
- first-built never used: caught
- built time = the chosen record's alone (stop, and `record_first_built`): **not caught** (finding 3)
- `DTS_CLOCK_START` moved to 13:29: caught
- held guard's clock-start check removed: caught
- held guard's input binding removed: caught (`test_r6_held.py`; the smoke step-1 tests alone do not catch it)
- smoke family not given marked passed: caught
- DTS retries only on 9001: caught
- smoke mutation counted on any error: caught

28 mutations, 25 caught, 3 not caught (findings 2 and 3: test gaps on code that is correct).

## 5. Hard rule 6

| Item | Status | Where |
|---|---|---|
| 1, B1: `results/value_sets.json` written by the regression, SHA in its record; held (every flag) and smoke refuse at steps 3 and 4; SHA in the attempt and the pass; smoke step 1 lists it; tests missing and changed | covered | `run_r6_held.py` `run_regression`, `value_sets_problem`, `value_sets_file_guard`, `value_sets_guard`; `run_r6_smoke.py` `_chain_problem`, `stage_problems`, `wiring_mutation`; tests in held, regression, smoke and seed-42 regression |
| 2, A nits, tests only: `QUANTITIES` pinned to §3/§4; apply tests for `disagreements` missing, null, non-empty and `n_quantities == 0`; nine did-not-beat names | covered | `test_r6_stats.py` `_check_quantities` and 5 text mutants; `test_r6_apply_rule.py` refusal cases, `TEXT_MUTANTS`, `test_did_not_beat_names_follow_rule_section_4` |
| 3, §9 time-box: first read refuses after Amsterdam 2026-10-15, exit 4, nothing written; flags and smoke not boxed | covered (test gap on the clock's timezone, finding 2) | `run_r6_held.py` `amsterdam_today`, `READ_DEADLINE`, `refuse_or_go` |
| 4, B nits: sensitivity input SHA bound; `held_started.json` records git HEAD and `git status --porcelain -- src` | covered (sensitivity by the existing `guard:per_episode_sha`) | `git_provenance`; `run_r6_sensitivity.py:61` |
| 5, C4: the first seed-42 DTS GPU input refuses without `results/dts_settings.json` equal to F's bytes; smoke exempt | partial: `list-input` only, not the verbaliser input writer (finding 1; the handoff orders the copy first) | `settings_copy_guard` |
| T15 fix round: waivers, DTS retries 9002 then 9003 then missing, record-guard negative tests, stop bound to all its inputs (smoke and held), first-built written once and used only while unchanged, clock pinned | covered (test gap on "later of", finding 3) | `run_r6_smoke.py` `stage_finish`, `dts_family`, `require_copy_free`, `listing_writer_check`; `run_r6_dts.py` `record_first_built`, `first_built`, `stage_stop`; `run_r6_held.py` `dts_stop_problem` |

No item contradicts its brief. Nothing unrequested beyond T15 review nits (`require_copy_free`, the listing-writer
smoke), which that review asked for.

## 6. Suite

Run file by file, `pytest -q -p no:cacheprovider`, logs in the scratchpad `rr/suite/`. `test_r6_common.py` ran with
`-k "not worktree"`. No heavy file was skipped.

| File | Result |
|---|---|
| test_r6_stats.py | 36 passed |
| test_r6_apply_rule.py | 108 passed |
| test_r6_held.py | 138 passed |
| test_r6_dts.py | 64 passed |
| test_r6_smoke.py | 30 passed |
| test_r6_regression.py | 38 passed, 1 skipped |
| test_r6_common.py | 34 passed, 4 deselected (worktree tests) |
| test_r6_episodes.py | 33 passed |
| test_r6_heads.py | 31 passed |
| test_r6_context.py | 28 passed |
| test_r6_bundle.py | 9 passed |
| test_r6_score.py | 36 passed |
| test_r6_external.py | 41 passed |
| test_r6_descriptive.py | 49 passed |
| test_r6_gpu.py | 80 passed |
| test_r6_gpu_inputs.py | 14 passed |
| test_r6_gpu_t12.py | 20 passed |
| test_r6_sync_ckpt.py | 12 passed |
| test_r6_held_synthetic.py | 5 passed |
| test_r6_held_smoke.py | 1 passed |
| test_r6_regression_seed42.py | 3 passed |
| test_r6_picks_seed42.py | 8 passed |
| test_r6_descriptive_seed42.py | 8 passed |
| test_r6_smoke_e2e.py | 2 passed |

Totals: 828 passed, 1 skipped, 4 deselected, 0 failed.

Storage: the scratchpad `rr/` holds the suite logs, the crafted DTS worlds, the mutation copies and their pytest temp
folders, and one `value_sets.json`. All of it is small, temporary and deletable. Nothing was written to F/results.
