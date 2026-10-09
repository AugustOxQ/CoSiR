# Round 6 final review, area A (verdict path): tickets 04, 08, 09

> Reviewer A (Opus), 2026-10-09 (Amsterdam). Snapshot: worktree `/project/CoSiR-r6-fr`, integration branch
> `r6-held-test` at b97b411 (ticket 15 not in it). Rule c394b60 (`DECISION_RULE.md`, SHA-256 7444a5e3…). Scripts:
> `final_review/fr_a_stats.py`, `fr_a_e2e.py`, `fr_a_apply.py`, `fr_a_guards.py`, `fr_a_mutate.py`; their JSON
> outputs beside them (gitignored); temp files in the scratchpad `fr/a/`.

## 1. Verdict

**CONFIRMED WITH FIXES.** Every number of rule §3 and §8.2 that the verdict rests on is reproduced bit for bit by my
own code written from the rule's text, both on crafted real-shape arrays through `r6_stats.pass_record` and on the
end-to-end synthetic held read through `run_r6_held.run_held`; the verdict step agrees with my own reading of §4 on 48
crafted pass files; every §8.1 and §9 refusal fires through the real entry points without writing a file. Two
safeguards the rule or the contracts name are not in the snapshot's code (findings 1 and 2): the apply step's
module-SHA check against the smoke record, and a pre-read check of the DTS stop (§6 item 6) in the held runner.

## 2. What I re-derived

| Check | How | Mine vs the code's | Result |
|---|---|---|---|
| §3 draws and ci95 | own loop from §3's text (unique/return_inverse clusters, rng 42, chunks of 250, Σ sums / Σ counts), against `cluster_bootstrap`'s own ci95 | equal, every check, every array | exact |
| §3 counts n_j | integer 4·Σ cluster sums ≤ 0 (own `np.add.at` int64), also the float count | equal to `held_pass.json`'s n_j in all 45 checks of 5 runs; float count = integer count everywhere | exact |
| §3 Holm | own integer Holm (order by n, ties P1..P7 / S1 first; stepdown; 40·(n+1)·(m+1−k) ≤ 5,001), also the float p-value form | 25,237 crafted count vectors (every n*_k − 1, n*_k, n*_k + 1 per rank under 60+ rank permutations, ties, all-equal, wide counts), P and S: 0 disagreements; float form never differs (40·(n+1)·(8−k) is never 5,001) | exact |
| §3 intervals | `numpy.percentile` at 100·0.025/(m+1−k), 100·(1−0.025/(m+1−k)), ×100 after; level 1 − 0.05/(m+1−k) | every `ci95`, `ci_holm`, `level_two_sided`, `holm_k`, `near_boundary`, `passes`, `own_count_passes`, `holm_order` equal | exact |
| §3 on real shapes | `pass_record` (mode held, 3 × 12,288 episodes, 8,770 Pareto-sized anchor clusters, quarter-valued differences) with planted shifts: GO at the boundaries (16, 19, 24, 30, 40, 61, 1; S 60, 124), first rank one over (17), a middle failure (31 at k 4), ties (12, 12, 12; 0, 0; S 3, 3) | 0 field differences in 4 runs | exact |
| §3 end to end | `run_held` on test_r6_held_synthetic's real-shape env (36,864 episodes, 13,506 anchor paintings): my numbers from `held_arrays.npz` against `held_pass.json` | 0 field differences (n, point, ci95, ci_holm, k, level, flags, order, n_clusters) | exact |
| §8.2 | own one-way split (R3 §6.1) and SE² = (σa²ΣM_p² + σε²N)/N², x = 3.532 SE, x₂ = 3.083 SE, x95 = 2.80 SE; z values from scipy (3.5317, 3.0830) | σ parts relative difference 0; SE, x, x₂, x95 relative difference 0 (via `sensitivity_held` and in the end-to-end `sensitivity_held.json`); N 36,864 and n_paintings equal; P has `x`, S has `x2` | exact |
| §4, §9 | own decision function from §4's text against `run_r6_apply_rule.main` on 48 crafted pass files: GO with S both/one/none passing, S ties, S not reached, S did not beat; each P failing alone at rank 7 with point > 0, < 0 and = 0, each P at n*_7 (passes); Holm stops at ranks 1, 2, 4 with the rest not reached; a not-reached check with point < 0; mixed NO-GO; all-inconclusive NO-GO; NO-GO whose S counts would pass | verdict, kind, every check's pass, reading, did-not-beat name, not-reached mark, S tested flag, licence, claim text, boundary report: 0 differences; a second run refuses and the file is unchanged in all 48 | exact |
| §8.3 | keys of `held_arrays.npz` and of the scores in the end-to-end read | exactly `CORE_SCORERS` × metrics + cl, pair_index, seed_index; no PM, no `r1_cf`; pass file has no per-seed or per-pair number | as the rule |

## 3. Findings

1. **`run_r6_apply_rule.py:357-395` (`apply`), should-fix (fix before the run).** Contracts §8 amendment 12:20 says
   the apply step, in real mode, asserts `r6_module_shas()` equals the latest smoke record's `module_sha256` and
   refuses (exit 4) otherwise, "T15 adds this check to both". The snapshot has it in the runner (`smoke_guard`) but not
   in the apply step: a real apply with a stale smoke record beside it writes the verdict (my case
   `stale_smoke_record_exit4_expected`: exit 0). Ticket 15's text does not list it; its controller note (16:23) reads
   as if the apply step already had it, so it can fall between T09 and T15. Failure in the run: an edit of
   `run_r6_apply_rule.py` (or any r6 module) after the smoke goes unnoticed and decides the verdict (rule §6.7 "any
   change to these files afterwards needs a new smoke"). Fix: in `apply()`, real and reserve mode only, after
   `assert_rule()`: pick the latest record as the runner does (`smoke_record_reserve.json` for `--reserve`;
   `smoke_record_fix1.json` when it exists; else `smoke_record_crash1.json` when it exists; else `smoke_record.json`),
   refuse unless it exists, has `"passed": true` and `module_sha256 == CM.r6_module_shas()` key for key; tests on tmp
   copies (stale, missing, failed, extra key; smoke mode not checked). Best: move `latest_smoke_record` and the
   comparison to `r6_common` so the runner, the apply step and the descriptive pass share one copy.
2. **`run_r6_held.py:1032-1037` (`run_read` step 3) and `seed42_guard` (:780), should-fix (confirm with reviewer C).**
   Rule §6 item 6 (the DTS stop of §7.6, §7.7) is a pre-read check whose failure "stops the work before the read", and
   §4's GO needs "§6's pre-read checks all passed". The held runner checks items 2 to 5 and the smoke record (item 7),
   never `dts_stop.json`; in the snapshot nothing in code refuses a read after a fired or unbuilt DTS stop. Ticket 15
   step 1 refuses the smoke unless the stop output "exists"; if it does not also require no stop and built, a
   smoke record can be `passed` with the stop fired. Fix: in `seed42_guard`, refuse unless `results/dts_stop.json`
   exists with `"built": true` and `"stop": false` and was written by the current bytes of `run_r6_dts.py`'s modules
   (`stale_modules`), and record its SHA-256 in the attempt; one test per clause.
3. **Rule §9 and §10.6 time-box, nit.** "The read has not started by Thu 2026-10-15: no read starts" is not enforced
   in code (process only). Optional: held mode (first read only) refuses after 2026-10-15 23:59 Amsterdam.
4. **`run_r6_apply_rule.py:299-307` `rederivation_n`, nit (deferred T09).** It is the runner's count copied, labelled
   as the re-derivation's; §8.5 asks for both counts. Equal by the phase-2 agreement (discrete quantities identical),
   so harmless; rename to `rederivation_n_by_agreement` or have the agreement record carry the re-derivation's n_j.
5. **Observation for reviewer B (no defect found).** On the synthetic end-to-end read `r1_fused__r1` equals `B__r1`
   bit for bit (gates closed on random features), so there S2 and P3 coincide; S2's wiring is pinned only by the
   seed-42 regression (r1_fused arrays and AFF − R1 against round 3), which the run chat reruns before the smoke.
   Held bundles also carry the nine PM terms before the verdict (no PM metric); rule §8.3 lists the held bundles as
   allowed, so I read it as within the rule.

## 4. Guards

Three sources: (a) my own scenarios through the real entry points (`fr_a_guards.py`: 53 runner scenarios through
`run_r6_held.run_held` on temp results, ledger and folder; `fr_a_apply.py`: 47 apply-step cases through
`run_r6_apply_rule.main`; 12 subprocess runs of both scripts from the worktree, all refused with exit 4 or 2 and no file
written); (b) the folder's own deletion mutants (`test_r6_held.py`, 95 passed here, one per `# guard:` marker); (c) my
36 semantic weakenings (`fr_a_mutate.py`), each on a copy of the folder, its test file run against the copy.

- Runner §8.1: verdict exists, pass exists (also with `--after-crash`), started exists, after-crash without a start,
  third attempt, H5 missing / twice / other SHA / this SHA not last / upper-case SHA / report cell filled or "pending",
  episode cell filled on a first read, committed copy exists, copy missing or different on `--after-crash`: **all fire**
  (exit 4, nothing written); `MAX_ATTEMPTS = 3`, SHA "anywhere in the cell", after-crash past an existing pass:
  **caught**.
- Runner `--fix 1`: needs the pass, refuses its own pass, needs `smoke_record_fix1.json`, refuses a verdict, needs the
  nine H5 hashes (pending or other hashes refused), refuses a second fix, stale fix-1 record: **all fire**.
- Runner `--reserve`: needs H5-R, needs `held_verdict.json`, refuses its started / verdict / pass file, its smoke record,
  partial hashes, an existing copy, a filled H5-R report cell: **all fire**.
- Runner module SHAs (exit 4): record missing, failed, one SHA stale, a module missing from the record, an extra module
  in the record, `smoke_record_crash1.json` used for `--after-crash`, `smoke_record_fix1.json` counted when present:
  **all fire**; weakening the comparison to the record's keys, ignoring crash1 or fix1: **caught**.
- Runner seed-42 currency: regression stale or failed, sensitivity not from the current regression, refit stale:
  **all fire**.
- Runner read order and scope: `include_pm=True` in the read, per-seed N in §8.2, skipped rerun-hash comparison:
  **caught**.
- Apply step: agreement missing; phase 1, "2", 2.0, true, missing; n_quantities 0, "9", true, missing; disagreements
  missing, null, {}, non-empty; pass_file wrong or missing; all_agree false, "true", 1; smoke flag true, "false",
  missing; SHA binding; real agreement in smoke mode; pass rule SHA, mode smoke or regression, seeds; tampered flags,
  order or boundary flag (exit 5); verdict exists; sensitivity missing; `--smoke-subdir` in real mode; `--reserve`
  without the original verdict or its own sensitivity; a fix-1 pass without its own agreement or with one naming the
  first pass: **all fire**; fix-1 pair, `--reserve` and `--smoke-subdir fix1` controls write their verdict.
- Apply module SHAs against the smoke record: **does not exist** (finding 1).
- Semantic weakenings caught (30 of 36): integer count `< 0`; Holm not stepdown; ties reversed; Holm interval at 0.05
  or at the wrong rank; level at the wrong rank; strict near-boundary; Z_P, Z_S; Σ M_p not squared; ×100 before the
  percentile; point ≥ 0 read as inconclusive; secondary tested after a NO-GO; passes-consistency check removed; fix-1
  pair ignored; P4's name; failed secondary licensed in the claim; kind "any" inconclusive; not-reached marked at the
  last failure; smoke flag ignored; and the runner's eleven above.
- **Not caught (6, test gaps; the code is right, shown by my own checks):** P7's comparator rca → cosine and S2's
  r1_fused → B in `r6_stats.QUANTITIES` (test_r6_stats passes; my `fr_a_stats.py` part 2 pins all nine with distinct
  comparators, and the C13 phase-1 re-derivation is the run's backstop); the apply step's `disagreements` "present"
  half and `n_quantities >= 1` (the real code refuses both, my cases); the did-not-beat names of P1 and P5 (deferred
  T09 nit; my 48-case table pins all nine names).

## 5. Hard rule 6

| ID / section | Status | Where |
|---|---|---|
| §3 draws, counts, Holm, intervals | covered | `r6_stats.bootstrap_draws`, `holm`, `holm_interval`, `pass_record` (re-derived exact) |
| §4 GO / NO-GO, C11 readings, not reached, kind, passed-in-NO-GO, claim | covered | `run_r6_apply_rule.decide`, `read_family`, `c11` (48 cases) |
| §4 secondary gatekeeping (Holm m = 2, ties S1 first, x₂, licences) | covered | `decide` (`if go`), mutation caught |
| §5.5 head checks before `held_started.json`, coefficient SHAs recorded and asserted | covered | `head_guard`, `start_attempt` record; post-row scope `read_seed_bundle` |
| §6.7 smoke-agreement clause | covered | `load_agreement` smoke flag, both directions tested |
| §6.7 / contracts §8 12:20 module SHAs in the apply step | **missing** | finding 1 |
| §6 item 6 before the read (DTS stop) | **missing** in the runner (maybe partial via T15) | finding 2 |
| §8.1 refusals, ledger H5, started file and copy, attempts, after-crash, fix 1 | covered | `refuse_or_go`, `ledger_guard`, `smoke_guard`, `EpisodeRecorder` (53 scenarios, entry runs) |
| §8.2 | covered | `sensitivity_held`, `r6_stats.detectable` (exact) |
| §8.3 order and allowed quantities | covered | `run_read` (allowed_keys guard), `score_seed(include_pm=False)`; no verdict written |
| §8.4 agreement before the verdict | covered (code side) | `load_agreement`, SHA binding |
| §8.5 boundaries | covered | `near_boundary`, `boundary_report` (S only after a GO) |
| §9 crash / fix / reserve rows, never overwritten | covered | `read_names`, `write_new`, `keep_or_write_*`, `write_once` |
| §9 time-box | partial (process only) | finding 3 |
| SC4, SC5, D6, D7, D16, D27 | covered | as §3, §4 |
| D2, D8 | covered | `HELD_SEEDS`, `N_PER_PAIR`, `_seed_ok`, `held_seed_bundle` on `split.held` |
| D21 safeguards in these files | partial | all but findings 1 and 2 |
| D26 reserve read | covered | `--reserve` (H5-R, `_reserve` names, its own smoke record, same episodes) |
| Contracts §6; §7 amendments 05:45, 16:22, 16:43, 17:15; §8 amendments 05:45, 12:20 | applied in every writer and reader I found, except 12:20's apply-step half (finding 1) | runner, apply step, `run_r6_descriptive` (reads `mode`, `pass_file`, `agreement_file`, `agreement_sha256`) |
| Unrequested | none of concern: stricter agent defaults in the runner (documented in its docstring) | |

## 6. Deferred nits (T04, T08, T09)

- T09, "did not beat" names of P1, P2, P3, P5 not pinned: **fix now** (one parametrized test over all nine names; P1
  and P5 mutants survive here). Cheap, and a wrong name would put a wrong sentence in the verdict file.
- T09, the real-mode `--smoke-subdir` clause is only refused by a later clause: **leave**. The clause exists
  (`out_dir`, :340) and fires through `main` (exit 4); with it deleted, the smoke-flag check still refuses.
- T09, `rederivation_n` copied from the runner's count: **leave or rename** (finding 4); equal by the agreement.
- T04, T08: no deferred nits listed. New test gaps from section 4, **fix now** (tests only, no code change): pin
  `r6_stats.QUANTITIES` literally against rule §3 / §4's table; apply-step tests for `disagreements` missing or null and
  `n_quantities == 0`.

## 7. Storage

Scratch outputs are in `/tmp/claude-0/-project-CoSiR/714507bd-709f-400b-92b9-40bd3c709f3b/scratchpad/fr/a/` (80 MB
after I removed my own pytest temp folders, 630 MB); nothing was written in the worktree outside `final_review/`, no
`.pyc`, no file in the main checkout.
