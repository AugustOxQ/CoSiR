VERDICT: PASS WITH NOTES

# Review of fix ticket fx-inf (worktree r9-fx-inf, commit 61431a1 on main d926d9e), 2026-10-14

The fix is correct and the tests catch every mutant I tried. I found no HIGH finding. The one MEDIUM finding is about
the run plan, not the code: three generations leave little room if the SigLIP smoke chain needs a code fix.
Paths below are relative to F = `src/test/20261202_r9_scope` in the worktree unless absolute.

## Findings

### HIGH
None.

### MEDIUM
**M1 (plan, not code). Generation 3 is the last name, and the SigLIP and Qwen-A28 smoke chains have never run.**
- The chains: the Qwen H9 path, and the SigLIP H7 and H10 paths.
  - The H6 base pre-read stopped in the H6 descriptive, before the H9 chain (`run_r9_preread.py:319`, loop over
    held_read, then a28_read).
  - SigLIP's smoke has never run.
- The scenario:
  1. Fix 2 is spent on H8, H6 and H9.
  2. The H7 pre-read (which must come before the H8 launch, rule 7.3) then needs a code fix.
  3. Every pre-read and the shared `smoke_record_fix2.json` become stale, and every read that has not yet passed its
     apply step goes to fix 3.
  4. A second such fix leaves no name. `--after-crash` cannot help: crash1 ranks below every fixN in all three
     selections (`run_r9_preread.py:430`, `run_r9_held.py:500`, `run_r9_apply_rule.py:566`).
- The cap of 3 is what the brief asked for. Decision rule section 6 sets no limit, so a fourth generation would be a
  new code change.
- Mitigation: run the scratch H6 pre-read first. It runs both Qwen chains, H6 and H9, so it tests the never-run H9
  path. Run a scratch H7 pre-read (`--results` a fresh folder) as soon as SigLIP stage 2 passes, before spending
  real names on SigLIP.

### LOW
**L1. A late code change is caught only at the end of a pre-read, and the read loses that name.**
- `publish_smoke` (`run_r9_preread.py:359-373`) checks that the existing `smoke_record_fix2.json` is current only
  after the full pre-read.
- The scenario: any r9 file changes between H8's fix 2 and the H7 or H10 fix 2. H7's fix 2 then runs for its full
  time and stops with "canonical smoke stale". `preread_H7_fix2.json` is then taken (failed).
- Before each later fix-2 pre-read, the controller should check that `results/smoke_record_fix2.json`
  `module_sha256` equals the current `r9_module_shas()`. If it does not, go straight to fix 3.
- The same applies at the apply step. `run_r9_apply_rule.py:650` requires the latest smoke record to match the bytes
  current at apply time. So no round-folder code change may happen between a read's launch and its apply step
  without a new smoke generation. This behaviour existed before this fix.

**L2. `report_plain` is global, so a NaN in a descriptive statistic is now written as "nan" instead of stopping the
run.**
- Code: `run_r9_descriptive.py:66-69`, used at `:527`.
- Example: `r3_fusion.open_shares` returns `float("nan")` for a pair with no episodes
  (`/project/CoSiR/src/test/20261121_round3_affect_gate/r3_fusion.py:96`). Under `allow_nan=False` the run used to
  stop; now it writes "nan".
- The brief allows this ("at the point where the report is built or serialised"), and the descriptive pass decides
  nothing. A hidden statistic bug would show as "nan" rather than as an exit. Accepted as is; recorded for the
  user.

**L3. The committed copy must be committed.**
- H8's `--fix 2` writes `F/smoke_record_fix2.json` beside the code (`run_r9_preread.py:362-363`). `.gitignore` does
  not ignore it (checked with `git check-ignore`).
- Rule 6 item 5 wants it committed. Add it to the controller's step after H8's fix 2, as was done for fix 1 (commit
  7e7ded6).

**L4. Concurrent first publishers race, as the hand-back says.**
- Two pre-reads that both find no `smoke_record_fix2.json` both reach `write_once`. `AR.write_once` uses an
  exclusive `os.link` (`run_r9_apply_rule.py:495-505`).
- The loser stops with "exists". The stop is caught at `run_r9_preread.py:410`, and the loser's name
  `preread_<Hn>_fix2.json` is taken (failed).
- The hand-back's order (H8 alone and finished first) prevents this. No code change is needed.

## Item 1: the inf fix

- **Source.** Confirmed as follows:
  - `r9_picks.py:98` turns "inf" into `float("inf")`.
  - `run_r9_descriptive.py:181` (`P.frozen_lambdas`) reads it from baselines.
  - `:238` puts it in `frozen`, `:521` puts it in the report, and `:527` dumps the report with `allow_nan=False`.
- **Real H6 failure.** The failed `preread_H6.json` holds `ValueError: Out of range float values are not JSON
  compliant: inf`.
- **Real Qwen file.** On the real `baselines_seed42.json`, `parse_lambda_picks` gives `wang: {0: 16.0, 1: inf}` and no
  other non-finite value. The old `RH._plain` dump fails; the new dump writes `{"0": 16.0, "1": "inf"}`.
- **Only the dump changes.** The scoring lambda stays `float("inf")`, because `score_inputs` passes `lambdas` to
  `S.score_seed` before the report is built.
- **The real picks file.** `picks_seed42.json` also holds `lambda_picks.wang.1 = "inf"`. `load_picks` returns only
  B, B0 and B1, so that value never reaches the report.
- **No finite byte changes. Proof:**
  - `report_plain(x) = json.loads(json.dumps(_norm(x)), parse_constant=f)` and
    `_plain(x) = json.loads(json.dumps(_norm(x)))` (`run_r9_held.py:105-107`).
  - `parse_constant` is called only for the tokens Infinity, -Infinity and NaN. With no such token the two are
    identical.
  - Check on a real report: the real CLIP report `results/smoke/descriptive/descriptive_H8_H8_fix1_H8.json`
    (3,954,146 bytes) serialises byte for byte the same through both functions.
- **Nothing else changed.** `run_r9_held._plain` and the held, apply and pre-read writers are unchanged. No pass,
  verdict, pin or score changes.
- **Held mode is covered.** The held path for H6, H9, H7 and H10 uses the same `run()`, which dumps at the same
  line, `:527`.
- **No other non-finite value from the frozen inputs.**
  - Cells are Python int pairs (`r9_score.py:105-117`).
  - Taus come from the pinned `rc_tau`.
  - The report embeds sensitivity only as source and digest.
  - Lambda parsing admits only finite grid values or "inf" (`r9_picks.py:199-204`).
  - SigLIP's `baselines_seed42.json`, `picks_seed42.json`, `cells_seed42.json` and `rc_tau.json` now exist and hold
    no "inf", "nan", Infinity or NaN (grep over all recipe JSON files).
  - Statistics (`r9_descriptive.py`) are finite by construction except for the empty-pair case in L2. Any
    non-finite value would now be encoded, not raised.

## Item 2: the generations

- **Names and CLI.** All as asked:
  - CLI: `--fix` choices (1, 2, 3) at `run_r9_preread.py:460`; kind `f'fix{a.fix}'` at `:466`.
  - VARIANTS: `:37`.
  - Name patterns: `preread_name` at `:71-72`, the own proof at `:406` and the smoke tag at `:320`.
  - Canonical names: `SMOKE_RECORDS` fix2 and fix3 at `run_r9_apply_rule.py:121`.
  - `--fix 1` gives the same kind 'fix1' and the same names as before.
  - `run()` does not require an earlier pre-read (`:376-381`).
- **Selection order.**
  - Both `latest_smoke_record` functions use heldfix1, fix3, fix2, fix1, crash1, then smoke
    (`run_r9_held.py:500`, `run_r9_apply_rule.py:566`). The reserve branch and the held `--fix 1` branch are
    unchanged (`run_r9_held.py:498-499`, `run_r9_apply_rule.py:564-565`).
  - The launch gate uses fix3, fix2, fix1, crash1, then base (`run_r9_preread.py:430`). The heldfix1 and reserve
    branches are unchanged (`:424-426`).
- **Protections kept.**
  - Exclusive writes: `write_once` uses `os.link`.
  - Selection is by existence, not by passed, so a failed or stale newer record blocks an older pass. Mutant M9
    confirms this.
  - Staleness is checked in `run_r9_held.py:507-513`, `run_r9_apply_rule.py:582-587` and
    `run_r9_preread.py:438-441`.
  - `publish_smoke` keeps a current canonical record and refuses a stale one, for both the results record and the
    committed copy (`:365-370`).
  - `run_r9_held.py:951` still allows only held `--fix 1`.
- **Other kind lists.** A grep of F, `rederive/` and `final_review/` found nothing missed:
  - `run_r9_preread.py:212` (`_fix1`, `_crash1` phase-1 agreement suffixes) and `run_r9_descriptive.py:91-97`
    (held pass variants) belong to other naming systems and are correctly left alone.
  - `test_r9_common.py:543` (gitignore check) lists only examples.
  - `rederive/rd9_commands.md:101-107` is updated. `fx_held_mutants.py` anchors are updated.

## Item 3: the planned sequence, after the merge and a stage 0a rerun

Starting state: every pre-read and smoke record is stale, because `r9_module_shas` covers every `run_r9_*.py`
(`r9_common.py:346-355`). Real files:
- `preread_H6.json`: failed.
- `preread_H8.json`: failed.
- `preread_H8_fix1.json`: passed, stale.
- `smoke_record_fix1.json`: passed, stale; F copy stale too.

Stage 0b and stage 2 stay valid, because `S2.assert_modules` checks only the recipe module set
(`run_r9_stage2.py:156-162`).

**(a) `--read H8 --fix 2`, alone.**
- New names: `preread_H8_fix2.json`, smoke folders `H8_fix2_H6`, `H8_fix2_H8` and `H8_fix2_mutation`, and
  descriptive files `descriptive_{H6,H8}_H8_fix2_*.json`. None exists.
- `inputs_gate` needs the new stage 0a record (`:101-108`).
- `publish_smoke` finds neither `results/smoke_record_fix2.json` nor `F/smoke_record_fix2.json` and writes both.
- `AR.smoke_guard` then selects fix2 (no heldfix1, no fix3). It is current, so it passes. The stale fix1 records are
  untouched.

**(b) `--read H6 --fix 2` and `--read H9 --fix 2` at the same time; later H7 and H10.**
- Each run writes only its own names:
  - its pre-read and its own proof;
  - smoke folders `<Hn>_fix2_{H6,H9}` (Qwen; SigLIP: `{H7,H10}`) and `<Hn>_fix2_mutation`;
  - its own descriptive files.
- The sigma copy is read-only in real mode. The target `results/<sigma name>` is the source file itself
  (`run_r9_held.py:531-532`), so it exists and only its SHA is compared (`run_r9_preread.py:327-329`).
- `publish_smoke` finds both canonical files present. It only reads and compares them (`:365-370`), and
  `AR.smoke_guard` only reads. There is no race.
- A failed run writes only `preread_<its read>_fix2.json` (failed) and its own folders. No canonical record is
  written on failure, so it cannot block the other reads.
- The same holds for H7 and H10 later, provided no r9 byte changed since (L1).
- Gates selected: the H6 and H9 fix-2 smoke steps run under `smoke_record_fix2.json` (current), which passes.

**(c) Held launch `run_r9_held.py --backbone <b> --read <Hn>` (first launch) and the apply step.**
- `preread_guard(results, Hn, 'held')` takes `preread_<Hn>_fix2.json`, the first existing of fix3 and fix2. It passed,
  its module and runner match, its inputs are unchanged, and its own proof is intact. It passes.
- For H8, fix2 outranks the stale passed `preread_H8_fix1.json`, so the stale record never authorises.
- For H6, fix2 outranks the failed base `preread_H6.json`.
- `smoke_guard(results, 'held')` and `AR.smoke_guard(results, None)` both select `smoke_record_fix2.json`, which is
  current. They pass.
- Without fix2 in any of the three lists, H8 would select the stale fix1 records and refuse. H6 would select the
  stale `smoke_record_fix1.json` and refuse in the smoke check.

## Item 4: frozen modules

- No stage 2 closure module and neither runner is in the diff. The diff touches only `run_r9_apply_rule.py`,
  `run_r9_descriptive.py`, `run_r9_held.py`, `run_r9_preread.py`, tests, mutant scripts, the hand-back, `rederive/rd9_commands.md`,
  `fx_inf_mutation_results.json` and `.claude/20261014_log.md`.
- Their round imports (grep of import lines) are all inside the closure or unchanged. For example, `run_r9_stage2`
  imports `run_r9_stage0b`, which is unchanged.
- No new module was added.

## Item 5: tests and mutants

Tests (worktree, `-q -p no:cacheprovider`, CPU):

| File | Passed |
|---|---:|
| `test_r9_descriptive.py` | 91 |
| `test_r9_preread.py` | 145 |
| `test_r9_apply_rule.py` | 167 |
| `test_r9_held.py` | 180 |
| `test_r9_held_a28.py` (extra) | 290 |

The worktree was left clean.

Mutants: my own set of nine, run in a copy of commit 61431a1 extracted by `git archive` into
`/project/CoSiR/.scratch/r9-run/rr_fx_inf_scratch/tree/`. Every file was restored and compared equal with `cmp`.
Script: `rr_fx_inf_scratch/bin/mut.py`; log: `rr_fx_inf_scratch/mut.log`. All nine were caught:

| Mutant | Failed tests (whole file) |
|---|---|
| M1 `report_plain` reverted to `RH._plain` | 6 (the 6 inf report tests) |
| M2 held selection without fix2 | 4 |
| M3 launch gate fix1 before fix2 | 2 (failed and stale fix2) |
| M4 apply selection fix2 before fix3 | 4 |
| M5 publish keeps any existing canonical record | 14 |
| M6 held smoke check skips the module check for fixN | 2 |
| M7 CLI admits `--fix 4` | 1 |
| M8 NaN mapped to "inf" | 1 |
| M9 launch gate skips a failed newer pre-read | 3 |

## Item 6: decision rule

Numbered generations do not conflict with the decision rule.
- Section 6: "Each failure stops the work before the read, except the smoke's, which is fixed and rerun". Both base
  failures, H8 and H6, failed at the smoke check.
- Section 6 item 5: "Any later change to these files needs a new smoke". It sets no limit and does not fix the file
  name beyond `smoke_record.json`, the R6 naming that `_fix1` already extended.
- The committed-copy duty applies to each generation (L3).
- Rule 7.3's held `--fix 1` keeps its own `heldfix1` pre-read and smoke record, unchanged.
- Rule 7.3's `--after-crash` is unaffected, apart from the existing precedence noted in M1.

## For the controller (not run here)

- The full stage 0a rerun and the real pre-reads are long. Commands and order are as in `F/fx_inf_handback.md`.
- After H8's fix 2, add the commit of `F/smoke_record_fix2.json` (L3).
- Before each later fix-2 pre-read, add the staleness check of L1.
- Scratch folders to remove later (mine): `/project/CoSiR/.scratch/r9-run/rr_fx_inf_scratch/` (84 MB tree copy, logs,
  mutant script).
