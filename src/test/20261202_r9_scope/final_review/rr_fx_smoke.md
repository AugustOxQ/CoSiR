VERDICT: PASS WITH NOTES

# Review of fix ticket fx-smoke (r9-fx-smoke, commit 4b1e1aa on main 7b198f5), 2026-10-14

Reviewer: verdict-code ticket reviewer, read-only on code. F = src/test/20261202_r9_scope (worktree /project/CoSiR-wt/r9-fx-smoke).

The fix is correct and minimal. Real (held) mode is exactly as strict as before. The CLIP-on-H6 smoke now
passes apply and the descriptive step on a copy of the REAL failed artifacts. No HIGH finding. The notes are
about the rerun procedure, not the code.

## HIGH

None.

## MEDIUM

M1. Stage 0a must be rerun after the merge, before `run_r9_preread.py --read H8 --fix 1`.
- Evidence: `run_r9_preread.py:102-106` (`guard:preread_stage0a_modules`) requires
  `stage0a_regression.json["module_sha256"] == R.r9_module_shas()`. `r9_common.py:346-355` hashes every
  `run_r9_*.py`, so both changed runners count. Main's current record (written 04:27 on 7b198f5) lists
  `run_r9_apply_rule.py` and `run_r9_descriptive.py` among its 45 keys.
- Failure scenario: merge, then launch `--fix 1` directly. The pre-read stops at "inputs" with
  "stage 0a modules changed", and the one corrected rerun is spent.
- Codex's hand-back states this prerequisite; the controller must act on it. Stage 0a overwrites its record
  (`run_r9_stage0a.py:188-191,213`). Stage 2 is not affected: it compares only its recipe closure
  (`run_r9_stage2.py:561`), and no closure module changed.

M2. The CLIP A28 smoke path (H8 on clip_b32) had never run on real data, because the failed pre-read stopped on
H6 first. I ran it in scratch with the worktree code (see "H8 scratch run"), and it passes end to end. What
remains untested is the real pre-read wrapper itself: the shared `state.env`, the patched `RH.setup`, and
`wiring_mutation` on CLIP. None of these depends on this fix. The full scratch pre-read command in Codex's
hand-back covers them; run it after the stage 0a rerun and before `--fix 1`.

## LOW

L1. `run_r9_descriptive.py:106`: the real-mode `backbone == RH.READ_BACKBONES[read]` clause is now behind
`AR.load_pass` (line 103), which refuses the same case first. Mutant M6 (clause removed) survives every test.
This redundancy existed before the fix and is harmless.

L2. `decide` (`run_r9_apply_rule.py:434`) and `claim_a28` (`:422`) choose the name from `rec["mode"]`, not from
the caller's smoke flag. This is safe because `load_pass:277-278` forces `mode == "held"` when `smoke` is False,
and `apply` passes only `load_pass` output (`:662`, `:679`). A future caller that calls `decide` on an unvalidated
record would lose that guarantee. No such caller exists today (see item 3).

L3. The smoke-mode seven-check claim text ("on frozen CLIP features") is tested only on a synthetic GO pass. The
real CLIP H6 smoke gives NO-GO with claim None, so that string never appears in the real smoke.

## Item-by-item evidence

1. Real mode is as strict as before.
- `load_pass` (`run_r9_apply_rule.py:272`): the clause `(not smoke and backbone != READ_BACKBONE_IDS[read])`
  is identical to the old check when smoke is False. The new known-id guard (`:274`) only adds a refusal.
- The mode check (`:276-278`) binds `rec["mode"]` to the caller's flag. A crafted real pass with
  `mode: "smoke"` is refused before `decide` or `claim_a28` reads mode.
- `apply` (`:641`, `:663-664`): `base["backbone"]` is overwritten only `if smoke`. Real verdicts keep
  `READS[read]["backbone"]`.
- `decide` and `claim_a28` (`:422`, `:434`): in real mode the record's mode is always "held", so the name is
  `READS[read]`.
- `bind_read` (`run_r9_descriptive.py:71-99`): real mode sets `verdict = read_json(vpath)` at line 75, after
  refusing a missing file at line 74. So `verdict` is never None in real mode, and `if not smoke:` (`:119`) equals
  the old `if verdict is not None:`. In smoke mode the old code always had `verdict = None`, so the old block was
  always skipped there too. The real-mode verdict identity (`:78-83`, ledger name of `READ_BACKBONES[read]`, mode
  "held") is unchanged.
- The held CLIP refusal tests pass. Mutant M2 (guard made smoke-only) is caught.

2. Smoke mode is consistent, and an unknown id is refused.
- `BACKBONE_NAMES` (`:113`) equals `BB.LEDGER_NAMES` for all three ids, its key set equals `BB.BACKBONES`, and
  `RH.READ_BACKBONES == AR.READ_BACKBONE_IDS` (checked in Python).
- On a copy of main's real `results/smoke/H8_smoke_H6/` (pass backbone clip_b32), using the worktree code:
  - `AR.apply("H6", smoke=True)` wrote a verdict with backbone "CLIP ViT-B/32" (NO-GO, pass_file
    held_pass_H6.json, sensitivity_H6.json).
  - `RD.bind_read` bound it (sources: pass and verdict).
  - `RD.run("H6", smoke=True, smoke_tag=...)` finished in 159 s with report backbone clip_b32. H5 was paired
    (main's R6 smoke episode hashes equal the CLIP H6 smoke's) and H7 was reported absent.
- Unknown id: `load_pass:274` refuses it (code 4, "unknown backbone id"), and the descriptive step refuses it
  through the same loader. Mutant M3 is caught.
- Nothing else got weaker in smoke mode. The only relaxation is read/backbone equality, replaced by "one of the
  three known ids". The descriptive step gained a smoke-verdict binding (`:107-118`): identity, ledger name of the
  pass backbone, pass SHA, seeds, and then the verdict-sensitivity binding in `load_bound_inputs:157-159`.

3. Other callers.
- `run_r9_held.py:888` (A28 producer, `guard:a28_pass_shape`): held mode is unchanged. In smoke it now accepts
  another known backbone, which matches `read_guard(smoke=True)` (`run_r9_held.py:433-435`). The backbone comes
  from argparse choices, so it is always a known id.
- `run_r9_descriptive.py:103`: covered under item 1.
- `rederive/` and `final_review/` use their own `decide` (rd9_stats, RS9) and do not call `AR.load_pass` or
  `claim_a28`.
- Test callers pass: `test_r9_a28_stats.py`, `test_r9_held_a28.py` and `test_r9_preread.py` together give 452
  passed.

4. Stage 2 closure.
- `git diff --name-only 7b198f5 4b1e1aa` lists only `.claude/20261014_log.md`, the hand-back,
  `run_r9_apply_rule.py`, `run_r9_descriptive.py` and the two test files.
- No production import was added.
- None of the 17 closure modules or `run_r9_stage2.py` imports `run_r9_apply_rule` or `run_r9_descriptive`
  (grep found nothing).

5. Tests and mutants: see the table below.

6. The rest of the CLIP H8 pre-read smoke chain (preread `smoke_gate`, `run_r9_preread.py:285-354`):
- Held runner smoke H6 on CLIP: it ran for real and wrote the pass (`read_guard` is exempted in smoke).
- `smoke_agreement` (`:268-282`): it has no backbone check.
- `AR.apply` smoke: it now passes (real artifact).
- `RD.run` smoke: it now passes (real artifact; `stage0b` is absent in the smoke folder and reported absent;
  `a28_report` is "not applicable" for H6).
- `wiring_mutation` (`:237-265`): it uses the first captured bundle and the CLIP picks, so it does not depend on
  read or backbone mapping.
- `publish_smoke`: it writes the results-root record and compares it with the current module SHAs.
- I found no further check on the H6 path that would refuse CLIP. For the H8 path, see "H8 scratch run".

## H8 scratch run (A28 path on CLIP, worktree code, scratch output)

The A28 path passes end to end on real inputs. I mimicked `smoke_gate` (`run_r9_preread.py:285-354`) step by step.
Outputs are under `/project/CoSiR/.scratch/r9-run/rr_fx_smoke_scratch/h8/`.

| Step | Result |
| --- | --- |
| `RH.run_smoke("clip_b32", "H8", results=scratch, subdir="rr_H8")` | exit 0 in 142 s; `held_pass_H8.json` has read H8, backbone clip_b32, mode smoke |
| `run_r9_preread.smoke_agreement(out, "H8", ...)` | written |
| `AR.apply("H8", smoke=True, results=out)` | `held_verdict_H8.json`, backbone "CLIP ViT-B/32", NO-GO |
| `RD.run("H8", smoke=True, ...)` with the pre-read's projected `a28_report` and `RD.R.RESULTS` set to scratch | report written, backbone clip_b32, H9 and H10 absent, ArtELingo-28 section present; 160 s |

Differences from the real wrapper: my run built its own environment, where the wrapper reuses `state.env`, and it
did not run `wiring_mutation` or `publish_smoke`.


## Tests and mutants

Tests (worktree, `-q -p no:cacheprovider`):

| File | Result |
| --- | --- |
| test_r9_apply_rule.py | 167 passed (5.3 s) |
| test_r9_descriptive.py | 84 passed (7.4 s) |
| test_r9_a28_stats.py + test_r9_held_a28.py + test_r9_preread.py | 452 passed (8.8 s) |
| Scratch copy, baseline of the two touched files | 251 passed |

Mutants: each was applied to a scratch copy (`.scratch/r9-run/rr_fx_smoke_scratch/wt`), run with `pytest -x` on
both touched files, then restored, with the restored bytes asserted equal to the original.

| Mutant | Result |
| --- | --- |
| M1 smoke exemption removed (old guard) | caught: test_smoke_pass_on_clip_uses_clip_verdict_and_claim[H6] |
| M2 real-mode guard made smoke-only | caught: test_pass_file_must_name_its_read_and_backbone[backbone-clip_b32-H6] |
| M3 known-id guard removed | caught: test_smoke_pass_with_unknown_backbone_is_refused[H6] |
| M4 `decide` short name always read-bound | caught: test_smoke_pass_on_clip_uses_clip_verdict_and_claim[H6] |
| M5 descriptive smoke verdict compared with the read's ledger name | caught: test_clip_smoke_binding_and_descriptive_backbone[H6] |
| M6 descriptive real-mode backbone clause removed | survives (equivalent: `AR.load_pass` refuses first; see L1) |
| M7 apply smoke `base["backbone"]` overwrite dropped | caught: test_smoke_pass_on_clip_uses_clip_verdict_and_claim[H6] |

## Scratch files (not deleted; the controller may remove them)

All are under `/project/CoSiR/.scratch/r9-run/rr_fx_smoke_scratch/`:

| Path | Contents |
| --- | --- |
| `real_h6/` | copy of main's failed smoke folder, now with a smoke verdict |
| `rd_out/` | descriptive report from the H6 run |
| `wt/` | scratch copy of the round folder used for the mutants |
| `h8/` | H8 scratch smoke |
| scripts and logs | `rd_h6.py`, `h8_step1.py`, `h8_step2.py`, `mutate.py`, `*.log` |
