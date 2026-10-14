# fx-smoke hand-back (2026-10-14)

Cause confirmed: CLIP's H8 pre-read deliberately exercises ArtELingo as H6/CLIP and A28 as H8/CLIP. Changing that selection or dropping H6 would lose ticket 22 coverage. The fix keeps both paths. No stage-2 closure module or `run_r9_stage2.py` was edited; no new production import was added.

## Whole H6/CLIP chain

| Boundary | Finding |
| --- | --- |
| `run_r9_preread.smoke_gate` | H6 fallback is intentional; the two fresh smoke directories and A28 input pins are checked. |
| `run_r9_held.run_smoke/run_read` | `read_guard(..., smoke=True)` already bypasses the read/backbone mapping. Known-backbone input/pins/parts checks, switched-off cells, CLIP head identity, sigma, values, episode distinctness and selection scoring remain active. Started and pass records both carry `clip_b32`. H6 smoke does not invoke the held ledger/H5 rebuild guards. The A28-only producer call to `AR.load_pass` is not on the H6 path. |
| `smoke_agreement` | Writes only the smoke stand-in: H6 read/pass filename, pass SHA, sensitivity SHA and counts. No backbone mapping check. |
| `AR.apply` | **Original refusal 1:** shared `load_pass` required H6/Qwen even in smoke. All rule, mode, seeds, design/schema, agreement identity/SHA/counts, Holm and sensitivity checks remain. **Downstream mislabel:** verdict backbone and seven-check claim used H6's Qwen name; neither was a refusal, but both were wrong. Smoke now uses the pass's known backbone for ledger name and claim. A28 smoke claims use the same actual-backbone rule. |
| `RD.bind_read/run` | **Original refusal 2:** its call to the same `AR.load_pass` would reject H6/CLIP. Its own `pass backbone` check already allowed any known backbone in smoke. fx-desc's ledger-name verdict identity check is held-only, so does not reject this smoke. Previously smoke ignored the verdict entirely. It now checks any present verdict's identity, actual backbone ledger name, pass SHA and seeds; the existing sensitivity binding then applies too. Pass-only smoke remains supported. Started/pass identity, arrays/episodes/sensitivity SHA, CLIP inputs/pins/coefficient reconstruction and score reproduction remain checked. `run` writes the pass's code ID (`clip_b32`) into the descriptive report. Optional paired reads check seed/episode equality; absent peers and absent smoke stage0b are disclosed. No other H6-to-Qwen mapping check exists on this smoke path. |

Thus both actual CLIP refusals were the shared pass guard, reached twice; the verdict and claim also needed correction to keep all three artifacts consistent. Held mode retains the original strict read/backbone guard and every descriptive verdict/agreement guard.

## Diff by file

`run_r9_apply_rule.py`: derive the three known IDs' ledger/short names from the existing tables; exempt only smoke from read/backbone equality, explicitly reject unknown IDs, and name smoke verdicts and claims from their validated pass. Real-mode claims and backbone names stay read-bound.

`run_r9_descriptive.py`: bind a present smoke verdict to the validated pass using the actual backbone's ledger name, rule/read/mode/status/variant/pass filename/fix flag, pass digest and seeds. Retain the verdict as a source, enabling the existing sensitivity filename/digest check. Real binding is unchanged.

`test_r9_apply_rule.py`: H6/H7 CLIP smoke success with correct verdict and claim, held H6/H7 CLIP refusal, unknown smoke ID refusal at both loader and apply boundaries, and full/small A28 smoke claims on a different known backbone.

`test_r9_descriptive.py`: real smoke stand-in and apply feed descriptive binding and report writing for H6/H7 CLIP; numerical reporting is stubbed for this identity test. Ten verdict-field mutations are refused; smoke without a verdict is still supported. All fixtures are confined to `tmp_path`.

`.claude/20261014_log.md`: appended per-file before/after snippets and reasons. This hand-back records the chain, validation and scratch command.

## Tests and mutations

Python prefix for all executed Python commands:
`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python`.

Final full touched files, each run with `-m pytest -q -p no:cacheprovider` and a 230-second timeout: `test_r9_apply_rule.py` **167 passed**, `test_r9_descriptive.py` **84 passed**; **251 total**. Initial unfixed H6/H7 positive tests: 2 failed with the reported `read and backbone` refusal. No broader suite or full pre-read was run.

| In-place mutant (targeted tests) | Result |
| --- | --- |
| Restore both production files to HEAD before this fix; apply positive and descriptive positive tests | Caught: 4 failed, 5.48 s |
| Change real-mode equality guard from `not smoke and ...` to `smoke and ...`; held CLIP refusal tests | Caught: 2 failed, 4.56 s |
| Remove explicit known-ID guard; unknown-backbone loader/apply tests | Caught: 2 failed, 4.58 s |
| Remove smoke verdict backbone assignment; apply positive tests | Caught: 2 failed, 4.63 s |
| Restore read-based seven-check claim name; apply positive tests | Caught: 2 failed, 4.63 s |
| Remove smoke verdict binding; ten descriptive inconsistency tests | Caught: 10 failed, 5.19 s |

All six mutants were restored in `finally` blocks; both source byte hashes matched their pre-mutation bytes after every run. Final touched-file suites ran again after restoration; the final diff contains no mutant.

## Main-checkout scratch pre-read (listed, not run)

After applying this commit in main, ensure default `F/results/stage0a_regression.json` is passed and records the **final main-checkout** `R.r9_module_shas()` and unchanged stage0a runner SHA. Stage0a includes the changed runners in its certificate, so an old certificate must be refreshed before this command. The selected CLIP phase-1 agreement must also pass and bind current pins and independent bytes (`_fix1`, then `_crash1`, then unsuffixed precedence). This fix leaves the stage0b/stage2 closure intact.

Exact command (fresh folder on every invocation; allow about 10–20 minutes CPU, exceeding this ticket's command limit):

```bash
cd /project/CoSiR
fx_smoke_scratch="$(mktemp -d /project/CoSiR/src/test/20261202_r9_scope/results/fx_smoke_H8_XXXXXXXX)"
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/run_r9_preread.py --read H8 --results "$fx_smoke_scratch"
```

`--results` redirects **outputs**, not certified inputs. Default locations supply stage0a (`F/results/stage0a_regression.json`), `F/pins_clip.json`, `F/cells_seed42_clip.json`, frozen CLIP parts and R6 references/sensitivity/value sets, A28 pins and their feature/table/translation/metadata files. Phase 1 reads `F/rederive/rd9_out/rd9_clip_b32_phase1_agreement{_fix1,_crash1,}.json` and its independent record/arrays there. It does not need certificates or inputs copied into scratch. Existing R6 H5 smoke artifacts, if present, are read for descriptive comparisons. CLIP skips the new-backbone stage0b/stage2 input gate; descriptive smoke looks for optional `stage0b.json` inside its own smoke directory and reports absence.

Scratch supplies the two generated passes, started files, episode/score arrays, sensitivities, smoke agreements and verdicts; descriptive reconstruction reads those files while retaining default certified input roots. Writes are all under scratch: `smoke/H8_smoke_H6/` and `smoke/H8_smoke_H8/` (including each started `folder/` copy), `smoke/descriptive/descriptive_H6_H8_smoke_H6.json` and `descriptive_H8_H8_smoke_H8.json`, `smoke/H8_smoke_mutation/r9_score.py`, `preread_H8_smoke_smoke_record.json`, `smoke_record.json`, and `preread_H8.json` (including any atomic temporary files). **No write outside scratch**: `publish_smoke` skips the adjacent-code copy for nondefault results, agreements stay in the smoke folders, RD's output root alone is patched, and the Python environment suppresses bytecode. It occupies no real `F/results/` output name or `F/rederive/rd9_out/` name needed by the later `--read H8 --fix 1` run. Refreshing stage0a is a separate prerequisite, not part of this scratch command.
