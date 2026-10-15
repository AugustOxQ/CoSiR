# fx-inf hand-back (2026-10-14)

The non-finite float is created at `r9_picks.py:98` (`"inf"` → `float("inf")`).
`run_r9_descriptive.py:181` obtains it through `P.frozen_lambdas`, `:238` returns it in
`frozen`, and `:521` inserts it into the report as `frozen.lambdas.wang.1`.
The failing strict dump was line 519 before this patch (now `:527`). The real Qwen
`baselines_seed42.json` contains `wang: {"0": 16.0, "1": "inf"}`; the real
`picks_seed42.json` read here contains B/B0/B1, with no top-level `wang`.
A read-only repro of the actual pinned Qwen files confirmed this sole infinite
lambda and the strict-JSON failure, then successfully serialized all frozen inputs
with the fix while the scoring lambda remained numeric infinity.

`report_plain` (`run_r9_descriptive.py:66`) keeps the existing JSON shape conversion
and maps Infinity/-Infinity/NaN to `"inf"`/`"-inf"`/`"nan"` only for the descriptive
dump. Finite payload bytes are unchanged. `run_r9_held._plain`, scoring numbers,
passes, verdicts, pins and every stage-2 closure module are unchanged.

H6/H9 use the same Qwen `score_inputs` path; H7/H10 use it with SigLIP's parts.
B/B0/B1 picks are members of the finite nested grid. Baseline lambda parsing admits
finite grid picks or `"inf"` only (rejects NaN/-inf); all such infinities are now
encoded. Cells become bounded Python integer pairs: sigma_star/gate_share are not
copied into `frozen.cells`. Taus must be increasing finite floats. Sensitivity is
strictly parsed and SHA-bound before reporting; its numerical record is not embedded,
only its source and digest. The held producer also requires finite non-negative
sigmas. SigLIP's baseline file was not present during this audit; these code-path
guarantees apply when its passed stage-2 files arrive. Full real held reads were
not run; six report-writing cases cover smoke H6/H9 and held H6/H9/H7/H10.

The live kind lists and consumers were audited, including `rederive/`:

| Place | Change |
| --- | --- |
| `run_r9_preread.py` docstring, VARIANTS, CLI choices/kind, preread_guard | Add fix2/fix3; first-launch order fix3, fix2, fix1, crash1, base. Direct entry allowed. Generic names, per-read proofs, publisher and smoke tags already interpolate kind. |
| `run_r9_apply_rule.py` SMOKE_RECORDS, module/function docstrings, latest_smoke_record | Add fix2/fix3 names; select heldfix1, fix3, fix2, fix1, crash1, smoke. |
| `run_r9_held.py` shared SMOKE_RECORDS, latest_smoke_record/docstring | Same order. read_names keeps held/fix1/reserve/smoke and its existing heldfix1 name. |
| `run_r9_descriptive.py` docstring/bind_read audit | Document pre-read generations; held/fix1/reserve pass matching remains unchanged. |
| `fx_held_mutants.py` two precedence anchors | Expanded tuples; both anchors checked unique. |
| `test_r9_preread.py` corrected-name/heldfix1 lists and new tests | Add fix2/fix3; publication, precedence, stale/failed proofs, exclusive writes, shared canonical preservation, tags and CLI refusal. |
| `rederive/rd9_commands.md` | Document generations and distinguish them from held/agreement variants. No rederive Python list needs expansion. |

Apply VARIANTS, held/descriptive pass lists, phase-1 `_fix1/_crash1` agreement
suffixes and comparator fix-pass tests describe different namespaces and are retained.
Historical review reports and existing records are retained. Exclusive writes and
all stale-proof checks remain; a newer failed/stale record never falls back to an
older pass. `publish_smoke` retains a current canonical record and refuses stale
results or adjacent-code copies.

Final focused suites use the required Python/env prefix, `-q -p no:cacheprovider`,
and a separate 230-second deadline per file:

| File | Passed |
| --- | ---: |
| `test_r9_descriptive.py` | 91 |
| `test_r9_preread.py` | 145 |
| `test_r9_apply_rule.py` | 167 |
| `test_r9_held.py` | 180 |
| Total | 583 |

The unfixed descriptive regression failed with the exact ValueError. Eight
targeted mutants were caught: reverted descriptive normalization; omitted fix2
in each of the three selectors; accepted stale fix2 in apply/held/launch guards;
accepted a stale fix2 canonical copy in the publisher. Every source was restored
byte-for-byte in `finally`; hashes and failing tests are in
`fx_inf_mutation_results.json`, reproduced by `fx_inf_mutants.py`.
Final suites ran after restoration; the diff contains no mutant.

Controller commands, **listed, not run** (full stage0a/pre-reads exceed this
ticket's four-minute limit), after applying this commit in main:

```bash
cd /project/CoSiR
F=src/test/20261202_r9_scope
r9_py() { PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python "$@"; }
r9_py "$F/run_r9_stage0a.py"
fx_inf_scratch="$(mktemp -d "$PWD/$F/results/fx_inf_H6_XXXXXXXX")"
r9_py "$F/run_r9_preread.py" --read H6 --results "$fx_inf_scratch"
r9_py "$F/run_r9_preread.py" --read H8 --fix 2
r9_py "$F/run_r9_preread.py" --read H6 --fix 2
r9_py "$F/run_r9_preread.py" --read H9 --fix 2
r9_py "$F/run_r9_preread.py" --read H7 --fix 2
r9_py "$F/run_r9_preread.py" --read H10 --fix 2
```

Proceed only after each prerequisite passes. H8 runs **alone and finishes first**,
publishing the shared `results/smoke_record_fix2.json` and `F/smoke_record_fix2.json`.
After it passes, H6/H9/H7/H10 may run concurrently: canonical records are already
current and retained, per-read proofs and `<Hn>_fix2_<read>` folders are distinct.
H7/H10 additionally await SigLIP's passed stage-2/sensitivity/pins and phase-1
agreement. The commands above show a safe sequential order. Stage0b and stage2
need no rerun for this patch; stage0a must certify the final changed runner bytes.

Scratch H6 writes **nothing outside its fresh root**: its sigma copy,
`smoke/H6_smoke_H6/`, `smoke/H6_smoke_H9/` (including started `folder/` copies,
episodes, arrays, sensitivities, agreements and verdicts), tagged descriptive
reports, `smoke/H6_smoke_mutation/r9_score.py`, its own smoke proof, canonical
scratch `smoke_record.json` and `preread_H6.json` all stay there. Stage certificates,
recipe inputs, A28 metadata/near-duplicates and phase-1 agreements are read at
normal owner locations. No adjacent-code canonical copy is written for scratch;
no real pre-read/smoke/held/agreement name is occupied. Refreshing stage0a is the
separate preceding command.
