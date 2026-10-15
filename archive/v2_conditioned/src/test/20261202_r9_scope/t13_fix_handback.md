# Ticket 13 fix round

Base: controller merge `f900a7f`; review `.scratch/r9-scope/notes/review_t13.md`. No recipe step, dry run, or GPU run
was performed. Files changed: `run_r9_stage2.py`, `test_r9_stage2.py`, `test_r9_pins.py`, the original hand-back and
both required logs. Added `r9_stage2_mutations.py` to reproduce the mutation table in scratch copies under `/tmp`.

## Findings

- **M1:** every frozen role is hashed against `pins_clip.json`, and against the R6 pin when its canonical path is
  an R6 input. The array comparison remains as a second check. A rewritten frozen file at CLIP's own path now fails.
- **M2:** exemptions skip only fitted-key inequality. Copied JSON keys still compare exactly to CLIP. Producer
  `keys_made` / `keys_copied` remain intact; `keys_tested_equal`, `keys_tested_differ`, `keys_tested_made`,
  `keys_tested_copied`, `keys_tested_nan` and metadata lists describe the actual checks. Reader JSON claims only
  the `feature_names` equality check. `picked.checkpoint_sha256` must match this backbone's A3 file.
- **M3:** the runtime AST import walk starts from the five recipe scripts, follows local imports and literal
  dynamic-import names, and visits exactly today's 17 certified modules. It stops at the lazy pins-to-driver
  consumer boundary. Unrelated held runners and newly added unimported modules do not enter the closure. Every
  step checks it before execution; provenance, record, and pins check it again. The closure is stored in each
  receipt, the stage 2 record and the pins. A mismatch gives the requested module-specific rerun-0b message.
- **M4:** `n_iter_` and array-valued `n_samples_seen_` are recorded as metadata that was not compared. Learned
  coefficients/scaler arrays must still differ; `classes_` must still equal CLIP.
- **M5:** the synthetic file world exercises real provenance, receipts, record creation and `run_pins`, with only
  feature loading/scoring and subprocess execution substituted. Tests also cover CLI gates, code changes between
  steps, lock arguments, CPU/GPU environment selection, failure exit codes, exclusive writes, state snapshots,
  switched-off termination, CLIP equality dispatch, sigma output dispatch/validity, input manifests and cell routing.
  The original 33 mutants and eight additional probes are replayed independently on scratch copies.
- **L1:** logical recipe inputs accept only the backbone's resolved file, never CLIP's alternative. Explicit
  CLIP paths for refitted inputs also refuse; explicit frozen aliases and the copied CSD source remain allowed. Unknown or
  malformed hash keys fail. A3's build record, declared paths and C-contiguous `local_groups` hash are resolved
  and checked explicitly.
- **L2:** a switched-off backbone writes `stage2_<backbone>.json` with `not run: switched off` and its gate share,
  then stops. `_score` is never called; there is no seed-42 statistics file, no sensitivity file, and no numerical
  pins. A separate pins request also refuses it.
- **L3:** an equal existing `stage2_inputs.json` is accepted on a cells retry; a changed manifest refuses. The
  manifest remains available to the existing CLI before final pins exist. A crashed producer's partial outputs
  must still be moved aside by the controller; the driver never deletes them.
- **L6:** the subprocess independence test also imports `run_r9_stage2`; it must load no R6 module or R6 sys.path entry.
- **L7:** the seed-42 B/B0/B1 means must exactly equal `picks_seed42.json.mean_r1`.

One schema conflict required a narrow distinction in M2: ticket 10 labels `picks_seed42.targets` as copied, but
both its producer and `r9_picks.load_picks` require `targets: null` on a new backbone, while CLIP holds numeric
regression targets. That field is explicitly checked as backbone metadata and is listed only in
`keys_tested_backbone_metadata`, not in the equality-to-CLIP list. All other copied keys compare to CLIP. No recipe
script was changed. This preserves the existing passed stage 0b certificate.

L4 and L5 are unchanged. Run stage 2 and the reads in the same checkout. A driver-byte change after a stage 2 record
still invalidates it. Case 1 is the CLIP dry-run case. DAS6 remains a later option.

## Tests and mutation results

Only `test_r9_stage2.py` and `test_r9_pins.py` ran. Each command and each mutation pytest invocation stayed below
four minutes; the harness limits each pytest invocation to 180 seconds. The three scratch copies use Git metadata
only for read-only main-checkout discovery; all mutated code and test outputs stay under `/tmp`.

```bash
cd /project/CoSiR-wt/r9-t13
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider --basetemp=src/test/20261202_r9_scope/results/t13_fix_final_stage2 src/test/20261202_r9_scope/test_r9_stage2.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider --basetemp=src/test/20261202_r9_scope/results/t13_fix_final_pins src/test/20261202_r9_scope/test_r9_pins.py
```

Worktree: **68 passed in 24.78s** in `test_r9_stage2.py`; **21 passed in 12.23s** in `test_r9_pins.py`.
The unmutated scratch baseline passed all **89 tests**. The review table is now **33 caught, 0 survived**; the eight
additional probes are **8 caught, 0 survived**. Every reported run matches the final driver, pins module and two
test-file SHA-256s in `t13_fix_mutation_results.json`. Setup/collection errors are not counted as caught mutants.
Each mutant starts from pristine copied files and is restored afterward; scratch logs are listed in the JSON.

## Controller dry-run commands

Stage 0b already passed case 1. These commands now explicitly read the passed certificate in the main checkout,
since this worktree has no `results/stage0b.json`. The 17-module closure is checked against it at runtime. If the
controller runs in another checkout, use that checkout's matching certificate and output directory. Record takes
8–12 minutes (CPU); pins takes under one minute. Both are intentionally unrun here.

```bash
cd /project/CoSiR-wt/r9-t13
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/run_r9_stage2.py --backbone clip_b32 --step record --dry-run --replay /project/CoSiR-wt/r9-replay/src/test/20261202_r9_scope/results/replay/clip_b32/ --stage0b /project/CoSiR/src/test/20261202_r9_scope/results/stage0b.json --out src/test/20261202_r9_scope/results/t13_clip_dryrun
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/run_r9_stage2.py --backbone clip_b32 --step pins --dry-run --replay /project/CoSiR-wt/r9-replay/src/test/20261202_r9_scope/results/replay/clip_b32/ --stage0b /project/CoSiR/src/test/20261202_r9_scope/results/stage0b.json --out src/test/20261202_r9_scope/results/t13_clip_dryrun
```

Dry-run writes are exclusive. If an earlier dry run left files, move them aside before running again, or use a fresh
`--out` under this round folder. The external replay stays read-only. The dry run recomputes exact CLIP values and
then preserves the stored sensitivity bytes exactly.

## Mutation table

| Mutant | Change | Result |
|---|---|---|
| m01 | `guard:made_differs` body → `pass` | caught |
| m02 | copied-equal check → `pass` | caught |
| m03 | `assert_frozen` npz comparison → `pass` | caught |
| m04 | frozen loop in `provenance` removed | caught |
| m05 | `guard:script_sha` → `True` | caught |
| m06 | `assert_script` call in `provenance` removed | caught |
| m07 | `assert_script` loop in `pins_evidence` removed | caught |
| m08 | `assert_part` call removed (producer lists used) | caught |
| m09 | `provenance(..., replay=True)` for every backbone | caught |
| m10 | `guard:stage2_passed` → `True` | caught |
| m11 | `guard:picks_passed` → `pass` | caught |
| m12 | record-file check → `pass` | caught |
| m13 | `build_pins` skips evidence for new backbones | caught |
| m14 | `pins_evidence` dry-run flag check removed | caught |
| m15 | `pins_evidence` "stage 2 file changed" check removed | caught |
| m16 | CLIP sigma bytes written for any run | caught |
| m17 | `clip_exact` skipped | caught |
| m18 | `main()` dry-run backbone gate removed | caught |
| m19 | GPU `flock` removed | caught |
| m20 | `CUDA_VISIBLE_DEVICES=0` on every step | caught |
| m21 | `--step all` ignores a failed step | caught |
| m22 | receipt file comparison removed | caught |
| m23 | `write_new` opens with `'w'` | caught |
| m24 | `rc_tau` added to EXEMPT | caught |
| m25 | pickle `coef_` treated as copied | caught |
| m26 | `S.CELLS` routing in `seed42_record` removed | caught |
| m27 | sigma finite / non-negative check removed | caught |
| m28 | extra `import r6_bundle` in the driver | caught |
| m29 | unrefit-NaN check removed | caught |
| m30 | feature-store lineage check removed | caught |
| m31 | status always `ready` | caught |
| m32 | manifest frozen-SHA check removed | caught |
| m33 | runner / rule SHA check in `pins_evidence` removed | caught |

Additional guard probes:

| Mutant | Change | Result |
|---|---|---|
| m34 | Frozen pins_clip SHA guard removed | caught |
| m35 | Closure check before each main step removed | caught |
| m36 | Closure check in provenance removed | caught |
| m37 | Closure check in pins removed | caught |
| m38 | Closure check in run_step removed | caught |
| m39 | Copied JSON comparisons skipped | caught |
| m40 | Iteration counts treated as fitted arrays | caught |
| m41 | B/B0/B1 mean cross-check removed | caught |

Reproduction (run small batches so each shell command remains below four minutes):

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/r9_stage2_mutations.py --init
# Substitute the printed scratch path; repeat for the remaining mutant IDs.
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/r9_stage2_mutations.py --scratch /tmp/r9-t13-mutations-PRINTED_PATH --mutants m01 m02
```
