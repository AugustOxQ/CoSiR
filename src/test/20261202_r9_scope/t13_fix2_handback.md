# Ticket 13 second fix round

Base: `42e214d`; re-review: `/project/CoSiR/.scratch/r9-scope/notes/review_t13_fix.md`.
No recipe step, CLIP dry run or GPU execution was performed. Nothing was written in the external replay folder.

## Findings

- **N1:** `_input_files(..., replay=True)` accepts either the resolved replay file or CLIP's canonical file for
  a known logical input. Live inputs still accept only the backbone's file; unknown keys and wrong hashes still
  refuse. Both `provenance` and `pins_evidence` forward their replay/dry-run flag. Tests model cells naming CLIP's
  frozen baselines while the replay baselines have different bytes, and exercise record through pins evidence.
- **N2:** feature lineage skips an input hash equal to the current role's own file hash only in replay mode.
  `skipped_self_input_sha256` records the skipped input names and hashes per role, separately from checked file
  inputs. The input file hash is still verified. Live self-cycles still refuse, and other cycles retain the guard.
- **N3:** two direct call-site tests change a valid `rc_tau` file after its receipt was written. The downstream
  baselines step must refuse before subprocess execution; record must refuse before scoring. Removing either
  receipt call fails its corresponding test.
- **N4:** a direct test deletes `r9_common.py` from the certificate's module map and requires the rerun-0b refusal.
  Accepting the missing module via `.get(module, sha)` fails this test. The production module checker is unchanged.

Files changed: `run_r9_stage2.py`, `test_r9_stage2.py`, `r9_stage2_mutations.py`, `t13_handback.md`, the round log and
`.claude/20261014_log.md`. Added this hand-back, the read-only diagnostic `t13_fix2_replay_check.py` and its result,
and the second-round mutation results. No recipe producer or closure module was edited.

## Real replay, read-only

`_input_files` and `provenance` passed for **all 20 recipe roles**. The only skipped self-input is partitions:
`/project/CoSiR/src/test/20261031_pseudo_partitions/results/partitions.npz`, SHA-256
`cfd57dbb21216cdc9d1c2e42d1b408648bc18147f62354b52202ade4ab8a9caa`.
The role list and full report are in `t13_fix2_replay_result.json`.

As in the re-review, this diagnostic isolates module certification: the main checkout's existing stage0b certificate
differs for **r9_pins.py** only. It reports that mismatch and substitutes only `assert_modules` for the read-only
provenance call. All input, producer-script, frozen-part, array and feature-lineage checks run unchanged. This is
input/provenance evidence, not a certified stage 2 pass. The production record/pins commands still refuse the stale
certificate; the controller must rerun the stage 0b compare after merging before running them.

```bash
cd /project/CoSiR-wt/r9-t13
timeout 230s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/t13_fix2_replay_check.py
```

## Tests and mutations

Both authorized suites ran under the four-minute limit, with bytecode/cache writes disabled:

```bash
timeout 230s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider --basetemp=src/test/20261202_r9_scope/results/t13_fix2_stage2 src/test/20261202_r9_scope/test_r9_stage2.py
timeout 230s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider --basetemp=src/test/20261202_r9_scope/results/t13_fix2_pins src/test/20261202_r9_scope/test_r9_pins.py
```

**74 stage-2 tests passed in 16.10s; 21 pins tests passed in 8.67s.** Six tests were added. The targeted pre-fix
run failed on both replay cases and passed the four tests for existing live/receipt/module checks.

Mutation results are recorded in `t13_fix2_mutation_results.json`; all changes ran on pristine scratch copies under
`/tmp/r9-t13-mutations-5n0svlfn`, then were restored. Every mutant ran both authorized suites (95 tests), with no
collection/setup errors counted as a catch. The original 41 probes were not rerun in this small round; their
previous results remain in `t13_fix_mutation_results.json`.
This round: **10 caught, 0 survived**, including all three re-review mutants that motivated N3/N4.

| Mutant | Change | Result |
|---|---|---|
| o5 | Remove `run_step` receipt check | caught |
| o6 | Accept a missing certified module with `.get(module, sha)` | caught |
| o8 | Remove `run_record` receipt check | caught |
| n1a | Remove replay's CLIP logical-file alternative | caught |
| n1b | Allow CLIP logical-file alternatives on live runs | caught |
| n1c | Do not forward replay to provenance input checks | caught |
| n1d | Do not forward dry-run to pins input checks | caught |
| n2a | Remove replay self-hash exception | caught |
| n2b | Skip self-hashes on live runs too | caught |
| n2c | Omit skipped self-hash evidence | caught |

Reproduction: initialize with `r9_stage2_mutations.py --init`, then use the printed scratch path. Run batches of
three or four under `timeout 230s`; both suites run for each mutant:

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/r9_stage2_mutations.py --init
timeout 230s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/r9_stage2_mutations.py --scratch /tmp/r9-t13-mutations-5n0svlfn --mutants o5 o6 o8
timeout 230s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/r9_stage2_mutations.py --scratch /tmp/r9-t13-mutations-5n0svlfn --mutants n1a n1b n1c n1d
timeout 230s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/r9_stage2_mutations.py --scratch /tmp/r9-t13-mutations-5n0svlfn --mutants n2a n2b n2c
```

## Controller dry-run commands

Unchanged commands. First merge this fix and refresh the stage 0b compare certificate in the checkout used for
the reads. Record takes approximately **8–12 minutes on CPU**; pins takes **under one minute**. These were not run.
Writes are exclusive: use a fresh `--out` if the listed output already contains files.

```bash
cd /project/CoSiR-wt/r9-t13
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/run_r9_stage2.py --backbone clip_b32 --step record --dry-run --replay /project/CoSiR-wt/r9-replay/src/test/20261202_r9_scope/results/replay/clip_b32/ --stage0b /project/CoSiR/src/test/20261202_r9_scope/results/stage0b.json --out src/test/20261202_r9_scope/results/t13_clip_dryrun
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/run_r9_stage2.py --backbone clip_b32 --step pins --dry-run --replay /project/CoSiR-wt/r9-replay/src/test/20261202_r9_scope/results/replay/clip_b32/ --stage0b /project/CoSiR/src/test/20261202_r9_scope/results/stage0b.json --out src/test/20261202_r9_scope/results/t13_clip_dryrun
```
