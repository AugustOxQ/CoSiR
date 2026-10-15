# Ticket 13 hand-back

Updated by two review fix rounds; see `t13_fix_handback.md` for M1–M5/L1–L7 and `t13_fix2_handback.md` for N1–N4,
the real replay check, current mutation results and controller commands.

Built `run_r9_stage2.py` and `test_r9_stage2.py`; extended `r9_pins.build_pins` for the new-backbone evidence. Default
CLIP pins retain their original schema. The existing independence test now allows exactly the three original R6
statistics producers in the stage 2 driver, as required by this ticket.

The driver runs 07–11 through their existing CLIs. GPU commands use
`flock -n -o -E 75 /tmp/gpu0.lock`; CPU steps use eight threads. The controller runs them. Each completed step gets
an exclusive receipt with every part's file and producer-record SHA-256, plus A3's history/gates and 07's build
record. Receipt hashes are checked before downstream steps and before final pins. No provisional final pins are
written. The cells CLI is executed via `runpy` under the driver's checked input-manifest context, because its
existing setup otherwise requires the final pins file. The recipe script itself is unchanged, keeping its 0b SHA.

The `record` step checks each made array against CLIP, copied arrays and frozen roles against CLIP, and every
producer script against the passed `stage0b.json`. It checks feature-store provenance through upstream records,
then scores only seed-42 selection episodes. R6's actual `pass_record`, `per_episode_arrays`, and
`sensitivity_record` functions produce the counts and sigma components; stage 0a's functions produce the means
and 96 array keys. The pins contain all shared roles, these numbers/hashes, the sigma file's path/hash, records,
receipts and gate share for running backbones. A switched-off backbone gets only a stage 2 status record with
`not run: switched off` and its gate share; scoring and numerical pins stop there. No held runner is invoked.

Choices beyond the fixed ticket: `--step record` and `--step all` are added; record is separate so it can be run
without repeating fits. Scipy graph NPZ `shape` and `format` are checked as copied container metadata; `data`,
`indices`, and `indptr` are checked as made arrays. Reader pickle classes are copied metadata; iteration/sample counts are un-compared metadata;
learned coefficients/scaler arrays are made. Unused style-head arrays follow 07's `keys_unrefit` NaN contract.
Picks, cells, `picked.json`, and `baselines_seed42.json` are exempt from differing from CLIP. All output creation
is exclusive. The live recipe directory is fixed; only dry-run record/pins take `--out`, within this round folder.
DAS6 is a later option; this driver launches no cluster work.

## Validation

Run from `/project/CoSiR-wt/r9-t13`:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider --basetemp=src/test/20261202_r9_scope/results/t13_pytest_final src/test/20261202_r9_scope/test_r9_stage2.py src/test/20261202_r9_scope/test_r9_pins.py
```

Original build: **33 passed in 15.73 seconds**. The review fix adds call-site coverage and a mutation sweep; current counts are in `t13_fix_handback.md`. The tests include each of the three provenance refusals, the pins step's missing-record/failed-stage/failed-picks
refusals, and exact equality of R6's sigma producer against all of CLIP's stored sigma components. CLI help is
smoke-checked. No GPU was run by the implementer.

## Controller commands (each long command is intentionally left to the controller)

The controller's merge `f900a7f` brings the completed replay and passed stage 0b case 1. The commands below
explicitly read `/project/CoSiR/src/test/20261202_r9_scope/results/stage0b.json`; this worktree has no local copy.
The runtime recipe import closure is checked against that certificate. The external replay is read-only.

Then the required CLIP dry run, writing only here. `record` takes 8–12 minutes, CPU; `pins` takes under a minute.
The record step asserts exact equality of the five means, every non-run-specific pass-count field, all 96 arrays,
and the entire sigma producer result (P1–P7, S1/S2, seed, N and painting count) against the stored CLIP references.
After that computation/equality check, the dry run writes CLIP's original sensitivity bytes, preserving the stored
provenance fields and formatting as well; the dry-run sigma file is therefore byte-for-byte CLIP's stored file.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/run_r9_stage2.py --backbone clip_b32 --step record --dry-run --replay /project/CoSiR-wt/r9-replay/src/test/20261202_r9_scope/results/replay/clip_b32/ --stage0b /project/CoSiR/src/test/20261202_r9_scope/results/stage0b.json --out src/test/20261202_r9_scope/results/t13_clip_dryrun
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/run_r9_stage2.py --backbone clip_b32 --step pins --dry-run --replay /project/CoSiR-wt/r9-replay/src/test/20261202_r9_scope/results/replay/clip_b32/ --stage0b /project/CoSiR/src/test/20261202_r9_scope/results/stage0b.json --out src/test/20261202_r9_scope/results/t13_clip_dryrun
```

After that, the two live chains (approximately 65–90 minutes each; estimates from the CLIP replay, the larger
feature dimensions may take longer): groups 12 minutes GPU, A3 12 minutes GPU, reader 14 minutes CPU,
baselines 10 minutes CPU, cells 5 minutes CPU, record 8–12 minutes CPU, pins under a minute. `--step all`
propagates failure/lock exit codes and stops immediately on any failed step. A completed step is never replaced;
use explicit `--step reader`, etc. to continue from successful preceding receipts.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/run_r9_stage2.py --backbone qwen3vl_emb_2b --step all --stage0b /project/CoSiR/src/test/20261202_r9_scope/results/stage0b.json
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/run_r9_stage2.py --backbone siglip2_so400m --step all --stage0b /project/CoSiR/src/test/20261202_r9_scope/results/stage0b.json
```

The driver sets `CUDA_VISIBLE_DEVICES=0` only in the two locked GPU subprocesses. No held row is scored by these
commands. New-backbone sigma files are `results/sensitivity_seed42_<backbone>.json`, in the held runner's schema.
