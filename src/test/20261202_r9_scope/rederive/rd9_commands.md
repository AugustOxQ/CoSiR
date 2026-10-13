# Controller commands

Run from `/project/CoSiR-wt/r9-t25` (or the integration checkout). Every output is write-once. The CLIP preparation,
phase 1, and selection replay already exist in this worktree's ignored `rederive/rd9_out/`; a fresh checkout needs
CLIP prepare/score first. New-backbone/real-read commands below are handed back, not launched in this ticket.
They may exceed four minutes, particularly head preparation on wider features and the three-seed read.
Do not score an A28 read before its official pass is available for the independent comparison.

## Tests and synthetic label-only smoke

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/20261202_r9_scope/rederive/rd9_test_arithmetic.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/rederive/rd9_smoke.py
```

## Phase 1

These emit `rd9_<backbone>_{bundle.npz,fits.pkl,prepare.json,seed42_arrays.npz,phase1.json}` in `rd9_out/`.
The pins adapter contract is in `inputs.md`. Use `--pins` if the controller chooses different pins-file names.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/rederive/rd9_phase1.py --backbone clip_b32 --stage prepare
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/rederive/rd9_phase1.py --backbone clip_b32 --stage score
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/rederive/rd9_phase1.py --backbone qwen3vl_emb_2b --stage prepare
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/rederive/rd9_phase1.py --backbone qwen3vl_emb_2b --stage score
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/rederive/rd9_phase1.py --backbone siglip2_so400m --stage prepare
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/rederive/rd9_phase1.py --backbone siglip2_so400m --stage score
```

CLIP phase-1 comparison in a fresh checkout (this job's proof is the committed `rd9_clip_b32_agreement.json`):

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/rederive/rd9_compare.py --clip-phase1 --derived src/test/20261202_r9_scope/rederive/rd9_out/rd9_clip_b32_phase1.json --derived-arrays src/test/20261202_r9_scope/rederive/rd9_out/rd9_clip_b32_seed42_arrays.npz --out src/test/20261202_r9_scope/rederive/rd9_out/rd9_clip_b32_controller_agreement.json
```

## Real A28 labels-only smoke, once tickets 14/16 provide the table and values

This writes hashes for seeds 9001–9003, 4,096 episodes per pair, for comparison with ticket 17. It scores no feature.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/rederive/rd9_episodes.py --seeds 9001 9002 9003 --total 4096 --out src/test/20261202_r9_scope/rederive/rd9_out/rd9_a28_smoke
```

## Phase 2, after each official pass

H6/H7 rebuild and check H5's episodes. H8 builds the independent A28 hashes; H9/H10 assert they equal its hashes.
These use frozen phase-1 fits/picks, and produce no official verdict. Default filenames of both pins sets are used;
`--pins` and `--a28-pins` override them. Each invocation may exceed four minutes and belongs to the controller.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/rederive/rd9_phase2.py --read H6
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/rederive/rd9_phase2.py --read H7
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/rederive/rd9_phase2.py --read H8
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/rederive/rd9_phase2.py --read H9
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/rederive/rd9_phase2.py --read H10
```

One agreement JSON per pass file (replace H8 with H6/H7/H9/H10 as appropriate). The output names are
`rd9_<Hn>_agreement.json` (original pass), `rd9_<Hn>_agreement_fix1.json` (fix pass) and
`rd9_<Hn>_agreement_reserve.json` (reserve read); the apply-rule step (ticket 19) looks for exactly these names in
`rederive/rd9_out/`. Give `--reported` the pass file that the agreement belongs to (`held_pass_<Hn>[_fix1|_reserve].json`).
The file holds `agreement`, `compared`, `failed`, `items`, `independent_identity`, `read`, `backbone` and the provenance
the apply-rule step reads: `phase` (2), `smoke` (bool, from the `mode` field of the `--reported` file: "smoke" is true,
"held" and "regression" are false, anything else or no field is a crash), `pass_file` (the name of the `--reported`
file), `held_pass_sha256` (SHA-256 of that file's bytes) and `rederived_counts` ({P1..P7: n_j of the independent pass,
from `--derived`; the same keys for the two-level reads}). The phase-1 modes do not write the provenance fields.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/rederive/rd9_compare.py --derived src/test/20261202_r9_scope/rederive/rd9_out/rd9_H8_read.json --derived-arrays src/test/20261202_r9_scope/rederive/rd9_out/rd9_H8_read_arrays.npz --reported src/test/20261202_r9_scope/results/held_pass_H8.json --reported-arrays src/test/20261202_r9_scope/results/held_arrays_H8.npz --out src/test/20261202_r9_scope/rederive/rd9_out/rd9_H8_agreement.json
```

The fix pass and the reserve read use the same command with the `_fix1` / `_reserve` file names, for example
`--reported .../held_pass_H8_fix1.json ... --out .../rd9_H8_agreement_fix1.json`. A synthetic smoke pass is compared
the same way, with `--out` in `rd9_out/` (the comparator writes only below `rederive/`; ticket 19 reads a smoke agreement from `results/smoke/`, so the controller copies it there); its agreement file then says `smoke: true`.

Comparison checks counts, points, both intervals, ordering and pass flags, episode hashes, and all 30 relevant
scorer/metric arrays plus clusters/pair/seed indices. Missing keys fail. R9 pass files must use the R6 field names
(or the documented aliases in the comparator); two-level files carry `holm_order` as the runner's
`{level1, level2}` dict (level 2's Holm is computed even after a level-1 fail); a check's `tested` defaults to true
when `holm_order` is present. An optional pair of `--derived-episodes`/`--reported-episodes` compares aggregate NPZ arrays.

For the already-completed phase-2 selection regression, use `rd9_phase2.py --read H8 --mode selection`; compare its
JSON to R6's `seed42_pass_counts.json` and its arrays to the independently validated phase-1 array archive. The
committed proof for this job is `rd9_H8_selection_agreement.json`.

## Review fix: generic phase-1 agreement (each backbone)

Run after that backbone's independent phase-1 output and pinned seed-42 record exist. This includes switched-off
backbones. Use `clip_b32`, `qwen3vl_emb_2b` or `siglip2_so400m`; the example shows Qwen.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/rederive/rd9_compare.py --phase1 --backbone qwen3vl_emb_2b --derived src/test/20261202_r9_scope/rederive/rd9_out/rd9_qwen3vl_emb_2b_phase1.json --derived-arrays src/test/20261202_r9_scope/rederive/rd9_out/rd9_qwen3vl_emb_2b_seed42_arrays.npz --out src/test/20261202_r9_scope/rederive/rd9_out/rd9_qwen3vl_emb_2b_phase1_agreement.json
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/20261202_r9_scope/rederive/rd9_test_arithmetic.py src/test/20261202_r9_scope/rederive/rd9_test_a28.py -o cache_dir=src/test/20261202_r9_scope/rederive/rd9_out/rd9_pytest_cache --basetemp src/test/20261202_r9_scope/rederive/rd9_out/rd9_test_temp
```

The real A28 labels-only smoke now defaults to 4,096 episodes per pair, matching ticket 17. `--total 64` remains
available for smaller wiring runs. The exact mutation recipe and selected pytest cases are in `rd9_mutations.py`;
its output is write-once and every edited source is restored in a `finally` block.
