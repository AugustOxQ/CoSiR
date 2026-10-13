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

One agreement JSON per read (replace H8 with H6/H7/H9/H10 as appropriate; add `_fix1` input suffixes if needed):

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/rederive/rd9_compare.py --derived src/test/20261202_r9_scope/rederive/rd9_out/rd9_H8_read.json --derived-arrays src/test/20261202_r9_scope/rederive/rd9_out/rd9_H8_read_arrays.npz --reported src/test/20261202_r9_scope/results/held_pass_H8.json --reported-arrays src/test/20261202_r9_scope/results/held_arrays_H8.npz --out src/test/20261202_r9_scope/rederive/rd9_out/rd9_H8_agreement.json
```

Comparison checks counts, points, both intervals, ordering and pass flags, episode hashes, and all 30 relevant
scorer/metric arrays plus clusters/pair/seed indices. Missing keys fail. R9 pass files must use the R6 field names
(or the documented aliases in the comparator); two-level files need `tested`, `holm_order_level1`,
`holm_order_level2`. An optional pair of `--derived-episodes`/`--reported-episodes` compares aggregate NPZ arrays.

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
