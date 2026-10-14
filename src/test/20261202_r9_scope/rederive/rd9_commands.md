# Controller commands

Run from the integration checkout (`/project/CoSiR`). The default inputs under `results/` (the read set, the value sets,
the two stage-1 claim files) are gitignored and exist only there; in a worktree the commands crash on the missing files
(no agreement is written), unless `--language-flags` and `--near-duplicates` point to the main checkout. Every output is write-once. The CLIP preparation,
phase 1, and selection replay are ignored outputs in `rederive/rd9_out/`; a fresh checkout needs
CLIP prepare/score first. New-backbone/real-read commands below are handed back, not launched in this ticket.
They may exceed four minutes, particularly head preparation on wider features and the three-seed read.
Do not score an A28 read before its official pass is available for the independent comparison.

## Tests and synthetic label-only smoke

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/20261202_r9_scope/rederive/rd9_test_arithmetic.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/20261202_r9_scope/rederive/rd9_test_compare.py src/test/20261202_r9_scope/rederive/rd9_test_arithmetic.py src/test/20261202_r9_scope/rederive/rd9_test_a28.py
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

H8 to H10 also require two stage-1 JSON inputs, read by the comparer itself (gitignored run outputs):
`--language-flags` defaults to this checkout's `src/test/20261202_r9_scope/results/language_flags_a28.json`;
its `flagged` list is compared with the pass's `flagged_languages`, both sorted. `--near-duplicates` defaults
to `src/test/20261202_r9_scope/results/near_duplicates_a28.json`; its `cos_099` and `cos_098` counts are compared
with the pass's `near_duplicates`, and its `train_side == 912` and `read_paintings == 746` are asserted.
The comparer reads the flags file from the bytes it hashes and asserts that its SHA-256 equals its entry in
`--a28-pins` (default `src/test/20261202_r9_scope/pins_a28.json`; found by file name, so a copy elsewhere passes only if
byte-equal). `near_duplicates_a28.json` is not in `pins_a28.json` by decision; its SHA-256 is pinned in `rd9_inputs.py`
(`NEAR_DUPLICATES_SHA256`, and the same value in `run_r9_held.py`), the comparer reads it with `read_pinned`, and the
digest is recorded. The
agreement file holds both digests as `language_flags_sha256` and `near_duplicates_sha256` (H8 to H10 only).
`rd9_phase2.py` for H8 to H10 also checks `read_set_a28.parquet` and `value_sets_a28.json` against `pins_a28.json` at
the moment it loads them (it parses the bytes it hashed).
Use the two arguments for inputs in another folder. A missing file crashes; a missing compared field on
either side writes a failed agreement. H6/H7 and the H8 selection replay skip these inputs and claim fields.

Set-asides come from `episode_records[read_seed][pair]["set_aside"]` in the derived JSON. Each entry's language,
the enclosing pair (`emotion__style`, `emotion__genre`, `style__genre`) and the enclosing read seed form
`(language, pair, seed)`, compared as a set with the pass's entries. The entry's builder seed (e.g. 55003) and
reason are not pass-file claim fields; the pass uses the read seed (e.g. 55). All three pairs must have an explicit
`set_aside` list for every derived read seed, even when empty. H8 to H10 must carry all three claim fields.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/rederive/rd9_compare.py --derived src/test/20261202_r9_scope/rederive/rd9_out/rd9_H8_read.json --derived-arrays src/test/20261202_r9_scope/rederive/rd9_out/rd9_H8_read_arrays.npz --reported src/test/20261202_r9_scope/results/held_pass_H8.json --reported-arrays src/test/20261202_r9_scope/results/held_arrays_H8.npz --reported-sensitivity src/test/20261202_r9_scope/results/sensitivity_H8.json --language-flags src/test/20261202_r9_scope/results/language_flags_a28.json --near-duplicates src/test/20261202_r9_scope/results/near_duplicates_a28.json --out src/test/20261202_r9_scope/rederive/rd9_out/rd9_H8_agreement.json
```

The fix pass and the reserve read use the same command with the `_fix1` / `_reserve` file names, for example
`--reported .../held_pass_H8_fix1.json ... --out .../rd9_H8_agreement_fix1.json`. A synthetic smoke pass is compared
the same way, with `--out` in `rd9_out/` (the comparator writes only below `rederive/`; ticket 19 reads a smoke agreement from `results/smoke/`, so the controller copies it there); its agreement file then says `smoke: true`.

Pre-read `run_r9_preread.py --fix N` accepts N = 1, 2, 3 without an earlier pre-read of that read.
These smoke generations write `preread_<Hn>_fixN.json`, `preread_<Hn>_fixN_smoke_record.json`,
`smoke_record_fixN.json` (plus its adjacent-code copy for default results) and smoke folders `<Hn>_fixN_<read>`.
They do not add held pass or phase-1/phase-2 agreement variants: those keep their existing `_fix1` / `_reserve` names.
First-launch pre-reads select fix3, fix2, fix1, crash1, base; smoke records select heldfix1, fix3, fix2, fix1, crash1, smoke.
Reserve and held `--fix 1` keep their own proofs. Never reuse a stale canonical name; run the first publisher of a
new smoke generation alone before other pre-reads share that current canonical record.

The command carries `--reported-sensitivity .../results/sensitivity_H8.json` (the runner's `sensitivity_<Hn>[_fix1|_reserve].json`;
for the fix pass and the reserve read, the file of that pass). The comparer then compares SE, x and x95 of P1 to P7 with the derived record's; without the file those
keys are differences, and the agreement records the file's SHA-256 as `sensitivity_sha256`. The pass file's `quantity`
label of each check is compared with the rule's text too, and a repeated `set_aside` entry is a difference.
The H8 selection replay (`--mode selection`) does not compare SE, x and x95: round 6's seed-42 sensitivity file holds only the
variance components, so there is nothing to compare with. The agreement then carries `margins_not_compared` with this reason.

Comparison checks counts, points, both intervals, ordering and pass flags, episode hashes, and all 30 relevant
scorer/metric arrays plus clusters/pair/seed indices. Missing keys fail. R9 pass files must use the R6 field names
(or the documented aliases in the comparator); two-level files carry `holm_order` as the runner's
`{level1, level2}` dict (level 2's Holm is computed even after a level-1 fail); a check's `tested` defaults to true
when `holm_order` is present. An optional pair of `--derived-episodes`/`--reported-episodes` compares aggregate NPZ arrays.

For the already-completed phase-2 selection regression, use `rd9_phase2.py --read H8 --mode selection`; compare its
JSON to R6's `seed42_pass_counts.json` and its arrays to the independently validated phase-1 array archive. The
committed proof for this job is `rd9_H8_selection_agreement.json`. That file was made before the margins and quantity
labels were compared and is not rewritten; a rerun now adds the seven quantity labels (136 items) and skips the margins.

## Review fix: generic phase-1 agreement (each backbone)

Run after that backbone's independent phase-1 output and pinned seed-42 record exist. The pins file is read through its
`parts` only (each `{path, sha256}`); the seed-42 sigma file is the pins' `sensitivity_seed42` record, read pinned
(CLIP: R6's `sensitivity_seed42.json`, SHA-256 in `rd9_inputs.py`). The phase-1 comparison covers the cells, the B and B0
picks, RCA's lambda, the taus, the five seed-42 means, the P-points and n_j, the sigma inputs (N, clusters, sigma_a2 and
sigma_e2 of P1 to P7) and the array hashes with their minimum sets. This includes switched-off
backbones. Use `clip_b32`, `qwen3vl_emb_2b` or `siglip2_so400m`; the example shows Qwen.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/rederive/rd9_compare.py --phase1 --backbone qwen3vl_emb_2b --derived src/test/20261202_r9_scope/rederive/rd9_out/rd9_qwen3vl_emb_2b_phase1.json --derived-arrays src/test/20261202_r9_scope/rederive/rd9_out/rd9_qwen3vl_emb_2b_seed42_arrays.npz --out src/test/20261202_r9_scope/rederive/rd9_out/rd9_qwen3vl_emb_2b_phase1_agreement.json
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/20261202_r9_scope/rederive/rd9_test_arithmetic.py src/test/20261202_r9_scope/rederive/rd9_test_a28.py -o cache_dir=src/test/20261202_r9_scope/rederive/rd9_out/rd9_pytest_cache --basetemp src/test/20261202_r9_scope/rederive/rd9_out/rd9_test_temp
```

The real A28 labels-only smoke now defaults to 4,096 episodes per pair, matching ticket 17. `--total 64` remains
available for smaller wiring runs. The exact mutation recipe and selected pytest cases are in `rd9_mutations.py`;
its output is write-once and every edited source is restored in a `finally` block.
