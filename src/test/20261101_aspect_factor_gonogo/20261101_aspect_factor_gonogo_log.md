# 20261101 aspect-factor go/no-go (E3) log

## Problem

E3 produces the Oct 9 go/no-go that picks the paper's branch (spec §4, §6). We train the aspect-factor grid (A1 to A6,
the held-out-genre model H1 and the supervision ablation S1) on scorer-train rows, pick one run on Task 9's seed-42
selection episodes, then test it on Task 9's seed-43 episodes against backbone-only, the GO baseline and its own
uniform-weight control, plus the K8 held-out-genre test. Every rule is fixed in `PREREGISTRATION.md`, committed
(994e138) before any training run, smoke included.

## Setup (2026-10-03)

- **Runner:** `run_gonogo.py --train RUN --seed S [--smoke]`, `--select [--smoke]`, `--gonogo [--smoke]` (Task 13).
  `run_grid.sh` trains the 8 runs at seed 42, at most 3 at once, each logging to `logs/<RUN>_seed42.log`.
- **Inputs:** E2 (`src/test/20261031_pseudo_partitions/results/`: partitions, graph, banks AIC/AI/IC; every SHA-256 is
  checked against its `build_record.json`; `local_groups` is checked against
  `np.unique(groups[scorer_train], return_inverse=True)[1]`; bank rows are checked to lie in 0..n-1). Task 9
  (`src/test/20261030_aspect_baselines/results/`: episodes, per-anchor arrays, baselines JSON, cached SE/C0/R3 codes).
- **Base config, plan vs C0 recipe (controller ruling R7).** The plan writes
  `dataclasses.replace(R3_CONFIG, painting_batches=True, ...)`; the factor-learning grid's C0 cell is
  `cell_config("C0", seed) = replace(R3_CONFIG, painting_batches=True, seed=seed, epochs=steps,
  agreement_level="pair", lambda_condition=0.0)`. We imported the grid module and compared `asdict` of both: **no field
  differs** (the two C0 fields equal the defaults). `run_config` builds the C0 form, and `--train` asserts that the
  base equals the config stored by the grid's real C0 run (`history_C0_seed42.json`) on every stored key except seed
  and epochs.
- **Grid:** 2,000 steps, `aspect_episodes_per_step = 32`, `aspect_beta = 0.3` (A5: 0.0); the table is in
  `PREREGISTRATION.md` §3.

## Smoke (2026-10-03)

- **GPU busy.** Task 14's MLLM probe held `/tmp/gpu0.lock` (pid 141988, 11.5 GB; about 40 minutes for 900 episodes),
  so the first training smoke ran on the CPU (`CUDA_VISIBLE_DEVICES=""`).
- **`--train A1 --seed 42 --smoke` (CPU):** 50 steps on E2's smoke AIC bank (768 episodes) in 134.0 s (load 3.3 s);
  codes finite on all 183,694 scorer-train rows; image codes 59% active (0 dead factors), caption codes 49% active
  (1 dead factor); loss 13.25 to 11.02, aspect loss 3.43 to 3.02, τ 0.027 to 0.029.
- **`--select --smoke`:** 192 pooled seed-42 smoke episodes (64 per pair), SHA-256s match Task 9's record; the A1 smoke
  checkpoint stands in for all 8 runs, so all rows tie (R@1 14.97 [12.50, 17.58], gain 3.78 [1.01, 6.67], uniform
  control R@1 16.02, gain 0.00) and the tie rule picks A1. 11.9 s on the CPU. `select_seed42.json`, `picked.json`
  and `per_anchor_select_seed42.npz` written to `results/smoke/`.
- **`run_grid.sh` logic** was checked with a stub trainer in the scratchpad (one run forced to fail): the other 7
  ran 3 at a time, the script exited 1 naming the missing run, and a relaunch trained only that run.

## Launch commands (controller)

```bash
flock -n -o -E 75 /tmp/gpu0.lock bash src/test/20261101_aspect_factor_gonogo/run_grid.sh
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python src/test/20261101_aspect_factor_gonogo/run_gonogo.py --select
```

## Results

Grid and pick: pending (the controller launches the grid).
