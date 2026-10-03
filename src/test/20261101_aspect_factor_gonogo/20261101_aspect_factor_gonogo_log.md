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
- **`--gonogo --smoke` (Task 13):** 1,536 pooled seed-43 smoke episodes (512 per pair; Task 9's smoke file),
  1,285 anchor paintings; the A1 smoke checkpoint stands in for picked, H1 and S1. Every section is present
  (`go`, `k8`, `ablation`, `value_sharing`, `verdict`), no NaN anywhere (asserted), 10.0 s on the CPU. The smoke GO
  baseline is xing, whose smoke λ picks were 0 on both halves, so its row equals cosine (checked: identical arrays).
  Smoke verdict (meaningless by design): GO false, K8 false.
- **Real-input checks (no trained run involved):** on Task 9's full outputs the GO baseline resolves to **rca**
  (mean of R@1 and gain 6.742 on seed 42); both episode files hold 12,288 pooled episodes (4,096 per pair) whose
  SHA-256s match; the recomputed cosine per-anchor arrays equal Task 9's exactly on seeds 42 and 43 (4,602 and
  4,575 anchor paintings); Task 9's cached SE/C0/R3 codes are finite on all selection rows; the value-sharing
  diagnostic sees 8 emotion, 23 style and 10 genre values on selection rows.
- **Guard checks** (scratch copies, monkeypatched folder): `--gonogo` refuses a `picked.json` whose run is not the
  rule's pick from its own table, whose checkpoint SHA-256 or path differs, or whose smoke flag differs; real
  `--train`, `--select` and `--gonogo` refuse to overwrite; `--select` stops before any work when a checkpoint is
  missing without a failed record and accepts a failed record; `check_base_config` catches a drifted recipe.

## Observations for the decision (from Task 9, before any trained run)

- The uniform-weight control is a strong R@1 bar: on seed 42, SE's uniform control reached R@1 16.30 against
  cosine 12.96 (Task 9 table), because factor similarity ranks both shared-aspect candidates high. Beating it on
  R@1 with a CI lower bound above 0 is likely the hardest of the three GO comparisons.
- Backbone-only R@1 on these episodes is 12.96 (seed 42) and 13.53 (seed 43), not the spec's 11.1 (an earlier
  spike's episodes). The rule compares against the measured cosine, so strong GO needs about R@1 17.5 on seed 43.
- The value-sharing diagnostic, as pre-registered (10% of the factor's largest value-mean, half of the values),
  returns shares of 0.95 to 1.00 for the smoke checkpoint and for SE, C0 and R3. Dense ReLU codes keep every value's
  mean above 10% of the maximum, so this diagnostic may not separate models; we report it as pre-registered.

## Review fix round 1: pre-registration addendum (2026-10-03, before any grid result)

- **Addendum** committed first (759ee83), while the grid was queued on the GPU lock and no grid checkpoint existed:
  the value-sharing share is disclosed as blind and gets two descriptive measures (mean η² over live factors and the
  value spread S, floor 1/V); `S1_vs_A1_emotion_pairs` becomes the clean ablation (banks IC vs AIC), with
  `H1_vs_A1_genre_pairs` as K8 context; K8's wording (the AI bank's image partition carries genre, AMI 0.397 vs
  0.161); failure handling (failed H1: K8 untestable and not holding; failed S1 or A1: rows omitted; GO always
  computed); the picked run against all nine GO-bar candidates and SE_uniform on seed 43; input and checkpoint
  SHA-256s; the measured backbone-only R@1 (12.96 / 13.53, so strong GO needs about 17.5); GO as an
  intersection-union test. Pick, GO, strong GO and K8 rules unchanged.
- **Code:** only the `--gonogo` section changed (`--train` and `--select` are byte-identical); the new file was built
  and smoke-tested under a temporary name in this folder and swapped in with `os.replace`, so a grid run starting at
  that moment reads a complete file.
- **Smoke (`--gonogo --smoke`, CPU, 12.1 s):** all sections present (`go`, `k8`, `ablation`, `baseline_context`,
  `value_sharing`, `verdict`, `inputs`, `checkpoints`, `run_status`), every float finite, no missing value. Value
  spread S on the real SE, C0 and R3 codes: 0.42 to 0.63 across aspects and modalities (reviewer: 0.44 to 0.65);
  mean η² 0.02 to 0.06 on emotion, up to 0.25 on genre (image). Synthetic check with our function: one factor per
  value S 0.128 (1/V 0.125), aspect block S 0.703; the blind share gave 0.75 and 1.00 there.
- **Failure paths** (monkeypatched status, outputs in a temporary smoke subfolder): H1 and S1 failed → GO computed, K8
  "untestable … counts as K8 not holding", all S1/H1 rows omitted; A1 failed with A2 picked → the two A1-based rows
  omitted, S1-vs-picked kept, K8 tested; a checkpoint missing without a failed record stops `--gonogo` before any work.

## Launch commands (controller)

```bash
flock -n -o -E 75 /tmp/gpu0.lock bash src/test/20261101_aspect_factor_gonogo/run_grid.sh
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python src/test/20261101_aspect_factor_gonogo/run_gonogo.py --select
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python src/test/20261101_aspect_factor_gonogo/run_gonogo.py --gonogo
```

## Results

Grid, pick and GO test: pending (the controller launches the grid, then `--select`, then `--gonogo`).

## Post-hoc diagnostic (not pre-registered): training-task fit

Descriptive only, outside the pre-registered E3 decision map; it changes no pick, gate or conclusion of the go/no-go.
Script `train_fit_diagnostic.py`, output `results/train_fit_diagnostic.json` (+ `.txt`). CPU. Scorer-train rows only (local
rows 0..n-1, matching the E2 banks). Fresh episodes are NEW episodes (seed 777 + pair index, 2,048 per pair, validated)
over training rows built from the E2 partitions, not the training bank's episodes; bank episodes are the first 2,048 of
each block of `bank_AIC.npz` (A1..A6 trained on them; H1 on AI, S1 on IC). Pairs: affect__caption (third image),
affect__image (third caption), caption__image (third affect). Scores are the E3 scorers: cross-fitted agreement rule
(z-fusion with cosine), and the training-time fixed-beta score (beta 0.3, no fusion). Pooled over 3 pairs, in points,
cluster bootstrap by painting; the uniform-weight control has gain 0.00 by construction (R@1 about 22.5 to 23.2 for every
model, above cosine, because the unconditioned factor term itself adds retrieval signal).

R@1 / condition gain, pooled (cosine R@1: fresh 18.41, bank 17.96; cosine gain 0 by construction):

| Model | fresh agreement | fresh fixed-beta | bank agreement | bank fixed-beta |
|---|---|---|---|---|
| A1 | 18.89 / 0.54 [-0.01, 1.13] | 17.57 / 0.54 | 18.09 / 0.90 | 18.01 / 1.33 |
| A2 | 18.38 / 0.02 | 17.89 / 0.33 | 18.75 / 1.14 | 18.61 / 1.81 |
| A3 | 19.04 / 0.36 [-0.12, 0.86] | 17.69 / 0.95 [0.28, 1.62] | 19.57 / 1.18 | 18.05 / 1.87 |
| A4 | 18.93 / 0.48 | 17.79 / 0.69 | 18.88 / 0.50 | 17.86 / 0.63 |
| A5 | 18.82 / 0.61 | 17.39 / 0.47 | 18.73 / 0.48 | 17.71 / 1.16 |
| A6 | 18.87 / -0.06 | 17.54 / 0.64 | 18.43 / 1.04 | 17.90 / 1.28 |
| H1 | 18.46 / -0.28 | 17.50 / -0.02 | 18.60 / -0.15 | 17.67 / 0.65 |
| S1 | 18.29 / -0.32 | 17.72 / 0.15 | 18.73 / 0.28 | 17.65 / 0.38 |
| C0 | 18.00 / -0.41 [-0.81, 0.00] | 16.56 / 0.04 | 18.31 / 0.04 | 16.57 / 0.28 |
| SE | 18.41 / -0.41 [-0.71, -0.12] | 15.70 / -0.20 | 18.17 / -0.24 | 16.09 / 0.16 |

Per pair (A3, fresh, agreement): affect__caption R@1 16.77 gain 0.31; affect__image 17.74 / 0.62; caption__image 22.62 / 0.15.
Full per-pair, other and swap values for every model are in the JSON.

Reading: the trained models did not fit the pseudo-aspect training task in a way a held-out set of its own episodes
shows: on fresh episodes no model ranks the pseudo-aspect target first clearly above cosine (best A3 R@1 19.04 vs 18.41)
and the condition gain is 0.36 to 0.61 for A1/A3/A4/A5 with CIs spanning 0 (A3 fixed-beta 0.95 [0.28, 1.62] is the only
interval above 0 on fresh episodes, with R@1 below cosine), and even the in-sample bank gain stays at 0.5 to 1.9 points.
That is the same order as the 0.26 gain on labelled aspects, so the near-zero labelled gain is not a transfer failure
from a well-fit training task; E3 mostly tested an implementation whose factors barely learned the pseudo-aspect task
(consistent with the 1 to 4.4% training-loss drop), and the NO-GO speaks to that recipe, not to the idea.
