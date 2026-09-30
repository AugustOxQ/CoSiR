# 20261018 affect factor learning: run script, prepare and timing smoke

## Problem
Cells E and SE train factor encoders (R3 config, pair agreement, condition episodes) with conditions drawn from
GoEmotions affect clusters of the captions. Task 2 builds `run_affect.py`, extracts the affect signal, clusters it,
measures diagnostics, and times the runs so the local-vs-DAS6 choice can be made.

## Steps
1. `run_affect.py`: imports `run_grid.py` as `grid`; `prepare()`, `affect_cell_config`, `affect_source`,
   `run_affect_cell` (overwrite guard on full runs), `smoke()`, `main()` (`--evaluate/--replicate/--tables` raise
   `NotImplementedError("Task 3")`, Controller Ruling 2).
2. `--prepare` (125 s total), then `--smoke`.

## Verified
- Row scope: only `cache["scorer_train"]` captions reach the model; asserted `len(captions) == len(st)` and that
  no selection/val/held row overlaps scorer-train. Probe split asserted inside scorer-train and painting-disjoint.
- Affect array (183,694, 28), finite, in [0, 1] (asserted). Extraction 84.7 s on the RTX 3090.
- C0 and S reference checkpoints: stored configs equal `grid.cell_config("C0"/"S", 42)` (asserted); SHA-256 in
  `cache/affect_prepare.json` (C0 7653caf0985b..., S 33d35943ec62...).
- Image and caption AMIs reproduce the spec's 0.035 / 0.318 and 0.056 / 0.058.

## Prepare diagnostics (measured, never used for a choice)
k-means k=64: 64 non-empty groups, all 64 have >= 200 rows; sizes min / median / max = 244 / 1690 / 21098.

| Partition | AMI with emotion | AMI with art style |
|---|---|---|
| affect k=64 | 0.196 | 0.016 |
| CLIP image | 0.035 | 0.318 |
| CLIP caption | 0.056 | 0.058 |

Probe (logistic, C=1, standardized; fit on 146,960 rows of 80% of scorer-train paintings, scored on 36,734 rows):

| Input -> emotion | Accuracy |
|---|---|
| affect-28 | 0.517 |
| CLIP caption features | 0.577 |
| majority class | 0.285 |

## Step 5 smoke table (local RTX 3090)

| Cell | s/step | fixed s | projected full run (min) | peak GiB |
|---|---|---|---|---|
| E | 0.2687 | 0.74 | 8.97 | 3.87 |
| SE | 0.2781 | 0.27 | 9.28 | 3.86 |
| C0 (2x2 smoke) | 0.2477 | 1.37 | 8.28 | 3.86 |

Projected: selection (E + SE) 18.2 min; replication 2 x (SE + C0) 35.1 min; total 53.4 min sequential; peak 3.87 GiB.
Thresholds: one run > 45 min, total > 3 h, peak > 20 GiB: none exceeded. Decision: `run_locally`.
