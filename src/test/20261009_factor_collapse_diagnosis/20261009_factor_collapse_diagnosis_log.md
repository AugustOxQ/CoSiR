# 2026-10-09 factor-collapse diagnosis log

## Problem

Candidate A's factor space (32 non-negative factors over frozen CLIP features of ArtELingo) collapsed onto essentially one axis (participation ratio 1.32, 374/496 factor pairs with |r| >= 0.9). The code review named five suspects: a non-contrastive (cosine) agreement loss, no decorrelation term, a balance loss that copies of one axis can satisfy, weak sparsity, and uncentered inputs. Task 3 tests them one variable at a time with the EXISTING training code before Task 4 builds a fix. No file under `src/` was changed.

## Investigation steps

1. `run_diagnosis.py` (repo root, CoSiR env, RTX 3090, seed 42): load ArtELingo, build leakage groups, `grouped_split(seed=42)` giving train/val/held = 216,107 / 30,872 / 61,744 rows and zero painting or image-vector leakage (asserted).
2. Content graph (`GraphConfig()`, 2,198,162 edges), Stage 1 (`Stage1Config()`) and Leiden communities (22) on the `train` rows only, built once and shared by all variants.
3. Seven variants (D0-D6), each one `train_factors` call on `train` rows with `FactorTrainingConfig` overrides exactly as pre-registered (seed 42, 32 factors, 2,000 epochs). `val` rows encoded by the returned model in eval mode, `torch.no_grad()`, 8,192-row batches. D6 passes `train - train_mean` to `train_factors` and encodes `val - train_mean`; the gates always receive the original uncentered features.
4. `evaluate_factor_gates` with default thresholds: fit = train rows (codes from `train_factors`, original train features), eval = val rows, community codes / labels = train. Added on top: PC1 variance share of the val codes per modality.
5. Each variant was run exactly once (whole run 288.6 s, of which graph + Stage 1 + communities 54.9 s, about 33 s per variant). Per-variant JSON is in `results/` (gitignored); `make_tables.py` regenerates every table in the report from it. `feature_norms.py` is a read-only context check (feature norms).

## Root cause

Under the pre-registered rule exactly one term is implicated: the cosine paired-agreement loss (`lambda_paired`). Removing it (D3) raises the smaller participation ratio from 1.309 to 17.074 and drops max |r| from 0.99934 to 0.79963, with 8 of 9 gates passing. Removing the balance term (D4) or the graph term (D5) does not help, and centering the encoder input (D6) does not help. The plain reconstruction autoencoder (D1) is not collapsed (participation ratio 5.98 / 3.20, above the pre-registered 3.0). The D0 collapse also reproduces on the painting-grouped split, so the earlier row-split leakage was not its cause. Pair retrieval fails in every variant (best 0.397 vs the 0.5 gate), so removal alone is not a repair. Full tables, the reading and caveats are in `docs/reports/2026-10-09_cosir_v2_candidate_a_factor_collapse_diagnosis.md`.

## Solution implemented

None by design: this task is diagnostic and changes no training code. Output is the script, this log and the report; the report recommends which Task 4 mechanisms the evidence supports.

## Files

- `run_diagnosis.py`, `make_tables.py`, `feature_norms.py` (committed)
- `results/`, `cache/`, `run_diagnosis.log`, `feature_norms.log` (gitignored, regenerable by re-running the scripts)
