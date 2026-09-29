# 2026-10-11 factor-repair grid log

## Problem

Candidate A's shared factor space (32 non-negative factors over frozen CLIP ArtELingo features) collapsed onto one axis. Task 3 traced the collapse to the cosine paired-agreement loss. Task 4 added four default-off mechanisms: InfoNCE agreement, a decorrelation penalty, TopK activation and input centering. This task trains the nine pre-registered configurations R0-R8 of plan Task 6, including R8 (`lambda_paired=0.0`), which was added with user approval at the Task 3 checkpoint.

A run is eligible only if it passes all nine geometry gates on `val`. Among eligible runs, the one with the largest condition-specific benefit on human-label episodes (emotion and art style) is selected. The selected recipe is then replicated across seeds.

## Investigation steps

1. `run_grid.py --smoke` (3 epochs, 256 episodes) exercised every code path once:
   - selection, forced in smoke mode only;
   - replication and Hungarian alignment;
   - the held check, using val rows as a stand-in;
   - checkpoint save and reload.

   Everything was written under `cache/smoke/`. Held rows were never encoded. The smoke numbers were discarded, and the smoke checkpoints were deleted afterwards so they cannot be mistaken for Task 7 inputs. Only `cache/smoke/smoke.log` remains.
2. The real run was `run_grid.py`, executed once (seed 42, RTX 3090, torch 2.11.0+cu130). Total runtime was 387.4 s.
   - **Data:** `load_artelingo()` gives 308,723 rows. Rows are grouped by `leakage_groups`, then `grouped_split(seed=42)` gives train/val/held = 216,107 / 30,872 / 61,744 rows. Painting and image-vector leakage is zero (asserted).
   - **Art style (Ruling 12):** joined positionally via `annotations[int(i)]["art_style"]` over `data.sample_ids`. There are 27 styles. The script asserts that each painting, and each leakage group, has exactly one style.
   - **Train-side graph:** content graph, Stage 1 and Leiden communities, rebuilt on the train rows (54.8 s): 2,198,162 edges, 22 communities. These are identical to Task 3's counts. Task 3's cache was not loaded.
   - **Validation label episodes (Ruling 13):** built once and shared by all nine runs.
     - `paintings=` is the full leakage-group array, and `exclude_target_paintings_from_negatives=True` for both label types.
     - Emotion: `exclude_target_labels=("something else",)`, 2,048 episodes over 8 target labels.
     - Art style: 2,048 episodes over 23 target labels. Four styles have fewer than 30 val paintings, so they are not targets.
     - Each episode has an anchor, a positive, 4 supports, 4 contrasts and 12 distractors, so there are 13 candidates and chance R@1 is 1/13.
     - The script asserts three properties. (a) Every episode row is a val row. (b) No painting group repeats within an episode. (c) Building the episodes from val-only arrays gives the same episodes once indices are mapped back through `val`. So no non-val row influences them.
   - **Runs R0-R8:** each run uses base `lambda_usage_balance=0.1`, seed 42, 32 factors and 2,000 epochs. `group_ids = leakage_groups(...)[train]` (int64). ORIGINAL features are always passed; R6's model stores the train means itself.
     - Val rows are encoded by the returned model in eval mode, under `no_grad`, in 8,192-row batches.
     - All train and val codes were asserted finite. All nine runs were finite, with finite losses throughout.
     - Gates: fit = train, eval = val, community codes/labels = train, default thresholds.
     - `condition_lift` was computed on both episode sets. Selection score = mean of the two `lift_mean` values.
   - **Pre-registered selection (Step 4)** was applied in code, in `select()`.

## Result (root cause of the stop)

**No run passes all nine gates on val.** Under Step 4.4, no recipe is selected and the task stops. These steps were not run: seed replication (Step 5), the held check, and the checkpoints `checkpoints/selected_seed42.pt` / `checkpoints/R0_seed42.pt` (Step 6). Held rows were never encoded. The gate sanity check (Step 4.5) does not apply because there is no passing run, so there is no flag.

| run | gates | binding gates | selection score (R@1 points) |
|---|---:|---|---:|
| R0 | 4/9 | participation_ratio, redundancy, readout (txt), sparsity, pair_retrieval | +0.55 |
| R1 | 7/9 | readout (txt +0.0160), sparsity (0.484) | +3.85 |
| R2 | 6/9 | participation_ratio (2.86), redundancy (0.950), sparsity (0.839) | +3.31 |
| R3 | 7/9 | readout (txt +0.0148), sparsity (0.487) | +4.17 |
| R4 | 7/9 | readout (img +0.0051, txt +0.0215), modality_private (9) | +2.37 |
| R5 | 7/9 | readout (img +0.0048, txt +0.0136), modality_private (14) | +2.70 |
| R6 | 7/9 | readout (txt +0.0156), sparsity (0.486) | +4.99 |
| R7 | 6/9 | readout (txt +0.0209), sparsity (0.405), modality_private (3) | +3.91 |
| R8 | 8/9 | pair_retrieval (0.397) | +3.38 |

Readout margins are the code readout minus the CLIP PCA-10 value (it must be <= 0). Sparsity is the larger per-modality active fraction (it must be <= 0.375).

Reproduction check: R0 reproduces Task 3's D0, and R8 reproduces D3, to every printed decimal. For example, R0 has PR 1.342/1.309, max |r| 0.9993 and ratio 0.3732; R8 has PR 20.629/17.074, max |r| 0.7996 and ratio 0.3974. Training on this setup is deterministic.

## Solution implemented

None. Per the pre-registered rule, the user decides the next step, and no runs were added and no thresholds changed. Full tables, the verdict and the options are in `docs/reports/2026-10-11_cosir_v2_candidate_a_factor_repair.md`. There is no selected `FactorTrainingConfig`.

## Files

- Committed: `run_grid.py`, this log, `.gitignore`.
- Gitignored: `results/*.json` (per-run results and `summary.json`), `run_grid.log`, `cache/smoke/smoke.log`.
- Tables can be regenerated with `run_grid.py --tables`. No `checkpoints/` directory exists because Step 6 was not reached.
