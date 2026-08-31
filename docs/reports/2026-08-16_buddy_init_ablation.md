# Does buddy-graph initialization beat generic (imgtxt) initialization?

**Date:** 2026-08-16 · **Datasets:** Impressions (N = 12,123), RedCaps-150k (N = 150,000) · **Branch:** `experiment/buddy_init_ablation`
**Code:** `scripts/run_init_ablation.sh`, `scripts/run_init_ablation_impressions.sh`, `scripts/run_init_ablation_redcaps.sh`, `scripts/analyze_init_ablation.py`
**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 1
**Plan:** `docs/superpowers/plans/2026-08-10-buddy-init-ablation.md` (Task 6)

---

## TL;DR

This is the paper's foundational, previously-unmeasured claim: does initializing each sample's trainable condition vector from the cross-modal buddy graph actually beat the prior generic (`imgtxt`) initialization, with every training-time buddy term held off? The result is **real but directionally mixed, not a clean win**:

- **RedCaps: unambiguous win.** Buddy-init beats imgtxt-init on both retrieval directions, seed-replicated, large effect (t2i mean Δ = **+4.00 ± 0.26** R1 pts, mean/SEM = +26.2; i2t mean Δ = **+4.57 ± 0.72**, mean/SEM = +10.9).
- **Impressions: split by direction.** i2t favors buddy-init (mean Δ = **+3.00 ± 1.18**, mean/SEM = +4.4) but t2i favors imgtxt-init instead (mean Δ = **−1.60 ± 0.62**, mean/SEM = −4.4) — both reliably above the ~0.1–0.7 R1 noise floor, i.e. both are real effects, just opposite in sign.
- **Net: 3 of 4 (dataset × direction) cells favor buddy-init**, including the two largest-magnitude effects (RedCaps). The one exception (Impressions t2i) is itself a seed-replicated, significant effect, not noise — a genuine caveat, not an artifact to wave away.

This does **not** cleanly satisfy the spec's strict "Positive" criterion (clean win on ≥2/3 datasets tested) as written, because Impressions isn't a clean win. It is also clearly not "Null" or "Negative" as written, since 3 of 4 cells show large, reliable, positive effects. See "Applying the decision rule" below for how this is resolved.

---

## Method

**Operating point (fixed across both datasets, per the spec's Experiment 1 design):**

| | value |
|---|---|
| `optimizer.lr` | 1e-3 |
| `optimizer.lr_label` | 1e-4 |
| `model.embedding_dim` | 16 |
| `train.buddies.alpha` | 0.5 |
| `model.num_layers` | 6 |
| training-time buddy terms | **off** — `lambda_buddy=0`, `lambda_buddy_con=0`, `buddy_refresh=False` (all default/absent, never overridden) |
| seeds | 1, 2, 3 |
| Impressions epochs / eval interval | 250 / 50 |
| RedCaps epochs / eval interval / test_ratio | 100 / 10 / 0.2 |

`initialization_strategy ∈ {imgtxt, buddies}` is a **template-compatibility key** (`src/hook/train_cosir.py:244-271`): a template built under one strategy is rejected and silently rebuilt under another, so each strategy gets its own `results_dir` to avoid two multirun processes racing on the same `template_embeddings/` directory (`scripts/run_init_ablation.sh`). Seeds are swept via Hydra multirun within each strategy's own results directory.

**Run environment:** executed locally (RTX 3090) rather than on the originally-planned cluster — the sandbox's `cuml`/`numba-cuda` stack was broken (libstdc++ symbol conflict + a `cudf-cu13`/`numba` version-drift install that didn't match this repo's pinned `cudf-cu12==25.8.*`) and was repaired earlier in this session; see the environment-fix discussion earlier in this thread for the root-cause breakdown. All 12 runs (2 strategies × 3 seeds × 2 datasets) completed with no errors, logged online to `wandb` (`entity=augustoxq`, `project=cosir_image`, `group="buddy-init ablation"`, tags `init-ablation-impressions` / `init-ablation-redcaps`).

**Analysis:** `scripts/analyze_init_ablation.py --tag <tag>` (Task 4) — paired-within-seed Δ (`buddies − imgtxt`) at the matched operating point, mean ± std across the 3 seeds, `mean/SEM` as the significance read (project convention: `|mean/SEM| ≥ 2` reads as significant), compared against the measured noise floor (~0.1–0.7 R1 from a duplicate-config run, not against zero) per `docs/reports/2026-06-24_buddy_progress_report.md` §8a.

## Results

### Impressions (250 epochs)

**`test_oracle/t2i_R1`**

| seed | imgtxt | buddies | Δ (buddies − imgtxt) |
|---:|---:|---:|---:|
| 1 | 63.00 | 61.60 | −1.40 |
| 2 | 62.90 | 60.60 | −2.30 |
| 3 | 61.80 | 60.70 | −1.10 |

Over 3 paired cells: buddies beats imgtxt in **0/3**. Mean Δ = **−1.60 ± 0.62** R1 pts, mean/SEM = **−4.4**.

**`test_oracle/i2t_R1`**

| seed | imgtxt | buddies | Δ (buddies − imgtxt) |
|---:|---:|---:|---:|
| 1 | 71.80 | 75.80 | +4.00 |
| 2 | 72.10 | 75.40 | +3.30 |
| 3 | 73.20 | 74.90 | +1.70 |

Over 3 paired cells: buddies beats imgtxt in **3/3**. Mean Δ = **+3.00 ± 1.18** R1 pts, mean/SEM = **+4.4**.

### RedCaps-150k (100 epochs)

**`test_oracle/t2i_R1`**

| seed | imgtxt | buddies | Δ (buddies − imgtxt) |
|---:|---:|---:|---:|
| 1 | 21.80 | 26.10 | +4.30 |
| 2 | 22.40 | 26.20 | +3.80 |
| 3 | 22.30 | 26.20 | +3.90 |

Over 3 paired cells: buddies beats imgtxt in **3/3**. Mean Δ = **+4.00 ± 0.26** R1 pts, mean/SEM = **+26.2**.

**`test_oracle/i2t_R1`**

| seed | imgtxt | buddies | Δ (buddies − imgtxt) |
|---:|---:|---:|---:|
| 1 | 9.00 | 14.40 | +5.40 |
| 2 | 10.10 | 14.20 | +4.10 |
| 3 | 10.00 | 14.20 | +4.20 |

Over 3 paired cells: buddies beats imgtxt in **3/3**. Mean Δ = **+4.57 ± 0.72** R1 pts, mean/SEM = **+10.9**.

## Applying the decision rule

The spec's §4 Experiment 1 decision rule (written before any data existed) is a three-way split: Positive (clean win, mean/SEM ≥ 2, on ≥2/3 datasets) / Null (no reliable difference) / Negative (imgtxt wins). The actual result doesn't fall cleanly into any one bucket — it's positive on 3 of 4 (dataset × direction) cells and reliably negative on the fourth, all four effects clearing the mean/SEM ≥ 2 significance bar. Treating this honestly:

- **Not "Positive" as strictly written** — that required a *clean* per-dataset win, and Impressions isn't clean (t2i loses).
- **Not "Null"** — every cell shows a reliable effect well above the measured noise floor; there's no cell where the true answer is "no difference."
- **Not "Negative"** — 3 of 4 cells, including both RedCaps cells (the largest effects observed anywhere in this ablation), favor buddy-init.
- **Best characterization: a real, direction-and-dataset-dependent effect, net positive.** RedCaps shows buddy-init is unambiguously better. Impressions shows buddy-init trading t2i performance for i2t performance — worth a follow-up look (not commissioned by this task) at whether this tracks the same near-duplicate-photo structure C4 found on Impressions (Impressions has repeated source photos; RedCaps does not), or reflects a genuine t2i/i2t trade-off in how the buddy-derived initialization shapes the condition space.

**Consequence for the spec's venue tiering (§3.2):** this does **not** unlock the stretch tier (ICLR/NeurIPS main track) as written — that gate required a *clean* ≥2/3-dataset win, and Impressions isn't clean. It **comfortably supports the primary TMLR framing** (§3.3), which was explicitly designed to survive any Experiment 1 outcome: this result *upgrades* the paper from "signal validated, training-use mostly doesn't work" (the pre-Experiment-1 framing) to "signal validated, and useful as an initializer on the stronger-signal dataset (RedCaps), with a direction-dependent trade-off on the dataset most affected by near-duplicate structure (Impressions) — itself a citable, on-theme finding given C4's near-duplicate-confound result." This is a strictly more interesting paper than either a clean positive or a null result would have been.

## Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
bash scripts/run_init_ablation_impressions.sh   # 2 strategies x 3 seeds, 250 epochs each
bash scripts/run_init_ablation_redcaps.sh       # 2 strategies x 3 seeds, 100 epochs each
python scripts/analyze_init_ablation.py --tag init-ablation-impressions
python scripts/analyze_init_ablation.py --tag init-ablation-redcaps
```

## Caveats

- COCO (spec's Experiment 5, stretch/gated) was not run — the "≥2/3 datasets" language in the original decision rule anticipated a possible third dataset that isn't in scope here; only 2 datasets were tested.
- The Impressions t2i regression is not explained by this task — it's flagged as a follow-up question, not diagnosed here.
- Runs used only the single previously-established operating point (lr, lr_label, dim, alpha) — no sensitivity sweep on whether the t2i/i2t split on Impressions is operating-point-specific.
