# Does the combiner-architecture win survive at real scale? — Full-Scale Validation

**Date:** 2026-09-03 · **Dataset:** RedCaps 500k-diverse (`redcaps_500k_diverse` / `redcaps_500k_diverse_cluster`) · **Branch:** `experiment/condition_drift_retrieval_correlation`
**Code:** `scripts/run_combiner_architecture_fullscale.sh`, `configs/dataset/redcaps_500k_diverse_cluster.yaml`, `scripts/analyze_combiner_architecture_smoke.py` (reused as-is, `--group` already supported)
**Prior report:** `docs/reports/auto/buddy/2026-09-02_combiner_architecture_ablation.md` (smoke-scale ablation this validates)
**Infra:** DAS6 node411, via `.claude/skills/cluster-run` — first fully successful end-to-end use of this skill (previous sessions never got past environment provisioning)

---

## TL;DR

The smoke-scale report's headline result — `lowrank` (rank-16 residual adapter) beats `legacy` on oracle i2t R1 — **survives at full scale, and by a larger margin**: +4.10 R1 at 500k/100 epochs vs. +3.10 R1 at 150k/30 epochs (n=3). Oracle t2i R1 stays a small, consistent regression in both regimes (-0.70 at full scale vs. -0.80 at smoke scale).

**The deployment-tier (`pre_diff`) cost is larger at full scale than the smoke sweep suggested**: i2t pre_diff regressed -8.00 R1 here vs. -3.63 R1 at smoke scale (t2i: -6.20 vs. -3.60). Both regimes agree on direction and that the deployment cost is well below `residual_control`/`film`'s collapse (-14 to -28 R1), but the gap is roughly 2x the smoke sweep's number, not a clean match.

**This is n=1 per arm, not the smoke sweep's n=3** — a first-signal pass, exactly as scoped going in. The direction and rough magnitude of the oracle win are corroborated by the smoke sweep's 3-seed result; the deployment-tier number is not yet corroborated at this scale and shouldn't be treated as settled.

---

## Method

Same fixed operating point as the smoke sweep (`K=30`, `alpha=0.5`, `buddy_dim=16`, `lr=1e-3`/`lr_label=1e-4`, `initialization_strategy=buddies`), scaled to this project's real training regime: `redcaps_500k_diverse`, 100 epochs, 1 seed per arm (`legacy` vs. `lowrank`). `film` and `residual_control` were not re-run at scale — the smoke sweep's deployment-tier collapse for both (-14 to -28 R1) was already large enough at n=3 not to need a scale check before ruling them out.

Run via the `cluster-run` skill on DAS6 node411:

```bash
SMOKE=1 bash scripts/run_combiner_architecture_fullscale.sh   # pipeline sanity (lowrank, 2 epochs) — confirmed clean first
bash scripts/run_combiner_architecture_fullscale.sh           # legacy + lowrank, 100 epochs, seed=1
python scripts/analyze_combiner_architecture_smoke.py --group 'combiner architecture fullscale'
```

Getting this far required fixing several gaps in the cluster-run infrastructure itself, none specific to this experiment:

- **Missing raw eval assets on the node**: `CoSiRValidationDataset` reads test images live every run, but `cluster_sync_up.sh`'s data sync only ever covered the pre-extracted feature cache. Pushed a 5,000-image subset + both annotation JSONs to `/var/scratch/wding/Dataset/redcaps_plus/` manually and wrote `configs/dataset/redcaps_500k_diverse_cluster.yaml` to point at them, keeping the same in-domain RedCaps test set the smoke report used (not the COCO test set the existing `redcaps_cluster.yaml` substitutes).
- **`requirements.txt` under-declared two hard imports**: `dask-cuda` (unconditionally pulled in via `src/model/__init__.py` → `clustering.py`) and `psutil` (`src/utils/feature_manager.py`) were both installed locally but missing from the file entirely (the former commented out). Added both.
- **`cuml-cu12==25.8.0` breaks under `scikit-learn>=1.9`**: a fresh env picked up the newest sklearn by default and `cuml.accel`'s estimator-proxy shim references a private sklearn API (`BaseEstimator._get_default_requests`) removed in 1.9. Pinned `scikit-learn<1.9` in `requirements.txt`.
- **`.clusterignore` let ~2.6GB of local debugging feature caches through**: `src/test/20260708_heldout_grid/heldout_feats/` and `src/test/20260623_redcaps_buddy/dino_feats.npy` — scratch data from unrelated debugging sessions (per this repo's `src/test/yyyymmdd_*/` convention), never touched by the training path. Now excluded, along with `multirun/` (Hydra's local sweep output, same category as the already-ignored `outputs/`) and `.history/` (editor local-history plugin data). Cut a code-only push from 2.58GB to ~78MB.
- **`cluster_launch.sh`'s node-autodetection only matched a fish-shell prompt** (`wding at node4XX`). The reservation's shell changed to bash mid-session (a deliberate switch, "fish is a bit messy on the cluster"), producing a different prompt format the regex never matched. Added a fallback that matches the tmux *window name* instead (`srun-node411-gpu-02`, set by `srun` itself) — more robust than parsing prompt text since it doesn't depend on shell/prompt configuration at all.

---

## Results

**n=1 per arm — see caveat above before treating this as settled.**

| metric | legacy | lowrank | Δ (lowrank − legacy) | smoke-scale Δ (n=3) |
|---|---:|---:|---:|---:|
| oracle t2i R1 | 26.50 | 25.80 | -0.70 | -0.80 |
| oracle i2t R1 | 10.60 | 14.70 | **+4.10** | +3.10 |
| pre_diff t2i R1 | -2.20 | -8.40 | -6.20 | -3.60 |
| pre_diff i2t R1 | -4.70 | -12.70 | -8.00 | -3.63 |

Runs: `res/CoSiR_combiner_architecture_fullscale/legacy/20260902_221847_CoSiR_Experiment/`, `res/CoSiR_combiner_architecture_fullscale/lowrank/20260902_235245_CoSiR_Experiment/` (wandb group `combiner architecture fullscale`; a `SMOKE=1` sanity run sharing `lowrank`/seed=1 was correctly excluded by the analyzer's existing max-epoch dedup).

**Reading:** the oracle i2t win — the report's central claim — replicates directionally and quantitatively at full scale, if anything strengthening. The deployment-tier cost is real in both regimes but roughly doubles at full scale; with n=1 here it's not possible to tell whether that's a genuine scale-dependent effect or single-seed variance landing on the larger side. Worth a multi-seed full-scale re-run before this number is used in anything paper-facing.

---

## What wasn't done, and why

- **Multi-seed at full scale**: this run was explicitly scoped as a first-signal pass to validate the pipeline and get a first real-scale read cheaply, not to reproduce the smoke sweep's n=3 rigor. The oracle-tier finding is corroborated well enough by the smoke sweep to act on; the deployment-tier number is not, and should get a 2-3 seed full-scale re-run before being treated as final — particularly since it moved more between regimes than the oracle number did.
- **`film`/`residual_control` at scale**: not re-run — already ruled out clearly enough at smoke scale (see TL;DR).
- **Hydra's per-run `outputs/<date>/<time>/` provenance**: same known, deliberately-unresolved gap noted in the cluster-run skill itself — not captured for this run. wandb's own config/summary logging is treated as sufficient for this validation pass.

## Recommendation

Treat `lowrank` as validated for the oracle-tier win the smoke report proposed adopting it for. Before flipping `configs/model/*.yaml`'s default away from `legacy`, get a multi-seed full-scale read on the deployment-tier (`pre_diff`) regression specifically — it's the one number here that didn't just confirm the smoke-scale read, and it's the number that matters most for any real deployment use of buddy conditioning.
