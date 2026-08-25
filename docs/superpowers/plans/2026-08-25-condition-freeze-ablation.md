# Condition Freeze Ablation (Experiment 11.1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer Experiment 11.1 from `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 — does post-init training of the per-sample condition table do anything, holding buddy-init geometry and every other hyperparameter identical between a **frozen** arm (conditions fixed at their buddy-init value for the whole run) and the **trained** arm (today's default)? Answer this on two independent axes: retrieval (`test_oracle/t2i_R1`, `test_oracle/i2t_R1`) and embedding geometry (how much conditioning shifts the combine-side embedding, how that shift distribution and its most/least-changed samples compare across arms and epochs) — because retrieval alone can miss a real divergence.

**Architecture:** No new training-time mechanism is needed — `em_interval` (`configs/train/default.yaml`, EM-alternation) already sets `embedding_manager.embeddings.requires_grad_(False)` whenever `epoch // em_interval` is even, so a sentinel value ≥ `epochs` keeps that freeze active for the whole run. Because `em_interval` is **not** part of the buddy-init template-compatibility key (`_extra` in `_init_embedding_manager`, `src/hook/train_cosir.py`), both arms can — and, for a true paired comparison, must — share one `results_dir` and therefore one buddy-init template: every run in this experiment starts from a byte-identical buddy-init, and only whether gradient reaches the condition table afterward differs. The only source change needed (Task 1) is threading the already-available `sample_ids` list into the per-epoch `condition_viz/epoch_XXXX.pt` snapshot that `_save_condition_viz_snapshot` already writes unconditionally — everything else (per-epoch condition table + combiner weights, frozen train-set CLIP features, buddy-graph edges) is already cached by existing code. From there: a sweep script toggles `em_interval` per arm (Task 2), a paired retrieval-analysis script mirrors the project's existing `analyze_*.py` pattern (Task 3), and a new post-hoc geometry-diagnostic script rebuilds each saved epoch's combiner to compute and rank per-sample conditioning shift (Tasks 4–5, split into an offline-testable pure-math core and the real-data integration).

**Tech Stack:** Python 3.10, Hydra/OmegaConf, PyTorch, wandb, numpy/pandas/scipy/matplotlib. Existing CoSiR training entrypoint `main_cosir.py`; existing RedCaps-150k `FeatureManager`/`redcaps_buddy.load_data()`; existing `reorder_features_to_z` (`src/metrics/regularizer.py`). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 11.1 (added 2026-08-25).

## Global Constraints

- Always run Python/bash training or analysis commands with `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR` first (per `.claude/CLAUDE.md`).
- Statistical standard (spec §5): ≥3 seeds, paired-within-seed Δ, report mean ± std and mean/SEM (flag `|z| ≥ 2` with `*`), compare against the noise floor (~0.1–0.7 R1), never against zero.
- Fixed operating point (matches Exp. 1/8/10's confirmed strong cell): `lr=1e-3`, `lr_label=1e-4`, `embedding_dim=16`, `alpha=0.5`. `initialization_strategy=buddies` fixed throughout.
- Every training-time buddy term stays OFF: never pass `+loss.lambda_buddy`, `+loss.lambda_buddy_con`, or `+loss.buddy_refresh*`. Omitting them gives the code's own default of `0.0`/`False`.
- **`train.em_interval` is the ONLY axis distinguishing the two arms, and it is deliberately NOT a template-compatibility key.** Unlike every prior sweep in this plan (`initialization_strategy`, `encoder_pair`, `distance_mode`, each requiring its own `results_dir` to avoid template races), both arms of this experiment point at **one shared `results_dir`** so they load the exact same cached buddy-init template. Do not "fix" this into a per-arm `results_dir` split — that would weaken the pairing this experiment exists to test.
- Scope: RedCaps-150k only (`dataset=redcaps_150k`), 3 seeds × 2 arms = 6 runs. This plan covers 11.1 only — 11.2 is explicitly gated on 11.1's result and out of scope here (per spec).
- One existing `src/` file is modified in this plan (`src/hook/train_cosir.py`). Per CLAUDE.md, log the change in `.claude/20260825_log.md` (one `# <path>` section).
- Debugging/verification scripts for this plan live in `src/test/20260825_condition_freeze_ablation/`, per CLAUDE.md's dated-folder convention. Reusable sweep/analysis scripts live in `scripts/`, matching every prior experiment in this plan.
- wandb defaults: `entity=augustoxq`, `project=cosir_image` (`configs/config.yaml`).
- Task 6 (launching the real 6-run sweep) is a multi-hour GPU commitment — **get explicit user confirmation before running it**, even though this is a plan already approved end-to-end. Do not auto-launch it as part of unattended execution.

---

### Task 1: Save `sample_ids` in the per-epoch condition_viz snapshot

**Files:**
- Modify: `src/hook/train_cosir.py:531-546` (`_save_condition_viz_snapshot` signature), `:581-618` (saved dict), `:933-934` (`get_all_embeddings()` call site), `:1039-1053` (call site)
- Create: `src/test/20260825_condition_freeze_ablation/verify_sample_ids_in_snapshot.py`
- Modify: `.claude/20260825_log.md` (create; log this change)

**Interfaces:**
- Consumes: `TrainableEmbeddingManager.get_all_embeddings() -> (sample_ids: List[int], embeddings: Tensor)` (unchanged, already returns `self.sample_ids` as its first element — currently discarded at the call site).
- Produces: `condition_viz/epoch_XXXX.pt` now includes `"sample_ids": List[int]`, row-aligned with `"label_embeddings_all"`. Consumed by Task 5's geometry script (`reorder_features_to_z` needs these as `z_ids`) and by Task 3's ranking-by-sample-id output.

- [ ] **Step 1: Modify `_eval_snapshot`'s `get_all_embeddings()` call**

In `src/hook/train_cosir.py`, find (around line 933-934):

```python
        print("Getting all embeddings")
        _, label_embeddings_all = embedding_manager.get_all_embeddings()
```

Replace with:

```python
        print("Getting all embeddings")
        label_ids_all, label_embeddings_all = embedding_manager.get_all_embeddings()
```

- [ ] **Step 2: Thread `label_ids_all` into `_save_condition_viz_snapshot`'s call site**

Find (around line 1039-1053):

```python
        _save_condition_viz_snapshot(
            cfg,
            epoch,
            experiment,
            model,
            all_img_emb,
            all_txt_emb,
            all_raw_text,
            image_to_text_map,
            text_to_image_map,
            test_set,
            label_embeddings_all,
            representatives,
            sample_types,
        )
```

Replace with:

```python
        _save_condition_viz_snapshot(
            cfg,
            epoch,
            experiment,
            model,
            all_img_emb,
            all_txt_emb,
            all_raw_text,
            image_to_text_map,
            text_to_image_map,
            test_set,
            label_embeddings_all,
            label_ids_all,
            representatives,
            sample_types,
        )
```

- [ ] **Step 3: Add the parameter to `_save_condition_viz_snapshot` and save it**

Find the function signature (around line 531-546):

```python
def _save_condition_viz_snapshot(
    cfg,
    epoch,
    experiment,
    model,
    all_img_emb,
    all_txt_emb,
    all_raw_text,
    image_to_text_map,
    text_to_image_map,
    test_set,
    label_embeddings_all,
    representatives,
    sample_types,
):
```

Replace with:

```python
def _save_condition_viz_snapshot(
    cfg,
    epoch,
    experiment,
    model,
    all_img_emb,
    all_txt_emb,
    all_raw_text,
    image_to_text_map,
    text_to_image_map,
    test_set,
    label_embeddings_all,
    label_ids_all,
    representatives,
    sample_types,
):
```

Then find, inside the same function, the per-epoch `torch.save` dict (around line 583-586):

```python
    epoch_path = cond_viz_dir / f"epoch_{epoch:04d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "label_embeddings_all": label_embeddings_all.cpu(),
```

Replace with:

```python
    epoch_path = cond_viz_dir / f"epoch_{epoch:04d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "label_embeddings_all": label_embeddings_all.cpu(),
            "sample_ids": list(label_ids_all),
```

- [ ] **Step 4: Write the verification script**

Create `src/test/20260825_condition_freeze_ablation/verify_sample_ids_in_snapshot.py`:

```python
"""
Verify Task 1 of the condition-freeze-ablation plan (docs/superpowers/plans/
2026-08-25-condition-freeze-ablation.md): condition_viz/epoch_*.pt now carries a
'sample_ids' field whose length and values exactly match the run's own persisted
training_embeddings/sample_ids.npy (the TrainableEmbeddingManager's ground-truth z-table
order) -- the CLAUDE.md 'sample ID consistency' check, applied to this new field.

Run against a completed smoke run's experiment directory:
    python src/test/20260825_condition_freeze_ablation/verify_sample_ids_in_snapshot.py <exp_dir>
"""
import sys
from pathlib import Path

import numpy as np
import torch


def verify(exp_dir: str) -> None:
    exp_path = Path(exp_dir)
    cond_viz_dir = exp_path / "condition_viz"
    epoch_files = sorted(cond_viz_dir.glob("epoch_*.pt"))
    assert epoch_files, f"no condition_viz/epoch_*.pt under {exp_dir}"

    truth_path = exp_path / "training_embeddings" / "sample_ids.npy"
    assert truth_path.exists(), f"missing ground-truth {truth_path}"
    truth_ids = sorted(int(x) for x in np.load(truth_path).tolist())

    for ef in epoch_files:
        snap = torch.load(ef, map_location="cpu")
        assert "sample_ids" in snap, f"{ef} missing 'sample_ids' key"
        ids = snap["sample_ids"]
        n_emb = snap["label_embeddings_all"].shape[0]
        assert len(ids) == n_emb, (
            f"{ef}: sample_ids length {len(ids)} != label_embeddings_all rows {n_emb}"
        )
        assert sorted(int(x) for x in ids) == truth_ids, (
            f"{ef}: sample_ids do not match training_embeddings/sample_ids.npy "
            f"(len {len(ids)} vs {len(truth_ids)}, or values differ)"
        )
        print(f"PASS {ef.name}: {len(ids)} sample_ids, all match ground truth")

    print(f"ALL {len(epoch_files)} SNAPSHOT(S) VERIFIED")


if __name__ == "__main__":
    assert len(sys.argv) == 2, "usage: verify_sample_ids_in_snapshot.py <exp_dir>"
    verify(sys.argv[1])
```

- [ ] **Step 5: Run a smoke training run and verify**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python main_cosir.py \
  dataset=redcaps_150k eval.evaluation_interval=1 eval.oracle_aggregation=max \
  eval.test_ratio=0.2 model=clip_base model.num_layers=6 model.embedding_dim=16 \
  optimizer.lr=1e-3 optimizer.lr_label=1e-4 seed=1 \
  train.initialization_strategy=buddies train.buddies.alpha=0.5 train.epochs=2 \
  experiment.results_dir=res/CoSiR_condition_freeze_ablation/_smoke/task1 \
  wandb.group="condition-freeze-ablation-smoke"
```

Note the printed experiment directory (e.g. `res/CoSiR_condition_freeze_ablation/_smoke/task1/<timestamp>_<name>/`), then:

```bash
python src/test/20260825_condition_freeze_ablation/verify_sample_ids_in_snapshot.py \
  res/CoSiR_condition_freeze_ablation/_smoke/task1/<timestamp>_<name>
```

Expected: `PASS epoch_0000.pt: <N> sample_ids, all match ground truth`, `PASS epoch_0001.pt: ...`, `ALL 2 SNAPSHOT(S) VERIFIED`.

- [ ] **Step 6: Log the change**

Create `.claude/20260825_log.md`:

```markdown
# /src/hook/train_cosir.py

## `_save_condition_viz_snapshot`: added `sample_ids` to the per-epoch snapshot

**Before:** `condition_viz/epoch_XXXX.pt` saved `label_embeddings_all` (the per-sample
condition table at that epoch) with no record of which training sample each row belongs to
— `get_all_embeddings()`'s `sample_ids` return value was discarded at the call site.

**After:** `_eval_snapshot` now keeps `label_ids_all` from `get_all_embeddings()` and passes
it through to `_save_condition_viz_snapshot`, which saves it as `"sample_ids"` alongside
`"label_embeddings_all"`. Backward-compatible for every existing consumer of the snapshot
(pure addition, no field removed/renamed).

**Why:** Experiment 11.1's condition-geometry diagnostic (`docs/superpowers/specs/
2026-08-04-buddy-publication-plan-design.md` §4) needs to reindex frozen CLIP train-set
features into the condition table's row order via `reorder_features_to_z`, which requires
knowing that order. See `docs/superpowers/plans/2026-08-25-condition-freeze-ablation.md`
Task 1.
```

- [ ] **Step 7: Commit**

```bash
git add src/hook/train_cosir.py src/test/20260825_condition_freeze_ablation/verify_sample_ids_in_snapshot.py .claude/20260825_log.md
git commit -m "feat: save sample_ids in per-epoch condition_viz snapshot (Experiment 11.1)"
```

---

### Task 2: Sweep script — `scripts/run_condition_freeze_ablation.sh`

**Files:**
- Create: `scripts/run_condition_freeze_ablation.sh`

**Interfaces:**
- Consumes: `train.em_interval` (existing config key, `configs/train/default.yaml`); `main_cosir.py`.
- Produces: 6 finished runs under one shared `experiment.results_dir`, tagged `+train.arm=trained|frozen`, wandb group `condition freeze ablation` — consumed by Task 3's and Task 5's analysis scripts.

- [ ] **Step 1: Write the script**

Create `scripts/run_condition_freeze_ablation.sh`:

```bash
#!/bin/bash
set -euo pipefail
# Experiment 11.1 (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md §4):
# does post-init training of the conditions do anything, holding buddy-init geometry, the
# frozen CLIP backbone, and every other hyperparameter identical between arms?
#
# train.em_interval is NOT a template-compatibility key (see src/hook/train_cosir.py's
# _extra dict, which only reads k/alpha/method/b_weight/encoder_pair/distance_mode) — unlike
# every other axis this project has swept (initialization_strategy, encoder_pair,
# distance_mode), toggling it does NOT change the buddy-init graph/embedding at all. This
# script deliberately points BOTH arms at the SAME results_dir so they load the exact same
# cached template (template_dir = experiment.directory.parent / "template_embeddings" in
# _init_embedding_manager) — every one of this sweep's 6 runs starts from a byte-identical
# buddy-init, and only the em_interval value governs whether the condition table can move
# after that. Do NOT give the two arms separate results_dir/template dirs — that would weaken
# the pairing this experiment is designed around.
#
# Two arms, both via train.em_interval (existing EM-alternation mechanism):
#   trained: em_interval=-1        (default; conditions update every step, as in every prior
#                                    experiment in this plan)
#   frozen:  em_interval=EPOCHS+1  (epoch // em_interval stays 0 for the whole run -> "network"
#                                    phase forever -> embedding_manager.embeddings.requires_grad_(False)
#                                    from epoch 0 onward, per the epoch-0 _prev_em_phase transition)
#
#   SMOKE=1 bash scripts/run_condition_freeze_ablation.sh   # 2 epochs, seed=1, both arms
#   bash scripts/run_condition_freeze_ablation.sh           # full 6-run sweep

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED_SWEEP="${SEED_SWEEP:-1,2,3}"

LR_SWEEP="${LR_SWEEP:-1e-3}"
LR_LABEL_SWEEP="${LR_LABEL_SWEEP:-1e-4}"
EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
ALPHA="${ALPHA:-0.5}"

DATASET="${DATASET:-redcaps_150k}"
TEST_RATIO="${TEST_RATIO:-0.2}"
# Deliberately ONE results_dir for both arms — see header comment.
RESULTS_DIR="${RESULTS_DIR:-res/CoSiR_condition_freeze_ablation/${DATASET}}"
WANDB_TAG="${WANDB_TAG:-condition-freeze-ablation-${DATASET}}"
WANDB_GROUP="${WANDB_GROUP:-condition freeze ablation}"

if [ -n "${SMOKE:-}" ]; then
  EPOCHS="${EPOCHS:-2}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
  SEED_SWEEP="${SEED_SWEEP_SMOKE:-1}"
  WANDB_TAG="${WANDB_TAG}-smoke"
  echo ">>> SMOKE: 2 epochs, seed=1, both arms — template-build + pipeline sanity"
else
  EPOCHS="${EPOCHS:-100}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
fi

FROZEN_EM_INTERVAL="$((EPOCHS + 1))"

echo "==================================================================="
echo "Condition freeze ablation ($DATASET): {trained, frozen} x seeds={$SEED_SWEEP}"
echo "  fixed: lr=$LR_SWEEP lr_label=$LR_LABEL_SWEEP dim=$EMBEDDING_DIM alpha=$ALPHA, initialization_strategy=buddies"
echo "  EPOCHS=$EPOCHS EVAL_INTERVAL=$EVAL_INTERVAL tag=$WANDB_TAG group=$WANDB_GROUP"
echo "  training-time buddy terms: OFF (not passed -> default lambda_buddy=0, lambda_buddy_con=0, buddy_refresh=False)"
echo "  results_dir (SHARED across both arms, by design): $RESULTS_DIR"
echo "==================================================================="

for ARM in trained frozen; do
  if [ "$ARM" = "trained" ]; then
    EM="-1"
  else
    EM="$FROZEN_EM_INTERVAL"
  fi
  echo ">>> arm=${ARM} (em_interval=${EM})"
  python main_cosir.py -m \
    dataset="$DATASET" \
    eval.evaluation_interval="$EVAL_INTERVAL" \
    eval.oracle_aggregation=max \
    eval.test_ratio="$TEST_RATIO" \
    model=clip_base \
    model.num_layers=6 \
    model.embedding_dim="$EMBEDDING_DIM" \
    optimizer.lr="$LR_SWEEP" \
    optimizer.lr_label="$LR_LABEL_SWEEP" \
    seed="$SEED_SWEEP" \
    train.initialization_strategy=buddies \
    train.buddies.alpha="$ALPHA" \
    train.em_interval="$EM" \
    train.epochs="$EPOCHS" \
    experiment.results_dir="$RESULTS_DIR" \
    wandb.group="$WANDB_GROUP" \
    +train.arm="$ARM" \
    ${WANDB_TAG:+++wandb.tags=[$WANDB_TAG]} \
    ${EXTRA_OVERRIDES:-}
done

echo "==================================================================="
echo "Done. Analyse retrieval (paired vs. trained, mean delta +/- std) with:"
echo "  python scripts/analyze_condition_freeze_ablation.py --tag $WANDB_TAG"
echo "Geometry diagnostic (per run, then compare):"
echo "  python scripts/analyze_condition_geometry.py --exp-dir <run_dir>"
echo "  python scripts/analyze_condition_geometry.py --compare <frozen_run_dir> <trained_run_dir>"
echo "==================================================================="
```

- [ ] **Step 2: Smoke-test it, and confirm both arms load the same template**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
SMOKE=1 bash scripts/run_condition_freeze_ablation.sh
```

Expected: two short runs (2 epochs each, `arm=trained` then `arm=frozen`), no traceback. Check the printed logs: the **first** job overall (`arm=trained`) should print `Initializing embeddings with buddies strategy...` (it builds the template); every job **after** the first — including `arm=frozen` — should print `Attempting to load from template embeddings...` with **no** `Template config mismatch` warning (confirming both arms genuinely share one buddy-init, not two independently-built ones). If a mismatch warning appears, stop and re-check that `train.arm`/`train.em_interval` haven't accidentally leaked into the `_extra` template-compatibility dict before proceeding to Task 3.

Also confirm the `frozen` run's log shows the epoch-0 EM transition: `[EM] Epoch 0: switching to NETWORK update phase` (confirms `embeddings.requires_grad_(False)` fired), and that it does **not** print a `[EM] Epoch 1: switching to CONDITIONS update phase` line (confirms it never flips back within the 2-epoch smoke run).

- [ ] **Step 3: Commit**

```bash
git add scripts/run_condition_freeze_ablation.sh
git commit -m "feat: add condition-freeze ablation sweep runner (Experiment 11.1)"
```

---

### Task 3: Retrieval paired-analysis script — `scripts/analyze_condition_freeze_ablation.py`

**Files:**
- Create: `scripts/analyze_condition_freeze_ablation.py`

**Interfaces:**
- Consumes: wandb runs from Task 2 (group `condition freeze ablation`, config key `train.arm`).
- Produces: printed paired Δ table (frozen vs. trained, mean ± std, mean/SEM) plus a frozen-arm `drift_from_init ≈ 0` sanity check — read directly by Task 7's report-writing step and by the spec's 11.1→11.2 gate decision (retrieval axis).

- [ ] **Step 1: Write the failing selftest**

Create `scripts/analyze_condition_freeze_ablation.py`:

```python
"""
Paired analysis for Experiment 11.1 (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md
S4): does post-init training of the conditions change retrieval, holding buddy-init geometry
and every other hyperparameter identical between the frozen and trained arms?

Reads wandb runs from the 'condition freeze ablation' group
(scripts/run_condition_freeze_ablation.sh), pairs the frozen arm against the trained arm
WITHIN each seed, and reports mean delta +/- std and mean/SEM (spec S5; delta = frozen -
trained, so positive means frozen wins). Also prints each run's drift_from_init (buddy_diag
section) as a sanity check -- the frozen arm's drift must be ~0 (embeddings never update); a
nonzero frozen-arm drift means the em_interval freeze did not actually take effect and the
whole ablation's premise is broken for that run.

Usage
-----
  python scripts/analyze_condition_freeze_ablation.py --tag condition-freeze-ablation-redcaps_150k
  python scripts/analyze_condition_freeze_ablation.py --selftest   # offline check, no wandb call

Requires: wandb, pandas, numpy (all already deps).
"""
import argparse

import numpy as np
import pandas as pd

T2I = "test_oracle/t2i_R1"
I2T = "test_oracle/i2t_R1"
DRIFT = "buddy_diag/drift_from_init"
BASELINE = "trained"
TREATMENT = "frozen"
CELL = [("arm", ("train", "arm")), ("seed", ("seed",))]


def _selftest():
    """Offline arithmetic check - no wandb call."""
    df = pd.DataFrame([
        {"arm": "trained", "seed": 1, T2I: 50.0},
        {"arm": "frozen", "seed": 1, T2I: 49.0},
        {"arm": "trained", "seed": 2, T2I: 48.0},
        {"arm": "frozen", "seed": 2, T2I: 48.5},
        {"arm": "trained", "seed": 3, T2I: 49.0},
        {"arm": "frozen", "seed": 3, T2I: 49.0},
    ])
    deltas = compute_paired_deltas(df, T2I)
    assert len(deltas) == 3, f"expected 3 paired cells, got {len(deltas)}"
    got = sorted(round(d, 2) for _, d in deltas)
    want = [-1.0, 0.0, 0.5]
    assert got == want, f"expected deltas {want}, got {got}"
    s = summarize(deltas)
    assert s["n"] == 3
    assert abs(s["mean"] - (-0.5 / 3)) < 1e-9
    assert s["wins"] == 1
    print("SELFTEST OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
```

- [ ] **Step 2: Run it to verify it fails**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python scripts/analyze_condition_freeze_ablation.py --selftest
```

Expected: `NameError: name 'compute_paired_deltas' is not defined`.

- [ ] **Step 3: Implement the core functions and CLI**

Add the following above `_selftest()` (after the `CELL` constant):

```python
def cget(cfg, path, default=None):
    d = cfg
    for p in path:
        if d is None:
            return default
        try:
            d = d.get(p) if hasattr(d, "get") else getattr(d, p, None)
        except Exception:
            return default
    return default if d is None else d


def sget(summ, key, default=np.nan):
    try:
        v = summ.get(key, default)
    except Exception:
        v = getattr(summ, key, default)
    return default if v is None else v


def fetch(entity, project, group, tag=None):
    import wandb
    api = wandb.Api()
    rows = []
    for run in api.runs(f"{entity}/{project}", filters={"group": group}):
        if tag and tag not in (run.tags or []):
            continue
        cfg, summ = run.config, run.summary
        arm = cget(cfg, ("train", "arm"))
        if not arm:
            continue
        row = {
            "run_id": run.id,
            "state": run.state,
            "arm": arm,
            T2I: float(sget(summ, T2I)) if not np.isnan(sget(summ, T2I)) else np.nan,
            I2T: float(sget(summ, I2T)) if not np.isnan(sget(summ, I2T)) else np.nan,
            DRIFT: float(sget(summ, DRIFT)) if not np.isnan(sget(summ, DRIFT)) else np.nan,
        }
        for cname, cpath in CELL:
            row[cname] = cget(cfg, cpath)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_paired_deltas(df, metric):
    """Pair BASELINE ('trained') vs TREATMENT ('frozen') within each seed. Returns list of
    (seed, delta) where delta = frozen - trained. Pure function - no wandb, no I/O."""
    deltas = []
    for seed, cell in df.groupby("seed"):
        by_arm = cell.groupby("arm")[metric].max()
        if BASELINE not in by_arm.index or TREATMENT not in by_arm.index:
            continue
        b, t = by_arm[BASELINE], by_arm[TREATMENT]
        if np.isnan(b) or np.isnan(t):
            continue
        deltas.append((seed, t - b))
    return deltas


def summarize(deltas):
    """mean, std, sem, z=mean/sem, win-rate - matches the project's existing convention."""
    n = len(deltas)
    if n == 0:
        return {"n": 0}
    arr = np.asarray([d for _, d in deltas], dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=1) if n > 1 else float("nan")
    sem = std / np.sqrt(n) if n > 1 else float("nan")
    z = mean / sem if (n > 1 and sem > 0) else float("nan")
    wins = int((arr > 0).sum())
    return {"n": n, "mean": mean, "std": std, "sem": sem, "z": z, "wins": wins}


def analyze(entity, project, group, tag=None):
    print(f"\n{'='*78}\nExperiment 11.1 - condition freeze ablation  group='{group}'"
          + (f"  tag='{tag}'" if tag else "") + f"\n{'='*78}")
    df = fetch(entity, project, group, tag=tag)
    if df.empty:
        print("  (no runs found - check --entity/--project/--tag)")
        return
    n_unfinished = (df["state"] != "finished").sum()
    print(f"  {len(df)} run(s); arms present: {sorted(df['arm'].unique())}."
          + (f"  [{n_unfinished} not finished -> best-so-far]" if n_unfinished else ""))

    frozen_drift = df.loc[df["arm"] == "frozen", DRIFT].dropna()
    if len(frozen_drift):
        bad = frozen_drift[frozen_drift.abs() > 1e-6]
        if len(bad):
            print(f"  !! WARNING: {len(bad)} frozen-arm run(s) show nonzero drift_from_init "
                  f"(max={bad.abs().max():.6f}) -- the em_interval freeze may not have taken "
                  f"effect; check those runs before trusting this comparison.")
        else:
            print(f"  OK: all {len(frozen_drift)} frozen-arm run(s) show drift_from_init == 0 "
                  f"(freeze confirmed to have taken effect).")

    for metric in (T2I, I2T):
        print(f"\n  --- {metric} (frozen - trained) ---")
        deltas = compute_paired_deltas(df, metric)
        s = summarize(deltas)
        if s["n"] == 0:
            print("    (no paired seeds found)")
            continue
        sig = f"  mean/SEM={s['z']:+.1f}{' *' if not np.isnan(s['z']) and abs(s['z']) >= 2 else ''}" if s["n"] > 1 else ""
        print(f"    mean delta = {s['mean']:+.2f} (n={s['n']}, wins={s['wins']}/{s['n']}){sig}")
        for seed, d in sorted(deltas):
            print(f"      seed {seed}: delta = {d:+.2f}")

    trained_drift = df.loc[df["arm"] == "trained", DRIFT].dropna()
    if len(trained_drift):
        print(f"\n  trained-arm drift_from_init: mean={trained_drift.mean():.4f}, "
              f"range=[{trained_drift.min():.4f}, {trained_drift.max():.4f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="augustoxq")
    ap.add_argument("--project", default="cosir_image")
    ap.add_argument("--group", default="condition freeze ablation")
    ap.add_argument("--tag", default=None, help="only include runs carrying this wandb tag")
    ap.add_argument("--selftest", action="store_true", help="offline arithmetic check, no wandb call")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    analyze(args.entity, args.project, args.group, tag=args.tag)


if __name__ == "__main__":
    main()
```

Delete the old `if __name__ == "__main__":` block from Step 1 — it's superseded by `main()` above.

- [ ] **Step 4: Run the selftest again to verify it passes**

```bash
python scripts/analyze_condition_freeze_ablation.py --selftest
```

Expected: `SELFTEST OK`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_condition_freeze_ablation.py
git commit -m "feat: add paired retrieval analysis for condition freeze ablation (Experiment 11.1)"
```

---

### Task 4: Geometry diagnostic — pure math core + selftest

**Files:**
- Create: `scripts/analyze_condition_geometry.py`

**Interfaces:**
- Produces: `compute_shift(comb_emb, unconditioned_emb) -> np.ndarray`, `effective_dims(x, variance_threshold=0.95) -> int`, `pairwise_sim_spread(x, n_sample=2000, seed=0) -> dict`, `rank_most_least_changed(shift, sample_ids, k=20) -> dict`, `correlate_shift(shift, other) -> dict` — all pure, no I/O. Consumed by Task 5's real-data integration in the same file.

- [ ] **Step 1: Write the failing selftest**

Create `scripts/analyze_condition_geometry.py`:

```python
"""
Post-hoc condition-embedding geometry diagnostic (Experiment 11.1, spec
docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md S4).

Retrieval numbers (test_oracle/*_R1) can miss a real difference between the frozen and
trained arms of Experiment 11.1 -- this script inspects the actual embedding geometry
instead: how much does conditioning shift the combine-side embedding, how does that shift
distribution compare across epochs/arms, and which samples are moved the most/least.

Two modes:
  --exp-dir PATH   analyze one run's condition_viz/ snapshots, write
                   condition_geometry/summary.json + a plot inside that run's directory.
  --compare A B    load two already-produced summary.json files (e.g. a frozen run and a
                   trained run, same seed) and print a paired diff.
  --selftest       offline arithmetic check of the pure math helpers, no data needed.

Usage:
  python scripts/analyze_condition_geometry.py --exp-dir <run_dir>
  python scripts/analyze_condition_geometry.py --compare <frozen_run_dir> <trained_run_dir>
  python scripts/analyze_condition_geometry.py --selftest
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr


def compute_shift(comb_emb: torch.Tensor, unconditioned_emb: torch.Tensor) -> np.ndarray:
    """Per-sample 1 - cos(conditioned, unconditioned). Both [N, D]; neither need be
    pre-normalized (this normalizes both)."""
    cond_n = F.normalize(comb_emb, dim=-1)
    uncond_n = F.normalize(unconditioned_emb, dim=-1)
    return (1.0 - (cond_n * uncond_n).sum(dim=-1)).cpu().numpy()


def effective_dims(x: np.ndarray, variance_threshold: float = 0.95) -> int:
    """Number of PCA components needed to explain >= variance_threshold of variance.
    x: [N, D]. Falls back to D if N <= D (PCA undefined)."""
    n, d = x.shape
    if n <= d:
        return d
    xc = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(xc, compute_uv=False)
    var = s ** 2
    ratio = var / var.sum()
    cumsum = np.cumsum(ratio)
    return int(np.argmax(cumsum >= variance_threshold) + 1)


def pairwise_sim_spread(x: np.ndarray, n_sample: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Mean/std of pairwise cosine similarity over a random subsample (full N^2 is wasteful
    at N~120k). x rows need not be pre-normalized."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    idx = rng.choice(n, size=min(n_sample, n), replace=False)
    xs = x[idx]
    xs = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-8)
    sims = xs @ xs.T
    iu = np.triu_indices(len(idx), k=1)
    off = sims[iu]
    return {"mean": float(off.mean()), "std": float(off.std())}


def rank_most_least_changed(shift: np.ndarray, sample_ids: List[int], k: int = 20) -> Dict[str, List[Dict]]:
    """Top-k / bottom-k samples by shift magnitude, paired with their sample id."""
    order = np.argsort(shift)  # ascending: least-changed first
    least = [{"sample_id": int(sample_ids[i]), "shift": float(shift[i])} for i in order[:k]]
    most = [{"sample_id": int(sample_ids[i]), "shift": float(shift[i])} for i in order[::-1][:k]]
    return {"most_changed": most, "least_changed": least}


def correlate_shift(shift: np.ndarray, other: np.ndarray) -> Dict[str, float]:
    """Pearson r between per-sample shift and another per-sample scalar (condition norm,
    buddy-graph degree, ...). Returns r=0/p=1 if either array has ~zero variance."""
    if shift.std() < 1e-8 or other.std() < 1e-8 or len(shift) < 2:
        return {"r": 0.0, "p": 1.0}
    r, p = pearsonr(shift, other)
    return {"r": float(r), "p": float(p)}


def _selftest():
    torch.manual_seed(0)
    # compute_shift: identical vectors -> shift 0; orthogonal -> shift 1.
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    s = compute_shift(a, b)
    assert abs(s[0] - 0.0) < 1e-6, s
    assert abs(s[1] - 1.0) < 1e-6, s

    # effective_dims: a rank-1 signal embedded in D=5 needs 1 component for 95% variance.
    rng = np.random.default_rng(0)
    direction = rng.normal(size=5)
    x = np.outer(rng.normal(size=500), direction) + rng.normal(scale=1e-4, size=(500, 5))
    assert effective_dims(x) == 1, effective_dims(x)

    # pairwise_sim_spread: identical rows -> mean sim == 1, std == 0.
    same = np.tile(rng.normal(size=(1, 8)), (100, 1))
    spread = pairwise_sim_spread(same, n_sample=50)
    assert abs(spread["mean"] - 1.0) < 1e-5, spread
    assert spread["std"] < 1e-5, spread

    # rank_most_least_changed: correct extremes and ids.
    shift = np.array([0.1, 0.9, 0.5, 0.0, 1.0])
    ids = [10, 11, 12, 13, 14]
    ranks = rank_most_least_changed(shift, ids, k=2)
    assert [r["sample_id"] for r in ranks["most_changed"]] == [14, 11], ranks
    assert [r["sample_id"] for r in ranks["least_changed"]] == [13, 10], ranks

    # correlate_shift: perfectly correlated inputs -> r ~= 1.
    x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x2 = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    c = correlate_shift(x1, x2)
    assert abs(c["r"] - 1.0) < 1e-6, c

    print("SELFTEST OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
```

- [ ] **Step 2: Run it to verify it fails**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python scripts/analyze_condition_geometry.py --selftest
```

Expected: fails only if there's a typo — this file is complete as written above, so this step should actually **pass**. Run it to confirm: expected `SELFTEST OK`. (Unlike Tasks 3/5, this task's core has no external dependency to stub out first — write it complete, then verify.)

- [ ] **Step 3: Commit**

```bash
git add scripts/analyze_condition_geometry.py
git commit -m "feat: add pure math core for condition-geometry diagnostic (Experiment 11.1)"
```

---

### Task 5: Geometry diagnostic — real-data integration + CLI

**Files:**
- Modify: `scripts/analyze_condition_geometry.py` (append; requires Task 4 and a completed smoke run from Task 2)

**Interfaces:**
- Consumes: Task 4's pure functions; Task 1's `sample_ids` field in `condition_viz/epoch_XXXX.pt`; `reorder_features_to_z` (`src/metrics/regularizer.py`); `Combiner_new` (`src/model/combiner.py`); `redcaps_buddy.load_data()` (`src/test/20260623_redcaps_buddy/redcaps_buddy.py`); `training_embeddings/buddy_edges.npy` (written by `TrainableEmbeddingManager._buddy_init`).
- Produces: `analyze_run(exp_dir, k_ranked=20) -> dict` (writes `condition_geometry/summary.json` + `plots/condition_geometry_trajectory.png` inside `exp_dir`), `compare_runs(dir_a, dir_b) -> None` (prints a paired diff) — consumed by Task 7's report-writing step.

- [ ] **Step 1: Append the real-data integration functions**

In `scripts/analyze_condition_geometry.py`, add near the top (after the existing imports):

```python
import os
import sys

_REDCAPS_BUDDY_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src", "test", "20260623_redcaps_buddy"
)
if _REDCAPS_BUDDY_DIR not in sys.path:
    sys.path.insert(0, _REDCAPS_BUDDY_DIR)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.metrics.regularizer import reorder_features_to_z
from src.model.combiner import Combiner_new
```

Then add the following above `_selftest()`:

```python
def _load_redcaps_train_features():
    """Frozen CLIP (img, txt) features + sample_ids for RedCaps-150k, in FeatureManager's
    own row order. Scope is RedCaps-150k only per spec S4 Experiment 11.1."""
    import redcaps_buddy as rb
    data = rb.load_data()
    return data.img, data.txt, data.sample_ids


def _rebuild_combiner(epoch_snapshot: dict) -> Combiner_new:
    cfg = epoch_snapshot["combiner_config"]
    combiner = Combiner_new(
        clip_feature_dim=cfg["clip_feature_dim"],
        projection_dim=cfg["projection_dim"],
        label_dim=cfg["label_dim"],
        hidden_dim=512,  # unused by Combiner_new's forward; harmless placeholder
        num_heads=8,     # unused by Combiner_new's forward; harmless placeholder
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    combiner.load_state_dict(epoch_snapshot["combiner_state_dict"])
    combiner.eval()
    return combiner


def _compute_comb_emb(combiner: Combiner_new, text_feat: torch.Tensor, conditions: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Chunked forward through the combiner. text_feat/conditions: [N, *], row-aligned."""
    n = text_feat.shape[0]
    out = None
    with torch.no_grad():
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            c = combiner(text_feat[s:e], None, conditions[s:e])
            if out is None:
                out = torch.empty((n,) + c.shape[1:], dtype=c.dtype)
            out[s:e] = c
    return out


def analyze_run(exp_dir: str, k_ranked: int = 20) -> dict:
    """Analyze one run's condition_viz/ snapshots. Writes condition_geometry/summary.json
    and a shift-trajectory plot inside exp_dir. Returns the summary dict."""
    exp_path = Path(exp_dir)
    cond_viz_dir = exp_path / "condition_viz"
    epoch_files = sorted(cond_viz_dir.glob("epoch_*.pt"))
    if not epoch_files:
        raise FileNotFoundError(f"no condition_viz/epoch_*.pt snapshots under {exp_dir}")

    img_np, txt_np, feat_sample_ids = _load_redcaps_train_features()
    img_t = torch.from_numpy(np.ascontiguousarray(img_np)).float()
    txt_t = torch.from_numpy(np.ascontiguousarray(txt_np)).float()

    edges_path = exp_path / "training_embeddings" / "buddy_edges.npy"
    buddy_edges = np.load(edges_path) if edges_path.exists() else None

    per_epoch = []
    for ef in epoch_files:
        snap = torch.load(ef, map_location="cpu")
        epoch = snap["epoch"]
        conditions = snap["label_embeddings_all"]  # [N, D]
        sample_ids = snap["sample_ids"]             # [N], added in Task 1
        n = conditions.shape[0]

        combine_side = snap.get("combine_side", "txt")
        raw_feat = img_t if combine_side == "img" else txt_t
        combine_feat = reorder_features_to_z(raw_feat, feat_sample_ids, sample_ids)

        combiner = _rebuild_combiner(snap)
        comb_emb = _compute_comb_emb(combiner, combine_feat, conditions)

        shift = compute_shift(comb_emb, combine_feat)
        cond_np = conditions.numpy()
        comb_np = comb_emb.numpy()
        raw_np = F.normalize(combine_feat, dim=-1).numpy()

        cond_norm = np.linalg.norm(cond_np, axis=1)
        norm_corr = correlate_shift(shift, cond_norm)

        degree_corr = {"r": None, "p": None}
        if buddy_edges is not None:
            degree = np.bincount(buddy_edges.flatten(), minlength=n).astype(float)
            degree_corr = correlate_shift(shift, degree)

        per_epoch.append({
            "epoch": int(epoch),
            "n_samples": int(n),
            "shift_mean": float(shift.mean()),
            "shift_std": float(shift.std()),
            "shift_p10": float(np.percentile(shift, 10)),
            "shift_p90": float(np.percentile(shift, 90)),
            "conditioned_effective_dims": effective_dims(comb_np),
            "unconditioned_effective_dims": effective_dims(raw_np),
            "condition_effective_dims": effective_dims(cond_np),
            "conditioned_pairwise_sim": pairwise_sim_spread(comb_np),
            "unconditioned_pairwise_sim": pairwise_sim_spread(raw_np),
            "shift_vs_condition_norm": norm_corr,
            "shift_vs_buddy_degree": degree_corr,
            "ranked": rank_most_least_changed(shift, sample_ids, k=k_ranked),
        })

    out_dir = exp_path / "condition_geometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"exp_dir": str(exp_path), "per_epoch": per_epoch}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _plot_trajectory(per_epoch, exp_path / "plots" / "condition_geometry_trajectory.png")
    print(f"Wrote {out_dir / 'summary.json'} ({len(per_epoch)} epochs)")
    return summary


def _plot_trajectory(per_epoch: List[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [e["epoch"] for e in per_epoch]
    shift_mean = [e["shift_mean"] for e in per_epoch]
    shift_std = [e["shift_std"] for e in per_epoch]
    eff_dims = [e["conditioned_effective_dims"] for e in per_epoch]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].errorbar(epochs, shift_mean, yerr=shift_std, marker="o")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("shift = 1 - cos(conditioned, unconditioned)")
    axes[0].set_title("Conditioning shift over training")

    axes[1].plot(epochs, eff_dims, marker="o")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("PCA effective dims (95% var)")
    axes[1].set_title("Conditioned-embedding effective dimensionality")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def compare_runs(dir_a: str, dir_b: str) -> None:
    """Print a paired epoch-by-epoch diff between two already-analyzed runs (e.g. frozen vs
    trained, same seed). Requires both to already have condition_geometry/summary.json (run
    analyze_run on each first)."""
    with open(Path(dir_a) / "condition_geometry" / "summary.json") as f:
        a = json.load(f)
    with open(Path(dir_b) / "condition_geometry" / "summary.json") as f:
        b = json.load(f)

    a_by_epoch = {e["epoch"]: e for e in a["per_epoch"]}
    b_by_epoch = {e["epoch"]: e for e in b["per_epoch"]}
    common = sorted(set(a_by_epoch) & set(b_by_epoch))
    if not common:
        print("No overlapping epochs between the two runs.")
        return

    print(f"\n{'='*78}\nCondition geometry comparison\n  A: {dir_a}\n  B: {dir_b}\n{'='*78}")
    for ep in common:
        ea, eb = a_by_epoch[ep], b_by_epoch[ep]
        d_mean = eb["shift_mean"] - ea["shift_mean"]
        d_dims = eb["conditioned_effective_dims"] - ea["conditioned_effective_dims"]
        ids_a = {r["sample_id"] for r in ea["ranked"]["most_changed"]}
        ids_b = {r["sample_id"] for r in eb["ranked"]["most_changed"]}
        overlap = len(ids_a & ids_b) / max(len(ids_a | ids_b), 1)
        print(f"  epoch {ep:>4}: shift_mean B-A={d_mean:+.4f}  eff_dims B-A={d_dims:+d}  "
              f"most-changed-set Jaccard(A,B)={overlap:.2f}")
```

- [ ] **Step 2: Replace the `--selftest`-only CLI with the full CLI**

Find the `if __name__ == "__main__":` block from Task 4:

```python
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
```

Replace with:

```python
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", default=None, help="analyze one run's condition_viz/ snapshots")
    ap.add_argument("--compare", nargs=2, default=None, metavar=("DIR_A", "DIR_B"),
                     help="print a paired diff between two already-analyzed run directories")
    ap.add_argument("--k-ranked", type=int, default=20)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if args.exp_dir:
        analyze_run(args.exp_dir, k_ranked=args.k_ranked)
        return
    if args.compare:
        compare_runs(args.compare[0], args.compare[1])
        return
    ap.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the selftest still passes**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python scripts/analyze_condition_geometry.py --selftest
```

Expected: `SELFTEST OK`.

- [ ] **Step 4: Smoke-test `--exp-dir` against Task 1/2's smoke run output**

Using the experiment directory from Task 2 Step 2's `SMOKE=1` run (note both the `arm=trained` and `arm=frozen` run directories printed by that run):

```bash
python scripts/analyze_condition_geometry.py --exp-dir <trained_smoke_run_dir>
python scripts/analyze_condition_geometry.py --exp-dir <frozen_smoke_run_dir>
```

Expected: both print `Wrote .../condition_geometry/summary.json (2 epochs)` with no traceback; `condition_geometry/summary.json` and `plots/condition_geometry_trajectory.png` exist in both directories.

- [ ] **Step 5: Smoke-test `--compare`**

```bash
python scripts/analyze_condition_geometry.py --compare <frozen_smoke_run_dir> <trained_smoke_run_dir>
```

Expected: prints the comparison header and one line per shared epoch (`epoch 0`, `epoch 1`) with `shift_mean B-A`, `eff_dims B-A`, and a Jaccard overlap in `[0, 1]`.

- [ ] **Step 6: Commit**

```bash
git add scripts/analyze_condition_geometry.py
git commit -m "feat: add real-data integration + CLI for condition-geometry diagnostic (Experiment 11.1)"
```

---

### Task 6: Launch the full 6-run sweep

**Files:** none (execution only).

**Interfaces:**
- Consumes: Task 2's sweep script.
- Produces: 6 finished wandb runs (2 arms × 3 seeds) under one shared `results_dir`, feeding Task 7's analysis.

- [ ] **Step 1: Get explicit user confirmation before launching**

This is a real, multi-hour GPU training commitment (6 runs × ~100 epochs on RedCaps-150k). Per this project's established practice, **stop here and get explicit user confirmation before running Step 2**, even though this plan itself was already approved — do not treat plan approval as launch approval.

- [ ] **Step 2: Confirm Tasks 1–5's smoke tests all passed**

Verify Task 1 Step 5 (`ALL 2 SNAPSHOT(S) VERIFIED`), Task 2 Step 2 (template-reuse + EM-transition checks), and Task 5 Steps 4–5 (`--exp-dir` and `--compare` smoke checks) all completed without error before spending full compute.

- [ ] **Step 3: Launch the full sweep**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
bash scripts/run_condition_freeze_ablation.sh
```

This runs 2 arms × 3 seeds = 6 runs at 100 epochs each. Long-running — launch with `run_in_background` or `nohup ... &` if executing interactively.

- [ ] **Step 4: Verify all 6 runs finished**

Check the wandb UI (project `cosir_image`, group `condition freeze ablation`, tag `condition-freeze-ablation-redcaps_150k`) or query via `wandb.Api()` that all 6 runs show `state == "finished"` with `test_oracle/t2i_R1`, `test_oracle/i2t_R1`, and `buddy_diag/drift_from_init` present.

---

### Task 7: Run all analyses, write the report, update the spec

**Files:**
- Create: `docs/reports/2026-08-25_condition_freeze_ablation.md` (adjust date to when Task 6 actually completes)
- Modify: `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` (add the Experiment 11.1 outcome to §2's claims table; resolve 11.2's gate per the widened either-axis rule)

**Interfaces:**
- Consumes: `scripts/analyze_condition_freeze_ablation.py` (Task 3) and `scripts/analyze_condition_geometry.py` (Tasks 4–5) output.
- Produces: the go/no-go input for Experiment 11.2, and a §2 claims-table row.

- [ ] **Step 1: Run the retrieval analysis**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python scripts/analyze_condition_freeze_ablation.py --tag condition-freeze-ablation-redcaps_150k
```

Capture the full printed output (frozen-arm drift sanity check, per-metric paired table, mean Δ ± std, mean/SEM for both `t2i_R1`/`i2t_R1`, trained-arm drift range).

- [ ] **Step 2: Run the geometry diagnostic on all 6 runs, then compare paired by seed**

For each of the 6 run directories from Task 6:

```bash
python scripts/analyze_condition_geometry.py --exp-dir <run_dir>
```

Then, for each seed, compare its frozen and trained run directories:

```bash
python scripts/analyze_condition_geometry.py --compare <frozen_seed1_dir> <trained_seed1_dir>
python scripts/analyze_condition_geometry.py --compare <frozen_seed2_dir> <trained_seed2_dir>
python scripts/analyze_condition_geometry.py --compare <frozen_seed3_dir> <trained_seed3_dir>
```

Capture all three comparisons' full printed output (per-epoch `shift_mean B-A`, `eff_dims B-A`, most-changed-set Jaccard overlap).

- [ ] **Step 3: Apply the spec's decision rule**

Per spec §4 Experiment 11.1's success criteria: **no real difference on either axis** (retrieval within noise floor AND no meaningful geometry divergence — small `shift_mean` deltas, stable `eff_dims`, high most-changed-set Jaccard overlap across the 3 seed-paired comparisons) → the current training-time pressure on conditions is not earning its complexity; report this as a clean simplification result, no 11.2 needed. **A real difference on either axis** (retrieval mean/SEM ≥ 2 exceeding the noise floor in either direction, OR a clear geometry divergence — large/growing `shift_mean` deltas, diverging `eff_dims`, or low most-changed-set Jaccard overlap — even with null retrieval) → 11.2 is gated open; note in the report which axis (or both) triggered it.

- [ ] **Step 4: Write the results report**

Create `docs/reports/2026-08-25_condition_freeze_ablation.md` following the structure of `docs/reports/2026-08-24_buddy_init_encoder_ablation.md` (method, retrieval results table, geometry-diagnostic results — shift distributions, effective-dims trajectory, most/least-changed samples, correlations against condition norm and buddy degree — interpretation, caveats, reproduction commands).

- [ ] **Step 5: Update the spec**

In `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`: add the outcome as a new row in §2's claims table (next available letter), citing the new report; in §4 Experiment 11.2, replace "Gate: Runs if 11.1 clears the noise floor on retrieval in either direction, or the geometry diagnostic shows a real divergence between arms even with null retrieval" with the actual resolved status (either "gate cleared — proceeding" with which axis triggered it, or "gate not cleared — 11.2 dropped from scope, see report for the clean-null result").

- [ ] **Step 6: Commit**

```bash
git add docs/reports/2026-08-25_condition_freeze_ablation.md docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md
git commit -m "results: condition freeze ablation (Experiment 11.1)"
```

---

## Self-Review

- **Spec coverage:** Task 1 provides the one missing piece of infrastructure (sample-id-aware per-epoch snapshots) the geometry diagnostic needs; Task 2 implements 11.1's core paired-arm design exactly as specced (shared results_dir/template, `em_interval` as the sole axis); Tasks 3 covers the retrieval axis of the success criteria; Tasks 4–5 cover the geometry axis (distribution comparison, per-sample shift, most/least-changed ranking, correlation against condition norm and buddy degree — all four diagnostics named in the spec's Tooling paragraph); Task 6 executes; Task 7 applies the widened either-axis decision rule and resolves 11.2's gate. 11.2 itself is explicitly out of scope for this plan, per spec.
- **Placeholder scan:** every code block is complete and runnable as written; no TBD/TODO. Task 4 Step 2's note that the selftest is expected to *pass* (not fail) is a deliberate, explained deviation from strict red-green TDD — the file has no external dependency to stub out first, unlike Tasks 3/5 — not a placeholder.
- **Type/interface consistency:** `_save_condition_viz_snapshot(..., label_ids_all, ...)` (Task 1) matches its call site's new `label_ids_all,` argument and the saved `"sample_ids": list(label_ids_all)` key, which Task 5's `analyze_run` reads back as `snap["sample_ids"]` and passes to `reorder_features_to_z(raw_feat, feat_sample_ids, sample_ids)` and `rank_most_least_changed(shift, sample_ids, k=...)` — same list-of-int type throughout. `compute_paired_deltas(df, metric) -> list[(seed, float)]` and `summarize(deltas) -> dict` (Task 3) mirror the signatures already established by `scripts/analyze_buddy_init_encoder_ablation.py`/`scripts/analyze_buddy_distance_mode_ablation.py`. `analyze_run(exp_dir, k_ranked=20) -> dict` and `compare_runs(dir_a, dir_b) -> None` (Task 5) are defined once and used identically in their smoke-test steps and in Task 7.
- **Scope check:** RedCaps-150k, 2 arms, 3 seeds (6 runs) only — 11.2 is explicitly out of scope for this plan (gated on this plan's own result, per spec). Experiments 0–10 are untouched.
- **Sample-ID consistency (CLAUDE.md's flagged failure mode):** addressed directly in three places — Task 1's verification script cross-checks the new `sample_ids` field against the run's own `training_embeddings/sample_ids.npy` ground truth; Task 5's `analyze_run` uses `reorder_features_to_z` (not a hand-rolled join) to align FeatureManager's row order against the condition table's row order; and `buddy_edges.npy`'s endpoints are already documented (`compute_buddy_init`'s docstring) as table positions in the same `output_sample_ids` order the condition table uses, so `np.bincount` needs no separate reindexing.
- **Template-sharing correctness (this plan's one genuine departure from every prior experiment's pattern):** called out in Global Constraints, the sweep script's header comment, and Task 2 Step 2's explicit smoke-test check (confirming the second arm loads rather than rebuilds the template) — the one place this plan could most easily be "corrected" into breaking its own pairing design if an implementer pattern-matched on Experiments 1/8/10 without reading the reasoning.
