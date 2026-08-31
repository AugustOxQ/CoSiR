# Bidirectional Table↔Predictor Coupling (Experiment 11.3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer Experiment 11.3 from `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` (Experiment 11.3 subsection) — does removing the stop-gradient on the condition-predictor distillation term (so the per-sample condition table is also pulled toward what `condition_predictor` can represent, not just the reverse) change `test_oracle` (the table's own held-out codebook quality) and/or `test_pre_diff` (the predictor's standalone usefulness vs. raw CLIP), relative to Experiment 11.1's existing frozen and trained arms?

**Architecture:** No new architecture — `condition_predictor` (`src/model/condition_predictor.py`) and its distillation loss (`lambda_pred`, `train_cosir.py`) already exist. The only change is a new boolean config flag, `loss.pred_stopgrad` (default `True` = today's one-way distillation), threaded through a small pure function extracted from the training loop into `src/metrics/loss.py` (Task 1) — mirroring how `imix_loss` already lives there as a pure, testable helper called from the training loop, rather than leaving the branch inline and untestable. A new sweep arm (`pred_coupled`: `train.em_interval=-1`, `loss.pred_stopgrad=false`) reuses Experiment 11.1's exact `results_dir` and wandb group so it shares the identical buddy-init template and can be compared directly against 11.1's already-completed `trained`/`frozen` runs without re-running them (Task 2). Both primary metrics (`test_oracle`, `test_pre_diff`) are already computed by the existing eval pipeline (`src/eval/pipeline.py`) with zero new eval code — the analysis script (Task 3) only needs to fetch and pair them from wandb, following the same pattern as `scripts/analyze_condition_freeze_ablation.py`.

**Tech Stack:** Python 3.10, Hydra/OmegaConf, PyTorch, wandb, numpy/pandas. Existing CoSiR training entrypoint `main_cosir.py`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`, Experiment 11.3 subsection (added 2026-08-26).

## Global Constraints

- Always run Python/bash training or analysis commands with `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR` first (per `.claude/CLAUDE.md`).
- Statistical standard (spec §5): ≥3 seeds, paired-within-seed Δ, report mean ± std and mean/SEM (flag `|z| ≥ 2` with `*`), compare against the noise floor (~0.1–0.7 R1), never against zero.
- Fixed operating point (matches Exp. 1/8/10/11.1's confirmed strong cell): `lr=1e-3`, `lr_label=1e-4`, `embedding_dim=16`, `alpha=0.5`. `initialization_strategy=buddies` fixed throughout. `train.em_interval=-1` (trained regime) — this experiment does not touch the freeze axis.
- Every training-time buddy term stays OFF: never pass `+loss.lambda_buddy`, `+loss.lambda_buddy_con`, or `+loss.buddy_refresh*`. Omitting them gives the code's own default of `0.0`/`False`.
- **`loss.pred_stopgrad` is the ONLY axis distinguishing this experiment's new arm from Experiment 11.1's existing `trained` arm, and — like `train.em_interval` before it — it is deliberately NOT a template-compatibility key** (`src/hook/train_cosir.py`'s `_extra` dict, built in `_init_embedding_manager`, only reads `k`/`alpha`/`method`/`b_weight`/`encoder_pair`/`distance_mode`). This new arm therefore points at the **same `results_dir`** as `scripts/run_condition_freeze_ablation.sh` (Experiment 11.1) so it loads the exact same cached buddy-init template, and the **same wandb group** (`condition freeze ablation`) so its 3 new runs can be pulled and paired against 11.1's already-completed `trained`/`frozen` runs directly. Do not give this arm its own `results_dir` or wandb group — that would break the paired comparison this experiment depends on.
- Scope: RedCaps-150k only (`dataset=redcaps_150k`), 1 new arm × 3 seeds = 3 runs. This plan only adds the `pred_coupled` arm — it does not re-run 11.1's `trained`/`frozen` arms.
- Two existing `src/` files are modified in this plan (`src/metrics/loss.py`, `src/metrics/__init__.py`, `src/hook/train_cosir.py`) and one config file (`configs/train/default.yaml`). Per CLAUDE.md, log the change in `.claude/20260826_log.md` (one `# <path>` section per file).
- wandb defaults: `entity=augustoxq`, `project=cosir_image` (`configs/config.yaml`).
- Task 4 (launching the real 3-run sweep) is a multi-hour GPU commitment — **get explicit user confirmation before running it**, even though this is a plan already approved end-to-end. Do not auto-launch it as part of unattended execution.

---

### Task 1: `predictor_consistency_loss` — pure function, config flag, wiring

**Files:**
- Modify: `src/metrics/loss.py` (insert new function after `imix_loss`, before `class LabelContrastiveLoss_enhance`, i.e. after line 54)
- Modify: `src/metrics/__init__.py`
- Modify: `src/hook/train_cosir.py:48` (import), `:1606-1621` (loss computation)
- Modify: `configs/train/default.yaml` (add `loss.pred_stopgrad`)
- Create: `src/test/test_loss_predictor_consistency.py`
- Create: `.claude/20260826_log.md`

**Interfaces:**
- Produces: `predictor_consistency_loss(pred_cond: Tensor, label_embeddings: Tensor, stopgrad: bool = True) -> Tensor` (`src/metrics/loss.py`), exported from `src.metrics`. Consumed by `train_cosir.py`'s training loop (this task) and by Task 1's own test.

- [ ] **Step 1: Write the failing test**

Create `src/test/test_loss_predictor_consistency.py`:

```python
"""Tests for src.metrics.loss.predictor_consistency_loss (Experiment 11.3,
docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md).

Verifies the actual autograd claim the spec's "What" section makes: with
stopgrad=True (today's default), only the predictor receives gradient from this
term (one-way distillation). With stopgrad=False, gradient also flows into the
condition table -- the mechanism Experiment 11.3 tests.

Run: python src/test/test_loss_predictor_consistency.py
"""
import torch

from src.metrics.loss import predictor_consistency_loss


def test_stopgrad_true_blocks_gradient_into_table():
    torch.manual_seed(0)
    label_embeddings = torch.randn(4, 8, requires_grad=True)
    pred_cond = torch.randn(4, 8, requires_grad=True)

    loss = predictor_consistency_loss(pred_cond, label_embeddings, stopgrad=True)
    loss.backward()

    assert pred_cond.grad is not None, "predictor must receive gradient regardless of stopgrad"
    assert not torch.allclose(pred_cond.grad, torch.zeros_like(pred_cond.grad)), "predictor gradient must be nonzero"
    assert label_embeddings.grad is None, "stopgrad=True must block gradient into the condition table"
    print("PASS: stopgrad=True blocks gradient into label_embeddings")


def test_stopgrad_false_allows_gradient_into_table():
    torch.manual_seed(0)
    label_embeddings = torch.randn(4, 8, requires_grad=True)
    pred_cond = torch.randn(4, 8, requires_grad=True)

    loss = predictor_consistency_loss(pred_cond, label_embeddings, stopgrad=False)
    loss.backward()

    assert label_embeddings.grad is not None, "stopgrad=False must let gradient flow into the condition table"
    assert not torch.allclose(label_embeddings.grad, torch.zeros_like(label_embeddings.grad)), "table gradient must be nonzero"
    print("PASS: stopgrad=False lets gradient flow into label_embeddings")


def test_stopgrad_default_is_true():
    torch.manual_seed(0)
    label_embeddings = torch.randn(4, 8, requires_grad=True)
    pred_cond = torch.randn(4, 8, requires_grad=True)

    loss_default = predictor_consistency_loss(pred_cond, label_embeddings)
    loss_explicit = predictor_consistency_loss(pred_cond, label_embeddings, stopgrad=True)
    assert torch.allclose(loss_default, loss_explicit), "default stopgrad must match today's one-way-distillation behavior"
    print("PASS: default stopgrad=True matches explicit stopgrad=True (backward-compatible default)")


def main():
    test_stopgrad_true_blocks_gradient_into_table()
    test_stopgrad_false_allows_gradient_into_table()
    test_stopgrad_default_is_true()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it to verify it fails**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python src/test/test_loss_predictor_consistency.py
```

Expected: `ImportError: cannot import name 'predictor_consistency_loss' from 'src.metrics.loss'`.

- [ ] **Step 3: Implement the function**

In `src/metrics/loss.py`, find (line 54, the end of `imix_loss`):

```python
    return lambda_imix * loss.mean()


class LabelContrastiveLoss_enhance(nn.Module):
```

Replace with:

```python
    return lambda_imix * loss.mean()


def predictor_consistency_loss(
    pred_cond: Tensor,
    label_embeddings: Tensor,
    stopgrad: bool = True,
) -> Tensor:
    """Cosine-distance consistency loss between the condition predictor's output and the
    per-sample condition table.

    stopgrad=True (default): only the predictor receives gradient from this term -- today's
    one-way distillation (predictor learns to reproduce the table).
    stopgrad=False: gradient also flows into label_embeddings, pulling the table toward what
    the predictor -- a bounded-capacity function of the frozen input feature -- can represent
    (Experiment 11.3, docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md).
    """
    target = label_embeddings.detach() if stopgrad else label_embeddings
    return (1 - F.cosine_similarity(pred_cond, target, dim=-1)).mean()


class LabelContrastiveLoss_enhance(nn.Module):
```

- [ ] **Step 4: Export it from `src.metrics`**

In `src/metrics/__init__.py`, find:

```python
from .loss import LabelContrastiveLoss_enhance
```

Replace with:

```python
from .loss import LabelContrastiveLoss_enhance, predictor_consistency_loss
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
python src/test/test_loss_predictor_consistency.py
```

Expected:
```
PASS: stopgrad=True blocks gradient into label_embeddings
PASS: stopgrad=False lets gradient flow into label_embeddings
PASS: default stopgrad=True matches explicit stopgrad=True (backward-compatible default)
```

- [ ] **Step 6: Wire it into the training loop**

In `src/hook/train_cosir.py`, find (line 48):

```python
from src.metrics import LabelContrastiveLoss_enhance
```

Replace with:

```python
from src.metrics import LabelContrastiveLoss_enhance, predictor_consistency_loss
```

Then find (around lines 1606-1621):

```python
            # Condition predictor distillation + L5 entropy diversity.
            # pred_cond is shared between both losses to avoid a second forward pass.
            lambda_pred = cfg.loss.lambda_pred
            lambda_ent = getattr(cfg.loss, "lambda_ent", 0.0)
            ent_tau = getattr(cfg.loss, "ent_tau", 5.0)

            pred_cond = None
            if lambda_pred > 0 or (lambda_ent > 0 and len(sample_types) > 0):
                pred_cond = model.predict_condition(combine_emb)

            if lambda_pred > 0 and pred_cond is not None:
                pred_loss = (
                    1
                    - F.cosine_similarity(pred_cond, label_embeddings.detach(), dim=-1)
                ).mean()
                loss = loss + lambda_pred * pred_loss
                loss_dict["loss_pred"] = pred_loss
```

Replace with:

```python
            # Condition predictor distillation + L5 entropy diversity.
            # pred_cond is shared between both losses to avoid a second forward pass.
            lambda_pred = cfg.loss.lambda_pred
            lambda_ent = getattr(cfg.loss, "lambda_ent", 0.0)
            ent_tau = getattr(cfg.loss, "ent_tau", 5.0)
            pred_stopgrad = getattr(cfg.loss, "pred_stopgrad", True)

            pred_cond = None
            if lambda_pred > 0 or (lambda_ent > 0 and len(sample_types) > 0):
                pred_cond = model.predict_condition(combine_emb)

            if lambda_pred > 0 and pred_cond is not None:
                pred_loss = predictor_consistency_loss(
                    pred_cond, label_embeddings, stopgrad=pred_stopgrad
                )
                loss = loss + lambda_pred * pred_loss
                loss_dict["loss_pred"] = pred_loss
```

- [ ] **Step 7: Add the config flag**

In `configs/train/default.yaml`, find:

```yaml
  lambda_pred: 1.0 # condition predictor distillation weight (0 = disabled)
```

Replace with:

```yaml
  lambda_pred: 1.0 # condition predictor distillation weight (0 = disabled)
  pred_stopgrad: true # if false, gradient also flows predictor->table too (Experiment 11.3 bidirectional coupling); default true = today's one-way distillation
```

- [ ] **Step 8: Log the change**

Create `.claude/20260826_log.md`:

```markdown
# /src/metrics/loss.py

## Added `predictor_consistency_loss`

**Before:** The condition-predictor distillation term was computed inline in
`train_cosir.py`'s training loop, always with `label_embeddings.detach()` — gradient could
only flow table→predictor, never predictor→table.

**After:** Extracted into a pure function `predictor_consistency_loss(pred_cond,
label_embeddings, stopgrad=True)`, mirroring how `imix_loss` already lives in this file as a
pure helper called from the training loop. `stopgrad=True` (default) preserves today's
behavior exactly; `stopgrad=False` lets gradient also flow into `label_embeddings`.

**Why:** Experiment 11.3 (`docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`)
needs this as a config-toggleable, unit-testable branch — see
`docs/superpowers/plans/2026-08-26-pred-stopgrad-ablation.md` Task 1.

# /src/metrics/__init__.py

## Exported `predictor_consistency_loss` alongside `LabelContrastiveLoss_enhance`.

# /src/hook/train_cosir.py

## Training loop now uses `predictor_consistency_loss` with a new `pred_stopgrad` flag

**Before:** `pred_loss = (1 - F.cosine_similarity(pred_cond, label_embeddings.detach(),
dim=-1)).mean()` — hardcoded one-way distillation.

**After:** `pred_stopgrad = getattr(cfg.loss, "pred_stopgrad", True)`; `pred_loss =
predictor_consistency_loss(pred_cond, label_embeddings, stopgrad=pred_stopgrad)`.
Backward-compatible: default `pred_stopgrad=True` reproduces the old behavior exactly.

**Why:** See `docs/superpowers/plans/2026-08-26-pred-stopgrad-ablation.md` Task 1.

# /configs/train/default.yaml

## Added `loss.pred_stopgrad: true`

**Why:** New config toggle for Experiment 11.3 — see
`docs/superpowers/plans/2026-08-26-pred-stopgrad-ablation.md` Task 1.
```

- [ ] **Step 9: Commit**

```bash
git add src/metrics/loss.py src/metrics/__init__.py src/hook/train_cosir.py configs/train/default.yaml src/test/test_loss_predictor_consistency.py .claude/20260826_log.md
git commit -m "feat: add bidirectional table<->predictor coupling toggle (Experiment 11.3)"
```

---

### Task 2: Sweep script — `scripts/run_pred_stopgrad_ablation.sh`

**Files:**
- Create: `scripts/run_pred_stopgrad_ablation.sh`

**Interfaces:**
- Consumes: `loss.pred_stopgrad` (Task 1's new config key); `main_cosir.py`.
- Produces: 3 finished runs (`+train.arm=pred_coupled`) sharing 11.1's `results_dir` and wandb group `condition freeze ablation` — consumed by Task 3's analysis script.

- [ ] **Step 1: Write the script**

Create `scripts/run_pred_stopgrad_ablation.sh`:

```bash
#!/bin/bash
set -euo pipefail
# Experiment 11.3 (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md,
# Experiment 11.3 subsection): does removing the stop-gradient on the condition-predictor
# distillation term (loss.pred_stopgrad=false) change test_oracle (the table's own held-out
# codebook quality) or test_pre_diff (the predictor's standalone usefulness vs. raw CLIP),
# relative to Experiment 11.1's existing trained/frozen arms?
#
# loss.pred_stopgrad is NOT a template-compatibility key (src/hook/train_cosir.py's _extra
# dict only reads k/alpha/method/b_weight/encoder_pair/distance_mode -- the same class of
# non-template-affecting knob as train.em_interval, per 11.1's own script). This script
# therefore points at the SAME results_dir as scripts/run_condition_freeze_ablation.sh so this
# arm's 3 runs load the exact same cached buddy-init template as 11.1's existing trained/frozen
# runs, and reuses the SAME wandb group so scripts/analyze_pred_stopgrad_ablation.py can pull
# all three arms from one query. Do NOT give this arm its own results_dir or wandb group --
# that would break the paired comparison against 11.1's already-completed runs.
#
# One new arm, added on top of 11.1's existing {trained, frozen}:
#   pred_coupled: em_interval=-1 (trained regime, same as 11.1's "trained" arm)
#                 + loss.pred_stopgrad=false (Experiment 11.3's bidirectional coupling)
#
#   SMOKE=1 bash scripts/run_pred_stopgrad_ablation.sh   # 2 epochs, seed=1
#   bash scripts/run_pred_stopgrad_ablation.sh           # full 3-run sweep

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED_SWEEP="${SEED_SWEEP:-1,2,3}"

LR_SWEEP="${LR_SWEEP:-1e-3}"
LR_LABEL_SWEEP="${LR_LABEL_SWEEP:-1e-4}"
EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
ALPHA="${ALPHA:-0.5}"

DATASET="${DATASET:-redcaps_150k}"
TEST_RATIO="${TEST_RATIO:-0.2}"
# Deliberately the SAME results_dir as scripts/run_condition_freeze_ablation.sh -- see header.
RESULTS_DIR="${RESULTS_DIR:-res/CoSiR_condition_freeze_ablation/${DATASET}}"
WANDB_TAG="${WANDB_TAG:-pred-stopgrad-ablation-${DATASET}}"
WANDB_GROUP="${WANDB_GROUP:-condition freeze ablation}"

if [ -n "${SMOKE:-}" ]; then
  EPOCHS="${EPOCHS:-2}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
  SEED_SWEEP="${SEED_SWEEP_SMOKE:-1}"
  WANDB_TAG="${WANDB_TAG}-smoke"
  echo ">>> SMOKE: 2 epochs, seed=1, pred_coupled arm only -- pipeline sanity"
else
  EPOCHS="${EPOCHS:-100}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
fi

echo "==================================================================="
echo "Pred-stopgrad ablation ($DATASET): pred_coupled x seeds={$SEED_SWEEP}"
echo "  fixed: lr=$LR_SWEEP lr_label=$LR_LABEL_SWEEP dim=$EMBEDDING_DIM alpha=$ALPHA, initialization_strategy=buddies"
echo "  EPOCHS=$EPOCHS EVAL_INTERVAL=$EVAL_INTERVAL tag=$WANDB_TAG group=$WANDB_GROUP"
echo "  training-time buddy terms: OFF (not passed -> default lambda_buddy=0, lambda_buddy_con=0, buddy_refresh=False)"
echo "  results_dir (SHARED with Experiment 11.1, by design): $RESULTS_DIR"
echo "==================================================================="

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
  train.em_interval=-1 \
  train.epochs="$EPOCHS" \
  loss.pred_stopgrad=false \
  experiment.results_dir="$RESULTS_DIR" \
  wandb.group="$WANDB_GROUP" \
  +train.arm="pred_coupled" \
  ${WANDB_TAG:+++wandb.tags=[$WANDB_TAG]} \
  ${EXTRA_OVERRIDES:-}

echo "==================================================================="
echo "Done. Analyse (pred_coupled vs 11.1's trained/frozen, mean delta +/- std) with:"
echo "  python scripts/analyze_pred_stopgrad_ablation.py --tag $WANDB_TAG"
echo "==================================================================="
```

- [ ] **Step 2: Smoke-test it, and confirm it loads 11.1's existing template**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
SMOKE=1 bash scripts/run_pred_stopgrad_ablation.sh
```

Expected: one short run (2 epochs), no traceback. Because Experiment 11.1 already built and cached the buddy-init template at this `results_dir`, the printed log should show `Attempting to load from template embeddings...` with **no** `Template config mismatch` warning — it should **not** print `Initializing embeddings with buddies strategy...` (that would mean it built a *new* template, which would break the paired comparison against 11.1's runs). If a mismatch or a fresh-template-build message appears, stop and re-check that `loss.pred_stopgrad`/`train.arm` haven't leaked into the `_extra` template-compatibility dict, and that `RESULTS_DIR` truly matches 11.1's, before proceeding to Task 3.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_pred_stopgrad_ablation.sh
git commit -m "feat: add pred-stopgrad ablation sweep runner (Experiment 11.3)"
```

---

### Task 3: Analysis script — `scripts/analyze_pred_stopgrad_ablation.py`

**Files:**
- Create: `scripts/analyze_pred_stopgrad_ablation.py`

**Interfaces:**
- Consumes: wandb runs from Task 2 (group `condition freeze ablation`, config key `train.arm` ∈ `{trained, frozen, pred_coupled}`).
- Produces: printed paired Δ tables (`pred_coupled` vs. `trained`, and `pred_coupled` vs. `frozen`) for `test_oracle` and `test_pre_diff` (both i2t/t2i), plus `drift_from_init` and final-step `loss/loss_pred` as free diagnostic context — read directly by Task 4's report-writing step.

- [ ] **Step 1: Write the failing selftest**

Create `scripts/analyze_pred_stopgrad_ablation.py`:

```python
"""
Paired analysis for Experiment 11.3 (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md,
Experiment 11.3 subsection): does removing the stop-gradient on the condition-predictor
distillation term change test_oracle or test_pre_diff, relative to Experiment 11.1's existing
trained/frozen arms?

Reads wandb runs from the 'condition freeze ablation' group (shared with
scripts/run_condition_freeze_ablation.sh), pairs the new 'pred_coupled' arm against BOTH
11.1's 'trained' and 'frozen' arms WITHIN each seed, and reports mean delta +/- std and
mean/SEM (spec S5; delta = pred_coupled - baseline, so positive means pred_coupled wins) for
test_oracle/{t2i,i2t}_R1 and test_pre_diff/{t2i,i2t}_R1. Also prints each arm's
buddy_diag/drift_from_init and final-step loss/loss_pred as free diagnostic context (does
coupling shrink drift toward frozen's ~0, and does the predictor's own reconstruction loss
converge or diverge).

Usage
-----
  python scripts/analyze_pred_stopgrad_ablation.py --tag pred-stopgrad-ablation-redcaps_150k
  python scripts/analyze_pred_stopgrad_ablation.py --selftest   # offline check, no wandb call

Requires: wandb, pandas, numpy (all already deps).
"""
import argparse

import numpy as np
import pandas as pd

T2I_ORACLE = "test_oracle/t2i_R1"
I2T_ORACLE = "test_oracle/i2t_R1"
T2I_PREDIFF = "test_pre_diff/t2i_R1"
I2T_PREDIFF = "test_pre_diff/i2t_R1"
DRIFT = "buddy_diag/drift_from_init"
PRED_LOSS = "loss/loss_pred"
METRICS = [T2I_ORACLE, I2T_ORACLE, T2I_PREDIFF, I2T_PREDIFF]
TREATMENT = "pred_coupled"
BASELINES = ["trained", "frozen"]


def _selftest():
    """Offline arithmetic check - no wandb call."""
    df = pd.DataFrame([
        {"arm": "trained", "seed": 1, T2I_ORACLE: 50.0},
        {"arm": "frozen", "seed": 1, T2I_ORACLE: 54.0},
        {"arm": "pred_coupled", "seed": 1, T2I_ORACLE: 52.0},
        {"arm": "trained", "seed": 2, T2I_ORACLE: 48.0},
        {"arm": "frozen", "seed": 2, T2I_ORACLE: 52.5},
        {"arm": "pred_coupled", "seed": 2, T2I_ORACLE: 51.0},
    ])
    deltas_vs_trained = compute_paired_deltas(df, T2I_ORACLE, "trained")
    assert len(deltas_vs_trained) == 2, f"expected 2 paired cells, got {len(deltas_vs_trained)}"
    got = sorted(round(d, 2) for _, d in deltas_vs_trained)
    assert got == [2.0, 3.0], f"expected deltas [2.0, 3.0] vs trained, got {got}"

    deltas_vs_frozen = compute_paired_deltas(df, T2I_ORACLE, "frozen")
    assert len(deltas_vs_frozen) == 2, f"expected 2 paired cells, got {len(deltas_vs_frozen)}"
    got2 = sorted(round(d, 2) for _, d in deltas_vs_frozen)
    assert got2 == [-2.0, -1.5], f"expected deltas [-2.0, -1.5] vs frozen, got {got2}"

    s = summarize(deltas_vs_trained)
    assert s["n"] == 2
    assert abs(s["mean"] - 2.5) < 1e-9
    assert s["wins"] == 2
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
python scripts/analyze_pred_stopgrad_ablation.py --selftest
```

Expected: `NameError: name 'compute_paired_deltas' is not defined`.

- [ ] **Step 3: Implement the core functions and CLI**

Add the following above `_selftest()` (after the `BASELINES` constant):

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
        cfg, summ = run.config, run.summary
        arm = cget(cfg, ("train", "arm"))
        if not arm:
            continue
        if arm == TREATMENT and tag and tag not in (run.tags or []):
            continue
        row = {"run_id": run.id, "state": run.state, "arm": arm, "seed": cget(cfg, ("seed",))}
        for metric in METRICS + [DRIFT, PRED_LOSS]:
            v = sget(summ, metric)
            row[metric] = float(v) if not np.isnan(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compute_paired_deltas(df, metric, baseline):
    """Pair TREATMENT ('pred_coupled') vs the given baseline arm within each seed. Returns
    list of (seed, delta) where delta = pred_coupled - baseline. Pure function - no wandb,
    no I/O."""
    deltas = []
    for seed, cell in df.groupby("seed"):
        by_arm = cell.groupby("arm")[metric].max()
        if baseline not in by_arm.index or TREATMENT not in by_arm.index:
            continue
        b, t = by_arm[baseline], by_arm[TREATMENT]
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
    print(f"\n{'='*78}\nExperiment 11.3 - bidirectional table<->predictor coupling  group='{group}'"
          + (f"  tag='{tag}'" if tag else "") + f"\n{'='*78}")
    df = fetch(entity, project, group, tag=tag)
    if df.empty:
        print("  (no runs found - check --entity/--project/--tag)")
        return
    n_unfinished = (df["state"] != "finished").sum()
    print(f"  {len(df)} run(s); arms present: {sorted(df['arm'].unique())}."
          + (f"  [{n_unfinished} not finished -> best-so-far]" if n_unfinished else ""))

    for baseline in BASELINES:
        print(f"\n  {'-'*70}\n  pred_coupled vs {baseline}\n  {'-'*70}")
        for metric in METRICS:
            print(f"\n    --- {metric} (pred_coupled - {baseline}) ---")
            deltas = compute_paired_deltas(df, metric, baseline)
            s = summarize(deltas)
            if s["n"] == 0:
                print("      (no paired seeds found)")
                continue
            sig = f"  mean/SEM={s['z']:+.1f}{' *' if not np.isnan(s['z']) and abs(s['z']) >= 2 else ''}" if s["n"] > 1 else ""
            print(f"      mean delta = {s['mean']:+.2f} (n={s['n']}, wins={s['wins']}/{s['n']}){sig}")
            for seed, d in sorted(deltas):
                print(f"        seed {seed}: delta = {d:+.2f}")

    print(f"\n  {'-'*70}\n  diagnostic context (not paired deltas)\n  {'-'*70}")
    for arm in ["frozen", "trained", TREATMENT]:
        drift = df.loc[df["arm"] == arm, DRIFT].dropna()
        pred_loss = df.loc[df["arm"] == arm, PRED_LOSS].dropna()
        drift_str = f"mean={drift.mean():.4f}" if len(drift) else "(none logged)"
        pred_loss_str = f"mean={pred_loss.mean():.4f}" if len(pred_loss) else "(none logged)"
        print(f"    {arm}: drift_from_init {drift_str}; final loss/loss_pred {pred_loss_str}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="augustoxq")
    ap.add_argument("--project", default="cosir_image")
    ap.add_argument("--group", default="condition freeze ablation")
    ap.add_argument("--tag", default=None, help="only include pred_coupled runs carrying this wandb tag")
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
python scripts/analyze_pred_stopgrad_ablation.py --selftest
```

Expected: `SELFTEST OK`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_pred_stopgrad_ablation.py
git commit -m "feat: add paired analysis for pred-stopgrad ablation (Experiment 11.3)"
```

---

### Task 4: Launch the sweep and write up results

**Files:**
- Modify: `docs/reports/2026-08-25_condition_freeze_ablation.md` (append new `## Experiment 11.3` section, matching the existing `## Experiment 11.2` section's structure)

**Interfaces:**
- Consumes: Task 2's sweep script, Task 3's analysis script.
- Produces: a results section other tasks/readers can cite, same as 11.1/11.2's sections in this file.

- [ ] **Step 1: Get explicit user confirmation, then launch the real sweep**

**STOP: confirm with the user before running this** — 3 runs, ~100-epoch RedCaps schedule, multi-hour GPU commitment.

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
bash scripts/run_pred_stopgrad_ablation.sh
```

- [ ] **Step 2: Run the analysis script and capture its output**

```bash
python scripts/analyze_pred_stopgrad_ablation.py --tag pred-stopgrad-ablation-redcaps_150k
```

Save the full printed output — it is the source of every number in Step 3.

- [ ] **Step 3: Append the results section**

Open `docs/reports/2026-08-25_condition_freeze_ablation.md` and append, after the existing `## Experiment 11.2` section's final subsection (`### Reproduction`), a new top-level section:

```markdown
## Experiment 11.3 — bidirectional table↔predictor coupling

### TL;DR for this section

[One paragraph: does removing the stop-gradient on the condition-predictor distillation term
change test_oracle and/or test_pre_diff, relative to 11.1's trained and frozen arms? State the
direction and significance (seed-replicated 3/3, mean/SEM vs. the noise floor) for each of the
four metrics x two baselines from Task 3's output, then state which of the spec's three named
outcome branches (real signal / clean null / reportable instability) applies.]

### Method

One new arm, `pred_coupled` (`train.em_interval=-1`, `loss.pred_stopgrad=false`), 3 seeds,
sharing Experiment 11.1's exact `results_dir` and buddy-init template
(`scripts/run_pred_stopgrad_ablation.sh`), compared against 11.1's already-completed `trained`
and `frozen` arms via `scripts/analyze_pred_stopgrad_ablation.py`. Same operating point as
11.1/11.2: RedCaps-150k, lr=1e-3, lr_label=1e-4, dim=16, alpha=0.5, all training-time buddy
terms off.

### Per-seed results

[Paste Task 3's full per-seed delta output for all four metrics x two baselines, verbatim.]

### Cross-seed synthesis

[Mean ± std and mean/SEM per metric x baseline, from Task 3's summarize() output. State
explicitly whether each clears the noise floor (~0.1-0.7 R1) and is seed-replicated (3/3 in
sign).]

### Diagnostic context

[drift_from_init and final loss/loss_pred for all three arms, from Task 3's output. State
whether pred_coupled's drift moved toward frozen's (near-zero) or stayed close to trained's,
and whether loss_pred's final value is lower/higher/similar to a plain trained run's -- read
this off the same run's own training curve if a single final-step number isn't conclusive.]

### Interpretation

[Which of the three success-criteria branches from the spec's Experiment 11.3 subsection
applies: real signal (escalate to a lambda_pred strength sweep, gated on this result) / clean
null (mechanism ruled out at minimal cost) / reportable instability (worse than trained on
either primary metric). State the practical recommendation for the paper.]

### Reproduction

```bash
bash scripts/run_pred_stopgrad_ablation.sh
python scripts/analyze_pred_stopgrad_ablation.py --tag pred-stopgrad-ablation-redcaps_150k
```
```

Fill in every bracketed section using Step 2's actual output — do not leave any bracket placeholder in the committed file.

- [ ] **Step 4: Commit**

```bash
git add docs/reports/2026-08-25_condition_freeze_ablation.md
git commit -m "results: bidirectional table<->predictor coupling (Experiment 11.3)"
```

---

## Self-review

- **Spec coverage:** Task 1 implements the spec's "What" (config-gated stop-gradient removal). Task 2 implements the spec's scope (1 arm x 3 seeds, shared results_dir/template/group). Task 3 implements the spec's two primary metrics (test_oracle, test_pre_diff) plus its two named free-diagnostic reads (drift_from_init, pred_loss trajectory). Task 4 implements the spec's three-branch success criteria as an explicit interpretation step, and produces the report deliverable the spec's §7 "Results reports" item requires.
- **Placeholder scan:** every code block is complete and runnable as written; Task 4's report template brackets are explicitly called out as required-to-fill, not left-in placeholders, and Step 3 explicitly instructs against committing them unfilled.
- **Type consistency:** `predictor_consistency_loss(pred_cond, label_embeddings, stopgrad=True) -> Tensor` (Task 1) is the only new function signature in this plan; Task 1's own wiring step and test are the only two call sites, and both use identical argument names/order.
