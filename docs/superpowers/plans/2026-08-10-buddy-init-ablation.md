# Buddy-Init Ablation (Experiments 0 & 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two gates on the critical path of `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`: (0) a fast related-work/prior-art check, and (1) the paper's foundational, currently-unmeasured claim — does buddy-graph spectral initialization beat the prior generic (`imgtxt`) initialization on retrieval, with every training-time buddy term held off?

**Architecture:** Experiment 0 is a literature-reading task with a written note as its only deliverable — no code. Experiment 1 reuses the existing Hydra (`main_cosir.py -m`) + wandb pipeline, following the established pattern for sweeping a *template-key* config axis (`scripts/run_blean_impressions.sh`'s `b_weight` loop): `initialization_strategy` is looped in bash (each value gets its own `results_dir` so its own `template_embeddings/`, avoiding template-compatibility races), while `seed` is a Hydra multirun axis inside each loop iteration (seed is not a template key, so seeds correctly share one template — that's intentional replication, not a bug). A new analysis script pairs `imgtxt` vs `buddies` within each `(lr, lr_label, dim, alpha, seed)` cell and reports mean Δ ± std / mean-SEM, per the project's existing statistical convention (`scripts/analyze_buddy_families.py`).

**Tech Stack:** Python 3.10, Hydra/OmegaConf, PyTorch, wandb, pandas/numpy. Existing CoSiR training entrypoint `main_cosir.py`; no new dependencies.

## Global Constraints

- Always run Python/bash training or analysis commands with `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR` first (per `.claude/CLAUDE.md`).
- Statistical standard for every paired comparison (spec §5): ≥3 seeds, paired-within-seed Δ, report mean ± std and mean/SEM (flag `|z| ≥ 2` with `*`), compare the resulting mean Δ against the measured noise floor (~0.1–0.7 R1, `docs/reports/2026-06-24_buddy_progress_report.md` §8a) — **never against zero**.
- Fixed operating point for Experiment 1 (the confirmed strong cell from prior sweeps): `lr=1e-3`, `lr_label=1e-4`, `embedding_dim=16`, `alpha=0.5`.
- Every training-time buddy term must stay OFF: never pass `+loss.lambda_buddy`, `+loss.lambda_buddy_con`, or `+loss.buddy_refresh*` as overrides. Omitting them gives the code's own default of `0.0` / `False` (`src/hook/train_cosir.py:1317-1370`) — this is the *actual* off state, not an approximation of it.
- wandb defaults for this project are `entity=augustoxq`, `project=cosir_image` (`configs/config.yaml:18-19`). New analysis tooling must default to these — `scripts/analyze_buddy_families.py`'s `--project` default (`cosir_scripts`) is stale relative to where runs actually land (every script that calls it overrides `--project cosir_image` explicitly); do not copy that stale default.
- `initialization_strategy` is a template-compatibility key (`src/hook/train_cosir.py:244-271`): a template built under one strategy is rejected and rebuilt under another. Never sweep it as a Hydra multirun axis sharing one `results_dir` — always give each value its own `results_dir`, exactly like `scripts/run_blean_impressions.sh` does for `b_weight`.
- This plan only adds new files under `scripts/`, `docs/reports/`, and `docs/superpowers/plans/`. No existing `src/` file is modified, so no `.claude/YYYYMMDD_log.md` change log is required per CLAUDE.md's rule (which applies to modified existing source files). If any task ends up touching an existing `src/` file, add that log before committing.

---

### Task 0: Related-work / prior-art grounding note (Experiment 0)

**Files:**
- Create: `docs/reports/2026-08-10_prior_art_note.md`

**Interfaces:**
- Consumes: nothing from other tasks — fully independent, can run in parallel with Tasks 1–5.
- Produces: a written verdict (`green` / `yellow` / `red` overlap risk) that feeds the Week-3 checkpoint in `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §6, and the eventual paper's related-work section.

This task has no code — the "test" is a completeness check against a fixed checklist (Step 5).

- [ ] **Step 1: Search for and read the NNCLR paper**

Use WebSearch for `"NNCLR" "nearest-neighbor contrastive learning" Dwibedi 2021` and WebFetch the paper (arXiv or the ICCV page). Read the abstract and method section. Note specifically: NNCLR uses nearest neighbors of an anchor's *own* embedding (drawn from a support set/queue) as additional positives in a single-modality (image) SSL contrastive loss.

- [ ] **Step 2: Search for and read mean-shift / prototype-based SSL (MSF, SwAV)**

Use WebSearch for `"mean shift for self-supervised learning" MSF` and `"SwAV" "swapping assignments between views"`. Read abstracts. Note whether either uses a *graph* structure (vs. a queue/prototype bank) and whether either is cross-modal.

- [ ] **Step 3: Search for graph-Laplacian / spectral embedding init precedent**

Use WebSearch for `"node2vec" graph embedding initialization` and `"Laplacian eigenmaps" initialization deep learning recommender`. Note any prior use of a mutual-kNN or Laplacian-eigenmap graph specifically to *initialize* a trainable per-sample embedding (as opposed to being the whole method).

- [ ] **Step 4: Write the note**

Create `docs/reports/2026-08-10_prior_art_note.md` with this structure (fill every section — no section may be left as a heading only):

```markdown
# Prior-Art Grounding Note — Conditional Buddies

**Date:** 2026-08-10
**Feeds:** docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md §3.4, §6 Week-3 checkpoint

## What exists

### NNCLR (nearest-neighbor positives in SSL)
[2-4 sentences: what it does, how positives are selected, single- vs cross-modal, what it optimizes for.]

### Mean-shift / prototype-based SSL (MSF, SwAV)
[2-4 sentences: same structure.]

### Graph-Laplacian / spectral embedding init precedent (node2vec, recsys)
[2-4 sentences: same structure — specifically note whether any prior work uses this class of
graph as an INITIALIZATION for a trainable embedding vs. as the training method itself.]

## Differentiation

[2-4 sentences: what is genuinely different about conditional-buddies — mutual-kNN graph
built jointly across TWO modalities (image and text, each independently), used ONLY to
INITIALIZE a per-sample trainable condition vector (not as an ongoing training loss in the
validated part of the project), validated for robustness via held-out encoders never used to
build the graph (C1-C3 in the spec) rather than proposed as a new SSL training method.]

## Connection to the C4 finding

[2-3 sentences: does the confound-diagnosis result on Family #2 (contrastive supervision) —
real win on Impressions, explained by near-duplicate structure, does not transfer to
RedCaps — read as a useful cautionary counterpoint to NNCLR-class neighbor-as-positive
claims? Is this worth a paragraph in the paper's related work / discussion?]

## Verdict

**Overlap risk: [green | yellow | red]**

[1-2 sentences justifying the verdict. green = no close prior art, safe to proceed as framed.
yellow = related but distinguishable, framing needs adjustment in the paper (specify what).
red = a near-identical prior method exists — flag for the Week-3 checkpoint before further
compute or writing investment.]
```

- [ ] **Step 5: Completeness check**

Confirm every bracketed placeholder in the template above has been replaced with actual content (no `[...]` remains in the file), and that a `Verdict` of exactly one of `green`/`yellow`/`red` is stated.

- [ ] **Step 6: Commit**

```bash
git add docs/reports/2026-08-10_prior_art_note.md
git commit -m "docs: prior-art grounding note (Experiment 0)"
```

---

### Task 1: Core sweep runner — `scripts/run_init_ablation.sh`

**Files:**
- Create: `scripts/run_init_ablation.sh`

**Interfaces:**
- Consumes: `main_cosir.py` (existing Hydra entrypoint), `configs/train/default.yaml` (`train.initialization_strategy`, `train.buddies.alpha`), `configs/dataset/*.yaml`.
- Produces: per-strategy experiment directories under `${BASE_RESULTS_DIR}/init_${STRAT}/`, wandb runs tagged `${WANDB_TAG}` in group `${WANDB_GROUP}` (default `buddy-init ablation`) — consumed by Task 4's analysis script and invoked by Tasks 2 and 3's wrapper scripts.

There is no pytest-style test for this file — the codebase's own convention for these Hydra-launcher shell scripts is a `SMOKE=1` fast pass (see `scripts/run_buddycon_redcaps.sh`, `scripts/run_blean_impressions.sh`), not a unit test. Step 3 below is that smoke test.

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
set -euo pipefail
# Experiment 1 (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md §4):
# does buddy-graph spectral initialization actually beat the prior generic (imgtxt) init on
# retrieval, with every training-time buddy term held OFF (lambda_buddy=0, lambda_buddy_con=0,
# buddy_refresh=False — all default/absent, never added as an override below)?
#
# initialization_strategy is a TEMPLATE-COMPATIBILITY key (see
# src/hook/train_cosir.py:244-271): a template built under one strategy is REJECTED and
# silently rebuilt under another, so sharing one results_dir across strategies would still be
# "correct" but risks two multirun processes racing on the same template_embeddings/ dir. We
# avoid that entirely by giving each strategy its OWN results_dir, exactly like the b_weight
# sweep does in scripts/run_blean_impressions.sh — a bash loop over the template-key axis
# (strategy), with an inner Hydra multirun over the non-template axis (seed).
#
#   SMOKE=1 DATASET=impressions bash scripts/run_init_ablation.sh   # 2 epochs, seed=1, both strategies
#   DATASET=impressions bash scripts/run_init_ablation.sh           # full sweep for one dataset
#
# Normally called by the per-dataset wrappers (run_init_ablation_impressions.sh,
# run_init_ablation_redcaps.sh), which set DATASET/EPOCHS/EVAL_INTERVAL/TEST_RATIO/
# BASE_RESULTS_DIR/WANDB_TAG. Safe to call directly for ad-hoc reruns.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Template-key axis (bash loop; each value = its own template + results_dir) ──
INIT_STRATEGY_SWEEP="${INIT_STRATEGY_SWEEP:-imgtxt buddies}"

# ── Non-template axis (Hydra multirun; reuses each strategy's template) ─────────
SEED_SWEEP="${SEED_SWEEP:-1,2,3}"

# ── Fixed operating point (confirmed strong cell from the family sweeps) ────────
LR_SWEEP="${LR_SWEEP:-1e-3}"
LR_LABEL_SWEEP="${LR_LABEL_SWEEP:-1e-4}"
EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
ALPHA="${ALPHA:-0.5}"

# ── Dataset + storage (set by the per-dataset wrapper; sane standalone defaults) ─
DATASET="${DATASET:-impressions}"
BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-res/CoSiR_init_ablation/${DATASET}}"
WANDB_TAG="${WANDB_TAG:-init-ablation-${DATASET}}"
WANDB_GROUP="${WANDB_GROUP:-buddy-init ablation}"

if [ -n "${SMOKE:-}" ]; then
  EPOCHS="${EPOCHS:-2}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
  SEED_SWEEP="${SEED_SWEEP_SMOKE:-1}"
  WANDB_TAG="${WANDB_TAG}-smoke"
  echo ">>> SMOKE: 2 epochs, seed=1, both strategies — template-build + pipeline sanity"
else
  EPOCHS="${EPOCHS:-100}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
fi

echo "==================================================================="
echo "Init-strategy ablation ($DATASET): {$INIT_STRATEGY_SWEEP} x seeds={$SEED_SWEEP}"
echo "  fixed: lr=$LR_SWEEP lr_label=$LR_LABEL_SWEEP dim=$EMBEDDING_DIM alpha=$ALPHA"
echo "  EPOCHS=$EPOCHS EVAL_INTERVAL=$EVAL_INTERVAL tag=$WANDB_TAG group=$WANDB_GROUP"
echo "  training-time buddy terms: OFF (not passed -> default lambda_buddy=0, lambda_buddy_con=0, buddy_refresh=False)"
echo "==================================================================="

for STRAT in $INIT_STRATEGY_SWEEP; do
  RD="${BASE_RESULTS_DIR}/init_${STRAT}"
  echo ">>> initialization_strategy=${STRAT}  ->  results_dir=${RD}"
  python main_cosir.py -m \
    dataset="$DATASET" \
    eval.evaluation_interval="$EVAL_INTERVAL" \
    eval.oracle_aggregation=max \
    model=clip_base \
    model.num_layers=6 \
    model.embedding_dim="$EMBEDDING_DIM" \
    optimizer.lr="$LR_SWEEP" \
    optimizer.lr_label="$LR_LABEL_SWEEP" \
    seed="$SEED_SWEEP" \
    train.initialization_strategy="$STRAT" \
    train.buddies.alpha="$ALPHA" \
    train.epochs="$EPOCHS" \
    experiment.results_dir="$RD" \
    wandb.group="$WANDB_GROUP" \
    +loss.log_buddy_preservation=true \
    ${TEST_RATIO:+eval.test_ratio=$TEST_RATIO} \
    ${WANDB_TAG:+++wandb.tags=[$WANDB_TAG]}
done

echo "==================================================================="
echo "Done. Analyse (paired within seed, imgtxt vs buddies, mean delta +/- std) with:"
echo "  python scripts/analyze_init_ablation.py --tag $WANDB_TAG"
echo "==================================================================="
```

- [ ] **Step 2: Make it executable-by-bash (no chmod needed — matches existing convention)**

Confirm the file is invoked as `bash scripts/run_init_ablation.sh` (not `./scripts/run_init_ablation.sh`), matching every other script in `scripts/`. No `chmod +x` step needed.

- [ ] **Step 3: Smoke-test it**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
SMOKE=1 DATASET=impressions bash scripts/run_init_ablation.sh
```
Expected: two short training runs (one per strategy), each printing `Created experiment: ...` and `Experiment directory: res/CoSiR_init_ablation/impressions/init_imgtxt-smoke/...` (and `init_buddies-smoke/...`), completing 2 epochs each with no traceback, ending with the `Done. Analyse ...` banner. The `buddies` arm's log should show the spectral template being built (`Initializing embeddings with buddies strategy...` or the template-load path, per `src/hook/train_cosir.py:298-299`) — confirm this line appears and the run does not error out.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_init_ablation.sh
git commit -m "feat: add core init-strategy ablation sweep runner (Experiment 1)"
```

---

### Task 2: Impressions wrapper — `scripts/run_init_ablation_impressions.sh`

**Files:**
- Create: `scripts/run_init_ablation_impressions.sh`

**Interfaces:**
- Consumes: `scripts/run_init_ablation.sh` (Task 1).
- Produces: wandb runs tagged `init-ablation-impressions`, results under `res/CoSiR_init_ablation/impressions/`.

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
set -euo pipefail
# Experiment 1 on Impressions — see scripts/run_init_ablation.sh for the mechanism.
# 250 epochs matches the Impressions operating point used throughout the family sweeps
# (e.g. scripts/run_buddy_seeds.sh).
export DATASET="impressions"
export EPOCHS="${EPOCHS:-250}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
export BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-res/CoSiR_init_ablation/impressions}"
export WANDB_TAG="${WANDB_TAG:-init-ablation-impressions}"

HERE="$(cd "$(dirname "$0")" && pwd)"
bash "$HERE/run_init_ablation.sh"
```

- [ ] **Step 2: Smoke-test it**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
SMOKE=1 bash scripts/run_init_ablation_impressions.sh
```
Expected: identical behavior to Task 1 Step 3 (this wrapper just fixes `DATASET=impressions` and calls the core script), tag ends up as `init-ablation-impressions-smoke`.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_init_ablation_impressions.sh
git commit -m "feat: add Impressions wrapper for init-strategy ablation"
```

---

### Task 3: RedCaps wrapper — `scripts/run_init_ablation_redcaps.sh`

**Files:**
- Create: `scripts/run_init_ablation_redcaps.sh`

**Interfaces:**
- Consumes: `scripts/run_init_ablation.sh` (Task 1).
- Produces: wandb runs tagged `init-ablation-redcaps`, results under `res/CoSiR_init_ablation/redcaps_150k/`.

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
set -euo pipefail
# Experiment 1 on RedCaps-150k — see scripts/run_init_ablation.sh for the mechanism.
# EPOCHS=100 (not 250) for the same documented reason as scripts/run_buddycon_redcaps.sh: a
# deadline-driven, NOT data-size-scaled schedule that can only shrink a true delta, never
# inflate one. TEST_RATIO=0.2 for the same eval-cost reason (redcaps_test is 25k pairs).
export DATASET="redcaps_150k"
export EPOCHS="${EPOCHS:-100}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
export TEST_RATIO="${TEST_RATIO:-0.2}"
export BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-res/CoSiR_init_ablation/redcaps_150k}"
export WANDB_TAG="${WANDB_TAG:-init-ablation-redcaps}"

HERE="$(cd "$(dirname "$0")" && pwd)"
bash "$HERE/run_init_ablation.sh"
```

- [ ] **Step 2: Smoke-test it**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
SMOKE=1 bash scripts/run_init_ablation_redcaps.sh
```
Expected: identical behavior pattern to Task 2 Step 2, but against `redcaps_150k` features (`configs/dataset/redcaps_150k.yaml`), tag ends up as `init-ablation-redcaps-smoke`. Since RedCaps has never previously been trained through the `buddies`-strategy path in this exact ablation form, specifically confirm the `buddies` arm's spectral template build completes without error (this is the first-ever run of `initialization_strategy=buddies` isolated from all training-time terms on this dataset).

- [ ] **Step 3: Commit**

```bash
git add scripts/run_init_ablation_redcaps.sh
git commit -m "feat: add RedCaps wrapper for init-strategy ablation"
```

---

### Task 4: Analysis script — `scripts/analyze_init_ablation.py`

**Files:**
- Create: `scripts/analyze_init_ablation.py`

**Interfaces:**
- Consumes: wandb runs produced by Tasks 1–3 (group `buddy-init ablation`, tags `init-ablation-impressions` / `init-ablation-redcaps`).
- Produces: printed paired Δ tables (mean ± std, mean/SEM) per metric (`test_oracle/t2i_R1`, `test_oracle/i2t_R1`) — read directly by Task 5's decision step and quoted in Task 5's results report.

This script has one pure, wandb-free function (`compute_paired_deltas` + `summarize`) that is unit-testable offline. Follow TDD for that function; the wandb-dependent `fetch()`/`analyze()` wrapper is exercised for real in Task 5 against live data (no pytest infra exists for wandb-fetching scripts anywhere in this codebase — `scripts/analyze_buddy_families.py` has none either — so don't invent one here; a real invocation against live runs in Task 5 is this codebase's actual verification method for that half).

- [ ] **Step 1: Write the failing selftest**

Create `scripts/analyze_init_ablation.py` with **only** the imports, constants, and the `_selftest()` function below (leave `compute_paired_deltas` and `summarize` undefined for now):

```python
"""
Paired analysis for Experiment 1 (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md
S4): does buddy-graph spectral initialization beat the prior generic (imgtxt) initialization on
retrieval, with every training-time buddy term held off?

Reads wandb runs from the 'buddy-init ablation' group (scripts/run_init_ablation.sh), pairs
imgtxt vs buddies WITHIN each (lr, lr_label, dim, alpha, seed) cell, and reports mean delta +/- std
and mean/SEM (the project's standard significance read - see spec S5). Compare the resulting
mean delta against the measured noise floor (~0.1-0.7 R1 from a duplicate-config run,
docs/reports/2026-06-24_buddy_progress_report.md S8a), NOT against zero.

Usage
-----
  python scripts/analyze_init_ablation.py --tag init-ablation-impressions
  python scripts/analyze_init_ablation.py --tag init-ablation-redcaps
  python scripts/analyze_init_ablation.py --selftest   # offline arithmetic check, no wandb call

Requires: wandb, pandas, numpy (all already deps).
"""
import argparse

import numpy as np
import pandas as pd

T2I = "test_oracle/t2i_R1"
I2T = "test_oracle/i2t_R1"
BASELINE = "imgtxt"
TREATMENT = "buddies"
CELL = [
    ("lr", ("optimizer", "lr")),
    ("lr_label", ("optimizer", "lr_label")),
    ("dim", ("model", "embedding_dim")),
    ("alpha", ("train", "buddies", "alpha")),
    ("seed", ("seed",)),
]


def _selftest():
    """Offline arithmetic check - no wandb call. Verifies compute_paired_deltas/summarize
    against hand-computed numbers before ever touching real run data."""
    df = pd.DataFrame([
        # seed=1: imgtxt=50.0, buddies=52.0 -> delta +2.0
        {"strategy": "imgtxt", "lr": 1e-3, "lr_label": 1e-4, "dim": 16, "alpha": 0.5, "seed": 1, T2I: 50.0},
        {"strategy": "buddies", "lr": 1e-3, "lr_label": 1e-4, "dim": 16, "alpha": 0.5, "seed": 1, T2I: 52.0},
        # seed=2: imgtxt=48.0, buddies=51.0 -> delta +3.0
        {"strategy": "imgtxt", "lr": 1e-3, "lr_label": 1e-4, "dim": 16, "alpha": 0.5, "seed": 2, T2I: 48.0},
        {"strategy": "buddies", "lr": 1e-3, "lr_label": 1e-4, "dim": 16, "alpha": 0.5, "seed": 2, T2I: 51.0},
        # seed=3: imgtxt=49.0, buddies=49.0 -> delta 0.0
        {"strategy": "imgtxt", "lr": 1e-3, "lr_label": 1e-4, "dim": 16, "alpha": 0.5, "seed": 3, T2I: 49.0},
        {"strategy": "buddies", "lr": 1e-3, "lr_label": 1e-4, "dim": 16, "alpha": 0.5, "seed": 3, T2I: 49.0},
    ])
    deltas = compute_paired_deltas(df, T2I)
    assert len(deltas) == 3, f"expected 3 paired cells, got {len(deltas)}"
    got = sorted(d for _, d in deltas)
    want = [0.0, 2.0, 3.0]
    assert got == want, f"expected deltas {want}, got {got}"
    s = summarize(deltas)
    assert s["n"] == 3
    assert abs(s["mean"] - (5.0 / 3)) < 1e-9, f"expected mean {5.0/3}, got {s['mean']}"
    expected_std = np.std([0.0, 2.0, 3.0], ddof=1)
    assert abs(s["std"] - expected_std) < 1e-9
    assert s["wins"] == 2, f"expected 2 wins (deltas > 0), got {s['wins']}"
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
python scripts/analyze_init_ablation.py --selftest
```
Expected: `NameError: name 'compute_paired_deltas' is not defined`.

- [ ] **Step 3: Implement `compute_paired_deltas`, `summarize`, and the rest of the script**

Add the following above `_selftest()` (after the `CELL` constant, before `def _selftest():`):

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
        strat = cget(cfg, ("train", "initialization_strategy"))
        if strat not in (BASELINE, TREATMENT):
            continue
        row = {
            "run_id": run.id,
            "state": run.state,
            "strategy": strat,
            T2I: float(sget(summ, T2I)) if not np.isnan(sget(summ, T2I)) else np.nan,
            I2T: float(sget(summ, I2T)) if not np.isnan(sget(summ, I2T)) else np.nan,
        }
        for cname, cpath in CELL:
            cv = cget(cfg, cpath)
            row[cname] = float(cv) if cv is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compute_paired_deltas(df, metric):
    """Pair BASELINE vs TREATMENT within each CELL. Returns list of (cell_key, delta) where
    delta = treatment - baseline. Pure function - no wandb, no I/O."""
    cell_cols = [c for c, _ in CELL]
    deltas = []
    for cell_key, cell in df.groupby(cell_cols, dropna=False):
        by_strat = cell.groupby("strategy")[metric].max()
        if BASELINE not in by_strat.index or TREATMENT not in by_strat.index:
            continue
        b, t = by_strat[BASELINE], by_strat[TREATMENT]
        if np.isnan(b) or np.isnan(t):
            continue
        deltas.append((cell_key, t - b))
    return deltas


def summarize(deltas):
    """mean, std, sem, z=mean/sem, win-rate - matches analyze_buddy_families.py's convention."""
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


def paired_table(df, metric):
    cell_cols = [c for c, _ in CELL]
    deltas = compute_paired_deltas(df, metric)
    print(f"\n  --- {metric} ---")
    header = cell_cols + [BASELINE, TREATMENT, "delta(buddies-imgtxt)"]
    print("    " + "  ".join(f"{h:>13}" for h in header))
    for cell_key, cell in df.groupby(cell_cols, dropna=False):
        by_strat = cell.groupby("strategy")[metric].max()
        if BASELINE not in by_strat.index or TREATMENT not in by_strat.index:
            continue
        b, t = by_strat[BASELINE], by_strat[TREATMENT]
        vals = list(cell_key) if isinstance(cell_key, tuple) else [cell_key]
        vals = [f"{v:g}" if not (isinstance(v, float) and np.isnan(v)) else "-" for v in vals]
        vals += [f"{b:.2f}" if not np.isnan(b) else "  -  ",
                 f"{t:.2f}" if not np.isnan(t) else "  -  ",
                 f"{(t - b):+.2f}" if not (np.isnan(b) or np.isnan(t)) else "  -  "]
        print("    " + "  ".join(f"{v:>13}" for v in vals))
    s = summarize(deltas)
    if s["n"] == 0:
        print("    (no paired cells with both imgtxt and buddies present)")
        return
    sig = f"  mean/SEM={s['z']:+.1f}{' *' if not np.isnan(s['z']) and abs(s['z']) >= 2 else ''}" if s["n"] > 1 else ""
    spread = f" +/- {s['std']:.2f}" if s["n"] > 1 else ""
    print(f"\n    Over {s['n']} paired cell(s): buddies beats imgtxt in {s['wins']}/{s['n']} "
          f"(mean delta = {s['mean']:+.2f}{spread} R1 pts){sig}")
    print("    Compare mean delta against the noise floor (~0.1-0.7 R1, NOT zero) - "
          "see docs/reports/2026-06-24_buddy_progress_report.md S8a.")


def analyze(entity, project, group, tag=None):
    print(f"\n{'='*78}\nExperiment 1 - buddy-init vs. imgtxt-init  group='{group}'"
          + (f"  tag='{tag}'" if tag else "") + f"\n{'='*78}")
    df = fetch(entity, project, group, tag=tag)
    if df.empty:
        print("  (no runs found - check --entity/--project/--tag)")
        return
    n_unfinished = (df["state"] != "finished").sum()
    print(f"  {len(df)} run(s); strategies present: {sorted(df['strategy'].unique())}."
          + (f"  [{n_unfinished} not finished -> best-so-far]" if n_unfinished else ""))
    paired_table(df, T2I)
    paired_table(df, I2T)
```

Then replace the `if __name__ == "__main__":` block at the bottom with:

```python
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="augustoxq")
    ap.add_argument("--project", default="cosir_image")
    ap.add_argument("--group", default="buddy-init ablation")
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

- [ ] **Step 4: Run the selftest again to verify it passes**

```bash
python scripts/analyze_init_ablation.py --selftest
```
Expected: `SELFTEST OK`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_init_ablation.py
git commit -m "feat: add paired analysis script for init-strategy ablation (Experiment 1)"
```

---

### Task 5: Launch the full sweep

**Files:** none (execution only).

**Interfaces:**
- Consumes: Tasks 1–3's scripts.
- Produces: 12 finished wandb runs (2 strategies x 3 seeds x 2 datasets) feeding Task 6's analysis.

- [ ] **Step 1: Confirm smoke tests passed**

Verify Tasks 1–3's smoke-test steps all completed without error before spending full compute (`SMOKE=1` runs for both wrapper scripts).

- [ ] **Step 2: Launch the Impressions sweep**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
bash scripts/run_init_ablation_impressions.sh
```
This runs 2 strategies x 3 seeds = 6 runs at 250 epochs each. Long-running — launch with `run_in_background` or `nohup ... &` if executing interactively.

- [ ] **Step 3: Launch the RedCaps sweep**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
bash scripts/run_init_ablation_redcaps.sh
```
This runs 2 strategies x 3 seeds = 6 runs at 100 epochs each.

- [ ] **Step 4: Verify all 12 runs finished**

Check the wandb UI (project `cosir_image`, group `buddy-init ablation`, tags `init-ablation-impressions` / `init-ablation-redcaps`) or query via `wandb.Api()` that all 12 runs show `state == "finished"` with `test_oracle/t2i_R1` and `test_oracle/i2t_R1` present in their summary.

---

### Task 6: Analyze results and write the report

**Files:**
- Create: `docs/reports/2026-08-10_buddy_init_ablation.md` (adjust date to when Task 5 actually completes)
- Modify: `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` (update §2's claims table or add a new row once the result is in — see Step 3)

**Interfaces:**
- Consumes: `scripts/analyze_init_ablation.py` (Task 4) output.
- Produces: the Week-3-checkpoint input for the venue-tier decision in the spec's §3.2/§6.

- [ ] **Step 1: Run the analysis for each dataset**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python scripts/analyze_init_ablation.py --tag init-ablation-impressions
python scripts/analyze_init_ablation.py --tag init-ablation-redcaps
```
Capture the full printed output (paired tables + mean Δ ± std + mean/SEM for both `t2i_R1` and `i2t_R1`, both datasets).

- [ ] **Step 2: Apply the spec's decision rule**

Per spec §4 Experiment 1's success criteria:
- **Positive**: buddy-init beats imgtxt-init, seed-replicated, `mean/SEM ≥ 2`, on ≥2/3 datasets tested → paper leads with this as the headline result; stretch-tier venue becomes live.
- **Null**: no reliable difference → reframe as "the signal is real and robust, but content-aware geometric initialization alone does not measurably improve retrieval over a generic PCA init."
- **Negative**: imgtxt beats buddies → same reframe, plus a short discussion note.

Determine which outcome applies, checking the mean Δ against the noise floor (~0.1–0.7 R1) as printed by the analysis script, not against zero.

- [ ] **Step 3: Write the results report**

Create `docs/reports/2026-08-10_buddy_init_ablation.md` following the structure of `docs/reports/2026-07-16_buddy_cross_vlm_survival.md` (method, results tables, interpretation, caveats, reproduction commands). Include: the exact operating point, the full paired tables from Step 1, the decision-rule outcome from Step 2, and a pointer back to the spec.

- [ ] **Step 4: Update the spec**

In `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`, add the Experiment 1 outcome as a new row in §2's claims table (or amend §3.2's TMLR/stretch-tier conditions to reflect the actual result rather than the hypothetical), citing the new report.

- [ ] **Step 5: Commit**

```bash
git add docs/reports/2026-08-10_buddy_init_ablation.md docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md
git commit -m "results: buddy-init vs imgtxt-init ablation (Experiment 1)"
```

---

## Self-Review

- **Spec coverage:** Task 0 covers spec §3.4/Experiment 0 (prior-art grounding). Tasks 1–6 cover spec §4 Experiment 1 end-to-end (script → wrapper scripts → analysis → execution → report → spec update).
- **Placeholder scan:** every code block is complete and runnable as written; Task 0's note template is the one place with bracketed placeholders, and Step 5 of that task explicitly requires them all to be resolved before commit — this is a content-completeness gate, not a plan placeholder.
- **Type/interface consistency:** `compute_paired_deltas(df, metric) -> list[(cell_key, float)]` and `summarize(deltas) -> dict` are defined once in Task 4 Step 3 and used identically in `paired_table`/`_selftest`; no renamed variants appear elsewhere.
- **Scope check:** Experiments 2–7 from the spec are explicitly out of scope for this plan (only Experiments 0 and 1 — the critical path — are covered here); each would get its own plan per the spec's §7 deliverable note ("one plan per experiment cluster").
