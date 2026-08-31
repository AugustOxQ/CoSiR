# Training-Validity Audit (Experiment 12.2/12.3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the tooling for Experiment 12.2 (per-epoch training-trajectory audit) and Experiment 12.3 (cross-recipe, cross-seed, signed replication of Experiment 12's bridge/delta_rank cross-reference), then run both for real and update Experiment 12's report.

**Architecture:** Two independent, additive pieces of tooling, both reusing only already-completed checkpoints/wandb runs (zero new training):
1. A new script, `scripts/analyze_training_trajectory.py`, pulls per-epoch (not just final-epoch) `test_oracle/{t2i,i2t}_R1` history from wandb for all 9 already-completed runs (11.1's 3 `trained` + 3 `frozen`, 11.3's 3 `pred_coupled`) and characterizes whether the trained/pred_coupled-vs-frozen i2t gap is present from the start or grows over training.
2. `scripts/analyze_polysemy_bridges.py`'s existing `correlate_polysemy_with_retrieval()` function gets a signed-`delta_rank` addition, and its `run()`/CLI gain support for cross-referencing against **multiple** per-sample dumps at once (one per seed/recipe), with a new pooling function that reports mean/std/sem/z across runs — the project's existing multi-seed convention.

**Tech Stack:** Python, numpy, pandas, wandb API, scipy.stats (already deps), matplotlib (`Agg` backend, already used for Experiment 12's figures).

**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 12.2/12.3 (12.4 is out of scope for this plan — it is gated on this plan's own results and not committed).

## Global Constraints

- **No new training, no new eval runs.** Every number this plan produces comes from wandb history already logged by 11.1/11.3's 9 completed runs, or from `--dump-per-sample` re-runs of `scripts/analyze_condition_retrieval_correlation.py` against already-completed checkpoints (that script and its `--dump-per-sample` flag already exist — Experiment 12's own Task 2 built it).
- **Reuse existing conventions exactly:** wandb `entity="augustoxq"`, `project="cosir_image"`, `group="condition freeze ablation"`; tags `condition-freeze-ablation-redcaps_150k` (trained/frozen) and `pred-stopgrad-ablation-redcaps_150k` (pred_coupled); mean/std/sem/`z=mean/sem` significance convention (flag `|z|>=2`); figures via matplotlib `Agg` backend, `fig.savefig(path, dpi=130)`, embedded in reports as `![alt](assets/<topic>/name.png)`.
- **Every new script/function ships with an offline `_selftest()`** (no network/wandb/GPU calls) exactly like `analyze_condition_freeze_ablation.py` and `analyze_polysemy_bridges.py` already do — this repo's convention is an embedded self-test function, not a separate pytest file, for these `scripts/analyze_*.py` diagnostics.
- **Do not create `.ccg/tasks/` scaffolding or extra archive commits** — one clean, focused commit per task, source files only.

---

### Task 1: `analyze_training_trajectory.py` — history fetch + pure gap computation

**Files:**
- Create: `scripts/analyze_training_trajectory.py`

**Interfaces:**
- Produces: `fetch_history(entity: str, project: str, group: str, tags: dict) -> pandas.DataFrame` where `tags` maps arm name -> wandb tag (e.g. `{"trained": "condition-freeze-ablation-redcaps_150k", "frozen": "condition-freeze-ablation-redcaps_150k", "pred_coupled": "pred-stopgrad-ablation-redcaps_150k"}`), returning one row per (run, logged-epoch) with columns `arm`, `seed`, `epoch`, `t2i` (`test_oracle/t2i_R1`), `i2t` (`test_oracle/i2t_R1`).
- Produces: `compute_epoch_gaps(df: pandas.DataFrame, baseline: str = "frozen", metric: str = "i2t") -> pandas.DataFrame` — pure function, no I/O, columns `arm`, `seed`, `epoch`, `gap` (`treatment - baseline` at that epoch, paired within seed; `arm` is the non-baseline arm name). Rows where a (seed, epoch) pair is missing from either side are dropped, not NaN-filled.

- [ ] **Step 1: Write the failing test for `compute_epoch_gaps`**

```python
def test_compute_epoch_gaps_pairs_within_seed():
    import pandas as pd
    df = pd.DataFrame([
        {"arm": "frozen", "seed": 1, "epoch": 0, "i2t": 10.0},
        {"arm": "frozen", "seed": 1, "epoch": 10, "i2t": 12.0},
        {"arm": "trained", "seed": 1, "epoch": 0, "i2t": 10.0},
        {"arm": "trained", "seed": 1, "epoch": 10, "i2t": 9.0},
        {"arm": "frozen", "seed": 2, "epoch": 0, "i2t": 11.0},
        {"arm": "trained", "seed": 2, "epoch": 0, "i2t": 11.5},
        # seed 2 has no epoch-10 trained row -> that (seed, epoch) must be dropped, not NaN
    ])
    gaps = compute_epoch_gaps(df, baseline="frozen", metric="i2t")
    got = sorted(
        (row.seed, row.epoch, round(row.gap, 4)) for row in gaps.itertuples()
    )
    want = [(1, 0, 0.0), (1, 10, -3.0), (2, 0, 0.5)]
    assert got == want, got
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /project/CoSiR && source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python -c "from scripts.analyze_training_trajectory import compute_epoch_gaps"` (or add the test body inline to `_selftest()` per this repo's convention — see Step 3)
Expected: `ImportError` / `NameError` — `compute_epoch_gaps` does not exist yet.

- [ ] **Step 3: Write `fetch_history` and `compute_epoch_gaps`**

```python
"""
Experiment 12.2 (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md,
Experiment 12.2 subsection): does the trained/pred_coupled-vs-frozen i2t retrieval
deficit (established in Experiment 11.1/11.3, docs/reports/2026-08-25_condition_freeze_ablation.md)
hold from the earliest logged epoch, or does it grow over the 100-epoch run?

Pulls ALREADY-LOGGED per-epoch test_oracle/{t2i,i2t}_R1 history (not just the final-epoch
summary 11.1/11.3's own scripts read) from wandb for all 9 already-completed runs (11.1's
3 trained + 3 frozen, 11.3's 3 pred_coupled). No new training, no new eval runs.

Usage
-----
  python scripts/analyze_training_trajectory.py --selftest   # offline check, no wandb call
  python scripts/analyze_training_trajectory.py \\
      --tag condition-freeze-ablation-redcaps_150k \\
      --pred-coupled-tag pred-stopgrad-ablation-redcaps_150k \\
      --out-fig docs/reports/assets/training_trajectory/i2t_gap_trajectory.png

Requires: wandb, pandas, numpy, matplotlib (all already deps).
"""
import argparse

import numpy as np
import pandas as pd

T2I = "test_oracle/t2i_R1"
I2T = "test_oracle/i2t_R1"


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


def fetch_history(entity: str, project: str, group: str, tags: dict) -> pd.DataFrame:
    """Pull per-epoch test_oracle/{t2i,i2t}_R1 history for every finished run in `group`
    whose tags include one of `tags`' values, tagging each row with the arm read from
    that run's own config (train.arm) and the seed. `tags` maps arm name -> wandb tag,
    e.g. {"trained": "...redcaps_150k", "frozen": "...redcaps_150k",
    "pred_coupled": "pred-stopgrad-ablation-redcaps_150k"} -- multiple arms may share one
    tag (11.1's trained/frozen both carry the same tag; only `train.arm` distinguishes
    them), so this fetches the UNION of tag values once, not once per arm name."""
    import wandb
    api = wandb.Api()
    wanted_tags = set(tags.values())
    rows = []
    for run in api.runs(f"{entity}/{project}", filters={"group": group}):
        if run.state != "finished":
            continue
        if not wanted_tags.intersection(run.tags or []):
            continue
        arm = cget(run.config, ("train", "arm"))
        seed = cget(run.config, ("seed",))
        if not arm or arm not in tags:
            continue
        hist = run.history(keys=[T2I, I2T], pandas=True)
        if hist is None or hist.empty:
            continue
        for _, hrow in hist.iterrows():
            epoch = hrow.get("_step")
            if epoch is None or pd.isna(epoch):
                continue
            rows.append({
                "arm": arm, "seed": seed, "epoch": int(epoch),
                "t2i": float(hrow[T2I]) if T2I in hrow and not pd.isna(hrow[T2I]) else np.nan,
                "i2t": float(hrow[I2T]) if I2T in hrow and not pd.isna(hrow[I2T]) else np.nan,
            })
    return pd.DataFrame(rows)


def compute_epoch_gaps(df: pd.DataFrame, baseline: str = "frozen", metric: str = "i2t") -> pd.DataFrame:
    """For every non-baseline arm present, pair it against `baseline` within (seed, epoch)
    and return the treatment-minus-baseline gap. Pure function -- no wandb, no I/O. Rows
    where the (seed, epoch) cell is missing on either side are dropped, not NaN-filled,
    since a partial-history run (e.g. an early crash) must not silently contribute a
    one-sided gap."""
    base = df[df["arm"] == baseline][["seed", "epoch", metric]].rename(columns={metric: "_base"})
    out_rows = []
    for arm in sorted(set(df["arm"]) - {baseline}):
        treat = df[df["arm"] == arm][["seed", "epoch", metric]].rename(columns={metric: "_treat"})
        merged = treat.merge(base, on=["seed", "epoch"], how="inner")
        for _, r in merged.iterrows():
            if pd.isna(r["_treat"]) or pd.isna(r["_base"]):
                continue
            out_rows.append({"arm": arm, "seed": r["seed"], "epoch": int(r["epoch"]),
                              "gap": float(r["_treat"] - r["_base"])})
    return pd.DataFrame(out_rows)
```

- [ ] **Step 4: Run test to verify it passes**

Add the Step 1 test body into `_selftest()` (see Task 2, Step 1) and run:
`python scripts/analyze_training_trajectory.py --selftest`
Expected: no assertion error for this check (full `_selftest()` completes in Task 2 once the rest of the file exists).

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_training_trajectory.py
git commit -m "feat: add wandb per-epoch history fetch + gap computation for Experiment 12.2"
```

---

### Task 2: `analyze_training_trajectory.py` — plotting, CLI, and full selftest

**Files:**
- Modify: `scripts/analyze_training_trajectory.py`

**Interfaces:**
- Consumes: `fetch_history`, `compute_epoch_gaps` from Task 1.
- Produces: `render_trajectory_figure(gaps: pandas.DataFrame, metric_label: str, out_path: str) -> None` — matplotlib line plot, one line per (arm, seed), x=epoch, y=gap, zero-reference horizontal line, `Agg` backend, `dpi=130`, matching Experiment 12's figure convention (`docs/reports/assets/polysemy_bridges/*.png` is the style template).
- Produces: `main()` CLI with `--entity` (default `augustoxq`), `--project` (default `cosir_image`), `--group` (default `condition freeze ablation`), `--tag` (default `condition-freeze-ablation-redcaps_150k`, covers both `trained`/`frozen`), `--pred-coupled-tag` (default `pred-stopgrad-ablation-redcaps_150k`), `--out-fig` (default `None`), `--out-json` (default `None`), `--selftest`.

- [ ] **Step 1: Write the failing test for `render_trajectory_figure` and extend `_selftest()`**

```python
def _selftest():
    """Offline arithmetic + plotting check -- no wandb call."""
    df = pd.DataFrame([
        {"arm": "frozen", "seed": 1, "epoch": 0, "i2t": 10.0},
        {"arm": "frozen", "seed": 1, "epoch": 10, "i2t": 12.0},
        {"arm": "trained", "seed": 1, "epoch": 0, "i2t": 10.0},
        {"arm": "trained", "seed": 1, "epoch": 10, "i2t": 9.0},
        {"arm": "frozen", "seed": 2, "epoch": 0, "i2t": 11.0},
        {"arm": "trained", "seed": 2, "epoch": 0, "i2t": 11.5},
    ])
    gaps = compute_epoch_gaps(df, baseline="frozen", metric="i2t")
    got = sorted((row.seed, row.epoch, round(row.gap, 4)) for row in gaps.itertuples())
    want = [(1, 0, 0.0), (1, 10, -3.0), (2, 0, 0.5)]
    assert got == want, got

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        fig_path = os.path.join(tmp, "gap.png")
        render_trajectory_figure(gaps, metric_label="i2t", out_path=fig_path)
        assert os.path.exists(fig_path) and os.path.getsize(fig_path) > 0

    print("SELFTEST OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python scripts/analyze_training_trajectory.py --selftest`
Expected: `NameError: name 'render_trajectory_figure' is not defined`.

- [ ] **Step 3: Implement `render_trajectory_figure` and `main()`**

```python
def render_trajectory_figure(gaps: pd.DataFrame, metric_label: str, out_path: str) -> None:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (arm, seed), g in gaps.groupby(["arm", "seed"]):
        g = g.sort_values("epoch")
        ax.plot(g["epoch"], g["gap"], marker="o", markersize=3,
                label=f"{arm} seed {seed}", alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{metric_label} gap (treatment - frozen)")
    ax.set_title(f"Per-epoch {metric_label} gap vs. frozen baseline")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="augustoxq")
    ap.add_argument("--project", default="cosir_image")
    ap.add_argument("--group", default="condition freeze ablation")
    ap.add_argument("--tag", default="condition-freeze-ablation-redcaps_150k",
                    help="wandb tag covering the trained/frozen runs (11.1)")
    ap.add_argument("--pred-coupled-tag", default="pred-stopgrad-ablation-redcaps_150k",
                    help="wandb tag covering the pred_coupled runs (11.3)")
    ap.add_argument("--out-fig", default=None, help="write the i2t gap trajectory figure here")
    ap.add_argument("--out-json", default=None, help="write the per-epoch gap table (JSON) here")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return

    tags = {"trained": args.tag, "frozen": args.tag, "pred_coupled": args.pred_coupled_tag}
    df = fetch_history(args.entity, args.project, args.group, tags)
    if df.empty:
        print("  (no runs found - check --entity/--project/--tag/--pred-coupled-tag)")
        return

    print(f"\n{'='*78}\nExperiment 12.2 - training-trajectory audit  group='{args.group}'\n{'='*78}")
    print(f"  {len(df)} (run, epoch) row(s); arms present: {sorted(df['arm'].unique())}.")

    for metric, label in ((I2T, "i2t"), (T2I, "t2i")):
        gaps = compute_epoch_gaps(df, baseline="frozen", metric=label)
        print(f"\n  --- {label} gap vs frozen (treatment - frozen), by (arm, seed, epoch) ---")
        for (arm, seed), g in gaps.groupby(["arm", "seed"]):
            g = g.sort_values("epoch")
            trail = ", ".join(f"e{int(r.epoch)}={r.gap:+.2f}" for r in g.itertuples())
            print(f"    {arm} seed {seed}: {trail}")
        if args.out_fig and label == "i2t":
            fig_path = args.out_fig
            render_trajectory_figure(gaps, metric_label=label, out_path=fig_path)
            print(f"  Wrote {fig_path}")
        if args.out_json:
            gaps.to_json(args.out_json.replace(".json", f"_{label}.json"), orient="records", indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python scripts/analyze_training_trajectory.py --selftest`
Expected: `SELFTEST OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_training_trajectory.py
git commit -m "feat: add trajectory figure + CLI for Experiment 12.2"
```

---

### Task 3: `analyze_polysemy_bridges.py` — signed `delta_rank` in the cross-reference

**Files:**
- Modify: `scripts/analyze_polysemy_bridges.py` (`correlate_polysemy_with_retrieval`, `_selftest`, `main`)

**Interfaces:**
- Consumes: existing `correlate_polysemy_with_retrieval(labels, sample_ids, npz_path, return_raw=False) -> dict` (this task extends its return value, does not change its signature).
- Produces: the result dict now also contains, per label, `"median_delta_rank"` (signed, alongside the existing `"median_abs_delta_rank"`), and at the top level `"corr_is_polysemic_vs_delta_rank"` (signed, alongside the existing `"corr_is_polysemic_vs_abs_delta_rank"`).

- [ ] **Step 1: Write the failing test (extend `_selftest`'s existing `correlate_polysemy_with_retrieval` check)**

```python
    # (inside _selftest, right after the existing correlate_polysemy_with_retrieval block)
    assert result["bridge"]["median_delta_rank"] == 10.0, result  # only one bridge row (id 100, delta_rank=10)
    assert "corr_is_polysemic_vs_delta_rank" in result, result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python scripts/analyze_polysemy_bridges.py --selftest`
Expected: `KeyError: 'median_delta_rank'`

- [ ] **Step 3: Implement the signed fields**

In `correlate_polysemy_with_retrieval`, inside the per-label loop, add one key next to the existing `"median_abs_delta_rank"`:

```python
        result[lbl] = {
            "n": int(mask.sum()),
            "median_abs_delta_rank": float(np.median(np.abs(delta_rank[mask]))),
            "median_delta_rank": float(np.median(delta_rank[mask])),
            "median_condition_drift": float(np.median(condition_drift[mask])),
            "median_embedding_shift": float(np.median(embedding_shift[mask])),
        }
```

And right after the existing `result["corr_is_polysemic_vs_abs_delta_rank"] = ...` line, add:

```python
    result["corr_is_polysemic_vs_delta_rank"] = spearman_correlate(is_polysemic, delta_rank)
```

Also update `main()`'s print block for the single-dump path to print the new signed median alongside the existing one:

```python
            if lbl in rc:
                print(f"    {lbl}: n={rc[lbl]['n']} median|delta_rank|={rc[lbl]['median_abs_delta_rank']:.1f} "
                      f"median_delta_rank={rc[lbl]['median_delta_rank']:+.1f} "
                      f"median_drift={rc[lbl]['median_condition_drift']:.4f}")
        c1 = rc["corr_is_polysemic_vs_abs_delta_rank"]
        print(f"    corr(is_polysemic, |delta_rank|): rho={c1['rho']:+.3f} p={c1['p']:.3e}")
        c2 = rc["corr_is_polysemic_vs_delta_rank"]
        print(f"    corr(is_polysemic, delta_rank):   rho={c2['rho']:+.3f} p={c2['p']:.3e}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python scripts/analyze_polysemy_bridges.py --selftest`
Expected: `SELFTEST OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_polysemy_bridges.py
git commit -m "feat: add signed delta_rank to Experiment 12 bridge/retrieval cross-reference"
```

---

### Task 4: `analyze_polysemy_bridges.py` — multi-run cross-reference + pooling (Experiment 12.3)

**Files:**
- Modify: `scripts/analyze_polysemy_bridges.py` (`run`, `main`, `_selftest`; new `pool_cross_references`)

**Interfaces:**
- Consumes: `correlate_polysemy_with_retrieval` (Task 3's version).
- Produces: `pool_cross_references(results: List[dict], tags: List[str]) -> dict` — pure function, no I/O. `results` is a list of `correlate_polysemy_with_retrieval(...)` return dicts, `tags` a parallel list of run identifiers (e.g. `"trained/seed1"`). Returns `{"n_runs": int, "per_run": {tag: result, ...}, "pooled": {"corr_is_polysemic_vs_abs_delta_rank": {"n", "mean", "std", "sem", "z"}, "corr_is_polysemic_vs_delta_rank": {...}}}` where the pooled stats are mean/std/sem/`z=mean/sem` of each run's own `rho` value across runs (the project's standard multi-seed convention, matching `summarize()` in `scripts/analyze_condition_freeze_ablation.py`).
- Changes `run()`'s `per_sample_npz` parameter from a single path to accept either a single path (`str`) or a list of paths (`List[str]`) — backward compatible; when a list of length > 1 is given, `result["retrieval_correlation"]` becomes the `pool_cross_references(...)` output instead of a single `correlate_polysemy_with_retrieval` dict, tagged by each npz path's parent-parent directory name (the run dir, e.g. `20260825_161846_CoSiR_Experiment`).
- CLI: `--per-sample-npz` becomes `nargs="+"` (still accepts one path, unchanged behavior for existing single-path invocations from Experiment 12's own report).

- [ ] **Step 1: Write the failing test for `pool_cross_references`**

```python
    # pool_cross_references: two runs' cross-reference results -> pooled mean/std/sem/z
    # of each run's own rho, matching the project's standard multi-seed convention.
    r1 = {"corr_is_polysemic_vs_abs_delta_rank": {"rho": 0.10, "p": 0.01},
          "corr_is_polysemic_vs_delta_rank": {"rho": 0.02, "p": 0.5}, "n_joined": 100}
    r2 = {"corr_is_polysemic_vs_abs_delta_rank": {"rho": 0.20, "p": 0.01},
          "corr_is_polysemic_vs_delta_rank": {"rho": -0.02, "p": 0.5}, "n_joined": 90}
    pooled = pool_cross_references([r1, r2], tags=["trained/seed1", "trained/seed2"])
    assert pooled["n_runs"] == 2
    assert set(pooled["per_run"].keys()) == {"trained/seed1", "trained/seed2"}
    abs_stats = pooled["pooled"]["corr_is_polysemic_vs_abs_delta_rank"]
    assert abs(abs_stats["mean"] - 0.15) < 1e-9, abs_stats
    assert abs_stats["n"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python scripts/analyze_polysemy_bridges.py --selftest`
Expected: `NameError: name 'pool_cross_references' is not defined`

- [ ] **Step 3: Implement `pool_cross_references` and wire it into `run()`/`main()`**

```python
def pool_cross_references(results: List[dict], tags: List[str]) -> dict:
    """Pool multiple already-computed correlate_polysemy_with_retrieval() results (one
    per run/seed) into a per-run table plus mean/std/sem/z of each run's own rho, across
    runs -- this project's standard multi-seed synthesis convention (see summarize() in
    scripts/analyze_condition_freeze_ablation.py). Does not re-touch any per-sample data;
    purely aggregates already-computed per-run dicts."""
    assert len(results) == len(tags), (len(results), len(tags))
    per_run = {tag: r for tag, r in zip(tags, results)}
    pooled = {}
    for corr_key in ("corr_is_polysemic_vs_abs_delta_rank", "corr_is_polysemic_vs_delta_rank"):
        rhos = np.array([r[corr_key]["rho"] for r in results], dtype=float)
        n = len(rhos)
        mean = float(rhos.mean())
        std = float(rhos.std(ddof=1)) if n > 1 else float("nan")
        sem = std / np.sqrt(n) if n > 1 and std == std else float("nan")
        z = mean / sem if sem == sem and sem > 0 else float("nan")
        pooled[corr_key] = {"n": n, "mean": mean, "std": std, "sem": sem, "z": z}
    return {"n_runs": len(results), "per_run": per_run, "pooled": pooled}
```

In `run()`, change the signature's `per_sample_npz: str = None` to `per_sample_npz=None` (accepting `str` or `List[str]`) and replace the existing:

```python
    retrieval_raw = None
    if per_sample_npz is not None:
        if save_raw is not None:
            result["retrieval_correlation"], retrieval_raw = correlate_polysemy_with_retrieval(
                labels, sample_ids, per_sample_npz, return_raw=True
            )
        else:
            result["retrieval_correlation"] = correlate_polysemy_with_retrieval(
                labels, sample_ids, per_sample_npz
            )
```

with:

```python
    retrieval_raw = None
    if per_sample_npz is not None:
        npz_list = [per_sample_npz] if isinstance(per_sample_npz, str) else list(per_sample_npz)
        if len(npz_list) == 1:
            if save_raw is not None:
                result["retrieval_correlation"], retrieval_raw = correlate_polysemy_with_retrieval(
                    labels, sample_ids, npz_list[0], return_raw=True
                )
            else:
                result["retrieval_correlation"] = correlate_polysemy_with_retrieval(
                    labels, sample_ids, npz_list[0]
                )
        else:
            per_run_results = [correlate_polysemy_with_retrieval(labels, sample_ids, p) for p in npz_list]
            tags = [Path(p).parent.parent.name for p in npz_list]
            result["retrieval_correlation"] = pool_cross_references(per_run_results, tags)
```

In `main()`, change `ap.add_argument("--per-sample-npz", default=None, ...)` to add `nargs="+"` so it collects one or more paths:

```python
    ap.add_argument("--per-sample-npz", default=None, nargs="+",
                    help="one or more Task 2 --dump-per-sample .npz paths for the retrieval-rank/"
                         "drift cross-reference (optional; multiple paths are pooled across runs)")
```

`main()`'s call to `run(...)` already passes `per_sample_npz=args.per_sample_npz` unchanged — argparse now hands it a list (or `None`), matching `run()`'s updated signature. Extend `main()`'s print block: after the existing single-dump print block, add an `elif` branch for the pooled case:

```python
    if "retrieval_correlation" in result:
        rc = result["retrieval_correlation"]
        if "n_runs" in rc:
            print(f"  retrieval cross-reference, pooled across {rc['n_runs']} run(s): "
                  f"{sorted(rc['per_run'].keys())}")
            for corr_key, human in (
                ("corr_is_polysemic_vs_abs_delta_rank", "|delta_rank|"),
                ("corr_is_polysemic_vs_delta_rank", "delta_rank"),
            ):
                p = rc["pooled"][corr_key]
                sig = f"  mean/SEM={p['z']:+.1f}{' *' if p['z'] == p['z'] and abs(p['z']) >= 2 else ''}" if p["n"] > 1 else ""
                print(f"    corr(is_polysemic, {human}) across runs: mean rho={p['mean']:+.3f} (n={p['n']}){sig}")
        else:
            print(f"  retrieval cross-reference (n_joined={rc['n_joined']}):")
            for lbl in ("neither", "img_only_only", "txt_only_only", "bridge"):
                if lbl in rc:
                    print(f"    {lbl}: n={rc[lbl]['n']} median|delta_rank|={rc[lbl]['median_abs_delta_rank']:.1f} "
                          f"median_delta_rank={rc[lbl]['median_delta_rank']:+.1f} "
                          f"median_drift={rc[lbl]['median_condition_drift']:.4f}")
            c1 = rc["corr_is_polysemic_vs_abs_delta_rank"]
            print(f"    corr(is_polysemic, |delta_rank|): rho={c1['rho']:+.3f} p={c1['p']:.3e}")
            c2 = rc["corr_is_polysemic_vs_delta_rank"]
            print(f"    corr(is_polysemic, delta_rank):   rho={c2['rho']:+.3f} p={c2['p']:.3e}")
```

(This replaces the plain `if "retrieval_correlation" in result:` block Task 3 already updated — the single-dump branch's body is the same code Task 3 wrote, now nested under the `else`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python scripts/analyze_polysemy_bridges.py --selftest`
Expected: `SELFTEST OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_polysemy_bridges.py
git commit -m "feat: support multi-run pooled cross-reference for Experiment 12.3"
```

---

### Task 5: Run 12.2 and 12.3 for real, update the report (NOT dispatched to a subagent — judgment-heavy, done in the main session)

**Files:**
- Modify: `docs/reports/2026-08-26_polysemy_bridge_diagnostic.md` (add "Experiment 12.2" and "Experiment 12.3" sections; narrow the existing Interpretation section's overclaim per the brainstorm memo's point 1)
- Create (if `--out-fig` is used): `docs/reports/assets/training_trajectory/i2t_gap_trajectory.png`

**Steps (not TDD — this is real-data execution and report writing):**

- [ ] **Step 1:** Run Task 1/2's script for real:
  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
  python scripts/analyze_training_trajectory.py \
    --tag condition-freeze-ablation-redcaps_150k \
    --pred-coupled-tag pred-stopgrad-ablation-redcaps_150k \
    --out-fig docs/reports/assets/training_trajectory/i2t_gap_trajectory.png
  ```
  Read off whether the i2t gap is present from the earliest logged epoch (flat) or grows over the run (late-onset).

- [ ] **Step 2:** Generate the 5 additional per-sample dumps (11.1's seed 2/3 trained-vs-frozen, 11.3's seed 1/2/3 pred_coupled-vs-frozen), reusing already-completed run dirs under `res/CoSiR_condition_freeze_ablation/redcaps_150k/`:
  ```bash
  python scripts/analyze_condition_retrieval_correlation.py --dump-per-sample --pair \
    res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_171558_CoSiR_Experiment \
    res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_163307_CoSiR_Experiment   # trained seed2
  python scripts/analyze_condition_retrieval_correlation.py --dump-per-sample --pair \
    res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_172950_CoSiR_Experiment \
    res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_164733_CoSiR_Experiment   # trained seed3
  python scripts/analyze_condition_retrieval_correlation.py --dump-per-sample --pair \
    res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_170212_CoSiR_Experiment \
    res/CoSiR_condition_freeze_ablation/redcaps_150k/20260826_100355_CoSiR_Experiment   # pred_coupled seed1
  python scripts/analyze_condition_retrieval_correlation.py --dump-per-sample --pair \
    res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_171558_CoSiR_Experiment \
    res/CoSiR_condition_freeze_ablation/redcaps_150k/20260826_102258_CoSiR_Experiment   # pred_coupled seed2
  python scripts/analyze_condition_retrieval_correlation.py --dump-per-sample --pair \
    res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_172950_CoSiR_Experiment \
    res/CoSiR_condition_freeze_ablation/redcaps_150k/20260826_103723_CoSiR_Experiment   # pred_coupled seed3
  ```
  (Seed 1 trained-vs-frozen's dump already exists from Experiment 12's own run — reuse it, do not regenerate.)

- [ ] **Step 3:** Run the pooled cross-reference across all 6 dumps (seed1/2/3 trained + seed1/2/3 pred_coupled, each vs. its matched frozen seed):
  ```bash
  python scripts/analyze_polysemy_bridges.py --n-bridge-sample 5000 --device cuda \
    --per-sample-npz \
      res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_161846_CoSiR_Experiment/condition_geometry/per_sample_retrieval_correlation.npz \
      res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_163307_CoSiR_Experiment/condition_geometry/per_sample_retrieval_correlation.npz \
      res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_164733_CoSiR_Experiment/condition_geometry/per_sample_retrieval_correlation.npz \
      res/CoSiR_condition_freeze_ablation/redcaps_150k/20260826_100355_CoSiR_Experiment/condition_geometry/per_sample_retrieval_correlation.npz \
      res/CoSiR_condition_freeze_ablation/redcaps_150k/20260826_102258_CoSiR_Experiment/condition_geometry/per_sample_retrieval_correlation.npz \
      res/CoSiR_condition_freeze_ablation/redcaps_150k/20260826_103723_CoSiR_Experiment/condition_geometry/per_sample_retrieval_correlation.npz \
    --out res/CoSiR_condition_freeze_ablation/redcaps_150k/polysemy_bridges_pooled_12_3.json
  ```

- [ ] **Step 4:** Write the "Experiment 12.2" and "Experiment 12.3" report sections (TL;DR, Method, Results, Interpretation, Caveats, Reproduction — matching the existing report's section style for Experiment 12), using the real printed output from Steps 1 and 3. Rewrite Experiment 12's own Interpretation section per the brainstorm memo's first point: state plainly that the original null was measured against one specific, already-flagged-as-underperforming training recipe at its final epoch, and that 12.2/12.3 either strengthen or narrow that claim (write whichever the real data supports — do not presuppose the direction).

- [ ] **Step 5:** Commit the report update and any new figure.

```bash
git add docs/reports/2026-08-26_polysemy_bridge_diagnostic.md docs/reports/assets/training_trajectory/
git commit -m "results: add Experiment 12.2/12.3 (training-trajectory audit + cross-recipe replication)"
```

## Self-review

**Spec coverage:** Task 1-2 implement 12.2's `analyze_training_trajectory.py` (spec: new script, per-epoch history, gap characterization, figure) in full. Task 3-4 implement 12.3's signed-`delta_rank` addition and multi-run pooling in full. Task 5 covers both experiments' real-data execution and report write-up. 12.4 is intentionally not planned here — the spec marks it tentative, gated on Task 5's own findings.

**Placeholder scan:** No TBD/TODO; every step has runnable code or an exact command.

**Type consistency:** `run()`'s `per_sample_npz` parameter is documented as accepting `str` or `List[str]` consistently between Task 4's Interfaces block and its Step 3 code. `pool_cross_references`'s return shape (`n_runs`/`per_run`/`pooled`) matches what Task 5's Step 3 command expects to read from `--out`'s JSON.
