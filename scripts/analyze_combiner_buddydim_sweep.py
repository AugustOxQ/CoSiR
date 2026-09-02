"""
Analysis for the combiner-architecture ablation Phase 5 buddy_dim sweep
(.planning/2026-09-01-combiner-architecture-ablation/task_plan.md): for the Phase 4 winner
(lowrank, rank=16), does buddy_dim change retrieval at a fixed operating point (K=30, alpha=0.5)?

Reads W&B runs from the 'combiner buddydim sweep' group
(scripts/run_combiner_buddydim_sweep.sh), groups finished runs by buddy_dim (model.embedding_dim),
and reports test_oracle/test_pre_diff t2i/i2t R1 as mean +/- std across seeds, plus a paired
delta from the dim=16 anchor (the dimension used throughout Phase 3/4).

Usage
-----
  python scripts/analyze_combiner_buddydim_sweep.py
  python scripts/analyze_combiner_buddydim_sweep.py --selftest   # offline check, no W&B call

Requires: wandb, pandas, numpy (all already deps).
"""
import argparse

import numpy as np
import pandas as pd

METRICS = [
    ("test_oracle t2i R1", "test_oracle/t2i_R1"),
    ("test_oracle i2t R1", "test_oracle/i2t_R1"),
    ("test_pre_diff t2i R1", "test_pre_diff/t2i_R1"),
    ("test_pre_diff i2t R1", "test_pre_diff/i2t_R1"),
]
DIM_ANCHOR = 16


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


def value_or_nan(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def fetch(entity, project, group):
    import wandb

    api = wandb.Api()
    rows = []
    skipped_unfinished = 0
    for run in api.runs(f"{entity}/{project}", filters={"group": group}):
        if run.state != "finished":
            skipped_unfinished += 1
            continue
        cfg, summ = run.config, run.summary
        dim = cget(cfg, ("model", "embedding_dim"))
        try:
            dim = int(dim)
        except (TypeError, ValueError):
            continue
        row = {"run_id": run.id, "buddy_dim": dim,
               "seed": value_or_nan(cget(cfg, ("seed",))),
               "epoch": value_or_nan(sget(summ, "epoch"))}
        for _, metric in METRICS:
            row[metric] = value_or_nan(sget(summ, metric))
        rows.append(row)
    if skipped_unfinished:
        print(f"  ({skipped_unfinished} non-finished run(s) in this group excluded from analysis)")
    df = pd.DataFrame(rows)
    return _drop_shorter_duplicate_runs(df)


def _drop_shorter_duplicate_runs(df):
    """Keep only the most-complete run per (buddy_dim, seed) - see analyze_combiner_architecture_
    smoke.py for why (a SMOKE=1 sanity run can share group/tag/seed with the real run)."""
    if df.empty:
        return df
    max_epoch = df.groupby(["buddy_dim", "seed"])["epoch"].transform("max")
    dropped = df[df["epoch"] < max_epoch]
    if len(dropped):
        print(f"  ({len(dropped)} shorter duplicate run(s) excluded): {dropped['run_id'].tolist()}")
    return df[df["epoch"] >= max_epoch].reset_index(drop=True)


def summarize(values):
    """Return mean, sample std, SEM, and mean/SEM for a numeric iterable."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return {"n": 0}
    mean = arr.mean()
    std = arr.std(ddof=1) if n > 1 else float("nan")
    sem = std / np.sqrt(n) if n > 1 else float("nan")
    z = mean / sem if n > 1 and sem > 0 else float("nan")
    return {"n": n, "mean": mean, "std": std, "sem": sem, "z": z}


def cell_values(cell, metric):
    """One final metric per seed; .max() mirrors existing ablation analyzers on reruns."""
    return cell.groupby("seed")[metric].max().dropna()


def paired_delta(df, dim, metric):
    """Return dim-minus-anchor deltas paired within seed."""
    anchor = cell_values(df[df["buddy_dim"] == DIM_ANCHOR], metric)
    treatment = cell_values(df[df["buddy_dim"] == dim], metric)
    paired = pd.concat([anchor.rename("anchor"), treatment.rename("treatment")], axis=1).dropna()
    return paired["treatment"] - paired["anchor"]


def format_summary(summary):
    if summary["n"] == 0:
        return "n=0"
    spread = f" +/- {summary['std']:.2f}" if summary["n"] > 1 else ""
    ratio = f" mean/SEM={summary['z']:+.1f}" if summary["n"] > 1 else ""
    return f"n={summary['n']} mean={summary['mean']:.2f}{spread}{ratio}"


def report_metric(df, label, metric):
    print(f"\n  --- {label} ({metric}) ---")
    for dim, cell in df.groupby("buddy_dim", sort=True):
        print(f"  buddy_dim={dim:<3}  {format_summary(summarize(cell_values(cell, metric)))}")
    if DIM_ANCHOR not in df["buddy_dim"].unique():
        print(f"\n  (no buddy_dim={DIM_ANCHOR} anchor present - skipping deltas)")
        return
    print(f"\n  Paired delta relative to buddy_dim={DIM_ANCHOR} (within seed):")
    for dim in sorted(df["buddy_dim"].unique()):
        if dim == DIM_ANCHOR:
            continue
        print(f"  buddy_dim={dim:<3}  {format_summary(summarize(paired_delta(df, dim, metric)))}")


def analyze(entity, project, group):
    print(f"\n{'=' * 78}\nCombiner architecture ablation Phase 5 - buddy_dim sweep  group='{group}'"
          f"\n{'=' * 78}")
    df = fetch(entity, project, group)
    if df.empty:
        print("  (no runs found - check --entity/--project/--group)")
        return
    print(f"  {len(df)} finished run(s); buddy_dims present: {sorted(df['buddy_dim'].unique())}")
    for label, metric in METRICS:
        report_metric(df, label, metric)


def _selftest():
    """Offline arithmetic check for per-cell and paired dim-minus-16 summaries."""
    df = pd.DataFrame([
        {"buddy_dim": 16, "seed": 1, "metric": 50.0},
        {"buddy_dim": 16, "seed": 2, "metric": 48.0},
        {"buddy_dim": 32, "seed": 1, "metric": 52.0},
        {"buddy_dim": 32, "seed": 2, "metric": 51.0},
    ])
    values = cell_values(df[df["buddy_dim"] == 32], "metric")
    summary = summarize(values)
    assert summary["n"] == 2
    assert abs(summary["mean"] - 51.5) < 1e-9
    deltas = paired_delta(df, 32, "metric")
    assert sorted(deltas.tolist()) == [2.0, 3.0]
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="augustoxq")
    ap.add_argument("--project", default="cosir_image")
    ap.add_argument("--group", default="combiner buddydim sweep")
    ap.add_argument("--selftest", action="store_true", help="offline arithmetic check, no W&B call")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    analyze(args.entity, args.project, args.group)


if __name__ == "__main__":
    main()
