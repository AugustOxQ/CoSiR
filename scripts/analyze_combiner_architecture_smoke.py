"""
Analysis for the combiner-architecture ablation Phase 4 smoke sweep
(.planning/2026-09-01-combiner-architecture-ablation/task_plan.md): does the fusion-module
family change retrieval at a fixed buddy-init operating point (K=30, alpha=0.5, buddy_dim=16)?

Reads W&B runs from the 'combiner architecture smoke' group
(scripts/run_combiner_architecture_smoke.sh), groups finished runs by combiner_type, and
reports test_oracle/test_pre_diff t2i/i2t R1 as mean +/- std across seeds. For every
combiner_type other than 'legacy', also reports the delta from 'legacy' (paired within seed
where possible, else an unpaired mean difference).

Usage
-----
  python scripts/analyze_combiner_architecture_smoke.py
  python scripts/analyze_combiner_architecture_smoke.py --selftest   # offline check, no W&B call

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
BASELINE = "legacy"


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
        combiner_type = cget(cfg, ("model", "combiner_type"))
        if not combiner_type:
            continue
        row = {"run_id": run.id, "combiner_type": combiner_type,
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
    """Keep only the most-complete run per (combiner_type, seed).

    A SMOKE=1 pipeline-sanity run shares the same group/tag/seed as its full run (e.g. an
    ad-hoc `SMOKE=1 ... combiner_type=lowrank` check before the real sweep) and would
    otherwise be blended in by .max() per metric, mixing a 2-epoch immature run's numbers
    into the same cell as the real 30-epoch run. Drop rows that aren't the max-epoch row
    within their (combiner_type, seed) group.
    """
    if df.empty:
        return df
    max_epoch = df.groupby(["combiner_type", "seed"])["epoch"].transform("max")
    dropped = df[df["epoch"] < max_epoch]
    if len(dropped):
        print(f"  ({len(dropped)} shorter duplicate run(s) excluded, e.g. a SMOKE=1 sanity "
              f"check under the same tag/seed): {dropped['run_id'].tolist()}")
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


def paired_or_unpaired_delta(df, combiner_type, metric):
    """Delta from the legacy baseline: paired within seed if both sides share seeds,
    else an unpaired difference of per-arm cell values (smoke sweeps may run 1 seed/arm)."""
    baseline = cell_values(df[df["combiner_type"] == BASELINE], metric)
    treatment = cell_values(df[df["combiner_type"] == combiner_type], metric)
    paired = pd.concat([baseline.rename("baseline"), treatment.rename("treatment")], axis=1).dropna()
    if len(paired) > 0:
        return paired["treatment"] - paired["baseline"], True
    if len(baseline) == 0 or len(treatment) == 0:
        return pd.Series(dtype=float), False
    return pd.Series([treatment.mean() - baseline.mean()]), False


def format_summary(summary):
    if summary["n"] == 0:
        return "n=0"
    spread = f" +/- {summary['std']:.2f}" if summary["n"] > 1 else ""
    ratio = f" mean/SEM={summary['z']:+.1f}" if summary["n"] > 1 else ""
    return f"n={summary['n']} mean={summary['mean']:.2f}{spread}{ratio}"


def report_metric(df, label, metric):
    print(f"\n  --- {label} ({metric}) ---")
    for combiner_type, cell in df.groupby("combiner_type", sort=True):
        print(f"  {combiner_type:>18}  {format_summary(summarize(cell_values(cell, metric)))}")
    if BASELINE not in df["combiner_type"].unique():
        print(f"\n  (no '{BASELINE}' baseline present - skipping deltas)")
        return
    print(f"\n  Delta relative to '{BASELINE}':")
    for combiner_type in sorted(df["combiner_type"].unique()):
        if combiner_type == BASELINE:
            continue
        deltas, paired = paired_or_unpaired_delta(df, combiner_type, metric)
        kind = "paired" if paired else "unpaired, single point"
        print(f"  {combiner_type:>18}  {format_summary(summarize(deltas))}  ({kind})")


def analyze(entity, project, group):
    print(f"\n{'=' * 78}\nCombiner architecture ablation Phase 4 - smoke sweep  group='{group}'"
          f"\n{'=' * 78}")
    df = fetch(entity, project, group)
    if df.empty:
        print("  (no runs found - check --entity/--project/--group)")
        return
    print(f"  {len(df)} finished run(s); combiner_types present: {sorted(df['combiner_type'].unique())}")
    for label, metric in METRICS:
        report_metric(df, label, metric)


def _selftest():
    """Offline arithmetic check for per-arm and baseline-relative summaries."""
    df = pd.DataFrame([
        {"combiner_type": "legacy", "seed": 1, "metric": 50.0},
        {"combiner_type": "lowrank", "seed": 1, "metric": 53.0},
        {"combiner_type": "film", "seed": 1, "metric": 48.0},
    ])
    values = cell_values(df[df["combiner_type"] == "lowrank"], "metric")
    summary = summarize(values)
    assert summary["n"] == 1
    assert summary["mean"] == 53.0
    deltas, paired = paired_or_unpaired_delta(df, "lowrank", "metric")
    assert paired is True
    assert deltas.tolist() == [3.0]
    deltas, paired = paired_or_unpaired_delta(df, "film", "metric")
    assert paired is True
    assert deltas.tolist() == [-2.0]
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="augustoxq")
    ap.add_argument("--project", default="cosir_image")
    ap.add_argument("--group", default="combiner architecture smoke")
    ap.add_argument("--selftest", action="store_true", help="offline arithmetic check, no W&B call")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    analyze(args.entity, args.project, args.group)


if __name__ == "__main__":
    main()
