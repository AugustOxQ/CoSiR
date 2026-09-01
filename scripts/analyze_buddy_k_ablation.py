"""
Analysis for Experiment 16.2 Stage B (spec docs/superpowers/specs/
2026-08-04-buddy-publication-plan-design.md §4): how does buddy-init K affect retrieval
across RedCaps scale, at the matched trained-buddies operating point?

Reads W&B runs from the 'buddy K ablation' group (scripts/run_buddy_k_ablation.sh), groups
finished runs by (dataset scale, K), and reports test_oracle/test_pre_diff t2i/i2t R1 as
mean +/- std and mean/SEM across seeds. For every K other than 30, it also reports a paired,
within-seed delta from that scale's K=30 anchor (K - 30), with mean +/- std and mean/SEM.

Usage
-----
  python scripts/analyze_buddy_k_ablation.py --tag buddy-k-ablation-redcaps_300k_diverse
  python scripts/analyze_buddy_k_ablation.py --tag buddy-k-ablation-redcaps_500k_diverse
  python scripts/analyze_buddy_k_ablation.py --selftest   # offline arithmetic check, no W&B call

Requires: wandb, pandas, numpy (all already deps).
"""
import argparse
import re

import numpy as np
import pandas as pd

METRICS = [
    ("test_oracle t2i R1", "test_oracle/t2i_R1"),
    ("test_oracle i2t R1", "test_oracle/i2t_R1"),
    ("test_pre_diff t2i R1", "test_pre_diff/t2i_R1"),
    ("test_pre_diff i2t R1", "test_pre_diff/i2t_R1"),
]
K_ANCHOR = 30


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


def dataset_scale(cfg):
    """Return the scale label from the scale-specific train annotation/store, if present."""
    values = [
        str(cget(cfg, ("data", "train_annotation_path"), "")),
        str(cget(cfg, ("featuremanager", "storage_dir"), "")),
        str(cget(cfg, ("dataset",), "")),
    ]
    for value in values:
        match = re.search(r"redcaps_(150k|300k|500k)(?:_diverse)?", value)
        if match:
            return match.group(0)
        match = re.search(r"redcaps_train_(150000|300000|500000)", value)
        if match:
            return f"redcaps_{int(match.group(1)) // 1000}k"
    return values[-1] or "unknown"


def value_or_nan(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def fetch(entity, project, group, tag=None):
    import wandb

    api = wandb.Api()
    rows = []
    skipped_unfinished = 0
    for run in api.runs(f"{entity}/{project}", filters={"group": group}):
        if tag and tag not in (run.tags or []):
            continue
        if run.state != "finished":
            skipped_unfinished += 1
            continue
        cfg, summ = run.config, run.summary
        k = cget(cfg, ("train", "buddies", "k"))
        try:
            k = int(k)
        except (TypeError, ValueError):
            continue
        row = {"run_id": run.id, "scale": dataset_scale(cfg), "k": k,
               "seed": value_or_nan(cget(cfg, ("seed",)))}
        for _, metric in METRICS:
            row[metric] = value_or_nan(sget(summ, metric))
        rows.append(row)
    if skipped_unfinished:
        print(f"  ({skipped_unfinished} non-finished run(s) under this tag excluded from analysis)")
    return pd.DataFrame(rows)


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


def paired_deltas(df, scale, k, metric):
    """Return K-minus-anchor deltas paired within seed for one scale and metric."""
    anchor = cell_values(df[(df["scale"] == scale) & (df["k"] == K_ANCHOR)], metric)
    treatment = cell_values(df[(df["scale"] == scale) & (df["k"] == k)], metric)
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
    for (scale, k), cell in df.groupby(["scale", "k"], sort=True):
        print(f"  {scale:>24}  K={k:<3}  {format_summary(summarize(cell_values(cell, metric)))}")
    print(f"\n  Paired delta relative to K={K_ANCHOR} (K - {K_ANCHOR}, within seed):")
    for scale in sorted(df["scale"].unique()):
        ks = sorted(df.loc[df["scale"] == scale, "k"].unique())
        if K_ANCHOR not in ks:
            print(f"  {scale:>24}  (no K={K_ANCHOR} anchor)")
            continue
        for k in ks:
            if k == K_ANCHOR:
                continue
            print(f"  {scale:>24}  K={k:<3}  {format_summary(summarize(paired_deltas(df, scale, k, metric)))}")


def analyze(entity, project, group, tag):
    print(f"\n{'=' * 78}\nExperiment 16.2 Stage B - buddy K ablation  group='{group}'"
          + (f"  tag='{tag}'" if tag else "") + f"\n{'=' * 78}")
    df = fetch(entity, project, group, tag=tag)
    if df.empty:
        print("  (no runs found - check --entity/--project/--tag)")
        return
    print(f"  {len(df)} finished run(s); scales present: {sorted(df['scale'].unique())}")
    for label, metric in METRICS:
        report_metric(df, label, metric)


def _selftest():
    """Offline arithmetic check for per-cell and paired K-minus-30 summaries."""
    df = pd.DataFrame([
        {"scale": "redcaps_300k_diverse", "k": 30, "seed": 1, "metric": 50.0},
        {"scale": "redcaps_300k_diverse", "k": 30, "seed": 2, "metric": 48.0},
        {"scale": "redcaps_300k_diverse", "k": 30, "seed": 3, "metric": 49.0},
        {"scale": "redcaps_300k_diverse", "k": 35, "seed": 1, "metric": 52.0},
        {"scale": "redcaps_300k_diverse", "k": 35, "seed": 2, "metric": 51.0},
        {"scale": "redcaps_300k_diverse", "k": 35, "seed": 3, "metric": 49.0},
    ])
    values = cell_values(df[df["k"] == 35], "metric")
    summary = summarize(values)
    assert summary["n"] == 3
    assert abs(summary["mean"] - (152.0 / 3)) < 1e-9
    deltas = paired_deltas(df, "redcaps_300k_diverse", 35, "metric")
    assert sorted(deltas.tolist()) == [0.0, 2.0, 3.0]
    delta_summary = summarize(deltas)
    assert abs(delta_summary["mean"] - (5.0 / 3)) < 1e-9
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="augustoxq")
    ap.add_argument("--project", default="cosir_image")
    ap.add_argument("--group", default="buddy K ablation")
    ap.add_argument(
        "--tag",
        default="buddy-k-ablation-redcaps_150k",
        help="only include runs carrying this W&B tag; pass --tag '' to disable filtering",
    )
    ap.add_argument("--selftest", action="store_true", help="offline arithmetic check, no W&B call")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    analyze(args.entity, args.project, args.group, tag=args.tag)


if __name__ == "__main__":
    main()
