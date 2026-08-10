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
