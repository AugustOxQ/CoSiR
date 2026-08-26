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
train_buddy_diag/drift_from_init and final-step train_loss/loss_pred as free diagnostic
context (does coupling shrink drift toward frozen's ~0, and does the predictor's own
reconstruction loss converge or diverge).

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
DRIFT = "train_buddy_diag/drift_from_init"
PRED_LOSS = "train_loss/loss_pred"
METRICS = [T2I_ORACLE, I2T_ORACLE, T2I_PREDIFF, I2T_PREDIFF]
TREATMENT = "pred_coupled"
BASELINES = ["trained", "frozen"]


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
    skipped_unfinished = 0
    for run in api.runs(f"{entity}/{project}", filters={"group": group}):
        # A crashed/killed/still-running run can still have partial summary metrics
        # (e.g. logged at epoch 0 before dying) that look like real numbers to
        # compute_paired_deltas's per-cell .max(). Excluding anything but a clean
        # finish is the only safe way to dedupe a cell that got re-run after a
        # crash — .max() alone can silently prefer the crashed run's number over
        # the real, fully-trained one (this happened: a driver-outage-killed run
        # logged epoch-0 metrics that were numerically higher, by epoch-0 noise,
        # than the finished run's converged ones).
        if run.state != "finished":
            skipped_unfinished += 1
            continue
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
    if skipped_unfinished:
        print(f"  ({skipped_unfinished} non-finished run(s) excluded from analysis)")
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
    print(f"  {len(df)} run(s); arms present: {sorted(df['arm'].unique())}.")

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
    main()
