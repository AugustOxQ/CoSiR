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
        if tag and tag not in (run.tags or []):
            continue
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
    if skipped_unfinished:
        print(f"  ({skipped_unfinished} non-finished run(s) excluded from analysis)")
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
    print(f"  {len(df)} run(s); arms present: {sorted(df['arm'].unique())}.")

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
        print("    Compare mean delta against the noise floor (~0.1-0.7 R1, NOT zero) - "
              "see docs/reports/2026-06-24_buddy_progress_report.md S8a.")
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
    main()
