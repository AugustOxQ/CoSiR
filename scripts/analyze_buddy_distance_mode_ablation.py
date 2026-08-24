"""
Paired analysis for Experiment 10 (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md
S4): does modality-provenance-aware distance mixing ("typed") beat the current fixed-alpha
blend ("blend") on retrieval, and does it narrow C6's gap to raw CLIP (test_pre_diff)?

Reads wandb runs from the 'buddy distance-mode ablation' group
(scripts/run_buddy_distance_mode_ablation.sh), pairs typed vs blend WITHIN each seed, and
reports mean delta +/- std and mean/SEM (spec S5) for test_oracle (retrieval, higher is
better) AND test_pre_diff (ours - CLIP, already logged automatically by every eval call
in src/eval/pipeline.py -- LESS NEGATIVE / higher delta = narrower gap to CLIP).

Usage
-----
  python scripts/analyze_buddy_distance_mode_ablation.py --tag buddy-distance-mode-ablation-redcaps_150k
  python scripts/analyze_buddy_distance_mode_ablation.py --selftest   # offline check, no wandb call

Requires: wandb, pandas, numpy (all already deps).
"""
import argparse

import numpy as np
import pandas as pd

METRICS = [
    "test_oracle/t2i_R1", "test_oracle/i2t_R1",
    "test_pre_diff/t2i_R1", "test_pre_diff/i2t_R1",
    "test_raw/t2i_R1", "test_raw/i2t_R1",
]
BASELINE = "blend"
TREATMENT = "typed"
CELL = [("distance_mode", ("train", "buddies", "distance_mode")), ("seed", ("seed",))]


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
        if run.state != "finished":
            skipped_unfinished += 1
            continue
        cfg, summ = run.config, run.summary
        mode = cget(cfg, ("train", "buddies", "distance_mode"))
        if mode not in (BASELINE, TREATMENT):
            continue
        row = {"run_id": run.id, "distance_mode": mode}
        for metric in METRICS:
            v = sget(summ, metric)
            row[metric] = float(v) if not (isinstance(v, float) and np.isnan(v)) else np.nan
        for cname, cpath in CELL:
            cv = cget(cfg, cpath)
            row[cname] = cv
        rows.append(row)
    if skipped_unfinished:
        print(f"  ({skipped_unfinished} non-finished run(s) under this group excluded from analysis)")
    return pd.DataFrame(rows)


def compute_paired_deltas(df, metric):
    """Pair BASELINE vs TREATMENT within each seed. Returns list of (seed, delta) where
    delta = treatment - baseline. Pure function - no wandb, no I/O."""
    deltas = []
    for seed, cell in df.groupby("seed"):
        by_mode = cell.groupby("distance_mode")[metric].max()
        if BASELINE not in by_mode.index or TREATMENT not in by_mode.index:
            continue
        b, t = by_mode[BASELINE], by_mode[TREATMENT]
        if np.isnan(b) or np.isnan(t):
            continue
        deltas.append((seed, t - b))
    return deltas


def summarize(deltas):
    """mean, std, sem, z=mean/sem, win-rate - matches this project's existing convention."""
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
    print(f"\n{'='*78}\nExperiment 10 - buddy distance-mode ablation  group='{group}'"
          + (f"  tag='{tag}'" if tag else "") + f"\n{'='*78}")
    df = fetch(entity, project, group, tag=tag)
    if df.empty:
        print("  (no runs found - check --entity/--project/--tag)")
        return
    modes = sorted(df["distance_mode"].unique())
    print(f"  {len(df)} finished run(s); modes present: {modes}")

    for label, metric in [
        ("retrieval t2i R1", "test_oracle/t2i_R1"),
        ("retrieval i2t R1", "test_oracle/i2t_R1"),
        ("gap-to-CLIP t2i (ours-CLIP)", "test_pre_diff/t2i_R1"),
        ("gap-to-CLIP i2t (ours-CLIP)", "test_pre_diff/i2t_R1"),
    ]:
        deltas = compute_paired_deltas(df, metric)
        s = summarize(deltas)
        if s["n"] == 0:
            print(f"\n  {label}: (no paired cells with both {BASELINE} and {TREATMENT} present)")
            continue
        sig = f"  mean/SEM={s['z']:+.1f}{' *' if not np.isnan(s['z']) and abs(s['z']) >= 2 else ''}" if s["n"] > 1 else ""
        spread = f" +/- {s['std']:.2f}" if s["n"] > 1 else ""
        print(f"\n  {label}: typed - blend, over {s['n']} seed(s), "
              f"{TREATMENT} wins {s['wins']}/{s['n']} "
              f"(mean delta = {s['mean']:+.2f}{spread}){sig}")

    for metric in ("test_raw/t2i_R1", "test_raw/i2t_R1"):
        vals = df[metric].dropna().unique()
        print(f"\n  {metric} distinct values across all runs: {sorted(vals)} "
              f"(sanity check - should be a single value, same frozen backbone/test set)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="augustoxq")
    ap.add_argument("--project", default="cosir_image")
    ap.add_argument("--group", default="buddy distance-mode ablation")
    ap.add_argument(
        "--tag",
        default="buddy-distance-mode-ablation-redcaps_150k",
        help="only include runs carrying this wandb tag (default excludes -smoke-tagged runs "
             "from run_buddy_distance_mode_ablation.sh's SMOKE=1 mode, which share this same "
             "wandb group/results dir and would otherwise silently mix in; pass --tag '' to "
             "disable filtering, or another value to target a different sweep)",
    )
    ap.add_argument("--selftest", action="store_true", help="offline arithmetic check, no wandb call")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    analyze(args.entity, args.project, args.group, tag=args.tag)


def _selftest():
    """Offline arithmetic check - no wandb call. Verifies compute_paired_deltas/summarize
    against hand-computed numbers, for both a retrieval metric and the gap-to-CLIP metric,
    before ever touching real run data."""
    df = pd.DataFrame([
        {"distance_mode": "blend", "seed": 1, "test_oracle/t2i_R1": 50.0, "test_pre_diff/t2i_R1": -10.0},
        {"distance_mode": "typed", "seed": 1, "test_oracle/t2i_R1": 52.0, "test_pre_diff/t2i_R1": -8.0},
        {"distance_mode": "blend", "seed": 2, "test_oracle/t2i_R1": 48.0, "test_pre_diff/t2i_R1": -11.0},
        {"distance_mode": "typed", "seed": 2, "test_oracle/t2i_R1": 51.0, "test_pre_diff/t2i_R1": -9.0},
        {"distance_mode": "blend", "seed": 3, "test_oracle/t2i_R1": 49.0, "test_pre_diff/t2i_R1": -10.5},
        {"distance_mode": "typed", "seed": 3, "test_oracle/t2i_R1": 49.0, "test_pre_diff/t2i_R1": -9.5},
    ])
    deltas = compute_paired_deltas(df, "test_oracle/t2i_R1")
    assert len(deltas) == 3, f"expected 3 paired cells, got {len(deltas)}"
    got = sorted(d for _, d in deltas)
    want = [0.0, 2.0, 3.0]
    assert got == want, f"expected deltas {want}, got {got}"
    s = summarize(deltas)
    assert s["n"] == 3
    assert abs(s["mean"] - (5.0 / 3)) < 1e-9

    pre_deltas = compute_paired_deltas(df, "test_pre_diff/t2i_R1")
    pre_s = summarize(pre_deltas)
    assert pre_s["n"] == 3
    assert pre_s["mean"] > 0, (
        "expected a positive test_pre_diff delta in this toy example "
        "(typed less negative than blend -> narrower gap to CLIP)"
    )
    print("SELFTEST OK")


if __name__ == "__main__":
    main()
