"""
Paired analysis for Experiment 8 (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md
S4): does the (vision, text) encoder pair used to build the buddy graph/init matter for
downstream retrieval, holding the frozen CLIP backbone and all training-time buddy terms off?

Reads wandb runs from the 'buddy-init encoder-pair ablation' group
(scripts/run_buddy_init_encoder_ablation.sh), pairs every non-baseline encoder pair against
the clip_img:clip_txt baseline WITHIN each seed, and reports mean delta +/- std and mean/SEM
(spec S5). Also joins each pair's C3 cross-VLM survival rate (mean off-diagonal chance-lift
of its union graph E against the other 15 cells, from
docs/reports/assets/buddy_cross_vlm/grid_agreement.json) against its measured retrieval delta.

Usage
-----
  python scripts/analyze_buddy_init_encoder_ablation.py --tag buddy-encoder-ablation-redcaps_150k
  python scripts/analyze_buddy_init_encoder_ablation.py --selftest   # offline check, no wandb call

Requires: wandb, pandas, numpy (all already deps).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

T2I = "test_oracle/t2i_R1"
I2T = "test_oracle/i2t_R1"
BASELINE = "clip_img:clip_txt"
CELL = [("encoder_pair", ("train", "buddies", "encoder_pair")),
        ("seed", ("seed",))]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SURVIVAL_JSON = os.path.join(ROOT, "docs/reports/assets/buddy_cross_vlm/grid_agreement.json")


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
        pair = cget(cfg, ("train", "buddies", "encoder_pair"))
        if not pair:
            continue
        row = {
            "run_id": run.id,
            "state": run.state,
            "encoder_pair": pair,
            T2I: float(sget(summ, T2I)) if not np.isnan(sget(summ, T2I)) else np.nan,
            I2T: float(sget(summ, I2T)) if not np.isnan(sget(summ, I2T)) else np.nan,
        }
        for cname, cpath in CELL:
            cv = cget(cfg, cpath)
            row[cname] = cv
        rows.append(row)
    if skipped_unfinished:
        print(f"  ({skipped_unfinished} non-finished run(s) under this group excluded from analysis)")
    return pd.DataFrame(rows)


def compute_paired_deltas(df, metric, treatment_pair):
    """Pair BASELINE vs treatment_pair within each seed. Returns list of (seed, delta) where
    delta = treatment - baseline. Pure function - no wandb, no I/O."""
    deltas = []
    for seed, cell in df.groupby("seed"):
        by_pair = cell.groupby("encoder_pair")[metric].max()
        if BASELINE not in by_pair.index or treatment_pair not in by_pair.index:
            continue
        b, t = by_pair[BASELINE], by_pair[treatment_pair]
        if np.isnan(b) or np.isnan(t):
            continue
        deltas.append((seed, t - b))
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


def survival_rate_per_cell(grid: dict) -> dict:
    """Mean off-diagonal chance-lift of each cell's union graph E against the other 15 cells
    (docs/reports/assets/buddy_cross_vlm/grid_agreement.json's 'E'.'lift' matrix), keyed by
    cell name (e.g. 'dinov2xbge'). Higher = more cross-VLM-consensus buddy structure."""
    cells = grid["cells"]
    lift = np.asarray(grid["E"]["lift"], dtype=float)
    n = len(cells)
    rates = {}
    for i, name in enumerate(cells):
        off = [lift[i, j] for j in range(n) if j != i]
        rates[name] = float(np.mean(off))
    return rates


def to_pair_key(encoder_pair: str) -> str:
    """'dinov2:bge' -> 'dinov2xbge' to match grid_agreement.json's cell-name convention."""
    v, t = encoder_pair.split(":")
    return f"{v}x{t}"


def analyze(entity, project, group, tag=None):
    print(f"\n{'='*78}\nExperiment 8 - buddy-init encoder-pair ablation  group='{group}'"
          + (f"  tag='{tag}'" if tag else "") + f"\n{'='*78}")
    df = fetch(entity, project, group, tag=tag)
    if df.empty:
        print("  (no runs found - check --entity/--project/--tag)")
        return
    n_unfinished = (df["state"] != "finished").sum()
    pairs = sorted(df["encoder_pair"].unique())
    print(f"  {len(df)} run(s); pairs present: {pairs}."
          + (f"  [{n_unfinished} not finished -> best-so-far]" if n_unfinished else ""))

    with open(SURVIVAL_JSON) as f:
        grid = json.load(f)
    surv = survival_rate_per_cell(grid)

    summary_rows = []
    for metric in (T2I, I2T):
        print(f"\n  --- {metric} (vs. {BASELINE}) ---")
        for pair in pairs:
            if pair == BASELINE:
                continue
            deltas = compute_paired_deltas(df, metric, pair)
            s = summarize(deltas)
            if s["n"] == 0:
                continue
            sig = f"  mean/SEM={s['z']:+.1f}{' *' if not np.isnan(s['z']) and abs(s['z']) >= 2 else ''}" if s["n"] > 1 else ""
            print(f"    {pair:>20}: mean delta = {s['mean']:+.2f} (n={s['n']}, wins={s['wins']}){sig}")
            summary_rows.append({"metric": metric, "encoder_pair": pair, "mean_delta": s["mean"],
                                   "z": s.get("z", np.nan), "survival_rate": surv.get(to_pair_key(pair), np.nan)})

    corr_df = pd.DataFrame(summary_rows)
    for metric in (T2I, I2T):
        sub = corr_df[corr_df["metric"] == metric].dropna(subset=["mean_delta", "survival_rate"])
        if len(sub) >= 3:
            r = np.corrcoef(sub["mean_delta"], sub["survival_rate"])[0, 1]
            print(f"\n  Correlation(mean delta [{metric}], C3 survival rate) over {len(sub)} pairs: r={r:+.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="augustoxq")
    ap.add_argument("--project", default="cosir_image")
    ap.add_argument("--group", default="buddy-init encoder-pair ablation")
    ap.add_argument("--tag", default=None, help="only include runs carrying this wandb tag")
    ap.add_argument("--selftest", action="store_true", help="offline arithmetic check, no wandb call")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    analyze(args.entity, args.project, args.group, tag=args.tag)


def _selftest():
    """Offline arithmetic check - no wandb call, no survival JSON needed."""
    df = pd.DataFrame([
        {"encoder_pair": "clip_img:clip_txt", "seed": 1, T2I: 50.0},
        {"encoder_pair": "dinov2:bge", "seed": 1, T2I: 52.0},
        {"encoder_pair": "clip_img:clip_txt", "seed": 2, T2I: 48.0},
        {"encoder_pair": "dinov2:bge", "seed": 2, T2I: 51.0},
        {"encoder_pair": "clip_img:clip_txt", "seed": 3, T2I: 49.0},
        {"encoder_pair": "dinov2:bge", "seed": 3, T2I: 49.0},
    ])
    deltas = compute_paired_deltas(df, T2I, "dinov2:bge")
    assert len(deltas) == 3, f"expected 3 paired cells, got {len(deltas)}"
    got = sorted(d for _, d in deltas)
    want = [0.0, 2.0, 3.0]
    assert got == want, f"expected deltas {want}, got {got}"
    s = summarize(deltas)
    assert s["n"] == 3
    assert abs(s["mean"] - (5.0 / 3)) < 1e-9
    assert s["wins"] == 2

    # survival_rate_per_cell: 3-cell toy grid, symmetric lift matrix.
    toy = {"cells": ["a", "b", "c"], "E": {"lift": [[1, 4, 6], [4, 1, 2], [6, 2, 1]]}}
    rates = survival_rate_per_cell(toy)
    assert abs(rates["a"] - 5.0) < 1e-9, rates   # mean(4, 6) off-diag
    assert abs(rates["b"] - 3.0) < 1e-9, rates   # mean(4, 2)
    assert abs(rates["c"] - 4.0) < 1e-9, rates   # mean(6, 2)
    print("SELFTEST OK")


if __name__ == "__main__":
    main()
