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
import sys

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


def _selftest():
    """Offline arithmetic check -- no wandb call."""
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

    print("SELFTEST OK")


if __name__ == "__main__" and "--selftest" in sys.argv:
    _selftest()
