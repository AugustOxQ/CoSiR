"""
Post-sweep analysis for the Family #1 buddy-graph smoothness regularizer (v4 sweep).

Pulls the sweep runs from W&B and answers the only question that matters:
does the buddy regularizer (lambda_buddy > 0) beat the paired baseline
(lambda_buddy = 0) at the SAME (lr, lr_label) — and does the benefit grow
with lr_label (the regime where z actually drifts from the buddy init)?

It prints three things:
  1. Paired per-cell table: best R1 at each lambda_buddy within each
     (lr, lr_label) cell, plus delta vs lambda_buddy=0.
  2. lambda_buddy x lr_label interaction: mean delta (averaged over lr).
     A benefit that grows down the lr_label column is the coherent-mechanism
     signal; random signs mean "doing nothing".
  3. Activity diagnostics: mean final drift_from_init by lambda_buddy. If
     higher lambda visibly shrinks drift, the regularizer is gripping z
     (so a flat R1 means "active but redundant", not "inert").

Usage
-----
  python scripts/analyze_buddyreg_sweep.py \
      --entity augustoxq --project cosir_image \
      [--group "buddy-reg sweep"] [--metric test_oracle/t2i_R1]

Requires: wandb (already a dep), pandas, numpy.
"""
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import wandb

DRIFT_KEY = "train_buddy_diag/drift_from_init"


def _get(d, key, default=None):
    """Safe getter for wandb config/summary (they behave like dicts)."""
    try:
        return d.get(key, default)
    except Exception:
        return getattr(d, key, default)


def fetch(entity, project, group, metric):
    api = wandb.Api()
    path = f"{entity}/{project}"
    filters = {"group": group} if group else None
    rows = []
    for run in api.runs(path, filters=filters):
        cfg, summ = run.config, run.summary
        lam = _get(cfg, "lambda_buddy")
        if lam is None:  # not a buddy-reg run
            continue
        m = _get(summ, metric)
        if m is None:
            print(f"  [skip] run {run.id} ({run.state}) has no '{metric}' yet")
            continue
        rows.append(
            {
                "run_id": run.id,
                "state": run.state,
                "lr": float(_get(cfg, "lr")),
                "lr_label": float(_get(cfg, "lr_label")),
                "lambda_buddy": float(lam),
                "metric": float(m),
                "drift": _get(summ, DRIFT_KEY, np.nan),
            }
        )
    return pd.DataFrame(rows)


def per_cell_table(df, metric):
    """For each (lr, lr_label) cell, R1 at each lambda + delta vs lambda=0."""
    lambdas = sorted(df["lambda_buddy"].unique())
    print(f"\n{'='*72}\n1. PAIRED PER-CELL TABLE  (metric = {metric}, best epoch)\n{'='*72}")
    header = ["lr", "lr_label"] + [f"λ={l:g}" for l in lambdas] + [
        f"Δλ={l:g}" for l in lambdas if l != 0
    ]
    print("  " + "  ".join(f"{h:>11}" for h in header))

    deltas = defaultdict(list)  # lambda -> list of (lr_label, delta)
    wins = defaultdict(int)
    n_cells = 0
    for (lr, lrl), cell in df.groupby(["lr", "lr_label"]):
        by_lam = cell.groupby("lambda_buddy")["metric"].max()
        if 0.0 not in by_lam.index:
            continue
        base = by_lam[0.0]
        n_cells += 1
        vals = [f"{lr:.0e}", f"{lrl:.0e}"]
        for l in lambdas:
            vals.append(f"{by_lam.get(l, np.nan):.4f}" if l in by_lam.index else "   -   ")
        for l in lambdas:
            if l == 0:
                continue
            if l in by_lam.index:
                d = by_lam[l] - base
                deltas[l].append((lrl, d))
                if d > 0:
                    wins[l] += 1
                vals.append(f"{d:+.4f}")
            else:
                vals.append("   -   ")
        print("  " + "  ".join(f"{v:>11}" for v in vals))

    print(f"\n  Direction consistency over {n_cells} paired cells:")
    for l in lambdas:
        if l == 0:
            continue
        n = len(deltas[l])
        if n:
            print(f"    λ={l:g}: beats baseline in {wins[l]}/{n} cells "
                  f"(mean Δ = {np.mean([d for _, d in deltas[l]]):+.4f})")
    return deltas


def interaction_table(deltas):
    """mean delta as a function of (lambda_buddy, lr_label), averaged over lr."""
    print(f"\n{'='*72}\n2. λ × lr_label INTERACTION  (mean Δ vs baseline, averaged over lr)\n{'='*72}")
    print("  Look for Δ that GROWS as lr_label increases → coherent mechanism.\n")
    recs = []
    for l, pairs in deltas.items():
        for lrl, d in pairs:
            recs.append({"lambda_buddy": l, "lr_label": lrl, "delta": d})
    if not recs:
        print("  (no paired deltas)")
        return
    piv = (
        pd.DataFrame(recs)
        .pivot_table(index="lr_label", columns="lambda_buddy", values="delta", aggfunc="mean")
        .sort_index()
    )
    with pd.option_context("display.float_format", lambda x: f"{x:+.4f}"):
        print(piv.to_string())


def drift_table(df):
    print(f"\n{'='*72}\n3. ACTIVITY DIAGNOSTIC  (mean final drift ‖z − z_init‖ by λ)\n{'='*72}")
    print("  Higher λ should shrink drift → regularizer is gripping z.")
    print("  If drift is flat across λ, the term is inert (λ too small).\n")
    g = df.dropna(subset=["drift"]).groupby("lambda_buddy")["drift"].agg(["mean", "std", "count"])
    if g.empty:
        print("  (no drift_from_init logged — older runs predate the diagnostic)")
        return
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(g.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="augustoxq")
    ap.add_argument("--project", default="cosir_scripts")
    ap.add_argument("--group", default=None, help="W&B group to filter (optional).")
    ap.add_argument("--metric", default="test_oracle/t2i_R1")
    args = ap.parse_args()

    print(f"Fetching runs from {args.entity}/{args.project}"
          + (f" (group='{args.group}')" if args.group else "") + " ...")
    df = fetch(args.entity, args.project, args.group, args.metric)
    if df.empty:
        print("No buddy-reg runs found (config must contain 'lambda_buddy').")
        return
    print(f"Loaded {len(df)} runs "
          f"({df['lambda_buddy'].nunique()} λ × {df['lr'].nunique()} lr × {df['lr_label'].nunique()} lr_label).")
    n_unfinished = (df["state"] != "finished").sum()
    if n_unfinished:
        print(f"  note: {n_unfinished} run(s) not in 'finished' state — metric is best-so-far.")

    deltas = per_cell_table(df, args.metric)
    interaction_table(deltas)
    drift_table(df)
    print(f"\n{'='*72}")
    print("VERDICT GUIDE:")
    print("  consistent +Δ (esp. growing with lr_label) ............ WORKS → try #2/#3")
    print("  Δ≈0 but drift shrinks with λ .......................... redundant → #2 is the real test")
    print("  Δ≈0 and drift flat across λ ........................... inert → raise λ, re-judge")
    print("  Δ<0 at high λ ......................................... over-constraining → #2's contrastive form")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
