"""
Combined post-ablation analysis for the THREE buddy training families.

Each family has a focused Hydra-multirun ablation with a built-in baseline arm
(the swept value 0), and all arms of a family share ONE buddy init template, so
the only thing changing within a family is that family's training term:

  Family #1  group 'buddy-reg ablation'      axis loss.lambda_buddy        (0 = no term, buddy-init only)
  Family #2  group 'buddy-con ablation'      axis loss.lambda_buddy_con    (0 = no term)
  Family #3  group 'buddy-refresh ablation'  axis loss.buddy_refresh_blend (0 = static #2 graph)

For each family this prints, for BOTH retrieval metrics (t2i_R1, i2t_R1):
  1. Paired table: best R1 at each axis value and Δ vs the family's baseline arm,
     paired within each (lr, lr_label, dim, alpha) cell.
  2. Activity diagnostic by axis value, so a flat ΔR1 can be read as
     "active but redundant" rather than "inert":
       #1 -> drift ‖z − z_init‖  (should SHRINK as λ grows if the term grips z)
       #2 -> buddy_con_alignment (cos anchor↔buddy; should RISE when active)
       #3 -> graph_new_edge_frac (>0 ⇒ the model's graph disagrees with CLIP,
             i.e. refresh is doing something) and graph_churn (stability).

The Hydra runners log the FULL nested cfg to wandb.config, so config is read
nested: config['loss']['lambda_buddy_con'], config['optimizer']['lr'], etc.
(This is why the older analyze_buddyreg_sweep.py — which reads FLAT keys from the
wandb-sweep-agent path — does not apply to these runners.)

Usage
-----
  python scripts/analyze_buddy_families.py --entity augustoxq --project cosir_scripts
  python scripts/analyze_buddy_families.py --only refresh
  python scripts/analyze_buddy_families.py --reg-group "buddy-reg ablation" \
      --con-group "buddy-con ablation" --refresh-group "buddy-refresh ablation"

Requires: wandb, pandas, numpy (all already deps).
"""
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import wandb

T2I = "test_oracle/t2i_R1"
I2T = "test_oracle/i2t_R1"

FAMILIES = {
    "reg": {
        "name": "Family #1 — smoothness regularizer (z-space)",
        "group": "buddy-reg ablation",
        "axis": ("loss", "lambda_buddy"),
        "axis_label": "lambda_buddy",
        "diags": {"drift": "train_buddy_diag/drift_from_init"},
        "diag_hint": "drift ‖z−z_init‖ should SHRINK as λ grows if #1 grips z "
                     "(flat drift across λ ⇒ term inert / λ too small).",
    },
    "con": {
        "name": "Family #2 — contrastive supervision (combined space)",
        "group": "buddy-con ablation",
        "axis": ("loss", "lambda_buddy_con"),
        "axis_label": "lambda_buddy_con",
        "diags": {"alignment": "train_loss/buddy_con_alignment"},
        "diag_hint": "alignment (cos anchor↔buddy positives) should RISE when the "
                     "term is active (near baseline ⇒ term inert).",
    },
    "refresh": {
        "name": "Family #3 — self-refreshing graph (feeds #2)",
        "group": "buddy-refresh ablation",
        "axis": ("loss", "buddy_refresh_blend"),
        "axis_label": "buddy_refresh_blend",
        "diags": {
            "new_edge_frac": "train_buddy_refresh/graph_new_edge_frac",
            "churn": "train_buddy_refresh/graph_churn",
        },
        "diag_hint": "new_edge_frac>0 ⇒ the refreshed graph disagrees with CLIP "
                     "(refresh is doing something); churn≈1 stable, low churn = thrashing. "
                     "Baseline (blend=0) logs new_edge_frac=0 by construction.",
    },
}

# cell coordinates: everything held fixed within a family's sweep
CELL = [
    ("lr", ("optimizer", "lr")),
    ("lr_label", ("optimizer", "lr_label")),
    ("dim", ("model", "embedding_dim")),
    ("alpha", ("train", "buddies", "alpha")),
]


def cget(cfg, path, default=None):
    """Read a nested key from a wandb run.config (nested dict)."""
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


def fetch(entity, project, fam):
    api = wandb.Api()
    rows = []
    for run in api.runs(f"{entity}/{project}", filters={"group": fam["group"]}):
        cfg, summ = run.config, run.summary
        axis = cget(cfg, fam["axis"])
        if axis is None:
            continue  # not part of this family's sweep
        row = {
            "run_id": run.id,
            "state": run.state,
            "axis": float(axis),
            T2I: float(sget(summ, T2I)) if not np.isnan(sget(summ, T2I)) else np.nan,
            I2T: float(sget(summ, I2T)) if not np.isnan(sget(summ, I2T)) else np.nan,
        }
        for cname, cpath in CELL:
            cv = cget(cfg, cpath)
            row[cname] = float(cv) if cv is not None else np.nan
        for dname, dkey in fam["diags"].items():
            row[dname] = float(sget(summ, dkey))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_table(df, metric, fam):
    axes = sorted(df["axis"].unique())
    base = min(axes)  # baseline arm is the swept value 0 (== min)
    print(f"\n  --- {metric} ---")
    cell_cols = [c for c, _ in CELL]
    header = cell_cols + [f"{fam['axis_label']}={a:g}" for a in axes] + \
             [f"Δ@{a:g}" for a in axes if a != base]
    print("    " + "  ".join(f"{h:>13}" for h in header))
    deltas = defaultdict(list)
    wins = defaultdict(int)
    n_cells = 0
    for cell_key, cell in df.groupby(cell_cols, dropna=False):
        by_axis = cell.groupby("axis")[metric].max()
        if base not in by_axis.index or np.isnan(by_axis.get(base, np.nan)):
            continue
        b = by_axis[base]
        n_cells += 1
        vals = [f"{v:g}" if not (isinstance(v, float) and np.isnan(v)) else "-"
                for v in (cell_key if isinstance(cell_key, tuple) else (cell_key,))]
        for a in axes:
            vals.append(f"{by_axis[a]:.2f}" if a in by_axis.index and not np.isnan(by_axis[a]) else "  -  ")
        for a in axes:
            if a == base:
                continue
            if a in by_axis.index and not np.isnan(by_axis[a]):
                d = by_axis[a] - b
                deltas[a].append(d)
                wins[a] += int(d > 0)
                vals.append(f"{d:+.2f}")
            else:
                vals.append("  -  ")
        print("    " + "  ".join(f"{v:>13}" for v in vals))
    if n_cells == 0:
        print("    (no paired cells with a baseline arm and a finite metric)")
        return
    print(f"\n    Over {n_cells} paired cell(s):")
    for a in axes:
        if a == base:
            continue
        n = len(deltas[a])
        if n:
            print(f"      {fam['axis_label']}={a:g}: beats baseline in {wins[a]}/{n} "
                  f"(mean Δ = {np.mean(deltas[a]):+.2f} R1 pts)")


def diag_table(df, fam):
    print(f"\n  --- activity diagnostic ---  ({fam['diag_hint']})")
    cols = list(fam["diags"].keys())
    have = [c for c in cols if not df[c].dropna().empty]
    if not have:
        print("    (no diagnostic logged — runs may predate it)")
        return
    g = df.groupby("axis")[have].mean()
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(g.to_string())


def analyze_family(entity, project, fam):
    print(f"\n{'='*78}\n{fam['name']}\n  group='{fam['group']}'  axis={fam['axis_label']}\n{'='*78}")
    df = fetch(entity, project, fam)
    if df.empty:
        print("  (no runs found for this group)")
        return
    n_unfinished = (df["state"] != "finished").sum()
    print(f"  {len(df)} run(s); {df['axis'].nunique()} {fam['axis_label']} value(s)."
          + (f"  [{n_unfinished} not finished → best-so-far]" if n_unfinished else ""))
    paired_table(df, T2I, fam)
    paired_table(df, I2T, fam)
    diag_table(df, fam)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="augustoxq")
    ap.add_argument("--project", default="cosir_scripts")
    ap.add_argument("--only", choices=list(FAMILIES), default=None,
                    help="analyze only one family (default: all three)")
    ap.add_argument("--reg-group", default=None)
    ap.add_argument("--con-group", default=None)
    ap.add_argument("--refresh-group", default=None)
    args = ap.parse_args()

    overrides = {"reg": args.reg_group, "con": args.con_group, "refresh": args.refresh_group}
    keys = [args.only] if args.only else list(FAMILIES)
    print(f"Fetching from {args.entity}/{args.project} ...")
    for k in keys:
        fam = dict(FAMILIES[k])
        if overrides[k]:
            fam["group"] = overrides[k]
        analyze_family(args.entity, args.project, fam)

    print(f"\n{'='*78}\nVERDICT GUIDE (per family)")
    print("  consistent +ΔR1 ................................. term WORKS")
    print("  ΔR1≈0 but diagnostic shows the term is active ... redundant (not the lever)")
    print("  ΔR1≈0 and diagnostic flat/inactive ............. inert (raise the knob, re-judge)")
    print("  ΔR1<0 ........................................... over-constraining / harmful")
    print("  #3 specifically: compare blend=1.0 vs blend=0 — blend=0 IS Family #2, so a")
    print("  positive Δ means the LIVE graph beats the static one; new_edge_frac says whether")
    print("  the model ever actually disagreed with CLIP.")
    print("="*78)


if __name__ == "__main__":
    main()
