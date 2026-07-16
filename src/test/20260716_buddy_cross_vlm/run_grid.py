"""
Cross-VLM buddy survival driver: build the 4x4 (vision x text) buddy grid on
RedCaps, compute chance-corrected pairwise agreement + consensus core (for B and
E), validate the core with subreddit lift, and write JSON + plots.

Usage:
  python run_grid.py --smoke 512      # fast pipeline sanity (magnitudes not interpreted)
  python run_grid.py                  # full RedCaps run

Design: docs/superpowers/specs/2026-07-16-buddy-cross-vlm-survival-design.md
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
import cross_vlm_buddy as cvb

ASSETS = os.path.join(ROOT, "docs", "reports", "assets", "buddy_cross_vlm")


def plot_heatmap(mat, names, title, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="viridis")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def plot_survival(survB, survE, path):
    t = np.arange(1, len(survB) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, survB, "o-", label="B (intersection)")
    ax.plot(t, survE, "s-", label="E (union)")
    ax.set_xlabel("consensus level t (edge present in >= t of 16 cells)")
    ax.set_ylabel("# surviving buddy edges")
    ax.set_yscale("log")
    ax.set_title("Buddy survival curve across the VLM grid")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def plot_core_lift(liftB, liftE, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for curve, lab, mk in ((liftB, "B (intersection)", "o-"),
                           (liftE, "E (union)", "s-")):
        t = [c["t"] for c in curve]
        lift = [c["lift"] for c in curve]
        ax.plot(t, lift, mk, label=lab)
    ax.axhline(1.0, color="grey", ls="--", lw=1, label="chance (random pairs)")
    ax.set_xlabel("consensus level t")
    ax.set_ylabel("same-subreddit lift of the >= t core")
    ax.set_title("Are surviving buddies semantically coherent?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--core_t", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(ASSETS, exist_ok=True)

    feats, sub_id, sub_names, vmask = cvb.load_grid_features(smoke=args.smoke)
    cell_B, cell_E, N = cvb.build_cell_graphs(feats, K=args.K, device=args.device)
    n_cells = len(cvb.CELLS)

    aggB = cvb.agreement_matrix(cell_B, N)
    aggE = cvb.agreement_matrix(cell_E, N)

    uB, cB = cvb.consensus_counts(list(cell_B.values()))
    uE, cE = cvb.consensus_counts(list(cell_E.values()))
    survB = cvb.survival_curve(cB, n_cells)
    survE = cvb.survival_curve(cE, n_cells)
    liftB = cvb.core_subreddit_lift(uB, cB, N, sub_id, sub_names, n_cells)
    liftE = cvb.core_subreddit_lift(uE, cE, N, sub_id, sub_names, n_cells)

    # smoke: assert the pipeline produced finite, well-shaped output; do not interpret.
    if args.smoke:
        assert aggB["jaccard"].shape == (n_cells, n_cells)
        assert np.isfinite(aggB["median_offdiag_jaccard"])
        assert survB.shape == (n_cells,)
        assert len(liftB) == n_cells
        print(f"[smoke] OK  N={N}  medianJ(B)={aggB['median_offdiag_jaccard']:.4f} "
              f"medianLift(B)={aggB['median_offdiag_lift']:.2f}")
        return

    summary = {
        "n_nodes": int(N),
        "K": args.K,
        "cells": [f"{v}x{t}" for v, t in cvb.CELLS],
        "B": {"median_offdiag_jaccard": aggB["median_offdiag_jaccard"],
              "median_offdiag_lift": aggB["median_offdiag_lift"],
              "jaccard": aggB["jaccard"].tolist(),
              "overlap": aggB["overlap"].tolist(),
              "lift": aggB["lift"].tolist(),
              "survival": survB.tolist(),
              "core_lift": liftB},
        "E": {"median_offdiag_jaccard": aggE["median_offdiag_jaccard"],
              "median_offdiag_lift": aggE["median_offdiag_lift"],
              "jaccard": aggE["jaccard"].tolist(),
              "overlap": aggE["overlap"].tolist(),
              "lift": aggE["lift"].tolist(),
              "survival": survE.tolist(),
              "core_lift": liftE},
    }
    with open(os.path.join(ASSETS, "grid_agreement.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {os.path.join(ASSETS, 'grid_agreement.json')}")

    plot_heatmap(aggB["jaccard"], aggB["names"], "B buddy-graph agreement (Jaccard)",
                 os.path.join(ASSETS, "agreement_B.png"))
    plot_heatmap(aggE["jaccard"], aggE["names"], "E buddy-graph agreement (Jaccard)",
                 os.path.join(ASSETS, "agreement_E.png"))
    plot_survival(survB, survE, os.path.join(ASSETS, "survival_curves.png"))
    plot_core_lift(liftB, liftE, os.path.join(ASSETS, "core_lift.png"))

    np.save(os.path.join(ASSETS, "core_edges_B.npy"),
            cvb.core_edges(uB, cB, args.core_t, N))
    np.save(os.path.join(ASSETS, "core_edges_E.npy"),
            cvb.core_edges(uE, cE, args.core_t, N))
    print(f"wrote core_edges_{{B,E}}.npy (t>={args.core_t})")
    print(f"[done] B: medianJ={aggB['median_offdiag_jaccard']:.4f} "
          f"medianLift={aggB['median_offdiag_lift']:.2f} | "
          f"E: medianJ={aggE['median_offdiag_jaccard']:.4f} "
          f"medianLift={aggE['median_offdiag_lift']:.2f}")


if __name__ == "__main__":
    main()
