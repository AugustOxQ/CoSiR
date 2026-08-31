"""
Score the held-out encoder grid: for each cached encoder, is a buddy pair closer
than a random pair in that encoder's (independent) space? Reuses each dataset's own
`build_graphs` and `heldout_distance_test`, so the Impressions within/cross-photo
split and the RedCaps single-number path are both preserved.

Usage:
  python score_grid.py --dataset impressions
  python score_grid.py --dataset redcaps --smoke 64

Emits docs/reports/assets/heldout_grid/<dataset>_grid.{json,png}.
Design: docs/superpowers/specs/2026-07-08-heldout-grid-design.md
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
sys.path.insert(0, os.path.join(ROOT, "src", "test", "20260622_buddy_analysis"))
sys.path.insert(0, os.path.join(ROOT, "src", "test", "20260623_redcaps_buddy"))

from heldout_models import MODELS
from extract_heldout import DATASETS, cache_path, _mod

ASSETS = os.path.join(ROOT, "docs", "reports", "assets", "heldout_grid")
os.makedirs(ASSETS, exist_ok=True)

ORDER = ["dinov2", "siglip_v", "vit_sup", "minilm", "bge", "e5"]


def score_full(dataset, data, G, heldout, mod):
    """Normalized {buddy, random, ratio, raw} for one encoder, full graphs."""
    result = {}
    for gname in ("B", "E"):
        e = mod.edges(G[gname])
        if dataset == "impressions":
            valid = np.abs(heldout).sum(1) > 0
            ev = e[valid[e[:, 0]] & valid[e[:, 1]]]
            wm = data.source_id[ev[:, 0]] == data.source_id[ev[:, 1]]
            raw = mod.heldout_distance_test(heldout, data, ev, wm)
            buddy, random = raw["buddy_cross"], raw["random_cross"]
        else:  # redcaps: every edge is cross-content
            raw = mod.heldout_distance_test(heldout, data, e)
            buddy, random = raw["buddy"], raw["random"]
        ratio = buddy / random if random and np.isfinite(random) else float("nan")
        result[gname] = {"buddy": buddy, "random": random, "ratio": ratio, "raw": raw}
    return result


def score_smoke(data, G, heldout_N, n, seed=42):
    """Pipeline sanity on the first-N subgraph (magnitudes noisy, not interpreted)."""
    h = heldout_N / (np.linalg.norm(heldout_N, axis=1, keepdims=True) + 1e-12)
    rng = np.random.default_rng(seed)

    def mean_dist(pairs):
        if len(pairs) == 0:
            return float("nan")
        sims = (h[pairs[:, 0]] * h[pairs[:, 1]]).sum(1)
        return float((1 - sims).mean())

    out = {}
    for gname in ("B", "E"):
        e = mod_edges_in_subgraph(G[gname], n)
        ri, rj = rng.integers(0, n, 2000), rng.integers(0, n, 2000)
        rp = np.stack([ri, rj], 1)[ri != rj]
        buddy, random = mean_dist(e), mean_dist(rp)
        out[gname] = {"buddy": buddy, "random": random,
                      "ratio": buddy / random if random else float("nan"),
                      "n_edges": int(len(e))}
    return out


def mod_edges_in_subgraph(graph, n):
    from scipy.sparse import triu
    U = triu(graph, k=1).tocoo()
    e = np.stack([U.row, U.col], 1).astype(np.int64)
    return e[(e[:, 0] < n) & (e[:, 1] < n)]


def plot_grid(dataset, grid, path):
    models = [m for m in ORDER if m in grid]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    x = np.arange(len(models)); w = 0.38
    for ax, gname in zip(axes, ("B", "E")):
        buddy = [grid[m][gname]["buddy"] for m in models]
        random = [grid[m][gname]["random"] for m in models]
        colors = ["#c44" if MODELS[m]["modality"] == "image" else "#48a" for m in models]
        ax.bar(x - w / 2, buddy, w, label="buddy", color=colors)
        ax.bar(x + w / 2, random, w, label="random", color="#ccc")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{dataset} — graph {gname}\n(red=vision, blue=text; grey=random)",
                     fontsize=9)
        ax.set_ylabel("mean held-out cosine distance")
        ax.legend(fontsize=8)
    fig.suptitle("Held-out encoder grid: buddy vs random (lower buddy = meaningful)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    mod = _mod(args.dataset)
    data = mod.load_data()
    G = mod.build_graphs(data, K=30)

    grid, missing = {}, []
    for key in ORDER:
        p = cache_path(args.dataset, key, args.smoke)
        if not os.path.exists(p):
            missing.append(key)
            continue
        heldout = np.load(p)
        if args.smoke:
            grid[key] = score_smoke(data, G, heldout, args.smoke)
        else:
            grid[key] = score_full(args.dataset, data, G, heldout, mod)
        print(f"{key:9s} B ratio={grid[key]['B']['ratio']:.3f}  "
              f"E ratio={grid[key]['E']['ratio']:.3f}")

    tag = f"smoke{args.smoke}_" if args.smoke else ""
    out_json = os.path.join(ASSETS, f"{tag}{args.dataset}_grid.json")
    with open(out_json, "w") as f:
        json.dump({"dataset": args.dataset, "smoke": args.smoke,
                   "missing": missing, "grid": grid}, f, indent=2)
    print(f"wrote {out_json}")
    if missing:
        print(f"NOT YET EXTRACTED: {missing}")
    if grid and not args.smoke:
        plot_grid(args.dataset, grid, os.path.join(ASSETS, f"{args.dataset}_grid.png"))


if __name__ == "__main__":
    main()
