"""
Add the held-out DINOv2 probe to an existing run_phase1 stats.json without
recomputing the (slow) spectral structure init. Rebuilds the buddy graphs
(KNN only) and computes buddy-vs-random DINO distance for B and E, then
regenerates the lift+DINO figure.

Run after extract_dino.py has produced dino_feats.npy.
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import redcaps_buddy as rb
import run_phase1 as rp

DINO = os.path.join(os.path.dirname(__file__), "dino_feats.npy")
STATS = os.path.join(rp.ASSETS, "stats.json")


def main():
    heldout = np.load(DINO)
    print(f"held-out DINO: {heldout.shape}")
    data = rb.load_data()
    G = rb.build_graphs(data, K=30)

    stats = json.load(open(STATS))
    for name in ("B", "E"):
        e = rb.edges(G[name])
        d = rb.heldout_distance_test(heldout, data, e)
        stats["graphs"][name]["dino"] = d
        print(f"[{name}] DINO buddy={d['buddy']:.3f} random={d['random']:.3f} "
              f"(n_edges={d['n_buddy_edges']:,})")

    json.dump(stats, open(STATS, "w"), indent=2)
    rp.fig_lift_and_dino(stats)
    print(f"Updated {STATS} + lift_and_dino.png")


if __name__ == "__main__":
    main()
