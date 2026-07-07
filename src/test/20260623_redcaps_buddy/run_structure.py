"""
Structure-only runner: computes the buddy spectral init on a small DENSE
subsample (so the arpack solver stays well-conditioned) and runs the
init-structure probes, merging them into the existing validation stats.json.

Separated from run_phase1 so the (cheap, robust) validation result is never
recomputed or risked by the (slow) spectral step.
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import redcaps_buddy as rb
import run_phase1 as rp
from src.conditional_buddy import compute_buddy_init

STATS = os.path.join(rp.ASSETS, "stats.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="subsample size (0 = full)")
    ap.add_argument("--k", type=int, default=30, help="mutual-KNN K (init default 30)")
    ap.add_argument("--eigen-solver", default="auto",
                    help="auto | arpack | amg | lobpcg")
    ap.add_argument("--no-connect", action="store_true",
                    help="disable ensure_connected (ablation; leaves E fragmented)")
    args = ap.parse_args()
    connect = not args.no_connect

    data = rb.load_data()
    stats = json.load(open(STATS))

    n = data.n if args.n in (0, None) or args.n >= data.n else args.n
    rng = np.random.default_rng(rp.SEED)
    sidx = np.arange(data.n) if n == data.n else \
        np.sort(rng.choice(data.n, size=n, replace=False))
    print(f"Computing buddy init [{len(sidx)},16] K={args.k} "
          f"solver={args.eigen_solver} connect={connect} (of {data.n}) …", flush=True)
    emb = compute_buddy_init(data.img[sidx], data.txt[sidx], n_dim=16,
                             method="spectral", K=args.k, alpha=0.5,
                             device="cuda", normalize_method="rank", seed=rp.SEED,
                             eigen_solver=args.eigen_solver, connect_components=connect)
    y_s = data.sub_id[sidx]
    stats["structure_meta"] = {"struct_n": int(len(sidx)), "struct_K": args.k,
                               "eigen_solver": args.eigen_solver,
                               "connect_components": connect}
    top, mi = rp.structure_probes(y_s, emb, data.sub_names, stats)
    rp.fig_structure(y_s, emb, data.sub_names, top, mi)

    json.dump(stats, open(STATS, "w"), indent=2)
    print(f"Merged structure probes into {STATS} + init_structure.png", flush=True)


if __name__ == "__main__":
    main()
