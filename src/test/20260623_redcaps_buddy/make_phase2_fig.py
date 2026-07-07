"""
Grouped-bar figure for the RedCaps Phase 2 VLM judge: buddy vs subreddit-random vs
plain-random GOOD rate, for graphs B and E. Reads phase2_vlm_{B,E}.json.
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                      "docs", "reports", "assets", "redcaps_buddy"))


def main():
    graphs = []
    for g in ("B", "E"):
        p = os.path.join(ASSETS, f"phase2_vlm_{g}.json")
        if os.path.exists(p):
            graphs.append(json.load(open(p)))
    if not graphs:
        print("no phase2_vlm_*.json found")
        return

    labels = ["buddy", "subreddit\nrandom", "plain\nrandom"]
    keys = ["buddy_good_rate", "subreddit_random_good_rate", "plain_random_good_rate"]
    colors = ["#4c72b0", "#dd8452", "#cccccc"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(graphs))
    w = 0.25
    for j, (k, lab) in enumerate(zip(keys, labels)):
        vals = [s[k] for s in graphs]
        bars = ax.bar(x + (j - 1) * w, vals, w, label=lab, color=colors[j])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0%}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s['graph']}  (n={s['n_buddy']} buddy pairs)" for s in graphs])
    ax.set_ylabel("caption judged GOOD match for anchor image")
    ax.set_ylim(0, 1.0)
    ax.set_title("Phase 2 — Qwen2.5-VL: does a candidate caption describe the anchor image?")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(ASSETS, "phase2_vlm.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
