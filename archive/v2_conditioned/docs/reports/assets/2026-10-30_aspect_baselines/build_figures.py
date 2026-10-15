"""Figures for the E1 report. Reads src/test/20261030_aspect_baselines/results/baselines_seed{42,43}.json."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parents[3] / "src/test/20261030_aspect_baselines/results"
ORDER = ["cosine", "diag", "diag_relu", "bilinear", "kissme", "rca", "xing", "wang", "probe", "tip",
         "value_prototype", "SE", "C0", "R3", "SE_uniform"]
PAIRS = ["emotion__style", "emotion__genre", "style__genre"]
recs = {s: json.load(open(RES / f"baselines_seed{s}.json")) for s in (42, 43)}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, s in zip(axes, (42, 43)):
    sc = recs[s]["scorers"]
    for i, name in enumerate(ORDER):
        o = sc[name]["overall"]
        color = "#555555" if name == "cosine" else "#e07b00" if name == "SE_uniform" else "#1f77b4"
        ax.errorbar(o["r1"]["point"], o["gain"]["point"],
                    xerr=[[o["r1"]["point"] - o["r1"]["ci95"][0]], [o["r1"]["ci95"][1] - o["r1"]["point"]]],
                    yerr=[[o["gain"]["point"] - o["gain"]["ci95"][0]], [o["gain"]["ci95"][1] - o["gain"]["point"]]],
                    fmt="o", color=color, ms=4, lw=1)
        ax.annotate(name, (o["r1"]["point"], o["gain"]["point"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"episode seed {s}")
    ax.set_xlabel("R@1 (%)")
axes[0].set_ylabel("condition gain (points)")
fig.suptitle("E1: R@1 against condition gain, 95% painting-clustered CI (grey = CLIP cosine, orange = SE uniform control)")
fig.tight_layout()
fig.savefig(HERE / "r1_vs_gain.png", dpi=150)

fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
names = [n for n in ORDER if n not in ("cosine", "SE_uniform")]
w = 0.27
for ax, s in zip(axes, (42, 43)):
    for j, p in enumerate(PAIRS):
        g = [recs[s]["scorers"][n]["per_pair"][p]["gain"] for n in names]
        pt = np.array([x["point"] for x in g])
        lo = pt - np.array([x["ci95"][0] for x in g]); hi = np.array([x["ci95"][1] for x in g]) - pt
        ax.bar(np.arange(len(names)) + (j - 1) * w, pt, w, yerr=[lo, hi], capsize=1.5, label=p.replace("__", " x "))
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel(f"gain, seed {s}")
axes[0].legend(fontsize=8)
axes[1].set_xticks(np.arange(len(names)), names, rotation=40, ha="right")
fig.tight_layout()
fig.savefig(HERE / "per_pair_gain.png", dpi=150)
