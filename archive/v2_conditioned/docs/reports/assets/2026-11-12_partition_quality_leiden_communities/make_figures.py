"""Figures for docs/reports/auto/v2/2026-11-12_partition_quality_leiden_communities.md.

Reads the gitignored result files of src/test/20261110_partition_profile/ and src/test/20261112_community_sweep/,
writes the numbers the figures need to figure_data.json next to this script (committed), then draws from that file.
Run from the repo root:  python docs/reports/assets/2026-11-12_partition_quality_leiden_communities/make_figures.py
With --from-data it redraws from figure_data.json alone (no result files needed).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PROFILE = ROOT / "src/test/20261110_partition_profile/results/profile.json"
SWEEP = ROOT / "src/test/20261112_community_sweep/results/sweep.json"
CELLS = ROOT / "src/test/20261112_community_sweep/results/cells"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]  # reference categorical slots 1 to 3 (validated, light)
CONTROL = "#8a8984"


def collect():
    prof = json.loads(PROFILE.read_text())
    lift = prof["check3"]["lift"]
    heads = prof["check3b"]["by_aspect"]
    combos = [("affect", "emotion"), ("image", "genre"), ("image", "style"), ("caption", "genre")]
    bars = [{"label": f"{p} grouping\nx {a}", "groups": lift[f"{p}|{a}"]["lift"],
             "heads": heads[f"{p}|{a}"]["ratio_same_over_diff"]} for p, a in combos]
    pick = json.loads((CELLS / "leiden_k40_r1.0.json").read_text())["pairs"]
    bars.insert(1, {"label": "affect (Leiden, 31)\nx emotion", "groups": pick["groups"]["lift"]["lift"],
                    "heads": pick["heads"]["by_aspect"]["ratio_same_over_diff"]})

    sweep = json.loads(SWEEP.read_text())
    cells = []
    for f in sorted(CELLS.glob("*.json")):
        c = json.loads(f.read_text())
        rec = {"cell": c["cell"], "kind": c["kind"], "n_groups": c["groups"]["n_groups"]}
        if c["kind"] == "leiden":
            rec["k"] = c["leiden"]["settings"]["k"]
            rec["resolution"] = c["leiden"]["settings"]["resolution_parameter"]
        for term in ("told", "reader"):
            m = c["eval"][term]["fusedT_vs_fusedTcf"]["r1"]
            rec[term] = [m["point"], *m["ci95"]]
        cells.append(rec)
    ref = sweep["reference"]["R0"]
    r0 = {"n_groups": ref["n_groups"], "told": [ref["told"]["point"], *ref["told"]["ci95"]],
          "reader": [ref["reader"]["point"], *ref["reader"]["ci95"]]}
    data = {"source": {"profile": str(PROFILE.relative_to(ROOT)), "sweep": str(SWEEP.relative_to(ROOT)),
                       "sweep_plan_sha256": sweep["plan_sha256"]},
            "lift_bars": bars, "cells": cells, "R0_kmeans64": r0}
    (HERE / "figure_data.json").write_text(json.dumps(data, indent=1))
    return data


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_lift(data):
    bars = data["lift_bars"]
    fig, ax = plt.subplots(figsize=(8.2, 3.6), facecolor=SURFACE)
    style(ax)
    w = 0.36
    xs = range(len(bars))
    for i, (key, name) in enumerate((("groups", "on the groups (perfect placement)"),
                                     ("heads", "through the heads (image vs caption)"))):
        vals = [b[key] for b in bars]
        pos = [x + (i - 0.5) * (w + 0.03) for x in xs]
        ax.bar(pos, vals, width=w, color=SLOT[i], label=name, zorder=2)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.12, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(1.0, color=INK2, linewidth=1, linestyle=(0, (3, 3)), zorder=1, label="1.0 = no signal")
    ax.set_xticks(list(xs), [b["label"] for b in bars], fontsize=8.5, color=INK)
    ax.set_ylabel("same value vs different value\n(ratio)", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, labelcolor=INK)
    ax.set_ylim(0, max(b["groups"] for b in bars) * 1.12)
    fig.tight_layout()
    fig.savefig(HERE / "lift_groups_vs_heads.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_margins(data):
    cells, r0 = data["cells"], data["R0_kmeans64"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), facecolor=SURFACE, sharex=True)
    for ax, term, title in zip(axes, ("told", "reader"),
                               ("Told margin (grouping given)", "Reader margin (label-free method)")):
        style(ax)
        km = sorted([c for c in cells if c["kind"] == "kmeans"] + [{"n_groups": r0["n_groups"], **r0}],
                    key=lambda c: c["n_groups"])
        ax.errorbar([c["n_groups"] for c in km], [c[term][0] for c in km],
                    yerr=[[c[term][0] - c[term][1] for c in km], [c[term][2] - c[term][0] for c in km]],
                    color=CONTROL, marker="s", markersize=5, linewidth=1.5, capsize=2, label="k-means (control)")
        for j, k in enumerate((10, 20, 40)):
            pts = sorted([c for c in cells if c["kind"] == "leiden" and c["k"] == k], key=lambda c: c["n_groups"])
            nudge = (0.965, 1.0, 1.035)[j]  # small horizontal offset so cells with equal counts stay visible
            ax.errorbar([c["n_groups"] * nudge for c in pts], [c[term][0] for c in pts],
                        yerr=[[c[term][0] - c[term][1] for c in pts], [c[term][2] - c[term][0] for c in pts]],
                        color=SLOT[j], marker="o", markersize=7, linestyle="none", capsize=2,
                        label=f"Leiden, graph k = {k}")
        if term == "reader":
            ax.axhline(0.5, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
            ax.text(11.5, 0.51, "development bar +0.5", ha="left", va="bottom", fontsize=8, color=INK2)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlim(11, 150)
        ax.minorticks_off()
        ax.set_xticks([14, 31, 64, 118], ["14", "31", "64", "118"])
        ax.set_xlabel("number of affect groups (log scale)", fontsize=9, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    axes[0].set_ylabel("R@1 margin over matched counterpart (pp)", fontsize=9, color=INK)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(HERE / "margins_vs_groups.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    d = json.loads((HERE / "figure_data.json").read_text()) if "--from-data" in sys.argv else collect()
    fig_lift(d)
    fig_margins(d)
    print("wrote", sorted(p.name for p in HERE.glob("*.png")))
