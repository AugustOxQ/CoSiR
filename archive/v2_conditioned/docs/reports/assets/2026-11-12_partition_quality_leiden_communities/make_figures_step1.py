"""Step-1 figure for docs/reports/auto/v2/2026-11-12_partition_quality_leiden_communities.md (§9).

Reads the gitignored results of src/test/20261116_grouping_step1_style/, writes the numbers the figure needs to
figure_data_step1.json next to this script (committed), then draws step1_told_reader.png from that file.
Run from the repo root:  python docs/reports/assets/2026-11-12_partition_quality_leiden_communities/make_figures_step1.py
With --from-data it redraws from figure_data_step1.json alone.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RES = ROOT / "src/test/20261116_grouping_step1_style/results"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1 to 4 (light, surface #fcfcfb)
ARMS = [("A0", "A0: today (affect L, image, caption)"), ("AR", "AR: + random 4th grouping"),
        ("A1", "A1: + CSD style grouping"), ("A1s", "A1s: + CSD, CSD image head"),
        ("A2", "A2: + Gram style grouping"), ("A2s", "A2s: + Gram, Gram image head"),
        ("A3", "A3: Leiden image and caption")]
GROUPINGS = [("affect", "affect"), ("image", "image (CLIP k-means 64)"), ("caption", "caption (CLIP k-means 64)"),
             ("style_csd", "style (CSD)")]
CONDS = [("emotion__style", "b", "style", "style_csd"), ("emotion__genre", "b", "genre", "image"),
         ("style__genre", "a", "style", "style_csd"), ("style__genre", "b", "genre", "image")]
PAIR_NAME = {"emotion__style": "emotion × style", "emotion__genre": "emotion × genre", "style__genre": "style × genre"}


def collect():
    style = json.loads((RES / "step1_eval_style.json").read_text())
    desc = json.loads((RES / "step1_eval_descriptive.json").read_text())
    arms = {**desc["arms"], **style["arms"]}
    rows = []
    for key, label in ARMS:
        ev = arms[key]["eval"]
        rec = {"arm": key, "label": label}
        for term in ("told", "reader"):
            m = ev[term]["fusedT_vs_fusedTcf"]["r1"]
            rec[term] = [m["point"], *m["ci95"]]
        rows.append(rec)
    picks = []
    pp = style["arms"]["A1"]["eval"]["pick"]["picked_partition"]
    for pair, cond, aspect, correct in CONDS:
        picks.append({"pair": pair, "condition": cond, "aspect": aspect, "correct": correct,
                      "shares": {g: pp[pair][cond][g] for g, _ in GROUPINGS}})
    data = {"source": {"style": str((RES / "step1_eval_style.json").relative_to(ROOT)),
                       "descriptive": str((RES / "step1_eval_descriptive.json").relative_to(ROOT)),
                       "plan_sha256": style.get("plan_sha256")},
            "margins": rows, "A1_picks": picks}
    (HERE / "figure_data_step1.json").write_text(json.dumps(data, indent=1))
    return data


def draw(data):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK, "text.color": INK})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.9), gridspec_kw={"width_ratios": [1.0, 1.15]})
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    rows = data["margins"]
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        for term, dy, color in (("told", 0.14, SLOT[0]), ("reader", -0.14, SLOT[1])):
            p, lo, hi = r[term]
            ax1.plot([lo, hi], [y + dy, y + dy], color=color, lw=2, solid_capstyle="round", zorder=2)
            ax1.plot(p, y + dy, "o", ms=6.5, color=color, mec=SURFACE, mew=1.5, zorder=3)
    ax1.axvline(0, color=INK2, lw=0.8, zorder=1)
    ax1.set_yticks(ys)
    ax1.set_yticklabels([r["label"] for r in rows])
    ax1.set_xlabel("R@1 margin over the matched counterpart (pp, 95% CI)")
    ax1.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax1.set_axisbelow(True)
    ax1.plot([], [], "o-", color=SLOT[0], label="told (oracle: right grouping given)")
    ax1.plot([], [], "o-", color=SLOT[1], label="reader (label-free, argmax Δ)")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=8.5)
    ax1.set_title("(a) Told and reader margins by arm (seed 42)", loc="left", fontsize=10)

    picks = data["A1_picks"]
    yb = list(range(len(picks)))[::-1]
    for y, pk in zip(yb, picks):
        left = 0.0
        for (g, _), color in zip(GROUPINGS, SLOT):
            w = pk["shares"][g]
            ax2.barh(y, w - 0.4, left=left + 0.2, height=0.62, color=color, zorder=2)
            if w >= 9:
                mark = " ✓" if g == pk["correct"] else ""
                ax2.text(left + w / 2, y, f"{w:.0f}%{mark}", ha="center", va="center", fontsize=8.5, color=INK,
                         fontweight="bold" if mark else "normal", zorder=3)
            left += w
    ax2.set_yticks(yb)
    ax2.set_yticklabels([f"{PAIR_NAME[p['pair']]}\n{p['aspect']} condition (right: "
                         f"{'style (CSD)' if p['correct'] == 'style_csd' else 'image'})" for p in picks])
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("share of rankings in which the reader picked each grouping (%)")
    handles = [Patch(facecolor=color, label=name) for (_, name), color in zip(GROUPINGS, SLOT)]
    ax2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=8.5)
    ax2.set_title("(b) A1: reader's pick per condition (✓ = told grouping)", loc="left", fontsize=10)
    fig.tight_layout(w_pad=3.0)
    fig.savefig(HERE / "step1_told_reader.png", dpi=180, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)


if __name__ == "__main__":
    data = json.loads((HERE / "figure_data_step1.json").read_text()) if "--from-data" in sys.argv else collect()
    draw(data)
