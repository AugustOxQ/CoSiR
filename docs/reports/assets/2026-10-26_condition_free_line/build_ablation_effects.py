"""Rebuild the ablation effects figure using only the task's supplied data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


COLORS = {"clear": "#1d6fd1", "small": "#d9a21b", "not resolved": "#a6a49e"}
INK, SECONDARY, GRID, REFERENCE = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
# Each aspect tuple is (effect, lower interval, upper interval, effect word).
ROWS = [
    ("Separate heads", [(-.0006, -.0041, .0031, "not resolved"), (-.0008, -.0097, .0088, "not resolved"), (.0049, -.0053, .0158, "not resolved")]),
    ("No teacher", [(-.0032, -.0069, .0009, "not resolved"), (-.0106, -.0210, -.0013, "small"), (.0063, -.0028, .0165, "not resolved")]),
    ("Default-view teacher", [(-.0041, -.0079, .0000, "not resolved"), (-.0048, -.0129, .0032, "not resolved"), (.0077, -.0053, .0223, "not resolved")]),
    ("Distillation only\n(no graph or caption terms)", [(-.0169, -.0215, -.0124, "clear"), (-.0995, -.1054, -.0896, "clear"), (-.0194, -.0280, -.0092, "small")]),
    ("No buddy terms", [(-.0214, -.0263, -.0166, "clear"), (-.1473, -.1562, -.1327, "clear"), (-.0449, -.0543, -.0342, "clear")]),
    ("No own-caption terms", [(.0055, .0009, .0105, "small"), (.0180, .0099, .0249, "clear"), (.0063, -.0033, .0180, "not resolved")]),
    ("Image endpoints only", [(.0057, .0014, .0103, "small"), (.0248, .0172, .0314, "clear"), (-.0086, -.0178, .0016, "not resolved")]),
    ("No content correction", [(-.0066, -.0115, -.0013, "small"), (-.0929, -.1015, -.0813, "clear"), (-.0049, -.0146, .0055, "not resolved")]),
    ("One default-view graph", [(-.0222, -.0280, -.0165, "clear"), (-.1681, -.1740, -.1557, "clear"), (-.0287, -.0394, -.0180, "clear")]),
    ("Random pairs, degrees kept", [(-.0296, -.0343, -.0248, "clear"), (-.1702, -.1801, -.1541, "clear"), (-.0625, -.0752, -.0484, "clear")]),
    ("No instruction anywhere", [(-.0229, -.0284, -.0174, "clear"), (-.1849, -.1905, -.1728, "clear"), (-.0299, -.0411, -.0181, "clear")]),
    ("Instructed inputs (three passes)", [(.0031, -.0014, .0077, "not resolved"), (.0160, .0088, .0225, "small"), (.0005, -.0103, .0113, "not resolved")]),
]
PANELS = [
    ("Emotion", .0962, (-.035, .015), [-.03, -.02, -.01, 0, .01]),
    ("Style", .3697, (-.20, .04), [-.20, -.10, 0, .04]),
    ("Genre", .4540, (-.08, .03), [-.08, -.04, 0, .03]),
]
# Give the two-line distillation label extra clearance on both sides.
ROW_POSITIONS = [0, 1, 2, 3.5, 5, 6, 7, 8, 9, 10, 11, 12]


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 14,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": SECONDARY, "ytick.color": INK,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    # The required one-line subtitle needs this width at the 14 pt minimum.
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), dpi=200)
    fig.subplots_adjust(left=.25, right=.98, top=.68, bottom=.17, wspace=.25)
    fig.suptitle(
        "The instructed graphs carry the gain; the teacher adds little;\n"
        "without the caption terms, style rises",
        x=.5, y=.98, fontsize=20, linespacing=1.12,
    )
    fig.text(.5, .84,
             "Effect on image AMI of changing one part of the first shared model; 5,000 development paintings, 3 seeds, 95% bootstrap intervals",
             ha="center", va="center", fontsize=14, color=SECONDARY)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="none", color=c,
                               markersize=7, label=word) for word, c in COLORS.items()],
               loc="center", bbox_to_anchor=(.5, .79), ncol=3, frameon=False,
               fontsize=14, handletextpad=.5, columnspacing=1.7)
    for j, (ax, (name, full, limits, ticks)) in enumerate(zip(axes, PANELS)):
        ax.text(.5, 1.105, name, transform=ax.transAxes, ha="center", fontsize=17, weight="bold")
        ax.text(.5, 1.035, f"full model: {full:.4f}", transform=ax.transAxes,
                ha="center", fontsize=14, color=SECONDARY)
        ax.axvline(0, color=REFERENCE, linewidth=1.1, zorder=1)
        for i, (_, values) in enumerate(ROWS):
            y = ROW_POSITIONS[i]
            effect, low, high, word = values[j]
            assert limits[0] <= low <= effect <= high <= limits[1]
            ax.hlines(y, *limits, color=GRID, linewidth=.6, zorder=0)
            ax.errorbar(effect, y, xerr=[[effect-low], [high-effect]], fmt="o",
                        color=COLORS[word], markersize=6, elinewidth=1.8,
                        capsize=3, capthick=1.2, zorder=3)
        ax.set_xlim(*limits)
        ax.set_ylim(ROW_POSITIONS[-1]+.5, -.5)
        ax.set_yticks(ROW_POSITIONS)
        ax.set_yticklabels([r[0] for r in ROWS] if j == 0 else [], fontsize=14)
        ax.tick_params(axis="y", length=0, pad=12)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{x:+.2f}".replace("-", "−")))
        ax.tick_params(axis="x", labelsize=14, length=4, pad=7)
        ax.set_xlabel("effect on AMI\n(variant minus\nthe full model)", fontsize=14, labelpad=10)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(REFERENCE)
    fig.savefig(Path(__file__).with_name("ablation_effects.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
