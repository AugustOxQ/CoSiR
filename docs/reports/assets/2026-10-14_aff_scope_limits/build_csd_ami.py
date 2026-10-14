"""Build the CSD style/genre agreement figure with matplotlib."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Source: /project/CoSiR/docs/reports/auto/v2/2026-11-26_design_l_style_source.md
# Table: "Groupings (rule §6, after the gate)", rows T, R0, V1;
# columns "AMI style" and "AMI genre", confirmed against the source table.
STYLE = [1.00, 0.34, 0.10]
GENRE = [0.22, 0.33, 0.08]
LABELS = [
    "True style labels\n(reference)",
    "CSD style groups\n(label-free)",
    "CSD, content\npart removed",
]

# Font and palette from 2026-10-12_affect_steering_line/build_figures.py.
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def main():
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    fig.subplots_adjust(left=0.135, right=0.98, bottom=0.18, top=0.73)
    fig.suptitle(
        "Label-free style groups match genre as much as style",
        x=0.5, y=0.97, fontsize=16,
    )
    width = 0.30
    for values, offset, color, legend_label in [
        (STYLE, -width / 2, BLUE, "agreement with the true style labels"),
        (GENRE, width / 2, ORANGE, "agreement with the true genre labels"),
    ]:
        bars = ax.bar(
            [i + offset for i in range(3)], values, width=width,
            color=color, label=legend_label, zorder=3,
        )
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=12, color=INK)

    fig.legend(
        *ax.get_legend_handles_labels(), loc="upper left",
        bbox_to_anchor=(0.135, 0.89), fontsize=11, frameon=False,
        handlelength=1.5, labelspacing=0.45,
    )
    ax.set_xticks(range(3), LABELS, fontsize=11.5)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("Agreement with the true labels (AMI, 0 to 1)", fontsize=11, labelpad=12)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.55, 2.55)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(INK2)
    fig.savefig(Path(__file__).with_name("csd_style_genre_ami.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
