"""Build the image-side AMI comparison in the reference report's style."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Values checked against:
# /project/CoSiR/docs/user_read/2026-10-19_percept_core_k40.md,
# table "The comparison at a glance": direct encoders Qwen/CLIP/Qwen;
# PercepT core P3/P2/P3 (best setting per aspect); and the Ours row.
# /project/CoSiR/src/test/20261019_emotion_ceiling/20261019_emotion_ceiling_log.md,
# column "Raw": Qwen emotion_focus, Qwen style_word, Qwen genre_word.
SERIES = [
    ("Best plain image encoder, clustered directly", [0.070, 0.300, 0.449], "#a6a49e"),
    ("PercepT core, 40 topics (best setting)", [0.077, 0.229, 0.436], "#c98b2e"),
    ("Our network (one pass, default image)", [0.081, 0.348, 0.458], "#1d6fd1"),
    ("Qwen image with the aspect's instruction (one pass per aspect)",
     [0.093, 0.395, 0.496], "#4a9c5b"),
]

# Font, ink, grid, and outer margins copied from build_csd_ami.py.
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
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
    # Reserve extra space for the longer title, subtitle, and four legend rows.
    fig.subplots_adjust(left=0.135, right=0.98, bottom=0.18, top=0.59)
    fig.suptitle(
        "Our network leads clearly only on style;\n"
        "the instructed Qwen image leads on all three",
        x=0.5, y=0.97, fontsize=16,
    )
    fig.text(
        0.5, 0.825,
        "Image-side AMI, k-means with 20 clusters, 5,000 development paintings, mean of 3 seeds",
        ha="center", va="center", fontsize=11, color=INK2,
    )
    for index, (label, values, color) in enumerate(SERIES):
        offset = (index - 1.5) * 0.22
        bars = ax.bar(
            [i + offset for i in range(3)], values, width=0.21,
            color=color, label=label, zorder=3,
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=11, color=INK)

    fig.legend(
        *ax.get_legend_handles_labels(), loc="upper left",
        bbox_to_anchor=(0.135, 0.795), fontsize=11, frameon=False,
        handlelength=1.5, labelspacing=0.45,
    )
    ax.set_xticks(range(3), ["Emotion", "Style", "Genre"], fontsize=11.5)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("AMI (0 = chance)", fontsize=11, labelpad=12)
    ax.set_ylim(0, 0.55)
    ax.set_xlim(-0.55, 2.55)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(INK2)
    fig.savefig(Path(__file__).with_name("image_side_ami.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
