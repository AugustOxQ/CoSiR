"""Rebuild timeline.png from the task's eighteen supplied steps only."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


INK = "#0b0b0b"
SECONDARY = "#52514e"
GRID = "#e4e3df"
COLORS = {"worked": "#1d6fd1", "did not work": "#eb6834", "check": "#a6a49e"}

# day, vertical position, short display label, outcome, held data
STEPS = [
    (17, 790, "Code audit and\npremise checks", "check", False),
    (17, 695, "One Qwen instruction\nper aspect", "check", False),
    (17, 600, "Image-only student\ntoward captions", "did not work", False),
    (17, 505, "Emotion-word\nmasking", "check", False),
    (18, 315, "Student on\npurified pairs", "did not work", False),
    (18, 215, "Aspect graph sheet", "check", False),
    (18, 115, "Graph follow-up,\ninstructed inputs", "check", False),
    (19, 885, "First PercepT core run\n(thread-order bug)", "check", False),
    (19, 790, "PercepT's own\ntopic-count rule", "check", False),
    (19, 695, "PercepT core\nwith 40 topics", "check", False),
    (19, 600, "Labelled ceiling probe", "check", False),
    (19, 505, "First shared model:\ntrunk, heads, teacher", "worked", False),
    (20, 315, "Held emotion claim test,\ntwo scales", "worked", True),
    (21, 695, "Framing note:\nwhat is claimable", "check", False),
    (21, 600, "Ablation and the\nfour Qwen views", "check", False),
    (21, 885, "Qwen attention maps", "check", False),
    (25, 215, "Recipe pick: SigLIP 2, Qwen pass;\nCLIP B/32 inconclusive", "worked", True),
    (26, 695, "CLIP L/14, SigLIP 1 pass;\nDINOv2 fails; genre term\n(development)", "worked", True),
]


def day_x(day):
    return 110 + (day - 17) * 240


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 14,
        "text.color": INK, "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(12, 6.3), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 2400), ylim=(0, 1260))
    ax.set_axis_off()
    ax.text(75, 1225,
            "Eighteen steps in ten days: what worked,\nwhat did not, and what was a check",
            fontsize=21, va="top", linespacing=1.14)
    ax.text(75, 1065,
            "The condition-free line, 17 to 26 October 2026; held results are marked",
            fontsize=14, va="center", color=SECONDARY)
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=8,
                      color=color, label=outcome)
               for outcome, color in COLORS.items()]
    handles.append(Line2D([], [], marker="*", linestyle="none", markersize=11,
                          color=INK, label="held data"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.026, 0.791),
              ncol=4, frameon=False, fontsize=14, handletextpad=0.5,
              columnspacing=1.6, borderaxespad=0)

    axis_y = 430
    ax.plot([day_x(17), day_x(26)], [axis_y, axis_y], color=SECONDARY, lw=1.1)
    for day in range(17, 27):
        x = day_x(day)
        ax.plot([x, x], [axis_y - 8, axis_y + 8], color=SECONDARY, lw=1)
        ax.text(x, axis_y - 25, f"{day} Oct", ha="center", va="top", fontsize=14)

    # One quiet vertical leader per date; event dots sit at the exact day.
    for day in sorted({step[0] for step in STEPS}):
        ys = [step[1] for step in STEPS if step[0] == day]
        ax.plot([day_x(day)] * 2, [min(axis_y, *ys), max(axis_y, *ys)],
                color=GRID, lw=1.1, zorder=1)

    # The only duration in the supplied data: 21–25 October.
    ax.plot([day_x(21), day_x(25)], [885, 885], color=COLORS["check"], lw=1.1)
    ax.plot([day_x(25)] * 2, [877, 893], color=COLORS["check"], lw=1.1)

    held_labels = []
    for day, y, label, outcome, held in STEPS:
        x = day_x(day)
        ax.scatter([x], [y], s=65, color=COLORS[outcome], zorder=4)
        right_aligned = day in (25, 26)
        tx = x - 26 if right_aligned else x + 26
        # A white patch keeps the duration line clear of its label.
        t = ax.text(tx, y, label, ha="right" if right_aligned else "left",
                    va="center", fontsize=14, linespacing=1.12,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=5)
        if held:
            held_labels.append((t, day, y))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label, day, y in held_labels:
        box = label.get_window_extent(renderer)
        # Put a filled star beside the label, with room for the date marker.
        sx = ax.transData.inverted().transform((box.x1, box.y1))[0] + 22
        if day in (25, 26):
            sx = day_x(day) + 32
        ax.scatter([sx], [y], marker="*", s=90, color=INK, zorder=6)

    fig.savefig(Path(__file__).with_name("timeline.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
