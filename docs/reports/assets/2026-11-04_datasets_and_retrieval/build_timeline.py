"""Render the stage timeline; adapted from 2026-10-26_condition_free_line.

Dates are Amsterdam experiment dates, not the auto reports' sequence dates.
Outcome colours and the independent event markers follow the original script.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


INK = "#0b0b0b"
SECONDARY = "#52514e"
GRID = "#e4e3df"
COLORS = {"worked": "#1d6fd1", "did not work": "#eb6834", "check": "#a6a49e"}

# Date, event labels with outcomes. Placement communicates order, not duration.
STEPS = [
    ("29 Oct", [
        ("Plain cosine retrieval:\nemotion below teacher", "did not work"),
        ("Validation read-outs:\nretrieval bar passed", "worked"),
        ("Wrong-head test passed;\nclass analysis checked", "worked"),
    ]),
    ("30 to 31 Oct", [
        ("Frozen c* on Affection:\nbelow teacher", "did not work"),
        ("CRL: painting bars passed;\nphoto WIN bar failed", "did not work"),
    ]),
    ("1 Nov", [
        ("Photo fine-tune:\nAffection bars passed", "worked"),
        ("Photo CRL bar\npassed", "worked"),
    ]),
    ("2 Nov", [
        ("Three more encoders:\nmain bars passed", "worked"),
        ("CLIP L/14 painting emotion\nretrieval below teacher", "did not work"),
        ("Frozen photo model:\nEmoSet bars passed", "worked"),
        ("Settings and ablations:\nno model selected", "check"),
    ]),
    ("3 Nov", [
        ("MMArt: four encoders\npassed main bars", "worked"),
        ("Below teacher in emotion retrieval:\nQwen and CLIP L/14", "did not work"),
        ("SemArt feasibility:\nnew artists, no result", "check"),
        ("Experiment overview:\ncoverage and plan only", "check"),
    ]),
]


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "text.color": INK, "figure.facecolor": "white",
                         "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(16, 8), dpi=180)
    fig.subplots_adjust(left=0.025, right=0.985, top=0.78, bottom=0.14)
    ax.set(xlim=(-0.15, 5.05), ylim=(-0.45, 4.5))
    ax.set_axis_off()
    fig.text(0.035, 0.95, "Retrieval passed with read-outs; photo training closed the Affection gap",
             fontsize=21, va="top")
    fig.text(0.035, 0.895, "29 October to 3 November 2026 | Main bars and teacher limits shown separately",
             color=SECONDARY, fontsize=13)
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=8,
                      color=color, label=outcome) for outcome, color in COLORS.items()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.029, 0.87),
               ncol=3, frameon=False, fontsize=12)
    for x, (date, events) in enumerate(STEPS):
        ax.text(x, 4.4, date, fontsize=15, weight="bold", va="top")
        ax.plot([x - 0.07, x - 0.07], [0.15, 4.0], color=GRID, lw=1.2)
        for row, (label, outcome) in enumerate(events):
            y = 3.7 - row * 1.05
            ax.scatter([x - 0.07], [y], s=65, color=COLORS[outcome], zorder=3)
            ax.text(x, y, label, fontsize=10.5, va="center", linespacing=1.3)
    fig.text(0.035, 0.075,
             "Source labels: r11 retrieval; r12 frozen photos and CRL; r13 photo fine-tune; "
             "r14 encoders, EmoSet and analysis; r15 MMArt; r16 overview.",
             fontsize=10, color=SECONDARY)
    fig.text(0.035, 0.045,
             "Later work, no results here: r17 new scores from saved outputs; "
             "r18 outside methods; r19 past our own picks. Positions show order only.",
             fontsize=10, color=SECONDARY)
    fig.savefig(Path(__file__).with_name("timeline.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
