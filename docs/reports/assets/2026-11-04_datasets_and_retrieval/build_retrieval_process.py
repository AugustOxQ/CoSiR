"""Render the number-free retrieval schematic in the recipe-sweep style.

Run with:
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python build_retrieval_process.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

INK, SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GREY, BLUE, GREEN = "#a6a49e", "#1d6fd1", "#4a9c5b"
PALE, ORANGE = "#dbe8f8", "#eb6834"
WIDTH, HEIGHT, DPI = 2400, 1150, 200


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 14,
        "text.color": INK, "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
    ax.axis("off")

    def text(x, y, label, size=14, weight="normal", color=INK, ha="center"):
        ax.text(x, y, label, fontsize=size, fontweight=weight, color=color,
                ha=ha, va="center", linespacing=1.16)

    def rounded(x, y, w, h, edge=GRID, fill="white", radius=18):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
            edgecolor=edge, facecolor=fill, linewidth=1.2))

    def arrow(start, end, color=SECONDARY, scale=14):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                     mutation_scale=scale, color=color, linewidth=1.5,
                     shrinkA=0, shrinkB=0))

    def chip(x, y, w, h, label, fill=BLUE, color="white", size=14):
        rounded(x, y, w, h, edge=fill, fill=fill, radius=10)
        text(x + w / 2, y + h / 2, label, size=size, color=color)

    def painting(x, y, w, h):
        # Generic grey image glyph, matching the approach figures.
        ax.add_patch(Rectangle((x, y), w, h, fc="#f0efed", ec=GREY, lw=1.2))
        ax.add_patch(Circle((x + .73*w, y + .27*h), .10*min(w, h),
                            fc=GREY, ec="none"))
        ax.add_patch(Polygon([(x+.08*w, y+.88*h), (x+.38*w, y+.40*h),
                              (x+.62*w, y+.75*h), (x+.79*w, y+.56*h),
                              (x+.92*w, y+.88*h)], fc=GREY, ec="none"))

    text(1200, 58, "Retrieval: a painting plus an aspect name ranks the other paintings",
         size=18, weight="bold")
    text(1200, 125, "The aspect name picks one head; the same head reads the query and the gallery",
         color=SECONDARY)
    xs, bw = [30, 635, 1240, 1845], 525
    for x, title in zip(xs, ["Query", "Read once", "Rank the gallery", "Score"]):
        rounded(x, 195, bw, 850)
        text(x + bw/2, 265, title, size=16, weight="bold")
    for x, next_x in zip(xs, xs[1:]):
        arrow((x+bw+10, 590), (next_x-10, 590))

    x = xs[0]
    painting(x+130, 390, 265, 290)
    rounded(x+30, 790, 465, 75, edge=GREY, fill="#f0efed", radius=10)
    text(x+bw/2, 827.5, "aspect name: style")

    x = xs[1]
    text(x+bw/2, 385, "plain image")
    arrow((x+bw/2, 430), (x+bw/2, 490), BLUE, scale=8)
    chip(x+175, 500, 175, 65, "trunk")
    ax.plot([x+bw/2, x+bw/2], [565, 615], color=BLUE, lw=1.6)
    centers = [x+94, x+262.5, x+431]
    ax.plot([centers[0], centers[-1]], [615, 615], color=BLUE, lw=1.6)
    for cx, name in zip(centers, ["emotion", "style", "genre"]):
        selected = name == "style"
        arrow((cx, 615), (cx, 655), BLUE if selected else "#afc9ea", scale=8)
        chip(cx-78, 665, 156, 65, name,
             fill=BLUE if selected else PALE,
             color="white" if selected else SECONDARY, size=12)
    text(x+bw/2, 845, "the name picks the head", size=12, color=SECONDARY)

    x = xs[2]
    text(x+bw/2, 385, "other paintings,\nsame head")
    for dx in [50, 205, 360]:
        painting(x+dx, 505, 115, 135)
    text(x+bw/2, 820, "compare with cosine,\nor a comparison method\npicked on validation", size=13)

    x = xs[3]
    for y, hit in zip([345, 450, 555, 660, 765], [True, False, True, True, False]):
        painting(x+115, y, 105, 80)
        # Draw the marks as strokes so glyph availability cannot change them.
        if hit:
            ax.plot([x+280, x+294, x+320], [y+40, y+55, y+24], color=GREEN, lw=2.8)
        else:
            ax.plot([x+283, x+315], [y+25, y+57], color=ORANGE, lw=2.8)
            ax.plot([x+283, x+315], [y+57, y+25], color=ORANGE, lw=2.8)
    text(x+bw/2, 915, "hit: same label", size=13)
    text(x+bw/2, 985, "mAP: how high the hits sit", size=12)

    text(1200, 1110,
         "The plain image gives one ranking for all aspects; the instructed Qwen image gives one embedding per aspect.",
         size=12, color=SECONDARY)
    fig.savefig(Path(__file__).with_name("retrieval_process.png"), dpi=DPI, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
