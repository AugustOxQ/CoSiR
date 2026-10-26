"""Build the condition-free model diagram; no data or external assets required.

Run with:
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python build_model_structure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


INK, SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GREY, BLUE, GREEN = "#a6a49e", "#1d6fd1", "#4a9c5b"
# Three tints of instructed-Qwen green, consistently ordered by aspect.
ASPECTS = [("emotion", GREEN), ("style", "#74b382"), ("genre", "#a2cdaa")]
WIDTH, HEIGHT, DPI = 2400, 1050, 200


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
        return ax.text(x, y, label, fontsize=size, fontweight=weight,
                       color=color, ha=ha, va="center", linespacing=1.16)

    def rounded(x, y, w, h, edge=GRID, fill="white", lw=1.2, radius=18):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
            edgecolor=edge, facecolor=fill, linewidth=lw,
        ))

    def arrow(start, end, color=SECONDARY, dashed=False, scale=14, lw=1.5):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=scale,
            color=color, linewidth=lw, linestyle="--" if dashed else "-",
            shrinkA=0, shrinkB=0,
        ))

    def chip(x, y, w, h, label, edge, fill="white", color=INK):
        rounded(x, y, w, h, edge=edge, fill=fill, radius=10)
        text(x + w / 2, y + h / 2, label, color=color)

    text(1200, 58,
         "Our model: one read of the default image gives emotion, style and genre spaces",
         size=18, weight="bold")
    text(1200, 125,
         "Train with instructed Qwen views; test with one default image and k-means per aspect",
         color=SECONDARY)

    xs, bw = [30, 635, 1240, 1845], 525
    headings = ["1  Four views,\none embedder", "2  One buddy graph\nper aspect",
                "3  One trunk,\nthree heads", "4  Read once\nat test time"]
    for x, heading in zip(xs, headings):
        rounded(x, 195, bw, 750)
        text(x + bw / 2, 265, heading, size=16, weight="bold")

    # Flow labels sit above the short arrows, outside all four boxes.
    for x, next_x, label in zip(xs, xs[1:], ["graphs", "training", "one pass"]):
        mid = (x + bw + next_x) / 2
        text(mid, 425, label, color=SECONDARY)
        arrow((x + bw + 10, 470), (next_x - 10, 470))

    # 1. Four embedding views share the same Qwen embedder.
    x = xs[0]
    text(x + bw / 2, 365, "Qwen3-VL-Embedding")
    chip(x + 25, 440, 475, 72, "default (no instruction)", GREY, "#f0efed")
    for i, (name, color) in enumerate(ASPECTS):
        chip(x + 12 + i * 169, 610, 163, 64, name, color)
    text(x + bw / 2, 723, "one instruction each", color=SECONDARY)
    text(x + bw / 2, 862, "images and captions")

    # 2. Schematic graph glyphs, with no implied measured graph statistics.
    x = xs[1]
    text(x + bw / 2, 373, "mutual 20-nearest\nneighbours")
    text(x + bw / 2, 480, "in the instructed view")
    text(x + bw / 2, 579, "content similarity\nsubtracted")
    nodes = [(-48, -10), (-16, -48), (41, -26), (43, 36), (-23, 45)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 4)]
    for i, (name, color) in enumerate(ASPECTS):
        cx, cy = x + 95 + 168 * i, 770
        for a, b in edges:
            pa, pb = nodes[a], nodes[b]
            ax.plot([cx + pa[0], cx + pb[0]], [cy + pa[1], cy + pb[1]],
                    color=color, lw=1.6, zorder=1)
        for dx, dy in nodes:
            ax.add_patch(Circle((cx + dx, cy + dy), 10, fc=color, ec="white", lw=0.8))
        text(cx, 870, name)

    # 3. One student trunk feeds all heads; teacher guidance enters from below.
    x = xs[2]
    text(x + bw / 2, 365, "trained on the\ndefault image")
    text(x + bw / 2, 455, "buddy terms\non each graph")
    text(x + bw / 2, 552, "teacher: the instructed\nimage (weight 1/5)")
    chip(x + 175, 615, 175, 60, "trunk", BLUE, BLUE, "white")
    ax.plot([x + bw / 2, x + bw / 2], [675, 695], color=BLUE, lw=1.6)
    centers = [x + 94, x + 263, x + 432]
    ax.plot([centers[0], centers[-1]], [695, 695], color=BLUE, lw=1.6)
    for cx, (name, _) in zip(centers, ASPECTS):
        arrow((cx, 695), (cx, 722), BLUE, scale=10)
        chip(cx - 81.5, 725, 163, 64, name, BLUE, BLUE, "white")
    chip(x + 83, 870, 359, 55, "instructed image", GREEN, "#f1f7f2")
    # Branch below the heads, so the dashed teacher edges never cross trunk edges.
    arrow((x + bw / 2, 869), (x + bw / 2, 820), GREEN, True, scale=10)
    ax.plot([centers[0], centers[-1]], [820, 820], color=GREEN, lw=1.4, ls="--")
    for cx in centers:
        arrow((cx, 820), (cx, 791), GREEN, True, scale=10)
    text(x + 413, 850, "teacher", color=GREEN)

    # 4. A single default image yields three student spaces.
    x = xs[3]
    text(x + bw / 2, 365, "default image in")
    text(x + bw / 2, 435, "three outputs out")
    chip(x + 34, 535, 175, 145, "default\nimage", GREY, "#f0efed")
    arrow((x + 220, 608), (x + 284, 608))
    for i, width in enumerate([168, 142, 120]):
        ax.add_patch(Rectangle((x + 302, 545 + i * 51), width, 31,
                               facecolor=BLUE, edgecolor="none"))
    text(x + bw / 2, 775, "k-means per aspect")
    text(x + bw / 2, 875, "scored against emotion,\nstyle, genre labels")

    text(1200, 1005,
         "Training needs Qwen's instructed views; the student's own image encoder can be Qwen, SigLIP or CLIP.",
         color=SECONDARY)
    fig.savefig(Path(__file__).with_name("model_structure.png"), dpi=DPI,
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
