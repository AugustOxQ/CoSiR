"""Build the recipe-sweep pick's model diagram.

Source: src/test/20261027_recipe_sweep/outputs/stage3/verdict.json,
pick_config.k11 = 40, checked against the PICK row and stage 3 table in
docs/reports/auto/percept/2026-12-28_recipe_sweep.md. Other labels are
inherited from the 2026-10-26 diagram, including its teacher weight 1/5.

Run with:
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python build_model_structure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


INK, SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GREY, BLUE, GREEN = "#a6a49e", "#1d6fd1", "#4a9c5b"
CHANGED, NEW = "#eb6834", "#1a9e8f"
# Three tints of instructed-Qwen green, consistently ordered by aspect.
ASPECTS = [("emotion", GREEN), ("style", "#74b382"), ("genre", "#a2cdaa")]
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
        rounded(x, 195, bw, 850)
        text(x + bw / 2, 265, heading, size=16, weight="bold")

    # Flow labels sit above the short arrows, outside all four boxes.
    for x, next_x, label in zip(xs, xs[1:], ["graphs", "train", "test"]):
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
    neighbour_label = HPacker(children=[
        TextArea("mutual ", textprops={"fontsize": 14, "color": INK}),
        TextArea("40-nearest", textprops={"fontsize": 14, "color": CHANGED}),
    ], align="center", pad=0, sep=0)
    ax.add_artist(AnnotationBbox(neighbour_label, (x + bw / 2, 350),
                                frameon=False, box_alignment=(0.5, 0.5)))
    text(x + bw / 2, 395, "neighbours")
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
    text(x + bw / 2, 460, "buddy terms\non each graph")
    text(x + bw / 2, 555, "teacher: instructed\nimage (weight 1/5)")
    text(x + bw / 2, 671, "genre: matches the\nteacher's in-batch\nsimilarities",
         color=NEW)
    chip(x + 175, 760, 175, 55, "trunk", BLUE, BLUE, "white")
    ax.plot([x + bw / 2, x + bw / 2], [815, 827], color=BLUE, lw=1.6)
    centers = [x + 105, x + 273, x + 441]
    ax.plot([centers[0], centers[-1]], [827, 827], color=BLUE, lw=1.6)
    for cx, (name, _) in zip(centers, ASPECTS):
        arrow((cx, 827), (cx, 842), BLUE, scale=6)
        chip(cx - 81.5, 845, 163, 54, name, BLUE, BLUE, "white")
    chip(x + 83, 977, 359, 44, "instructed image", GREEN, "#f1f7f2")
    # Branch below the heads, so the dashed teacher edges never cross trunk edges.
    arrow((x + bw / 2, 976), (x + bw / 2, 925), GREEN, True, scale=6)
    ax.plot([centers[0], centers[-1]], [925, 925], color=GREEN, lw=1.4, ls="--")
    for cx in centers:
        arrow((cx, 925), (cx, 901), GREEN, True, scale=6)
    text(x + 413, 951, "teacher", color=GREEN)

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

    text(30, 1070, "orange: changed; teal: new", size=12, color=SECONDARY, ha="left")
    text(1200, 1118,
         "Training needs Qwen's instructed views; the student's own image encoder can be Qwen, SigLIP or CLIP.",
         color=SECONDARY)
    fig.savefig(Path(__file__).with_name("model_structure.png"), dpi=DPI,
                facecolor="white")
    fig.savefig(Path(__file__).with_name("model_structure.svg"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
