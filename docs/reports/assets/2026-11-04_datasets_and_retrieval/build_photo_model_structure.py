"""Build the dev-log photo-model figure with the recipe-sweep figure's style.

Run with:
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python build_photo_model_structure.py

Step digits are omitted to satisfy the requested number-free figure.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

INK, SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GREY, BLUE = "#a6a49e", "#1d6fd1"
CHANGED, NEW = "#eb6834", "#1a9e8f"
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

    def arrow(start, end, color=SECONDARY, scale=14, lw=1.5):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=scale,
            color=color, linewidth=lw, shrinkA=0, shrinkB=0,
        ))

    def chip(x, y, w, h, label, edge, fill="white", color=INK, size=14):
        rounded(x, y, w, h, edge=edge, fill=fill, radius=10)
        text(x + w / 2, y + h / 2, label, color=color, size=size)

    text(1200, 58,
         "Our photo model: the emotion path, fine-tuned on unlabelled photos",
         size=18, weight="bold")
    text(1200, 125,
         "Train on photos and their explanations; test with one plain image",
         color=SECONDARY)

    xs, bw = [30, 645, 1260, 1875], 495
    headings = ["Start from the\npainting model", "Unlabelled\nphotos",
                "Fine-tune the\nemotion path", "Read once\nat test time"]
    for index, (x, heading) in enumerate(zip(xs, headings)):
        rounded(x, 195, bw, 850)
        text(x + bw / 2, 265, heading, size=16, weight="bold",
             color=CHANGED if index == 2 else NEW if index == 1 else INK)

    for x, next_x, label in zip(xs, xs[1:], ["fine-\ntune", "graphs", "test"]):
        mid = (x + bw + next_x) / 2
        text(mid, 420, label, size=12, color=SECONDARY)
        arrow((x + bw + 10, 480), (next_x - 10, 480))

    # Each inherited side keeps its own trunk and emotion output.
    x = xs[0]
    text(x + bw / 2, 420, "image encoder fixed", color=SECONDARY)
    chip(x + 20, 540, 215, 105, "image\ntrunk", BLUE, BLUE, "white")
    chip(x + 260, 540, 215, 105, "explanation\ntrunk", BLUE, BLUE, "white", size=12)
    arrow((x + 127.5, 645), (x + 127.5, 712), BLUE, scale=8)
    arrow((x + 367.5, 645), (x + 367.5, 712), BLUE, scale=8)
    chip(x + 20, 715, 215, 90, "emotion\nhead", BLUE, BLUE, "white", size=12)
    chip(x + 260, 715, 215, 90, "emotion\nhead", BLUE, BLUE, "white", size=12)
    text(x + bw / 2, 870, "style and genre heads\nnot used")

    # Teal identifies the newly introduced photo/explanation data.
    x = xs[1]
    text(x + bw / 2, 410, "Affection train photos", color=NEW)
    text(x + bw / 2, 535, "free-text explanations", color=NEW)
    text(x + bw / 2, 660, "near-copies removed", color=NEW)
    text(x + bw / 2, 785, "no emotion labels", color=NEW)

    # Orange identifies the changed emotion training path.
    x = xs[2]
    text(x + bw / 2, 390, "buddy graphs from\nthe explanations", color=CHANGED)
    text(x + bw / 2, 520, "content similarity\nsubtracted", color=CHANGED)
    text(x + bw / 2, 650, "buddy terms\non the graph", color=CHANGED)
    text(x + bw / 2, 780, "teacher:\ninstructed image", color=CHANGED)
    text(x + bw / 2, 865, "full training loop, stopped\non validation loss", size=12)
    nodes = [(-65, -10), (-22, -50), (55, -28), (58, 42), (-30, 50)]
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 4)]:
        pa, pb = nodes[a], nodes[b]
        ax.plot([x + bw / 2 + pa[0], x + bw / 2 + pb[0]],
                [975 + pa[1], 975 + pb[1]], color=CHANGED, lw=1.6, zorder=1)
    for dx, dy in nodes:
        ax.add_patch(Circle((x + bw / 2 + dx, 975 + dy), 10,
                            fc=CHANGED, ec="white", lw=0.8))

    # A plain image produces a single emotion representation.
    x = xs[3]
    text(x + bw / 2, 410, "plain image in")
    text(x + bw / 2, 510, "emotion output out", color=CHANGED)
    text(x + bw / 2, 690, "clustering: new k-means\non the test photos", size=13)
    text(x + bw / 2, 870, "retrieval: rank\nthe test photos")

    text(30, 1070, "orange: changed; teal: new", size=12, color=SECONDARY, ha="left")
    text(1200, 1118,
         "The photo model is a separate model from the painting model.",
         color=SECONDARY)
    fig.savefig(Path(__file__).with_name("photo_model_structure.png"), dpi=DPI,
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
