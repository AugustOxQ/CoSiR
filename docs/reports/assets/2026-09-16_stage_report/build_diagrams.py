"""Build the two architecture schematics for the 2026-09-16 stage report."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path(__file__).parent
FROZEN = "#e5e7eb"
TRAINABLE = "#dbeafe"
OUTPUT = "#ecfdf5"
LOSS = "#fef3c7"
EDGE = "#475569"
TEXT = "#1e293b"


def box(ax, xy, width, height, label, color, fontsize=9, rounded=True):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.035" if rounded else "square,pad=0.02",
        facecolor=color, edgecolor=EDGE, linewidth=1.25,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center",
            color=TEXT, fontsize=fontsize, linespacing=1.18, wrap=True)
    return (x, y, width, height)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.35, color=EDGE, shrinkA=2, shrinkB=3))


def elbow_arrow(ax, points):
    """Draw a right-angled connector with an arrowhead on its final segment."""
    for start, end in zip(points, points[1:-1]):
        ax.plot((start[0], end[0]), (start[1], end[1]), color=EDGE, linewidth=1.35)
    arrow(ax, points[-2], points[-1])


def right(rect):
    return rect[0] + rect[2], rect[1] + rect[3] / 2


def left(rect):
    return rect[0], rect[1] + rect[3] / 2


def bottom(rect):
    return rect[0] + rect[2] / 2, rect[1]


def top(rect):
    return rect[0] + rect[2] / 2, rect[1] + rect[3]


def table(ax):
    rect = box(
        ax, (0.55, 0.90), 2.85, 1.48,
        "Per-sample condition table\n(one 16-d row per training sample)\ninitialized from the buddy graph\ntrained directly, own learning rate",
        TRAINABLE, fontsize=8.2,
    )
    # A small stacked-row icon conveys that the table has many independent rows.
    icon_x, icon_y = 0.70, 1.10
    for row in range(4):
        ax.add_patch(Rectangle((icon_x, icon_y + row * 0.16), 0.34, 0.11,
                               facecolor="#93c5fd", edgecolor=EDGE, linewidth=0.55))
    return rect


def common(ax, title):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=15, fontweight="bold", color=TEXT, pad=14)

    image = box(ax, (0.45, 5.25), 1.05, 0.62, "Image", OUTPUT)
    image_encoder = box(ax, (1.90, 5.02), 1.75, 1.08, "Frozen CLIP\nimage encoder", FROZEN)
    image_embedding = box(ax, (4.05, 5.20), 1.75, 0.70, "image embedding (512-d)", OUTPUT, fontsize=8.5)

    text_input = box(ax, (0.45, 3.55), 1.05, 0.62, "Text", OUTPUT)
    text_encoder = box(ax, (1.90, 3.32), 1.75, 1.08, "Frozen CLIP\ntext encoder", FROZEN)
    text_embedding = box(ax, (4.05, 3.50), 1.75, 0.70, "text embedding (512-d)", OUTPUT, fontsize=8.5)

    combiner = box(ax, (7.00, 4.55), 2.35, 1.30,
                   "Combiner\n(low-rank residual adapter)\ncondition vector -> small additive correction",
                   TRAINABLE, fontsize=8.3)
    combined = box(ax, (9.62, 4.88), 1.80, 0.66, "combined image embedding", OUTPUT, fontsize=7.6)
    other_proj = box(ax, (7.00, 3.20), 2.35, 0.90,
                     "other_proj\n(identity-initialized linear layer)", TRAINABLE, fontsize=8.4)
    projected = box(ax, (9.62, 3.32), 1.80, 0.66, "projected text embedding", OUTPUT, fontsize=7.6)
    loss = box(ax, (9.45, 1.45), 1.95, 0.80, "Contrastive retrieval loss", LOSS, fontsize=8.4)

    arrow(ax, right(image), left(image_encoder))
    arrow(ax, right(image_encoder), left(image_embedding))
    arrow(ax, right(text_input), left(text_encoder))
    arrow(ax, right(text_encoder), left(text_embedding))
    arrow(ax, right(image_embedding), left(combiner))
    arrow(ax, right(text_embedding), left(other_proj))
    arrow(ax, right(combiner), left(combined))
    arrow(ax, right(other_proj), left(projected))
    arrow(ax, bottom(combined), top(loss))
    arrow(ax, bottom(projected), (10.70, 2.25))

    ax.text(0.52, 0.38, "Legend:  grey = frozen/fixed CLIP components     blue = trainable components",
            fontsize=8, color="#475569")
    return image_embedding, text_embedding, combiner


def before():
    fig, ax = plt.subplots(figsize=(11, 6.6), constrained_layout=True)
    image_embedding, text_embedding, combiner = common(ax, "CoSiR before Experiment 18 — free per-sample condition vectors")
    condition_table = table(ax)
    arrow(ax, right(condition_table), (7.00, 4.85))
    fig.savefig(OUT_DIR / "architecture_before.png", dpi=150, facecolor="white")
    plt.close(fig)


def after():
    fig, ax = plt.subplots(figsize=(11, 6.6), constrained_layout=True)
    image_embedding, text_embedding, combiner = common(ax, "Experiment 18 — prototype-pooled conditioning (only this block changes)")
    bank = box(ax, (0.45, 0.98), 2.05, 1.24,
               "Prototype bank\n16 shared, learned prototypes\n(each has a key + a value vector)", TRAINABLE, fontsize=8.2)
    query = box(ax, (3.02, 1.23), 1.35, 0.74, "query projection\n(from mean of image\n+ text embeddings)", TRAINABLE, fontsize=7.6)
    attention = box(ax, (4.80, 0.93), 1.80, 1.34,
                    "softmax attention\n(temperature-controlled)\nover the 16 prototype keys", TRAINABLE, fontsize=8.0)
    weighted = box(ax, (5.05, 2.68), 1.55, 0.92,
                   "weighted sum of\nthe 16 prototypes' value vectors", TRAINABLE, fontsize=7.8)

    arrow(ax, (4.18, 5.20), top(query))
    elbow_arrow(ax, [(4.60, 3.50), (4.60, 2.35), (3.3, 2.35), (3.3, 1.97)])
    elbow_arrow(ax, [right(bank), (2.68, 1.60), (2.68, 2.48), (5.70, 2.48), top(attention)])
    arrow(ax, right(query), left(attention))
    arrow(ax, top(attention), bottom(weighted))
    arrow(ax, right(weighted), (7.00, 4.85))
    ax.text(0.52, 2.55, "same module for every sample —\nno per-sample table.",
            fontsize=8.2, color="#475569", style="italic")

    fig.savefig(OUT_DIR / "architecture_after.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    before()
    after()
