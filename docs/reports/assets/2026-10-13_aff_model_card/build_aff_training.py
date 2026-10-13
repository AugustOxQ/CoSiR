"""Build the AFF training companion from training_facts.md, using matplotlib.

Run this file to regenerate aff_training.png in this directory. All text is
manually wrapped; coordinates are inches on the 22 × 11 inch canvas.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath


HERE = Path(__file__).resolve().parent
INK, SECONDARY = "#0b0b0b", "#52514e"
GREY, GREY_FILL = "#8a8984", "#f0efed"
BLUE, BLUE_FILL = "#2a78d6", "#e6f0fb"
GREEN = "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
fig = plt.figure(figsize=(22, 11), dpi=150, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 22), ylim=(0, 11))
ax.axis("off")
boxes = {}
text_checks = []


def text(x, y, value, size=10.5, color=INK, bold=False, **kwargs):
    artist = ax.text(x, y, value, fontsize=size, color=color,
                     fontweight="bold" if bold else "normal",
                     ha="left", va="top", linespacing=1.22, **kwargs)
    return artist


def box(key, x, y, width, height, title, body, role="shared", size=10.5):
    edge, fill = {
        "frozen": (GREY, GREY_FILL), "base": (GREY, GREY_FILL),
        "steer": (BLUE, BLUE_FILL), "shared": (SECONDARY, "white"),
        "control": (GREEN, "#edf9f4"),
        "test": (SECONDARY, "white"),
    }[role]
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0,rounding_size=0.11",
        linewidth=1.8, edgecolor=edge, facecolor=fill,
        linestyle="--" if role == "test" else "-", zorder=3))
    title_artist = text(x + .14, y + height - .13, title, size=12,
                        color=edge if role in ("steer", "control") else INK,
                        bold=True, zorder=4)
    body_artist = text(x + .14, y + height - .43, body, size=size, zorder=4)
    boxes[key] = (x, y, width, height)
    text_checks.extend([(key, title_artist), (key, body_artist)])


def route(points, color=SECONDARY, arrow=True, width=1.4, clearance=True):
    """Orthogonal routes with white clearance at unrelated line crossings."""
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    artist = FancyArrowPatch(path=path, arrowstyle="-|>" if arrow else "-",
                             mutation_scale=11, linewidth=width, color=color,
                             capstyle="round", joinstyle="round", zorder=5)
    if clearance:
        artist.set_path_effects([pe.Stroke(linewidth=width + 2.5, foreground="white"),
                                pe.Normal()])
    ax.add_patch(artist)


def dot(x, y, color=SECONDARY):
    ax.plot(x, y, "o", markersize=3.3, color=color, zorder=6)


text(.30, 10.80,
     "How AFF is trained: label-free groupings, heads and a reader fitted on training rows; "
     "weights picked on development episodes; then frozen", size=13.5)

columns = [(.30, 2.75, "1  Frozen inputs", "(never trained)"),
           (3.75, 2.80, "2  Groupings", "(label-free, seed 42)"),
           (7.20, 4.05, "3  Fitted on scorer-train", ""),
           (12.05, 5.00, "4  Picked on development episodes", ""),
           (18.00, 3.65, "5  Frozen, then tested", "")]
for x, width, title, subtitle in columns:
    text(x, 10.27, title, size=12, bold=True)
    if subtitle:
        text(x, 10.00, subtitle, size=10.5, color=SECONDARY)
    ax.plot([x, x + width], [9.77, 9.77], color="#e4e3df", lw=1)
text(12.05, 10.00, "Seed 42 · 12,288 episodes · cross-fitted by halves",
     size=10, color=SECONDARY)

box("rows", .30, 8.28, 2.75, 1.07, "ArtELingo scorer-train",
    "183,694 painting-caption rows\nRows split by painting.", "frozen")
box("go", .30, 6.70, 2.75, 1.23, "GoEmotions",
    "Public RoBERTa; Reddit-trained.\n28 emotion probabilities\nper caption; used offline.", "frozen")
box("clip", .30, 4.43, 2.75, 1.54, "CLIP ViT-B/32",
    "Frozen 512-d image and\ncaption features.\nUnit-normalised before\nevery head.", "frozen")

box("affect", 3.75, 8.05, 2.80, 1.30, "affect",
    "Leiden on a 20-NN graph of\nGoEmotions probabilities.\n41 groups; <200 rows merged.", "steer", size=10)
box("akm", 3.75, 6.25, 2.80, 1.20, "affect-km",
    "k-means 64 on GoEmotions\nprobabilities.\nB only.", "base")
box("image", 3.75, 4.55, 2.80, 1.10, "image",
    "k-means 64 on\nCLIP image features.")
box("caption", 3.75, 2.85, 2.80, 1.10, "caption",
    "k-means 64 on\nCLIP caption features.")

box("heads", 7.20, 7.92, 4.05, 1.43, "Heads: 8 logistic regressions",
    "4 groupings × image/caption; target = group id.\n"
    "C = 1; 300 iterations; 60,000-row draw (seed 0).\n"
    "Output p_h(x); s_h(q,k) = p_h(q)·p_h(k).\n"
    "Pair agreement a_h(i,t) = p_h(i)·p_h(t).", size=10)
box("factor", 7.20, 6.00, 4.05, 1.33, "Factor model A3 (earlier method A)",
    "2,000 steps; model seed 42.\n65,536 pseudo-aspect episodes from k-means:\n"
    "affect-km, image, caption.\nGives the centred uniform factor term.", "base", size=10)
box("practice", 7.20, 4.40, 4.05, 1.26, "Reader's practice episodes",
    "AFF's groupings: affect, image, caption.\n4 supports share a group of one grouping;\n"
    "4 contrasts share a group of another.\n49,152 per painting half.", "steer", size=10)
box("reader", 7.20, 2.12, 4.05, 1.84, "Reader",
    "Heads re-fitted on each painting half.\n18 numbers in (6 per grouping).\n"
    "StandardScaler + multinomial logistic regression.\nC by 5-fold CV (half 0: 1; half 1: 100).\n"
    "Two half-readers averaged into P(h).\nTop-two margin m from P(h).", "steer", size=10)
box("term", 7.20, .91, 4.05, .76, "Steering term T",
    "T = Σ_h P(h)·s_h", "steer", size=11)

box("base", 12.05, 7.84, 5.00, 1.51, "B: condition-free base",
    "z(cos) + λ_u·z(factor term) + λ_a·z(head agreement)\n"
    "Head agreement averaged over affect-km, image, caption.\n56 settings (7 × 8 weights); pick best R@1.\n"
    "B′(A0): head agreement over AFF's own three groupings.", "base")
box("tau", 12.05, 6.70, 5.00, .76, "τ: four threshold choices",
    "0/25/50/75th percentiles of R1's top-two margins.", "steer")
box("gate", 12.05, 5.56, 5.00, .76, "Gate: the one-sided rule",
    "Open if m ≥ τ AND top grouping = affect.", "steer")
box("aff", 12.05, 2.77, 5.00, 2.22, "AFF: fused score and pick",
    "S = (1 + λ_u)·z(B) + λ_a·gate·z(T)\n"
    "z = z-score over the 13 candidates; gate closed: B.\n"
    "224 settings (4 τ × 7 λ_u × 8 λ_a).\n"
    "λ_u: 0, 0.5, 1, 2, 4, 8, 16\nλ_a: 0, 0.25, 0.5, 1, 2, 4, 8, 16\n"
    "Pick = largest min(R@1 gain over B,\ncondition gain).", "steer")
box("control", 12.05, .91, 5.00, 1.35, "Matched control",
    "Same 224 settings; gated term averaged over the\ntwo conditions. Each half picks by R@1.\n"
    "Every pick: tune one parity half, score the other;\nno held row is read.", "control")

box("frozen", 18.00, 6.83, 3.65, 2.52, "AFF's seed-42 settings",
    "Tune half 0 → score parity 1:\nτ_0, λ_u = 4, λ_a = 16\n\n"
    "Tune half 1 → score parity 0:\nτ_2, λ_u = 0, λ_a = 16\n\n"
    "τ_0 ≈ 0.00004; τ_2 = 0.480\nUsed unchanged on the held read.", "steer", size=11)
box("fresh", 18.00, 4.53, 3.65, 1.70, "Fresh test",
    "Selection rows; seeds 49 to 51.\n36,864 episodes. Reader, heads,\nτ and grid frozen; weights\nre-picked by halves on each seed.\n"
    "R@1: AFF 18.88 vs B′(A0) 18.29", "test", size=11)
box("held", 18.00, 1.95, 3.65, 1.98, "Held read",
    "12,281 held paintings.\nSeeds 52 to 54; 36,864 episodes.\n\n"
    "R@1: AFF 19.44 vs B′(A0) 18.84\n+0.60 [+0.46, +0.73]", "test", size=11)

# CLIP bypasses the grouping column to reach all eight heads.
route([(3.05, 5.45), (3.20, 5.45), (3.20, 9.57), (9.23, 9.57), (9.23, 9.35)])
route([(3.20, 5.12), (3.75, 5.12)])
route([(3.20, 5.45), (3.20, 3.40), (3.75, 3.40)])
dot(3.20, 5.45)
dot(3.20, 5.12)
# Frozen GoEmotions outputs feed the two affect partitions only.
route([(3.05, 7.30), (3.51, 7.30), (3.51, 8.72), (3.75, 8.72)], BLUE)
route([(3.51, 7.30), (3.51, 6.85), (3.75, 6.85)], GREY)
dot(3.51, 7.30)

# Scorer-train rows are the fitting data, independently of frozen CLIP features.
route([(1.68, 9.35), (1.68, 9.68), (8.00, 9.68), (8.00, 9.35)], GREY)

# One grouping bus; labelled arrows state each destination's input subset.
route([(6.70, 3.40), (6.70, 9.40)], arrow=False, clearance=False)
for y in (8.72, 6.85, 5.12, 3.40):
    route([(6.55, y), (6.70, y)], arrow=False, clearance=False)
route([(6.70, 9.40), (7.05, 9.40), (7.05, 9.12), (7.20, 9.12)], clearance=False)
text(6.85, 9.54, "all 4 groupings", size=9, color=SECONDARY, zorder=6)
route([(6.70, 7.50), (10.85, 7.50), (10.85, 7.33)], GREY, clearance=False)
text(7.35, 7.78, "affect-km, image, caption", size=9, color=SECONDARY, zorder=6)
route([(6.70, 5.77), (10.85, 5.77), (10.85, 5.66)], BLUE, clearance=False)
text(7.35, 5.98, "affect, image, caption", size=9, color=BLUE, zorder=6)

# Heads supply agreement features to the reader and T, and to B.
route([(11.25, 8.66), (12.05, 8.66)], GREY)
route([(11.25, 8.10), (11.43, 8.10), (11.43, 3.08), (11.25, 3.08)])
route([(11.43, 3.08), (11.43, 1.29), (11.25, 1.29)], BLUE)
dot(11.43, 3.08)
route([(11.25, 6.65), (11.66, 6.65), (11.66, 8.10), (12.05, 8.10)], GREY)
route([(9.23, 4.40), (9.23, 3.96)], BLUE)
route([(9.23, 2.12), (9.23, 1.67)], BLUE)
route([(11.25, 3.57), (11.84, 3.57), (11.84, 5.93), (12.05, 5.93)], BLUE)
# T and B feed the fused score; τ and reader probabilities define the gate.
route([(11.25, 1.15), (11.93, 1.15), (11.93, 3.13), (12.05, 3.13)], BLUE)
route([(14.55, 6.70), (14.55, 6.32)], BLUE)
route([(14.55, 5.56), (14.55, 4.99)], BLUE)
route([(14.55, 2.77), (14.55, 2.26)], GREEN)
route([(17.05, 8.66), (17.25, 8.66), (17.25, 4.48), (17.05, 4.48)], GREY)
# The fresh test re-picks the weights on each seed; only the held read reuses the seed-42 settings.
route([(17.05, 3.70), (17.65, 3.70), (17.65, 7.84), (18.00, 7.84)], BLUE)
route([(17.65, 5.38), (18.00, 5.38)], BLUE)
route([(21.65, 7.12), (21.83, 7.12), (21.83, 2.94), (21.65, 2.94)], BLUE)

# One-line role legend, consistent with the inference companion's palette.
legend = [(0.30, GREY, GREY_FILL, "Frozen inputs / condition-free base"),
          (5.95, BLUE, BLUE_FILL, "AFF steering"),
          (9.15, SECONDARY, "white", "Shared heads / groupings"),
          (13.60, GREEN, "#edf9f4", "Matched control"),
          (17.20, SECONDARY, "white", "Tests")]
for x, edge, fill, label in legend:
    ax.add_patch(Rectangle((x, .29), .23, .15, ec=edge, fc=fill, lw=1.5,
                           linestyle="--" if label == "Tests" else "-"))
    text(x + .34, .46, label, size=10.5, color=SECONDARY)

# Check actual glyph bounds as well as the manually chosen coordinates.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for key, artist in text_checks:
    x, y, w, h = boxes[key]
    bounds = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
    if not (x + .08 <= bounds.x0 and bounds.x1 <= x + w - .08
            and y + .07 <= bounds.y0 and bounds.y1 <= y + h - .07):
        print(f"Text exceeds padding in {key}: {artist.get_text()!r}")
fig.savefig(HERE / "aff_training.png", dpi=150, facecolor="white")
plt.close(fig)
