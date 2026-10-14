import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

TEAL, ORANGE, GREY = "#1baf7a", "#eb6834", "#8a8984"
INK, INK2, LINE = "#0b0b0b", "#52514e", "#e4e3df"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aff_structure.png")

fig = plt.figure(figsize=(8, 3.25), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 80)
ax.set_ylim(0, 32.5)
ax.axis("off")


def box(x, w, y=3, h=20, ec=LINE, lw=1.2, fc="white"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
        ec=ec, fc=fc, lw=lw,
    ))


def arrow(x0, x1, y=13, c=INK2):
    ax.add_patch(FancyArrowPatch(
        (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=12,
        color=c, lw=1.6,
    ))


def t(x, y, s, c=INK, fs=11, w="bold", ha="center"):
    ax.text(x, y, s, color=c, fontsize=fs, fontweight=w,
            ha=ha, va="center")


t(40, 29.5,
  "Our current method (AFF): add the emotion term only when the reader is confident",
  fs=11.5)

X = [1, 21, 43, 63]
W = [17, 19, 17, 16]
heads = ["1  CLIP features", "2  Reader", "3  Steer?", "4  Rank"]
for x, width, heading in zip(X, W, heads):
    box(x, width)
    t(x + width / 2, 21, heading, fs=11)
for i in range(3):
    arrow(X[i] + W[i] + 0.3, X[i + 1] - 0.3)

# Input keys match the sister figure exactly.
items = [("Query", INK), ("Example pairs", TEAL),
         ("Contrast pairs", ORANGE), ("Candidates", GREY)]
for k, (label, colour) in enumerate(items):
    y = 17 - k * 3.6
    ax.add_patch(Rectangle((X[0] + 1.2, y - 1.1), 2.2, 2.2,
                           fc=colour, ec="none"))
    t(X[0] + 4.3, y, label, fs=10.5, w="normal", ha="left")

# Reader question above the three grouping choices.
t(X[1] + W[1] / 2, 17,
  "Which grouping do\nthe example pairs\nshare?", fs=10.5, w="normal")
for k, name in enumerate(["emotion", "image", "caption"]):
    y = 12.6 - k * 3.6
    colour = TEAL if k == 0 else GREY
    box(X[1] + 1, W[1] - 2, y - 1.6, 3.2,
        ec=colour, lw=2 if k == 0 else 1.2)
    t(X[1] + W[1] / 2, y, name, c=colour, fs=11)

# Steering gate, with both outcomes spelled out.
t(X[2] + W[2] / 2, 17,
  "Only if confident\nand the pick is\nemotion", fs=10.5, w="normal")
t(X[2] + W[2] / 2, 10.5,
  "yes: add the\nemotion term", c=TEAL, fs=10.5, w="normal")
t(X[2] + W[2] / 2, 5.8,
  "no: condition-free\nbar alone", c=GREY, fs=10.5, w="normal")

# Final ranking uses the same bar geometry as the sister figure.
t(X[3] + W[3] / 2, 17,
  "Condition-free bar,\nplus the emotion\nterm", fs=10.5, w="normal")
for k in range(3):
    ax.add_patch(Rectangle((X[3] + 2, 12.2 - k * 2.4),
                           12 - k * 3, 1.8,
                           fc=TEAL if k == 0 else GREY, ec="none"))

fig.savefig(out, dpi=200, facecolor="white")
plt.close(fig)
