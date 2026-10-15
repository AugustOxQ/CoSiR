import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

TEAL, ORANGE, GREY = "#1baf7a", "#eb6834", "#8a8984"
INK, INK2, LINE = "#0b0b0b", "#52514e", "#e4e3df"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "head_reader.png")

fig = plt.figure(figsize=(8, 3.25), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 80); ax.set_ylim(0, 32.5); ax.axis("off")

def box(x, w, y=3, h=20, ec=LINE, lw=1.2, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                ec=ec, fc=fc, lw=lw))
def arrow(x0, x1, y=13, c=INK2):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=12, color=c, lw=1.6))
def t(x, y, s, c=INK, fs=11, w="bold", ha="center"):
    ax.text(x, y, s, color=c, fontsize=fs, fontweight=w, ha=ha, va="center")

t(40, 29.5, "The head reader: find the grouping the example pairs share, then rank by it", fs=11.5)

X = [1, 21, 43, 63]; W = [17, 19, 17, 16]
heads = ["1  CLIP features", "2  Three groupings", "3  Pick", "4  Rank"]
for x, w, h in zip(X, W, heads):
    box(x, w)
    t(x + w/2, 21, h, fs=11)
for i in range(3):
    arrow(X[i] + W[i] + 0.3, X[i+1] - 0.3)

# 1: inputs
items = [("Query", INK), ("Example pairs", TEAL), ("Contrast pairs", ORANGE), ("Candidates", GREY)]
for k, (s, c) in enumerate(items):
    y = 17 - k*3.6
    ax.add_patch(Rectangle((X[0]+1.2, y-1.1), 2.2, 2.2, fc=c, ec="none"))
    t(X[0]+4.3, y, s, c=INK, fs=10.5, w="normal", ha="left")

# 2: groupings
names = ["mood", "image", "caption"]
for k, n in enumerate(names):
    y = 16.5 - k*4
    ec = TEAL if k == 0 else GREY
    box(X[1]+1, W[1]-2, y-1.6, 3.2, ec=ec, lw=2 if k == 0 else 1.2)
    t(X[1]+W[1]/2, y, n, c=ec, fs=11)
t(X[1]+W[1]/2, 4.8, "64 clusters each", c=INK2, fs=10.5, w="normal")

# 3: pick scores
sc = [0.9, 0.35, 0.3]
for k, n in enumerate(names):
    y = 16.5 - k*4
    c = TEAL if k == 0 else GREY
    t(X[2]+1, y, n, c=c, fs=10.5, ha="left")
    ax.add_patch(Rectangle((X[2]+7.5, y-1.2), 8*sc[k], 2.4, fc=c, ec="none"))
t(X[2]+W[2]/2, 4.8, "examples agree,\ncontrast differs", c=INK2, fs=10.5, w="normal")
ax.add_patch(FancyBboxPatch((X[2]+0.4, 14.9), W[2]-0.8, 3.2, boxstyle="round,pad=0,rounding_size=0.6", ec=TEAL, fc="none", lw=1.8))

# 4: rank
t(X[3]+W[3]/2, 17, "Match the query\nin the picked\ngrouping", c=INK, fs=10.5, w="normal")
for k in range(3):
    ax.add_patch(Rectangle((X[3]+2, 12.2-k*2.4), 12-k*3, 1.8, fc=TEAL if k == 0 else GREY, ec="none"))
t(X[3]+W[3]/2, 4.8, "ranked candidates", c=INK2, fs=10.5, w="normal")
fig.savefig(out, dpi=200, facecolor="white")
