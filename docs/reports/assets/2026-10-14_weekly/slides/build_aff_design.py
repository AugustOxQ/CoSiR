#!/usr/bin/env python3
"""Build the retrieval-method slide using only Matplotlib primitives.

Run: /root/miniconda3/envs/CoSiR/bin/python build_aff_design.py
Output: aff_design.png, 2666 x 1500 pixels (13.33 x 7.5 inches, 200 dpi).
"""
from pathlib import Path
import os

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, FancyArrowPatch

W, H, DPI = 13.33, 7.5, 200
INK = "#0b0b0b"
SECONDARY = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
BLUE_LIGHT = "#e6f0fb"
BASE = "#8a8984"
BASE_LIGHT = "#f1f0ed"
CLUSTERS = ("#b79360", "#77958a", "#ad8297")
plt.rcParams.update({"font.family": "Ubuntu Condensed", "font.size": 13,
                     "text.color": INK, "axes.linewidth": 0.8})
fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(0, H))
ax.set_axis_off()
labels = []

def text(x, y, s, size=13, color=INK, ha="center", va="center", weight="normal", **kw):
    assert size >= 13, s
    t = ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                fontweight=weight, linespacing=1.13, zorder=kw.pop("zorder", 8), **kw)
    labels.append(t)
    return t

def box(x, y, w, h, fill="white", edge=INK, lw=1.15, radius=.10, z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z)
    ax.add_patch(p)
    return p

def number(n, x, y):
    ax.add_patch(Circle((x, y), .145, facecolor=INK, edgecolor="white", lw=1, zorder=10))
    text(x, y, str(n), 13, "white", weight="bold", zorder=11)

def line(points, color=SECONDARY, lw=1.3, dash=None, z=3):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=lw, linestyle=dash or "-", zorder=z,
            solid_capstyle="round", solid_joinstyle="round")

def arrow(points, color=SECONDARY, lw=1.5, z=4):
    if len(points) > 2:
        line(points[:-1], color, lw, z=z)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                mutation_scale=11, linewidth=lw, color=color,
                                shrinkA=0, shrinkB=0, zorder=z))

def painting(x, y, w=.42, h=.31):
    box(x, y, w, h, "#f8f7f3", SECONDARY, .9, .035, z=5)
    ax.add_patch(Circle((x+w*.76, y+h*.73), h*.085, facecolor="#b79360", edgecolor="none", zorder=6))
    ax.add_patch(Polygon([(x+w*.08,y+h*.13), (x+w*.35,y+h*.64),
                          (x+w*.53,y+h*.32), (x+w*.68,y+h*.51),
                          (x+w*.93,y+h*.13)], closed=True,
                         facecolor="#77958a", edgecolor="none", zorder=6))

def caption(x, y, w=.35, h=.27, edge=SECONDARY, fill="white"):
    box(x, y, w, h, fill, edge, .8, .025, z=5)
    for k, frac in enumerate((.74, .59, .69)):
        line([(x+w*.16,y+h*(.74-k*.23)), (x+w*frac,y+h*(.74-k*.23))],
             BASE, .8, z=6)

def pairs(y):
    for x in (.29, .90, 1.51, 2.12):
        painting(x, y, .28, .24)
        caption(x+.32, y, .24, .24)

def bracket(y):
    line([(.31,y+.08),(.31,y),(2.65,y),(2.65,y+.08)], BASE, 1)

def cloud(cx, cy, scale=1):
    # Three visually separate, unlabelled clusters within each grouping.
    centers = [(-.16,.055), (.01,-.045), (.17,.045)]
    for c, (dx, dy) in zip(CLUSTERS, centers):
        for ox, oy in ((-.035,0),(.026,.035),(.022,-.037)):
            ax.add_patch(Circle((cx+(dx+ox)*scale, cy+(dy+oy)*scale),
                                .027*scale, facecolor=c, edgecolor="none", zorder=6))

# Title and overall columns.
text(.27, 7.13,
     "A base score ranks the candidates; an emotion bonus is added only when the examples clearly share an emotion",
     20, ha="left")
text(.29, 6.51, "One episode", 18, ha="left")
text(12.45, 6.51, "Ranking", 18)
line([(2.82,1.01),(2.82,6.35)], GRID, 1)

# The episode is a collection of icons, without enclosing text boxes.
text(.29, 6.02, "query", 13, ha="left")
painting(.48, 5.20, 1.12, .66)
text(.29, 4.89, "4 example pairs", 16, ha="left")
pairs(4.39)
bracket(4.24)
text(1.48, 3.91, "share one aspect, never named", 13)
text(.29, 3.44, "4 contrast pairs", 16, ha="left")
pairs(2.94)
bracket(2.79)
text(1.48, 2.53, "share another aspect", 13)
text(.29, 1.96, "13 candidates", 16, ha="left")
for dx, dy in ((.12,.12),(.06,.06),(0,0)):
    caption(.39+dx, 1.20+dy, .51, .36)
text(1.21, 1.38, "captions to rank", 13, ha="left")

# Episode feeds both lanes via a short rail outside the icon column.
line([(2.97,2.84),(2.97,5.59)], SECONDARY, 1.3)
arrow([(2.82,5.59),(3.20,5.59)], BASE)
arrow([(2.82,2.84),(3.20,2.84)], BLUE)

# Always-on base lane.
box(3.04, 4.69, 8.63, 1.80, BASE_LIGHT, "none", radius=.12, z=1)
text(3.22, 6.32, "always on", 13, SECONDARY, ha="left")
box(3.20, 4.90, 8.27, 1.18, "white", BASE, radius=.08)
number(3, 3.37, 6.04)
text(3.70, 5.90, "Base score B", 18, ha="left")
chips = [(3.73,2.10,"CLIP similarity"), (6.42,1.90,"factor match"), (8.93,2.10,"group match")]
for x, w, label in chips:
    box(x, 5.30, w, .38, BASE_LIGHT, GRID, .9, .06)
    text(x+w/2, 5.49, label, 14)
text(6.10,5.49,"+",17,SECONDARY)
text(8.64,5.49,"+",17,SECONDARY)
text(7.34,5.06,"uses the examples, ignores which aspect they show",13,SECONDARY)

# Steering lane and its five stages.
box(3.04, 1.14, 8.63, 3.33, BLUE_LIGHT, "none", radius=.12, z=1)
text(3.22, 4.28, "steering, used only when the gate opens",13,BLUE,ha="left")
Y, BH = 1.94, 2.00
cards = [(3.20,1.72,1,"Groupings",INK),
         (5.15,1.38,2,"Heads",INK),
         (6.76,1.65,4,"Reader",BLUE),
         (10.10,1.53,6,"Emotion bonus",BLUE)]
for x, w, n, label, edge in cards:
    box(x,Y,w,BH,"white",edge,radius=.085)
    number(n,x+.15,Y+BH-.02)
    text(x+w/2,3.57,label,16)

for yy, label in ((3.17,"emotion"),(2.78,"image"),(2.39,"caption")):
    cloud(3.58, yy, .91)
    text(3.91,yy,label,13,ha="left")
text(4.06,1.57,"clusters of training\ndata, no labels",13,SECONDARY)

painting(5.34,2.85,.32,.28)
arrow([(5.72,2.99),(6.07,2.99)],SECONDARY,1.1)
for dx, dy in ((-.045,0),(.025,.04),(.033,-.039),(.093,.015),(.092,-.053)):
    ax.add_patch(Circle((6.15+dx,2.99+dy),.022,
                        facecolor=CLUSTERS[1],edgecolor="none",zorder=6))
text(5.84,1.57,"place any new item\nin a group",13,SECONDARY)

# Same heads also provide the base lane's group match. The route uses the
# open corridor above the row and ends under the corresponding chip.
line([(5.84,3.94),(5.84,4.12),(10.68,4.12),(10.68,5.19),
      (9.98,5.19),(9.98,5.30)],BASE,1.0,"--",z=3)

# Reader: three bars, with emotion clearly dominant.
for yy, label, length, color in ((3.18,"emotion",1.14,BLUE),
                                 (2.77,"image",.39,BASE),
                                 (2.36,"caption",.29,BASE)):
    text(6.94, yy+.075, label,13,ha="left")
    box(6.94, yy-.15, length, .085, color,"none",radius=.025,z=5)
text(7.585,1.57,"which grouping do the\nexample pairs share?",13,SECONDARY)

# Gate diamond, numbered like the other blocks.
number(5,8.76,3.92)
text(9.25,3.57,"Gate",16)
ax.add_patch(Polygon([(9.25,3.39),(9.95,2.84),(9.25,2.29),(8.55,2.84)],
                     closed=True,facecolor="white",edgecolor=BLUE,lw=1.3,zorder=5))
text(9.25,2.84,"emotion, and\nconfident?",13)

# Bonus glyphs: only captions in the query's emotion group receive '+'.
for yy, bonus in ((3.09,True),(2.66,False),(2.23,True)):
    caption(10.41,yy,.59,.29,edge=BLUE if bonus else SECONDARY)
    if bonus:
        ax.add_patch(Circle((11.12,yy+.145),.105,facecolor=BLUE,edgecolor="white",lw=.7,zorder=7))
        text(11.12,yy+.145,"+",13,"white")
text(10.865,1.57,"lift candidates in the\nquery’s emotion group",13,SECONDARY)

for a,b in ((4.92,5.15),(6.53,6.76),(8.41,8.55)):
    arrow([(a,2.84),(b,2.84)],BLUE)
arrow([(9.95,2.84),(10.10,2.84)],BLUE)
text(9.80,3.14,"yes",13,BLUE)

# Three non-crossing routes into the sum node: base, gated bonus, bypass.
arrow([(11.47,5.59),(12.45,5.59),(12.45,5.39)],BASE,1.6)
arrow([(11.63,2.84),(11.76,2.84),(11.76,5.20),(12.25,5.20)],BLUE,1.6)
arrow([(9.25,2.29),(9.25,.96),(12.98,.96),(12.98,5.20),(12.65,5.20)],BLUE,1.4)
text(9.36,2.08,"no",13,BLUE,ha="left")
text(10.26,1.035,"B alone",13,SECONDARY,va="bottom")
ax.add_patch(Circle((12.45,5.20),.19,facecolor="white",edgecolor=INK,lw=1.3,zorder=6))
text(12.45,5.20,"+",18)
arrow([(12.45,5.01),(12.45,4.80)],INK,1.4)
text(12.45,4.65,"top 5 of 13",13,SECONDARY)

# Top five of the thirteen candidates. One blue caption rises in the list.
for i, width in enumerate((.78,.78,.71,.76,.66)):
    yy = 4.22-i*.32
    highlight = i == 1
    box(12.06,yy,width,.23,BLUE_LIGHT if highlight else BASE_LIGHT,
        BLUE if highlight else GRID,.9,.035,z=5)
    line([(12.18,yy+.115),(12.06+width-.12,yy+.115)],BLUE if highlight else BASE,.9,z=6)
arrow([(11.91,3.58),(11.91,4.03)],BLUE,1.3)
text(12.45,2.54,"final ranking",13)

# One-line legend.
legend_y=.39
for x, fill, edge, label in ((2.06,BASE_LIGHT,BASE,"base score, always on"),
                            (5.05,BLUE_LIGHT,BLUE,"steering, only when the gate opens"),
                            (9.22,"white",INK,"shared parts")):
    box(x,legend_y-.08,.25,.16,fill,edge,.9,.02)
    text(x+.36,legend_y,label,13,SECONDARY,ha="left")

def validate_bounds():
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t in labels:
        bb = t.get_window_extent(renderer)
        if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > W*DPI or bb.y1 > H*DPI:
            raise ValueError(f"Text exceeds canvas: {t.get_text()!r}")
    print(f"Validated {len(labels)} text labels; minimum font {min(t.get_fontsize() for t in labels):g} pt.")

if __name__ == "__main__":
    validate_bounds()
    out = HERE / "aff_design.png"
    fig.savefig(out, dpi=DPI, facecolor="white")
    print(f"Saved {out} ({round(W*DPI)} x {round(H*DPI)} pixels)")
