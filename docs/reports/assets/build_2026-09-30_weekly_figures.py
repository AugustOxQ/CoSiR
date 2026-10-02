#!/usr/bin/env python3
"""Build the figures for the 2026-09-30 weekly report (Sep 23 to 30).

Figures are written to docs/reports/assets/2026-09-30_weekly/.
Run: python docs/reports/assets/build_2026-09-30_weekly_figures.py [name ...]
With no names, every figure is rebuilt.

Colour convention used in every architecture diagram of this report:
  grey   = an input or component that is the same in both designs
  orange = a component that the new design removes or replaces
  teal   = a component that is new in the design being introduced
  purple = a downstream component kept unchanged
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parent / "2026-09-30_weekly"
SLIDES = False  # set by --slides: date-free figure variants for the slide deck, in OUT_DIR/slides


def S(report_text, slides_text):
    """Pick the report wording or the slide-deck wording."""
    return slides_text if SLIDES else report_text

INK = "#1F2328"
MUTED = "#57606A"
GREY_FILL, GREY_EDGE = "#F2F3F5", "#8C959F"
ORANGE_FILL, ORANGE_EDGE = "#FFF1E5", "#D2691E"
TEAL_FILL, TEAL_EDGE = "#E3F6F3", "#1B8A7A"
PURPLE_FILL, PURPLE_EDGE = "#F0EBFA", "#6F52B5"
LANE_FILL = "#FAFBFC"

STYLES = {
    "grey": (GREY_FILL, GREY_EDGE),
    "orange": (ORANGE_FILL, ORANGE_EDGE),
    "teal": (TEAL_FILL, TEAL_EDGE),
    "purple": (PURPLE_FILL, PURPLE_EDGE),
}

plt.rcParams["font.family"] = "DejaVu Sans"


def canvas(width, height, scale=0.1):
    """Axes in data units (1 unit = scale inches) with no frame."""
    fig = plt.figure(figsize=(width * scale, height * scale), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, body="", style="grey", title_size=9.5, body_size=8.2,
        dashed=False):
    """Rounded box with a bold title line and an optional smaller body."""
    fill, edge = STYLES[style]
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
        facecolor=fill, edgecolor=edge, linewidth=1.6,
        linestyle=(0, (4, 2)) if dashed else "-", zorder=2,
    )
    ax.add_patch(patch)
    cx = x + w / 2
    if body:
        # centre the title + body block vertically (units: 1 unit = 7.2 pt)
        title_h = title_size / 7.2
        gap = 1.1
        body_h = (body.count("\n") + 1) * body_size * 1.35 / 7.2
        top = y + h / 2 + (title_h + gap + body_h) / 2
        ax.text(cx, top, title, ha="center", va="top", fontsize=title_size,
                fontweight="bold", color=INK, zorder=3)
        ax.text(cx, top - title_h - gap, body, ha="center", va="top",
                fontsize=body_size, color=INK, linespacing=1.35, zorder=3)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=INK, linespacing=1.3, zorder=3)
    return (x, y, w, h)


def anchor(b, side, frac=0.5):
    x, y, w, h = b
    if side == "right":
        return (x + w, y + h * frac)
    if side == "left":
        return (x, y + h * frac)
    if side == "top":
        return (x + w * frac, y + h)
    return (x + w * frac, y)


def arrow(ax, p, q, label="", color=INK, dashed=False, rad=0.0, label_xy=None,
          label_size=7.8, lw=1.4, label_color=None):
    patch = FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=11, linewidth=lw, color=color,
        linestyle=(0, (4, 2)) if dashed else "-", connectionstyle=f"arc3,rad={rad}",
        shrinkA=1, shrinkB=1, zorder=1.5,
    )
    ax.add_patch(patch)
    if label:
        lx, ly = label_xy if label_xy else ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 1.2)
        ax.text(lx, ly, label, ha="center", va="bottom", fontsize=label_size,
                color=label_color or color, zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))


def elbow(ax, pts, color=INK, dashed=False, label="", label_xy=None, label_color=None,
          lw=1.4, label_size=7.8):
    """Polyline through pts with an arrow head on the last segment."""
    ls = (0, (4, 2)) if dashed else "-"
    xs, ys = zip(*pts[:-1])
    ax.plot(list(xs) + [pts[-2][0]], list(ys) + [pts[-2][1]], color=color, lw=lw, ls=ls,
            zorder=1.5, solid_capstyle="round")
    arrow(ax, pts[-2], pts[-1], color=color, dashed=dashed, lw=lw)
    if label:
        ax.text(*label_xy, label, ha="center", va="bottom", fontsize=label_size,
                color=label_color or color, zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))


def lane(ax, x, y, w, h, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                facecolor=LANE_FILL, edgecolor="#D0D7DE", linewidth=1.0,
                                zorder=0))
    ax.text(x + 1.5, y + h - 1.4, label, ha="left", va="top", fontsize=10.5,
            fontweight="bold", color=INK, zorder=3)


def legend(ax, x, y, entries, size=8.5):
    for i, (style, text) in enumerate(entries):
        fill, edge = STYLES[style]
        ax.add_patch(FancyBboxPatch((x, y - 0.9), 3.2, 1.8,
                                    boxstyle="round,pad=0,rounding_size=0.4",
                                    facecolor=fill, edgecolor=edge, linewidth=1.4))
        ax.text(x + 4.2, y, text, ha="left", va="center", fontsize=size, color=INK)
        x += 8.0 + len(text) * 0.78


def save(fig, name):
    folder = OUT_DIR / "slides" if SLIDES else OUT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------------
# Figure 1: PercepT (our replication) vs the first buddy model
# --------------------------------------------------------------------------
def fig_percept_vs_buddy():
    W, H = 172, 98
    fig, ax = canvas(W, H)

    # Column headers
    ax.text(12, 95, "Inputs per painting", ha="center", fontsize=10, color=MUTED,
            fontweight="bold")
    ax.text(72, 95, "Stage 1: topic formation (image + captions, no labels)",
            ha="center", fontsize=10, color=MUTED, fontweight="bold")
    ax.text(152, 95, "Stage 2: topic mapping (image only)", ha="center", fontsize=10,
            color=MUTED, fontweight="bold")
    ax.plot([136.5, 136.5], [6, 93], color="#D0D7DE", lw=1.0, ls=(0, (3, 3)), zorder=0)

    # ---------------- lane (a): PercepT ----------------
    lane(ax, 1, 51, 134, 41, "(a) PercepT, our replication (standing config K = 60/40)")
    a_img = box(ax, 3, 76, 19, 8, "CLIP image", "512-d")
    a_txt = box(ax, 3, 66, 19, 8, "CLIP text", "512-d, mean of captions")
    a_aff = box(ax, 3, 53.5, 19, 10.5, "Affect text encoder",
                "RoBERTa (GoEmotions)\ntoken-mean embedding\n768-d", body_size=7.6)

    a_fuse = box(ax, 27, 63, 17, 14, "Fuse",
                 "h_C = L2[img, txt]\nh_E = affect\nL2[h_C, h_C, h_E]\n2,816-d", style="orange")
    a_ae = box(ax, 49, 63, 20, 14, "Autoencoder",
               "encoder -> Z, 128-d\ndecoder rebuilds\nthe 2,816-d input", style="orange")
    a_km = box(ax, 74, 63, 15, 14, "K-means", "init K = 60\ncentres in Z", style="orange")
    a_dec = box(ax, 94, 63, 19, 14, "DEC training",
                "KL self-sharpening\n+ reconstruction\n+ balance term\n(lambda = 1000)",
                style="orange")
    a_top = box(ax, 118, 63, 16, 14, "Topics",
                "prune to 40\ncentres; topic =\nnearest centre", style="orange")

    arrow(ax, anchor(a_img, "right"), anchor(a_fuse, "left", 0.75))
    arrow(ax, anchor(a_txt, "right"), anchor(a_fuse, "left", 0.5))
    arrow(ax, anchor(a_aff, "right"), anchor(a_fuse, "left", 0.12))
    arrow(ax, anchor(a_fuse, "right"), anchor(a_ae, "left"))
    arrow(ax, anchor(a_ae, "right"), anchor(a_km, "left"))
    arrow(ax, anchor(a_km, "right"), anchor(a_dec, "left"))
    arrow(ax, anchor(a_dec, "right"), anchor(a_top, "left"))
    ax.text(81, 55.5, "Z, the centres and the topics are learned together;\n"
            "held-out paintings are encoded and sent to the nearest centre",
            ha="center", va="center", fontsize=7.8, color=ORANGE_EDGE, style="italic")

    # ---------------- lane (b): first buddy model ----------------
    lane(ax, 1, 6, 134, 43, "(b) First buddy model (Attention-h1 student + Leiden)")
    b_img = box(ax, 3, 33, 19, 8, "CLIP image", "512-d")
    b_txt = box(ax, 3, 23.5, 19, 8, "CLIP text", "512-d, mean of captions")
    b_aff = box(ax, 3, 10, 19, 11.5, "Affect classifier",
                "GoEmotions, 28 emotion\nprobabilities, mean\nof captions", style="teal",
                body_size=7.6)

    b_st = box(ax, 30, 26, 40, 17, "Student (Attention-h1)",
               "content: L2[img, txt] -> PCA 50 -> linear -> 32-d\n"
               "affect: 28-d -> linear -> 32-d\n"
               "the two 32-d tokens -> 1-head self-attention\n"
               "-> mean -> LayerNorm -> L2 normalise\n"
               "output z: 32-d", style="teal", body_size=7.9)
    b_cg = box(ax, 30, 8, 19, 12.5, "Content teacher",
               "the buddy graph:\nmutual kNN (K = 20) on\nimage, union text", style="teal",
               body_size=7.6)
    b_ag = box(ax, 51, 8, 19, 12.5, "Affect teacher",
               "mutual kNN (K = 20)\non the 28-d affect\nvectors", style="teal",
               body_size=7.6)
    b_top = box(ax, 104, 25.5, 30, 18, "Topics",
                "mutual kNN (K = 20) on z\n-> Leiden communities\n(19 topics on train)\n"
                "held-out paintings: k-NN\nvote (k = 20) onto the\ntrain topics",
                style="teal", body_size=7.9)

    arrow(ax, anchor(b_img, "right"), anchor(b_st, "left", 0.8))
    arrow(ax, anchor(b_txt, "right"), anchor(b_st, "left", 0.45))
    arrow(ax, anchor(b_aff, "right", 0.85), anchor(b_st, "left", 0.08))
    arrow(ax, anchor(b_cg, "top"), anchor(b_cg, "top")[:1] + (26,), color=TEAL_EDGE,
          dashed=True)
    arrow(ax, anchor(b_ag, "top"), anchor(b_ag, "top")[:1] + (26,), color=TEAL_EDGE,
          dashed=True)
    ax.text(87, 14.2, "each teacher edge is a positive pair;\n"
            "loss = InfoNCE(content edges)\n+ InfoNCE(affect edges), tau = 0.1",
            ha="center", va="center", fontsize=7.8, color=TEAL_EDGE)
    arrow(ax, anchor(b_st, "right"), (104, 34.5), label="z", label_xy=(87, 34.9))

    # ---------------- Stage 2 (shared, unchanged) ----------------
    s2 = box(ax, 139, 36, 31, 28, "Stage 2 mapper",
             "same code for (a) and (b)\n\nCLIP ViT-B/32 patch\ntokens, 50 x 512\n"
             "-> attention pooling\n(1 learned query)\n-> linear head\n-> topic scores\n\n"
             "loss: BCE vs the Stage 1\ntopic of the painting",
             style="purple", body_size=7.9)
    arrow(ax, anchor(a_top, "right"), anchor(s2, "left", 0.8), label="targets",
          label_xy=(136.8, 67.2), color=PURPLE_EDGE)
    arrow(ax, anchor(b_top, "right"), anchor(s2, "left", 0.15), label="targets",
          label_xy=(136.8, 38.0), color=PURPLE_EDGE)
    ax.text(154.5, 29, "input at test time:\nthe painting image only\n\n"
            "metric: held-out\nmacro AUC over topics",
            ha="center", va="center", fontsize=7.8, color=MUTED)

    legend(ax, 3, 2.2, [
        ("grey", "same input in both"),
        ("orange", "PercepT part that (b) replaces"),
        ("teal", "new or changed in (b)"),
        ("purple", "kept unchanged"),
    ])
    save(fig, "percept_vs_first_buddy_model")


# --------------------------------------------------------------------------
# Chart palette (validated with the dataviz validator, light mode, all pairs)
# --------------------------------------------------------------------------
BUDDY_C = "#1baf7a"     # aqua: buddy-side methods
PERCEPT_C = "#eb6834"   # orange: PercepT replication
REF_C = "#8C959F"       # neutral grey: reference points
GRID_C = "#E6E8EB"
DATA_JSON = OUT_DIR / "data" / "community_composition_and_occupancy.json"


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#B8BEC6")
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(color=GRID_C, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------
# PercepT Stage 1 and its three loss terms
# --------------------------------------------------------------------------
def fig_percept_losses():
    W, H = 172, 80
    fig, ax = canvas(W, H)
    ax.text(86, 77.5, "PercepT Stage 1 as we replicated it: what is trained, and by which losses",
            ha="center", fontsize=10.5, fontweight="bold", color=INK)

    i_img = box(ax, 3, 60, 19, 8, "CLIP image", "512-d")
    i_txt = box(ax, 3, 50.5, 19, 8, "CLIP text", "512-d, mean of captions")
    i_aff = box(ax, 3, 38.5, 19, 10, "Affect text encoder",
                "RoBERTa (GoEmotions)\ntoken-mean, 768-d", body_size=7.6)
    fuse = box(ax, 27, 47, 16, 16, "Fuse: x", "L2[h_C, h_C, h_E]\ncontent : affect\n= 2 : 1\n2,816-d",
               style="orange", body_size=7.8)
    enc = box(ax, 48, 48.5, 14, 13, "Encoder", "2,816 -> 128", style="orange", body_size=7.8)
    z = box(ax, 67, 48.5, 13, 13, "Z", "128-d latent", style="orange", body_size=7.8)
    dec = box(ax, 64, 33, 19, 10, "Decoder", "rebuilds x from Z", style="orange", body_size=7.6)
    q = box(ax, 85, 46, 24, 17, "Soft assignment q",
            "Student's t similarity of\neach point in Z to K\ncentres (start: K-means,\nK = 60)",
            style="orange", body_size=7.6)
    p = box(ax, 114, 46, 21, 17, "Target p",
            "q squared, divided\nby cluster size, then\nrenormalised: confident\nassignments count more", style="orange",
            body_size=7.6)
    top = box(ax, 140, 46, 30, 17, "After training",
              "prune 60 -> 40 centres;\ntopic = most likely centre;\n"
              "held-out paintings go\nthrough the same encoder", style="orange", body_size=7.6)

    for src, frac in ((i_img, 0.8), (i_txt, 0.5), (i_aff, 0.2)):
        arrow(ax, anchor(src, "right"), anchor(fuse, "left", frac))
    arrow(ax, anchor(fuse, "right"), anchor(enc, "left"))
    arrow(ax, anchor(enc, "right"), anchor(z, "left"))
    arrow(ax, anchor(z, "bottom"), (73.5, 43))
    arrow(ax, anchor(z, "right"), anchor(q, "left"))
    arrow(ax, anchor(q, "right"), anchor(p, "left"))
    arrow(ax, anchor(p, "right"), anchor(top, "left"))

    # loss row: each loss sits under the parts it reads
    ax.add_patch(FancyBboxPatch((25, 3), 112, 28, boxstyle="round,pad=0,rounding_size=1.5",
                                facecolor=LANE_FILL, edgecolor="#D0D7DE", linewidth=1.0, zorder=0))
    l_rec = box(ax, 28, 16, 33, 12, "Reconstruction (paper)",
                "|| x - decoder(encoder(x)) ||^2,\nweight 1: keeps Z faithful to x",
                style="orange", body_size=7.6)
    l_bal = box(ax, 86, 16, 22, 12, "Balance term (ours)",
                "KL(mean q || uniform),\nweight 1000: stops\nclusters emptying",
                style="orange", body_size=7.4, dashed=True)
    l_dec = box(ax, 112, 16, 23, 12, "DEC loss (paper)",
                "KL(P || Q): pulls each\npainting towards its\nmost likely centre",
                style="orange", body_size=7.4)
    arrow(ax, anchor(fuse, "bottom", 0.5), (35, 28), color=ORANGE_EDGE, dashed=True)
    arrow(ax, anchor(dec, "bottom", 0.2), (55, 28), color=ORANGE_EDGE, dashed=True)
    arrow(ax, anchor(q, "bottom", 0.3), (92.2, 28), color=ORANGE_EDGE, dashed=True)
    arrow(ax, anchor(q, "bottom", 0.85), (116, 28), color=ORANGE_EDGE, dashed=True)
    arrow(ax, anchor(p, "bottom", 0.6), (126.6, 28), color=ORANGE_EDGE, dashed=True)
    ax.text(27, 12.6, "All three terms are summed and trained together, moving the encoder, "
            "the decoder and the centres at once.", ha="left", va="center", fontsize=8,
            color=INK)
    ax.text(27, 7.6, "Without the balance term DEC collapsed: 65 of 67 topics held under 1% of "
            "the paintings.\nThe paper has no such term; we added it after that collapse"
            + S(" (night of 22 to 23 Sep).", "."), ha="left", va="center", fontsize=7.8,
            color=ORANGE_EDGE, style="italic", linespacing=1.3)

    # deviations and later-found bugs
    ax.add_patch(FancyBboxPatch((140, S(3, 20)), 30, S(40, 23), boxstyle="round,pad=0,rounding_size=1.5",
                                facecolor="white", edgecolor="#D0D7DE", linewidth=1.0, zorder=0))
    ax.text(141.8, 41.5, "Where our replication\ndiffers from the paper", ha="left",
            va="top", fontsize=8, fontweight="bold", color=INK, linespacing=1.25)
    ax.text(141.8, 37.2,
            "- RoBERTa instead of the\n  paper's ModernBERT\n"
            "- concatenation instead of\n  a 2:1 weighted sum\n"
            "- stop when < 0.1% of\n  paintings change cluster\n"
            "- the balance term; K = 60/40",
            ha="left", va="top", fontsize=7.3, color=INK, linespacing=1.3)
    if not SLIDES:
        ax.text(141.8, 18.2, "Bugs found on 27 Sep (3.3)", ha="left", va="top", fontsize=8,
                fontweight="bold", color=ORANGE_EDGE)
        ax.text(141.8, 14.6,
                "- pruning kept the highest-\n  norm centres; the paper\n  drops them\n"
                "- reconstruction averaged\n  over 2,816 dims (~2,816x\n  too weak)",
                ha="left", va="top", fontsize=7.3, color=INK, linespacing=1.3)
    save(fig, "percept_stage1_losses")


# --------------------------------------------------------------------------
# Six ways the emotion signal was added on the buddy side (22 Sep)
# --------------------------------------------------------------------------
def fig_affect_routes():
    W, H = 172, 96
    fig, ax = canvas(W, H)
    ax.text(3, 93.5, "Method (where emotion enters)", ha="left", fontsize=9.5, fontweight="bold",
            color=MUTED)
    ax.text(90, 93.5, "Pipeline", ha="center", fontsize=9.5, fontweight="bold", color=MUTED)
    ax.text(160, 93.5, "train AMI\nemotion / genre", ha="center", va="center", fontsize=8.8,
            fontweight="bold", color=MUTED, linespacing=1.2)

    rows = [
        ("(a) Content only", "reference: no emotion input",
         [("CLIP image + text", "grey"), ("buddy graph\nmutual kNN, K = 20", "grey"),
          ("Leiden", "grey"), ("28 communities", "grey")], "0.059 / 0.438"),
        ("(b) Early fusion", "feature level",
         [("CLIP image, CLIP text\n+ w x affect (28-d)", "teal"), ("mutual kNN\ngraph", "grey"),
          ("Leiden", "grey"), ("communities", "grey")], "w = 1: 0.076 / 0.403\nw = 4: 0.116 / 0.087"),
        ("(c) Late fusion", "graph level",
         [("content graph\n+ affect graph", "teal"), ("union of\nthe edges", "teal"),
          ("Leiden", "grey"), ("communities", "grey")], "0.124 / 0.139"),
        ("(d) Hierarchical refinement", "partition level",
         [("content graph\n-> Leiden, 28 parents", "grey"),
          ("affect Leiden inside\neach parent only", "teal"), ("~400 child\ncommunities", "grey")],
         "0.107 / 0.195"),
        ("(e) DEC in place of Leiden", "clustering step, affect only",
         [("affect only\n(28-d)", "teal"), ("autoencoder\n28 -> 16", "teal"),
          ("DEC, K = 28\n(PercepT's clustering)", "teal"), ("clusters", "grey")],
         "0.149 / 0.055\n(Leiden, same input:\n0.118 / 0.040)"),
        ("(f) Two-teacher student", "learned embedding",
         [("CLIP + affect features;\ntwo teacher graphs", "teal"), ("student z, 32-d\n(InfoNCE on edges)", "teal"),
          ("kNN graph on z\n-> Leiden", "grey"), ("communities", "grey")],
         "Linear: 0.128 / 0.280\nAttention-h1:\n0.135 / 0.240"),
    ]
    row_h = 14.2
    for i, (name, where, steps, result) in enumerate(rows):
        y0 = 88 - (i + 1) * row_h
        ax.add_patch(FancyBboxPatch((1, y0 + 0.6), 170, row_h - 1.2,
                                    boxstyle="round,pad=0,rounding_size=1.2",
                                    facecolor=LANE_FILL if i % 2 == 0 else "white",
                                    edgecolor="#E1E4E8", linewidth=0.8, zorder=0))
        ax.text(3, y0 + row_h / 2 + 1.6, name, ha="left", va="center", fontsize=9,
                fontweight="bold", color=INK)
        ax.text(3, y0 + row_h / 2 - 2.0, where, ha="left", va="center", fontsize=7.8,
                color=TEAL_EDGE if i else MUTED, style="italic")
        x = 42
        widths = [26 if len(steps) == 4 else 30] * len(steps)
        prev = None
        for (label, style), w in zip(steps, widths):
            b = box(ax, x, y0 + 2.2, w - 3, row_h - 4.4, label, style=style, title_size=7.6)
            if prev:
                arrow(ax, anchor(prev, "right"), anchor(b, "left"))
            prev = b
            x += w + 2
        ax.text(160, y0 + row_h / 2, result, ha="center", va="center", fontsize=8, color=INK,
                linespacing=1.3)
    legend(ax, 3, 2.5, [("grey", "same as (a)"), ("teal", "what the method adds or changes")])
    save(fig, "affect_routes")


# --------------------------------------------------------------------------
# What the communities contain: emotion and genre lift heatmaps
# --------------------------------------------------------------------------
def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "blue_grey_red", ["#1c5cab", "#86b6ef", "#f0efec", "#f19a98", "#b8302f"])


def fig_community_composition():
    import json
    import numpy as np
    data = json.loads(DATA_JSON.read_text())
    emotions, genres = data["emotions"], data["genres"]
    cmap = _diverging_cmap()
    panels = [("content_only", "Content-only buddy graph, 28 communities",
               "AMI 0.059", "AMI 0.438"),
              ("attention_h1", "Attention-h1, 19 communities",
               "AMI 0.135", "AMI 0.240")]
    fig = plt.figure(figsize=(13.5, 8.6), dpi=200)
    left, gap_x, right = 0.13, 0.06, 0.06
    usable = 1 - left - gap_x - right
    heat_h, bar_h = 0.27, 0.05
    tops = [0.90, 0.44]
    for r, (key, title, emo_ami, gen_ami) in enumerate(panels):
        emo = np.array(data[key]["emotion"]["counts"], dtype=float)   # communities x emotions
        gen = np.array(data[key]["genre"]["counts"], dtype=float)
        order = np.argsort(-emo.sum(1))
        emo, gen = emo[order], gen[order]
        n = len(order)
        width = usable / 2 * n / 28
        for c, (mat, vocab, label, ami) in enumerate(
                ((emo, emotions, "emotion", emo_ami), (gen, genres, "genre", gen_ami))):
            x0 = left + c * (usable / 2 + gap_x)
            base = mat.sum(0) / mat.sum()
            share = mat / np.maximum(mat.sum(1, keepdims=True), 1)
            lift = np.clip(100 * (share - base), -30, 30)
            small = mat.sum(1) < 15
            lift[small, :] = np.nan
            axb = fig.add_axes((x0, tops[r] - bar_h, width, bar_h))
            axb.bar(np.arange(n), mat.sum(1), color=REF_C, width=0.8)
            axb.set_xlim(-0.5, n - 0.5)
            axb.axis("off")
            axb.text(0, 1.25, f"{title}: {label} ({ami})", transform=axb.transAxes,
                     fontsize=9.5, fontweight="bold", color=INK, va="bottom")
            axb.text(1.0, 1.02, f"bars: {'paintings' if c == 0 else 'genre-labelled paintings'}"
                     " per community", transform=axb.transAxes, fontsize=7.5, color=MUTED,
                     ha="right", va="bottom")
            axh = fig.add_axes((x0, tops[r] - bar_h - heat_h - 0.005, width, heat_h))
            masked = np.ma.masked_invalid(lift.T)
            cm = cmap.copy()
            cm.set_bad("#FFFFFF")
            im = axh.imshow(masked, aspect="auto", cmap=cm, vmin=-30, vmax=30,
                            interpolation="nearest")
            for j in np.where(small)[0]:
                axh.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, len(vocab), fill=False,
                                            hatch="////", edgecolor="#C9CED4", linewidth=0))
            axh.set_yticks(range(len(vocab)))
            axh.set_yticklabels([v.replace("_", " ") for v in vocab], fontsize=8, color=INK)
            axh.set_xticks([])
            axh.set_xlabel("communities, largest to smallest", fontsize=8, color=MUTED)
            for s in axh.spines.values():
                s.set_visible(False)
            # thin white separators between cells
            axh.set_xticks(np.arange(-0.5, n, 1), minor=True)
            axh.set_yticks(np.arange(-0.5, len(vocab), 1), minor=True)
            axh.grid(which="minor", color="white", linewidth=1.2)
            axh.tick_params(which="minor", length=0)
    cax = fig.add_axes((0.30, 0.035, 0.40, 0.018))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-30, -15, 0, 15, 30])
    cb.ax.set_xticklabels(["-30 points or less", "-15", "same as overall", "+15",
                           "+30 points or more"], fontsize=8, color=INK)
    cb.outline.set_visible(False)
    fig.text(0.5, 0.075, "Cell colour: share of the label inside the community minus its share "
             "in the whole training set, in percentage points (hatched: fewer than 15 labelled "
             "paintings)",
             ha="center", fontsize=8.5, color=MUTED)
    save(fig, "community_composition")


# --------------------------------------------------------------------------
# Emotion AMI vs genre AMI, buddy side against the PercepT baseline
# --------------------------------------------------------------------------
def _pt(ax, x, y, label, kind, at=None, ha="left", bold=False, size=7.8, xerr=None,
        yerr=None, faint=False):
    """Scatter one point; `at` places its label (a thin leader line is drawn to it)."""
    color, marker = {"buddy": (BUDDY_C, "o"), "percept": (PERCEPT_C, "s"),
                     "ref": (REF_C, "D")}[kind]
    if xerr is not None or yerr is not None:
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor=color,
                    elinewidth=0.9 if faint else 1.3, alpha=0.4 if faint else 1.0,
                    capsize=0 if faint else 3, zorder=3)
    if not faint:
        ax.scatter([x], [y], s=70 if bold else 46, marker=marker, color=color,
                   edgecolor="white", linewidth=1.4, zorder=4)
    if label:
        ax.annotate(label, (x, y), xytext=at, fontsize=size, color=INK, ha=ha, va="center",
                    fontweight="bold" if bold else "normal", zorder=5, linespacing=1.2,
                    arrowprops=dict(arrowstyle="-", color="#B8BEC6", lw=0.7, shrinkA=2,
                                    shrinkB=4))


def _bar(ax, xlim, ylim, note_xy, note_ha):
    ax.axvline(0.1236, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.axhline(0.1954, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.fill_between([0.1236, 1], 0.1954, 1, color="#EAF7F2", zorder=0)
    ax.text(*note_xy, "clears the Pareto bar", fontsize=8, color=TEAL_EDGE, ha=note_ha,
            va="top", fontweight="bold")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("emotion AMI (agreement with the 9 human emotion labels)", fontsize=9,
                  color=INK)
    ax.set_ylabel("genre AMI (agreement with the 9 genre labels)", fontsize=9, color=INK)


def fig_ami_plane():
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 6.6), dpi=200)
    for ax in (a, b):
        style_axes(ax)
    _bar(a, (0.02, 0.18), (0.0, 0.47), (0.177, 0.215), "right")
    _bar(b, (0.100, 0.1345), (0.15, 0.345), (0.1340, 0.343), "right")

    a.set_title(S("(a) Training split: every method from 22 to 24 September", "(a) Training split: every method tried"), fontsize=10.5,
                fontweight="bold", color=INK, loc="left")
    ws = [(0.0593, 0.4384), (0.0614, 0.4045), (0.0759, 0.4025), (0.1119, 0.2042),
          (0.1160, 0.0867)]
    a.plot(*zip(*ws), color=BUDDY_C, lw=1.4, ls=(0, (3, 2)), zorder=3)
    for (x, y), w, off in zip(ws[1:], ("0.5", "1", "2", "4"),
                              ((4, -9), (4, -9), (-26, -3), (5, -6))):
        a.scatter([x], [y], s=22, color=BUDDY_C, zorder=4)
        a.annotate(f"w={w}", (x, y), xytext=off, textcoords="offset points", fontsize=7,
                   color=MUTED)
    a.text(0.080, 0.418, "(b) early fusion, weight w", fontsize=7.8, color=INK)
    _pt(a, 0.0593, 0.4384, "(a) content only", "ref", at=(0.030, 0.458))
    _pt(a, 0.1180, 0.0396, "affect only", "ref", at=(0.128, 0.018))
    _pt(a, 0.1236, 0.1394, "(c) late fusion, union", "buddy", at=(0.131, 0.120))
    _pt(a, 0.0510, 0.2399, "late fusion,\nintersection", "buddy", at=(0.030, 0.172))
    _pt(a, 0.1072, 0.1954, "(d) hierarchical", "buddy", at=(0.080, 0.160), ha="center")
    _pt(a, 0.1492, 0.0554, "(e) DEC on affect only", "buddy", at=(0.150, 0.088), ha="center")
    _pt(a, 0.1284, 0.2799, "Linear", "buddy", at=(0.112, 0.300), ha="right")
    _pt(a, 0.1230, 0.2319, "MLP-64", "buddy", at=(0.100, 0.252), ha="right")
    _pt(a, 0.1142, 0.2039, "MLP-128", "buddy", at=(0.140, 0.178))
    _pt(a, 0.1328, 0.2441, "Attention-h4", "buddy", at=(0.152, 0.270))
    _pt(a, 0.1351, 0.2397, "Attention-h1", "buddy", at=(0.152, 0.238), bold=True)
    _pt(a, 0.1462, 0.3258, "PercepT K = 60/40\n(seed 42)", "percept", at=(0.152, 0.350),
        bold=True)
    _pt(a, 0.1373, 0.3902, "PercepT, paper's schedule", "percept", at=(0.141, 0.425))
    _pt(a, 0.0478, 0.3281, "PercepT without the\nbalance term (collapsed)", "percept",
        at=(0.030, 0.292))

    b.set_title("(b) Held-out split, zoomed on the bar: the fair comparison", fontsize=10.5,
                fontweight="bold", color=INK, loc="left")
    _pt(b, 0.1095, 0.2901, "Linear", "buddy", at=(0.1030, 0.305))
    _pt(b, 0.1046, 0.2087, "MLP-64", "buddy", at=(0.1020, 0.228))
    _pt(b, 0.1114, 0.1604, "MLP-128", "buddy", at=(0.1150, 0.162))
    _pt(b, 0.1216, 0.1875, "Attention-h4", "buddy", at=(0.1140, 0.178), ha="right")
    _pt(b, 0.1249, 0.2404, "Attention-h1\n(seed 42)", "buddy", at=(0.1150, 0.250), ha="right",
        bold=True)
    _pt(b, 0.1242, 0.2517, "", "percept", faint=True,
        xerr=[[0.1242 - 0.1157], [0.1289 - 0.1242]],
        yerr=[[0.2517 - 0.2192], [0.2750 - 0.2517]])
    _pt(b, 0.1252, 0.2486, "PercepT K = 60/40, 4 seeds\n(bars: min to max)", "percept",
        at=(0.1275, 0.292), bold=True, xerr=[[0.1252 - 0.1238], [0.1272 - 0.1252]],
        yerr=[[0.2486 - 0.2328], [0.2617 - 0.2486]])
    b.text(0.1293, 0.228, "faint bars: the same\nconfig over 14 seeds;\n10 of 14 clear the bar",
           fontsize=7.4, color=MUTED, linespacing=1.2)
    _pt(b, 0.1235, 0.2089, "PercepT K = 100/67,\nbalance, 4 seeds", "percept",
        at=(0.1275, 0.180))
    _pt(b, 0.1092, 0.3288, "PercepT, paper's schedule", "percept", at=(0.1125, 0.334))

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BUDDY_C, markersize=7,
                   label="buddy side (ours)"),
        plt.Line2D([], [], marker="s", ls="", color=PERCEPT_C, markersize=7,
                   label="PercepT replication (baseline)"),
        plt.Line2D([], [], marker="D", ls="", color=REF_C, markersize=6, label="reference"),
        plt.Line2D([], [], color=INK, lw=1.0, ls=(0, (4, 3)),
                   label="Pareto bar: emotion > 0.1236 and genre > 0.1954"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "ami_plane_vs_percept")


# --------------------------------------------------------------------------
# Held-out occupancy of topics (section 3.1)
# --------------------------------------------------------------------------
def fig_occupancy():
    import json
    import numpy as np
    data = json.loads(DATA_JSON.read_text())["occupancy"]
    n = data["n_heldout"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2), dpi=200,
                             gridspec_kw={"width_ratios": [28, 67]}, sharey=True)
    for ax, key, color, title in (
            (axes[0], "buddy_knn_vote", BUDDY_C, "Buddy model: 19 topics"),
            (axes[1], "percept_paper_recipe", PERCEPT_C, "PercepT, paper's schedule (collapsed): 67 topics")):
        occ = np.sort(np.array(data[key]))[::-1]
        style_axes(ax)
        ax.grid(axis="x", visible=False)
        shown = np.where(occ > 0, occ, 0.6)
        bars = ax.bar(np.arange(len(occ)), shown, color=np.where(occ > 0, color, "#D0D7DE"),
                      width=0.8, zorder=3)
        ax.set_yscale("log")
        ax.axhline(0.01 * n, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=4)
        ax.set_xlim(-0.8, len(occ) - 0.2)
        ax.set_xticks([])
        ax.set_xlabel("topics, largest to smallest", fontsize=8.5, color=MUTED)
        empty = int((occ == 0).sum())
        below = int((occ < 0.01 * n).sum())
        ax.set_title(f"{title}\nmedian {int(np.median(occ))} paintings per topic\n"
                     f"{below} below 1%, {empty} empty", fontsize=9.5, fontweight="bold",
                     color=INK, loc="left")
        if empty:
            ax.text(len(occ) - empty / 2 - 0.5, 1.1, f"{empty} topics\nwith no held-out\npainting",
                    ha="center", va="bottom", fontsize=7.8, color=MUTED)
    axes[0].set_ylabel("held-out paintings in the topic (log scale)", fontsize=9, color=INK)
    axes[0].set_ylim(0.5, 3000)
    axes[1].text(21, 0.01 * n * 1.15, "1% of the 9,365 held-out paintings", fontsize=7.8,
                 color=INK, va="bottom")
    fig.tight_layout()
    save(fig, "heldout_topic_occupancy")


# --------------------------------------------------------------------------
# CoSiR v2 as first designed (28 Sep), against the old CoSiR
# --------------------------------------------------------------------------
def fig_v2_design():
    W, H = 172, 124
    fig, ax = canvas(W, H)

    # (a) old CoSiR
    lane(ax, 1, 96, 170, 26, S("(a) CoSiR up to 27 Sep (Exp 1 to 18)", "(a) CoSiR before v2 (Exp 1 to 18)") + ": condition = one free vector per sample")
    o_img = box(ax, 3, 99, 20, 9, "CLIP image", "frozen, 512-d", style="purple")
    o_pred = box(ax, 30, 108.3, 27, 9, "Condition predictor",
                 "MLP: CLIP image -> z, used when\na sample has no table entry", style="orange",
                 body_size=7.2, dashed=True)
    o_z = box(ax, 30, 97.8, 27, 9.5, "Condition table",
              "one trainable 2 to 16-d z per sample,\ninitialised from the buddy graph",
              style="orange", body_size=7.2)
    o_comb = box(ax, 64, 97.8, 30, 20.5, "Combiner (image side)",
                 "image feature x and z:\ntowers z -> 128, x -> 128,\n[z; x] -> 512, gated mix\n"
                 "-> conditioned image\nembedding", style="orange", body_size=7.3)
    o_loss = box(ax, 101, 97.8, 68, 20.5, "Loss",
                 "paired image-text contrastive loss + several regularisers,\n"
                 "against the frozen CLIP text feature (through an identity-\n"
                 "initialised linear layer); similarity is still 'is this my caption'",
                 style="orange", body_size=7.3)
    arrow(ax, (23, 106), (30, 112.5))
    arrow(ax, anchor(o_z, "right"), anchor(o_comb, "left", 0.25))
    arrow(ax, anchor(o_pred, "right"), anchor(o_comb, "left", 0.75), color=ORANGE_EDGE, dashed=True)
    elbow(ax, [(13, 99), (13, 97.0), (61, 97.0), (61, 100.5), (64, 100.5)], color=GREY_EDGE, lw=1.1)
    arrow(ax, anchor(o_comb, "right"), anchor(o_loss, "left"))

    # (b) Block 1
    lane(ax, 1, 66, 170, 28, "(b) CoSiR v2, Block 1: Stage 1 rebuilt from the buddy model, content only")
    b_img = box(ax, 3, 69, 20, 9, "CLIP image", "frozen, 512-d", style="purple")
    b_txt = box(ax, 3, 80, 20, 8, "CLIP text", "frozen, 512-d", style="purple")
    b_cg = box(ax, 30, 69, 28, 19, "Content teacher graph",
               "the buddy graph: mutual kNN\n(K = 30) per modality,\nunion + repair\n\nno affect teacher in the\nfirst, fully unsupervised arm",
               style="grey", body_size=7.4)
    b_st = box(ax, 65, 69, 38, 19, "AttentionFusionStudent",
               "image 512 -> 32, text 512 -> 32\n(two tokens, as in Attention-h1\nbut image + text instead of\ncontent + affect)\n1-head attention -> LayerNorm -> L2",
               style="teal", body_size=7.4)
    b_top = box(ax, 110, 69, 26, 19, "Leiden communities",
                "kNN graph (k = 20) on z\n-> Leiden communities\n(frozen output of Block 1)", style="grey",
                body_size=7.4)
    b_s2 = box(ax, 143, 69, 26, 19, "Stage 2 topic mapper",
               "not carried into v2:\ntopic classification is\nnot CoSiR's target", style="orange",
               body_size=7.4, dashed=True)
    arrow(ax, anchor(b_img, "right"), anchor(b_cg, "left", 0.3))
    arrow(ax, anchor(b_txt, "right"), anchor(b_cg, "left", 0.78))
    elbow(ax, [(13, 69), (13, 67.3), (63, 67.3), (63, 73), (65, 73)], color=GREY_EDGE, lw=1.1)
    ax.text(40, 67.6, "the same image and text features are the student's input", ha="center",
            va="bottom", fontsize=7.3, color=MUTED)
    arrow(ax, anchor(b_cg, "right", 0.6), anchor(b_st, "left", 0.6), color=TEAL_EDGE, dashed=True,
          label="InfoNCE\non edges", label_xy=(61.5, 81.2), label_color=TEAL_EDGE)
    arrow(ax, anchor(b_st, "right"), anchor(b_top, "left"), label="z", label_xy=(106.5, 79.6))

    # (c) Candidate A
    lane(ax, 1, 12, 170, 52, "(c) Block 2 = Candidate A")
    c_img = box(ax, 3, 42, 20, 9, "CLIP image I", "frozen, 512-d", style="purple")
    c_txt = box(ax, 3, 29, 20, 9, "CLIP text T", "frozen, 512-d", style="purple")
    c_fac = box(ax, 30, 27, 36, 26, "(1) Factor discovery",
                "a_I: linear 512 -> L, ReLU\na_T: linear 512 -> L, ReLU\n"
                "shared, non-negative factors (L = 32)\nseparate decoders L -> 512\n\n"
                "losses: reconstruction, paired\nagreement, graph consistency\n(content graph), anti-split, L1",
                style="teal", body_size=7.3)
    c_sup = box(ax, 73, 46.5, 30, 14, "Condition c",
                "a few support pairs that\nshare the aspect, optionally\ncontrast pairs that do not",
                style="teal", body_size=7.3)
    c_cond = box(ax, 73, 22, 30, 18, "(2) Condition interface",
                 "set encoder over the support\nand contrast factor codes\n-> weight per factor\nw(c) >= 0, length L",
                 style="teal", body_size=7.3)
    c_score = box(ax, 110, 27, 30, 26, "(3) Score",
                  "s(I, T | c) =\n  beta * cos(CLIP I, CLIP T)\n+ sum over factors l of\n  w_l(c) * a_I,l(I) * a_T,l(T)\n\n"
                  "same form for i2t and t2i",
                  style="teal", body_size=7.4)
    c_train = box(ax, 145, 27, 24, 26, "(4) Training",
                  "self-mined episodes:\nsupport, query,\ncandidate pool with\nhard negatives\n\nranking loss\n+ swap term",
                  style="teal", body_size=7.3)
    arrow(ax, anchor(c_img, "right"), anchor(c_fac, "left", 0.7))
    arrow(ax, anchor(c_txt, "right"), anchor(c_fac, "left", 0.3))
    arrow(ax, anchor(c_sup, "bottom"), anchor(c_cond, "top"))
    arrow(ax, (66, 44), (110, 44), label="codes a_I(I), a_T(T)", label_xy=(76.5, 43.2))
    arrow(ax, anchor(c_fac, "right", 0.25), anchor(c_cond, "left", 0.5))
    arrow(ax, anchor(c_cond, "right"), anchor(c_score, "left", 0.25), label="w(c)",
          label_xy=(106.5, 30.8))
    arrow(ax, anchor(c_score, "right"), anchor(c_train, "left"))
    arrow(ax, (44, 69), (44, 53.2), color=GREY_EDGE, dashed=True, lw=1.1)
    ax.text(45.2, 66.0, "content graph neighbours (graph consistency loss)", ha="left", va="center",
            fontsize=7.6, color=MUTED)
    ax.text(50, 62.6, "s(I, T | c): score an image and a text under a condition c",
            ha="left", va="top", fontsize=8.6, color=INK)
    elbow(ax, [(13, 29), (13, 18.6), (125, 18.6), (125, 27)], color=GREY_EDGE, lw=1.1,
          label="CLIP image and text (cosine term)", label_xy=(70, 18.8), label_color=MUTED)
    ax.text(86, 14.8, "Example of the target task: under the condition 'shared colour', an image of a red car "
            "should rank a caption about a red bicycle above one about a blue car.",
            ha="center", va="center", fontsize=8, color=INK, style="italic")

    # build order strip
    ax.text(3, 9.2, S("Build order in the spec, and where each stage stood on 30 Sep:", "Build order in the spec, and where each stage stands:"), ha="left",
            va="center", fontsize=8.6, fontweight="bold", color=INK)
    steps = [("(a) Block 1", S("no gain over raw CLIP (4.3)", "no gain over raw CLIP")),
             ("(b) factors", S("collapse found, repaired 29 Sep (4.4, 4.6)", "collapse found and repaired")),
             ("(c) interface", S("trained head no better than naive (4.5)", "trained head no better than naive")),
             ("(d) trained scorer", S("gain not shown (4.7)", "gain not shown")),
             ("(e) human-judged set", "not started")]
    x = 3
    for name, state in steps:
        ax.text(x, 4.6, name, ha="left", va="center", fontsize=8, fontweight="bold",
                color=TEAL_EDGE)
        ax.text(x, 1.8, state, ha="left", va="center", fontsize=7.4, color=INK)
        x += 34
    legend(ax, 100, 9.2, [("purple", "frozen CLIP"), ("orange", "removed"),
                          ("grey", "reused"), ("teal", "new in v2")], size=7.8)
    save(fig, "cosir_v2_first_design")


# --------------------------------------------------------------------------
# Stage (d): the trained scorer on frozen R3 factors
# --------------------------------------------------------------------------
V2_C = "#2a78d6"        # blue: the v2 model under test
V2_LIGHT = "#9ec5f4"
STAGE_D_JSON = OUT_DIR / "data" / "stage_d.json"


def fig_stage_d_scorer():
    W, H = 172, 70
    fig, ax = canvas(W, H)
    ax.text(3, 67.5, "Stage (d): what is trained on top of the naive rule (403 parameters: the MLP, "
            "beta and a temperature; factors and CLIP stay frozen)", ha="left", fontsize=10, fontweight="bold", color=INK)
    sup = box(ax, 3, 44, 24, 16, "Condition c",
              "4 support items and\n4 contrast items,\neach as a pair code\n(image + text) / 2",
              style="purple", body_size=7.4)
    gap = box(ax, 35, 46, 28, 12, "Naive gap per factor",
              "mean(support) - mean(contrast)", style="grey", body_size=7.6)
    evid = box(ax, 35, 24, 28, 16, "Evidence per factor (6)",
               "gap, support mean, contrast\nmean, support spread, and\nhow often the factor fires\nin supports / contrasts",
               style="grey", body_size=7.3)
    mlp = box(ax, 70, 24, 24, 16, "Tied MLP",
              "6 -> 16 -> 16 -> 1,\nsame weights for all\n32 factors; last layer\nstarts at zero",
              style="teal", body_size=7.3)
    add = box(ax, 70, 46, 24, 12, "gap + correction",
              "then ReLU and L1 norm\n-> weights w(c)", style="grey", body_size=7.4)
    score = box(ax, 102, 38, 32, 22, "Score",
                "s = beta * cos(CLIP q, CLIP k)\n+ sum_l w_l * q_l * k_l\n\n"
                "beta learned (starts at 0.3);\na learned temperature scales\nthe logits (no effect on rank);\nq, k = frozen R3 codes",
                style="teal", body_size=7.4)
    train = box(ax, 140, 38, 30, 22, "Training data",
                "self-generated conditions:\nfactor combinations,\nCLIP k-means clusters or\nBlock 1 communities\n"
                "negatives: 6 CLIP-hard\n+ 6 random; optional swap",
                style="grey", body_size=7.2)
    arrow(ax, anchor(sup, "right", 0.75), anchor(gap, "left"))
    arrow(ax, anchor(sup, "right", 0.25), anchor(evid, "left", 0.7))
    arrow(ax, anchor(gap, "right"), anchor(add, "left"))
    arrow(ax, anchor(evid, "right"), anchor(mlp, "left"))
    arrow(ax, anchor(mlp, "top"), anchor(add, "bottom"), label="correction", label_xy=(88.5, 41.5))
    arrow(ax, anchor(add, "right"), anchor(score, "left", 0.6), label="w(c)", label_xy=(98, 50))
    arrow(ax, anchor(score, "right"), anchor(train, "left"))
    ax.add_patch(FancyBboxPatch((102, 11), 68, 21, boxstyle="round,pad=0,rounding_size=1.5",
                                facecolor="white", edgecolor="#D0D7DE", linewidth=1.0, zorder=0))
    ax.text(104, 30, "Evaluation (human labels, never used in training)", ha="left", va="top",
            fontsize=8.4, fontweight="bold", color=INK)
    ax.text(104, 25.2,
            "label episodes: supports share an emotion or an art style\n"
            "with the anchor; 13 candidates, 1 positive, random negatives\n"
            "(chance R@1 7.7%)\n\n"
            "swap test: one anchor, an emotion condition and a style\n"
            "condition; success = the ranking flips the right way twice",
            ha="left", va="top", fontsize=7.5, color=INK, linespacing=1.35)
    ax.text(3, 17.5, "At step 0 the correction is zero and beta = 0.3, so the model starts as the\n"
            "naive rule exactly. Only two things can change the ranking: the learned\n"
            "correction to the gap, and the learned CLIP weight beta.",
            ha="left", va="center", fontsize=8, color=INK, linespacing=1.35)
    ax.text(3, 7.5, "Mismatch found afterwards: half the training negatives are CLIP-nearest,\n"
            + S("while every evaluation negative is random (section 4.7).", "while every evaluation negative is random."),
            ha="left", va="center", fontsize=8, color=ORANGE_EDGE, style="italic", linespacing=1.35)
    legend(ax, 3, 1.6, [("purple", "frozen input"), ("grey", "fixed computation"),
                        ("teal", "contains parameters trained in stage (d)")], size=7.8)
    save(fig, "stage_d_scorer")


def fig_stage_d_selection():
    import json
    import numpy as np
    d = json.loads(STAGE_D_JSON.read_text())
    runs = d["runs"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.4), dpi=200)
    (a, b), (c, e) = axes
    for ax in axes.flat:
        style_axes(ax)

    # (a) decomposition of the condition-use gain
    a.set_title("(a) Condition-use gain on the selection split (vs naive, R@1 points)",
                fontsize=10, fontweight="bold", color=INK, loc="left")
    y = np.arange(len(runs))[::-1]
    for yi, r in zip(y, runs):
        dec = d["decomposition"][r]
        drop = dec["beta_drop"]["point"]
        beyond = dec["beyond_beta_drop"]["point"]
        tot = dec["total_vs_naive0.3"]
        a.barh(yi, drop, color="#C9CED4", height=0.55, zorder=3)
        a.barh(yi, beyond, left=drop if beyond >= 0 else 0, color=V2_C if beyond >= 0 else "#E8A0A0",
               height=0.55, zorder=3)
        a.errorbar(tot["point"], yi, xerr=[[tot["point"] - tot["ci95"][0]],
                                           [tot["ci95"][1] - tot["point"]]],
                   fmt="o", color=INK, ms=5, capsize=3, lw=1.1, zorder=4)
    a.axvspan(2.00, 3.00, color="#EAF2FC", zorder=0)
    a.text(2.5, len(runs) - 0.45, "tie band\n(within 1 point\nof the best)", ha="center",
           va="top", fontsize=7.4, color=V2_C)
    a.axvline(0, color=INK, lw=1.0)
    a.axvline(0.5, color=INK, lw=0.9, ls=(0, (4, 3)))
    a.text(0.56, len(runs) - 0.5, "stop\npoint\n+0.5", fontsize=7.2, color=MUTED, va="top")
    a.set_yticks(y)
    a.set_yticklabels([f"{r}: {d['sources'][r]}" + ("  (selected)" if r == "G3" else "")
                       for r in runs], fontsize=8.4)
    a.set_xlabel("gain in own condition use over the naive rule (points)", fontsize=9)
    a.set_xlim(-1.2, 4.3)
    a.set_ylim(-0.6, len(runs) - 0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#C9CED4"), plt.Rectangle((0, 0), 1, 1, color=V2_C),
               plt.Rectangle((0, 0), 1, 1, color="#E8A0A0"),
               plt.Line2D([], [], marker="o", ls="", color=INK)]
    a.legend(handles, ["part explained by the lower beta alone", "part beyond the beta drop",
                       "beyond the beta drop, negative", "total, 95% CI"], fontsize=7.6,
             frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2)

    # (b) beta during training
    b.set_title("(b) The learned CLIP weight beta during training", fontsize=10,
                fontweight="bold", color=INK, loc="left")
    for r in runs:
        h = d["beta_history"][r]
        is3 = r == "G3"
        b.plot(h["step"], h["beta"], color=V2_C if is3 else "#B8BEC6", lw=2.2 if is3 else 1.2,
               zorder=4 if is3 else 3)
    b.text(3040, 0.0496, "G3 (blue)\nother runs\n(grey)", fontsize=7.6, color=V2_C,
           va="center")
    b.axhline(0.3, color=INK, lw=0.9, ls=(0, (4, 3)))
    b.text(60, 0.305, "start = the naive rule's beta, 0.3", fontsize=7.6, color=INK)
    b.set_xlabel("training step", fontsize=9)
    b.set_ylabel("beta", fontsize=9)
    b.set_xlim(0, 3450)
    b.set_ylim(0, 0.33)

    # (c) mechanism: why training lowers beta
    c.set_title("(c) Why training lowers beta (CLIP-cluster training episodes)", fontsize=10,
                fontweight="bold", color=INK, loc="left")
    m = d["mechanism_clip_cluster"]
    betas = ["0.3", "0.2", "0.1", "0.05", "0.02", "0"]
    xs = [float(v) for v in betas]
    for kind, lab, color in (("pos_over_hard", "positive ranked above a CLIP-hard negative", V2_C),
                             ("pos_over_random", "positive ranked above a random negative", REF_C)):
        for direction, ls in (("i2t", "-"), ("t2i", (0, (4, 2)))):
            ys = [100 * m["naive"][bv][direction][kind] for bv in betas]
            c.plot(xs, ys, color=color, lw=1.8, ls=ls, marker="o", ms=4, zorder=3)
        c.text(0.295, 100 * m["naive"]["0.3"]["i2t"][kind] + (2.4 if kind == "pos_over_random" else -6.5),
               lab, fontsize=7.8, color=color, ha="left")
    for direction, mk in (("i2t", "o"), ("t2i", "^")):
        c.scatter([0.0496], [100 * m["G3_learned"][direction]["pos_over_hard"]], marker=mk, s=60,
                  color="white", edgecolor=V2_C, linewidth=1.8, zorder=5)
    c.text(0.058, 100 * m["G3_learned"]["i2t"]["pos_over_hard"] + 1.6, "G3 (open markers)",
           fontsize=7.6, color=V2_C)
    for direction, v in (("i2t", m["clip_only"]["i2t"]["pos_over_hard"]),):
        c.axhline(100 * v, color=V2_C, lw=0.8, ls=(0, (1, 2)))
    c.text(0.295, 100 * m["clip_only"]["i2t"]["pos_over_hard"] + 1.2,
           "CLIP alone beats a hard negative only 31% of the time (i2t)", fontsize=7.4,
           color=V2_C, ha="left")
    c.invert_xaxis()
    c.set_xlabel("beta used by the naive rule (lower = less CLIP)", fontsize=9)
    c.set_ylabel("pairwise accuracy (%)", fontsize=9)
    c.set_ylim(25, 92)
    c.plot([], [], color=INK, ls="-", label="i2t")
    c.plot([], [], color=INK, ls=(0, (4, 2)), label="t2i")
    c.legend(fontsize=7.6, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.1))

    # (e) oracles
    e.set_title("(d) How much room do the frozen factors leave? (R@1)",
                fontsize=10, fontweight="bold", color=INK, loc="left")
    o = d["oracle_r1"]
    sel = d["selection_r1"]
    clip = d["clip_only_r1"]
    rows = [("chance (1 of 13)", 7.69, 7.69, REF_C),
            ("CLIP only", 100 * clip["i2t"]["r1"], 100 * clip["t2i"]["r1"], REF_C),
            ("naive rule", o["naive@0.3"]["i2t"], o["naive@0.3"]["t2i"], "#8C959F"),
            ("G3 (trained)", 100 * sel["G3"]["i2t"]["r1"], 100 * sel["G3"]["t2i"]["r1"], V2_C),
            ("label oracle: best fixed weights\nper human label (cross-validated)",
             o["label_oracle"]["i2t"], o["label_oracle"]["t2i"], V2_LIGHT),
            ("per-episode oracle, random\ntarget (flexibility alone)",
             o["ceiling_null"]["i2t"], o["ceiling_null"]["t2i"], "#E6E8EB"),
            ("per-episode oracle, true target", o["ceiling"]["i2t"], o["ceiling"]["t2i"], "#E6E8EB")]
    yy = np.arange(len(rows))[::-1]
    for yi, (lab, i2t, t2i, color) in zip(yy, rows):
        mean = (i2t + t2i) / 2
        e.barh(yi, mean, color=color, height=0.6, zorder=3)
        e.scatter([i2t], [yi], marker="o", s=18, color=INK, zorder=4)
        e.scatter([t2i], [yi], marker="^", s=22, color=INK, zorder=4)
        e.text(max(i2t, t2i) + 2.0 if mean < 60 else mean - 9.0, yi, f"{mean:.1f}",
               va="center", ha="left" if mean < 60 else "right", fontsize=7.8, color=INK)
    e.set_yticks(yy)
    e.set_yticklabels([r[0] for r in rows], fontsize=8)
    e.set_xlabel("R@1 (%), bar = mean of the two directions; dot i2t, triangle t2i", fontsize=8.6)
    e.set_xlim(0, 95)
    fig.tight_layout(h_pad=2.2, w_pad=2.5)
    save(fig, "stage_d_selection")


def fig_stage_d_final():
    import json
    import numpy as np
    d = json.loads(STAGE_D_JSON.read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.6), dpi=200,
                               gridspec_kw={"width_ratios": [1.15, 1]})
    for ax in (a, b):
        style_axes(ax)
    a.set_title("(a) Criterion 1: condition-use gain on held paintings (G3 vs naive)",
                fontsize=10, fontweight="bold", color=INK, loc="left")
    seeds = [("G3_seed42", "seed 42 (judged)"), ("G3_seed43", "seed 43"), ("G3_seed44", "seed 44")]
    ypos = []
    for i, (key, lab) in enumerate(seeds):
        for j, (direction, mk, sel_pt) in enumerate((("i2t", "o", 1.83), ("t2i", "^", 3.25))):
            yi = 2 * (len(seeds) - 1 - i) + (0.35 if j == 0 else -0.35)
            v = d["criterion1"][key][direction]
            pt, lo, hi = 100 * v["point"], 100 * v["ci95"][0], 100 * v["ci95"][1]
            a.errorbar(pt, yi, xerr=[[pt - lo], [hi - pt]], fmt=mk, color=V2_C, ms=6, capsize=3,
                       lw=1.3, zorder=4)
            if i == 0:
                a.scatter([sel_pt], [yi], marker=mk, s=40, color="white", edgecolor=MUTED,
                          linewidth=1.3, zorder=4)
            ypos.append((yi, f"{lab}, {direction}"))
    a.axvline(0, color=INK, lw=1.0)
    a.set_yticks([p for p, _ in ypos])
    a.set_yticklabels([l for _, l in ypos], fontsize=8)
    a.set_xlabel("gain in own condition use over naive (R@1 points), 95% CI", fontsize=8.8)
    a.text(3.3, 4.75, "open markers: the same\nquantity on the selection\nsplit (+1.83, +3.25)",
           fontsize=7.4, color=MUTED, va="center")
    a.text(-1.9, -1.35, "met only if every lower bound is above 0 in both directions: not met "
           "(power about 0.38)", fontsize=7.6, color=INK)
    a.set_ylim(-1.7, 5.0)

    b.set_title("(b) Criterion 2: human swap test (success rate, %)", fontsize=10,
                fontweight="bold", color=INK, loc="left")
    held = d["criterion2_rates"]
    sw = d["swap_selection"]
    bars = [("held: naive", 100 * np.mean(list(held["naive"].values())), "#8C959F"),
            ("held: G3 seed 42", 100 * np.mean(list(held["G3_seed42"].values())), V2_C),
            ("held: G3 seed 43", 100 * np.mean(list(held["G3_seed43"].values())), V2_C),
            ("held: G3 seed 44", 100 * np.mean(list(held["G3_seed44"].values())), V2_C),
            ("selection: naive, beta 0.3", np.mean(list(sw["naive@0.3"].values())), "#8C959F"),
            ("selection: naive at G3's beta 0.05", np.mean(list(sw["naive@G3beta"].values())),
             V2_LIGHT),
            ("selection: G3", np.mean(list(sw["G3"].values())), V2_C)]
    yy = np.arange(len(bars))[::-1]
    for yi, (lab, v, color) in zip(yy, bars):
        b.barh(yi, v, color=color, height=0.6, zorder=3)
        b.text(v + 0.4, yi, f"{v:.1f}", va="center", fontsize=7.8, color=INK)
    b.axhline(2.5, color="#D0D7DE", lw=1.0)
    b.set_yticks(yy)
    b.set_yticklabels([x[0] for x in bars], fontsize=8)
    b.set_xlabel("episodes where both conditions flip the ranking correctly (%), mean of "
                 "i2t and t2i", fontsize=8.4)
    b.set_xlim(0, 32)
    fig.tight_layout(w_pad=3)
    save(fig, "stage_d_final")


# --------------------------------------------------------------------------
# Week at a glance
# --------------------------------------------------------------------------
def _t(day, hh, mm=0):
    return day + (hh + mm / 60) / 24


def _events(ax, events, y, above=True, levels=(0.9, 1.75, 2.6)):
    sign = 1 if above else -1
    for i, ev in enumerate(events):
        t, label, kind = ev[:3]
        lvl = ev[3] if len(ev) > 3 else i % len(levels)
        color = {"buddy": BUDDY_C, "percept": PERCEPT_C, "v2": V2_C, "ref": REF_C}[kind]
        ax.scatter([t], [y], s=46, color=color, edgecolor="white", linewidth=1.2, zorder=4,
                   marker="s" if kind == "percept" else "o")
        ly = y + sign * levels[lvl]
        ax.plot([t, t], [y, ly - sign * 0.18], color="#C9CED4", lw=0.8, zorder=2)
        ax.text(t, ly, label, ha="center", va="bottom" if above else "top", fontsize=7.3,
                color=INK, linespacing=1.15, zorder=5)


def fig_timeline():
    fig, (a, b) = plt.subplots(2, 1, figsize=(14, 8.2), dpi=200,
                               gridspec_kw={"height_ratios": [1.05, 1]})
    for ax in (a, b):
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_yticks([])
        ax.tick_params(colors=MUTED, labelsize=8.5)
    a.set_title("(a) The percept line, 22 to 30 September (branch experiment/percept_topic_pipeline)",
                fontsize=10, fontweight="bold", color=INK, loc="left")
    a.axhline(0, color="#B8BEC6", lw=1.2, zorder=1)
    ev = [(_t(22, 15, 17), "buddy graph alone:\ngenre 0.44, emotion 0.06", "buddy"),
          (_t(22, 21, 11), "Attention-h1 clears\nthe bar (1 seed)", "buddy"),
          (_t(23, 2, 19), "PercepT K = 60/40:\n4 of 4 seeds clear", "percept"),
          (_t(24, 22, 30), "PercepT paper schedule:\nsilhouette 0.51, collapsed", "percept"),
          (_t(26, 6, 19), "first buddy model in\nPercepT's pipeline", "buddy"),
          (_t(26, 23, 46), "six DEC hybrids:\nall fail", "buddy"),
          (_t(27, 2, 8), "RedCaps B1:\nno gain over raw CLIP", "buddy", 0),
          (_t(27, 15, 40), "review finds two\nPercepT bugs", "percept", 1),
          (_t(27, 21, 15), "buddy Stage 2\ntuned: 0.853", "buddy", 2),
          (_t(28, 0, 50), "PercepT tuned the\nsame way: 0.923", "percept", 3),
          (_t(28, 3, 45), "joint sweep launched\n(2,267 trials by 29 Sep)", "buddy", 0),
          (_t(30, 1, 27), "sweep winner\nstress-tested", "buddy", 2),
          (_t(30, 3, 33), "check: winner's pass\nis a yardstick artefact", "buddy", 0),
          (_t(30, 4, 40), "matched head-to-head\nlaunched (finished\nthat evening, §3.6)", "buddy", 1)]
    _events(a, ev, 0, above=True, levels=(0.9, 1.75, 2.6, 3.45))
    a.plot([_t(28, 22, 44), _t(30, 9)], [-0.45, -0.45], color=V2_C, lw=3, solid_capstyle="butt")
    a.text(_t(29, 16), -0.62, "CoSiR v2 ran on main in parallel: see (b)", ha="center", va="top",
           fontsize=7.6, color=V2_C)
    a.set_ylim(-1.3, 4.4)
    a.set_xlim(_t(22, 6), _t(30, 12))
    days = list(range(22, 31))
    a.set_xticks(days)
    a.set_xticklabels([f"{d} Sep" for d in days])

    b.set_title("(b) CoSiR v2 on main, 28 Sep 22:00 to 30 Sep 09:00", fontsize=10,
                fontweight="bold", color=INK, loc="left")
    b.axhline(0, color="#B8BEC6", lw=1.2, zorder=1)
    ev2 = [(_t(28, 22, 44), "v2 spec:\nstop patching", "v2"),
           (_t(28, 23, 39), "Block 1 = raw\nCLIP (+0.001)", "v2"),
           (_t(29, 0, 0), "factors: 2 of 32\nhold 88% of mass", "v2"),
           (_t(29, 1, 58), "other encoders\nwiden the gap", "v2"),
           (_t(29, 5, 26), "usage balance:\n'fixed' by the checks", "v2"),
           (_t(29, 6, 12), "condition interface: 19 of\n32 factors never recovered", "v2"),
           (_t(29, 12, 11), "ranking test:\nnaive rule wins", "v2"),
           (_t(29, 17, 38), "CLIP-only\nbaseline", "v2"),
           (_t(29, 19, 43), "gain was mostly\nscale: +3.6", "v2"),
           (_t(29, 20, 50), "review: factors\ncollapsed (PR 1.3)", "v2"),
           (_t(29, 21, 25), "cause: cosine\nagreement loss", "v2"),
           (_t(29, 23, 23), "R3 repair\n(amended gates)", "v2"),
           (_t(29, 23, 41), "held test: +4.4\nt2i only", "v2"),
           (_t(30, 4, 23), "stage (d) scorer:\ngain not shown", "v2"),
           (_t(30, 8, 20), "decision: condition-\naware factor learning", "v2")]
    _events(b, ev2, 0, above=True, levels=(0.9, 1.9, 2.9))
    b.set_ylim(-0.6, 4.0)
    b.set_xlim(_t(28, 22), _t(30, 9, 30))
    ticks = [_t(29, 0), _t(29, 6), _t(29, 12), _t(29, 18), _t(30, 0), _t(30, 6)]
    b.set_xticks(ticks)
    b.set_xticklabels(["29 Sep 00:00", "06:00", "12:00", "18:00", "30 Sep 00:00", "06:00"])
    handles = [plt.Line2D([], [], marker="o", ls="", color=BUDDY_C, label="buddy side"),
               plt.Line2D([], [], marker="s", ls="", color=PERCEPT_C, label="PercepT baseline"),
               plt.Line2D([], [], marker="o", ls="", color=V2_C, label="CoSiR v2")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.5)
    save(fig, "timeline")


# --------------------------------------------------------------------------
# Section 3.2: the six DEC hybrids against the buddy baseline and PercepT
# --------------------------------------------------------------------------
def fig_dec_hybrids():
    import numpy as np
    rows = [
        ("Attention-h1 baseline", "buddy", [0.1249, 0.1213, 0.1238, 0.1264],
         [0.2404, 0.2544, 0.2386, 0.2289], [0.0377, 0.0397, 0.0375, 0.0438]),
        ("+ cosine LR schedule", "buddy", [0.1306, 0.1244, 0.1334, 0.1222],
         [0.1973, 0.2583, 0.2452, 0.2576], [0.0488, 0.0438, 0.0487, 0.0451]),
        ("+ schedule + Leiden pseudo-labels", "buddy", [0.1210, 0.1141, 0.1180, 0.1107],
         [0.2623, 0.3111, 0.3487, 0.2822], [0.0789, 0.0721, 0.0692, 0.0753]),
        ("(1) Euclidean DEC on z", "dec", [0.1162, 0.1232, 0.1125, 0.1122], [0.1321],
         [-0.0232, -0.0266, -0.0202, -0.0450]),
        ("(2) vMF DEC on z", "dec", [0.1186, 0.1260, 0.1247, 0.1167],
         [0.0895, 0.1638, 0.1908, 0.1575], [0.0273, 0.0323, 0.0324, 0.0271]),
        ("(3) separate cluster head", "dec", [0.0845], [0.0423],
         [-0.1345, -0.1634, -0.1718, -0.1577]),
        ("(4) same, gradient detached", "dec", [0.0828], [0.0481], [-0.1529]),
        ("(5) PercepT autoencoder + DEC on z (1 seed)", "dec", [0.1482], [0.1765], [-0.0252]),
        ("(6) reconstruction-anchored head", "dec", [0.1041, 0.1150, 0.1307, 0.0991],
         [0.1264, 0.0920, 0.0808, 0.1404], [-0.0872, -0.0720, -0.0933, -0.0647]),
        ("(6b) same, PercepT bugs fixed", "dec", [0.1197, 0.1173, 0.1178, 0.1047],
         [0.1500, 0.0648, 0.0986, 0.1464], [-0.0198, -0.0082, 0.0026, -0.0215]),
        ("PercepT, paper's schedule (seed 42)", "percept", [0.1092], [0.3288], [0.5120]),
        ("PercepT, same, bugs fixed (seed 42)", "percept", [0.1097], [0.3764], [0.2224]),
    ]
    if SLIDES:
        rows = [r for r in rows if "bugs fixed" not in r[0]]
    n_p = sum(r[1] == "percept" for r in rows)
    n_dec = sum(r[1] == "dec" for r in rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.4), dpi=200, sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})
    y = np.arange(len(rows))[::-1]
    titles = ("held-out emotion AMI", "held-out genre AMI", "held-out silhouette (buddy rows in z)")
    for k, ax in enumerate(axes):
        style_axes(ax)
        ax.grid(axis="y", visible=False)
        for yi, row in zip(y, rows):
            vals = row[2 + k]
            color = {"buddy": BUDDY_C, "dec": "#0f6b52", "percept": PERCEPT_C}[row[1]]
            marker = "s" if row[1] == "percept" else ("D" if row[1] == "dec" else "o")
            if len(vals) > 1:
                ax.scatter(vals, [yi] * len(vals), s=14, color=color, alpha=0.45, zorder=3)
            ax.scatter([np.mean(vals)], [yi], s=58, marker=marker, color=color,
                       edgecolor="white", linewidth=1.2, zorder=4)
        ax.set_title(titles[k], fontsize=9.6, fontweight="bold", color=INK, loc="left")
        ax.axhline(n_p + n_dec - 0.5, color="#D0D7DE", lw=1.0)
        ax.axhline(n_p - 0.5, color="#D0D7DE", lw=1.0)
    axes[0].axvline(0.1236, color=INK, lw=1.0, ls=(0, (4, 3)))
    axes[1].axvline(0.1954, color=INK, lw=1.0, ls=(0, (4, 3)))
    axes[2].axvline(0, color=INK, lw=1.0)
    axes[0].text(0.1242, len(rows) - 0.25, "bar", fontsize=7.6, color=INK)
    axes[1].text(0.2, len(rows) - 0.25, "bar", fontsize=7.6, color=INK)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=8.2)
    axes[0].set_xlim(0.075, 0.155)
    axes[1].set_xlim(0.0, 0.40)
    axes[2].set_xlim(-0.2, 0.56)
    axes[2].text(0.52, n_p + n_dec / 2 - 0.5, "DEC hybrids\n(diamonds)", fontsize=7.6, color="#0f6b52", ha="right")
    fig.text(0.5, 0.01, "Large marker: mean over seeds; small dots: individual seeds where the reports list them "
             "(not for attempt 1 genre, the AMIs of attempts 3 and 4, or one-seed runs). PercepT silhouette is in "
             "its own 128-d Z. Dashed lines: the Pareto bar.",
             ha="center", fontsize=8.2, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1), w_pad=1.2)
    save(fig, "dec_hybrids")


# --------------------------------------------------------------------------
# Section 3.4: the Stage 2 race
# --------------------------------------------------------------------------
def fig_stage2_race():
    steps = [(S("27 Sep 00:54\n", "") + "first wiring", 0.5978, 0.5690, True),
             (S("27 Sep 15:47\nPercepT bugs fixed", "PercepT: paper-faithful\npruning and loss scale"), 0.5978, 0.5925, True),
             (S("27 Sep 20:37\n", "") + "buddy: merge small\ntopics + class weights", 0.6334, 0.5925, False),
             (S("27 Sep 20:42\n", "") + "buddy: lr 1e-2,\n400 epochs", 0.8461, 0.5925, False),
             (S("27 Sep 21:15\n", "") + "buddy: multi-label\ntraining targets", 0.8534, 0.5925, False),
             (S("28 Sep 00:50\n", "") + "PercepT: same\nlr / epoch sweep", 0.8534, 0.9226, True)]
    fig, ax = plt.subplots(figsize=(13.5, 5.6), dpi=200)
    style_axes(ax)
    xs = list(range(len(steps)))
    for i, (_, bu, pe, matched) in enumerate(steps):
        if not matched:
            ax.axvspan(i - 0.5, i + 0.5, color="#FFF6EE", zorder=0)
    ax.step(xs, [s[1] for s in steps], where="mid", color=BUDDY_C, lw=2.2, zorder=3)
    ax.step(xs, [s[2] for s in steps], where="mid", color=PERCEPT_C, lw=2.2, zorder=3)
    for i, (_, bu, pe, _m) in enumerate(steps):
        ax.scatter([i], [bu], s=48, color=BUDDY_C, edgecolor="white", zorder=4)
        ax.scatter([i], [pe], s=48, marker="s", color=PERCEPT_C, edgecolor="white", zorder=4)
        ax.text(i + 0.07, bu + 0.013, f"{bu:.4f}", fontsize=7.8, color=INK)
        ax.text(i + 0.07, pe + (0.013 if pe > bu else -0.03), f"{pe:.4f}", fontsize=7.8,
                color=INK)
    ax.axhline(0.8256, color=REF_C, lw=1.0, ls=(0, (1, 2)))
    ax.text(-0.45, 0.832, S("PercepT on 23 Sep", "PercepT earlier,") + " with multi-label targets and lr 3e-3: 0.8256 "
            "(different targets, not used)", fontsize=7.4, color=MUTED)
    ax.axhline(0.5, color=INK, lw=0.8)
    ax.text(-0.45, 0.505, "guessing (0.5)", fontsize=7.4, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in steps], fontsize=7.8)
    ax.set_ylabel("held-out Stage 2 macro AUC", fontsize=9)
    ax.set_ylim(0.48, 0.97)
    ax.set_xlim(-0.5, len(steps) - 0.5)
    ax.text(3, 0.945, "shaded: only the buddy side was changed (unmatched comparison); steps are in "
            "order, not to scale in time", ha="center", fontsize=8, color=ORANGE_EDGE)
    handles = [plt.Line2D([], [], color=BUDDY_C, lw=2.2, marker="o", label="buddy topics"),
               plt.Line2D([], [], color=PERCEPT_C, lw=2.2, marker="s",
                          label="PercepT topics (baseline)")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5)
    fig.tight_layout()
    save(fig, "stage2_race")


# --------------------------------------------------------------------------
# Section 3.5: RedCaps B1
# --------------------------------------------------------------------------
def fig_redcaps():
    import numpy as np
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 4.6), dpi=200,
                               gridspec_kw={"width_ratios": [1.05, 1]})
    for ax in (a, b):
        style_axes(ax)
    a.set_title("(a) Subreddit lift of a kNN graph on the validation embeddings", fontsize=9.8,
                fontweight="bold", color=INK, loc="left")
    rows = [("raw CLIP image + text\n(no training)", 27.114, REF_C),
            ("raw CLIP image", 24.314, REF_C), ("raw CLIP text", 18.153, REF_C),
            ("mean-pool student", 18.973, BUDDY_C), ("attention student (B1)", 18.391, BUDDY_C)]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, c) in zip(y, rows):
        a.barh(yi, v, color=c, height=0.6, zorder=3)
        a.text(v + 0.3, yi, f"{v:.1f}x", va="center", fontsize=8, color=INK)
    a.set_yticks(y)
    a.set_yticklabels([r[0] for r in rows], fontsize=8.2)
    a.set_xlabel("same-subreddit edge share over chance (x)", fontsize=8.8)
    a.set_xlim(0, 31)
    b.set_title("(b) Communities vs Leiden resolution (graph-only arm)",
                fontsize=9.8, fontweight="bold", color=INK, loc="left")
    res = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005]
    comms = [1423, 1410, 1403, 1398, 1397, 1397, 1397]
    b.plot(res, comms, marker="o", color=BUDDY_C, lw=1.8, label="teacher graph as built")
    b.axhline(1397, color=INK, lw=0.9, ls=(0, (4, 3)))
    b.text(0.0055, 700, "dashed: 1,397 connected components\n(1,354 of them single nodes), a floor\nthat no resolution can go below",
           fontsize=7.6, color=INK, va="top")
    b.scatter([1.0], [71], s=60, marker="D", color=V2_C, zorder=4)
    b.text(0.62, 71, "after linking isolated nodes: 71", fontsize=7.8, color=V2_C, ha="right",
           va="center")
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlabel("Leiden resolution (log)", fontsize=8.8)
    b.set_ylabel("communities on 120,000 train items", fontsize=8.8)
    b.set_ylim(40, 3000)
    fig.tight_layout(w_pad=3)
    save(fig, "redcaps_b1")


# --------------------------------------------------------------------------
# Section 3.6: sweep finalists and the two yardsticks
# --------------------------------------------------------------------------
def fig_sweep_qc():
    import numpy as np
    fin = [("m8x7ifx4", 4, 0.9355, 0.1295), ("bfrae3fr", 3, 0.9399, 0.12359),
           ("j4fp661m", 3, 0.9357, 0.1273), ("1vavykgu", 2, 0.9482, 0.1220),
           ("knp9y2wp", 2, 0.9360, 0.1231), ("wokudgi4", 2, 0.9309, 0.1238),
           ("whysrv0g", 1, 0.9584, 0.1111), ("i3zajuzr", 1, 0.9513, 0.1189),
           ("woy8z3lw", 1, 0.9486, 0.1195), ("2lvvlkmy", 1, 0.9431, 0.1198)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 4.9), dpi=200,
                               gridspec_kw={"width_ratios": [1, 1.1]})
    for ax in (a, b):
        style_axes(ax)
    a.set_title("(a) The 10 sweep finalists over 4 seeds", fontsize=9.8, fontweight="bold",
                color=INK, loc="left")
    shades = {4: "#0f6b52", 3: BUDDY_C, 2: "#8fd9bd", 1: "#d6f2e7"}
    for name, passes, auc, emo in fin:
        a.scatter([emo], [auc], s=40 + 22 * passes, color=shades[passes], edgecolor=INK,
                  linewidth=0.6, zorder=4)
    a.annotate("winner m8x7ifx4\n(4 of 4 seeds pass)", (0.1295, 0.9355), xytext=(0.1255, 0.9265),
               fontsize=7.8, arrowprops=dict(arrowstyle="-", color="#B8BEC6", lw=0.7))
    a.annotate("sweep's #1 (1 of 4)", (0.1111, 0.9584), xytext=(0.1135, 0.9595), fontsize=7.8,
               arrowprops=dict(arrowstyle="-", color="#B8BEC6", lw=0.7))
    a.axvline(0.1236, color=INK, lw=1.0, ls=(0, (4, 3)))
    a.text(0.1239, 0.962, "emotion bar", fontsize=7.6, color=INK)
    a.text(0.1265, 0.957, "r = -0.85 between\nemotion AMI and AUC", fontsize=7.8, color=MUTED)
    for passes in (4, 3, 2, 1):
        a.scatter([], [], s=40 + 22 * passes, color=shades[passes], edgecolor=INK, linewidth=0.6,
                  label=f"{passes} of 4 seeds pass")
    a.legend(fontsize=7.4, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.0))
    a.set_xlabel("mean emotion AMI, sweep yardstick (k-NN vote, k tuned)", fontsize=8.6)
    a.set_ylabel("mean Stage 2 macro AUC", fontsize=8.8)
    a.set_xlim(0.108, 0.133)
    a.set_ylim(0.922, 0.966)

    b.set_title("(b) Same models, two ways of scoring held-out emotion", fontsize=9.8,
                fontweight="bold", color=INK, loc="left")
    rows = [("untuned Attention-h1 pilot\n(seeds 42, 7, 123, 2024)", 0.1230, 0.1341),
            ("winner, its original 4 seeds", 0.1164, 0.1295),
            ("winner, 4 new seeds", 0.1213, 0.1314)]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, ind, gate) in zip(y, rows):
        b.plot([ind, gate], [yi, yi], color="#C9CED4", lw=2.4, zorder=2)
        b.scatter([ind], [yi], s=70, color=INK, zorder=4)
        b.scatter([gate], [yi], s=70, color="white", edgecolor=BUDDY_C, linewidth=2, zorder=4)
        b.text(ind - 0.0006, yi + 0.2, f"{ind:.4f}", ha="right", fontsize=7.8, color=INK)
        b.text(gate + 0.0006, yi + 0.2, f"{gate:.4f}", ha="left", fontsize=7.8, color=BUDDY_C)
    b.axvline(0.1236, color=INK, lw=1.0, ls=(0, (4, 3)))
    b.text(0.12375, 2.55, "bar 0.1236", fontsize=7.6, color=INK)
    b.set_yticks(y)
    b.set_yticklabels([r[0] for r in rows], fontsize=8.2)
    b.set_xlim(0.112, 0.138)
    b.set_ylim(-0.6, 2.8)
    b.set_xlabel("mean held-out emotion AMI", fontsize=8.8)
    b.scatter([], [], s=60, color=INK, label="independent re-clustering (how the bar was set)")
    b.scatter([], [], s=60, color="white", edgecolor=BUDDY_C, linewidth=2,
              label="sweep gate: merged topics, k = 40 vote")
    b.legend(fontsize=7.6, frameon=False, loc="lower right")
    fig.tight_layout(w_pad=3)
    save(fig, "sweep_yardsticks")


# --------------------------------------------------------------------------
# Section 3.6: matched head-to-head, final
# --------------------------------------------------------------------------
H2H_FLOOR = 0.1117
H2H_WINNERS = {"primary": {"buddy_k16": "6edlxmyv", "buddy_k40": "89mkiavu",
                           "percept_k16": "x0oa5511", "percept_k40": "0biqmu50"},
               "constrained": {"buddy_k16": "mssv0f7s", "buddy_k40": "4fu1936m",
                               "percept_k16": "usnsbxdq", "percept_k40": "i1rigwnq"}}


def fig_h2h_final():
    import json
    data = json.loads((OUT_DIR / "data" / "h2h_final.json").read_text())
    rows, test = data["trials"], data["test"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=200,
                             gridspec_kw={"width_ratios": [1, 1, 0.85]})
    for ax, k, tag in zip(axes[:2], ("16", "40"), ("a", "b")):
        style_axes(ax)
        for impl, cell, color, marker, fill, lab in (
                ("percept", f"percept_k{k}", PERCEPT_C, "s", True, "PercepT"),
                ("harness", f"buddy_k{k}", BUDDY_C, "o", True, "buddy, sweep-harness Stage 1"),
                ("pilot", f"buddy_k{k}", "#0f6b52", "o", False, "buddy, pilot-faithful Stage 1")):
            pts = [r for r in rows if r["cell"] == cell and r["impl"] == impl]
            ax.scatter([p["emo"] for p in pts], [p["auc"] for p in pts], s=16, marker=marker,
                       color=color if fill else "white", edgecolor=color, linewidth=1.0,
                       alpha=0.7, zorder=3, label=f"{lab} (n={len(pts)})")
        for sel, mk in (("primary", "*"), ("constrained", "P")):
            for system in ("buddy", "percept"):
                rid = H2H_WINNERS[sel][f"{system}_k{k}"]
                hit = [r for r in rows if r["id"] == rid]
                if hit:
                    ax.scatter([hit[0]["emo"]], [hit[0]["auc"]], s=190, marker=mk,
                               color=BUDDY_C if system == "buddy" else PERCEPT_C,
                               zorder=5, edgecolor=INK, linewidth=1.2)
        ax.axvline(H2H_FLOOR, color="#9aa3ad", lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.text(H2H_FLOOR - 0.001, 0.62, "emotion floor 0.1117\n(fixed on val\nbefore test)",
                fontsize=7.2, color=MUTED, ha="right")
        ax.set_title(f"({tag}) all sweep trials, K = {k} topics", fontsize=10, fontweight="bold",
                     color=INK, loc="left")
        ax.set_xlabel("independent emotion AMI on the val half", fontsize=8.8)
        ax.set_ylim(0.6, 1.0)
        ax.legend(fontsize=7.4, frameon=False, loc="lower left")
    axes[0].set_ylabel("val Stage 2 macro AUC (mean of 2 search seeds)", fontsize=8.8)
    c = axes[2]
    style_axes(c)
    groups = [("plain AUC\nK = 16", "primary", "16"), ("plain AUC\nK = 40", "primary", "40"),
              ("emotion floor\nK = 16", "constrained", "16"), ("emotion floor\nK = 40", "constrained", "40")]
    for i, (lab, sel, k) in enumerate(groups):
        for off, system, color, marker in ((-0.16, "buddy", BUDDY_C, "o"), (0.16, "percept", PERCEPT_C, "s")):
            t = test[f"test_{sel}_{system}_k{k}"]
            c.errorbar(i + off, t["auc_mean"], yerr=[[t["auc_mean"] - t["auc_lo"]], [t["auc_hi"] - t["auc_mean"]]],
                       fmt=marker, color=color, ms=7, capsize=3, lw=1.3, zorder=3)
            c.text(i + off, t["auc_lo"] - 0.003, f"emo\n{t['emo_mean']:.3f}", ha="center", va="top",
                   fontsize=6.6, color=MUTED)
    c.axvline(1.5, color="#B8BEC6", lw=0.9)
    c.set_xticks(range(len(groups)))
    c.set_xticklabels([g[0] for g in groups], fontsize=7.8)
    c.set_ylim(0.915, 1.0)
    c.set_ylabel("test Stage 2 macro AUC (5 seeds, 95% CI)", fontsize=8.8)
    c.set_title("(c) cell winners on the test half", fontsize=10, fontweight="bold", color=INK, loc="left")
    c.scatter([], [], marker="o", color=BUDDY_C, label="buddy")
    c.scatter([], [], marker="s", color=PERCEPT_C, label="PercepT (baseline)")
    c.legend(fontsize=7.4, frameon=False, loc="upper right")
    fig.text(0.5, 0.005, "Large stars: plain-AUC winners; large crosses: emotion-floor winners (black outline, coloured by system). "
             "The emotion-floor selection is a secondary analysis designed after an interim look; "
             "its floor was fixed on val before any test run. Buddy topic counts are matched within 2.",
             ha="center", fontsize=7.8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=2.0)
    save(fig, "h2h_final")


# --------------------------------------------------------------------------
# Section 4.3: Block 1 against its matched raw-feature baseline
# --------------------------------------------------------------------------
def fig_block1():
    import numpy as np
    pairs = [("CLIP + CLIP", 0.035781, 0.036942), ("DINOv2 + e5", 0.032652, 0.040973),
             ("SigLIP + e5", 0.034434, 0.040173), ("ViT (supervised) + e5", 0.031267, 0.052282)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.4), dpi=200,
                               gridspec_kw={"width_ratios": [1.5, 1]})
    for ax in (a, b):
        style_axes(ax)
        ax.grid(axis="x", visible=False)
    a.set_title("(a) Emotion AMI of communities: raw features vs the trained Block 1 student",
                fontsize=9.6, fontweight="bold", color=INK, loc="left")
    x = np.arange(len(pairs))
    a.bar(x - 0.19, [p[1] for p in pairs], width=0.36, color=REF_C, label="raw features, same clustering",
          zorder=3)
    a.bar(x + 0.19, [p[2] for p in pairs], width=0.36, color=V2_C, label="trained student", zorder=3)
    for xi, (_, raw, tr) in zip(x, pairs):
        a.text(xi + 0.19, tr + 0.0008, f"+{tr - raw:.4f}", ha="center", fontsize=7.8, color=INK)
    a.set_xticks(x)
    a.set_xticklabels([p[0] for p in pairs], fontsize=8.4)
    a.set_ylabel("in-sample emotion AMI (308,723 caption rows)", fontsize=8.6)
    a.set_ylim(0, 0.06)
    a.legend(fontsize=7.8, frameon=False, loc="upper left")
    b.set_title("(b) CLIP + CLIP: longer training does not help", fontsize=9.6,
                fontweight="bold", color=INK, loc="left")
    ep = [200, 2000, 10000]
    ami = [0.036942, 0.036682, 0.035851]
    b.plot(ep, ami, marker="o", color=V2_C, lw=1.8, zorder=3)
    b.axhline(0.035781, color=REF_C, lw=1.2, ls=(0, (4, 3)))
    b.text(260, 0.03555, "raw CLIP 0.0358", fontsize=7.8, color=MUTED, va="top")
    b.set_xscale("log")
    b.set_xlabel("training steps (log)", fontsize=8.6)
    b.set_ylim(0.034, 0.0385)
    b.set_ylabel("emotion AMI", fontsize=8.6)
    fig.tight_layout(w_pad=3)
    save(fig, "block1")


# --------------------------------------------------------------------------
# Section 4.6: the factor collapse, seen directly
# --------------------------------------------------------------------------
def fig_factor_collapse():
    import json
    import numpy as np
    d = json.loads((OUT_DIR / "data" / "factors.json").read_text())
    fig = plt.figure(figsize=(14, 7.6), dpi=200)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15], hspace=0.45, wspace=0.35)
    cmap = _diverging_cmap()
    for j, (key, title) in enumerate((("R0", "(a) R0 recipe: factor correlations"),
                                      ("R3", "(b) repaired R3: factor correlations"))):
        ax = fig.add_subplot(gs[0, j])
        c = np.array(d[key]["corr"])
        im = ax.imshow(c, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=9.6, fontweight="bold", color=INK, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"32 factors; {d[key]['pairs_ge_09']} of 496 pairs with |r| >= 0.9",
                      fontsize=8, color=MUTED)
    cax = fig.add_axes((0.08, 0.53, 0.012, 0.3))
    cb = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1])
    cb.ax.tick_params(labelsize=7.5)
    cax.yaxis.set_ticks_position("left")
    ax = fig.add_subplot(gs[0, 2])
    style_axes(ax)
    for key, color, lab in (("R0", PERCEPT_C, "R0"), ("R3", V2_C, "R3")):
        s = np.array(d[key]["spectrum"])
        ax.plot(np.arange(1, len(s) + 1), s, marker="o", ms=3, color=color, lw=1.6,
                label=f"{lab}: participation ratio {d[key]['pr']:.1f} (pair codes)")
    ax.set_yscale("log")
    ax.set_title("(c) Share of variance per principal component", fontsize=9.6,
                 fontweight="bold", color=INK, loc="left")
    ax.set_xlabel("component", fontsize=8.6)
    ax.set_ylabel("share of code variance (log)", fontsize=8.6)
    ax.legend(fontsize=7.8, frameon=False)
    ax = fig.add_subplot(gs[1, :])
    style_axes(ax)
    ax.grid(axis="x", visible=False)
    names = {"D0": "D0\nR0 recipe", "D1": "D1\nreconstruction only",
             "D2": "D2\nD1 + L1", "D3": "D3\nR0 without\ncosine agreement",
             "D4": "D4\nR0 without\nusage balance", "D5": "D5\nR0 without\ngraph term",
             "D6": "D6\nR0 with\ncentred input"}
    keys = list(names)
    vals = [d["diagnosis"][k]["pr"] for k in keys]
    colors = [V2_C if k == "D3" else "#B8BEC6" for k in keys]
    ax.bar(range(len(keys)), vals, color=colors, width=0.6, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.08, f"{v:.2f}", ha="center", fontsize=8, color=INK)
    ax.axhline(8, color=INK, lw=1.0, ls=(0, (4, 3)))
    ax.text(6.45, 8.6, "gate: >= 8", ha="right", fontsize=7.8, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 40)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([names[k] for k in keys], fontsize=8.2)
    ax.set_ylabel("participation ratio,\nweaker modality (log)", fontsize=8.6)
    ax.set_title("(d) One change at a time from the R0 recipe (validation rows, one seed each)",
                 fontsize=9.6, fontweight="bold", color=INK, loc="left")
    fig.text(0.5, 0.005, "(a) pair codes of the R0 recipe from the stage (c) run (all 308,723 rows); "
             "(b) R3 seed 42 on the 30,872 validation rows. Participation ratio = effective number "
             "of dimensions; CLIP's own is about 40. D2's 1.00 comes from dead codes, not from copies of one axis.", ha="center", fontsize=7.8, color=MUTED)
    save(fig, "factor_collapse")


# --------------------------------------------------------------------------
# Section 4.6: the repair grid and its gates
# --------------------------------------------------------------------------
def fig_repair_grid():
    import json
    import numpy as np
    from matplotlib.colors import ListedColormap
    d = json.loads((OUT_DIR / "data" / "factors.json").read_text())
    gates = ["participation_ratio", "redundancy", "readout", "sparsity", "dead",
             "modality_private", "usage_concentration", "community_spanning", "pair_retrieval"]
    glabels = ["PR >= 8", "max |r| <= .9", "readout", "sparsity", "no dead", "private <= 1",
               "top-2 mass", "spanning", "retrieval"]
    runs = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]
    desc = {"R0": "R0 (collapsed recipe)", "R1": "R1 InfoNCE", "R2": "R2 decorrelation",
            "R3": "R3 InfoNCE + decorrelation", "R4": "R4 InfoNCE, TopK 8",
            "R5": "R5 R4 + decorrelation", "R6": "R6 R3 + centring", "R7": "R7 R3 + L1 0.1",
            "R8": "R8 no agreement term"}
    am = {r["run"]: r for r in d["amended"]}
    mat = np.zeros((len(runs), len(gates)))
    for i, r in enumerate(runs):
        for j, g in enumerate(gates):
            orig = am[r]["original_passed"][g]
            amended = am[r]["passed"][g]
            mat[i, j] = 2 if orig else (1 if amended else 0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=200,
                               gridspec_kw={"width_ratios": [2.3, 1]}, sharey=True)
    cmap = ListedColormap(["#F4C7C3", "#FDE9B8", "#BFE8D7"])
    a.imshow(mat, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    for i in range(len(runs)):
        for j in range(len(gates)):
            a.text(j, i, {2: "pass", 1: "pass*", 0: "fail"}[int(mat[i, j])], ha="center",
                   va="center", fontsize=7.4, color=INK)
    a.set_xticks(range(len(gates)))
    a.set_xticklabels(glabels, fontsize=8, rotation=25, ha="right")
    a.set_yticks(range(len(runs)))
    a.set_yticklabels([desc[r] for r in runs], fontsize=8.3)
    a.set_xticks(np.arange(-0.5, len(gates), 1), minor=True)
    a.set_yticks(np.arange(-0.5, len(runs), 1), minor=True)
    a.grid(which="minor", color="white", linewidth=2)
    a.tick_params(which="minor", length=0)
    for s in a.spines.values():
        s.set_visible(False)
    a.set_title("(a) Gates on validation rows (pass* = passes only under the amended rule)",
                fontsize=9.6, fontweight="bold", color=INK, loc="left")
    style_axes(b)
    b.grid(axis="y", visible=False)
    scores = [100 * am[r]["selection_score"] for r in runs]
    colors = [V2_C if r == "R3" else ("#8fb8ea" if am[r]["all_passed"] else "#C9CED4") for r in runs]
    b.barh(range(len(runs)), scores, color=colors, height=0.6, zorder=3)
    for i, v in enumerate(scores):
        b.text(v + 0.08, i, f"+{v:.2f}", va="center", fontsize=7.8, color=INK)
    b.set_xlim(0, 6)
    b.set_title("(b) Selection score (R@1 points)", fontsize=9.6, fontweight="bold",
                color=INK, loc="left")
    b.set_xlabel("naive minus uniform on val label episodes", fontsize=8.4)
    b.text(5.95, 0.3, "blue: passes all\namended gates;\ndark blue: selected", ha="right",
           fontsize=7.4, color=MUTED, va="center")
    fig.tight_layout(w_pad=1.5)
    save(fig, "repair_grid")


# --------------------------------------------------------------------------
# Sections 4.5 and 4.6: condition use on mined and on human-label episodes
# --------------------------------------------------------------------------
def fig_condition_eval():
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.9), dpi=200,
                             gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    a, b, c = axes
    for ax in axes:
        style_axes(ax)
    a.set_title("(a) Mined episodes, R0 factors, row split\n(" + S("29 Sep, ", "") + "scale-fair, R@1 %)",
                fontsize=9.4, fontweight="bold", color=INK, loc="left")
    rows = [("naive rule", 58.1, 49.6, V2_C), ("uniform weights", 54.5, 46.4, REF_C),
            ("oracle (true factor)", 54.7, 48.4, REF_C), ("trained head", 55.3, 46.2, "#C9CED4"),
            ("shuffled condition", 54.4, 46.0, "#C9CED4"), ("CLIP only", 13.7, 21.7, REF_C)]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, i2t, t2i, color) in zip(y, rows):
        a.barh(yi + 0.17, i2t, height=0.32, color=color, zorder=3)
        a.barh(yi - 0.17, t2i, height=0.32, color=color, alpha=0.55, zorder=3)
        a.text(i2t + 0.8, yi + 0.17, f"{i2t:.1f}", va="center", fontsize=7.2, color=INK)
        a.text(t2i + 0.8, yi - 0.17, f"{t2i:.1f}", va="center", fontsize=7.2, color=INK)
    a.set_yticks(y)
    a.set_yticklabels([r[0] for r in rows], fontsize=8.2)
    a.set_xlim(0, 70)
    a.axvline(100 / 13, color=INK, lw=0.8, ls=(0, (1, 2)))
    a.set_xlabel("R@1 (%): upper bar i2t, lower bar t2i; dotted = chance", fontsize=8)

    b.set_title("(b) Human-label episodes, held paintings\n(painting split, R@1 %)", fontsize=9.4,
                fontweight="bold", color=INK, loc="left")
    rows = [("R3 naive", 17.6, 20.9, V2_C), ("R3 uniform", 15.6, 15.7, "#8fb8ea"),
            ("R0 naive", 13.1, 16.7, PERCEPT_C), ("R0 uniform", 11.9, 15.9, "#f5b99e"),
            ("CLIP only", 11.5, 14.9, REF_C)]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, i2t, t2i, color) in zip(y, rows):
        b.barh(yi + 0.17, i2t, height=0.32, color=color, zorder=3)
        b.barh(yi - 0.17, t2i, height=0.32, color=color, alpha=0.6, zorder=3)
        b.text(i2t + 0.3, yi + 0.17, f"{i2t:.1f}", va="center", fontsize=7.2, color=INK)
        b.text(t2i + 0.3, yi - 0.17, f"{t2i:.1f}", va="center", fontsize=7.2, color=INK)
    b.axvline(100 / 13, color=INK, lw=0.8, ls=(0, (1, 2)))
    b.set_yticks(y)
    b.set_yticklabels([r[0] for r in rows], fontsize=8.2)
    b.set_xlim(0, 25)
    b.set_xlabel("R@1 (%): upper bar i2t, lower bar t2i; dotted = chance", fontsize=8)

    c.set_title("(c) Paired differences, held (95% CI)", fontsize=9.4, fontweight="bold",
                color=INK, loc="left")
    diffs = [("R3 naive - R0 naive, i2t", 4.5, 2.7, 6.3),
             ("R3 naive - R0 naive, t2i", 4.2, 2.2, 6.2),
             ("condition-specific part\n(interaction), i2t", 0.78, -1.03, 2.64),
             ("condition-specific part\n(interaction), t2i", 4.39, 2.39, 6.40)]
    y = np.arange(len(diffs))[::-1]
    for yi, (lab, pt, lo, hi) in zip(y, diffs):
        col = V2_C if lo > 0 else REF_C
        c.errorbar(pt, yi, xerr=[[pt - lo], [hi - pt]], fmt="o", color=col, ms=6, capsize=3,
                   lw=1.3, zorder=3)
        c.text(hi + 0.2, yi, f"{pt:+.1f}", va="center", fontsize=7.6, color=INK)
    c.axvline(0, color=INK, lw=1.0)
    c.set_yticks(y)
    c.set_yticklabels([d[0] for d in diffs], fontsize=8)
    c.set_xlim(-2, 8.5)
    c.set_xlabel("R@1 points", fontsize=8.4)
    fig.tight_layout(w_pad=1.5)
    save(fig, "condition_eval")


FIGURES = {
    "timeline": fig_timeline,
    "dec_hybrids": fig_dec_hybrids,
    "stage2_race": fig_stage2_race,
    "redcaps": fig_redcaps,
    "sweep_qc": fig_sweep_qc,
    "h2h": fig_h2h_final,
    "block1": fig_block1,
    "factor_collapse": fig_factor_collapse,
    "repair_grid": fig_repair_grid,
    "condition_eval": fig_condition_eval,
    "stage_d_scorer": fig_stage_d_scorer,
    "stage_d_selection": fig_stage_d_selection,
    "stage_d_final": fig_stage_d_final,
    "v2_design": fig_v2_design,
    "percept_losses": fig_percept_losses,
    "affect_routes": fig_affect_routes,
    "composition": fig_community_composition,
    "ami_plane": fig_ami_plane,
    "occupancy": fig_occupancy,
    "percept_vs_buddy": fig_percept_vs_buddy,
}

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--slides" in args:
        args.remove("--slides")
        SLIDES = True
    names = args or [n for n in FIGURES if not (SLIDES and n == "timeline")]
    for name in names:
        FIGURES[name]()
