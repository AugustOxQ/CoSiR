"""Figures 7 to 9 of the affect steering stage report: protocol, after_go, mllm.
Imported by build_figures.py (which writes figure_data.json); run build_figures.py.
Held numbers come from src/test/20261125_artelingo_held_test/results/descriptive.json; the development-episode numbers
are typed from the auto reports named in each function (checked against them on 2026-10-12).
"""
import json, os, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = "/project/CoSiR"
OUT = os.path.dirname(os.path.abspath(__file__))
HELD = json.load(open(f"{ROOT}/src/test/20261125_artelingo_held_test/results/descriptive.json"))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
AFF, ORANGE, MLLM, DTS, GREY = "#2a78d6", "#e07b39", "#c4576b", "#d9a21b", "#8a8984"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white"})


def sg(v, d=2):
    """Signed number, rounded half up, with a Unicode minus."""
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(repr(round(v, 6))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return (f"+{q}" if q >= 0 else f"{q}").replace("-", "\u2212")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- protocol
def protocol():
    LH, HEAD = 0.235, 0.62
    W = 19.4
    bw, gap, x0 = 4.2, 3.3, 0.15
    boxes = [
        ("1  Development episodes", GREY, "#f1f0ee",
         ["Seed 42, 12,288 episodes from the 6,451 development paintings.",
          "Reused to choose methods (about 50 variants were read here)."]),
        ("2  Fresh test", AFF, "#e6f0fb",
         ["New seeds from the same paintings, used once (AFF: seeds 49 to 51, 36,864 episodes).",
          "Decision rule committed before the episodes were built.", "Seven checks."]),
        ("3  Held read", "#1d8f78", "#e5f3ef",
         ["The 12,281 held paintings, never used to fit, choose or tune. Every weight frozen.",
          "Seven checks with Holm.",
          "One main read and one reserve read (AFF: seeds 52 to 54, 36,864 episodes)."]),
    ]
    wrap = int(bw * 7.6)
    lines = [[textwrap.fill(p, wrap) for p in b[3]] for b in boxes]
    nl = max(sum(p.count("\n") + 1 for p in ls) for ls in lines)
    bh = HEAD + nl * LH + 0.2 + 0.1 * 3
    strip_h, note_h = 1.15, 0.62
    H = 0.3 + note_h + 0.35 + strip_h + 0.5 + bh + 0.5
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    by = H - 0.5 - bh
    arrows = ["margin ≥ +0.5 R@1 over the strongest condition-free comparator, lower bound > 0", "GO"]
    for i, ((t, ec, fc, _), ls) in enumerate(zip(boxes, lines)):
        x = x0 + i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, by), bw, bh, boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=ec, lw=2.8,
                                    zorder=2))
        ax.text(x + 0.18, by + bh - 0.14, t, fontsize=14, fontweight="bold", color=INK, va="top", zorder=3)
        yy = by + bh - HEAD
        for p in ls:
            ax.text(x + 0.18, yy, p, fontsize=12, color=INK, va="top", zorder=3, linespacing=1.25)
            yy -= (p.count("\n") + 1) * LH + 0.1
        if i < 2:
            xa, xb = x + bw + 0.08, x + bw + gap - 0.08
            ax.add_patch(FancyArrowPatch((xa, by + bh / 2), (xb, by + bh / 2), arrowstyle="-|>", mutation_scale=26,
                                         lw=3, color=INK2, zorder=1, shrinkA=0, shrinkB=0))
            lab = textwrap.fill(arrows[i], 25) if i == 0 else arrows[i]
            ax.text((xa + xb) / 2, by + bh / 2 + 0.15, lab, fontsize=12, fontweight="bold" if i else "normal",
                    ha="center", va="bottom", color=INK)
    sy = by - 0.35 - strip_h
    ax.add_patch(FancyBboxPatch((x0, sy), W - 2 * x0, strip_h, boxstyle="round,pad=0,rounding_size=0.1", fc="white",
                                ec=INK2, lw=1.8, zorder=2))
    ax.text(x0 + 0.2, sy + strip_h - 0.15, "The seven checks", fontsize=13, fontweight="bold", va="top")
    ax.text(x0 + 0.2, sy + strip_h - 0.5, textwrap.fill(
        "AFF minus cosine, RCA, B, B′(A0), its matched control on R@1; condition gain over the condition-free "
        "scores and over RCA. Every 95% lower bound must be above 0.", 150), fontsize=12, va="top", linespacing=1.3)
    ny = sy - 0.3 - note_h
    ax.add_patch(FancyBboxPatch((x0, ny), 11.2, note_h, boxstyle="round,pad=0,rounding_size=0.1", fc="white",
                                ec=GREY, lw=1.6, ls="--", zorder=2))
    ax.text(x0 + 0.2, ny + note_h / 2, "Side note, training split: 36,518 training paintings: groupings, heads, "
            "reader, factors; no evaluation labels", fontsize=11.5, va="center", color=INK2)
    ax.set_title("Every claim passes a development bar, a fresh test and one held read", fontsize=16, loc="left",
                 pad=10)
    save(fig, "protocol.png")
    return dict(note="hand-typed diagram; no plotted data", stages=[b[0] for b in boxes])


# --------------------------------------------------------------------------- after_go
def after_go():
    """Development episodes, seed 42, bar margins in R@1 points.
    Sources: 2026-11-22_round4_aff_vetoes.md (AFF over B'(A0) +0.700 [+0.460,+0.937]; over B'(A1) +0.332 [+0.048,+0.625];
    V4 +0.663 [+0.429,+0.894]; V2 +0.295 [+0.010,+0.578]; V24 +0.262 [-0.016,+0.541]; V4 net -18 vs AFF),
    2026-11-23_idea3_goemotions.md (G-T +0.356 [+0.124,+0.583]; G-TF +0.427 [+0.185,+0.667]),
    2026-11-26_design_l_style_source.md (S1 best +0.25 [+0.05,+0.46], over B'(A1))."""
    rows = [
        ("AFF itself\n(over B′(A0))", .700, .460, .937, True, ""),
        ("AFF itself\n(over B′(A1))", .332, .048, .625, True, ""),
        ("Round 4: abstain when both sides agree on\nthe image grouping (over B′(A0))", .663, .429, .894, False,
         "lost 18 net rankings to AFF"),
        ("Round 4: steer only if a CSD-aware reader\nalso picks affect (over B′(A1))", .295, .010, .578, False, ""),
        ("Round 4: both vetoes\n(over B′(A1))", .262, -.016, .541, False, ""),
        ("Round 5: GoEmotions caption placement,\nsteering term only (over B′(A0))", .356, .124, .583, False, ""),
        ("Round 5: GoEmotions caption placement,\nalso in the reader (over B′(A0))", .427, .185, .667, False, ""),
        ("Round 7: best of 12 style-grouping\nscorers (over B′(A1))", .25, .05, .46, False, ""),
    ]
    ys = [0, 1, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6]
    fig, ax = plt.subplots(figsize=(14, 8))
    data = []
    for (lab, v, lo, hi, isaff, note), y in zip(rows, ys):
        col = AFF if isaff else ORANGE
        ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="o", ms=11, color=col, ecolor=col, elinewidth=3, capsize=5,
                    zorder=4)
        txt = f"{sg(v)} [{sg(lo)}, {sg(hi)}]"
        ax.text(max(hi, 0.5) + 0.04, y - (0.12 if note else 0), txt, va="center", ha="left", fontsize=12, color=INK)
        if note:
            ax.text(max(hi, 0.5) + 0.04, y + 0.3, note, va="center", ha="left", fontsize=12, color=ORANGE, style="italic")
        data.append(dict(row=lab.replace("\n", " "), value=v, ci95=[lo, hi], colour=col))
    ax.axvline(0, color=INK, lw=1.6, zorder=2)
    ax.axvline(0.5, color=INK2, lw=1.8, ls="--", zorder=2)
    ax.text(0.5, -1.6, "development bar +0.5", fontsize=12, ha="center", va="bottom", color=INK2)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=12)
    ax.set_ylim(8.2, -2.0); ax.set_xlim(-0.25, 1.65)
    ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_xticklabels([sg(t, 1) if t else "0" for t in
                                                                           [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]],
                                                                          fontsize=12)
    ax.set_xlabel("Margin over each row's own bar comparator, R@1 points (95% interval)", fontsize=13)
    ax.grid(axis="x", color=GRID, zorder=0); ax.tick_params(axis="y", length=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=10, color=AFF, label="AFF"),
                       Line2D([], [], marker="o", ls="", ms=10, color=ORANGE, label="attempt to improve AFF (did not work)")],
              loc="upper right", fontsize=12, frameon=False)
    ax.set_title("After the GO, no attempt to improve AFF passed the development bar and beat AFF", fontsize=15,
                 loc="left", pad=12)
    fig.text(0.02, -0.01, "Development episodes, seed 42 (12,288 episodes). B′(A0) and B′(A1) are the "
             "condition-free scores without and with the style (CSD) features.", fontsize=11.5, color=INK2, va="top")
    save(fig, "after_go.png")
    return data


# --------------------------------------------------------------------------- mllm
def mllm():
    H = HELD["rows"]
    cos52 = H["cosine"]["per_seed"]["52"]["r1"]["point"]
    mllm_r1, mllm_g = H["MLLM"]["pooled"]["r1"]["point"], H["MLLM"]["pooled"]["gain"]
    dts_r1 = H["DTS"]["pooled"]["r1"]["point"]
    dts_g = H["DTS"]["pooled"]["gain"]
    cosp = H["cosine"]["pooled"]["r1"]["point"]
    aff_d, aff_g = H["cosine"]["pooled"]["aff_minus"]["r1"], H["aff_fused"]["pooled"]["gain"]
    # label, colour, r1-minus-cosine (point, ci or None), gain (point, ci)
    rows = [
        ("Qwen3-VL-2B in context\n(fresh seed 44, 900 episodes)", MLLM, (0.22, (-1.13, 1.53)), (-0.53, (-1.68, 0.66))),
        ("Qwen3-VL-8B in context\n(fresh seed 46, 1,800 episodes)", MLLM, (1.07, (0.17, 1.93)), (0.21, (-0.51, 0.94))),
        ("Qwen3-VL-8B in context, held\n(seed 52, 12,288 episodes)", MLLM, (mllm_r1 - cos52, None),
         (mllm_g["point"], mllm_g["ci95"])),
        ("describe-then-score, held\n(seeds 52 to 54)", DTS, (dts_r1 - cosp, None), (dts_g["point"], dts_g["ci95"])),
        ("AFF, held, for reference\n(seeds 52 to 54)", AFF, (aff_d["point"], aff_d["ci95"]), (aff_g["point"], aff_g["ci95"])),
    ]
    ys = [0, 1, 2, 3, 4.3]
    fig, axs = plt.subplots(1, 2, figsize=(15, 7), sharey=True, gridspec_kw=dict(wspace=0.06))
    xl = [(-2.3, 4.6), (-2.3, 3.6)]
    data = []
    for k, (ax, name, ttl) in enumerate(zip(axs, ["r1", "gain"],
                                            ["R@1 minus cosine on the same episodes", "Condition gain"])):
        for (lab, col, a, b), y in zip(rows, ys):
            v, ci = (a if k == 0 else b)
            if col == AFF:
                ax.text(xl[k][1] - (1.75 if k == 0 else 1.15), y, f"{sg(v)} [{sg(ci[0])}, {sg(ci[1])}]\noff the axis",
                        va="center", ha="right", fontsize=12, color=AFF, fontweight="bold")
                ax.annotate("", xy=(xl[k][1] - 0.05, y), xytext=(xl[k][1] - (1.65 if k == 0 else 1.05), y),
                            arrowprops=dict(arrowstyle="-|>", color=AFF, lw=2.5))
            else:
                if ci:
                    ax.errorbar(v, y, xerr=[[v - ci[0]], [ci[1] - v]], fmt="o", ms=10, color=col, ecolor=col,
                                elinewidth=3, capsize=5, zorder=4)
                    txt = f"{sg(v)} [{sg(ci[0])}, {sg(ci[1])}]"
                    ax.text(ci[1] + 0.08, y, txt, va="center", ha="left", fontsize=12)
                else:
                    ax.plot(v, y, "o", ms=10, color=col, zorder=4)
                    ax.text(v + (0.12 if v >= 0 else -0.12), y, f"{sg(v)} (no interval)", va="center", ha="left" if v >= 0 else "right", fontsize=12)
            data.append(dict(panel=name, row=lab.replace("\n", " "), value=v, ci95=list(ci) if ci else None))
        ax.axvline(0, color=INK, lw=1.6, zorder=2)
        ax.set_xlim(*xl[k]); ax.grid(axis="x", color=GRID, zorder=0)
        ax.set_title(ttl, fontsize=14, loc="left")
        ax.set_xlabel("R@1 points (95% interval)" if k == 0 else "points (95% interval)", fontsize=13)
        ax.tick_params(axis="x", labelsize=12)
        ax.set_xticks(list(range(-2, 5)) if k == 0 else list(range(-2, 4)))
        ax.set_xticklabels([sg(t, 0) if t else "0" for t in (range(-2, 5) if k == 0 else range(-2, 4))])
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
    axs[0].set_yticks(ys); axs[0].set_yticklabels([r[0] for r in rows], fontsize=12)
    axs[1].tick_params(axis="y", length=0); axs[0].tick_params(axis="y", length=0); axs[0].set_ylim(5.0, -0.7)
    fig.suptitle("MLLMs find aspect candidates a little more often than cosine, but none chooses the shown aspect",
                 fontsize=15, x=0.02, ha="left", y=0.99)
    fig.text(0.02, 0.0,
             f"Held MLLM row: R@1 {mllm_r1:.2f} against cosine {cos52:.2f} on the same seed 52 ({sg(mllm_r1 - cos52)}); "
             f"describe-then-score: R@1 {dts_r1:.2f} against cosine {cosp:.2f}, pooled over seeds 52 to 54 "
             f"({sg(dts_r1 - cosp)}).\nNo interval is available for these two differences. Condition-free cosine has a "
             "condition gain of 0. Held rows: 36,864 episodes (MLLM: 12,288).",
             fontsize=11.5, color=INK2, ha="left", va="top")
    save(fig, "mllm.png")
    return data
