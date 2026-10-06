"""Figures for docs/user_read/2026-10-06_reader_fix.md, the user-read briefing on the CoSiR v2 reader fix
(full report: docs/reports/auto/v2/2026-11-18_reader_fix_csd.md).

Every chart reads stored outputs; nothing is recomputed except differences and sums of stored values.
  src/test/20261117_reader_fix_csd/results/cand_<name>.json      the seven candidates and the AR runs
  src/test/20261117_reader_fix_csd/results/ra_summary.json       picks that go to the empty grouping (AR check)
  src/test/20261117_reader_fix_csd/results/rb_reader_<cfg>.json  learned reader's accuracy on held-out practice episodes
  src/test/20261116_grouping_step1_style/results/step1_eval_style.json  the current (step-1 arg-max) reader
The history of told and reader margins (figure 1) is typed in from the full report's section 2 table. The script
asserts that the stored values match the numbers the full report prints.

Run from the repo root:
  OMP_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python docs/user_read/assets/2026-10-06_reader_fix/build_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

ROOT = Path(__file__).resolve().parents[4]
RES = ROOT / "src/test/20261117_reader_fix_csd/results"
STEP1 = ROOT / "src/test/20261116_grouping_step1_style/results"
OUT = Path(__file__).resolve().parent
DPI = 150

# Colours: the dataviz reference palette (validated colour-blind-safe), one hue per reader, grey for the current
# reader. Text stays in ink colours; every mark is also labelled directly.
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
C_CUR = "#8a8984"   # current reader
C_NS = "#2a78d6"    # noise-scaled reader (R-a)
C_LA = "#eb6834"    # learned reader, top pick (R-b arg-max)
C_LE = "#1baf7a"    # learned reader, weighted mix (R-b expected)
C_CG = "#4a3aa7"    # confidence-gated reader (R-c)
C_GAIN = "#2a78d6"  # diverging pair: gain bought (blue) against either rate paid (red)
C_COST = "#e34948"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.labelsize": 13,
    "xtick.color": INK2,
    "ytick.color": INK,
    "xtick.labelsize": 12.5,
    "ytick.labelsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def load(p):
    return json.loads(Path(p).read_text())


def close(a, b, tol=6e-4):
    assert abs(a - b) < tol, (a, b)


def sg(v, d=3):
    """Signed number with a true minus sign."""
    return f"{v:+.{d}f}".replace("-", "−")


def title(fig, text, y=0.975):
    fig.text(0.02, y, text, fontsize=14.5, fontweight="bold", va="top", color=INK)


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print("wrote", name)


def clean_y(ax):
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


WHITE_BG = dict(fc="white", ec="none", pad=1.5)


# ----------------------------------------------------------------------------------------------- data
def cur_row(step1, cfg):
    """The current (step-1 arg-max) reader on one configuration, with its terms against its bar comparator."""
    arm = step1["arms"][cfg]
    assert arm["bar"]["comparator"] == "B_prime"
    npz = np.load(STEP1 / "step1_eval_style.npz")
    either_f = npz[f"{cfg}__reader__fused__r1"] + npz[f"{cfg}__reader__fused__other"]
    either_bp = npz[f"{cfg}__Bprime__r1"] + npz[f"{cfg}__Bprime__other"]
    return dict(bar=arm["bar"]["r1"], gain=arm["bar"]["gain_vs_comparator"],
                either_bar=100.0 * float(np.mean(either_f - either_bp)))


def cand_row(key):
    d = load(RES / f"cand_{key}.json")
    comp = d["bar"]["comparator"]
    vs = {"B_prime": d["fused_vs_Bprime"], "counterpart": d["margin"], "B": d["fused_vs_B"]}[comp]
    return dict(bar=d["bar"]["r1"], gain=d["gain_statistic"], either_bar=vs["either"]["point"], raw=d)


def all_rows():
    step1 = load(STEP1 / "step1_eval_style.json")
    rows = {"cur_A0": cur_row(step1, "A0"), "cur_A1": cur_row(step1, "A1")}
    for k in ["Rc_Rb_expected_A0", "Rb_expected_A0", "Ra_A0", "Rb_argmax_A0",
              "Rb_argmax_A1", "Rb_expected_A1", "Ra_A1"]:
        rows[k] = cand_row(k)
    for r in rows.values():  # bar margin = (gain + either change) / 2 against the bar comparator, every row
        close((r["gain"]["point"] + r["either_bar"]) / 2, r["bar"]["point"], 1e-9)
    # numbers the full report prints (Table 1, section 3.2)
    for k, v in {"Rc_Rb_expected_A0": 0.444, "Rb_expected_A0": 0.313, "Ra_A0": 0.230, "Rb_argmax_A0": 0.144,
                 "Rb_argmax_A1": 0.181, "Rb_expected_A1": 0.098, "Ra_A1": -0.045, "cur_A0": 0.313,
                 "cur_A1": 0.008}.items():
        close(rows[k]["bar"]["point"], v)
    close(rows["cur_A0"]["either_bar"], -0.710)
    close(rows["Rb_expected_A0"]["either_bar"], -1.485)
    close(rows["Rc_Rb_expected_A0"]["either_bar"], -1.780)
    close(rows["cur_A0"]["gain"]["point"], 1.337)
    close(rows["Rc_Rb_expected_A0"]["gain"]["point"], 2.667)
    # the gap to the bar in rankings: 12,288 episodes x 4 rankings = 49,152
    n_rank = rows["Rc_Rb_expected_A0"]["raw"]["n_episodes"] * 4
    assert n_rank == 49152, n_rank
    net = rows["Rc_Rb_expected_A0"]["bar"]["point"] / 100 * n_rank
    close(net, 218, 1e-6)
    assert int(np.ceil(0.5 / 100 * n_rank)) == 246
    return rows


# ----------------------------------------------------------------------------------------------- 1 context
def fig1_history():
    # told and label-free reader margins over the matched control; full report section 2 table
    # (stage report section 14; grouping report sections 4 and 9)
    steps = ["4 Oct: first groupings", "5 Oct: new emotion-like grouping", "5 Oct: plus the style grouping"]
    told = [(1.14, 0.90, 1.41), (1.64, 1.37, 1.92), (2.23, 1.93, 2.56)]
    reader = [(0.14, -0.04, 0.32), (0.35, 0.15, 0.57), (0.06, -0.15, 0.28)]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.76, bottom=0.15)
    ys = np.array([2, 1, 0]) * 1.3
    for y, t, r, s in zip(ys, told, reader, steps):
        ax.text(-0.55, y + 0.45, s, fontsize=12.5, va="center", color=INK)
        for (pt, lo, hi), c in [(t, C_NS), (r, C_LA)]:
            ax.plot([lo, hi], [y, y], color=c, lw=2.4, solid_capstyle="round")
            ax.plot(pt, y, "o", ms=11, color=c, mec="white", mew=1.8, zorder=4)
        ax.text(t[2] + 0.06, y, sg(t[0], 2), va="center", fontsize=12.5, color=INK)
        ax.text(r[2] + 0.06, y, sg(r[0], 2), va="center", fontsize=12.5, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(-0.6, 2.95)
    ax.set_ylim(-0.5, ys.max() + 0.8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("R@1 margin over the matched control (points)")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    fig.legend(handles=[Line2D([], [], color=C_NS, marker="o", lw=2.4, ms=10, label="told the right grouping"),
                        Line2D([], [], color=C_LA, marker="o", lw=2.4, ms=10, label="reader")],
               loc="upper left", bbox_to_anchor=(0.02, 0.9), ncol=2, frameon=False, fontsize=12.5)
    title(fig, "The ceiling rose at each step; the reader stayed low")
    save(fig, "fig1_ceiling_vs_reader.png")


# ----------------------------------------------------------------------------------------------- 2 result 1
def fig2_bar_margins(rows):
    groups = [
        ("Without the style grouping", [("confidence-gated", "Rc_Rb_expected_A0", C_CG),
                                        ("learned, weighted", "Rb_expected_A0", C_LE),
                                        ("noise-scaled", "Ra_A0", C_NS),
                                        ("current (reference)", "cur_A0", C_CUR)]),
        ("With the style grouping", [("learned, top pick", "Rb_argmax_A1", C_LA),
                                     ("current (reference)", "cur_A1", C_CUR)]),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    fig.subplots_adjust(left=0.36, right=0.97, top=0.86, bottom=0.12)
    y, ys, labels, heads = 0.0, [], [], []
    for head, items in groups:
        heads.append(y)
        y -= 0.8
        for lab, k, c in items:
            r = rows[k]
            lo, hi = r["bar"]["ci95"]
            ax.plot([lo, hi], [y, y], color=c, lw=2.4, solid_capstyle="round")
            ax.plot(r["bar"]["point"], y, "o", ms=11, color=c, mec="white", mew=1.8, zorder=4)
            ax.text(hi + 0.025, y, sg(r["bar"]["point"]), va="center", fontsize=12, color=INK, bbox=WHITE_BG)
            ys.append(y)
            labels.append(lab)
            y -= 1.0
        y -= 0.3
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    for yh, (head, _) in zip(heads, groups):
        ax.text(0.02, yh, head, fontsize=12.5, fontweight="bold", va="center", ha="left", transform=tr)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.axvline(0.5, color=INK, lw=1.4, ls=(0, (5, 3)))
    ax.text(0.515, heads[0], "bar +0.5", fontsize=12, va="center", color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    clean_y(ax)
    ax.set_xlim(-0.33, 0.95)
    ax.set_ylim(min(ys) - 0.6, heads[0] + 0.5)
    ax.set_xlabel("bar margin, R@1 points (95% interval)")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    title(fig, "No candidate reached the +0.5 bar;\nthe best fell 0.056 short")
    save(fig, "fig2_bar_margins.png")


# ----------------------------------------------------------------------------------------------- 3 result 2
def fig3_gain_either(rows):
    items = [("current reader", "cur_A0"), ("learned, weighted", "Rb_expected_A0"),
             ("confidence-gated", "Rc_Rb_expected_A0")]
    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    fig.subplots_adjust(left=0.3, right=0.97, top=0.72, bottom=0.15)
    ys = np.array([2, 1, 0])
    for y, (lab, k) in zip(ys, items):
        r = rows[k]
        g, e, m = r["gain"]["point"], r["either_bar"], r["bar"]["point"]
        ax.barh(y, g, height=0.5, color=C_GAIN, zorder=2)
        ax.barh(y, e, height=0.5, color=C_COST, zorder=2)
        ax.text(g + 0.06, y, sg(g), va="center", fontsize=12, color=INK)
        ax.text(e - 0.06, y, sg(e), va="center", ha="right", fontsize=12, color=INK)
        ax.plot(m, y, "o", ms=10, color=INK, mec="white", mew=1.6, zorder=4)
        ax.text(m, y + 0.37, f"bar margin {sg(m)}", ha="center", va="bottom", fontsize=11, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([i[0] for i in items])
    clean_y(ax)
    ax.set_xlim(-2.95, 3.35)
    ax.set_ylim(-0.5, 2.75)
    ax.set_xlabel("change against the strongest aspect-blind scorer\n(R@1 points)")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    fig.legend(handles=[Patch(color=C_GAIN, label="condition gain"),
                        Patch(color=C_COST, label="change in either rate"),
                        Line2D([], [], color=INK, marker="o", lw=0, ms=9, label="bar margin = half the sum")],
               loc="upper left", bbox_to_anchor=(0.02, 0.87), ncol=2, frameon=False, fontsize=12)
    title(fig, "More gain, but most of it went back in either rate")
    save(fig, "fig3_gain_vs_either.png")


# ----------------------------------------------------------------------------------------------- 4 result 3
def fig4_empty_grouping():
    ar = load(RES / "ra_summary.json")["ar_check"]
    rb = load(RES / "cand_Rb_argmax_AR.json")["pick_share"]
    rbe = load(RES / "cand_Rb_expected_AR.json")["pick_share"]
    assert rb["overall"]["rand"] == rbe["overall"]["rand"]  # the two scorings share the pick
    vals = [("current reader", ar["reader_step1_argmax"]["overall"]["rand"], C_CUR),
            ("learned", rb["overall"]["rand"], C_LA),
            ("noise-scaled", ar["reader_Ra"]["overall"]["rand"], C_NS)]
    for (_, v, _), want in zip(vals, [20.3, 27.3, 30.5]):
        close(v, want, 0.051)
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    fig.subplots_adjust(left=0.3, right=0.95, top=0.8, bottom=0.2)
    ys = np.array([2, 1, 0])
    for y, (lab, v, c) in zip(ys, vals):
        ax.barh(y, v, height=0.55, color=c)
        ax.text(v + 0.6, y, f"{v:.1f}%", va="center", fontsize=12.5, color=INK, bbox=WHITE_BG, zorder=5)
    ax.axvline(25, color=INK, lw=1.4, ls=(0, (5, 3)))
    ax.text(25.4, 2.55, "chance 25%", fontsize=12, va="center", color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([v[0] for v in vals])
    clean_y(ax)
    ax.set_xlim(0, 37)
    ax.set_ylim(-0.5, 2.85)
    ax.set_xlabel("picks that go to the empty grouping (%)")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    title(fig, "Dividing by noise made the empty grouping\nwin more often, not less")
    save(fig, "fig4_empty_grouping.png")


def fig5_practice_vs_real():
    cfgs = [("A0", "without the\nstyle grouping", 100 / 3), ("A1", "with the\nstyle grouping", 25.0)]
    want = {"A0": (79.71, 79.08, 51.26), "A1": (62.87, 62.70, 48.69)}
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.66, bottom=0.15)
    w = 0.27
    for i, (c, lab, ch) in enumerate(cfgs):
        h = load(RES / f"rb_reader_{c}.json")["halves"]
        b0, b1 = h["0"]["oof_accuracy_at_chosen_C"], h["1"]["oof_accuracy_at_chosen_C"]
        pk = load(RES / f"cand_Rb_argmax_{c}.json")["pick_accuracy"]["correct_share"]["point"]
        for v, t in zip((b0, b1, pk), want[c]):
            close(v, t, 0.006)
        for j, (v, fc, hatch) in enumerate([(b0, C_NS, None), (b1, C_NS, "//"), (pk, C_LA, None)]):
            x = i + (j - 1) * (w + 0.03)
            ax.bar(x, v, width=w, color=fc, hatch=hatch, edgecolor="white", lw=0)
            ax.text(x, v + 1.2, f"{v:.1f}", ha="center", fontsize=12, color=INK)
        ax.plot([i - 0.47, i + 0.47], [ch, ch], color=INK, lw=1.4, ls=(0, (5, 3)))
    ax.set_xticks(range(2))
    ax.set_xticklabels([c[1] for c in cfgs], fontsize=12.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 90)
    ax.set_ylabel("right grouping picked (%)")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    fig.legend(handles=[Patch(color=C_NS, label="practice episodes, copy 1"),
                        Patch(fc=C_NS, hatch="//", ec="white", label="practice episodes, copy 2"),
                        Patch(color=C_LA, label="real development episodes"),
                        Line2D([], [], color=INK, lw=1.4, ls=(0, (5, 3)), label="chance")],
               loc="upper left", bbox_to_anchor=(0.02, 0.87), ncol=1, frameon=False, fontsize=12)
    title(fig, "The learned reader scores much lower on real\nepisodes, on a stricter measure")
    save(fig, "fig5_practice_vs_real.png")


# ----------------------------------------------------------------------------------------------- 5 result 4
def fig6_style_grouping(rows):
    items = [("current reader", "cur_A0", "cur_A1", C_CUR),
             ("noise-scaled", "Ra_A0", "Ra_A1", C_NS),
             ("learned, top pick", "Rb_argmax_A0", "Rb_argmax_A1", C_LA),
             ("learned, weighted", "Rb_expected_A0", "Rb_expected_A1", C_LE)]
    # A1 minus A0 bar margins as the full report prints them (Table 4)
    want = {"current reader": -0.305, "noise-scaled": -0.275, "learned, top pick": 0.037,
            "learned, weighted": -0.216}
    rbs = load(RES / "rb_summary.json")["A1_minus_A0"]
    close(rbs["argmax"]["bar_margin_r1"]["point"], 0.037)
    close(rbs["expected"]["bar_margin_r1"]["point"], -0.216)
    close(load(RES / "ra_summary.json")["A1_minus_A0_under_Ra"]["bar_r1"]["point"], -0.275)
    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    fig.subplots_adjust(left=0.32, right=0.95, top=0.7, bottom=0.15)
    ys = np.array([3, 2, 1, 0])
    for y, (lab, a0, a1, c) in zip(ys, items):
        v0, v1 = rows[a0]["bar"]["point"], rows[a1]["bar"]["point"]
        close(v1 - v0, want[lab])
        ax.annotate("", xy=(v1, y), xytext=(v0, y),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2, shrinkA=7, shrinkB=7, mutation_scale=16))
        ax.plot(v0, y, "o", ms=11, mfc="white", mec=c, mew=2.2, zorder=4)
        ax.plot(v1, y, "o", ms=11, color=c, mec="white", mew=1.6, zorder=4)
        lo_x, hi_x = (v1, v0) if v1 < v0 else (v0, v1)
        ax.text(hi_x + 0.03, y, f"{sg(v1 - v0)}", va="center", fontsize=12, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([i[0] for i in items])
    clean_y(ax)
    ax.set_xlim(-0.12, 0.55)
    ax.set_ylim(-0.5, 3.5)
    ax.set_xlabel("bar margin, R@1 points")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    fig.legend(handles=[Line2D([], [], marker="o", lw=0, ms=11, mfc="white", mec=INK2, mew=2,
                               label="without the style grouping"),
                        Line2D([], [], marker="o", lw=0, ms=11, color=INK2, label="with it")],
               loc="upper left", bbox_to_anchor=(0.02, 0.86), ncol=2, frameon=False, fontsize=12)
    title(fig, "With the style grouping, bar margins\nfell or stayed flat")
    save(fig, "fig6_style_grouping.png")


def main():
    rows = all_rows()
    fig1_history()
    fig2_bar_margins(rows)
    fig3_gain_either(rows)
    fig4_empty_grouping()
    fig5_practice_vs_real()
    fig6_style_grouping(rows)


if __name__ == "__main__":
    main()
