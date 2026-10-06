"""Figures for docs/user_read/2026-10-06_reader_fix_affect_steering.md, the user-read briefing on the CoSiR v2 reader-fix
line (rounds 1 and 2, the brainstorm and round 3). Full report: docs/reports/auto/v2/2026-11-21_round3_affect_gate.md.

Every chart reads stored outputs; nothing is recomputed except differences of stored values.
  src/test/20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.json   round 1's confidence-gated reader (seed 42)
  src/test/20261118_reader_fix_round2/results/cand_R{1,2,3}_A0.json        round 2's three readers (seed 42)
  src/test/20261120_r1_levers_brainstorm/results/bs_03_sxg.json            the brainstorm's per-pair, per-condition split
  src/test/20261121_round3_affect_gate/results/regression_check.json      affect steering on seed 42 (brainstorm numbers)
  src/test/20261121_round3_affect_gate/results/go_pooled.json             the seven GO checks and the secondary check
  src/test/20261121_round3_affect_gate/results/descriptive.json           per pair, R1 on the test seeds, the control
The script asserts that the stored values match the numbers the full reports print.

Run from the repo root:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python docs/user_read/assets/2026-10-06_reader_fix_affect_steering/build_figures.py
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
R1RES = ROOT / "src/test/20261117_reader_fix_csd/results"
R2RES = ROOT / "src/test/20261118_reader_fix_round2/results"
BSRES = ROOT / "src/test/20261120_r1_levers_brainstorm/results"
R3RES = ROOT / "src/test/20261121_round3_affect_gate/results"
OUT = Path(__file__).resolve().parent
DPI = 150

# Colours: the dataviz reference palette, first three categorical slots (validated all-pairs, light mode). The
# confidence-gated reader keeps the violet it had in the round-1 briefing. Every mark is also labelled directly.
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
C_AFF = "#2a78d6"   # one-sided affect steering
C_GATE = "#4a3aa7"  # confidence-gated reader
C_RAND = "#1baf7a"  # random one-sided control
C_REST = "#8a8984"  # non-highlighted bars (single series)

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
WHITE_BG = dict(fc="white", ec="none", pad=1.5)
PAIRS = ["emotion__style", "emotion__genre", "style__genre"]
PAIR_LABEL = {"emotion__style": "emotion × style", "emotion__genre": "emotion × genre",
              "style__genre": "style × genre"}


def load(p):
    return json.loads(Path(p).read_text())


def close(a, b, tol=6e-4):
    assert abs(a - b) < tol, (a, b)


def sg(v, d=3):
    """Signed number with a true minus sign."""
    return f"{v:+.{d}f}".replace("-", "−")


def pci(d):
    """(point, lower, upper) from a {'point', 'ci95'} record."""
    return d["point"], d["ci95"][0], d["ci95"][1]


def title(fig, text, y=0.975):
    fig.text(0.02, y, text, fontsize=14.5, fontweight="bold", va="top", color=INK)


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print("wrote", name)


def clean_y(ax):
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def dot(ax, y, p, lo, hi, c, hollow=False, label_d=3):
    ax.plot([lo, hi], [y, y], color=c, lw=2.4, solid_capstyle="round", zorder=3)
    if hollow:
        ax.plot(p, y, "o", ms=11, mfc="white", mec=c, mew=2.4, zorder=4)
    else:
        ax.plot(p, y, "o", ms=11, color=c, mec="white", mew=1.8, zorder=4)
    ax.text(hi + 0.03, y, sg(p, label_d), va="center", fontsize=12, color=INK, bbox=WHITE_BG)


# ----------------------------------------------------------------------------------------------- data
def data():
    d = {}
    # round 1: the confidence-gated reader (R-c) on seed 42, against its matched counterpart
    rc = load(R1RES / "cand_Rc_Rb_expected_A0.json")
    assert rc["bar"]["comparator"] == "counterpart"
    d["r1_gate"] = pci(rc["bar"]["r1"])
    # round 2: R1 (the same fused reader), R2 and R3 on seed 42
    for k in ("R1", "R2", "R3"):
        c = load(R2RES / f"cand_{k}_A0.json")
        d[f"r2_{k}"] = pci(c["bar"]["r1"])
        d[f"r2_{k}_pick"] = c["pick_accuracy"]["correct_share"]["point"]
    # affect steering on seed 42, the brainstorm's numbers as the round-3 regression check reproduced them
    reg = load(R3RES / "regression_check.json")
    assert reg["passed"]
    got = {c["name"]: c["got"] for c in reg["comparisons"] if c["item"] == 3}
    d["aff_42"] = tuple(got["bar_margin"])
    # the brainstorm's split of the gated reader's margin (fused minus matched counterpart) by pair and condition
    bs = load(BSRES / "bs_03_sxg.json")["per_condition"]
    d["bs_split"] = {(p, c): bs[p][c]["fused"]["r1"] - bs[p][c]["cf"]["r1"] for p in PAIRS for c in "ab"}
    # round 3, pooled over the fresh seeds 49 to 51
    go = load(R3RES / "go_pooled.json")
    assert go["go"] and go["secondary"]["pass"]
    d["aff_fresh"] = pci(go["checks"]["r1_vs_Bprime"])
    d["aff_vs_cf"] = pci(go["checks"]["r1_vs_counterpart"])
    d["secondary"] = pci(go["secondary"])
    desc = load(R3RES / "descriptive.json")
    r1c = desc["item3_R1_checks"]
    assert r1c["bar_margin_pooled"]["comparator"] == "B_prime"
    d["gate_fresh"] = pci(r1c["bar_margin_pooled"]["r1"])
    pp = desc["item1_per_seed_and_per_pair"]["per_pair_pooled"]
    d["aff_pair"] = {p: pci(pp[p]["checks"]["r1_vs_Bprime"]) for p in PAIRS}
    d["gate_pair"] = {p: pci(r1c["bar_margin_pooled"]["per_pair_r1"][p]) for p in PAIRS}
    rs = desc["item7_random_share_control"]["draws"]
    d["rand"] = [pci(rs[r]["bar_margin_pooled"]["r1"]) for r in ("r0", "r1")]
    d["aff_minus_rand"] = [pci(rs[r]["AFF_minus_control_bar_margin"]) for r in ("r0", "r1")]
    for r in ("r0", "r1"):
        assert rs[r]["bar_margin_pooled"]["comparator"] == "B_prime"

    # numbers the full reports print
    close(d["r1_gate"][0], 0.444); close(d["r1_gate"][1], 0.216); close(d["r1_gate"][2], 0.674)
    close(d["r2_R1"][0], 0.472); close(d["r2_R2"][0], 0.116); close(d["r2_R3"][0], 0.077)
    close(d["r2_R1_pick"], 51.3, 0.05); close(d["r2_R2_pick"], 47.2, 0.05); close(d["r2_R3_pick"], 55.7, 0.05)
    close(d["aff_42"][0], 0.700); close(d["aff_42"][1], 0.460); close(d["aff_42"][2], 0.937)
    for (p, c), v in {("emotion__style", "a"): 1.40, ("emotion__style", "b"): 0.01, ("emotion__genre", "a"): 2.70,
                      ("emotion__genre", "b"): -0.26, ("style__genre", "a"): -0.18,
                      ("style__genre", "b"): -1.013}.items():
        close(d["bs_split"][(p, c)], v, 0.006)
    close(d["aff_fresh"][0], 0.591); close(d["aff_fresh"][1], 0.462); close(d["aff_fresh"][2], 0.729)
    close(d["aff_vs_cf"][0], 0.796)
    close(d["secondary"][0], 0.202); close(d["secondary"][1], 0.093); close(d["secondary"][2], 0.309)
    close(d["gate_fresh"][0], 0.389); close(d["gate_fresh"][1], 0.254); close(d["gate_fresh"][2], 0.532)
    for p, v in zip(PAIRS, (0.810, 1.544, -0.580)):
        close(d["aff_pair"][p][0], v)
    for p, v in zip(PAIRS, (0.690, 1.086, -0.608)):
        close(d["gate_pair"][p][0], v)
    close(d["rand"][0][0], 0.633); close(d["rand"][1][0], 0.562)
    close(d["aff_minus_rand"][0][0], -0.042); close(d["aff_minus_rand"][1][0], 0.029)
    close(d["aff_fresh"][0] / d["aff_42"][0], 0.84, 0.005)  # "kept 84% of its seed-42 value"
    return d


# ----------------------------------------------------------------------------------------------- 1 the line
def fig1_line(d):
    groups = [
        ("Development episodes (reused)", [
            ("Round 1: confidence-gated reader", d["r1_gate"], C_GATE, False),
            ("Round 2: same reader, still best", d["r2_R1"], C_GATE, False),
            ("Brainstorm: affect steering\n(picked from ~50 variants)", d["aff_42"], C_AFF, True),
        ]),
        ("Fresh episodes (three new draws)", [
            ("Round 3: affect steering", d["aff_fresh"], C_AFF, False),
            ("Round 3: confidence-gated\nreader, run beside it", d["gate_fresh"], C_GATE, False),
        ]),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 5.9))
    fig.subplots_adjust(left=0.42, right=0.97, top=0.85, bottom=0.12)
    y, ys, labels, heads, fresh_ys = 0.0, [], [], [], []
    for gi, (head, items) in enumerate(groups):
        heads.append(y)
        y -= 0.85
        for lab, (p, lo, hi), c, hollow in items:
            dot(ax, y, p, lo, hi, c, hollow)
            ys.append(y)
            labels.append(lab)
            if gi == 1:
                fresh_ys.append(y)
            y -= 1.05
        y -= 0.35
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    for yh, (head, _) in zip(heads, groups):
        ax.text(0.02, yh, head, fontsize=12.5, fontweight="bold", va="center", ha="left", transform=tr)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.axvline(0.5, color=INK, lw=1.4, ls=(0, (5, 3)))
    ax.text(0.515, heads[0], "bar +0.5", fontsize=12, va="center", color=INK)
    # the prior written before the fresh test: about +0.3
    y0, y1 = min(fresh_ys) - 0.45, max(fresh_ys) + 0.45
    ax.plot([0.3, 0.3], [y0, y1], color=INK2, lw=1.4, ls=(0, (1, 2)))
    ax.text(0.29, y0 - 0.08, "expected\nbeforehand +0.3", fontsize=11, ha="right", va="top", color=INK2,
            bbox=WHITE_BG)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=12)
    clean_y(ax)
    ax.set_xlim(-0.08, 1.12)
    ax.set_ylim(min(ys) - 1.45, heads[0] + 0.5)
    ax.set_xlabel("bar margin, R@1 points (95% interval)")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    fig.legend(handles=[Line2D([], [], color=C_GATE, marker="o", lw=2.4, ms=10, label="confidence-gated reader"),
                        Line2D([], [], color=C_AFF, marker="o", lw=2.4, ms=10, label="affect steering"),
                        Line2D([], [], color=C_AFF, marker="o", lw=0, ms=10, mfc="white", mew=2.2,
                               label="selected on these episodes")],
               loc="upper left", bbox_to_anchor=(0.02, 0.905), ncol=3, frameon=False, fontsize=10.5,
               handlelength=1.6, columnspacing=1.0)
    title(fig, "Affect steering kept most of its margin on fresh episodes")
    save(fig, "fig1_line.png")


# ----------------------------------------------------------------------------------------------- 2 brainstorm
def fig2_where_margin(d):
    cond = {("emotion__style", "a"): "shows\nemotion", ("emotion__style", "b"): "shows\nstyle",
            ("emotion__genre", "a"): "shows\nemotion", ("emotion__genre", "b"): "shows\ngenre",
            ("style__genre", "a"): "shows\nstyle", ("style__genre", "b"): "shows\ngenre"}
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.8, bottom=0.26)
    xs, w = [], 0.72
    for i, p in enumerate(PAIRS):
        for j, c in enumerate("ab"):
            x = i * 2.6 + j * 1.0
            v = d["bs_split"][(p, c)]
            col = C_GATE if c == "a" and p != "style__genre" else C_REST
            ax.bar(x, v, width=w, color=col, zorder=2)
            ax.text(x, v + (0.08 if v >= 0 else -0.08), sg(v, 1), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=12, color=INK)
            ax.text(x, -1.6, cond[(p, c)], ha="center", va="top", fontsize=11, color=INK2)
            xs.append(x)
        ax.text(i * 2.6 + 0.5, -2.32, PAIR_LABEL[p], ha="center", va="top", fontsize=12.5, color=INK,
                fontweight="bold")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.6, xs[-1] + 0.6)
    ax.set_ylim(-1.5, 3.1)
    ax.set_ylabel("R@1 over the matched control\n(points)")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    fig.legend(handles=[Patch(color=C_GATE, label="the examples show emotion"),
                        Patch(color=C_REST, label="the examples show style or genre")],
               loc="upper left", bbox_to_anchor=(0.02, 0.89), ncol=2, frameon=False, fontsize=11.5)
    title(fig, "The gated reader's margin came from the emotion side")
    save(fig, "fig2_where_margin.png")


# ----------------------------------------------------------------------------------------------- 3 per pair
def fig3_per_pair(d):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    fig.subplots_adjust(left=0.36, right=0.96, top=0.8, bottom=0.12)
    y, ys, labels, heads = 0.0, [], [], []
    for p in PAIRS:
        heads.append(y)
        y -= 0.8
        for lab, v, c in (("affect steering", d["aff_pair"][p], C_AFF),
                          ("confidence-gated reader", d["gate_pair"][p], C_GATE)):
            dot(ax, y, *v, c)
            ys.append(y)
            labels.append(lab)
            y -= 0.95
        y -= 0.3
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    for yh, p in zip(heads, PAIRS):
        ax.text(0.02, yh, PAIR_LABEL[p], fontsize=12.5, fontweight="bold", va="center", ha="left", transform=tr)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=12)
    clean_y(ax)
    ax.set_xlim(-1.0, 2.15)
    ax.set_ylim(min(ys) - 0.6, heads[0] + 0.5)
    ax.set_xlabel("bar margin, R@1 points (95% interval)")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    fig.legend(handles=[Line2D([], [], color=C_AFF, marker="o", lw=2.4, ms=10, label="affect steering"),
                        Line2D([], [], color=C_GATE, marker="o", lw=2.4, ms=10, label="confidence-gated reader")],
               loc="upper left", bbox_to_anchor=(0.02, 0.885), ncol=2, frameon=False, fontsize=11.5)
    title(fig, "The gain sits on the two emotion pairs;\nstyle × genre is below the comparator")
    save(fig, "fig3_per_pair.png")


# ----------------------------------------------------------------------------------------------- 4 control
def fig4_control(d):
    rows = [("affect steering", d["aff_fresh"], C_AFF),
            ("random one-sided gate, draw 1", d["rand"][0], C_RAND),
            ("random one-sided gate, draw 2", d["rand"][1], C_RAND),
            ("confidence-gated reader", d["gate_fresh"], C_GATE)]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    fig.subplots_adjust(left=0.42, right=0.96, top=0.8, bottom=0.17)
    ys = np.arange(len(rows))[::-1].astype(float)
    for y, (lab, v, c) in zip(ys, rows):
        dot(ax, y, *v, c)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=12)
    clean_y(ax)
    ax.set_xlim(-0.05, 0.98)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("bar margin, R@1 points (95% interval)")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    title(fig, "A random gate that steers the same side as often\nmatched affect steering")
    save(fig, "fig4_control.png")


def main():
    d = data()
    fig1_line(d)
    fig2_where_margin(d)
    fig3_per_pair(d)
    fig4_control(d)


if __name__ == "__main__":
    main()
