"""Figures for docs/reports/auto/v2/2026-11-18_reader_fix_csd.md (CoSiR v2 reader fix with the CSD grouping).

Reads stored outputs only (no recomputation of any reader):
  src/test/20261117_reader_fix_csd/results/cand_<name>.json      seven candidates and three AR runs
  src/test/20261117_reader_fix_csd/results/ra_summary.json       R-a AR check (picks to rand)
  src/test/20261117_reader_fix_csd/results/rb_summary.json       R-b AR check (picks to rand)
  src/test/20261117_reader_fix_csd/results/rb_reader_<cfg>.json  R-b out-of-fold bank accuracy
  src/test/20261116_grouping_step1_style/results/step1_eval_style.{json,npz}  step-1 arg-max reader (reference rows)

The only quantity computed here is the step-1 reader's either-rate change against B' (mean of the per-episode
(R@1 + other) difference from step1_eval_style.npz); the script checks R@1 = (either + gain) / 2 on every row.

Run from the repo root:
  OMP_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python docs/reports/assets/2026-11-18_reader_fix_csd/build_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

ROOT = Path(__file__).resolve().parents[4]
RES = ROOT / "src/test/20261117_reader_fix_csd/results"
STEP1 = ROOT / "src/test/20261116_grouping_step1_style/results"
OUT = Path(__file__).resolve().parent
DPI = 150

# Colour-blind-safe categorical slots (dataviz reference palette, validated all-pairs in light mode),
# assigned by reader in fixed order; grey for the step-1 reference reader.
C_REF = "#8a8984"
C_RA = "#2a78d6"
C_RBA = "#eb6834"
C_RBE = "#1baf7a"
C_RC = "#4a3aa7"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

READER = {
    "ref": ("arg-max reader (step 1, reference)", C_REF),
    "Ra": ("R-a, scaled Δ", C_RA),
    "Rb_argmax": ("R-b arg-max", C_RBA),
    "Rb_expected": ("R-b expected", C_RBE),
    "Rc": ("R-c, gate on R-b expected", C_RC),
}
MARK = {"A1": "o", "A0": "s", "AR": "^"}
PAIRS = ["emotion__style", "emotion__genre", "style__genre"]
PAIR_LABEL = {"emotion__style": "emotion × style", "emotion__genre": "emotion × genre",
              "style__genre": "style × genre"}
COMP_LABEL = {"B_prime": "B′", "counterpart": "counterpart", "B": "B"}


def load(p):
    return json.loads(Path(p).read_text())


def ref_row(step1, npz, cfg):
    arm = step1["arms"][cfg]
    rd = arm["eval"]["reader"]
    fused = arm["fused_r1"]["reader"]["fused"]["r1"]["point"]
    cf = arm["fused_r1"]["reader"]["counterpart"]["r1"]["point"]
    bp = arm["B_prime"]["describe"]["r1"]["point"]
    comp = arm["bar"]["comparator"]
    assert comp == "B_prime"
    either_f = npz[f"{cfg}__reader__fused__r1"] + npz[f"{cfg}__reader__fused__other"]
    either_bp = npz[f"{cfg}__Bprime__r1"] + npz[f"{cfg}__Bprime__other"]
    either_bar = 100.0 * float(np.mean(either_f - either_bp))
    return dict(
        key=f"ref_{cfg}", reader="ref", config=cfg, fused=fused, cf=cf, B=step1["B"]["r1"]["point"], Bp=bp,
        margin=rd["fusedT_vs_fusedTcf"]["r1"], gain=rd["fusedT_vs_fusedTcf"]["gain"],
        either_margin=rd["fusedT_vs_fusedTcf"]["either"]["point"], bar=arm["bar"]["r1"], comp=comp,
        either_bar=either_bar, pick=arm["eval"]["pick"]["correct_share"]["point"],
        per_pair={p: arm["bar"]["per_pair_r1"][p] for p in PAIRS},
    )


def cand_row(name):
    d = load(RES / f"cand_{name}.json")
    comp = d["bar"]["comparator"]
    either_bar = {"B_prime": d["fused_vs_Bprime"], "counterpart": d["margin"], "B": d["fused_vs_B"]}[comp]
    reader = name.rsplit("_", 1)[0]
    if reader.startswith("Rc"):
        reader = "Rc"
    return dict(
        key=name, reader=reader, config=d["config"], fused=d["r1_means"]["fused"], cf=d["r1_means"]["counterpart"],
        B=d["r1_means"]["B"], Bp=d["r1_means"]["B_prime"], margin=d["margin"]["r1"], gain=d["gain_statistic"],
        either_margin=d["margin"]["either"]["point"], bar=d["bar"]["r1"], comp=comp,
        either_bar=either_bar["either"]["point"], pick=d["pick_accuracy"]["correct_share"]["point"],
        per_pair={p: d["bar"]["per_pair_r1"][p] for p in PAIRS},
    )


def rows():
    step1 = load(STEP1 / "step1_eval_style.json")
    npz = np.load(STEP1 / "step1_eval_style.npz")
    order = [
        ("A1", ["ref", "Ra_A1", "Rb_argmax_A1", "Rb_expected_A1"]),
        ("A0", ["ref", "Ra_A0", "Rb_argmax_A0", "Rb_expected_A0", "Rc_Rb_expected_A0"]),
        ("AR", ["ref", "Ra_AR", "Rb_argmax_AR", "Rb_expected_AR"]),
    ]
    out = []
    for cfg, names in order:
        for n in names:
            r = ref_row(step1, npz, cfg) if n == "ref" else cand_row(n)
            # R@1 = (either + gain) / 2 per episode, so the paired differences obey the same identity
            assert abs((r["gain"]["point"] + r["either_bar"]) / 2 - r["bar"]["point"]) < 1e-9, r["key"]
            assert abs((r["gain"]["point"] + r["either_margin"]) / 2 - r["margin"]["point"]) < 1e-9, r["key"]
            out.append(r)
    return out


CFG_TITLE = {
    "A1": "A1: affect, image, caption, csd",
    "A0": "A0: affect, image, caption",
    "AR": "AR: affect, image, caption, rand (control, not candidates)",
}


def y_layout(rs):
    """y positions top to bottom with a gap and a header slot per configuration."""
    ys, headers, y = [], [], 0.0
    last = None
    for r in rs:
        if r["config"] != last:
            if last is not None:
                y -= 0.6
            headers.append((y, r["config"]))
            y -= 1.0
            last = r["config"]
        ys.append(y)
        y -= 1.0
    return np.array(ys), headers


def style_forest_axis(ax, rs, ys, headers, xlim, show_labels=True):
    ax.set_ylim(ys.min() - 0.8, headers[0][0] + 0.7)
    ax.set_xlim(*xlim)
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    ax.axvline(0.5, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    if show_labels:
        ax.set_yticks(ys)
        ax.set_yticklabels([READER[r["reader"]][0] for r in rs])
        tr = blended_transform_factory(ax.figure.transFigure, ax.transData)
        for y, cfg in headers:
            ax.text(0.02, y, CFG_TITLE[cfg], fontsize=10, fontweight="bold", color=INK, va="center", ha="left",
                    transform=tr)
    else:
        ax.tick_params(axis="y", labelleft=False)


def draw_interval(ax, y, pt, lo, hi, color, hollow):
    ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round", zorder=3)
    ax.plot(pt, y, marker="o", ms=8, color=color, mfc="white" if hollow else color, mew=2, zorder=4)


def fig_bar_margins(rs):
    ys, headers = y_layout(rs)
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    fig.subplots_adjust(left=0.36, right=0.74, top=0.9, bottom=0.1)
    xlim = (-0.45, 0.85)
    style_forest_axis(ax, rs, ys, headers, xlim)
    for y, r in zip(ys, rs):
        c = READER[r["reader"]][1]
        lo, hi = r["bar"]["ci95"]
        draw_interval(ax, y, r["bar"]["point"], lo, hi, c, hollow=(r["config"] == "AR"))
        txt = f"{r['bar']['point']:+.3f} [{lo:+.3f}, {hi:+.3f}]  vs {COMP_LABEL[r['comp']]}"
        ax.text(1.02, y, txt, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=9, color=INK)
    ax.text(1.02, headers[0][0] + 0.55, "bar margin [95% interval], comparator", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=9, color=INK2)
    ax.text(0.5, headers[0][0] + 0.55, "development bar +0.5", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xlabel("Bar margin: fused reader minus bar comparator, R@1 (percentage points)")
    fig.suptitle("Bar margins on seed 42 (development): no candidate reaches +0.5", fontsize=12, x=0.02, ha="left",
                 color=INK)
    fig.text(0.02, 0.025, "Hollow markers: AR runs (random-grouping control, never candidates). "
             "Grey: the step-1 arg-max reader on the same configuration.", fontsize=8.5, color=INK2)
    fig.savefig(OUT / "bar_margins.png", dpi=DPI)
    plt.close(fig)


def fig_gain_either(rs):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.2, wspace=0.08)
    panels = [("either_margin", "Either-rate change vs matched counterpart (pp)", "margin", "(a) Against the matched counterpart"),
              ("either_bar", "Either-rate change vs bar comparator (pp)", "bar margin", "(b) Against the bar comparator (decides the bar)")]
    xs = np.linspace(-2.2, 0.6, 50)
    for ax, (key, xlabel, what, title) in zip(axes, panels):
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.plot(xs, -xs, color=INK2, lw=1.0)
        ax.plot(xs, 1.0 - xs, color=INK, lw=1.2, ls=(0, (4, 3)))
        ax.text(-2.1, 2.1 - 0.12, f"{what} = 0", color=INK2, fontsize=8.5, rotation=-33, ha="left", va="top",
                rotation_mode="anchor")
        ax.text(-0.25, 1.25 + 0.08, f"{what} = +0.5", color=INK, fontsize=8.5, rotation=-33, ha="left",
                va="bottom", rotation_mode="anchor")
        for r in rs:
            c = READER[r["reader"]][1]
            ax.plot(r[key], r["gain"]["point"], marker=MARK[r["config"]], ms=9, color=c,
                    mfc="white" if r["config"] == "AR" else c, mew=2, ls="none", zorder=4)
        rc = [r for r in rs if r["reader"] == "Rc"][0]
        ax.annotate("R-c", (rc[key], rc["gain"]["point"]), xytext=(8, 4), textcoords="offset points", fontsize=9,
                    color=INK)
        ax.set_xlim(-2.2, 0.6)
        ax.set_ylim(0, 3.1)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10.5, loc="left", color=INK)
    axes[0].set_ylabel("Gain statistic (pp)")
    handles = [Line2D([], [], color=READER[k][1], marker="o", ls="none", ms=8, label=READER[k][0]) for k in READER]
    handles += [Line2D([], [], color=INK2, marker=MARK[c], ls="none", ms=8, mfc=("white" if c == "AR" else INK2),
                       mew=2, label={"A1": "A1", "A0": "A0", "AR": "AR (control)"}[c]) for c in MARK]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("What the fused readers gain and lose: R@1 difference = (either change + gain) / 2", fontsize=12,
                 x=0.02, ha="left", color=INK)
    fig.text(0.02, 0.9, "Points above the dashed line would clear +0.5. Every reader buys condition gain at a cost in "
             "either rate; the gate (R-c) buys the most gain and pays the most either rate.", fontsize=9, color=INK2)
    fig.savefig(OUT / "gain_vs_either.png", dpi=DPI)
    plt.close(fig)


def fig_per_pair(rs):
    ys, headers = y_layout(rs)
    fig, axes = plt.subplots(1, 3, figsize=(14, 7.2), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.99, top=0.88, bottom=0.1, wspace=0.07)
    xlim = (-1.5, 1.7)
    for i, (ax, p) in enumerate(zip(axes, PAIRS)):
        style_forest_axis(ax, rs, ys, headers, xlim, show_labels=(i == 0))
        for y, r in zip(ys, rs):
            v = r["per_pair"][p]
            draw_interval(ax, y, v["point"], v["ci95"][0], v["ci95"][1], READER[r["reader"]][1],
                          hollow=(r["config"] == "AR"))
        ax.set_title(PAIR_LABEL[p], fontsize=11, color=INK)
        ax.set_xlabel("Bar margin, R@1 (pp)")
    # short configuration names instead of the long headers of the single-panel figure
    for t in list(axes[0].texts):
        if t.get_text() in CFG_TITLE.values():
            t.set_text(t.get_text().split(":")[0] + (" (control)" if t.get_text().startswith("AR") else ""))
    fig.suptitle("Bar margins per aspect pair (same comparator as the pooled bar; descriptive, not tested)",
                 fontsize=12, x=0.02, ha="left", color=INK)
    fig.text(0.02, 0.025, "Dashed line: +0.5. Hollow markers: AR runs. On style × genre every A0 and A1 reader has a "
             "negative bar margin (point estimate).", fontsize=8.5, color=INK2)
    fig.savefig(OUT / "per_pair_bar_margins.png", dpi=DPI)
    plt.close(fig)


def fig_levels(rs):
    ys, headers = y_layout(rs)
    fig, ax = plt.subplots(figsize=(11, 7.2))
    fig.subplots_adjust(left=0.36, right=0.97, top=0.88, bottom=0.17)
    xlim = (-0.15, 0.75)
    ax.set_ylim(ys.min() - 0.8, headers[0][0] + 0.7)
    ax.set_xlim(*xlim)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_yticks(ys)
    ax.set_yticklabels([READER[r["reader"]][0] for r in rs])
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    for y, cfg in headers:
        ax.text(0.02, y, CFG_TITLE[cfg], fontsize=10, fontweight="bold", color=INK, va="center", ha="left",
                transform=tr)
    for y, r in zip(ys, rs):
        c = READER[r["reader"]][1]
        B = r["B"]
        comp_val = {"B_prime": r["Bp"], "counterpart": r["cf"], "B": B}[r["comp"]]
        ax.plot([comp_val - B, r["fused"] - B], [y, y], color=c, lw=2, alpha=0.55, zorder=2,
                solid_capstyle="round")
        ax.plot(r["Bp"] - B, y, marker="D", ms=7, color=INK2, mfc=INK2, ls="none", zorder=3)
        ax.plot(r["cf"] - B, y, marker="o", ms=8, color=c, mfc="white", mew=2, ls="none", zorder=4)
        ax.plot(r["fused"] - B, y, marker="o", ms=8, color=c, mfc=c, mew=2, ls="none", zorder=5)
    ax.set_xlabel("R@1 minus B (percentage points; B = 18.341)")
    handles = [Line2D([], [], color=INK2, marker="D", ls="none", ms=7, label="B′ (B rebuilt with the configuration's groupings)"),
               Line2D([], [], color=INK2, marker="o", ls="none", ms=8, mfc="white", mew=2, label="matched counterpart"),
               Line2D([], [], color=INK2, marker="o", ls="none", ms=8, mew=2, label="fused reader"),
               Line2D([], [], color=INK2, lw=2, alpha=0.55, label="bar margin (bar comparator to fused reader)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5, frameon=False, bbox_to_anchor=(0.62, 0.0))
    fig.suptitle("Where the comparators sit: CSD lifts B′ and the counterparts as much as the readers",
                 fontsize=12, x=0.02, ha="left", color=INK)
    fig.savefig(OUT / "comparator_levels.png", dpi=DPI)
    plt.close(fig)


def fig_diagnostics():
    ra = load(RES / "ra_summary.json")["ar_check"]
    rb_rand = load(RES / "cand_Rb_argmax_AR.json")["pick_share"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.27, wspace=0.25)

    # (a) AR check: share of picks that go to rand
    ax = axes[0]
    readers = [("arg-max (step 1)", ra["reader_step1_argmax"], C_REF), ("R-a", ra["reader_Ra"], C_RA),
               ("R-b", rb_rand, C_RBA)]
    cats = [("condition a", lambda s: s["per_condition"]["a"]["rand"]),
            ("condition b", lambda s: s["per_condition"]["b"]["rand"]),
            ("overall", lambda s: s["overall"]["rand"])]
    w = 0.26
    for j, (lab, s, c) in enumerate(readers):
        vals = [f(s) for _, f in cats]
        x = np.arange(len(cats)) + (j - 1) * (w + 0.02)
        ax.bar(x, vals, width=w, color=c, label=lab, zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.axhline(25, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.text(1.0, 26.2, "chance 25%", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels([c for c, _ in cats])
    ax.set_ylabel("Share of (episode, condition) picks going to rand (%)")
    ax.set_ylim(0, 48)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("(a) AR check: how often each reader picks the empty grouping", fontsize=10.5, loc="left",
                 color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    # (b) R-b bank accuracy vs seed-42 pick accuracy
    ax = axes[1]
    cfgs = ["A1", "A0", "AR"]
    chance = {"A1": 25.0, "A0": 100 / 3, "AR": 25.0}
    bank0, bank1, pick = [], [], []
    for c in cfgs:
        halves = load(RES / f"rb_reader_{c}.json")["halves"]
        bank0.append(halves["0"]["oof_accuracy_at_chosen_C"])
        bank1.append(halves["1"]["oof_accuracy_at_chosen_C"])
        pick.append(load(RES / f"cand_Rb_argmax_{c}.json")["pick_accuracy"]["correct_share"]["point"])
    w = 0.26
    x = np.arange(len(cfgs))
    series = [(bank0, "bank accuracy, half-reader 0 (out-of-fold)", C_RA, -1),
              (bank1, "bank accuracy, half-reader 1 (out-of-fold)", C_RA, 0),
              (pick, "seed-42 pick accuracy (told mapping, diagnostic)", C_RBA, 1)]
    for vals, lab, c, off in series:
        xx = x + off * (w + 0.02)
        ax.bar(xx, vals, width=w, color=c, label=lab if off != 0 else None, zorder=3,
               hatch="//" if off == 0 else None, edgecolor="white", lw=0)
        for xi, v in zip(xx, vals):
            ax.text(xi, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    for xi, c in zip(x, cfgs):
        ax.plot([xi - 0.45, xi + 0.45], [chance[c]] * 2, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=4)
    ax.text(x[0] - 0.45, chance["A1"] + 0.8, "chance", ha="left", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c} ({ {'A1': 4, 'A0': 3, 'AR': 4}[c]} classes)" for c in cfgs])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("(b) R-b: accuracy on its pseudo-aspect bank and on seed 42", fontsize=10.5, loc="left", color=INK)
    h, l = ax.get_legend_handles_labels()
    h.insert(1, plt.Rectangle((0, 0), 1, 1, facecolor=C_RA, hatch="//", edgecolor="white", lw=0))
    l.insert(1, "bank accuracy, half-reader 1 (out-of-fold)")
    ax.legend(h, l, frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
    fig.suptitle("Label-free and told-mapping diagnostics (enter no rule)", fontsize=12, x=0.02, ha="left", color=INK)
    fig.savefig(OUT / "diagnostics.png", dpi=DPI)
    plt.close(fig)


def main():
    rs = rows()
    fig_bar_margins(rs)
    fig_gain_either(rs)
    fig_per_pair(rs)
    fig_levels(rs)
    fig_diagnostics()
    for r in rs:
        print(f"{r['key']:20s} {r['config']}  bar {r['bar']['point']:+.4f} vs {r['comp']:11s} "
              f"gain {r['gain']['point']:+.4f} either_bar {r['either_bar']:+.4f} either_margin {r['either_margin']:+.4f}")


if __name__ == "__main__":
    main()
