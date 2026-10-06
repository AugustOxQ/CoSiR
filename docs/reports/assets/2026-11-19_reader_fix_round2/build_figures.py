"""Figures for docs/reports/auto/v2/2026-11-19_reader_fix_round2.md (CoSiR v2 reader fix, round 2).

Reads stored outputs only and asserts every plotted number against its source:
  src/test/20261118_reader_fix_round2/results/cand_R{1,2,3}_A0.json, cand_R1_A1.json   candidates and the ablation
  src/test/20261118_reader_fix_round2/results/rule_application.json                    the verdict (cross-checked)
  src/test/20261118_reader_fix_round2/results/probs_R2_A0.json, probs_R3_A0.json       R2 shift report, R3 probabilities
  src/test/20261118_reader_fix_round2/results/r3_k_A0.json                             D(k) table, mean |Delta|
  src/test/20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.json, cand_Rb_expected_A0.json   round 1
  src/test/20261116_grouping_step1_style/results/step1_eval_style.json                 step-1 arg-max reader
  docs/reports/assets/2026-11-19_reader_fix_round2/diagnostics.json                    report diagnostics (diagnostics.py)

Run from the repo root (after diagnostics.py):
  OMP_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python docs/reports/assets/2026-11-19_reader_fix_round2/build_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
RES = ROOT / "src/test/20261118_reader_fix_round2/results"
R1RES = ROOT / "src/test/20261117_reader_fix_csd/results"
STEP1 = ROOT / "src/test/20261116_grouping_step1_style/results/step1_eval_style.json"
OUT = Path(__file__).resolve().parent
DPI = 150

# dataviz reference palette (validated: all checks pass in light mode; aqua below 3:1 contrast, so every mark is
# direct-labelled or listed in a table), assigned by reader in fixed order; grey for references.
C_REF = "#8a8984"
C_RC = "#4a3aa7"   # round-1 R-c
C_R1 = "#2a78d6"
C_R2 = "#eb6834"
C_R3 = "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"R1": C_R1, "R2": C_R2, "R3": C_R3}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.facecolor": "white", "savefig.facecolor": "white",
})


def load(p):
    return json.loads(Path(p).read_text())


def close(a, b, tol=1e-12):
    assert abs(a - b) <= tol, (a, b)


# ---------------------------------------------------------------- data

def data():
    cand = {n: load(RES / f"cand_{n}_A0.json") for n in ("R1", "R2", "R3")}
    a1 = load(RES / "cand_R1_A1.json")
    app = load(RES / "rule_application.json")
    diag = load(OUT / "diagnostics.json")
    rc = load(R1RES / "cand_Rc_Rb_expected_A0.json")
    rbe = load(R1RES / "cand_Rb_expected_A0.json")
    step1 = load(STEP1)["arms"]["A0"]
    # the verdict file and the candidate files agree; the diagnostics reproduce the stored candidates
    for n, c in cand.items():
        a = app["candidates"][n]
        close(a["bar_margin"], c["bar"]["r1"]["point"])
        assert a["bar_ci95"] == c["bar"]["r1"]["ci95"] and a["comparator"] == c["bar"]["comparator"]
        close(a["gain_statistic"], c["gain_statistic"]["point"])
        assert a["clears"] is False
        d = diag["readers"][n]["own_896"]
        close(d["bar_margin"]["point"], c["bar"]["r1"]["point"])
        assert d["bar_margin"]["ci95"] == c["bar"]["r1"]["ci95"]
        close(diag["readers"][n]["reader"]["pick_accuracy"], c["pick_accuracy"]["correct_share"]["point"], 1e-9)
    # R1 restricted to k_top = 13 is round-1 R-c (regression check, re-derived by diagnostics.py)
    k13 = diag["readers"]["R1"]["k13_only"]
    close(k13["bar_margin"]["point"], rc["bar"]["r1"]["point"])
    assert k13["bar_margin"]["ci95"] == rc["bar"]["r1"]["ci95"]
    close(rc["bar"]["r1"]["point"], 0.4435221354166667)
    return cand, a1, app, diag, rc, rbe, step1


# ---------------------------------------------------------------- figure 1: what changed

def box(ax, x, y, w, h, text, kind, fs=8.6):
    fill = {"same": ("#ecebe8", "#8a8984"), "unchanged": ("#e3e0f4", C_RC), "replaced": ("#fde3d7", C_R2),
            "new": ("#d3f1e6", "#12876a")}[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", fc=fill[0],
                                ec=fill[1], lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=1.1, color=INK2))


def fig_what_changed():
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    h = 0.15
    # row 1: inputs and reader
    y1 = 0.70
    box(ax, 0.01, y1, 0.15, h, "Episode on seed 42\nquery, 4 support pairs,\n4 contrast pairs,\n13 candidates", "same")
    box(ax, 0.20, y1, 0.17, h, "A0 groupings and heads\naffect, image, caption;\ngrouping scores s_h", "same")
    box(ax, 0.41, y1, 0.15, h, "18 reader features\nper grouping: S, C, Δ,\ntwo spreads, match share", "same")
    box(ax, 0.60, y1 + 0.14, 0.38, 0.07, "R1: round 1's two half-readers, frozen", "unchanged")
    box(ax, 0.60, y1 + 0.02, 0.38, 0.085, "R2: same half-readers, inputs re-standardised on seed 42,\nEM class-prior correction (π̂)",
        "replaced")
    box(ax, 0.60, y1 - 0.10, 0.38, 0.085, "R3: half-readers retrained on impure banks\n(k of 4 pairs keep the group; k chosen without labels)",
        "replaced")
    ax.text(0.79, y1 + 0.245, "reader probabilities P^c(h): the only part that differs between candidates",
            ha="center", fontsize=9, color=INK)
    for x0, x1 in ((0.16, 0.20), (0.37, 0.41), (0.56, 0.60)):
        arrow(ax, x0, y1 + h / 2, x1, y1 + h / 2)
    # row 2: fusion
    y2 = 0.30
    box(ax, 0.01, y2, 0.15, h, "weighted term\nT^c = Σ_h P^c(h)·s_h,\nz-scored per row", "unchanged")
    box(ax, 0.20, y2, 0.15, h, "confidence gate\ng^c = 1[m^c ≥ τ],\nτ = 0/25/50/75th pct\nof own margins", "unchanged")
    box(ax, 0.39, y2, 0.17, h, "fused score\n(1+λ_u)·z(B) + λ_a·g·z(T),\n56 weight cells", "unchanged")
    box(ax, 0.60, y2, 0.16, h, "top-k restriction\nfirst place only from\nB's k_top best\n(k_top 13, 5, 3, 2)", "new")
    box(ax, 0.80, y2, 0.18, h, "cross-fit on parity halves\n896 cells (was 224),\ninteger criteria\n(min-margin rule)", "replaced")
    for x0, x1 in ((0.16, 0.20), (0.35, 0.39), (0.56, 0.60), (0.76, 0.80)):
        arrow(ax, x0, y2 + h / 2, x1, y2 + h / 2)
    arrow(ax, 0.60, y1 - 0.06, 0.085, y2 + h + 0.005)
    # row 3: comparators
    y3 = 0.05
    box(ax, 0.20, y3, 0.35, 0.13, "matched counterpart (round-1 R-c's definition):\nG_cf = (g^a z(T^a) + g^b z(T^b)) / 2; extended in round 2\n"
        "to 896 cells, shared top-k sets, max-R@1 cross-fit", "unchanged")
    box(ax, 0.59, y3, 0.39, 0.13, "comparators B and B′(A0); bar comparator = the largest of\nB, B′, counterpart; bar +0.5, both lower bounds above 0",
        "same")
    handles = [Patch(fc="#ecebe8", ec="#8a8984", label="grey: same as round 1 (data, heads, comparators, bar)"),
               Patch(fc="#e3e0f4", ec=C_RC, label="purple: unchanged from round-1 R-c"),
               Patch(fc="#fde3d7", ec=C_R2, label="orange: replaced in round 2"),
               Patch(fc="#d3f1e6", ec="#12876a", label="teal: new in round 2")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("What round 2 changed relative to round-1 R-c (the confidence-gated learned reader on A0)",
                 fontsize=12, x=0.02, ha="left", color=INK)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.07)
    fig.savefig(OUT / "what_changed.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2: bar margins

def fig_bar_margins(cand, a1, rc, step1):
    rows = [
        ("step-1 arg-max reader (round-1 reference)", step1["bar"]["r1"], "B′", C_REF, False),
        ("round-1 R-c (224 cells)", rc["bar"]["r1"], "counterpart", C_RC, False),
        ("R1: round-1 reader, 896 cells", cand["R1"]["bar"]["r1"], "counterpart", C_R1, False),
        ("R2: adapted reader", cand["R2"]["bar"]["r1"], "B′", C_R2, False),
        ("R3: impure-bank reader", cand["R3"]["bar"]["r1"], "B′", C_R3, False),
        ("R1 on A1 (ablation, descriptive)", a1["bar"]["r1"], "counterpart", C_R1, True),
    ]
    assert cand["R1"]["bar"]["comparator"] == "counterpart" and cand["R2"]["bar"]["comparator"] == "B_prime"
    assert cand["R3"]["bar"]["comparator"] == "B_prime" and a1["bar"]["comparator"] == "counterpart"
    assert step1["bar"]["comparator"] == "B_prime" and rc["bar"]["comparator"] == "counterpart"
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    fig.subplots_adjust(left=0.30, right=0.72, top=0.85, bottom=0.16)
    ys = -np.arange(len(rows), dtype=float)
    ys[-1] -= 0.4
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    ax.axvline(0.5, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for y, (lab, b, comp, c, hollow) in zip(ys, rows):
        lo, hi = b["ci95"]
        ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round", zorder=3)
        ax.plot(b["point"], y, marker="o", ms=8, color=c, mfc="white" if hollow else c, mew=2, zorder=4)
        ax.text(1.02, y, f"{b['point']:+.3f} [{lo:+.3f}, {hi:+.3f}]  vs {comp}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=9, color=INK)
    ax.text(1.02, 0.75, "bar margin [95% interval], comparator", transform=ax.get_yaxis_transform(), va="center",
            ha="left", fontsize=9, color=INK2)
    ax.text(0.5, 0.75, "development bar +0.5", ha="center", va="center", fontsize=9, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(ys.min() - 0.7, 1.1)
    ax.set_xlim(-0.3, 0.8)
    ax.set_xlabel("Bar margin: fused reader minus bar comparator, R@1 (percentage points)")
    fig.suptitle("Bar margins on seed 42 (development): no round-2 candidate reaches +0.5", fontsize=12, x=0.02,
                 ha="left", color=INK)
    fig.text(0.02, 0.03, "Grey and purple: round 1, for reference. Hollow: the A1 ablation (with the CSD style grouping), "
             "never a candidate.", fontsize=8.5, color=INK2)
    fig.savefig(OUT / "bar_margins.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figure 3: pick accuracy against bar margin

def fig_pick_vs_bar(cand, rc, rbe, step1):
    pts = [
        ("step-1 arg-max", step1["eval"]["pick"]["correct_share"]["point"], step1["bar"]["r1"], C_REF, "s"),
        ("round-1 learned reader,\nweighted term (no gate)", rbe["pick_accuracy"]["correct_share"]["point"], rbe["bar"]["r1"], C_REF, "D"),
        ("round-1 R-c", rc["pick_accuracy"]["correct_share"]["point"], rc["bar"]["r1"], C_RC, "o"),
        ("R1", cand["R1"]["pick_accuracy"]["correct_share"]["point"], cand["R1"]["bar"]["r1"], C_R1, "o"),
        ("R2", cand["R2"]["pick_accuracy"]["correct_share"]["point"], cand["R2"]["bar"]["r1"], C_R2, "o"),
        ("R3", cand["R3"]["pick_accuracy"]["correct_share"]["point"], cand["R3"]["bar"]["r1"], C_R3, "o"),
    ]
    close(pts[0][1], 54.703776041666664, 1e-9)
    close(pts[3][1], 51.261393229166664, 1e-9)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.86, bottom=0.12)
    ax.axhline(0.5, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.text(45.6, 0.515, "development bar +0.5", fontsize=9, color=INK, va="bottom")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    offs = {"step-1 arg-max": (8, 6), "round-1 learned reader,\nweighted term (no gate)": (14, -36),
            "round-1 R-c": (-130, -4), "R1": (10, 2), "R2": (10, -4), "R3": (10, -4)}
    jitter = {"round-1 R-c": -0.12, "R1": 0.12, "round-1 learned reader,\nweighted term (no gate)": 0.0}
    for lab, x, b, c, m in pts:
        xx = x + jitter.get(lab, 0.0)
        ax.plot([xx, xx], b["ci95"], color=c, lw=2, solid_capstyle="round", zorder=3)
        ax.plot(xx, b["point"], marker=m, ms=8, color=c, mew=2, ls="none", zorder=4)
        ax.annotate(f"{lab} ({b['point']:+.3f})", (xx, b["point"]), xytext=offs[lab], textcoords="offset points",
                    fontsize=8.8, color=INK)
    ax.axvline(100 / 3, color=INK2, lw=0.8, ls=":")
    ax.set_xlim(45, 58)
    ax.set_ylim(-0.2, 0.8)
    ax.set_xlabel("Pick accuracy under the told mapping (%; diagnostic, chance 33.3%)")
    ax.set_ylabel("Bar margin, R@1 (pp), with 95% interval")
    fig.suptitle("Higher pick accuracy did not buy a higher bar margin (A0, seed 42)", fontsize=12, x=0.02, ha="left",
                 color=INK)
    fig.text(0.02, 0.905, "R3 picks the told grouping most often yet has the lowest bar margin. R1, round-1 R-c and the round-1 "
             "learned reader share one set of picks.", fontsize=8.5, color=INK2)
    fig.savefig(OUT / "pick_vs_bar.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figure 4: reader confidence and where the picks go

def fig_reader_confidence(cand, diag):
    p2 = load(RES / "probs_R2_A0.json")["shift_report"]["iii_top_probability"]
    p3 = load(RES / "probs_R3_A0.json")["top_probability"]
    close(p2["R1_seed42"]["both_conditions"]["mean"], diag["readers"]["R1"]["reader"]["top_probability_mean"], 1e-12)
    close(p2["R2_adapted_seed42"]["both_conditions"]["mean"], diag["readers"]["R2"]["reader"]["top_probability_mean"], 1e-12)
    close(p3["seed42"]["both_conditions"]["mean"], diag["readers"]["R3"]["reader"]["top_probability_mean"], 1e-12)
    dec = [str(k) for k in range(10, 100, 10)]
    lines = [("R1 on seed 42", p2["R1_seed42"]["both_conditions"], C_R1, "-", "o"),
             ("R2 on seed 42", p2["R2_adapted_seed42"]["both_conditions"], C_R2, "-", "o"),
             ("R3 on seed 42", p3["seed42"]["both_conditions"], C_R3, "-", "o"),
             ("round-1 bank, out of fold (R1, R2's half-readers)", p2["round1_bank_oof_pooled_halves"], C_R1, "--", None),
             ("R3's impure bank, out of fold", p3["bank_oof_at_chosen_C_pooled_halves"], C_R3, "--", None)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.27, wspace=0.22)
    ax = axes[0]
    for lab, s, c, ls, m in lines:
        ax.plot([int(k) for k in dec], [s["deciles"][k] for k in dec], color=c, ls=ls, lw=2, marker=m, ms=5,
                label=f"{lab}: mean {s['mean']:.3f}")
    ax.set_xticks(range(10, 100, 10))
    ax.set_xlabel("Decile of the (episode, condition) values")
    ax.set_ylabel("Top probability max_h P(h)")
    ax.set_ylim(0.38, 1.0)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("(a) Reader confidence: deciles of the top probability", fontsize=10.5, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8.2, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)
    ax = axes[1]
    cats = [(c, h) for c in ("a", "b") for h in ("affect", "image", "caption")]
    w = 0.26
    x = np.arange(len(cats))
    for j, n in enumerate(("R1", "R2", "R3")):
        ps = cand[n]["pick_share"]["per_condition"]
        vals = [ps[c][h] for c, h in cats]
        xx = x + (j - 1) * (w + 0.02)
        ax.bar(xx, vals, width=w, color=COL[n], label=n, zorder=3)
        for xi, v in zip(xx, vals):
            ax.text(xi, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=7.8, color=INK)
    close(cand["R2"]["pick_share"]["overall"]["caption"], 33.94368489583333, 1e-9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"condition {c}\n{h}" for c, h in cats], fontsize=8.8)
    ax.axvline(2.5, color=INK2, lw=0.8)
    ax.set_ylabel("Share of picks (%)")
    ax.set_ylim(0, 92)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.8, loc="upper right")
    ax.set_title("(b) Where the picks go (caption is never the told grouping on A0)", fontsize=10.5, loc="left",
                 color=INK)
    fig.suptitle("R2 became more confident and moved picks to caption; R3 is matched to its bank but flatter",
                 fontsize=12, x=0.02, ha="left", color=INK)
    fig.savefig(OUT / "reader_confidence.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figure 5: R3's choice of purity

def fig_purity():
    rk = load(RES / "r3_k_A0.json")
    p3 = load(RES / "probs_R3_A0.json")
    assert rk["k_star"] == p3["k_star"] == 2 and rk["D"] == p3["D"]
    ks = ["1", "2", "3", "4"]
    D = [rk["D"][k] for k in ks]
    assert int(np.argmin(D)) + 1 == 2
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.15, wspace=0.25)
    ax = axes[0]
    cols = [C_R3 if k == "2" else "#b9b8b3" for k in ks]
    ax.bar([1, 2, 3, 4], D, width=0.55, color=cols, zorder=3)
    for i, v in enumerate(D):
        ax.text(i + 1, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["k = 1", "k = 2 (chosen)", "k = 3", "k = 4\n(round 1's bank)"])
    ax.set_ylabel("D(k): mean |SMD| over the 18 reader features")
    ax.set_ylim(0, 0.32)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("(a) Label-free match of bank to seed-42 features", fontsize=10.5, loc="left", color=INK)
    ax = axes[1]
    mad = rk["mean_abs_delta"]
    gcol = {"affect": C_RC, "image": C_R1, "caption": C_R2}
    for g in ("image", "caption", "affect"):
        ax.plot([1, 2, 3, 4], [mad["bank"][k][g] for k in ks], color=gcol[g], lw=2, marker="o", ms=6,
                label=f"{g}: bank of purity k")
        ax.axhline(mad["seed42"][g], color=gcol[g], lw=1.2, ls=(0, (4, 3)))
        ax.text(4.12, mad["seed42"][g] + 0.0012, f"seed 42 ({g})", va="bottom", fontsize=8.3, color=INK)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.8, 4.9)
    ax.set_xlabel("Purity k (support and contrast pairs that keep their shared group, of 4)")
    ax.set_ylabel("Mean |Δ_h| (support minus contrast agreement)")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("(b) Diagnostic: signal strength |Δ| by purity (enters no rule)", fontsize=10.5, loc="left", color=INK)
    fig.suptitle("R3's purity: D(k) chose k = 2, and the separate |Δ| diagnostic points the same way for image",
                 fontsize=12, x=0.02, ha="left", color=INK)
    fig.text(0.02, 0.9, "D(k) matches feature levels; the three Δ features have SMD 0 for every k by construction. "
             "Dashed lines in (b): the seed-42 value of each grouping.", fontsize=8.5, color=INK2)
    fig.savefig(OUT / "r3_purity.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figure 6: the readers at the same fusion settings

def fig_term_weight(diag):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.81, bottom=0.2, wspace=0.22)
    la = [c["lambda_a"] for c in diag["readers"]["R1"]["curve_tau2_lu0_k13"]]
    xs = np.arange(len(la))
    B = diag["readers"]["R1"]["curve_tau2_lu0_k13"][0]["r1"]
    close(B, 18.341064453125, 1e-9)
    for n in ("R1", "R2", "R3"):
        cv = diag["readers"][n]["curve_tau2_lu0_k13"]
        assert [c["lambda_a"] for c in cv] == la
        axes[0].plot(xs, [c["r1"] for c in cv], color=COL[n], lw=2, marker="o", ms=6, label=f"{n}, fused reader")
        axes[0].plot(xs, [c["cf_r1"] for c in cv], color=COL[n], lw=1.2, ls=(0, (3, 2)), marker="o", ms=4,
                     mfc="white", label=f"{n}, matched counterpart")
        axes[1].plot(xs, [c["gain"] for c in cv], color=COL[n], lw=2, marker="o", ms=6, label=n)
        axes[0].annotate(n, (xs[-1], cv[-1]["r1"]), xytext=(6, 0), textcoords="offset points", fontsize=9,
                         color=INK, va="center")
        axes[1].annotate(n, (xs[-1], cv[-1]["gain"]), xytext=(6, 0), textcoords="offset points", fontsize=9,
                         color=INK, va="center")
    axes[0].axhline(B, color=INK2, lw=0.8)
    axes[0].text(0.1, B - 0.06, "B (λ_a = 0)", fontsize=8.5, color=INK2, va="top")
    for ax, yl, t in ((axes[0], "R@1 on all 12,288 episodes (pp)", "(a) R@1 of the fused reader and its counterpart"),
                      (axes[1], "Condition gain (pp)", "(b) Condition gain of the fused reader")):
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{v:g}" for v in la])
        ax.set_xlabel("Term weight λ_a (λ_u = 0, gate at τ_2, no top-k restriction)")
        ax.set_ylabel(yl)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(t, fontsize=10.5, loc="left", color=INK)
        ax.set_xlim(-0.3, len(la) - 0.4)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8.3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("At every positive term weight R1 stays above R2 and R3: the readers, not the cross-fit, set the order",
                 fontsize=12, x=0.02, ha="left", color=INK)
    fig.text(0.02, 0.885, "Diagnostic, in sample (all episodes, no cross-fit; optimistic). Cells (k_top 13, τ_2, λ_u 0, "
             "λ_a) of each reader's own family, from diagnostics.py.", fontsize=8.5, color=INK2)
    fig.savefig(OUT / "term_weight.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figure 7: what the top-k restriction flips

def fig_topk(diag):
    ra = diag["readers"]["R1"]["ranks"]
    assert ra["n_rankings"] == 49152
    ks = [5, 3, 2]
    gain = [ra[f"restrict_top{k}"]["miss_to_hit"] for k in ks]
    loss = [ra[f"restrict_top{k}"]["hit_to_miss"] for k in ks]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.18)
    x = np.arange(len(ks))
    w = 0.34
    ax.bar(x - w / 2 - 0.01, gain, width=w, color=C_R1, label="miss → hit (first place came from outside B's top k; "
           "the target is the best inside it)", zorder=3)
    ax.bar(x + w / 2 + 0.01, loss, width=w, color=C_REF, label="hit → miss (the target itself sat outside B's top k)",
           zorder=3)
    for xi, g, l_, k in zip(x, gain, loss, ks):
        ax.text(xi - w / 2 - 0.01, g + 12, f"{g}", ha="center", fontsize=9, color=INK)
        ax.text(xi + w / 2 + 0.01, l_ + 12, f"{l_}", ha="center", fontsize=9, color=INK)
        net = ra[f"restrict_top{k}"]["net_rankings"]
        ax.text(xi, max(g, l_) + 75, f"net {net:+d} rankings\n({ra[f'restrict_top{k}']['net_r1_pp']:+.3f} pp R@1)",
                ha="center", fontsize=8.8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k_top = {k}\n{ra[f'first_outside_top{k}_share']:.1f}% of first places\noutside B's top {k}"
                        for k in ks], fontsize=9)
    ax.set_ylabel("Rankings of 49,152")
    ax.set_ylim(0, 1250)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.6, loc="upper left")
    fig.suptitle("Restricting R1's chosen cells to B's top k would lose more hits than it gains", fontsize=12,
                 x=0.02, ha="left", color=INK)
    fig.text(0.02, 0.86, "Diagnostic on R1's assembled fused scores (cells 116 and 119), all episodes; from "
             "diagnostics.py. Enters no rule.", fontsize=8.5, color=INK2)
    fig.savefig(OUT / "topk_flips.png", dpi=DPI)
    plt.close(fig)


def main():
    cand, a1, app, diag, rc, rbe, step1 = data()
    fig_what_changed()
    fig_bar_margins(cand, a1, rc, step1)
    fig_pick_vs_bar(cand, rc, rbe, step1)
    fig_reader_confidence(cand, diag)
    fig_purity()
    fig_term_weight(diag)
    fig_topk(diag)
    for n in ("R1", "R2", "R3"):
        c = cand[n]
        print(f"{n}: bar {c['bar']['r1']['point']:+.4f} vs {c['bar']['comparator']}, gain "
              f"{c['gain_statistic']['point']:+.4f}, pick {c['pick_accuracy']['correct_share']['point']:.2f}")
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
