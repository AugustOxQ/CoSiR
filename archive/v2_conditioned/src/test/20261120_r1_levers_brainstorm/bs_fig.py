"""Figures for docs/reports/auto/v2/2026-11-20_r1_levers_brainstorm.md (exploratory, seed 42, decides nothing).
Reads results/bs_*.json of this folder; writes PNGs to docs/reports/assets/2026-11-20_r1_levers_brainstorm/."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE.parents[2] / "docs/reports/assets/2026-11-20_r1_levers_brainstorm"
OUT.mkdir(parents=True, exist_ok=True)
GRAY, BLUE, ORANGE, AQUA, INK, MUTED = "#8a8985", "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def load(name):
    return json.loads((RES / f"{name}.json").read_text())


# ---- Figure 1: where R1's margin comes from, and what one-sided steering changes (per pair x condition)
b5 = load("bs_05_aff")["A0"]["per_pair_condition"]
pairs = ["emotion__style", "emotion__genre", "style__genre"]
target = {("emotion__style", "a"): "emotion", ("emotion__style", "b"): "style", ("emotion__genre", "a"): "emotion",
          ("emotion__genre", "b"): "genre", ("style__genre", "a"): "style", ("style__genre", "b"): "genre"}
labels, r1d, affd, cfr = [], [], [], []
for p in pairs:
    for c in ("a", "b"):
        labels.append(f"{p.replace('__', ' × ').replace('emotion', 'e').replace('style', 's').replace('genre', 'g')}\n"
                      f"{c}: {target[(p, c)]}")
        r1d.append(b5["R1"][p][c]["fused"]["r1"] - b5["R1"][p][c]["cf"]["r1"])
        affd.append(b5["AFF"][p][c]["fused"]["r1"] - b5["AFF"][p][c]["cf"]["r1"])
        cfr.append(b5["R1"][p][c]["cf"]["r1"])
x = np.arange(len(labels))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
w = 0.38
ax1.bar(x - w / 2, r1d, w, color=GRAY, label="R1 (gate on any pick)", edgecolor="white", linewidth=1)
ax1.bar(x + w / 2, affd, w, color=BLUE, label="one-sided: gate only on affect picks", edgecolor="white", linewidth=1)
ax1.axhline(0, color=MUTED, lw=0.8)
ax1.set_xticks(x, labels, fontsize=7.5)
ax1.set_ylabel("fused minus counterpart, R@1 (pp)")
ax1.set_title("(a) Gain per pair and condition (target aspect named)", fontsize=9, color=INK, loc="left")
ax1.legend(frameon=False, fontsize=8, loc="upper right")
ax2.bar(x, cfr, 0.6, color=GRAY, edgecolor="white", linewidth=1)
for i, v in enumerate(cfr):
    ax2.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=7.5, color=INK)
ax2.set_xticks(x, labels, fontsize=7)
ax2.set_ylabel("R1's counterpart R@1 (pp)")
ax2.set_title("(b) The condition-free score already ranks genre first", fontsize=9, color=INK, loc="left")
fig.tight_layout()
fig.savefig(OUT / "per_pair_condition.png", dpi=170)
plt.close(fig)

# ---- Figure 2: bar margins of the exploratory variants, by family
b2, b3, b4 = load("bs_02_gates"), load("bs_03_sxg"), load("bs_04_readers")["results"]
b5a, b6, b7 = load("bs_05_aff")["A0"], load("bs_06_aff_ceiling")["results"], load("bs_07_detector")["results"]
b8, b10, b1 = load("bs_08_csd_evidence")["results"], load("bs_10_subsets"), load("bs_01_decouple")
rows = [
    ("R1 (reference)", b2["R1"], GRAY, False),
    ("two-sided: gates on min margin", b2["variants"]["sym_min: both gates on min margin"], ORANGE, False),
    ("two-sided: gates on TV(P^a, P^b)", b2["variants"]["tv: both gates on TV(P^a,P^b)"], ORANGE, False),
    ("two-sided: soft gate (weight m^c)", b2["variants"]["soft: weight = own margin m^c"], ORANGE, False),
    ("decoupled weights, anchored", b1["anchored"], ORANGE, False),
    ("joint exclusive decoding (eps 0)", b4["J_eps0.0"], ORANGE, False),
    ("R1 + R3 geometric ensemble", b4["ENS_geo"], ORANGE, False),
    ("H = image picks only", b10["H=image"], ORANGE, False),
    ("one-sided: affect picks (AFF)", b5a["AFF"], BLUE, False),
    ("one-sided: H chosen by the cross-fit", b5a["family_H_crossfit"], BLUE, False),
    ("one-sided: AFF on R2", b6["AFF_R2"], BLUE, False),
    ("one-sided: P(affect) gate, z(s_aff)", b7["R1aff"], BLUE, False),
    ("one-sided: bank LR detector (18)", b7["BANK_LR18"], BLUE, False),
    ("one-sided: bank GB detector + csd (24)", b8["BANK_A1_GB24"], BLUE, False),
    ("s×g abstention: min(S_img, C_img) p75", b4["IMGABST_q75"], AQUA, False),
    ("s×g abstention: min(S_csd, C_csd) p75", b4["CSDABST_q75"], AQUA, False),
    ("control: random gate, AFF's share per cond", b10["random_seed0"], GRAY, True),
    ("control: same, another seed", b10["random_seed1"], GRAY, True),
    ("oracle: gates shut on s×g", b3["oracles"]["O_sxg_closed"], INK, True),
    ("oracle: supervised 18-feature reader", b3["oracles"]["O_sup18"], INK, True),
    ("oracle: perfect emotion gate, z(s_aff)", b6["O_emo_gate_aff"], INK, True),
]
fig, ax = plt.subplots(figsize=(7.6, 6.4))
for i, (nm, r, col, hollow) in enumerate(rows):
    y = len(rows) - 1 - i
    p, lo, hi = r["bar"]["point"], r["bar"]["ci95"][0], r["bar"]["ci95"][1]
    ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
    ax.plot(p, y, "o", ms=7, mfc="white" if hollow else col, mec=col, mew=2)
    ax.text(2.2, y, f"{p:+.2f}", va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=8)
ax.axvline(0.5, color=MUTED, ls="--", lw=1)
ax.axvline(0.444, color=GRAY, ls=":", lw=1)
ax.text(0.52, len(rows) - 0.4, "bar +0.5", fontsize=7.5, color=MUTED)
ax.set_xlim(-0.35, 2.45)
ax.set_xlabel("bar margin, R@1 (pp), with 95% interval\nhollow = control or declared oracle (not a method)")
ax.set_title("Exploratory variants on seed 42 (decide nothing)", fontsize=9, color=INK, loc="left")
from matplotlib.lines import Line2D  # noqa: E402
ax.legend(handles=[Line2D([], [], color=c, marker="o", lw=2, label=l) for c, l in
                   ((ORANGE, "two-sided or reader transforms"), (BLUE, "one-sided affect steering"),
                    (AQUA, "s×g abstention"), (GRAY, "R1 and controls"), (INK, "declared oracles"))],
          frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.45, 0.99))
fig.tight_layout()
fig.savefig(OUT / "variants.png", dpi=170)
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
