"""Figure for the r21 user-read report: style AMI change from saved c*, with 95% intervals.

Reads scores/stage2_lam0.json (the stage 2 table with the lambda = 0 control) and draws one bar per arm.
Usage: python r21_userread_fig.py OUT.png
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCORES = "/data/SSD2/cosir_r21/20261111_dec_heads/scores/stage2_lam0.json"
ARMS = [
    ("dec-plain_l10_w5_S2", "DEC, plain"),
    ("dec-start_l10_w5_S2", "DEC, start pull"),
    ("dec-teacher_l10_w5_S2", "DEC, instructed-\nimage pull"),
    ("dec-plain_l0_w5_S2", "Control\n(DEC weight 0)"),
]


def main(out):
    d = json.load(open(SCORES))
    cells = d["aspects"]["style"]["arms"]
    labels, ch, lo, hi = [], [], [], []
    for arm, label in ARMS:
        c = cells[arm]["ami_class_count"]
        labels.append(label)
        ch.append(c["change"])
        lo.append(c["change"] - c["lo"])
        hi.append(c["hi"] - c["change"])
    cstar = cells[ARMS[0][0]]["ami_class_count"]["cstar"]
    # Palette (docs/reports/assets/palette.md): our network #1d6fd1; reference / neutral #8a8984.
    colors = ["#1d6fd1", "#1d6fd1", "#1d6fd1", "#8a8984"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    x = range(len(labels))
    ax.bar(x, ch, color=colors, width=0.6)
    ax.errorbar(x, ch, yerr=[lo, hi], fmt="none", ecolor="black", capsize=5, lw=1.2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Style AMI change from c*")
    ax.set_title(
        "The DEC forms raise style AMI; the control does not\n"
        f"(test, 5 seeds, 95% interval; c* style AMI {cstar:.3f})",
        fontsize=10,
    )
    for xi, v in zip(x, ch):
        ax.text(xi + 0.32, v, f"{v:+.4f}", va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    print(out, [round(v, 4) for v in ch])


if __name__ == "__main__":
    main(sys.argv[1])
