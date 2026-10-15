"""Figure 4 of docs/reports/auto/v2/2026-11-17_ars_reader_fix_plan_review.md: the reader-fix decision rule after the
ARS review's fixes, coloured by what changed against the draft rule of the 2026-10-06 handoff (§5.4).
Grey = same as the draft, orange = replaced, teal = new.

    /root/miniconda3/envs/CoSiR/bin/python docs/reports/assets/2026-11-17_ars_reader_fix_plan_review/build_rule_flow.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

OUT = Path(__file__).with_name("rule_flow.png")
COL = {"same": ("#eceff1", "#607d8b"), "replaced": ("#ffe0b2", "#e65100"), "new": ("#b2dfdb", "#00695c")}


def box(ax, x, y, w, h, text, kind, size=9.5, bold_first=True):
    face, edge = COL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=face, edgecolor=edge, linewidth=1.6))
    lines = text.split("\n")
    if bold_first:
        ax.text(x + 0.18, y + h - 0.22, lines[0], ha="left", va="top", fontsize=size + 0.5, fontweight="bold")
        ax.text(x + 0.18, y + h - 0.55, "\n".join(lines[1:]), ha="left", va="top", fontsize=size, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, label=None, lx=0.0, ly=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, linewidth=1.4,
                                 color="#37474f"))
    if label:
        ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label, fontsize=9, ha="center", va="center",
                color="#37474f", style="italic")


def main():
    fig, ax = plt.subplots(figsize=(13, 9.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9.6)
    ax.axis("off")

    ax.text(0.1, 9.55, "Seed 42 (development)", fontsize=11, fontweight="bold", va="top", color="#37474f")
    top, h1 = 7.45, 1.75
    box(ax, 0.1, top, 3.05, h1, "R-a  scaled Δ\nΔ_h divided by its noise-only\nspread (pooled within-episode\n"
        "standard error); was: RMS of Δ_h", "replaced", size=9)
    box(ax, 3.35, top, 3.05, h1, "R-b  learned reader\nfrozen before seed-42 numbers:\n60,000-row cross-fitted heads per\n"
        "half, fixed bank, C by CV on bank", "replaced", size=9)
    box(ax, 6.6, top, 3.05, h1, "R-c  confidence gate\nhard gate, τ on a percentile grid;\ncounterpart = mean over a and b\n"
        "of g·z(T); was: gate on T_cf", "replaced", size=9)
    box(ax, 9.85, top, 3.05, h1, "Diagnostics only\npick accuracy, bank accuracy,\nlabel-free shift report, A1 − A0,\n"
        "random-slot (AR) check", "new", size=9)
    arrow(ax, 1.6, top, 1.6, 7.0)
    arrow(ax, 4.85, top, 4.85, 7.0)
    arrow(ax, 8.1, top, 5.9, 7.0)

    h = 1.45
    box(ax, 0.1, 5.55, 6.3, h, "Development bar, per candidate, on A1 and A0\nbar margin ≥ +0.5 with 95% lower bound > 0, against\n"
        "the largest of B, B′ and the matched counterpart\n(B added; B′ = B rebuilt; reader fused on B)", "replaced", size=9)
    box(ax, 6.6, 5.55, 6.3, h, "Kill rule\nno candidate clears the bar: no test, the user decides\n"
        "(was: plus a 10-point pick-accuracy clause and an R-b\nkill on 'high' and 'does not move')", "replaced", size=9)
    arrow(ax, 6.4, 6.27, 6.6, 6.27)

    box(ax, 0.1, 3.55, 6.3, h, "Carry one configuration\nbest A1 candidate that clears the bar; A0 only if no\n"
        "A1 candidate clears; ties within 0.05 go to R-a,\nthen R-b arg-max, R-b expected, R-c", "replaced", size=9)
    box(ax, 6.6, 3.55, 6.3, h, "Cutoff and timeline steps\ncandidates without numbers by the cutoff are dropped;\n"
        "the user approves the rule commit; projected test\nsensitivity written down before the seeds are built", "new", size=9)
    arrow(ax, 3.25, 5.55, 3.25, 5.0, "one or more clear", lx=1.05)

    box(ax, 0.1, 1.55, 6.3, h, "Test on seeds 49, 50, 51, pooled\nonly GO quantities computed before the verdict; GO =\n"
        "R@1 and gain lower bounds > 0 against cosine, RCA,\nB, B′ and the matched counterpart", "replaced", size=9)
    box(ax, 6.6, 1.55, 6.3, h, "Before Friday's decision\ncontroller re-derives the test numbers, then the whole-\n"
        "branch final review; a NO-GO with a positive pooled\npoint is reported as inconclusive", "new", size=9)
    arrow(ax, 3.25, 3.55, 3.25, 3.0)
    arrow(ax, 6.4, 2.27, 6.6, 2.27)

    box(ax, 0.1, 0.05, 12.8, 1.15, "Kept from the draft rule\ndevelopment on seed 42; seeds 49 to 51 built once with an SHA-256 check; "
        "one cluster per painting across seeds;\nevery seed reported alone; counterparts of R-a and R-b = the two-condition "
        "mean; bar +0.5 with lower bound > 0; gain counted once", "same", size=9)

    ax.legend(handles=[Patch(facecolor=COL[k][0], edgecolor=COL[k][1], label=lab) for k, lab in
                       (("same", "same as the draft rule"), ("replaced", "replaced by a review fix"),
                        ("new", "new (review fix or controller addition)"))],
              loc="upper right", bbox_to_anchor=(1.0, 1.005), fontsize=9.5, frameon=False, ncol=3)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
