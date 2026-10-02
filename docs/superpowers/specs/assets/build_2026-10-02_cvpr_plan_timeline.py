"""Timeline figure for docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md.

Run from the repository root:
    python docs/superpowers/specs/assets/build_2026-10-02_cvpr_plan_timeline.py
Writes docs/superpowers/specs/assets/2026-10-02_cvpr_plan_timeline.png. The rows mirror the spec's §11 table;
edit both together.
"""

from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT = Path(__file__).with_name("2026-10-02_cvpr_plan_timeline.png")
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PHASES = {                                   # categorical slots 1 to 6, validated (light surface)
    "0 setup": "#2a78d6",
    "1 go/no-go": "#eb6834",
    "2 generalize": "#1baf7a",
    "3 final tests": "#eda100",
    "4 writing and review": "#e87ba4",
    "5 supplementary": "#008300",
}
D = lambda m, d: date(2026, m, d)  # noqa: E731
ROWS = [  # (label, start, end inclusive, phase)
    ("E0 setup, episode module, Qwen fidelity", D(10, 2), D(10, 4), "0 setup"),
    ("E1 tier-1 baselines (ArtELingo)", D(10, 3), D(10, 6), "1 go/no-go"),
    ("E2 pseudo-partitions (ArtELingo)", D(10, 4), D(10, 5), "1 go/no-go"),
    ("E3 method A, go/no-go runs", D(10, 5), D(10, 9), "1 go/no-go"),
    ("E4 feature extraction on DAS6", D(10, 5), D(10, 12), "1 go/no-go"),
    ("E5 replication seeds", D(10, 10), D(10, 11), "2 generalize"),
    ("E6 CUB (unseen species)", D(10, 10), D(10, 16), "2 generalize"),
    ("E7 GeneCIS focus attribute", D(10, 13), D(10, 20), "2 generalize"),
    ("E8 SemArt", D(10, 15), D(10, 22), "2 generalize"),
    ("E9 Qwen backbone runs", D(10, 12), D(10, 22), "2 generalize"),
    ("E10 tier-2 baselines (names)", D(10, 12), D(10, 20), "2 generalize"),
    ("E11 ablations (K6, K7)", D(10, 14), D(10, 23), "2 generalize"),
    ("E12 C3 modality analysis", D(10, 16), D(10, 30), "2 generalize"),
    ("E13 pre-registration, power", D(10, 21), D(10, 23), "3 final tests"),
    ("E14 final held reads", D(10, 24), D(11, 1), "3 final tests"),
    ("E15 writing", D(10, 26), D(11, 16), "4 writing and review"),
    ("E16 final review, fix wave", D(11, 11), D(11, 14), "4 writing and review"),
    ("E17 supplementary", D(11, 16), D(11, 23), "5 supplementary"),
]
MILESTONES = [  # (date, label, emphasis)
    (D(10, 9), "GO / NO-GO", True),
    (D(10, 16), "CUB check", False),
    (D(10, 23), "methods frozen", False),
    (D(11, 1), "experiments frozen", False),
    (D(11, 7), "abstract text fixed", False),
    (D(11, 10), "abstract registration", True),
    (D(11, 16), "paper deadline", True),
    (D(11, 23), "supplementary", True),
]
TODAY = D(10, 2)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK})
    fig, ax = plt.subplots(figsize=(13.5, 7.6), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    n = len(ROWS)
    for i, (label, start, end, phase) in enumerate(ROWS):
        y = n - 1 - i
        width = (end - start).days + 1
        ax.barh(y, width, left=mdates.date2num(start), height=0.62, color=PHASES[phase],
                edgecolor=SURFACE, linewidth=2, zorder=3)
        span = f"{start:%b %d}" if width == 1 else f"{start:%b %d} to {end:%b %d}"
        ax.text(mdates.date2num(end) + 1.25, y, span, va="center", ha="left", fontsize=8.2, color=INK_2, zorder=4,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in reversed(ROWS)])
    ax.tick_params(axis="y", length=0)
    top = n - 0.3
    for k, (when, label, strong) in enumerate(MILESTONES):
        x = mdates.date2num(when) + 1.0   # end of the milestone day, where bars ending that day stop
        ax.axvline(x, color=INK if strong else INK_2, linewidth=1.4 if strong else 1.0,
                   linestyle="-" if strong else (0, (3, 3)), zorder=2)
        ax.text(x, top + (1.55 if k % 2 else 0.55), f"{label}\n{when:%b %d}", ha="center", va="bottom",
                fontsize=8.4, color=INK if strong else INK_2, fontweight="bold" if strong else "normal",
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
    ax.axvline(mdates.date2num(TODAY), color=INK_2, linewidth=0.8, linestyle=":", zorder=2)
    ax.text(mdates.date2num(TODAY) + 0.2, -0.95, "today", fontsize=8, color=INK_2, va="center")
    ax.set_xlim(mdates.date2num(D(10, 1)), mdates.date2num(D(11, 26)))
    ax.set_ylim(-1.3, n + 2.6)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.grid(axis="x", which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel("2026 (major ticks on Mondays)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    handles = [Patch(facecolor=c, edgecolor="none", label=f"phase {p}") for p, c in PHASES.items()]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.17), ncol=6, frameon=False, fontsize=8.6,
              handlelength=1.2, columnspacing=1.4)
    ax.set_title("CoSiR v2 CVPR plan: experiments E0 to E17 and decision points (solid lines: hard deadlines)",
                 loc="left", fontsize=11.5, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(OUT, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
