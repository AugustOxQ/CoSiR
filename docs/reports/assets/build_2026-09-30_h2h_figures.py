"""Figures for docs/reports/auto/percept/2026-09-30_matched_percept_buddy_h2h.md.

Fig 1 (sweep): every trial of the four equal-budget sweeps, val Stage 2 macro
AUC (2-seed mean) against val independent emotion AMI, one panel per topic
count, with each system's Pareto front and the emotion floor used by the
secondary selection.

Fig 2 (test): the cell winners and the two references on the test half (not
used for selection in this experiment), one point per test seed plus the
mean, for the primary and the emotion-constrained selection.

Fig 3 (val stress vs test): each of the eight cell winners on the val half
(4 stress seeds) and on the test half (5 test seeds), for Stage 2 AUC and
both independent AMIs, so a reader can see which differences replicate
across the two halves.

Colours: the validated default categorical palette, slots 1-3 (PercepT blue,
buddy pilot Stage 1 orange, buddy harness Stage 1 aqua); each system also has
its own marker, so identity never rests on colour alone.

Usage: python docs/reports/assets/build_2026-09-30_h2h_figures.py [fig1|fig2|fig3|all]
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "src/test/20260930_matched_h2h"
OUT = Path(__file__).resolve().parent / "2026-09-30_h2h"
EMOTION_FLOOR = 0.1117
PARETO_BAR_EMOTION = 0.1236

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {
    "percept": {"color": "#2a78d6", "marker": "o", "label": "PercepT"},
    "buddy_pilot": {"color": "#eb6834", "marker": "^", "label": "buddy, pilot Stage 1"},
    "buddy_harness": {"color": "#1baf7a", "marker": "s", "label": "buddy, §6i harness Stage 1"},
}


def _axes_style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _series(row):
    if row["cell"].startswith("percept"):
        return "percept"
    return "buddy_pilot" if row["config"].get("buddy_impl") == "pilot" else "buddy_harness"


def _pareto(points):
    front, best = [], -1.0
    for emo, auc in sorted(points, key=lambda p: (-p[0], -p[1])):
        if auc > best:
            front.append((emo, auc))
            best = auc
    return front


def fig1():
    rows = json.loads((DATA / "sweep_runs.json").read_text())
    valid = [r for r in rows if r["state"] == "finished" and r["objective"] is not None
             and r["objective"] > -1 and r["summary"].get("ind_emo") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor=SURFACE)
    for ax, k in zip(axes, (16, 40)):
        _axes_style(ax)
        cell_rows = [r for r in valid if r["cell"].endswith(f"k{k}")]
        counts = {}
        for key in ("buddy_harness", "buddy_pilot", "percept"):
            pts = [(r["summary"]["ind_emo"], r["objective"]) for r in cell_rows if _series(r) == key]
            st = STYLE[key]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=16, marker=st["marker"],
                       color=st["color"], alpha=0.55, linewidths=0, label=st["label"])
            counts[key] = len(pts)
        for system, color in (("percept", STYLE["percept"]["color"]), ("buddy", INK)):
            pts = [(r["summary"]["ind_emo"], r["objective"]) for r in cell_rows
                   if (_series(r) == "percept") == (system == "percept")]
            front = _pareto(pts)
            ax.step([p[0] for p in front], [p[1] for p in front], where="post", color=color,
                    linewidth=2, label=f"{'PercepT' if system == 'percept' else 'buddy (both Stage 1s)'} Pareto front")
        ax.axvline(EMOTION_FLOOR, color=INK_2, linestyle="--", linewidth=1.2)
        ax.text(EMOTION_FLOOR + 0.001, 0.62, f"emotion floor {EMOTION_FLOOR}", color=INK_2, fontsize=8, rotation=90)
        ax.axvline(PARETO_BAR_EMOTION, color=INK_2, linestyle=":", linewidth=1.2)
        ax.text(PARETO_BAR_EMOTION + 0.001, 0.62, f"Pareto bar {PARETO_BAR_EMOTION}", color=INK_2, fontsize=8, rotation=90)
        ax.set_title(f"K = {k}   (n: PercepT {counts['percept']}, buddy pilot {counts['buddy_pilot']}, "
                     f"buddy harness {counts['buddy_harness']})", color=INK, fontsize=10)
        ax.set_xlabel("val independent emotion AMI (pilots' re-clustering)", color=INK, fontsize=9)
    axes[0].set_ylabel("val Stage 2 macro AUC (2-seed mean)", color=INK, fontsize=9)
    axes[0].set_ylim(0.6, 1.0)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Every sweep trial: Stage 2 AUC against emotion structure (300 trials per system per K)",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig1_auc_vs_emotion.png", dpi=160, facecolor=SURFACE)
    print(f"wrote {OUT / 'fig1_auc_vs_emotion.png'}")


def _test_seeds():
    """tag -> list of per-seed rows, from the H2H_SEED lines of the test logs."""
    out = {}
    for path in sorted((DATA / "logs").glob("h2h-test-*.log")):
        for line in path.read_text().splitlines():
            if line.startswith("H2H_SEED "):
                row = json.loads(line[len("H2H_SEED "):])
                out.setdefault(row["tag"], []).append(row)
    return out


def fig2():
    seeds = _test_seeds()
    groups = [("K=16\nplain AUC", "test_primary_{}_k16"), ("K=16\nemotion-\nconstrained", "test_constrained_{}_k16"),
              ("K=40\nplain AUC", "test_primary_{}_k40"), ("K=40\nemotion-\nconstrained", "test_constrained_{}_k40")]
    refs = [("§6i winner\nm8x7ifx4\nK=14-15", "ref_m8x7ifx4", "buddy_harness"),
            ("§6g PercepT\nconfig\nK=40", "ref_percept_6g", "percept")]
    metrics = (("auc_primary", "test Stage 2 macro AUC"), ("ind_emo", "test independent emotion AMI"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    for ax, (metric, ylabel) in zip(axes, metrics):
        _axes_style(ax)
        xticks, xlabels, x = [], [], 0.0
        for label, pattern in groups:
            for offset, system in ((-0.18, "percept"), (0.18, "buddy")):
                rows = seeds.get(pattern.format(system), [])
                vals = [r[metric] for r in rows if r.get(metric) is not None]
                if not vals:
                    continue
                key = "percept" if system == "percept" else "buddy_harness"
                st = STYLE[key]
                ax.scatter([x + offset] * len(vals), vals, s=18, marker=st["marker"], color=st["color"], alpha=0.6, linewidths=0)
                mean = sum(vals) / len(vals)
                ax.hlines(mean, x + offset - 0.12, x + offset + 0.12, color=st["color"], linewidth=2.5)
            xticks.append(x)
            xlabels.append(label)
            x += 1.0
        for label, tag, key in refs:
            vals = [r[metric] for r in seeds.get(tag, []) if r.get(metric) is not None]
            if vals:
                st = STYLE[key]
                ax.scatter([x] * len(vals), vals, s=18, marker=st["marker"], color=st["color"], alpha=0.6, linewidths=0)
                ax.hlines(sum(vals) / len(vals), x - 0.12, x + 0.12, color=st["color"], linewidth=2.5)
            xticks.append(x)
            xlabels.append(label)
            x += 1.0
        if metric == "ind_emo":
            ax.axhline(EMOTION_FLOOR, color=INK_2, linestyle="--", linewidth=1.2)
            ax.axhline(PARETO_BAR_EMOTION, color=INK_2, linestyle=":", linewidth=1.2)
            ax.text(x - 0.6, EMOTION_FLOOR + 0.001, "emotion floor", fontsize=7, color=INK_2, ha="right")
            ax.text(x - 0.6, PARETO_BAR_EMOTION + 0.001, "Pareto bar", fontsize=7, color=INK_2, ha="right")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=8, color=INK)
        ax.set_ylabel(ylabel, color=INK, fontsize=9)
    handles = [plt.Line2D([], [], marker=STYLE[k]["marker"], color=STYLE[k]["color"], linestyle="none", markersize=6,
                          label=lab) for k, lab in (("percept", "PercepT"), ("buddy_harness", "buddy (winning configs use the harness Stage 1)"))]
    handles.append(plt.Line2D([], [], color=INK_2, linewidth=2.5, label="mean over 5 test seeds"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Test half (not used for selection here), 5 fresh seeds: cell winners and the two earlier references",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig2_test_results.png", dpi=160, facecolor=SURFACE)
    print(f"wrote {OUT / 'fig2_test_results.png'}")


WINNERS = {  # (selection, K) -> (buddy run id, PercepT run id); from stress_summary.md
    ("primary", 16): ("6edlxmyv", "x0oa5511"), ("primary", 40): ("89mkiavu", "0biqmu50"),
    ("constrained", 16): ("mssv0f7s", "usnsbxdq"), ("constrained", 40): ("4fu1936m", "i1rigwnq"),
}


def _seed_rows(pattern):
    """(run_id, subset) -> per-seed rows, from the H2H_SEED lines of matching logs."""
    out = {}
    for path in sorted((DATA / "logs").glob(pattern)):
        for line in path.read_text().splitlines():
            if line.startswith("H2H_SEED "):
                row = json.loads(line[len("H2H_SEED "):])
                out.setdefault((row["run_id"], row["subset"]), []).append(row)
    return out


def fig3():
    rows = {**_seed_rows("h2h-stress-*.log"), **_seed_rows("h2h-test-*.log")}
    metrics = (("auc_primary", "Stage 2 macro AUC"), ("ind_emo", "independent emotion AMI"),
               ("ind_genre", "independent genre AMI"))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), facecolor=SURFACE)
    for r, selection in enumerate(("primary", "constrained")):
        for c, (metric, title) in enumerate(metrics):
            ax = axes[r, c]
            _axes_style(ax)
            xticks, xlabels = [], []
            for g, k in enumerate((16, 40)):
                base = 3.0 * g
                for offset, system, run_id in ((-0.08, "percept", WINNERS[(selection, k)][1]),
                                               (0.08, "buddy", WINNERS[(selection, k)][0])):
                    st = STYLE["percept" if system == "percept" else "buddy_harness"]
                    means = []
                    for j, subset in enumerate(("val", "test")):
                        x = base + j + offset
                        vals = [row[metric] for row in rows[(run_id, subset)] if row.get(metric) is not None]
                        ax.scatter([x] * len(vals), vals, s=16, marker=st["marker"], color=st["color"],
                                   alpha=0.45, linewidths=0)
                        means.append(sum(vals) / len(vals))
                        ax.hlines(means[-1], x - 0.12, x + 0.12, color=st["color"], linewidth=2.5)
                    ax.plot([base + offset, base + 1 + offset], means, color=st["color"], linewidth=1.5)
                xticks += [base, base + 1]
                xlabels += [f"K={k}\nval stress", f"K={k}\ntest"]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels, fontsize=8, color=INK)
            ax.set_title(f"{'plain-AUC' if selection == 'primary' else 'emotion-constrained'} winners: {title}",
                         color=INK, fontsize=9)
    handles = [plt.Line2D([], [], marker=STYLE[k]["marker"], color=STYLE[k]["color"], linestyle="none",
                          markersize=6, label=lab)
               for k, lab in (("percept", "PercepT (baseline)"), ("buddy_harness", "buddy (harness Stage 1)"))]
    handles.append(plt.Line2D([], [], color=INK_2, linewidth=2.5, label="mean (4 val seeds, 5 test seeds)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("The same eight winners on the val half (stress seeds) and the test half (test seeds)",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig3_val_vs_test.png", dpi=160, facecolor=SURFACE)
    print(f"wrote {OUT / 'fig3_val_vs_test.png'}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("fig1", "all"):
        fig1()
    if which in ("fig2", "all"):
        fig2()
    if which in ("fig3", "all"):
        fig3()


if __name__ == "__main__":
    main()
