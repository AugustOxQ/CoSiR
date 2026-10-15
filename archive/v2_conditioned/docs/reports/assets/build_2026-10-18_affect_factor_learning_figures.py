"""Figures for docs/reports/auto/v2/2026-10-18_candidate_a_affect_factor_learning_selection.md and
docs/reports/auto/v2/2026-10-19_candidate_a_affect_factor_learning_held.md (held_*.png).

Reads the affect run's stored results (gitignored, local only):
    src/test/20261018_affect_factor_learning/results/selection_results.json   (run_affect.py --evaluate)
    src/test/20261018_affect_factor_learning/results/replication.json         (run_affect.py --replicate)
    src/test/20261019_affect_factor_learning_held/results/held_results.json   (run_held.py --run; if present)
    src/test/20261018_affect_factor_learning/results/posthoc_affect.json      (run_posthoc_affect.py --run; if present)
and writes the PNGs to docs/reports/assets/2026-10-18_affect_factor_learning/.

Run from the repository root:
    python docs/reports/assets/build_2026-10-18_affect_factor_learning_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "src/test/20261018_affect_factor_learning/results/selection_results.json"
REPLICATION = ROOT / "src/test/20261018_affect_factor_learning/results/replication.json"
HELD_RESULTS = ROOT / "src/test/20261019_affect_factor_learning_held/results/held_results.json"
POSTHOC = ROOT / "src/test/20261018_affect_factor_learning/results/posthoc_affect.json"
OUT = ROOT / "docs/reports/assets/2026-10-18_affect_factor_learning"

SERIES_1, SERIES_2, SERIES_3 = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots (light surface)
INK, MUTED, GRID, SURFACE, REFERENCE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff", "#a3a29d"
BARS = ("C0", "S", "E", "SE")
BAR_COLOR = {"C0": REFERENCE, "S": SERIES_3, "E": SERIES_1, "SE": SERIES_2}
CELL_NAMES = {"R3": "R3\n(original)", "C0": "C0\n(control)", "S": "S\nimage clusters\n(reference)",
              "E": "E\naffect clusters", "SE": "SE\naffect + image"}
SCOPE_NAMES = {"pooled": "Pooled (8,192 episodes)", "emotion": "Emotion episodes (4,096)",
               "art_style": "Art-style episodes (4,096)"}

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def point_ci(block: dict) -> tuple[float, float, float]:
    return block["point"], block["ci95"][0], block["ci95"][1]


def signed(value: float) -> str:
    return f"{value:+.2f}".replace("-", "−")


def naive_figure(r: dict) -> None:
    """Naive R@1 at beta 0.3 per model and label type, with the C0 / R3 / CLIP-only / chance lines."""
    head = r["headline_beta0.3"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.8), dpi=200, sharey=True)
    x = np.arange(len(BARS))
    for ax, scope in zip(axes, ("pooled", "emotion", "art_style")):
        stats = [point_ci(head[m]["naive"][scope]["mean"]) for m in BARS]
        pts = np.array([s[0] for s in stats])
        err = np.array([[p - lo for p, lo, _ in stats], [hi - p for p, _, hi in stats]])
        ax.bar(x, pts, 0.6, color=[BAR_COLOR[m] for m in BARS], zorder=2, edgecolor=SURFACE, linewidth=1)
        ax.errorbar(x, pts, yerr=err, fmt="none", ecolor=INK, elinewidth=0.9, capsize=3, zorder=3)
        for xi, p in zip(x, pts):                       # values inside the bars: no clash with the lines
            ax.text(xi, 1.0, f"{p:.1f}", ha="center", va="bottom", fontsize=9, color=SURFACE, fontweight="bold")
        refs = ((head["C0"]["naive"][scope]["mean"]["point"], "-", "C0 (matched control)"),
                (head["R3"]["naive"][scope]["mean"]["point"], "--", "original R3 (current system)"),
                (head["clip_only"][scope]["mean"]["point"], "-.", "CLIP only"),
                (r["chance_r1"], ":", "chance"))
        for value, style, name in refs:
            ax.axhline(value, color=MUTED, linestyle=style, linewidth=1.0, zorder=1, label=f"{name}: {value:.1f}")
        ax.set_xticks(x, [CELL_NAMES[m] for m in BARS], fontsize=8)
        ax.set_title(SCOPE_NAMES[scope], fontsize=11, color=INK, loc="left", fontweight="bold")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_ylim(0, 44)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=1)
    axes[0].set_ylabel("Naive R@1 (%) at β 0.3, mean of i2t and t2i")
    fig.suptitle("Naive-rule R@1 on the selection episodes (seed 42, 95% bootstrap CIs)",
                 fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "naive_r1.png", facecolor=SURFACE)
    plt.close(fig)


def criterion_figure(r: dict) -> None:
    """D_emo and D_style per cell vs C0 with the 0 and -1.5 reference lines; pooled as context; S as reference."""
    rows = ("E", "SE", "S")
    series = ((r["d_emo"], SERIES_1, "D_emo: emotion (criterion, lower bound must be > 0)"),
              (r["d_style"], SERIES_2, "D_style: art style (guard, lower bound must be > −1.5)"),
              (r["d_pooled"], SERIES_3, "pooled (context, not gating)"))
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=200)
    y = np.arange(len(rows))[::-1].astype(float)
    offsets = (0.22, 0.0, -0.22)
    lows, highs = [], []
    for (blocks, color, name), off in zip(series, offsets):
        stats = [point_ci(blocks[c]) for c in rows]
        pts = np.array([s[0] for s in stats])
        lo, hi = np.array([s[1] for s in stats]), np.array([s[2] for s in stats])
        lows.append(lo.min())
        highs.append(hi.max())
        ax.errorbar(pts, y + off, xerr=[pts - lo, hi - pts], fmt="o", color=color, ecolor=color, elinewidth=1.6,
                    capsize=3, markersize=6, label=name, zorder=3)
        for yi, p, h in zip(y + off, pts, hi):
            ax.text(h + 0.08, yi, signed(p), va="center", fontsize=8, color=INK)
    ax.axvline(0.0, color=INK, linewidth=1.0, zorder=1)
    ax.axvline(-1.5, color=MUTED, linewidth=1.0, linestyle="--", zorder=1)
    top = len(rows) - 0.45
    ax.text(0.05, top, "0 (criterion)", color=INK, fontsize=8, va="bottom", ha="left")
    ax.text(-1.45, top, "−1.5 (style guard)", color=MUTED, fontsize=8, va="bottom", ha="left")
    gates = r["gates"]
    labels = []
    for c in rows:
        note = f"{gates[c]['n_binding_passed']}/8 binding gates"
        labels.append(f"{c} − C0\n({note})" + ("\nreference, not a candidate" if c == "S" else ""))
    ax.set_yticks(y, labels, fontsize=9)
    left, right = min(min(lows), -1.5) - 0.6, max(max(highs), 0.0) + 1.0
    ax.set_xlim(left, right)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("Paired difference in naive R@1 at β 0.3 (points), mean of i2t and t2i, 95% bootstrap CI")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
    fig.tight_layout()
    fig.savefig(OUT / "criterion.png", facecolor=SURFACE)
    plt.close(fig)


def beta_grid_figure(r: dict) -> None:
    """Cell - C0 in naive R@1 at every beta of the grid, emotion and art style, with 95% CIs."""
    betas = list(r["vs_c0_beta_grid"]["E"])
    xpos = np.arange(len(betas), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), dpi=200, sharey=True)
    for ax, scope, title in zip(axes, ("emotion", "art_style"), ("Emotion episodes", "Art-style episodes")):
        for cell, color, off in (("E", SERIES_1, -0.12), ("SE", SERIES_2, 0.0), ("S", SERIES_3, 0.12)):
            stats = [point_ci(r["vs_c0_beta_grid"][cell][b][scope]["mean"]) for b in betas]
            pts = np.array([s[0] for s in stats])
            lo, hi = np.array([s[1] for s in stats]), np.array([s[2] for s in stats])
            name = f"{cell} − C0" + (" (reference)" if cell == "S" else "")
            ax.errorbar(xpos + off, pts, yerr=[pts - lo, hi - pts], fmt="o-", color=color, ecolor=color,
                        elinewidth=1.3, capsize=3, markersize=5, linewidth=1.2, label=name, zorder=3)
        ax.axhline(0.0, color=INK, linewidth=1.0, zorder=1)
        if scope == "art_style":
            ax.axhline(-1.5, color=MUTED, linewidth=1.0, linestyle="--", zorder=1, label="−1.5 (style guard at β 0.3)")
        ax.axvspan(betas.index("0.3") - 0.3, betas.index("0.3") + 0.3, color=GRID, alpha=0.6, zorder=0)
        ax.set_xticks(xpos, [f"β {b}" + ("\n(pre-registered)" if b == "0.3" else "\n(scale-free)" if b == "0" else "")
                             for b in betas], fontsize=8.5)
        ax.set_title(title, fontsize=11, color=INK, loc="left", fontweight="bold")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    axes[0].set_ylabel("Cell − C0, naive R@1 (points), mean of directions")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=4)
    fig.suptitle("Differences to C0 across β (context, not gating): is a difference in the code or in its scale?",
                 fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "beta_grid.png", facecolor=SURFACE)
    plt.close(fig)


def per_target_figure(r: dict) -> None:
    """E - C0 and SE - C0 per emotion target at beta 0.3 and beta 0, with per-target CIs."""
    pt = r["extras"]["per_target"]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=200, sharey=True)
    order = sorted(pt["E"]["emotion"]["naive@0.3"], key=lambda t: pt["E"]["emotion"]["naive@0.3"][t]["diff"]["point"])
    y = np.arange(len(order), dtype=float)
    for ax, cell in zip(axes, ("E", "SE")):
        rows = pt[cell]["emotion"]
        for col, color, off, name in (("naive@0.3", SERIES_1, 0.17, "naive, β 0.3 (the pre-registered β)"),
                                      ("naive@0", SERIES_2, -0.17, "naive, β 0 (factor term only, scale-free)")):
            pts = np.array([rows[col][t]["diff"]["point"] for t in order])
            lo = np.array([rows[col][t]["diff"]["ci95"][0] for t in order])
            hi = np.array([rows[col][t]["diff"]["ci95"][1] for t in order])
            ax.errorbar(pts, y + off, xerr=[pts - lo, hi - pts], fmt="o", color=color, ecolor=color,
                        elinewidth=1.1, capsize=2, markersize=4, label=name, zorder=3)
        ax.axvline(0.0, color=INK, linewidth=1.0, zorder=1)
        ax.set_title(f"{cell} − C0, emotion targets", fontsize=11, color=INK, loc="left", fontweight="bold")
        ax.set_xlabel(f"{cell} − C0, naive R@1 (points), mean of directions, 95% CI per target")
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    names = [f"{t} (n={pt['E']['emotion']['naive@0.3'][t]['n']})" for t in order]
    axes[0].set_yticks(y, names, fontsize=8.5)
    axes[0].set_ylim(-0.7, len(order) - 0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.01, 0.945), ncol=2)
    fig.suptitle("Where the affect cells gain and lose against C0, per target emotion",
                 fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "per_target_emotion.png", facecolor=SURFACE)
    plt.close(fig)


def oracle_figure(r: dict) -> None:
    """Naive at beta 0.3 vs the label oracle at beta 0 per model and label type, with R3's oracle and the null."""
    head = r["headline_beta0.3"]
    models = ("R3", "C0", "S", "E", "SE")
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), dpi=200, sharey=True)
    x = np.arange(len(models))
    width = 0.38
    for ax, scope in zip(axes, ("pooled", "emotion", "art_style")):
        for offset, field, color, name in ((-width / 2 - 0.01, "naive", SERIES_1, "naive rule, β 0.3"),
                                           (width / 2 + 0.01, "oracle_0", SERIES_2, "label oracle, β 0")):
            stats = [point_ci(head[m][field][scope]["mean"]) for m in models]
            pts = np.array([s[0] for s in stats])
            err = np.array([[p - lo for p, lo, _ in stats], [hi - p for p, _, hi in stats]])
            ax.bar(x + offset, pts, width, color=color, label=name, zorder=2, edgecolor=SURFACE, linewidth=1)
            ax.errorbar(x + offset, pts, yerr=err, fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5, zorder=3)
            for xi, p in zip(x + offset, pts):           # values inside the bars: no clash with the lines
                ax.text(xi, 1.0, f"{p:.1f}", ha="center", va="bottom", fontsize=7, color=SURFACE,
                        fontweight="bold", rotation=90)
        null = np.mean([head[m]["oracle_null_0"][scope]["mean"]["point"] for m in models])
        ax.axhline(head["C0"]["oracle_0"][scope]["mean"]["point"], color=MUTED, linestyle="--", linewidth=1.0,
                   zorder=1, label=f"C0 label oracle: {head['C0']['oracle_0'][scope]['mean']['point']:.1f}")
        ax.axhline(null, color=MUTED, linestyle=":", linewidth=1.0, zorder=1,
                   label=f"oracle null (mean of models): {null:.1f}")
        ax.set_xticks(x, [CELL_NAMES[m] for m in models], fontsize=7.5)
        ax.set_title(SCOPE_NAMES[scope], fontsize=11, color=INK, loc="left", fontweight="bold")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_ylim(0, 42)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=1)
    axes[0].set_ylabel("R@1 (%), mean of i2t and t2i")
    fig.suptitle("How much label information can the cross-modal score use? Naive rule vs cross-validated label oracle",
                 fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "label_oracle.png", facecolor=SURFACE)
    plt.close(fig)


def training_figure(r: dict) -> None:
    """Condition loss and the learned tau over training for S, E and SE."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9), dpi=200)
    for cell, color in (("E", SERIES_1), ("SE", SERIES_2), ("S", SERIES_3)):
        h = r["training"][cell]["history"]
        name = cell + (" (reference)" if cell == "S" else "")
        axes[0].plot(h["step"], h["condition_loss"], color=color, linewidth=1.4, marker="o", markersize=2.5,
                     label=name)
        axes[1].plot(h["step"], h["tau"], color=color, linewidth=1.4, marker="o", markersize=2.5, label=name)
    axes[0].axhline(np.log(4), color=MUTED, linestyle=":", linewidth=1.0, label="uniform scores: log 4 = 1.386")
    axes[0].set_ylabel("condition episode loss (one batch)")
    axes[1].set_ylabel("learned temperature τ")
    for ax, title in zip(axes, ("Condition loss over training", "Learned τ over training")):
        ax.set_xlabel("training step")
        ax.set_title(title, fontsize=11, color=INK, loc="left", fontweight="bold")
        ax.grid(color=GRID, linewidth=0.8, zorder=0)
    axes[0].set_ylim(0.2, 2.0)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2)
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "condition_training.png", facecolor=SURFACE)
    plt.close(fig)


def held_criterion_figure(h: dict, r: dict, rep: dict) -> None:
    """Held test (2026-10-19 report): SE - C0 on held vs selection, per seed, and plain naive R@1 on held per model."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14.0, 5.6), dpi=200, gridspec_kw={"width_ratios": [1.15, 1.0]})
    rows = (("held, seed 42\n(the criterion)", h["d_emo"], h["d_style"], h["d_pooled"]),
            ("held, seed 43\n(context)", h["seeds_vs_c0"]["43"]["d_emo"], h["seeds_vs_c0"]["43"]["d_style"],
             h["seeds_vs_c0"]["43"]["d_pooled"]),
            ("held, seed 44\n(context)", h["seeds_vs_c0"]["44"]["d_emo"], h["seeds_vs_c0"]["44"]["d_style"],
             h["seeds_vs_c0"]["44"]["d_pooled"]),
            ("selection, seed 42\n(picked SE)", r["d_emo"]["SE"], r["d_style"]["SE"], r["d_pooled"]["SE"]),
            ("selection, seed 43", rep["per_seed"]["43"]["d_emo"], rep["per_seed"]["43"]["d_style"],
             rep["per_seed"]["43"]["d_pooled"]),
            ("selection, seed 44", rep["per_seed"]["44"]["d_emo"], rep["per_seed"]["44"]["d_style"],
             rep["per_seed"]["44"]["d_pooled"]))
    y = np.array([6.0, 5.0, 4.0, 2.4, 1.4, 0.4])
    series = ((1, SERIES_1, "D_emo: emotion (criterion: lower bound > 0)"),
              (2, SERIES_2, "D_style: art style (guard: lower bound > −1.5)"),
              (3, SERIES_3, "pooled (context)"))
    for (idx, color, name), off in zip(series, (0.24, 0.0, -0.24)):
        stats = [point_ci(row[idx]) for row in rows]
        pts = np.array([s[0] for s in stats])
        lo, hi = np.array([s[1] for s in stats]), np.array([s[2] for s in stats])
        ax.errorbar(pts, y + off, xerr=[pts - lo, hi - pts], fmt="o", color=color, ecolor=color, elinewidth=1.6,
                    capsize=3, markersize=5.5, label=name, zorder=3)
        for yi, p, top in zip(y + off, pts, hi):
            ax.text(top + 0.07, yi, signed(p), va="center", fontsize=7.5, color=INK)
    ax.axvline(0.0, color=INK, linewidth=1.0, zorder=1)
    ax.axvline(-1.5, color=MUTED, linewidth=1.0, linestyle="--", zorder=1)
    ax.axhline(3.2, color=GRID, linewidth=1.2, zorder=0)
    ax.text(0.05, 6.55, "0 (criterion)", color=INK, fontsize=8, va="bottom", ha="left")
    ax.text(-1.45, 6.55, "−1.5 (style guard)", color=MUTED, fontsize=8, va="bottom", ha="left")
    ax.text(3.95, 3.45, f"held: {h['meta']['n_per_label']:,} fresh episodes per label", fontsize=8, color=MUTED,
            ha="right", va="bottom")
    ax.text(3.95, 2.95, "selection: 4,096 episodes per label", fontsize=8, color=MUTED, ha="right", va="top")
    ax.set_yticks(y, [row[0] for row in rows], fontsize=8.5)
    ax.set_xlim(-2.0, 4.0)
    ax.set_ylim(-0.2, 6.9)
    ax.set_xlabel("SE − C0 of the same seed, naive R@1 at β 0.3 (points), mean of i2t and t2i, 95% CI")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_title("(a) SE minus the matched control C0", fontsize=11, color=INK, loc="left", fontweight="bold")

    # Baselines in neutral grays, SE in the one accent that panel (a) does not use (categorical slot 7, violet).
    models = (("clip_only", "CLIP only", "#c9c8c2", INK), ("R3", "original R3\n(current system)", REFERENCE, INK),
              ("C0_seed42", "C0\n(matched control)", MUTED, SURFACE), ("SE_seed42", "SE\n(picked)", "#4a3aa7", SURFACE))
    scopes = (("emotion", "emotion"), ("art_style", "art style"), ("pooled", "pooled"))
    x = np.arange(len(scopes), dtype=float)
    width = 0.2
    for i, (m, name, color, text_color) in enumerate(models):
        blocks = [(h["clip_only_r1"] if m == "clip_only" else h["naive_r1"][m])[s]["mean"] for s, _ in scopes]
        stats = [point_ci(b) for b in blocks]
        pts = np.array([s[0] for s in stats])
        err = np.array([[p - lo for p, lo, _ in stats], [top - p for p, _, top in stats]])
        pos = x + (i - 1.5) * width
        bx.bar(pos, pts, width * 0.94, color=color, label=name.replace("\n", " "), zorder=2, edgecolor=SURFACE,
               linewidth=0.8)
        bx.errorbar(pos, pts, yerr=err, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=3)
        for xi, p in zip(pos, pts):
            bx.text(xi, 0.8, f"{p:.1f}", ha="center", va="bottom", fontsize=7, color=text_color, fontweight="bold",
                    rotation=90)
    bx.axhline(h["chance_r1"], color=MUTED, linestyle=":", linewidth=1.0, zorder=1,
               label=f"chance: {h['chance_r1']:.1f}")
    bx.set_xticks(x, [name for _, name in scopes], fontsize=9.5)
    bx.set_ylabel("naive R@1 (%) at β 0.3 on held, mean of i2t and t2i")
    bx.set_ylim(0, 31)
    bx.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    bx.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    bx.set_title("(b) plain R@1 on the held episodes (seed 42 models)", fontsize=11, color=INK, loc="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "held_criterion.png", facecolor=SURFACE)
    plt.close(fig)


def held_per_target_figure(h: dict, r: dict) -> None:
    """SE - C0 per target emotion at beta 0.3: held (seed-43 episodes) beside selection (seed-42 episodes)."""
    held, sel = h["per_target"]["emotion"]["naive@0.3"], r["extras"]["per_target"]["SE"]["emotion"]["naive@0.3"]
    order = sorted(held, key=lambda t: held[t]["diff"]["point"])
    y = np.arange(len(order), dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=200)
    for rows, color, off, name in ((held, SERIES_1, 0.17, "held (8,192 episodes, seed 43)"),
                                   (sel, REFERENCE, -0.17, "selection (4,096 episodes, seed 42)")):
        pts = np.array([rows[t]["diff"]["point"] for t in order])
        lo = np.array([rows[t]["diff"]["ci95"][0] for t in order])
        hi = np.array([rows[t]["diff"]["ci95"][1] for t in order])
        ax.errorbar(pts, y + off, xerr=[pts - lo, hi - pts], fmt="o", color=color, ecolor=color, elinewidth=1.2,
                    capsize=2, markersize=4.5, label=name, zorder=3)
    ax.axvline(0.0, color=INK, linewidth=1.0, zorder=1)
    ax.set_yticks(y, [f"{t} (held n={held[t]['n']})" for t in order], fontsize=8.5)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("SE − C0 (seed 42), naive R@1 at β 0.3 (points), mean of directions, 95% CI per target")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("Where SE's emotion gain over C0 comes from, per target emotion", fontsize=11, color=INK,
                 loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "held_per_target_emotion.png", facecolor=SURFACE)
    plt.close(fig)


def support_curve_figure(p: dict) -> None:
    """Post-hoc: naive R@1 at beta 0.3 against the number of supports (= contrasts), with each model's stored oracle."""
    c = p["support_curve"]
    counts = [int(k) for k in c["per_count"]]
    xpos = np.arange(len(counts), dtype=float)
    models = (("C0", REFERENCE, -0.08), ("E", SERIES_1, 0.0), ("SE", SERIES_2, 0.08))
    names = {"C0": "C0 (control)", "E": "E (affect clusters)", "SE": "SE (affect + image clusters)"}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), dpi=200)
    for ax, scope, title in zip(axes, ("emotion", "art_style"), ("Emotion episodes", "Art-style episodes")):
        for m, color, off in models:
            stats = [point_ci(c["per_count"][str(k)]["0.3"]["r1"][m][scope]["mean"]) for k in counts]
            pts = np.array([s[0] for s in stats])
            lo, hi = np.array([s[1] for s in stats]), np.array([s[2] for s in stats])
            ax.errorbar(xpos + off, pts, yerr=[pts - lo, hi - pts], fmt="o-", color=color, ecolor=color,
                        elinewidth=1.3, capsize=3, markersize=5, linewidth=2.0, label=f"{names[m]}, naive rule",
                        zorder=3)
            ax.axhline(c["stored_4096"][m]["oracle@0.3"][scope], color=color, linestyle="--", linewidth=1.2,
                       zorder=2)
        ax.set_xticks(xpos, [f"{k}" + ("\n(the evaluated setting)" if k == 4 else "") for k in counts], fontsize=8.5)
        ax.set_xlabel("supports per episode (and as many contrasts)", fontsize=9)
        ax.set_title(title, fontsize=11, color=INK, loc="left", fontweight="bold")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    axes[0].set_ylabel("Naive R@1 (%) at β 0.3, mean of directions")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=MUTED, linestyle="--", linewidth=1.2))
    labels.append("same model's label oracle, β 0.3 (stored, 4,096 episodes)")
    fig.legend(handles, labels, frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=2)
    fig.suptitle("Post-hoc, selection rows: more supports close the gap to the label oracle (2,048 new episodes "
                 "per label and count)", fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(OUT / "support_curve.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r = json.loads(RESULTS.read_text())
    naive_figure(r)
    criterion_figure(r)
    beta_grid_figure(r)
    per_target_figure(r)
    oracle_figure(r)
    training_figure(r)
    if HELD_RESULTS.exists():                          # the 2026-10-19 held report's figures
        h = json.loads(HELD_RESULTS.read_text())
        held_criterion_figure(h, r, json.loads(REPLICATION.read_text()))
        held_per_target_figure(h, r)
    if POSTHOC.exists():                               # post-hoc diagnostics (final-review fix wave)
        support_curve_figure(json.loads(POSTHOC.read_text()))
    print(f"wrote {sorted(p.name for p in OUT.glob('*.png'))} to {OUT}")


if __name__ == "__main__":
    main()
