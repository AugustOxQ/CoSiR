"""Figures 1 to 3 and 5 to 9 of docs/reports/auto/v2/2026-11-17_ars_reader_fix_plan_review.md (Figure 4 is
build_rule_flow.py). Diagrams use the report's change colours (grey same, orange replaced, teal new); charts use the
validated categorical slots blue, orange, aqua.

Numbers: Figure 2 from src/test/20261116_grouping_step1_style/20261116_grouping_step1_style_log.md (Arms table,
seed 42); Figures 6 and 7 from the methodology seat's summary-level estimates in
src/test/20261117_reader_fix_csd/ars_review/phase2_methodology.md (W3, W6), re-derived in the review log.

    /root/miniconda3/envs/CoSiR/bin/python docs/reports/assets/2026-11-17_ars_reader_fix_plan_review/build_figures.py
"""
from math import erf, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

HERE = Path(__file__).parent
COL = {"same": ("#eceff1", "#607d8b"), "replaced": ("#ffe0b2", "#e65100"), "new": ("#b2dfdb", "#00695c"),
       "ctrl": ("#ffffff", "#607d8b")}
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, body, kind, size=9, dashed=False, title_size=None):
    face, edge = COL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", facecolor=face,
                                edgecolor=edge, linewidth=1.5, linestyle="--" if dashed else "-"))
    ty = y + h - 0.16
    if title:
        ax.text(x + 0.14, ty, title, ha="left", va="top", fontsize=title_size or size + 0.6, fontweight="bold",
                color=INK)
        ty -= 0.36
    if body:
        ax.text(x + 0.14, ty, body, ha="left", va="top", fontsize=size, color=INK, linespacing=1.4)


def arrow(ax, x0, y0, x1, y1, label=None, dx=0.0, dy=0.12, color="#37474f", style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=13, linewidth=1.3,
                                 color=color))
    if label:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, label, fontsize=8.5, ha="center", va="bottom", color=INK2,
                style="italic")


def change_legend(ax, kinds, y=None, x=None):
    labels = {"same": "unchanged by the plan", "replaced": "replaced by the plan or a review fix",
              "new": "new (review fix or controller addition)", "ctrl": "comparison, not part of the score"}
    hs = [Patch(facecolor=COL[k][0], edgecolor=COL[k][1], linestyle="--" if k == "ctrl" else "-", label=labels[k])
          for k in kinds]
    ax.legend(handles=hs, loc="upper right", bbox_to_anchor=(1.0, 1.02), fontsize=9, frameon=False, ncol=len(kinds))


def save(fig, name):
    out = HERE / name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out)


# ------------------------------------------------------------------ Figure 1: how one episode is scored

def fig_pipeline():
    fig, ax = canvas(15.5, 7.0)
    top = 6.25
    box(ax, 0.1, 2.0, 2.75, 4.0, "1. Episode",
        "query: one image or one\ncaption of a painting\n\n4 support pairs: image of\none painting + caption of\n"
        "another, sharing a value\nof the hidden aspect A\n\n4 contrast pairs: the same\nfor a second aspect B\n\n"
        "13 candidates (other\nmodality) to rank", "same")
    box(ax, 3.25, 2.0, 3.0, 4.0, "2. Four groupings, heads",
        "each grouping splits the\nscorer-train rows into groups\nwithout evaluation labels:\n\n"
        "  affect: Leiden, GoEmotions, 41\n  image: k-means, CLIP, 64\n  caption: k-means, CLIP, 64\n"
        "  CSD: Leiden, CSD style, 17\n\nper grouping an image head and\na caption head give every item\n"
        "group probabilities p_h", "same")
    box(ax, 6.65, 4.05, 3.1, 1.95, "3. Evidence per grouping",
        "agreement of a pair =\np_h(image) · p_h(caption)\nΔ_h = mean over support pairs\n"
        "  − mean over contrast pairs\n(condition b: Δ_h flips sign)", "same")
    box(ax, 6.65, 2.0, 3.1, 1.75, "B: condition-free score",
        "cosine + centred factor term +\nheads averaged over groupings;\nweights cross-fitted; R@1 18.34;\n"
        "ignores the aspect shown", "same")
    box(ax, 10.15, 4.05, 2.6, 1.95, "4. Reader",
        "decides which grouping the\nsupports share (R-a, R-b,\nR-c: Figure 3) → reader\nterm T = query-candidate\n"
        "agreement there", "replaced")
    box(ax, 12.95, 2.0, 2.45, 4.0, "5. Fused score",
        "(1 + λ_u)·z(B) + λ_a·z(T)\n\nz: z-score within each\nranking; λ picked on one\nhalf of the episodes,\n"
        "used on the other half\n\nrank the 13 candidates;\nthe target should be\nfirst (R@1)", "same")
    box(ax, 0.1, 0.45, 15.3, 1.15, "How the score is judged (comparators, not part of the score)",
        "matched counterpart: the same fused score with T replaced by (T under a + T under b) / 2, so it keeps every "
        "ingredient and drops only the condition\nB′: B rebuilt with the heads averaged over all four groupings   ·   "
        "pass on seed 42: the fused score beats the larger of the two (and B) by ≥ +0.5 R@1 with a lower bound > 0",
        "ctrl", size=8.8, dashed=True)
    arrow(ax, 2.85, 4.0, 3.25, 4.0)
    arrow(ax, 6.25, 5.0, 6.65, 5.0)
    arrow(ax, 6.25, 2.9, 6.65, 2.9)
    arrow(ax, 9.75, 5.0, 10.15, 5.0)
    arrow(ax, 12.75, 5.0, 12.95, 5.0)
    arrow(ax, 9.75, 2.9, 12.95, 2.9)
    ax.text(0.1, 6.95, "How one episode is scored: the plan changes only the reader (step 4)", fontsize=11,
            fontweight="bold", va="top", color=INK)
    hs = [Patch(facecolor=COL[k][0], edgecolor=COL[k][1], linestyle="--" if k == "ctrl" else "-", label=l)
          for k, l in (("same", "unchanged by the plan"), ("replaced", "replaced by the plan"),
                       ("ctrl", "comparison, not part of the score"))]
    ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.0, 0.955), fontsize=9, frameon=False, ncol=3)
    save(fig, "method_pipeline.png")


# ------------------------------------------------------------------ Figure 2: where seed 42 stands

def fig_comparators():
    names = ["A0 (affect, image, caption)", "AR (A0 + random grouping)", "A1 (A0 + CSD style)"]
    cf = [0.05, 0.02, 0.41]      # matched counterpart minus B
    bp = [0.10, 0.11, 0.46]      # B′ minus B
    rd = [0.41, 0.27, 0.47]      # fused reader minus B
    bar = ["+0.31 [0.10, 0.53]", "+0.16 [−0.02, 0.35]", "+0.01 [−0.23, 0.24]"]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(3)
    w = 0.24
    for k, (vals, col, lab) in enumerate(((cf, BLUE, "matched counterpart"), (bp, ORANGE, "B′ (B rebuilt)"),
                                          (rd, AQUA, "fused reader (current arg-max)"))):
        xs = x + (k - 1) * (w + 0.02)
        ax.bar(xs, vals, width=w, color=col, label=lab, zorder=3)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.02, f"+{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    for i in range(3):
        need = max(cf[i], bp[i]) + 0.5
        ax.plot([i - 0.42, i + 0.42], [need, need], color=INK, linestyle=(0, (4, 3)), linewidth=1.4, zorder=4)
        ax.text(i, need + 0.015, f"pass line +{need:.2f}", fontsize=8.5, va="bottom", ha="center", color=INK)
    ax.axhline(0, color=INK2, linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\nbar margin {m}" for n, m in zip(names, bar)])
    ax.tick_params(axis="x", pad=6, length=0)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlim(-0.55, 2.6)
    ax.set_ylabel("R@1 above B (points), seed 42")
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0\n(B = 18.34)", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=3, bbox_to_anchor=(0.0, 1.1))
    save(fig, "comparators_seed42.png")


# ------------------------------------------------------------------ Figure 3: the reader candidates

def fig_readers():
    fig, ax = canvas(15.6, 7.0)
    cols = [
        ("Current reader (step 1)", "same",
         "pick the grouping with the\nlargest raw Δ_h",
         "agreement of query and\ncandidate on that grouping",
         "(T under a + T under b) / 2",
         "coarse groupings win by\nnoise; CSD wins genre\nconditions"),
        ("R-a  scaled Δ", "replaced",
         "divide each Δ_h by σ_h, its\nnoise-only spread (pooled\nwithin-episode standard error,\n"
         "frozen from seed 42); pick\nthe largest",
         "agreement on the picked\ngrouping",
         "(T under a + T under b) / 2",
         "failure 1 (noise)\ncost: minutes"),
        ("R-b  learned reader", "replaced",
         "logistic regression on features\nof all four groupings → P(h);\ntrained on bank episodes\n"
         "(scorer-train rows, cross-\nfitted heads), frozen first",
         "arg-max: agreement on the most\nprobable grouping; expected:\nΣ_h P(h)·s_h (s_h = agreement\non grouping h)",
         "arg-max: (T_a + T_b) / 2;\nexpected: Σ_h P_avg(h)·s_h,\nP_avg = mean of P over a and b",
         "failures 1 and 2 (CSD vs\nimage for genre)\ncost: hours"),
        ("R-c  confidence gate", "replaced",
         "on the best of R-a and R-b:\ng_c = 1 if the top-two margin\nunder condition c ≥ τ, else 0;\n"
         "τ: 0/25/50/75th percentile",
         "gated term g_c·z(T_c), T from\nthe base reader; fused over\n56 weights × 4 τ = 224 cells",
         "(g_a·z(T_a) + g_b·z(T_b)) / 2\n(draft used g_c·z(T_cf),\nwhich is not condition-free)",
         "wrong picks that remain\ncost: minutes"),
    ]
    rows = [("Rule", 4.55, 1.75), ("Reader term", 3.1, 1.25), ("Matched counterpart", 1.7, 1.2),
            ("Targets", 0.4, 1.1)]
    x0, w, gap = 1.75, 3.3, 0.12
    for r, (label, y, h) in enumerate(rows):
        ax.text(1.6, y + h / 2, label, ha="right", va="center", fontsize=9.5, fontweight="bold", color=INK2)
    for c, (title, kind, rule, term, cf, target) in enumerate(cols):
        x = x0 + c * (w + gap)
        ax.text(x + w / 2, 6.55, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
        for (label, y, h), text in zip(rows, (rule, term, cf, target)):
            k = kind if label != "Targets" else ("same" if c == 0 else "ctrl")
            box(ax, x, y, w, h, None, text, k, size=8.9)
    change_legend(ax, ["same", "replaced"])
    ax.text(0.1, 7.0, "Four readers; the decision rule picks which one is carried to the test", fontsize=11,
            fontweight="bold", va="top", color=INK)
    save(fig, "reader_candidates.png")


# ------------------------------------------------------------------ Figure 5: R-c's counterpart

def fig_rc_counterpart():
    fig, ax = canvas(14.0, 6.0)
    ax.text(0.1, 5.95, "Why the draft control of R-c reads the condition (toy numbers, scaled Δ, τ = 0.010)",
            fontsize=11, fontweight="bold", va="top", color=INK)
    hdr = ["", "affect", "image", "caption", "CSD", "top two", "margin", "gate"]
    rows = [["condition a: Δ", "+0.020", "+0.012", "−0.004", "+0.010", "affect, image", "0.008", "g_a = 0"],
            ["condition b: −Δ", "−0.020", "−0.012", "+0.004", "−0.010", "caption, CSD", "0.014", "g_b = 1"]]
    xs = [0.2, 2.4, 3.5, 4.6, 5.8, 7.0, 8.9, 10.2]
    for j, h in enumerate(hdr):
        ax.text(xs[j], 5.2, h, fontsize=9.5, fontweight="bold", color=INK2, va="center")
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            ax.text(xs[j], 4.75 - 0.42 * i, v, fontsize=9.5, color=INK, va="center",
                    fontweight="bold" if j == 7 else "normal")
    ax.text(0.2, 3.75, "Δ flips sign between the conditions, so the gap between the two largest values differs, "
            "and the gate opens under b but not under a.", fontsize=9.5, color=INK2, va="center")
    box(ax, 0.2, 0.65, 6.6, 2.65, "Draft: the same gate applied to T_cf",
        "counterpart term = g_c · z(T_cf)\n\nunder a:  0 · z(T_cf) = 0\nunder b:  1 · z(T_cf)\n\n"
        "different under a and b, so it still reads the\ncondition; the code's condition-free check\n"
        "(crossfit_condition_free) raises an error", "replaced")
    box(ax, 7.2, 0.65, 6.6, 2.65, "Fix: the two-condition mean of the gated term",
        "G_cf = (g_a·z(T_a) + g_b·z(T_b)) / 2\n\nunder a:  (0 + z(T_b)) / 2\nunder b:  (0 + z(T_b)) / 2\n\n"
        "identical under a and b: removes only the\ncondition, the same operator as every other\ncounterpart",
        "new")
    save(fig, "rc_counterpart.png")


# ------------------------------------------------------------------ Figure 6: R-a's spread

def fig_ra_spread():
    noise, rms, signal = 0.017, 0.025, 0.0236
    xs = np.linspace(-3.5, 5.0, 600)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=True)
    for ax, (scale, title) in zip(axes, ((rms, "Planned: divide by the RMS of Δ (0.025)"),
                                          (noise, "Fix: divide by the noise-only spread (0.017)"))):
        mu_i, sd_i = signal / scale, noise / scale
        pu = 1 - phi(mu_i / sqrt(1 + sd_i ** 2))
        dens_u = np.exp(-xs ** 2 / 2) / sqrt(2 * np.pi)
        dens_i = np.exp(-((xs - mu_i) / sd_i) ** 2 / 2) / (sd_i * sqrt(2 * np.pi))
        ax.fill_between(xs, dens_u, color=ORANGE, alpha=0.18, linewidth=0)
        ax.plot(xs, dens_u, color=ORANGE, linewidth=2, label="a grouping that carries nothing")
        ax.fill_between(xs, dens_i, color=BLUE, alpha=0.18, linewidth=0)
        ax.plot(xs, dens_i, color=BLUE, linewidth=2, label="image grouping, genre condition")
        ax.axvline(mu_i, color=BLUE, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(mu_i + 0.08, 0.6, f"mean {mu_i:.2f}", color=INK, fontsize=9)
        ax.text(-3.4, 0.55, f"the empty grouping\noutranks image in\n{100 * pu:.0f}% of episodes", fontsize=9.5,
                color=INK, va="top")
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("scaled Δ (Δ divided by the spread)")
        ax.yaxis.grid(True, color=GRID)
    axes[0].set_ylabel("density")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.32), frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    save(fig, "ra_spread.png")


# ------------------------------------------------------------------ Figure 7: test sensitivity

def fig_power():
    m = np.linspace(0, 0.6, 300)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for se, col, lab in ((0.066, BLUE, "episode noise dominates (half-width ≈ 0.13)"),
                         (0.097, ORANGE, "painting variation dominates (half-width ≈ 0.19)")):
        p = [phi(v / se - 1.96) for v in m]
        ax.plot(m, p, color=col, linewidth=2, label=lab)
        for v in (0.15, 0.25):
            pv = phi(v / se - 1.96)
            ax.plot([v], [pv], "o", color=col, markersize=7, markeredgecolor="white", markeredgewidth=1.5, zorder=4)
            ax.text(v + 0.008, pv - 0.035, f"{pv:.2f}", fontsize=9, color=INK, va="top")
    for v, lab in ((0.15, "+0.15"), (0.25, "+0.25\nhalf the bar"), (0.5, "+0.5\ndevelopment bar")):
        ax.axvline(v, color=INK2, linewidth=0.9, linestyle=(0, (3, 3)))
        ax.text(v, 1.03, lab, ha="center", va="bottom", fontsize=8.8, color=INK2)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("true margin on the fresh seeds (R@1 points)")
    ax.set_ylabel("chance one comparison's lower bound > 0")
    ax.yaxis.grid(True, color=GRID)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    save(fig, "test_power.png")


# ------------------------------------------------------------------ Figure 8: timeline

def fig_timeline():
    fig, ax = plt.subplots(figsize=(14.5, 5.0))
    day = 24.0

    def t(d, hh):
        return d * day + hh

    lanes = ["Rule", "R-a, R-c", "R-b", "Checks", "Test", "Decision"]
    items = [
        (0, t(0, 8), t(0, 13), "rewrite §5 as DECISION_RULE.md, user approves, commit", "new"),
        (1, t(0, 13), t(0, 19), "R-a on A1 and A0 (+ AR check)", "replaced"),
        (1, t(1, 13), t(1, 20), "R-c on the best of R-a, R-b", "replaced"),
        (2, t(0, 13), t(1, 10), "cross-fitted heads and banks (A1, A0)", "replaced"),
        (2, t(1, 10), t(1, 20), "train, evaluate R-b", "replaced"),
        (3, t(1, 20), t(2, 0), "re-derive dev numbers", "same"),
        (3, t(2, 18), t(2, 22), "re-derive\ntest numbers", "new"),
        (3, t(3, 8), t(3, 12), "final whole-branch review", "new"),
        (4, t(2, 12), t(2, 14), "apply rule, project sensitivity", "new", "left"),
        (4, t(2, 14), t(2, 18), "seeds 49 to 51: the one test", "same"),
        (5, t(3, 12), t(3, 16), "user decides: GO or NO-GO", "same"),
    ]
    for d, lab in enumerate(("Tue 6 Oct", "Wed 7 Oct", "Thu 8 Oct", "Fri 9 Oct")):
        ax.axvspan(t(d, 0), t(d, 24), color="#f6f6f4" if d % 2 == 0 else "white", zorder=0)
        ax.text(t(d, 12), len(lanes) - 0.25, lab, ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
    for item in items:
        lane, a0, b0, lab, kind = item[:5]
        side = item[5] if len(item) > 5 else "right"
        y = len(lanes) - 1 - lane
        face, edge = COL[kind]
        ax.barh(y, b0 - a0, left=a0, height=0.56, color=face, edgecolor=edge, linewidth=1.3, zorder=3)
        inside = (b0 - a0) >= 9
        if inside:
            ax.text((a0 + b0) / 2, y, lab, ha="center", va="center", fontsize=8.6, color=INK, zorder=4)
        elif side == "left":
            ax.text(a0 - 0.6, y, lab, ha="right", va="center", fontsize=8.6, color=INK, zorder=4)
        else:
            ax.text(b0 + 0.6, y, lab, ha="left", va="center", fontsize=8.6, color=INK, zorder=4)
    for hh, d, lab in ((23, 1, "cutoff option Wed 23:00"), (12, 2, "cutoff option Thu 12:00")):
        ax.axvline(t(d, hh), color="#d03b3b", linewidth=1.3, linestyle=(0, (4, 3)), zorder=2)
        ax.text(t(d, hh) - 0.5, -0.75, lab, ha="right", va="center", fontsize=8.5, color="#d03b3b")
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(lanes[::-1])
    ax.set_xticks([t(d, h) for d in range(4) for h in (0, 12)])
    ax.set_xticklabels([f"{h:02d}:00" for d in range(4) for h in (0, 12)], fontsize=8, color=INK2)
    ax.set_xlim(t(0, 6), t(3, 24))
    ax.set_ylim(-1.1, len(lanes) + 0.2)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Patch(facecolor=COL[k][0], edgecolor=COL[k][1], label=l) for k, l in
                       (("same", "in the draft timeline"), ("replaced", "in the draft, method changed by the review"),
                        ("new", "new step from the review"))],
              loc="upper left", bbox_to_anchor=(0.0, 1.14), frameon=False, ncol=3, fontsize=9)
    save(fig, "timeline.png")


# ------------------------------------------------------------------ Figure 9: what each outcome means

def fig_outcomes():
    fig, ax = canvas(15.0, 6.4)
    good, bad = ("#e8f5e9", "#0ca30c"), ("#fbe9e7", "#d03b3b")
    COL["good"], COL["bad"], COL["neutral"] = good, bad, ("#ffffff", "#607d8b")
    box(ax, 0.1, 2.65, 3.0, 1.25, "Seed 42: apply the rule",
        "every candidate on A1 and A0\nagainst the development bar", "neutral")
    box(ax, 4.0, 4.35, 4.4, 1.45, "No candidate clears the bar",
        "no test is built (seeds 49 to 51 stay\nunused); Friday: choose between\ndesign L and other options", "bad")
    box(ax, 4.0, 0.6, 4.4, 1.45, "One or more clear",
        "carry one: best A1 candidate, A0 only\nif no A1 one clears; then the test on\nseeds 49 to 51, pooled", "neutral")
    box(ax, 9.3, 4.1, 5.6, 1.7, "GO: every lower bound above 0",
        "R@1 against cosine, RCA, B, B′ and the matched\ncounterpart, and the gain. Licensed claim: the reader\n"
        "beats its control on new episodes of the same 6,451\npaintings, pooled over the three aspect pairs",
        "good")
    box(ax, 9.3, 2.25, 5.6, 1.3, "NO-GO with a pooled point above 0",
        "inconclusive at the projected detectable margin\n(Figure 7); not evidence that the reader fails",
        "neutral")
    box(ax, 9.3, 0.35, 5.6, 1.3, "NO-GO with a pooled point at or below 0",
        "the reader fix did not hold on fresh episodes;\nFriday chooses the next route", "bad")
    arrow(ax, 3.1, 3.55, 4.0, 5.0, "none clears", dx=-0.35)
    arrow(ax, 3.1, 3.0, 4.0, 1.4, "one or more", dx=-0.35, dy=-0.35)
    arrow(ax, 8.4, 1.6, 9.3, 4.9)
    arrow(ax, 8.4, 1.35, 9.3, 2.9)
    arrow(ax, 8.4, 1.1, 9.3, 1.0)
    ax.text(0.1, 1.75, "If an A0 configuration\nis carried, a GO supports\nthe reader fix but leaves\n"
            "the CSD question open.", fontsize=9, color=INK2, va="top")
    ax.text(0.1, 6.35, "What each outcome would mean on Friday 9 October", fontsize=11, fontweight="bold", va="top",
            color=INK)
    save(fig, "outcomes.png")


if __name__ == "__main__":
    fig_pipeline()
    fig_comparators()
    fig_readers()
    fig_rc_counterpart()
    fig_ra_spread()
    fig_power()
    fig_timeline()
    fig_outcomes()
