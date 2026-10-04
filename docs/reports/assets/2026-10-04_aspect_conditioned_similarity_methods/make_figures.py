"""Reads only figure_data.json; writes the PNGs. No dashes in on-figure text."""
import json, os, numpy as np
OUT = os.path.dirname(os.path.abspath(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

D = json.load(open(os.path.join(OUT, "figure_data.json")))
PTS = {p["label"]: p for p in D["points"]}
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
TEAL, ORANGE, GREY = "#1baf7a", "#eb6834", "#b9b8b0"
FAM = {"baseline": ("#898781", "Raw baselines (E1)"), "factor": ("#2a78d6", "Factor methods (A3, A prime, N1)"),
       "free": ("#4a3aa7", "Condition-free centered scores"), "label": ("#1baf7a", "Label diagnostics (hollow)"),
       "partition": ("#eb6834", "Partition heads (N6, N6c)"), "mllm": ("#e87ba4", "MLLM probes (hollow, other episodes)")}
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": MUTED, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
S = lambda v: f"{v:.2f}".replace("-", "-")  # plain hyphen-minus for negatives

def box(ax, x, y, w, h, fc, ec=None, text="", fs=9, bold=False, tc=INK, ha="center", lw=1.0, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec or fc, lw=lw, ls=ls, zorder=z))
    if text:
        tx = x + w / 2 if ha == "center" else x + 1.2
        ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=tc, zorder=z + 1)
def arrow(ax, p, q, color=INK2, lw=1.4, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, color=color, lw=lw, connectionstyle=f"arc3,rad={rad}", zorder=1))
def tint(c, a=0.18):
    c = np.array(matplotlib.colors.to_rgb(c)); return tuple(1 - a * (1 - c))

# ---------------------------------------------------------------- 1 schematic
def fig_schematic():
    fig, ax = plt.subplots(figsize=(15, 8.6)); ax.set_xlim(0, 160); ax.set_ylim(0, 92); ax.axis("off")
    ax.text(2, 89, "One aspect episode", fontsize=15, fontweight="bold")
    ax.text(2, 85, "Aspects A and B are two of emotion, style, genre. Each painting has a value for every aspect.", fontsize=10, color=INK2)
    box(ax, 2, 40, 26, 22, tint(GREY, .35), GREY, "Query image\n(a painting)\n\nA = a0\nB = b0", fs=10, bold=True)
    # supports
    ax.text(36, 79.5, "4 support pairs for aspect A", fontsize=11, fontweight="bold", color=TEAL)
    ax.text(36, 76, "image + caption, different paintings, one shared A value", fontsize=8.5, color=INK2)
    for i in range(4):
        box(ax, 36, 66.5 - i * 5.6, 52, 4.6, tint(TEAL, .25), TEAL, f"support {i+1}: image + caption, A = a{i+1}", fs=8.8, ha="left")
    ax.text(36, 44, "The four A values are all different and never a0.", fontsize=8.5, color=INK2)
    ax.text(36, 36.8, "4 contrast pairs for aspect B", fontsize=11, fontweight="bold", color=ORANGE)
    ax.text(36, 33.2, "image + caption, different paintings, one shared B value", fontsize=8.5, color=INK2)
    for i in range(4):
        box(ax, 36, 23.8 - i * 5.6, 52, 4.6, tint(ORANGE, .25), ORANGE, f"contrast {i+1}: image + caption, B = b{i+1}", fs=8.8, ha="left")
    ax.text(36, 3.2, "Again four different B values, never b0.", fontsize=8.5, color=INK2)
    # candidates
    ax.text(104, 79.5, "13 candidate captions", fontsize=11, fontweight="bold")
    rows = [("n", None)] * 4 + [("pA", 0)] + [("n", None)] * 4 + [("pB", 0)] + [("n", None)] * 3
    ni = 0
    for i, (k, _) in enumerate(rows):
        y = 73.5 - i * 5.2
        if k == "pA": box(ax, 104, y, 54, 4.2, TEAL, TEAL, "p_A: shares the query's A value (a0)", fs=8.8, bold=True, tc="white", ha="left")
        elif k == "pB": box(ax, 104, y, 54, 4.2, ORANGE, ORANGE, "p_B: shares the query's B value (b0)", fs=8.8, bold=True, tc="white", ha="left")
        else:
            ni += 1; box(ax, 104, y, 54, 4.2, tint(GREY, .4), GREY, f"negative {ni}: shares neither a0 nor b0", fs=8.5, ha="left", tc=INK2)
    ax.text(104, 4.2, "1 + 1 + 11 = 13 candidates", fontsize=8.5, color=INK2)
    arrow(ax, (28.5, 54), (35.5, 68)); arrow(ax, (28.5, 46), (35.5, 26)); arrow(ax, (88.5, 56), (103.5, 56), lw=1.8)
    ax.text(89.5, 59, "score\ncandidates", fontsize=8.5, color=INK2, ha="left")
    # swap banner
    box(ax, 2, 4, 31, 30, SURF, MUTED, "", ls="--")
    ax.text(3.2, 30, "Swap test", fontsize=10.5, fontweight="bold")
    ax.text(3.2, 25.5, "Swap the supports and the\ncontrasts: the episode now\nconditions on aspect B.", fontsize=8.8, va="top", color=INK2)
    ax.text(3.2, 14.5, "p_B must move to the top", fontsize=9.5, fontweight="bold", color=ORANGE, va="top")
    ax.text(3.2, 10.5, "A scorer that only measures\nsimilarity cannot do this.", fontsize=8.8, va="top", color=INK2)
    # legend
    for i, (c, t) in enumerate([(TEAL, "aspect A"), (ORANGE, "aspect B"), (GREY, "negatives (neither)")]):
        ax.add_patch(Rectangle((2 + i * 22, 0.2), 3, 2, fc=c, ec="none")); ax.text(6 + i * 22, 1.2, t, fontsize=9, va="center", color=INK2)
    fig.savefig(os.path.join(OUT, "task_schematic.png"), dpi=170, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 2 chain
def fig_chain():
    steps = [("Task redefined", "Aspect episodes replace value episodes.\nLabel ceiling 23.09 vs CLIP 11.13 (spike)", "ctx"),
             ("E1: nine pair-metric baselines", "Best R@1: RCA 13.38 (gain 0.10)\nCLIP cosine 12.96. RCA sets the GO bar", "ctx"),
             ("E3: factor basis + agreement rule", "NO-GO: 13.76 vs its uniform control 16.72\n(seed 43)", "fail"),
             ("A prime: nested score", "R@1 16.52 vs control 16.55\nPre-registered branch 3", "fail"),
             ("MLLM in context", "Qwen3-VL-8B: R@1 +1.07, gain +0.21\nNot working by the pre-registered rule", "fail"),
             ("D0: label probes", "Inferred labels recover 13.33 of the\n21.06 gain that told labels give", "pos"),
             ("N1: centered rule", "Fresh-seed NO-GO\n-1.15 R@1 vs the matched control", "fail"),
             ("N6: partition heads", "Term-only gain 4.41\nNested R@1 +0.23 [-0.01, 0.47] vs control", "pos"),
             ("N6c: gate", "+0.15 [-0.06, 0.38] R@1 vs matched control\nGate not passed", "fail")]
    col = {"ctx": (GREY, "#e9e8e2"), "fail": (ORANGE, tint(ORANGE, .2)), "pos": (TEAL, tint(TEAL, .22))}
    fig, ax = plt.subplots(figsize=(15, 8)); ax.set_xlim(0, 150); ax.set_ylim(-6, 84); ax.axis("off")
    ax.text(1, 81, "Three days of experiments, in order (2026-10-02 to 2026-10-04)", fontsize=14, fontweight="bold")
    W, H = 44, 17
    pos = []
    for i, (t, r, k) in enumerate(steps):
        row, c = divmod(i, 3); x = 2 + c * 49; y = 56 - row * 26
        pos.append((x, y)); ec, fc = col[k]
        box(ax, x, y, W, H, fc, ec, lw=1.8)
        ax.text(x + 1.8, y + H - 3.2, f"{i+1}. {t}", fontsize=10.5, fontweight="bold", va="center")
        ax.text(x + 1.8, y + 6.2, r, fontsize=9.3, va="center", color=INK2, linespacing=1.4)
    for i in range(8):
        (x, y), (x2, y2) = pos[i], pos[i + 1]
        if (i + 1) % 3: arrow(ax, (x + W + .3, y + H / 2), (x2 - .3, y2 + H / 2))
        else: arrow(ax, (x + W / 2, y - .3), (x2 + W / 2, y2 + H + .3), rad=0.0) if False else arrow(ax, (x + W - 6, y - .3), (x2 + 6, y2 + H + .3), rad=-0.0)
    for j, (c, t) in enumerate([("ctx", "context or baseline"), ("fail", "failed its GO or gate"), ("pos", "new positive finding")]):
        ax.add_patch(Rectangle((2 + j * 38, -4.5), 4, 3, fc=col[c][1], ec=col[c][0], lw=1.6)); ax.text(7.5 + j * 38, -3, t, fontsize=9.5, va="center", color=INK2)
    fig.savefig(os.path.join(OUT, "chain.png"), dpi=170, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 3 frontier
OFF = {"x":0}
OFF.update({"cosine": (-34, 26, "right"), "rca": (20, 50, "left"), "wang": (-30, 45, "right"), "A3 (E3 score)": (-6, 12, "right"),
       "A3 agree term": (7, -3, "left"), "A' nested on A3": (-90, 45, "right"), "N1 nested on A3": (-20, 90, "right"),
       "A3 matched control": (0, 60, "center"), "C2 (N6c matched)": (25, 45, "left"), "N6 T6 term": (8, 3, "left"),
       "N6 nested": (-9, 7, "right"), "N6c": (7, 6, "left"), "D0 told": (-10, -3, "right"), "Qwen3-VL-8B": (30, -40, "left")})
SHORT = {"cosine": "cosine", "rca": "RCA (GO bar)", "wang": "Wang (best gain)", "A3 (E3 score)": "A3, E3 score", "A3 agree term": "A3 agree term",
         "A' nested on A3": "A prime nested", "N1 nested on A3": "N1 nested A3", "A3 matched control": "matched control 17.94",
         "C2 (N6c matched)": "C2 18.34", "N6 T6 term": "N6 T6 term", "N6 nested": "N6 nested", "N6c": "N6c", "D0 told": "D0 told",
         "Qwen3-VL-8B": "Qwen3-VL-8B (seed 46)"}
FAN = {}
def make_fan():
    grpA = sorted([l for l, p in PTS.items() if p["either"] and p.get("frontier") is not False and l not in SHORT and 25 < p["either"] < 27 and p["gain"] < .2], key=lambda l: PTS[l]["either"])
    grpB = sorted([l for l, p in PTS.items() if p["either"] and p.get("frontier") is not False and l not in SHORT and p["either"] > 32 and abs(p["gain"]) < .05], key=lambda l: PTS[l]["either"])
    for grp, x0, dx in ((grpA, 24.5, .42), (grpB, 31.9, .55)):
        for i, l in enumerate(grp): FAN[l] = (x0 + i * dx, -0.36 - 0.1 * (i % 2))
make_fan()
def draw_pts(ax, numbered, zoom):
    for lab, p in PTS.items():
        if p["either"] is None or p.get("frontier") is False: continue
        c = FAM[p["family"]][0]; ctrl = p["kind"] == "control"
        sz = 28 if p["family"] == "baseline" else 85
        mk = "s" if ctrl else ("D" if p["family"] == "mllm" else "o")
        ax.scatter(p["either"], p["gain"], s=sz, marker=mk, facecolors="none" if p["hollow"] else c, edgecolors=c, linewidths=1.8 if p["hollow"] else 0.8, zorder=4)
        if zoom:
            if lab in SHORT:
                dx, dy, ha = OFF[lab]
                ax.annotate(SHORT[lab], (p["either"], p["gain"]), (dx, dy), textcoords="offset points", ha=ha, fontsize=8.8, color=INK,
                            arrowprops=dict(arrowstyle="-", color=MUTED, lw=.7), zorder=6)
            elif lab in FAN:
                tx, ty = FAN[lab]
                ax.annotate(str(numbered[lab]), (p["either"], p["gain"]), (tx, ty), textcoords="data", fontsize=8, color=INK2, ha="center", va="center",
                            arrowprops=dict(arrowstyle="-", color=GRID if False else MUTED, lw=.5), zorder=6)
            elif lab in numbered:
                ax.annotate(str(numbered[lab]), (p["either"], p["gain"]), (4, 3), textcoords="offset points", fontsize=8, color=INK2, zorder=6)
def fig_frontier():
    numbered = {}
    for lab, p in PTS.items():
        if p["either"] is not None and p.get("frontier") is not False and lab not in SHORT: numbered[lab] = len(numbered) + 1
    fig = plt.figure(figsize=(17, 8.6)); gs = fig.add_gridspec(1, 3, width_ratios=[2.4, 6.6, 3.0], wspace=0.14)
    a0, a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2]); a2.axis("off")
    for ax in (a0, a1): ax.grid(color=GRID, lw=.6, zorder=0); ax.set_axisbelow(True)
    draw_pts(a0, numbered, False)
    a0.set_xlim(19.5, 42); a0.set_ylim(-1.5, 23); a0.set_xlabel("either rate (pp)"); a0.set_ylabel("condition gain (pp)")
    a0.set_title("Full range", fontsize=10.5, loc="left", color=INK2)
    a0.add_patch(Rectangle((20.2, -0.55), 38.6 - 20.2, 5.25, fill=False, ec=INK2, lw=1.1, ls="--", zorder=5))
    a0.annotate("D0 told\n(40.27, 21.06)", (PTS["D0 told"]["either"], PTS["D0 told"]["gain"]), (-8, -4), textcoords="offset points", ha="right", fontsize=8.5)
    a0.text(20.4, 5.4, "zoom area", fontsize=8.5, color=INK2)
    # zoom
    xs = np.linspace(21, 38.5, 50)
    for r, name in [(12.96, "cosine R@1 12.96"), (16.55, "E3 control R@1 16.55"), (18.34, "best condition-free R@1 18.34")]:
        a1.plot(xs, 2 * r - xs, color=MUTED, lw=1.1, ls=":", zorder=1)
    a1.text(22.0, 2 * 12.96 - 22.0 + .1, "R@1 = 12.96 (cosine)", fontsize=8.5, color=INK2, rotation=-31, rotation_mode="anchor", va="bottom")
    a1.text(29.3, 2 * 16.55 - 29.3 + .15, "R@1 = 16.55 (E3 control)", fontsize=8.5, color=INK2, rotation=-31, rotation_mode="anchor", va="bottom")
    a1.text(33.3, 2 * 18.34 - 33.3 + .15, "R@1 = 18.34 (best condition-free)", fontsize=8.5, color=INK2, rotation=-31, rotation_mode="anchor", va="bottom")
    draw_pts(a1, numbered, True)
    a1.set_xlim(20.2, 38.6); a1.set_ylim(-0.55, 4.7); a1.set_xlabel("either rate (pp): share of episodes with p_A or p_B ranked first"); a1.set_ylabel("condition gain (pp)")
    a1.set_title("Zoom on the dense region (seed-42 development episodes)", fontsize=10.5, loc="left", color=INK2)
    a1.axhline(0, color=MUTED, lw=.8)
    # key
    h = [Line2D([], [], marker="o", ls="", mfc=FAM[k][0], mec=FAM[k][0], ms=7) if k not in ("label", "mllm") else
         Line2D([], [], marker="o" if k == "label" else "D", ls="", mfc="none", mec=FAM[k][0], mew=1.8, ms=7) for k in FAM]
    h += [Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, ms=7), Line2D([], [], marker="s", ls="", mfc=INK2, mec=INK2, ms=7)]
    a2.legend(h, [FAM[k][1] for k in FAM] + ["condition-aware score (circle)", "condition-free control (square)"], loc="upper left", fontsize=8.8, frameon=False, bbox_to_anchor=(0, 1.0))
    key = "Numbered points\n" + "\n".join(f"{n:>2}  {l}  ({PTS[l]['either']:.1f}, {PTS[l]['gain']:.2f})" for l, n in numbered.items())
    a2.text(0, 0.60, key, fontsize=7.9, va="top", family="DejaVu Sans Mono", color=INK2, transform=a2.transAxes)
    a2.text(0, 0.02, "Qwen3-VL-8B is on seed-46 episodes (cosine there: either 26.28),\nnot comparable one to one. The 2B model has no absolute\neither rate and is omitted here.", fontsize=8, color=INK2, transform=a2.transAxes)
    fig.suptitle("Frontier: condition gain against either rate, one point per scorer", x=0.07, ha="left", fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(os.path.join(OUT, "frontier.png"), dpi=170, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 4 gain progression
def fig_gain():
    G = [("Raw baselines (E1)", [("wang", "Wang metric", 0), ("probe", "Linear probe", 0)]),
         ("Factor rules, term only", [("C0", None, 0), ("SE", None, 0), ("A3 agree term", "A3 agreement term", 0), ("A3 N1 term", "A3 N1 term", 0)]),
         ("Label-trained factors (diagnostic)", [("L3 agree term", "L3 agreement term", 1), ("L3 N1 term", "L3 N1 term", 1)]),
         ("MLLMs (other episodes)", [("Qwen3-VL-2B v2", "Qwen3-VL-2B, seed 44", 1), ("Qwen3-VL-8B", "Qwen3-VL-8B, seed 46", 1)]),
         ("Partition heads, term only", [("N6 T6 term", "N6 T6", 0), ("N6 T6soft term", "N6 T6soft", 0)]),
         ("Label probes (diagnostic)", [("D0 hard", "D0 hard (inferred labels)", 1), ("D0 told", "D0 told (true labels)", 1)])]
    gcol = ["#898781", "#2a78d6", "#1baf7a", "#e87ba4", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(11.5, 8.2)); y = 0; yt, yl = [], []
    for gi, (gname, items) in enumerate(G):
        ax.text(-0.5, y + .15, gname, fontsize=10, fontweight="bold", color=INK, va="center", ha="right" if False else "left", transform=ax.get_yaxis_transform())
        y += 1
        for lab, nice, hatch in items:
            p = PTS[lab]; ci = p["gain_ci"]; nice = nice or lab
            ax.barh(y, p["gain"], height=.62, color=tint(gcol[gi], .55) if hatch else gcol[gi], edgecolor=gcol[gi], lw=1.2, hatch="///" if hatch else None, zorder=3)
            ax.plot(ci, [y, y], color=INK, lw=1.4, zorder=4); ax.plot([ci[0]] * 2, [y - .14, y + .14], color=INK, lw=1.4, zorder=4); ax.plot([ci[1]] * 2, [y - .14, y + .14], color=INK, lw=1.4, zorder=4)
            ax.text(max(ci[1], p["gain"]) + .35, y, f'{S(p["gain"])}  [{S(ci[0])}, {S(ci[1])}]', va="center", fontsize=9, color=INK2)
            yt.append(y); yl.append(nice); y += 1
        y += .6
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=9.5); ax.invert_yaxis(); ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlim(-2.2, 27); ax.set_xlabel("condition gain (pp), 95% cluster-bootstrap CI"); ax.grid(axis="x", color=GRID, lw=.6, zorder=0)
    ax.set_title("Condition gain by method family (seed-42 development episodes unless noted)", fontsize=12.5, fontweight="bold", loc="left")
    ax.tick_params(axis="y", length=0)
    for sp in ("left",): ax.spines[sp].set_visible(False)
    fig.text(0.01, 0.005, "Hatched = diagnostic (uses labels). MLLM bars are on different episodes (2B seed 44, 8B seed 46). Term-only: the term scored alone.", fontsize=8.5, color=INK2)
    fig.savefig(os.path.join(OUT, "gain_progression.png"), dpi=170, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 5 margins
def fig_margins():
    M = D["margins"]
    names = ["A prime nested A3\n(seed 42)", "N1 nested A3\n(seed 42)", "N1 nested A3\n(fresh seeds, pooled)", "N6 nested\n(seed 42)", "N6c\n(seed 42)"]
    keyf = lambda m: (m["label"], m["control"])
    order = [("A' nested A3", "declared control"), ("N1 nested A3", "declared control"), ("N1 nested A3", "matched control"),
             ("N1 nested A3, fresh seeds", "declared control"), ("N1 nested A3, fresh seeds", "matched control"),
             ("N6 nested", "declared control"), ("N6c", "declared control (C1)"), ("N6c", "matched control (C2)")]
    rows = {keyf(m): m for m in M}
    ylab = {0: "A prime nested A3 (seed 42)", 1: "N1 nested A3 (seed 42)", 2: "N1 nested A3 (fresh seeds, pooled)", 3: "N6 nested (seed 42)", 4: "N6c (seed 42)"}
    ypos = {("A' nested A3", "declared control"): 0, ("N1 nested A3", "declared control"): 1, ("N1 nested A3", "matched control"): 1,
            ("N1 nested A3, fresh seeds", "declared control"): 2, ("N1 nested A3, fresh seeds", "matched control"): 2,
            ("N6 nested", "declared control"): 3, ("N6c", "declared control (C1)"): 4, ("N6c", "matched control (C2)"): 4}
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True, gridspec_kw={"wspace": 0.06})
    for ai, (ax, kd, kc, ttl) in enumerate([(axs[0], "d_r1", "d_r1_ci", "R@1 minus control (pp)"), (axs[1], "d_gain", "d_gain_ci", "Condition gain minus control (pp)")]):
        for k in order:
            m = rows[k]; y = ypos[k]; matched = "matched" in k[1]; off = .17 if matched else -.17
            if ypos[k] in (0, 3): off = 0
            c = "#4a3aa7" if matched else "#2a78d6"
            ax.errorbar(m[kd], y + off, xerr=[[m[kd] - m[kc][0]], [m[kc][1] - m[kd]]], fmt="o" if not matched else "s", color=c, ecolor=c, capsize=3.5, ms=7, lw=1.6, zorder=4)
            ax.text(m[kc][1] + (.04 if ai == 0 else .05), y + off, f'{m[kd]:+.2f} [{m[kc][0]:+.2f}, {m[kc][1]:+.2f}]', fontsize=8.3, va="center", color=INK2)
        ax.axvline(0, color=INK, lw=1); ax.grid(axis="x", color=GRID, lw=.6); ax.set_xlabel(ttl); ax.set_axisbelow(True)
        ax.set_ylim(4.6, -.6); ax.set_title(ttl.split(" (")[0], fontsize=11, loc="left", fontweight="bold")
    axs[0].set_xlim(-1.75, 1.7); axs[1].set_xlim(-.4, 2.9)
    axs[0].set_yticks(range(5)); axs[0].set_yticklabels([ylab[i] for i in range(5)], fontsize=9.5); axs[0].tick_params(axis="y", length=0)
    fig.legend([Line2D([], [], marker="o", ls="", color="#2a78d6", ms=7), Line2D([], [], marker="s", ls="", color="#4a3aa7", ms=7)],
                  ["vs declared control (circle)", "vs matched control (square)"], loc="lower center", ncol=2, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("The wall: fused methods against their condition-free controls, 95% cluster-bootstrap CIs", x=0.06, ha="left", fontsize=13, fontweight="bold", y=1.0)
    fig.savefig(os.path.join(OUT, "margins.png"), dpi=170, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 6 per pair
def fig_pair():
    H = D["heat"]; Z = np.array([h["gain"] for h in H]); pairs = H[0]["pairs"]
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    cm = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f1f6fd", "#9ec5f4", "#2a78d6", "#0d366b"])
    im = ax.imshow(Z, cmap=cm, aspect="auto", vmin=0, vmax=Z.max())
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            v = Z[i, j]; ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10.5, color="white" if v > Z.max() * .5 else INK, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(pairs); ax.set_yticks(range(len(H))); ax.set_yticklabels([h["label"] for h in H])
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    for yy in (1.5, 3.5): ax.axhline(yy, color=SURF, lw=3)
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.03); cb.set_label("condition gain (pp)", color=INK2); cb.outline.set_visible(False)
    ax.set_title("Condition gain by aspect pair (seed 42, 4,096 episodes per pair)", fontsize=12, fontweight="bold", loc="left", pad=34)
    fig.text(.01, -.01, "D0 rows are label diagnostics. Term rows score the term alone; N6 nested and N6c are fused scores.", fontsize=8.3, color=INK2)
    fig.savefig(os.path.join(OUT, "per_pair.png"), dpi=170, bbox_inches="tight"); plt.close(fig)

for f in (fig_schematic, fig_chain, fig_frontier, fig_gain, fig_margins, fig_pair): f()
