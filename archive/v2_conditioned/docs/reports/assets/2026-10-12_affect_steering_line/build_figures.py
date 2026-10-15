"""Build the figures for the affect steering stage report (2026-10-12).
Run: PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python build_figures.py
Numbers come from the JSON files of round 3 (fresh) and round 6 (held); the chain and pipeline text is hand-typed.
"""
import json, textwrap, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D

ROOT = "/project/CoSiR"
OUT = os.path.dirname(os.path.abspath(__file__))
HELD = json.load(open(f"{ROOT}/src/test/20261125_artelingo_held_test/results/descriptive.json"))
FRESH = json.load(open(f"{ROOT}/src/test/20261121_round3_affect_gate/results/descriptive.json"))
DPI = 200
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white"})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C = dict(AFF="#2a78d6", R1="#eb6834", CTRL="#1baf7a", COS="#555555", RCA="#7b5ea7", B="#8a8984",
         B0="#b3b1ab", B1="#cfcdc7", MLLM="#c4576b", DTS="#d9a21b", FT="#3d9fb0")
FD = {}  # figure_data


def sg(v, d=2):
    """Signed number with a Unicode minus."""
    return f"{v:+.{d}f}".replace("-", "\u2212")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- 1 chain
def chain():
    steps = [
        ("Jun to Sep", "Earlier CoSiR lines (conditional buddies, topic prototypes)", "diag"),
        ("28 Sep", "CoSiR v2: conditioned scores on frozen CLIP features", "ok"),
        ("30 Sep to 1 Oct", "Shared factors on value episodes: pass, but the examples give the answer away", "no"),
        ("2 Oct", "Reframe: aspect episodes, condition gain, matched control", "ok"),
        ("2 to 3 Oct", "Baselines: metric from pairs, in-context MLLM, trained factors (method A): none reads the aspect", "no"),
        ("4 Oct", "Label-free groupings and heads; a reader reads the condition, but fusion loses", "diag"),
        ("5 Oct", "Affect grouping: Leiden communities of GoEmotions caption scores", "ok"),
        ("6 Oct", "Learned reader with a confidence gate: +0.44, short of the +0.5 bar", "no"),
        ("6 Oct", "One-sided affect steering (AFF) found: +0.70 on development episodes", "ok"),
        ("6 Oct", "Round 3, fresh test: GO, +0.59 R@1", "ok"),
        ("7 Oct", "Round 4, vetoes on the gate: none beats AFF", "no"),
        ("7 Oct", "Round 5, GoEmotions placement of captions: below the bar", "no"),
        ("7 Oct", "Fine-tuned CLIP as a comparator: 15 R@1, far below AFF", "diag"),
        ("10 Oct", "Round 6, held paintings: GO, +0.60 R@1", "ok"),
        ("12 Oct", "Round 7, label-free style groupings: NO-GO", "no"),
    ]
    ncol, nrow = 5, 3
    FH = 7.6
    fig, ax = plt.subplots(figsize=(17, FH))
    ax.set_xlim(0, ncol); ax.set_ylim(0, nrow); ax.axis("off")
    bw = 0.87
    nl = max(len(textwrap.wrap(t, 22)) for _, t, _ in steps)
    bh = (0.42 + nl * 0.215 + 0.08) / (FH * 0.86 / nrow)  # box height in data units, sized to the longest text
    pos = []
    for i in range(len(steps)):
        r, c = divmod(i, ncol)
        if r % 2 == 1:
            c = ncol - 1 - c
        pos.append((c + 0.5, nrow - r - 0.5))
    for i, ((d, t, k), (x, y)) in enumerate(zip(steps, pos)):
        if k == "ok":
            fc, ec, tc, lw = "#1d8f78", "#1d8f78", "white", 2
        elif k == "no":
            fc, ec, tc, lw = "white", "#e07b39", INK, 3.5
        else:
            fc, ec, tc, lw = "white", "#5b6b82", INK, 3.5
        ax.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh, boxstyle="round,pad=0.0,rounding_size=0.05",
                                    fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(x - bw / 2 + 0.04, y + bh / 2 - 0.07 / (FH * 0.86 / nrow), f"{i + 1}. {d}", fontsize=13, fontweight="bold", color=tc,
                va="top", ha="left", zorder=3)
        ax.text(x - bw / 2 + 0.04, y + bh / 2 - 0.42 / (FH * 0.86 / nrow), textwrap.fill(t, 22), fontsize=12, color=tc, va="top",
                ha="left", zorder=3, linespacing=1.25)
        if i + 1 < len(steps):
            x2, y2 = pos[i + 1]
            if y2 == y:
                s = 1 if x2 > x else -1
                a, b = (x + s * bw / 2, y), (x2 - s * bw / 2, y2)
            else:
                a, b = (x, y - bh / 2), (x2, y2 + bh / 2)
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=22, lw=2.2, color=INK2, zorder=1,
                                         shrinkA=3, shrinkB=3))
    ax.set_title("From a fixed score to one-sided affect steering, 28 September to 12 October", fontsize=16,
                 loc="left", pad=14)
    leg = [Patch(fc="#1d8f78", ec="#1d8f78", label="worked"),
           Patch(fc="white", ec="#e07b39", lw=3, label="did not work"),
           Patch(fc="white", ec="#5b6b82", lw=3, label="comparator or diagnostic")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=13, frameon=False)
    FD["chain"] = [dict(n=i + 1, date=d, text=t, outcome=k) for i, (d, t, k) in enumerate(steps)]
    save(fig, "chain.png")


# --------------------------------------------------------------------------- 2 pipeline
def pipeline():
    GREY, GFILL, BFILL = "#8a8984", "#f1f0ee", "#e6f0fb"
    LH, HEAD = 0.225, 0.55  # line height and title block, in inches (= data units; 1 unit = 1 inch)
    W, H = 19.0, 7.0
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    def paras(body, w):
        out = []
        for t, bold in body:
            out.append((textwrap.fill(t, int(w * 8.8)), bold))
        return out

    def need(body, w):
        n = sum(p.count("\n") + 1 for p, _ in paras(body, w))
        return HEAD + n * LH + 0.3

    def box(x, y, w, h, title, body, ec, fc, tcol):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=ec, lw=2.5,
                                    zorder=2))
        ax.text(x + 0.15, y + h - 0.12, title, fontsize=13, fontweight="bold", color=tcol, va="top", ha="left",
                zorder=3)
        yy = y + h - HEAD
        for t, bold in paras(body, w):
            ax.text(x + 0.15, yy, t, fontsize=11.5, color=INK, va="top", ha="left", zorder=3, linespacing=1.25,
                    fontweight="bold" if bold else "normal")
            yy -= (t.count("\n") + 1) * LH + 0.06

    def arrow(a, b, col, **kw):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=24, lw=2.5, color=col, zorder=1,
                                     shrinkA=0, shrinkB=0, **kw))

    inp = [("Query, 4 example pairs, 4 contrast pairs, 13 candidates (frozen CLIP ViT-B/32 features)", False)]
    bsc = [("Cosine plus a factor term centred on the example items, plus grouping heads, averaged. "
            "Uses the examples, ignores which aspect they show.", False)]
    texts = [
        ("Three groupings", "Label-free groupings of the training rows. Affect: Leiden communities of GoEmotions "
         "scores of the captions, 41 groups. Image and caption: k-means on CLIP, 64 groups each."),
        ("Heads", "Place any image or caption in each grouping (logistic regression on CLIP)."),
        ("Reader", "From how the example pairs agree against the contrast pairs, gives a probability per grouping. "
         "Trained on practice episodes built from the groupings, no evaluation labels."),
        ("Gate", "Open only when the reader is confident and its top grouping is affect."),
        ("Grouping score T", "The reader's probability-weighted match of query and candidate."),
    ]
    w, g = 2.6, 0.22
    fx, fw = 14.45, 4.45
    fused = [("(1 + \u03bbu)\u00b7z(B) + \u03bba\u00b7gate\u00b7z(T)", True), ("gate closed: B alone", False),
             ("Weights chosen on one half of the episodes, applied to the other.", False)]
    th = max(need(inp, 4.6), need(bsc, 8.2))
    bh = max(max(need([(b, False)], w) for _, b in texts), need(fused, fw))
    bus_gap = 1.1
    by = 0.85
    ty = by + bh + bus_gap
    ax.set_ylim(0, ty + th + 0.35)
    fig.set_size_inches(W, ty + th + 0.35)
    xs = [0.1 + i * (w + g) for i in range(5)]
    box(0.1, ty, 4.6, th, "Input", inp, GREY, GFILL, INK)
    box(5.3, ty, 8.2, th, "Condition-free score B", bsc, GREY, GFILL, INK)
    arrow((4.7, ty + th / 2), (5.3, ty + th / 2), GREY)
    for x, (t, b) in zip(xs, texts):
        box(x, by, w, bh, t, [(b, False)], C["AFF"], BFILL, C["AFF"])
    for i in range(4):
        arrow((xs[i] + w, by + bh / 2), (xs[i + 1], by + bh / 2), C["AFF"])
    # Input down into the steering lane: a bus over groupings, heads and reader
    busy = by + bh + bus_gap / 2
    ix = 0.1 + 2.3
    ax.plot([ix, ix], [ty, busy], color=GREY, lw=2.5, zorder=1)
    ax.plot([xs[0] + w / 2, xs[2] + w / 2], [busy] * 2, color=GREY, lw=2.5, zorder=1)
    for i in range(3):
        arrow((xs[i] + w / 2, busy), (xs[i] + w / 2, by + bh), GREY)
    ax.text(xs[2] + w / 2 + 0.3, busy + 0.06, "same query, example pairs and candidates", fontsize=11.5, color=INK2,
            va="bottom", ha="left")
    # fused
    box(fx, by, fw, bh, "Fused score", fused, INK, "white", INK)
    ax.plot([13.5, fx + 1.3], [ty + th / 2] * 2, color=GREY, lw=2.5, zorder=1)
    arrow((fx + 1.3, ty + th / 2), (fx + 1.3, by + bh), GREY)
    arrow((xs[4] + w, by + bh / 2), (fx, by + bh / 2), C["AFF"])
    ax.set_title("One-sided affect steering: a condition-free score plus a gated affect term", fontsize=16,
                 loc="left", pad=10)
    ax.text(0.1, 0.3, "Grey = condition-free base, always on.   Blue = steering branch, added only when the gate opens.",
            fontsize=13, color=INK, va="center")
    FD["aff_pipeline"] = dict(note="hand-typed diagram; no data")
    save(fig, "aff_pipeline.png")


# --------------------------------------------------------------------------- 3 held comparison
ROWS = [  # key, label, colour, group
    ("cosine", "cosine (CLIP)", C["COS"], 0),
    ("rca", "RCA (metric from the example pairs)", C["RCA"], 0),
    ("DTS", "describe-then-score (DTS)", C["DTS"], 0),
    ("MLLM", "in-context MLLM †", C["MLLM"], 0),
    ("FT-LP", "fine-tuned CLIP, linear probe", C["FT"], 0),
    ("FT-LB", "fine-tuned CLIP, last block", C["FT"], 0),
    ("FT-LoRA", "fine-tuned CLIP, LoRA", C["FT"], 0),
    ("B", "B", C["B"], 1),
    ("B0", "B′(A0)", C["B0"], 1),
    ("aff_cf", "matched control of AFF", C["CTRL"], 1),
    ("B1", "B′(A1)", C["B1"], 1),
    ("r1_fused", "R1 (gated reader)", C["R1"], 2),
    ("aff_fused", "AFF", C["AFF"], 2),
]
GROUPS = ["External baselines", "Condition-free scores (ours)", "Reading the condition (ours)"]


def val(key, what):
    p = HELD["rows"][key]["pooled"][what]
    return p["point"], p.get("ci95")


def held_comparison():
    n = len(ROWS)
    # y positions with gaps between groups
    ys, y, prev = [], 0.0, 0
    for _, _, _, g in ROWS:
        if g != prev:
            y += 0.7
            prev = g
        ys.append(y)
        y += 1
    fig, axs = plt.subplots(1, 2, figsize=(15, 9.5), sharey=True, gridspec_kw=dict(wspace=0.06))
    data = []
    for ax, what, xl in zip(axs, ["r1", "gain"], ["R@1 (%)", "Condition gain (points)"]):
        aff = val("aff_fused", what)[0]
        for (k, lab, col, g), yy in zip(ROWS, ys):
            v, ci = val(k, what)
            has_ci = ci is not None and not (ci[0] == ci[1])
            ax.barh(yy, v, height=0.68, color=col, zorder=3, hatch="//" if k == "MLLM" and False else None)
            if has_ci:
                ax.errorbar(v, yy, xerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor=INK, elinewidth=1.6,
                            capsize=3.5, zorder=4)
            xr = ci[1] if has_ci else v
            ax.text(xr + (0.012 if what == "r1" else 0.012) * ax.get_xlim()[1] if False else xr, yy,
                    f"  {v:.2f}", va="center", ha="left", fontsize=12, color=INK, zorder=5)
            data.append(dict(panel=what, scorer=k, label=lab, value=v, ci95=ci if has_ci else None))
        ax.axvline(aff, color=C["AFF"], ls="--", lw=1.8, zorder=2)
        ax.set_xlabel(xl, fontsize=13)
        ax.tick_params(axis="x", labelsize=12)
        ax.grid(axis="x", color=GRID, zorder=0)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axs[0].set_xlim(0, 25)
    axs[1].set_xlim(0, 4.3)
    axs[0].set_yticks(ys)
    axs[0].set_yticklabels([r[1] for r in ROWS], fontsize=12.5)
    axs[0].set_ylim(ys[-1] + 0.8, -1.25)
    # group bands
    gy = {}
    for (k, l, c, g), yy in zip(ROWS, ys):
        gy.setdefault(g, []).append(yy)
    for g, lst in gy.items():
        lo, hi = min(lst) - 0.5, max(lst) + 0.5
        for ax in axs:
            ax.axhspan(lo, hi, color="#f4f3f0" if g % 2 == 0 else "#eaf1fa", zorder=0, lw=0)
        axs[0].text(0.01, lo - 0.04, GROUPS[g], transform=axs[0].get_yaxis_transform(), fontsize=12, style="italic",
                    color=INK2, va="bottom", ha="left", zorder=6)
    axs[0].set_title("R@1 (%)", fontsize=14, loc="left")
    axs[1].set_title("Condition gain (points)", fontsize=14, loc="left")
    axs[0].tick_params(axis="y", length=0)
    fig.suptitle("On held paintings AFF is far above every external baseline and about 0.6 R@1 points above\n"
                 "the pre-registered condition-free bars", fontsize=16, x=0.02, ha="left", y=0.99)
    fig.text(0.02, -0.005,
             "36,864 held episodes (seeds 52 to 54; MLLM: seed 52, 12,288 episodes). "
             "Condition-free scores have a condition gain of exactly 0.\n"
             "† Seed 52 only. Bars show 95% intervals where available. Dashed line: AFF.",
             fontsize=12, color=INK2, ha="left", va="top")
    FD["held_comparison"] = data
    save(fig, "held_comparison.png")


# --------------------------------------------------------------------------- 4 frontier
def frontier():
    pts = {}
    for k, lab, col, g in ROWS:
        r1, _ = val(k, "r1"); gn, _ = val(k, "gain")
        pts[k] = (gn, 2 * r1 - gn, r1)
    dn = ("DTS-N",) + tuple(val("DTS-N", w)[0] for w in ["gain", "r1"])
    r1n, gn = val("DTS-N", "r1")[0], val("DTS-N", "gain")[0]
    ptsn = (gn, 2 * r1n - gn, r1n)
    fig, ax = plt.subplots(figsize=(13, 9))
    xmax = 4.1
    for c in [13, 15, 17, 19, 21]:
        ax.plot([-1.2, xmax], [2 * c + 1.2, 2 * c - xmax], ls="--", color="#b5b4b0", lw=1.4, zorder=1)
        ax.text(xmax + 0.03, 2 * c - xmax, f"R@1 = {c}", fontsize=12, color=INK2, va="center", ha="left")
    # label offsets (points): dx, dy, ha
    off = {
        "cosine": (-14, -6, "right"), "rca": (14, 12, "left"), "DTS": (-14, -10, "right"),
        "MLLM": (16, 4, "left"), "FT-LP": (-14, 0, "right"), "FT-LB": (-14, -12, "right"), "FT-LoRA": (-14, 12, "right"),
        "B": (18, -30, "left"), "B0": (18, -2, "left"), "aff_cf": (-18, -8, "right"), "B1": (18, 8, "left"),
        "r1_fused": (-14, -22, "right"), "aff_fused": (16, 14, "left"),
    }
    names = {k: l.replace(" †", "") for k, l, _, _ in ROWS}
    names.update(cosine="cosine", rca="RCA", DTS="DTS", MLLM="MLLM (seed 52)", **{"FT-LP": "FT linear probe",
                 "FT-LB": "FT last block", "FT-LoRA": "FT LoRA"}, B0="B′(A0)", B1="B′(A1)",
                 aff_cf="matched control", r1_fused="R1", aff_fused="AFF")
    data = []
    for (k, lab, col, g) in ROWS:
        x, y, r1 = pts[k]
        ax.scatter(x, y, s={"B": 380, "B0": 260}.get(k, 170), color=col, zorder=4, edgecolor=INK if k in ("B1", "B0") else "white", lw=1.2)
        dx, dy, ha = off[k]
        ax.annotate(names[k], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=12.5, color=INK, ha=ha,
                    va="center", arrowprops=dict(arrowstyle="-", color="#999", lw=0.8, shrinkA=0, shrinkB=6))
        data.append(dict(scorer=k, gain=x, either=y, r1=r1))
    ax.scatter(ptsn[0], ptsn[1], s=190, facecolor="white", edgecolor=C["DTS"], lw=2.6, zorder=4)
    ax.annotate("DTS-N (told the true aspect name; privileged)", (ptsn[0], ptsn[1]), xytext=(16, -14),
                textcoords="offset points", fontsize=12.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color="#999", lw=0.8, shrinkB=6))
    data.append(dict(scorer="DTS-N", gain=ptsn[0], either=ptsn[1], r1=ptsn[2]))
    ax.set_xlim(-1.2, xmax); ax.set_ylim(21, 42); ax.set_yticks(range(22, 43, 2))
    ax.set_xlabel("Condition gain (points): choosing the shown aspect", fontsize=13)
    ax.set_ylabel("Either rate (%): finding a candidate that shares either aspect", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.grid(color=GRID, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("Condition-free scores find aspect candidates; only R1 and AFF also choose the shown one",
                 fontsize=15, loc="left", pad=12, wrap=True)
    fig.text(0.08, 0.005, "R@1 = (either rate + condition gain) / 2. Either rate: a candidate that shares either "
             "aspect comes first.\nHeld paintings, 36,864 episodes (MLLM: 12,288).", fontsize=12, color=INK2,
             ha="left", va="top")
    FD["frontier"] = data
    save(fig, "frontier.png")


# --------------------------------------------------------------------------- 5 per pair
def per_pair():
    fa = FRESH["item2_bar_margin_cells_frozen"]["AFF_bar_margin_pooled"]
    ha = HELD["bar_margin"]["AFF"]["pooled"]
    cats = [("emotion__style", "emotion × style"), ("emotion__genre", "emotion × genre"),
            ("style__genre", "style × genre"), (None, "pooled")]
    rows = []
    for key, lab in cats:
        f = fa["r1"] if key is None else fa["per_pair_r1"][key]
        h = ha["r1"] if key is None else ha["per_pair_r1"][key]
        rows.append((lab, f, h))
    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    h = 0.36
    data = []
    for i, (lab, f, hh) in enumerate(rows):
        for off, d, col, name in [(-h / 2, f, "#9cc3ec", "Fresh episodes (round 3, seeds 49 to 51)"),
                                  (h / 2, hh, C["AFF"], "Held paintings (round 6, seeds 52 to 54)")]:
            v, ci = d["point"], d["ci95"]
            ax.barh(i + off, v, height=h, color=col, zorder=3, label=name if i == 0 else None)
            ax.errorbar(v, i + off, xerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor=INK, elinewidth=1.6, capsize=3.5,
                        zorder=4)
            xt = ci[1] + 0.04 if v >= 0 else ci[0] - 0.04
            ax.text(xt, i + off, sg(v), va="center", ha="left" if v >= 0 else "right", fontsize=12.5)
            data.append(dict(group=lab, test=name, value=v, ci95=ci))
    ax.axvline(0, color=INK, lw=1.4, zorder=5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(-1.3, 2.3)
    ax.set_xlabel("AFF minus B′(A0), R@1 points (95% interval)", fontsize=13)
    ax.tick_params(axis="x", labelsize=12); ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, zorder=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=12, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.33))
    ax.set_title("AFF gains on the two emotion pairs and loses on style × genre, on fresh and held paintings alike",
                 fontsize=15, loc="left", pad=12, wrap=True)
    FD["per_pair_fresh_held"] = data
    save(fig, "per_pair_fresh_held.png")


# --------------------------------------------------------------------------- 6 random gate
def random_gate():
    i7 = FRESH["item7_random_share_control"]["draws"]
    r1 = FRESH["item3_R1_checks"]["bar_margin_pooled"]["r1"]
    aff = FRESH["item2_bar_margin_cells_frozen"]["AFF_bar_margin_pooled"]["r1"]
    bars = [("R1", r1, C["R1"], None), ("AFF", aff, C["AFF"], None),
            ("Random gate,\ndraw 1", i7["r0"]["bar_margin_pooled"]["r1"], C["CTRL"], "///"),
            ("Random gate,\ndraw 2", i7["r1"]["bar_margin_pooled"]["r1"], C["CTRL"], "///")]
    fig, ax = plt.subplots(figsize=(11, 7.5))
    data = []
    for i, (lab, d, col, hatch) in enumerate(bars):
        v, ci = d["point"], d["ci95"]
        ax.bar(i, v, width=0.62, color=col, hatch=hatch, edgecolor="white" if hatch else col, lw=0, zorder=3)
        ax.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor=INK, elinewidth=1.8, capsize=5, zorder=4)
        ax.text(i, ci[1] + 0.015, sg(v), va="bottom", ha="center", fontsize=13)
        data.append(dict(bar=lab.replace("\n", " "), value=v, ci95=ci))
    ax.set_xticks(range(4)); ax.set_xticklabels([b[0] for b in bars], fontsize=13)
    ax.set_xlim(-0.6, 3.9)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Margin over B′(A0), R@1 points (95% interval)", fontsize=13)
    ax.tick_params(axis="y", labelsize=12); ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("A random gate that steers the same side as often matches AFF", fontsize=16, loc="left", pad=12)
    d1, d2 = i7["r0"]["AFF_minus_control_bar_margin"], i7["r1"]["AFF_minus_control_bar_margin"]
    ci = lambda d: f"[{sg(d['ci95'][0])}, {sg(d['ci95'][1])}]"
    fig.text(0.1, 0.0, "Fresh episodes, round 3, seeds 49 to 51. The random gate opens each condition as often as AFF "
             "does, at random.\nIt knows which one is condition a, which no method may. AFF minus draw 1: "
             f"{sg(d1['point'])} {ci(d1)}; draw 2: {sg(d2['point'])} {ci(d2)}.",
             fontsize=11.5, color=INK2, ha="left", va="top")
    FD["random_gate"] = data + [dict(bar="AFF minus draw 1", **i7["r0"]["AFF_minus_control_bar_margin"]),
                                dict(bar="AFF minus draw 2", **i7["r1"]["AFF_minus_control_bar_margin"])]
    save(fig, "random_gate.png")


from build_figures_2 import protocol, after_go, mllm  # noqa: E402  (new figures, share helpers via this module)

if __name__ == "__main__":
    chain(); pipeline(); held_comparison(); frontier(); per_pair(); random_gate()
    FD["protocol"] = protocol(); FD["after_go"] = after_go(); FD["mllm"] = mllm()
    json.dump(FD, open(f"{OUT}/figure_data.json", "w"), indent=1)
