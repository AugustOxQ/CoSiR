"""Generate the four NEW result figures for the buddy slides deck (report §8).

All numbers are hard-coded from docs/reports/2026-06-24_buddy_progress_report.md §8 — no wandb
needed. Writes PNGs into docs/reports/assets/slides/. Edit the tuples if final numbers shift
(e.g. after the last RedCaps run lands). Run from repo root in the CoSiR env:
    python scripts/make_slide_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUT = "docs/reports/assets/slides"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 15, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 140, "savefig.bbox": "tight"})
GRN, RED, GRY, BLU = "#2e8b57", "#c0392b", "#7f8c8d", "#1f6f9c"


def seed_replication():
    fams = ["#1 smooth", "#2 contrast", "#3 refresh"]
    t2i = [(-0.10, 0.12), (1.23, 0.18), (-0.03, 0.15)]   # (mean, sem)
    i2t = [(0.90, 0.49), (1.07, 0.46), (0.03, 0.20)]
    x = np.arange(len(fams)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (dat, lab, off, a) in enumerate([(t2i, "t2i R1", -w / 2, 0.9),
                                            (i2t, "i2t R1", w / 2, 0.55)]):
        m = [d[0] for d in dat]; e = [d[1] for d in dat]
        cols = [GRN if v > 0.3 else GRY for v in m]
        ax.bar(x + off, m, w, yerr=e, capsize=4, label=lab, color=cols,
               alpha=a, edgecolor="k", linewidth=0.5)
    ax.axhline(0, color="k", lw=1); ax.set_xticks(x); ax.set_xticklabels(fams)
    ax.set_ylabel("mean Δ R1 vs baseline (3 seeds)"); ax.legend()
    ax.set_title("Only #2 clears the noise floor")
    fig.savefig(f"{OUT}/seed_replication.png"); plt.close(fig)


def peak_curve():
    lam = [0.3, 0.5, 1.0, 2.0, 4.0]
    dt2i = [1.1, 1.6, 2.3, 1.2, -0.3]
    di2t = [1.3, 2.3, 3.2, 0.9, -1.8]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.axvspan(1.0, 4.0, color=RED, alpha=0.06)  # 'harm' zone past the peak
    ax.plot(lam, dt2i, "-o", color=GRN, label="Δ t2i R1", lw=2)
    ax.plot(lam, di2t, "-s", color=BLU, label="Δ i2t R1", lw=2)
    ax.axhline(0, color="k", lw=1); ax.axvline(1.0, color=GRY, ls="--", lw=1)
    ax.annotate("peak", (1.0, 3.35), color=GRY, ha="center")
    ax.set_xscale("log"); ax.set_xticks(lam)
    ax.set_xticklabels([str(v) for v in lam])
    ax.set_xlabel("λ_con"); ax.set_ylabel("Δ R1 vs λ_con=0")
    ax.set_title("#2 dose-response: peaks at 1.0, then harms"); ax.legend()
    fig.savefig(f"{OUT}/peak_curve.png"); plt.close(fig)


def transfer_bars():
    metrics = ["Δ t2i", "Δ i2t"]; imp = [2.3, 3.2]; red = [0.4, -0.9]
    x = np.arange(2); w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(x - w / 2, imp, w, label="Impressions", color=GRN,
           edgecolor="k", linewidth=0.5)
    ax.bar(x + w / 2, red, w, label="RedCaps (1:1)",
           color=[GRN if v > 0 else RED for v in red], alpha=0.65,
           edgecolor="k", linewidth=0.5)
    ax.axhline(0, color="k", lw=1); ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("Δ R1 at λ_con=1.0"); ax.legend()
    ax.set_title("The win does not transfer off near-duplicates")
    fig.savefig(f"{OUT}/transfer_bars.png"); plt.close(fig)


def nearup_enrichment():
    labels = ["E (union,\nused by #2)", "B (strict)", "RedCaps"]
    pct = [40.6, 79.9, 0.0]; enr = ["279×", "550×", "0×"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(labels, pct, color=[RED, "#8e2b1e", GRN],
                  edgecolor="k", linewidth=0.5)
    ax.axhline(0.145, color=GRY, ls="--", lw=1)
    ax.annotate("random 0.145%", (1.4, 4), color=GRY)
    for b, e in zip(bars, enr):
        ax.annotate(e, (b.get_x() + b.get_width() / 2, b.get_height() + 2),
                    ha="center", fontweight="bold")
    ax.set_ylabel("% buddy edges within same source photo")
    ax.set_title("40.6% of buddy edges are the same photo"); ax.set_ylim(0, 92)
    fig.savefig(f"{OUT}/nearup_enrichment.png"); plt.close(fig)


def laplacian_eigenmaps():
    """Slide 3 teaching figure: a synthetic buddy-like graph (left, drawn as a coordinate-free
    hairball) vs its Laplacian-Eigenmaps embedding (right, neighbors land close). Illustrates
    'embed the graph so connected nodes sit near each other'."""
    rng = np.random.default_rng(3)
    cols = ["#1f6f9c", "#2e8b57", "#e08214"]  # 3 latent communities
    per, k = 42, 6
    centers = np.array([[0, 2.3], [-2.0, -1.15], [2.0, -1.15]])
    pts = np.vstack([c + rng.normal(scale=1.05, size=(per, 2)) for c in centers])
    lab = np.repeat(np.arange(3), per)
    n = pts.shape[0]

    # symmetric kNN adjacency on latent distances (this is the "graph" — coords then discarded)
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :k]
    A = np.zeros((n, n))
    for i in range(n):
        A[i, nn[i]] = 1
    A = np.maximum(A, A.T)
    ei, ej = np.where(np.triu(A, 1) > 0)

    # Laplacian Eigenmaps: bottom non-trivial eigenvectors of the normalized Laplacian
    deg = A.sum(1); dinv2 = 1.0 / np.sqrt(deg)
    Lsym = np.eye(n) - (A * dinv2[:, None]) * dinv2[None, :]
    _, V = np.linalg.eigh(Lsym)
    emb = dinv2[:, None] * V[:, 1:3]  # eigvecs 1,2 (skip trivial 0)

    # coordinate-free hairball layout for the left panel: nodes on a circle in shuffled order
    order = rng.permutation(n)
    ang = np.zeros(n); ang[order] = 2 * np.pi * np.arange(n) / n
    circ = np.c_[np.cos(ang), np.sin(ang)]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2))
    for ax, P, title, sub in [
        (axes[0], circ, "The graph E", "who is whose buddy — no coordinates"),
        (axes[1], emb, "Laplacian Eigenmaps", "neighbors → nearby coordinates"),
    ]:
        ax.axis("off"); ax.grid(False); ax.set_aspect("equal")
        for a, b in zip(ei, ej):
            ax.plot(*zip(P[a], P[b]), color="#c4c4c4", lw=0.4, zorder=1)
        ax.scatter(P[:, 0], P[:, 1], c=[cols[l] for l in lab], s=42,
                   edgecolor="k", linewidth=0.4, zorder=2)
        ax.set_title(title, fontsize=14, pad=10)
        ax.text(0.5, -0.04, sub, transform=ax.transAxes, ha="center",
                va="top", fontsize=10, color="#555")
    axes[1].annotate("", xy=(0.04, 0.5), xytext=(-0.09, 0.5),
                     xycoords="axes fraction", textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", color="k", lw=2))
    fig.suptitle(r"Embed the graph so connected nodes sit close:  "
                 r"minimize  $\sum_{(i,j)\in E} w_{ij}\,\|z_i-z_j\|^2$",
                 fontsize=13.5, y=1.0)
    fig.subplots_adjust(top=0.82, wspace=0.02, bottom=0.06)
    fig.savefig(f"{OUT}/laplacian_eigenmaps.png"); plt.close(fig)


def seed_replication_table():
    """Slide 7 detailed table: per-seed paired ΔR1 for #1 and #2 (seed-replicated), with
    mean ± std, mean/SEM, and verdict. #3 is a footnote (grid-confirmed null, not seed-rep)."""
    cols = ["Family", "Metric", "seed 1", "seed 2", "seed 3",
            "mean Δ ± std", "mean/SEM", "verdict"]
    rows = [
        ["#1  Smoothness", "t2i", "−0.30", "+0.10", "−0.10", "−0.10 ± 0.20", "−0.9", "null"],
        ["#1  Smoothness", "i2t", "+0.00", "+1.70", "+1.00", "+0.90 ± 0.85", "+1.8", "n.s."],
        ["#2  Contrastive", "t2i", "+0.90", "+1.50", "+1.30", "+1.23 ± 0.31", "+7.0 *", "WIN"],
        ["#2  Contrastive", "i2t", "+1.90", "+0.30", "+1.00", "+1.07 ± 0.80", "+2.3 *", "WIN"],
    ]
    widths = [0.19, 0.08, 0.085, 0.085, 0.085, 0.17, 0.11, 0.11]
    fig, ax = plt.subplots(figsize=(11.2, 3.2))
    ax.axis("off"); ax.grid(False)
    tbl = ax.table(cellText=rows, colLabels=cols, colWidths=widths,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11.5); tbl.scale(1, 2.0)
    hdr, g1, g2 = "#2c3e50", "#eef1f2", "#e2f0e8"
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("white"); cell.set_linewidth(1.5)
        if r == 0:  # header
            cell.set_facecolor(hdr); cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            win = r >= 3  # #2 rows
            cell.set_facecolor(g2 if win else g1)
            if c == 0:
                cell.get_text().set_fontweight("bold")
            if c == 7:  # verdict column
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_color(GRN if win else GRY)
            if c == 6 and win:  # significant mean/SEM
                cell.get_text().set_color(GRN)
    ax.set_title("Seed replication (3 seeds · lr=1e-3, lr_label=1e-4): paired ΔR1 vs baseline",
                 fontsize=14, pad=14)
    fig.text(0.5, 0.02,
             "#3 self-refresh: grid-confirmed null (blend 0 vs 1: t2i −0.03, i2t +0.03) — "
             "dropped from replication.",
             ha="center", fontsize=9.5, color="#666", style="italic")
    fig.savefig(f"{OUT}/seed_replication_table.png"); plt.close(fig)


def training_families():
    """Slide 6 schematic: the three ways to use buddies during training — #1 smoothness in
    z-space, #2 contrastive in retrieval space, #3 self-refreshing graph. Neutral colors
    (the '#2 wins' reveal is a later slide)."""
    INK, PULL, PUSH = "#2c3e50", "#2e8b57", "#c0392b"
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.2))
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)

    # ── #1 Smoothness: pull buddies' z together in condition space ──────────────
    ax = axes[0]
    ax.set_title("#1  Smoothness", fontsize=14, color=INK, pad=8)
    ax.text(0.5, 0.9, "regularizer in condition (z) space", ha="center", fontsize=10, color="#555")
    zpos = np.array([[0.30, 0.55], [0.62, 0.60], [0.44, 0.30], [0.75, 0.35]])
    pairs = [(0, 1), (2, 3)]
    for a, b in pairs:
        pa, pb = zpos[a], zpos[b]; mid = (pa + pb) / 2
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#bbb", lw=1, zorder=1)
        for p in (pa, pb):  # inward arrows = "pull together"
            ax.annotate("", xy=mid, xytext=p,
                        arrowprops=dict(arrowstyle="-|>", color=PULL, lw=1.8, shrinkA=8, shrinkB=8))
    ax.scatter(zpos[:, 0], zpos[:, 1], s=170, c="#dfe6ea", edgecolor=INK, linewidth=1.2, zorder=3)
    for i, (x, y) in enumerate(zpos):
        ax.text(x, y, f"$z_{i}$", ha="center", va="center", fontsize=10, color=INK, zorder=4)
    ax.text(0.5, 0.08, r"minimize $\|z_i - z_j\|^2$ over buddies", ha="center", fontsize=10.5, color=PULL)

    # ── #2 Contrastive: buddy positives pulled in, others pushed out ────────────
    ax = axes[1]
    ax.set_title("#2  Contrastive", fontsize=14, color=INK, pad=8)
    ax.text(0.5, 0.9, "buddy positives in retrieval space", ha="center", fontsize=10, color="#555")
    anc = np.array([0.5, 0.52])
    pos = np.array([[0.28, 0.68], [0.30, 0.42], [0.46, 0.74]])   # buddies (+)
    neg = np.array([[0.82, 0.66], [0.80, 0.36]])                  # non-buddies (−)
    for p in pos:
        ax.annotate("", xy=anc + (p - anc) * 0.42, xytext=p,
                    arrowprops=dict(arrowstyle="-|>", color=PULL, lw=1.8))
    for p in neg:
        ax.annotate("", xy=p + (p - anc) * 0.30, xytext=p,
                    arrowprops=dict(arrowstyle="-|>", color=PUSH, lw=1.8))
    ax.scatter(*anc, marker="*", s=420, c="#f1c40f", edgecolor=INK, linewidth=1.2, zorder=4)
    ax.text(anc[0], anc[1] - 0.12, "anchor", ha="center", fontsize=9, color=INK)
    ax.scatter(pos[:, 0], pos[:, 1], s=150, c=PULL, edgecolor="k", linewidth=0.5, zorder=3)
    ax.scatter(neg[:, 0], neg[:, 1], s=150, c=PUSH, edgecolor="k", linewidth=0.5, zorder=3)
    ax.text(0.30, 0.24, "+ buddies", color=PULL, fontsize=10, ha="center")
    ax.text(0.81, 0.22, "− others", color=PUSH, fontsize=10, ha="center")
    ax.text(0.5, 0.06, "InfoNCE: pull buddies' images together", ha="center", fontsize=10.5, color=INK)

    # ── #3 Self-refresh: rebuild the graph from the model's own features ────────
    ax = axes[2]
    ax.set_title("#3  Self-refresh", fontsize=14, color=INK, pad=8)
    ax.text(0.5, 0.9, "graph rebuilt from the model's features", ha="center", fontsize=10, color="#555")

    def mini_graph(cx, cy, seed, color):
        r = np.random.default_rng(seed)
        p = np.c_[cx + r.uniform(-0.09, 0.09, 5), cy + r.uniform(-0.13, 0.13, 5)]
        for a in range(5):
            for b in range(a + 1, 5):
                if r.random() < 0.5:
                    ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], color="#ccc", lw=0.8, zorder=1)
        ax.scatter(p[:, 0], p[:, 1], s=90, c=color, edgecolor="k", linewidth=0.5, zorder=2)
    mini_graph(0.26, 0.50, 1, "#9db4c0"); ax.text(0.26, 0.26, "frozen CLIP\ngraph", ha="center", fontsize=9, color="#555")
    mini_graph(0.74, 0.50, 7, "#7fb08a"); ax.text(0.74, 0.26, "model's current\ngraph", ha="center", fontsize=9, color="#555")
    ax.annotate("", xy=(0.60, 0.58), xytext=(0.40, 0.58),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2,
                                connectionstyle="arc3,rad=-0.4"))
    ax.text(0.5, 0.72, "refresh\nevery R epochs", ha="center", fontsize=9.5, color=INK)
    ax.text(0.5, 0.06, "graph co-evolves with training", ha="center", fontsize=10.5, color=INK)

    fig.suptitle("Three ways to use buddies during training  —  all share ONE buddy init",
                 fontsize=14.5, y=1.0)
    fig.subplots_adjust(top=0.80, wspace=0.04, bottom=0.04)
    fig.savefig(f"{OUT}/training_families.png"); plt.close(fig)


def buddy_venn():
    """Slide 2 schematic: B = A_img ∩ A_txt (strict) vs E = A_img ∪ A_txt (broad, used for init)."""
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.set_aspect("equal"); ax.axis("off"); ax.grid(False)
    r = 1.35
    c_img, c_txt = (-0.6, 0.15), (0.6, 0.15)
    ax.add_patch(Circle(c_img, r, facecolor=BLU, alpha=0.35, edgecolor=BLU, lw=2.5))
    ax.add_patch(Circle(c_txt, r, facecolor=GRN, alpha=0.35, edgecolor=GRN, lw=2.5))
    # per-modality labels over the non-overlapping lobes
    ax.text(-1.35, 1.35, "$A_{img}$", ha="center", color=BLU, fontsize=17, fontweight="bold")
    ax.text(-1.35, 1.02, "image mutual-KNN", ha="center", color=BLU, fontsize=10)
    ax.text(1.35, 1.35, "$A_{txt}$", ha="center", color=GRN, fontsize=17, fontweight="bold")
    ax.text(1.35, 1.02, "text mutual-KNN", ha="center", color=GRN, fontsize=10)
    # B = intersection, in the lens
    ax.text(0, 0.35, "B", ha="center", va="center", fontsize=16, fontweight="bold", color="k")
    ax.text(0, -0.05, "∩\nstrict\nsparse", ha="center", va="center", fontsize=9, color="#222")
    # E = union callout (points at the whole shape)
    ax.annotate("E = $A_{img}$ ∪ $A_{txt}$\nbroad · well-connected\n→ used for init",
                xy=(1.55, -0.6), xytext=(2.15, -1.7), ha="center", va="top", fontsize=11,
                color=GRY, arrowprops=dict(arrowstyle="->", color=GRY, lw=1.2))
    ax.annotate("", xy=(-1.55, -0.6), xytext=(1.9, -1.55),
                arrowprops=dict(arrowstyle="->", color=GRY, lw=1.2))
    ax.set_xlim(-3.0, 3.4); ax.set_ylim(-2.4, 1.9)
    ax.set_title("Two graphs from one mutual-KNN (K=30)", fontsize=15)
    fig.savefig(f"{OUT}/buddy_venn.png"); plt.close(fig)


if __name__ == "__main__":
    seed_replication(); peak_curve(); transfer_bars(); nearup_enrichment()
    buddy_venn(); laplacian_eigenmaps(); training_families(); seed_replication_table()
    print("wrote 8 figures to", OUT)
