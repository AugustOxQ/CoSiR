"""Two figures for the dev-log primer page. Run:
cd /project/CoSiR && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python \
    docs/reports/assets/2026-10-12_devlog_primer/build_primer_figures.py
Writes artelingo_example.{png,json} and condition_gain.png next to this script.

Painting choice: kitagawa-utamaro_upon-my-oath (development painting) passes every rule: genre label, 5 English
captions, 3 emotions (awe, contentment, amusement). Two of its 5 captions are over 20 words, so only the 3 short ones
are shown. No scan for another painting was needed.
"""
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.artelingo import load_artelingo  # noqa: E402
from src.data.artelingo_splits import artelingo_aspect_labels, artelingo_splits  # noqa: E402
from src.data.wikiart_genre import GENRE_NAMES  # noqa: E402

PAINTING = "kitagawa-utamaro_upon-my-oath"
ANN = Path("/data/PDD/artelingo/artelingo_train.json")
IMAGES = Path("/data/PDD/wikiart_proj/wikiart")
FD = ROOT / "docs/reports/assets/2026-10-12_affect_steering_line/figure_data.json"
TEAL, ORANGE, GREY, INK, INK2, GRID = "#1baf7a", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"
AFF, B0 = "#2a78d6", "#b3b1ab"
plt.rcParams["font.family"] = "DejaVu Sans"
DPI = 200


def pretty(s):
    if s == "Ukiyo_e":
        return "Ukiyo-e"
    return s.replace("_painting", "").replace("_", " ")


def canvas(w, h):
    fig = plt.figure(figsize=(w, h), dpi=DPI, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    return fig, ax


def fig1():
    data = load_artelingo()
    sid = np.asarray(data.sample_ids, dtype=np.int64)
    ann = json.loads(ANN.read_text())
    sp = artelingo_splits(data)
    lab = artelingo_aspect_labels(data)
    rows = [r for r in range(len(sid)) if ann[int(sid[r])]["painting"] == PAINTING]   # positional join
    held_groups = set(sp.groups[sp.held].tolist())
    assert not any(int(sp.groups[r]) in held_groups or r in set(sp.held.tolist()) for r in rows), "held painting"
    sel, trn = set(sp.selection.tolist()), set(sp.scorer_train.tolist())
    assert all(r in sel or r in trn for r in rows)
    split = "development" if all(r in sel for r in rows) else "training"
    gcode = {int(lab["genre"][r]) for r in rows}
    assert len(gcode) == 1 and -1 not in gcode, "genre label"
    genre = pretty(GENRE_NAMES[gcode.pop()])
    recs = [ann[int(sid[r])] for r in rows]
    assert len(recs) >= 4
    assert len({a["emotion"] for a in recs} - {"something else"}) >= 3
    styles = {a["art_style"] for a in recs}; assert len(styles) == 1
    style = pretty(styles.pop())
    shown, seen = [], set()
    for a in recs:                                   # <= 20 words; one caption per emotion first
        if len(a["caption"].split()) <= 20 and a["emotion"] != "something else":
            shown.append(a)
    shown = shown[:5]
    assert len({a["emotion"] for a in shown}) >= 3
    img = Image.open(IMAGES / recs[0]["image"]).convert("RGB")

    W, H = 8.0, 5.0
    fig, ax = canvas(W, H)
    ax.text(0.25, 0.4, "Style and genre belong to the painting; each caption carries its writer's emotion",
            fontsize=11.5, fontweight="bold", color=INK, va="center", wrap=False)
    # image
    iw, ih = img.size
    bw, bh = 2.6, 3.9
    s = min(bw / iw, bh / ih); dw, dh = iw * s, ih * s
    x0, y0 = 0.25, 0.85
    ax.imshow(np.asarray(img.resize((int(dw * 200), int(dh * 200)))), extent=(x0, x0 + dw, y0 + dh, y0), aspect="auto", zorder=2)
    # per painting
    rx = 3.25
    ax.text(rx, 0.95, "PER PAINTING", fontsize=9, fontweight="bold", color=INK2, va="center")
    ax.text(rx, 1.28, f"Style: {style}", fontsize=11.5, color=INK, va="center")
    ax.text(rx, 1.6, f"Genre: {genre}", fontsize=11.5, color=INK, va="center")
    ax.plot([rx, W - 0.25], [1.88, 1.88], color=GRID, lw=1.2)
    ax.text(rx, 2.12, "PER VIEWER: caption and the emotion its writer felt", fontsize=9, fontweight="bold", color=INK2, va="center")
    y = 2.4
    for a in shown:
        lines = textwrap.wrap("“" + a["caption"] + "”", 42)
        ax.text(rx, y, "\n".join(lines), fontsize=10, color=INK, va="top", linespacing=1.25)
        yt = y + 0.19 * len(lines) + 0.07
        t = ax.text(rx + 0.08, yt + 0.11, a["emotion"], fontsize=9, fontweight="bold", color="white", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc=TEAL, ec="none"))
        y = yt + 0.36
    assert y < H - 0.05, y
    fig.savefig(HERE / "artelingo_example.png", dpi=DPI, facecolor="white")
    (HERE / "artelingo_example.json").write_text(json.dumps({
        "painting": PAINTING, "split": split, "style": style, "genre": genre, "image_file": recs[0]["image"],
        "captions_shown": [{"caption": a["caption"], "emotion": a["emotion"]} for a in shown],
        "captions_all": [{"caption": a["caption"], "emotion": a["emotion"], "words": len(a["caption"].split())} for a in recs],
        "rule": "row r -> annotations[sample_ids[r]]; no row of the painting in sp.held"}, indent=1, ensure_ascii=False))


def fig2():
    fd = json.loads(FD.read_text())["frontier"]
    g = {r["scorer"]: r for r in fd}
    vals = {}
    for k in ("B0", "aff_fused"):
        r = g[k]
        assert abs(r["r1"] - (r["either"] + r["gain"]) / 2) < 1e-6
        vals[k] = (r["r1"], (r["either"] - r["gain"]) / 2, r["gain"])
    assert [round(x, 2) for x in vals["B0"]] == [18.84, 18.84, 0.0]
    assert [round(x, 2) for x in vals["aff_fused"]] == [19.44, 16.37, 3.06]

    W, H = 8.0, 8.6
    fig, ax = canvas(W, H)
    ax.text(W / 2, 0.4, "Condition gain: does the first place follow the examples?", fontsize=12.5,
            fontweight="bold", color=INK, ha="center", va="center")
    lx, cw, cgap = 0.2, 3.0, 0.4
    cols = [lx + 1.5 + 0.05, lx + 1.5 + 0.05 + 3.1 + 0.0]
    cols = [1.6, 4.75]
    cwid = 3.0
    ax.text(cols[0] + 1.4, 1.0, "condition a:\nthe examples share an emotion", fontsize=10, fontweight="bold",
            color=INK, ha="center", va="center", linespacing=1.2)
    ax.text(cols[1] + 1.4, 1.0, "condition b:\nthe examples share a genre", fontsize=10, fontweight="bold",
            color=INK, ha="center", va="center", linespacing=1.2)
    role_col = {"target": TEAL, "other": ORANGE, "neg": GREY}

    def card(x, y, rank, name, role):
        c = role_col[role]
        ax.add_patch(FancyBboxPatch((x, y), 0.88, 0.8, boxstyle="round,pad=0,rounding_size=0.07", fc=c, ec="none"))
        ax.text(x + 0.44, y + 0.2, rank, fontsize=9.5, color="white", ha="center", va="center", fontweight="bold")
        ax.text(x + 0.44, y + 0.55, name, fontsize=9.5, color="white", ha="center", va="center")

    def row(y, label, result, rescol, cond_b_first):
        ax.text(lx, y + 0.4, label, fontsize=10, color=INK, va="center", linespacing=1.25, fontweight="bold")
        ax.text(lx, y + 1.15, result, fontsize=10, color=rescol, va="center", fontweight="bold")
        for ci, cond in enumerate("ab"):
            x0 = cols[ci]
            if cond == "a":
                order = [("emotion", "target"), ("negative", "neg"), ("genre", "other")]
                status, scol = "1st place: the target (a hit)", TEAL
            else:
                if cond_b_first == "emotion":
                    order = [("emotion", "other"), ("negative", "neg"), ("genre", "target")]
                    status, scol = "1st place: the other-aspect\ncandidate (a miss)", ORANGE
                else:
                    order = [("genre", "target"), ("negative", "neg"), ("emotion", "other")]
                    status, scol = "1st place: the target (a hit)", TEAL
            for k, (nm, rl) in enumerate(order):
                card(x0 + k * 0.97, y, ["1st", "2nd", "3rd"][k], nm, rl)
            ax.text(x0, y + 0.98, status, fontsize=9.5, color=scol, va="center", fontweight="bold", linespacing=1.2)

    row(1.5, "a score that\nignores\nthe aspect", "condition gain 0", INK, "emotion")
    ax.plot([0.2, W - 0.2], [3.3, 3.3], color=GRID, lw=1.2)
    row(3.5, "a score that\nreads\nthe aspect", "condition gain\nabove 0", TEAL, "genre")
    ax.text(W / 2, 5.0, "teal = target     orange = other-aspect candidate     grey = negative", fontsize=9.5,
            color=INK2, ha="center", va="center")
    ax.text(0.2, 5.5, "Real numbers", fontsize=11, fontweight="bold", color=INK, va="center")
    # bars
    bax = fig.add_axes([0.1, 0.11, 0.87, 0.23])
    bax.set_facecolor("white")
    names = [("B′(A0)", B0, vals["B0"]), ("AFF", AFF, vals["aff_fused"])]
    xs, w = [0, 1], 0.34
    for x, (nm, col, (r1, oth, gain)) in zip(xs, names):
        bax.bar(x - w / 2 - 0.01, r1, w, color=col)
        bax.bar(x + w / 2 + 0.01, oth, w, color=col, alpha=0.45, hatch="///", edgecolor="white", lw=0)
        bax.text(x - w / 2 - 0.01, r1 + 0.5, f"{r1:.2f}", ha="center", va="bottom", fontsize=9.5, color=INK)
        bax.text(x + w / 2 + 0.01, oth + 0.5, f"{oth:.2f}", ha="center", va="bottom", fontsize=9.5, color=INK)
        bax.text(x, 29.2, f"condition gain {gain:.2f}", ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
        bax.text(x - w / 2 - 0.01, -1.2, "target\nranked first", ha="center", va="top", fontsize=9, color=INK2)
        bax.text(x + w / 2 + 0.01, -1.2, "other-aspect\nranked first", ha="center", va="top", fontsize=9, color=INK2)
    bax.set_xticks(xs); bax.set_xticklabels([])
    for x, (nm, *_ ) in zip(xs, names):
        bax.text(x, -8.2, nm, ha="center", va="top", fontsize=11, fontweight="bold", color=INK, clip_on=False)
    bax.set_ylim(0, 31); bax.set_xlim(-0.6, 1.6)
    bax.set_yticks([0, 10, 20]); bax.tick_params(axis="y", labelsize=9.5, colors=INK2, length=0)
    bax.tick_params(axis="x", length=0)
    bax.set_ylabel("ranked first (%)", fontsize=10, color=INK2)
    bax.yaxis.grid(True, color=GRID, lw=1); bax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        bax.spines[s].set_visible(False)
    bax.spines["bottom"].set_color(GRID)
    fig.text(0.5, 0.012, "held paintings", ha="center", fontsize=9.5, color=INK2)
    fig.savefig(HERE / "condition_gain.png", dpi=DPI, facecolor="white")
    print(vals)


if __name__ == "__main__":
    fig1(); fig2()
