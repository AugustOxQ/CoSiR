"""Supplementary figure: clustering AMI against the number of clusters K on ArtELingo (held test split).

One panel per aspect (emotion, style, genre); one line per method. All numbers come from:
  - R17 sweep  (plain, teacher, matched -> PercepT (Qwen3), cstar -> CoSiR (ours))
  - records/cells.json of this round (TURTLE, TPIPS, CLAY, CRL, VLM2Vec, BERTopic, PercepT (CLIP L/14) from r26)
  - optional --percept-clip JSON list of {aspect, k, mean} (PercepT (CLIP L/14), round r26)

Writes <out-dir>/supp_cluster_counts.pdf, .png (300 dpi) and values.md (plotted means, 3 decimals).
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
R17 = Path("/data/SSD2/cosir_r17/20261103_new_scores/outputs/stage3/sweep/artelingo__qwen.json")
KR = HERE / "records" / "cells.json"

ASPECTS = ["emotion", "style", "genre"]
ASPECT_TITLE = {"emotion": "Emotion", "style": "Style", "genre": "Genre"}
GRID = {"emotion": [5, 8, 9, 16, 20], "style": [10, 20, 27, 40], "genre": [5, 10, 16, 20]}
CLASS_COUNT = {"emotion": 9, "style": 27, "genre": 10}

R17_ARMS = {"plain": "Plain embedding", "teacher": "Teacher (instructed)",
            "matched": "PercepT (Qwen3)", "cstar": "CoSiR (ours)"}
PERCEPT_CLIP = "PercepT (CLIP L/14)"
KR_ROWS = {"turtle": "Plain + TURTLE", "tpips": "TPIPS", "clay": "CLAY", "crl": "CRL",
           "vlm2vec": "VLM2Vec", "bertopic": "BERTopic", "percept_clip": PERCEPT_CLIP}
CLASS_BAND = "#DEDEDE"

# Drawing order (back to front) and legend order. Rows not loaded are skipped.
ROW_ORDER = ["Plain embedding", "Plain + TURTLE", "PercepT (Qwen3)", PERCEPT_CLIP, "TPIPS", "CLAY",
             "CRL", "VLM2Vec", "BERTopic", "Teacher (instructed)", "CoSiR (ours)"]
LEGEND_ORDER = ["CoSiR (ours)", "Teacher (instructed)", "PercepT (Qwen3)", PERCEPT_CLIP,
                "Plain embedding", "Plain + TURTLE", "TPIPS", "CLAY", "CRL", "VLM2Vec", "BERTopic"]

# Okabe-Ito based qualitative colours; markers and dashes keep lines apart in greyscale.
STYLE = {
    "CoSiR (ours)":         dict(color="#D55E00", ls="-", marker="o", lw=1.9, ms=3.8, z=20, mfc=None),
    "Teacher (instructed)": dict(color="#1A1A1A", ls=(0, (4.0, 1.6)), marker="s", lw=1.3, ms=3.0, z=15, mfc=None),
    "PercepT (Qwen3)":      dict(color="#0072B2", ls="-", marker="^", lw=1.0, ms=3.2, z=10, mfc=None),
    PERCEPT_CLIP:           dict(color="#56B4E9", ls=(0, (3.0, 1.2, 1.0, 1.2)), marker="v", lw=1.0, ms=3.2, z=10, mfc=None),
    "Plain embedding":      dict(color="#8C8C8C", ls="-", marker="o", lw=1.0, ms=3.2, z=8, mfc="white"),
    "Plain + TURTLE":       dict(color="#8C8C8C", ls=(0, (1.0, 1.2)), marker="D", lw=1.0, ms=3.0, z=8, mfc=None),
    "TPIPS":                dict(color="#009E73", ls=(0, (3.0, 1.2)), marker="P", lw=1.0, ms=3.3, z=9, mfc=None),
    "CLAY":                 dict(color="#E69F00", ls="-", marker="X", lw=1.0, ms=3.3, z=9, mfc=None),
    "CRL":                  dict(color="#CC79A7", ls=(0, (1.0, 1.2)), marker="h", lw=1.0, ms=3.3, z=9, mfc=None),
    "VLM2Vec":              dict(color="#7B4F2C", ls=(0, (3.0, 1.2, 1.0, 1.2)), marker="d", lw=1.0, ms=3.3, z=9, mfc=None),
    "BERTopic":             dict(color="#8E7CC3", ls="-", marker="*", lw=1.0, ms=4.2, z=9, mfc=None),
}


def setup_fonts():
    """Times-like serif: use the first family that exists (register system TTFs matplotlib has not cached)."""
    for d in ["/usr/share/fonts/truetype/freefont", "/usr/share/fonts/truetype/liberation",
              "/usr/share/fonts/opentype", "/usr/share/fonts/type1"]:
        if os.path.isdir(d):
            for f in font_manager.findSystemFonts(fontpaths=[d]):
                try:
                    font_manager.fontManager.addfont(f)
                except Exception:
                    pass
    have = {f.name for f in font_manager.fontManager.ttflist}
    wanted = ["Times New Roman", "Times", "Nimbus Roman", "Nimbus Roman No9 L", "TeX Gyre Termes",
              "FreeSerif", "Liberation Serif", "STIXGeneral"]
    chosen = [f for f in wanted if f in have]
    plt.rcParams.update({
        "font.family": "serif", "font.serif": chosen + ["DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    })
    return chosen[0] if chosen else "DejaVu Serif"


def load_rows(percept_clip=None):
    data = defaultdict(dict)  # label -> {(aspect, k): mean}
    r17 = json.loads(R17.read_text())
    assert r17["class_count"] == CLASS_COUNT, r17["class_count"]
    for r in r17["rows"]:
        if r["metric"] == "ami" and r["arm"] in R17_ARMS:
            data[R17_ARMS[r["arm"]]][(r["aspect"], int(r["k"]))] = float(r["mean"])
    for c in json.loads(KR.read_text())["cells"]:
        assert c["id"].split("/")[2] == "ami", c["id"]
        data[KR_ROWS[c["row"]]][(c["aspect"], int(c["k"]))] = float(c["mean"])
    if percept_clip:
        obj = json.loads(Path(percept_clip).read_text())
        cells = obj["cells"] if isinstance(obj, dict) else obj
        for c in cells:
            data[PERCEPT_CLIP][(c["aspect"], int(c["k"]))] = float(c["mean"])
    return data


def check_coverage(data):
    print(f"{'row':<22}" + "".join(f"{a:>10}" for a in ASPECTS))
    for label in ROW_ORDER:
        if label not in data:
            print(f"{label:<22}" + "".join(f"{'absent':>10}" for _ in ASPECTS))
            continue
        cells = []
        for a in ASPECTS:
            have = [k for k in GRID[a] if (a, k) in data[label]]
            cells.append(f"{len(have)}/{len(GRID[a])}")
            missing = sorted(set(GRID[a]) - set(have))
            assert not missing, f"{label} {a}: missing K {missing}"
        print(f"{label:<22}" + "".join(f"{c:>10}" for c in cells))
    unknown = set(data) - set(ROW_ORDER)
    assert not unknown, unknown


def write_values(data, path, extra=None):
    lines = ["# Plotted AMI means (ArtELingo held test split), generated by kr_figure.py", "",
             f"Sources: `{R17}` (R17 sweep), `{KR}`" + (f" and `{extra}`" if extra else ""), ""]
    for a in ASPECTS:
        lines += [f"## {ASPECT_TITLE[a]} (class count {CLASS_COUNT[a]})", "",
                  "| Method | " + " | ".join(f"K={k}" for k in GRID[a]) + " |",
                  "|---|" + "---:|" * len(GRID[a])]
        for label in LEGEND_ORDER:
            if label in data:
                lines.append(f"| {label} | " + " | ".join(f"{data[label][(a, k)]:.3f}" for k in GRID[a]) + " |")
        lines.append("")
    path.write_text("\n".join(lines))


def draw(data, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(6.875, 2.35))
    for ax, a in zip(axes, ASPECTS):
        ks = GRID[a]
        ax.axvline(CLASS_COUNT[a], color=CLASS_BAND, lw=3.0, zorder=0, solid_capstyle="butt")
        for label in ROW_ORDER:
            if label not in data:
                continue
            s = STYLE[label]
            ax.plot(ks, [data[label][(a, k)] for k in ks], color=s["color"], ls=s["ls"], lw=s["lw"],
                    marker=s["marker"], ms=s["ms"], mfc=s["mfc"] or s["color"], mec=s["color"], mew=0.7,
                    zorder=s["z"], clip_on=False)
        ax.set_xticks(ks)
        ax.set_xlim(ks[0] - 0.04 * (ks[-1] - ks[0]), ks[-1] + 0.04 * (ks[-1] - ks[0]))
        ax.set_title(ASPECT_TITLE[a], pad=3)
        ax.grid(axis="y", color="#EBEBEB", lw=0.5, zorder=-1)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[1].set_xlabel("Number of clusters", labelpad=1.5)
    axes[0].set_ylabel("AMI", labelpad=2)
    handles = []
    for label in LEGEND_ORDER:
        if label in data:
            s = STYLE[label]
            handles.append(Line2D([], [], color=s["color"], ls=s["ls"], lw=s["lw"], marker=s["marker"],
                                  ms=s["ms"], mfc=s["mfc"] or s["color"], mec=s["color"], mew=0.7, label=label))
    handles.append(Patch(facecolor=CLASS_BAND, edgecolor="none", label="Class count"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=2.6, columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.2)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "supp_cluster_counts.pdf")
    fig.savefig(out_dir / "supp_cluster_counts.png", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--percept-clip", type=str, default=None,
                    help="JSON list of {aspect, k, mean} for PercepT (CLIP L/14) (round r26)")
    args = ap.parse_args()
    font = setup_fonts()
    print(f"font: {font}")
    data = load_rows(args.percept_clip)
    check_coverage(data)
    draw(data, args.out_dir)
    write_values(data, args.out_dir / "values.md", args.percept_clip)
    print(f"wrote {args.out_dir}/supp_cluster_counts.pdf, .png, values.md")


if __name__ == "__main__":
    main()
