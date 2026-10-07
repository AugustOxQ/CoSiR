#!/usr/bin/env python3
"""Build the controller-specified weekly report figures, without source inference.

Usage: python build_2026-10-07_weekly_figures.py [figure_name]
Names may include the .png suffix. No argument builds every figure.
All scientific values and displayed precision are fixed by the controller.
"""

import argparse
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image


OUT = Path(__file__).resolve().parent / "2026-10-07_weekly"
TEAL = "#1B8A7A"
ORANGE = "#D9822B"
SLATE = "#5B6C8F"
GREY = "#9AA0A6"
PURPLE = "#8E7CC3"
INK = "#263444"
LIGHT = "#F2F3F4"
DPI = 200

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.edgecolor": "#BCC2C8",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": DPI,
    "svg.fonttype": "none",
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=DPI)
    plt.close(fig)


def xgrid(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E6E9EC", linewidth=0.8)
    ax.tick_params(axis="both", length=0, pad=8)


def factor_line_end():
    fig, axs = plt.subplots(1, 3, figsize=(24, 10))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.79,
                        bottom=0.17, wspace=0.20)
    fig.suptitle("Why the factor line stopped", fontsize=25,
                 fontweight="bold", y=0.97)
    panels = [
        ("(a) Value episodes:\nthe query adds little", "Selection rows, pooled", [
            ("CLIP only", "13.45", GREY, False),
            ("SE factors", "21.22", TEAL, False),
            ("Prototype of the examples, query ignored", "22.83", SLATE, False),
            ("Prototype + query", "23.40", SLATE, False),
            ("Logistic probe on the 4 + 4 examples, raw CLIP", "24.10", SLATE, False),
        ], 40, None),
        ("(b) Aspect episodes:\nno factor model selects the aspect", "Emotion × style | 4,096 episodes | pooled", [
            ("CLIP only", "11.13", GREY, False),
            ("R3", "11.05", TEAL, False),
            ("C0", "11.09", TEAL, False),
            ("SE", "11.32", TEAL, False),
            ("Raw-CLIP agreement rule", "11.16", GREY, False),
            ("Names given as text (privileged)", "12.63", LIGHT, True),
            ("Label probes, told the aspect (ceiling)", "23.09", LIGHT, True),
        ], 28, 7.69),
        ("(c) Method A's GO test\n(fresh seed 43)", "", [
            ("Cosine", "13.53", GREY, False),
            ("RCA", "13.52", GREY, False),
            ("Method A (A3)", "13.76", TEAL, False),
            ("Its condition-free control", "16.72", SLATE, False),
        ], 21, None),
    ]
    for index, (ax, (heading, detail, rows, xmax, chance)) in enumerate(zip(axs, panels)):
        ax.set_title(heading, loc="left", pad=52, fontsize=18)
        ax.text(0, 1.055, detail, transform=ax.transAxes, fontsize=13)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-1.0, 7.8)
        ax.set_yticks([])
        ax.set_ylabel("Scorer", labelpad=15)
        ax.set_xlabel("R@1 (%)", labelpad=10)
        xgrid(ax)
        for row_index, (label, value, color, hatch) in enumerate(rows):
            y = 7.1 - row_index
            edge = GREY if hatch else (ORANGE if index == 2 and row_index == 3 else color)
            ax.barh(y - 0.20, float(value), height=0.34, color=color,
                    edgecolor=edge, linewidth=2 if index == 2 and row_index == 3 else 1,
                    hatch="////" if hatch else None, zorder=3)
            ax.text(0, y + 0.10, label, fontsize=13.5, va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.2), zorder=4)
            inside = index == 0 and row_index in (2, 3)
            ax.text(float(value) - 0.35 if inside else float(value) + 0.30,
                    y - 0.20, value, fontsize=15, va="center",
                    ha="right" if inside else "left",
                    color="white" if inside else INK, fontweight="bold")
        if chance is not None:
            ax.axvline(chance, linestyle=":", color=GREY, linewidth=1.8)
            ax.text(chance, -0.74, "Chance 7.69", ha="center", fontsize=13,
                    bbox=dict(facecolor="white", edgecolor="none", pad=2))
        if index == 0:
            ax.plot([24.8, 25.5, 25.5, 24.8], [4.9, 4.9, 3.9, 3.9],
                    color=SLATE, linewidth=1.6)
            ax.text(26.2, 4.4, "query adds\n+0.57\n[+0.20, +0.96]",
                    fontsize=14.5, fontweight="bold", color=SLATE, va="center")
        if index == 2:
            ax.text(0, 1.9, "A3 − control:\n−2.96 [−3.29, −2.64]", fontsize=16,
                    color=ORANGE, fontweight="bold", va="top")
    fig.legend(handles=[
        Patch(facecolor=TEAL, label="Factor method"),
        Patch(facecolor=SLATE, label="Condition-free comparator"),
        Patch(facecolor=GREY, label="External baseline / raw agreement"),
        Patch(facecolor=LIGHT, edgecolor=GREY, hatch="////",
              label="Evaluation-label reference (diagnostic ceiling)"),
        Patch(facecolor=SLATE, edgecolor=ORANGE, linewidth=2,
              label="Method A's own control"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.018), ncol=3,
        frameon=False, fontsize=13.5)
    save(fig, "factor_line_end")


def modality_asymmetry():
    fig, ax = plt.subplots(figsize=(17, 9))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.79, bottom=0.27)
    fig.suptitle("Emotion and style are each carried mainly by one modality, for every backbone",
                 y=0.96, fontsize=23, fontweight="bold")
    labels = ["Emotion from captions\n(strong)\nspread 5.6",
              "Emotion from images\n(weak)\nspread 1.5",
              "Style from images\n(strong)\nspread 11.0",
              "Style from captions\n(weak)\nspread 0.9"]
    series = [
        ("CLIP ViT-B/32", "#1B8A7A", ["56.9", "35.1", "60.9", "25.4"]),
        ("SigLIP 2", "#65BFB0", ["58.7", "36.3", "70.7", "25.9"]),
        ("PE-Core", "#5B6C8F", ["58.9", "36.6", "71.9", "26.3"]),
        ("Qwen3-VL-Embedding-2B", "#9AA8CA", ["62.5", "36.4", "64.2", "26.1"]),
    ]
    positions = np.arange(4)
    width = 0.18
    for index, (name, color, values) in enumerate(series):
        xs = positions + (index - 1.5) * width
        bars = ax.bar(xs, [float(value) for value in values], width=width,
                      color=color, label=name, zorder=3)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, float(value) + 1.3,
                    value, ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.set_xticks(positions, labels, fontsize=16, linespacing=1.5)
    ax.set_ylim(0, 82)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylabel("Probe accuracy (%)")
    ax.set_xlabel("Aspect and input modality | spread in percentage points", labelpad=18)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E6E9EC")
    ax.tick_params(length=0, pad=12)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.89), ncol=4,
               frameon=False, fontsize=15)
    save(fig, "modality_asymmetry")


def methods_vs_controls():
    fig, ax = plt.subplots(figsize=(21, 11))
    fig.subplots_adjust(left=0.405, right=0.805, top=0.79, bottom=0.17)
    fig.suptitle("Each method against its own condition-free control (development seed 42)",
                 y=0.97, fontsize=22, fontweight="bold")
    rows = [
        ("Method A (A3), 3 Oct", "16.55", "13.39", "13.39 vs 16.55"),
        ("Repair A′ on A3, 3 Oct", "16.55", "16.52", "16.52 vs 16.55"),
        ("Centered rule N1, 4 Oct (matched control)", "17.94", "16.79", "−1.15"),
        ("Partition-head reader N6, 4 Oct", "17.07", "17.30", "+0.23"),
        ("N6 on the centered base (N6c), 4 Oct", "18.34", "18.49", "+0.15"),
        ("Gated learned reader (R-c), round 1, 6 Oct", "18.475", "18.919", "+0.444"),
        ("One-sided affect steering (AFF), 6 Oct", "18.437", "19.137", "+0.700"),
        ("AFF on fresh seeds 49 to 51 (pooled)", "18.288", "18.880", "+0.591 [+0.462, +0.729]"),
    ]
    ys = [7, 6, 5, 4, 3, 2, 1, -0.5]
    ax.set_xlim(12.5, 19.5)
    ax.set_ylim(-1.3, 8.25)
    ax.set_xlabel("R@1 (%)", labelpad=14)
    ax.set_ylabel("Method and evaluation", labelpad=22)
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=15)
    ax.tick_params(axis="y", pad=20)
    xgrid(ax)
    for value, label, height in [(12.96, "Cosine 12.96", 1.085),
                                  (13.38, "RCA 13.38", 1.025),
                                  (18.34, "B 18.34", 1.085)]:
        ax.axvline(value, color=GREY, linestyle=":", linewidth=1.7, zorder=1)
        ax.text(value, height, label, transform=ax.get_xaxis_transform(),
                ha="center", color="#67717C", fontsize=14)
    ax.text(1.03, 1.085, "Reported comparison", transform=ax.transAxes,
            fontsize=14, fontweight="bold")
    for index, ((label, control, method, difference), y) in enumerate(zip(rows, ys)):
        below = float(method) < float(control)
        color = ORANGE if below else TEAL
        marker = "s" if index == 7 else "o"
        ax.plot([float(control), float(method)], [y, y], color="#BBC3CD",
                linewidth=3, zorder=2)
        ax.scatter(float(control), y, marker=marker, s=185, facecolor="white",
                   edgecolor=SLATE, linewidth=2.5, zorder=3)
        ax.scatter(float(method), y, marker=marker, s=85, facecolor=color,
                   edgecolor="white", linewidth=1, zorder=4)
        ax.text(1.03, y, difference, transform=ax.get_yaxis_transform(),
                va="center", fontsize=15, color=color, fontweight="bold")
        if index == 6:
            ax.text(-0.025, y - 0.37, "Control: strongest condition-free, B′(A0)",
                    transform=ax.get_yaxis_transform(), ha="right",
                    color=SLATE, fontsize=12.5)
    ax.axhline(0.25, color="#BCC2C8", linewidth=1)
    fig.legend(handles=[
        Line2D([], [], marker="o", color=SLATE, markerfacecolor="white",
               markeredgewidth=2, linestyle="none", markersize=10,
               label="Condition-free control (same score, condition removed)"),
        Line2D([], [], marker="o", color=TEAL, linestyle="none", markersize=10, label="Method"),
        Line2D([], [], marker="o", color=ORANGE, linestyle="none", markersize=10,
               label="Method below its control"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=3,
        frameon=False, fontsize=14)
    save(fig, "methods_vs_controls")


def go_checks():
    fig, axs = plt.subplots(1, 3, figsize=(24, 11.5),
                            gridspec_kw={"width_ratios": [1.0, 1.55, 1.35]})
    fig.subplots_adjust(left=0.055, right=0.97, top=0.75,
                        bottom=0.20, wspace=0.23)
    fig.suptitle("One-sided affect steering passed all seven pre-registered checks on fresh episodes",
                 fontsize=23, fontweight="bold", y=0.975)
    fig.text(0.5, 0.905, "Pooled over seeds 49 to 51 | 36,864 episodes | 95% intervals",
             ha="center", fontsize=17)
    external = [
        ("Cosine", "+5.840", "+5.611", "+6.069", False),
        ("RCA", "+5.722", "+5.495", "+5.946", False),
    ]
    comparators = [
        ("B", "+0.806", "+0.686", "+0.930", False),
        ("B′(A0), strongest condition-free\ncomparator", "+0.591", "+0.462", "+0.729", False),
        ("Its matched control", "+0.796", "+0.670", "+0.920", False),
        ("Secondary: gated reader R1\n(not a GO check)", "+0.202", "+0.093", "+0.309", True),
    ]
    gain = [
        ("Condition-free scorers\n(B, B′, control; all 0)", "+3.319", "+3.130", "+3.518", False),
        ("RCA", "+3.286", "+3.076", "+3.494", False),
    ]
    specs = [
        ("External baselines", external, [3.7, 1.4], 6.4,
         "R@1 difference (percentage points)"),
        ("Condition-free comparators", comparators, [4.6, 3.2, 1.8, 0.4], 1.05,
         "R@1 difference (percentage points)"),
        ("Condition gain", gain, [3.7, 1.4], 3.85,
         "Condition-gain difference (percentage points)"),
    ]
    for panel, (ax, (title, rows, ys, xmax, xlabel)) in enumerate(zip(axs, specs)):
        ax.set_title(title, fontsize=18, loc="left", pad=25)
        ax.set_xlim(-xmax * 0.035, xmax)
        ax.set_ylim(-0.4, 5.5)
        ax.set_yticks([])
        ax.set_ylabel("Comparator", labelpad=15)
        ax.set_xlabel(xlabel, fontsize=14.5, labelpad=15)
        ax.axvline(0, color="#67717C", linewidth=1.5)
        xgrid(ax)
        for label, mean, low, high, secondary in rows:
            y = ys[rows.index((label, mean, low, high, secondary))]
            color = GREY if secondary else TEAL
            ax.plot([float(low), float(high)], [y, y], color=color, linewidth=3, zorder=3)
            ax.plot([float(low), float(high)], [y, y], linestyle="none", marker="|",
                    markersize=12, markeredgewidth=2, color=color, zorder=3)
            ax.scatter(float(mean), y, s=95, edgecolor=color,
                       facecolor="white" if secondary else color, linewidth=2, zorder=4)
            bold = label.startswith("B′(A0)")
            ax.text(0.01, y + 0.32, label, transform=ax.get_yaxis_transform(),
                    fontsize=14.5, color="#7B8187" if secondary else INK,
                    fontweight="bold" if bold else "normal", va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none", pad=2))
            ax.text(0.01, y - 0.30, f"{mean} [{low}, {high}]",
                    transform=ax.get_yaxis_transform(), fontsize=14.5,
                    color="#67717C" if secondary else color, va="top",
                    fontweight="bold" if bold else "normal",
                    bbox=dict(facecolor="white", edgecolor="none", pad=2))
    fig.text(0.31, 0.83, "R@1: AFF minus comparator", ha="center", fontsize=17,
             fontweight="bold")
    fig.text(0.81, 0.83, "AFF condition gain minus comparator", ha="center",
             fontsize=17, fontweight="bold")
    fig.legend(handles=[
        Line2D([], [], marker="o", color=TEAL, linewidth=2, markersize=9,
               label="Pre-registered GO check"),
        Line2D([], [], marker="o", color=GREY, markerfacecolor="white",
               linewidth=2, markersize=9, label="Secondary comparison (not a GO check)"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=2,
        frameon=False, fontsize=15)
    save(fig, "go_checks")


def aff_scoring():
    dot = r'''digraph AFF {
  graph [rankdir=LR, bgcolor="white", dpi=200, pad=0.35,
         nodesep=0.45, ranksep=0.60, splines=polyline,
         fontname="DejaVu Sans", fontsize=20, labelloc=b,
         label=<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="10">
           <TR><TD COLSPAN="4">The two conditions of an episode see mirrored evidence, so the affect pick, and AFF's gate,</TD></TR>
           <TR><TD COLSPAN="4">opens mostly on one side (about 81% of first sides, 30% of second sides).</TD></TR>
           <TR><TD BGCOLOR="#F1F3F4"><FONT COLOR="#6B7279">Grey: same as earlier pipeline</FONT></TD>
               <TD BGCOLOR="#F1EDF8"><FONT COLOR="#8E7CC3">Purple: unchanged from gated reader R1</FONT></TD>
               <TD BGCOLOR="#FFF2E3"><FONT COLOR="#D9822B">Orange: replaced</FONT></TD>
               <TD BGCOLOR="#E7F6F2"><FONT COLOR="#1B8A7A">Teal: new in AFF</FONT></TD></TR>
         </TABLE>>];
  node [shape=box, style="rounded,filled", fontname="DejaVu Sans",
        fontsize=19, margin="0.20,0.17", color="#9AA0A6",
        fillcolor="#F1F3F4", fontcolor="#263444", penwidth=2];
  edge [color="#9AA0A6", arrowsize=0.8, penwidth=1.7,
        fontname="DejaVu Sans", fontsize=14];
  subgraph cluster_pipeline {
    label="How one-sided affect steering scores a candidate";
    labelloc=t; fontsize=27; fontname="DejaVu Sans";
    pencolor="white"; margin=20;
    query [label="Query\n(image or caption)"];
    examples [label="4 support pairs\n+ 4 contrast pairs"];
    candidates [label="13 candidates"];
    features [label="Frozen CLIP\nViT-B/32 features"];
    groupings [fontsize=17, label="Three label-free groupings\nof the training rows\n\naffect: Leiden communities of\nGoEmotions caption probabilities\n(41 groups)\n\nimage: k-means on CLIP\nimage features (64)\n\ncaption: k-means on CLIP\ncaption features (64)"];
    heads [label="Heads: logistic regressions\nplace any image or caption\nin each grouping"];
    agreement [color="#8E7CC3", fillcolor="#F1EDF8",
      label="Agreement of support pairs\nminus contrast pairs,\nper grouping"];
    reader [color="#8E7CC3", fillcolor="#F1EDF8",
      label="Learned reader: trained on\npractice episodes built\nfrom the groupings\n(no evaluation labels)\n-> probability per grouping"];
    score_t [color="#8E7CC3", fillcolor="#F1EDF8",
      label="Weighted grouping\nscore T"];
    gate_r1 [color="#D9822B", fillcolor="#FFF2E3",
      label="R1's gate: open if the\nreader is confident\n(top minus second\nprobability >= tau)"];
    gate_aff [color="#1B8A7A", fillcolor="#E7F6F2",
      label="AFF: and only if\nthe top grouping\nis affect"];
    base [label="Condition-free score B:\ncosine + factor term\n+ averaged heads\n(uses the examples,\nignores the aspect)"];
    output [color="#1B8A7A", fillcolor="#E7F6F2",
      label="Score = z(B) + λu·z(B)\n+ λa·gate·z(T)\n\ngate closed -> B alone"];
    {rank=same; query; examples; candidates;}
    {rank=same; features; groupings;}
    {rank=same; score_t; base; gate_r1;}
    query -> features;
    examples -> features;
    candidates -> features;
    groupings -> heads;
    features -> heads;
    heads -> agreement;
    agreement -> reader [color="#8E7CC3"];
    reader -> score_t [color="#8E7CC3"];
    heads -> score_t [color="#8E7CC3", label="grouping scores",
                      tailport=n, headport=n];
    reader -> gate_r1 [color="#D9822B"];
    gate_r1 -> gate_aff [color="#1B8A7A"];
    features -> base;
    heads -> base;
    score_t -> output [color="#8E7CC3"];
    gate_aff -> output [color="#1B8A7A"];
    base -> output;
  }
}'''
    source = OUT / "aff_scoring.dot"
    source.write_text(dot, encoding="utf-8")
    for suffix in ("png", "svg"):
        subprocess.run(["/root/miniconda3/bin/dot", f"-T{suffix}", str(source),
                        "-o", str(OUT / f"aff_scoring.{suffix}")], check=True)
    # Graphviz renders at graph dpi but omits PNG physical-resolution metadata.
    # Preserve its pixels and attach the same 200 dpi to the PNG header.
    png = OUT / "aff_scoring.png"
    with Image.open(png) as rendered:
        rendered.load()
        rendered.save(png, dpi=(DPI, DPI))


def ablation_ladder():
    steps = [
        ("Step 1 | Reader on the label-free groupings", [
            ("rule reader: pick the grouping with the largest\nsupport-minus-contrast agreement", "+0.313", "+0.102", "+0.528", "kept"),
            ("noise-scaled rule", "+0.230", "+0.028", "+0.437", "alt"),
            ("rule reader with the CSD style grouping added", "+0.008", "−0.225", "+0.243", "alt"),
        ]),
        ("Step 2 | Learned reader (trained on practice episodes)", [
            ("learned reader, probability-weighted grouping score", "+0.313", "+0.076", "+0.550", "kept"),
            ("learned reader, top pick only", "+0.144", "−0.033", "+0.327", "alt"),
        ]),
        ("Step 3 | Confidence gate", [
            ("steer only when the reader is confident (R1)", "+0.444", "+0.216", "+0.674", "kept"),
            ("reader adapted to real episodes, gated", "+0.116", "−0.073", "+0.301", "alt"),
            ("reader retrained on impure practice episodes, gated", "+0.077", "−0.111", "+0.271", "alt"),
            ("gated reader with the CSD style grouping added", "−0.010", "−0.169", "+0.156", "alt"),
        ]),
        ("Step 4 | One-sided steering", [
            ("steer only when the confident pick is affect (AFF)", "+0.700", "+0.460", "+0.937", "kept"),
            ("AFF plus an image-agreement veto", "+0.663", "+0.429", "+0.894", "alt"),
            ("random one-sided gate, draw 0\ncontrol, knows the side", "+0.665", None, None, "control"),
            ("random one-sided gate, draw 1\ncontrol, knows the side", "+0.564", None, None, "control"),
        ]),
    ]
    fig, ax = plt.subplots(figsize=(20, 13))
    fig.subplots_adjust(left=0.485, right=0.805, top=0.83, bottom=0.13)
    fig.suptitle("One-sided affect steering, built up one component at a time",
                 fontsize=24, fontweight="bold", y=0.965)
    fig.text(0.5, 0.913, "Development episodes (seed 42); margin = R@1 minus the strongest "
             "condition-free comparator; 95% intervals", ha="center", fontsize=15)
    ax.set_xlim(-0.4, 1.0)
    ax.set_ylim(-1.7, 20.0)
    ax.set_yticks([])
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0],
                  ["−0.4", "−0.2", "0", "+0.2", "+0.4", "+0.6", "+0.8", "+1.0"])
    ax.set_xlabel("Margin in R@1 points", labelpad=15)
    xgrid(ax)
    ax.axvline(0, color=INK, linewidth=1.5)
    ax.axvline(0.5, color=ORANGE, linestyle="--", linewidth=1.8)
    ax.text(0.5, 1.015, "development bar +0.5", ha="center",
            transform=ax.get_xaxis_transform(), color=ORANGE, fontsize=13)
    ax.text(1.045, 1.015, "Margin [95% interval]", fontsize=13,
            transform=ax.transAxes, color=INK, fontweight="bold")

    def draw_row(label, mean, low, high, kind, y):
        color = TEAL if kind in ("kept", "fresh") else SLATE if kind == "control" else GREY
        marker = "s" if kind == "fresh" else "D" if kind == "control" else "o"
        if low is not None:
            endpoints = [float(v.replace("−", "-")) for v in (low, high)]
            ax.plot(endpoints, [y, y], color=color, linewidth=2.5, zorder=3)
            ax.plot(endpoints, [y, y], marker="|", markersize=10,
                    markeredgewidth=1.7, linestyle="none", color=color, zorder=3)
        ax.scatter(float(mean.replace("−", "-")), y, marker=marker,
                   s=95 if kind in ("kept", "fresh") else 63,
                   facecolor="white" if kind == "alt" else color,
                   edgecolor=color, linewidth=1.8, zorder=4)
        ax.text(-0.035, y, label, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=13.5 if kind in ("kept", "fresh") else 12.5,
                color=INK if kind in ("kept", "fresh") else color,
                fontweight="bold" if kind in ("kept", "fresh") else "normal")
        value = f"{mean} [{low}, {high}]" if low is not None else f"{mean}\nno interval reported"
        ax.text(1.045, y, value, transform=ax.get_yaxis_transform(),
                va="center", fontsize=13, color=color,
                fontweight="bold" if kind in ("kept", "fresh") else "normal")

    y = 19.0
    for heading, rows in steps:
        ax.text(-0.035, y, heading, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=14.5, fontweight="bold", color=TEAL)
        y -= 1.1
        for row in rows:
            draw_row(*row, y)
            y -= 1.0
        y -= 0.45
    ax.axhline(y + 0.25, xmin=-1.45, xmax=1.60, clip_on=False,
               color="#BCC2C8", linewidth=1)
    y -= 0.45
    draw_row("AFF on fresh episodes (seeds 49 to 51, pooled)",
             "+0.591", "+0.462", "+0.729", "fresh", y)
    fig.legend(handles=[
        Line2D([], [], marker="o", color=TEAL, linestyle="none", markersize=9, label="kept in AFF"),
        Line2D([], [], marker="o", color=GREY, markerfacecolor="white", linestyle="none",
               markeredgewidth=1.8, markersize=8, label="alternative tried"),
        Line2D([], [], marker="D", color=SLATE, linestyle="none", markersize=8,
               label="mechanism control (not a method)"),
        Line2D([], [], marker="s", color=TEAL, linestyle="none", markersize=9, label="fresh test"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=4, frameon=False, fontsize=14)
    save(fig, "ablation_ladder")


def aff_design():
    import html
    from matplotlib.font_manager import FontProperties, findfont
    from PIL import ImageFont

    # Fixed Graphviz positions keep the pipeline and alternatives in clean rows.
    # Explicit cubic segments keep long scoring edges clear of the boxes.
    dot = [r'''digraph AFF_design {
  graph [layout=neato, bgcolor="white", dpi=200, pad=0.25,
         overlap=true, splines=true, fontname="DejaVu Sans", fontsize=25,
         labelloc=t, label="One-sided affect steering as one design, and where each component came from"];
  node [shape=box, style="rounded,filled", fontname="DejaVu Sans",
        fontsize=16, margin="0.10,0.10", color="#1B8A7A", fillcolor="white",
        fontcolor="#263444", penwidth=2, width=2.5, height=5.15, fixedsize=true];
  edge [color="#1B8A7A", penwidth=1.7, arrowsize=0.7,
        fontname="DejaVu Sans", fontsize=15];''']

    def lines(text, size=16, *, bold=False, width=148):
        font = ImageFont.truetype(findfont(FontProperties(
            family="DejaVu Sans", weight="bold" if bold else "normal")), size * 10)
        wrapped = []
        for paragraph in text.splitlines():
            line = ""
            for word in paragraph.split():
                trial = f"{line} {word}" if line else word
                if line and font.getlength(trial) / 10 > width:
                    wrapped.append(line)
                    line = word
                else:
                    line = trial
            wrapped.append(line)
        return '<BR ALIGN="LEFT"/>'.join(html.escape(part) for part in wrapped) + '<BR ALIGN="LEFT"/>'

    def box(key, name, body, provenance, x, y, *, base=False, fused=False, height=5.15):
        color = "#9AA0A6" if base else "#1B8A7A"
        fill = "#F2F3F4" if base else "#1B8A7A" if fused else "white"
        ink = "white" if fused else "#263444"
        prov = (f'<TR><TD ALIGN="LEFT"><FONT COLOR="#747B82" POINT-SIZE="13">'
                f'{lines(provenance, 13)}</FONT></TD></TR>') if provenance else ""
        label = (f'<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="5">'
                 f'<TR><TD ALIGN="LEFT"><FONT COLOR="{ink}" POINT-SIZE="17"><B>{lines(name, 17, bold=True)}</B></FONT></TD></TR>'
                 f'<TR><TD ALIGN="LEFT"><FONT COLOR="{ink}" POINT-SIZE="16">{lines(body)}</FONT></TD></TR>'
                 f'{prov}</TABLE>')
        dot.append(f'{key} [pos="{x},{y}!", height={height}, color="{color}", fillcolor="{fill}", label=<{label}>];')

    xs = [110 + 198 * i for i in range(8)]
    box("inputs", "Inputs", "query; 4 support and 4 contrast pairs; 13 candidates; frozen CLIP ViT-B/32 features",
        "", xs[0], 500)
    box("groupings", "Groupings (label-free)",
        "affect: Leiden communities on GoEmotions caption probabilities (41 groups); image and caption: k-means on CLIP features (64 each)",
        "from: grouping study, 5 Oct; evidence: Leiden told margin +1.64 vs k-means +1.10 at 41 groups", xs[1], 500)
    box("heads", "Heads", "logistic regressions on CLIP features (60,000 training rows) place any image or caption in each grouping; agreement = image posterior · caption posterior",
        "from: partition-head reader, 4 Oct; evidence: told the right grouping, the heads beat their control by +1.14", xs[2], 500)
    box("reader_features", "Reader features", "per grouping: support agreement, contrast agreement, their difference, two spreads, a match share (18 numbers)",
        "", xs[3], 500)
    box("reader", "Learned reader", "multinomial logistic regression trained on practice episodes built from the groupings (49,152 per painting half; no evaluation labels) gives a probability per grouping",
        "from: reader round 1, 6 Oct; evidence: 79% right on practice episodes, 51.3% on real ones", xs[4], 500)
    box("gate", "Confidence gate", "open when the top probability leads the second by at least tau (a percentile of the development margins)",
        "from: reader round 1, 6 Oct; evidence: +0.444 vs +0.313 without the gate", xs[5], 500)
    box("rule", "One-sided rule", "steer only when the top grouping is affect, the grouping least redundant with B",
        "from: brainstorm and round 3, 6 Oct; evidence: +0.700 development, +0.591 fresh", xs[6], 500)
    box("base", "Condition-free score B", "cosine + centered factor term + averaged heads; uses the examples, ignores the aspect",
        "from: method A and partition heads, 3 to 4 Oct; 18.34 R@1 on development episodes", xs[0], 150, base=True, height=3.8)
    box("fused", "Fused score", "(1 + λu)·z(B) + λa·gate·z(T); weights from 224 cells, chosen on one parity half of the episodes and applied to the other",
        "", xs[7], 500, fused=True)

    pipeline = ["inputs", "groupings", "heads", "reader_features", "reader", "gate", "rule", "fused"]
    for index, (left, right) in enumerate(zip(pipeline, pipeline[1:])):
        start, end = xs[index] + 90, xs[index + 1] - 90
        dot.append(f'{left} -> {right} [pos="e,{end},500 {start},500 {start + 6},500 {end - 12},500 {end - 7},500"];')
    dot.append('inputs -> base [color="#9AA0A6", pos="e,110,286.8 110,314.6 110,307 110,301 110,294"];')
    dot.append('heads -> fused [label="grouping score T", lp="1010,772", pos="e,1461,685.4 506,685.4 506,710 506,750 506,750 506,750 1461,750 1461,750 1461,750 1461,710 1461,693"];')
    dot.append('base -> fused [color="#9AA0A6", pos="e,1528,685.4 20,150 -10,150 -10,150 -10,150 -10,150 -10,820 -10,820 -10,820 1528,820 1528,820 1528,820 1528,720 1528,693"];')

    alternatives = [
        ("km", "k-means affect grouping: told +1.10, reader +0.16", "groupings", xs[1]),
        ("csd", "CSD style grouping: told ceiling +2.23, but reader +0.06", "groupings", xs[2]),
        ("leiden", "Leiden for image and caption: reader −0.32", "groupings", xs[3]),
        ("other_readers", "noise-scaled rule: +0.230; adapted reader: +0.116; retrained reader: +0.077", "reader", xs[4]),
        ("veto", "vetoes on AFF (round 4): none beat AFF (net −18, −18, −34 rankings)", "rule", xs[6]),
    ]
    dot.append('tried [shape=plain, fixedsize=false, width=0, height=0, pos="110,-110!", label=<<FONT POINT-SIZE="18" COLOR="#747B82"><B>Tried and<BR/>left out</B></FONT>>];')
    for key, body, target, x in alternatives:
        dot.append(f'{key} [pos="{x},-110!", width=2.45, height=2.0, style="rounded,dashed,filled", color="#9AA0A6", fillcolor="white", label=<<FONT POINT-SIZE="15" COLOR="#747B82">{lines(body, 15, width=152)}</FONT>>];')
        tx = xs[pipeline.index(target)]
        tx += -28 if key == "km" else 28 if key == "leiden" else 0
        lane = 235 if key == "km" else 260 if key == "csd" else 285 if key == "leiden" else 290
        dot.append(f'{target} -> {key} [style=dotted, color="#9AA0A6", arrowhead=none, pos="{tx},314.6 {tx},{lane} {tx},{lane} {tx},{lane} {tx},{lane} {x},{lane} {x},{lane} {x},{lane} {x},-20 {x},-38"];')
    dot.append('legend [shape=plain, fixedsize=false, width=0, height=0, pos="800,-245!", label=<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="15"><TR><TD><FONT COLOR="#1B8A7A" POINT-SIZE="16">teal border = component of AFF</FONT></TD><TD><FONT COLOR="#747B82" POINT-SIZE="16">grey = shared condition-free base</FONT></TD><TD><FONT COLOR="#747B82" POINT-SIZE="16">dashed grey = tried and left out</FONT></TD></TR></TABLE>>];')
    dot.append("}")
    source = OUT / "aff_design.dot"
    source.write_text("\n".join(dot), encoding="utf-8")
    for suffix in ("png", "svg"):
        subprocess.run(["/root/miniconda3/bin/dot", "-Kneato", "-n2", f"-T{suffix}", str(source),
                        "-o", str(OUT / f"aff_design.{suffix}")], check=True)
    png = OUT / "aff_design.png"
    with Image.open(png) as rendered:
        rendered.load()
        rendered.save(png, dpi=(DPI, DPI))


def setup_overview():
    """Introduce the task with schematic pairs rather than scores or formulas."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    # Coordinates are display pixels, so text remains legible at the target size.
    fig = plt.figure(figsize=(8, 3.65), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1600), ylim=(0, 730))
    ax.set_axis_off()

    def text(x, y, label, size=6.5, **kwargs):
        return ax.text(x, y, label, fontsize=size, va="center",
                       linespacing=1.25, **kwargs)

    def box(x, y, width, height, color, *, fill="white", rounded=True):
        if rounded:
            patch = FancyBboxPatch(
                (x, y), width, height,
                boxstyle="round,pad=0,rounding_size=7",
                edgecolor=color, facecolor=fill, linewidth=0.7)
        else:
            patch = Rectangle((x, y), width, height, edgecolor=color,
                              facecolor=fill, linewidth=0.7)
        ax.add_patch(patch)
        return patch

    def image_tile(x, y, width, height, color):
        box(x, y, width, height, color, fill=LIGHT, rounded=False)
        text(x + width / 2, y + height / 2, "image", 5.6,
             ha="center", color=INK)

    def pair_group(heading, heading_y, row_ys, color, fill, description):
        text(44, heading_y, heading, 7.0, color=color, fontweight="bold")
        for y in row_ys:
            image_tile(44, y, 60, 28, color)
            box(115, y, 190, 28, color, fill=fill)
            text(210, y + 14, "[caption, other painting]", 5.6,
                 ha="center", color=INK)
        bottom, top = min(row_ys) - 3, max(row_ys) + 31
        ax.plot([316, 326, 326, 316], [top, top, bottom, bottom],
                color=color, linewidth=0.85)
        text(343, (top + bottom) / 2, description, 6.5, color=color)

    text(800, 687,
         "Conditional similarity shown by examples: the task at a glance",
         9.5, ha="center", fontweight="bold")
    for x, width, heading in [
        (32, 518, "What the user gives"),
        (610, 432, "What the system does"),
        (1100, 468, "What comes out"),
    ]:
        text(x, 625, heading, 8.0, fontweight="bold")
        ax.plot([x, x + width], [604, 604], color=GREY, linewidth=0.6)

    text(44, 580, "Query", 7.0, fontweight="bold")
    image_tile(44, 521, 68, 45, SLATE)
    text(128, 544, "a painting (or a caption)", 7.0)
    pair_group("4 example pairs", 490, [449, 416, 383, 350],
               TEAL, "#E7F6F2",
               "alike in one respect\n(here: the feeling\nthey evoke)")
    pair_group("4 contrast pairs", 290, [249, 216, 183, 150],
               ORANGE, "#FFF2E3",
               "alike in another\nrespect (here:\nart style)")
    text(44, 113, "the respect is never named, and the examples\nnever show the query's own value", 6.0,
         fontstyle="italic", color=SLATE)

    box(610, 185, 432, 380, SLATE, fill="#F6F7FA")
    # Ordinal words preserve the restriction on numeric text in this overview.
    text(633, 529, "First", 7.2, color=SLATE, fontweight="bold")
    text(633, 465,
         "read which respect the example\npairs share, and which the\ncontrast pairs share",
         7.0)
    text(633, 375, "Second", 7.2, color=SLATE, fontweight="bold")
    text(633, 325,
         "compare the query with every\ncandidate in that respect", 7.0)
    ax.plot([633, 1019], [269, 269], color=GREY, linewidth=0.5)
    text(633, 226,
         "frozen image and caption encoders;\nno labels of the evaluated respects", 6.2,
         color=SLATE)
    for start, end in [(557, 601), (1050, 1091)]:
        ax.add_patch(FancyArrowPatch(
            (start, 362), (end, 362), arrowstyle="-|>",
            mutation_scale=8, linewidth=0.9, color=SLATE))

    text(1100, 580, "13 candidate captions, ranked", 7.0)
    rows = [
        ("first", 500, 62, TEAL, "#E7F6F2",
         "shares the query's feeling (the target)"),
        ("second", 445, 42, GREY, LIGHT, "shares neither"),
        ("third", 390, 42, GREY, LIGHT, "shares neither"),
        ("fourth", 315, 62, ORANGE, "#FFF2E3",
         "shares the query's style (the other respect)"),
        ("fifth", 260, 42, GREY, LIGHT, "shares neither"),
        ("sixth", 205, 42, GREY, LIGHT, "shares neither"),
    ]
    for rank, y, height, color, fill, description in rows:
        text(1100, y + height / 2, rank, 6.2, color=color,
             fontweight="bold" if color != GREY else "normal")
        box(1180, y, 388, height, color, fill=fill)
        text(1193, y + height - 13, "[caption of another painting]", 5.8)
        if color == TEAL:
            description = "shares the query's feeling\n(the target)"
        elif color == ORANGE:
            description = "shares the query's style\n(the other respect)"
        text(1193, y + (23 if height == 62 else 11), description,
             6.3 if height == 62 else 5.8,
             color=color if color != GREY else SLATE)
    text(1374, 188, "...", 7.5, ha="center", color=GREY)

    box(1100, 102, 468, 76, ORANGE, fill="#FFF9F1")
    ax.add_patch(FancyArrowPatch(
        (1121, 128), (1157, 149), connectionstyle="arc3,rad=-0.8",
        arrowstyle="-|>", mutation_scale=8, linewidth=0.9, color=ORANGE))
    text(1179, 140,
         "swap examples and contrasts:\nthe style candidate should\nnow come first", 6.2)

    box(32, 24, 1536, 57, GREY, fill=LIGHT, rounded=False)
    text(50, 64,
         "Benchmark: ArtELingo, WikiArt paintings with viewer captions. "
         "Respects scored: emotion, art style, genre.", 6.1)
    text(50, 41,
         "Both directions: an image query ranks captions, "
         "a caption query ranks images.", 6.1)
    fig.savefig(OUT / "setup_overview.png", dpi=DPI * 2.5)  # same layout, resolution matched to the other figures
    plt.close(fig)


def aff_structure():
    """Model structure of one-sided affect steering, in the style of the CoSiR v2 design figure."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig = plt.figure(figsize=(17, 9.6), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1700), ylim=(0, 960))
    ax.set_axis_off()
    styles = {
        "frozen": ("#6B4FA0", "#EFEAF8"),
        "reused": ("#8A9097", "#F1F2F4"),
        "new": (TEAL, "#E4F4F1"),
        "aff": (TEAL, "#C9EBE4"),
        "input": ("#8A9097", "#FFFFFF"),
    }

    def node(x, y, w, h, kind, title, body, lw=1.6, dashed=False):
        edge, fill = styles[kind]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=12",
                                    linewidth=lw, edgecolor=edge, facecolor=fill,
                                    linestyle="--" if dashed else "-"))
        ax.text(x + w / 2, y + h - 26, title, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 48, body, ha="center", va="top", fontsize=8.2, color=INK,
                linespacing=1.35)

    def arrow(x0, y0, x1, y1, label=None, color=INK, style="-", lx=0, ly=10):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.3, color=color, linestyle=style,
                                     shrinkA=0, shrinkB=0))
        if label:
            ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label, ha="center", va="bottom",
                    fontsize=8, color="#4A5563")

    ax.text(850, 935, "One-sided affect steering: model structure", ha="center", va="center",
            fontsize=16, fontweight="bold", color=INK)
    ax.text(850, 905, "score an image and a caption under a condition shown by example pairs; "
            "no evaluation labels anywhere", ha="center", va="center", fontsize=10, color="#4A5563")

    # inputs
    node(30, 690, 230, 160, "input", "Episode",
         "query (image or caption)\n4 support pairs\n4 contrast pairs\n13 candidates\n(other modality)")
    node(30, 505, 230, 120, "frozen", "CLIP image encoder", "frozen ViT-B/32\n512-d feature")
    node(30, 350, 230, 120, "frozen", "CLIP text encoder", "frozen\n512-d feature")
    arrow(145, 690, 145, 625)
    arrow(260, 565, 330, 565)
    arrow(260, 410, 330, 470)

    # condition-free base (top lane)
    node(330, 690, 600, 160, "reused", "Condition-free base score B",
         "CLIP cosine of query and candidate\n+ centred factor term (method A's factors, uniform weights)\n"
         "+ head agreement averaged over the groupings\nuses the examples, ignores which aspect they show")
    # steering lane
    node(330, 380, 280, 250, "new", "Label-free groupings",
         "built once from the training rows\n\naffect: Leiden communities on\neach caption's GoEmotions\n"
         "emotion scores (41 groups)\n\nimage: k-means on CLIP image (64)\ncaption: k-means on CLIP text (64)")
    node(650, 380, 280, 250, "new", "Heads",
         "one logistic regression per\ngrouping and modality\n(fitted on 60,000 training rows)\n\n"
         "image or caption feature\n-> probability over groups\n\nagreement of two items =\noverlap of their probabilities")
    node(970, 380, 270, 250, "new", "Reader features",
         "for each grouping:\nsupport-pair agreement\ncontrast-pair agreement\ntheir difference\ntwo spreads\n"
         "match share\n\n18 numbers per episode\nand condition")
    node(1280, 380, 390, 250, "new", "Learned reader",
         "multinomial logistic regression\ntrained on practice episodes built\nfrom the groupings (answer known,\n"
         "no labels; 49,152 per painting half)\n\noutput: probability per grouping\nP(affect), P(image), P(caption)")
    arrow(610, 505, 650, 505)
    arrow(930, 505, 970, 505)
    arrow(1240, 505, 1280, 505)
    arrow(260, 610, 420, 690, color="#8A9097")
    arrow(260, 770, 330, 770, color="#8A9097")

    # gate and score (bottom lane)
    node(1280, 130, 390, 200, "aff", "Gate (the one-sided rule)",
         "open only if\n1. the reader is confident: top probability\n    leads the second by at least a threshold\n"
         "2. the top grouping is affect\n\nopens mostly on one side of an episode", lw=2.6)
    node(650, 130, 590, 200, "new", "Grouping score T",
         "mix of the three grouping scores, weighted by the\nreader's probabilities:\n"
         "for each grouping, overlap of the query's and the\ncandidate's group probabilities, times P(grouping)")
    node(30, 130, 580, 200, "new", "Fused score",
         "z-scored base B, plus the z-scored grouping score T\nwhen the gate is open; B alone when it is closed\n\n"
         "two weights and the threshold are chosen on one half\nof the episodes and applied to the other half\n"
         "-> ranking of the 13 candidates")
    arrow(1475, 380, 1475, 330)
    arrow(790, 380, 790, 330, label="group probabilities", lx=-80, ly=-8)
    arrow(1280, 230, 1240, 230)
    arrow(650, 230, 610, 230)
    arrow(1380, 380, 1150, 330, color=TEAL, style="--", label="P(grouping)", lx=40, ly=-14)
    ax.plot([330, 300, 300], [720, 720, 360], color="#8A9097", linewidth=1.3)
    arrow(300, 360, 300, 330, color="#8A9097")
    ax.text(292, 520, "base B", rotation=90, ha="right", va="center", fontsize=8, color="#4A5563")

    # legend and example
    ly = 60
    for i, (kind, label) in enumerate([("frozen", "frozen CLIP"), ("reused", "condition-free base"),
                                       ("new", "label-free steering branch"),
                                       ("aff", "the one-sided rule that defines AFF")]):
        edge, fill = styles[kind]
        x = 40 + i * 300
        ax.add_patch(FancyBboxPatch((x, ly - 12), 36, 24, boxstyle="round,pad=0,rounding_size=5",
                                    linewidth=2.2 if kind == "aff" else 1.4, edgecolor=edge, facecolor=fill))
        ax.text(x + 48, ly, label, ha="left", va="center", fontsize=9.5, color=INK)
    ax.text(850, 18, "Example: the support pairs share a feeling and the contrast pairs a style; the reader picks "
            "affect with high confidence, the gate opens, and candidates whose groups match the query's rise.",
            ha="center", va="center", fontsize=9, fontstyle="italic", color="#4A5563")
    fig.savefig(OUT / "aff_structure.png", dpi=DPI)
    plt.close(fig)


SLIDES_OUT = OUT / "slides"


def _slide_save(fig, name):
    """Keep the slide canvas and physical font sizes fixed, with no tight crop."""
    SLIDES_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(SLIDES_OUT / f"{name}.png", dpi=DPI, facecolor="white")
    plt.close(fig)


def _slide_canvas():
    fig = plt.figure(figsize=(13, 5), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 13), ylim=(0, 5))
    ax.set_axis_off()
    return fig, ax


def _slide_box(ax, x, y, width, height, color, fill="white", linewidth=1.5):
    from matplotlib.patches import FancyBboxPatch
    patch = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0,rounding_size=0.08",
        edgecolor=color, facecolor=fill, linewidth=linewidth)
    ax.add_patch(patch)
    return patch


def slide_pitfall():
    """Examples solve the old test; the factor model fails on the fair test."""
    fig = plt.figure(figsize=(13, 5), dpi=DPI)
    panels = [
        (0.045, "Old test: the examples share\nthe query's value", [
            ("plain CLIP", 13.45, GREY, False),
            ("factor model", 21.22, TEAL, False),
            ("examples only, query ignored", 22.83, SLATE, False),
            ("examples plus query", 23.40, SLATE, False),
        ], "the query adds only +0.57"),
        (0.535, "Fair test: the examples show\nother values", [
            ("plain CLIP", 11.13, GREY, False),
            ("factor model", 11.32, TEAL, False),
            ("label probes told the respect (ceiling)", 23.09, "white", True),
        ], "the factor model stays at CLIP level"),
    ]
    fig.text(0.045, 0.95, "Old test against fair test", fontsize=23,
             fontweight="bold", va="top")
    for left, title, rows, note in panels:
        fig.text(left, 0.84, title, fontsize=17, fontweight="bold", va="top")
        ax = fig.add_axes([left, 0.19, 0.42, 0.49])
        ax.set(xlim=(0, 28), ylim=(-0.45, 3.9), xticks=[], yticks=[])
        ax.spines[["left", "right", "top", "bottom"]].set_visible(False)
        # Identical row positions and scales make the panels directly comparable.
        for i, (label, value, color, ceiling) in enumerate(rows):
            y = 3.25 - i
            ax.text(0, y + 0.26, label, fontsize=14, va="bottom")
            ax.barh(y, value, height=0.30, color=color,
                    edgecolor=GREY if ceiling else color,
                    hatch="///" if ceiling else None, linewidth=1.2)
            ax.text(value + 0.35, y, f"{value:.2f}", fontsize=17,
                    fontweight="bold", va="center")
        fig.text(left, 0.14, "top-1 accuracy (%)", fontsize=13, color=SLATE)
        fig.text(left, 0.055, note, fontsize=16, fontweight="bold", color=ORANGE)
    _slide_save(fig, "pitfall")


def slide_structure():
    """Two clean scoring paths, with the gate as the visual focus."""
    from matplotlib.patches import FancyArrowPatch
    fig, ax = _slide_canvas()
    ax.text(0.35, 4.65, "One-sided affect steering", fontsize=23, fontweight="bold")

    def node(x, y, width, title, line, color=TEAL, fill="#E7F6F2", strong=False,
             title_size=17, body_size=12.5):
        _slide_box(ax, x, y, width, 0.88, color, fill, 3 if strong else 1.6)
        ax.text(x + width / 2, y + 0.59, title, fontsize=title_size, fontweight="bold",
                ha="center", va="center", color=color if strong else INK)
        if line:
            ax.text(x + width / 2, y + 0.25, line, fontsize=body_size,
                    ha="center", va="center")

    def route(points, color):
        if len(points) > 2:
            ax.plot(*zip(*points[:-1]), color=color, linewidth=1.6)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                     mutation_scale=15, color=color, linewidth=1.6,
                                     shrinkA=0, shrinkB=2))

    node(0.3, 2.20, 1.7, "CLIP features", "frozen", SLATE, LIGHT, title_size=15)
    node(2.25, 3.40, 9.95, "Base score",
         "ranks candidates using the examples, ignoring which respect they show", SLATE, LIGHT)
    node(2.25, 1.82, 1.45, "Groupings", "", fill="white", title_size=16)
    node(4.00, 1.82, 1.25, "Heads", "", fill="white")
    node(5.55, 1.82, 3.55, "Reader", "which grouping do the examples share?",
         fill="white", body_size=12)
    node(9.40, 1.82, 2.80, "Gate", "confident and emotion-like?", strong=True)
    node(2.25, 0.35, 9.95, "Final score",
         "base score, plus the steering score only when the gate opens")
    ax.text(2.25, 2.95, "Label-free steering branch", fontsize=14,
            fontweight="bold", color=TEAL)
    route([(2.0, 2.79), (2.12, 2.79), (2.12, 3.84), (2.25, 3.84)], SLATE)
    route([(2.0, 2.49), (2.12, 2.49), (2.12, 2.26), (2.25, 2.26)], TEAL)
    for left, right in [(3.70, 4.00), (5.25, 5.55), (9.10, 9.40)]:
        route([(left, 2.26), (right, 2.26)], TEAL)
    route([(12.2, 3.84), (12.65, 3.84), (12.65, 0.79), (12.2, 0.79)], SLATE)
    route([(10.80, 1.82), (10.80, 1.23)], TEAL)
    _slide_save(fig, "structure")


def slide_fresh_test():
    """The fresh-test gain is the headline; all requested scorers remain visible."""
    rows = [
        ("plain CLIP", 13.04, GREY),
        ("best metric learned from the example pairs", 13.16, GREY),
        ("base score", 18.07, SLATE),
        ("its matched control", 18.08, SLATE),
        ("strongest condition-free comparator", 18.29, SLATE),
        ("confidence-gated reader", 18.68, TEAL),
        ("one-sided affect steering", 18.88, TEAL),
    ]
    fig = plt.figure(figsize=(13, 5), dpi=DPI)
    fig.text(0.045, 0.93, "Fresh test", fontsize=23, fontweight="bold")
    fig.text(0.045, 0.86, "36,864 fresh episodes", fontsize=14, color=SLATE)
    fig.text(0.52, 0.93, "Steering gain: +0.59 points", fontsize=22,
             fontweight="bold", color=ORANGE)
    fig.text(0.52, 0.86, "[+0.46, +0.73] over the strongest condition-free comparator",
             fontsize=13, color=ORANGE)
    ax = fig.add_axes([0.435, 0.15, 0.49, 0.64])
    baseline = 0
    for i, (label, value, color) in enumerate(rows):
        y = 6 - i
        ax.barh(y, value - baseline, left=baseline, height=0.57, color=color,
                edgecolor=ORANGE if i in (4, 6) else color,
                linewidth=2 if i in (4, 6) else 0)
        ax.text(value + 0.10, y, f"{value:.2f}", fontsize=16, va="center",
                fontweight="bold", color=TEAL if i == 6 else INK)
    ax.set(xlim=(baseline, 22.83), ylim=(-0.55, 6.55), xticks=[], yticks=[])
    for i, (label, _, _) in enumerate(rows):
        ax.text(-0.03, 6 - i, label, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=13.5,
                fontweight="bold" if i == 6 else "normal")
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.set_xlabel("top-1 accuracy (%)  |  axis starts at zero", fontsize=13, labelpad=10)
    ax.plot([21.90, 22.10, 22.10, 21.90], [2, 2, 0, 0], color=ORANGE, linewidth=1.8)
    _slide_save(fig, "fresh_test")


def slide_buildup():
    """Only the five requested steps, with fresh episodes separated by color."""
    steps = [
        ("rule reader", 0.31, TEAL),
        ("learned reader", 0.31, TEAL),
        ("+ confidence\ngate", 0.44, TEAL),
        ("+ one-sided rule\n(development\nepisodes)", 0.70, TEAL),
        ("one-sided affect\nsteering on\nfresh episodes", 0.59, ORANGE),
    ]
    fig = plt.figure(figsize=(13, 5), dpi=DPI)
    fig.text(0.045, 0.93, "Building the method up", fontsize=23, fontweight="bold")
    fig.text(0.045, 0.85, "Margin over the strongest condition-free comparator (points)",
             fontsize=14, color=SLATE)
    ax = fig.add_axes([0.065, 0.23, 0.89, 0.54])
    for i, (_, value, color) in enumerate(steps):
        ax.bar(i, value, color=color, width=0.52)
        ax.text(i, value + 0.025, f"+{value:.2f}", ha="center", va="bottom",
                fontsize=21, fontweight="bold", color=color, zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=2))
    ax.axhline(0.5, color=SLATE, linestyle="--", linewidth=1.4, zorder=0)
    ax.text(-0.48, 0.52, "development bar +0.5", fontsize=13, color=SLATE,
            bbox=dict(facecolor="white", edgecolor="none", pad=2))
    ax.set(xlim=(-0.6, 4.6), ylim=(0, 0.85), yticks=[0], yticklabels=["0"])
    ax.set_xticks(range(len(steps)), [step[0] for step in steps], fontsize=14)
    ax.tick_params(axis="both", length=0, pad=10, labelsize=14)
    ax.spines[["left", "top", "right"]].set_visible(False)
    _slide_save(fig, "buildup")


def slide_value_vs_aspect():
    """The old examples reveal the value; the new examples identify the respect."""
    fig, ax = _slide_canvas()
    columns = [(0.35, "Value episode (old)"), (6.85, "Aspect episode (new)")]
    for x, heading in columns:
        ax.text(x, 4.65, heading, fontsize=22, fontweight="bold")
        _slide_box(ax, x, 3.90, 5.8, 0.48, SLATE, LIGHT)
        ax.text(x + 2.9, 4.14, "query: a sad painting", fontsize=16,
                ha="center", va="center")

    ax.text(0.35, 3.54, "Examples", fontsize=14, fontweight="bold", color=TEAL)
    for i in range(4):
        x = 0.35 + i * 1.5
        _slide_box(ax, x, 2.98, 1.3, 0.42, TEAL, "#E7F6F2")
        ax.text(x + 0.65, 3.19, "sad", fontsize=15, color=TEAL,
                ha="center", va="center")

    def pair_row(x, top, labels, color, fill):
        for i, label in enumerate(labels):
            px = x + i * 1.5
            for dx in (0.23, 0.73):
                _slide_box(ax, px + dx, top - 0.25, 0.38, 0.27, color, fill)
            ax.plot([px + 0.61, px + 0.73], [top - 0.115, top - 0.115],
                    color=color, linewidth=1.2)
            ax.text(px + 0.65, top - 0.40, label, fontsize=12.5, color=color,
                    ha="center", va="center")

    ax.text(6.85, 3.54, "Example pairs: emotion", fontsize=14,
            fontweight="bold", color=TEAL)
    pair_row(6.85, 3.30, ["calm", "fear", "joy", "awe"], TEAL, "#E7F6F2")
    ax.text(6.85, 2.58, "Contrast pairs: art style", fontsize=14,
            fontweight="bold", color=ORANGE)
    pair_row(6.85, 2.33, ["Baroque", "Cubism", "Impressionism", "Ukiyo-e"],
             ORANGE, "#FFF2E3")

    for x, _ in columns:
        ax.text(x, 1.62, "Candidates", fontsize=14, fontweight="bold")
        _slide_box(ax, x, 1.10, 5.8, 0.36, TEAL, "#E7F6F2")
        ax.text(x + 0.15, 1.28, "sad (target)", fontsize=14,
                fontweight="bold", color=TEAL, va="center")
    _slide_box(ax, 0.35, 0.66, 5.8, 0.36, GREY, LIGHT)
    ax.text(0.50, 0.84, "other candidates", fontsize=14, color=SLATE, va="center")
    _slide_box(ax, 6.85, 0.66, 5.8, 0.36, ORANGE, "#FFF2E3")
    ax.text(7.0, 0.84, "same style as the query (distractor)", fontsize=14,
            color=ORANGE, va="center")
    ax.text(0.35, 0.28, "the examples already show the answer;\nthe query is barely needed",
            fontsize=15, fontweight="bold", va="center", linespacing=1.25)
    ax.text(6.85, 0.28, "the examples show which respect matters;\nthe system must apply it to the query",
            fontsize=15, fontweight="bold", va="center", linespacing=1.25)
    _slide_save(fig, "value_vs_aspect")


def slide_dataset():
    """One split bar and three cards describe ArtELingo at a glance."""
    fig, ax = _slide_canvas()
    ax.text(0.35, 4.65, "ArtELingo at a glance", fontsize=23, fontweight="bold")
    ax.text(0.35, 4.03, "61,402 paintings", fontsize=22, fontweight="bold")
    splits = [
        ("training", 36518, TEAL, "36,518"),
        ("development and fresh test", 6451, ORANGE, "6,451"),
        ("held for the paper", 12281, SLATE, "12,281"),
        ("not used here", 61402 - 36518 - 6451 - 12281, GREY, None),
    ]
    from matplotlib.patches import Rectangle
    left = 0.35
    for _, count, color, _ in splits:
        width = 6.15 * count / 61402
        ax.add_patch(Rectangle((left, 3.15), width, 0.52,
                               facecolor=color, edgecolor="white", linewidth=2))
        left += width
    for i, (label, _, color, value) in enumerate(splits):
        y = 2.65 - i * 0.47
        ax.add_patch(Rectangle((0.35, y - 0.07), 0.17, 0.17, facecolor=color))
        ax.text(0.70, y, label, fontsize=16, va="center")
        if value:
            ax.text(6.50, y, value, fontsize=18, fontweight="bold", ha="right", va="center")
    cards = [
        ("Emotion: 8 values", "labelled per caption: the writer's feeling", TEAL, "#E7F6F2"),
        ("Art style: 23 values", "per painting", ORANGE, "#FFF2E3"),
        ("Genre: 10 values", "per painting; about 81% of paintings", SLATE, LIGHT),
    ]
    for i, (title, line, color, fill) in enumerate(cards):
        y = 3.25 - i * 1.12
        _slide_box(ax, 7.10, y, 5.5, 0.92, color, fill)
        ax.text(7.35, y + 0.60, title, fontsize=18, fontweight="bold", color=color)
        ax.text(7.35, y + 0.23, line, fontsize=14)
    ax.text(6.5, 0.27,
            "308,723 image and caption pairs; captions written by viewers of WikiArt paintings",
            fontsize=14, ha="center", va="center")
    _slide_save(fig, "dataset")


BUILDERS = {
    "factor_line_end": factor_line_end,
    "modality_asymmetry": modality_asymmetry,
    "methods_vs_controls": methods_vs_controls,
    "go_checks": go_checks,
    "aff_scoring": aff_scoring,
    "ablation_ladder": ablation_ladder,
    "aff_design": aff_design,
    "setup_overview": setup_overview,
    "aff_structure": aff_structure,
    "slide_pitfall": slide_pitfall,
    "slide_structure": slide_structure,
    "slide_fresh_test": slide_fresh_test,
    "slide_buildup": slide_buildup,
    "slide_value_vs_aspect": slide_value_vs_aspect,
    "slide_dataset": slide_dataset,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", nargs="?", help="Figure name, optionally ending in .png")
    args = parser.parse_args()
    name = args.figure.removesuffix(".png") if args.figure else None
    if name and name not in BUILDERS:
        parser.error("Unknown figure. Choose: " + ", ".join(BUILDERS))
    if not name or not name.startswith("slide_"):
        OUT.mkdir(parents=True, exist_ok=True)
    for selected in ([name] if name else BUILDERS):
        BUILDERS[selected]()
        if selected.startswith("slide_"):
            output = SLIDES_OUT / f"{selected.removeprefix('slide_')}.png"
        else:
            output = OUT / f"{selected}.png"
        print(output)


if __name__ == "__main__":
    main()
