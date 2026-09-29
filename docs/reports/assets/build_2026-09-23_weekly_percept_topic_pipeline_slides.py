#!/usr/bin/env python3
"""Build the presenter-oriented 2026-09-23 PercepT weekly-report deck."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "2026-09-23_weekly_percept_topic_pipeline_slides.pptx"
DIAGRAMS = Path(__file__).resolve().parent / "diagrams"
PROGRESSION = DIAGRAMS / "percept_fusion_mechanism_progression.png"
K_SWEEP = DIAGRAMS / "percept_k_sweep_heldout_ami.png"
ATTENTION_ARCH = DIAGRAMS / "percept_attention_architecture_heldout_ami.png"
FUSION_FRONTIER = Path(__file__).resolve().parent / "fusion_pareto_frontier.png"

W, H = Inches(13.333), Inches(7.5)
BLACK = RGBColor(0x1A, 0x1A, 0x1A)
GRAY = RGBColor(0x59, 0x59, 0x59)
LIGHT_GRAY = RGBColor(0xE8, 0xE8, 0xE8)
BORDER = RGBColor(0xBF, 0xBF, 0xBF)
BLUE = RGBColor(0x1F, 0x77, 0xB4)
ORANGE = RGBColor(0xFF, 0x7F, 0x0E)
RED = RGBColor(0xC0, 0x30, 0x30)
WHITE = RGBColor(255, 255, 255)

MPL_BLUE, MPL_ORANGE, MPL_RED, MPL_GRAY = "#1f77b4", "#ff7f0e", "#c03030", "#595959"


def set_font(paragraph, size=12, bold=False, color=BLACK, align=PP_ALIGN.LEFT):
    paragraph.alignment = align
    paragraph.space_after = Pt(0)
    for run in paragraph.runs:
        run.font.name = "Calibri"
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color


def text(slide, value, x, y, w, h, size=12, bold=False, color=BLACK,
         align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.text = value
    set_font(paragraph, size, bold, color, align)
    return box


def bullet_text(slide, items, x, y, w, h, size=14, color=BLACK):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    for index, item in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = "•  " + item
        paragraph.space_after = Pt(9)
        set_font(paragraph, size, False, color)
    return box


def border_cell(cell):
    cell_pr = cell._tc.get_or_add_tcPr()
    for edge in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        cell_pr.append(parse_xml(
            '<%s %s w="12700" cap="flat" cmpd="sng" algn="ctr">'
            '<a:solidFill><a:srgbClr val="BFBFBF"/></a:solidFill>'
            '<a:prstDash val="solid"/><a:round/>'
            '<a:headEnd type="none" w="med" len="med"/>'
            '<a:tailEnd type="none" w="med" len="med"/></%s>'
            % (edge, nsdecls("a"), edge)
        ))


def table(slide, headers, rows, x, y, w, h, widths=None, font_size=12):
    shape = slide.shapes.add_table(
        len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(w), Inches(h)
    )
    tbl = shape.table
    if widths:
        for column, width in zip(tbl.columns, widths):
            column.width = Inches(width)
    for row_index, values in enumerate([headers] + rows):
        for column_index, value in enumerate(values):
            cell = tbl.cell(row_index, column_index)
            cell.text = str(value)
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT_GRAY if row_index == 0 else WHITE
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            set_font(
                cell.text_frame.paragraphs[0], font_size, row_index == 0, BLACK,
                PP_ALIGN.CENTER if column_index > 0 else PP_ALIGN.LEFT,
            )
            border_cell(cell)
    return shape


def base_slide(prs, kicker, title_text):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    background.fill.solid()
    background.fill.fore_color.rgb = WHITE
    background.line.fill.background()
    text(slide, kicker.upper(), 0.5, 0.28, 10.7, 0.25, 11, False, GRAY)
    text(slide, title_text, 0.5, 0.55, 12.25, 0.55, 24, True)
    rule = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.18), Inches(12.33), Inches(0.012)
    )
    rule.fill.solid()
    rule.fill.fore_color.rgb = BORDER
    rule.line.fill.background()
    return slide


def footer(slide, page, total):
    text(slide, f"{page} / {total}", 12.2, 7.15, 0.9, 0.2, 10, False, GRAY, PP_ALIGN.RIGHT)


def note(slide, value, y=6.05, h=0.5):
    return text(slide, value, 0.5, y, 12.25, h, 12, False, BLACK)


def make_charts():
    DIAGRAMS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    labels = ["Base\n100/67", "Balance\nλ=1", "Balance\nλ=500", "λ=1000\none seed",
              "λ=1000\n4-seed", "K=60/40\n4-seed"]
    emotion = [0.0363, 0.0925, 0.1142, 0.1242, 0.1235, 0.1252]
    genre = [0.3081, 0.2631, 0.2116, 0.2466, 0.2089, 0.2486]
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(10.8, 4.7), constrained_layout=True)
    width = 0.35
    axis.bar(x - width / 2, emotion, width, label="Emotion AMI", color=MPL_BLUE)
    axis.bar(x + width / 2, genre, width, label="Genre AMI", color=MPL_ORANGE)
    axis.axhline(0.1236, color=MPL_BLUE, linestyle="--", linewidth=1.4, label="Emotion Pareto bar")
    axis.axhline(0.1954, color=MPL_ORANGE, linestyle="--", linewidth=1.4, label="Genre Pareto bar")
    axis.set_ylabel("Held-out AMI")
    axis.set_ylim(0, 0.35)
    axis.set_xticks(x, labels)
    axis.set_title("Fusion-mechanism progression: collapse → stable K=60/40")
    axis.grid(axis="y", color="#e8e8e8")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(ncol=2, frameon=False, loc="upper left")
    fig.savefig(PROGRESSION, dpi=220, facecolor="white")
    plt.close(fig)

    labels = ["20/13", "30/20", "40/27", "50/33", "55/37", "60/40", "65/43", "70/47", "80/53", "100/67"]
    emotion = [0.1152, 0.1220, 0.1220, 0.1202, 0.1208, 0.1238, 0.1238, 0.1237, 0.1220, 0.1242]
    genre = [0.2273, 0.2577, 0.2736, 0.2830, 0.2622, 0.2617, 0.2376, 0.2357, 0.2396, 0.2466]
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(11.4, 4.7), constrained_layout=True)
    axis.plot(x, emotion, marker="o", color=MPL_BLUE, linewidth=2.2, label="Emotion AMI")
    axis.plot(x, genre, marker="o", color=MPL_ORANGE, linewidth=2.2, label="Genre AMI")
    axis.axhline(0.1236, color=MPL_BLUE, linestyle="--", linewidth=1.2, label="Emotion Pareto bar")
    axis.axhline(0.1954, color=MPL_ORANGE, linestyle="--", linewidth=1.2, label="Genre Pareto bar")
    axis.axhline(0.1249, color=MPL_BLUE, linestyle=":", linewidth=1.3, label="Attention-h1 emotion")
    axis.axhline(0.2404, color=MPL_ORANGE, linestyle=":", linewidth=1.3, label="Attention-h1 genre")
    axis.scatter([5], [emotion[5]], color=MPL_RED, s=65, zorder=5)
    axis.annotate("standing", (5, emotion[5]), xytext=(5.3, 0.107), color=MPL_RED, fontsize=9)
    axis.set_ylabel("Held-out AMI")
    axis.set_xlabel("Initial / surviving clusters")
    axis.set_ylim(0.10, 0.30)
    axis.set_xticks(x, labels)
    axis.set_title("Held-out AMI across the consolidated K sweep")
    axis.grid(axis="y", color="#e8e8e8")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20))
    fig.savefig(K_SWEEP, dpi=220, facecolor="white")
    plt.close(fig)

    labels = ["Stage 1", "Stage 2\nMLP-64", "MLP-128", "Attention-h4", "Attention-h1"]
    emotion = [0.1095, 0.1046, 0.1114, 0.1216, 0.1249]
    genre = [0.2901, 0.2087, 0.1604, 0.1875, 0.2404]
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(11.4, 4.7), constrained_layout=True)
    width = 0.34
    emotion_bars = axis.bar(x - width / 2, emotion, width, label="Emotion AMI", color=MPL_BLUE)
    genre_bars = axis.bar(x + width / 2, genre, width, label="Genre AMI", color=MPL_ORANGE)
    for bars in (emotion_bars, genre_bars):
        bars[-1].set_edgecolor(MPL_RED)
        bars[-1].set_linewidth(2.8)
    axis.axhline(0.1236, color=MPL_BLUE, linestyle="--", linewidth=1.3, label="Emotion Pareto bar")
    axis.axhline(0.1954, color=MPL_ORANGE, linestyle="--", linewidth=1.3, label="Genre Pareto bar")
    axis.annotate("only configuration clearing both", (4, 0.2404), xytext=(2.2, 0.326),
                  arrowprops={"arrowstyle": "->", "color": MPL_RED, "lw": 1.4},
                  color=MPL_RED, fontsize=9, fontweight="bold")
    axis.set_ylabel("Held-out AMI")
    axis.set_ylim(0.08, 0.35)
    axis.set_xticks(x, labels)
    axis.set_title("Held-out AMI: linear heads, MLP heads, and attention")
    axis.grid(axis="y", color="#e8e8e8")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(ncol=2, frameon=False, loc="upper left")
    fig.savefig(ATTENTION_ARCH, dpi=220, facecolor="white")
    plt.close(fig)


def build():
    make_charts()
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    total = 13

    # 1 — title
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    background.fill.solid(); background.fill.fore_color.rgb = WHITE; background.line.fill.background()
    text(slide, "Reproducing CoSiR in PercepT", 0.7, 2.35, 11.9, 0.65, 30, True)
    text(slide, "ArtELingo mechanism-fidelity investigation", 0.7, 3.32, 11.8, 0.35, 18, False, GRAY)
    text(slide, "2026-09-23  ·  experiment/percept_topic_pipeline", 0.7, 5.2, 11.8, 0.3, 12, False, GRAY)
    footer(slide, 1, total)

    # 2
    slide = base_slide(prs, "Context", "What is PercepT?")
    text(slide, "Each painting begins with two complementary embeddings.", 0.7, 1.52, 8.4, 0.35, 18, True)
    table(slide, ["Signal", "Role"], [
        ["Factual (CLIP)", "What the painting depicts"],
        ["Affective (GoEmotions-tuned text encoder)", "Emotion-related language around the painting"],
    ], 0.7, 2.03, 11.85, 1.25, [4.25, 7.6], 14)
    text(slide, "Stage 1 — unsupervised topic formation", 0.7, 3.78, 5.2, 0.28, 14, True)
    text(slide, "Fuse the two embeddings → autoencode the fusion → use Deep Embedded Clustering (DEC) to sharpen an initial K-means grouping into topics. Human emotion and genre labels are used only afterward to evaluate the topics.", 0.7, 4.13, 11.55, 0.8, 15)
    text(slide, "Stage 2 — image-only topic mapping", 0.7, 5.36, 4.8, 0.28, 14, True)
    text(slide, "Freeze those Stage 1 assignments and train an image-only classifier to recover them. No text is needed at inference.", 0.7, 5.7, 11.55, 0.48, 16, True, BLUE)
    footer(slide, 2, total)

    # 3
    slide = base_slide(prs, "Motivation", "Why test this mechanism on ArtELingo?")
    text(slide, "CoSiR’s conditional buddy graph already has an established result on this exact held-out split and criterion:", 0.7, 1.52, 11.45, 0.38, 16)
    table(slide, ["Same-split reference", "Emotion AMI", "Genre AMI"], [
        ["Attention-h1 buddy graph", "0.1249", "0.2404"],
    ], 0.7, 2.08, 8.2, 0.72, [4.7, 1.75, 1.75], 14)
    text(slide, "Question", 0.7, 3.43, 1.8, 0.25, 14, True)
    text(slide, "Can PercepT’s independent fusion → autoencoder → DEC mechanism find an equally useful factual-plus-affective partition on the same data?", 0.7, 3.78, 11.45, 0.52, 18, True, BLUE)
    text(slide, "This is a mechanism-fidelity check—not a paper benchmark.", 0.7, 4.78, 7.4, 0.3, 15, True)
    bullet_text(slide, [
        "The paper used a different dataset and label taxonomy.",
        "This replication substitutes the affect backbone, fusion formula, and DEC stopping schedule.",
    ], 0.7, 5.15, 11.3, 0.9, 14)
    footer(slide, 3, total)

    # 4
    slide = base_slide(prs, "Stage 1 journey", "Early observation: collapse → stable K=60/40")
    table(slide, ["Waypoint", "Emotion", "Genre", "Takeaway"], [
        ["Base K=100/67", "0.0363", "0.3081", "Collapsed"],
        ["Balance λ=1", "0.0925", "0.2631", "Still collapsed"],
        ["Balance λ=500", "0.1142", "0.2116", "Still collapsed"],
        ["λ=1000, one seed", "0.1242", "0.2466", "Success, fragile evidence"],
        ["λ=1000, 4-seed", "0.1235", "0.2089", "Only 1/4 jointly clear"],
        ["K=60/40, 4-seed", "0.1252", "0.2486", "4/4 jointly clear"],
    ], 0.5, 1.48, 5.2, 3.9, [1.72, 0.72, 0.72, 2.04], 10)
    slide.shapes.add_picture(str(PROGRESSION), Inches(5.95), Inches(1.48), width=Inches(6.75), height=Inches(4.45))
    note(slide, "Changing cluster count—not merely increasing regularization—was the stability fix.", 6.08, 0.38)
    footer(slide, 4, total)

    # 5
    slide = base_slide(prs, "Interpretation", "How to read AMI—and what “collapse” means")
    text(slide, "Adjusted Mutual Information (AMI) is chance-adjusted agreement between discovered topics and a human label: 0 = chance; 1 = perfect.", 0.7, 1.52, 11.5, 0.55, 17, True)
    text(slide, "DEC sharpens its own confident assignments. Unchecked, that self-training pressure can empty almost every cluster into a few; a run may converge yet still be a failed partition.", 0.7, 2.52, 11.5, 0.85, 17)
    table(slide, ["Base K=100/67 example", "Observed result"], [
        ["Convergence", "Converged, but not useful"],
        ["Topic occupancy", "65 / 67 surviving topics below 1% of nodes"],
        ["Median topic size", "0"],
        ["Held-out AMI", "Emotion 0.0363  ·  Genre 0.3081"],
    ], 0.7, 4.05, 11.45, 1.82, [4.2, 7.25], 14)
    note(slide, "The genre number alone did not rescue a clustering whose topics had effectively disappeared.", 6.12, 0.38)
    footer(slide, 5, total)

    # 6
    slide = base_slide(prs, "Baseline provenance", "Five fusion mechanisms—and the CCA pivot")
    bullet_text(slide, [
        "Late fusion: union improved emotion; intersection was degenerate (98.96% isolated before repair).",
        "Hierarchical refinement: real affect signal, but much genre cost was a granularity artifact.",
        "SNF + co-regularized spectral: evidence-based SKIPPED; mutual-kNN intersection was near-empty.",
        "CCA audit: top held-out canonical correlation 0.7285 vs. permutation-null 0.0699—licensed learning.",
        "Outcome: a learned two-teacher contrastive student.",
    ], 0.55, 1.39, 6.0, 2.05, 12)
    table(slide, ["Train-split method", "Emotion", "Genre"], [
        ["Content-only", "0.0593", "0.4384"],
        ["GoEmotions-only", "0.1180", "0.0396"],
        ["Late fusion — union", "0.1236", "0.1394"],
        ["Hierarchical refinement", "0.1072", "0.1954"],
    ], 0.55, 3.78, 5.95, 2.05, [3.45, 1.22, 1.28], 11)
    slide.shapes.add_picture(str(FUSION_FRONTIER), Inches(6.8), Inches(1.46), width=Inches(5.95), height=Inches(4.15))
    note(slide, "Frontier figure predates the later architecture sweep: Stage 1 is correctly shown as the frontier at that time; the next slide shows Attention-h1.", 6.15, 0.42)
    footer(slide, 6, total)

    # 7
    slide = base_slide(prs, "Baseline provenance", "From linear heads to attention: why Attention-h1 stands")
    table(slide, ["Configuration", "Train E / G", "Held-out E / G"], [
        ["Stage 1 — linear + gate", "0.1284 / 0.2799", "0.1095 / 0.2901"],
        ["Stage 2 — MLP-64 + gate", "0.1230 / 0.2319", "0.1046 / 0.2087"],
        ["MLP-128 + gate", "0.1142 / 0.2039", "0.1114 / 0.1604"],
        ["Attention-h4 — linear", "0.1328 / 0.2441", "0.1216 / 0.1875"],
        ["Attention-h1 — linear", "0.1351 / 0.2397", "0.1249 / 0.2404"],
    ], 0.5, 1.45, 5.85, 3.25, [2.5, 1.67, 1.68], 10)
    slide.shapes.add_picture(str(ATTENTION_ARCH), Inches(6.55), Inches(1.38), width=Inches(6.25), height=Inches(3.75))
    text(slide, "Per-view MLP capacity consistently hurt both axes. Self-attention helped only with one head; four heads lost the genre-axis gain.", 0.55, 5.13, 12.05, 0.42, 14, True, BLUE)
    note(slide, "Attention-h1 is the first configuration to clear both held-out bars. It does not strictly dominate Stage 1 (genre 0.2901 remains higher); it is standing by the predeclared Pareto-bar criterion.", 5.85, 0.55)
    footer(slide, 7, total)

    # 8
    slide = base_slide(prs, "Fine K-sweep", "What qualifies as Stage 1 success?")
    text(slide, "Held-out Pareto bar: emotion AMI > 0.1236 AND genre AMI > 0.1954, simultaneously, on val+test with zero train overlap.", 0.7, 1.48, 11.55, 0.5, 17, True, BLUE)
    table(slide, ["Configuration", "Emotion AMI", "Genre AMI", "Joint clears"], [
        ["Attention-h1 buddy baseline", "0.1249", "0.2404", "Same-split reference"],
        ["K=60/40 (standing)", "0.1252", "0.2486", "4/4"],
        ["K=65/43 (best fine-sweep)", "0.1234", "0.2435", "3/4"],
    ], 0.7, 2.46, 11.65, 1.72, [4.35, 1.65, 1.65, 4.0], 13)
    text(slide, "K=65/43 was the best nearby point, but its like-for-like 4-seed means were lower on both axes. K=60/40 remains standing.", 0.7, 5.02, 11.5, 0.75, 17)
    footer(slide, 8, total)

    # 9
    slide = base_slide(prs, "Fine K-sweep", "Full K sweep: why K=60/40 remains standing")
    slide.shapes.add_picture(str(K_SWEEP), Inches(0.65), Inches(1.42), width=Inches(12.05), height=Inches(4.95))
    note(slide, "Attention-h1 reference: emotion 0.1249, genre 0.2404. Only 65/43 and 70/47 cleared both bars at seed 42; 65/43 then reached 3/4 joint clears, versus K=60/40’s 4/4.", 6.45, 0.38)
    footer(slide, 9, total)

    # 10
    slide = base_slide(prs, "Stage 1 robustness", "Best-supported is not seed-robust")
    table(slide, ["Statistic", "Emotion AMI", "Genre AMI"], [
        ["Mean (14 seeds)", "0.1242", "0.2517"],
        ["Min / max", "0.1157 / 0.1289", "0.2192 / 0.2750"],
        ["Sample standard deviation", "0.0032", "0.0156"],
        ["Individual bar clears", "10/14 (71.4%)", "14/14 (100%)"],
        ["Both bars clear simultaneously", "10/14 (71.4%)", "—"],
    ], 0.7, 1.55, 11.55, 2.55, [4.25, 3.65, 3.65], 13)
    text(slide, "Genre clears in every seed. Emotion misses in four, so the original 4/4 result becomes a more cautious 10/14 joint-clearance finding.", 0.7, 4.82, 11.4, 0.7, 18, True, BLUE)
    footer(slide, 10, total)

    # 11
    slide = base_slide(prs, "Stage 2 reproducibility", "The gate is a safety check, not a formality")
    text(slide, "Before any Stage 2 image-classifier training, re-fit the exact frozen Stage 1 K=60/40 seed-42 encoder from scratch. It must reproduce the known 0.1238 emotion / 0.2617 genre held-out AMI within ±0.002.", 0.7, 1.47, 11.55, 0.72, 16, True)
    table(slide, ["Gate result", "Emotion AMI", "Genre AMI"], [
        ["Known citation", "0.1238", "0.2617"],
        ["Repeated re-fit", "0.1225 (diff 0.0013)", "0.2274 (diff 0.0343)"],
    ], 0.7, 2.55, 11.35, 1.2, [4.0, 3.65, 3.7], 13)
    text(slide, "Genre missed by over 17× the tolerance, so the gate stopped mapper training. The reproducible re-fit became the re-based target and all 14 mapper seeds were retrained against it.", 0.7, 4.12, 11.45, 0.62, 15)
    text(slide, "Resolution: five separate-process launches of the exact refit were byte-identical (0.1225 / 0.2274; zero spread). The Stage 1 seed effect is genuine—not GPU noise—and the citation mismatch most likely reflects a different execution environment or library version.", 0.7, 5.2, 11.5, 0.78, 15, True, BLUE)
    footer(slide, 11, total)

    # 12
    slide = base_slide(prs, "Rebased Stage 2", "Image-only mapping remains seed-robust")
    text(slide, "One learnable query attention-pools 50 CLIP ViT-B/32 patch tokens, followed by a linear 40-topic head. Multi-label targets use q > 1.2/40 and always include the argmax topic.", 0.7, 1.47, 11.55, 0.58, 15)
    table(slide, ["Statistic", "Held-out macro AUC"], [
        ["Mean (14 seeds)", "0.8290"],
        ["Min / max", "0.8266 / 0.8315"],
        ["Sample standard deviation", "0.0014"],
        ["95% CI (normal approximation)", "[0.8283, 0.8298]"],
    ], 0.7, 2.46, 8.1, 2.05, [4.4, 3.7], 14)
    text(slide, "All 14 mapper-initialization seeds exceed the 0.5000 train-marginal-frequency baseline and the old threshold-sweep maximum of 0.5760.", 0.7, 5.2, 11.55, 0.45, 16, True, BLUE)
    footer(slide, 12, total)

    # 13
    slide = base_slide(prs, "Rebased Stage 2", "Exact scope of the rebased result")
    table(slide, ["Conclusion", "Evidence"], [
        ["Fixed Stage 1 target", "Deterministic re-fit of K=60/40, seed 42: emotion 0.1225 / genre 0.2274"],
        ["Stage 2 target construction", "Frozen 40 topics; image-only mapper; q > 1.2/40 multi-label targets"],
        ["Mapper evidence", "14 newly trained mapper-init seeds against the same rebased target set"],
        ["Held-out result", "Macro AUC mean 0.8290; min/max 0.8266 / 0.8315; sample SD 0.0014"],
        ["Robustness conclusion", "Seed-robust mapper result; no mixture of old and rebased target sets"],
    ], 0.6, 1.52, 12.05, 3.82, [3.4, 8.65], 13)
    text(slide, "The Stage 2 conclusion remains intact: the image-only mapper is stable across mapper initializations even though the underlying Stage 1 partition is sensitive to its seed.", 0.7, 5.92, 11.45, 0.56, 16, True, BLUE)
    footer(slide, 13, total)

    prs.save(OUT)
    print(f"Wrote {OUT}")
    print(f"Wrote {PROGRESSION}")
    print(f"Wrote {K_SWEEP}")
    print(f"Wrote {ATTENTION_ARCH}")


if __name__ == "__main__":
    build()
