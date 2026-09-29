#!/usr/bin/env python3
"""Build the 2026-09-16 Experiment 18 stage-report slide deck."""
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls

OUT = Path(__file__).resolve().parents[1] / "2026-09-16_stage_report_prototype_conditioning_slides.pptx"
ASSETS = Path(__file__).resolve().parent
W, H = Inches(13.333), Inches(7.5)
BLACK = RGBColor(0x1A, 0x1A, 0x1A)
GRAY = RGBColor(0x59, 0x59, 0x59)
LIGHT_GRAY = RGBColor(0xE8, 0xE8, 0xE8)
BORDER = RGBColor(0xBF, 0xBF, 0xBF)
BLUE = RGBColor(0x1F, 0x77, 0xB4)
ORANGE = RGBColor(0xFF, 0x7F, 0x0E)
RED = RGBColor(0xC0, 0x30, 0x30)
WHITE = RGBColor(255, 255, 255)


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
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.text = value
    set_font(p, size, bold, color, align)
    return box


def bullet_text(slide, items, x, y, w, h, size=14, color=BLACK):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear(); tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item
        p.level = 0
        p.space_after = Pt(9)
        set_font(p, size, False, color)
        p.text = "•  " + item
    return box


def border_cell(cell):
    tc_pr = cell._tc.get_or_add_tcPr()
    for edge in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        tc_pr.append(parse_xml(
            '<%s %s w="12700" cap="flat" cmpd="sng" algn="ctr">'
            '<a:solidFill><a:srgbClr val="BFBFBF"/></a:solidFill>'
            '<a:prstDash val="solid"/><a:round/><a:headEnd type="none" w="med" len="med"/>'
            '<a:tailEnd type="none" w="med" len="med"/></%s>' % (edge, nsdecls('a'), edge)))


def table(slide, headers, rows, x, y, w, h, widths=None, font_size=12):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(w), Inches(h))
    t = shape.table
    if widths:
        for col, width in zip(t.columns, widths): col.width = Inches(width)
    for r, values in enumerate([headers] + rows):
        for c, value in enumerate(values):
            cell = t.cell(r, c)
            cell.text = str(value)
            cell.fill.solid(); cell.fill.fore_color.rgb = LIGHT_GRAY if r == 0 else WHITE
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            numeric = c > 0
            set_font(p, font_size, r == 0, BLACK, PP_ALIGN.CENTER if numeric else PP_ALIGN.LEFT)
            border_cell(cell)
    return shape


def bar_chart(slide, categories, series, x, y, w, h, title=None, minimum=None, maximum=None):
    data = CategoryChartData()
    data.categories = categories
    for name, values in series:
        data.add_series(name, values)
    chart = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), data).chart
    chart.has_legend = len(series) > 1
    if chart.has_legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.name = "Calibri"; chart.legend.font.size = Pt(10)
    chart.has_title = bool(title)
    if title:
        chart.chart_title.text_frame.text = title
        set_font(chart.chart_title.text_frame.paragraphs[0], 12, True)
    chart.value_axis.has_major_gridlines = True
    chart.value_axis.tick_labels.font.name = "Calibri"; chart.value_axis.tick_labels.font.size = Pt(10)
    chart.category_axis.tick_labels.font.name = "Calibri"; chart.category_axis.tick_labels.font.size = Pt(10)
    if minimum is not None: chart.value_axis.minimum_scale = minimum
    if maximum is not None: chart.value_axis.maximum_scale = maximum
    chart.plots[0].has_data_labels = True
    chart.plots[0].data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    chart.plots[0].data_labels.font.name = "Calibri"; chart.plots[0].data_labels.font.size = Pt(9)
    colors = [BLUE, ORANGE, RED]
    for s, color in zip(chart.series, colors):
        s.format.fill.solid(); s.format.fill.fore_color.rgb = color
        s.format.line.color.rgb = color
    return chart


def base_slide(prs, kicker, title_text):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid(); bg.fill.fore_color.rgb = WHITE; bg.line.fill.background()
    text(slide, kicker.upper(), 0.5, 0.28, 10.7, 0.25, 11, False, GRAY)
    text(slide, title_text, 0.5, 0.55, 12.25, 0.55, 24, True)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.18), Inches(12.33), Inches(0.012))
    line.fill.solid(); line.fill.fore_color.rgb = BORDER; line.line.fill.background()
    return slide


def footer(slide, page, total):
    text(slide, f"{page} / {total}", 12.2, 7.15, 0.9, 0.2, 10, False, GRAY, PP_ALIGN.RIGHT)


def add_note(slide, value, y=5.85, h=0.8):
    return text(slide, value, 0.5, y, 12.25, h, 12, False, BLACK)


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    total = 11

    # 1
    s = base_slide(prs, "Recap", "Where CoSiR stood ~2 months ago")
    text(s, "CoSiR keeps a frozen pretrained CLIP model — the image and text encoders are never fine-tuned — and attaches a small trainable condition vector to every training sample. That vector enters a small trainable combiner module, which nudges one side's CLIP embedding before the usual contrastive retrieval loss.", 0.5, 1.52, 12.15, 1.1, 15)
    text(s, "Why", 0.7, 3.02, 1.5, 0.25, 13, True)
    text(s, "Aim for retrieval better than raw CLIP while staying close to CLIP's own embedding space.", 0.7, 3.35, 11.45, 0.42, 16, True, BLUE)
    text(s, "Starting point", 0.7, 4.18, 2.0, 0.25, 13, True)
    text(s, "Each condition vector starts from the buddy graph: mutual nearest-neighbor connections in CLIP's image space, text space, or both. Samples CLIP already considers close therefore begin training with similar conditions rather than random ones.", 0.7, 4.51, 11.55, 0.82, 16)
    add_note(s, "This buddy-graph idea, and its validation, is the established starting point — not new this report.", 5.9, 0.45)
    footer(s, 1, total)

    # 2
    s = base_slide(prs, "Recap", "Two months of hardening the foundation (mid-July → early Sept)")
    text(s, "The following months strengthened this basis without changing the buddy-graph construction Experiment 18 still relies on:", 0.5, 1.43, 12.2, 0.34, 13, False, GRAY)
    table(s, ["Milestone", "One-line takeaway"], [
        ["Cross-encoder buddy-graph check", "Buddy structure holds across 16 different vision/text encoder pairs — not a CLIP-specific fluke"],
        ["Buddy-init vs. generic init", "Buddy-graph initialization wins on RedCaps and becomes the standing default; neither init beats raw CLIP outright at this scale"],
        ["Robustness/diagnostic ablations", "Graph-edge, encoder-pair, and bridge-node checks confirmed the construction was sound; none changed the architecture"],
        ["Neighborhood-size (K) scaling", "Fixed K=30 is measurably suboptimal at 500k samples (predicted K≈39 wins i2t R1 by +0.97) — real, still open, not yet in configs"],
        ["Combiner redesign", "A rank-16 low-rank residual adapter beat the older two-MLP-tower combiner, confirmed at full 500k/100-epoch scale — selected for Experiment 18, not yet a global default"],
    ], 0.5, 1.98, 12.25, 3.55, [3.35, 8.9], 11)
    add_note(s, "Also standing, deliberately: conditioning is asymmetric — it is fused only into the image embedding (combine_side=\"img\"); text passes through an identity-initialized other_proj layer. A parallel line of work suggests this may cause retrieval asymmetry — a forward-pointer, not resolved here.", 5.85, 0.72)
    footer(s, 2, total)

    # 3
    s = base_slide(prs, "Before Experiment 18", "Architecture right before Experiment 18")
    s.shapes.add_picture(str(ASSETS / "2026-09-16_stage_report" / "architecture_before.png"), Inches(0.7), Inches(1.48), width=Inches(7.8), height=Inches(4.68))
    text(s, "How it worked", 8.95, 1.65, 2.4, 0.25, 13, True)
    text(s, "A large lookup table holds one trainable 16-d condition vector per training sample, initialized from the buddy graph and trained directly at its own, much larger learning rate. The image embedding and condition enter the low-rank combiner; text passes through other_proj untouched; both final embeddings train against a contrastive retrieval loss.", 8.95, 1.98, 3.4, 2.05, 14)
    text(s, "The problem", 8.95, 4.52, 2.4, 0.25, 13, True)
    text(s, "Every table row is independent: there is no general meaning for a condition, and a brand-new sample needs a separately-trained condition_predictor to approximate the table.", 8.95, 4.85, 3.35, 1.05, 14, True, BLUE)
    footer(s, 3, total)

    # 4
    s = base_slide(prs, "Experiment 18 · idea", "Replace the lookup table with shared learned archetypes")
    text(s, "Replace the giant independent lookup table with a small, shared bank of 16 learnable \"prototype\" vectors — a small set of learned archetypes. Each sample's condition becomes a soft, weighted blend of those 16 shared prototypes, computed fresh from the sample's own CLIP feature rather than looked up from a table.", 0.5, 1.48, 12.1, 1.05, 16)
    text(s, "The bet", 0.7, 3.08, 2.0, 0.25, 13, True)
    bullet_text(s, [
        "With 16 shared archetypes instead of one row per sample, the condition space should naturally organize into interpretable clusters.",
        "Because the condition is a function of the sample's own features, it works automatically for new, unseen samples — no separate predictor network needed.",
    ], 0.7, 3.45, 11.4, 1.45, 17)
    text(s, "Only the condition source is being redesigned; the downstream retrieval architecture stays in place.", 0.7, 5.65, 11.5, 0.35, 15, True, BLUE)
    footer(s, 4, total)

    # 5
    s = base_slide(prs, "Experiment 18 · implementation", "Attention over a small prototype bank")
    s.shapes.add_picture(str(ASSETS / "2026-09-16_stage_report" / "architecture_after.png"), Inches(0.7), Inches(1.48), width=Inches(7.8), height=Inches(4.68))
    text(s, "How a condition is made", 8.95, 1.65, 3.2, 0.25, 13, True)
    text(s, "Each prototype has a learned key and value. A sample's averaged image/text CLIP feature becomes a query; comparison to all 16 keys produces scores; a temperature-controlled softmax turns them into weights summing to 1; the condition is the attention-weighted sum of value vectors.", 8.95, 1.98, 3.4, 2.25, 14)
    text(s, "What does not change", 8.95, 4.62, 3.1, 0.25, 13, True)
    text(s, "The low-rank combiner, other_proj, and retrieval loss are completely unchanged. Only the condition vector's source changes.", 8.95, 4.95, 3.35, 0.75, 14, True, BLUE)
    footer(s, 5, total)

    # 6
    s = base_slide(prs, "Vocabulary", "A few terms this report leans on")
    bullet_text(s, [
        "Oracle retrieval (Recall@1/5/10): retrieval accuracy given the best-available condition vector per query — always read against raw retrieval (plain, unconditioned CLIP) as the floor to beat.",
        "Effective dimensions (95% variance): directions needed to explain 95% of condition-vector spread. Close to 16 = full capacity; collapsing to 1 = nearly every vector differs along one shared direction.",
        "Silhouette score: a standard clustering-quality number — positive and high means clean, separated groups; near-zero or negative means no real structure.",
        "Probe selectivity: a simple classifier's real-label accuracy minus shuffled-label accuracy from only the condition vector. Meaningfully positive = the space encodes that distinction.",
        "warmth / register: RedCaps proxy groupings used as content-separation probes, not validated measures of emotion or formality; dedicated audits found both are largely explained by plain photo content or carry little signal beyond it.",
    ], 0.65, 1.48, 11.75, 4.95, 13)
    footer(s, 6, total)

    # 7
    s = base_slide(prs, "Experiment 18 · first pass", "Retrieval flat, interpretability collapsed")
    text(s, "Default settings, 3 random seeds, RedCaps-150k. Retrieval was roughly on par with the old lookup-table design — no clear win or loss, both designs slightly below plain unconditioned CLIP at this scale.", 0.5, 1.43, 12.25, 0.54, 14, False, GRAY)
    text(s, "Interpretability — the actual point of the redesign — failed outright:", 0.5, 2.13, 8.0, 0.3, 15, True)
    table(s, ["", "seed 1", "seed 2", "seed 3"], [
        ["Effective dimensions (of 16)", "1", "1", "1"],
        ["Silhouette score", "0.12", "0.31", "−0.13"],
        ["warmth probe", "weak positive", "null", "positive"],
        ["register probe", "null", "null", "null"],
    ], 0.5, 2.62, 12.25, 2.25, [4.8, 2.48, 2.48, 2.49], 13)
    text(s, "Nearly every sample ended up with almost the same condition vector, regardless of content — clearly worse than the old table, which reliably encoded both proxy properties.", 0.7, 5.38, 11.45, 0.58, 16, True, BLUE)
    footer(s, 7, total)

    # 8
    s = base_slide(prs, "Diagnosis", "Why the built-in health check missed it")
    text(s, "The design's one health-check metric, usage entropy (how spread out each sample's soft attention is over 16 prototypes), looked reassuring: 92–93% of maximum spread, \"not concentrated.\"", 0.5, 1.52, 12.0, 0.72, 16)
    text(s, "But on 88–99.5% of all 150,000 samples, the single strongest prototype match was the same one or two prototypes. Because the softmax stayed close to uniform everywhere, the final blended output still became nearly identical for nearly every sample.", 0.5, 2.9, 12.0, 0.9, 16)
    text(s, "Lesson", 0.7, 4.55, 1.6, 0.25, 13, True)
    text(s, "A spread-out-attention metric and a diverse final output are not the same thing — the original monitoring only checked the first.", 0.7, 4.9, 11.55, 0.62, 19, True, BLUE)
    footer(s, 8, total)

    # 9
    s = base_slide(prs, "Root cause", "A hidden 1,000× learning-rate gap")
    text(s, "The prototype bank's own parameters were quietly being trained 1,000× slower than the old per-sample table — an optimizer-configuration gap, not a flaw in the idea itself.", 0.5, 1.55, 12.0, 0.7, 18, True)
    text(s, "What changed", 0.7, 3.02, 2.2, 0.25, 13, True)
    text(s, "Give the bank its own, correctly-scaled learning rate, plus a tunable starting \"sharpness\" for its attention.", 0.7, 3.35, 11.4, 0.45, 17)
    text(s, "Quick single-seed sweep", 0.7, 4.43, 3.1, 0.25, 13, True)
    text(s, "As learning rate increased, entropy sharpened, attention concentration spread across more prototypes, and silhouette roughly quadrupled — a clear, monotonic trend.", 0.7, 4.78, 11.4, 0.68, 17, True, BLUE)
    footer(s, 9, total)

    # 10
    s = base_slide(prs, "Experiment 18 · 3-seed results", "Best setting, 3-seed confirmed: real interpretability gains")
    table(s, ["", "original design (n=3)", "best fix: temp=0.3, lr_prototype=1e-2 (n=3)"], [
        ["Silhouette score", "0.12, 0.31, −0.13", "0.55, 0.56, 0.70"],
        ["register probe", "null, null, null", "positive, positive, null"],
        ["warmth probe", "1 of 3 positive", "0 of 3 positive"],
    ], 0.5, 1.48, 12.25, 1.95, [2.5, 3.7, 6.05], 12)
    text(s, "What improved", 0.7, 4.05, 2.0, 0.25, 13, True)
    text(s, "Silhouette moved to consistently positive and much higher across all 3 seeds — the condition space broke out of its 1-dimensional collapse. register became recoverable for the first time in this whole investigation.", 0.7, 4.4, 11.55, 0.72, 17)
    text(s, "The smaller real regression", 0.7, 5.65, 3.0, 0.25, 13, True)
    text(s, "warmth's one prior weak positive signal disappeared alongside the gain.", 0.7, 5.98, 11.55, 0.32, 16, True, BLUE)
    footer(s, 10, total)

    # 11
    s = base_slide(prs, "Experiment 18 · trade-off", "...bundled with a serious, seed-replicated retrieval cost")
    table(s, ["oracle i2t Recall@1", "value"], [
        ["Original prototype design", "16.8"],
        ["Best fix (temp=0.3, lr_prototype=1e-2)", "10.4"],
        ["No conditioning at all (raw CLIP)", "17.8"],
    ], 0.5, 1.62, 6.2, 1.85, [4.55, 1.65], 14)
    text(s, "At this operating point, conditioning is actively worse than doing nothing on image-to-text retrieval — a real, tight drop, consistent across all 3 seeds, not seed noise. Text-to-image retrieval is unaffected either way.", 0.7, 4.15, 11.55, 0.78, 18, True)
    add_note(s, "Bottom line: this is a genuine, seed-replicated trade-off, not a finished fix. The learning-rate diagnosis is real and the interpretability problem is fixable in principle, but this exact setting is not yet usable as a replacement for the old design.", 5.65, 0.7)
    footer(s, 11, total)

    prs.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
