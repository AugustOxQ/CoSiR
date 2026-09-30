#!/usr/bin/env python3
"""Build the 2026-09-02 Conditional Buddies weekly report slide deck."""
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

OUT = Path(__file__).resolve().parents[1] / "pptx" / "2026-09-02_conditional_buddies_slides.pptx"
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
    total = 15

    # 1
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid(); bg.fill.fore_color.rgb = WHITE; bg.line.fill.background()
    text(s, "Conditional Buddies — Weekly Results", 0.7, 2.6, 11.9, 0.65, 32, True)
    text(s, "Why does training sometimes hurt, and what should default settings be?", 0.7, 3.35, 11.9, 0.4, 18, False, GRAY)
    text(s, "Bridge honesty  ·  symmetric conditioning  ·  K scaling  ·  fusion architecture", 0.7, 4.9, 11.9, 0.3, 14)
    text(s, "2026-09-02 · branch experiment/condition_drift_retrieval_correlation", 0.7, 5.3, 11.9, 0.3, 12, False, GRAY)
    footer(s, 1, total)

    # 2
    s = base_slide(prs, "This week", "Two mysteries explained; two new levers found")
    text(s, "This week tracked down the training/retrieval mystery, stress-tested graph honesty, and found settings that move real retrieval.", 0.5, 1.45, 12.25, 0.35, 13, False, GRAY)
    table(s, ["Thread", "Bottom line"], [
        ["Bridge diagnostic (12) + positive control (14)", "Indirect pairs pull together — but a real direct edge is favored by ~31%"],
        ["combine_side=txt + symmetric fix (13)", "Training-hurts-retrieval is substantially a one-sided-fusion artifact, not shown to be a buddy-graph property"],
        ["Buddy-supervision attribution (15)", "Located the source of honesty erosion and recovered part of it"],
        ["K ablation (16)", "K should scale (gently) with dataset size; the predicted 500k/i2t fix wins"],
        ["Fusion-module search", "Fusion family matters much more than buddy-vector dimension"],
    ], 0.5, 2.05, 12.25, 3.25, [4.2, 8.05], 12)
    add_note(s, "All retrieval reads use paired same-seed deltas and mean/SEM (≥2 is “real”); measured noise floor: ~0.1–0.7 R1.", 5.65, 0.45)
    footer(s, 2, total)

    # 3
    s = base_slide(prs, "Experiment 12 · bridge diagnostic", "Does the buddy graph invent relationships it shouldn’t?")
    bullet_text(s, [
        "A bridge node’s nearest image neighbors and nearest caption neighbors disagree: its image resembles one group while its caption resembles another.",
        "False transitivity asks whether two samples linked only through that bridge get treated as related to each other — despite no direct connection.",
    ], 0.7, 1.48, 11.65, 1.15, 14)
    s.shapes.add_picture(str(ASSETS / "diagrams" / "exp12_bridge_abc.png"), Inches(0.9), Inches(2.38), width=Inches(8.5), height=Inches(4.21))
    text(s, "Test", 9.75, 2.55, 1.5, 0.22, 13, True)
    text(s, "80.2% of RedCaps-150k samples are bridge nodes. We sampled 5,000 B–C pairs that share a bridge but have no edge in either modality.", 9.75, 2.88, 2.65, 1.15, 14)
    text(s, "Question", 9.75, 4.55, 1.5, 0.22, 13, True)
    text(s, "How much does training pull such never-connected pairs together, versus a matched pair with nothing shared?", 9.75, 4.88, 2.65, 0.75, 14)
    footer(s, 3, total)

    # 4
    s = base_slide(prs, "Experiment 12 · results", "The pull is nearly universal — and barely explained by real overlap")
    table(s, ["Bridge-pair read", "Value"], [
        ["Pairs pulled closer", "91.4% of 5,000"], ["Mean pull", "+1.98"],
        ["Significance", "mean/SEM +102.1"], ["Jaccard correlation", "ρ = +0.076; ρ² ≈ 0.6%"],
    ], 0.5, 1.62, 5.5, 2.4, [3.05, 2.45], 13)
    text(s, "Behavioral cross-reference", 6.55, 1.68, 3.4, 0.25, 13, True)
    text(s, "Pooling 6 training runs: this pull has a small, real effect on retrieval ranking — but almost entirely for samples with only an image-side connection and no caption-side one (14.5% of the graph), not for bridge nodes generally.", 6.55, 2.05, 5.75, 1.15, 15)
    text(s, "Four companion figures", 6.55, 3.55, 3.0, 0.25, 13, True)
    text(s, "Node-type counts, pull-distance distribution, Jaccard-vs-pull scatter, and retrieval-rank change by node type are in assets/polysemy_bridges/.", 6.55, 3.88, 5.75, 0.8, 14, False, GRAY)
    text(s, "Verdict", 0.7, 4.95, 1.5, 0.25, 13, True)
    text(s, "This is a real, worth-stating limitation: soft “guilt by association” occurs, but it is expected for graph smoothing rather than evidence that the graph is broken.", 0.7, 5.28, 11.55, 0.65, 16, True)
    footer(s, 4, total)

    # 5
    s = base_slide(prs, "Experiment 14 · positive control", "A fairer version of Experiment 12: does a real connection count more?")
    text(s, "Experiment 12 had no directly connected comparison. Here, closed triangles share a bridge and a real edge; controls share a bridge but have no edge of any kind.", 0.5, 1.42, 12.1, 0.42, 13, False, GRAY)
    s.shapes.add_picture(str(ASSETS / "diagrams" / "exp14_positive_control_abcd.png"), Inches(0.9), Inches(2.05), width=Inches(8.65), height=Inches(4.01))
    text(s, "Positive control", 9.9, 2.2, 2.2, 0.25, 13, True)
    text(s, "Closed triangle\nC–D has a real edge", 9.9, 2.58, 2.3, 0.7, 15)
    text(s, "Control", 9.9, 3.8, 1.2, 0.25, 13, True)
    text(s, "Genuinely unconnected\nC–D has no edge of any kind", 9.9, 4.18, 2.3, 0.7, 15)
    text(s, "Design: 3 sampling seeds · RedCaps-150k", 9.9, 5.45, 2.3, 0.45, 12, False, GRAY)
    footer(s, 5, total)

    # 6
    s = base_slide(prs, "Experiment 14 · results", "The graph favors direct evidence by a real, seed-stable margin")
    table(s, ["Statistic", "Closed triangle", "Genuinely unconnected", "Contrast"], [
        ["Pooled mean pull (± SEM)", "+3.1772 ± 0.0034", "+2.4248 ± 0.0096", "+0.7524 ± 0.0063"],
        ["mean/SEM", "+927.0", "+253.3", "+119.7"],
        ["Relative ratio", "", "", "1.31×"],
    ], 0.5, 1.58, 12.25, 1.85, [3.25, 2.9, 3.25, 2.85], 12)
    bullet_text(s, [
        "The original “open” group was 51.7% contaminated by another edge type; requiring no edge of any kind widened the gap from ~1.2× to ~1.31×.",
        "Dose response: both (+3.26, 1.35×) > img_only (+3.17, 1.32×) > txt_only (+2.73, 1.13×) > unconnected (+2.41).",
    ], 0.7, 3.9, 11.55, 1.45, 14)
    text(s, "Verdict: direct evidence wins, even though indirect evidence is not ignored.", 0.7, 5.72, 11.5, 0.35, 16, True)
    footer(s, 6, total)

    # 7
    s = base_slide(prs, "Side test · combine_side=txt", "One-sided fusion amplifies the problem — but does not create it")
    text(s, "Why run it? Every earlier training run fused the trainable signal into image only; switching to text tests whether the retrieval-direction and subgroup effects mirror.", 0.5, 1.42, 12.25, 0.45, 14)
    table(s, ["Frozen − trained", "Image-combine", "Text-combine"], [
        ["t2i mean Δ", "−0.27; mean/SEM −2.0", "+0.13; mean/SEM +0.3"],
        ["i2t mean Δ", "+4.67; mean/SEM +32.1", "+0.40; mean/SEM +4.0"],
    ], 1.0, 2.25, 11.3, 1.5, [3.0, 4.15, 4.15], 14)
    text(s, "The i2t effect remains significant but shrinks to roughly 1/11th; the specific bridge-subgroup pattern flips cleanly across all 3 seeds.", 0.7, 4.35, 11.5, 0.55, 16)
    text(s, "Bridge to Experiment 13: remove the asymmetry altogether, rather than merely moving it to the other side.", 0.7, 5.55, 11.5, 0.4, 15, True, BLUE)
    footer(s, 7, total)

    # 8
    s = base_slide(prs, "Experiment 13 · symmetric conditioning", "Removing one-sided fusion resolves last week’s headline mystery")
    text(s, "One shared table and tied-weight combiner now condition both image and text, with auxiliary losses made symmetric.", 0.5, 1.42, 12.25, 0.35, 13, False, GRAY)
    table(s, ["Original: image only", "Side test: text only", "Both, symmetrically"], [
        ["+4.67 R1\nmean/SEM +32.1", "+0.40 R1\nmean/SEM +4.0", "+0.53 R1\nmean/SEM +0.8 (n.s.; 2/3 wins)"],
    ], 0.5, 2.05, 12.25, 1.45, [4.08, 4.08, 4.09], 15)
    text(s, "Frozen − trained image-to-text effect", 0.7, 3.72, 5.5, 0.25, 13, True)
    text(s, "Verdict", 0.7, 4.65, 1.5, 0.25, 13, True)
    text(s, "Continued training does not generically hurt retrieval: the regression was substantially an artifact of the current one-sided combiner design, not buddy-graph structure.", 0.7, 4.98, 11.55, 0.72, 16, True)
    footer(s, 8, total)

    # 9
    s = base_slide(prs, "Experiment 15 · supervision attribution", "Why the graph’s honesty erodes: ordinary retrieval training is the main pressure")
    text(s, "Measure: closed/unconnected discrimination ratio in retrieval (comb_all) space, pooled across 11.1’s 3 seeds per arm.", 0.5, 1.42, 12.25, 0.35, 13, False, GRAY)
    table(s, ["Arm", "Epoch", "Closed / unconnected ratio"], [
        ["Trained or Frozen", "0000 (init)", "2.4793 ± 0.0031"],
        ["Trained", "0099 (final)", "2.2629 ± 0.0017"],
        ["Frozen", "0099 (final)", "2.2682 ± 0.0032"],
    ], 1.15, 2.05, 11.0, 2.0, [3.2, 2.6, 5.2], 14)
    text(s, "Both arms fall from 2.48 to ~2.26 even when buddy losses are inactive. The ~0.005 trained/frozen difference is negligible beside the ~0.21 shared drop.", 0.7, 4.62, 11.55, 0.75, 16)
    text(s, "Interpretation: ordinary retrieval pressure, not buddy-specific training, causes most of the erosion.", 0.7, 5.78, 11.55, 0.35, 15, True, BLUE)
    footer(s, 9, total)

    # 10
    s = base_slide(prs, "Experiment 15 · test and fix", "Only static contrastive supervision helps — type-aware refinement adds a little more")
    table(s, ["Family / arm", "Ratio", "Δ vs. buddy-off", "mean/SEM"], [
        ["Baseline (buddy off)", "2.2629 ± 0.0017", "—", "—"],
        ["Family #1 only", "2.2605 ± 0.0009", "−0.0024", "≈1.3 (n.s.)"],
        ["Family #2 only", "2.2760 ± 0.0008", "+0.0131", "≈7.0"],
        ["Family #2 + refresh", "2.2628 ± 0.0009", "+0.0001", "n.s. — cancels"],
        ["Typed #2 + repair-excluded", "2.2795 ± 0.0014", "+0.0166", "≈7.5"],
    ], 0.5, 1.55, 12.25, 2.85, [3.7, 2.45, 2.55, 3.55], 12)
    text(s, "Verdict / next", 0.7, 4.85, 2.2, 0.25, 13, True)
    text(s, "Keep Family #2 static; refresh creates a feedback loop that cancels its gain. The approved targeted repulsion test (15.4) is not yet run.", 0.7, 5.18, 11.55, 0.6, 16, True)
    footer(s, 10, total)

    # 11
    s = base_slide(prs, "Experiment 16 · mutual-KNN K", "A structural K(N) prediction lands exactly on a real 500k retrieval win")
    text(s, "Structural K(N) prediction: preserve 150k/K=30 strict-buddy degree → K≈35 at 300k and K≈39 at 500k (sublinear, not proportional to N).", 0.5, 1.42, 12.25, 0.35, 14)
    table(s, ["500k Stage B", "Result"], [
        ["K=39 vs. K=30, i2t R1", "+0.97; mean/SEM +14.5"],
        ["K=39 vs. K=30, t2i R1", "+0.10; mean/SEM +0.9 (n.s.)"],
    ], 0.5, 2.0, 5.7, 1.45, [3.35, 2.35], 13)
    bar_chart(s, ["K=30", "K=39"], [("Δ i2t R1 vs. K=30", [0.00, 0.97])], 6.7, 1.85, 5.8, 3.15, "500k i2t result", 0, 1.2)
    add_note(s, "Caveat: the effect is scale- and direction-specific — 500k’s clean win is i2t-only; 300k’s smaller, borderline wins are t2i-only.", 5.55, 0.45)
    footer(s, 11, total)

    # 12
    s = base_slide(prs, "Combiner architecture", "Fusion family matters much more than buddy-vector dimension")
    table(s, ["Combiner family", "Oracle i2t R1 Δ", "Deployment pre_diff i2t R1 Δ"], [
        ["residual_control", "+0.63 (n.s.)", "−23.87 (mean/SEM −84)"],
        ["lowrank (rank-16 residual)", "+3.10 (mean/SEM +15)", "−3.63 (mean/SEM −11)"],
        ["film", "+2.97 (mean/SEM +20)", "−24.63 (mean/SEM −78)"],
    ], 0.5, 1.55, 12.25, 1.95, [3.8, 3.7, 4.75], 13)
    bar_chart(s, ["residual", "lowrank", "film"], [("Oracle i2t Δ", [0.63, 3.10, 2.97]), ("Deployment pre_diff i2t Δ", [-23.87, -3.63, -24.63])], 2.5, 3.9, 8.35, 2.05, "Oracle gain vs. deployment read", -28, 5)
    text(s, "Verdict: lowrank wins on both axes; vector size 8/16/32 changed nothing. It is not yet a validated default.", 0.7, 6.12, 11.5, 0.3, 14, True)
    footer(s, 12, total)

    # 13
    s = base_slide(prs, "Synthesis", "What this week established")
    table(s, ["Puzzle / lever", "Resolution"], [
        ["Does training hurt retrieval?", "Mostly one-sided fusion: symmetric conditioning reduces the i2t result to noise."],
        ["Does the graph invent relationships?", "Indirect pull is real, but direct edges still receive ~31% more pull; ordinary training causes most erosion."],
        ["K", "A gentle K(N) rule predicted 500k/K=39, which delivers +0.97 i2t R1 (mean/SEM +14.5)."],
        ["Fusion module", "Family matters, dimension does not; lowrank is the only candidate strong on oracle and comparatively robust in deployment."],
    ], 0.5, 1.7, 12.25, 3.05, [3.3, 8.95], 13)
    text(s, "Publication reading", 0.7, 5.2, 2.4, 0.25, 13, True)
    text(s, "Two worries are now explained rather than merely flagged; two promising levers are validated but not yet adopted as defaults.", 0.7, 5.53, 11.55, 0.5, 16, True)
    footer(s, 13, total)

    # 14
    s = base_slide(prs, "Next", "Close the remaining gates")
    bullet_text(s, [
        "Scope and run Experiment 15’s targeted repulsion fix for genuinely-unconnected hub pairs.",
        "Validate the low-rank fusion module at the project’s real training scale before considering it as the new default.",
        "Investigate why one-sided fusion creates the regression; Experiment 13 identifies the cause class, not its mechanism.",
        "Decide whether 500k/K=39 needs a third confirming seed before it enters the paper.",
    ], 0.7, 1.55, 11.55, 3.8, 16)
    footer(s, 14, total)

    # 15
    s = base_slide(prs, "Questions", "The publication-safe claim and the sharper open question")
    text(s, "Current publication-safe claim", 0.7, 1.65, 4.2, 0.25, 13, True)
    text(s, "Buddy-graph structure is a robust, content-grounded initializer and a better in-model starting point than the generic alternative.", 0.7, 2.0, 11.4, 0.65, 18, True)
    text(s, "This week’s sharper question", 0.7, 3.75, 4.2, 0.25, 13, True)
    text(s, "Now that the training-hurts-retrieval mystery is explained away, what is the cleanest way to state the graph’s remaining honesty gap as a limitation rather than a flaw?", 0.7, 4.1, 11.4, 0.75, 20, True, BLUE)
    footer(s, 15, total)

    prs.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
