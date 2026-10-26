"""The detailed companion of framework.png: formulas and small examples for the views, the residual graphs, the
network, the losses, stopping and the test.

Facts: the claim-test graph code (ct2_graphs._partition), the model-sheet trainer (ms_train: symmetric, distill_loss,
head_loss, the stopping loop), itp_train.CONFIG, recipe R6-5 (rb_rows), the stage report of 2026-10-26.
Run: PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python build_framework_detail_figure.py
"""
import re

import numpy as np

from build_framework_figure import (SVG, GREEN, DEFAULT_BLUE, DEFAULT_GREY, DATA_STROKE, TRAINED_STROKE, LOSS_STROKE,
                                    OP_STROKE, INK)

FORMULA_FILL = "#1A1A1A"


class DetailSVG(SVG):
    def formula(self, x, y, s, fs=12.5, anchor="start", fill=FORMULA_FILL, weight="normal"):
        """Text with _{...} subscripts and ^{...} superscripts (tspans with dy shifts)."""
        esc = lambda t: t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        small = fs * 0.72
        out, pos = [], 0
        for m in re.finditer(r"([_^])\{([^}]*)\}", s):
            out.append(esc(s[pos:m.start()]))
            dy = 4 if m.group(1) == "_" else -5
            out.append(f'<tspan dy="{dy}" style="font-size:{small}px">{esc(m.group(2))}</tspan><tspan dy="{-dy}"> </tspan>')
            pos = m.end()
        out.append(esc(s[pos:]))
        style = f"font-family:Lato, DejaVu Sans, sans-serif;font-size:{fs}px;font-weight:{weight};"
        self.parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" xml:space="preserve" style="{style}">{"".join(out)}</text>')

    def formulas(self, x, y, lines, fs=12.5, lh=None):
        lh = lh or fs * 1.55
        for i, s in enumerate(lines):
            self.formula(x, y + i * lh, s, fs)

    def box(self, x, y, w, h, lines, kind="data", fs=14, trained=False, math=False):
        if not math:
            return super().box(x, y, w, h, lines, kind, fs, trained)
        from build_framework_figure import KINDS
        fill, stroke = KINDS[kind]
        self.rect(x, y, w, h, fill, stroke, dash=(kind == "note"))
        lh = fs * 1.3
        y0 = y + h / 2 - (len(lines) - 1) * lh / 2 + fs * 0.35
        for i, s in enumerate(lines):
            self.formula(x + w / 2, y0 + i * lh, s, fs, "middle", INK, "bold" if i == 0 else "normal")
        if trained:
            self.badge(x + w - 8, y - 9, "trained")

    def matrix(self, x, y, n=4, cell=22, row_label="i", col_label="j"):
        for r in range(n):
            for c in range(n):
                fill = "#9ED39A" if r == c else "#EDEDED"
                self.parts.append(f'<rect x="{x+c*cell}" y="{y+r*cell}" width="{cell-2}" height="{cell-2}" fill="{fill}" stroke="#BBBBBB" stroke-width="0.8"/>')
            self.formula(x - 6, y + r * cell + cell * 0.68, f"{row_label}_{{{r+1}}}", 11, "end")
            self.formula(x + r * cell + cell / 2 - 1, y - 6, f"{col_label}_{{{r+1}}}", 11, "middle")

    def scatter(self, x, y, w, h, seed, colors, label):
        rng = np.random.default_rng(seed)
        self.rect(x, y, w, h, "#FFFFFF", DATA_STROKE, rx=6)
        centres = rng.uniform(0.2, 0.8, size=(len(colors), 2))
        for (cx, cy), col in zip(centres, colors):
            pts = rng.normal(0, 0.07, size=(14, 2)) + (cx, cy)
            for px, py in pts:
                px, py = min(max(px, 0.04), 0.96), min(max(py, 0.04), 0.9)
                self.parts.append(f'<circle cx="{x+px*w:.1f}" cy="{y+py*h:.1f}" r="3" fill="{col}" opacity="0.85"/>')
        self.text(x + w / 2, y + h - 6, label, 11, "middle", "bold")

    def lambda_example(self, x, y):
        """Anchor i with four plain emotion buddies (a, b, c, d) and three content buddies (a, b, e); legend below."""
        cx, cy = x + 120, y + 85
        nodes = dict(a=(cx - 85, cy - 50), b=(cx + 85, cy - 50), c=(cx - 95, cy + 40), d=(cx + 95, cy + 40), e=(cx, cy + 75))
        for name in ("a", "b", "e"):
            nx, ny = nodes[name]
            self.parts.append(f'<line x1="{cx}" y1="{cy}" x2="{nx}" y2="{ny}" stroke="#8A8A8A" stroke-width="3.2" stroke-dasharray="5,4"/>')
        for name in ("a", "b", "c", "d"):
            nx, ny = nodes[name]
            self.parts.append(f'<line x1="{cx}" y1="{cy}" x2="{nx}" y2="{ny}" stroke="{GREEN[0]}" stroke-width="1.8"/>')
        for name, (nx, ny) in nodes.items():
            ring = name in ("a", "b")
            self.parts.append(f'<circle cx="{nx}" cy="{ny}" r="12" fill="#FFFFFF" stroke="{"#333333" if ring else DATA_STROKE}" stroke-width="{2.4 if ring else 1.4}"/>')
            self.text(nx, ny + 4, name, 12, "middle", "bold")
        self.parts.append(f'<circle cx="{cx}" cy="{cy}" r="14" fill="#FDE7CF" stroke="{TRAINED_STROKE}" stroke-width="1.8"/>')
        self.text(cx, cy + 4, "i", 13, "middle", "bold")
        lx, ly = x + 5, y + 190
        self.parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+28}" y2="{ly}" stroke="{GREEN[0]}" stroke-width="1.8"/>')
        self.text(lx + 34, ly + 4, "plain emotion buddy of i (mutual top-20 in E alone)", 11, "start", fill="#444444")
        self.parts.append(f'<line x1="{lx}" y1="{ly+20}" x2="{lx+28}" y2="{ly+20}" stroke="#8A8A8A" stroke-width="3.2" stroke-dasharray="5,4"/>')
        self.text(lx + 34, ly + 24, "content buddy of i (mutual top-20 in D or in I)", 11, "start", fill="#444444")
        self.parts.append(f'<circle cx="{lx+14}" cy="{ly+40}" r="7" fill="#FFFFFF" stroke="#333333" stroke-width="2.4"/>')
        self.text(lx + 34, ly + 44, "both: a and b", 11, "start", fill="#444444")
        self.formula(cx, y + 272, "\u03bb_{i} = 2 / 4 = 0.5", 14, "middle", weight="bold")


def figure():
    s = DetailSVG(1600, 1600)
    s.text(800, 36, "The condition-free model in detail: views, residual graphs, network, losses, test", 22, weight="bold")
    s.text(800, 62, "Formulas as in the code on 2026-10-26 (claim-test graphs, model-sheet trainer, current recipe: no own-caption terms, "
                    "teacher weight 0.25). All embeddings and outputs are unit vectors; \u03c4 = 0.05.", 14, fill="#444444")
    s.legend_row(420, 78)
    G = ["emo", "sty", "gen"]

    # ------------------------------------------------------------------ 1. views (full width)
    s.panel(30, 102, 1540, 300, "1  Views: six Qwen embeddings per painting p, from one frozen embedder")
    s.painting(50, 150, 120, 80, "painting p")
    s.text(50, 258, "its captions (example text)", 11, "start", fill="#444444", italic=True)
    for k, cap in enumerate(["\u201ca calm lake at dusk\u201d", "\u201clonely and quiet\u201d", "\u201csoft evening light\u201d"]):
        s.box(50, 266 + k * 32, 150, 26, [cap], "data", 11)
    s.box(215, 150, 185, 56, ["Qwen3-VL-Embedding-2B", "frozen; one pass per view"], "frozen", 12.5)
    s.arrow([(170, 178), (213, 178)])
    s.arrow([(200, 310), (207, 310), (207, 192), (213, 192)])
    s.box(420, 150, 420, 100, ["the system prompt makes the view",
                               "default: \u201cRepresent the user\u2019s input\u201d",
                               "emotion: \u201cRepresent the emotion of this artwork, or of this description",
                               "of an artwork. Focus only on the emotion.\u201d",
                               "style, genre: \u201cDescribe the <aspect> of this artwork in one word\u201d"], "op", 10.5)
    s.arrow([(418, 178), (402, 178)])
    s.arrow([(840, 200), (868, 200)], "six views", 11, lx=854, ly=186)
    bx, fx = 880, 955
    s.text(bx, 142, "default view: the network\u2019s input", 11, "start", fill="#444444", italic=True)
    s.emb_bar(bx, 150, 66, 11, DEFAULT_BLUE, "")
    s.formula(fx, 160, "x_{p} = Qwen(image_{p})", 12)
    s.emb_bar(bx, 172, 66, 11, DEFAULT_GREY, "")
    s.formula(fx, 182, "c_{p} = norm( mean_{k} Qwen(caption_{p,k}) ):  the painting mean", 12)
    s.text(bx, 213, "instructed images: the teacher targets; the style and genre graphs", 11, "start", fill="#444444", italic=True)
    for k, (a, col) in enumerate(zip(G, GREEN)):
        s.emb_bar(bx, 221 + k * 22, 66, 11, col, "")
        s.formula(fx, 231 + k * 22, f"t^{{{a}}}_{{p}} = Qwen_{{{a}}}(image_{{p}})", 12)
    s.text(bx, 300, "instructed captions: the emotion graph", 11, "start", fill="#444444", italic=True)
    s.emb_bar(bx, 308, 66, 11, GREEN[0], "")
    s.formula(fx, 318, "e_{p} = norm( mean_{k} Qwen_{emo}(caption_{p,k}) )", 12)
    s.box(215, 262, 625, 128, ["why the mean, why the instruction",
                               "a painting has several captions by different viewers; their mean embedding is one caption",
                               "vector per painting, so images and captions pair one to one. An instruction changes which",
                               "paintings Qwen places near each other: on 5,000 development paintings the emotion instruction",
                               "raised emotion AMI from 0.0696 to 0.0929 (image side), the style instruction raised style AMI",
                               "from 0.2217 to 0.3952, and each lowered most other-aspect AMIs (stage report, step 2).",
                               "Qwen never trains; the model learns these geometries from the default view alone."], "note", 11)

    # ------------------------------------------------------------------ 2. graphs (full width)
    s.panel(30, 422, 1540, 340, "2  Buddy graphs: one per aspect, built on the training paintings only, no labels")
    s.formulas(50, 472, [
        "scaled similarity in a space V:   sim_{V}(i, j) = cos(v_{i}, v_{j}) / sd_{V},     sd_{V} = std of cos over 200,000 random pairs (seed 0)",
        "content(i, j) = \u00bd [ sim_{D}(i, j) + sim_{I}(i, j) ]                D: the default caption means c,  I: the default images x",
        "emotion score(i, j) = sim_{E}(i, j) \u2212 \u03bb_{i} \u00b7 content(i, j)          E: the emotion caption means e",
        "style score(i, j) = sim_{S}(i, j) \u2212 0.5 \u00b7 content(i, j)            S: the style-instructed images t^{sty}",
        "genre score(i, j) = cos(t^{gen}_{i}, t^{gen}_{j})                        nothing subtracted",
    ], 12)
    s.formulas(50, 588, [
        "\u03bb_{i} = | plain emotion buddies of i \u2229 content buddies of i |  /  | plain emotion buddies of i |",
        "plain emotion buddies: mutual top-20 in E alone;  content buddies: mutual top-20 in D or in I;",
        "a painting with no plain emotion buddy gets the mean \u03bb.  Example at right: 2 of 4, so \u03bb_{i} = 0.5 and half",
        "of content(i, j) is subtracted from every score of anchor i.",
    ], 12)
    s.box(740, 472, 450, 84, ["from scores to a graph, per aspect",
                              "for each anchor i, the 20 highest scores (self excluded);",
                              "keep the pair (i, j) only if each is in the other\u2019s 20 (mutual);",
                              "every kept pair has weight 1"], "op", 11.5)
    s.box(740, 572, 450, 110, ["what the subtraction does",
                               "a content twin of i (same scene, little emotion in common) loses",
                               "score and drops out of the top-20; an emotion twin with a different",
                               "scene gains and comes in. The weight \u03bb of i is large when its emotion",
                               "neighbours are mostly content neighbours, so the subtraction is",
                               "strongest where emotion and content coincide."], "note", 11)
    s.lambda_example(1230, 462)

    # ------------------------------------------------------------------ 3. network
    s.panel(30, 782, 740, 440, "3  The network: two sides; a first layer shared by the aspects, a second layer per aspect")
    s.formulas(50, 832, [
        "image side:    h = GELU( W_{1} x_{p} ), dropout 0.1;     z^{a}_{img}(p) = norm( W^{a}_{2} h ),   a \u2208 {emotion, style, genre}",
        "caption side:  the same shape on c_{p} with its own weights W\u2032_{1}, W\u2032^{a}_{2}; the sides share nothing",
        "W_{1}: 2048 \u00d7 1024, one per side, shared by the three aspects;   W^{a}_{2}: 1024 \u00d7 2048, one per aspect and side",
        "parameters: 8.4 M per side, 16.8 M in all; three separate two-layer MLPs per side would be 25.2 M (not chosen, step 12)",
    ], 12)
    for side, y0, bar_col, W, zs in (("img", 900, DEFAULT_BLUE, "W", "x_{p}"), ("cap", 1030, DEFAULT_GREY, "W\u2032", "c_{p}")):
        s.emb_bar(60, y0 + 30, 66, 11, bar_col, "")
        s.formula(60, y0 + 56, zs, 12)
        s.arrow([(130, y0 + 35), (173, y0 + 35)])
        s.box(175, y0 + 12, 150, 46, [f"{W}_{{1}}, GELU", "2048 \u2192 1024"], "trained", 12, trained=True, math=True)
        for k, (a, col) in enumerate(zip(G, GREEN)):
            yh = y0 + k * 36
            s.box(360, yh, 110, 28, [f"{W}^{{{a}}}_{{2}}"], "trained", 12, math=True)
            s.arrow([(325, y0 + 35), (358, yh + 14)])
            s.arrow([(470, yh + 14), (498, yh + 14)])
            s.emb_bar(500, yh + 9, 66, 11, col, "")
            s.formula(576, yh + 19, f"z^{{{a}}}_{{{side}}}(p)", 12)
    s.text(395, 1016, "W\u2081 is updated by every aspect\u2019s loss; each W\u2082 only by its own aspect\u2019s", 11, "middle", fill="#444444", italic=True)
    s.box(50, 1138, 700, 72, ["another image encoder as the student\u2019s input",
                              "x_{p} comes from SigLIP, CLIP or DINOv2 instead of Qwen; only W_{1} changes its input width (SigLIP 2: 1,152;",
                              "CLIP B/32: 512; DINOv2-large: 1,024). The caption side, the graphs and the teacher targets stay Qwen\u2019s."], "note", 11, math=True)

    # ------------------------------------------------------------------ 4. losses
    s.panel(790, 782, 780, 440, "4  The losses, for one aspect a and one batch of B = 512 buddy pairs (i, j) from a\u2019s graph")
    s.formulas(805, 832, [
        "InfoNCE between two lists of outputs u_{k}, v_{k} (k = 1 \u2026 B):",
        "NCE(u, v) = \u00bd mean_{k} [ \u2212log softmax_{l}( u_{k}\u00b7v_{l} / \u03c4 )_{k}  \u2212  log softmax_{l}( v_{k}\u00b7u_{l} / \u03c4 )_{k} ]",
        "the k-th pair is the positive; the other B \u2212 1 paintings of the batch are the negatives",
    ], 12)
    s.matrix(1400, 845, 4, 22, "i", "j")
    s.formula(1444, 950, "u_{k}\u00b7v_{l} / \u03c4:  the diagonal is the positive", 10.5, "middle", "#444444")
    s.text(1444, 964, "(4 of the 512 pairs drawn)", 10.5, "middle", fill="#444444", italic=True)
    s.box(805, 908, 485, 96, ["buddy loss, 4/5 of the weight (both sides, cross pairs included)",
                              "L^{a}_{buddy} = \u00bc [ NCE(z_{img}(i), z_{img}(j)) + NCE(z_{img}(i), z_{cap}(j))",
                              "+ NCE(z_{cap}(i), z_{img}(j)) + NCE(z_{cap}(i), z_{cap}(j)) ]",
                              "a painting\u2019s outputs move toward its buddy\u2019s, away from the batch\u2019s others"], "loss", 11.5, math=True)
    s.box(805, 1016, 485, 96, ["teacher loss, 1/5 of the weight (image side only; direct distillation)",
                               "L^{a}_{teach} = mean_{n} \u2212log softmax_{m}( z^{a}_{img}(p_{n}) \u00b7 t^{a}_{p_m} / \u03c4 )_{n}",
                               "over the 2B paintings of the batch (both endpoints of every pair);",
                               "the same painting is masked out of the negatives"], "loss", 11.5, math=True)
    s.box(1310, 1016, 245, 96, ["stopping, per aspect",
                                "after each epoch: L^{a} on the stopping",
                                "paintings\u2019 pairs (6,152 paintings),",
                                "no labels; a new minimum saves the",
                                "heads and the trunk; stop after",
                                "15 epochs without one"], "note", 10.5, math=True)
    s.formulas(805, 1140, [
        "L^{a} = L^{a}_{buddy} + 0.25 \u00b7 L^{a}_{teach};    one batch per aspect per step, the three aspects in turn",
        "own-caption terms, \u00bd [ NCE(z_{img}(i), z_{cap}(i)) + NCE(z_{img}(j), z_{cap}(j)) ], have weight 0 in the current recipe",
    ], 12)
    s.text(1180, 1200, "settings: AdamW, learning rate 1e-3, weight decay 0.05, batch 512, at most 200 epochs, patience 15", 11, "middle", fill="#444444", italic=True)

    # ------------------------------------------------------------------ 5. test
    s.panel(30, 1242, 1540, 330, "5  Test time: one default-image pass, k-means per aspect, AMI against the labels")
    s.painting(50, 1300, 120, 80, "painting p")
    s.arrow([(170, 1340), (203, 1340)])
    s.box(205, 1312, 170, 56, ["image encoder, frozen", "Qwen; or SigLIP, CLIP"], "frozen", 12)
    s.arrow([(375, 1340), (408, 1340)])
    s.emb_bar(410, 1335, 66, 11, DEFAULT_BLUE, "")
    s.formula(443, 1365, "x_{p}", 12, "middle")
    s.arrow([(480, 1340), (513, 1340)])
    s.box(515, 1300, 300, 80, ["trunk and three heads, as trained",
                               "z^{a}(p) = norm( W^{a}_{2} GELU( W_{1} x_{p} ) )",
                               "for a = emotion, style, genre; one trunk pass"], "trained", 12, trained=True, math=True)
    s.line([(815, 1340), (835, 1340), (835, 1282), (860 + 300 + 65, 1282)])
    for k, (col, lab) in enumerate(zip(GREEN, ["emotion space", "style space", "genre space"])):
        sx = 860 + k * 150
        s.arrow([(sx + 65, 1282), (sx + 65, 1293)])
        s.scatter(sx, 1295, 130, 90, 10 + k, [col, "#C98B2E", "#1D6FD1", "#A6A49E"], lab)
    s.text(1080, 1402, "k-means with 20 clusters per aspect, on the aspect\u2019s outputs of the test paintings (4 clusters drawn)", 11, "middle", fill="#444444", italic=True)
    s.arrow([(1292, 1340), (1343, 1340)])
    s.box(1345, 1300, 210, 80, ["score per aspect", "AMI( clusters_{a} , labels_{a} )", "adjusted mutual information"], "op", 12, math=True)
    s.box(50, 1430, 740, 120, ["the labels and the sets (labels appear only here)",
                               "emotion: the mode of the caption labels of the painting (nine ArtELingo emotions); style and genre: WikiArt\u2019s.",
                               "held set: 12,281 ArtELingo paintings, read with 5 model seeds; development: 29,232 fresh paintings.",
                               "an instruction-free pass: no caption, no prompt, no PercepT topic classifier at test time."], "note", 11)
    s.box(810, 1430, 745, 120, ["where it stands (held emotion AMI, 2026-10-26)",
                                "ours, Qwen input: 0.1029;  SigLIP 1: 0.1036;  SigLIP 2: 0.1024;  CLIP L/14: 0.0996  (passes)",
                                "bar: PercepT core\u2019s own reading on the Qwen image, 0.0874; a pass needs +0.0100 with the interval above zero",
                                "CLIP B/16 (0.0945) and B/32 (0.0885) inconclusive; DINOv2-large fails. Style and genre: development data only."], "note", 11)
    return s.save("framework_detail")


if __name__ == "__main__":
    print(figure())
