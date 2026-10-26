"""Illustration of the condition-free model's framework (low detail), in the style of MultiMAE's idea figures.

Facts: stage report 2026-10-26_condition_free_line.md, "The design, as it stands on 2026-10-26"; weight sharing
checked in src/test/20261019_model_sheet/ms_train.py (two TrunkSide modules, 'img' and 'cap').
Run: PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python build_framework_figure.py
"""
from pathlib import Path
import subprocess

OUT = Path(__file__).parent

TRAINED_FILL, TRAINED_STROKE = "#FDE7CF", "#E07B20"   # trained module
FROZEN_FILL, FROZEN_STROKE = "#EEE8F6", "#7B5EA7"     # frozen encoder
DATA_FILL, DATA_STROKE = "#F3F3F3", "#7A7A7A"          # data, embeddings
OP_FILL, OP_STROKE = "#DCEBF7", "#3A78B5"              # fixed operation
LOSS_FILL, LOSS_STROKE = "#E3F2E1", "#3C8C3C"          # loss
NOTE_STROKE = "#9A9A9A"
INK = "#222222"
FONT = "Lato, DejaVu Sans, sans-serif"
KINDS = {"trained": (TRAINED_FILL, TRAINED_STROKE), "frozen": (FROZEN_FILL, FROZEN_STROKE),
         "data": (DATA_FILL, DATA_STROKE), "op": (OP_FILL, OP_STROKE), "loss": (LOSS_FILL, LOSS_STROKE),
         "note": ("#FFFFFF", NOTE_STROKE)}
# Instructed-Qwen green and its tints, as in the stage report's model figure
GREEN = ["#4a9c5b", "#74b382", "#a2cdaa"]
DEFAULT_BLUE, DEFAULT_GREY = "#5B8DB8", "#9AA8B4"


class SVG:
    def __init__(self, w, h):
        self.w, self.h, self.parts = w, h, []

    def text(self, x, y, s, fs=14, anchor="middle", weight="normal", fill=INK, italic=False):
        style = f"font-family:{FONT};font-size:{fs}px;font-weight:{weight};" + ("font-style:italic;" if italic else "")
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" style="{style}">{s}</text>')

    def rect(self, x, y, w, h, fill, stroke, dash=False, rx=8, sw=1.6):
        d = ' stroke-dasharray="7,5"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def box(self, x, y, w, h, lines, kind="data", fs=14, trained=False):
        fill, stroke = KINDS[kind]
        self.rect(x, y, w, h, fill, stroke, dash=(kind == "note"))
        lh = fs * 1.3
        y0 = y + h / 2 - (len(lines) - 1) * lh / 2 + fs * 0.35
        for i, s in enumerate(lines):
            self.text(x + w / 2, y0 + i * lh, s, fs, "middle", "bold" if i == 0 else "normal")
        if trained:
            self.badge(x + w - 8, y - 9, "trained")

    def badge(self, xr, y, label):
        bw = 58
        self.rect(xr - bw, y, bw, 18, TRAINED_STROKE, TRAINED_STROKE, rx=9)
        self.text(xr - bw / 2, y + 13, label, 11, "middle", "bold", "#FFFFFF")

    def arrow(self, pts, label=None, fs=12, lx=None, ly=None, anchor="middle", dash=False, color=INK, sw=1.6):
        p = " ".join(f"{x},{y}" for x, y in pts)
        d = ' stroke-dasharray="6,5"' if dash else ""
        self.parts.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}" marker-end="url(#ah)"{d}/>')
        if label:
            (x1, y1), (x2, y2) = pts[0], pts[-1]
            self.text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2 - 6,
                      label, fs, anchor, "normal", "#444444", italic=True)

    def bracket(self, x, y0, y1, tick=7, color="#555555"):
        self.line([(x - tick, y0), (x, y0), (x, y1), (x - tick, y1)], color=color, sw=1.4)

    def line(self, pts, color=INK, sw=1.6):
        p = " ".join(f"{x},{y}" for x, y in pts)
        self.parts.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}"/>')

    def painting(self, x, y, w, h, label):
        self.rect(x, y, w, h, "#FFFFFF", DATA_STROKE, rx=6)
        self.parts.append(f'<rect x="{x+3}" y="{y+3}" width="{w-6}" height="{h-6}" rx="4" fill="#E8EEF3"/>')
        self.parts.append(f'<line x1="{x+8}" y1="{y+h*0.62}" x2="{x+w-8}" y2="{y+h*0.62}" stroke="#8AA0B0" stroke-width="1.5"/>')
        self.parts.append(f'<circle cx="{x+w*0.72}" cy="{y+h*0.33}" r="{min(w,h)*0.1}" fill="#F0C060"/>')
        self.text(x + w / 2, y + h - 8, label, 13, "middle", "bold")

    def emb_bar(self, x, y, w, h, color, label, segs=12, label_fs=12):
        sw = w / segs
        for i in range(segs):
            self.parts.append(f'<rect x="{x+i*sw:.1f}" y="{y}" width="{sw-1:.1f}" height="{h}" fill="{color}"/>')
        self.rect(x - 1, y - 1, w + 1, h + 2, "none", DATA_STROKE, rx=2, sw=1)
        self.text(x + w + 8, y + h / 2 + 4, label, label_fs, "start")

    def legend_row(self, x, y):
        items = [("trained", "trained module", 110), ("frozen", "frozen encoder", 110), ("data", "data, embeddings", 120),
                 ("op", "fixed operation", 110), ("loss", "loss", 50), ("note", "note", 50)]
        for kind, lab, adv in items:
            fill, stroke = KINDS[kind]
            self.rect(x, y, 26, 14, fill, stroke, dash=(kind == "note"), rx=3)
            self.text(x + 34, y + 11, lab, 12, "start", fill="#444444")
            x += 34 + adv

    def panel(self, x, y, w, h, title):
        self.rect(x, y, w, h, "#FAFAFA", "#CCCCCC", rx=12, sw=1.2)
        self.text(x + 14, y + 24, title, 17, "start", "bold")

    def heading(self, x, y, s):
        self.text(x, y, s, 15, "start", "bold", "#444444")

    def save(self, name):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
                '<defs><marker id="ah" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto" markerUnits="userSpaceOnUse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker></defs>'
                f'<rect width="{self.w}" height="{self.h}" fill="#FFFFFF"/>')
        p = OUT / f"{name}.svg"
        p.write_text(head + "\n".join(self.parts) + "</svg>")
        subprocess.run(["rsvg-convert", "-w", str(int(self.w * 1.6)), str(p), "-o", str(OUT / f"{name}.png")], check=True)
        return p


def figure():
    s = SVG(1600, 950)
    s.text(800, 36, "The condition-free model: train on Qwen’s instructed views, read the default image once", 22, weight="bold")
    s.text(800, 62, "The instructed views build the training signal only (buddy graphs and teacher targets). "
                    "At test time one default image gives emotion, style and genre spaces.", 15, fill="#444444")
    s.legend_row(420, 78)

    # ---------------------------------------------------------------- training
    s.panel(30, 102, 1540, 590, "Training: on the training paintings only, no labels; the same recipe for each aspect (emotion, style, genre)")

    # 1. four views
    s.heading(60, 158, "1  Four views of each painting from one frozen embedder")
    s.painting(60, 185, 130, 84, "painting")
    s.box(60, 285, 130, 46, ["its captions", "several per painting"], "data", 12)
    s.box(225, 212, 190, 60, ["Qwen3-VL-Embedding-2B", "frozen; one pass per view"], "frozen", 12.5)
    s.arrow([(190, 225), (223, 235)])
    s.arrow([(190, 308), (223, 252)])
    s.box(225, 318, 330, 92, ["one prompt per aspect makes the instructed views",
                              "“Represent the emotion of this artwork …”",
                              "“Describe the style of this artwork in one word”",
                              "“Describe the genre of this artwork in one word”"], "op", 11)
    s.arrow([(390, 316), (390, 274)])
    s.box(60, 440, 635, 56, ["why instructed views: an instruction makes Qwen cluster the paintings by one aspect",
                             "(own-aspect AMI up, the other aspects down); the network learns that geometry from the",
                             "default view, so it needs no instruction at test time"], "note", 11)
    s.text(430, 177, "default view, no instruction: the network’s input", 11, "start", fill="#444444", italic=True)
    s.emb_bar(430, 186, 90, 12, DEFAULT_BLUE, "default image")
    s.emb_bar(430, 206, 90, 12, DEFAULT_GREY, "default caption (mean)")
    s.text(430, 244, "instructed views: the training signal only", 11, "start", fill="#444444", italic=True)
    s.emb_bar(430, 253, 90, 12, GREEN[0], "emotion view")
    s.emb_bar(430, 273, 90, 12, GREEN[1], "style view")
    s.emb_bar(430, 293, 90, 12, GREEN[2], "genre view")
    s.arrow([(415, 232), (428, 204)])
    s.arrow([(415, 250), (428, 281)])
    # default view -> network inputs (a bracket gathers each group of bars)
    s.bracket(676, 184, 220)
    s.arrow([(676, 192), (748, 202)])
    s.arrow([(676, 212), (715, 212), (715, 332), (748, 332)])
    # instructed views -> graph and teacher
    s.bracket(676, 251, 307)
    s.arrow([(676, 279), (700, 279), (700, 512), (748, 512)])
    s.arrow([(700, 512), (700, 672), (1500, 672), (1500, 626)],
            "the instructed image embedding of the aspect is the teacher target", 12, lx=712, ly=666, anchor="start")

    # 2. the network
    s.heading(750, 158, "2  The network: two sides; each a shared trunk and three aspect heads (trained)")
    s.box(750, 180, 130, 44, ["default image", "embedding"], "data", 12)
    s.arrow([(880, 202), (913, 202)])
    s.box(915, 174, 125, 56, ["image trunk", "2048 → 1024"], "trained", 12.5, trained=True)
    for i, (name, yh) in enumerate([("emotion head", 165), ("style head", 201), ("genre head", 237)]):
        s.box(1075, yh, 125, 30, [name], "trained", 12)
        s.arrow([(1040, 202), (1073, yh + 15)])
    s.text(1000, 283, "each head 1024 → 2048; the two sides share no weights", 11, fill="#444444", italic=True)
    s.box(750, 310, 130, 44, ["default caption", "embedding"], "data", 12)
    s.arrow([(880, 332), (913, 332)])
    s.box(915, 304, 125, 56, ["caption trunk", "2048 → 1024"], "trained", 12.5, trained=True)
    for i, (name, yh) in enumerate([("emotion head", 295), ("style head", 331), ("genre head", 367)]):
        s.box(1075, yh, 125, 30, [name], "trained", 12)
        s.arrow([(1040, 332), (1073, yh + 15)])
    # outputs -> losses: a bracket gathers the three heads of a side; every head enters its aspect's losses
    s.bracket(1206, 163, 269)
    s.arrow([(1206, 216), (1450, 216), (1450, 460)], "image outputs: to both losses", 12, lx=1240, ly=208, anchor="start")
    s.arrow([(1206, 216), (1225, 216), (1225, 460)])
    s.bracket(1206, 293, 399)
    s.arrow([(1206, 346), (1215, 346), (1215, 460)], "caption outputs: to the buddy loss only", 12, lx=1240, ly=338, anchor="start")

    # 3. the signal
    s.heading(750, 448, "3  The training signal, from the instructed views")
    s.box(750, 462, 330, 100, ["buddy graph in the instructed view",
                               "mutual 20-nearest neighbours among the training",
                               "paintings (emotion on captions, style and genre on",
                               "images); emotion and style first subtract the",
                               "default view’s content similarity"], "op", 11.5)
    s.arrow([(915, 562), (915, 578)])
    s.box(750, 580, 330, 48, ["each painting’s buddies", "the paintings close to it in that aspect"], "data", 12)
    s.arrow([(1080, 604), (1098, 604)])

    # 4. the losses
    s.box(1100, 462, 230, 164, ["buddy loss (InfoNCE): 4/5 of the weight",
                                "the aspect\u2019s image and caption heads:",
                                "a painting\u2019s outputs are pulled toward",
                                "its buddies\u2019 outputs on that aspect\u2019s",
                                "graph; the other paintings in the",
                                "batch are pushed away"], "loss", 11.5)
    s.box(1345, 462, 205, 164, ["teacher loss: 1/5 of the weight",
                                "the aspect\u2019s image head only:",
                                "its output is pulled toward the",
                                "instructed image embedding of",
                                "that aspect (direct distillation)"], "loss", 11.5)
    s.heading(1250, 648, "4  The losses, one pair per aspect")

    # ---------------------------------------------------------------- test time
    s.panel(30, 710, 1540, 222, "Test time: one pass of the default image; no instruction, no caption")
    s.painting(60, 775, 120, 80, "painting")
    s.arrow([(180, 815), (213, 815)])
    s.box(215, 785, 200, 60, ["image encoder, frozen", "Qwen; or SigLIP, CLIP"], "frozen", 12.5)
    s.arrow([(415, 815), (448, 815)])
    s.box(450, 793, 130, 44, ["default image", "embedding"], "data", 12)
    s.arrow([(580, 815), (613, 815)])
    s.box(615, 787, 125, 56, ["image trunk"], "trained", 12.5, trained=True)
    for name, yh in [("emotion head", 762), ("style head", 798), ("genre head", 834)]:
        s.box(775, yh, 125, 30, [name], "trained", 12)
        s.arrow([(740, 815), (773, yh + 15)])
        s.arrow([(900, yh + 15), (943, 800 + (yh - 762) * 0.42)])
    s.box(945, 779, 200, 72, ["k-means per aspect", "20 clusters on each", "head’s output"], "op", 12)
    s.arrow([(1145, 815), (1178, 815)])
    s.box(1180, 775, 370, 80, ["scored by AMI against the paintings’",
                               "emotion, style and genre labels (ArtELingo);",
                               "labels enter only here, never training"], "note", 12)
    s.box(215, 878, 930, 40, ["a student on SigLIP or CLIP changes only the image input layer of the trunk; "
                              "graphs, teacher and captions stay Qwen’s at training"], "note", 11)
    return s.save("framework")


if __name__ == "__main__":
    print(figure())
