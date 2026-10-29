"""Eight kept changes from the weekly draft's FIGURE SPEC, without shortening.

Uses DejaVu Sans and ink from ../2026-10-28_recipe_sweep/build_model_structure.py.
The explicit spec's teal (#1baf7a) takes precedence over the reference's teal.
Run with PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python build_build_up.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GREY, BLUE, PREVIOUS, GREEN, PERCEPT = "#a6a49e", "#1d6fd1", "#8fb8ea", "#4a9c5b", "#c98b2e"
CHANGED, NEW = "#eb6834", "#1baf7a"
WIDTH, DPI, SIZE, LEADING = 1600, 200, 12.5, 40
TITLE = "Each step kept one change; the instructed graphs and the relational genre term carry the most"
SUBTITLE = "Development paintings of each step; each row against its own reference, so rows are not isolated effects"
FOOTER = "Numbers: development AMI (triples are emotion/style/genre) or mean gain in seed ranges (emotion 0.0046, style 0.0082, genre 0.0128)."
# Dates, kept change, evidence, alternatives: exact supplied wording and numbers.
ROWS = [
    ("10-18", "Instructed graphs, one per aspect",
     "one default-view graph in their place: style −0.1681, emotion −0.0222, genre −0.0287 (ablation 10-21)",
     ["random pairs, degrees kept: style −0.1702", "no instruction anywhere: style −0.1849"]),
    ("10-18", "Content correction (residual graphs) for emotion and style",
     "target AMI, three seeds: emotion 0.137 vs plain 0.118, style 0.260 vs plain 0.213",
     ["plain aspect graph", "genre: image 0.75 + caption 0.25 mix 0.420 vs image graph 0.428",
      "no content correction: style −0.0929 (ablation 10-21)"]),
    ("10-19", "Shared trunk",
     "trunk without teacher 0.0930/0.3591/0.4603 vs separate heads 0.0894/0.3539/0.4574; 16,791,552 vs 25,184,256 parameters",
     ["separate heads"]),
    ("10-19", "Direct teacher, weight 0.5",
     "emotion 0.0962 vs 0.0930 without teacher",
     ["weight 2: lower on every aspect", "relational distillation at 1: 0.0884/0.3700/0.4891",
      "distillation only: 0.0825/0.2751/0.4366 (1 seed)"]),
    ("10-25", "Own-caption terms removed, teacher share 1/5",
     "+0.0061/+0.0240/+0.0070, mean gain 1.60 seed ranges, seed 0",
     ["teacher share 1/3: fails style and genre guards (−1.36)", "image endpoints only: 1.55",
      "no own-caption terms at 1/3: 0.10"]),
    ("10-26", "Relational genre term, weight 8",
     "genre 0.5022 vs 0.4694, +0.0328 [+0.0235, +0.0442], seeds 1 to 3",
     ["teacher weight x2: 0.4594", "teacher weight x4: 0.4478", "regression term: 0.4675 (fails style guard)",
      "relational 0.5: fails emotion guard", "weights 2 and 4: 0.4948, 0.4938 (seed 0)"]),
    ("10-27", "Sharper teacher row, temperature 0.02",
     "genre 0.5050 vs weight 8's 0.5022 (seeds 1 to 3; within one seed range)",
     ["temperature 0.01: 0.4882, fails style guard", "weight 16: 0.5022, weight 32: 0.4964 (seed 0)",
      "relational term alone: 0.4972", "4,096 teacher anchors: 0.4900"]),
    ("10-27", "40 graph neighbours",
     "mean gain +0.683 [+0.452, +0.959] over the previous recipe, seeds 1 to 3",
     ["+ style relational 2 + trunk 512: +0.369", "+ teacher row 0.035: +0.272, does not qualify",
      "30 neighbours, 10 neighbours (seed 0)"]),
]


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": SIZE,
                         "text.color": INK, "figure.facecolor": "white", "savefig.facecolor": "white"})
    renderer = RendererAgg(WIDTH, 100, DPI)

    def wrap(label, width, size=SIZE, weight="normal"):
        prop = FontProperties(family="DejaVu Sans", size=size, weight=weight)
        lines, line = [], ""
        for word in label.split():
            # Preserve numbers verbatim while allowing long AMI triplets to
            # break at their existing slash separators.
            while renderer.get_text_width_height_descent(word, prop, False)[0] > width:
                if line:
                    lines.append(line)
                    line = ""
                cuts = [j + 1 for j, char in enumerate(word) if char == "/"
                        and renderer.get_text_width_height_descent(word[:j + 1], prop, False)[0] <= width]
                assert cuts, f"No safe wrap boundary: {word}"
                cut = max(cuts)
                lines.append(word[:cut])
                word = word[cut:]
            test = f"{line} {word}".strip()
            if line and renderer.get_text_width_height_descent(test, prop, False)[0] > width:
                lines.append(line)
                line = word
            else:
                line = test
        return "\n".join(lines + [line])

    # Date gutter + three columns. Each alternative receives its own dashed box.
    left_x, left_w, mid_x, mid_w, right_x, right_w = 170, 340, 550, 470, 1060, 500
    title = wrap(TITLE, 1520, 18, "bold")
    subtitle = wrap(SUBTITLE, 1520)
    legend_y = 36 + len(title.splitlines()) * 58 + 22 + len(subtitle.splitlines()) * LEADING + 26
    heading_y = legend_y + LEADING + 34
    y = heading_y + 66
    layouts = []
    for date, change, evidence, alternatives in ROWS:
        change = wrap(change, left_w - 40, weight="bold")
        evidence = wrap(evidence, mid_w - 32)
        alternatives = [wrap(label, right_w - 40) for label in alternatives]
        alt_heights = [len(label.splitlines()) * LEADING + 34 for label in alternatives]
        height = max(len(change.splitlines()) * LEADING + 44,
                     len(evidence.splitlines()) * LEADING + 44,
                     sum(alt_heights) + 16 * (len(alternatives) - 1))
        layouts.append((date, change, evidence, alternatives, alt_heights, y, height))
        y += height + 42
    footer = wrap(FOOTER, 1520)
    height = int(y + len(footer.splitlines()) * LEADING + 30)
    fig = plt.figure(figsize=(WIDTH / DPI, height / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(height, 0))
    ax.axis("off")
    text_checks, box_checks = [], []

    def text(x, y, label, size=SIZE, weight="normal", color=INK):
        t = ax.text(x, y, label, va="top", fontsize=size, fontweight=weight,
                    color=color, linespacing=LEADING / (SIZE * DPI / 72))
        text_checks.append(t)
        return t

    def box(x, y, w, h, edge, teal_fill=False, dashed=False):
        # Teal tint uses the spec colour; no extra palette hue.
        fill = (*matplotlib.colors.to_rgb(NEW), 0.08) if teal_fill else "white"
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=12",
                          facecolor=fill, edgecolor=edge, linewidth=1.3,
                          linestyle=(0, (4, 3)) if dashed else "-")
        ax.add_patch(p)
        return p

    text(40, 36, title, 18, "bold")
    text(40, 36 + len(title.splitlines()) * 58 + 22, subtitle, color=SECONDARY)
    legend_x = 40
    legend_prop = FontProperties(family="DejaVu Sans", size=SIZE)
    for label, edge, dashed in [
        ("in the model by 10-26", BLUE, False),
        ("new on 10-27", NEW, False),
        ("changed by the sweep", CHANGED, False),
        ("alternative tried", GREY, True),
    ]:
        box(legend_x, legend_y + 4, 32, 26, edge, dashed=dashed)
        text(legend_x + 44, legend_y, label)
        legend_x += 44 + renderer.get_text_width_height_descent(label, legend_prop, False)[0] + 24
    text(left_x, heading_y, "Kept change", weight="bold")
    text(mid_x + 16, heading_y, "Evidence", weight="bold")
    text(right_x + 20, heading_y, "Alternatives tried", weight="bold")
    for i, (date, change, evidence, alternatives, alt_heights, top, row_h) in enumerate(layouts):
        edge = NEW if i == 6 else CHANGED if i == 7 else BLUE
        kept = box(left_x, top, left_w, row_h, edge, teal_fill=True)
        t = text(left_x + 20, top + 22, change, weight="bold")
        box_checks.append((t, kept))
        text(40, top + 22, date, color=SECONDARY)
        # The ladder occupies the gutter, away from labels and dashed boxes.
        if i < len(layouts) - 1:
            ax.add_patch(FancyArrowPatch((left_x + left_w / 2, top + row_h + 7),
                                        (left_x + left_w / 2, top + row_h + 35),
                                        arrowstyle="-|>", mutation_scale=12,
                                        linewidth=1.2, color=SECONDARY))
        evidence_text = text(mid_x + 16, top + 22, evidence)
        # Check column bounds as well as figure bounds (long numeric tokens
        # can otherwise cross into the alternatives without being clipped).
        box_checks.append((evidence_text, ax.add_patch(FancyBboxPatch(
            (mid_x, top), mid_w, row_h, boxstyle="square,pad=0",
            facecolor="none", edgecolor="none"))))
        ax.plot([mid_x, mid_x], [top, top + row_h], color=GRID, lw=0.9)
        alt_y = top
        for label, alt_h in zip(alternatives, alt_heights):
            alternative = box(right_x, alt_y, right_w, alt_h, GREY, dashed=True)
            t = text(right_x + 20, alt_y + 17, label)
            box_checks.append((t, alternative))
            alt_y += alt_h + 16
        ax.plot([40, 1560], [top + row_h + 21, top + row_h + 21], color=GRID, lw=0.7, zorder=0)
    text(40, y, footer, color=SECONDARY)
    fig.canvas.draw()
    actual_renderer = fig.canvas.get_renderer()
    for t in text_checks:
        b = t.get_window_extent(actual_renderer)
        assert b.x0 >= 0 and b.x1 <= WIDTH and b.y0 >= 0 and b.y1 <= height, t.get_text()
    for t, p in box_checks:
        b, outer = t.get_window_extent(actual_renderer), p.get_window_extent(actual_renderer)
        assert outer.contains(b.x0, b.y0) and outer.contains(b.x1, b.y1), t.get_text()
    output = Path(__file__).with_name("build_up.png")
    fig.savefig(output, dpi=DPI)
    plt.close(fig)
    print(f"{output}: {WIDTH} x {height} px; {len(ROWS)} rows; "
          f"{sum(len(r[3]) for r in ROWS)} alternatives; {SIZE} pt body text")


if __name__ == "__main__":
    main()
