"""Weekly timeline from the FIGURE SPEC in the 2026-10-28 draft.

Adapted from ../2026-10-26_condition_free_line/build_timeline.py:
DejaVu Sans, ink, outcome dots, held stars and quiet grey leaders retained.
Run with PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python build_timeline.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties

INK, SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"worked": "#1d6fd1", "did not work": "#eb6834", "check": "#a6a49e", "decision": INK}
WIDTH, DPI, SIZE, LEADING = 1600, 200, 12.5, 40
TITLE = "Two weeks of the condition-free line: checks, model changes, held tests and ablations"
SUBTITLE = "15 to 28 October 2026, Amsterdam dates; held results starred"
FOOTNOTE = "Worked: the step met its rule or bar; check: a measurement with no bar."
# Exact supplied labels, left to right; no interpretation of outcomes or dates.
ROWS = [
    (15, [("Conditioned line stopped and archived", "decision", False)]),
    (16, []),
    (17, [("Code audit, premise checks", "check", False),
          ("One Qwen instruction per aspect", "check", False),
          ("Image-only student toward captions", "did not work", False),
          ("Emotion-word masking", "check", False)]),
    (18, [("Student on purified pairs", "did not work", False),
          ("Aspect graph sheet", "check", False),
          ("Graph follow-up", "check", False)]),
    (19, [("First PercepT core run", "check", False),
          ("PercepT's own topic-count rule", "check", False),
          ("PercepT core, 40 topics", "check", False),
          ("Labelled ceiling probe", "check", False),
          ("First shared model", "worked", False)]),
    (20, [("Held emotion claim test, two scales", "worked", True)]),
    (21, [("Framing note", "check", False),
          ("Ablation and four Qwen views", "check", False),
          ("Attention maps (to 10-25)", "check", False)]),
    (22, None), (23, None), (24, None),
    (25, [("Recipe pick; SigLIP 2 and Qwen pass", "worked", True)]),
    (26, [("Four more encoders; genre term (development)", "worked", True)]),
    (27, [("Relational genre term, held genre level", "worked", True),
          ("Previous recipe's held emotion test", "worked", True),
          ("Recipe sweep, 13 settings and grid", "check", False),
          ("Sweep's pick, held test", "worked", True),
          ("Development ablation", "check", False)]),
    (28, [("Held ablation (to 03:53)", "check", True), ("Reports", "check", False)]),
]


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": SIZE,
                         "text.color": INK, "figure.facecolor": "white", "savefig.facecolor": "white"})
    renderer = RendererAgg(WIDTH, 100, DPI)

    def wrap(label, width, size=SIZE):
        prop = FontProperties(family="DejaVu Sans", size=size)
        lines, line = [], ""
        for word in label.split():
            test = f"{line} {word}".strip()
            if line and renderer.get_text_width_height_descent(test, prop, False)[0] > width:
                lines.append(line)
                line = word
            else:
                line = test
        return "\n".join(lines + [line])

    title, subtitle = wrap(TITLE, 1520, 18), wrap(SUBTITLE, 1520)
    y = 36 + len(title.splitlines()) * 58 + 18 + len(subtitle.splitlines()) * LEADING + 24
    legend_y = y + 17
    y += 77
    layouts = []
    for day, steps in ROWS:
        if steps:
            # Keep sparse days concise and dense days in five aligned columns.
            cell_w = 278 if len(steps) > 2 else 620
            labels = [wrap(s[0], cell_w - 26) for s in steps]
            h = 60 + max(len(s.splitlines()) for s in labels) * LEADING + 28
        else:
            cell_w, labels, h = 0, [], 88
        layouts.append((day, steps, y, h, cell_w, labels))
        y += h
    foot = wrap(FOOTNOTE, 1520)
    height = int(y + 28 + len(foot.splitlines()) * LEADING + 30)
    fig = plt.figure(figsize=(WIDTH / DPI, height / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(height, 0))
    ax.axis("off")
    texts = []

    def text(x, y, label, size=SIZE, color=INK):
        t = ax.text(x, y, label, va="top", fontsize=size, color=color,
                    linespacing=LEADING / (SIZE * DPI / 72))
        texts.append(t)
        return t

    text(40, 36, title, 18)
    text(40, 36 + len(title.splitlines()) * 58 + 18, subtitle, color=SECONDARY)
    for x, label, marker, color in [(50, "worked", "o", COLORS["worked"]),
                                    (310, "did not work", "o", COLORS["did not work"]),
                                    (670, "check", "o", COLORS["check"]),
                                    (925, "decision", "D", INK),
                                    (1240, "held data", "*", INK)]:
        ax.scatter([x], [legend_y], s=65 if marker != "*" else 90, marker=marker, color=color)
        text(x + 24, legend_y - 19, label)
    for day, steps, top, h, cell_w, labels in layouts:
        ax.plot([40, 1560], [top, top], color=GRID, lw=0.8)
        text(40, top + 20, f"10-{day}", color=SECONDARY)
        if steps:
            ax.plot([173, 1560], [top + 25, top + 25], color=GRID, lw=1.1, zorder=1)
            for i, ((_, outcome, held), label) in enumerate(zip(steps, labels)):
                x = 180 + i * cell_w
                ax.scatter([x], [top + 25], s=65, marker="D" if outcome == "decision" else "o",
                           color=COLORS[outcome], zorder=4)
                if held:
                    ax.scatter([x + 32], [top + 25], marker="*", s=90, color=INK, zorder=4)
                text(x - 7, top + 53, label)
        elif steps is None:
            ax.plot([180, 430], [top + 38, top + 38], color=COLORS["check"], lw=1.1)
            text(455, top + 18, "Attention maps", color=SECONDARY)
        else:
            ax.plot([180, 430], [top + 38, top + 38], color=GRID, lw=1.1)
            text(455, top + 18, "no step", color=COLORS["check"])
    text(40, y + 28, foot, color=SECONDARY)
    fig.canvas.draw()
    for t in texts:
        b = t.get_window_extent(fig.canvas.get_renderer())
        assert b.x0 >= 0 and b.x1 <= WIDTH and b.y0 >= 0 and b.y1 <= height, t.get_text()
    output = Path(__file__).with_name("timeline.png")
    fig.savefig(output, dpi=DPI)
    plt.close(fig)
    print(f"{output}: {WIDTH} x {height} px; {sum(len(s) for _, s in ROWS if s)} steps; "
          f"{sum(s[2] for _, ss in ROWS if ss for s in ss)} held stars")


if __name__ == "__main__":
    main()
