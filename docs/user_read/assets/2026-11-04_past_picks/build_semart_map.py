"""SemArt new artists, genre mAP: plain image, c* and teacher per encoder. Values come from records/outcomes.json."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = json.loads((ROOT / "src/test/20261103_past_picks/records/outcomes.json").read_text())

PLAIN, CSTAR, TEACHER = "#a6a49e", "#1d6fd1", "#4a9c5b"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORDER = ["qwen", "siglip2", "siglip1", "clip_l14", "dinov2_large", "clip_b32", "clip_b16"]
NAMES = {
    "qwen": "Qwen",
    "siglip2": "SigLIP 2",
    "siglip1": "SigLIP 1",
    "clip_l14": "CLIP L/14",
    "dinov2_large": "DINOv2",
    "clip_b32": "CLIP B/32",
    "clip_b16": "CLIP B/16",
}

rows = [c for c in OUT["cells"] if c["data"] == "semart_new" and c["task"] == "ret"]
plain, cstar, teacher, outcome = {}, {}, {}, {}
for c in rows:
    enc = c["encoder"]
    if c["comparator"].startswith("frozen"):
        plain[enc] = c["comparator_value"]
        cstar[enc] = c["candidate_value"]
    elif "teacher" in c["comparator"]:
        teacher[enc] = c["comparator_value"]
        outcome[enc] = c["outcome"]
assert set(plain) == set(ORDER) == set(teacher), (sorted(plain), sorted(teacher))
assert len(set(round(v, 6) for v in teacher.values())) == 1
line = next(iter(teacher.values()))

fig, ax = plt.subplots(figsize=(8.0, 4.1), dpi=200)
ys = list(range(len(ORDER)))[::-1]
for y, enc in zip(ys, ORDER):
    ax.plot([plain[enc], cstar[enc]], [y, y], color=GRID, lw=2.5, zorder=1, solid_capstyle="round")
    ax.scatter(plain[enc], y, s=70, color=PLAIN, zorder=3, label="Plain image" if y == ys[0] else None)
    ax.scatter(cstar[enc], y, s=70, color=CSTAR, zorder=3, label="c*" if y == ys[0] else None)
    ax.text(
        0.728, y, outcome[enc], va="center", ha="left", fontsize=9,
        color=INK if outcome[enc] == "LEVEL" else INK2,
    )
ax.axvline(line, color=TEACHER, lw=2, zorder=2, label="Teacher (Qwen image, instructed for genre)")
ax.text(0.728, len(ORDER) - 0.35, "c* against\nteacher", va="bottom", ha="left", fontsize=8.5, color=INK2)
ax.set_yticks(ys, [NAMES[e] for e in ORDER], fontsize=10, color=INK)
ax.set_xlim(0.54, 0.72)
ax.set_ylim(-0.6, len(ORDER) - 0.4)
ax.set_xlabel("Genre mAP on SemArt new artists (higher is better)", fontsize=10, color=INK2)
ax.tick_params(axis="x", labelsize=9, colors=INK2)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.legend(loc="upper center", bbox_to_anchor=(0.46, -0.17), ncol=3, frameon=False, fontsize=9)
ax.set_title(
    "c* beats the plain image on all seven encoders; three reach LEVEL with the teacher",
    fontsize=11, color=INK, loc="left", pad=30,
)
fig.savefig(HERE / "semart_genre_map.png", bbox_inches="tight", facecolor="white")
for e in ORDER:
    print(NAMES[e], round(plain[e], 4), round(cstar[e], 4), round(teacher[e], 4), outcome[e])
