"""Figure for the r11 user-read report: c* minus the instructed Qwen image, per aspect, at both read-outs.

Reads the verdict files of the r11 run (stage 1: plain cosine; stage 5: read-out picked on validation).
Usage: python make_teacher_gap.py [outputs_dir]
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else '/data/SSD2/cosir_r11/20261029_retrieval/outputs')
HERE = Path(__file__).resolve().parent
OURS = '#1d6fd1'  # palette.md: our network
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
ASPECTS = ['emotion', 'style', 'genre']
STAGES = [('stage1', 'plain cosine', 'SC4 SC5 SC6'.split(), 'white'),
          ('stage5', 'read-out picked on validation', 'SC13 SC14 SC15'.split(), OURS)]
WORD = {'WIN': 'win', 'LEVEL': 'level', 'BELOW': 'below'}

fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
for i, aspect in enumerate(ASPECTS):
    y0 = len(ASPECTS) - 1 - i
    r = None
    for j, (stage, label, ids, face) in enumerate(STAGES):
        c = json.load(open(OUT / stage / 'verdict.json'))['sc'][ids[i]]['comparisons'][0]
        assert c['aspect'] == aspect and c['comparator'][0] == 'instructed'
        r = c['r']
        y = y0 + (0.15 if j == 0 else -0.15)
        lo, hi = c['interval']
        ax.plot([lo * 1e3, hi * 1e3], [y, y], color=OURS, lw=2, zorder=3)
        ax.scatter([c['d'] * 1e3], [y], s=46, facecolor=face, edgecolor=OURS, lw=1.6, zorder=4,
                   label=label if i == 0 else None)
        ax.text(hi * 1e3 + 0.12, y, WORD[c['outcome']], va='center', fontsize=8.5, color=INK2)
    ax.add_patch(plt.Rectangle((-r * 1e3, y0 - 0.4), r * 1e3, 0.8, color=GRID, zorder=1, lw=0))
ax.axvline(0, color=INK2, lw=0.9, zorder=2)
ax.set_yticks(range(len(ASPECTS)))
ax.set_yticklabels(ASPECTS[::-1])
ax.set_ylim(-0.6, len(ASPECTS) - 0.4)
ax.set_xlim(-4.0, 7.0)
ax.set_xlabel('c* minus the instructed Qwen image (mAP, x 0.001), 95% interval')
ax.set_title('After the read-out, c* is level with or above its teacher on every aspect',
             fontsize=10, color=INK, loc='left')
handles, labels = ax.get_legend_handles_labels()
handles.append(plt.Rectangle((0, 0), 1, 1, color=GRID))
labels.append('"level" zone: within one seed range below zero')
ax.legend(handles, labels, fontsize=8, frameon=False, loc='upper right', bbox_to_anchor=(1.0, 1.02))
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.tick_params(colors=INK2, labelsize=9)
fig.text(0.01, 0.01, '12,281 held ArtELingo paintings (genre: 9,949), 5 seeds of c*.', fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(HERE / 'teacher_gap.png')
print('wrote', HERE / 'teacher_gap.png')
