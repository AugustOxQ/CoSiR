"""Slide adaptation of the source retrieval figure; same verdict data and level zones.

Reads the verdict files of the r11 run (stage 1: plain cosine; stage 5: post-processing picked on validation).
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
          ('stage5', 'post-processing picked on validation', 'SC13 SC14 SC15'.split(), OURS)]
WORD = {'WIN': 'clearly above', 'LEVEL': 'level', 'BELOW': 'below'}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 20})
fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
fig.subplots_adjust(left=0.14, right=0.97, bottom=0.22, top=0.69)
for i, aspect in enumerate(ASPECTS):
    y0 = len(ASPECTS) - 1 - i
    r = None
    for j, (stage, label, ids, face) in enumerate(STAGES):
        c = json.load(open(OUT / stage / 'verdict.json'))['sc'][ids[i]]['comparisons'][0]
        assert c['aspect'] == aspect and c['comparator'][0] == 'instructed'
        r = c['r']
        y = y0 + (0.15 if j == 0 else -0.15)
        lo, hi = c['interval']
        ax.plot([lo * 1e3, hi * 1e3], [y, y], color=OURS, lw=3, zorder=3)
        ax.scatter([c['d'] * 1e3], [y], s=115, facecolor=face, edgecolor=OURS, lw=2.3, zorder=4,
                   label=label if i == 0 else None)
        ax.text(hi * 1e3 + 0.12, y, WORD[c['outcome']], va='center', fontsize=20, color=INK2)
    ax.add_patch(plt.Rectangle((-r * 1e3, y0 - 0.4), r * 1e3, 0.8, color=GRID, zorder=1, lw=0))
ax.axvline(0, color=INK2, lw=0.9, zorder=2)
ax.set_yticks(range(len(ASPECTS)))
ax.set_yticklabels([a.capitalize() for a in ASPECTS[::-1]])
ax.set_ylim(-0.6, len(ASPECTS) - 0.4)
ax.set_xlim(-4.0, 7.0)
ax.set_xlabel('Painting model minus the teacher (mAP × 0.001), 95% interval', labelpad=16)
fig.suptitle('With the picked post-processing, emotion moves from below to level;\nstyle moves from clearly above to level', x=0.035, y=0.96, ha='left', fontsize=25, color=INK)
handles, labels = ax.get_legend_handles_labels()
handles.append(plt.Rectangle((0, 0), 1, 1, color=GRID))
labels.append('"level" zone: within one seed range below zero')
ax.legend(handles, labels, fontsize=18, frameon=False, loc='upper left', bbox_to_anchor=(0.13, 0.84), bbox_transform=fig.transFigure)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.tick_params(colors=INK2, labelsize=20)
fig.text(0.035, 0.045, '12,281 held ArtELingo paintings (genre: 9,949), five model seeds.', fontsize=18, color=INK2)
fig.savefig(HERE / 'teacher_gap_slide.png')
plt.close(fig)
print('wrote', HERE / 'teacher_gap_slide.png')
