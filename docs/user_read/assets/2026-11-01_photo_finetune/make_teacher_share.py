"""Share of teacher gain on held photos and paintings (verdict point values).

The photo model is c* fine-tuned on Affection's train photos.
Usage: python make_teacher_share.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
PHOTO_MODEL = Path('/data/SSD2/cosir_r13/20261031_photo_finetune/outputs/stage3/verdict.json')
FROZEN = Path('/data/SSD2/cosir_r12/20261030_affection_confirmation/outputs/stage2/verdict.json')
PAINTINGS = Path('/data/SSD2/cosir_r11/20261029_retrieval/outputs/stage5/verdict.json')
OURS, PHOTO, PLAIN, TEACHER = '#1d6fd1', '#0b3d91', '#a6a49e', '#4a9c5b'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def teacher_share(candidate, plain, teacher):
    assert teacher > plain
    return 100 * (candidate - plain) / (teacher - plain)


def photo_share(sc, plain_id, teacher_id, metric):
    plain, teacher = sc[plain_id], sc[teacher_id]
    assert plain['comparator'] == 'plain' and plain['metric'] == metric
    assert teacher['comparator'] == 'teacher_artwork' and teacher['metric'] == metric
    assert plain['point_c'] == teacher['point_c']
    return teacher_share(plain['point_c'], plain['point_t'], teacher['point_t'])


frozen_sc = json.loads(FROZEN.read_text())['sc']
model_sc = json.loads(PHOTO_MODEL.read_text())['sc']
frozen_ami = photo_share(frozen_sc, 'SC1', 'SC3', 'ami')
frozen_map = photo_share(frozen_sc, 'SC2', 'SC4', 'map')
model_ami = photo_share(model_sc, 'SC1', 'SC3', 'ami')
model_map = photo_share(model_sc, 'SC2', 'SC4', 'map')
painting_map = json.loads(PAINTINGS.read_text())['map']
art_map = teacher_share(*(painting_map[key]['emotion']['mean']
                          for key in ('cstar', 'plain', 'instructed')))
assert [round(v, 1) for v in (frozen_ami, frozen_map, art_map)] == [54.9, 61.5, 103.4]
assert [round(v, 1) for v in (model_ami, model_map)] == [127.5, 145.9]
for key in ('SC1', 'SC2', 'SC3', 'SC4'):
    assert model_sc[key]['point_t'] == frozen_sc[key]['point_t'], key

rows = [('Paintings (ArtELingo), retrieval mAP', art_map, None),
        ('Photos (Affection), retrieval mAP', frozen_map, model_map),
        ('Photos (Affection), clustering AMI', frozen_ami, model_ami)]
fig, ax = plt.subplots(figsize=(11, 5.1), dpi=200)
fig.subplots_adjust(left=0.36, right=0.98, bottom=0.43, top=0.83)
for i, (label, frozen, model) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.plot([0, max(100, frozen, model or 0)], [y, y], color=GRID, lw=1, zorder=1)
    ax.scatter([0, 100], [y, y], s=32, c=[PLAIN, TEACHER], zorder=3)
    ax.scatter([frozen], [y], s=65, edgecolors=OURS, facecolors='white',
               linewidths=1.6, zorder=4)
    points = [(frozen, 8, 0, 'left')]
    if model is not None:
        ax.scatter([model], [y], s=65, color=PHOTO, zorder=4)
        points.append((model, 8, 0, 'left'))
    for value, dx, dy, align in points:
        ax.annotate(f'{value:.1f}%', (value, y), xytext=(dx, dy),
                    textcoords='offset points', va='center', ha=align,
                    fontsize=10, color=INK)
for x in (0, 100):
    ax.axvline(x, color=GRID, lw=0.9, zorder=0)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([label for label, _, _ in rows][::-1])
ax.set_ylim(-0.5, 2.7)
ax.set_xlim(-7, max(model_ami, model_map, art_map) + 27)
ax.set_xticks([0, 25, 50, 75, 100, 125, 150])
ax.set_xlabel("Share of the teacher's gain over the plain image (%)", fontsize=11, color=INK)
ax.xaxis.set_label_coords(0.5, 0.325, transform=fig.transFigure)
fig.suptitle('On held photos, the photo model passes its teacher in both tasks',
             x=0.015, y=0.97, ha='left', fontsize=14, color=INK)
handles = [
    Line2D([], [], linestyle='', marker='o', color=PLAIN, markersize=6,
           label='plain Qwen image'),
    Line2D([], [], linestyle='', marker='o', markeredgecolor=OURS,
           markerfacecolor='white', markeredgewidth=1.6, markersize=8,
           label='frozen c* (5 seeds, averaged)'),
    Line2D([], [], linestyle='', marker='o', color=PHOTO, markersize=8,
           label='photo model (c* fine-tuned on photos, 5 seeds, averaged)'),
    Line2D([], [], linestyle='', marker='o', color=TEACHER, markersize=6,
           label='teacher (Qwen image instructed for emotion)'),
]
ax.legend(handles=handles, fontsize=10, frameon=False, loc='upper center',
          bbox_to_anchor=(0.5, 0.29), bbox_transform=fig.transFigure)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
ax.spines['left'].set_color(INK2)
ax.spines['bottom'].set_color(INK2)
ax.tick_params(colors=INK2, labelsize=10)
fig.text(0.015, 0.025,
         'Point values. 8,501 Affection test photos; 12,281 held ArtELingo paintings. Aspect: emotion.',
         fontsize=10, color=INK2)
fig.savefig(HERE / 'teacher_share.png')
plt.close(fig)
for label, frozen, model in rows:
    print(f'{label}: frozen c* {frozen:.1f}%' +
          (f'; photo model {model:.1f}%' if model is not None else ''))
print('Comparators: plain 0.0%; teacher 100.0%; r13 and r12 point values identical.')
print('wrote', HERE / 'teacher_share.png')
