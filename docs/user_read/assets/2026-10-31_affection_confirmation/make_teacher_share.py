"""Figure for the affection confirmation report: share of the teacher's gain kept by c*.

Reads the Affection stage 2 and ArtELingo stage 5 verdict files (point values).
Usage: python make_teacher_share.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PHOTOS = Path('/data/SSD2/cosir_r12/20261030_affection_confirmation/outputs/stage2/verdict.json')
PAINTINGS = Path('/data/SSD2/cosir_r11/20261029_retrieval/outputs/stage5/verdict.json')
OURS, PLAIN, TEACHER = '#1d6fd1', '#a6a49e', '#4a9c5b'
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


sc = json.loads(PHOTOS.read_text())['sc']
photo_ami = photo_share(sc, 'SC1', 'SC3', 'ami')
photo_map = photo_share(sc, 'SC2', 'SC4', 'map')
painting_map = json.loads(PAINTINGS.read_text())['map']
art_map = teacher_share(*(painting_map[key]['emotion']['mean']
                          for key in ('cstar', 'plain', 'instructed')))
assert [round(v, 1) for v in (photo_ami, photo_map, art_map)] == [54.9, 61.5, 103.4]

rows = [('Paintings (ArtELingo), retrieval mAP', art_map),
        ('Photos (Affection), retrieval mAP', photo_map),
        ('Photos (Affection), clustering AMI', photo_ami)]
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
fig.subplots_adjust(left=0.46, right=0.98, bottom=0.40, top=0.83)
for i, (label, share) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.plot([0, 100], [y, y], color=GRID, lw=1, zorder=1)
    ax.scatter([0], [y], s=24, color=PLAIN, zorder=3,
               label='plain Qwen image' if i == 0 else None)
    ax.scatter([share], [y], s=50, color=OURS, zorder=4,
               label='c* (5 seeds, averaged)' if i == 0 else None)
    ax.scatter([100], [y], s=24, color=TEACHER, zorder=3,
               label='teacher (Qwen image instructed for emotion)' if i == 0 else None)
    ax.annotate(f'{share:.1f}%', (share, y), xytext=(7, 0),
                textcoords='offset points', va='center', fontsize=8.5, color=INK)
for x in (0, 100):
    ax.axvline(x, color=GRID, lw=0.9, zorder=0)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([label for label, _ in rows][::-1])
ax.set_ylim(-0.55, 2.55)
ax.set_xlim(-5, 128)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("Share of the teacher's gain over the plain image that c* keeps (%)",
              fontsize=9, color=INK)
ax.xaxis.set_label_coords(0.5, 0.275, transform=fig.transFigure)
fig.suptitle("On photos, c* keeps about half of its teacher's gain; on paintings, all of it",
             x=0.01, y=0.97, ha='left', fontsize=10, color=INK)
ax.legend(fontsize=8, frameon=False, loc='upper center',
          bbox_to_anchor=(0.5, 0.235), bbox_transform=fig.transFigure)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.spines['left'].set_color(INK2)
ax.spines['bottom'].set_color(INK2)
ax.tick_params(colors=INK2, labelsize=9)
fig.text(0.01, 0.01,
         'Point values. Photos: 8,501 Affection test photos. Paintings: 12,281 held ArtELingo paintings, emotion.',
         fontsize=7.5, color=INK2)
fig.savefig(HERE / 'teacher_share.png')
for label, share in rows:
    print(f'{label}: {share:.1f}%')
print('wrote', HERE / 'teacher_share.png')
