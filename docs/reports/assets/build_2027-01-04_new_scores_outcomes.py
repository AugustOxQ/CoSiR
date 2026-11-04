"""r17 user-read figure: outcome counts per comparison, from records/verdicts.json (read-only input)."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V = json.load(open('/project/CoSiR/src/test/20261103_new_scores/records/verdicts.json'))['cells']
ROWS = [
    ('Plain image, clustering', lambda c: c['sc'] == 'SC1'),
    ('PercepT, clustering', lambda c: c['sc'] == 'SC2'),
    ('Teacher, AMI and NMI', lambda c: c['sc'] == 'SC3' and c['metric'] in ('ami', 'nmi')),
    ('Teacher, ARI and ACC', lambda c: c['sc'] == 'SC3' and c['metric'] in ('ari', 'acc')),
    ('Photo model, clustering', lambda c: c['sc'] == 'SC4'),
    ('CRL, clustering', lambda c: c['sc'] == 'SC5'),
    ('Retrieval (CRL, PercepT, untold)', lambda c: c['sc'] in ('SC6', 'SC7', 'SC8')),
]
COL = {'WIN': '#1baf7a', 'LEVEL': '#b3b1ab', 'BELOW': '#eb6834'}
fig, ax = plt.subplots(figsize=(8.4, 3.5))
for i, (name, keep) in enumerate(ROWS):
    cells = [c for c in V if keep(c)]
    left = 0.
    for o in COL:
        n = sum(c['outcome'] == o for c in cells)
        w = 100 * n / len(cells)
        ax.barh(i, w, left=left, color=COL[o], label=o if i == 0 else None, edgecolor='white')
        if n:
            ax.text(left + w / 2, i, str(n), ha='center', va='center', fontsize=9)
        left += w
    ax.text(101, i, f'{len(cells)} cells', va='center', fontsize=9)
ax.set_yticks(range(len(ROWS)), [r[0] for r in ROWS])
ax.invert_yaxis()
ax.set_xlim(0, 113)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel('Share of cells (%); numbers in the bars are cell counts')
ax.set_title('c* wins most comparisons; ARI and ACC against the teacher are mixed', fontsize=11)
ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.45, -0.2), frameon=False)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig('/project/CoSiR/docs/user_read/assets/2026-11-04_new_scores/outcomes.png', dpi=160)
