"""r30 figure: r21's own emotion UMAPs on validation (c*, long DEC plain), coloured by genre and by positive caption share.

Writes OUT/islands_emotion.png. Coordinates come from r21's figure cache (one UMAP fit per panel, as the user saw them).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r30_followup as R
F = R.F


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    F.SC._deps(); F.T._deps()
    import r20_data
    names = r20_data._artelingo_class_names()['genre']
    rt = F.SC.rt_data
    shares, _ = R.shares_and_labels(rt.emotion_dist('val'), sorted(rt.rs_vh.EMOTIONS))
    genre = np.asarray(F.SC.ns_data.labels(F.SC.SET, 'val', 'genre'))
    palette = list(plt.get_cmap('tab10').colors); palette[7] = '#172033'   # r21's genre colours
    arms = [('cstar', 'c* (no DEC)'), ('dec-plain_l10_w5_S1', 'DEC plain, 50 DEC epochs')]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle('Long DEC splits emotion outputs by subject (people against scenes), not by valence\n'
                 'ArtELingo validation, seed 0, r21 emotion UMAPs (one fit per panel)', fontsize=14)
    for row, (arm, title) in enumerate(arms):
        with np.load(F.OUT / 'cache' / f'val__{arm}__emotion__s0.npz', allow_pickle=False) as z:
            xy = z['coords']
        ax = axes[row, 0]
        m = genre < 0
        ax.scatter(xy[m, 0], xy[m, 1], s=1.5, c='#d5d5d5', alpha=.45, linewidths=0, rasterized=True)
        for c in range(len(names)):
            m = genre == c
            ax.scatter(xy[m, 0], xy[m, 1], s=1.5, color=palette[c], alpha=.7, linewidths=0, rasterized=True)
        ax.set_title(f'{title}: coloured by genre', fontsize=12)
        ax = axes[row, 1]
        sc = ax.scatter(xy[:, 0], xy[:, 1], s=1.5, c=shares[:, 0], cmap='RdBu', vmin=0, vmax=1, alpha=.8,
                        linewidths=0, rasterized=True)
        ax.set_title(f'{title}: coloured by positive caption share', fontsize=12)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(sc, ax=axes[:, 1], fraction=.04, pad=.02)
    cbar.set_label('share of captions with a positive emotion\n(amusement, awe, contentment, excitement)')
    handles = [Line2D([], [], marker='o', color='w', markerfacecolor=palette[i], markersize=7,
                      label=n.replace('_', ' ')) for i, n in enumerate(names)]
    handles.append(Line2D([], [], marker='o', color='w', markerfacecolor='#c9c9c9', markersize=7, label='unlabelled'))
    fig.legend(handles=handles, loc='lower left', ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(.04, .0))
    fig.subplots_adjust(left=.03, right=.86, top=.89, bottom=.11, wspace=.06, hspace=.12)
    fig.savefig(R.OUT / 'islands_emotion.png', dpi=150)
    print('wrote', R.OUT / 'islands_emotion.png')


if __name__ == '__main__':
    main()
