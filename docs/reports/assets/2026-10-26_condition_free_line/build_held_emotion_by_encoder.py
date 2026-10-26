"""Rebuild the held/development encoder comparison using only task-supplied data."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

INK, SECONDARY, GRID = '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, GREY = '#1d6fd1', '#c98b2e', '#a6a49e'
YELLOW, FAIL = '#d9a21b', '#eb6834'
ENCODERS = ['SigLIP 1', 'Qwen', 'SigLIP 2', 'CLIP L/14', 'CLIP B/16', 'CLIP B/32', 'DINOv2-large']
HELD = [0.1036, 0.1029, 0.1024, 0.0996, 0.0945, 0.0885, 0.0814]
DEVELOPMENT = np.array([
    [0.0631, 0.0784, 0.1073],
    [0.0690, 0.0844, 0.1050],
    [0.0571, 0.0756, 0.1042],
    [0.0648, 0.0772, 0.1031],
    [0.0643, 0.0728, 0.0952],
    [0.0624, 0.0708, 0.0905],
    [0.0587, 0.0722, 0.0839],
])
# Reported lifts are supplied independently; preserve their stated precision.
LIFTS = ['+0.0288', '+0.0206', '+0.0286', '+0.0259', '+0.0225', '+0.0196', '+0.0117']


def main():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 14,
        'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': SECONDARY, 'ytick.color': SECONDARY,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    })
    fig = plt.figure(figsize=(14, 7), dpi=200)
    ax = fig.add_axes([0.125, 0.255, 0.335, 0.405])
    bx = fig.add_axes([0.630, 0.255, 0.340, 0.405])
    fig.text(0.025, 0.965, 'A   Held emotion: four encoders pass the bar,\n     two are inconclusive, one fails',
             fontsize=17, fontweight='bold', va='top', linespacing=1.25)
    fig.text(0.535, 0.965, 'B   On development paintings every student\n     lifts over PercepT on its own encoder',
             fontsize=17, fontweight='bold', va='top', linespacing=1.25)
    fig.text(0.025, 0.843,
             'Emotion AMI on 12,281 held ArtELingo paintings, 5 seeds;\nfull scale (DINOv2-large: sample scale)',
             fontsize=14, color=SECONDARY, va='top', linespacing=1.3)
    fig.text(0.535, 0.843,
             'Emotion AMI on 29,232 development paintings,\nsample scale, 5 seeds',
             fontsize=14, color=SECONDARY, va='top', linespacing=1.3)
    y = np.arange(7)
    for axis, lower, upper in [(ax, .06, .11), (bx, .04, .12)]:
        axis.set_xlim(lower, upper)
        axis.set_ylim(6.8, -.65)
        axis.set_yticks(y, ENCODERS)
        axis.tick_params(axis='y', length=0, pad=10, labelsize=14)
        axis.tick_params(axis='x', length=0, pad=9, labelsize=14)
        axis.set_xlabel('Emotion AMI', fontsize=14, labelpad=10)
        axis.set_axisbelow(True)
        axis.grid(axis='x', color=GRID, linewidth=.8)
        for side in ['top', 'left', 'right']:
            axis.spines[side].set_visible(False)
        axis.spines['bottom'].set_color(GRID)
    ax.set_yticklabels(ENCODERS[:-1] + ['DINOv2-large\n(sample scale)'])
    ax.set_xticks([.06, .07, .08, .09, .10, .11])
    bx.set_xticks([.04, .06, .08, .10, .12])
    ax.barh(y, np.array(HELD)-.06, left=.06, height=.52,
            color=[BLUE]*4+[YELLOW]*2+[FAIL], zorder=3)
    ax.axvline(.0874, color=ORANGE, linewidth=2, zorder=4)
    ax.axvline(.0974, color=SECONDARY, linewidth=1.6, linestyle=(0, (5, 4)), zorder=4)
    # Small keyed line segments above the plot keep long labels off the bars.
    for ypos, label, color, style in [
        (1.16, 'PercepT core, Qwen image (the bar)', ORANGE, '-'),
        (1.065, 'pass: bar + 0.0100', SECONDARY, (0, (5, 4))),
    ]:
        ax.plot([.0, .055], [ypos, ypos], transform=ax.transAxes,
                color=color, linestyle=style, linewidth=2, clip_on=False)
        ax.text(.075, ypos, label, transform=ax.transAxes, va='center', fontsize=14)
    for row, value in enumerate(HELD):
        if row == 6:
            ax.text(value-.0006, row, f'{value:.4f}', fontsize=14, va='center',
                    ha='right', color='white', zorder=6)
        else:
            ax.text(value+.00065, row, f'{value:.4f}', fontsize=14, va='center', zorder=6,
                    bbox=dict(facecolor='white', edgecolor='none', pad=.8))
    ax.vlines(.0855, 5.77, 6.23, color=ORANGE, linewidth=3, zorder=7)
    ax.text(.0858, 6.48, 'bar at sample scale', color=SECONDARY, fontsize=14, va='center',
            bbox=dict(facecolor='white', edgecolor='none', pad=.5), zorder=8)
    for col, (color, offset) in enumerate([(GREY, -.23), (ORANGE, 0), (BLUE, .23)]):
        bx.barh(y+offset, DEVELOPMENT[:, col]-.04, left=.04, height=.205,
                color=color, zorder=3)
    for row, lift in enumerate(LIFTS):
        bx.text(DEVELOPMENT[row, 2]+.001, row+.23, lift, fontsize=14, va='center', color=BLUE)
    for axis, entries in [
        (ax, [(BLUE, 'pass (interval above zero)'),
              (YELLOW, 'inconclusive (0 < gain < 0.0100)'), (FAIL, 'fail')]),
        (bx, [(GREY, 'frozen image, clustered directly'),
              (ORANGE, 'PercepT core on this encoder'), (BLUE, 'our student on this encoder')]),
    ]:
        axis.legend(handles=[Patch(facecolor=c, label=t) for c,t in entries],
                    loc='upper left', bbox_to_anchor=(0, -.245), frameon=False,
                    fontsize=14, borderaxespad=0, handlelength=1.2, labelspacing=.6)
    fig.savefig(Path(__file__).with_name('held_emotion_by_encoder.png'), dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
