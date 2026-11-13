"""Shared c* UMAPs and emotion-head valence analysis, seed 0."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '20261111_r21_dec_heads'))
import r21_figures as F

OUT = Path('/data/SSD2/cosir_r21/20261111_dec_heads/r30_followup')
SPLITS = [('held', F.TEST), ('val', F.VAL)]
POS = {'amusement', 'awe', 'contentment', 'excitement'}
NEG = {'anger', 'disgust', 'fear', 'sadness'}
VALENCE = ('positive', 'negative', 'other')
STYLE = ['#d60000', '#018700', '#b500ff', '#05acc6', '#97ff00', '#ffa52f', '#ff8ec8', '#79525e', '#00fdcf', '#afa5ff', '#93ac83', '#9a6900', '#366962', '#d3008c', '#fdf490', '#c86e66', '#9ee2ff', '#00c846', '#a877ac', '#b8ba00', '#f497ff', '#d2b993', '#f10062', '#6c00c4', '#004c99', '#7b4f00', '#172033']


def load(split, arm, aspect, limit):
    if split == 'held':
        actual = F.T.CSTAR_ARM if arm == 'cstar' else arm
        root = F.T.R10_FULL if arm == 'cstar' else F.T.R21_FULL
        x, _ = F.T.load_held(root, actual, aspect, 0)
    else:
        x, _ = F.SC.load_val(arm, aspect, 0, None if arm == 'cstar' else F.TRAIN)
        x = x.astype(np.float32)
    return x[:limit]


def shares_and_labels(dist, names):
    names = list(names)
    assert len(names) == 9 and len(set(names)) == 9 and POS | NEG <= set(names)
    other = set(names) - POS - NEG
    assert len(other) == 1
    shares = np.stack([dist[:, [i for i, n in enumerate(names) if n in group]].sum(axis=1)
                       for group in (POS, NEG, other)], axis=1)
    maxima = shares.max(axis=1, keepdims=True)
    labels = shares.argmax(axis=1).astype(int)
    # Shares are sums of count/total fractions: equal counts can differ in the last bit, so ties use a tolerance.
    labels[np.isclose(shares, maxima, rtol=0, atol=1e-12).sum(axis=1) > 1] = -1
    return shares, labels


def pos_gap(shares, groups):
    return float(abs(shares[groups == 1, 0].mean() - shares[groups == 0, 0].mean()))


def cache_path(out, split, arm, aspect):
    return out / 'cache' / f'{split}__{arm}__{aspect}__s0.npz'


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def umap_run(out, limit):
    import umap
    (out / 'cache').mkdir(parents=True, exist_ok=True)
    for aspect in F.SC.ASPECTS:
        for split, cells in SPLITS:
            base = load(split, 'cstar', aspect, limit)
            base_sha = digest(base)
            model = None
            for arm, _ in cells:
                start = time.monotonic()
                x = base if arm == 'cstar' else load(split, arm, aspect, limit)
                identity = dict(cstar_sha256=base_sha, arm_sha256=digest(x), umap=F.SETTINGS)
                dest = cache_path(out, split, arm, aspect)
                if dest.exists():
                    with np.load(dest, allow_pickle=False) as z:
                        if json.loads(str(z['identity'])) == identity:
                            print(f'{split} {arm} {aspect}: cached', flush=True)
                            continue
                if model is None:
                    model = umap.UMAP(**F.SETTINGS).fit(base)
                coords = model.embedding_ if arm == 'cstar' else model.transform(x)
                np.savez_compressed(dest, coords=coords, identity=json.dumps(identity, sort_keys=True))
                print(f'{split} {arm} {aspect}: {time.monotonic()-start:.1f}s', flush=True)


def valence_run(out, limit):
    import r20_data
    rt = F.SC.rt_data
    class_names = r20_data._artelingo_class_names()
    emotion_names = sorted(rt.rs_vh.EMOTIONS)
    assert emotion_names == list(class_names['emotion'])
    result = {'seed': 0, 'limit': limit, 'emotion_names': emotion_names, 'valence_names': VALENCE,
              'definitions': {'positive': sorted(POS), 'negative': sorted(NEG), 'other': sorted(set(emotion_names)-POS-NEG),
                              'tie': 'top two caption shares within 1e-12; excluded from valence AMI'}, 'splits': {}}
    for split, cells in SPLITS:
        names = rt.names(split)
        assert names == list(F.SC.ns_data.keys(F.SC.SET, split))
        if split == 'val':
            assert names == F.SC.val_names()
        else:
            assert rt.rs_vh.names_sha(names) == F.T.held_sha()
        shares, val = shares_and_labels(rt.emotion_dist(split)[:limit], emotion_names)
        labels = {a: np.asarray(F.SC.ns_data.labels(F.SC.SET, split, a))[:limit] for a in F.SC.ASPECTS}
        targets = {'valence': val, **labels}
        split_result = {'rows': len(val), 'valence_ties': int((val < 0).sum()), 'arms': {}}
        result['splits'][split] = split_result
        for arm, title in cells:
            x = load(split, arm, 'emotion', limit)
            assert len(x) == len(val)
            x = F.SC.ns_data._unit32(x)
            parts = F.SC.ns_metrics.partitions(x, 2, F.SC.SET)
            metrics = {}
            for name, y in targets.items():
                mask = y >= 0
                scores = [F.SC.ns_metrics.ami(y[mask], p[mask]) for p in parts]
                metrics[name] = {'n': int(mask.sum()), 'states': scores, 'mean': float(np.mean(scores))}
            groups = []
            for g in (0, 1):
                mask = parts[0] == g
                assert mask.any()
                groups.append({'group': g, 'size': int(mask.sum()),
                    'majority_emotion_shares': {name: float((labels['emotion'][mask] == i).mean()) for i, name in enumerate(emotion_names)},
                    'mean_caption_shares': {name: float(shares[mask, i].mean()) for i, name in enumerate(VALENCE)},
                    'valence_label_shares': {**{name: float((val[mask] == i).mean()) for i, name in enumerate(VALENCE)}, 'tie': float((val[mask] < 0).mean())}})
            split_result['arms'][arm] = {'title': title, 'ami': metrics, 'pos_share_gap': pos_gap(shares, parts[0]), 'groups': groups}
            print(f'{split} {arm}: valence AMI {metrics["valence"]["mean"]:.4f}', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'valence.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# Valence analysis', '', 'AMI values are means over k-means states 0–2; group summaries use state 0. Ties are excluded from valence AMI and included in group share denominators.', '']
    for split, _ in SPLITS:
        d = result['splits'][split]
        lines += [f'## {split} (n={d["rows"]}, valence ties={d["valence_ties"]})', '', '| Arm | Valence AMI (n) | Emotion AMI (n) | Style AMI (n) | Genre AMI (n) | Positive share gap |', '|---|---:|---:|---:|---:|---:|']
        for arm, row in d['arms'].items():
            vals = [f'{row["ami"][a]["mean"]:.4f} ({row["ami"][a]["n"]})' for a in ('valence', 'emotion', 'style', 'genre')]
            lines.append('| ' + ' | '.join([arm, *vals, f'{row["pos_share_gap"]:.4f}']) + ' |')
        for arm, row in d['arms'].items():
            lines += ['', f'### {arm}', '', '| Group | Size | ' + ' | '.join(emotion_names) + ' | Mean pos | Mean neg | Mean other | Positive | Negative | Other | Tie |', '|---:|---:|' + '---:|' * (len(emotion_names)+7)]
            for g in row['groups']:
                values = list(g['majority_emotion_shares'].values()) + list(g['mean_caption_shares'].values()) + list(g['valence_label_shares'].values())
                lines.append(f'| {g["group"]} | {g["size"]} | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |')
        lines.append('')
    (out / 'valence.md').write_text('\n'.join(lines) + '\n')


def plot(out, limit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import r20_data
    names = r20_data._artelingo_class_names()
    for aspect in F.SC.ASPECTS:
        palette = STYLE if aspect == 'style' else list(plt.get_cmap('tab10').colors)
        if aspect != 'style': palette[7] = '#172033'
        labels = {s: np.asarray(F.SC.ns_data.labels(F.SC.SET, s, aspect))[:limit] for s, _ in SPLITS}
        fig, axes = plt.subplots(2, 5, figsize=(22, 9), squeeze=False)
        fig.suptitle(f'{aspect.capitalize()} - one UMAP map fitted on c* - seed 0', fontsize=20, y=.98)
        fig.text(.5, .93, 'c* fitted; other arms projected with UMAP transform', ha='center', fontsize=12)
        fig.text(.012, .70, 'Test', rotation=90, va='center', fontsize=15, weight='bold')
        fig.text(.012, .34, 'Validation', rotation=90, va='center', fontsize=15, weight='bold')
        for row, (split, cells) in enumerate(SPLITS):
            coords = []
            for arm, _ in cells:
                with np.load(cache_path(out, split, arm, aspect), allow_pickle=False) as z:
                    xy = z['coords'].copy()
                assert len(xy) == len(labels[split]), (split, arm, aspect)
                coords.append(xy)
            all_xy = np.concatenate(coords)
            lo, hi = all_xy.min(axis=0), all_xy.max(axis=0)
            margin = np.maximum((hi-lo)*.04, .01)
            for col, ((arm, title), xy) in enumerate(zip(cells, coords)):
                ax = axes[row, col]; ax.set_title(title, fontsize=12, pad=8)
                y = labels[split]; missing = y < 0
                ax.scatter(xy[missing, 0], xy[missing, 1], s=1.5, c='#d5d5d5', alpha=.45, linewidths=0, rasterized=True)
                for c in range(len(names[aspect])):
                    mask = y == c
                    ax.scatter(xy[mask, 0], xy[mask, 1], s=1.5, color=palette[c], alpha=.65, linewidths=0, rasterized=True)
                ax.set_xlim(lo[0]-margin[0], hi[0]+margin[0]); ax.set_ylim(lo[1]-margin[1], hi[1]+margin[1])
                ax.set_xticks([]); ax.set_yticks([])
            for col in range(len(cells), 5): axes[row, col].axis('off')
        handles = [Line2D([], [], marker='o', color='w', markerfacecolor=palette[i], markersize=7, label=n.replace('_', ' ')) for i, n in enumerate(names[aspect])]
        if any((y < 0).any() for y in labels.values()):
            handles.append(Line2D([], [], marker='o', color='w', markerfacecolor='#c9c9c9', markersize=7, label='Unlabelled'))
        fig.legend(handles=handles, loc='lower center', ncol=7 if aspect == 'style' else 6, frameon=False, fontsize=10, columnspacing=1.2, handletextpad=.4, bbox_to_anchor=(.5, .015))
        fig.subplots_adjust(left=.045, right=.99, top=.88, bottom=.20 if aspect == 'style' else .14, wspace=.19, hspace=.34)
        for ext in ('png', 'pdf'): fig.savefig(out / f'shared_umap_{aspect}.{ext}', dpi=200)
        plt.close(fig)
        print(f'rendered {aspect}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('umap', 'valence', 'plot'):
        p = sub.add_parser(command)
        p.add_argument('--out', type=Path, default=OUT)
        p.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 16: parser.error('--limit must be at least 16')
    F.SC._deps(); F.T._deps()
    args.out.mkdir(parents=True, exist_ok=True)
    {'umap': umap_run, 'valence': valence_run, 'plot': plot}[args.command](args.out, args.limit)


if __name__ == '__main__': main()
