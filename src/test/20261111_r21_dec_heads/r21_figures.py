"""Render r21 figures: python r21_figures.py compute [--aspects emotion style genre]

Then run: python r21_figures.py plot
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import r21_score as SC
import r21_table as T

OUT = Path('/data/SSD2/cosir_r21/20261111_dec_heads/figures')
TRAIN = T.R21_FULL / 'train'
TEST = [('cstar', 'c* (no DEC)'), ('dec-plain_l0_w5_S2', 'λ = 0 control'),
        ('dec-plain_l10_w5_S2', 'DEC plain'), ('dec-start_l10_w5_S2', 'DEC start'),
        ('dec-teacher_l10_w5_S2', 'DEC teacher')]
VAL = [('cstar', 'c* (no DEC), validation'), ('dec-plain_l10_w5_S1', 'DEC plain, 50 epochs, validation')]
SETTINGS = dict(n_components=2, metric='cosine', n_neighbors=15, min_dist=0.1, random_state=0)


def path(split, arm, aspect):
    return OUT / 'cache' / f'{split}__{arm}__{aspect}__s0.npz'


def silhouette_rows(x, labels):
    # Exact cosine distances from normalized rows to float64 cluster sums.
    x = np.asarray(x, dtype=np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    _, inv = np.unique(labels, return_inverse=True)
    size = np.bincount(inv).astype(np.float64)
    sums = np.stack([x[inv == c].sum(axis=0) for c in range(len(size))])
    result = np.empty(len(x))
    for lo in range(0, len(x), 1024):
        hi = min(lo + 1024, len(x))
        ids = inv[lo:hi]
        total = size - x[lo:hi] @ sums.T
        own = total[np.arange(hi-lo), ids] / np.maximum(size[ids]-1, 1)
        other = total / size
        other[np.arange(hi-lo), ids] = np.inf
        nearest = other.min(axis=1)
        den = np.maximum(own, nearest)
        result[lo:hi] = np.where((size[ids] > 1) & (den > 0), (nearest-own)/np.where(den > 0, den, 1), 0)
    return result


def reference(split, arm, aspect):
    if split == 'held':
        doc = json.loads((T.SCORES / 'diag.json').read_text())
        try:
            return doc['sources'][arm][aspect]['seeds']['0']['silhouette']
        except KeyError:
            return None
    doc = json.loads((T.SCORES / 'val_internal.json').read_text())
    return doc['arms'][arm][aspect]['si']


def compute(aspects):
    import umap
    SC._deps(); T._deps()
    (OUT / 'cache').mkdir(parents=True, exist_ok=True)
    for aspect in aspects:
        k = SC.ns_paths.CLASS_COUNT[(SC.SET, aspect)]
        for split, cells in [('held', TEST), ('val', VAL)]:
            for arm, _ in cells:
                start = time.monotonic()
                if split == 'held':
                    actual = T.CSTAR_ARM if arm == 'cstar' else arm
                    root = T.R10_FULL if arm == 'cstar' else T.R21_FULL
                    if not T.held_path(root, actual, aspect, 0).exists():
                        print(f'{split} {arm} {aspect}: missing, skipped', flush=True)
                        continue
                    x, meta = T.load_held(root, actual, aspect, 0)
                else:
                    x, _ = SC.load_val(arm, aspect, 0, None if arm == 'cstar' else TRAIN)
                    x = x.astype(np.float32)
                sha = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
                identity = dict(sha256=sha, umap=SETTINGS, k=int(k), partition='ns_metrics.partitions state 0', silhouette='cosine exact float64')
                dest = path(split, arm, aspect)
                if dest.exists():
                    with np.load(dest, allow_pickle=False) as z:
                        if json.loads(str(z['identity'])) == identity:
                            mean = float(z['mean']); ref = reference(split, arm, aspect)
                            print(f'{split} {arm} {aspect}: cached mean={mean:.9f} reference={ref} |diff|={abs(mean-ref) if ref is not None else "unavailable"} ({time.monotonic()-start:.1f}s)', flush=True)
                            if ref is not None:
                                assert abs(mean-ref) < 1e-6
                            continue
                if split == 'held':
                    labels = T.get_partitions(T.SCORES / 'cache', actual, aspect, 0, k, meta, x)[0]
                else:
                    labels = SC.ns_metrics.partitions(x, k, SC.SET)[0]
                s = silhouette_rows(x, labels)
                mean = float(s.mean()); exact = T.silhouette_cosine(x, labels)
                ref = reference(split, arm, aspect)
                print(f'{split} {arm} {aspect}: mean={mean:.9f} T={exact:.9f} reference={ref} |diff|={abs(mean-ref) if ref is not None else "unavailable"}', flush=True)
                assert abs(mean-exact) < 1e-6
                if ref is not None:
                    assert abs(mean-ref) < 1e-6
                coords = umap.UMAP(**SETTINGS).fit_transform(x)
                elapsed = time.monotonic()-start
                np.savez_compressed(dest, coords=coords, silhouette=s, identity=json.dumps(identity),
                                    mean=mean, reference=np.nan if ref is None else ref, seconds=elapsed)
                print(f'{split} {arm} {aspect}: {elapsed:.1f}s', flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    SC._deps(); T._deps()
    import r20_data
    OUT.mkdir(parents=True, exist_ok=True)
    names = r20_data._artelingo_class_names()
    # Fixed categorical colours; grey is reserved exclusively for unlabelled rows.
    style_palette = [
        '#d60000', '#018700', '#b500ff', '#05acc6', '#97ff00', '#ffa52f',
        '#ff8ec8', '#79525e', '#00fdcf', '#afa5ff', '#93ac83', '#9a6900',
        '#366962', '#d3008c', '#fdf490', '#c86e66', '#9ee2ff', '#00c846',
        '#a877ac', '#b8ba00', '#f497ff', '#d2b993', '#f10062', '#6c00c4',
        '#004c99', '#7b4f00', '#172033',
    ]
    small_palette = list(plt.get_cmap('tab10').colors)
    small_palette[7] = '#172033'
    records = []
    unlabelled = []
    for aspect in SC.ASPECTS:
        palette = style_palette if aspect == 'style' else small_palette
        split_labels = {split: np.asarray(SC.ns_data.labels(SC.SET, split, aspect))
                        for split in ('held', 'val')}
        for split, y in split_labels.items():
            count = int((y < 0).sum())
            unlabelled.append((aspect, split, count))
            print(f'{aspect} {split}: unlabelled {count} / {len(y)}', flush=True)
        data = {}
        for split, cells in [('held', TEST), ('val', VAL)]:
            for arm, _ in cells:
                p = path(split, arm, aspect)
                if p.exists():
                    with np.load(p, allow_pickle=False) as z:
                        data[split, arm] = {key: z[key].copy() for key in ('coords', 'silhouette', 'mean', 'reference', 'seconds')}
                    d = data[split, arm]
                    records.append((aspect, split, arm, float(d['mean']), float(d['reference']), float(d['seconds'])))
        for kind in ('umap', 'silhouette'):
            fig, axes = plt.subplots(2, 5, figsize=(22, 9), squeeze=False)
            fig.suptitle(f'{aspect.capitalize()} — {"UMAP" if kind == "umap" else "cosine silhouette"} — seed 0', fontsize=20, y=.98)
            fig.text(.012, .70, 'Test', rotation=90, va='center', fontsize=15, weight='bold')
            fig.text(.012, .34, 'Validation', rotation=90, va='center', fontsize=15, weight='bold')
            all_s = np.concatenate([d['silhouette'] for d in data.values()]) if data else np.array([-0.2, 0.5])
            lo, hi = float(all_s.min()), float(all_s.max())
            margin = max(.03, .06 * (hi - lo))
            bins = np.linspace(lo - margin, hi + margin, 45)
            for cell, d in data.items():
                values = d['silhouette']
                assert np.isfinite(values).all() and ((values > bins[0]) & (values < bins[-1])).all(), cell
                # Leave an empty bin at either edge, including for the c* outlines.
                assert values.min() > bins[1] and values.max() < bins[-2], cell
            for row, (split, cells) in enumerate([('held', TEST), ('val', VAL)]):
                y = split_labels[split]
                for col, (arm, title) in enumerate(cells):
                    ax = axes[row, col]; ax.set_title(title, fontsize=12, pad=8)
                    d = data.get((split, arm))
                    if d is None:
                        ax.text(.5, .5, 'not run yet', ha='center', va='center', transform=ax.transAxes, color='#777777', fontsize=13)
                        ax.set_xticks([]); ax.set_yticks([])
                        continue
                    if kind == 'umap':
                        xy = d['coords']; missing = y < 0
                        ax.scatter(xy[missing, 0], xy[missing, 1], s=1.5, c='#d5d5d5', alpha=.45, linewidths=0, rasterized=True)
                        for c in range(len(names[aspect])):
                            mask = y == c
                            ax.scatter(xy[mask, 0], xy[mask, 1], s=1.5, color=palette[c], alpha=.65, linewidths=0, rasterized=True)
                        ax.set_xticks([]); ax.set_yticks([])
                    else:
                        s = d['silhouette']; base = data.get((split, 'cstar'))
                        ax.hist(s, bins=bins, density=True, color='#4c78a8', alpha=.65)
                        if arm != 'cstar' and base is not None:
                            ax.hist(base['silhouette'], bins=bins, density=True, histtype='step', color='#222222', linewidth=1.4)
                        mean = float(d['mean']); ax.axvline(mean, color='#b22222', linewidth=1.5)
                        ax.text(.97, .96, f'Mean {mean:.3f}\nbelow 0: {(s < 0).mean():.1%}', ha='right', va='top', transform=ax.transAxes, fontsize=10,
                                bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
                        ax.set_xlim(bins[0], bins[-1]); ax.tick_params(labelsize=9)
                        if col == 0: ax.set_ylabel('Density', fontsize=10)
                        ax.set_xlabel('Cosine silhouette', fontsize=10)
                for col in range(len(cells), 5): axes[row, col].axis('off')
            if kind == 'silhouette':
                ymax = max(ax.get_ylim()[1] for row in axes for ax in row if ax.has_data())
                for row in axes:
                    for ax in row:
                        if ax.has_data(): ax.set_ylim(0, ymax)
            if kind == 'umap':
                handles = [Line2D([], [], marker='o', color='w', markerfacecolor=palette[i], markersize=7, label=n.replace('_', ' ')) for i, n in enumerate(names[aspect])]
                if any((y < 0).any() for y in split_labels.values()):
                    handles.append(Line2D([], [], marker='o', color='w', markerfacecolor='#c9c9c9', markersize=7, label='Unlabelled'))
                fig.legend(handles=handles, loc='lower center', ncol=7 if aspect == 'style' else 6, frameon=False, fontsize=10, columnspacing=1.2, handletextpad=.4, bbox_to_anchor=(.5, .015))
                fig.subplots_adjust(left=.045, right=.99, top=.90, bottom=.20 if aspect == 'style' else .14, wspace=.19, hspace=.34)
            else:
                fig.legend(handles=[Line2D([], [], color='#4c78a8', linewidth=6, alpha=.65, label='this arm'), Line2D([], [], color='#222222', label='c* on the same split'), Line2D([], [], color='#b22222', label='mean of this arm')], loc='lower center', ncol=3, frameon=False, fontsize=11)
                fig.subplots_adjust(left=.055, right=.99, top=.90, bottom=.12, wspace=.3, hspace=.42)
            for ext in ('png', 'pdf'):
                fig.savefig(OUT / f'{kind}_{aspect}.{ext}', dpi=200)
            plt.close(fig)
    lines = ['# r21 figures', '', 'UMAP figures show true aspect labels (grey: unlabelled); silhouette figures show each arm against c* on the same split. Missing exports appear as empty panels.', '', 'UMAP: 2 components, cosine metric, 15 neighbours, min_dist 0.1, random_state 0; fitted to all rows separately per cell. Partitions: MiniBatchKMeans at the aspect class count, state 0. Silhouette: exact per-row cosine distance, all rows. Cache identities contain the input array SHA-256 and settings.', '', '| Aspect | Split | Arm | Mean | Reference | Absolute difference | Seconds |', '|---|---|---|---:|---:|---:|---:|']
    for aspect, split, arm, mean, ref, seconds in records:
        lines.append(f'| {aspect} | {split} | {arm} | {mean:.9f} | {ref:.9f} | {abs(mean-ref):.9g} | {seconds:.1f} |' if np.isfinite(ref) else f'| {aspect} | {split} | {arm} | {mean:.9f} | unavailable | unavailable | {seconds:.1f} |')
    lines += ['', 'Shared silhouette bins cover the minimum and maximum across all cached arms and both splits, with a margin and an empty bin at each edge. All values are asserted to lie inside the bins.', '', '| Aspect | Split | Unlabelled rows |', '|---|---|---:|']
    lines += [f'| {aspect} | {split} | {count} |' for aspect, split, count in unlabelled]
    (OUT / 'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest='command', required=True)
    c = subs.add_parser('compute'); c.add_argument('--aspects', nargs='+', choices=SC.ASPECTS, default=SC.ASPECTS)
    subs.add_parser('plot')
    args = parser.parse_args()
    compute(args.aspects) if args.command == 'compute' else plot()


if __name__ == '__main__':
    main()
