"""r30 add-on: do the two UMAP islands of long DEC (r21's own emotion map, validation) equal the k = 2 k-means groups?

Also: genre and style make-up of each group, and the islands' AMI against valence, emotion, style and genre.
Writes OUT/islands.json and prints a short summary.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r30_followup as R
F = R.F


def main():
    F.SC._deps(); F.T._deps()
    import r20_data
    from sklearn.cluster import KMeans
    names = r20_data._artelingo_class_names()
    rt = F.SC.rt_data
    emotion_names = sorted(rt.rs_vh.EMOTIONS)
    shares, val = R.shares_and_labels(rt.emotion_dist('val'), emotion_names)
    labels = {a: np.asarray(F.SC.ns_data.labels(F.SC.SET, 'val', a)) for a in F.SC.ASPECTS}
    targets = {'valence': val, **labels}
    out = {}
    for arm in ('cstar', 'dec-plain_l10_w5_S1'):
        x = F.SC.ns_data._unit32(R.load('val', arm, 'emotion', None))
        part = F.SC.ns_metrics.partitions(x, 2, F.SC.SET)[0]
        with np.load(F.OUT / 'cache' / f'val__{arm}__emotion__s0.npz', allow_pickle=False) as z:
            xy = z['coords']
        islands = KMeans(2, n_init=10, random_state=0).fit(xy).labels_
        gap = None
        if arm != 'cstar':
            # How far apart the two islands are: smallest cross distance over a sample, against the median spread.
            rng = np.random.default_rng(0)
            a = xy[islands == 0][rng.choice((islands == 0).sum(), 1500, replace=False)]
            b = xy[islands == 1][rng.choice((islands == 1).sum(), 1500, replace=False)]
            cross = np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1)).min()
            gap = float(cross)
        row = {'kmeans_vs_islands_ami': F.SC.ns_metrics.ami(part, islands),
               'island_min_gap_2d': gap, 'island_sizes': np.bincount(islands).tolist(),
               'islands_ami': {}, 'groups': {}}
        for t, y in targets.items():
            m = y >= 0
            row['islands_ami'][t] = F.SC.ns_metrics.ami(y[m], islands[m])
        for name, g in (('kmeans', part), ('islands', islands)):
            row['groups'][name] = []
            for c in (0, 1):
                m = g == c
                gl = labels['genre'][m]; gl = gl[gl >= 0]
                sl = labels['style'][m]
                genre = {names['genre'][i]: float((gl == i).mean()) for i in range(len(names['genre']))}
                style = {names['style'][i]: float((sl == i).mean()) for i in range(len(names['style']))}
                row['groups'][name].append({
                    'size': int(m.sum()), 'mean_pos': float(shares[m, 0].mean()), 'mean_neg': float(shares[m, 1].mean()),
                    'genre_top': sorted(genre.items(), key=lambda kv: -kv[1])[:4],
                    'style_top': sorted(style.items(), key=lambda kv: -kv[1])[:4]})
        out[arm] = row
    dest = R.OUT / 'islands.json'
    dest.write_text(json.dumps(out, indent=2) + '\n')
    for arm, row in out.items():
        print(f"== {arm}: k-means vs islands AMI {row['kmeans_vs_islands_ami']:.3f}; gap {row['island_min_gap_2d']}; "
              f"island AMI " + ', '.join(f"{k} {v:.4f}" for k, v in row['islands_ami'].items()))
        for name, gs in row['groups'].items():
            for i, g in enumerate(gs):
                print(f"  {name} {i}: n={g['size']} pos={g['mean_pos']:.3f} neg={g['mean_neg']:.3f} "
                      f"genre {[(k, round(v, 2)) for k, v in g['genre_top']]} style {[(k, round(v, 2)) for k, v in g['style_top'][:3]]}")


if __name__ == '__main__':
    main()
