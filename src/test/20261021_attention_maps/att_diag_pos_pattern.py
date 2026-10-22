"""Is part of each image map a fixed position pattern shared by all paintings with the same grid size?

Per view, band layers 16-27, image-only maps. Groups = grid_hw with >= 40 paintings.
- rho_loo: Spearman between a painting's map and the mean map of the other paintings in its group.
- rho_pair: Spearman between two different paintings' maps (random pairs in the group).
- Moran's I of the raw map and of the ratio map (map / leave-one-out group mean).
"""
import sys
from collections import defaultdict
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, '/project/CoSiR/src/test/20261021_attention_maps')
import att_core as C

ROOT = '/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_attention/'
A, B = 16, 28


def load(k):
    out = {}
    with np.load(C.painting_path(ROOT, k)) as z:
        g = tuple(int(x) for x in z['grid_hw'])
        for v in C.VIEWS:
            off = z[f'{v}__img__off']
            a, b = int(off[0]), int(off[1])
            L = z[f'{v}__img__layers'][A:B, a:b].astype(np.float64).sum(0)
            tt = z[f'{v}__img__ttype'][a:b]
            m = L[tt == 2]
            out[v] = (m / m.sum()).astype(np.float32)
    return k, g, out


def rank(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind='stable')] = np.arange(len(x))
    return r


def spear(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def moran(m, g):
    x = m.reshape(g) - m.mean()
    num = (x[:, 1:] * x[:, :-1]).sum() + (x[1:, :] * x[:-1, :]).sum()
    w = x[:, 1:].size + x[1:, :].size
    return float(m.size / w * num / (x ** 2).sum())


def q(a):
    a = np.asarray(a)
    return f'{np.median(a):.3f} [{np.percentile(a, 10):.3f}, {np.percentile(a, 90):.3f}]'


def main():
    with Pool(8) as p:
        rows = p.map(load, range(5000), chunksize=20)
    groups = defaultdict(list)
    for k, g, out in rows:
        groups[g].append(out)
    big = sorted([g for g in groups if len(groups[g]) >= 40], key=lambda g: -len(groups[g]))
    print(f'groups with >= 40 paintings: {len(big)}, covering {sum(len(groups[g]) for g in big)} paintings; '
          f'largest: {[(g, len(groups[g])) for g in big[:6]]}')
    rng = np.random.default_rng(0)
    for v in C.VIEWS:
        loo, pair, mi_raw, mi_ratio, r2 = [], [], [], [], []
        for g in big:
            M = np.stack([o[v] for o in groups[g]])           # (n, tokens)
            n = len(M)
            S = M.sum(0)
            for i in range(n):
                mu = (S - M[i]) / (n - 1)
                loo.append(spear(M[i], mu))
                mi_raw.append(moran(M[i], g))
                mi_ratio.append(moran(M[i] / mu, g))
                c = np.corrcoef(M[i], mu)[0, 1]
                r2.append(c * c)
            for _ in range(min(200, n * (n - 1) // 2)):
                i, j = rng.choice(n, 2, replace=False)
                pair.append(spear(M[i], M[j]))
        print(f'{v:14s} rho_loo {q(loo)}  rho_pair {q(pair)}  R2(value) {q(r2)}  '
              f'Moran raw {q(mi_raw)}  Moran ratio {q(mi_ratio)}')
    # cross-view: is the shared pattern the same across views? (group mean maps, largest group)
    g = big[0]
    means = {v: np.stack([o[v] for o in groups[g]]).mean(0) for v in C.VIEWS}
    print('largest group', g, 'Spearman of group-mean maps between views:',
          {f'{a[:4]}-{b[:4]}': round(spear(means[a], means[b]), 3)
           for i, a in enumerate(C.VIEWS) for b in C.VIEWS[i + 1:]})
    np.savez('/project/CoSiR/src/test/20261021_attention_maps/outputs/spatial_diag/pos_means.npz',
             grid=np.array(g), **{v: means[v] for v in C.VIEWS})


if __name__ == '__main__':
    main()
