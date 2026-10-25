"""End-to-end test of rb_sheet on small synthetic data."""
import json

import numpy as np

import rb_sheet


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def synthetic(n=300, w=16, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n).astype(np.int64)
    centers = rng.normal(size=(3, w)) * 3
    default = unit(centers[y] + rng.normal(size=(n, w)) + 2.0)
    views = {'default': default}
    for k in rb_sheet.VIEWS[1:]:
        views[k] = unit(default + 0.3 * rng.normal(size=(n, w)))
    labels = {'emotion': y, 'style': (y + 1) % 3, 'genre': np.where(rng.random(n) < 0.2, -1, y).astype(np.int64)}
    return views, labels


def test_statistics_match_brute_force():
    views, labels = synthetic()
    x, y = views['emotion_focus'], labels['genre']
    n = len(x)
    cos = x @ x.T
    off = ~np.eye(n, dtype=bool)
    an = rb_sheet.anisotropy(x)
    assert abs(an['mean_pairwise_cosine'] - cos[off].mean()) < 1e-6
    ev = np.linalg.eigvalsh(np.cov(x.T.astype(np.float64)))
    assert abs(an['top_pc_variance_share'] - ev[-1] / ev.sum()) < 1e-5
    nbr = rb_sheet.knn(x, 20)
    masked = np.where(off, cos, -np.inf)
    ref = np.argsort(-masked, axis=1)[:, :20]
    assert all(set(a) == set(b) for a, b in zip(nbr, ref))
    # silhouette against the textbook definition
    d = 1 - cos
    lab = np.flatnonzero(y >= 0)
    s = []
    for i in lab:
        same = lab[(y[lab] == y[i]) & (lab != i)]
        a = d[i, same].mean()
        b = min(d[i, lab[y[lab] == c]].mean() for c in set(y[lab]) if c != y[i])
        s.append((b - a) / max(a, b))
    assert abs(rb_sheet.silhouette(x, y)['overall'] - np.mean(s)) < 1e-5


def test_sheet_end_to_end(tmp_path):
    views, labels = synthetic()
    norms = {k: rb_sheet.norm_summary(np.linspace(0.9, 1.1, 50)) for k in rb_sheet.VIEWS}
    names = [f'p{i:03d}' for i in range(300)]
    sheet = rb_sheet.build({'toy': (names, views, norms, labels)})
    path = rb_sheet.write_sheet(tmp_path, sheet)
    assert path == tmp_path / 'part1' / 'sheet.json' and (tmp_path / 'part1' / 'sheet.md').exists()
    got = json.loads(path.read_text())['sets']['toy']
    assert got['n'] == 300 and set(got['views']) == set(rb_sheet.VIEWS)
    assert 'cosine_to_default' not in got['views']['default'] and 'cosine_to_default' in got['views']['genre_word']
    assert got['views']['default']['purity_20nn']['emotion']['purity'] > 0.6
    assert got['views']['default']['purity_20nn']['emotion']['chance'] < 0.5
    assert got['views']['default']['silhouette_cosine']['emotion']['overall'] > 0
    assert got['names_sha256'] and got['views']['default']['hubness_20nn']['skew'] == got['views']['default']['hubness_20nn']['skew']


def test_stored_norms_first_store_wins(tmp_path):
    import rb_paths  # noqa: F401
    import ct2_data
    for d, rows in (('a', {'x': 2.0}), ('b', {'x': 5.0, 'y': 3.0})):
        p = tmp_path / d
        p.mkdir()
        names = list(rows)
        arr = np.zeros((len(names), 2048), dtype=np.float16)
        for i, nme in enumerate(names):
            arr[i, 0] = rows[nme]
        np.save(p / 'img.npy', arr)
        np.savez(p / 'ids.npz', painting_names=np.asarray(names))
    got = rb_sheet.stored_image_norms([tmp_path / 'a', tmp_path / 'b'], ['y', 'x'])
    assert np.allclose(got, [3.0, 2.0])
