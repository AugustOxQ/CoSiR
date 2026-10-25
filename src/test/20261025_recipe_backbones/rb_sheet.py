"""Four-view sheet (spec §3.6, D14, D34): descriptive statistics of the default and the three instructed Qwen views.

Views: default, emotion_focus, style_word, genre_word. Sets: `P_image` (the 29,232 paintings, images only),
`old5000_image` and `old5000_caption` (painting means of the captions; P lacks style and genre instructed captions).
Per view: norms as stored (the raw float16 store rows before any normalization; caption side: every stored caption
of the old 5,000 paintings), then on unit vectors: anisotropy (mean pairwise cosine over distinct pairs, exact via the
sum vector; the top principal component's share of the total variance), each instructed view's cosine to the default
(mean and quartiles, row by row), 20-NN (cosine, self excluded) label purity per aspect, cosine silhouette per label
(exact, via per-class sums) and hubness (skew of the 20-NN in-degree). Writes `part1/sheet.json` and `part1/sheet.md`.
Exploration only: decides nothing.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import rb_paths
import rb_data
import ct2_data

VIEWS = ('default', 'emotion_focus', 'style_word', 'genre_word')
ASPECTS = rb_paths.RB_ASPECTS
K = 20
THREADS = 8
BLOCK = 1024
SMOKE_ROWS = 1000
_CODE = ('rb_sheet.py', 'rb_data.py')


# ------------------------------------------------------------------ statistics (unit vectors)
def _quartiles(v):
    q = np.percentile(np.asarray(v, dtype=np.float64), [25, 50, 75])
    return {'q25': float(q[0]), 'median': float(q[1]), 'q75': float(q[2])}


def norm_summary(norms):
    """Summary of stored norms (float64 over a float32 vector of row norms)."""
    v = np.asarray(norms, dtype=np.float64)
    assert v.ndim == 1 and len(v) > 0 and np.isfinite(v).all()
    return {'n': int(len(v)), 'mean': float(v.mean()), 'std': float(v.std()), 'min': float(v.min()),
            'max': float(v.max()), **_quartiles(v)}


def anisotropy(x):
    """Mean pairwise cosine over distinct pairs (exact, from the sum vector) and the top component's variance share."""
    n = x.shape[0]
    assert n > 2
    total = x.sum(axis=0, dtype=np.float64)
    mean_cos = float((total @ total - n) / (n * (n - 1)))      # unit rows: |sum|^2 = n + sum of ordered pair cosines
    xc = x - x.mean(axis=0, dtype=np.float64).astype(np.float32)
    cov = (xc.T @ xc).astype(np.float64)
    eig = np.linalg.eigvalsh(cov)
    return {'mean_pairwise_cosine': mean_cos, 'top_pc_variance_share': float(eig[-1] / eig.sum())}


def cosine_to(x, d):
    """Row-wise cosine of unit rows of x to the same rows of d: mean and quartiles."""
    assert x.shape == d.shape
    c = np.einsum('ij,ij->i', x, d, dtype=np.float64)
    return {'mean': float(c.mean()), **_quartiles(c)}


def knn(x, k=K):
    """Indices [n, k] of each row's k nearest rows by cosine (unit rows), self excluded."""
    n = x.shape[0]
    assert n > k
    out = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, BLOCK):
        stop = min(start + BLOCK, n)
        sims = x[start:stop] @ x.T
        sims[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        out[start:stop] = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    assert not (out == np.arange(n)[:, None]).any()
    return out


def purity(nbr, y):
    """Mean over labelled rows of the share of labelled neighbours with the row's label; and the chance level."""
    ln = y[nbr]
    valid = ln >= 0
    same = (ln == y[:, None]) & valid
    keep = (y >= 0) & (valid.sum(axis=1) > 0)
    p = same.sum(axis=1)[keep] / valid.sum(axis=1)[keep]
    lab = y[y >= 0]
    share = np.bincount(lab) / len(lab)
    return {'purity': float(p.mean()), 'chance': float((share ** 2).sum()), 'rows': int(keep.sum()),
            'labelled': int(len(lab))}


def silhouette(x, y):
    """Cosine silhouette (distance 1 - cos) of the labelled rows, exact via class sums; per label and overall.

    A singleton label scores 0 for its row (the usual convention). Overall = mean over rows.
    """
    keep = y >= 0
    xl, yl = x[keep], y[keep]
    classes, inv, counts = np.unique(yl, return_inverse=True, return_counts=True)
    assert len(classes) >= 2, 'silhouette needs two labels'
    sums = np.zeros((len(classes), x.shape[1]), dtype=np.float64)
    np.add.at(sums, inv, xl)
    mean_cos = (xl @ sums.T.astype(np.float32)).astype(np.float64) / counts[None, :]     # [rows, classes]
    idx = np.arange(len(yl))
    own_n = counts[inv]
    a = np.where(own_n > 1, 1 - (mean_cos[idx, inv] * own_n - 1) / np.maximum(own_n - 1, 1), 0.0)
    other = 1 - mean_cos
    other[idx, inv] = np.inf
    b = other.min(axis=1)
    s = np.where(own_n > 1, (b - a) / np.maximum(np.maximum(a, b), 1e-12), 0.0)
    per = {str(int(c)): {'n': int(counts[i]), 'silhouette': float(s[inv == i].mean())} for i, c in enumerate(classes)}
    return {'overall': float(s.mean()), 'per_label': per}


def skew(v):
    v = np.asarray(v, dtype=np.float64)
    m = v - v.mean()
    return float((m ** 3).mean() / (m ** 2).mean() ** 1.5)


def hubness(nbr, n):
    """Skew of the k-NN in-degree, with its maximum and the share of rows nobody lists."""
    deg = np.bincount(nbr.ravel(), minlength=n)
    assert deg.sum() == nbr.size
    return {'skew': skew(deg), 'max_in_degree': int(deg.max()), 'share_zero_in_degree': float((deg == 0).mean())}


# ------------------------------------------------------------------ one set of views
def compute_set(views, norms, labels):
    """Every statistic for one set. `views` {view: unit float32 [n, w]}, `norms` {view: norm summary},
    `labels` {aspect: int64 [n]} (-1 masked); all in one row order."""
    assert tuple(views) == VIEWS and tuple(norms) == VIEWS and tuple(labels) == ASPECTS
    n = views['default'].shape[0]
    for v in views.values():
        rb_data.check_unit(v, n)
    for y in labels.values():
        assert y.shape == (n,) and y.dtype == np.int64
    out = {}
    with threadpool_limits(limits=THREADS):
        for view, x in views.items():
            nbr = knn(x)
            block = {'norms_as_stored': norms[view], 'anisotropy': anisotropy(x),
                     'purity_20nn': {a: purity(nbr, labels[a]) for a in ASPECTS},
                     'silhouette_cosine': {a: silhouette(x, labels[a]) for a in ASPECTS},
                     'hubness_20nn': hubness(nbr, n)}
            if view != 'default':
                block['cosine_to_default'] = cosine_to(x, views['default'])
            out[view] = block
    return {'n': int(n), 'labelled': {a: int((labels[a] >= 0).sum()) for a in ASPECTS}, 'views': out}


# ------------------------------------------------------------------ stored norms from the raw stores
def stored_image_norms(paths, wanted, sealed_key=None):
    """Norms of the raw float16 store rows for `wanted` names, first store wins (as `ct2_data.instructed_images`)."""
    wanted = [str(w) for w in wanted]
    assert len(set(wanted)) == len(wanted)
    lookup = {name: row for row, name in enumerate(wanted)}
    norms = np.zeros(len(wanted), dtype=np.float64)
    seen = np.zeros(len(wanted), dtype=bool)
    for path in paths:
        if sealed_key is not None:
            ct2_data._sealed(path, sealed_key)
        names, raw = ct2_data._image_store(path)
        source = np.asarray([r for r, name in enumerate(names) if str(name) in lookup], dtype=np.int64)
        target = np.asarray([lookup[str(names[r])] for r in source], dtype=np.int64)
        if len(source) == 0:          # e.g. the fit/stop store holds no P painting
            continue
        values = np.asarray(raw[source], dtype=np.float32)
        assert np.isfinite(values).all()
        new = ~seen[target]
        norms[target[new]] = np.linalg.norm(values, axis=1)[new]
        seen[target] = True
    assert seen.all(), f'stored norms: {int((~seen).sum())} paintings without a row'
    assert (norms > 0).all()
    return norms


def _image_paths(key, full):
    if key == 'default':
        return [ct2_data.QWEN], None
    paths = [ct2_data.ASPECT / key, ct2_data.ASPECT / f'{key}_trainimg']
    if full:
        paths.append(ct2_data.ASPECT / f'{key}_s2img')
    return paths, key


def _image_norms(key, names, full):
    paths, sealed = _image_paths(key, full)
    return norm_summary(stored_image_norms(paths, names, sealed))


# ------------------------------------------------------------------ the real sets
def load_p_image():
    names = rb_data.p_names()
    views = {'default': rb_data.p_default_images(), **{k: rb_data.p_instructed_images(k) for k in VIEWS[1:]}}
    norms = {k: _image_norms(k, names, True) for k in VIEWS}
    return names, views, norms, rb_data.p_labels()


def load_old_image():
    names = rb_data.old5000_names()
    views = {'default': rb_data.old5000_default_images(), **{k: rb_data.old5000_instructed_images(k) for k in VIEWS[1:]}}
    norms = {k: _image_norms(k, names, False) for k in VIEWS}
    return names, views, norms, rb_data.old5000_labels()


def load_old_caption():
    """Painting means of the captions (unit) for the old 5,000; norms are those of every stored caption of them."""
    s = rb_data._sample()
    names = rb_data.old5000_names()
    in_old = np.isin(s['pid'], np.asarray(s['subset_pid']))
    views, norms = {}, {}
    for key in VIEWS:
        if key == 'default':
            raw = ct2_data.default_captions(s, per_caption=True)
        else:
            raw = ct2_data.instructed_captions(s, key, per_caption=True)
        norms[key] = norm_summary(np.linalg.norm(raw[in_old], axis=1))
        mean = ct2_data.l2(ct2_data.paint_mean(raw, s['pid'], len(s['first'])))
        views[key] = rb_data.select_role('old5000', mean)
        del raw
    return names, views, norms, rb_data.old5000_labels()


LOADERS = {'P_image': load_p_image, 'old5000_image': load_old_image, 'old5000_caption': load_old_caption}


def subsample(names, views, labels, rows):
    """Smoke only: every stride-th row of one set, the same rows in every array."""
    sel = np.arange(0, len(names), max(1, len(names) // rows))[:rows]
    return ([names[i] for i in sel], {k: v[sel] for k, v in views.items()}, {a: y[sel] for a, y in labels.items()})


# ------------------------------------------------------------------ output
def code_sha256():
    here = Path(__file__).resolve().parent
    return {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in _CODE}


def fmt(x):
    return f'{x:.4f}'


def render_md(sheet):
    lines = ['# Four-view sheet', '', f"smoke: {sheet['smoke']}. Exploration only; decides nothing.", '']
    for sname, st in sheet['sets'].items():
        lines += [f"## {sname} (n = {st['n']}; labelled {st['labelled']})", '']
        lines += ['| view | norm mean | norm min..max | mean pair cos | top PC share | cos to default (mean; q25/med/q75) '
                  '| hub skew | silhouette emo/sty/gen |', '|---|---|---|---|---|---|---|---|']
        for v, b in st['views'].items():
            nm, an = b['norms_as_stored'], b['anisotropy']
            cd = b.get('cosine_to_default')
            cds = '-' if cd is None else f"{fmt(cd['mean'])}; {fmt(cd['q25'])}/{fmt(cd['median'])}/{fmt(cd['q75'])}"
            sil = '/'.join(fmt(b['silhouette_cosine'][a]['overall']) for a in ASPECTS)
            lines.append(f"| {v} | {fmt(nm['mean'])} | {fmt(nm['min'])}..{fmt(nm['max'])} | {fmt(an['mean_pairwise_cosine'])} "
                         f"| {fmt(an['top_pc_variance_share'])} | {cds} | {fmt(b['hubness_20nn']['skew'])} | {sil} |")
        lines += ['', '20-NN label purity (chance in brackets)', '', '| view | emotion | style | genre |', '|---|---|---|---|']
        for v, b in st['views'].items():
            cells = [f"{fmt(b['purity_20nn'][a]['purity'])} ({fmt(b['purity_20nn'][a]['chance'])})" for a in ASPECTS]
            lines.append(f'| {v} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines.append('Per-label silhouettes are in sheet.json.')
    return '\n'.join(lines) + '\n'


def write_sheet(outputs, sheet):
    folder = Path(outputs) / 'part1'
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in (('sheet.json', json.dumps(sheet, indent=1) + '\n'), ('sheet.md', render_md(sheet))):
        tmp = folder / f'{name}.pending'
        tmp.write_text(text)
        tmp.replace(folder / name)
    return folder / 'sheet.json'


def build(sets, smoke=False):
    """`sets` {set name: (names, views, norms, labels)} -> the sheet dictionary."""
    out = {}
    for sname, (names, views, norms, labels) in sets.items():
        t0 = time.time()
        if smoke:
            names, views, labels = subsample(names, views, labels, SMOKE_ROWS)
        res = compute_set(views, norms, labels)
        res['names_sha256'] = rb_data.names_sha256(list(names))
        res['seconds'] = round(time.time() - t0, 1)
        out[sname] = res
        print(f'{sname}: n={res["n"]} {res["seconds"]} s', flush=True)
    return {'smoke': bool(smoke), 'k': K, 'views': list(VIEWS), 'code_sha256': code_sha256(), 'sets': out}


def main(argv=None):
    ap = argparse.ArgumentParser(description='Write part1/sheet.json and sheet.md (four-view sheet).')
    ap.add_argument('--outputs', type=Path, default=rb_paths.OUTPUTS)
    ap.add_argument('--smoke', action='store_true', help='1,000 rows per set; outputs under /tmp/claude-0/')
    args = ap.parse_args(argv)
    if args.smoke:
        assert str(args.outputs.resolve()).startswith('/tmp/claude-0/'), 'smoke needs --outputs under /tmp/claude-0/'
    rb_paths.check_imports()
    p = Path(args.outputs) / 'part1' / 'P.json'
    if p.exists():
        rb_data.read_p_json(args.outputs, expect=rb_data.p_names())
    sets = {}
    for sname, loader in LOADERS.items():
        names, views, norms, labels = loader()
        sets[sname] = (names, views, norms, labels)
    sheet = build(sets, smoke=args.smoke)
    print(write_sheet(args.outputs, sheet))


if __name__ == '__main__':
    sys.exit(main())
