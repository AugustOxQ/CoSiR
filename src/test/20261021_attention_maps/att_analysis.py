"""Summary analysis of the Qwen3-VL-Embedding-2B attention maps (descriptive; nothing here tunes or selects).

Run:  python att_analysis.py --root <dir with p0000.npz ...> [--out <dir>] [--workers 8] [--limit N] [--min-class 30]
Outputs in --out (default outputs/<root name>/ next to this file): analysis.json, analysis.md, figures *.png,
per_painting.csv.gz.

Measures (one section each in analysis.md / key m1..m7 in analysis.json); every measure is given for the three bands of
att_core.BANDS (layers 16-27 is the default):
  m1 shares of the pooled token's raw attention by layer (first token, instruction, image, caption, other template)
  m2 shape of the image map (normalised entropy, top-1% mass, spatial spread, vertical centre, border excess, hottest on ring)
  m3 change between views (L1, Jensen-Shannon base 2, top-5% Jaccard) with a shuffled-map baseline
  m4 img map against the joint (image+caption) pooled image map, and the joint image / caption split
  m5 caption-word weights of the pooled token (txt and joint): highest, and rising against the default view
  m6 labels (emotion, style, genre): class means and Kruskal-Wallis tests
  m7 cross maps (joint): caption-word maps against the pooled image map

Conventions: token weights in m5 are the pooled token's band-summed attention on caption tokens, renormalised to sum 1
over the caption (att_core.band_map(..., 'caption')), averaged per occurrence. Image tokens are read row-major on the
grid_hw grid. For joint, shape and change measures are computed per caption and averaged within a painting first.
"""
import argparse
import gzip
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from att_core import BANDS, VIEWS, N_LAYERS, load_painting, band_map  # noqa: E402

REPO = Path('/project/CoSiR')
BAND_NAMES = [b[0] for b in BANDS]
NB = len(BANDS)
NOISE_L1 = {'layers 16-27': 0.09, 'last layer': 0.14, 'all layers': 0.06}   # bf16 against float32 (round log)
CATS = ('first token', 'instruction', 'image', 'caption', 'other template')
SHAPE_MEAS = ('norm_entropy', 'top1pct_mass', 'spread_row', 'spread_col', 'vertical_centre', 'border_excess', 'hottest_on_ring')
PAIRS = (('default', 'emotion_focus'), ('default', 'style_word'), ('default', 'genre_word'),
         ('emotion_focus', 'style_word'), ('emotion_focus', 'genre_word'), ('style_word', 'genre_word'))
PAIR_NAMES = [f'{a} vs {b}' for a, b in PAIRS]
PMETRICS = ('l1', 'js', 'jaccard5')
MIN_OCC = 50
SEED = 0


# ================================================================= small map measures
def norm_entropy(p):
    p = np.asarray(p, np.float64)
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / math.log(len(p))) if len(p) > 1 else float('nan')


def l1(a, b):
    return float(np.abs(a - b).sum())


def js2(a, b):
    m = 0.5 * (a + b)

    def kl(x):
        s = x > 0
        return float((x[s] * np.log2(x[s] / m[s])).sum())
    return 0.5 * kl(a) + 0.5 * kl(b)


def top_set(a, frac):
    n = len(a)
    m = max(1, int(round(frac * n)))
    return set(np.argpartition(-a, m - 1)[:m].tolist())


def jaccard_top(a, b, frac=0.05):
    sa, sb = top_set(a, frac), top_set(b, frac)
    return len(sa & sb) / len(sa | sb)


def ring_mask(h, w):
    r = np.zeros((h, w), bool)
    r[0, :] = r[-1, :] = True
    r[:, 0] = r[:, -1] = True
    return r.reshape(-1)


def shape_measures(m, h, w):
    """m: map (sum 1, row-major h x w). Returns the 7 values of SHAPE_MEAS."""
    n = h * w
    ent = norm_entropy(m)
    top1 = float(np.sort(m)[::-1][:max(1, int(round(0.01 * n)))].sum())
    rr, cc = np.divmod(np.arange(n), w)
    mr, mc = float((m * rr).sum()), float((m * cc).sum())
    sr = math.sqrt(max(float((m * (rr - mr) ** 2).sum()), 0.0)) / h
    sc = math.sqrt(max(float((m * (cc - mc) ** 2).sum()), 0.0)) / w
    vc = mr / (h - 1) if h > 1 else float('nan')
    ring = ring_mask(h, w)
    share = ring.sum() / n
    be = float(m[ring].sum() / share) if share < 1 else float('nan')
    hot = float(ring[int(np.argmax(m))])
    return [ent, top1, sr, sc, vc, be, hot]


def shares_by_layer(p):
    """(28, 5) raw attention of the pooled token on first token, instruction, image, caption, other template."""
    L = np.asarray(p['layers'], np.float64)
    tt = np.asarray(p['ttype'])
    idx = np.arange(len(tt))
    return np.stack([L[:, 0], L[:, tt == 1].sum(1), L[:, tt == 2].sum(1), L[:, tt == 3].sum(1),
                     L[:, (tt == 0) & (idx > 0)].sum(1)], 1)


# ================================================================= per painting
def analyse_painting(args):
    root, k = args
    try:
        P = load_painting(root, k)
    except Exception as e:  # incomplete or unreadable file
        return dict(k=k, error=repr(e))
    if any(v not in P for v in VIEWS):
        return dict(k=k, error='missing views')
    pa = P['painting']
    h, w = (int(x) for x in pa['grid_hw'])
    n_img = h * w
    ncap = len(pa['cap_emo'])
    assert len(P['default']['txt']) == ncap and len(P['default']['joint']) == ncap and len(P['default']['img']) == 1
    sty, gen = np.unique(pa['sty']), np.unique(pa['gen'])
    assert len(sty) == 1 and len(gen) == 1, 'style/genre not constant per painting'
    emo = int(np.bincount(np.asarray(pa['cap_emo'], np.int64), minlength=9).argmax())
    out = dict(k=int(k), painting_id=str(pa['painting_id']), emo=emo, sty=int(sty[0]), gen=int(gen[0]), ncap=ncap,
               grid=(h, w))
    nv = len(VIEWS)
    shares = np.zeros((nv, 3, N_LAYERS, 5))
    shape = np.full((nv, 2, NB, len(SHAPE_MEAS)), np.nan)
    add = np.full((nv, NB, 2), np.nan)
    cross = np.full((nv, NB, 3), np.nan)       # mean word-map entropy, mean pooled entropy, mean word-vs-pooled correlation
    cross_drop = np.zeros((nv, NB))
    words = {}      # (view, mod, band) -> (ids, weights)
    cross_tok = {}  # (view, band) -> (ids, entropies)
    imgmap = np.zeros((nv, NB, n_img))
    for vi, v in enumerate(VIEWS):
        d = P[v]
        pi = d['img'][0]
        shares[vi, 0] = shares_by_layer(pi)
        shares[vi, 1] = np.mean([shares_by_layer(p) for p in d['txt']], 0)
        shares[vi, 2] = np.mean([shares_by_layer(p) for p in d['joint']], 0)
        for bi, band in enumerate(BANDS):
            mi, _ = band_map(pi, band, 'image')
            assert len(mi) == n_img
            imgmap[vi, bi] = mi
            shape[vi, 0, bi] = shape_measures(mi, h, w)
            sj, l1s, jss = [], [], []
            ent_w, ent_p, cor_w, ids_all, ent_all = [], [], [], [], []
            for p in d['joint']:
                mj, _ = band_map(p, band, 'image')
                sj.append(shape_measures(mj, h, w))
                l1s.append(l1(mi, mj))
                jss.append(js2(mi, mj))
                cm = np.asarray(p['cross'][bi], np.float64)       # (T, n_img)
                ids_c = np.asarray(p['ids'])[np.asarray(p['ttype']) == 3]
                tot = cm.sum(1)
                ok = tot > 0
                cross_drop[vi, bi] += int((~ok).sum())
                if ok.any():
                    cm = cm[ok] / tot[ok, None]
                    es = np.array([norm_entropy(r) for r in cm])
                    ent_w.append(es.mean())
                    ent_p.append(norm_entropy(mj))
                    z = cm - cm.mean(1, keepdims=True)
                    zp = mj - mj.mean()
                    den = np.sqrt((z ** 2).sum(1)) * math.sqrt((zp ** 2).sum())
                    cor_w.append(np.mean(np.where(den > 0, (z @ zp) / np.maximum(den, 1e-300), np.nan)))
                    ids_all.append(ids_c[ok])
                    ent_all.append(es)
            shape[vi, 1, bi] = np.nanmean(np.array(sj), 0)
            add[vi, bi] = (np.mean(l1s), np.mean(jss))
            if ent_w:
                cross[vi, bi] = (np.mean(ent_w), np.mean(ent_p), np.nanmean(cor_w))
                cross_tok[(v, BAND_NAMES[bi])] = (np.concatenate(ids_all).astype(np.int32),
                                                  np.concatenate(ent_all).astype(np.float32))
            for m in ('txt', 'joint'):
                ids_l, w_l = [], []
                for p in d[m]:
                    tt = np.asarray(p['ttype'])
                    if not (tt == 3).any():
                        continue
                    wm, _ = band_map(p, band, 'caption')
                    ids_l.append(np.asarray(p['ids'])[tt == 3])
                    w_l.append(wm)
                if ids_l:
                    words[(v, m, BAND_NAMES[bi])] = (np.concatenate(ids_l).astype(np.int32),
                                                     np.concatenate(w_l).astype(np.float32))
    # change between views (img maps) and the shuffled baseline
    rng = np.random.default_rng([SEED, int(k)])
    vidx = {v: i for i, v in enumerate(VIEWS)}
    pair = np.zeros((len(PAIRS), NB, 3))
    pair_sh = np.zeros((len(PAIRS), NB, 3))
    for pi_, (a, b) in enumerate(PAIRS):
        perm = rng.permutation(n_img)
        for bi in range(NB):
            A, B = imgmap[vidx[a], bi], imgmap[vidx[b], bi]
            Bs = B[perm]
            pair[pi_, bi] = (l1(A, B), js2(A, B), jaccard_top(A, B))
            pair_sh[pi_, bi] = (l1(A, Bs), js2(A, Bs), jaccard_top(A, Bs))
    out.update(shares=shares, shape=shape, add=add, cross=cross, cross_drop=cross_drop, words=words,
               cross_tok=cross_tok, pair=pair, pair_sh=pair_sh)
    return out


# ================================================================= aggregation helpers
def stat(x, axis=0):
    """mean, median, p10, p90 over `axis` (nan-aware) -> dict of arrays."""
    x = np.asarray(x, np.float64)
    q = np.nanpercentile(x, [10, 50, 90], axis=axis)
    return dict(mean=np.nanmean(x, axis=axis), median=q[1], p10=q[0], p90=q[2])


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def label_names():
    """emotion, style, genre names. Uses att_label_names.json when present (viewer worker), else builds them."""
    f = HERE / 'att_label_names.json'
    if f.exists():
        try:
            j = json.loads(f.read_text())
            out = {}
            for key, long in (('emotion', 'emo'), ('style', 'sty'), ('genre', 'gen')):
                for kk in j:
                    if kk.lower().startswith(long):
                        v = j[kk]
                        out[key] = ({int(a): b for a, b in v.items()} if isinstance(v, dict) else dict(enumerate(v)))
            if len(out) == 3:
                return out, 'att_label_names.json'
        except Exception as e:
            print('att_label_names.json not usable:', e, flush=True)
    # same names as apc_data.load_sample(min_caps=3) (emo_names, sty_names), without loading its feature arrays
    sys.path.insert(0, str(REPO))
    from src.data.artelingo import load_artelingo
    from src.data.artelingo_splits import encode_labels
    from src.data.wikiart_genre import GENRE_NAMES
    d = load_artelingo()
    _, en = encode_labels(d.emotions)
    _, sn = encode_labels(d.art_styles)
    return dict(emotion=dict(enumerate(en)), style=dict(enumerate(sn)), genre=dict(enumerate(GENRE_NAMES))), \
        'built from load_artelingo / encode_labels / GENRE_NAMES'


def token_strings(ids):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-VL-Embedding-2B', local_files_only=True)
    return {int(i): tok.decode([int(i)]).strip().replace('\n', '\\n') or '(space)' for i in ids}


# ================================================================= main analysis
def aggregate(res, names, min_class, tok_cache):
    N = len(res)
    A = {}
    nv = len(VIEWS)
    vi = {v: i for i, v in enumerate(VIEWS)}
    shares = np.stack([r['shares'] for r in res])          # N, view, mod(img,txt,joint), layer, cat
    shape = np.stack([r['shape'] for r in res])            # N, view, mod(img,joint), band, meas
    pair = np.stack([r['pair'] for r in res])              # N, pair, band, metric
    pair_sh = np.stack([r['pair_sh'] for r in res])
    add = np.stack([r['add'] for r in res])                # N, view, band, (l1, js)
    cross = np.stack([r['cross'] for r in res])            # N, view, band, 3
    mods = ('img', 'txt', 'joint')

    # sanity numbers
    sums = shares.sum(-1)
    A['sanity'] = dict(n_paintings=N, max_abs_share_sum_minus_1=float(np.abs(sums - 1).max()),
                       cross_rows_dropped=float(np.sum([r['cross_drop'] for r in res])),
                       noise_floor_l1_bf16_vs_float32=NOISE_L1)

    # ---- m1
    m1 = {}
    for mi, m in enumerate(mods):
        m1[m] = {}
        for v in VIEWS:
            s = stat(shares[:, vi[v], mi], 0)                # each (layer, cat)
            m1[m][v] = {c: {k: s[k][:, ci] for k in s} for ci, c in enumerate(CATS)}
    A['m1_layer_shares'] = m1
    # band-level summary of shares (mean over the band's layers, per painting)
    m1b = {}
    for mi, m in enumerate(mods):
        for v in VIEWS:
            for bi, (bn, a, b) in enumerate(BANDS):
                bs = shares[:, vi[v], mi, a:b].mean(1)       # N, cat
                m1b[f'{m}|{v}|{bn}'] = {c: stat(bs[:, ci]) for ci, c in enumerate(CATS)}
    A['m1_band_shares'] = m1b

    # ---- m2
    m2 = {}
    for bi, bn in enumerate(BAND_NAMES):
        for mi, m in enumerate(('img', 'joint')):
            for v in VIEWS:
                m2[f'{bn}|{m}|{v}'] = {me: stat(shape[:, vi[v], mi, bi, j]) for j, me in enumerate(SHAPE_MEAS)}
    A['m2_shape'] = m2

    # ---- m3
    m3 = {}
    for bi, bn in enumerate(BAND_NAMES):
        for pi_, pn in enumerate(PAIR_NAMES):
            m3[f'{bn}|{pn}'] = dict(
                real={me: stat(pair[:, pi_, bi, j]) for j, me in enumerate(PMETRICS)},
                shuffled={me: stat(pair_sh[:, pi_, bi, j]) for j, me in enumerate(PMETRICS)},
                noise_floor_l1=NOISE_L1[bn])
    A['m3_view_change'] = m3

    # ---- m4
    m4 = {}
    for bi, (bn, a, b) in enumerate(BANDS):
        for v in VIEWS:
            js_ = shares[:, vi[v], 2, a:b].mean(1)           # N, cat (joint)
            m4[f'{bn}|{v}'] = dict(
                l1=stat(add[:, vi[v], bi, 0]), js=stat(add[:, vi[v], bi, 1]), noise_floor_l1=NOISE_L1[bn],
                joint_share={c: stat(js_[:, ci]) for ci, c in enumerate(CATS)})
    A['m4_img_vs_joint'] = m4

    # ---- m5 words
    def table_for(key_fn):
        ids_l, v_l = [], []
        for r in res:
            t = key_fn(r)
            if t is not None:
                ids_l.append(t[0])
                v_l.append(t[1])
        ids = np.concatenate(ids_l)
        vals = np.concatenate(v_l).astype(np.float64)
        V = int(ids.max()) + 1
        return np.bincount(ids, minlength=V), np.bincount(ids, weights=vals, minlength=V)

    need_ids = set()
    m5 = {}
    for bn in BAND_NAMES:
        for m in ('txt', 'joint'):
            tabs = {v: table_for(lambda r, v=v, m=m, bn=bn: r['words'].get((v, m, bn))) for v in VIEWS}
            V = max(len(t[0]) for t in tabs.values())
            cnt = {v: np.pad(tabs[v][0], (0, V - len(tabs[v][0]))) for v in VIEWS}
            sm = {v: np.pad(tabs[v][1], (0, V - len(tabs[v][1]))) for v in VIEWS}
            mean = {v: sm[v] / np.maximum(cnt[v], 1) for v in VIEWS}
            ent = {}
            for v in VIEWS:
                ok = np.flatnonzero(cnt[v] >= MIN_OCC)
                top = ok[np.argsort(-mean[v][ok])[:30]]
                rise = []
                if v != 'default':
                    okr = np.flatnonzero((cnt[v] >= MIN_OCC) & (cnt['default'] >= MIN_OCC))
                    d = mean[v][okr] - mean['default'][okr]
                    rise_ids = okr[np.argsort(-d)[:20]]
                    rise = [dict(id=int(i), n=int(cnt[v][i]), mean_weight=float(mean[v][i]),
                                 default_weight=float(mean['default'][i]), rise=float(mean[v][i] - mean['default'][i]))
                            for i in rise_ids]
                    need_ids |= set(int(i) for i in rise_ids)
                need_ids |= set(int(i) for i in top)
                ent[v] = dict(top=[dict(id=int(i), n=int(cnt[v][i]), mean_weight=float(mean[v][i])) for i in top],
                              rise_vs_default=rise, n_eligible_tokens=int(len(ok)))
            m5[f'{bn}|{m}'] = ent
    A['m5_words'] = m5

    # ---- m6 labels
    labs = dict(emotion=np.array([r['emo'] for r in res]), style=np.array([r['sty'] for r in res]),
                genre=np.array([r['gen'] for r in res]))
    from scipy.stats import kruskal
    imgshare_band = {bn: shares[:, :, 0, a:b, 2].mean(2) for bn, a, b in BANDS}     # N, view  (img modality image share)
    l1_def = {bn: np.stack([pair[:, 0, bi, 0], pair[:, 1, bi, 0], pair[:, 2, bi, 0]], 1) for bi, bn in enumerate(BAND_NAMES)}
    # l1_def columns: emotion_focus, style_word, genre_word (PAIRS 0..2 are default vs each)
    inst = VIEWS[1:]
    m6, tests = {}, []
    for ln, lv in labs.items():
        classes = [c for c in np.unique(lv) if c >= 0 and (lv == c).sum() >= min_class]
        for bn in BAND_NAMES:
            cl = {}
            for c in classes:
                sel = lv == c
                cl[names[ln].get(int(c), str(c))] = dict(
                    code=int(c), n=int(sel.sum()),
                    img_share={v: float(imgshare_band[bn][sel, vi[v]].mean()) for v in VIEWS},
                    l1_to_default={v: float(l1_def[bn][sel, j].mean()) for j, v in enumerate(inst)})
            m6[f'{ln}|{bn}'] = cl
            if len(classes) >= 2:
                for v in VIEWS:
                    g = [imgshare_band[bn][lv == c, vi[v]] for c in classes]
                    H, p = kruskal(*g)
                    tests.append(dict(label=ln, band=bn, view=v, metric='image share', H=float(H), p=float(p),
                                      n_classes=len(classes)))
                for j, v in enumerate(inst):
                    g = [l1_def[bn][lv == c, j] for c in classes]
                    H, p = kruskal(*g)
                    tests.append(dict(label=ln, band=bn, view=v, metric='L1 to default', H=float(H), p=float(p),
                                      n_classes=len(classes)))
    nt = len(tests)
    for t in tests:
        t['p_bonferroni'] = min(1.0, t['p'] * nt)
        t['flag_p_bonf_below_0.001'] = bool(t['p_bonferroni'] < 0.001)
    A['m6_labels'] = dict(classes=m6, tests=tests, n_tests=nt, min_class_size=min_class,
                          class_counts={ln: {names[ln].get(int(c), str(c)) if c >= 0 else 'unknown': int((lv == c).sum())
                                             for c in np.unique(lv)} for ln, lv in labs.items()})

    # ---- m7 cross
    m7 = {}
    for bi, bn in enumerate(BAND_NAMES):
        for v in VIEWS:
            row = dict(word_map_entropy=stat(cross[:, vi[v], bi, 0]), pooled_map_entropy=stat(cross[:, vi[v], bi, 1]),
                       corr_word_vs_pooled=stat(cross[:, vi[v], bi, 2]))
            tabs = table_for(lambda r, v=v, bn=bn: r['cross_tok'].get((v, bn)))
            cnt, sm = tabs
            mean = sm / np.maximum(cnt, 1)
            ok = np.flatnonzero(cnt >= MIN_OCC)
            order = ok[np.argsort(mean[ok])]
            f = [dict(id=int(i), n=int(cnt[i]), mean_entropy=float(mean[i])) for i in order[:20]]
            lf = [dict(id=int(i), n=int(cnt[i]), mean_entropy=float(mean[i])) for i in order[::-1][:20]]
            need_ids |= set(x['id'] for x in f + lf)
            row.update(most_focused=f, least_focused=lf)
            m7[f'{bn}|{v}'] = row
    A['m7_cross'] = m7

    # token strings
    ts = token_strings(sorted(need_ids))
    A['token_strings'] = {str(k): v for k, v in ts.items()}
    return A, dict(shares=shares, shape=shape, pair=pair, pair_sh=pair_sh, add=add, cross=cross, labs=labs)


# ================================================================= per-painting csv
def write_csv(path, res, names, arrs):
    import pandas as pd
    vi = {v: i for i, v in enumerate(VIEWS)}
    rows = []
    for n, r in enumerate(res):
        d = dict(k=r['k'], painting_id=r['painting_id'], n_captions=r['ncap'], emotion=r['emo'],
                 emotion_name=names['emotion'].get(r['emo'], ''), style=r['sty'], style_name=names['style'].get(r['sty'], ''),
                 genre=r['gen'], genre_name=names['genre'].get(r['gen'], 'unknown'),
                 grid_rows=r['grid'][0], grid_cols=r['grid'][1])
        for bi, (bn, a, b) in enumerate(BANDS):
            tag = bn.replace(' ', '_')
            for v in VIEWS:
                d[f'{v}|{tag}|img_share'] = float(arrs['shares'][n, vi[v], 0, a:b, 2].mean())
                d[f'{v}|{tag}|img_entropy'] = float(arrs['shape'][n, vi[v], 0, bi, 0])
            for j, v in enumerate(VIEWS[1:]):
                d[f'{v}|{tag}|l1_to_default'] = float(arrs['pair'][n, j, bi, 0])
                d[f'{v}|{tag}|js_to_default'] = float(arrs['pair'][n, j, bi, 1])
        rows.append(d)
    pd.DataFrame(rows).to_csv(path, index=False, compression='gzip', float_format='%.6g')


# ================================================================= figures
def figures(A, arrs, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    vi = {v: i for i, v in enumerate(VIEWS)}
    cols = dict(zip(VIEWS, ('k', 'tab:red', 'tab:blue', 'tab:green')))
    x = np.arange(N_LAYERS)
    N = A['sanity']['n_paintings']
    for fname, ci, mods, ttl in (('fig_image_share_by_layer.png', 2, ('img', 'joint'), 'image'),
                                 ('fig_caption_share_by_layer.png', 3, ('txt', 'joint'), 'caption')):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for a, m in zip(ax, mods):
            for v in VIEWS:
                a.plot(x, A['m1_layer_shares'][m][v][CATS[ci]]['mean'], color=cols[v], label=v)
            a.axvline(15.5, color='grey', ls=':', lw=1)
            a.set_title(f'Pooled token: share of raw attention on {ttl} tokens\n{m} modality (mean of {N} paintings)', fontsize=9)
            a.set_xlabel('layer (0 to 27)')
        ax[0].set_ylabel(f'{ttl} share')
        ax[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=130)
        plt.close(fig)
    # change from default
    for fname, mi, lab in (('fig_change_l1.png', 0, 'L1 (0 to 2)'), ('fig_change_js.png', 1, 'Jensen-Shannon (bits)'),
                           ('fig_change_jaccard.png', 2, 'top-5% Jaccard')):
        fig, ax = plt.subplots(1, NB, figsize=(13, 4), sharey=True)
        for bi, bn in enumerate(BAND_NAMES):
            a = ax[bi]
            xs = np.arange(len(PAIRS))
            real = [A['m3_view_change'][f'{bn}|{p}']['real'][PMETRICS[mi]] for p in PAIR_NAMES]
            sh = [A['m3_view_change'][f'{bn}|{p}']['shuffled'][PMETRICS[mi]] for p in PAIR_NAMES]
            a.bar(xs - 0.2, [r['mean'] for r in real], 0.4, label='views of one painting',
                  yerr=[[r['mean'] - r['p10'] for r in real], [r['p90'] - r['mean'] for r in real]], capsize=2)
            a.bar(xs + 0.2, [r['mean'] for r in sh], 0.4, label='shuffled baseline', color='lightgrey',
                  yerr=[[r['mean'] - r['p10'] for r in sh], [r['p90'] - r['mean'] for r in sh]], capsize=2)
            if mi == 0:
                a.axhline(NOISE_L1[bn], color='r', ls='--', lw=1, label='bf16 noise floor')
            a.set_xticks(xs)
            a.set_xticklabels([p.replace(' vs ', '\nvs\n').replace('_', ' ') for p in PAIR_NAMES], fontsize=6)
            a.set_title(f'{bn}', fontsize=9)
        ax[0].set_ylabel(f'{lab}, mean with p10 to p90')
        ax[0].legend(fontsize=7)
        fig.suptitle(f'Image maps: difference between instruction views, {N} paintings', fontsize=10)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=130)
        plt.close(fig)
    # shape distributions, default band
    bi = 0
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, j in zip(ax, (0, 1, 5)):
        data, labs_, pos = [], [], []
        for mi, m in enumerate(('img', 'joint')):
            for v in VIEWS:
                data.append(arrs['shape'][:, vi[v], mi, bi, j])
                labs_.append(f'{m}\n{v.replace("_", " ")}')
        a.boxplot([d[np.isfinite(d)] for d in data], showfliers=False)
        a.set_xticks(range(1, len(data) + 1))
        a.set_xticklabels(labs_, fontsize=6)
        a.set_ylabel(SHAPE_MEAS[j].replace('_', ' '))
        a.set_title(f'{SHAPE_MEAS[j]} of the pooled image map ({BAND_NAMES[bi]})', fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'fig_map_shape.png', dpi=130)
    plt.close(fig)
    # img vs joint
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    xs = np.arange(NB)
    for k_, v in enumerate(VIEWS):
        ax[0].bar(xs + (k_ - 1.5) * 0.2, [A['m4_img_vs_joint'][f'{bn}|{v}']['l1']['mean'] for bn in BAND_NAMES], 0.2,
                  color=cols[v], label=v)
        ax[1].bar(xs + (k_ - 1.5) * 0.2, [A['m4_img_vs_joint'][f'{bn}|{v}']['js']['mean'] for bn in BAND_NAMES], 0.2,
                  color=cols[v], label=v)
    ax[0].plot(xs, [NOISE_L1[bn] for bn in BAND_NAMES], 'r_', ms=30, label='bf16 noise floor')
    for a, t in zip(ax, ('L1', 'Jensen-Shannon (bits)')):
        a.set_xticks(xs)
        a.set_xticklabels(BAND_NAMES)
        a.set_ylabel(t)
        a.set_title(f'Image map without vs with the caption (mean of {N} paintings)', fontsize=9)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'fig_img_vs_joint.png', dpi=130)
    plt.close(fig)
    # cross entropy
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k_, v in enumerate(VIEWS):
        ax[0].bar(xs + (k_ - 1.5) * 0.2, [A['m7_cross'][f'{bn}|{v}']['word_map_entropy']['mean'] for bn in BAND_NAMES], 0.2,
                  color=cols[v], label=v)
        ax[1].bar(xs + (k_ - 1.5) * 0.2, [A['m7_cross'][f'{bn}|{v}']['corr_word_vs_pooled']['mean'] for bn in BAND_NAMES],
                  0.2, color=cols[v], label=v)
    ax[0].plot(xs, [np.mean([A['m7_cross'][f'{bn}|{v}']['pooled_map_entropy']['mean'] for v in VIEWS]) for bn in BAND_NAMES],
               'k_', ms=30, label='pooled map (mean of views)')
    ax[0].set_ylabel('mean normalised entropy of a caption-word map')
    ax[1].set_ylabel('mean Pearson correlation, word map vs pooled map')
    for a in ax:
        a.set_xticks(xs)
        a.set_xticklabels(BAND_NAMES)
        a.set_title(f'Joint cross maps over the image ({N} paintings)', fontsize=9)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'fig_cross.png', dpi=130)
    plt.close(fig)


# ================================================================= markdown
def f3(x):
    return '' if x is None or not np.isfinite(x) else f'{x:.3f}'


def md_table(head, rows):
    s = '| ' + ' | '.join(head) + ' |\n|' + '|'.join('---' for _ in head) + '|\n'
    for r in rows:
        s += '| ' + ' | '.join(str(c) for c in r) + ' |\n'
    return s + '\n'


def sst(d):
    return [f3(d['mean']), f3(d['median']), f3(d['p10']), f3(d['p90'])]


def write_md(A, names, names_src, out, root, t_run):
    ts = A['token_strings']

    def tk(i):
        return ts[str(i)].replace('|', '\\|')
    N = A['sanity']['n_paintings']
    L = [f'# Attention-map analysis\n\nRoot: `{root}`. Paintings: {N}. Written {time.strftime("%Y-%m-%d %H:%M")}.\n']
    L.append('Maps are the pooled last token\'s raw head-mean attention summed over a band of layers. Bands: '
             + ', '.join(BAND_NAMES) + ' (default: layers 16-27). Statistics are over paintings: mean, median, p10, p90.\n')
    L.append('Noise floor: bf16 rounding alone moves a map by L1 about ' + ', '.join(
        f'{NOISE_L1[b]:.2f} ({b})' for b in BAND_NAMES) + ' against float32.\n')
    L.append(f'Label names: {names_src}. Max |share sum - 1| over all layer rows: '
             f'{A["sanity"]["max_abs_share_sum_minus_1"]:.2e}. Caption-word cross rows dropped (zero sum): '
             f'{A["sanity"]["cross_rows_dropped"]:.0f}.\n')
    # m1
    L.append('## 1. Shares of the pooled token\'s attention\n')
    L.append('Mean over paintings of the share on each token type per layer (txt and joint: mean over a painting\'s captions first). '
             'Median, p10 and p90 per layer are in analysis.json (m1_layer_shares). Figures: fig_image_share_by_layer.png, '
             'fig_caption_share_by_layer.png.\n')
    L.append('### Band summary (mean over the band\'s layers, then statistics over paintings)\n')
    for ci, c in enumerate(CATS):
        rows = []
        for m in ('img', 'txt', 'joint'):
            if (m == 'img' and c == 'caption') or (m == 'txt' and c == 'image'):
                continue
            for v in VIEWS:
                for bn in BAND_NAMES:
                    rows.append([m, v, bn] + sst(A['m1_band_shares'][f'{m}|{v}|{bn}'][c]))
        L.append(f'Share on {c}.\n')
        L.append(md_table(['modality', 'view', 'band', 'mean', 'median', 'p10', 'p90'], rows))
    L.append('### Mean share by layer\n')
    for m in ('img', 'txt', 'joint'):
        for v in VIEWS:
            rows = [[l] + [f3(A['m1_layer_shares'][m][v][c]['mean'][l]) for c in CATS] for l in range(N_LAYERS)]
            L.append(f'{m}, view {v}: mean share by layer.\n')
            L.append(md_table(['layer'] + list(CATS), rows))
    # m2
    L.append('## 2. Shape of the image map\n')
    L.append('Normalised entropy (1 = flat), mass on the top 1% of tokens, spread (mass-weighted standard deviation of row / '
             'column position divided by grid rows / cols), vertical centre (0 top, 1 bottom), border excess (mass on the outer '
             'ring divided by the ring\'s share of tokens), and the fraction of maps whose hottest token is on the ring. '
             'Joint: per caption, then mean over a painting\'s captions. Figure: fig_map_shape.png.\n')
    for bn in BAND_NAMES:
        rows = []
        for m in ('img', 'joint'):
            for v in VIEWS:
                for me in SHAPE_MEAS:
                    rows.append([m, v, me] + sst(A['m2_shape'][f'{bn}|{m}|{v}'][me]))
        L.append(f'Band: {bn}.\n')
        L.append(md_table(['modality', 'view', 'measure', 'mean', 'median', 'p10', 'p90'], rows))
    # m3
    L.append('## 3. Change from the default view (img modality)\n')
    L.append('Per painting, L1 (0 to 2), Jensen-Shannon divergence (base 2, 0 to 1) and Jaccard overlap of the top-5% tokens '
             'between the maps of two views. "Shuffled" compares view A\'s map with a random permutation of view B\'s map of the '
             'same painting (seed 0). The bf16 noise floor for L1 is given in each caption. Figures: fig_change_*.png.\n')
    for bi, bn in enumerate(BAND_NAMES):
        for me, nm in zip(PMETRICS, ('L1', 'Jensen-Shannon (bits)', 'top-5% Jaccard')):
            rows = []
            for pn in PAIR_NAMES:
                e = A['m3_view_change'][f'{bn}|{pn}']
                rows.append([pn] + sst(e['real'][me]) + [f3(e['shuffled'][me]['mean']), f3(e['shuffled'][me]['median'])])
            cap = f'Band {bn}, {nm}.' + (f' bf16 noise floor L1 about {NOISE_L1[bn]:.2f}.' if me == 'l1' else '')
            L.append(cap + '\n')
            L.append(md_table(['pair', 'mean', 'median', 'p10', 'p90', 'shuffled mean', 'shuffled median'], rows))
    # m4
    L.append('## 4. Adding the caption\n')
    L.append('Per view: L1 and Jensen-Shannon between the img map and the joint pooled image map (mean over a painting\'s '
             'captions), and how the joint pass splits the pooled token\'s attention (mean over the band\'s layers). '
             'Figure: fig_img_vs_joint.png.\n')
    for bn in BAND_NAMES:
        rows = []
        for v in VIEWS:
            e = A['m4_img_vs_joint'][f'{bn}|{v}']
            rows.append([v] + sst(e['l1']) + sst(e['js']))
        L.append(f'Band {bn}: img map against joint image map. bf16 noise floor L1 about {NOISE_L1[bn]:.2f}.\n')
        L.append(md_table(['view', 'L1 mean', 'median', 'p10', 'p90', 'JS mean', 'median', 'p10', 'p90'], rows))
        rows = [[v] + [f3(A['m4_img_vs_joint'][f'{bn}|{v}']['joint_share'][c]['mean']) for c in CATS] for v in VIEWS]
        L.append(f'Band {bn}: joint pass, mean share on each token type.\n')
        L.append(md_table(['view'] + list(CATS), rows))
    # m5
    L.append('## 5. Caption words\n')
    L.append(f'Weight of a caption token = the pooled token\'s attention on it, renormalised over the caption\'s tokens, '
             f'averaged per occurrence. Tokens with at least {MIN_OCC} occurrences. "Top" = highest mean weight; "Rise" = '
             'weight in the instructed view minus weight in the default view.\n')
    for bn in BAND_NAMES:
        for m in ('txt', 'joint'):
            e = A['m5_words'][f'{bn}|{m}']
            rows = []
            for i in range(30):
                rows.append([i + 1] + [f'{tk(e[v]["top"][i]["id"])} ({e[v]["top"][i]["mean_weight"]:.3f}, n={e[v]["top"][i]["n"]})'
                                       if i < len(e[v]['top']) else '' for v in VIEWS])
            L.append(f'Band {bn}, {m}: 30 caption tokens with the highest mean weight (weight, occurrences).\n')
            L.append(md_table(['rank'] + list(VIEWS), rows))
            rows = []
            for i in range(20):
                rows.append([i + 1] + [f'{tk(e[v]["rise_vs_default"][i]["id"])} (+{e[v]["rise_vs_default"][i]["rise"]:.3f}, '
                                       f'n={e[v]["rise_vs_default"][i]["n"]})' if i < len(e[v]['rise_vs_default']) else ''
                                       for v in VIEWS[1:]])
            L.append(f'Band {bn}, {m}: 20 tokens whose weight rises most against the default view (rise, occurrences).\n')
            L.append(md_table(['rank'] + list(VIEWS[1:]), rows))
    # m6
    L.append('## 6. Labels\n')
    m6 = A['m6_labels']
    L.append(f'Painting-level classes: style and genre are constant per painting; emotion is the majority of a painting\'s '
             f'captions (ties: lowest code). Genre -1 (unknown) is left out of the class tables and tests. Classes with at '
             f'least {m6["min_class_size"]} paintings. Class sizes (all classes):\n')
    for ln, cc in m6['class_counts'].items():
        L.append(f'- {ln}: ' + ', '.join(f'{k} {v}' for k, v in cc.items()) + '\n')
    L.append('\n')
    for ln in ('emotion', 'style', 'genre'):
        for bn in BAND_NAMES:
            cl = m6['classes'][f'{ln}|{bn}']
            if not cl:
                L.append(f'{ln}, band {bn}: no class reaches the minimum size.\n\n')
                continue
            rows = [[c, e['n']] + [f3(e['img_share'][v]) for v in VIEWS] + [f3(e['l1_to_default'][v]) for v in VIEWS[1:]]
                    for c, e in cl.items()]
            L.append(f'{ln}, band {bn}: mean image share per view (img modality) and mean L1 to the default view.\n')
            L.append(md_table(['class', 'n'] + [f'share {v}' for v in VIEWS] + [f'L1 {v}' for v in VIEWS[1:]], rows))
    L.append(f'Kruskal-Wallis tests across classes: {m6["n_tests"]} tests; Bonferroni-adjusted p = min(1, p x {m6["n_tests"]}); '
             'flag = adjusted p < 0.001.\n')
    rows = [[t['label'], t['band'], t['view'], t['metric'], f'{t["H"]:.2f}', f'{t["p"]:.2e}', f'{t["p_bonferroni"]:.2e}',
             t['n_classes'], 'yes' if t['flag_p_bonf_below_0.001'] else ''] for t in m6['tests']]
    L.append(md_table(['label', 'band', 'view', 'metric', 'H', 'p', 'p Bonferroni', 'classes', 'flag'], rows))
    # m7
    L.append('## 7. Cross maps (joint)\n')
    L.append('Each caption token\'s raw attention over the image tokens, summed over the band, renormalised ("word map"). '
             'Entropy is normalised; correlation is Pearson between a word map and the pooled image map of the same caption '
             '(mean over words, then over paintings). Figure: fig_cross.png.\n')
    rows = []
    for bn in BAND_NAMES:
        for v in VIEWS:
            e = A['m7_cross'][f'{bn}|{v}']
            rows.append([bn, v, f3(e['word_map_entropy']['mean']), f3(e['word_map_entropy']['median']),
                         f3(e['pooled_map_entropy']['mean']), f3(e['pooled_map_entropy']['median']),
                         f3(e['corr_word_vs_pooled']['mean']), f3(e['corr_word_vs_pooled']['median']),
                         f3(e['corr_word_vs_pooled']['p10']), f3(e['corr_word_vs_pooled']['p90'])])
    L.append('Word-map entropy against pooled-map entropy, and word-pooled correlation.\n')
    L.append(md_table(['band', 'view', 'word entropy mean', 'median', 'pooled entropy mean', 'median', 'corr mean', 'median',
                       'p10', 'p90'], rows))
    for bn in BAND_NAMES:
        for v in VIEWS:
            e = A['m7_cross'][f'{bn}|{v}']
            rows = [[i + 1,
                     (f'{tk(e["most_focused"][i]["id"])} ({e["most_focused"][i]["mean_entropy"]:.3f}, n={e["most_focused"][i]["n"]})'
                      if i < len(e['most_focused']) else ''),
                     (f'{tk(e["least_focused"][i]["id"])} ({e["least_focused"][i]["mean_entropy"]:.3f}, n={e["least_focused"][i]["n"]})'
                      if i < len(e['least_focused']) else '')] for i in range(20)]
            L.append(f'Band {bn}, view {v}: 20 most and 20 least focused caption tokens (mean normalised entropy, occurrences).\n')
            L.append(md_table(['rank', 'most focused', 'least focused'], rows))
    L.append(f'\nRun time: {t_run:.0f} s.\n')
    (out / 'analysis.md').write_text('\n'.join(L))


# ================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--min-class', type=int, default=30)
    a = ap.parse_args()
    root = Path(a.root)
    out = Path(a.out) if a.out else HERE / 'outputs' / root.name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ks = sorted(int(p.stem[1:]) for p in root.glob('p[0-9][0-9][0-9][0-9].npz'))
    if a.limit:
        ks = ks[:a.limit]
    print(f'{len(ks)} painting files in {root}', flush=True)
    res, bad = [], []
    workers = max(1, min(a.workers, 8))
    with Pool(workers) as pool:
        for n, r in enumerate(pool.imap(analyse_painting, [(str(root), k) for k in ks], chunksize=4), 1):
            (bad if 'error' in r else res).append(r)
            if n % 500 == 0 or n == len(ks):
                print(f'{n}/{len(ks)} paintings, {time.time() - t0:.0f} s', flush=True)
    print(f'{len(res)} analysed, {len(bad)} skipped', flush=True)
    for b in bad[:10]:
        print('skipped', b, flush=True)
    names, src = label_names()
    A, arrs = aggregate(res, names, a.min_class, None)
    A['skipped'] = [dict(k=b['k'], error=b['error']) for b in bad]
    A['meta'] = dict(root=str(root), bands=BAND_NAMES, views=list(VIEWS), label_names=names, label_names_source=src,
                     shuffle_seed=SEED, min_occurrences=MIN_OCC)
    (out / 'analysis.json').write_text(json.dumps(clean(A)))
    write_csv(out / 'per_painting.csv.gz', res, names, arrs)
    figures(A, arrs, out)
    write_md(A, names, src, out, root, time.time() - t0)
    print(f'done in {time.time() - t0:.0f} s; outputs in {out}', flush=True)


if __name__ == '__main__':
    main()
