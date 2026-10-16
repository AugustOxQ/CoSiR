"""Shared loading for the aspect premise checks: ArtELingo train side, painting-subsampled (seed 0, <=100k captions)."""
import sys, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/project/CoSiR")
from src.data.artelingo import load_artelingo
from src.data.artelingo_splits import artelingo_splits, encode_labels
from src.data.wikiart_genre import load_wikiart_genre

HERE = Path(__file__).parent
MAX_CAPTIONS = 100_000
STYLE_ROOT = Path("/data/SSD2/pre_extract/artelingo")


def l2(x, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def load_sample(min_caps=1, max_captions=MAX_CAPTIONS, seed=0, with_splits=False):
    """Return dict with row indices (into load_artelingo order) of the sampled train captions plus labels."""
    t0 = time.time()
    d = load_artelingo()
    sp = artelingo_splits(d)
    train = np.sort(np.concatenate([sp.scorer_train, sp.selection]))
    paint_all = np.asarray(d.paintings)
    pt = paint_all[train]
    uniq, inv, cnt = np.unique(pt, return_inverse=True, return_counts=True)
    keep_p = np.flatnonzero(cnt >= min_caps)
    rng = np.random.default_rng(seed)
    order = rng.permutation(keep_p)
    cum = np.cumsum(cnt[order])
    chosen = order[cum <= max_captions]
    mask = np.isin(inv, chosen)
    rows = train[mask]
    pid = np.unique(inv[mask], return_inverse=True)[1]  # dense painting ids 0..P-1 for sampled captions
    emo, emo_names = encode_labels(d.emotions)  # all 9 classes incl. 'something else'
    sty, sty_names = encode_labels(d.art_styles)
    gen = load_wikiart_genre(paint_all.astype(object))
    out = dict(d=d, sp=sp, rows=rows, pid=pid, emo=emo[rows], sty=sty[rows], gen=gen[rows],
               emo_names=emo_names, sty_names=sty_names, n_paint=int(pid.max()) + 1,
               t=l2(d.txt_features[rows]), v=l2(d.img_features[rows]), sample_ids=d.sample_ids[rows])
    print(f"[load_sample] {len(rows)} captions, {out['n_paint']} paintings, {time.time()-t0:.0f}s", flush=True)
    return out


def paint_majority(labels, pid, n_paint, n_cls=None, ignore=-1):
    """Majority label per painting (ties -> smallest id); ignore label counted out; -1 if none."""
    n_cls = int(labels.max()) + 1 if n_cls is None else n_cls
    m = labels != ignore
    cnt = np.zeros((n_paint, n_cls), dtype=np.int64)
    np.add.at(cnt, (pid[m], labels[m]), 1)
    res = cnt.argmax(1)
    res[cnt.sum(1) == 0] = -1
    return res


def paint_first(labels, pid, n_paint):
    """Per-painting constant label (first caption's)."""
    res = np.full(n_paint, -2, dtype=np.int64)
    res[pid[::-1]] = labels[::-1]
    return res


def paint_mean(x, pid, n_paint):
    s = np.zeros((n_paint, x.shape[1]), dtype=np.float64)
    np.add.at(s, pid, x)
    c = np.bincount(pid, minlength=n_paint)[:, None]
    return (s / c).astype(np.float32)
