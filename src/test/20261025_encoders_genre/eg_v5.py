"""V5 (rule §4.1): validity check of the image stores of the encoders.

    python eg_v5.py --calibrate                          # CLIP B/32 and SigLIP 2, both datasets (read only)
    python eg_v5.py --store clip_b16 --dataset artelingo_full [--dataset artelingo28]
Writes <outputs>/stores/v5.json (merged by key; the file is rewritten whole, atomically):
  calibration {clip_b32, siglip2_so400m: top-1 on the full split}, calibration_a28 (same, descriptive),
  bar_void (either calibration < 1%), stores {"<dataset>/<store>": {pass, checks, ...}}. Exit 1 on a failed store.

Structure (all stores): row count, unique names equal in order to the reference (clip_b32 for the full split, the
rb_encoders A28 list for ArtELingo-28), finite rows of the stated width, nonzero rows, not a smoke store.
Content, text-bearing new stores (clip_b16, clip_l14, siglip_so400m): top-1 retrieval image -> own caption mean among
the paintings, pass at >= 1% unless the bar is void. Full split: the first 1,000 paintings of P in sorted order.
ArtELingo-28: all 746 paintings (P is a full-split list; the 746 are the whole set).
DINOv2 (dinov2_large, full split): per-painting cosine to the archived store on the paintings both hold, every
cosine >= 0.99. On ArtELingo-28 it has the structure checks only (no archived store).
The calibration stores get the structure checks; their content number defines the bar and decides nothing.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import eg_paths
import eg_encoders
import rb_data
import rb_encoders

FULL, A28 = 'artelingo_full', 'artelingo28'
COUNT = {FULL: 61402, A28: 746}
CALIBRATION = ('clip_b32', 'siglip2_so400m')
BAR = 0.01
N_P = 1000
DINO_MIN_COS = 0.99
REFERENCE_FULL = 'clip_b32'
STYLE_CSD_META = eg_paths.PRE / 'artelingo/style_csd_vitl/meta.json'   # image_paths: the archived DINOv2 row order


def stem(path):
    return path.rsplit('/', 1)[-1].rsplit('.', 1)[0]


def index_of(dataset, store):
    return json.loads((eg_paths.PRE / dataset / store / 'index.json').read_text())


def width_of(store):
    return eg_encoders.BY_STORE[store].width if store in eg_encoders.BY_STORE else rb_encoders.WIDTH[store]


def reference_names(dataset):
    if dataset == FULL:
        idx = index_of(FULL, REFERENCE_FULL)
        names = [r['painting'] for r in idx['images']]
        assert len(names) == COUNT[FULL] == len(set(names))
        return names
    return rb_encoders.a28_names()


def structure(dataset, store):
    """The structural checks; returns (checks dict, names, normalized float32 image rows or None)."""
    path = eg_paths.PRE / dataset / store
    idx = index_of(dataset, store)
    names = [r['painting'] for r in idx['images']]
    width = width_of(store)
    c = {}
    c['not_smoke'] = (not idx['smoke']) and idx['backbone'] == store and idx['dataset'] == dataset
    c['rows'] = idx['n_images'] == len(names) == COUNT[dataset]
    c['names_unique'] = len(set(names)) == len(names)
    c['names_equal_reference_in_order'] = names == reference_names(dataset)
    raw = np.load(path / 'img.npy', mmap_mode='r')
    c['width'] = idx['dim'] == width and raw.shape == (COUNT[dataset], width) and raw.dtype == np.float16
    finite, nonzero, x = True, True, None
    if c['width'] and c['rows']:
        x = np.empty(raw.shape, dtype=np.float32)
        for i in range(0, len(raw), 8192):
            block = np.asarray(raw[i:i + 8192], dtype=np.float32)
            finite &= bool(np.isfinite(block).all())
            norm = np.linalg.norm(block, axis=1, keepdims=True)
            nonzero &= bool((norm > 0).all())
            x[i:i + 8192] = block / np.where(norm > 0, norm, 1)
    c['finite'] = finite
    c['nonzero'] = nonzero
    return c, names, x


def caption_means(dataset, store, wanted, names):
    """Unit-norm mean of each wanted painting's text rows [len(wanted), width]; texts joined by painting name."""
    path = eg_paths.PRE / dataset / store
    idx = index_of(dataset, store)
    texts = idx['texts']
    txt = np.load(path / 'txt.npy', mmap_mode='r')
    assert txt.shape == (idx['n_texts'], idx['dim']) == (len(texts), width_of(store)), (path, txt.shape)
    row_of = {n: i for i, n in enumerate(names)}
    rows = {n: [] for n in wanted}
    for k, t in enumerate(texts):
        assert names[t['img']] == t['painting'], f'{path}: text {k} img index and painting differ'
        if t['painting'] in rows:
            rows[t['painting']].append(k)
    assert all(rows.values()), f'{path}: a painting has no text rows'
    out = np.empty((len(wanted), idx['dim']), dtype=np.float32)
    for i, n in enumerate(wanted):
        t = np.asarray(txt[np.asarray(rows[n])], dtype=np.float32)
        assert np.isfinite(t).all()
        t = t / np.linalg.norm(t, axis=1, keepdims=True)
        m = t.mean(axis=0)
        out[i] = m / np.linalg.norm(m)
    del row_of
    return out


def retrieval(dataset, store, x, names):
    """Top-1 image -> own caption mean among the chosen paintings; returns (top1, n)."""
    if dataset == FULL:
        wanted = rb_data.p_names()[:N_P]
        assert wanted == sorted(wanted) and len(wanted) == N_P
    else:
        wanted = list(names)
    pos = {n: i for i, n in enumerate(names)}
    assert all(n in pos for n in wanted), 'a chosen painting is missing from the store'
    img = x[np.asarray([pos[n] for n in wanted])]
    cap = caption_means(dataset, store, wanted, names)
    hit = (img @ cap.T).argmax(axis=1) == np.arange(len(wanted))
    return float(hit.mean()), len(wanted)


def dino_archive():
    """The archived DINOv2-large store: painting names (via the CSD image paths) and float32 rows."""
    meta = json.loads((eg_paths.DINO_ARCHIVE / 'meta.json').read_text())
    emb = np.load(eg_paths.DINO_ARCHIVE / 'embeddings.npy', mmap_mode='r')
    index = np.load(eg_paths.DINO_ARCHIVE / 'image_index.npy')
    paths = json.loads(STYLE_CSD_META.read_text())['images']['image_paths']
    assert emb.shape == (meta['n'], meta['dim']) == (len(index), 1024) and not meta['smoke'], emb.shape
    assert len(paths) == COUNT[FULL] and 0 <= index.min() and index.max() < len(paths)
    names = [stem(paths[j]) for j in index]
    assert len(set(names)) == len(names)
    return names, emb


def dino_check(x, names):
    arch_names, emb = dino_archive()
    pos = {n: i for i, n in enumerate(names)}
    keep = [k for k, n in enumerate(arch_names) if n in pos]
    cos = np.empty(len(keep), dtype=np.float32)
    for s in range(0, len(keep), 8192):
        k = np.asarray(keep[s:s + 8192])
        a = np.asarray(emb[k], dtype=np.float32)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b = x[np.asarray([pos[arch_names[i]] for i in k])]
        cos[s:s + len(k)] = (a * b).sum(axis=1)
    return {'min_cosine': float(cos.min()), 'mean_cosine': float(cos.mean()), 'n': len(keep),
            'archive_n': len(arch_names)}


def check_store(dataset, store, bar_void, calibration_store=False):
    """One store's result: {pass, checks, ...}. The content number of a calibration store decides nothing."""
    checks, names, x = structure(dataset, store)
    out = {'checks': checks}
    ok = all(checks.values())
    enc = eg_encoders.BY_STORE.get(store)
    if ok and store == 'dinov2_large' and dataset == FULL:
        d = dino_check(x, names)
        out['dino_vs_archive'] = d
        checks['dino_cosine_ge_0.99'] = d['min_cosine'] >= DINO_MIN_COS and d['n'] == d['archive_n']
    elif ok and (calibration_store or (enc is not None and enc.texts)):
        top1, n = retrieval(dataset, store, x, names)
        out['retrieval'] = {'top1': top1, 'n_paintings': n, 'chance': 1.0 / n}
        if not calibration_store:
            out['bar_void'] = bool(bar_void)
            checks['retrieval_ge_1pct'] = bool(bar_void) or top1 >= BAR
    out['pass'] = all(checks.values())
    return out


def git_head():
    here = Path(__file__).resolve().parent
    return subprocess.run(['git', '-C', str(here), 'rev-parse', 'HEAD'], check=True, capture_output=True,
                          text=True).stdout.strip()


def load(path):
    return json.loads(path.read_text()) if path.exists() else {'stores': {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs', type=Path, default=eg_paths.OUTPUTS)
    ap.add_argument('--store', choices=sorted(eg_encoders.BY_STORE))
    ap.add_argument('--dataset', action='append', choices=(FULL, A28))
    ap.add_argument('--calibrate', action='store_true')
    a = ap.parse_args()
    assert a.calibrate or (a.store and a.dataset), 'give --calibrate, or --store with --dataset'
    eg_paths.check_imports()
    path = a.outputs / 'stores/v5.json'
    res = load(path)
    res.setdefault('stores', {})
    failed = []
    res['code_commit'] = git_head()
    res['rule_commit'] = eg_paths.RULE_COMMIT

    def save():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
        with os.fdopen(fd, 'w') as f:
            json.dump(res, f, indent=1, sort_keys=True)
        os.replace(tmp, path)

    def checked(dataset, store, **kw):
        try:
            return check_store(dataset, store, **kw)
        except Exception as err:
            return {'pass': False, 'checks': {}, 'cause': f'{type(err).__name__}: {err}'}

    def record(dataset, store, r):
        if not r['pass']:
            r.setdefault('cause', ', '.join(k for k, ok in r.get('checks', {}).items() if not ok) or 'check failed')
        res['stores'][f'{dataset}/{store}'] = r
        save()
        print(f'{dataset}/{store}: pass={r["pass"]} {json.dumps(r.get("retrieval", r.get("dino_vs_archive", "")))}',
              flush=True)
        if not r['pass']:
            failed.append(f'{dataset}/{store}')

    if a.calibrate:
        res.setdefault('calibration', {})
        res.setdefault('calibration_a28', {})
        for dataset in (FULL, A28):
            for store in CALIBRATION:
                r = checked(dataset, store, bar_void=False, calibration_store=True)
                if 'retrieval' in r:
                    (res['calibration'] if dataset == FULL else res['calibration_a28'])[store] = r['retrieval']['top1']
                record(dataset, store, r)
    if 'calibration' not in res or any(s not in res['calibration'] for s in CALIBRATION):
        raise SystemExit('no calibration in v5.json: run --calibrate first')
    res['bar_void'] = any(res['calibration'][s] < BAR for s in CALIBRATION)
    if a.store:
        # A hard kill between two datasets must not leave the old block of the dataset in flight (the stores queue
        # skips V5 when both old blocks pass): drop this store's blocks first, then check and save per dataset.
        for dataset in a.dataset:
            res['stores'].pop(f'{dataset}/{a.store}', None)
        save()
        for dataset in a.dataset:
            r = checked(dataset, a.store, bar_void=res['bar_void'])
            if dataset == A28 and not r['pass']:
                full = res['stores'].get(f'{FULL}/{a.store}')
                r['fails_on_28_only'] = bool(full and full['pass'])   # removes only the A outcome
            record(dataset, a.store, r)
    save()
    print(f'bar_void={res["bar_void"]} calibration={res["calibration"]} -> {path}')
    if failed:
        print('FAILED:', failed)
        sys.exit(1)


if __name__ == '__main__':
    main()
