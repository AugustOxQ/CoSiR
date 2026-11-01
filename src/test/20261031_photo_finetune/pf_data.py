"""Affection photo data provider of r13 (spec 2026-10-31-photo-finetune-design.md §3.3, §3.4 training inputs).

Keys are `<source>/<image_name>`; every join is by key (C6), explanation rows are joined by `csv_row`. Train labels are
never read: from the default store's index.json only `img`, `csv_row`, `split` of the text rows and `source`,
`image_name`, `splits` of the image rows are used (the stored `emotion` and `workerid` are never touched). Val and test
keys are r12's (`ac_data.keys`); train keys follow r12's rule (an image's split is its one non-ignored row split) read
from the index instead of the annotation CSV, so that no train annotation (label) is read.
"""
import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import pf_paths
import ac_data
import ac_paths
import rs_vh
import rt_data

CODE = {'train': 0, 'val': 1, 'test': 2, 'ignored': 3}
IGNORED = CODE['ignored']
WIDTH = 2048


@lru_cache(maxsize=2)
def _index(path, mtime_ns, size):
    """The projected default-store index: image keys and splits, and per text row `img`, `csv_row`, `split`."""
    del mtime_ns, size   # cache identity
    side = json.loads(Path(path).read_text())
    images, texts = side['images'], side['texts']
    assert len(images) == side['n_images'] and len(texts) == side['n_texts'], 'index counts'
    keys = [f"{r['source']}/{r['image_name']}" for r in images]
    assert len(set(keys)) == len(keys), 'duplicate image keys'
    t_img = np.fromiter((t['img'] for t in texts), np.int64, len(texts))
    t_csv = np.fromiter((t['csv_row'] for t in texts), np.int64, len(texts))
    t_code = np.fromiter((CODE[t['split']] for t in texts), np.int8, len(texts))
    assert ((t_img >= 0) & (t_img < len(images))).all(), 'text image link'
    # r12's rule: an image has exactly one non-ignored split (D3c: no image has only ignored rows).
    lo, hi = np.full(len(images), 9, np.int8), np.full(len(images), -1, np.int8)
    live = t_code != IGNORED
    np.minimum.at(lo, t_img[live], t_code[live])
    np.maximum.at(hi, t_img[live], t_code[live])
    assert (lo == hi).all(), 'every image needs exactly one non-ignored split'
    names = {v: k for k, v in CODE.items()}
    assert all(names[int(c)] in r['splits'] for c, r in zip(lo, images)), 'image splits list'
    assert (np.bincount(t_img, minlength=len(images)) == [r['n_texts'] for r in images]).all(), 'image n_texts'
    return SimpleNamespace(keys=keys, pos={k: i for i, k in enumerate(keys)}, split=lo,
                           t_img=t_img, t_csv=t_csv, t_code=t_code)


def _ix():
    assert pf_paths.STORE == ac_paths.STORE, 'pf_paths and ac_paths must name the same store'
    path = Path(pf_paths.STORE) / 'index.json'
    stat = path.stat()
    return _index(path, stat.st_mtime_ns, stat.st_size)


def keys(split):
    """Sorted `<source>/<image_name>` keys of a split. Train: smoke (PF_SMOKE=N) gives the first N."""
    assert split in pf_paths.SPLITS, split
    ix = _ix()
    in_index = sorted(k for k, c in zip(ix.keys, ix.split) if c == CODE[split])
    if split != 'train':
        result = ac_data.keys(split)
        assert result == in_index, f'{split} keys: r12 annotations and index disagree'
        return result
    assert len(in_index) == pf_paths.N['train'], 'train key count'
    n = pf_paths.SMOKE
    assert n <= len(in_index), f'PF_SMOKE {n} exceeds the train keys'
    return in_index[:n] if n > 0 else in_index


def text_rows(split):
    """Every explanation row of the split's photos, in default-index order (train or val).

    Returns `key` (list[str]), `csv_row`, `store_row` (the row in the default store's txt.npy; int64), `eligible`
    (bool: the row's own split is not "ignored").
    """
    assert split in ('train', 'val'), split
    ix = _ix()
    member = np.zeros(len(ix.keys), dtype=bool)
    member[[ix.pos[k] for k in keys(split)]] = True
    rows = np.flatnonzero(member[ix.t_img])
    code = ix.t_code[rows]
    assert ((code == CODE[split]) | (code == IGNORED)).all(), 'a row of this split\'s photos has another split'
    result = dict(key=[ix.keys[i] for i in ix.t_img[rows]], csv_row=ix.t_csv[rows], store_row=rows.astype(np.int64),
                  eligible=code != IGNORED)
    assert len(set(result['csv_row'].tolist())) == len(rows), 'csv_row must be unique'
    return result


def _unit(x):
    return ac_data._unit(x)


def default_images(requested):
    """Unit float32 default-store image rows for any keys (train, val or test), joined by key (C6)."""
    requested = list(requested)
    return _unit(ac_data._plain(requested))


def _eligible_rows(requested):
    """Eligible explanation rows of the requested train/val photos: key, csv_row, store_row (default-index order)."""
    wanted = set(requested)
    assert len(wanted) == len(requested), 'duplicate keys'
    out = dict(key=[], csv_row=[], store_row=[])
    known = set()
    for split in ('train', 'val'):
        rows = text_rows(split)
        known.update(keys(split))
        take = [i for i, (k, e) in enumerate(zip(rows['key'], rows['eligible'])) if e and k in wanted]
        out['key'] += [rows['key'][i] for i in take]
        out['csv_row'].append(rows['csv_row'][take])
        out['store_row'].append(rows['store_row'][take])
    unknown = sorted(wanted - known)
    assert not unknown, f'unknown key (not a train or val photo): {unknown[:3]}'
    return dict(key=out['key'], csv_row=np.concatenate(out['csv_row']), store_row=np.concatenate(out['store_row']))


def _group_mean(requested, row_keys, row_pos, source, chunk=4096):
    """Per requested key: raw float16 rows `source[row_pos]` summed in float64 in row order, divided by the row count,
    cast float32 (ct2_data.paint_mean's arithmetic, chunked), then L2 (ct2_data.l2). Float32 [n, 2048], unit."""
    index = {k: i for i, k in enumerate(requested)}
    kidx = np.asarray([index[k] for k in row_keys], dtype=np.int64)
    count = np.bincount(kidx, minlength=len(requested))
    assert (count > 0).all(), f'a key has no eligible row: {[requested[i] for i in np.flatnonzero(count == 0)[:3]]}'
    total = np.zeros((len(requested), WIDTH), dtype=np.float64)
    row_pos = np.asarray(row_pos, dtype=np.int64)
    for a in range(0, len(kidx), chunk):
        raw = source[row_pos[a:a + chunk]]
        assert raw.dtype == np.float16 and raw.shape[1] == WIDTH, 'explanation source dtype/shape'
        np.add.at(total, kidx[a:a + chunk], raw.astype(np.float64))
    mean = (total / count[:, None]).astype(np.float32)
    return ac_data._array(rt_data.ct2_data.l2(mean), len(requested), WIDTH, np.float32, unit=True)


def _default_texts(requested):
    """D: default explanation means, from the default store's txt.npy by `store_row`."""
    requested = list(requested)
    rows = _eligible_rows(requested)
    txt = np.load(Path(pf_paths.STORE) / 'txt.npy', mmap_mode='r', allow_pickle=False)
    assert txt.dtype == np.float16 and txt.shape[1] == WIDTH and txt.shape[0] == len(_ix().t_img), 'default txt.npy'
    return _group_mean(requested, rows['key'], rows['store_row'], txt)


def _check_meta(side, required, count_shape):
    """The meta gate of a new view (adapted from ac_data._view): identity, extraction gate and shape."""
    assert set(required) <= set(side), f'missing metadata fields: {sorted(set(required) - set(side))}'
    gate = side['gate']
    assert isinstance(gate, dict) and gate.get('ok') is True, 'view extraction gate'
    assert {'ok', 'n_nonfinite', 'n_bad_norm', 'max_norm_dev'} <= set(gate), 'view gate fields'
    assert all(type(gate[f]) is int and gate[f] == 0 for f in ('n_nonfinite', 'n_bad_norm')), 'view gate counts'
    dev = gate['max_norm_dev']
    assert isinstance(dev, (int, float)) and np.isfinite(dev) and 0 <= dev <= 1e-3, 'view gate norm deviation'
    assert side['smoke'] is (pf_paths.SMOKE > 0), 'view smoke flag'
    assert side['dtype'] == 'float16' and side['shape'] == list(count_shape), 'view shape/dtype'
    assert side['model_id'] == 'Qwen/Qwen3-VL-Embedding-2B' and \
        side['model_revision'] == rt_data.ct2_data.QWEN_REVISION, 'view encoder'
    # D8: the artwork wording (c*'s own instruction) for both images and explanations.
    assert side['instruction'] == rt_data.ct2_data.itp_data.PROMPTS['emotion_focus'], 'view instruction'


def _train_view():
    """TRAIN_IMG_VIEW checked: (keys.json list, float16 [n, 2048] unit array)."""
    path = Path(pf_paths.TRAIN_IMG_VIEW)
    side = ac_data._json(path / 'meta.json', ('keys_sha256', 'img_sha256', 'shape'))
    stored = json.loads((path / 'keys.json').read_text())
    ac_data._key_list(stored)
    assert sorted(stored) == keys('train'), 'train view keys are not the train keys'
    assert side['keys_sha256'] == rs_vh.names_sha(stored), 'view keys hash'
    _check_meta(side, ('view', 'instruction', 'model_id', 'model_revision', 'dtype', 'shape', 'keys_sha256',
                       'img_sha256', 'gate', 'smoke'), (len(stored), WIDTH))
    x = np.load(path / 'img.npy', mmap_mode='r', allow_pickle=False)
    ac_data._array(x, len(stored), WIDTH, np.float16, unit=True)
    assert side['img_sha256'] == rs_vh.array_sha(x), 'view image array hash'
    return stored, x


def _teacher_images(requested):
    """T: instructed emotion images. Train keys from TRAIN_IMG_VIEW, val keys from r12's teacher view, by key."""
    requested = list(requested)
    assert Path(pf_paths.VAL_TEST_TEACHER_VIEW) == Path(ac_paths.VIEWS) / ac_paths.WORDINGS['artwork'], 'teacher view'
    train, val = set(keys('train')), set(keys('val'))
    unknown = [k for k in requested if k not in train and k not in val]
    assert not unknown, f'unknown key (not a train or val photo): {unknown[:3]}'
    out = np.empty((len(requested), WIDTH), dtype=np.float32)
    for part, members in (('train', train), ('val', val)):
        mask = np.fromiter((k in members for k in requested), bool, len(requested))
        if not mask.any():
            continue
        names = [k for k in requested if k in members]
        if part == 'train':
            stored, x = _train_view()
            raw = x[ac_data._gather(stored, names)]
        else:
            raw = ac_data._view('teacher_artwork', names)      # r12's checked view, joined by key
        out[mask] = _unit(np.asarray(raw))
    return out


def _instructed_texts(requested):
    """E: instructed explanation means, from TXT_VIEW joined by csv_row (rows.json checked against the index)."""
    requested = list(requested)
    path = Path(pf_paths.TXT_VIEW)
    side = ac_data._json(path / 'meta.json', ('txt_sha256', 'shape'))
    rows_json = json.loads((path / 'rows.json').read_text())
    assert set(rows_json) == {'csv_row', 'key'}, 'rows.json fields'
    view_csv, view_key = rows_json['csv_row'], rows_json['key']
    assert len(view_csv) == len(view_key) and len(set(view_csv)) == len(view_csv), 'rows.json csv_row'
    expect = [text_rows(s) for s in ('train', 'val')]
    index_key = dict(zip(np.concatenate([e['csv_row'] for e in expect]).tolist(), expect[0]['key'] + expect[1]['key']))
    assert dict(zip(view_csv, view_key)) == index_key, 'rows.json disagrees with the index text rows (csv_row -> key)'
    _check_meta(side, ('view', 'instruction', 'model_id', 'model_revision', 'dtype', 'shape', 'txt_sha256',
                       'rows_sha256', 'gate', 'smoke'), (len(view_csv), WIDTH))
    x = np.load(path / 'txt.npy', mmap_mode='r', allow_pickle=False)
    ac_data._array(x, len(view_csv), WIDTH, np.float16)
    assert side['txt_sha256'] == rs_vh.array_sha(x), 'view text array hash'
    rows = _eligible_rows(requested)
    where = {c: i for i, c in enumerate(view_csv)}
    view_pos = [where[c] for c in rows['csv_row'].tolist()]
    return _group_mean(requested, rows['key'], view_pos, x)


def photo_arrays(requested):
    """Training inputs for train and/or val keys, float32 unit [n, 2048], rows in `requested` order.

    I default image, D default explanation mean, E instructed explanation mean, T instructed emotion image.
    """
    requested = list(requested)
    return dict(I=default_images(requested), D=_default_texts(requested), E=_instructed_texts(requested),
                T=_teacher_images(requested))


def fit_keys():
    """The sorted train keys minus the near-copies that 02 recorded (D12)."""
    removed = ac_data._json(pf_paths.NEARCOPY_FILE, ('removed_keys',))['removed_keys']
    ac_data._key_list(removed)
    train = keys('train')
    assert set(removed) <= set(train), 'removed keys must be train keys'
    drop = set(removed)
    return [k for k in train if k not in drop]
