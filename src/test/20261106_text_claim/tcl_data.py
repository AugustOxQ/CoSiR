"""r23 data door (ticket 02; spec §3.3, D6, D9, gate (iv), C6).

Every later module reads sets, labels, texts and images only through here.

    frame(set, kind)            n_i, n_t, tp, ip, labels {aspect: {'t', 'i'}}, names, text (+ sample_id for ArtELingo)
    plain(set, kind)            (text, image): plain Qwen unit rows in frame order
    teacher(set, kind, aspect)  (text, image) under the aspect's instruction
    side(set, row)              ArtELingo only: clip_l14, siglip2_so400m, siglip_so400m (text, image); roberta (text, None)
    names_label(set, kind, a)   bool[n_t]: the text names its own label (r22's patterns)
    gate_iv()                   spec §3.5 (iv): set sizes

Sets: art_val, art_held (ArtELingo, painting names from rt_data, caption ids from the plain store's index by painting
name), mm_val, mm_test (MMArt keys in r15's sha256 order, one text per painting per kind). Joins are by sample id or
key (C6), never by row position. TCL_SMOKE=N (tcl_paths.SMOKE) keeps the first N paintings and their texts; a store
marked smoke is read only then. tcl_paths.STORE_ROOT, when set, is the base of the new stores (folder names below it
are the stores' own names).
"""
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

import tcl_paths as P
import tx_data
import mt_data
import mt_paths
import rt_data
import rs_vh
from pav_prompts import PROMPTS

FULL = tx_data.FULL
ART_SETS = ('art_val', 'art_held')
MM_SETS = ('mm_val', 'mm_test')
SPLIT = {'art_val': 'val', 'art_held': 'held', 'mm_val': 'val', 'mm_test': 'test'}
PLAIN_WIDTH, SIDE_WIDTH = 2048, dict(clip_l14=768, siglip2_so400m=1152, siglip_so400m=1152, roberta=768)
SIDE_IMAGE_ARM = dict(clip_l14='clip', siglip2_so400m='siglip2')


# ------------------------------------------------------------------------------------------------ small readers
def unit(x):
    return tx_data.unit(x)


def read_store(path, wanted, id_kind, instruction, width, dtype):
    """Unit float32 rows of the text store `path` for the ids `wanted`, in that order, joined by id (`id_kind`: 'ids'
    reads ids.npz `sample_ids`, 'keys' reads keys.json). Asserts: done.json instruction, smoke flag (a smoke store only
    under TCL_SMOKE), shape, dtype, finite rows, unique stored ids, every wanted id found."""
    path = Path(path)
    done = tx_data.read_json(path / 'done.json')
    assert done['smoke'] is False or P.SMOKE, f'smoke text store outside TCL_SMOKE: {path}'
    assert done['instruction'] == instruction, f'instruction mismatch: {path}'
    if id_kind == 'ids':
        with np.load(path / 'ids.npz', allow_pickle=False) as z:
            stored = z['sample_ids'].tolist()
    else:
        assert id_kind == 'keys'
        stored = tx_data.read_json(path / 'keys.json')
    x = np.load(path / 'txt.npy', mmap_mode='r', allow_pickle=False)
    assert x.ndim == 2 and x.dtype == dtype and x.shape[1] == width, (path, x.shape, x.dtype)
    assert x.shape[0] == len(stored) == done['count'], f'store rows, ids and count differ: {path}'
    rows = tx_data.gather(stored, list(wanted))
    values = np.asarray(x[rows], dtype=np.float32)
    assert values.shape == (len(wanted), width) and np.isfinite(values).all()
    return unit(values)


def store_base(store):
    """The base of a new store: tcl_paths' constant, or its own folder name under STORE_ROOT (smoke)."""
    assert P.STORE_ROOT is None or P.SMOKE, 'TCL_STORES requires TCL_SMOKE'
    return store if P.STORE_ROOT is None else Path(P.STORE_ROOT) / store.name


def art_teacher_path(split, aspect):
    instruction = P.INSTRUCTIONS[aspect]
    if split == 'held':
        return store_base(P.ART_TEACHER_STORE) / f'{instruction}_heldtxt'
    return tx_data.text_path('teacher', aspect, out=P.R22_OUT)


def mm_store_path(kind, instruction, split):
    """Plain validation text is r22's `text/mmart_val/e_<kind>`; everything else is a new store."""
    if instruction == 'default' and split == 'val':
        return P.R22_OUT / 'text/mmart_val' / ('e_' + kind)
    return store_base(P.MMART_TEXT_STORE) / kind / instruction / split


# ------------------------------------------------------------------------------------------------ ArtELingo frames
@lru_cache(maxsize=1)
def _plain_index():
    return tx_data.read_json(FULL / tx_data.STORES['plain'] / 'index.json')


@lru_cache(maxsize=1)
def _annotations():
    return tx_data.read_json(rs_vh.ANNOTATIONS)


@lru_cache(maxsize=2)
def _art_captions(split):
    """r22's `val_captions`, for validation or held: caption ids of the split's paintings from the plain store's index by
    painting name, sorted by sample id; emotion from the annotations; style and genre checked against rt_data.labels."""
    assert split in ('val', 'held')
    ns = rt_data.names(split)
    lookup = {name: i for i, name in enumerate(ns)}
    index = _plain_index()
    selected = [r for r in index['texts'] if r['painting'] in lookup]
    selected.sort(key=lambda r: r['sample_id'])
    assert len(selected) == P.N_CAPTIONS['art_' + split]
    ids = np.asarray([r['sample_id'] for r in selected], dtype=np.int64)
    assert (np.diff(ids) > 0).all()
    annotations = _annotations()
    rows = [annotations[int(sid)] for sid in ids]
    assert all(ann['painting'] == r['painting'] for ann, r in zip(rows, selected))
    emotion_names = sorted(rs_vh.EMOTIONS)
    emotion = np.asarray([emotion_names.index(r['emotion']) for r in rows], dtype=np.int64)
    pid = np.asarray([lookup[r['painting']] for r in selected], dtype=np.int64)
    counts = np.zeros((len(ns), len(emotion_names)), dtype=np.int64)
    np.add.at(counts, (pid, emotion), 1)
    labels = rt_data.labels(split)
    assert (counts.sum(1) > 0).all()
    assert np.array_equal(counts.argmax(1), labels['emotion'])
    styles = sorted(tx_data.STYLE_PATTERNS)
    assert {r['art_style'] for r in rows} <= set(styles), 'style codebook differs'
    if split == 'val':
        assert sorted({r['art_style'] for r in rows}) == styles
    assert np.array_equal(np.asarray([styles.index(r['art_style']) for r in rows]), labels['style'][pid])
    return dict(sample_id=ids, pid=pid, text=[r['caption'] for r in rows], emotion=emotion, names=ns, labels=labels)


# ------------------------------------------------------------------------------------------------ MMArt frames
@lru_cache(maxsize=2)
def _mm_texts(set_name):
    """The five texts of a set's keys, from the raw parquet joined by image_id (key order)."""
    import pandas as pd
    keys = mt_data.keys(SPLIT[set_name])
    cols = [f'e_{k}' for k in P.KINDS]
    frame = pd.read_parquet('/data/SSD/mmart/raw/data/data.parquet', columns=['image_id', *cols],
                            filters=[('image_id', 'in', keys)])
    assert frame['image_id'].is_unique and not frame[cols].isna().any().any()
    rows = tx_data.gather(frame['image_id'].tolist(), keys)
    frame = frame.iloc[rows].reset_index(drop=True)
    assert frame['image_id'].tolist() == keys
    return {k: frame[f'e_{k}'].tolist() for k in P.KINDS}


@lru_cache(maxsize=4)
def _frame_full(set_name, kind):
    """The unsliced frame of a set and text kind."""
    assert set_name in P.SETS
    assert kind in P.kinds(P.GROUP[set_name]), (set_name, kind)
    if set_name in ART_SETS:
        c = _art_captions(SPLIT[set_name])
        painting = c['labels']
        labels = {}
        for a in P.ASPECTS:
            t = c['emotion'].copy() if a == 'emotion' else painting[a][c['pid']]
            labels[a] = dict(t=t, i=painting[a].copy())
        n_i = len(c['names'])
        f = dict(n_i=n_i, n_t=len(c['pid']), tp=c['pid'].copy(), ip=np.arange(n_i, dtype=np.int64), labels=labels,
                 names=list(c['names']), text=list(c['text']), sample_id=c['sample_id'].copy())
    else:
        keys = mt_data.keys(SPLIT[set_name])
        n = len(keys)
        labels = {}
        for a in P.ASPECTS:
            y = mt_data.labels(keys, a)
            labels[a] = dict(t=y, i=y.copy())
        f = dict(n_i=n, n_t=n, tp=np.arange(n, dtype=np.int64), ip=np.arange(n, dtype=np.int64), labels=labels,
                 names=list(keys), text=list(_mm_texts(set_name)[kind]))
    for value in f.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    for lab in labels.values():
        for value in lab.values():
            value.setflags(write=False)
    return f


def frame(set_name, kind):
    """See the module doc. Under TCL_SMOKE=N: the first N paintings and their texts.

    Returns fresh dicts, lists and writable array copies; cached frame arrays are read-only.
    """
    f = _frame_full(set_name, kind)
    if not P.SMOKE:
        return dict({k: v.copy() if isinstance(v, np.ndarray) else v for k, v in f.items()},
                    names=list(f['names']), text=list(f['text']),
                    labels={a: {row: y.copy() for row, y in v.items()} for a, v in f['labels'].items()})
    keep = f['tp'] < P.SMOKE
    out = dict(n_i=min(P.SMOKE, f['n_i']), n_t=int(keep.sum()), tp=f['tp'][keep], names=f['names'][:P.SMOKE],
               text=[t for t, k in zip(f['text'], keep) if k],
               labels={a: dict(t=v['t'][keep], i=v['i'][:P.SMOKE].copy()) for a, v in f['labels'].items()})
    out['ip'] = np.arange(out['n_i'], dtype=np.int64)
    if 'sample_id' in f:
        out['sample_id'] = f['sample_id'][keep]
    return out


# ------------------------------------------------------------------------------------------------ texts
def _art_plain_style_rows(split, name, f, width):
    """Rows of an all-rows ArtELingo store (index.json + txt.npy), joined by sample id, painting name checked."""
    path = FULL / tx_data.STORES[name]
    index = tx_data.read_json(path / 'index.json')
    rows = tx_data.gather([r['sample_id'] for r in index['texts']], f['sample_id'].tolist())
    assert all(index['texts'][i]['painting'] == f['names'][p] for i, p in zip(rows, f['tp']))
    x = np.load(path / 'txt.npy', mmap_mode='r', allow_pickle=False)
    assert x.ndim == 2 and x.dtype == np.float16 and x.shape[1] == width and x.shape[0] == len(index['texts'])
    values = np.asarray(x[rows], dtype=np.float32)
    assert values.shape == (f['n_t'], width) and np.isfinite(values).all()
    return unit(values)


def plain_text(set_name, kind):
    f = frame(set_name, kind)
    if set_name in ART_SETS:
        return _art_plain_style_rows(SPLIT[set_name], 'plain', f, PLAIN_WIDTH)
    return read_store(mm_store_path(kind, 'default', SPLIT[set_name]), f['names'], 'keys', PROMPTS['default'],
                      PLAIN_WIDTH, np.float16)


def teacher_text(set_name, kind, aspect):
    assert aspect in P.ASPECTS
    f = frame(set_name, kind)
    instruction = P.INSTRUCTIONS[aspect]
    if set_name in ART_SETS:
        return read_store(art_teacher_path(SPLIT[set_name], aspect), f['sample_id'].tolist(), 'ids',
                          PROMPTS[instruction], PLAIN_WIDTH, np.float16)
    return read_store(mm_store_path(kind, instruction, SPLIT[set_name]), f['names'], 'keys', PROMPTS[instruction],
                      PLAIN_WIDTH, np.float16)


def _smoke_cut(x):
    return x[:P.SMOKE] if P.SMOKE else x


def _art_side_image(split, row, aspect='emotion'):
    if row in ('plain', 'teacher'):
        return rt_data.embed('plain' if row == 'plain' else 'instructed', split, aspect)
    if row in SIDE_IMAGE_ARM:
        return rt_data.embed(SIDE_IMAGE_ARM[row], split, aspect)
    assert row == 'siglip_so400m'
    path = FULL / tx_data.STORES[row]
    index = tx_data.read_json(path / 'index.json')
    rows = tx_data.gather([r['painting'] for r in index['images']], rt_data.names(split))
    x = np.load(path / 'img.npy', mmap_mode='r', allow_pickle=False)
    assert x.dtype == np.float16 and x.shape[1] == 1152
    return unit(x[rows])


def image(set_name, row, aspect='emotion'):
    """Unit float32 image rows in frame (painting) order. row: 'plain', 'teacher' (aspect), or an ArtELingo side row."""
    if set_name in ART_SETS:
        return _smoke_cut(_art_side_image(SPLIT[set_name], row, aspect))
    keys = frame(set_name, 'narrative')['names']
    if row == 'plain':
        return mt_data.features('qwen_default', keys)
    assert row == 'teacher' and aspect in P.ASPECTS
    return mt_data.features(mt_paths.TEACHER_VIEW[aspect], keys)


def plain(set_name, kind):
    return plain_text(set_name, kind), image(set_name, 'plain')


def teacher(set_name, kind, aspect):
    return teacher_text(set_name, kind, aspect), image(set_name, 'teacher', aspect)


def side(set_name, row):
    assert set_name in ART_SETS and row in SIDE_WIDTH, (set_name, row)
    f = frame(set_name, P.CAPTION)
    if row != 'roberta':
        return _art_plain_style_rows(SPLIT[set_name], row, f, SIDE_WIDTH[row]), image(set_name, row)
    path = FULL / 'roberta_go_emotions_s2' if set_name == 'art_val' else store_base(P.ART_ROBERTA_STORE)
    return read_store(path, f['sample_id'].tolist(), 'ids', None, 768, np.float32), None


# ------------------------------------------------------------------------------------------------ own-label words
def r22_names(aspect):
    if aspect == 'emotion':
        return sorted(rs_vh.EMOTIONS)
    if aspect == 'style':
        return sorted(tx_data.STYLE_PATTERNS)
    assert aspect == 'genre'
    assert set(tx_data.GENRE_NAMES) == set(tx_data.GENRE_PATTERNS)
    return list(tx_data.GENRE_NAMES)


def class_map(aspect):
    """MMArt class name -> r22 class name, explicit; each MMArt class must have one (and so a pattern)."""
    own = r22_names(aspect)
    lower = {n.lower(): n for n in own}
    assert len(lower) == len(own)
    mapping = {c: lower[c.lower()] for c in mt_paths.CLASSES[aspect] if c.lower() in lower}
    assert set(mapping) == set(mt_paths.CLASSES[aspect]), f'{aspect}: MMArt classes without an r22 name'
    if aspect != 'emotion':
        table = tx_data.STYLE_PATTERNS if aspect == 'style' else tx_data.GENRE_PATTERNS
        assert all(v in table for v in mapping.values()), f'{aspect}: class without a pattern'
    return mapping


@lru_cache(maxsize=24)
def _names_label_full(set_name, kind, aspect):
    f = _frame_full(set_name, kind)
    codes = f['labels'][aspect]['t']
    names = r22_names(aspect)
    if set_name in MM_SETS:                     # MMArt's class index -> r22's index of the same class
        classes = mt_paths.CLASSES[aspect]
        mapping = class_map(aspect)
        to_r22 = np.asarray([names.index(mapping[c]) for c in classes], dtype=np.int64)
        codes = np.where(codes < 0, -1, to_r22[np.maximum(codes, 0)])
    if aspect == 'emotion':
        return np.asarray([tx_data.emotion_named(t, y) for t, y in zip(f['text'], codes)], dtype=bool)
    table = tx_data.STYLE_PATTERNS if aspect == 'style' else tx_data.GENRE_PATTERNS
    assert all(names[int(y)] in table for y in np.unique(codes) if y >= 0)
    patterns = [re.compile(r'\b(?:' + table[name] + r')\b', re.IGNORECASE) for name in names]
    return np.asarray([y >= 0 and bool(patterns[int(y)].search(t)) for t, y in zip(f['text'], codes)], dtype=bool)


def names_label(set_name, kind, aspect):
    """bool[n_t]: the text names its own label (r22's patterns, both data sets)."""
    out = _names_label_full(set_name, kind, aspect)
    if not P.SMOKE:
        return out.copy()
    return out[_frame_full(set_name, kind)['tp'] < P.SMOKE]


def label_share(set_name, kind, aspect):
    """Share of labelled rows whose text names its own label (spec D9)."""
    f = frame(set_name, kind)
    got = names_label(set_name, kind, aspect)
    labelled = f['labels'][aspect]['t'] >= 0
    return float(got[labelled].mean())


# ------------------------------------------------------------------------------------------------ gate (iv)
def gate_iv():
    """Spec §3.5 (iv), on the full sets whatever TCL_SMOKE says. Returns the measured sizes; asserts them."""
    held = rt_data.names('held')
    captions = _art_captions('held')
    val = _art_captions('val')
    test = mt_data.keys('test')
    got = dict(art_held_paintings=len(held), art_held_captions=len(captions['sample_id']),
               art_held_genre_labelled=int((rt_data.labels('held')['genre'] >= 0).sum()),
               mm_test_keys=len(test), art_val_paintings=len(val['names']), art_val_captions=len(val['sample_id']),
               mm_val_keys=len(mt_data.keys('val')))
    want = dict(art_held_paintings=P.N_PAINTINGS['art_held'], art_held_captions=P.N_CAPTIONS['art_held'],
                art_held_genre_labelled=9949, mm_test_keys=P.N_PAINTINGS['mm_test'],
                art_val_paintings=P.N_PAINTINGS['art_val'], art_val_captions=P.N_CAPTIONS['art_val'],
                mm_val_keys=P.N_PAINTINGS['mm_val'])
    assert got == want, (got, want)
    assert len(set(test)) == len(test), 'MMArt test keys are not unique'
    # r15 cmd_cutsets hashes lexical key order; evaluation order sorts each key's digest.
    cutsets = tx_data.read_json(P.R15_ROUND / 'records/cutsets.json')
    digest = hashlib.sha256('\n'.join(sorted(test)).encode('utf-8')).hexdigest()
    assert digest == cutsets['test']['sha256'], 'MMArt test membership differs from r15 cutsets'
    assert test == sorted(test, key=lambda k: hashlib.sha256(k.encode('utf-8')).hexdigest()), \
        "MMArt test keys not in r15's order"
    return got
