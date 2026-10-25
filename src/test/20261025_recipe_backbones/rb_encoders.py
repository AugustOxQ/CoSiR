"""Encoder images by painting name: full split and the 746 ArtELingo-28 paintings.

An encoder image is the float32 store row divided by its float32 norm (rule §2), the CT's expression
(`ct2_data.default_images`: float16 -> float32, `apc_data.l2`). Stores use the generic layout (`img.npy` float16,
`index.json` with `images[i]['painting']`). Store roots are globals for model-free tests.
"""
import hashlib
import functools
import json

import numpy as np
import pandas as pd

import rb_paths
import ct2_data
from apc_data import l2

WIDTH = {'siglip2_so400m': 1152, 'clip_b32': 512, 'qwen3vl_emb_2b': 2048, 'dinov2_large': 1024}
FULL_ROOT = rb_paths.PRE / 'artelingo_full'
A28_ROOT = rb_paths.PRE / 'artelingo28'
FULL_COUNT = 61402
READ_SET = A28_ROOT / 'read_set_a28.parquet'
A28_COUNT = 746
A28_SHA256 = 'c1f1b27ed2af3057192f7684e25a327ca4be42127eacbe5ab0134f89f6ba2b70'   # the data's, 2026-10-25
A28_REFERENCE = 'siglip2_so400m'   # the complete A28 store whose index is cross-checked against the read set
A28_KEYS = ('emotion_focus', 'style_word', 'genre_word')


def read_store(path, width, names, count):
    """Rows of a generic image store for `names` (unique, caller order), unit norm float32 [n, width]."""
    index = json.loads((path / 'index.json').read_text())
    assert index['backbone'] == path.name and index['dataset'] == path.parent.name, (path, index['dataset'])
    assert not index['smoke'] and index['dtype'] == 'float16' and index['dim'] == width
    stored = [row['painting'] for row in index['images']]
    assert len(stored) == count == index['n_images'], (path, len(stored), count)
    assert len(set(stored)) == count, f'{path}: duplicate painting names'
    raw = np.load(path / 'img.npy', mmap_mode='r')
    assert raw.shape == (count, width) and raw.dtype == np.float16, (path, raw.shape, raw.dtype)
    assert len(set(names)) == len(names), 'duplicate requested names'
    lookup = {name: row for row, name in enumerate(stored)}
    missing = [name for name in names if name not in lookup]
    assert not missing, f'{path}: {len(missing)} names missing, e.g. {missing[:3]}'
    rows = np.asarray([lookup[name] for name in names], dtype=np.int64)
    x = np.asarray(raw[rows], dtype=np.float32)
    assert np.isfinite(x).all(), f'{path}: nonfinite features'
    assert (np.linalg.norm(x, axis=1) > 0).all(), f'{path}: zero images'
    x = l2(x)
    assert np.isfinite(x).all() and x.shape == (len(names), width)
    return x


def full_images(backbone, names):
    """Each encoder's default image of the full-split paintings `names`, float32 [n, WIDTH], unit norm.

    The Qwen image goes through the CT's own loader (regression path unchanged); the others read their
    `artelingo_full` store with the same normalization.
    """
    assert backbone in WIDTH, backbone
    names = [str(name) for name in names]
    assert len(set(names)) == len(names), 'duplicate requested names'
    if backbone == 'qwen3vl_emb_2b':
        x = ct2_data.default_images({'all_painting_ids': np.asarray(names)})
    else:
        x = read_store(FULL_ROOT / backbone, WIDTH[backbone], names, FULL_COUNT)
    assert x.dtype == np.float32 and x.shape == (len(names), WIDTH[backbone]) and np.isfinite(x).all()
    return x


def _sha256(names):
    return hashlib.sha256('\n'.join(names).encode('utf-8')).hexdigest()


@functools.lru_cache(maxsize=None)
def _a28_list(read_set, root):
    """The 746 painting names, sorted: the read set's `painting` column, checked against a store's index.

    A painting name is the `painting` string in both places: the read-set parquet column and the store index
    `images[i]['painting']` (also the image file's stem). The index's texts carry the parquet's `row_id` and
    `painting` per caption row; those must match row for row (row_id is the CSV position, never a sample_id).
    """
    frame = pd.read_parquet(read_set, columns=['row_id', 'painting'])
    assert frame.row_id.is_unique and frame.painting.notna().all()
    names = sorted(frame.painting.unique())
    assert len(names) == A28_COUNT, len(names)
    index = json.loads((root / A28_REFERENCE / 'index.json').read_text())
    assert index['dataset'] == 'artelingo28' and index['backbone'] == A28_REFERENCE and not index['smoke']
    stored = [row['painting'] for row in index['images']]
    assert len(stored) == len(set(stored)) == index['n_images'], 'duplicate or miscounted store images'
    assert set(stored) == set(names), 'store index and read-set painting names differ'
    stems = [row['image'].rsplit('/', 1)[-1].rsplit('.', 1)[0] for row in index['images']]
    assert stems == stored, 'image file stems differ from painting names'
    texts = index['texts']
    assert len(texts) == index['n_texts'] == len(frame), 'caption row count differs from the read set'
    by_row = dict(zip(frame.row_id.tolist(), frame.painting))
    assert all(by_row[row['row_id']] == row['painting'] for row in texts), 'row_id painting differs from read set'
    assert {row['row_id'] for row in texts} == set(by_row), 'row_id sets differ'
    assert _sha256(names) == A28_SHA256, 'A28 list sha256 changed'
    return tuple(names)


def a28_names():
    """The 746 ArtELingo-28 paintings, sorted; the order of every A28 output."""
    return list(_a28_list(READ_SET, A28_ROOT))


def a28_sha256():
    """SHA-256 of '\\n'.join(names).encode('utf-8'), as for the held list; equals the pinned `A28_SHA256`."""
    return _sha256(a28_names())


def a28_images(backbone):
    """Each encoder's image of the 746 paintings in `a28_names()` order, float32 [746, WIDTH], unit norm.

    The own ArtELingo-28 store of every backbone, Qwen included (its image differs from the full split's).
    """
    assert backbone in WIDTH, backbone
    return read_store(A28_ROOT / backbone, WIDTH[backbone], a28_names(), A28_COUNT)


def a28_instructed(key):
    """The instructed Qwen image (`key` in A28_KEYS) of the 746 paintings in `a28_names()` order, float32 unit norm.

    Store `A28_ROOT/qwen3vl_emb_2b_aspect_prompts/<key>` read with the CT's sealed check (done.json: the CT's
    instruction, not a smoke store, the CT's Qwen revision, count and ids hash) and name reader (`ids.npz` painting_names); one row per painting, 746 exactly.
    """
    assert key in A28_KEYS, key
    path = ct2_data._sealed(A28_ROOT / 'qwen3vl_emb_2b_aspect_prompts' / key, key)
    done = json.loads((path / 'done.json').read_text())
    stored, raw = ct2_data._image_store(path)
    names = a28_names()
    assert len(stored) == A28_COUNT, (key, len(stored))
    assert done['model_revision'] == ct2_data.QWEN_REVISION, f'{key}: model revision {done["model_revision"]}'
    assert done['count'] == len(stored) == len(raw), (key, done['count'], len(stored), len(raw))
    assert done['ids_sha256'] == ct2_data.ids_sha256({'painting_names': stored}), f'{key}: ids hash mismatch'
    lookup = {str(name): row for row, name in enumerate(stored)}
    missing = [name for name in names if name not in lookup]
    assert not missing, f'{key}: {len(missing)} names missing, e.g. {missing[:3]}'
    x = np.asarray(raw[np.asarray([lookup[name] for name in names], dtype=np.int64)], dtype=np.float32)
    assert np.isfinite(x).all(), f'{key}: nonfinite features'
    assert (np.linalg.norm(x, axis=1) > 0).all(), f'{key}: zero instructed images'
    x = l2(x)
    assert np.isfinite(x).all() and x.shape == (A28_COUNT, 2048)
    return x
