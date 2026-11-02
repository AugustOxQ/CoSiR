"""MMArt key joins; return stored values as float32, without normalising again.

Like bs_stores.load, requests must be nonempty and unique; every key must occur
exactly once. Older ArtELingo stores are read only. Roots/index are injectable.
"""
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INDEX = Path('/data/SSD/mmart/mmart_index.parquet')
ART_ROOT = Path('/data/SSD2/pre_extract/artelingo_full')
DEFAULT_ROOT = Path('/data/SSD2/pre_extract/mmart')
FOLDERS = dict(qwen_default='qwen3vl_emb_2b', emotion_focus='qwen3vl_emb_2b_aspect_prompts/emotion_focus',
               style_word='qwen3vl_emb_2b_aspect_prompts/style_word',
               genre_word='qwen3vl_emb_2b_aspect_prompts/genre_word', siglip2='siglip2_so400m',
               siglip1='siglip_so400m', clip_l14='clip_l14')
VIEWS = tuple(FOLDERS)
TEACHERS = ('emotion_focus', 'style_word', 'genre_word')
WIDTH = dict(qwen_default=2048, emotion_focus=2048, style_word=2048, genre_word=2048,
             siglip2=1152, siglip1=1152, clip_l14=768)


def store_root():
    return Path(os.environ.get('MP_STORE_ROOT') or DEFAULT_ROOT)


def folder(view, which='new', root=None):
    assert view in VIEWS and which in ('new', 'parity')
    base = Path(root) if root is not None else store_root()
    return base / ('parity' if which == 'parity' else '') / FOLDERS[view]


def reference_folder(view, art_root=None):
    base = Path(art_root) if art_root is not None else ART_ROOT
    return base / (FOLDERS[view] + ('_held' if view in TEACHERS else ''))


def read_index(index=None):
    import pandas as pd
    table = pd.read_parquet(INDEX if index is None else index) if not hasattr(index, 'columns') else index
    assert table.key.is_unique and table.painting.is_unique, 'duplicate MMArt key or painting'
    assert set(table.artelingo) <= {'none', 'held', 'trainval'}
    return table


def read_store(path, teacher=False, reference=False):
    path = Path(path)
    if teacher:
        with np.load(path / 'ids.npz', allow_pickle=False) as ids:
            keys = ids['painting_names'].tolist()
    elif reference:
        keys = [r['painting'] for r in json.loads((path / 'index.json').read_text())['images']]
    else:
        keys = json.loads((path / 'keys.json').read_text())
    assert len(keys) == len(set(keys)), f'duplicate keys in {path}'
    x = np.load(path / 'img.npy', mmap_mode='r')
    assert x.ndim == 2 and x.shape[0] == len(keys), f'rows vs keys: {path}'
    return keys, x


def join(path, keys, view, **format_args):
    names, x = read_store(path, **format_args)
    assert x.shape[1] == WIDTH[view], f'wrong width: {path}'
    lookup = {k: i for i, k in enumerate(names)}
    missing = [k for k in keys if k not in lookup]
    assert not missing, f'{path}: keys not found: {missing[:3]}'
    out = np.asarray(x[[lookup[k] for k in keys]], dtype=np.float32)
    assert np.isfinite(out).all(), f'nonfinite features: {path}'
    return out


def load(view, keys, *, index=None, root=None, art_root=None):
    """Join MMArt and ArtELingo by key/painting; teacher trainval is unavailable."""
    assert view in VIEWS
    keys = list(keys)
    assert keys and len(keys) == len(set(keys)), 'keys must be nonempty and unique'
    table = read_index(index).set_index('key')
    assert all(k in table.index for k in keys), 'MMArt keys not found in index'
    rows = table.loc[keys]
    out = np.empty((len(keys), WIDTH[view]), dtype=np.float32)
    for group in ('none', 'held', 'trainval'):
        positions = np.flatnonzero(rows.artelingo.to_numpy() == group)
        if not len(positions):
            continue
        if group == 'none':
            out[positions] = join(folder(view, root=root), [keys[i] for i in positions], view)
        else:
            assert not (group == 'trainval' and view in TEACHERS), f'{view}: no trainval teacher features'
            out[positions] = join(reference_folder(view, art_root), rows.iloc[positions].painting.tolist(), view,
                                  teacher=view in TEACHERS, reference=True)
    return out
