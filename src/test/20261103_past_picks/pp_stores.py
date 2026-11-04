"""r19 store reader: `load(dataset, view, keys) -> float32 [n, d]`, one reader for every old and new feature store.

Rows come back in the order of `keys`, joined by key (constitution C6), L2-normalised once (the stored float16 rows cast
to float32 and divided by their float32 norm, the expression of `rb_encoders.full_images` and `mt_data._unit`, so the
result is bit-equal to those readers on the stores they share).

Datasets and keys
  artelingo  painting names                      stores: `artelingo_full/<encoder store>` (index.json, `painting`);
                                                 teachers `<view>_s2img` and `<view>_held` (ids.npz, painting names)
  mmart      MMArt keys `Style/stem.jpg`         keys of the MMArt index with `artelingo == 'none'` read the MMArt
                                                 stores; the others (held, trainval) read the ArtELingo stores by
                                                 painting (teachers: `_s2img` for trainval, `_held` for held)
  affection  `source/image_name`                 old folders (index.json or keys.json), CLIP L/14's two folders,
                                                 new folders (`pp_paths.NEW_STORE`)
  semart     `image_file`                        old index.json stores, new keys.json stores
Views: the encoder keys of `pp_paths.ENCODERS` (default image; `qwen_default` is accepted for `qwen`), and the Qwen
instructed views `emotion_focus`, `style_word`, `genre_word` (Affection also `emotion_focus_photo`).

Folder formats, told apart by their files: `keys.json` (+ `meta.json`), `index.json` (`images` list; the key field
depends on the dataset), `ids.npz` (`painting_names`). New stores: `pp_paths.NEW_STORE`, `img.npy` float16 unit rows.

`check(dataset, view)` runs the store checks of gate (i) and returns a dict (it records problems, it does not raise on
bad data): one row per key, finite, unit within `pp_paths.UNIT_TOL`, keys join, `meta.json` hashes, row count and key set
of a new store.
"""
import functools
import hashlib
import json
from pathlib import Path

import numpy as np

import pp_paths as PP

ART_ROOT = PP.ART
AFF_ROOT = PP.AFF
MMART_ROOT = PP.MMART_PRE
SEMART_ROOT = PP.SEMART_PRE
MMART_INDEX = PP.MMART_INDEX
NEW_STORE = dict(PP.NEW_STORE)           # (dataset, encoder or teacher view) -> folder; tests may point it elsewhere
EXPECTED_ROWS = dict(PP.STORE_ROWS)      # gate (i): rows of a NEW store, and the key set it must cover

TEACHERS = tuple(PP.TEACHER_VIEW.values())               # emotion_focus, style_word, genre_word
DATASETS = ('artelingo', 'mmart', 'affection', 'semart')
ALIAS = {'qwen_default': 'qwen'}                         # r14's and r15's name of Qwen's default-image view
PROMPTS = 'qwen3vl_emb_2b_aspect_prompts'
WIDTH = {**PP.WIDTH, **{v: 2048 for v in TEACHERS}, 'emotion_focus_photo': 2048}
ART_FOLDER = dict(PP.ART_STORE)                          # encoder -> folder under artelingo_full (and old stores)
OLD_AFFECTION = {'qwen': ('qwen3vl_emb_2b',), 'siglip2': ('siglip2_so400m',), 'siglip1': ('siglip_so400m',),
                 'clip_l14': ('clip_l14_valtest', 'clip_l14_train')}
OLD_AFFECTION_TEACHER = {'emotion_focus': ('emotion_focus', 'emotion_focus_train'),
                         'emotion_focus_photo': ('emotion_focus_photo',)}
OLD_SEMART = {'qwen': 'qwen3vl_emb_2b', 'siglip2': 'siglip2_so400m', 'clip_b32': 'clip_b32'}
OLD_MMART = {'qwen': 'qwen3vl_emb_2b', 'siglip2': 'siglip2_so400m', 'siglip1': 'siglip_so400m', 'clip_l14': 'clip_l14'}

_CACHE = {}


def canonical(view):
    view = ALIAS.get(view, view)
    assert view in WIDTH, f'unknown view {view!r}'
    return view


# ------------------------------------------------------------------------------------------------ folders
def _art_sources(view):
    if view in TEACHERS:
        base = Path(ART_ROOT) / PROMPTS
        return [base / f'{view}_s2img', base / f'{view}_held']
    assert view in ART_FOLDER, f'no ArtELingo store for {view!r}'
    return [Path(ART_ROOT) / ART_FOLDER[view]]


def _mmart_sources(view):
    if ('mmart', view) in NEW_STORE:
        return [Path(NEW_STORE[('mmart', view)])]
    if view in TEACHERS:
        return [Path(MMART_ROOT) / PROMPTS / view]
    assert view in OLD_MMART, f'no MMArt store for {view!r}'
    return [Path(MMART_ROOT) / OLD_MMART[view]]


def _affection_sources(view):
    if ('affection', view) in NEW_STORE:
        return [Path(NEW_STORE[('affection', view)])]
    if view in OLD_AFFECTION_TEACHER:
        return [Path(AFF_ROOT) / PROMPTS / f for f in OLD_AFFECTION_TEACHER[view]]
    assert view in OLD_AFFECTION, f'no Affection store for {view!r}'
    return [Path(AFF_ROOT) / f for f in OLD_AFFECTION[view]]


def _semart_sources(view):
    if ('semart', view) in NEW_STORE:
        return [Path(NEW_STORE[('semart', view)])]
    assert view in OLD_SEMART, f'no SemArt store for {view!r}'
    return [Path(SEMART_ROOT) / OLD_SEMART[view]]


def sources(dataset, view):
    """The store folders of (dataset, view), in lookup order. For `mmart`: the MMArt stores (keys of the `new` split)."""
    assert dataset in DATASETS, f'unknown dataset {dataset!r}'
    view = canonical(view)
    return {'artelingo': _art_sources, 'mmart': _mmart_sources, 'affection': _affection_sources,
            'semart': _semart_sources}[dataset](view)


# ------------------------------------------------------------------------------------------------ one folder
def _stamp(folder):
    files = [folder / n for n in ('keys.json', 'index.json', 'ids.npz', 'img.npy')]
    return tuple((f.name, f.stat().st_mtime_ns, f.stat().st_size) for f in files if f.exists())


def _index_key(dataset, m):
    if dataset in ('artelingo', 'mmart'):
        return m['painting']
    if dataset == 'affection':
        return f"{m['source']}/{m['image_name']}"
    return m['image_file']


def folder_keys(folder, dataset):
    """Keys of a store folder in row order (keys.json, else index.json `images`, else ids.npz). Unique, asserted."""
    folder = Path(folder)
    assert folder.is_dir(), f'store folder missing: {folder}'
    if (folder / 'keys.json').exists():
        keys = json.loads((folder / 'keys.json').read_text())
    elif (folder / 'index.json').exists():
        keys = [_index_key(dataset, m) for m in json.loads((folder / 'index.json').read_text())['images']]
    elif (folder / 'ids.npz').exists():
        with np.load(folder / 'ids.npz', allow_pickle=False) as z:
            keys = z['painting_names'].tolist()
    else:
        raise AssertionError(f'{folder}: no keys.json, index.json or ids.npz')
    assert all(isinstance(k, str) for k in keys), f'{folder}: keys are not strings'
    assert len(set(keys)) == len(keys), f'duplicate keys in {folder}'
    return keys


def _folder(folder, dataset):
    """(keys, key -> row, img memmap) of a folder, cached by the stamp of its files."""
    folder = Path(folder)
    stamp = (str(folder), dataset, _stamp(folder) if folder.is_dir() else None)
    hit = _CACHE.get(stamp)
    if hit is None:
        keys = folder_keys(folder, dataset)
        img = np.load(folder / 'img.npy', mmap_mode='r')
        assert img.ndim == 2 and img.shape[0] == len(keys), f'{folder}: rows {img.shape} vs {len(keys)} keys'
        hit = _CACHE[stamp] = (keys, {k: i for i, k in enumerate(keys)}, img)
    return hit


def _unit(x):
    x = np.asarray(x, dtype=np.float32)
    assert np.isfinite(x).all(), 'non-finite feature rows (finite check)'
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    assert (norms > 0).all(), 'zero feature row'
    return x / np.maximum(norms, 1e-12)


def _gather(folders, dataset, keys, width, what):
    """Raw float32 rows of `keys`, each found in exactly one of `folders`; asserts width, not found, not unique."""
    folders = [Path(f) for f in folders]
    out = np.empty((len(keys), width), dtype=np.float32)
    found = np.zeros(len(keys), dtype=np.int64)
    for folder in folders:
        _, lookup, img = _folder(folder, dataset)
        assert img.shape[1] == width, f'{folder}: width {img.shape[1]}, expected {width}'
        want = [(j, lookup[k]) for j, k in enumerate(keys) if k in lookup]
        if not want:
            continue
        js = np.asarray([w[0] for w in want])
        rows = np.asarray([w[1] for w in want])
        found[js] += 1
        order = np.argsort(rows)                           # sorted reads from the memmap
        out[js[order]] = np.asarray(img[rows[order]], dtype=np.float32)
    assert (found <= 1).all(), f'{what}: keys found in more than one folder'
    missing = [k for k, f in zip(keys, found) if f == 0]
    assert not missing, f'{what}: {len(missing)} of {len(keys)} keys not found, e.g. {missing[:2]}'
    return out


# ------------------------------------------------------------------------------------------------ the MMArt index
@functools.lru_cache(maxsize=2)
def _read_index(path, stamp):
    import pandas as pd
    return pd.read_parquet(path, columns=['key', 'painting', 'split', 'artelingo'])


def mmart_index(index=None):
    """MMArt index with columns key, painting, split, artelingo; unique keys and paintings, artelingo in
    none/held/trainval."""
    if index is None:
        index = _read_index(str(MMART_INDEX), Path(MMART_INDEX).stat().st_mtime_ns)
    missing = [c for c in ('key', 'painting', 'split', 'artelingo') if c not in index.columns]
    assert not missing, f'MMArt index lacks columns {missing}'
    assert index.key.is_unique and index.painting.is_unique, 'duplicate MMArt key or painting'
    assert set(index.artelingo) <= {'none', 'held', 'trainval'}, set(index.artelingo)
    return index


# ------------------------------------------------------------------------------------------------ load
def load(dataset, view, keys, folder_list=None, index=None):
    """float32 [len(keys), d]: the unit rows of `view` for `keys`, in the order of `keys`, joined by key.

    `folder_list` replaces the folders (a check of one store; not for `mmart`). `index` is the MMArt index frame.
    Asserts: non-empty unique keys; each key found in exactly one folder; width; finite; non-zero rows."""
    assert dataset in DATASETS, f'unknown dataset {dataset!r}'
    view = canonical(view)
    keys = [str(k) for k in keys]
    assert keys and len(set(keys)) == len(keys), 'keys must be non-empty and unique'
    what = f'{dataset}/{view}'
    if dataset == 'mmart':
        assert folder_list is None, 'folder_list is not available for mmart'
        return _unit(_mmart_rows(view, keys, mmart_index(index)))
    folders = sources(dataset, view) if folder_list is None else [Path(f) for f in folder_list]
    return _unit(_gather(folders, dataset, keys, WIDTH[view], what))


def _mmart_rows(view, keys, index):
    table = index.set_index('key')
    unknown = [k for k in keys if k not in table.index]
    assert not unknown, f'{len(unknown)} keys are not in the MMArt index, e.g. {unknown[:2]}'
    rows = table.loc[keys]
    split = set(rows.split)      # r15 reads only new, val and held keys (mt_data.py:240, mp_stores.py:91)
    assert split <= {'new', 'val', 'held'}, f'MMArt: only new, val and held keys, got {split - {"new", "val", "held"}}'
    group = rows.artelingo.to_numpy()
    out = np.empty((len(keys), WIDTH[view]), dtype=np.float32)
    mm = np.flatnonzero(group == 'none')
    art = np.flatnonzero(group != 'none')
    if len(mm):
        out[mm] = _gather(_mmart_sources(view), 'mmart', [keys[i] for i in mm], WIDTH[view], f'mmart/{view}')
    if len(art):
        paintings = rows.painting.to_numpy()[art].tolist()
        out[art] = _gather(_art_sources(view), 'artelingo', paintings, WIDTH[view], f'mmart/{view} (ArtELingo side)')
    return out


# ------------------------------------------------------------------------------------------------ check (gate (i))
def _sha_keys(keys):
    return hashlib.sha256('\n'.join(keys).encode()).hexdigest()


def _sha_array(x):
    h = hashlib.sha256()
    for start in range(0, len(x), 2048):
        h.update(np.ascontiguousarray(x[start:start + 2048]).tobytes())
    return h.hexdigest()


def _reference_keys(dataset, index=None):
    """The key set a new store must cover: the sealed Qwen default store (MMArt: the index keys with no ArtELingo copy)."""
    if dataset == 'mmart':
        t = mmart_index(index)
        return set(t.key[t.artelingo == 'none'])
    return set(folder_keys(sources(dataset, 'qwen')[0], dataset))


def check(dataset, view, folder_list=None, index=None, sample=200):
    """Gate (i) store checks of (dataset, view). Returns a dict; `ok` is True when `problems` is empty.

    Per folder: exists, keys readable and unique, rows == keys, width, float16, all values finite, rows unit within
    `pp_paths.UNIT_TOL`, `meta.json` (when present) n / keys_sha256 / img_sha256. Across the folders: keys disjoint;
    a deterministic sample of keys joins back through `load` in shuffled order. For a store in `NEW_STORE`: the
    rows equal `EXPECTED_ROWS[dataset]` and the key set equals the reference's (the sealed Qwen store; MMArt: the new keys)."""
    view = canonical(view)
    folders = [Path(f) for f in (sources(dataset, view) if folder_list is None else folder_list)]
    problems = []
    all_keys, n_nonfinite, max_dev, d = [], 0, 0.0, None
    for folder in folders:
        try:
            keys = folder_keys(folder, dataset)
        except AssertionError as e:
            problems.append(str(e))
            continue
        img = np.load(folder / 'img.npy', mmap_mode='r')
        if img.ndim != 2 or img.shape[0] != len(keys):
            problems.append(f'{folder}: rows {img.shape} vs {len(keys)} keys')
            continue
        d = int(img.shape[1])
        if d != WIDTH[view]:
            problems.append(f'{folder}: width {d}, expected {WIDTH[view]}')
        if img.dtype != np.float16:
            problems.append(f'{folder}: dtype {img.dtype}, expected float16')
        bad = 0
        dev = 0.0
        for start in range(0, len(keys), 4096):
            block = np.asarray(img[start:start + 4096], dtype=np.float32)
            finite = np.isfinite(block).all(axis=1)
            bad += int((~finite).sum())
            if finite.any():
                dev = max(dev, float(np.abs(np.linalg.norm(block[finite], axis=1) - 1).max()))
            else:
                dev = max(dev, 1.0)
        n_nonfinite += bad
        max_dev = max(max_dev, dev)
        if bad:
            problems.append(f'{folder}: {bad} rows with non-finite values')
        if dev > PP.UNIT_TOL:
            problems.append(f'{folder}: rows not unit, max deviation {dev:.2e} > {PP.UNIT_TOL}')
        meta_file = folder / 'meta.json'
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            if 'n' in meta and meta['n'] != len(keys):
                problems.append(f'{folder}: meta n {meta["n"]} vs {len(keys)} keys')
            if 'keys_sha256' in meta and meta['keys_sha256'] != _sha_keys(keys):
                problems.append(f'{folder}: meta keys_sha256 does not match keys.json')
            if 'img_sha256' in meta and meta['img_sha256'] != _sha_array(img):
                problems.append(f'{folder}: meta img_sha256 does not match img.npy')
        all_keys.append(keys)
    flat = [k for ks in all_keys for k in ks]
    if len(set(flat)) != len(flat):
        problems.append('duplicate keys across the store folders')
    if not problems:
        problems += _join_problems(dataset, view, flat, folder_list, index, sample)
    if (dataset, view) in NEW_STORE and dataset in EXPECTED_ROWS:
        if len(flat) != EXPECTED_ROWS[dataset]:
            problems.append(f'{len(flat)} rows, expected {EXPECTED_ROWS[dataset]}')
        try:
            ref = _reference_keys(dataset, index)
            if set(flat) != ref:
                problems.append(f'key set differs from the reference: {len(set(flat) - ref)} extra, '
                                f'{len(ref - set(flat))} missing')
        except AssertionError as e:
            problems.append(f'reference keys unreadable: {e}')
    return dict(dataset=dataset, view=view, folders=[str(f) for f in folders], n=len(flat), d=d,
                n_nonfinite=n_nonfinite, max_unit_dev=max_dev, unit_tol=PP.UNIT_TOL, problems=problems,
                ok=not problems)


def _join_problems(dataset, view, flat, folder_list, index, sample):
    """A deterministic sample of keys, asked in shuffled order, comes back as the stored rows of those keys."""
    if not flat:
        return ['no keys']
    rng = np.random.default_rng(0)
    pick = rng.choice(len(flat), size=min(sample, len(flat)), replace=False)
    ask = [flat[i] for i in pick]
    try:
        if dataset == 'mmart':
            got = load('mmart', view, ask, index=index)
        else:
            got = load(dataset, view, ask, folder_list=folder_list)
    except AssertionError as e:
        return [f'keys do not join: {e}']
    folders = sources(dataset, view) if folder_list is None else [Path(f) for f in folder_list]
    for folder in folders:
        keys, lookup, img = _folder(folder, dataset)
        here = [(j, lookup[k]) for j, k in enumerate(ask) if k in lookup]
        for j, r in here:
            if not np.array_equal(got[j], _unit(np.asarray(img[r], dtype=np.float32)[None])[0]):
                return [f'{folder}: key {ask[j]!r} does not join to its own row']
    return []
