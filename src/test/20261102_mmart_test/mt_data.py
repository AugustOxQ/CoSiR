"""r15 MMArt data door (ticket 01; spec §3.3, §3.5 0(i) and 0(v), §3.6, D5, D8, D9, D10, D14, D19).

Every other r15 module reads MMArt keys, order, labels, cut sets and feature rows only through here.

    keys(name)            MMArt keys in evaluation order; names: val, new, test, far, unseen_artist, shared_held
    order(keys)           ascending by sha256 of the key string (D9; as r14's bs_emoset.order)
    labels(keys, aspect)  int64 class index in mt_paths.CLASSES[aspect], -1 = no label (read from the index columns)
    features(view, keys)  float32 unit rows, joined by key: MMArt stores for `new` keys, ArtELingo stores (by painting
                          name) for `val` and `held` keys; teacher views on `val` come from the `<view>_s2img` stores
    artelingo_name(key)   the ArtELingo painting name of a key that is also an ArtELingo painting
    cut(threshold)        the test cut in §3.3's order: exact copies, near copies (cosine >= threshold), md5 groups
    gate_i(), gate_v()    stage 0 gates (i) and (v)

The index and the near-copy file are read lazily and cached. Feature rows are normalised as r11's `rb_encoders.
full_images` does (float32, divided by max(norm, 1e-12)); `mp_stores.load` alone does not normalise.
"""
import functools
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mt_paths as P  # noqa: E402

sys.path.insert(0, str(P.MP_ROUND))
import mp_stores as MPS  # noqa: E402  (the prep round's loader; read only)

RETRIEVAL_ROUND = P.MAIN / 'src/test/20261029_retrieval'   # rt_data / ct2_data: ArtELingo's own split lists
HELD_LIST = P.MAIN / 'src/test/20261020_claim_test/outputs/held/held_paintings.json'
SETS = ('val', 'new', 'test', 'far', 'unseen_artist', 'shared_held')
INDEX_COLUMNS = ('key', 'painting', 'md5', 'split', 'artelingo', 'exact_dup_trainval', 'emotion', 'style', 'genre')
PARITY_BAR = 0.999
UNIT_TOL = 1e-3
N_PARITY = 200
N_STORE_CHECKS = 14
N_VAL = P.SPLIT_COUNTS['val']
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')


# ------------------------------------------------------------------------------------------------ files, lazily
@functools.lru_cache(maxsize=1)
def _index_cached():
    return MPS.read_index(pd.read_parquet(P.MMART_INDEX))


@functools.lru_cache(maxsize=1)
def _nearcopy_cached():
    return pd.read_parquet(P.MMART_NEARCOPY)


def _index(index=None):
    table = _index_cached() if index is None else index
    missing = [c for c in INDEX_COLUMNS if c not in table.columns]
    assert not missing, f'index columns missing: {missing}'
    return table


_LOOKUP = {}


def _lookup(index):
    """key -> row position. One entry, held with its frame (so a reused id cannot match another frame)."""
    held = _LOOKUP.get('entry')
    if held is None or held[0] is not index:
        assert index.key.is_unique, 'duplicate index keys'
        _LOOKUP['entry'] = held = (index, {k: i for i, k in enumerate(index.key.tolist())})
    return held[1]


def _rows(keys, index):
    keys = list(keys)
    lookup = _lookup(index)
    missing = [k for k in keys if k not in lookup]
    assert not missing, f'{len(missing)} keys are not in the MMArt index, e.g. {missing[:2]}'
    return index.iloc[[lookup[k] for k in keys]]


# ------------------------------------------------------------------------------------------------ order
def order(keys):
    """Keys sorted ascending by the sha256 hex digest of the key string (D9, D10). Keys must be unique."""
    keys = list(keys)
    assert len(set(keys)) == len(keys), 'order: keys must be unique'
    return sorted(keys, key=lambda k: hashlib.sha256(k.encode('utf-8')).hexdigest())


# ------------------------------------------------------------------------------------------------ the cut
def cut(threshold, *, index=None, nearcopy=None):
    """The test cut (§3.3, D5, D10, D19). Of the `new` paintings, in this order: (1) `exact`: byte copies of an
    ArtELingo train or validation image; (2) `near`: of the rest, default-Qwen cosine >= threshold (inclusive, the
    float32 value compared in float64 as bs_emoset.removed_mask) with an ArtELingo train or validation painting;
    (3) `md5`: of the rest, every md5 group keeps its first key in sha256 order. Returns {'threshold', 'n_new',
    'kept', 'exact', 'near', 'md5'}, each list in sha256 order. The near-copy table must hold exactly one finite
    row per `new` key (joined by key)."""
    default = index is None and nearcopy is None
    if default:
        return _cut_default(float(threshold))
    return _cut(float(threshold), _index(index), _nearcopy_cached() if nearcopy is None else nearcopy)


@functools.lru_cache(maxsize=4)
def _cut_default(threshold):
    return _cut(threshold, _index(), _nearcopy_cached())


def _cut(threshold, idx, near):
    new = idx[idx.split == 'new']
    new_keys = new.key.tolist()
    assert near.key.is_unique and set(near.key) == set(new_keys) and len(near) == len(new_keys), \
        'near-copy table: need exactly one row per new key'
    cos = near.set_index('key').max_cos.astype(np.float64)
    assert np.isfinite(cos.to_numpy()).all(), 'near-copy table: non-finite cosine'
    ordered = order(new_keys)
    flags = new.set_index('key')
    exact = [k for k in ordered if bool(flags.exact_dup_trainval[k])]
    gone = set(exact)
    near_keys = [k for k in ordered if k not in gone and cos[k] >= threshold]
    gone |= set(near_keys)
    first, md5 = {}, []
    for k in ordered:                                         # sha256 order: the first key seen of a group is kept
        if k in gone:
            continue
        m = flags.md5[k]
        if m in first:
            md5.append(k)
        else:
            first[m] = k
    gone |= set(md5)
    kept = [k for k in ordered if k not in gone]
    assert len(kept) + len(exact) + len(near_keys) + len(md5) == len(new_keys)
    return dict(threshold=threshold, n_new=len(new_keys), kept=kept, exact=exact, near=near_keys, md5=md5)


# ------------------------------------------------------------------------------------------------ artists, subsets
def artist_of(name):
    """The artist of a WikiArt key or file name: the part of the file name before the first underscore (§3.6, D14).
    Folder underscores (`Abstract_Expressionism/`) do not count."""
    base = str(name).rsplit('/', 1)[-1]
    for ext in IMAGE_EXTENSIONS:
        if base.lower().endswith(ext):
            base = base[:-len(ext)]
            break
    return base.split('_', 1)[0]


def unseen(test_keys, train_names):
    """The keys of `test_keys` (kept in their order) whose artist has no painting among `train_names`."""
    seen = {artist_of(n) for n in train_names}
    return [k for k in test_keys if artist_of(k) not in seen]


@functools.lru_cache(maxsize=1)
def artelingo_train_names():
    """ArtELingo's own train list: the 42,969 train paintings of r10's and r11's split (ct2_data.split2('full'), the
    list `rt_data` and `rb_student` derive from; hash-checked there). Not the MMArt `split == 'train'` rows (40,155)."""
    for path in (str(RETRIEVAL_ROUND), str(P.MAIN / 'src/test/20261020_claim_test')):
        if path not in sys.path:
            sys.path.append(path)
    import rt_data  # noqa: F401  (sets the r10/r11 module paths)
    import ct2_data
    s = ct2_data.split2('full')
    names = s['all_painting_ids'][s['fit']].tolist()
    assert len(names) == len(set(names)) == ct2_data.FULL_COUNTS[0][0] == 42969
    return tuple(names)


def artelingo_name(key, *, index=None):
    """The ArtELingo painting name of `key` (the WikiArt file stem). Only for keys that are ArtELingo paintings."""
    row = _rows([key], _index(index)).iloc[0]
    assert row.artelingo != 'none', f'{key} is not an ArtELingo painting'
    return str(row.painting)


# ------------------------------------------------------------------------------------------------ sets
def keys(name):
    """MMArt keys of set `name` in evaluation order (`order`). val 5,769; new 16,782 (no cut); test 16,384 (cut at
    0.95); far 10,403 (cut at 0.85); unseen_artist (subset of test, D14); shared_held 11,528."""
    assert name in SETS, f'unknown set {name!r}'
    return list(_keys_cached(name))


@functools.lru_cache(maxsize=8)
def _keys_cached(name):
    idx = _index()
    if name in ('val', 'new'):
        out = order(idx[idx.split == name].key)
        assert len(out) == P.SPLIT_COUNTS[name]
    elif name == 'shared_held':
        out = order(idx[idx.split == 'held'].key)
        assert len(out) == P.SPLIT_COUNTS['held']
    elif name == 'test':
        out = cut(P.CUT_TEST)['kept']
    elif name == 'far':
        out = cut(P.CUT_FAR)['kept']
    else:
        out = unseen(cut(P.CUT_TEST)['kept'], artelingo_train_names())
    return tuple(out)


# ------------------------------------------------------------------------------------------------ labels
def labels(keys, aspect, *, index=None):
    """int64 [n] class index of each key in mt_paths.CLASSES[aspect]; -1 where MMArt has no label. Read from the
    index's `emotion`, `style` and `genre` columns by key; never from the key's text."""
    assert aspect in P.ASPECTS, f'unknown aspect {aspect!r}'
    rows = _rows(keys, _index(index))
    if aspect == 'genre':
        g = rows.genre.to_numpy().astype(np.int64)
        assert g.max(initial=-1) < len(P.CLASSES['genre'])
        return np.where(g < 0, -1, g)
    classes = {c: i for i, c in enumerate(P.CLASSES[aspect])}
    values = rows[aspect].tolist()
    unknown = {v for v in values if v != '' and v not in classes}
    assert not unknown, f'{aspect}: classes not in mt_paths.CLASSES: {sorted(unknown)[:3]}'
    return np.asarray([classes.get(v, -1) for v in values], dtype=np.int64)


# ------------------------------------------------------------------------------------------------ features
def _unit(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    assert np.isfinite(x).all() and (norms > 0).all(), 'non-finite or zero feature row'
    return x / np.maximum(norms, 1e-12)


def _raw_features(view, keys, index, root, art_root):
    """Stored values as float32 (not normalised), rows in the order of `keys`; every key found, finite."""
    assert view in P.VIEWS, f'unknown view {view!r}'
    keys = list(keys)
    assert keys and len(set(keys)) == len(keys), 'keys must be nonempty and unique'
    rows = _rows(keys, index)
    split = rows.split.to_numpy()
    assert set(split) <= {'new', 'val', 'held'}, f'features: only new, val and held keys, got {set(split) - {"new", "val", "held"}}'
    out = np.empty((len(keys), MPS.WIDTH[view]), dtype=np.float32)
    teacher_val = np.flatnonzero((split == 'val')) if view in MPS.TEACHERS else np.empty(0, dtype=np.int64)
    rest = np.setdiff1d(np.arange(len(keys)), teacher_val)
    if len(rest):
        out[rest] = MPS.load(view, [keys[i] for i in rest], index=index, root=root, art_root=art_root)
    if len(teacher_val):                                      # D8: r11's validation teacher views, by painting name
        base = Path(art_root) if art_root is not None else MPS.ART_ROOT
        store = base / (MPS.FOLDERS[view] + '_s2img')
        out[teacher_val] = MPS.join(store, rows.painting.iloc[teacher_val].tolist(), view, teacher=True)
    return out


def features(view, keys, *, index=None, root=None, art_root=None):
    """float32 [n, d] unit rows of `view` for `keys`, in the order of `keys`, joined by key. `new` keys read the MMArt
    stores; `val` and `held` keys read the ArtELingo stores by painting name (teacher views: `val` from
    `<view>_s2img`, `held` from `<view>_held`). Asserts: known view and keys, every key found, finite rows."""
    return _unit(_raw_features(view, keys, _index(index), root, art_root))


# ------------------------------------------------------------------------------------------------ gates
def check_parity(doc):
    """Gate (i), parity: each of the seven views has min cosine >= 0.999 on the 200 shared paintings (recomputed from
    the numbers; the stored flag is not trusted)."""
    evidence = {}
    for v in P.VIEWS:
        e = doc.get(v)
        evidence[v] = None if e is None else float(e['min_cosine'])
    ok = all(m is not None and m >= PARITY_BAR for m in evidence.values())
    ok = ok and all(doc[v]['n'] == N_PARITY for v in P.VIEWS if v in doc)
    return dict(min_cosine=evidence, bar=PARITY_BAR, **{'pass': bool(ok)})


def check_stores(doc):
    """Gate (i), store check: 14 of 14 stores (seven views, new and parity sets) pass."""
    stores = doc.get('stores', [])
    n_ok = sum(bool(s.get('ok')) for s in stores)
    return dict(n=len(stores), n_ok=n_ok, **{'pass': bool(doc.get('ok')) and len(stores) == N_STORE_CHECKS == n_ok})


def check_nearcopy(index, near, reference, expect_reference):
    """Gate (i), near-copy file: one row per new key, finite cosines, reference of `expect_reference` paintings."""
    new = set(index[index.split == 'new'].key)
    one_per_key = near.key.is_unique and set(near.key) == new and len(near) == len(new)
    cos = near.max_cos.to_numpy().astype(np.float64)
    finite = bool(np.isfinite(cos).all()) and bool(near.nearest_painting.notna().all())
    counts = {f'{c:.2f}': int((cos >= c).sum()) for c in (.85, .90, .95, .99)} if finite else None
    return dict(n=len(near), n_new=len(new), one_row_per_key=bool(one_per_key), finite=finite, reference=int(reference),
                expect_reference=int(expect_reference), counts=counts,
                **{'pass': bool(one_per_key and finite and reference == expect_reference)})


def check_unit(x, tol=UNIT_TOL):
    """Rows finite and unit norm within tol."""
    x = np.asarray(x)
    finite = bool(np.isfinite(x).all())
    dev = float(np.abs(np.linalg.norm(x.astype(np.float64), axis=1) - 1).max()) if finite else float('nan')
    return dict(n=int(len(x)), finite=finite, max_norm_deviation=dev, **{'pass': bool(finite and dev < tol)})


def reference_count():
    """ArtELingo train and validation paintings: the store's paintings (61,402) minus the held list (12,281), as
    mp_extract.nearcopy computes its reference (not stored in any JSON)."""
    names, _ = MPS.read_store(MPS.reference_folder('qwen_default'), reference=True)
    held = set(json.loads(HELD_LIST.read_text())['names'])
    assert len(held) == 12281 and held <= set(names)
    return len(names) - len(held)


def gate_i():
    """Stage 0 gate (i): prep parity, store check (14 of 14), the near-copy file (one row per new key, finite cosines,
    reference 49,121 recomputed, counts agree with mp_nearcopy.json) and the cut counts (16,384 at 0.95, 10,403 at
    0.85, 34 exact copies)."""
    idx, near = _index(), _nearcopy_cached()
    parity = check_parity(json.loads(P.PREP_PARITY.read_text()))
    stores = check_stores(json.loads(P.PREP_STORES_CHECK.read_text()))
    nc = check_nearcopy(idx, near, reference_count(), P.NEARCOPY_REFERENCE)
    stored = json.loads(P.PREP_NEARCOPY.read_text())
    nc['counts_match_prep_json'] = bool(stored['n'] == len(near) and nc['counts'] == stored['counts'])
    nc['pass'] = nc['pass'] and nc['counts_match_prep_json']
    cuts = {}
    for name, thr, expect in (('test', P.CUT_TEST, P.N_TEST), ('far', P.CUT_FAR, P.N_FAR)):
        c = cut(thr)
        cuts[name] = dict(threshold=thr, n=len(c['kept']), expect=expect, exact=len(c['exact']), near=len(c['near']),
                          md5=len(c['md5']), **{'pass': len(c['kept']) == expect and len(c['exact']) == P.N_EXACT})
    return dict(parity=parity, stores=stores, nearcopy=nc, cut=cuts,
                **{'pass': bool(parity['pass'] and stores['pass'] and nc['pass'] and all(c['pass'] for c in cuts.values()))})


def gate_v():
    """Stage 0 gate (v): 5,769 MMArt validation keys found in each teacher view's `<view>_s2img` store (by painting
    name), finite, unit norm within 1e-3; the store's done.json must say its extraction parity gate passed."""
    val = keys('val')
    assert len(val) == N_VAL
    idx = _index()
    names = _rows(val, idx).painting.tolist()
    views = {}
    for aspect, view in P.TEACHER_VIEW.items():
        store = MPS.ART_ROOT / (MPS.FOLDERS[view] + '_s2img')
        done = json.loads((store / 'done.json').read_text())
        x = MPS.join(store, names, view, teacher=True)               # asserts every name is found and finite
        unit = check_unit(x)
        manifest = bool(done['parity_gate']['passed']) and not done['smoke']
        views[view] = dict(aspect=aspect, n_found=int(len(x)), manifest_parity_gate=manifest, **unit)
        views[view]['pass'] = bool(unit['pass'] and len(x) == N_VAL and manifest)
    return dict(views=views, n_val=N_VAL, **{'pass': all(v['pass'] for v in views.values())})
