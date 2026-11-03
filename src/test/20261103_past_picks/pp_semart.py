"""The SemArt set (r19 ticket 03; spec docs/superpowers/specs/2026-11-03-past-picks-design.md §3.3, §3.5 (vi), D5,
D10, D14, D19).

Rule (`sc_check.py`'s, imported; nothing re-implemented but the mask): a SemArt painting is in the new-artist set when
its nearest cosine to ArtELingo train and validation is < 0.95 (strict), its author matches none of `sc_check`'s three
name tiers against the artists of ArtELingo's 42,969 train paintings, and the author does not START with "UNKNOWN".
The cosines are read from `sc_nearcopy.parquet` (21,382 rows); the tiers are recomputed with `sc_check.artist_tiers`
and must equal the parquet's. Gate (vi): 11,162 paintings by 2,717 artists, type counts equal to `sc_check.json`'s
`new_artist_types`. Then the md5 step (r15 D10's form, `mt_data._cut`): in sha256 order of the key, the first
painting of each image-file md5 group stays (expected 11,159). Keys are `image_file`, in sha256 order everywhere.

    keys(name)     'gate' (11,162), 'set' (after md5), 'close' (set keys at nearest cosine >= 0.93), 'far' (the rest)
    labels(keys)   int64 [n], index in pp_paths.SEMART_TYPES (the store index's `type`, checked against the CSV)
    gate_vi()      dict: pass, counts, type counts, md5 groups
    record()       JSON-able dict for records/semart_set.json (ticket 12 writes it)

CLI:  python pp_semart.py gate | record     (prints JSON; writes nothing)
"""
import pp_paths  # noqa: F401,I001  (first: sets the older rounds' output roots)

import functools
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import sc_check

QWEN_STORE = pp_paths.SEMART_PRE / 'qwen3vl_emb_2b'      # the store the cosines of sc_nearcopy.parquet come from
IMAGES = pp_paths.SEMART_DATA / 'Images'
N_FILES = 21382
UNKNOWN_PREFIX = 'UNKNOWN'
KEY_SETS = ('gate', 'set', 'close', 'far')


# ---------------------------------------------------------------------------------------------- pure logic
def sha_order(keys):
    """Keys ascending by sha256 hex of the key string (r15 D9; `pp_paths.order`). Keys must be unique."""
    keys = list(keys)
    assert len(set(keys)) == len(keys), 'sha_order: keys must be unique'
    return pp_paths.order(keys)


def unknown_author(author):
    """sc_check's rule: the author, upper-cased, STARTS with "UNKNOWN" (20 forms; not equality with one of them)."""
    return str(author).upper().startswith(UNKNOWN_PREFIX)


def new_artist_mask(max_cos, tier, author):
    """Boolean mask of the new-artist set: nearest cosine < 0.95 (strict, float32 as sc_check compares it), tier
    'none', author not starting with UNKNOWN."""
    max_cos = np.asarray(max_cos)
    assert max_cos.dtype == np.float32, 'the cosines are float32 (as sc_check compares them)'
    tier = np.asarray(tier)
    unknown = np.array([unknown_author(a) for a in author], dtype=bool)
    assert len(max_cos) == len(tier) == len(unknown)
    return (max_cos < pp_paths.SEMART_CUT) & (tier == 'none') & ~unknown


def recompute_tiers(authors, train_artists):
    """sc_check's name tier of each author (list, same order) against the set of train artist slugs."""
    train_artists = set(train_artists)
    train_tokens = {a: set(a.split('-')) for a in train_artists}
    memo = {}
    for a in set(map(str, authors)):
        memo[a] = sc_check.artist_tiers(a, train_artists, train_tokens)
    return [memo[str(a)] for a in authors]


def md5_step(keys, md5_of):
    """r15 D10's form: in sha256 order, keep the first key of each md5 group. Returns (kept in sha256 order, groups)
    where groups lists the md5 groups of two or more as {'md5', 'keep', 'dropped'} (dropped in sha256 order)."""
    ordered = sha_order(keys)
    missing = [k for k in ordered if k not in md5_of]
    assert not missing, f'no md5 for {len(missing)} keys, e.g. {missing[:2]}'
    first, members = {}, {}
    kept = []
    for k in ordered:
        m = md5_of[k]
        members.setdefault(m, []).append(k)
        if m not in first:
            first[m] = k
            kept.append(k)
    groups = [{'md5': m, 'keep': ks[0], 'dropped': ks[1:]} for m, ks in members.items() if len(ks) > 1]
    return kept, groups


def split_close(keys, cos_of, cut=pp_paths.SEMART_CLOSE):
    """D14's subsets of `keys`: (close, far), each in sha256 order; close = nearest cosine >= cut (inclusive,
    float32). `cos_of` maps key -> float32 nearest cosine."""
    ordered = sha_order(keys)
    cos = np.array([cos_of[k] for k in ordered], dtype=np.float32)
    flag = cos >= cut
    return [k for k, f in zip(ordered, flag) if f], [k for k, f in zip(ordered, flag) if not f]


def type_labels(keys, type_of):
    """int64 [n]: index of each key's type in `pp_paths.SEMART_TYPES`. Joined by key; no label comes from the key text."""
    keys = list(keys)
    missing = [k for k in keys if k not in type_of]
    assert not missing, f'{len(missing)} keys without a type, e.g. {missing[:2]}'
    bad = {type_of[k] for k in keys} - set(pp_paths.SEMART_TYPES)
    assert not bad, f'types outside SEMART_TYPES: {sorted(bad)}'
    return np.array([pp_paths.SEMART_TYPES.index(type_of[k]) for k in keys], dtype=np.int64)


def check_types(keys, index_type, csv_type):
    """The store index's type and the CSV's TYPE agree for every key (one value per painting)."""
    diff = [k for k in keys if index_type[k] != csv_type[k]]
    assert not diff, f'{len(diff)} paintings whose index type differs from the CSV TYPE, e.g. {diff[:3]}'


def inner_pairs(x, cut=pp_paths.SEMART_CUT, block=2048):
    """Pairs (i < j) of rows of `x` with cosine >= cut, and the number of rows in at least one pair (sc_check's
    count: float32, rows normalised, upper triangle)."""
    q = np.array(x, dtype=np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    n_pairs, members = 0, set()
    for start in range(0, len(q), block):
        s = q[start:start + block] @ q.T
        for r in range(s.shape[0]):
            s[r, :start + r + 1] = -1
        ii, jj = np.nonzero(s >= cut)
        n_pairs += len(ii)
        members.update((start + ii).tolist())
        members.update(jj.tolist())
    return n_pairs, len(members)


# ---------------------------------------------------------------------------------------------- real data
def _read_csv():
    csv = pd.concat([pd.read_csv(pp_paths.SEMART_DATA / f'semart_{s}.csv', sep='\t', encoding='latin-1')
                     for s in ('train', 'val', 'test')])
    assert csv.IMAGE_FILE.is_unique, 'CSV: an IMAGE_FILE listed twice'
    return csv.set_index('IMAGE_FILE')


@functools.lru_cache(maxsize=1)
def _index():
    """The Qwen store index: (image files in store order, {file: type})."""
    meta = json.loads((QWEN_STORE / 'index.json').read_text())
    files = [r['image_file'] for r in meta['images']]
    assert len(files) == len(set(files)) == meta['n_images'] == N_FILES
    return files, {r['image_file']: r['type'] for r in meta['images']}


@functools.lru_cache(maxsize=1)
def _frame():
    """sc_nearcopy.parquet joined by key to the store index and the CSV, with the tiers recomputed. Columns:
    image_file, author, type, max_cos (float32), artist_tier (parquet), tier (recomputed), new (rule mask).
    Row order = the parquet's, asserted equal to the store index's."""
    import mt_data
    df = pd.read_parquet(pp_paths.SC_NEARCOPY)
    files, index_type = _index()
    assert len(df) == N_FILES and df.image_file.is_unique and df.max_cos.dtype == np.float32
    assert set(df.image_file) == set(files) and df.image_file.tolist() == files, 'parquet rows vs store index'
    assert np.isfinite(df.max_cos.to_numpy()).all()
    csv = _read_csv()
    assert set(files) <= set(csv.index)
    csv_type = csv.TYPE.astype(str).to_dict()
    check_types(files, index_type, csv_type)
    assert (df.type.to_numpy() == np.array([index_type[k] for k in df.image_file])).all()
    train_artists = {mt_data.artist_of(n) for n in mt_data.artelingo_train_names()}
    df = df.assign(tier=recompute_tiers(df.author.tolist(), train_artists))
    df['new'] = new_artist_mask(df.max_cos.to_numpy(np.float32), df.tier.to_numpy(), df.author.to_numpy())
    return df.reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def _md5():
    """md5 of the raw bytes of `Images/<image_file>` for the gate paintings: ({key: md5})."""
    df = _frame()
    files = df.image_file[df['new']].tolist()

    def one(f):
        return hashlib.md5((IMAGES / f).read_bytes()).hexdigest()
    with ThreadPoolExecutor(8) as ex:
        return dict(zip(files, ex.map(one, files)))


@functools.lru_cache(maxsize=1)
def _sets():
    df = _frame()
    gate = sha_order(df.image_file[df['new']])
    kept, groups = md5_step(gate, _md5())
    cos = dict(zip(df.image_file, df.max_cos))
    close, far = split_close(kept, cos)
    close_gate, _ = split_close(gate, cos)
    return {'gate': gate, 'set': kept, 'close': close, 'far': far, 'groups': groups, 'close_gate': close_gate}


def keys(name):
    """The keys of `name` in {'gate', 'set', 'close', 'far'}, in sha256 order (a fresh list)."""
    assert name in KEY_SETS, f'unknown SemArt key set {name!r}; one of {KEY_SETS}'
    return list(_sets()[name])


def labels(keys_):
    """int64 [n]: the type index (position in `pp_paths.SEMART_TYPES`) of each key, joined by key. `_frame()` runs
    first, so the index-against-CSV type check has run before any label is given out."""
    _frame()
    return type_labels(keys_, _index()[1])


def _store_indexes():
    """{store folder (relative to SEMART_PRE): list of keys} for every SemArt store index found (index.json or
    keys.json), the new stores included once they exist."""
    out = {}
    for path in sorted(pp_paths.SEMART_PRE.glob('**/index.json')) + sorted(pp_paths.SEMART_PRE.glob('**/keys.json')):
        if path.name == 'index.json':
            meta = json.loads(path.read_text())
            ks = [r['image_file'] for r in meta['images']]
        else:
            ks = json.loads(path.read_text())
        out[str(path.parent.relative_to(pp_paths.SEMART_PRE))] = ks
    return out


def check_store_indexes(key_list):
    """Every key is in every SemArt store index found (C6), once. Returns the sorted list of store folders checked."""
    stores = _store_indexes()
    for name, ks in stores.items():
        assert len(ks) == len(set(ks)), f'{name}: duplicate keys in the index'
        missing = set(key_list) - set(ks)
        assert not missing, f'{name}: {len(missing)} keys missing from the store index, e.g. {sorted(missing)[:2]}'
    return sorted(stores)


def gate_vi():
    """Gate (vi): exactly 11,162 paintings by 2,717 artists with type counts equal to sc_check.json's
    `new_artist_types`; the tiers recomputed equal the parquet's; the md5 step's count recorded (11,159 expected)."""
    df, sets = _frame(), _sets()
    new = df[df['new']]
    sc = json.loads(pp_paths.SC_CHECK.read_text())
    types = {k: int(v) for k, v in Counter(new.type).most_common()}
    tiers_match = bool((df.tier == df.artist_tier).all())
    ok = (len(new) == pp_paths.SEMART_GATE['paintings'] and new.author.nunique() == pp_paths.SEMART_GATE['artists']
          and types == sc['new_artist_types'] and len(new) == sc['new_artist_kept']
          and new.author.nunique() == sc['new_artist_artists'] and tiers_match)
    return {
        'pass': bool(ok), 'paintings': int(len(new)), 'artists': int(new.author.nunique()), 'types': types,
        'expected': {**pp_paths.SEMART_GATE, 'types': sc['new_artist_types']}, 'tiers_match_parquet': tiers_match,
        'n_set': len(sets['set']), 'n_set_expected': pp_paths.N_SEMART,
        'n_set_as_expected': len(sets['set']) == pp_paths.N_SEMART, 'md5_groups': sets['groups'],
    }


def _qwen_rows(key_list):
    """float32 rows of the default Qwen store for `key_list`, joined by key through the store index."""
    files, _ = _index()
    pos = {k: i for i, k in enumerate(files)}
    x = np.load(QWEN_STORE / 'img.npy', mmap_mode='r')
    assert x.shape == (N_FILES, 2048)
    return np.asarray(x[[pos[k] for k in key_list]], dtype=np.float32)    # inner_pairs normalises once


def record():
    """JSON-able record of the set (for records/semart_set.json): gate, counts, md5 groups, D14 subsets, inner
    pairs, the stores whose indexes hold every key, the key lists' sha256."""
    sets = _sets()
    gate = gate_vi()
    n_pairs, n_members = inner_pairs(_qwen_rows(sets['set']))

    def digest(ks):
        return hashlib.sha256('\n'.join(ks).encode('utf-8')).hexdigest()
    return {
        'rule': 'nearest cosine < 0.95 and artist tier none and author not starting with UNKNOWN (sc_check.py)',
        'gate': gate, 'n_gate': len(sets['gate']), 'n_set': len(sets['set']),
        'artists_set': int(_frame().set_index('image_file').loc[sets['set'], 'author'].nunique()),
        'md5_groups': sets['groups'], 'n_md5_dropped': len(sets['gate']) - len(sets['set']),
        'close_before_md5': len(sets['close_gate']), 'close': len(sets['close']), 'far': len(sets['far']),
        'close_cut': pp_paths.SEMART_CLOSE,
        'set_types': {k: int(v) for k, v in Counter(
            _frame().set_index('image_file').loc[sets['set'], 'type']).most_common()},
        'inner_pairs_ge_0.95': {'pairs': int(n_pairs), 'paintings': int(n_members), 'rows': 'Qwen default image',
                                'n': len(sets['set'])},
        'stores_checked': check_store_indexes(sets['gate']),
        'keys_sha256': {n: digest(sets[n]) for n in KEY_SETS},
    }


def main(argv):
    assert len(argv) == 1 and argv[0] in ('gate', 'record'), 'usage: pp_semart.py gate | record'
    print(json.dumps(gate_vi() if argv[0] == 'gate' else record(), indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
