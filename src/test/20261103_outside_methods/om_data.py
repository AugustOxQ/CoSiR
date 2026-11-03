"""r18 data (ticket 06; spec §3.3, §3.4, C6): the four sets in r17's keys and orders, with every input the cells need.

Sets `artelingo` (splits `held`, `val`) and `mmart` (splits `test`, `val`). Every array follows `keys(set, split)`;
joins are by key (painting name or MMArt key), never by position across stores.

Re-exported from r17's `ns_data` (same signatures, not rewritten; `keys` asserts `om_paths.N_ROWS`):
    keys, labels, plain, cstar

New here:
    tpips(set, split, aspect, roots=None, name_of=None) -> float32 [n, 4096]
        Rows of the TPIPS factor of `aspect` (ticket 03's reader `om_tpips.load`). ArtELingo splits from the ArtELingo
        store by painting name; MMArt test from the MMArt store by MMArt key; MMArt val from the ArtELingo store by the
        painting name of each MMArt key (`mt_data.artelingo_name`, the map r15 and r17 use; every name must be an
        ArtELingo validation painting). `roots` {set: store folder} overrides `om_paths.TPIPS_STORE` (tests).
    clay(set, split, aspect, text_set, out=None) -> float32 [n, 768]
        `om_clay.vectors` of the CLIP L/14 plain rows of the split. The alignment mean runs over every row of the split
        (no label is read). Cached in OUT/clay/<set>_<split>/<text_set>/<aspect>.npz (`keys`, `emb`) with a sidecar
        .json (input, text and embedding sha256, rank, n); a cache whose input or text differs is recomputed.
    save_parts(labels, ...) and turtle_parts(set, enc, arm, k, aspect=None, method='turtle', out=None)
        Partition files OUT/parts/<method>/<set>/<enc>/<arm>/<name>.npz (`keys`, `labels` int32) and <name>.json.
        Name `k<K>__start<s>` (arm plain, starts 0 to 2) or `<aspect>__s<seed>__k<K>` (arm cstar, start 0, seeds 0 to
        4). The split is the set's held split (`HELD_SPLIT`). The reader returns int32 [3, n] (plain) or [5, 1, n]
        (c*), refuses a missing file, a sidecar of another fit, a wrong npz sha256, keys that differ from
        `keys(set, split)` in value or order, labels outside [0, K). Arm names with the method suffix
        (`plain_turtle`, `cstar_turtle`) are accepted and stored as `plain`, `cstar`.
        K: `om_paths.plain_ks(set)` (plain) or `om_paths.label_ks(set, aspect)` (c*, spec §3.3: 11 aspect-K pairs).
"""
import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np

import om_clay
import om_paths as op

import ns_data  # noqa: E402  (om_paths put r17's folder and the older rounds on sys.path)
from ns_data import keys, labels, plain, cstar  # noqa: E402,F401

FACTORS = tuple(op.TPIPS_FACTORS.values())
HELD_SPLIT = op.HELD_SPLIT
METHODS = ('turtle', 'temi')
ARMS = ('plain', 'cstar')

assert {k: v for k, v in ns_data.ns_paths.N_ROWS.items() if k[0] in op.SETS} == op.N_ROWS


def array_sha256(x):
    x = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(f'{x.dtype.str}{x.shape}'.encode())
    h.update(x.tobytes())
    return h.hexdigest()


def _atomic(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        with tmp.open('xb') as f:
            write(f)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_json(path, obj):
    _atomic(path, lambda f: f.write((json.dumps(obj, indent=1, sort_keys=True) + '\n').encode()))


def _split_check(set_, split):
    assert set_ in op.SETS and split in (HELD_SPLIT[set_], op.DEV_SPLIT), (set_, split)


# ------------------------------------------------------------------------------------------------ TPIPS
def _artelingo_name(key):
    import mt_data
    return mt_data.artelingo_name(key)


def tpips(set_, split, aspect, roots=None, name_of=None):
    """TPIPS rows of the factor of `aspect`, float32 [n, 4096] unit rows in `keys(set, split)` order."""
    _split_check(set_, split)
    assert aspect in op.ASPECTS, aspect
    import om_tpips
    factor = op.TPIPS_FACTORS[aspect]
    roots = roots or {}
    ks = keys(set_, split)
    if set_ == 'mmart' and split == op.DEV_SPLIT:
        # C6: the ArtELingo store is keyed by painting name; the MMArt key maps to its name, and only extraction
        # keys are used. MMArt validation is a subset of ArtELingo validation.
        name_of = name_of or _artelingo_name
        names = [name_of(k) for k in ks]
        val = set(keys('artelingo', op.DEV_SPLIT))
        bad = [m for m in names if m not in val]
        assert not bad, f'{len(bad)} MMArt validation names are not ArtELingo validation paintings, first {bad[:3]}'
        x = om_tpips.load('artelingo', factor, names, root=roots.get('artelingo'))
    else:
        x = om_tpips.load(set_, factor, ks, root=roots.get(set_))
    assert x.shape == (len(ks), op.TPIPS_WIDTH) and x.dtype == np.float32, (x.shape, x.dtype)
    return x


# ------------------------------------------------------------------------------------------------ CLAY
def _clay_paths(set_, split, aspect, text_set, out):
    d = Path(out if out is not None else op.OUT) / 'clay' / f'{set_}_{split}' / text_set
    return d / f'{aspect}.npz', d / f'{aspect}.json'


def clay(set_, split, aspect, text_set, out=None):
    """CLAY vectors of the CLIP L/14 rows of the split, float32 [n, 768] unit rows (see the module docstring)."""
    _split_check(set_, split)
    assert aspect in op.ASPECTS and text_set in ('A', 'B'), (aspect, text_set)
    ks = keys(set_, split)
    images = plain(set_, split, op.CLAY_ENCODER)          # every row of the split, whatever its labels
    assert images.shape == (len(ks), om_clay.D), images.shape
    text = om_clay.text_set(text_set, aspect)
    rank = op.clay_rank(text.shape[0])
    want = dict(set=set_, split=split, aspect=aspect, text_set=text_set, n=len(ks), rank=rank,
                input_sha256=array_sha256(images), text_sha256=array_sha256(text))
    npz, side_path = _clay_paths(set_, split, aspect, text_set, out)
    if npz.exists() and side_path.exists():
        side = json.loads(side_path.read_text())
        if all(side.get(k) == v for k, v in want.items()):
            assert side['emb_sha256'] == op.sha256_file(npz), f'{npz}: file differs from its sidecar sha256'
            with np.load(npz, allow_pickle=False) as z:
                assert z.files == ['keys', 'emb'] and z['keys'].tolist() == ks, f'{npz}: keys differ from keys()'
                emb = z['emb']
            assert emb.shape == (len(ks), om_clay.D) and emb.dtype == np.float32, (npz, emb.shape)
            return emb
    emb = om_clay.vectors(images, aspect, text_set)
    assert emb.shape == (len(ks), om_clay.D) and emb.dtype == np.float32 and np.isfinite(emb).all()
    _atomic(npz, lambda f: np.savez(f, keys=np.array(ks), emb=emb))
    _write_json(side_path, dict(want, emb_sha256=op.sha256_file(npz)))
    return emb


# ------------------------------------------------------------------------------------------------ partition files
def _arm(arm, method):
    base = arm[:-len(method) - 1] if arm.endswith(f'_{method}') else arm
    assert base in ARMS, f'arm {arm!r}: plain or cstar (with or without the _{method} suffix)'
    return base


def _name(arm, k, aspect, start, seed):
    return f'k{k}__start{start}' if arm == 'plain' else f'{aspect}__s{seed}__k{k}'


def _path(out, method, set_, enc, arm, name):
    return Path(out if out is not None else op.OUT) / 'parts' / method / set_ / enc / arm / f'{name}.npz'


def _check_fit(set_, enc, arm, k, aspect, start, seed):
    assert enc in op.ENCODERS, enc
    assert isinstance(k, (int, np.integer)) and k >= 2, k
    if arm == 'plain':
        assert aspect is None and seed is None, 'a plain-image fit has no aspect and no seed'
        assert start in op.TURTLE_PLAIN_STARTS, f'plain starts are {op.TURTLE_PLAIN_STARTS}, got {start}'
        assert k in op.plain_ks(set_), f'plain K {k} is not one of {op.plain_ks(set_)}'
    else:
        assert aspect in op.ASPECTS, f'a c* fit needs an aspect, got {aspect!r}'
        assert seed in op.SEEDS, f'a c* fit needs a seed in {op.SEEDS}, got {seed!r}'
        assert start == op.TURTLE_CSTAR_START, f'c* fits use start {op.TURTLE_CSTAR_START}, got {start}'
        assert k in op.label_ks(set_, aspect), f'c* K {k} is not one of {op.label_ks(set_, aspect)} for {aspect}'


def _check_labels(y, n, k, what):
    assert y.shape == (n,) and y.dtype == np.int32, (what, y.shape, y.dtype)
    assert y.min() >= 0 and y.max() < k, f'{what}: labels outside [0, {k}): {int(y.min())} to {int(y.max())}'


def save_parts(labels_, *, method, set_, enc, arm, k, aspect=None, start, seed=None, input_sha256, loss, entropy,
               n_clusters, seconds, device, out=None, overwrite=False):
    """Write one fit's partition (`labels_` int [n] in `keys(set, held split)` order) and its sidecar; returns the npz
    path. Refuses an existing file unless `overwrite`."""
    assert method in METHODS, method
    arm = _arm(arm, method)
    _check_fit(set_, enc, arm, k, aspect, start, seed)
    split = HELD_SPLIT[set_]
    ks = keys(set_, split)
    y = np.asarray(labels_)
    assert y.ndim == 1 and np.issubdtype(y.dtype, np.integer), (y.shape, y.dtype)
    y = y.astype(np.int32)
    _check_labels(y, len(ks), k, f'{method}/{set_}/{enc}/{arm}')
    path = _path(out, method, set_, enc, arm, _name(arm, k, aspect, start, seed))
    assert overwrite or not (path.exists() or path.with_suffix('.json').exists()), f'{path}: exists'
    _atomic(path, lambda f: np.savez(f, keys=np.array(ks), labels=y))
    side = dict(method=method, set=set_, split=split, enc=enc, arm=arm, aspect=aspect, K=int(k), start=int(start),
                seed=None if seed is None else int(seed), n=len(ks), sha256=op.sha256_file(path),
                input_sha256=input_sha256, loss=float(loss), entropy=float(entropy), n_clusters=int(n_clusters),
                seconds=float(seconds), device=device)
    _write_json(path.with_suffix('.json'), side)
    return path


def _read_one(path, expect, ks):
    if not (path.exists() and path.with_suffix('.json').exists()):
        raise FileNotFoundError(f'{path} (or its sidecar) is missing')
    side = json.loads(path.with_suffix('.json').read_text())
    for key, v in expect.items():
        assert side.get(key) == v, f'{path}: sidecar {key} is {side.get(key)!r}, expected {v!r}'
    assert side['n'] == len(ks), (path, side['n'])
    got = op.sha256_file(path)
    assert side['sha256'] == got, f'{path}: npz sha256 {got} differs from its sidecar {side["sha256"]}'
    with np.load(path, allow_pickle=False) as z:
        assert z.files == ['keys', 'labels'], (path, z.files)
        file_keys, y = z['keys'].tolist(), z['labels']
    assert file_keys == ks, f'{path}: keys differ from keys({expect["set"]}, {expect["split"]}) in value or order'
    _check_labels(y, len(ks), expect['K'], str(path))
    return y


def turtle_parts(set_, enc, arm, k, aspect=None, method='turtle', out=None):
    """Partitions of `method` on the set's held split: int32 [3, n] (plain: starts 0 to 2) or [5, 1, n] (c*: seeds 0
    to 4, start 0), rows in `keys(set, split)` order."""
    assert method in METHODS, method
    arm = _arm(arm, method)
    split = HELD_SPLIT[set_]
    if arm == 'plain':
        runs = [dict(start=s, seed=None) for s in op.TURTLE_PLAIN_STARTS]
    else:
        runs = [dict(start=op.TURTLE_CSTAR_START, seed=s) for s in op.SEEDS]
    ks = keys(set_, split)
    ys = []
    for r in runs:
        _check_fit(set_, enc, arm, k, aspect, r['start'], r['seed'])
        path = _path(out, method, set_, enc, arm, _name(arm, k, aspect, r['start'], r['seed']))
        expect = dict(method=method, set=set_, split=split, enc=enc, arm=arm, aspect=aspect, K=int(k), **r)
        expect['K'], expect['start'] = int(k), int(r['start'])
        ys.append(_read_one(path, expect, ks))
    y = np.stack(ys)
    return y if arm == 'plain' else y[:, None, :]
