"""r17 painting data (ticket 03; spec §3.2, §3.3, D8, D10, stage 0 (iii) class counts and (v)).

Sets `artelingo` (splits `held`, `val`) and `mmart` (splits `test`, `val`); encoders `ns_paths.ENCODERS`; aspects
`ns_paths.ASPECTS`. Every array follows `keys(set, split)`; joins are by key (painting name or MMArt key), never by
position across stores. The older loaders do the reading, so the numbers are the ones those rounds fed:

    keys(set, split)                  ArtELingo: sorted painting names (`rt_data.names`: held 12,281, val 6,152).
                                      MMArt: sha256 order of the key (`mt_data.keys`: test 16,384, val 5,769).
    labels(set, split, aspect)        int64 [n], -1 = no label. ArtELingo `rt_data.labels`; MMArt `mt_data.labels`.
    cstar(set, split, enc, aspect, seed, unit=True)
                                      float32 [n, 2048]. unit=True: unit rows as r11's `rt_data.embed('cstar')` and r15's
                                      `mt_ret.cstar_rows` (stored float16 cast to float32, divided by the float32 row norm):
                                      the RETRIEVAL form. unit=False: the stored float16 values cast to float32, not
                                      divided (norms within 1e-3 of 1): the CLUSTERING form, as r10's `rs_stage0.
                                      load_candidate`, r14's `bs_artclu.cstar_fixed` and r15's `mt_clu.cstar_fixed`.
    plain(set, split, enc)            float32 unit rows. ArtELingo: r14's `bs_artret.frozen_embed` (bit-equal to r11's
                                      `rt_data.embed('plain')` for Qwen and to r14's `bs_artclu.legacy_loader`, checked);
                                      MMArt: r15's `mt_data.features(default view, keys)`. Used by clustering and retrieval alike.
    teacher(set, split, aspect)       float32 unit rows of the Qwen image instructed for the aspect. ArtELingo:
                                      `rt_data.embed('instructed')`; MMArt: `mt_data.features(teacher view, keys)`.
    percept_probs(set, split, enc, seed)
                                      float32 [n, 40], the RAW probabilities of X's matched PercepT arm. held (ArtELingo): the
                                      stored `prob` files (key `prob`); test (MMArt): r15's CPU exports (`mt_matched.
                                      load_probs`, key `probs`); val: ticket 05's files `OUT/percept_val/<enc>/<arm>__s<seed>.npz`
                                      (`names`, `probs`), joined by painting name (MMArt validation = 5,769 of the 6,152).
                                      Retrieval divides by the float32 row norm (r11's form); clustering normalises itself.
    crl(set, split, aspect, raw=False)
                                      r11's CRL vector: `ac_crl.vectors(unit CLIP L/14 rows, rt_crl.text_array(aspect))`,
                                      float64 cosines divided by their float64 norm, one cast to float32 [n, 50]; never
                                      normalised again. raw=True: the float64 cosines (= `rt_data.raw('crl', ...)`).
    untold(set, split, enc, seed)     D8: the three heads of seed `seed`, each unit-normalised, joined (emotion, style,
                                      genre) and divided by sqrt(3): float32 [n, 6144]. Refuses heads whose key lists differ.
    gate_v(), class_counts(), class_counts_by_split()

Splits: the 'held' split of ArtELingo and the 'test' split of MMArt are the held sets of `ns_paths.HELD_SPLIT`.
"""
import contextlib
import hashlib
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_paths  # noqa: E402

ns_paths.add_rounds()
import rs_vh  # noqa: E402  (r10: the names hash)
import rt_data  # noqa: E402  (r11: ArtELingo names, labels, c*, teacher, CLIP rows)
import rt_crl  # noqa: E402  (r11: the CRL text arrays)
import ac_crl  # noqa: E402  (r12: r11's CRL vector form)
import bs_artret  # noqa: E402  (r14: frozen_embed, cstar_root)
import mt_data  # noqa: E402  (r15: MMArt keys, labels, rows)
import mt_export  # noqa: E402  (r15: MMArt c* exports)
import mt_matched  # noqa: E402  (r15: MMArt matched probabilities)

SPLITS = {'artelingo': ('held', 'val'), 'mmart': ('test', 'val')}
SQRT3 = math.sqrt(3.0)
N_HEADS = len(ns_paths.ASPECTS)
CSTAR_ROOT = {'qwen': ns_paths.R10_OUT / 'full', 'siglip2': ns_paths.R14_OUT / 'cstar/siglip2',
              'siglip1': ns_paths.R14_OUT / 'cstar/siglip1', 'clip_l14': ns_paths.R14_OUT / 'cstar/clip_l14'}
VIEW = {'qwen': 'qwen_default', 'siglip2': 'siglip2', 'siglip1': 'siglip1', 'clip_l14': 'clip_l14'}
TEACHER_VIEW = {'emotion': 'emotion_focus', 'style': 'style_word', 'genre': 'genre_word'}
EXPORT_KEYSET = {'test': 'new', 'val': 'val'}              # MMArt c* exports: test rows are a subset of `new`


def _check(set_, split):
    assert set_ in SPLITS and split in SPLITS[set_], f'{set_}/{split}: sets and splits are {SPLITS}'


def _check_args(set_, split, enc=None, aspect=None, seed=None):
    _check(set_, split)
    assert enc is None or enc in ns_paths.ENCODERS, enc
    assert aspect is None or aspect in ns_paths.ASPECTS, aspect
    assert seed is None or seed in ns_paths.SEEDS, seed


# ------------------------------------------------------------------------------------------------ keys, labels
@lru_cache(maxsize=None)
def _keys(set_, split):
    k = rt_data.names(split) if set_ == 'artelingo' else mt_data.keys(split)
    assert len(k) == len(set(k)) == ns_paths.N_ROWS[(set_, split)], (set_, split, len(k))
    if set_ == 'artelingo':
        assert k == sorted(k)
    else:
        assert k == mt_data.order(k), 'MMArt keys must be in sha256 order'
    return tuple(k)


def keys(set_, split):
    """Keys of (set, split) in the evaluation order of the set's earlier round (see the module docstring)."""
    _check(set_, split)
    return list(_keys(set_, split))


@lru_cache(maxsize=None)
def _labels(set_, split, aspect):
    n = ns_paths.N_ROWS[(set_, split)]
    if set_ == 'artelingo':
        y = np.asarray(rt_data.labels(split)[aspect], dtype=np.int64)
    else:
        y = np.asarray(mt_data.labels(list(_keys(set_, split)), aspect), dtype=np.int64)
    assert y.shape == (n,) and y.min() >= -1, (set_, split, aspect)
    assert y.max() < ns_paths.CLASS_COUNT[(set_, aspect)], (set_, split, aspect, int(y.max()))
    return y


def labels(set_, split, aspect):
    """int64 [n] class codes in `keys` order; -1 = no label (ArtELingo genre, MMArt emotion and genre)."""
    _check_args(set_, split, aspect=aspect)
    return _labels(set_, split, aspect).copy()


def _class_count(y):
    return int(len(np.unique(y[y >= 0])))


# ------------------------------------------------------------------------------------------------ rows
def _unit32(x):
    """Float32 rows divided by max(float32 row norm, 1e-12): r11's `rt_data.embed`, r15's `mt_ret.unit`."""
    x = np.asarray(x).astype(np.float32)
    assert x.ndim == 2 and np.isfinite(x).all()
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _check_rows(x, n, d, what, tol=ns_paths.TOL_UNIT):
    assert x.shape == (n, d) and x.dtype == np.float32 and np.isfinite(x).all(), (what, x.shape, x.dtype)
    dev = float(np.abs(np.linalg.norm(x.astype(np.float64), axis=1) - 1).max())
    assert dev < tol, f'{what}: rows not unit norm (max deviation {dev})'
    return x


def cstar(set_, split, enc, aspect, seed, unit=True):
    """X-c*'s head `aspect`, seed `seed`: float32 [n, 2048] (see the module docstring for unit=True or False)."""
    _check_args(set_, split, enc, aspect, seed)
    n = ns_paths.N_ROWS[(set_, split)]
    if set_ == 'artelingo':
        with bs_artret.cstar_root(CSTAR_ROOT[enc]):
            x16 = rt_data._cstar(split, aspect, seed, rt_data.names(split))     # checked: sidecar, shape, unit norm
    else:
        x16 = mt_export.load(EXPORT_KEYSET[split], enc, aspect, seed, _keys(set_, split), root=ns_paths.R15_OUT)
    assert x16.dtype == np.float16 and x16.shape == (n, ns_paths.CSTAR_WIDTH), (x16.dtype, x16.shape)
    x = _unit32(x16) if unit else np.asarray(x16).astype(np.float32)
    return _check_rows(x, n, ns_paths.CSTAR_WIDTH, f'cstar {set_}/{split}/{enc}/{aspect}/s{seed}')


def plain(set_, split, enc):
    """X's default image, float32 unit rows [n, WIDTH[enc]]."""
    _check_args(set_, split, enc)
    n = ns_paths.N_ROWS[(set_, split)]
    if set_ == 'artelingo':
        x = bs_artret.frozen_embed(VIEW[enc], split)
    else:
        x = mt_data.features(VIEW[enc], list(_keys(set_, split)))
    return _check_rows(x, n, ns_paths.WIDTH[enc], f'plain {set_}/{split}/{enc}', tol=1e-5)


def teacher(set_, split, aspect):
    """The Qwen image instructed for `aspect`, float32 unit rows [n, 2048]."""
    _check_args(set_, split, aspect=aspect)
    n = ns_paths.N_ROWS[(set_, split)]
    if set_ == 'artelingo':
        x = rt_data.embed('instructed', split, aspect)
    else:
        x = mt_data.features(TEACHER_VIEW[aspect], list(_keys(set_, split)))
    return _check_rows(x, n, 2048, f'teacher {set_}/{split}/{aspect}', tol=1e-5)


# ------------------------------------------------------------------------------------------------ PercepT
def _names_sha(names):
    return rs_vh.names_sha(names)


def _sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _check_probs(p, n, what):
    assert p.shape == (n, ns_paths.N_TOPICS) and p.dtype == np.float32 and np.isfinite(p).all(), (what, p.shape, p.dtype)
    assert ((p >= 0) & (p <= 1)).all(), f'{what}: probabilities outside [0, 1]'
    return p


def percept_val_path(enc, seed):
    return ns_paths.OUT / 'percept_val' / enc / f'{ns_paths.MATCHED[enc]}__s{seed}.npz'


def _percept_val_by_name(enc, seed):
    """Ticket 05's validation file as (names list, probs): sidecar names the npz hash; names unique."""
    path = percept_val_path(enc, seed)
    side_path = path.with_suffix('.json')
    if not path.is_file() or not side_path.is_file():
        raise FileNotFoundError(f'validation PercepT probabilities missing: {path} (ticket 05 writes them)')
    side = json.loads(side_path.read_text())
    assert _sha_file(path) in {v for v in side.values() if isinstance(v, str)}, f'{side_path}: no entry equal to the sha256 of {path.name}'
    with np.load(path, allow_pickle=False) as z:
        assert sorted(z.files) == ['names', 'probs'], (path, z.files)
        names, probs = z['names'].tolist(), z['probs']
    assert len(names) == len(set(names)) == len(probs), f'{path}: names must be unique and match the rows'
    assert 'n' not in side or side['n'] == len(names), f'{side_path}: n {side["n"]} vs {len(names)} rows'
    return names, _check_probs(probs, len(names), str(path))


def percept_probs(set_, split, enc, seed):
    """Raw matched-PercepT probabilities, float32 [n, 40], rows in `keys` order (see the module docstring)."""
    _check_args(set_, split, enc, seed=seed)
    n = ns_paths.N_ROWS[(set_, split)]
    if split == 'val':
        names, probs = _percept_val_by_name(enc, seed)
        if set_ == 'artelingo':
            want = _keys(set_, split)
        else:
            want = [mt_data.artelingo_name(k) for k in _keys(set_, split)]
        where = {nm: i for i, nm in enumerate(names)}
        missing = [w for w in want if w not in where]
        assert not missing, f'{len(missing)} validation paintings not in the PercepT file, e.g. {missing[:2]}'
        p = probs[[where[w] for w in want]]
    elif set_ == 'artelingo':
        path = mt_matched.held_prob_path(enc, seed)
        ns = rt_data.names('held')
        side = json.loads(path.with_suffix('.json').read_text())
        if 'sha256' in side:
            assert side['sha256'] == _sha_file(path), f'{path}: sidecar sha256 differs from the file'
        list_hash = side.get('list_sha256', side.get('held_sha256'))
        assert list_hash == _names_sha(ns), f'{path}: sidecar list hash is not the held list'
        with np.load(path, allow_pickle=False) as z:
            assert z.files == ['prob'], (path, z.files)
            p = z['prob']
    else:
        p = mt_matched.load_probs(enc, seed, list(_keys(set_, split)), root=ns_paths.R15_OUT)
    return _check_probs(np.ascontiguousarray(p), n, f'percept {set_}/{split}/{enc}/s{seed}')


# ------------------------------------------------------------------------------------------------ CRL
def _clip_unit(set_, split):
    if set_ == 'artelingo':
        return rt_data.embed('clip', split, 'emotion')                       # aspect-free store rows, float32 unit
    return mt_data.features('clip_l14', list(_keys(set_, split)))


def crl(set_, split, aspect, raw=False):
    """r11's CRL vector [n, 50] (float32 unit rows), or with raw=True the float64 cosines."""
    _check_args(set_, split, aspect=aspect)
    n = ns_paths.N_ROWS[(set_, split)]
    img, text = _clip_unit(set_, split), rt_crl.text_array(aspect)
    assert img.shape == (n, 768) and text.shape == (ns_paths.N_CRL_VALUES, 768)
    if raw:
        return img.astype(np.float64) @ text.astype(np.float64).T
    x = ac_crl.vectors(img, text)
    return _check_rows(x, n, ns_paths.N_CRL_VALUES, f'crl {set_}/{split}/{aspect}', tol=1e-6)


# ------------------------------------------------------------------------------------------------ untold
def ids_equal(ids):
    """True when every identity in `ids` (key lists or list hashes) equals the first."""
    return all(i == ids[0] for i in ids[1:])


def join_heads(rows, ids):
    """D8: three head matrices [n, 2048] (emotion, style, genre), each unit-normalised, joined and divided by sqrt(3).
    Refuses: not three heads, differing identities (key lists or list hashes), differing row counts, input rows not
    unit within TOL_UNIT."""
    assert len(rows) == len(ids) == N_HEADS, f'untold joins {N_HEADS} heads, got {len(rows)}'
    assert ids_equal(ids), 'the three heads do not share one key list'
    n, d = rows[0].shape
    assert all(r.shape == (n, d) for r in rows), [r.shape for r in rows]
    parts = []
    for r in rows:
        _check_rows(np.asarray(r, dtype=np.float32), n, d, 'untold head')
        parts.append(_unit32(r))
    out = np.concatenate(parts, axis=1) / np.float32(SQRT3)
    return np.ascontiguousarray(out, dtype=np.float32)


def _head_id(set_, split, enc, aspect, seed):
    """What identifies the rows of a head file: MMArt, the `keys` array inside the export; ArtELingo held, the sidecar's
    `list_sha256`; ArtELingo validation, the sidecar's validation-names hash (and its row count, checked on load)."""
    _check(set_, split)
    if set_ == 'mmart':
        path = mt_export.export_path(ns_paths.R15_OUT, EXPORT_KEYSET[split], enc, aspect, seed)
        with np.load(path, allow_pickle=False) as z:
            return z['keys'].tolist()
    root = CSTAR_ROOT[enc]
    if split == 'held':
        side = json.loads((root / f'held/k11=40/{aspect}__s{seed}.json').read_text())
        return side['list_sha256']
    side = json.loads((root / f'train/k11=40/R6-5/{aspect}__{rt_data.GRAPHS[aspect]}__s{seed}.json').read_text())
    return side['config']['names_sha256']['dev']


def untold(set_, split, enc, seed):
    """float32 [n, 6144]: X-c*'s three heads of `seed` joined (D8). Refuses heads that do not share their key list."""
    _check_args(set_, split, enc, seed=seed)
    n = ns_paths.N_ROWS[(set_, split)]
    ids = [_head_id(set_, split, enc, a, seed) for a in ns_paths.ASPECTS]
    rows = [cstar(set_, split, enc, a, seed, unit=True) for a in ns_paths.ASPECTS]
    out = join_heads(rows, ids)
    return _check_rows(out, n, ns_paths.UNTOLD_WIDTH, f'untold {set_}/{split}/{enc}/s{seed}', tol=1e-5)


# ------------------------------------------------------------------------------------------------ gates
def gate_v(encoders=ns_paths.ENCODERS, seeds=ns_paths.SEEDS):
    """Stage 0 gate (v), per encoder: MMArt (test export `new` and validation) the `keys` arrays of the three heads equal;
    ArtELingo held the sidecars' `list_sha256` equal (and equal to the held list's); ArtELingo validation the row counts
    equal the sorted validation list (and the names hash equal); stored head rows unit norm within TOL_UNIT (all heads,
    seeds and the four splits). Returns {enc: 'ok'|'failed', ..., 'evidence': {enc: {...}}}."""
    held_sha = _names_sha(rt_data.names('held'))
    val_sha = _names_sha(rt_data.names('val'))
    status, evidence = {}, {}
    for enc in encoders:
        ev = dict(mmart_keys_equal=True, artelingo_held_list_sha256_equal=True, artelingo_val_rows=None,
                  artelingo_val_names_equal=True, max_unit_deviation=0.0, files=0)
        for s in seeds:
            for split in ('test', 'val'):
                ids = [_head_id('mmart', split, enc, a, s) for a in ns_paths.ASPECTS]
                ev['mmart_keys_equal'] &= ids_equal(ids)
            held = [_head_id('artelingo', 'held', enc, a, s) for a in ns_paths.ASPECTS]
            ev['artelingo_held_list_sha256_equal'] &= ids_equal(held) and held[0] == held_sha
            val = [_head_id('artelingo', 'val', enc, a, s) for a in ns_paths.ASPECTS]
            ev['artelingo_val_names_equal'] &= ids_equal(val) and val[0] == val_sha
            for a in ns_paths.ASPECTS:
                for set_, split in (('artelingo', 'held'), ('artelingo', 'val'), ('mmart', 'test'), ('mmart', 'val')):
                    x = cstar(set_, split, enc, a, s, unit=False)            # raises on a wrong row count or width
                    ev['max_unit_deviation'] = max(ev['max_unit_deviation'],
                                                   float(np.abs(np.linalg.norm(x.astype(np.float64), axis=1) - 1).max()))
                    ev['files'] += 1
                    if (set_, split) == ('artelingo', 'val'):
                        ev['artelingo_val_rows'] = int(len(x))
        ok = (ev['mmart_keys_equal'] and ev['artelingo_held_list_sha256_equal'] and ev['artelingo_val_names_equal']
              and ev['artelingo_val_rows'] == len(rt_data.names('val')) and ev['max_unit_deviation'] < ns_paths.TOL_UNIT)
        status[enc], evidence[enc] = ('ok' if ok else 'failed'), ev
    return dict(status, evidence=evidence)


def class_counts_by_split():
    """{'<set>/<split>/<aspect>': classes with a labelled row} for the painting sets (held or test, and val)."""
    out = {}
    for set_ in ns_paths.PAINTING_SETS:
        for split in (ns_paths.HELD_SPLIT[set_], ns_paths.DEV_SPLIT):
            for a in ns_paths.ASPECTS:
                out[f'{set_}/{split}/{a}'] = _class_count(_labels(set_, split, a))
    return out


def class_counts():
    """{'<set>/<aspect>': n} on the held or test split (gate (iii): equal to `ns_paths.CLASS_COUNT`, and the validation
    split has the same count; refuses otherwise)."""
    by = class_counts_by_split()
    out = {}
    for set_ in ns_paths.PAINTING_SETS:
        for a in ns_paths.ASPECTS:
            held, val = by[f'{set_}/{ns_paths.HELD_SPLIT[set_]}/{a}'], by[f'{set_}/{ns_paths.DEV_SPLIT}/{a}']
            assert held == val, f'{set_}/{a}: held or test has {held} classes, validation {val}'
            out[f'{set_}/{a}'] = held
    return out
