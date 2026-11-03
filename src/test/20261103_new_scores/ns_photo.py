"""r17 ticket 04: photo data (Affection and EmoSet): keys, labels, photo-model rows, plain rows, teacher rows.

Spec docs/superpowers/specs/2026-11-03-new-scores-design.md §3.2 (photo model and teacher rows), §3.3 (roles, keys, order),
stage 0 (iii) class counts. Nothing is rewritten from r12 to r14: each reader calls the function that fed the stored
numbers, and every join is by key, never by position.

  sets, splits   affection (test 8,501; val 4,251; four encoders), emoset (test 17,348; val 5,770 after r14's near-copy
                 cut; Qwen only). `split` is 'test' or 'val'.
  keys           affection: `ac_data.keys` (Python-sorted `<dataset>/<image_name>`, r12). emoset: `bs_emoset.keys(split,
                 cut=True)` (sha256 of the key string, r14 D17).
  labels         affection: `ac_data.labels` (plurality of votes, 9 classes). emoset: `bs_emoset.labels` class names, coded
                 as the position in `bs_paths.EMOSET_CLASSES` (r14's class order), 8 classes. int64, no -1.
  photo          r14's `exports/<set>/photo_<enc>/<split>__s<seed>.npz` (`keys`, `emb` float16), joined by key. Returned
                 as the stored float16 cast to float32, NOT renormalised (the form r14 fed k-means).
  plain          the encoder's default image from the stores (`bs_stores.load`, joined by key), cast to float32 and divided
                 by the row norm (r12/r14's `_unit`): unit float32.
  teacher        the Qwen image in the "artwork" wording (view `emotion_focus`), unit float32, the same rows for every
                 encoder. Wording: r12 `stage1/picks.json` affection.teacher_wording.clustering == "artwork"; r14
                 `records/stage2_emoset.json` wording.clustering == "artwork". Both are read and asserted on first use.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_paths as NP  # noqa: E402

NP.add_rounds()
for _var in ('BS_OUTPUTS', 'BS_SMOKE', 'BS_STORE_ROOT'):
    if os.environ.get(_var):
        raise RuntimeError(f'{_var} is set: r17 reads r14 as it was run, not a smoke redirect')
import bs_paths as BP  # noqa: E402
import bs_stores  # noqa: E402
import bs_emoset  # noqa: E402
import bs_emoset_score  # noqa: E402
import ac_data  # noqa: E402
import rs_vh  # noqa: E402

SPLITS = ('test', 'val')
SEEDS = tuple(NP.SEEDS)
EXPORT_ROOT = NP.R14_OUT / 'exports'
PLAIN_VIEW = {'qwen': 'qwen_default', 'siglip2': 'siglip2', 'siglip1': 'siglip1', 'clip_l14': 'clip_l14'}
TEACHER_VIEW = {'artwork': 'emotion_focus', 'photo': 'emotion_focus_photo'}     # r12/r14's names for the two wordings
R12_PICKS = NP.R12_OUT / 'stage1/picks.json'
R14_STAGE2_EMOSET = NP.MAIN_TEST / '20261102_backbones_sweep/records/stage2_emoset.json'
N_CLASSES = {'affection': 9, 'emoset': 8}


# ------------------------------------------------------------------------------------------------ checks
def _check(set_name, split, enc=None, seed=None):
    assert set_name in NP.PHOTO_SETS, f'photo sets are {NP.PHOTO_SETS}, not {set_name!r}'
    assert split in SPLITS, f"split must be one of {SPLITS} (held belongs to ArtELingo), got {split!r}"
    if enc is not None:
        assert enc in NP.PHOTO_ENCODERS[set_name], f'{set_name} is scored for {NP.PHOTO_ENCODERS[set_name]}, not {enc!r}'
    if seed is not None:
        assert seed in SEEDS and not isinstance(seed, bool), f'seed must be in {SEEDS}, got {seed!r}'


def _unit(x):
    """float32 unit rows from stored rows: cast to float32, divide by the row norm (r12's `ac_data._unit`)."""
    return ac_data._unit(x)


def join_rows(file_keys, rows, keys, what='export', exact=None):
    """`rows` re-ordered to `keys` by key. Refuses: duplicate keys in the file or in the request, a requested key the
    file lacks, a row count that differs from the key count, and (`exact`, a set of the keys the file must hold) a file
    whose key set differs from `exact`."""
    file_keys = [str(k) for k in file_keys]
    keys = list(keys)
    assert len(file_keys) == len(rows), f'{what}: {len(file_keys)} keys for {len(rows)} rows'
    assert len(set(file_keys)) == len(file_keys), f'{what}: duplicate keys in the file'
    assert keys and len(set(keys)) == len(keys), f'{what}: requested keys must be unique'
    if exact is not None:
        assert set(file_keys) == set(exact), f'{what}: the file holds a different key set than the split file'
    lookup = {k: i for i, k in enumerate(file_keys)}
    missing = [k for k in keys if k not in lookup]
    assert not missing, f'{what}: {len(missing)} requested keys are not in the file, e.g. {missing[:2]}'
    return rows[np.fromiter((lookup[k] for k in keys), dtype=np.int64, count=len(keys))]


# ------------------------------------------------------------------------------------------------ keys, labels
_KEYS = {}


def keys(set_name, split):
    """Keys of a split in the set's evaluation order (list[str]). Affection: Python-sorted; EmoSet: sha256 of the
    key, near-copies cut."""
    _check(set_name, split)
    if (set_name, split) not in _KEYS:
        if set_name == 'affection':
            out = list(ac_data.keys(split))
            assert out == sorted(out), 'affection keys must be Python-sorted'
        else:
            out = list(bs_emoset.keys(split, cut=True))
            bs_emoset_score.check_order(out)
        assert len(out) == len(set(out)) == NP.N_ROWS[(set_name, split)], (set_name, split, len(out))
        _KEYS[(set_name, split)] = out
    return list(_KEYS[(set_name, split)])


def labels(set_name, split):
    """int64 [n] class ints in `keys(set_name, split)` order. Affection: r12's plurality labels (codes alphabetical:
    amusement ... something else). EmoSet: position in r14's `bs_paths.EMOSET_CLASSES`."""
    _check(set_name, split)
    k = keys(set_name, split)
    if set_name == 'affection':
        y = np.asarray(ac_data.labels(split), dtype=np.int64)
    else:
        y = bs_emoset_score.encode_labels(bs_emoset.labels(k))
    assert y.dtype == np.int64 and y.shape == (len(k),) and y.min() >= 0 and y.max() < N_CLASSES[set_name], (y.shape,)
    return y


def class_counts():
    """{'<set>/emotion': n classes with a labelled row}. Counted on test and on val; refuses a split that differs from
    the other (the count is the same on each split, §3.3)."""
    out = {}
    for s in NP.PHOTO_SETS:
        n = {}
        for sp in SPLITS:
            y = labels(s, sp)
            n[sp] = int(np.unique(y[y >= 0]).size)
        assert n['test'] == n['val'], f'{s}: class count differs between test and val: {n}'
        out[f'{s}/emotion'] = n['test']
    return out


# ------------------------------------------------------------------------------------------------ rows
def export_path(set_name, enc, split, seed):
    return EXPORT_ROOT / set_name / f'photo_{enc}' / f'{split}__s{seed}.npz'


def _uncut_keys(set_name, split):
    """The keys an export file must hold: Affection's split keys; EmoSet's split file uncut."""
    return set(ac_data.keys(split)) if set_name == 'affection' else set(bs_emoset.keys(split, cut=False))


def photo_stored(set_name, split, enc, seed):
    """The photo model's checked float16 rows [n, 2048] in `keys(set_name, split)` order (joined by key). Asserts: the
    sidecar's sha256 equals the file's, float16 finite 2-D rows, width 2048, unit norm within `TOL_UNIT`."""
    _check(set_name, split, enc, seed)
    path = export_path(set_name, enc, split, seed)
    side = json.loads(path.with_suffix('.json').read_text())
    assert side['sha256'] == rs_vh.file_sha(path), f'{path}: file differs from its sidecar'
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'keys', 'emb'}, (path, z.files)
        file_keys, emb = z['keys'].tolist(), z['emb']
    assert emb.dtype == np.float16 and emb.ndim == 2 and emb.shape[1] == NP.CSTAR_WIDTH and np.isfinite(emb).all(), \
        (path, emb.dtype, emb.shape)
    out = join_rows(file_keys, emb, keys(set_name, split), str(path), exact=_uncut_keys(set_name, split))
    assert np.abs(np.linalg.norm(out.astype(np.float32), axis=1) - 1).max() < NP.TOL_UNIT, f'{path}: rows are not unit'
    return out


def photo(set_name, split, enc, seed):
    """Photo-model rows float32 [n, 2048]: the stored float16 cast to float32, not renormalised (r14's form for
    k-means)."""
    return photo_stored(set_name, split, enc, seed).astype(np.float32)


def plain(set_name, split, enc):
    """The encoder's default image, unit float32 [n, width] in `keys(set_name, split)` order."""
    _check(set_name, split, enc)
    x = _unit(bs_stores.load(set_name, PLAIN_VIEW[enc], keys(set_name, split)))
    assert x.shape == (NP.N_ROWS[(set_name, split)], NP.WIDTH[enc]) and x.dtype == np.float32
    return x


_WORDING = {}


def teacher_wording(set_name):
    """The picked clustering wording of a photo set, read from the earlier round's record and asserted "artwork":
    Affection from r12 `stage1/picks.json` (affection.teacher_wording.clustering), EmoSet from r14
    `records/stage2_emoset.json` (wording.clustering). Returns the view name (`emotion_focus`)."""
    if set_name not in _WORDING:
        if set_name == 'affection':
            w = json.loads(R12_PICKS.read_text())['affection']['teacher_wording']['clustering']
            views = TEACHER_VIEW
        else:
            rec = json.loads(R14_STAGE2_EMOSET.read_text())
            w, views = rec['wording']['clustering'], rec['teacher']['views']
        assert w == 'artwork' and views[w] == 'emotion_focus', (set_name, w, views)
        _WORDING[set_name] = views[w]
    return _WORDING[set_name]


def teacher(set_name, split, enc):
    """The Qwen image in the picked wording (artwork, `emotion_focus`), unit float32 [n, 2048]. The same rows for every
    encoder (r14 SC6 used r12's Qwen teacher against each); `enc` is only checked."""
    _check(set_name, split, enc)
    x = _unit(bs_stores.load(set_name, teacher_wording(set_name), keys(set_name, split)))
    assert x.shape == (NP.N_ROWS[(set_name, split)], NP.CSTAR_WIDTH) and x.dtype == np.float32
    return x
