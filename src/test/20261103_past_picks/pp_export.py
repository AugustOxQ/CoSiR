"""r19 c* exports: the new three on ArtELingo, Affection and MMArt, all seven on SemArt, and stage 0 gate (iii)
(ticket 06; spec §3.2, §3.5 (iii), §3.7).

Nothing is retrained and no inference code is rewritten. The exports are written by r14's and r15's own code, driven
through `pp_register` (the new three are in their tables): `bs_cstar.export_held` (ArtELingo held), `bs_photo.export_arm`
(Affection), `mt_export.export` (MMArt) and `mt_export.infer` (SemArt: r14's `_infer_sides` on CPU, batches of 512,
float16, after `bs_cstar._check_weights`). All on CPU. Input of a head = X's default image of the set, read by key
(`pp_stores.load`), unit rows.

Layout (root = `pp_paths.OUT` unless given; `<c*>` = `mt_paths.CSTAR_ROOT[enc]`)
  ArtELingo held   <c*>/held/k11=40/<aspect>__s<seed>.npz  r14's file and sidecar (`.json`: list_sha256, weights_sha256)
                   plus `<aspect>__s<seed>.pp.json`: sha256 of the npz, checkpoint path and sha256, view, n
  Affection        <root>/exports/affection/<kind>_<enc>/<split>__s<seed>.npz  (kind photo or cstar; keys, emb) with r14's
                   sidecar and a `.pp.json` (same fields as above)
  MMArt            <root>/mmart/exports/{val,new}/<enc>/cstar/<aspect>__s<seed>.{npz,json}  (r15's layout and sidecar)
  SemArt           <root>/exports/semart/<enc>/cstar/genre__s<seed>.{npz,json}: `keys` str in sha256 order, `emb` float16
                   [11159, 2048]; sidecar: sha256 of the npz, checkpoint path and sha256, view, n, keys_sha256
  validation       the ArtELingo validation export of a c* (the trainer's file under `<c*>/train/k11=40/R6-5/`)

Readers join by key and assert that every key is present; they refuse a file whose sidecar sha256 is not the file's, whose
sidecar names another encoder, aspect, seed, view or n, or whose checkpoint (path, sha256) is not the current one.

    export_artelingo(enc, seeds, out_root)       export_affection(enc, seeds, splits, kinds, root)
    export_mmart(enc, keysets, seeds, root)      export_semart(enc, seeds, root)
    load(set, enc, aspect, seed, keys)           float16 rows in the order of `keys`; sets: artelingo_held, affection_val,
                                                 affection_test, mmart_val, mmart_new, semart
    validation(enc, aspect, seed)                (names in `rt_data.names('val')` order, float32 [6152, 2048])
    gate_iii(part)                               OUT/stage0/gate_iii.json, parts `r10` (Qwen c*, 15 held exports) and `r14`
                                                 (CLIP L/14: 15 held exports and 5 Affection test exports), each within 1e-3

CLI (CPU only):
    python pp_export.py artelingo|affection|mmart --encoder E [--seeds ..]
    python pp_export.py semart --encoder E|all [--seeds ..]
    python pp_export.py gate-iii [--part r10|r14|all]
"""
import pp_paths as PP
import pp_register  # noqa: F401  (registers the new three in r14's and r15's tables)
import pp_stores

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')        # as bs_cstar's CLI; unused on CPU
import bs_artclu  # noqa: E402
import bs_cstar  # noqa: E402
import bs_photo  # noqa: E402
import mt_export  # noqa: E402
import mt_paths  # noqa: E402
import rt_data  # noqa: E402

ARM = PP.CSTAR_ARM                                  # 'k11=40'
SEEDS = PP.SEEDS
TOL = mt_export.TOL                                 # 1e-3 (r12's bar, r14's 0(ix), spec §3.5 (iii))
EMB_WIDTH = 2048
SETS = ('artelingo_held', 'affection_val', 'affection_test', 'mmart_val', 'mmart_new', 'semart')
sha_file = mt_export.sha_file
keys_sha256 = mt_export.keys_sha256


def _atomic_json(path, doc):
    path = Path(path)
    pending = path.with_name(path.stem + '.pending.json')
    pending.write_text(json.dumps(doc, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def _root(root):
    return Path(PP.OUT if root is None else root)


def checkpoint_path(enc, aspect, seed):
    """`<c*>/weights/k11=40/<aspect>__s<seed>.pt`: r10's layout, r14's X-c* use it too (`mt_paths.CSTAR_ROOT`)."""
    return mt_export.checkpoint_path(enc, aspect, seed)


def _cstar_root(enc):
    return Path(mt_paths.CSTAR_ROOT[enc])


def _pp_sidecar(npz):
    npz = Path(npz)
    return npz.with_name(npz.stem + '.pp.json')


def _write_pp_sidecar(npz, **fields):
    doc = dict(sha256=sha_file(npz), code_sha256=sha_file(__file__), device='cpu', **fields)
    _atomic_json(_pp_sidecar(npz), doc)
    return doc


# ------------------------------------------------------------------------------------------------ checks shared by readers
def _check_fields(path, meta, want):
    for field, value in want.items():
        assert meta.get(field) == value, f'{path}: sidecar {field} {meta.get(field)!r}, wanted {value!r}'


def _check_checkpoint(path, meta, ckpt, ckpt_sha=None):
    ckpt = Path(ckpt)
    sha = sha_file(ckpt) if ckpt_sha is None else ckpt_sha
    assert meta['checkpoint'] == str(ckpt) and meta['checkpoint_sha256'] == sha, \
        f'{path}: stale export, its checkpoint is not the current {ckpt} (sidecar checkpoint {meta["checkpoint"]}, sha256 differs)'


def _check_pp(path, enc, view, n, ckpt, ckpt_sha, required, **want):
    """The `.pp.json` of a held or Affection export: file sha256, view, n, checkpoint. Absent: allowed only for the older
    encoders' own files (r10, r14 wrote none) when `required` is False."""
    side = _pp_sidecar(path)
    if not side.exists():
        assert not required, f'{side}: missing (every r19 export has one)'
        return
    meta = json.loads(side.read_text())
    assert meta['sha256'] == sha_file(path), f'{path}: sha256 differs from its sidecar'
    _check_fields(path, meta, dict(encoder=enc, view=view, n=n, **want))
    _check_checkpoint(path, meta, ckpt, ckpt_sha)


def _join(path, have, emb, keys):
    keys = [str(k) for k in keys]
    assert keys and len(set(keys)) == len(keys), 'keys must be non-empty and unique'
    where = {k: i for i, k in enumerate(have)}
    assert len(where) == len(have), f'{path}: duplicate keys in the file'
    missing = [k for k in keys if k not in where]
    assert not missing, f'{path}: {len(missing)} keys not exported, e.g. {missing[:2]}'
    return emb[[where[k] for k in keys]]


# ------------------------------------------------------------------------------------------------ SemArt
def semart_path(enc, aspect, seed, root=None):
    assert aspect == 'genre', 'SemArt exports are the genre head only'
    assert enc in PP.ENCODERS and seed in SEEDS, (enc, seed)
    return _root(root) / 'exports' / 'semart' / enc / 'cstar' / f'genre__s{seed}.npz'


def semart_keys():
    """The 11,159 keys of the SemArt set in sha256 order (`pp_semart.keys('set')`)."""
    import pp_semart
    keys = pp_semart.keys('set')
    assert len(keys) == PP.N_SEMART, f'SemArt set has {len(keys)} keys, expected {PP.N_SEMART}'
    return keys


def export_semart(enc, seeds=SEEDS, root=None, *, keys=None, rows=None, log=print):
    """The genre head of X-c* (seed s) on X's default SemArt image for each seed. `keys`/`rows` replace the SemArt set and
    its store (tests). Returns the written paths."""
    keys = semart_keys() if keys is None else [str(k) for k in keys]
    assert keys and len(set(keys)) == len(keys), 'SemArt keys must be unique'
    assert keys == PP.order(keys), 'SemArt keys must be in sha256 order (pp_paths.order)'
    x = pp_stores.load('semart', enc, keys) if rows is None else rows
    assert x.dtype == np.float32 and x.shape == (len(keys), PP.WIDTH[enc]), (x.dtype, x.shape, len(keys))
    paths = []
    for seed in seeds:
        start = time.monotonic()
        emb, sha = mt_export.infer(enc, 'genre', seed, x)
        ckpt = checkpoint_path(enc, 'genre', seed)
        assert sha == sha_file(ckpt) and emb.shape == (len(keys), EMB_WIDTH)
        path = semart_path(enc, 'genre', seed, root)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                same = z['keys'].tolist() == keys and np.array_equal(z['emb'], emb)
            assert same, f'{path}: exists with other content (move it aside to re-export)'
        else:
            pending = path.with_name(path.stem + '.pending.npz')
            np.savez(pending, keys=np.asarray(keys), emb=emb)
            pending.replace(path)
        _atomic_json(path.with_suffix('.json'), dict(
            sha256=sha_file(path), n=len(keys), width=int(emb.shape[1]), encoder=enc, aspect='genre', seed=int(seed),
            set='semart', view=enc, checkpoint=str(ckpt), checkpoint_sha256=sha, keys_sha256=keys_sha256(keys),
            device='cpu', code_sha256=sha_file(__file__)))
        paths.append(path)
        log(f'{enc} semart genre s{seed}: n={len(keys)} {time.monotonic() - start:.1f}s')
    return paths


def _load_semart(enc, aspect, seed, keys, root):
    path = semart_path(enc, aspect, seed, root)
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['sha256'] == sha_file(path), f'{path}: sha256 differs from its sidecar'
    _check_fields(path, meta, dict(encoder=enc, aspect=aspect, seed=seed, view=enc, set='semart'))
    ckpt = checkpoint_path(enc, aspect, seed)
    _check_checkpoint(path, meta, ckpt)
    with np.load(path, allow_pickle=False) as z:
        assert sorted(z.files) == ['emb', 'keys'], f'{path}: arrays {z.files}'
        have, emb = z['keys'].tolist(), z['emb']
    assert emb.dtype == np.float16 and emb.shape == (len(have), meta['width']), path
    assert len(have) == meta['n'], f'{path}: sidecar n {meta["n"]} but the file has {len(have)} rows'
    return _join(path, have, emb, keys)


# ------------------------------------------------------------------------------------------------ ArtELingo held
def held_path(enc, aspect, seed, root=None):
    """r14's held file; `root` is the folder that holds `held/` (default: the encoder's c* root)."""
    assert enc in PP.ENCODERS and aspect in PP.ASPECTS and seed in SEEDS, (enc, aspect, seed)
    return (_cstar_root(enc) if root is None else Path(root)) / 'held' / ARM / f'{aspect}__s{seed}.npz'


def export_artelingo(enc, seeds=SEEDS, out_root=None, log=print):
    """The 12,281 ArtELingo held paintings through the saved heads of `enc`: r14's `bs_cstar.export_held` on CPU. Files at
    `out_root` (default the encoder's c* root) `held/k11=40/`. `.pp.json` sidecars are added. Returns the paths."""
    root = _cstar_root(enc)
    out_root = root if out_root is None else Path(out_root)
    start = time.monotonic()
    with threadpool_limits(limits=8):
        bs_cstar.export_held(enc, bs_cstar.REF, list(seeds), root, device='cpu', out_root=out_root)
    paths = []
    for seed in seeds:
        for aspect in PP.ASPECTS:
            path = held_path(enc, aspect, seed, out_root)
            _, info = bs_cstar.rb_student.read_outputs(path)
            ckpt = checkpoint_path(enc, aspect, seed)
            ckpt_sha = sha_file(ckpt)
            assert info['weights_sha256'] == ckpt_sha and info['rows'] == len(bs_cstar.rb_student.held_names()[0])
            _write_pp_sidecar(path, encoder=enc, aspect=aspect, seed=int(seed), view=bs_cstar.backbone(enc), n=info['rows'],
                              set='artelingo_held', checkpoint=str(ckpt), checkpoint_sha256=ckpt_sha,
                              list_sha256=info['list_sha256'])
            paths.append(path)
    log(f'{enc} artelingo held: {len(paths)} files {time.monotonic() - start:.0f}s')
    return paths


def _load_held(enc, aspect, seed, keys, root, old_ok):
    path = held_path(enc, aspect, seed, root)
    arrays, info = bs_cstar.rb_student.read_outputs(path)               # array sha256 against its record
    names, names_sha = bs_cstar.rb_student.held_names()
    ckpt = checkpoint_path(enc, aspect, seed)
    ckpt_sha = sha_file(ckpt)
    assert info['list_sha256'] == names_sha and info['rows'] == len(names), f'{path}: not the held list'
    _check_fields(path, info, dict(encoder=bs_cstar.backbone(enc), aspect=aspect, seed=seed, arm=ARM))
    assert info['weights_sha256'] == ckpt_sha, f'{path}: stale export, its checkpoint is not the current {ckpt}'
    emb = arrays['img']
    assert emb.dtype == np.float16 and emb.shape == (len(names), EMB_WIDTH), path
    _check_pp(path, enc, bs_cstar.backbone(enc), len(names), ckpt, ckpt_sha,
              required=not (old_ok and enc in PP.OLD_ENCODERS), set='artelingo_held', aspect=aspect, seed=seed)
    return _join(path, names, emb, keys)               # the file carries no names: its rows are the held list's (list sha asserted)


# ------------------------------------------------------------------------------------------------ Affection
def export_affection(enc, seeds=SEEDS, splits=('val', 'test'), kinds=('photo', 'cstar'), root=None, cstar_root=None,
                     log=print):
    """r14's `bs_photo.export_arm('affection', '<kind>_<enc>', ...)`: the photo model and frozen X-c* emotion head on the 4,251
    validation and 8,501 test photos. Photo checkpoints are read from `<root>/photo/<enc>/s<seed>/`, X-c* from the encoder's
    c* root. Returns the paths."""
    root = _root(root)
    cstar_root = _cstar_root(enc) if cstar_root is None else Path(cstar_root)
    paths = []
    for kind in kinds:
        assert kind in ('photo', 'cstar'), kind
        arm = f'{kind}_{enc}'
        start = time.monotonic()
        with threadpool_limits(limits=8):
            written = bs_photo.export_arm('affection', arm, list(splits), list(seeds), root=root, cstar_root=cstar_root)
        for (split, seed), path in zip([(sp, s) for sp in splits for s in seeds], written):
            ckpt, _ = bs_photo.arm_checkpoint(arm, seed, root, cstar_root)
            ckpt_sha = bs_photo.checkpoint_sha(arm, seed, root, cstar_root)
            side = json.loads(path.with_suffix('.json').read_text())
            assert side['checkpoint_sha256'] == ckpt_sha and side['checkpoint'] == str(ckpt)
            _write_pp_sidecar(path, encoder=enc, arm=arm, split=split, seed=int(seed), view=bs_photo.VIEW[enc],
                              n=side['rows'], set=f'affection_{split}', checkpoint=str(ckpt), checkpoint_sha256=ckpt_sha)
            paths.append(path)
        log(f'{enc} affection {arm}: {len(written)} files {time.monotonic() - start:.0f}s')
    return paths


def _load_affection(set_, enc, seed, keys, root, kind, cstar_root, old_ok):
    split = set_.partition('_')[2]
    arm = f'{kind}_{enc}'
    root = _root(root)
    cstar_root = _cstar_root(enc) if cstar_root is None else Path(cstar_root)
    path = bs_photo.export_path('affection', arm, split, seed, root)
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['sha256'] == sha_file(path), f'{path}: sha256 differs from its sidecar'
    _check_fields(path, meta, dict(dataset='affection', arm=arm, split=split, seed=seed))
    ckpt, _ = bs_photo.arm_checkpoint(arm, seed, root, cstar_root)
    ckpt_sha = bs_photo.checkpoint_sha(arm, seed, root, cstar_root)
    _check_checkpoint(path, meta, ckpt, ckpt_sha)
    with np.load(path, allow_pickle=False) as z:
        assert sorted(z.files) == ['emb', 'keys'], f'{path}: arrays {z.files}'
        have, emb = z['keys'].tolist(), z['emb']
    assert emb.dtype == np.float16 and emb.shape == (len(have), EMB_WIDTH) and len(have) == meta['rows'], path
    assert bs_photo.rs_vh.names_sha(have) == meta['keys_sha256'], f'{path}: keys differ from the sidecar'
    _check_pp(path, enc, bs_photo.VIEW[enc], len(have), ckpt, ckpt_sha, required=not (old_ok and enc in PP.OLD_ENCODERS),
              set=set_, arm=arm, split=split, seed=seed)
    return _join(path, have, emb, keys)


# ------------------------------------------------------------------------------------------------ MMArt
def export_mmart(enc, keysets=('val', 'new'), seeds=SEEDS, root=None, log=print):
    """r15's `mt_export.export` on CPU: `val` (5,769) and `new` (16,782) keys, `<root>/mmart/exports/<keyset>/<enc>/cstar/`."""
    out = _root(root) / 'mmart'
    return mt_export.export(enc, list(keysets), list(seeds), out, log=log)


# ------------------------------------------------------------------------------------------------ readers
def load(set_, enc, aspect, seed, keys, root=None, *, kind='photo', cstar_root=None, old_ok=True):
    """float16 [len(keys), 2048], rows in the order of `keys`, joined by key; every key must be exported. `set_` is one of
    `SETS`. `root`: the output folder (default `pp_paths.OUT`; for `artelingo_held` the folder that holds `held/`, default
    the encoder's c* root). Affection: `kind` photo or cstar, aspect `emotion`. `old_ok`: a held or Affection file of an older
    encoder (r10, r14) may lack the `.pp.json` sidecar."""
    assert set_ in SETS, f'set {set_!r}: one of {SETS}'
    assert enc in PP.ENCODERS and aspect in PP.ASPECTS and seed in SEEDS, (enc, aspect, seed)
    if set_ == 'semart':
        return _load_semart(enc, aspect, seed, keys, root)
    if set_ in ('mmart_val', 'mmart_new'):
        return mt_export.load(set_.partition('_')[2], enc, aspect, seed, keys, root=_root(root) / 'mmart')
    if set_ == 'artelingo_held':
        return _load_held(enc, aspect, seed, keys, root, old_ok)
    assert aspect == 'emotion', 'Affection exports are the emotion head'
    return _load_affection(set_, enc, seed, keys, root, kind, cstar_root, old_ok)


def validation(enc, aspect, seed, root=None):
    """The ArtELingo validation export (image side) of the c* head: (names in `rt_data.names('val')` order, float32
    [6152, 2048]), from the trainer's file `<c*>/train/k11=40/R6-5/<aspect>__<graph>__s<seed>.npz` (r10's for Qwen, r14's or
    this round's for the others), read by r11's `rt_data._cstar` (sidecar, names hash, file hash and unit norm checked)."""
    assert enc in PP.ENCODERS and aspect in PP.ASPECTS and seed in SEEDS, (enc, aspect, seed)
    names = rt_data.names('val')
    with bs_artclu.cstar_root(_cstar_root(enc) if root is None else root):
        x = rt_data._cstar('val', aspect, seed, names)
    assert x.shape == (len(names), EMB_WIDTH)
    return names, x.astype(np.float32)


# ------------------------------------------------------------------------------------------------ gate (iii)
GATE_ITEMS = {'r10': [f'artelingo_held/{a}__s{s}' for s in SEEDS for a in PP.ASPECTS]}
GATE_ITEMS['r14'] = GATE_ITEMS['r10'] + [f'affection_test/emotion__s{s}' for s in SEEDS]
GATE_ENC = {'r10': 'qwen', 'r14': 'clip_l14'}
FROZEN_OUTPUTS = (PP.R10_OUT, PP.R11_OUT, PP.R12_OUT, PP.R13_OUT, PP.R14_OUT, PP.R15_OUT)


def compare(ours, stored, tol=TOL):
    """Max absolute difference of two float16 arrays of one shape: pass = every difference finite and at most `tol`
    (inclusive)."""
    assert ours.shape == stored.shape and ours.dtype == stored.dtype, (ours.shape, stored.shape, ours.dtype, stored.dtype)
    diff = np.abs(ours.astype(np.float64) - stored.astype(np.float64))
    finite = bool(np.isfinite(diff).all())
    biggest = float(diff.max()) if finite else None
    return {'pass': bool(finite and biggest <= tol), 'max_abs_diff': biggest, 'rows': int(len(ours))}


def gate_part(part, items, tol=TOL):
    """One part of the gate from per-item results `{'<set>/<aspect>__s<seed>': {'pass', 'max_abs_diff', 'rows'}}`; every item of
    `GATE_ITEMS[part]` must be present; pass only if all pass."""
    want = GATE_ITEMS[part]
    assert sorted(items) == sorted(want), f'gate (iii) {part} needs the {len(want)} items {sorted(want)}, got {sorted(items)}'
    failed = [k for k in want if not items[k]['pass']]
    diffs = [items[k]['max_abs_diff'] for k in want]
    return {'part': part, 'encoder': GATE_ENC[part], 'tol': tol, 'pass': not failed, 'failed': failed,
            'max_abs_diff': None if None in diffs else max(diffs), 'items': {k: items[k] for k in want}}


def gate_work(out_root, part, enc, run=None):
    """A fresh folder for the gate's own exports, under `out_root`, never inside an older round's outputs."""
    out_root = Path(out_root).resolve()
    for frozen in FROZEN_OUTPUTS:
        assert not out_root.is_relative_to(Path(frozen).resolve()), f'{out_root} is inside {frozen}'
    return out_root / 'stage0' / 'gate_iii' / part / enc / (run or time.strftime('run_%Y%m%d_%H%M%S'))


def write_gate_part(out_root, part, doc):
    """Put `doc` under `part` in OUT/stage0/gate_iii.json (the other part is kept); top-level `pass` = all parts present pass."""
    path = Path(out_root) / 'stage0' / 'gate_iii.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    whole = json.loads(path.read_text()) if path.exists() else {}
    whole[part] = doc
    parts = [p for p in ('r10', 'r14') if p in whole]
    whole['parts'] = parts
    whole['pass'] = all(whole[p]['pass'] for p in parts)
    _atomic_json(path, whole)
    return whole


def _held_items(enc, stored_root, work):
    """The held export of `enc` from the checkpoints under `stored_root` (r14's path, CPU) against the stored held files."""
    names, names_sha = bs_cstar.rb_student.held_names()
    stored_root = Path(stored_root)
    with threadpool_limits(limits=8):
        got = bs_cstar.export_held(enc, bs_cstar.REF, list(SEEDS), stored_root, device='cpu', out_root=work)
    items = {}
    for seed in SEEDS:
        for aspect in PP.ASPECTS:
            stored_path = stored_root / 'held' / ARM / f'{aspect}__s{seed}.npz'
            stored, info = bs_cstar.rb_student.read_outputs(stored_path)
            assert info['list_sha256'] == names_sha, f'{stored_path}: stored list differs from the held list'
            ours = json.loads(held_path(enc, aspect, seed, work).with_suffix('.json').read_text())
            res = compare(got[aspect, seed]['img'], stored['img'])
            res.update(weights_sha256_equal=bool(ours['weights_sha256'] == info['weights_sha256']),
                       weights_sha256=ours['weights_sha256'], stored_file=str(stored_path),
                       stored_device=info.get('export_device', info.get('device')))
            items[f'artelingo_held/{aspect}__s{seed}'] = res
    return items


def _affection_items(enc, stored_root, work):
    """The photo model's Affection test export from the checkpoints of `stored_root` (r14's `export_checkpoint`) against the
    stored `exports/affection/photo_<enc>/test__s<seed>.npz`, joined by photo key."""
    arm = f'photo_{enc}'
    keys = bs_photo.export_keys('affection', 'test')
    x = bs_photo.images('affection', enc, keys)
    items = {}
    for seed in SEEDS:
        ckpt, width = bs_photo.arm_checkpoint(arm, seed, stored_root)
        sha = bs_photo.checkpoint_sha(arm, seed, stored_root)
        path = bs_photo.export_checkpoint('affection', arm, 'test', seed, ckpt, sha, width, keys, x, root=work)
        stored_path = bs_photo.export_path('affection', arm, 'test', seed, stored_root)
        side = json.loads(stored_path.with_suffix('.json').read_text())
        with np.load(path, allow_pickle=False) as a, np.load(stored_path, allow_pickle=False) as b:
            have, ours = a['keys'].tolist(), a['emb']
            want, theirs = b['keys'].tolist(), b['emb']
        assert have == keys and set(want) == set(have) and len(set(want)) == len(want)
        where = {k: i for i, k in enumerate(want)}
        res = compare(ours, theirs[[where[k] for k in have]])
        res.update(checkpoint_sha256=sha, checkpoint_sha256_equal=bool(side['checkpoint_sha256'] == sha),
                   stored_file=str(stored_path))
        items[f'affection_test/emotion__s{seed}'] = res
    return items


def gate_iii(part, out_root=None, run=None, log=print):
    """Stage 0 gate (iii), part `r10` (Qwen c*: the 15 held exports against r10's stored ones) or `r14` (CLIP L/14: its 15
    held exports and its 5 photo-model Affection test exports against r14's stored ones). The export code runs from the
    stored checkpoints on CPU into a fresh folder under `out_root`; a difference above 1e-3 in any item fails the part.
    Writes OUT/stage0/gate_iii.json (the other part is kept). Returns the part's dict."""
    assert part in GATE_ITEMS, part
    out_root = _root(out_root)
    enc = GATE_ENC[part]
    work = gate_work(out_root, part, enc, run)
    start = time.monotonic()
    stored_root = PP.R10_OUT / 'full' if part == 'r10' else PP.CSTAR_ROOT[enc]
    items = _held_items(enc, stored_root, work)
    if part == 'r14':
        items.update(_affection_items(enc, PP.R14_OUT, work))
    doc = gate_part(part, items)
    doc.update(run_folder=str(work), checkpoint_root=str(stored_root), export_device='cpu',
               weights_all_equal=all(v.get('weights_sha256_equal', True) and v.get('checkpoint_sha256_equal', True)
                                     for v in items.values()),
               seconds=round(time.monotonic() - start, 1))
    write_gate_part(out_root, part, doc)
    log(f'gate (iii) {part}: pass={doc["pass"]} max|diff|={doc["max_abs_diff"]} failed={doc["failed"]} {doc["seconds"]}s')
    return doc


# ------------------------------------------------------------------------------------------------ command line
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gate-iii', help='stage 0 gate (iii): the export code against r10 and r14 stored exports')
    g.add_argument('--part', choices=('r10', 'r14', 'all'), default='all')
    g.add_argument('--out', default=None, help='output root (default pp_paths.OUT, i.e. PP_OUTPUTS)')
    for name in ('artelingo', 'affection', 'mmart', 'semart'):
        p = sub.add_parser(name)
        p.add_argument('--encoder', required=True, choices=(PP.ENCODERS + ('all',)) if name == 'semart' else PP.NEW_ENCODERS)
        p.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
        p.add_argument('--out', default=None, help='output root (default pp_paths.OUT, i.e. PP_OUTPUTS)')
        if name == 'mmart':
            p.add_argument('--keysets', nargs='+', choices=('val', 'new'), default=['val', 'new'])
        if name == 'affection':
            p.add_argument('--splits', nargs='+', choices=('val', 'test'), default=['val', 'test'])
            p.add_argument('--kinds', nargs='+', choices=('photo', 'cstar'), default=['photo', 'cstar'])
    a = ap.parse_args(argv)
    if a.cmd == 'gate-iii':
        parts = ('r10', 'r14') if a.part == 'all' else (a.part,)
        return 0 if all(gate_iii(p, a.out)['pass'] for p in parts) else 1
    encs = PP.ENCODERS if a.encoder == 'all' else (a.encoder,)
    for enc in encs:
        start = time.monotonic()
        if a.cmd == 'artelingo':
            paths = export_artelingo(enc, a.seeds, None if a.out is None else Path(a.out) / 'cstar' / enc)
        elif a.cmd == 'affection':
            paths = export_affection(enc, a.seeds, a.splits, a.kinds, a.out)
        elif a.cmd == 'mmart':
            paths = export_mmart(enc, a.keysets, a.seeds, a.out)
        else:
            paths = export_semart(enc, a.seeds, a.out)
        print(f'{enc} {a.cmd}: wrote {len(paths)} files in {time.monotonic() - start:.0f}s, e.g. {paths[0]}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
