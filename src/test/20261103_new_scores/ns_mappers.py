"""r17 ticket 05: PercepT matched mappers on CPU, validation probabilities, stage 0 gate (iv) (spec §3.7, 0(iv), D9).

The 20 saved matched mappers (`ns_paths.MATCHED`: Qwen P3, SigLIP 2 M2, SigLIP 1 M8, CLIP L/14 M7; seeds 0 to 4) run
frozen on CPU with r15's `mt_matched.infer_cpu` (the trainer's own inference: the unit-norm image row is divided by
its norm a second time, r15 code map trap 2). Nothing is retrained.

    export(out)              validation probabilities of all 20 mappers on the 6,152 ArtELingo validation paintings
                             (sorted names; the encoder's plain store, unit rows) -> OUT/percept_val/<enc>/matched__s<seed>.{npz,json}
    read_val(out, enc, seed, names=None)   (names, float32 [n, 40]); checks the sidecar; rows by name if `names` is given
    gate_iv(out)             {'status': {enc: 'ok'|'failed'}, 'evidence': {enc: {...}}}; written to OUT/stage0/gate_iv.json
                             (a) each mapper on CPU against its stored held probabilities within TOL_MAPPER;
                             (b) Qwen's validation export against r11's stored validation probabilities within TOL_MAPPER.

CLI: python ns_mappers.py export|gate [--out DIR]     (DIR defaults to ns_paths.OUT, i.e. NS_OUTPUTS)
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')       # r15's mt_matched / `base.determinism` assert it

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_paths as NP  # noqa: E402

VAL_DIR = 'percept_val'
ARM = 'matched'
N_VAL = NP.N_ROWS[('artelingo', 'val')]          # 6,152
N_TOPICS = NP.N_TOPICS


# ------------------------------------------------------------------------------------------------ small helpers
def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def names_sha(names):
    return hashlib.sha256('\n'.join(names).encode('utf-8')).hexdigest()


def max_abs_diff(a, b):
    """Max absolute difference (float64) of two same-shape arrays; inf if any entry is not finite."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.ndim == 2, (a.shape, b.shape)
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float(d.max()) if np.isfinite(d).all() else float('inf')


def _root(out):
    return Path(NP.OUT if out is None else out)


# ------------------------------------------------------------------------------------------------ the files
def val_path(out, enc, seed):
    """OUT/percept_val/<enc>/matched__s<seed>.npz (the index's `<enc>/<arm>__s<seed>.npz`, arm = matched)."""
    assert enc in NP.ENCODERS and seed in NP.SEEDS, (enc, seed)
    return _root(out) / VAL_DIR / enc / f'{ARM}__s{seed}.npz'


def mapper_weights_sha(enc, seed):
    """`weights_sha256` of the saved mapper (enc, seed), from its sidecar."""
    mm = _r15_matched()
    return json.loads(mm.mapper_path(enc, seed).with_suffix('.json').read_text())['weights_sha256']


def write_val(out, enc, seed, names, probs, *, mapper, mapper_sha256, store):
    """Save the validation probabilities (float32 [6152, 40], rows = `names`, sorted and unique) and a sidecar with
    the mapper (path, weights hash), the input store, n and the npz's sha256. Pending files first."""
    names = [str(n) for n in names]
    assert len(names) == N_VAL and names == sorted(names) and len(set(names)) == N_VAL, 'validation names'
    assert probs.dtype == np.float32 and probs.shape == (N_VAL, N_TOPICS), (probs.dtype, probs.shape)
    assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all(), 'probabilities outside [0, 1]'
    path = val_path(out, enc, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.stem + '.pending.npz')
    np.savez_compressed(pending, names=np.asarray(names), probs=probs)
    pending.replace(path)
    side = dict(encoder=enc, seed=int(seed), arm=ARM, n=N_VAL, width=N_TOPICS, device='cpu', sha256=sha_file(path),
                names_sha256=names_sha(names), mapper=str(mapper), mapper_sha256=mapper_sha256,
                input_store=str(store), code_sha256=sha_file(__file__))
    pending = path.with_name(path.stem + '.pending.json')
    pending.write_text(json.dumps(side, indent=1, allow_nan=False) + '\n')
    pending.replace(path.with_suffix('.json'))
    return path


def read_val(out, enc, seed, names=None):
    """(names, probs float32 [n, 40]) of one (enc, seed). Refuses: a missing sidecar, an npz whose sha256 differs from
    the sidecar's, a sidecar for another encoder or seed or whose mapper hash is not the current mapper's, a row count
    other than 6,152 or other than the sidecar's n, names not sorted and unique. With `names`, rows are returned in
    that order, joined by name (every name must be exported)."""
    path = val_path(out, enc, seed)
    side = json.loads(path.with_suffix('.json').read_text())
    assert side['sha256'] == sha_file(path), f'{path}: sha256 differs from its sidecar (stale sidecar)'
    assert side['encoder'] == enc and side['seed'] == seed, \
        f"{path}: sidecar is for {side['encoder']!r} seed {side['seed']!r}, wanted {enc!r} seed {seed!r}"
    assert side['mapper_sha256'] == mapper_weights_sha(enc, seed), f'{path}: sidecar mapper hash is not the current mapper\'s'
    with np.load(path, allow_pickle=False) as z:
        assert sorted(z.files) == ['names', 'probs'], z.files
        have, probs = z['names'].tolist(), z['probs']
    assert len(have) == probs.shape[0] == side['n'] == N_VAL, \
        f'{path}: rows {len(have)}/{probs.shape[0]}, sidecar n {side["n"]}, wanted {N_VAL}'
    assert probs.dtype == np.float32 and probs.shape[1] == N_TOPICS
    assert have == sorted(have) and len(set(have)) == len(have), f'{path}: names not sorted and unique'
    assert side['names_sha256'] == names_sha(have), f'{path}: names differ from the sidecar\'s names hash'
    if names is None:
        return have, probs
    where = {k: i for i, k in enumerate(have)}
    names = [str(k) for k in names]
    missing = [k for k in names if k not in where]
    assert not missing, f'{path}: {len(missing)} names not exported, e.g. {missing[:2]}'
    return names, probs[[where[k] for k in names]]


# ------------------------------------------------------------------------------------------------ r15 / r11 code
_R15 = {}


def _r15_matched():
    """r15's `mt_matched` (MAIN checkout), imported once; it brings `rb_matched`, `rb_encoders`, `ct_*`, `bs_artclu`."""
    if 'mm' not in _R15:
        NP.add_rounds()
        import mt_matched
        _R15['mm'] = mt_matched
    return _R15['mm']


def val_names():
    """The 6,152 ArtELingo validation painting names, sorted (r11's `rt_data.names('val')`)."""
    NP.add_rounds()
    import rt_data
    names = rt_data.names('val')
    assert len(names) == N_VAL == len(set(names)) and names == sorted(names)
    return names


def val_images(enc, names):
    """float32 [n, d] unit rows of the encoder's plain image store for the ArtELingo validation `names` (the
    reference store under `mp_stores.ART_ROOT`, joined by painting name; unit rows as `mt_data.features`)."""
    mm = _r15_matched()
    import mp_stores
    import mt_data
    view = mm.P.DEFAULT_VIEW[enc]
    store = mp_stores.reference_folder(view)
    x = mt_data._unit(mp_stores.join(store, names, view, reference=True))
    assert x.dtype == np.float32 and x.shape == (len(names), NP.WIDTH[enc])
    return x, store


def export(out=None, encoders=NP.ENCODERS, seeds=NP.SEEDS):
    """Run the saved mappers on the ArtELingo validation paintings and write the files. Returns the paths."""
    mm = _r15_matched()
    names = val_names()
    paths = []
    for enc in encoders:
        x, store = val_images(enc, names)
        import rb_encoders
        import eg_encoders
        eg_encoders.extend_rb_maps()
        # the rows the mappers saw in training (r15's loader), bit for bit (review of ticket 05, note 1)
        assert np.array_equal(x, rb_encoders.full_images(mm._spec(enc)['store'], names)), enc
        for seed in seeds:
            mpath = mm.mapper_path(enc, seed)
            sha = json.loads(mpath.with_suffix('.json').read_text())['weights_sha256']
            prob = mm.infer_cpu(mpath, x)
            paths.append(write_val(out, enc, seed, names, prob, mapper=mpath, mapper_sha256=sha, store=store))
    return paths


# ------------------------------------------------------------------------------------------------ gate (iv)
def judge(held, val_qwen, tol=None):
    """Gate (iv) from the measured differences. `held` {enc: {seed: max abs diff of the CPU mapper from its stored held
    probabilities}}, `val_qwen` {seed: max abs diff of Qwen's validation export from r11's stored} or None. An encoder
    is 'ok' only if all five seeds are present, finite and at most `tol` (default ns_paths.TOL_MAPPER); Qwen
    also needs its five validation differences. Returns {enc: {'status', 'evidence'}} for all four encoders."""
    tol = NP.TOL_MAPPER if tol is None else tol

    def check(diffs):
        diffs = {} if diffs is None else diffs
        failed = [s for s in NP.SEEDS if s not in diffs or not np.isfinite(diffs[s]) or not diffs[s] <= tol]
        present = [float(diffs[s]) for s in NP.SEEDS if s in diffs]
        top = max(present) if present else None
        return failed, top, {int(s): float(diffs[s]) for s in diffs}

    out = {}
    for enc in NP.ENCODERS:
        failed, top, per_seed = check(held.get(enc))
        ev = dict(tol=tol, held_max_abs_diff=top, failed_seeds=failed, held_per_seed=per_seed,
                  val_max_abs_diff=None, val_failed_seeds=[], val_per_seed=None)
        ok = not failed
        if enc == 'qwen':
            vfailed, vtop, vper = check(val_qwen)
            ev.update(val_max_abs_diff=vtop, val_failed_seeds=vfailed, val_per_seed=vper)
            ok = ok and not vfailed
        out[enc] = dict(status='ok' if ok else 'failed', evidence=ev)
    return out


def held_diffs(enc):
    """{seed: max abs diff} of the CPU mapper from its stored held probabilities (ArtELingo held, 12,281 rows; the
    images through the loader the mappers were trained with, as r15's gate (iv) a)."""
    mm = _r15_matched()
    import ct_held
    import rb_encoders
    import eg_encoders
    names, sha = ct_held.read_held_list(mm.base_held_list())
    eg_encoders.extend_rb_maps()
    images = rb_encoders.full_images(mm._spec(enc)['store'], names)
    assert len(names) == NP.N_ROWS[('artelingo', 'held')] == len(images)
    out = {}
    for s in NP.SEEDS:
        side = json.loads(mm.held_prob_path(enc, s).with_suffix('.json').read_text())
        assert side.get('list_sha256', side.get('held_sha256')) == sha, f'{enc} seed {s}: stored held list differs'
        with np.load(mm.held_prob_path(enc, s), allow_pickle=False) as z:
            stored = z['prob']
        out[s] = max_abs_diff(mm.infer_cpu(mm.mapper_path(enc, s), images), stored)
    return out


def qwen_val_diffs(out=None):
    """{seed: max abs diff} of Qwen's exported validation probabilities from r11's stored ones (`percept_val_full`)."""
    NP.add_rounds()
    import rs_vh
    names = val_names()
    res = {}
    for s in NP.SEEDS:
        stored_path = NP.R11_OUT / f'percept_val_full/P3__s{s}.npz'
        side = json.loads(stored_path.with_suffix('.json').read_text())
        assert side['val_sha256'] == rs_vh.names_sha(names) and side['rows'] == N_VAL
        with np.load(stored_path, allow_pickle=False) as z:
            stored = z['prob']
        _, ours = read_val(out, 'qwen', s, names=names)
        res[s] = max_abs_diff(ours, stored)
    return res


def gate_iv(out=None, write=True):
    """Stage 0 gate (iv). Needs the validation export in `out` (for Qwen's comparison with r11). Returns
    {'status': {enc: 'ok'|'failed'}, 'evidence': {enc: {...}}}; a missing or unreadable Qwen export fails Qwen."""
    held = {enc: held_diffs(enc) for enc in NP.ENCODERS}
    try:
        val = qwen_val_diffs(out)
        problem = None
    except (AssertionError, FileNotFoundError, KeyError) as e:
        val, problem = None, f'{type(e).__name__}: {e}'
    judged = judge(held, val)
    if problem:
        judged['qwen']['evidence']['val_problem'] = problem
    res = dict(status={e: v['status'] for e, v in judged.items()}, evidence={e: v['evidence'] for e, v in judged.items()})
    if write:
        NP.write_json(_root(out) / 'stage0/gate_iv.json', res)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', choices=('export', 'gate'))
    ap.add_argument('--out', default=None, help='output root (default ns_paths.OUT, i.e. NS_OUTPUTS)')
    args = ap.parse_args(argv)
    if args.cmd == 'export':
        paths = export(args.out)
        print(f'wrote {len(paths)} files, e.g. {paths[0]}')
    else:
        res = gate_iv(args.out)
        for enc in NP.ENCODERS:
            ev = res['evidence'][enc]
            print(f"{enc}: {res['status'][enc]}  held max|diff|={ev['held_max_abs_diff']:.3g} failed={ev['failed_seeds']}"
                  + (f"  val-vs-r11 max|diff|={ev['val_max_abs_diff']}" if enc == 'qwen' else ''))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
