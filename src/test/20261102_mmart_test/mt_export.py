"""r15 c* exports on MMArt, four encoders, and stage 0 gate (ii) (ticket 02; spec §3.2 X-c*, §3.5 0(ii), §3.7).

c* is r10's arm `k11=40`: Qwen's own checkpoints (r10 `full`), and for SigLIP 2, SigLIP 1 and CLIP L/14 the X-c* checkpoints
r14 trained (`mt_paths.CSTAR_ROOT`). Nothing is retrained and no inference code is rewritten: the image side of each head
runs through r14's `bs_cstar._infer_sides` (the function that made r14's held exports; batches of 512, float16, the saved
best view), on CPU, on the encoder's default image rows `mt_data.features(view, keys)` (float32, L2-normalised as
`rb_encoders.full_images` does), after `bs_cstar._check_weights` (checkpoint checksum, aspect, seed, encoder, width, arm).

    export(enc, keysets, seeds, root)   OUT/exports/<keyset>/<enc>/cstar/<aspect>__s<seed>.{npz,json}; keysets val, new,
                                        shared_held; npz keys `keys` (str, in `mt_data.order`) and `emb` (float16 [n, 2048])
    load(keyset, enc, aspect, seed, keys, root=None)
                                        float16 rows in the order of `keys`, joined by key, every key present; refuses a file
                                        whose sidecar sha256 is not the file's, whose sidecar names another encoder, aspect,
                                        seed, keyset or view, or whose checkpoint (path, sha256) is not the current one
    keyset_rows(enc, keyset)            (keys, float32 rows): what `export` feeds the heads; the gate's own-path items use it too
    gate_ii(enc)                        stage 0 gate (ii), 21 items, each at most 1e-3 (inclusive), `pass` needs all 21:
                                        `held_export_held` (15: 5 seeds x 3 aspects): r14's own export path
                                          (`bs_cstar.export_held`, device cpu, images from `rb_encoders.full_images`) on the
                                          12,281 ArtELingo held paintings against the stored held exports of the checkpoints
                                          (r10's for Qwen, r14's for the others); this does not test `export`'s code;
                                        `own_path_shared_held` (3: seed 0, 3 aspects): `keyset_rows` + `infer`, the code
                                          `export` runs, on the 11,528 shared_held keys against the same stored held exports;
                                        `own_path_val` (3: seed 0, 3 aspects): the same on the 5,769 val keys against the stored
                                          validation exports (`img` side, r10's `train/<arm>` file of the checkpoint root).
                                        MMArt keys are joined to the stored order by `mt_data.artelingo_name`. The stored files
                                        carry no painting names: the held files' row order is the list whose sha256 their record
                                        names, asserted equal to the held list's, so the join is positional under that assertion;
                                        the validation files (the trainer's own, no list hash) are in the trainer's dev order
                                        (`ct2_data.split2('full')`, 6,152 rows, asserted), which r14's validation export reuses;
                                        their sidecar's `output_sha256` is checked against the file.
                                        Each run writes its held export to a fresh `OUT/stage0/gate_ii/<enc>/<run>/` folder;
                                        the JSON says per item whether the file was reused or recomputed.
                                        Writes OUT/stage0/gate_ii_<enc>.json

The sidecar `.json` holds `sha256` (of the npz itself; ticket 04's ranges file copies it), `checkpoint` and
`checkpoint_sha256`, `view`, `n`, `width`, `encoder`, `aspect`, `seed`, `keyset`, `keys_sha256`, `device`, `code_sha256`.

CLI (everything on the local CPU, no GPU):
    python mt_export.py export  --enc E|all [--keysets val new shared_held] [--seeds 0 1 2 3 4] [--out DIR]
    python mt_export.py gate-ii --enc E|all [--out DIR]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mt_paths as P  # noqa: E402
import mt_data  # noqa: E402

if str(P.BS_ROUND) not in sys.path:
    sys.path.insert(0, str(P.BS_ROUND))                       # older round's code, read only
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')    # as bs_cstar's CLI; unused on CPU
import bs_cstar  # noqa: E402  (r14: the c* trainer's exports; puts r10 and the older rounds on sys.path)

KEYSETS = ('val', 'new', 'shared_held')
TOL = bs_cstar.TOL_EXPORT                                      # 1e-3 (r12's bar, r14's 0(ix) and spec §3.5 0(ii))
DEVICE = 'cpu'
EMB_WIDTH = 2048


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def keys_sha256(keys):
    return hashlib.sha256('\n'.join(keys).encode('utf-8')).hexdigest()


# ------------------------------------------------------------------------------------------------ files
def checkpoint_path(enc, aspect, seed):
    """`<CSTAR_ROOT[enc]>/weights/k11=40/<aspect>__s<seed>.pt` (r10's layout; r14's X-c* use it too)."""
    assert enc in P.ENCODERS and aspect in P.ASPECTS and seed in P.SEEDS, (enc, aspect, seed)
    return bs_cstar.rb_student.weights_path(P.CSTAR_ROOT[enc], P.CSTAR_ARM, aspect, seed)


def export_path(root, keyset, enc, aspect, seed):
    assert keyset in KEYSETS, f'keyset {keyset!r}: exports exist for {KEYSETS}'
    return Path(root) / 'exports' / keyset / enc / 'cstar' / f'{aspect}__s{seed}.npz'


def write_export(root, keyset, enc, aspect, seed, keys, emb, *, checkpoint, view):
    """Save `emb` (float16 [n, 2048], rows in the order of `keys`) and its sidecar (pending files first, then renamed).
    `keys` must be unique and in `mt_data.order`. The sidecar records the sha256 of the npz, the checkpoint path and the
    sha256 of the checkpoint file as it is now."""
    keys = [str(k) for k in keys]
    assert keys and len(set(keys)) == len(keys) and keys == mt_data.order(keys), 'keys must be in mt_data.order (D9)'
    assert emb.dtype == np.float16 and emb.ndim == 2 and emb.shape[0] == len(keys), (emb.dtype, emb.shape, len(keys))
    assert np.isfinite(emb).all(), 'non-finite rows'
    checkpoint = Path(checkpoint)
    path = export_path(root, keyset, enc, aspect, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.stem + '.pending.npz')
    np.savez(pending, keys=np.asarray(keys), emb=emb)
    pending.replace(path)
    meta = dict(sha256=sha_file(path), n=len(keys), width=int(emb.shape[1]), encoder=enc, aspect=aspect, seed=int(seed),
                keyset=keyset, view=view, checkpoint=str(checkpoint), checkpoint_sha256=sha_file(checkpoint),
                keys_sha256=keys_sha256(keys), device=DEVICE, code_sha256=sha_file(__file__))
    pending = path.with_name(path.stem + '.pending.json')
    pending.write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
    pending.replace(path.with_suffix('.json'))
    return path


def load(keyset, enc, aspect, seed, keys, root=None):
    """float16 [len(keys), 2048], rows in the order of `keys`, joined by key (every key must be in the file). Refuses a stale
    or foreign file: the sidecar's sha256 must be the npz's, the sidecar must name this keyset, encoder, aspect, seed and
    the encoder's default view, and its checkpoint (path and sha256) must be the current checkpoint of (enc, aspect, seed)."""
    path = export_path(P.OUT if root is None else root, keyset, enc, aspect, seed)
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['sha256'] == sha_file(path), f'{path}: sha256 differs from its sidecar'
    for field, want in (('keyset', keyset), ('encoder', enc), ('aspect', aspect), ('seed', seed),
                        ('view', P.DEFAULT_VIEW[enc])):
        assert meta[field] == want, f'{path}: sidecar {field} {meta[field]!r}, wanted {want!r}'
    ckpt = checkpoint_path(enc, aspect, seed)
    assert meta['checkpoint'] == str(ckpt) and meta['checkpoint_sha256'] == sha_file(ckpt), \
        f'{path}: stale export, its checkpoint is not the current {ckpt} (sidecar checkpoint {meta["checkpoint"]}, sha256 differs)'
    with np.load(path, allow_pickle=False) as z:
        assert sorted(z.files) == ['emb', 'keys'], f'{path}: arrays {z.files}'
        have, emb = z['keys'].tolist(), z['emb']
    assert emb.dtype == np.float16 and emb.shape == (len(have), meta['width']) and len(have) == meta['n'], path
    where = {k: i for i, k in enumerate(have)}
    assert len(where) == len(have), f'{path}: duplicate keys'
    keys = [str(k) for k in keys]
    missing = [k for k in keys if k not in where]
    assert not missing, f'{path}: {len(missing)} keys not exported, e.g. {missing[:2]}'
    return emb[[where[k] for k in keys]]


# ------------------------------------------------------------------------------------------------ inference
def infer(enc, aspect, seed, x):
    """The image side of one saved head on `x` (float32 [n, width]) on CPU, by r14's inference (`bs_cstar._infer_sides`
    after `_check_weights`), inside r14's export context (`training_context(export=True)`). Returns (float16 [n, 2048],
    sha256 of the checkpoint file used)."""
    with threadpool_limits(limits=8), bs_cstar.training_context(enc, bs_cstar.REF, export=True) as ctx:
        weights, _, sha = bs_cstar._check_weights(P.CSTAR_ROOT[enc], ctx.arm, aspect, seed, ctx.backbone, False)
        assert Path(weights) == checkpoint_path(enc, aspect, seed)
        assert x.shape[1] == ctx.width == P.WIDTH[enc], (x.shape, ctx.width)
        emb = bs_cstar._infer_sides(weights, aspect, ctx.width, {'img': x}, DEVICE)['img']
    problem = bs_cstar.rg_full.vc3_problem(emb, n_rows=len(x))       # rows, float16, finite, unit norm (r14's held check)
    assert problem is None, f'{enc} {aspect} s{seed}: {problem}'
    return emb, sha


def keyset_rows(enc, keyset):
    """(keys in `mt_data.order`, float32 [n, width]): the encoder's default image rows of one keyset, as `export` and the
    gate's own-path items feed them to the heads."""
    keys = mt_data.keys(keyset)
    assert keys == mt_data.order(keys), f'{keyset}: keys are not in mt_data.order'
    x = mt_data.features(P.DEFAULT_VIEW[enc], keys)
    assert x.dtype == np.float32 and x.shape == (len(keys), P.WIDTH[enc]), (x.dtype, x.shape)
    return keys, x


def export(enc, keysets=KEYSETS, seeds=P.SEEDS, root=None, log=print):
    """Run the saved c* heads of `enc` on the default image of each keyset (`mt_data.features`), seeds x three aspects, and save
    them. Returns the written paths."""
    root = Path(P.OUT if root is None else root)
    view = P.DEFAULT_VIEW[enc]
    paths = []
    for keyset in keysets:
        keys, x = keyset_rows(enc, keyset)
        for seed in seeds:
            for aspect in P.ASPECTS:
                start = time.monotonic()
                emb, sha = infer(enc, aspect, seed, x)
                assert sha == sha_file(checkpoint_path(enc, aspect, seed)) and emb.shape == (len(keys), EMB_WIDTH)
                paths.append(write_export(root, keyset, enc, aspect, seed, keys, emb, checkpoint=checkpoint_path(enc, aspect, seed),
                                          view=view))
                log(f'{enc} {keyset} {aspect} s{seed}: n={len(keys)} {time.monotonic() - start:.1f}s')
    return paths


# ------------------------------------------------------------------------------------------------ gate (ii)
def compare_by_name(names_a, a, names_b, b, tol=TOL):
    """Max absolute difference of two float16 export arrays joined by painting name (rows of `b` are put in the order of
    `names_a`; unique names, the same set). pass = every difference finite and at most `tol` (inclusive)."""
    names_a, names_b = [str(n) for n in names_a], [str(n) for n in names_b]
    assert len(set(names_a)) == len(names_a) and len(set(names_b)) == len(names_b), 'duplicate painting names'
    assert set(names_a) == set(names_b), 'the two exports cover other paintings'
    assert a.shape == (len(names_a),) + a.shape[1:] and b.shape == (len(names_b),) + a.shape[1:], (a.shape, b.shape)
    where = {n: i for i, n in enumerate(names_b)}
    diff = np.abs(a.astype(np.float64) - b[[where[n] for n in names_a]].astype(np.float64))
    finite = bool(np.isfinite(diff).all())
    biggest = float(diff.max()) if finite else None
    return {'pass': bool(finite and biggest <= tol), 'max_abs_diff': biggest, 'rows': len(names_a)}


def compare_to_stored(names, ours, stored_names, stored, tol=TOL):
    """Like `compare_by_name`, for `names` (unique) a subset of `stored_names` (unique): row i of `ours` against the stored row
    of painting `names[i]`."""
    names, stored_names = [str(n) for n in names], [str(n) for n in stored_names]
    assert len(set(names)) == len(names) and len(set(stored_names)) == len(stored_names), 'duplicate painting names'
    where = {n: i for i, n in enumerate(stored_names)}
    missing = [n for n in names if n not in where]
    assert not missing, f'{len(missing)} paintings not in the stored export, e.g. {missing[:2]}'
    assert stored.shape[0] == len(stored_names) and ours.shape == (len(names),) + stored.shape[1:], (ours.shape, stored.shape)
    return compare_by_name(names, ours, names, stored[[where[n] for n in names]], tol)


GATE_PATHS = {'held_export_held': P.SEEDS, 'own_path_shared_held': (0,), 'own_path_val': (0,)}   # path -> seeds
GATE_ITEMS = [f'{path}/{a}__s{s}' for path, seeds in GATE_PATHS.items() for s in seeds for a in P.ASPECTS]


def gate_result(enc, items, tol=TOL):
    """The gate dict from per-item results `{'<path>/<aspect>__s<seed>': {'pass', 'max_abs_diff'}}`; all 21 items (`GATE_ITEMS`:
    15 held_export_held, 3 own_path_shared_held, 3 own_path_val) must be present; pass only if every item passes."""
    assert sorted(items) == sorted(GATE_ITEMS), f'gate (ii) needs the {len(GATE_ITEMS)} items {sorted(GATE_ITEMS)}, got {sorted(items)}'
    diffs = [items[k]['max_abs_diff'] for k in GATE_ITEMS]
    failed = [k for k in GATE_ITEMS if not items[k]['pass']]
    return dict(encoder=enc, tol=tol, passed=not failed, max_abs_diff=None if None in diffs else max(diffs),
                failed=failed, items={k: items[k] for k in GATE_ITEMS}, **{'pass': not failed})


def gate_ii(enc, out_root=None, write=True, run=None):
    """Stage 0 gate (ii) for `enc`, 21 items (see the module doc): r14's export path on the ArtELingo held paintings
    (`held_export_held`) and `export`'s own path (`keyset_rows` + `infer`) on shared_held and val (`own_path_*`), all on CPU
    from the stored checkpoints, against the stored held (r10's for Qwen, r14's for the others) and validation exports.
    The stored files carry no names, so the join to stored rows is positional: under the asserted `list_sha256` for the held
    files, and by the trainer's dev order (6,152 rows asserted, `output_sha256` checked) for the validation files; MMArt keys
    reach it through `mt_data.artelingo_name`. The held export goes to a fresh folder `<run>` (default a timestamp) under
    OUT/stage0/gate_ii/<enc>/; each `held_export_held` item records `held_export_reused` (True: `bs_cstar._export` found its
    file there, checked its metadata and did not run inference)."""
    out_root = Path(P.OUT if out_root is None else out_root)
    root = Path(P.CSTAR_ROOT[enc])
    names, names_sha = bs_cstar.rb_student.held_names()
    run = run or time.strftime('run_%Y%m%d_%H%M%S')
    work = out_root / 'stage0' / 'gate_ii' / enc / run
    held_file = lambda aspect, seed: work / 'held' / P.CSTAR_ARM / f'{aspect}__s{seed}.npz'
    reused = {(a, s): held_file(a, s).exists() for s in P.SEEDS for a in P.ASPECTS}
    with threadpool_limits(limits=8):
        got = bs_cstar.export_held(enc, bs_cstar.REF, list(P.SEEDS), root, device=DEVICE, out_root=work)
    items = {}
    for seed in P.SEEDS:
        for aspect in P.ASPECTS:
            stored_path = root / 'held' / P.CSTAR_ARM / f'{aspect}__s{seed}.npz'
            stored, info = bs_cstar.rb_student.read_outputs(stored_path)
            assert info['list_sha256'] == names_sha, f'{stored_path}: stored list differs from the held list'
            ours_info = json.loads(held_file(aspect, seed).with_suffix('.json').read_text())
            res = compare_by_name(names, got[aspect, seed]['img'], names, stored['img'])   # positional under the sha assert
            res['weights_sha256_equal'] = bool(ours_info['weights_sha256'] == info['weights_sha256'])
            res['weights_sha256'] = ours_info['weights_sha256']
            res['export_device'] = ours_info['export_device']
            res['stored_device'] = info.get('export_device', info.get('device'))
            res['held_export_reused'] = reused[aspect, seed]
            items[f'held_export_held/{aspect}__s{seed}'] = res
    # export's own path: keyset_rows (mt_data.features rows, key order) + infer, MMArt key -> painting by artelingo_name
    split = bs_cstar.ct2_data.split2('full')
    dev_names = [str(n) for n in split['all_painting_ids'][split['dev']]]
    for path, keyset, ref_names, ref_sha in (('own_path_shared_held', 'shared_held', names, names_sha),
                                              ('own_path_val', 'val', dev_names, None)):
        keys, x = keyset_rows(enc, keyset)
        painting = [mt_data.artelingo_name(k) for k in keys]
        for seed in GATE_PATHS[path]:
            for aspect in P.ASPECTS:
                stored_path = (root / 'held' / P.CSTAR_ARM / f'{aspect}__s{seed}.npz' if keyset == 'shared_held'
                               else bs_cstar.paths_of(root, P.CSTAR_ARM, aspect, seed)[1])
                if keyset == 'shared_held':
                    stored, info = bs_cstar.rb_student.read_outputs(stored_path)
                    assert info['list_sha256'] == ref_sha, f'{stored_path}: stored list differs from the list used'
                else:       # the trainer's own file (ms_train.save_output): no list hash; its sidecar has the file's sha256
                    info = json.loads(stored_path.with_suffix('.json').read_text())
                    assert info['output_sha256'] == sha_file(stored_path), f'{stored_path}: file differs from its sidecar'
                    with np.load(stored_path, allow_pickle=False) as z:
                        stored = {k: z[k] for k in z.files}
                    assert sorted(stored) == ['cap', 'img'] and stored['img'].shape == (len(ref_names), EMB_WIDTH), stored_path
                emb, sha = infer(enc, aspect, seed, x)
                res = compare_to_stored(painting, emb, ref_names, stored['img'])
                res['checkpoint_sha256'] = sha
                res['stored_file'] = str(stored_path)
                items[f'{path}/{aspect}__s{seed}'] = res
    result = gate_result(enc, items)
    result.update(held_rows=len(names), held_list_sha256=names_sha, checkpoint_root=str(root), run_folder=str(work),
                  stored_source='r10 full held' if enc == 'qwen' else 'r14 X-c* held', export_device=DEVICE,
                  weights_all_equal=all(v['weights_sha256_equal'] for k, v in items.items() if k.startswith('held_export_held/')),
                  item_paths={path: [k for k in items if k.startswith(path + '/')] for path in GATE_PATHS})
    if write:
        path = out_root / 'stage0' / f'gate_ii_{enc}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    return result


# ------------------------------------------------------------------------------------------------ command line
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('export', help='c* exports on the default image of val, new, shared_held (CPU)')
    p.add_argument('--keysets', nargs='+', choices=KEYSETS, default=list(KEYSETS))
    p.add_argument('--seeds', nargs='+', type=int, choices=P.SEEDS, default=list(P.SEEDS))
    q = sub.add_parser('gate-ii', help='stage 0 gate (ii): CPU export against the stored held exports')
    for s in (p, q):
        s.add_argument('--enc', required=True, choices=P.ENCODERS + ('all',))
        s.add_argument('--out', default=None, help='output root (default mt_paths.OUT, i.e. MT_OUTPUTS)')
    args = ap.parse_args(argv)
    encs = P.ENCODERS if args.enc == 'all' else (args.enc,)
    status = 0
    for enc in encs:
        start = time.monotonic()
        if args.cmd == 'export':
            paths = export(enc, args.keysets, args.seeds, args.out)
            print(f'{enc}: wrote {len(paths)} files in {time.monotonic() - start:.0f}s, e.g. {paths[0]}', flush=True)
        else:
            res = gate_ii(enc, args.out)
            print(f"{enc}: (ii) pass={res['pass']} max|diff|={res['max_abs_diff']} failed={res['failed']} "
                  f"weights_all_equal={res['weights_all_equal']} {time.monotonic() - start:.0f}s", flush=True)
            status |= 0 if res['pass'] else 1
    return status


if __name__ == '__main__':
    raise SystemExit(main())
