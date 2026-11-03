"""r19 ticket 04: the node402 jobs of the new three (DINOv2, CLIP B/32, CLIP B/16): X-c*, X photo, gate (ii).

Thin wrapper over r14's `bs_node` (units, rerun rule) and trainers (`bs_cstar.train`, `bs_photo.train`), imported
unchanged through `pp_register`. Run from the repo root, on the node inside `scripts/run_pp_node.sh`:

    python src/test/20261103_past_picks/pp_node.py part-a --encoder E [--device cuda|cpu]
    python src/test/20261103_past_picks/pp_node.py photo --encoder E
    python src/test/20261103_past_picks/pp_node.py regress-ii [--no-compare]

    part-a --encoder E    X-c* for seeds 0 to 4 (keep all: weights, validation exports as r14's trainer writes them,
                          run__s<seed>.json) under pp_paths.CSTAR_ROOT[E] = OUT/cstar/E. Resumes: a verified seed is
                          skipped. Writes OUT/node/cstar_<E>.json (`status` ok | failed).
    photo --encoder E     X photo model for seeds 0 to 4 (train seed 100 + s, started from X-c* emotion checkpoint of
                          seed s under OUT/cstar/E; r14's data and graphs) under pp_paths.PHOTO_ROOT[E] =
                          OUT/photo/E/s<seed>. Needs E's Affection store and X-c*'s emotion weights. Writes
                          OUT/node/photo_<E>.json. A failed photo seed stops the later photo seeds only.
    regress-ii            gate (ii): r14's CLIP L/14 c* seed 0 and photo seed 0 retrained under OUT/stage0/ii (the
                          photo model starts from this run's own c* seed 0, as r14's chain did), then compared with
                          r14's stored runs: per head best epoch equal and best development loss within 1e-4 (photo:
                          its one run.json), c* validation exports (img, cap) within 1e-3. Writes
                          OUT/stage0/gate_ii.json. `--no-compare` (the node job: r14's stored runs are not on the
                          node) trains only; the comparison is the same command run on the local copy after the pull
                          (it skips the finished training and compares).

Rerun rule (r14 D18, spec §3.4): a unit that raises (a non-finite loss included) runs once more with the same seed
(`bs_node.attempt`); a second failure is recorded as failed. A failed X-c* seed stops the part (the later seeds do not
run); the run record says which. Smoke switches are r14's (`BS_NODE_SMOKE=1`: 2 epochs; needs `PP_OUTPUTS` and
`PP_SMOKE=1`); `--seeds` is for smoke only. Exit code 1 when any unit failed twice.
"""
import argparse
import json
import socket
import sys
from pathlib import Path

import pp_paths as PP                       # first: it sets BS_OUTPUTS and the other roots before r14's modules load
import pp_register                          # noqa: F401,E402  (registers the new three)
import bs_node as N                         # noqa: E402  (units, attempt = the rerun rule)
import bs_paths as P                        # noqa: E402

OUT = PP.OUT
assert N.OUT == OUT, 'bs_node.OUT must be r19\'s OUT (pp_paths sets BS_OUTPUTS first)'
REGRESS_ENC = 'clip_l14'
REGRESS_ROOT = OUT / 'stage0' / 'ii'
TOL_TRAIN = 1e-4                            # best development loss (r14's TOL_TRAIN)
TOL_EXPORT = 1e-3                           # validation exports
EXPORT_VARIANT = {'emotion': 'Gres', 'style': 'Fres05', 'genre': 'R0o'}   # r14's R6-5 file names
R14_REF = PP.R14_OUT


def seeds_of(arg):
    if arg is None:
        return tuple(PP.SEEDS)
    assert N.smoke_on(), '--seeds is for smoke runs (BS_NODE_SMOKE=1) only'
    assert set(arg) <= set(PP.SEEDS), arg
    return tuple(arg)


def write_status(name, doc):
    N.atomic_write(N.NODE / name, N.jdump(dict(doc, finished_at=N.now(), host=socket.gethostname(), smoke=N.smoke_on())))


# ------------------------------------------------------------------------------------------------------- X-c*
def unit_cstar(enc, seed, device, root=None, kind='cstar'):
    import bs_cstar
    root = Path(root or PP.CSTAR_ROOT[enc])

    def fn():
        existed = bs_cstar.record_path(root, bs_cstar.STAR_ARM, seed).is_file()      # verified weights: train() skips
        rec = bs_cstar.train(enc, 'ref', [seed], device, root, keep='all', **N.cstar_args())[seed].record
        return dict(skipped=existed, best_epoch=rec['best_epoch'], train_seconds=rec['seconds'])
    return N.attempt(f'{kind}__{enc}__s{seed}', fn, dict(kind=kind, enc=enc, seed=seed))


def part_a(enc, device, seeds):
    assert enc in PP.NEW_ENCODERS, enc
    done, failed = [], None
    for seed in seeds:
        rec = unit_cstar(enc, seed, device)
        if rec['status'] == 'failed':
            failed = dict(label=rec['label'], errors=rec['errors'])
            break
        done.append(rec['label'])
    write_status(f'cstar_{enc}.json', dict(encoder=enc, status='failed' if failed else 'ok', failed=failed, done=done,
                                           seeds=list(seeds)))
    print(f'PP_NODE part-a {enc}: {"failed" if failed else "ok"}', flush=True)
    return 1 if failed else 0


# ----------------------------------------------------------------------------------------------------- X photo
def unit_photo(enc, seed, device, pdata, root=None, cstar_root=None, kind='xphoto'):
    import bs_photo
    root = Path(root or OUT)                         # bs_photo.run_dir: <root>/photo/<enc>/s<seed> = PP.PHOTO_ROOT[enc]
    cstar_root = Path(cstar_root or PP.CSTAR_ROOT[enc])

    def fn():
        run_json = bs_photo.run_dir(enc, bs_photo.REF, seed, root) / 'run.json'
        if run_json.is_file():                       # finished: skip before the data load
            return dict(skipped=True, best_epoch=json.loads(run_json.read_text())['best_epoch'])
        existed = False
        start = bs_photo.start_root(enc, cstar_root) / f'emotion__s{seed}.pt'
        assert start.is_file(), f'X-c* emotion checkpoint of seed {seed} missing: {start}'
        data, graphs = pdata.get(enc)
        out = bs_photo.train(enc, [bs_photo.REF], [seed], device, keep='all', root=root, max_epochs=N.photo_epochs(),
                             data=data, graphs=graphs, cstar_root=cstar_root)
        rec = out[(bs_photo.REF, seed)].record
        return dict(skipped=existed, best_epoch=rec['best_epoch'], epochs=N.history_len(rec), train_seconds=rec['seconds'])
    return N.attempt(f'{kind}__{enc}__s{seed}', fn, dict(kind=kind, enc=enc, seed=seed))


def photo(enc, device, seeds):
    assert enc in PP.NEW_ENCODERS, enc
    assert PP.PHOTO_ROOT[enc] == OUT / 'photo' / enc
    pdata, done, failed = N.PhotoData(), [], None
    for seed in seeds:
        if failed:
            print(f'PP_NODE photo {enc}: s{seed} skipped (after {failed["label"]} failed)', flush=True)
            continue
        rec = unit_photo(enc, seed, device, pdata)
        if rec['status'] == 'failed':
            failed = dict(label=rec['label'], errors=rec['errors'])
            continue
        done.append(rec['label'])
    write_status(f'photo_{enc}.json', dict(encoder=enc, status='failed' if failed else 'ok', failed=failed, done=done,
                                           seeds=list(seeds)))
    print(f'PP_NODE photo {enc}: {"failed" if failed else "ok"}', flush=True)
    return 1 if failed else 0


# ------------------------------------------------------------------------------------------------------ gate (ii)
def compare_ii(root, reference=None):
    """Gate (ii) comparison: r14's stored CLIP L/14 seed 0 against the runs under `root` (`stage0/ii`). Returns the
    gate dict (`pass`, per-head training items, photo item, validation-export items)."""
    import numpy as np
    import bs_cstar
    reference = Path(reference or R14_REF)
    cref = reference / 'cstar' / REGRESS_ENC
    cstar = bs_cstar.compare_training(root / 'cstar' / REGRESS_ENC, reference=cref, arm=bs_cstar.STAR_ARM, seed=0,
                                      tol=TOL_TRAIN)
    ours = json.loads((root / 'photo' / REGRESS_ENC / 's0' / 'run.json').read_text())
    theirs = json.loads((reference / 'photo' / REGRESS_ENC / 's0' / 'run.json').read_text())
    diff = abs(ours['best_dev_loss'] - theirs['best_dev_loss'])
    photo_item = {'pass': bool(ours['best_epoch'] == theirs['best_epoch'] and diff <= TOL_TRAIN),
                  'best_epoch': [ours['best_epoch'], theirs['best_epoch']],
                  'best_dev_loss': [ours['best_dev_loss'], theirs['best_dev_loss']], 'abs_diff': diff,
                  'weights_sha256_equal': ours['weights_sha256'] == theirs['weights_sha256'],      # recorded, not gated
                  'cstar_weights_sha256_equal': ours['cstar_weights_sha256'] == theirs['cstar_weights_sha256']}
    exports = {}
    for aspect, variant in EXPORT_VARIANT.items():
        name = f'{aspect}__{variant}__s0.npz'
        a = root / 'cstar' / REGRESS_ENC / 'train' / bs_cstar.STAR_ARM / 'R6-5' / name
        b = cref / 'train' / bs_cstar.STAR_ARM / 'R6-5' / name
        with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
            for side in ('img', 'cap'):
                u, v = x[side], y[side]
                ok = u.shape == v.shape and u.dtype == v.dtype
                d = float(np.abs(u.astype(np.float64) - v.astype(np.float64)).max()) if ok else None
                exports[f'{aspect}/{side}'] = {'pass': bool(ok and d is not None and d <= TOL_EXPORT), 'max_abs_diff': d,
                                               'shape': list(u.shape)}
    passed = bool(cstar['pass'] and photo_item['pass'] and all(i['pass'] for i in exports.values()))
    return {'pass': passed, 'tol_train': TOL_TRAIN, 'tol_export': TOL_EXPORT, 'cstar': cstar, 'photo': photo_item,
            'exports': exports}


def regress_ii(device, compare=True):
    import bs_cstar
    import bs_photo
    root, gate = REGRESS_ROOT, OUT / 'stage0' / 'gate_ii.json'
    croot = root / 'cstar' / REGRESS_ENC
    have_c = bs_cstar.record_path(croot, bs_cstar.STAR_ARM, 0).is_file()
    have_p = (bs_photo.run_dir(REGRESS_ENC, bs_photo.REF, 0, root) / 'run.json').is_file()
    failed = []
    if not (have_c and have_p):                 # a finished retraining (the node's, pulled) is compared, not redone
        rec = unit_cstar(REGRESS_ENC, 0, device, root=croot, kind='ii_cstar')
        if rec['status'] == 'failed':
            failed.append(rec)
        else:
            rec = unit_photo(REGRESS_ENC, 0, device, N.PhotoData(), root=root, cstar_root=croot, kind='ii_xphoto')
            if rec['status'] == 'failed':
                failed.append(rec)
    if failed:
        doc = dict(status='failed', errors=failed[0]['errors'], label=failed[0]['label'], at=N.now())
        doc['pass'] = False
        N.atomic_write(gate, N.jdump(doc))
        print('PP_NODE regress-ii: FAILED twice', flush=True)
        return 1
    if not compare:
        write_status('ii_train.json', dict(status='ok', root=str(root), note='trained; compare with r14 locally'))
        print('PP_NODE regress-ii: trained (no compare)', flush=True)
        return 0
    doc = dict(compare_ii(root), status='ok', at=N.now(), host=socket.gethostname(), smoke=N.smoke_on(),
               reference=str(R14_REF), encoder=REGRESS_ENC, seed=0)
    N.atomic_write(gate, N.jdump(doc))
    print(f'PP_NODE regress-ii: pass={doc["pass"]} -> {gate}', flush=True)
    return 0 if doc['pass'] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('mode', choices=['part-a', 'photo', 'regress-ii'])
    ap.add_argument('--encoder', choices=list(PP.NEW_ENCODERS))
    ap.add_argument('--seeds', type=int, nargs='+', default=None, help='smoke only (BS_NODE_SMOKE=1)')
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--no-compare', action='store_true')
    a = ap.parse_args(argv)
    if a.mode in ('part-a', 'photo'):
        assert a.encoder, f'{a.mode} needs --encoder'
        return (part_a if a.mode == 'part-a' else photo)(a.encoder, a.device, seeds_of(a.seeds))
    return regress_ii(a.device, compare=not a.no_compare)


if __name__ == '__main__':
    sys.exit(main())
