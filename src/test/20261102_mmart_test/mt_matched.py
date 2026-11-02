"""r15 matched arms on CPU (ticket 03; spec §3.2 matched-arm row, §3.5 0(iv), D5, D11, D18).

The matched arm of encoder X is PercepT on X's default image, applied frozen: the saved Stage 2 mappers (20 files,
trained on CUDA, with their ArtELingo Qwen topics; Qwen P3, SigLIP 2 M2, SigLIP 1 M8, CLIP L/14 M7; seeds 0 to 4) run on
CPU on MMArt `new`. A mapper takes the encoder's unit-norm image row and divides it by its norm again (the trainer's
own `rb_matched.infer`); `infer_cpu` is that function with the device check relaxed to allow CPU.

    infer_cpu(path, images)             float32 [n, 40] sigmoid probabilities on CPU
    readings(prob)                      {'r20': int [3, n], 'own': int [1, n]}: r12's PercepT row code
                                        (`ct_score.output_partitions(prob, 'percept')`: argmax; 20-cluster k-means on
                                        the L2-normalised probabilities, states 0 to 2)
    partitions(enc, keys)               {seed: {'own': int32 [n], 'k20': int32 [3, n]}} from the exported probabilities,
                                        rows in the order of `keys`, k-means fitted on exactly these rows
    write_export(...)                   OUT/exports/new/<enc>/matched/probs__s<seed>.{npz,json}
    gate_iv(enc)                        stage 0 gate (iv): (a) CPU against stored held probabilities (1e-5 max abs),
                                        (b) the reading code on the stored probabilities against the stored partitions
                                        (exactly) and `reported_stage4` (1e-9), (c) readings of the CPU probabilities
                                        recorded, no gate; writes OUT/stage0/gate_iv_<enc>.json

D11 fallback (spec §3.5 0(iv)). If (a) fails for a mapper, retrain it from the arm's saved Stage 1 topics with the
recorded settings of the arm's trainer, on the local GPU: `flock -o -w <s> /tmp/gpu0.lock <command>`. Trainer entry
points (not run here): SigLIP 2 M2: `20261025_recipe_backbones/rb_matched.py --scale full --arm M2 --seeds <s>`; SigLIP 1
M8 and CLIP L/14 M7: `20261025_encoders_genre/eg_matched.py --scale full --arm M8|M7 --seeds <s>`; Qwen P3: the CT's
`20261020_claim_test/ct_percept.py --stage 2 --settings core --cut k40 --version qwen_qwen --arm P3 --seeds <s>`
(driver `ct2_run.py core`). Then the check: the retrained mapper's held readings (own and 20-cluster, each aspect) must
lie within the stored arm's maximum minus minimum over seeds 0 to 4 of that reading; the report flags the mapper.

CLI (CPU only; run with CUBLAS_WORKSPACE_CONFIG=:4096:8 or let this module set it):
    python mt_matched.py export  --enc E|all [--out DIR]
    python mt_matched.py gate-iv --enc E|all [--out DIR]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')       # `base.determinism` asserts it

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mt_paths as P  # noqa: E402

for _path in (P.BS_ROUND, P.MAIN / 'src/test/20261025_recipe_backbones', P.MAIN / 'src/test/20261025_encoders_genre'):   # older rounds' code, read only
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
import bs_artclu  # noqa: E402  (r14: stored readers, `arm_means`, `held_setup`; puts r10 to r12 and older rounds on sys.path)
import rb_matched  # noqa: E402  (10-25: the trainer's own inference)
import rb_encoders  # noqa: E402  (the loader the mappers were trained with)
import ct_held  # noqa: E402
import ct_score  # noqa: E402  (the PercepT reading code)
import ct_percept as base  # noqa: E402
import eg_encoders  # noqa: E402  (10-26: adds SigLIP 1 and CLIP L/14 to `rb_encoders`' loader, as it did for those mappers)

TOL_PROB = 1e-5          # 0(iv)(a): max absolute
TOL_READING = 1e-9       # 0(iv)(b)
READINGS = ('r20', 'own')

# Per encoder: where the saved Stage 2 mappers, their stored held probabilities and stored partitions live (older
# rounds' folders in the main checkout), and the names `reported_stage4` uses. Qwen's P3 is the claim test's own; its
# stored partitions are 10-25's B1 (the same held probabilities).
_CT_FULL = P.MAIN / 'src/test/20261020_claim_test/outputs/stage2/full'
_RB = bs_artclu.RB_PART2
_EG = bs_artclu.EG_PART2


def _spec(enc):
    e = bs_artclu.ENC[enc]
    if enc == 'qwen':
        mapper_dir, held_dir = _CT_FULL / 'percept/core/stage2/k40', _CT_FULL / 'held/percept/core__k40'
    else:
        mapper_dir, held_dir = e['root'] / 'full/matched' / e['matched'], e['root'] / 'full/held' / e['matched']
    return dict(store=e['store'], root=e['root'], cand=e['cand'], matched=e['matched'], mapper_dir=mapper_dir,
                held_dir=held_dir, view=P.DEFAULT_VIEW[enc])


def mapper_path(enc, seed):
    return _spec(enc)['mapper_dir'] / f'P3__s{seed}.npz'


def held_prob_path(enc, seed):
    return _spec(enc)['held_dir'] / f'P3__s{seed}.npz'


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ inference
def infer_cpu(path, images):
    """`rb_matched.infer`'s body with the device check relaxed to CPU: the sealed mapper (hash-verified by the CT's
    `verify_output`), images float32 [n, input_dim], divided by the norm again unless the mapper's input is a CLIP
    image; one batch, caller order; float32 sigmoid probabilities in [0, 1]."""
    meta = base.verify_output(path)
    config = meta['config']
    assert images.dtype == np.float32 and images.shape == (len(images), config['input_dim']) and len(images) > 0
    assert np.isfinite(images).all()
    x = images / np.linalg.norm(images, axis=1, keepdims=True) if rb_matched.renormalizes(config['input']) else images
    base.determinism(config['seed'])
    with threadpool_limits(limits=8):
        mapper = base.build_mapper(config, meta['K'])
        state = torch.load(Path(path).with_suffix('.pt'), map_location='cpu', weights_only=True)
        mapper.load_state_dict(state['mapper'])
        mapper.to('cpu').eval()
        with torch.no_grad():
            prob = mapper(torch.from_numpy(np.ascontiguousarray(x))).sigmoid().cpu().numpy()
    assert prob.shape == (len(images), meta['K']) and prob.dtype == np.float32
    assert np.isfinite(prob).all() and base.in_unit_interval(prob)
    return prob


# ------------------------------------------------------------------------------------------------ readings
def readings(prob):
    """r12's PercepT row: own = argmax of the probabilities; r20 = 20-cluster k-means (states 0 to 2) on the L2-normalised
    probabilities (`ct_score.output_partitions`, the code behind the stored partitions)."""
    out = ct_score.output_partitions(np.asarray(prob, dtype=np.float32), 'percept')
    assert out['r20'].shape == (3, len(prob)) and out['own'].shape == (1, len(prob))
    return out


def reading_means(fixed, labels):
    """{reading: mean over seeds of the mean AMI over the reading's partitions} (r14's `arm_means`: labels < 0 masked)."""
    return bs_artclu.arm_means(fixed, labels)


def reported_readings(doc, cand, matched):
    """{aspect: {'r20', 'own'}} from a `reported_stage4.json` document: emotion in the seed spread's mean, style and
    genre in margin b_matched's A_e (`bs_artclu.replay_0vii_matched` reads the same places)."""
    rep = doc['blocks'][cand]['aspects']
    out = {'emotion': {r: rep['emotion']['point_spread'][matched][r]['mean'] for r in READINGS}}
    for aspect in ('style', 'genre'):
        out[aspect] = {r: rep[aspect]['margins']['b_matched']['A_e'][f'{matched}-{r}'] for r in READINGS}
    return out


def compare_probs(cpu, stored, tol=TOL_PROB):
    """0(iv)(a) for one mapper: max absolute difference of the CPU probabilities from the stored ones."""
    assert cpu.shape == stored.shape and cpu.ndim == 2, (cpu.shape, stored.shape)
    diff = float(np.abs(cpu.astype(np.float64) - stored.astype(np.float64)).max())
    return dict(max_abs_diff=diff, tol=tol, ok=bool(diff <= tol))


def gate_b(probs, stored, labels, reported):
    """0(iv)(b): the reading code, fed the stored held probabilities `probs` {seed: [n, 40]}, gives the stored partitions
    `stored` {seed: {'r20', 'own'}} exactly and `reported` {aspect: {reading: mean}} within 1e-9. `labels` {aspect: int64}."""
    assert sorted(probs) == sorted(stored)
    ours, parts = {}, {}
    for seed in sorted(probs):
        got = readings(probs[seed])
        ours[seed] = got
        for r in READINGS:
            same = bool(got[r].shape == stored[seed][r].shape and np.array_equal(got[r], stored[seed][r]))
            parts[f'{seed}/{r}'] = dict(equal=same)
    items = {}
    for aspect in P.ASPECTS:
        means = reading_means(ours, labels[aspect])
        for r in READINGS:
            diff = float(abs(means[r] - reported[aspect][r]))
            items[f'{aspect}/{r}'] = dict(ours=float(means[r]), theirs=float(reported[aspect][r]), abs_diff=diff,
                                          ok=bool(diff < TOL_READING))
    equal = all(v['equal'] for v in parts.values())
    ok = all(v['ok'] for v in items.values())
    return dict(passed=bool(equal and ok), partitions_equal=equal, readings_ok=ok, partitions=parts, readings=items,
                max_reading_diff=max(v['abs_diff'] for v in items.values()))


# ------------------------------------------------------------------------------------------------ exports
def export_path(root, enc, seed):
    return Path(root) / f'exports/new/{enc}/matched/probs__s{seed}.npz'


def write_export(root, enc, seed, keys, probs, *, mapper, mapper_sha256, view):
    """Save `probs` (float32 [n, 40], rows in the order of `keys`) and its sidecar (pending files first)."""
    keys = [str(k) for k in keys]
    assert probs.dtype == np.float32 and probs.shape == (len(keys), P.N_TOPICS) and len(set(keys)) == len(keys)
    assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()
    path = export_path(root, enc, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.stem + '.pending.npz')
    np.savez_compressed(pending, keys=np.asarray(keys), probs=probs)
    pending.replace(path)
    meta = dict(sha256=sha_file(path), n=len(keys), width=int(probs.shape[1]), seed=int(seed), encoder=enc,
                mapper=str(mapper), mapper_sha256=mapper_sha256, view=view, device='cpu',
                keys_sha256=hashlib.sha256('\n'.join(keys).encode('utf-8')).hexdigest(),
                code_sha256=sha_file(__file__))
    pending = path.with_name(path.stem + '.pending.json')
    pending.write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
    pending.replace(path.with_suffix('.json'))
    return path


def load_probs(enc, seed, keys, root=None):
    """The exported probabilities of one seed, rows in the order of `keys` (joined by key; every key must be present;
    the sidecar's sha256 must equal the file's, and the sidecar must name this encoder, seed and default view and the
    weights hash of the current saved mapper of (enc, seed))."""
    path = export_path(P.OUT if root is None else root, enc, seed)
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['sha256'] == sha_file(path), f'{path}: sha256 differs from its sidecar'
    assert meta['encoder'] == enc, f"{path}: sidecar encoder {meta['encoder']!r}, wanted {enc!r}"
    assert meta['seed'] == seed, f"{path}: sidecar seed {meta['seed']!r}, wanted {seed!r}"
    assert meta['view'] == P.DEFAULT_VIEW[enc], f"{path}: sidecar view {meta['view']!r}, wanted {P.DEFAULT_VIEW[enc]!r}"
    current = json.loads(mapper_path(enc, seed).with_suffix('.json').read_text())['weights_sha256']
    assert meta['mapper_sha256'] == current, f'{path}: sidecar mapper_sha256 is not the current mapper\'s weights_sha256'
    with np.load(path, allow_pickle=False) as z:
        assert sorted(z.files) == ['keys', 'probs']
        have, probs = z['keys'].tolist(), z['probs']
    assert probs.dtype == np.float32 and probs.shape == (len(have), P.N_TOPICS) == (meta['n'], meta['width'])
    where = {k: i for i, k in enumerate(have)}
    assert len(where) == len(have), f'{path}: duplicate keys'
    keys = [str(k) for k in keys]
    missing = [k for k in keys if k not in where]
    assert not missing, f'{path}: {len(missing)} keys not exported, e.g. {missing[:2]}'
    return probs[[where[k] for k in keys]]


def partitions(enc, keys, root=None):
    """{seed: {'own': int32 [n], 'k20': int32 [3, n]}}, rows in the order of `keys`: own = argmax of the 40
    probabilities; k20 = r12's PercepT row code (20 clusters, states 0 to 2, L2-normalised probabilities) fitted on
    exactly these rows."""
    out = {}
    for seed in P.SEEDS:
        prob = load_probs(enc, seed, keys, root)
        got = readings(prob)
        out[seed] = dict(own=got['own'][0].astype(np.int32), k20=got['r20'].astype(np.int32))
        assert np.array_equal(out[seed]['own'], prob.argmax(axis=1))
    return out


def export(enc, root=None):
    """Run the five saved mappers of `enc` on MMArt `new` (default view of X, `mt_data.features`) and save them."""
    import mt_data
    root = P.OUT if root is None else Path(root)
    keys = mt_data.keys('new')
    x = mt_data.features(P.DEFAULT_VIEW[enc], keys)
    assert x.dtype == np.float32 and x.shape == (len(keys), P.WIDTH[enc])
    for seed in P.SEEDS:
        path = mapper_path(enc, seed)
        meta = json.loads(path.with_suffix('.json').read_text())
        prob = infer_cpu(path, x)
        write_export(root, enc, seed, keys, prob, mapper=path, mapper_sha256=meta['weights_sha256'],
                     view=P.DEFAULT_VIEW[enc])
    return [export_path(root, enc, s) for s in P.SEEDS]


# ------------------------------------------------------------------------------------------------ gate (iv)
def stored_partitions_name(enc):
    """The arm folder of the stored partitions: the arm's own name, except Qwen's P3, whose partitions are 10-25's B1."""
    return 'B1' if enc == 'qwen' else _spec(enc)['matched']


def stored_partitions(enc, n_rows):
    """{seed: {'r20', 'own'}} stored partitions of the arm (10-25 and 10-26), each bound to the held probability file
    it came from by the sidecar's source sha256."""
    spec = _spec(enc)
    out = {}
    for s in P.SEEDS:
        base_dir = spec['root'] / 'partitions/full' / stored_partitions_name(enc)
        out[s] = {r: bs_artclu.read_stored_set(base_dir / f'percept__s{s}__{r}.npz', n_rows, 1 if r == 'own' else 3,
                                               held_prob_path(enc, s)) for r in READINGS}
    return out


def gate_iv(enc, out_root=None, write=True):
    """Stage 0 gate (iv) for encoder `enc`, on the real ArtELingo held rows. Returns the gate dict (key `pass`)."""
    spec = _spec(enc)
    names, sha = ct_held.read_held_list(base_held_list())
    n = len(names)
    eg_encoders.extend_rb_maps()                                    # idempotent; 10-26 trained M7 and M8 with it
    images = rb_encoders.full_images(spec['store'], names)          # the loader the mappers were trained with
    held_names, _, labels = bs_artclu.held_setup()
    assert list(names) == list(held_names), 'the CT held list differs from r14\'s held_setup names (order included)'
    stored_prob, cpu_prob, a = {}, {}, {}
    for s in P.SEEDS:
        side = json.loads(held_prob_path(enc, s).with_suffix('.json').read_text())
        listed = side['list_sha256'] if 'list_sha256' in side else side['held_sha256']   # the CT's own sidecar: held_sha256
        assert listed == sha, f'seed {s}: stored held list differs from the CT held list'
        with np.load(held_prob_path(enc, s), allow_pickle=False) as z:
            stored_prob[s] = z['prob']
        cpu_prob[s] = infer_cpu(mapper_path(enc, s), images)
        a[str(s)] = compare_probs(cpu_prob[s], stored_prob[s])
    gate_a = dict(tol=TOL_PROB, max_abs_diff=max(v['max_abs_diff'] for v in a.values()),
                  passed=bool(all(v['ok'] for v in a.values())), failed_seeds=[int(s) for s, v in a.items() if not v['ok']],
                  per_seed=a)
    stored = stored_partitions(enc, n)
    reported = reported_readings(bs_artclu.reported_stage4(spec['root']), spec['cand'], spec['matched'])
    gate_b_res = gate_b(stored_prob, stored, labels, reported)
    # (c): readings of the CPU probabilities, beside the stored ones; no gate
    cpu_fixed = {s: readings(cpu_prob[s]) for s in P.SEEDS}
    stored_means = {a_: reading_means(stored, labels[a_]) for a_ in P.ASPECTS}
    cpu_means = {a_: reading_means(cpu_fixed, labels[a_]) for a_ in P.ASPECTS}
    gate_c = dict(gate=False,
                  partitions_equal_stored={f'{s}/{r}': bool(np.array_equal(cpu_fixed[s][r], stored[s][r]))
                                           for s in P.SEEDS for r in READINGS},
                  cpu_readings=cpu_means, stored_readings=stored_means,
                  max_abs_reading_diff=max(abs(cpu_means[a_][r] - stored_means[a_][r])
                                           for a_ in P.ASPECTS for r in READINGS))
    result = {'encoder': enc, 'arm': 'P3' if enc == 'qwen' else spec['matched'], 'stored_partitions': stored_partitions_name(enc), 'held_rows': n, 'held_list_sha256': sha,
              'a': gate_a, 'b': gate_b_res, 'c': gate_c, 'pass': bool(gate_a['passed'] and gate_b_res['passed'])}
    if write:
        path = Path(P.OUT if out_root is None else out_root) / f'stage0/gate_iv_{enc}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    return result


def base_held_list():
    import rb_paths
    return rb_paths.CT_OUT / 'held/held_paintings.json'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('export', 'gate-iv'):
        p = sub.add_parser(name)
        p.add_argument('--enc', required=True, choices=P.ENCODERS + ('all',))
        p.add_argument('--out', default=None, help='output root (default mt_paths.OUT, i.e. MT_OUTPUTS)')
    args = ap.parse_args(argv)
    encs = P.ENCODERS if args.enc == 'all' else (args.enc,)
    for enc in encs:
        if args.cmd == 'export':
            paths = export(enc, args.out)
            print(f'{enc}: wrote {len(paths)} files, e.g. {paths[0]}')
        else:
            res = gate_iv(enc, args.out)
            print(f"{enc}: (iv) pass={res['pass']} a: max|diff|={res['a']['max_abs_diff']:.3g} "
                  f"failed={res['a']['failed_seeds']}; b: partitions_equal={res['b']['partitions_equal']} "
                  f"max reading diff={res['b']['max_reading_diff']:.3g}; c: max reading diff cpu-vs-stored="
                  f"{res['c']['max_abs_reading_diff']:.3g}")


if __name__ == '__main__':
    main()
