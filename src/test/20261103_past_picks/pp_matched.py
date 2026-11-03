"""r19 matched arms (ticket 05; spec §3.2 matched-arm row, §3.5 (v), D12, D18, D23).

The matched arm of encoder X is PercepT's Stage 2 mapper on X's default image (Qwen P3, SigLIP 2 M2, SigLIP 1 M8, CLIP
L/14 M7, CLIP B/32 M3, CLIP B/16 M6, DINOv2 M5). This module (1) trains DINOv2's M5 at full scale, (2) runs gate (v), (3)
applies every arm's saved mappers frozen to the new sets and reads the probabilities.

M5 (node402 job, GPU; imported code, unchanged): `eg_matched.run(scale, 'M5', seeds, outputs, device, gated=False,
kinds=('held',))` = 10-26's driver over 10-25's `rb_matched.train_stage2` (Stage 2 settings: MLP Linear(w, 1024), ReLU,
Linear(1024, 40), full-batch Adam lr 0.003, 100 epochs, mean BCE) on the claim test's saved Qwen Stage 1 topics. Then the
fixed partitions are written as M3's and M6's are (`rb_score.fixed_partitions`). Layout under the outputs folder:
    part2/<scale>/matched/M5/P3__s<k>.{npz,json,pt}   part2/<scale>/held/M5/P3__s<k>.{npz,json}
    part2/partitions/<scale>/M5/percept__s<k>__{r20,own}.{npz,json}      (full scale only)
Full scale goes to `pp_paths.M5_ROOT` (so `bs_artclu.ENC['dinov2_large']['root'] = M5_ROOT/part2` reads it); the
sample-scale seed-0 rerun for gate (v) goes to `SAMPLE_ROOT`, its own folder.

    python pp_matched.py m5 --scale full --seeds 0 1 2 3 4 --device cuda        (node402; CUBLAS_WORKSPACE_CONFIG=:4096:8)
    python pp_matched.py m5 --scale sample --seeds 0 --device cuda
    python pp_matched.py gate-v --enc clip_b32|clip_b16|dinov2_large [--out DIR]   (CPU; writes stage0/gate_v_<enc>.json)
    python pp_matched.py export --set semart|mmart_new --enc E|all [--out DIR]     (CPU)

Readers (index "Matched arms"):
    held_probs(enc)                      {seed: float32 [12281, 40]}  ArtELingo held, r14's held order, stored files
    partitions(set, enc, keys)           {seed: {'own': int32 [1, n], 'r20': int32 [3, n]}}  own = argmax; r20 =
                                         `ct_score.output_partitions(prob, 'percept')['r20']` on exactly these rows
    load_probs(set, enc, seed, keys)     float32 [len(keys), 40], joined by key
Exports: OUT/exports/<set>/<enc>/matched/probs__s<seed>.{npz,json}, sets `semart` (all seven encoders; the 11,159
paintings of `pp_semart.keys('set')`) and `mmart_new` (the new three; the 16,782 `new` keys), rows in sha256 order.

Gate (v) (spec §3.5): M3 and M6 give `reported_stage4` readings from their stored outputs and partitions within 1e-9 and
their ten mappers on CPU give their stored held probabilities within 1e-5 (r15's `mt_matched.gate_iv`, reused; r15's
fallback when a mapper misses is `fallback_commands`). DINOv2: the sample-scale seed-0 rerun's held readings differ from
10-26's stored seed-0 readings by at most the stored arm's max minus min over seeds 0 to 4 (D23, differences recorded);
the full outputs pass the store checks (`check_m5_outputs`) and the five mappers on CPU give their stored held
probabilities within 1e-5.
"""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')        # `base.determinism` asserts it

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import pp_paths as PP   # first: sets the older rounds' output roots
import pp_register      # noqa: F401  (adds the new three to r14's and r15's tables)
import pp_stores

import mt_matched as MM      # r15: infer_cpu, compare_probs, readings, gate_iv (registered tables)
import bs_artclu
import rb_paths
import rb_score
import ct_percept as base

READINGS = MM.READINGS                       # ('r20', 'own')
TOL_PROB = MM.TOL_PROB                       # 1e-5
TOL_READING = MM.TOL_READING                 # 1e-9
N_HELD = 12281
SAMPLE_ROOT = PP.OUT / 'matched_sample'      # gate (v): the sample-scale M5 seed-0 rerun (part2/sample/...)
M5_PART2 = PP.M5_ROOT / 'part2'              # full-scale M5 (bs_artclu.ENC['dinov2_large']['root'])
SETS = {'semart': PP.ENCODERS, 'mmart_new': PP.NEW_ENCODERS}
DATASET = {'semart': 'semart', 'mmart_new': 'mmart'}
STATE_KEYS = ('input', 'input_dim', 'source_commit')   # the settings that may differ between two matched arms' mappers
_FROZEN_OUTPUT_ROOTS = (PP.EG_PART2.parent, PP.EG_ROUND / 'outputs', PP.RB_ROUND / 'outputs', PP.R10_OUT, PP.R11_OUT,
                        PP.R12_OUT, PP.R13_OUT, PP.R14_OUT, PP.R15_OUT, PP.MAIN / 'src/test/20261020_claim_test/outputs')


def sha_file(path):
    return MM.sha_file(path)


def held_sha256():
    return rb_paths.HELD_SHA256


# ------------------------------------------------------------------------------------------------ readings
def readings(prob):
    """r12's PercepT row (`mt_matched.readings`): {'own': int [1, n] argmax, 'r20': int [3, n]}: 20-cluster k-means on the
    L2-normalised probabilities, states 0 to 2."""
    return MM.readings(prob)


def reading_means(fixed, labels):
    """{reading: mean over the seeds of the mean AMI over the reading's partitions}; labels < 0 are masked."""
    return MM.reading_means(fixed, labels)


def reading_table(probs, labels):
    """{aspect: {reading: mean AMI}} of `probs` {seed: [n, 40]} against `labels` {aspect: int64 [n]}."""
    fixed = {s: readings(p) for s, p in probs.items()}
    return {a: reading_means(fixed, labels[a]) for a in labels}


# ------------------------------------------------------------------------------------------------ D23
def stored_sample_readings(path=None):
    """{aspect: {reading: {'seed0', 'min', 'max'}}}: 10-26's stored sample-scale M5 held readings (`reported_stage3.json`,
    block K5, the seed spread of M5; mean AMI against the held labels)."""
    path = Path(PP.EG_PART2 / 'reported_stage3.json' if path is None else path)
    doc = json.loads(path.read_text())
    assert doc['scale'] == 'sample' and doc['held_sha256'] == held_sha256(), f'{path}: not the sample-scale document'
    out = {}
    for aspect in PP.ASPECTS:
        spread = doc['blocks']['K5']['aspects'][aspect]['point_spread']['M5']
        out[aspect] = {r: dict(seed0=float(spread[r]['per_seed']['0']), min=float(spread[r]['min']),
                               max=float(spread[r]['max'])) for r in READINGS}
    return out


def d23_check(rerun, stored):
    """D23: each held reading of the rerun differs from the stored seed-0 reading by at most the stored arm's maximum
    minus minimum over seeds 0 to 4 of that reading. `rerun` {aspect: {reading: value}}, `stored` as
    `stored_sample_readings`. All six (aspect, reading) items are required; the differences are recorded."""
    assert set(rerun) == set(stored) == set(PP.ASPECTS), (sorted(rerun), sorted(stored))
    items = {}
    for aspect in PP.ASPECTS:
        assert set(rerun[aspect]) == set(stored[aspect]) == set(READINGS), (aspect, sorted(rerun[aspect]))
        for r in READINGS:
            s = stored[aspect][r]
            value = float(rerun[aspect][r])
            assert np.isfinite(value) and np.isfinite([s['seed0'], s['min'], s['max']]).all(), (aspect, r)
            assert s['min'] <= s['seed0'] <= s['max'], f'{aspect}/{r}: stored seed 0 is outside the arm\'s own range'
            tolerance = s['max'] - s['min']
            diff = abs(value - s['seed0'])
            items[f'{aspect}/{r}'] = dict(rerun=value, stored_seed0=s['seed0'], stored_min=s['min'], stored_max=s['max'],
                                          signed_diff=value - s['seed0'], abs_diff=diff, tolerance=tolerance,
                                          ok=bool(diff <= tolerance))
    return {'pass': all(v['ok'] for v in items.values()), 'items': items,
            'max_abs_diff': max(v['abs_diff'] for v in items.values())}


# ------------------------------------------------------------------------------------------------ CPU replay
def replay_probs(cpu, stored, tol=TOL_PROB):
    """Gate (v) / r15 (iv)(a): per seed, the max absolute difference of the CPU probabilities from the stored ones."""
    assert sorted(cpu) == sorted(stored), (sorted(cpu), sorted(stored))
    per = {str(s): MM.compare_probs(cpu[s], stored[s], tol) for s in sorted(cpu)}
    return dict(tol=tol, max_abs_diff=max(v['max_abs_diff'] for v in per.values()),
                passed=bool(all(v['ok'] for v in per.values())), failed_seeds=[int(s) for s, v in per.items() if not v['ok']],
                per_seed=per)


def settings_diff(a, b):
    """Sorted keys where two mapper configs differ, leaving out the input name and width and the code commit."""
    return sorted(k for k in set(a) | set(b) if k not in STATE_KEYS and a.get(k) != b.get(k))


# ------------------------------------------------------------------------------------------------ mappers and stored held probabilities
def mapper_path(enc, seed):
    return MM.mapper_path(enc, seed)


def mapper_weights_sha256(enc, seed):
    return json.loads(Path(mapper_path(enc, seed)).with_suffix('.json').read_text())['weights_sha256']


def held_probs(enc):
    """{seed: float32 [12281, 40]}: the arm's stored ArtELingo held probabilities (r14's held order = the CT's held list;
    each sidecar must carry that list's hash)."""
    out = {}
    for seed in PP.SEEDS:
        path = MM.held_prob_path(enc, seed)
        assert Path(path).is_file(), f'{path}: missing'
        side = json.loads(Path(path).with_suffix('.json').read_text())
        listed = side['list_sha256'] if 'list_sha256' in side else side['held_sha256']     # the CT's own sidecar: held_sha256
        assert listed == held_sha256(), f'{path}: not the held list (sha256 {listed})'
        with np.load(path, allow_pickle=False) as z:
            assert z.files == ['prob'], path
            prob = z['prob']
        assert prob.dtype == np.float32 and prob.shape == (N_HELD, PP.N_TOPICS), (path, prob.shape, prob.dtype)
        assert np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all(), path
        out[seed] = prob
    return out


# ------------------------------------------------------------------------------------------------ exports on new sets
def export_path(root, set_name, enc, seed):
    assert enc in SETS.get(set_name, ()), f'set {set_name!r} has no encoder {enc!r} (sets: {sorted(SETS)})'
    return Path(root) / f'exports/{set_name}/{enc}/matched/probs__s{seed}.npz'


def write_export(root, set_name, enc, seed, keys, probs, *, mapper, mapper_sha256, view):
    """Save `probs` (float32 [n, 40], rows in the order of `keys`) and its sidecar (pending files first)."""
    keys = [str(k) for k in keys]
    assert probs.dtype == np.float32 and probs.shape == (len(keys), PP.N_TOPICS) and len(set(keys)) == len(keys)
    assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()
    path = export_path(root, set_name, enc, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.stem + '.pending.npz')
    np.savez_compressed(pending, keys=np.asarray(keys), probs=probs)
    pending.replace(path)
    meta = dict(sha256=sha_file(path), n=len(keys), width=int(probs.shape[1]), seed=int(seed), encoder=enc, set=set_name,
                mapper=str(mapper), mapper_sha256=mapper_sha256, view=view, device='cpu',
                keys_sha256=hashlib.sha256('\n'.join(keys).encode('utf-8')).hexdigest(), code_sha256=sha_file(__file__))
    pending = path.with_name(path.stem + '.pending.json')
    pending.write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
    pending.replace(path.with_suffix('.json'))
    return path


def load_probs(set_name, enc, seed, keys, root=None):
    """The exported probabilities of one seed, rows in the order of `keys` (joined by key; every key must be present;
    the sidecar's sha256 must equal the file's, and the sidecar must name this encoder, set, seed and view and the
    weights hash of the current saved mapper of (enc, seed))."""
    path = export_path(PP.OUT if root is None else root, set_name, enc, seed)
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['sha256'] == sha_file(path), f'{path}: sha256 differs from its sidecar'
    assert meta['encoder'] == enc and meta['set'] == set_name, f"{path}: sidecar names {meta['set']}/{meta['encoder']}"
    assert meta['seed'] == seed, f"{path}: sidecar seed {meta['seed']!r}, wanted {seed!r}"
    assert meta['view'] == enc, f"{path}: sidecar view {meta['view']!r}, wanted {enc!r}"
    assert meta['mapper_sha256'] == mapper_weights_sha256(enc, seed), \
        f'{path}: sidecar mapper_sha256 is not the current mapper\'s weights_sha256'
    with np.load(path, allow_pickle=False) as z:
        assert sorted(z.files) == ['keys', 'probs']
        have, probs = z['keys'].tolist(), z['probs']
    assert probs.dtype == np.float32 and probs.shape == (len(have), PP.N_TOPICS) == (meta['n'], meta['width'])
    where = {k: i for i, k in enumerate(have)}
    assert len(where) == len(have), f'{path}: duplicate keys'
    keys = [str(k) for k in keys]
    missing = [k for k in keys if k not in where]
    assert not missing, f'{path}: {len(missing)} keys not exported, e.g. {missing[:2]}'
    return probs[[where[k] for k in keys]]


def partitions(set_name, enc, keys, root=None):
    """{seed: {'own': int32 [1, n], 'r20': int32 [3, n]}}, rows in the order of `keys`: own = argmax of the 40
    probabilities; r20 = r12's PercepT row code (20 clusters, states 0 to 2, L2-normalised probabilities), fitted on
    exactly these rows."""
    out = {}
    for seed in PP.SEEDS:
        prob = load_probs(set_name, enc, seed, keys, root)
        got = readings(prob)
        out[seed] = dict(own=got['own'].astype(np.int32), r20=got['r20'].astype(np.int32))
        assert out[seed]['own'].shape == (1, len(prob)) and out[seed]['r20'].shape == (3, len(prob))
        assert np.array_equal(out[seed]['own'][0], prob.argmax(axis=1))
    return out


def default_keys(set_name):
    if set_name == 'semart':
        import pp_semart
        return pp_semart.keys('set')
    import mt_data
    return mt_data.keys('new')


def export(set_name, enc, root=None, keys=None, loader=None, infer=None):
    """Run the five saved mappers of `enc` on the set (X's default image from `pp_stores`) and save the probabilities."""
    assert enc in SETS.get(set_name, ()), f'set {set_name!r} has no encoder {enc!r}'
    root = PP.OUT if root is None else Path(root)
    keys = [str(k) for k in (default_keys(set_name) if keys is None else keys)]
    assert keys == PP.order(keys), 'keys must be in sha256 order (pp_paths.order)'
    x = (pp_stores.load if loader is None else loader)(DATASET[set_name], enc, keys)
    assert x.dtype == np.float32 and x.shape == (len(keys), PP.WIDTH[enc]), (x.shape, x.dtype)
    infer = MM.infer_cpu if infer is None else infer
    for seed in PP.SEEDS:
        path = mapper_path(enc, seed)
        write_export(root, set_name, enc, seed, keys, infer(path, x), mapper=path,
                     mapper_sha256=mapper_weights_sha256(enc, seed), view=enc)
    return [export_path(root, set_name, enc, s) for s in PP.SEEDS]


# ------------------------------------------------------------------------------------------------ M5 training job
def check_out_root(path):
    """An output folder must not lie inside, or contain, a frozen round's outputs (V4: no write into 10-25, 10-26, r10
    to r15 or the claim test)."""
    p = Path(path).resolve()
    for frozen in _FROZEN_OUTPUT_ROOTS:
        f = Path(frozen).resolve()
        assert not (p == f or p.is_relative_to(f) or f.is_relative_to(p)), f'{path} overlaps the frozen outputs {frozen}'
    return Path(path)


def default_root(scale):
    return M5_PART2.parent if scale == 'full' else SAMPLE_ROOT


def write_partitions(root, scale, seeds, arm='M5'):
    """The fixed partitions of the arm's held outputs as M3's and M6's are saved: `<root>/part2/partitions/<scale>/<arm>/
    percept__s<k>__{r20,own}.npz` (`rb_score.fixed_partitions`, sources bound by hash)."""
    root = Path(root)
    done = {}
    for seed in seeds:
        path = root / f'part2/{scale}/held/{arm}/P3__s{seed}.npz'
        x = rb_score.load_output(path, arm, N_HELD, held_sha256(), scale, seed, 'percept')
        done[seed] = rb_score.fixed_partitions(x, arm, root / f'part2/partitions/{scale}/{arm}', 'percept', seed,
                                               held_sha256(), rb_score.sha_file(path))
    return done


def run_m5(scale, seeds, outputs, device, smoke=False):
    """Train (or resume) M5 at `scale` with the imported 10-26 driver, ungated, held outputs only; then, at full scale,
    write the fixed partitions."""
    import eg_matched
    assert scale in ('sample', 'full')
    outputs = check_out_root(outputs)
    paths = eg_matched.run(scale, 'M5', list(seeds), outputs, device, smoke=smoke, epochs=2 if smoke else None,
                           gated=False, kinds=('held',))
    if scale == 'full':
        write_partitions(outputs, scale, seeds)
    return paths


# ------------------------------------------------------------------------------------------------ gate (v)
def check_m5_outputs(root, seeds=PP.SEEDS, scale='full', reference_arm='M6'):
    """Gate (v): the new M5 outputs pass the store checks (gate (i)'s kind, for output files): per seed, the mapper verifies
    (`base.verify_output`: weights hash), has the settings and Stage 1 topics of the 10-26 round's `reference_arm` mapper
    of the same seed (full scale; only the input differs), the held file has the held list's rows, hash, a finite [0, 1]
    array whose sidecar names the mapper's weights, and the stored partitions are bound to that held file by hash."""
    root = Path(root)
    items = {}
    for seed in seeds:
        problems = []
        mapper = root / f'part2/{scale}/matched/M5/P3__s{seed}.npz'
        meta = base.verify_output(mapper)
        cfg = meta['config']
        for key, want in (('input', 'dinov2_large default image'), ('input_dim', 1024), ('arm', 'P3'), ('epochs', 100),
                          ('device', 'cuda'), ('smoke', False)):
            if cfg.get(key) != want:
                problems.append(f'config {key} is {cfg.get(key)!r}, wanted {want!r}')
        if meta['K'] != PP.N_TOPICS:
            problems.append(f"K is {meta['K']}")
        if scale == 'full':
            ref = base.verify_output(PP.EG_PART2 / f'full/matched/{reference_arm}/P3__s{seed}.npz')
            diff = settings_diff(cfg, ref['config'])
            if diff:
                problems.append(f'settings differ from {reference_arm}: {diff}')
            for key in ('stage1_sha256', 'eval_names_sha256'):
                if meta.get(key) != ref.get(key):
                    problems.append(f'{key} differs from {reference_arm}')
        held = root / f'part2/{scale}/held/M5/P3__s{seed}.npz'
        try:
            rb_score.load_output(held, 'M5', N_HELD, held_sha256(), scale, seed, 'percept')
            side = json.loads(held.with_suffix('.json').read_text())
            if side.get('stage2_weights_sha256') != meta['weights_sha256']:
                problems.append('held sidecar does not name this mapper\'s weights')
            if side.get('stage1_sha256') != meta.get('stage1_sha256'):
                problems.append('held sidecar and mapper disagree on the Stage 1 file')
        except (AssertionError, OSError, KeyError) as exc:
            problems.append(f'held output: {exc}')
        if scale == 'full':
            try:
                bs_artclu.stored_m(root / 'part2', 'M5', N_HELD, seeds=(seed,))
            except (AssertionError, OSError, KeyError) as exc:
                problems.append(f'stored partitions: {exc}')
        items[str(seed)] = dict(ok=not problems, problems=problems, weights_sha256=meta['weights_sha256'])
    return dict(passed=bool(all(v['ok'] for v in items.values())), per_seed=items)


def _held_images(enc):
    import ct_held
    import eg_encoders
    import rb_encoders
    names, sha = ct_held.read_held_list(MM.base_held_list())
    assert sha == held_sha256()
    eg_encoders.extend_rb_maps()
    return names, rb_encoders.full_images(bs_artclu.ENC[enc]['store'], names)


def fallback_commands(enc, seeds, outputs):
    """r15's mapper fallback (spec §3.5 (v)): retrain a mapper that missed 1e-5 from the arm's saved Stage 1 topics on the
    local GPU into a new folder; the retrained mapper's held readings must lie within the stored arm's max minus min over
    seeds 0 to 4; the report flags the mapper. Commands (run through `res-queue gpu`), never into the frozen folders."""
    arm = PP.MATCHED[enc]
    src = 'recipe_backbones/rb_matched.py' if arm in ('M2', 'M3') else 'encoders_genre/eg_matched.py'
    kinds = '' if arm in ('M2', 'M3') else ' --kinds held'          # rb_matched's CLI has no --kinds
    return [f'CUBLAS_WORKSPACE_CONFIG=:4096:8 python src/test/20261025_{src} --scale full --arm {arm} --seeds {s} '
            f'--outputs {outputs} --device cuda{kinds}' for s in seeds]


def gate_v_stored(enc, out_root=None, write=True):
    """Gate (v) for M3 (clip_b32) and M6 (clip_b16): r15's `gate_iv` on their stored outputs: (a) their five mappers on CPU
    against the stored held probabilities (1e-5), (b) the reading code on the stored probabilities against the stored
    partitions (exactly) and `reported_stage4` (1e-9)."""
    assert enc in ('clip_b32', 'clip_b16'), enc
    res = MM.gate_iv(enc, out_root, write=False)
    result = dict(encoder=enc, arm=PP.MATCHED[enc], kind='stored', a=res['a'], b=res['b'], c=res['c'],
                  fallback=(fallback_commands(enc, res['a']['failed_seeds'], '<new outputs folder>')
                            if res['a']['failed_seeds'] else []), **{'pass': bool(res['pass'])})
    return _write_gate(result, out_root, write)


def gate_v_dinov2(part2=None, sample_root=None, out_root=None, write=True):
    """Gate (v) for DINOv2's M5: D23 on the sample-scale rerun, the full outputs' store checks, the five mappers on CPU
    against their node402 held probabilities (1e-5). Recorded, not gated: the readings of the CPU probabilities and
    whether the stored partitions equal fresh ones. A missing rerun or output fails the gate."""
    part2 = M5_PART2 if part2 is None else Path(part2)
    sample_root = SAMPLE_ROOT if sample_root is None else Path(sample_root)
    held_names, _, labels = bs_artclu.held_setup()
    result = dict(encoder='dinov2_large', arm='M5', kind='rerun')
    # D23: the sample rerun against the stored seed 0
    try:
        rerun_path = sample_root / 'part2/sample/held/M5/P3__s0.npz'
        x = rb_score.load_output(rerun_path, 'M5', N_HELD, held_sha256(), 'sample', 0, 'percept')
        table = reading_table({0: x}, labels)
        d23 = d23_check(table, stored_sample_readings())
        stored_w = json.loads((PP.EG_PART2 / 'sample/matched/M5/P3__s0.json').read_text())['weights_sha256']
        new_w = json.loads((sample_root / 'part2/sample/matched/M5/P3__s0.json').read_text())['weights_sha256']
        d23['rerun_mapper_equals_stored'] = bool(stored_w == new_w)
        d23['rerun_device'] = json.loads((sample_root / 'part2/sample/matched/M5/P3__s0.json').read_text())['config']['device']
    except (AssertionError, OSError) as exc:
        d23 = {'pass': False, 'error': str(exc)}
    result['d23'] = d23
    # the full outputs
    try:
        outputs = check_m5_outputs(part2.parent if part2.name == 'part2' else part2)
    except (AssertionError, OSError) as exc:
        outputs = {'passed': False, 'error': str(exc)}
    result['outputs'] = outputs
    # CPU replay and recorded readings
    try:
        stored = held_probs('dinov2_large')
        names, images = _held_images('dinov2_large')
        assert list(names) == list(held_names)
        cpu = {s: MM.infer_cpu(mapper_path('dinov2_large', s), images) for s in PP.SEEDS}
        result['cpu_replay'] = replay_probs(cpu, stored)
        stored_fixed = bs_artclu.stored_m(part2, 'M5', N_HELD)
        cpu_fixed = {s: readings(cpu[s]) for s in PP.SEEDS}
        result['recorded'] = dict(gate=False,
                                  partitions_equal_stored={f'{s}/{r}': bool(np.array_equal(cpu_fixed[s][r], stored_fixed[s][r]))
                                                           for s in PP.SEEDS for r in READINGS},
                                  cpu_readings={a: reading_means(cpu_fixed, labels[a]) for a in PP.ASPECTS},
                                  stored_readings={a: reading_means(stored_fixed, labels[a]) for a in PP.ASPECTS})
    except (AssertionError, OSError, KeyError) as exc:
        result['cpu_replay'] = {'passed': False, 'error': str(exc)}
    result['pass'] = bool(result['d23']['pass'] and result['outputs']['passed'] and result['cpu_replay']['passed'])
    return _write_gate(result, out_root, write)


def _write_gate(result, out_root, write):
    if write:
        path = Path(PP.OUT if out_root is None else out_root) / f"stage0/gate_v_{result['encoder']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1, allow_nan=False, default=_jsonable) + '\n')
    return result


def _jsonable(x):
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    raise TypeError(type(x))


# ------------------------------------------------------------------------------------------------ command line
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('m5', help='train DINOv2 M5 (GPU, node402) and write its partitions')
    p.add_argument('--scale', required=True, choices=('sample', 'full'))
    p.add_argument('--seeds', type=int, nargs='+', choices=range(5), default=list(range(5)))
    p.add_argument('--outputs', type=Path, default=None, help='default: pp_paths.M5_ROOT (full), OUT/matched_sample (sample)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', action='store_true', help='seed 0, 2 epochs, outputs under /tmp/claude-0')
    p = sub.add_parser('gate-v')
    p.add_argument('--enc', required=True, choices=PP.NEW_ENCODERS)
    p.add_argument('--out', default=None)
    p = sub.add_parser('export')
    p.add_argument('--set', required=True, choices=sorted(SETS))
    p.add_argument('--enc', required=True, choices=PP.ENCODERS + ('all',))
    p.add_argument('--out', default=None)
    args = ap.parse_args(argv)
    if args.cmd == 'm5':
        assert len(args.seeds) == len(set(args.seeds)), 'duplicate seeds'
        outputs = args.outputs or default_root(args.scale)
        if args.smoke:
            import eg_paths
            assert eg_paths.smoke_ok(outputs), '--smoke outputs must be under /tmp/claude-0'
        assert args.device.startswith('cuda'), 'Stage 2 training and the mapper inference of M5 need the GPU'
        run_m5(args.scale, [0] if args.smoke else args.seeds, outputs, args.device, smoke=args.smoke)
    elif args.cmd == 'gate-v':
        res = gate_v_stored(args.enc, args.out) if args.enc != 'dinov2_large' else gate_v_dinov2(out_root=args.out)
        print(f"{args.enc}: gate (v) pass={res['pass']}")
        raise SystemExit(0 if res['pass'] else 1)
    else:
        for enc in (PP.ENCODERS if args.enc == 'all' else (args.enc,)):
            if enc not in SETS[args.set]:
                assert args.enc == 'all', f'set {args.set} has no encoder {enc}'
                continue
            paths = export(args.set, enc, args.out)
            print(f'{args.set}/{enc}: wrote {len(paths)} files, e.g. {paths[0]}')


if __name__ == '__main__':
    main()
