"""Independent stage-iii re-derivation (EG rule §§2.3, 3.3, 4.3; RB §3.1).

No scoring implementation is imported. Partitions are recomputed before caches
are opened; records under examination never choose arms, draws or outcomes.
MiniBatchKMeans arguments not fixed by the rule use sklearn defaults, as
rb_ami.py does (init, max_iter, tol, max_no_improvement, init_size,
reassignment_ratio, compute_labels, verbose). Saved rows are never normalized.
"""
import argparse
import hashlib
import json
import multiprocessing
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
import sklearn
import threadpoolctl
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_limits

import eg_paths as paths
import eg_variants as variants

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = 'R6-5'
ASPECTS = ('emotion', 'style', 'genre')
RANGES = dict(emotion=0.0046, style=0.0082, genre=0.0128)
SEEDS = (1, 2, 3)
_DRAW_INPUT = None


def mean_ami(labels, partitions, indices):
    assert labels.ndim == 1 and partitions.shape == (3, len(labels))
    assert len(indices) and (labels[indices] >= 0).all()
    value = float(np.mean([adjusted_mutual_info_score(labels[indices], p[indices])
                           for p in partitions]))
    assert np.isfinite(value)
    return value


def _draw_chunk(task):
    labels, parts, fresh = _DRAW_INPUT
    start, stop = task
    indices = np.flatnonzero(labels >= 0)
    rng = np.random.default_rng(0)
    values = {arm: [] for arm in parts if arm != BASE}
    with threadpool_limits(limits=1):
        for b in range(stop):
            idx = indices[rng.integers(0, len(indices), size=len(indices))]
            seeds = rng.choice(SEEDS, size=3, replace=True) if fresh else (0,)
            if b < start:
                continue
            scores = {arm: float(np.mean([mean_ami(labels, by_seed[int(s)], idx) for s in seeds]))
                      for arm, by_seed in parts.items()}
            for arm in values:
                values[arm].append(scores[arm] - scores[BASE])
    return values


def score_effects(labels, parts, n_draws=2000, workers=8, fresh=False):
    """Score one aspect; own paired painting/seed draws, shared across arms.

    parts[arm][seed] is int64 [3,n], including masked rows. Fresh-seed draws
    resample the same seeds for both arms, after drawing paintings each time.
    """
    assert 1 <= workers <= 8 and n_draws > 0 and BASE in parts
    for by_seed in parts.values():
        assert set(by_seed) == set(SEEDS if fresh else (0,))
    idx = np.flatnonzero(labels >= 0)
    point = {arm: {str(s): mean_ami(labels, p, idx) for s, p in by_seed.items()}
             for arm, by_seed in parts.items()}
    means = {arm: float(np.mean(list(scores.values()))) for arm, scores in point.items()}
    global _DRAW_INPUT
    _DRAW_INPUT = labels, parts, fresh
    edges = np.linspace(0, n_draws, min(workers, n_draws) + 1, dtype=int)
    tasks = list(zip(edges[:-1], edges[1:]))
    if workers == 1:
        chunks = [_draw_chunk(tasks[0])]
    else:
        with multiprocessing.get_context('fork').Pool(len(tasks)) as pool:
            chunks = pool.map(_draw_chunk, tasks)
    out = {}
    for arm in parts:
        if arm == BASE:
            if not fresh:
                out[arm] = dict(A=means[arm], e=0., lo=0., hi=0.)
            continue
        draws = np.concatenate([c[arm] for c in chunks])
        lo, hi = map(float, np.percentile(draws, [2.5, 97.5]))
        entry = dict(e=means[arm] - means[BASE], lo=lo, hi=hi)
        if fresh:
            entry.update(A={a: point[a] for a in (arm, BASE)},
                         Abar={a: means[a] for a in (arm, BASE)})
        else:
            entry['A'] = means[arm]
        out[arm] = entry
    return out


def variant_result(run, effects):
    record = {k: run.get(k) for k in ('status', 'cause', 'best_epoch')}
    record.update(effects=effects, A_genre=None, e_genre=None, guarded=False)
    if run['status'] == 'ok':
        assert effects is not None
        record.update(A_genre=effects['genre']['A'], e_genre=effects['genre']['e'],
                      guarded=all(effects[a]['e'] >= -RANGES[a] for a in ('emotion', 'style')))
    else:
        assert run['status'] == 'failed' and run['cause'] and effects is None
    return record


def choose_first(records):
    usable = [v for v in variants.FIRST if records[v]['status'] == 'ok']
    guarded = [v for v in usable if records[v]['guarded']]
    if not usable:
        return dict(closest=None, method=None, rung=None, guarded_pool=False,
                    closest_guarded=False, extras=[])
    closest = max(guarded or usable, key=lambda v: records[v]['A_genre'])
    method, rung = variants.parse(closest)
    ladder = variants.LADDERS[method]
    run = {variants.parse(v)[1] for v in variants.FIRST if variants.parse(v)[0] == method}
    above = [r for r in ladder if r > rung and r not in run]
    below = [r for r in ladder if r < rung and r not in run]
    if rung == max(run) and records[closest]['guarded']:
        extras = above[:2]
    else:
        extras = below[-1:] + above[:1]
    return dict(closest=closest, method=method, rung=rung, guarded_pool=bool(guarded),
                closest_guarded=records[closest]['guarded'],
                extras=[variants.name(method, r) for r in extras])


def choose_all(records, extras, target):
    order = list(variants.FIRST) + list(extras)
    guarded = [v for v in order if records[v]['guarded']]
    winner = max(guarded, key=lambda v: records[v]['A_genre']) if guarded else None
    a = records[winner]['A_genre'] if winner else None
    e = records[winner]['e_genre'] if winner else None
    level = target - RANGES['genre']
    kept = [v for v in order if records[v]['A_genre'] is not None and records[v]['A_genre'] >= level]
    return dict(winner=winner, A_genre_winner=a, e_genre_winner=e,
                outcome='III-GO' if a is not None and a >= target else
                        'III-PROM' if a is not None and a >= level else 'III-NONE',
                fresh_seeds=bool(winner and e >= RANGES['genre']), kept_weights=kept,
                pending_deletions=[v for v in order if v not in kept])


def c_outcome(effects):
    guard = all(effects[a]['e'] >= -RANGES[a] for a in ('emotion', 'style'))
    if guard and effects['genre']['e'] >= RANGES['genre'] and effects['genre']['lo'] > 0:
        return 'C-HOLDS'
    return 'C-QUAL' if guard and effects['genre']['e'] > 0 else 'C-FAILS'


def fresh_reading(seed0_outcome, outcome):
    sentence = f'Seed-0 outcome {seed0_outcome} stands; '
    if outcome is None:
        return sentence + 'no C outcome is stated.'
    if seed0_outcome == 'III-NONE':
        return sentence + f'{outcome} does not make it GO or promising.'
    if outcome == 'C-FAILS':
        return sentence + 'not repeated on fresh seeds (C-FAILS).'
    return sentence + f'fresh seeds: {outcome}.'


def versions():
    return dict(sklearn=sklearn.__version__, numpy=np.__version__, scipy=scipy.__version__,
                threadpoolctl=threadpoolctl.__version__)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def array_sha(array):
    return sha(np.ascontiguousarray(array).tobytes())


def finite(value):
    if isinstance(value, dict):
        for v in value.values():
            finite(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            finite(v)
    elif isinstance(value, (int, float, np.number)):
        assert np.isfinite(value), 'non-finite value'


def differences(ours, theirs, path=''):
    """Compare complete subtrees, including missing/extra keys and strict booleans."""
    missing = '<missing>'
    out = []
    if isinstance(ours, dict) and isinstance(theirs, dict):
        for key in sorted(set(ours) | set(theirs)):
            out.extend(differences(ours.get(key, missing), theirs.get(key, missing),
                                   f'{path}/{key}' if path else key))
        return out
    if isinstance(ours, list) and isinstance(theirs, list):
        if len(ours) != len(theirs):
            return [dict(path=path, ours=ours, theirs=theirs)]
        for i, (a, b) in enumerate(zip(ours, theirs)):
            out.extend(differences(a, b, f'{path}/{i}'))
        return out
    if isinstance(ours, bool) or isinstance(theirs, bool):
        equal = type(ours) is type(theirs) and ours == theirs
    elif isinstance(ours, (int, float)) and isinstance(theirs, (int, float)):
        equal = np.isfinite(ours) and np.isfinite(theirs) and abs(ours - theirs) <= 1e-6
    else:
        equal = ours == theirs
    return [] if equal else [dict(path=path, ours=ours, theirs=theirs)]


class Inputs:
    """Read/validate raw inputs, retain their hashes and our independent partitions."""

    def __init__(self, outputs, rb, n_rows, width, p_sha, bad):
        self.outputs, self.rb = Path(outputs), Path(rb)
        self.n_rows, self.width, self.p_sha = n_rows, width, p_sha
        self.hashes, self.parts, self.img_hashes, self.runs = {}, {}, {}, {}
        self.bad = bad

    def bytes(self, path):
        path = Path(path)
        data = path.read_bytes()
        self.hashes[str(path)] = sha(data)
        return data

    def json(self, path, keys):
        data = json.loads(self.bytes(path))
        assert isinstance(data, dict) and set(keys) <= data.keys(), f'{path}: required keys {keys}'
        finite(data)
        return data

    def check(self, ours, theirs, path):
        self.bad.extend(differences(ours, theirs, path))

    def run_record(self, variant, seed):
        key = variant, seed
        if key not in self.runs:
            path = paths.iii_run(self.outputs, variant, seed) / 'run.json'
            run = self.json(path, ('variant', 'seed', 'switches', 'status', 'best_epoch'))
            assert run['variant'] == variant and run['seed'] == seed, f'{path}: run identity'
            assert run['switches'] == variants.switches(variant), f'{path}: switches'
            assert run['status'] in ('ok', 'failed')
            assert set(run['best_epoch']) == set(ASPECTS)
            assert all(v is None or (type(v) is int and v >= 0) for v in run['best_epoch'].values())
            if run['status'] == 'failed':
                assert isinstance(run.get('cause'), str) and run['cause'], f'{path}: failed run needs cause'
                assert run.get('failure_kind') in ('nonfinite_loss', 'vg3'), f'{path}: failure_kind'
            else:
                assert run.get('cause') is None, f'{path}: successful run has failure cause'
            self.check(self.p_sha, run.get('p_sha256', '<missing>'), f'inputs/{variant}/s{seed}/p_sha256')
            if not isinstance(run.get('code_commit'), str) or not run['code_commit']:
                self.check('nonempty commit', run.get('code_commit', '<missing>'),
                           f'inputs/{variant}/s{seed}/code_commit')
            self.runs[key] = run
        return self.runs[key]

    def partitions(self, variant, seed, aspect):
        key = variant, seed, aspect
        if key in self.parts:
            return self.parts[key]
        if variant == BASE and seed == 0:
            folder = self.rb / 'train' / BASE
            meta = self.json(folder / f'{aspect}__s0.json', ('recipe', 'aspect', 'seed', 'p_sha256'))
            assert (meta['recipe'], meta['aspect'], meta['seed'], meta['p_sha256']) == (
                BASE, aspect, 0, self.p_sha), 'frozen export identity'
            path = folder / f'{aspect}__s0__P.npz'
        else:
            run = self.run_record(variant, seed)
            assert run['status'] == 'ok', f'{variant}/s{seed}: failed run cannot be scored'
            path = paths.iii_run(self.outputs, variant, seed) / f'{aspect}__P.npz'
        # Hash the actual file as well as the raw saved array (the writer's identity).
        self.bytes(path)
        with np.load(path, allow_pickle=False) as z:
            assert 'img' in z.files and set(z.files) <= {'img', 'cap'}, f'{path}: keys'
            x = z['img']
            assert x.dtype == np.float16 and x.shape == (self.n_rows, self.width), f'{path}: shape/dtype'
            if 'cap' in z.files:
                assert z['cap'].shape == x.shape and z['cap'].dtype == np.float16
                assert np.isfinite(z['cap']).all()
        self.img_hashes[key] = array_sha(x)
        x = x.astype(np.float32)
        assert np.isfinite(x).all(), f'{path}: nonfinite image'
        assert np.all(np.abs(np.linalg.norm(x, axis=1) - 1) < 1e-3), f'{path}: unit norm'
        with threadpool_limits(limits=8):
            parts = np.stack([MiniBatchKMeans(n_clusters=20, random_state=s, n_init=3,
                                              batch_size=4096).fit_predict(x) for s in (0, 1, 2)])
        parts = parts.astype(np.int64)
        assert parts.shape == (3, self.n_rows) and ((parts >= 0) & (parts < 20)).all()
        self.parts[key] = parts
        return parts

    def compare_partitions(self):
        # This is called only after *all* independent partitions/statistics exist.
        for (v, s, a), ours in self.parts.items():
            path = self.outputs / 'stage_iii/partitions' / v / f's{s}' / f'{a}__partitions.npz'
            prefix = f'partitions/{v}/s{s}/{a}'
            if not path.exists():
                self.check('partition cache', '<missing>', prefix)
                continue
            self.bytes(path)
            with np.load(path, allow_pickle=False) as z:
                assert set(z.files) == {'labels'}, f'{path}: keys'
                theirs = z['labels']
            if theirs.dtype != np.int64 or theirs.shape != ours.shape:
                self.check(dict(shape=list(ours.shape), dtype='int64'),
                           dict(shape=list(theirs.shape), dtype=str(theirs.dtype)), prefix)
            elif not np.array_equal(ours, theirs):
                mismatch = np.argwhere(ours != theirs)
                self.bad.append(dict(path=prefix, ours=dict(sha256=array_sha(ours)),
                                     theirs=dict(sha256=array_sha(theirs), n_different=len(mismatch),
                                                 first_indices=mismatch[:8].tolist())))
            side = self.json(path.with_suffix('.json'), ('identity', 'versions', 'k', 'states', 'array'))
            assert all(isinstance(side[k], dict) for k in ('identity', 'versions', 'array')), f'{path}: sidecar'
            identity = dict(variant=v, aspect=a, seed=s, p_sha256=self.p_sha,
                            img_sha256=self.img_hashes[v, s, a], sklearn=sklearn.__version__, numpy=np.__version__)
            self.check(identity, {k: side['identity'].get(k, '<missing>') for k in identity}, prefix+'/identity')
            self.check(versions(), side['versions'], prefix+'/versions')
            self.check(dict(k=20, states=[0, 1, 2], array=dict(shape=list(theirs.shape),
                       dtype=str(theirs.dtype), sha256=array_sha(theirs))),
                       {k: side[k] for k in ('k', 'states', 'array')}, prefix+'/metadata')


def seed0_scores(inputs, labels, order, n_draws, workers):
    runs = {v: inputs.run_record(v, 0) for v in order}
    effects = {v: {} for v in order if runs[v]['status'] == 'ok'}
    base_a = {}
    for a in ASPECTS:
        parts = {v: {0: inputs.partitions(v, 0, a)} for v in [BASE]+list(effects)}
        scored = score_effects(labels[a], parts, n_draws, workers)
        base_a[a] = scored[BASE]['A']
        for v in effects:
            effects[v][a] = scored[v]
    return base_a, {v: variant_result(runs[v], effects.get(v)) for v in order}


def seed0_identity(inputs, labels_sha, order, n_draws):
    return dict(p_sha256=inputs.p_sha, label_sha256=labels_sha, n_draws=n_draws, variants=order,
                img_sha256={v: {a: inputs.img_hashes[v, 0, a] for a in ASPECTS}
                            for v in [BASE]+order if (v, 0, 'emotion') in inputs.img_hashes},
                run_sha256={v: inputs.hashes[str(paths.iii_run(inputs.outputs, v, 0)/'run.json')] for v in order},
                failed={v: inputs.runs[v, 0]['cause'] for v in order if inputs.runs[v, 0]['status'] == 'failed'})


def weights_equal(inputs, fresh_path, frozen_path):
    """Compare state tensor bytes, independent of torch's archive serialization."""
    import torch
    for path in (fresh_path, frozen_path):
        inputs.bytes(path)
    fresh = torch.load(fresh_path, map_location='cpu', weights_only=True)
    frozen = torch.load(frozen_path, map_location='cpu', weights_only=True)
    assert isinstance(fresh, dict) and isinstance(frozen, dict) and fresh and frozen
    assert all(isinstance(t, torch.Tensor) and torch.isfinite(t).all() for t in fresh.values())
    assert all(isinstance(t, torch.Tensor) and torch.isfinite(t).all() for t in frozen.values())
    return bool(fresh.keys() == frozen.keys() and all(
        fresh[k].shape == frozen[k].shape and fresh[k].dtype == frozen[k].dtype and
        torch.equal(fresh[k].contiguous().view(torch.uint8), frozen[k].contiguous().view(torch.uint8))
        for k in fresh))


def parity_differences(ours, theirs, path='fresh_parity'):
    """Only parity decisions enter VG2; trainer bookkeeping is descriptive."""
    if not isinstance(theirs, dict):
        return differences(ours, theirs, path)
    reference = theirs.get('reference')
    bad = []
    if not isinstance(reference, str) or Path(reference).parts[-4:] != ('part2', 'sample', 'weights', 'K4'):
        bad.append(dict(path=path+'/reference', ours='path ending part2/sample/weights/K4', theirs=reference))
    expected, actual = {'pass': ours['pass'], 'seeds': {}}, {'pass': theirs.get('pass'), 'seeds': {}}
    seeds = theirs.get('seeds', {})
    assert isinstance(seeds, dict)
    for s in sorted(set(ours['seeds']) | set(seeds)):
        mine, saved = ours['seeds'].get(s, {}), seeds.get(s, {})
        assert isinstance(saved, dict)
        keys = ['pass', 'run_status'] + (['weights_equal'] if 'weights_equal' in saved else [])
        expected['seeds'][s] = {k: mine.get(k, '<missing>') for k in keys}
        actual['seeds'][s] = {k: saved.get(k, '<missing>') for k in keys}
    return bad + differences(expected, actual, path)


def fresh_scores(inputs, labels, label_sha, header, seed0, n_draws, workers, rb_part2):
    winner = seed0['winner']
    ours = dict(header, winner=winner, seeds=list(SEEDS), seed0_outcome=seed0['outcome'],
                A_genre_seed0=seed0['A_genre_winner'], c_outcome=None, no_outcome_cause=None,
                effects=None, genre_vs_targets=None, runs={}, parity=None)
    identity = dict(p_sha256=inputs.p_sha, label_sha256=label_sha, n_draws=n_draws,
                    winner=winner, seeds=list(SEEDS), failed={}, run_sha256={}, img_sha256={})
    if not seed0['fresh_seeds']:
        ours['no_outcome_cause'] = 'fresh seeds not triggered by the independently computed seed-0 effects'
    else:
        parity_seeds = {}
        for v in (winner, BASE):
            ours['runs'][v], identity['run_sha256'][v] = {}, {}
            for s in SEEDS:
                run = inputs.run_record(v, s)
                digest = inputs.hashes[str(paths.iii_run(inputs.outputs, v, s)/'run.json')]
                ours['runs'][v][str(s)] = {k: run.get(k) for k in ('status', 'cause', 'best_epoch')}
                ours['runs'][v][str(s)]['run_sha256'] = digest
                identity['run_sha256'][v][str(s)] = digest
                if run['status'] == 'failed':
                    identity['failed'][f'{v}/s{s}'] = run['cause']
        for s in SEEDS:
            run = inputs.runs[BASE, s]
            equal = {}
            for a in ASPECTS:
                try:
                    equal[a] = run['status'] == 'ok' and weights_equal(inputs,
                        paths.iii_run(inputs.outputs, BASE, s)/f'{a}.pt',
                        Path(rb_part2)/'sample/weights/K4'/f'{a}__s{s}.pt')
                except FileNotFoundError as error:
                    inputs.bad.append(dict(path=f'parity/s{s}/{a}', ours='raw weights for independent parity',
                                           theirs=str(error)))
                    equal[a] = False
            parity_seeds[str(s)] = {'pass': all(equal.values()), 'run_status': run['status'],
                                   'run_cause': run.get('cause'), 'weights_equal': equal}
        parity_ok = all(p['pass'] for p in parity_seeds.values())
        expected_parity = {'pass': parity_ok, 'seeds': parity_seeds}
        # Draw fresh effects from raw exports, even when the parity gate fails.
        if not identity['failed']:
            effects = {}
            for a in ASPECTS:
                parts = {v: {s: inputs.partitions(v, s, a) for s in SEEDS} for v in (winner, BASE)}
                effects[a] = score_effects(labels[a], parts, n_draws, workers, fresh=True)[winner]
            ours['effects'] = effects
            abar = effects['genre']['Abar'][winner]
            target, level = header['t_g'], header['promising']
            ours['genre_vs_targets'] = dict(A_bar=abar, t_g=target, promising=level,
                diff_t_g=abar-target, diff_promising=abar-level, reaches_t_g=abar >= target,
                reaches_promising=abar >= level)
            if parity_ok:
                ours['c_outcome'] = c_outcome(effects)
            else:
                ours['no_outcome_cause'] = 'fresh R6-5 weights differ from frozen K4 weights'
        else:
            ours['no_outcome_cause'] = 'at least one fresh-seed run failed'
        for v in (winner, BASE):
            identity['img_sha256'][v] = {
                str(s): {a: inputs.img_hashes[v, s, a] for a in ASPECTS}
                for s in SEEDS if (v, s, 'emotion') in inputs.img_hashes}
        parity_path = inputs.outputs/'stage_iii/fresh_parity.json'
        parity = inputs.json(parity_path, ('pass', 'seeds', 'reference'))
        inputs.bad.extend(parity_differences(expected_parity, parity))
        digest = inputs.hashes[str(parity_path)]
        ours['parity'] = dict(parity, sha256=digest)
        identity['fresh_parity_sha256'] = digest
    # This file's hash authenticates the compared record; no score is used here.
    identity['score_all_sha256'] = sha(inputs.bytes(inputs.outputs/'stage_iii/score_all.json'))
    ours['identity'] = identity
    ours['reading'] = fresh_reading(ours['seed0_outcome'], ours['c_outcome'])
    return ours


def compare_record(inputs, ours, theirs, phase):
    """Compare rule-defined content; check explanatory prose by its rule meaning.

    The rule specifies outcomes and the D34 failure phrase, not arbitrary cause
    wording or a template for the rest of the reading sentence.
    """
    expected = dict(ours)
    if phase == 'fresh':
        parity = expected.pop('parity')
        if parity is None:
            inputs.check(None, theirs.get('parity', '<missing>'), 'score_fresh/parity')
        else:
            saved_parity = theirs.get('parity')
            inputs.bad.extend(parity_differences(parity, saved_parity, 'score_fresh/parity'))
            inputs.check(parity['sha256'], saved_parity.get('sha256') if isinstance(saved_parity, dict) else None,
                         'score_fresh/parity/sha256')
        expected.pop('reading')
        cause = expected.pop('no_outcome_cause')
        actual = theirs.get('no_outcome_cause', '<missing>')
        valid = ('no_outcome_cause' in theirs and
                 (actual is None if cause is None else isinstance(actual, str) and bool(actual)))
        if not valid:
            inputs.check(cause, actual, 'score_fresh/no_outcome_cause')
        reading = theirs.get('reading')
        tokens = [ours['seed0_outcome']]
        if ours['c_outcome']:
            tokens.append(ours['c_outcome'])
        if ours['seed0_outcome'] in ('III-GO', 'III-PROM') and ours['c_outcome'] == 'C-FAILS':
            tokens.append('not repeated on fresh seeds')
        if not isinstance(reading, str) or not all(t in reading for t in tokens):
            inputs.check(dict(required_phrases=tokens), reading, 'score_fresh/reading')
    inputs.check(expected, {k: theirs.get(k, '<missing>') for k in expected}, f'score_{phase}')
    for key, kind in (('code_commit', str), ('code_dirty', bool), ('code_sha256', str)):
        if not isinstance(theirs.get(key), kind):
            inputs.check(kind.__name__, theirs.get(key, '<missing>'), f'score_{phase}/{key}')


def run(outputs, phase, *, smoke=False, workers=8, rb_part1=None, annotations=None,
        genre_loader=None, expected_p_sha=paths.P_SHA256, n_rows=29232, width=2048,
        report_outputs=None, rb_part2=None):
    """Re-derive from raw inputs, then compare and write a phase-specific report.

    Explicit data/path/shape arguments support tiny synthetic datasets. CLI runs
    always use the frozen population, width and P hash and enforce smoke paths.
    """
    assert phase in ('first', 'all', 'fresh') and 1 <= workers <= 8
    n_draws = 20 if smoke else 2000
    bad = []
    inputs = Inputs(outputs, paths.RB_PART1 if rb_part1 is None else rb_part1,
                    n_rows, width, expected_p_sha, bad)
    complete = False
    ours = None
    label_sha = {}
    try:
        for rule in (paths.RULE, Path(__file__).parent.parent/'20261025_recipe_backbones/DECISION_RULE.md',
                     Path(__file__).parent.parent/'20261020_claim_test/DECISION_RULE.md',
                     Path(variants.__file__)):
            inputs.bytes(rule)
        p = inputs.json(inputs.rb/'P.json', ('names', 'sha256'))
        names = p['names']
        assert len(names) == n_rows and names == sorted(set(names)), 'P row count/order/unique names'
        assert sha('\n'.join(names).encode()) == p['sha256'] == expected_p_sha, 'P hash'
        pick = inputs.json(inputs.rb/'pick.json', ('p_sha256', 'n_rows', 'label_sha256'))
        assert pick['p_sha256'] == expected_p_sha and pick['n_rows'] == n_rows
        reference = inputs.json(inputs.rb/'reported.json', ('p_sha256', 'references_p'))
        assert reference['p_sha256'] == expected_p_sha
        target = reference['references_p']['genre_word']['genre']
        assert isinstance(target, float) and 0 <= target <= 1
        inputs.check(target, variants.T_G, 'eg_variants/T_G')
        inputs.check(RANGES, variants.RANGES, 'eg_variants/RANGES')
        if annotations is None:
            from src.data.artelingo import ANNOTATIONS_PATH
            annotations = json.loads(inputs.bytes(ANNOTATIONS_PATH))
        assert isinstance(annotations, list) and annotations
        assert all({'painting', 'emotion', 'art_style'} <= row.keys() for row in annotations)
        if genre_loader is None:
            from src.data.wikiart_genre import GENRE_CSVS, load_wikiart_genre
            for path in GENRE_CSVS:
                inputs.bytes(path)
            genre_loader = load_wikiart_genre
        from rb_vp1 import labels_from_annotations
        labels = labels_from_annotations(names, annotations, genre_loader(names))
        assert set(labels) == set(ASPECTS)
        for y in labels.values():
            assert y.shape == (n_rows,) and y.dtype == np.int64 and (y >= -1).all() and (y >= 0).any()
        label_sha = {a: array_sha(labels[a]) for a in ASPECTS}
        inputs.check(label_sha, pick['label_sha256'], 'RB_PART1/pick/label_sha256')
        header = dict(stage=phase, smoke=smoke, n_draws=n_draws, reference=BASE, seed=0, t_g=target,
                      promising=target-RANGES['genre'], ranges=RANGES, versions=versions(),
                      label_counts={a: int((labels[a] >= 0).sum()) for a in ASPECTS})
        base_a, records = seed0_scores(inputs, labels, list(variants.FIRST), n_draws, workers)
        first = choose_first(records)
        if phase == 'first':
            order = list(variants.FIRST)
            ours = dict(header, base_A=base_a, variants=order, results=records, **first,
                        identity=seed0_identity(inputs, label_sha, order, n_draws))
        else:
            extras = first['extras']
            if extras:
                _, extra_records = seed0_scores(inputs, labels, extras, n_draws, workers)
                records.update(extra_records)
            order = list(variants.FIRST)+extras
            all_values = choose_all(records, extras, target)
            ours = dict(header, base_A=base_a, variants=order, extras=extras, results=records,
                        **all_values, identity=seed0_identity(inputs, label_sha, order, n_draws))
            if phase == 'fresh':
                all_ours = dict(ours, stage='all')
                ours = fresh_scores(inputs, labels, label_sha, header, ours, n_draws, workers,
                                    paths.RB_PART2 if rb_part2 is None else rb_part2)
        finite(ours)
        # Scores and caches under examination are first read here.
        record_path = Path(outputs)/f'stage_iii/score_{phase}.json'
        theirs = inputs.json(record_path, ('stage', 'n_draws'))
        compare_record(inputs, ours, theirs, phase)
        if phase == 'fresh':
            all_theirs = inputs.json(Path(outputs)/'stage_iii/score_all.json', ('stage', 'n_draws'))
            compare_record(inputs, all_ours, all_theirs, 'all')
        inputs.compare_partitions()
        complete = True
    except (AssertionError, FileNotFoundError, KeyError, ValueError, TypeError) as error:
        bad.append(dict(path='inputs', ours='valid complete raw inputs and record',
                        theirs=f'{type(error).__name__}: {error}'))
    report = dict(phase=phase, agree=complete and not bad, differences=bad, complete=complete,
                  code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  n_draws=n_draws, smoke=smoke, input_hashes=inputs.hashes,
                  input_array_hashes=dict(p_sha256=expected_p_sha, label_sha256=label_sha,
                      img_sha256={f'{v}/s{s}/{a}': h for (v, s, a), h in inputs.img_hashes.items()}),
                  recomputed=ours)
    finite(report)
    dest = Path(outputs if report_outputs is None else report_outputs)/f'stage_iii/rederive_vg2_{phase}.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('first', 'all', 'fresh'), required=True)
    parser.add_argument('--outputs', type=Path, default=paths.OUTPUTS)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    parser.add_argument('--smoke', action='store_true', help='20 independent draws; reports only under /tmp/claude-0/')
    parser.add_argument('--report-outputs', type=Path,
                        help='write stage_iii/rederive_vg2_<phase>.json here; read inputs from --outputs')
    args = parser.parse_args(argv)
    if args.smoke and not paths.smoke_ok(args.report_outputs or args.outputs):
        parser.error('--smoke refuses report writes outside /tmp/claude-0/')
    paths.check_imports()
    report = run(args.outputs, args.phase, smoke=args.smoke, workers=args.workers,
                 report_outputs=args.report_outputs)
    print(f'VG2 phase={args.phase} agree={report["agree"]}; {len(report["differences"])} differences; '
          f'{report["n_draws"]} draws; complete={report["complete"]}')
    return 0 if report['agree'] else 1


if __name__ == '__main__':
    sys.exit(main())
